"""
Part 3 – Semi-automatic 3-D tumour segmentation with MONAI Label
=================================================================
Project 11763 – Medical Image Processing.

Pipeline
--------
1.  Load the cached arrays produced by ``part1_dicom.py`` and the
    registered PET / inverse field saved by ``part2_coregistration.py``.
2.  Display every axial slice of the *last* PET frame and let the user
    select the slice in which the tumour is most visible.
3.  Let the user draw a 2-D bounding box around the tumour on that
    slice.  The box is propagated to 3-D using a small symmetric margin
    along the slice axis (``BOX_Z_MARGIN``).
4.  Compute the **centroid** and the **3-D bounding box** of the tumour
    in registered PET / MR space (both share the same voxel grid).
5.  Save the MR volume as a NIfTI file inside the MONAI Label
    ``studies`` folder and call the ``vista3d`` (or ``deepedit``) model
    of a running MONAI Label server, passing the centroid as a
    foreground click and the bounding box as the prompt.
6.  Load the predicted mask, transform it to the *original* PET space
    using the inverse deformation field from part 2, and visualise it
    on both modalities.
7.  Build a coarse reference mask from the PET image (region-grown
    threshold inside the bounding box) and compare it to the
    automatic mask with **Dice**, **IoU**, **sensitivity** and
    **Hausdorff distance**.

Pre-requisites (MONAI Label setup)
----------------------------------
The simplest way to install and start a MONAI Label server is::

    pip install monailabel monai nibabel
    monailabel apps --download --name radiology --output apps
    mkdir -p studies/brain
    monailabel start_server \\
        --app apps/radiology \\
        --studies studies/brain \\
        --conf models vista3d
        
    windows cmd
    monailabel apps --download --name radiology --output apps

    mkdir studies\brain

    monailabel start_server ^
        --app apps\radiology ^
        --studies studies\brain ^
        --conf models vista3d

The script will then talk to the server through the official
``monailabel.client.MONAILabelClient`` Python API.

If the server is **not** running the script automatically falls back to
an *offline* mode that uses a simple 3-D region-growing algorithm so
that the rest of the project can still be executed end-to-end.
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
import numpy as np
import nibabel as nib
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector


# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------
PROJECT_DIR     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR         = os.path.join(PROJECT_DIR, "outputs3")
STUDIES_DIR     = os.path.join(PROJECT_DIR, "studies", "brain")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(STUDIES_DIR, exist_ok=True)

MONAI_LABEL_SERVER = os.environ.get("MONAI_LABEL_SERVER", "http://127.0.0.1:8000")
# ``deepedit`` is shipped with the stock ``radiology`` sample app and only
# needs point prompts.  Set MONAI_LABEL_MODEL=vista3d if you launched the
# server with ``--app apps/monaibundle --conf models vista3d`` instead
# (VISTA-3D also accepts the bounding-box prompt).
MONAI_LABEL_MODEL  = os.environ.get("MONAI_LABEL_MODEL",  "vista3d")
BOX_Z_MARGIN       = 4    # slices above/below the 2-D box used as 3-D extent


# -----------------------------------------------------------------------------
# Interactive prompt selection
# -----------------------------------------------------------------------------
def select_tumor_slice(pet_last_frame: np.ndarray) -> int:
    """
    Show every axial slice of the last PET frame in a grid so that the
    user can identify the slice where the tumour is most visible.
    """
    n = pet_last_frame.shape[0]/3.27
    
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2.5 * rows))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(pet_last_frame[i+40], cmap="hot", aspect="equal")
            ax.set_title(f"z={i}", fontsize=7)
        ax.axis("off")
    plt.suptitle("PET – last frame, all axial slices (find the bright tumour)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "part3_pet_all_slices.png"), dpi=120)
    plt.show()

    z = int(input("Enter the z slice index where the tumor is clearly visible: "))
    return 85


def draw_bbox(image: np.ndarray, z: int) -> dict:
    """Interactive bounding-box selection on a single 2-D image."""
    bbox: dict = {}

    def on_select(eclick, erelease):
        x_min = int(min(eclick.xdata, erelease.xdata))
        x_max = int(max(eclick.xdata, erelease.xdata))
        y_min = int(min(eclick.ydata, erelease.ydata))
        y_max = int(max(eclick.ydata, erelease.ydata))
        bbox.update(x_min=x_min, x_max=x_max,
                    y_min=y_min, y_max=y_max, z=z)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap="hot")
    ax.set_title(f"Draw a box around the tumour  (z={z})\n"
                 "Close the window when done", fontsize=11)
    _ = RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords="pixels",
        interactive=True,
        props=dict(edgecolor="cyan", linewidth=2, fill=False),
    )
    plt.tight_layout()
    plt.show()

    if not bbox:
        raise RuntimeError("No bounding box was drawn.")

    bbox["x_c"] = (bbox["x_min"] + bbox["x_max"]) // 2
    bbox["y_c"] = (bbox["y_min"] + bbox["y_max"]) // 2
    return bbox


# -----------------------------------------------------------------------------
# MONAI Label client wrapper
# -----------------------------------------------------------------------------
def save_mr_as_nifti(mr_vol: np.ndarray, name: str = "mr") -> str:
    """
    Save the MR volume as a NIfTI file inside the MONAI Label ``studies``
    folder and return the absolute path.  We use an identity affine because
    the MR voxel size is 1×1×1 mm.
    """
    out = os.path.join(STUDIES_DIR, f"{name}.nii.gz")
    nib.save(nib.Nifti1Image(mr_vol.astype(np.float32), np.eye(4)), out)
    return out


def call_monailabel(image_path: str, centroid_zyx: tuple,
                    bbox: dict) -> np.ndarray | None:
    """
    Send an inference request to a running MONAI Label server.

    Returns the predicted label volume aligned with the input MR or
    ``None`` if the server is not reachable.
    """
    try:
        from monailabel.client import MONAILabelClient
    except ImportError:
        print("monailabel is not installed – running offline fallback.")
        return None

    image_id = os.path.basename(image_path).replace(".nii.gz", "")

    try:
        client = MONAILabelClient(MONAI_LABEL_SERVER)
        info   = client.info()
        if MONAI_LABEL_MODEL not in info.get("models", {}):
            print(f"Model {MONAI_LABEL_MODEL!r} not found in the running "
                  f"server.  Available: {list(info.get('models', {}).keys())}")
            return None

        # Upload the MR (idempotent).
        try:
            client.create_image(image_id, image_path)
        except Exception:
            pass    # already uploaded

        # DeepEdit accepts foreground / background point clicks; VISTA-3D
        # additionally accepts a 6-component bounding box.  We always send
        # the foreground click (centroid) and only include ``box`` when
        # the user explicitly opted into VISTA-3D.
        cz, cy, cx = centroid_zyx
        params = {
            "label_info": [{"name": "tumor", "idx": 1}],
            "foreground": [[int(cx), int(cy), int(cz)]],
            "background": [],
        }
        if MONAI_LABEL_MODEL.lower() == "vista3d":
            params["box"] = [int(bbox["x_min"]), int(bbox["y_min"]),
                             int(bbox["z_min"]),
                             int(bbox["x_max"]), int(bbox["y_max"]),
                             int(bbox["z_max"])]
        print(f"Calling {MONAI_LABEL_MODEL} on {image_id} ...")
        result_file, _ = client.infer(MONAI_LABEL_MODEL, image_id,
                                      params=params)
        mask = nib.load(result_file).get_fdata().astype(np.uint8)
        # We saved the MR with an identity affine, so nibabel returns the
        # data in the same (z, y, x) ordering we use everywhere else in
        # the project – no transpose required.
        return mask

    except Exception as exc:
        print(f"MONAI Label inference failed: {exc}")
        return None


# -----------------------------------------------------------------------------
# Offline fallback: 3-D region growing inside the bbox
# -----------------------------------------------------------------------------
def offline_region_growing(volume: np.ndarray, bbox: dict,
                           rel_thr: float = 0.5) -> np.ndarray:
    """
    Connected-component segmentation inside the bounding box used as
    fallback when MONAI Label is not available.  Voxels above
    ``rel_thr * volume.max()`` inside the box are kept; the largest
    connected component is returned.
    """
    mask = np.zeros_like(volume, dtype=np.uint8)
    z0, z1 = bbox["z_min"], bbox["z_max"] + 1
    y0, y1 = bbox["y_min"], bbox["y_max"] + 1
    x0, x1 = bbox["x_min"], bbox["x_max"] + 1

    cube = volume[z0:z1, y0:y1, x0:x1]
    thr  = rel_thr * cube.max()
    bin_cube = cube > thr

    labels, n = nd.label(bin_cube)
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    biggest = counts.argmax()
    mask[z0:z1, y0:y1, x0:x1] = (labels == biggest).astype(np.uint8)
    return mask


# -----------------------------------------------------------------------------
# Reference mask (used as “provided” mask for numerical evaluation)
# -----------------------------------------------------------------------------
def pet_threshold_reference(pet_in_mr_space: np.ndarray, bbox: dict,
                            rel_thr: float = 0.5) -> np.ndarray:
    """
    Build a coarse reference tumour mask from the PET image by
    thresholding inside the bounding box (PET tumour appears very
    bright, so a relative threshold works well).
    """
    return offline_region_growing(pet_in_mr_space, bbox, rel_thr=rel_thr)


# -----------------------------------------------------------------------------
# Numerical evaluation
# -----------------------------------------------------------------------------
def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return float(2 * inter / s) if s > 0 else 0.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    inter = np.logical_and(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def sensitivity(reference: np.ndarray, prediction: np.ndarray) -> float:
    r = reference.astype(bool); p = prediction.astype(bool)
    tp = np.logical_and(r, p).sum()
    fn = np.logical_and(r, ~p).sum()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate Hausdorff distance via Euclidean distance transforms."""
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    da = nd.distance_transform_edt(~a.astype(bool))
    db = nd.distance_transform_edt(~b.astype(bool))
    return float(max(da[b.astype(bool)].max(), db[a.astype(bool)].max()))


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------
def show_segmentation(mr: np.ndarray, mask: np.ndarray,
                      title: str = "MR + Segmented Tumor Mask",
                      out_name: str = "part3_segmentation.png") -> None:
    """Axial / coronal / sagittal mid-views of the segmentation."""
    if mask.sum() == 0:
        z = mr.shape[0] // 2; y = mr.shape[1] // 2; x = mr.shape[2] // 2
    else:
        zc, yc, xc = np.argwhere(mask).mean(axis=0).astype(int)
        z, y, x = zc, yc, xc

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(mr[z], cmap="gray")
    ax[0].imshow(np.ma.masked_where(mask[z] == 0, mask[z]), cmap="autumn", alpha=0.6)
    ax[0].set_title("Axial");  ax[0].axis("off")
    ax[1].imshow(mr[:, y, :], cmap="gray")
    ax[1].imshow(np.ma.masked_where(mask[:, y, :] == 0, mask[:, y, :]),
                 cmap="autumn", alpha=0.6)
    ax[1].set_title("Coronal"); ax[1].axis("off")
    ax[2].imshow(mr[:, :, x], cmap="gray")
    ax[2].imshow(np.ma.masked_where(mask[:, :, x] == 0, mask[:, :, x]),
                 cmap="autumn", alpha=0.6)
    ax[2].set_title("Sagittal"); ax[2].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=120)
    plt.show()


def show_two_masks(mr: np.ndarray,
                   auto: np.ndarray,
                   ref:  np.ndarray) -> None:
    """Side-by-side comparison of automatic and reference masks."""
    z = int(np.argwhere(auto + ref).mean(axis=0)[0]) if (auto + ref).any() \
        else mr.shape[0] // 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(mr[z], cmap="gray")
    ax[0].imshow(np.ma.masked_where(auto[z] == 0, auto[z]),
                 cmap="autumn", alpha=0.6)
    ax[0].set_title("MONAI Label automatic mask")
    ax[0].axis("off")
    ax[1].imshow(mr[z], cmap="gray")
    ax[1].imshow(np.ma.masked_where(ref[z] == 0, ref[z]),
                 cmap="winter", alpha=0.6)
    ax[1].set_title("PET-threshold reference mask")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "part3_compare_masks.png"), dpi=120)
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    pet_4d        = np.load(os.path.join(OUT_DIR, "part1_pet_volume.npy"))
    mr_vol        = np.load(os.path.join(OUT_DIR, "part2_mr_resampled.npy"))
    registered_pet = np.load(os.path.join(OUT_DIR, "part2_registered_pet.npy"))

    # Use the *last* PET frame (highest tracer uptake in the tumour,
    # registered to MR space) for the interactive prompt selection.  Because
    # the user views the *registered* PET, the picked coordinates are already
    # in MR space.
    last_frame_path = os.path.join(OUT_DIR, "part2_registered_pet_last.npy")
    if os.path.exists(last_frame_path):
        last_frame_in_mr = np.load(last_frame_path)
    else:
        # Fall back to the registered mean if part 2 has not produced the
        # last-frame volume yet.
        last_frame_in_mr = registered_pet

    z = select_tumor_slice(last_frame_in_mr)
    bbox = draw_bbox(last_frame_in_mr[z], z)

    bbox["z_min"] = max(0,  z - BOX_Z_MARGIN)
    bbox["z_max"] = min(last_frame_in_mr.shape[0] - 1, z + BOX_Z_MARGIN)
    bbox["z_c"]   = z
    centroid = (bbox["z_c"], bbox["y_c"], bbox["x_c"])

    print("\nBounding box (MR voxel space):")
    for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        print(f"  {k:<6s}: {bbox[k]}")
    print(f"  centroid (z, y, x): {centroid}")

    # ---- run MONAI Label (or fallback) ----------------------------------
    mr_nifti = save_mr_as_nifti(mr_vol, name="mr_brain")
    auto_mask = call_monailabel(mr_nifti, centroid, bbox)
    # auto_mask = np.transpose(auto_mask, (2, 0, 1))

    if auto_mask is None:
        print("\n>>> Falling back to 3-D region growing on the MR image.")
        auto_mask = offline_region_growing(mr_vol, bbox, rel_thr=0.7)

    # ---- reference mask from PET hotspot --------------------------------
    ref_mask = pet_threshold_reference(registered_pet, bbox, rel_thr=0.5)
    
    print(auto_mask.shape)
    print(ref_mask.shape)

    # ---- numerical evaluation -------------------------------------------
    metrics = {
        "Dice":             dice(auto_mask, ref_mask),
        "IoU":              iou(auto_mask, ref_mask),
        "Sensitivity":      sensitivity(ref_mask, auto_mask),
        "Hausdorff (vox)":  hausdorff_distance(auto_mask, ref_mask),
        "Voxels (auto)":    int(auto_mask.sum()),
        "Voxels (ref)":     int(ref_mask.sum()),
    }
    print("\nSegmentation metrics:")
    lines = ["Segmentation metrics", "--------------------"]
    for k, v in metrics.items():
        line = f"{k:<18s}: {v:.4f}" if isinstance(v, float) else f"{k:<18s}: {v}"
        print("  " + line)
        lines.append(line)
    with open(os.path.join(OUT_DIR, "part3_metrics.txt"), "w") as f:
        f.write("\n".join(lines))

    # ---- visualisations -------------------------------------------------
    show_segmentation(mr_vol, auto_mask,
                      title="MR + automatically segmented tumour")
    show_two_masks(mr_vol, auto_mask, ref_mask)

    # ---- bring the mask back to the original PET space ----------------
    inv_field_path = os.path.join(OUT_DIR, "part2_inverse_field.npy")
    if os.path.exists(inv_field_path):
        inv_field = np.load(inv_field_path)         # (3, z, y, x)
        zz, yy, xx = np.meshgrid(np.arange(mr_vol.shape[0]),
                                 np.arange(mr_vol.shape[1]),
                                 np.arange(mr_vol.shape[2]),
                                 indexing="ij")
        coords = np.stack([zz + inv_field[0],
                           yy + inv_field[1],
                           xx + inv_field[2]], axis=0)
        mask_in_pet = nd.map_coordinates(auto_mask.astype(np.float32),
                                         coords, order=0, mode="constant")
        mask_in_pet = (mask_in_pet > 0.5).astype(np.uint8)
        np.save(os.path.join(OUT_DIR, "part3_auto_mask_in_pet.npy"), mask_in_pet)
        print("\nMask successfully transformed back to PET (input) space.")

        # ---- visualise the transformed mask on the original PET --------
        show_segmentation(registered_pet, mask_in_pet,
                          title="Transformed mask on PET (input space)",
                          out_name="part3_mask_on_pet.png")

    np.save(os.path.join(OUT_DIR, "part3_auto_mask_in_mr.npy"), auto_mask)
    np.save(os.path.join(OUT_DIR, "part3_ref_mask_in_mr.npy"),  ref_mask)
    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
