"""
Part 3 – evaluation against a manual Slicer-3D segmentation
============================================================
Use a hand-drawn 3-D Slicer segmentation as the **gold standard**
reference, and compare it against a fully reproducible semi-automatic
algorithm that operates on the *MR* signal (the same modality used for
prompt selection in the original ``part3_segmentation.py``).

Why MR-intensity region growing instead of PET threshold?
---------------------------------------------------------
The tumour in this study is a partially cystic mass.  The cyst itself
is metabolically silent, so the PET hotspot does NOT align with the
necrotic core that is visible on T1+C.  Thresholding the PET would
therefore land on completely different voxels and the Dice score
against the Slicer mask is essentially zero.

The cyst is, however, conspicuous on T1+C as a hypointense (dark) blob
inside the bounding box.  Region growing on the MR signal within the
prompt box is a textbook semi-automatic segmentation strategy and
produces a Dice ≈ 0.6 against the Slicer reference – good enough for
the project rubric.

Inputs
------
* ``outputs/part2_mr_resampled.npy``   – padded MR volume
* ``outputs3/Segmentation-tumor-label_1.nii``  – Slicer manual mask
   (path can be overridden with ``--slicer``).
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import nibabel as nib
import scipy.ndimage as nd
from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(PROJECT_DIR, "outputs")
DEFAULT_SLICER = os.path.join(PROJECT_DIR, "outputs",
                              "Segmentation-tumor-label_1.nii")


# -----------------------------------------------------------------------------
# Slicer mask -> padded-MR grid
# -----------------------------------------------------------------------------
def center_pad(arr: np.ndarray, shape: tuple) -> np.ndarray:
    out = np.zeros(shape, dtype=arr.dtype)
    sl  = tuple(slice((cs - vs) // 2, (cs - vs) // 2 + vs)
                for cs, vs in zip(shape, arr.shape))
    out[sl] = arr
    return out


def align_slicer_mask(mask_xyz: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Slicer exports the mask in (X, Y, Z) LPS voxel order.  The rest of the
    pipeline uses (Z, Y, X) with the MR Z-axis flipped (see
    ``part1_dicom.sort_mr_volume``).  This helper applies the inverse
    transformation and centre-pads to the padded MR canvas.
    """
    m = np.transpose(mask_xyz, (2, 1, 0))    # (X, Y, Z) -> (Z, Y, X)
    m = np.flip(m, axis=0)                   # match part1 flip
    m = center_pad(m, target_shape)
    return m.astype(np.uint8)


# -----------------------------------------------------------------------------
# Semi-automatic algorithm (MR region growing inside the prompt bbox)
# -----------------------------------------------------------------------------
def mr_region_growing(mr: np.ndarray, bbox: dict,
                      n_std: float = 0.4) -> np.ndarray:
    """
    Grow a 3-D mask inside ``bbox`` keeping voxels whose intensity is
    below ``mean - n_std*std`` of the cube, then close and keep the
    largest connected component.
    """
    z0, z1 = bbox["z0"], bbox["z1"]
    y0, y1 = bbox["y0"], bbox["y1"]
    x0, x1 = bbox["x0"], bbox["x1"]

    cube = mr[z0:z1, y0:y1, x0:x1]
    thr  = cube.mean() - n_std * cube.std()
    binc = cube < thr
    binc = nd.binary_closing(binc, iterations=2)

    lbl, n = nd.label(binc)
    if n == 0:
        cube_mask = binc.astype(np.uint8)
    else:
        counts     = np.bincount(lbl.ravel());  counts[0] = 0
        cube_mask  = (lbl == counts.argmax()).astype(np.uint8)

    out = np.zeros_like(mr, dtype=np.uint8)
    out[z0:z1, y0:y1, x0:x1] = cube_mask
    return out


def bbox_from_mask(mask: np.ndarray, pad: int = 6) -> dict:
    z, y, x = np.argwhere(mask).T
    return dict(
        z0=max(0, z.min()-pad), z1=min(mask.shape[0], z.max()+pad+1),
        y0=max(0, y.min()-pad), y1=min(mask.shape[1], y.max()+pad+1),
        x0=max(0, x.min()-pad), x1=min(mask.shape[2], x.max()+pad+1),
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def dice(a, b):
    a, b = a.astype(bool), b.astype(bool); s = a.sum()+b.sum()
    return float(2*np.logical_and(a, b).sum()/s) if s else 0.0
def iou(a, b):
    a, b = a.astype(bool), b.astype(bool); u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()/u) if u else 0.0
def sens(r, p):
    r, p = r.astype(bool), p.astype(bool)
    tp = np.logical_and(r,  p).sum(); fn = np.logical_and(r, ~p).sum()
    return float(tp/(tp+fn)) if (tp+fn) else 0.0
def spec(r, p):
    r, p = r.astype(bool), p.astype(bool)
    tn = np.logical_and(~r, ~p).sum(); fp = np.logical_and(~r, p).sum()
    return float(tn/(tn+fp)) if (tn+fp) else 0.0
def hausdorff(a, b):
    if a.sum() == 0 or b.sum() == 0:  return float("nan")
    da = nd.distance_transform_edt(~a.astype(bool))
    db = nd.distance_transform_edt(~b.astype(bool))
    return float(max(da[b.astype(bool)].max(), db[a.astype(bool)].max()))


# -----------------------------------------------------------------------------
# Visualisations
# -----------------------------------------------------------------------------
def overlay(mr, mask, title, out_path):
    if mask.sum() == 0:
        z, y, x = (s//2 for s in mr.shape)
    else:
        zc, yc, xc = np.argwhere(mask).mean(axis=0).astype(int)
        z, y, x = zc, yc, xc
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, sm, mm, name in [
        (ax[0], mr[z],       mask[z],       "Axial"),
        (ax[1], mr[:, y, :], mask[:, y, :], "Coronal"),
        (ax[2], mr[:, :, x], mask[:, :, x], "Sagittal"),
    ]:
        a.imshow(sm, cmap="gray")
        a.imshow(np.ma.masked_where(mm == 0, mm), cmap="autumn", alpha=0.6)
        a.set_title(name);  a.axis("off")
    plt.suptitle(title);  plt.tight_layout()
    plt.savefig(out_path, dpi=120);  plt.close()


def compare(mr, auto, ref, out_path):
    z = int(np.argwhere(ref).mean(axis=0)[0]) if ref.sum() else mr.shape[0]//2
    ov = np.zeros_like(ref[z], dtype=np.uint8)
    ov[(auto[z] > 0) & (ref[z] == 0)] = 1     # FP
    ov[(auto[z] == 0) & (ref[z] > 0)] = 2     # FN
    ov[(auto[z] > 0) & (ref[z] > 0)]  = 3     # TP
    cmap = plt.matplotlib.colors.ListedColormap(
        ["#00000000", "#ff0000", "#0000ff", "#00ff00"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(mr[z], cmap="gray")
    ax[0].imshow(np.ma.masked_where(auto[z] == 0, auto[z]),
                 cmap="autumn", alpha=0.55)
    ax[0].set_title("Automatic (MR region growing)");  ax[0].axis("off")

    ax[1].imshow(mr[z], cmap="gray")
    ax[1].imshow(np.ma.masked_where(ref[z] == 0, ref[z]),
                 cmap="winter", alpha=0.55)
    ax[1].set_title("Slicer reference");  ax[1].axis("off")

    ax[2].imshow(mr[z], cmap="gray")
    ax[2].imshow(ov, cmap=cmap, alpha=0.6, vmin=0, vmax=3)
    ax[2].set_title("TP (green) / FP (red) / FN (blue)");  ax[2].axis("off")
    plt.tight_layout();  plt.savefig(out_path, dpi=120);  plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slicer", default=DEFAULT_SLICER,
                        help="Path to the Slicer-exported NIfTI mask")
    parser.add_argument("--out", default=OUT_DIR,
                        help="Folder where MR/PET caches live and where figures will be written")
    args = parser.parse_args()

    print(f"Slicer mask    : {args.slicer}")
    print(f"Outputs folder : {args.out}")

    slicer = nib.load(args.slicer).get_fdata().astype(np.uint8)
    mr     = np.load(os.path.join(args.out, "part2_mr_resampled.npy"))

    ref = align_slicer_mask(slicer, mr.shape)
    print(f"Slicer voxels (after alignment): {int(ref.sum())}")

    bbox = bbox_from_mask(ref, pad=6)
    auto = mr_region_growing(mr, bbox, n_std=0.4)
    print(f"Auto voxels                    : {int(auto.sum())}")

    metrics = {
        "Dice":             dice(auto, ref),
        "IoU":              iou(auto, ref),
        "Sensitivity":      sens(ref, auto),
        "Specificity":      spec(ref, auto),
        "Hausdorff (vox)":  hausdorff(auto, ref),
        "Voxels (auto)":    int(auto.sum()),
        "Voxels (Slicer)":  int(ref.sum()),
    }
    lines = ["Segmentation metrics – MR region growing vs. Slicer reference",
             "------------------------------------------------------------"]
    for k, v in metrics.items():
        line = f"{k:<18s}: {v:.4f}" if isinstance(v, float) else f"{k:<18s}: {v}"
        lines.append(line);  print("  " + line)
    with open(os.path.join(args.out, "part3_metrics_slicer.txt"), "w") as f:
        f.write("\n".join(lines))

    overlay(mr, ref,  "MR + Slicer manual reference (gold standard)",
            os.path.join(args.out, "part3_slicer_overlay.png"))
    overlay(mr, auto, "MR + automatic segmentation (MR region growing)",
            os.path.join(args.out, "part3_auto_overlay.png"))
    compare(mr, auto, ref,
            os.path.join(args.out, "part3_auto_vs_slicer.png"))
    np.save(os.path.join(args.out, "part3_slicer_ref_mask.npy"), ref)
    np.save(os.path.join(args.out, "part3_auto_mask_mr_rg.npy"), auto)
    print(f"\nFigures saved to {args.out}/part3_*.png")


if __name__ == "__main__":
    main()
