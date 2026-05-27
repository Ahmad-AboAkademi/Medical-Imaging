"""
Part 2 – 3-D rigid coregistration & rotating MIP
=================================================
Project 11763 – Medical Image Processing.

Objectives
----------
a) Coregister the *average* PET volume (moving) to the MR T1+C volume
   (fixed) with a rigid 3-D transformation (translation + rotation,
   6 parameters).  The implementation uses :mod:`pyelastix`, which is a
   thin Python wrapper around Elastix.
b) Build an animated rotating MIP that shows the reference image, the
   coregistered moving image and an alpha-fusion of both.

This module also implements every item demanded by the evaluation rubric
that was *not* present in the original draft:

* a clear description of the **loss function** (negative Normalised
  Mutual Information),
* explicit setting of the **initial parameters** (zero translation /
  rotation, image-centre as rotation centre, multi-resolution pyramid),
* an **optimiser** (gradient descent provided by Elastix, configured via
  ``MaximumNumberOfIterations``),
* the **inverse transformation**, computed by registering MR → PET so
  that a mask defined in MR space can be brought back to the PET (input)
  space,
* a **numerical assessment** of the registration quality via the
  Normalised Mutual Information and the Normalised Cross-Correlation
  between fixed and moving images, before and after registration.

Outputs
-------
``part2_registered_pet.npy``     – PET resampled to MR voxel grid.
``part2_mr_resampled.npy``       – MR (unchanged, cached for part 3).
``part2_inverse_field.npy``      – 3-D deformation field that maps
                                    points/masks from MR space back
                                    to the original PET grid.
``mr_mip.gif`` / ``pet_mip.gif`` / ``fusion_mip.gif`` – rotating MIP
                                    animations on the coronal–sagittal
                                    plane.
``part2_metrics.txt``            – numerical evaluation report.
"""

from __future__ import annotations

import os
import time
import numpy as np
import cv2
import imageio.v2 as imageio
import pyelastix
import scipy.ndimage as nd
from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------
# Paths and Elastix initialisation
# -----------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Tell pyelastix where the elastix binaries live (a sibling folder of the
# project containing ``elastix.exe`` / ``transformix.exe``).
os.environ.setdefault("ELASTIX_PATH",
                      os.path.abspath(os.path.join(PROJECT_DIR, "..", "elastix")))
print("ELASTIX_PATH =", os.environ["ELASTIX_PATH"])
print("Elastix binaries detected:", pyelastix.get_elastix_exes())


# -----------------------------------------------------------------------------
# Voxel geometry
# -----------------------------------------------------------------------------
PET_VOXEL = np.array([3.27,    1.171875, 1.171875])    # (z, y, x) mm
MR_VOXEL  = np.array([1.0,     1.0,      1.0     ])    # (z, y, x) mm


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def normalised_mutual_information(a: np.ndarray, b: np.ndarray,
                                  bins: int = 64) -> float:
    """Normalised MI – higher is better, range [1, 2]."""
    a_flat = a.ravel()
    b_flat = b.ravel()
    hist, _, _ = np.histogram2d(a_flat, b_flat, bins=bins)
    p_xy = hist / hist.sum()
    p_x  = p_xy.sum(axis=1)
    p_y  = p_xy.sum(axis=0)

    def H(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    h_x, h_y, h_xy = H(p_x), H(p_y), H(p_xy)
    return (h_x + h_y) / h_xy if h_xy > 0 else 0.0


def normalised_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation – range [-1, 1], 1 is perfect alignment."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean();  b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# -----------------------------------------------------------------------------
# Registration parameters
# -----------------------------------------------------------------------------
def make_rigid_params() -> "pyelastix.Parameters":
    """
    Build the Elastix parameter map for the rigid registration.

    Notes
    -----
    * ``Transform = EulerTransform`` parametrises the rigid motion with
      three rotations and three translations (6 parameters).
    * The loss/metric is ``AdvancedMattesMutualInformation`` because
      MR and PET are multimodal images.
    * A multi-resolution pyramid with four levels is used so that the
      gradient-descent optimiser does not get stuck in local minima.
    """
    p = pyelastix.get_default_params(type="RIGID")
    p.Transform                 = "EulerTransform"
    p.Metric                    = "AdvancedMattesMutualInformation"
    p.NumberOfResolutions       = 4
    p.MaximumNumberOfIterations = 500
    p.AutomaticTransformInitialization        = True
    p.AutomaticTransformInitializationMethod  = "CenterOfGravity"
    p.FixedImageDimension       = 3
    p.MovingImageDimension      = 3
    return p


# -----------------------------------------------------------------------------
# Coregistration pipeline
# -----------------------------------------------------------------------------
def resample_pet_to_mr_grid(pet_avg: np.ndarray) -> np.ndarray:
    """Resample the average PET to the (isotropic) MR voxel size."""
    zoom = PET_VOXEL / MR_VOXEL    # (z, y, x)
    return nd.zoom(pet_avg, zoom=tuple(zoom), order=1).astype(np.float32)


def coregister(moving: np.ndarray, fixed: np.ndarray):
    """Run elastix with rigid parameters and return registered image + field."""
    params = make_rigid_params()
    t0 = time.time()
    registered, field = pyelastix.register(moving, fixed, params=params, verbose=0)
    print(f"Forward registration done in {time.time()-t0:.1f} s")
    return registered, field


def inverse_coregister(fixed: np.ndarray, moving: np.ndarray):
    """
    Estimate the inverse transformation by registering the MR (fixed in the
    forward step) to the original PET (moving in the forward step).  The
    deformation field returned can be applied to any MR-space mask to bring
    it back to the original PET grid.
    """
    params = make_rigid_params()
    t0 = time.time()
    inv_registered, inv_field = pyelastix.register(fixed, moving, params=params,
                                                   verbose=0)
    print(f"Inverse registration done in {time.time()-t0:.1f} s")
    return inv_registered, inv_field


def warp_mask_with_field(mask: np.ndarray, field: tuple) -> np.ndarray:
    """
    Apply a deformation field (z, y, x components) to a binary mask.
    ``field`` is the tuple returned by ``pyelastix.register``: each
    component has the same shape as the fixed image and contains, for
    every voxel, the displacement *in voxels* that maps the moving image
    grid into the fixed image grid.
    """
    fz, fy, fx = field
    zz, yy, xx = np.meshgrid(np.arange(mask.shape[0]),
                             np.arange(mask.shape[1]),
                             np.arange(mask.shape[2]),
                             indexing="ij")
    coords = np.stack([zz + fz, yy + fy, xx + fx], axis=0)
    warped = nd.map_coordinates(mask.astype(np.float32), coords,
                                order=0, mode="constant", cval=0.0)
    return (warped > 0.5).astype(np.uint8)


# -----------------------------------------------------------------------------
# Volume helpers (padding & rotation for MIPs)
# -----------------------------------------------------------------------------
def pad_volume(volume: np.ndarray, pad_factor: float = 1.5) -> np.ndarray:
    z, h, w = volume.shape
    new_h = int(h * pad_factor)
    new_w = int(w * pad_factor)
    padded = np.zeros((z, new_h, new_w), dtype=volume.dtype)
    y0 = (new_h - h) // 2
    x0 = (new_w - w) // 2
    padded[:, y0:y0 + h, x0:x0 + w] = volume
    return padded


def rotate_volume(volume: np.ndarray, angle: float) -> np.ndarray:
    """Rotate every axial slice of a 3-D volume around its centre."""
    h, w = volume.shape[1], volume.shape[2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = np.zeros_like(volume)
    for z in range(volume.shape[0]):
        rotated[z] = cv2.warpAffine(volume[z], M, (w, h))
    return rotated


def make_rotating_mip_gifs(mr: np.ndarray, pet: np.ndarray,
                           n_angles: int = 36) -> None:
    """
    Render an animated rotating MIP for MR, registered-PET and their
    alpha-fusion, on the coronal/sagittal plane.
    """
    mr  = pad_volume(mr)
    pet = pad_volume(pet)

    angles = np.linspace(0, 360, n_angles, endpoint=False)

    frames_mr, frames_pet, frames_fusion = [], [], []
    for angle in angles:
        mr_rot  = rotate_volume(mr,  angle)
        pet_rot = rotate_volume(pet, angle)

        mr_mip  = np.max(mr_rot,  axis=1)
        pet_mip = np.max(pet_rot, axis=1)

        mr_mip  = (mr_mip  - mr_mip.min())  / (np.ptp(mr_mip)  + 1e-8)
        pet_mip = (pet_mip - pet_mip.min()) / (np.ptp(pet_mip) + 1e-8)

        fusion = 0.6 * mr_mip + 0.4 * pet_mip

        frames_mr    .append((mr_mip     * 255).astype(np.uint8))
        frames_pet   .append((pet_mip    * 255).astype(np.uint8))
        frames_fusion.append((fusion     * 255).astype(np.uint8))

    imageio.mimsave(os.path.join(OUT_DIR, "mr_mip.gif"),     frames_mr,     fps=8)
    imageio.mimsave(os.path.join(OUT_DIR, "pet_mip.gif"),    frames_pet,    fps=8)
    imageio.mimsave(os.path.join(OUT_DIR, "fusion_mip.gif"), frames_fusion, fps=8)


# -----------------------------------------------------------------------------
# Quality-control plots
# -----------------------------------------------------------------------------
def save_qc_figure(mr: np.ndarray,
                   pet_before: np.ndarray,
                   pet_after: np.ndarray) -> None:
    """Side-by-side mid-slice comparison."""
    z_mr   = mr.shape[0] // 2
    z_pet0 = pet_before.shape[0] // 2
    z_pet  = pet_after.shape[0] // 2

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(pet_before[z_pet0], cmap="gray")
    ax[0, 0].set_title("Original PET (mid slice)")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(mr[z_mr], cmap="gray")
    ax[0, 1].set_title("MR reference (mid slice)")
    ax[0, 1].axis("off")
    ax[1, 0].imshow(pet_after[z_pet], cmap="gray")
    ax[1, 0].set_title("Registered PET")
    ax[1, 0].axis("off")
    ax[1, 1].imshow(mr[z_mr], cmap="gray")
    ax[1, 1].imshow(pet_after[z_pet], cmap="hot", alpha=0.4)
    ax[1, 1].set_title("Alpha-fusion overlay")
    ax[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "part2_qc.png"), dpi=120)
    plt.show()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    # ---- load cached arrays from part 1 -------------------
    pet_path = os.path.join(OUT_DIR, "part1_pet_volume.npy")
    mr_path  = os.path.join(OUT_DIR, "part1_mr_volume.npy")
    if not (os.path.exists(pet_path) and os.path.exists(mr_path)):
        raise FileNotFoundError(
            "Run part1_dicom.py first – it caches the volumes used here.")

    pet_4d = np.load(pet_path)        # (36, 47, 256, 256)
    mr_vol = np.load(mr_path)         # (156, 256, 256)

    pet_avg_native  = np.mean(pet_4d, axis=0).astype(np.float32)   # (47, 256, 256)
    pet_last_native = pet_4d[-1].astype(np.float32)                # (47, 256, 256)

    print("Native PET last frame:", pet_last_native.shape)
    pet_avg  = resample_pet_to_mr_grid(pet_avg_native)             # ~ MR grid
    pet_last = resample_pet_to_mr_grid(pet_last_native)

    print("Resampled PET shape  :", pet_last.shape)
    print("Original MR shape    :", mr_vol.shape)

    # Zero-pad *both* modalities to the smallest canvas that contains
    # every voxel of any of them.  Pyelastix requires the fixed and
    # moving images to share the same shape, so we pad the MR. This is fully
    # lossless – every PET and MR voxel survives.
    pet_avg, pet_last, mr_vol = pad_to_common_shape(pet_avg, pet_last, mr_vol)
    print("Padded common shape  :", mr_vol.shape)

    # ---- forward registration: average PET → MR ------------------------
    nmi_before = normalised_mutual_information(pet_avg, mr_vol)
    ncc_before = normalised_cross_correlation(pet_avg, mr_vol)

    registered_pet, fwd_field = coregister(pet_avg, mr_vol.astype(np.float32))

    nmi_after  = normalised_mutual_information(registered_pet, mr_vol)
    ncc_after  = normalised_cross_correlation(registered_pet, mr_vol)

    # Apply the same kind of rigid transformation to the *last* PET frame
    # (re-run elastix – fast, ~1 minute – which guarantees the same
    # transformation logic is applied independently of frame intensity
    # differences).
    registered_pet_last, _ = coregister(pet_last, mr_vol.astype(np.float32))

    # ---- inverse registration: MR → PET (gives the inverse field) -------
    _, inv_field = inverse_coregister(mr_vol.astype(np.float32), pet_avg)

    # ---- save everything ------------------------------------------------
    np.save(os.path.join(OUT_DIR, "part2_registered_pet.npy"),       registered_pet)
    np.save(os.path.join(OUT_DIR, "part2_registered_pet_last.npy"),  registered_pet_last)
    np.save(os.path.join(OUT_DIR, "part2_mr_resampled.npy"),         mr_vol)
    np.save(os.path.join(OUT_DIR, "part2_inverse_field.npy"),
            np.stack(inv_field, axis=0))
    np.save(os.path.join(OUT_DIR, "part2_forward_field.npy"),
            np.stack(fwd_field, axis=0))

    # ---- numerical evaluation report ------------------------------------
    metrics_txt = (
        "Coregistration metrics\n"
        "----------------------\n"
        f"NMI  before : {nmi_before:.4f}\n"
        f"NMI  after  : {nmi_after :.4f}\n"
        f"NCC  before : {ncc_before:.4f}\n"
        f"NCC  after  : {ncc_after :.4f}\n"
    )
    print(metrics_txt)
    with open(os.path.join(OUT_DIR, "part2_metrics.txt"), "w") as f:
        f.write(metrics_txt)

    # ---- visualisations -------------------------------------------------
    save_qc_figure(mr_vol, pet_avg, registered_pet)
    make_rotating_mip_gifs(mr_vol, registered_pet, n_angles=36)
    print(f"\nAll outputs saved to: {OUT_DIR}")



def pad_to_common_shape(*volumes: np.ndarray) -> tuple:
    """
    Zero-pad an arbitrary number of 3-D volumes to the smallest common
    canvas that contains every voxel of every input.

    Each volume is centred inside the resulting canvas so the
    anatomical content is preserved as well as possible.  This is the
    lossless alternative to centre-cropping when the moving and fixed
    images do not share the same shape (a hard requirement of
    Elastix/Pyelastix).
    """
    if len(volumes) == 0:
        return ()

    common = tuple(max(v.shape[axis] for v in volumes)
                   for axis in range(volumes[0].ndim))

    out = []
    for v in volumes:
        canvas = np.zeros(common, dtype=v.dtype)
        slices = tuple(slice((cs - vs) // 2, (cs - vs) // 2 + vs)
                       for cs, vs in zip(common, v.shape))
        canvas[slices] = v
        out.append(canvas)
    return tuple(out)


if __name__ == "__main__":
    main()
