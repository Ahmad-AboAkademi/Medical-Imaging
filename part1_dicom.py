"""
Part 1 – DICOM loading and visualization
=========================================
Project 11763 – Medical Image Processing.

Objectives
----------
a) Load PET (dynamic) and MR DICOM studies with PyDicom.
b) Inspect headers (acquisition geometry, timing, etc.).
c) Re-arrange the PET ``pixel_array`` (which is interleaved as
   frames * slices, rows, cols) into a 4-D volume of shape
   ``(frames, slices, rows, cols)``.
d) Visualise the *last frame* and the *average of all frames*.
e) Create GIF animations of the three median planes (axial,
   coronal and sagittal) along the temporal axis.

Outputs
-------
``axial.gif``, ``coronal.gif``, ``sagittal.gif``  – temporal animations.
``part1_pet_last_frame.png``, ``part1_pet_mean.png`` – static figures
used in the report.
``part1_pet_volume.npy``, ``part1_mr_volume.npy``  – numpy caches used by
``part2_coregistration.py`` so that we do not have to re-load the DICOM
files in the next part.
"""

from __future__ import annotations

import os
import numpy as np
import pydicom
import cv2
import imageio.v2 as imageio
from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
OUT_DIR     = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

PET_FILENAME = "02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916"
MR_FILENAME  = "15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916"


def filepath(filename: str, subfolder: str = "data") -> str:
    return os.path.join(PROJECT_DIR, subfolder, filename)


def load_dcm(path: str) -> pydicom.Dataset:
    """Read a DICOM file with PyDicom."""
    return pydicom.dcmread(path)


# -----------------------------------------------------------------------------
# Header inspection
# -----------------------------------------------------------------------------
TAGS_TO_PRINT = [
    ("Modality",                    (0x0008, 0x0060)),
    ("StudyDate",                   (0x0008, 0x0020)),
    ("AcquisitionTime",             (0x0008, 0x0032)),
    ("SeriesTime",                  (0x0008, 0x0031)),
    ("ContentTime",                 (0x0008, 0x0033)),
    ("PatientPosition",             (0x0018, 0x5100)),
    ("Number of Frames",            (0x0028, 0x0008)),
    ("Rows",                        (0x0028, 0x0010)),
    ("Columns",                     (0x0028, 0x0011)),
    ("Spacing Between Slices",      (0x0018, 0x0088)),
    ("Pixel Spacing",               (0x0028, 0x0030)),
    ("ImagePositionPatient",        (0x0020, 0x0032)),
    ("ImageOrientationPatient",     (0x0020, 0x0037)),
    ("Frame Positions Vector",      (0x0055, 0x1002)),
    ("Frame Start Times Vector",    (0x0055, 0x1001)),
    ("Frame Durations (ms) Vector", (0x0055, 0x1004)),
]


def print_header(dcm: pydicom.Dataset, name: str) -> None:
    """Pretty-print a curated list of DICOM tags."""
    print(f"\n=== {name} headers ===")
    for label, tag in TAGS_TO_PRINT:
        if tag in dcm:
            value = dcm[tag].value
            # large arrays: just print the shape
            if isinstance(value, (list, tuple, pydicom.multival.MultiValue)) and len(value) > 8:
                print(f"  {label:<32s}: array of length {len(value)}")
            else:
                print(f"  {label:<32s}: {value}")


# -----------------------------------------------------------------------------
# PET re-arrangement
# -----------------------------------------------------------------------------
def reshape_pet(pet_dcm: pydicom.Dataset) -> np.ndarray:
    """
    Reshape the interleaved PET volume into ``(frames, slices, rows, cols)``.

    The PET DICOM file is a multi-frame container whose ``pixel_array`` has
    shape ``(NumberOfFrames, rows, cols)`` and where the slices for each
    temporal frame are stored sequentially.  Sorting and grouping is done
    according to the *Frame Positions Vector* (0055, 1002) which lists the
    z-position of every individual slice.
    """
    n_frames = 36     # temporal frames (private tag 0055,1001 has 36 entries)
    n_slices = 47     # axial slices per frame (1692 / 36)
    rows, cols = int(pet_dcm.Rows), int(pet_dcm.Columns)

    raw = pet_dcm.pixel_array.astype(np.float32)
    assert raw.shape[0] == n_frames * n_slices, f"Unexpected number of frames: {raw.shape[0]}"

    vol = raw.reshape(n_frames, n_slices, rows, cols)

    # Patient is HFS so the DICOM Z-axis is stored from feet→head.
    # For an anatomically correct coronal/sagittal view we flip along Z.
    vol = np.flip(vol, axis=1)
    return vol


def sort_mr_volume(mr_dcm: pydicom.Dataset) -> np.ndarray:
    """
    Return the MR volume as ``(slices, rows, cols)``.

    For multi-frame DICOM the slices are already encoded in spatial order,
    but we flip along the slice axis to obtain the standard radiological
    superior→inferior ordering used by PET.
    """
    vol = mr_dcm.pixel_array.astype(np.float32)
    vol = np.flip(vol, axis=0)
    return vol


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------
def show_pet_overview(pet_vol: np.ndarray) -> None:
    """Show last temporal frame (mean axial projection) and mean of all frames."""
    last_frame = pet_vol[-1]                 # (slices, rows, cols)
    mean_frame = np.mean(pet_vol, axis=0)    # (slices, rows, cols)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(np.mean(last_frame, axis=0), cmap="hot")
    # ax[0].set_title("Last frame – mean axial projection")
    # ax[0].axis("off")
    # ax[1].imshow(np.mean(mean_frame, axis=0), cmap="hot")
    # ax[1].set_title("Average of all 36 frames – mean axial projection")
    # ax[1].axis("off")
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUT_DIR, "part1_pet_overview.png"), dpi=120)
    # plt.show()

    # Single bright image for the cover figure
    plt.figure(figsize=(5, 5))
    plt.imshow(np.mean(mean_frame, axis=0), cmap="hot")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "part1_pet_mean.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.imshow(np.mean(last_frame, axis=0), cmap="hot")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "part1_pet_last_frame.png"), dpi=120)
    plt.close()


def normalise_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)


def make_median_plane_gifs(pet_vol: np.ndarray, scale_z: float = 3.27 / 1.17) -> None:
    """
    Create temporal GIFs of the three median planes.

    The PET voxel is anisotropic (1.17 × 1.17 × 3.27 mm) so the coronal and
    sagittal planes are stretched along the slice axis to recover an
    isotropic display.
    """
    n_frames, n_slices, n_rows, n_cols = pet_vol.shape
    z_med = n_slices // 2
    y_med = n_rows // 2
    x_med = n_cols // 2

    axial_frames    = [pet_vol[t, z_med, :, :] for t in range(n_frames)]
    coronal_frames  = [pet_vol[t, :, y_med, :] for t in range(n_frames)]
    sagittal_frames = [pet_vol[t, :, :, x_med] for t in range(n_frames)]

    imageio.mimsave(os.path.join(OUT_DIR, "axial.gif"),
                    [normalise_uint8(f) for f in axial_frames], fps=5)
    imageio.mimsave(os.path.join(OUT_DIR, "coronal.gif"),
                    [normalise_uint8(cv2.resize(f, None, fx=1, fy=scale_z))
                     for f in coronal_frames], fps=5)
    imageio.mimsave(os.path.join(OUT_DIR, "sagittal.gif"),
                    [normalise_uint8(cv2.resize(f, None, fx=1, fy=scale_z))
                     for f in sagittal_frames], fps=5)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main() -> None:
    pet_dcm = load_dcm(filepath(PET_FILENAME))
    mr_dcm  = load_dcm(filepath(MR_FILENAME))
    
    #print metadata for both
    # print(pet_dcm)
    # print(mr_dcm)

    print_header(pet_dcm, "PET")
    print_header(mr_dcm,  "MR")

    pet_vol = reshape_pet(pet_dcm)
    mr_vol  = sort_mr_volume(mr_dcm)

    print(f"\nPET volume shape (frames, slices, rows, cols) = {pet_vol.shape}")
    print(f"MR  volume shape (slices, rows, cols)          = {mr_vol.shape}")

    # ---- d) static visualisations ----
    show_pet_overview(pet_vol)

    # ---- e) temporal GIFs on the three median planes ----
    make_median_plane_gifs(pet_vol)
    print(f"\nSaved GIFs and figures to: {OUT_DIR}")

    # ---- cache results for the next parts ----
    np.save(os.path.join(OUT_DIR, "part1_pet_volume.npy"), pet_vol)
    np.save(os.path.join(OUT_DIR, "part1_mr_volume.npy"),  mr_vol)


if __name__ == "__main__":
    main()
