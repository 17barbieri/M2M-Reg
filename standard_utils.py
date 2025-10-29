import os
import sys
import time
import numpy as np
import nibabel as nib
import itk
from scipy.ndimage import affine_transform

def nib_load(path, dtype=np.float32, raw=True, orientation=('A', 'R', 'S'), library = ''):
    """
    Load a NIfTI image with NiBabel, reorient to a standard orientation,
    optionally flip axes and update the affine accordingly.

    Parameters
    ----------
    path : str
        Path to the NIfTI image.
    dtype : np.dtype
        Desired data type.
    raw : bool
        If True, return Nifti1Image object; else return (Nifti1Image, np.array).
    orientation : tuple
        Target orientation (default is ('A', 'R', 'S')).
    flip_axes : tuple
        Axes to flip (0=x, 1=y, 2=z).

    Returns
    -------
    volume : nib.Nifti1Image
        NiBabel image with corrected orientation and affine.
    data : np.array (optional)
        Numpy array of image data if raw=False.
    """
    volume = nib.load(path)
    
    # Reorient image
    wanted_orientation = nib.orientations.axcodes2ornt(orientation)
    current_orientation = nib.orientations.io_orientation(volume.affine)
    if not np.array_equal(current_orientation, wanted_orientation):
        transformation = nib.orientations.ornt_transform(current_orientation, wanted_orientation)
        volume = volume.as_reoriented(transformation)
    
    volume_data = volume.get_fdata().copy().astype(dtype)
    affine = volume.affine.copy()
    
    if library == 'ANTs':
        # Flip axes if requested
        for axis in (0,1,2):
            if axis < 0 or axis > 2:
                raise ValueError("flip_axes must be 0, 1, or 2")
            shift = (volume_data.shape[axis] - 1) * volume.affine[:3, axis]  # use original direction
            affine[:3, 3] += shift
            affine[:, axis] *= -1
            volume_data = np.flip(volume_data, axis=axis)
        affine[0]*=-1
        affine[1]*=-1
    
    # Create new NiBabel image with updated affine
    volume = nib.Nifti1Image(volume_data, affine, volume.header)
    
    if raw:
        return volume
    return volume, volume_data

from nibabel.processing import resample_from_to
def match_nii_images(fixed_nib, moving_nib, fixed_seg_nib=None, moving_seg_nib=None):
    """
    Match voxel size, affine, and shape of moving to fixed image.
    Optionally processes corresponding segmentations.
    """
    # --- 1. Compute resampling target ---
    target_affine = fixed_nib.affine
    target_shape = fixed_nib.shape
    target = (target_shape, target_affine)

    # --- 2. Resample moving image and segmentation into fixed space ---
    resampled_moving = resample_from_to(moving_nib, target, order=1)
    
    resampled_moving_seg = None
    if moving_seg_nib is not None:
        resampled_moving_seg = resample_from_to(moving_seg_nib, target, order=0)
        return fixed_nib, fixed_seg_nib, resampled_moving, resampled_moving_seg
    
    return fixed_nib, resampled_moving

def zscore_normalize(volume_nib: nib.nifti1.Nifti1Image, eps=1e-8) -> np.ndarray:
    """Apply zscore normalization and reshape data"""
    # compute mean / std only over non-zero voxels (common for brain MRI)
    volume = volume_nib.get_fdata().copy()
    mask = volume != 0
    if mask.sum() == 0:
        v = volume
        mean = v.mean()
        std = v.std()
    else:
        v = volume[mask]
        mean = v.mean()
        std = v.std()
    if std < eps:
        std = eps
    out = (volume - mean) / std

    # clip extreme values for stability
    out = np.clip(out, -5.0, 5.0)

    out = nib.Nifti1Image(out.astype(np.float32), volume_nib.affine, volume_nib.header)
    return out