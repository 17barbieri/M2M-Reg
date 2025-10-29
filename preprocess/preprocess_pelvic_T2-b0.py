#%%
import os
import sys
import argparse
import glob
import time
import logging
import h5py
import numpy as np
from tqdm import tqdm
sys.path.append('..')
from standard_utils import *
import matplotlib.pyplot as plt
#%%
def pad_or_crop(img, target_shape):
    """Pad or center-crop a 3D numpy array to target_shape."""
    current_shape = np.array(img.shape)
    target_shape = np.array(target_shape)
    diff = target_shape - current_shape

    pad_before = np.maximum(diff // 2, 0)
    pad_after = np.maximum(diff - pad_before, 0)
    crop_before = np.maximum(-diff // 2, 0)
    crop_after = np.maximum(-(diff - (-diff // 2)), 0)

    img_cropped = img[
        crop_before[0]:img.shape[0]-crop_after[0] if crop_after[0] > 0 else None,
        crop_before[1]:img.shape[1]-crop_after[1] if crop_after[1] > 0 else None,
        crop_before[2]:img.shape[2]-crop_after[2] if crop_after[2] > 0 else None,
    ]
    img_padded = np.pad(
        img_cropped,
        [(pad_before[i], pad_after[i]) for i in range(3)],
        mode='constant'
    )
    return img_padded

def preprocess_and_save(data_dir, out_path, considered_patient_ids = [], target_shape = (256, 256, 100), exclude_list = []):
    """
    Convert paired T2/b0 images into an HDF5 dataset compatible with M2M-Reg.
    Output structure matches generate_hdf5_from_sheet:
        T2_dataset: moving (T2)
        b0_dataset: fixed (b0)
        T2_seg_dataset: segmentation (T2_seg)
        PatientID: subject IDs
    """
    start_t = time.time()
    subjects = sorted(glob.glob(os.path.join(data_dir, '**')))
    if considered_patient_ids == []:
        considered_patient_ids = [os.path.basename(p).split('_b0')[0] for p in glob.glob(os.path.join(data_dir, '**/*b0.nii.gz'))]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # For logging
    log_path = os.path.join(os.path.dirname(out_path),
                            f"hdf5_pelvis_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s:%(message)s',
                        handlers=[logging.FileHandler(log_path),
                                  logging.StreamHandler()])
    logging.info(f"Found {len(subjects)} subjects.")

    T2_dataset, b0_dataset, T2_seg_dataset, b0_seg_dataset, subject_ids = [], [], [], [], []

    for subj_path in tqdm(subjects, desc="Processing subjects"):
        subj = os.path.basename(subj_path)
        if subj in exclude_list:
            print(f'Excluded: {subj}')
            continue
        if subj not in considered_patient_ids:
            continue

        b0_path = os.path.join(subj_path, f"{subj}_b0.nii.gz")
        b0_seg_path = os.path.join(subj_path, f"{subj}_b0_seg.nii.gz")
        T2_path = os.path.join(subj_path, f"{subj}_coroT2Cube.nii.gz")
        T2_seg_path = os.path.join(subj_path, f"{subj}_Segmentation.nii.gz")

        if not (os.path.exists(T2_path) and os.path.exists(b0_path)):
            logging.warning(f"[{subj}] Missing T2 or b0 image, skipping.")
            continue

        try:
            # Load
            b0_nib = nib_load(b0_path)
            b0_seg_nib = nib_load(b0_seg_path) if os.path.exists(b0_seg_path) else None
            T2_nib = nib_load(T2_path)
            T2_seg_nib = nib_load(T2_seg_path) if os.path.exists(T2_seg_path) else None
            

            # Match orientations, voxel size, and FOV
            _, _, T2_matched, T2_seg_matched = match_nii_images(
                b0_nib, T2_nib, moving_seg_nib=T2_seg_nib
            )

            # Normalize
            b0_preproc = zscore_normalize(b0_nib)
            T2_preproc = zscore_normalize(T2_matched)

            # fig, ax = plt.subplots(2, 2)
            # z = b0_preproc.shape[2]//2
            # ax = ax.flatten()
            # ax[0].imshow(b0_preproc.get_fdata()[:,:,z], cmap = plt.cm.Greys_r)
            # # ax[1].imshow(b0_preproc[:,:,z], cmap = plt.cm.Greys_r)
            # ax[2].imshow(T2_preproc.get_fdata()[:,:,z], cmap = plt.cm.Greys_r)
            # ax[3].imshow(T2_seg_nib.get_fdata()[:,:,z], cmap = plt.cm.Greys_r)
            # os.makedirs('./figures', exist_ok = True)
            # plt.savefig(f'./figures/{subj}.png', bbox_inches='tight', dpi = 300)

            # Convert to arrays (force consistent orientation)
            T2 = np.nan_to_num(T2_preproc.get_fdata()).astype(np.float32)
            b0 = np.nan_to_num(b0_preproc.get_fdata()).astype(np.float32)
            T2_seg = np.nan_to_num(T2_seg_matched.get_fdata()).astype(np.int16)
            b0_seg = (
                np.nan_to_num(b0_seg_nib.get_fdata()).astype(np.int16)
                if b0_seg_nib
                else np.zeros_like(b0, dtype=np.int16)
            )

            # --- Enforce consistent size ---
            T2 = pad_or_crop(T2, target_shape)
            b0 = pad_or_crop(b0, target_shape)
            T2_seg = pad_or_crop(T2_seg, target_shape)
            b0_seg = pad_or_crop(b0_seg, target_shape)

            # Final shape sanity check
            if not (T2.shape == b0.shape == T2_seg.shape == target_shape):
                logging.warning(f"[{subj}] Shape mismatch after pad/crop: "
                                f"T2={T2.shape}, b0={b0.shape}, seg={T2_seg.shape}")
                continue

            # Append
            T2_dataset.append(T2)
            b0_dataset.append(b0)
            T2_seg_dataset.append(T2_seg)
            b0_seg_dataset.append(b0_seg)
            subject_ids.append(subj)

        except Exception as e:
            logging.error(f"Error processing {subj}: {e}")
            continue

    # Convert lists to arrays
    T2_dataset = np.asarray(T2_dataset, dtype=np.float32)
    b0_dataset = np.asarray(b0_dataset, dtype=np.float32)
    T2_seg_dataset = np.asarray(T2_seg_dataset, dtype=np.int16)
    b0_seg_dataset = np.asarray(b0_seg_dataset, dtype=np.int16)

    # logging.info(f"Shapes — T2: {T2_dataset.shape}, b0: {b0_dataset.shape}, T2_seg: {seg_dataset.shape}")

    # Save to HDF5 (flat structure)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("T2_dataset", data=T2_dataset, dtype=np.float32, compression="gzip")
        f.create_dataset("b0_dataset", data=b0_dataset, dtype=np.float32, compression="gzip")
        f.create_dataset("T2_seg_dataset", data=T2_seg_dataset, dtype=np.int16, compression="gzip")
        f.create_dataset("b0_seg_dataset", data=b0_seg_dataset, dtype=np.int16, compression="gzip")

        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("PatientID", data=np.array(subject_ids, dtype=object), dtype=dt, compression="gzip")

    elapsed = time.time() - start_t
    logging.info(f"✅ Preprocessing complete. Saved {len(subject_ids)} subjects to {out_path} in {elapsed:.2f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess pelvic T2→b0 dataset for M2M-Reg (HDF5 flat format).")
    parser.add_argument("--data_dir", type=str, required=False, default="../datasets/T2-b0_nifti",
                        help="Path to folder containing patient subfolders.")
    parser.add_argument("--out_dir", type=str, required=False, default="../datasets",
                        help="Output HDF5 file path.")
    args = parser.parse_args()

    # Exclude cases because they are not in the same orientation
    exclude_list = ['1-2-43_190114', '1-4-14_160822', '1-4-15_151130', '1-4-15_160829', '1-4-37_201116']

    # Identify training and validation data
    val_patient_ids = [os.path.basename(p).split('_b0_seg')[0] for p in glob.glob(os.path.join(args.data_dir, '**/*b0_seg.nii.gz'))]
    
    train_patient_ids = [os.path.basename(p).split('_b0')[0] for p in glob.glob(os.path.join(args.data_dir, '**/*b0.nii.gz'))]
    train_patient_ids = [x for x in train_patient_ids if x not in val_patient_ids]

    preprocess_and_save(args.data_dir, out_path = os.path.join(args.out_dir, 'pelvic_T2-b0_train.hdf5'), 
                        considered_patient_ids = train_patient_ids, exclude_list=exclude_list)
    preprocess_and_save(args.data_dir, out_path = os.path.join(args.out_dir, 'pelvic_T2-b0_val.hdf5'), 
                        considered_patient_ids = val_patient_ids, exclude_list=exclude_list)
