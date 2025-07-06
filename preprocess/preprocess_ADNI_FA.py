import os
import time
import logging
import numpy as np
import nibabel as nib
import torchio as tio
import pandas as pd
import argparse

ROI_mapping = np.zeros(3000, dtype=np.int16)
ROI_mapping[1000:] = 2
ROI_mapping[2] = 1     # "Cerebral-White-Matter(lh)",
ROI_mapping[3] = 2     # "Cerebral-Cortex(lh)",
ROI_mapping[4] = 3     # "Lateral-Ventricle(lh)",
ROI_mapping[5] = 3     # "Inf-Lat-Vent(lh)",
ROI_mapping[7] = 4     # "Cerebellum-White-Matter(lh)",
ROI_mapping[8] = 5     # "Cerebellum-Cortex(lh)",
ROI_mapping[10] = 6    # "Thalamus-Proper(lh)",
ROI_mapping[11] = 7    # "Caudate(lh)",
ROI_mapping[12] = 8    # "Putamen(lh)",
ROI_mapping[13] = 1    # "Pallidum(lh)",
ROI_mapping[14] = 3    # "3rd-Ventricle",
ROI_mapping[15] = 3    # "4th-Ventricle",
ROI_mapping[16] = 9    # "Brain-Stem",
ROI_mapping[17] = 10    # "Hippocampus(lh)",
ROI_mapping[18] = 11    # "Amygdala(lh)",
ROI_mapping[24] = 0    # "CSF",
ROI_mapping[26] = 7    # "Accumbens-area(lh)",
ROI_mapping[28] = 1    # "VentralDC(lh)",
ROI_mapping[31] = 3    # "choroid-plexus(lh)",
ROI_mapping[41] = 1    # "Cerebral-White-Matter(rh)",
ROI_mapping[42] = 2    # "Cerebral-Cortex(rh)",
ROI_mapping[43] = 3    # "Lateral-Ventricle(rh)",
ROI_mapping[44] = 3    # "Inf-Lat-Vent(rh)",
ROI_mapping[46] = 4    # "Cerebellum-White-Matter(rh)",
ROI_mapping[47] = 5    # "Cerebellum-Cortex(rh)",
ROI_mapping[49] = 6    # "Thalamus-Proper(rh)",
ROI_mapping[50] = 7    # "Caudate(rh)",
ROI_mapping[51] = 8    # "Putamen(rh)",
ROI_mapping[52] = 1    # "Pallidum(rh)",
ROI_mapping[53] = 10    # "Hippocampus(rh)",
ROI_mapping[54] = 11    # "Amygdala(rh)",
ROI_mapping[58] = 7    # "Accumbens-area(rh)",
ROI_mapping[60] = 1    # "VentralDC(rh)",
ROI_mapping[63] = 3    # "choroid-plexus(rh)",
ROI_mapping[72] = 3    # "5th-Ventricle",
ROI_mapping[77] = 1    # "WM-hypointensities",
ROI_mapping[173] = 9    # brainstem
ROI_mapping[174] = 9    # brainstem
ROI_mapping[175] = 9    # brainstem


def map_image(img, out_affine, out_shape, ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
              order=1):
    """
    Function to map image to new voxel space (RAS orientation)

    :param nibabel.MGHImage img: the src 3D image with data and affine set
    :param np.ndarray out_affine: trg image affine
    :param np.ndarray out_shape: the trg shape information
    :param np.ndarray ras2ras: ras2ras an additional maping that should be applied (default=id to just reslice)
    :param int order: order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: np.ndarray new_data: mapped image data array
    """
    from scipy.ndimage import affine_transform
    from numpy.linalg import inv

    # compute vox2vox from src to trg
    vox2vox = inv(out_affine) @ ras2ras @ img.affine

    # here we apply the inverse vox2vox (to pull back the src info to the target image)
    image_data = np.asanyarray(img.dataobj)
    # convert frames to single image
    if len(image_data.shape) > 3:
        if any(s != 1 for s in image_data.shape[3:]):
            raise ValueError(f'Multiple input frames {tuple(image_data.shape)} not supported!')
        image_data = np.squeeze(image_data, axis=tuple(range(3,len(image_data.shape))))

    new_data = affine_transform(image_data, inv(vox2vox), output_shape=out_shape, order=order)
    return new_data, vox2vox

def resample_nifti(src_path, dst_path, spacing):
    orig_nii = tio.Subject(t1=tio.ScalarImage(src_path))
    transform = tio.Resample(spacing)
    resampled_img = transform(orig_nii)
    resampled_img.t1.save(dst_path)
    print("Write file :", dst_path)

import torchio as tio
import nibabel as nib

def resample_with_mask(img_path, mask_path, out_img_path, out_mask_path, spacing):
    # Load MRI and segmentation mask using torchio
    mri = tio.ScalarImage(img_path)
    mask = tio.LabelMap(mask_path)  # LabelMap for discrete segmentation mask

    # Resample mask to match MRI resolution
    resampler = tio.Resample(spacing)  # Use MRI as the reference
    resampled_mri = resampler(mri)  # Resample mri
    resampled_mask = resampler(mask)  # Resample mask

    # Save resampled images
    resampled_mri.save(out_img_path)
    resampled_mask.save(out_mask_path)
    
    print(f"Resampled MRI saved to: {out_img_path}")
    print(f"Resampled Mask saved to: {out_mask_path}")
    
    
def conform(source_path, target_path, save_files, order=1):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.

    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader
    img = nib.load(source_path)
    cwidth = 256
    csize = 1
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([cwidth, cwidth, cwidth, 1])
    h1.set_zooms([csize, csize, csize])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = cwidth
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # get scale for conversion on original input before mapping to be more similar to mri_convert
    mapped_data, orig_to_256 = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)

    new_img = nib.MGHImage(mapped_data, h1.get_affine(), h1)
    if save_files:
        nib.save(new_img, target_path)
        print("Write file :", target_path)
    
    # return nib.load(new_img, mgh_format='mgh')
    return type(img)(mapped_data, affine=h1.get_affine(), header=h1)

def pad_and_crop_to_target(array, target_shape=(128, 128, 128), pad_value=0):
    """
    Apply symmetric padding and cropping to an array to make it match the target shape.

    Parameters:
    array (np.ndarray): The input 3D array.
    target_shape (tuple): The desired shape (x, y, z).
    pad_value (int or float): The value to use for padding.

    Returns:
    np.ndarray: The padded and/or cropped array.
    """
    # Calculate the padding required for each dimension
    padding = [(max(0, target - size) // 2, max(0, target - size) - max(0, target - size) // 2) for size, target in zip(array.shape, target_shape)]
    
    # Apply padding
    padded_array = np.pad(array, padding, mode='constant', constant_values=pad_value)

    # Calculate the cropping required for each dimension
    cropping = [(max(0, size - target) // 2, max(0, size - target) - max(0, size - target) // 2) for size, target in zip(padded_array.shape, target_shape)]

    # Apply cropping
    cropped_array = padded_array[
        cropping[0][0]:padded_array.shape[0]-cropping[0][1],
        cropping[1][0]:padded_array.shape[1]-cropping[1][1],
        cropping[2][0]:padded_array.shape[2]-cropping[2][1]
    ]

    return cropped_array

def percentile_minmax_normalization(image, min_percentile=0, max_percentile=99):
    """
    이미지의 하위 min_percentile 및 상위 max_percentile 값을 사용하여 min-max 정규화 수행
    """
    if min_percentile == 0:
        min_val = np.min(image)
    else:
        min_val = np.percentile(image, min_percentile)
    max_val = np.percentile(image, max_percentile)

    # 예외 처리: max_val이 min_val보다 작거나 같으면 정규화 수행 안 함
    if max_val <= min_val:
        return image
    logging.info(f"min: {min_val}, max: {max_val}")
    # Min-Max 정규화
    norm_image = (image - min_val) / (max_val - min_val)
    norm_image = np.clip(norm_image, 0, 1)  # 값 범위를 [0,1]로 제한
    return norm_image

def skull_strip(resampled_nifti_path, strip_path, mask_path):
    time.sleep(1)
    while not os.path.exists(resampled_nifti_path):
        time.sleep(1)

    inst = f'bash /usr/local/freesurfer/7.4.0/bin/mri_synthstrip -i {resampled_nifti_path} -o {strip_path} -m {mask_path}'
    os.system(inst)
    
    time.sleep(1)
    while not os.path.exists(mask_path):
        time.sleep(1)
    if os.path.exists(strip_path):
        os.remove(strip_path)

def crop_with_mask(resampled_nifti_path, mask_path, preprocessed_nifti_path, target_shape, patient_id, img_type, pons=None):
    resampled_nii = nib.load(resampled_nifti_path)
    resampled_np = resampled_nii.get_fdata()
    
    mask_np = nib.load(mask_path).get_fdata()
    mask_np = mask_np.astype(np.int16)
    
    if img_type == 'MR':
        resampled_np = resampled_np.astype(np.uint8)
    elif img_type == 'seg':
        resampled_np = resampled_np.astype(np.int16)

    coords = np.argwhere(mask_np > 0)
    start = coords.min(axis=0)
    end = coords.max(axis=0) + 1  # 슬라이스는 종료 인덱스에서 하나 더 크게 설정

    # 뇌 마스크의 크기 계산
    brain_size = end - start

    # 뇌 마스크의 중심 계산
    center = (start + end) // 2

    # 원하는 크기의 절반 계산
    half_size = np.array(target_shape) // 2  # [48, 48, 48]

    # 새로운 시작과 종료 인덱스 계산
    new_start = center - half_size
    new_end = center + half_size

    # 이미지 범위를 벗어나지 않도록 조정
    for i in range(3):
        if new_start[i] < 0:
            new_start[i] = 0
            new_end[i] = target_shape[i]
        if new_end[i] > resampled_np.shape[i]:
            new_end[i] = resampled_np.shape[i]
            new_start[i] = resampled_np.shape[i] - target_shape[i]
            if new_start[i] < 0:
                new_start[i] = 0  # 시작 인덱스는 음수가 될 수 없음

    # 이미지 자르기
    cropped_np = resampled_np[new_start[0]:new_end[0], new_start[1]:new_end[1], new_start[2]:new_end[2]]

    # 실제 자른 이미지의 크기
    actual_shape = cropped_np.shape

    # 원하는 크기보다 작은 경우 패딩 추가
    if actual_shape != target_shape:
        padding = []
        for i in range(3):
            pad_before = max(0, -new_start[i])
            pad_after = target_shape[i] - actual_shape[i] - pad_before
            padding.append((pad_before, pad_after))
        cropped_np = np.pad(cropped_np, padding, mode='constant', constant_values=0)

    # 뇌 마스크 크기가 원하는 크기를 초과하는 경우 경고 출력
    if any(brain_size[i] > target_shape[i] for i in range(3)):
        logging.warning(f"Warning: Brain size {brain_size} exceeds target shape {target_shape} along some axes for {patient_id}")
        raise ValueError("Brain size exceeds target shape")

    # if os.path.exists(mask_path):
    #     os.remove(mask_path)
    cropped_np = np.nan_to_num(cropped_np)
    if img_type == 'PET':
        cropped_np = np.clip(cropped_np, 0, None)
        cropped_np = percentile_minmax_normalization(cropped_np, min_percentile=0, max_percentile=99.9)
        # cropped_np = np.clip(cropped_np / pons, 0, 2.5)
        bitpix = 32
    elif img_type == 'MR':
        bitpix = 8
    elif img_type == 'seg':
        bitpix = 16

    resampled_nii.header['bitpix'] = bitpix
    resampled_nii.header['scl_slope'] = 1
    resampled_nii.header['scl_inter'] = 0
    save_nii = type(resampled_nii)(cropped_np, \
                                    affine=resampled_nii.affine, \
                                    header=resampled_nii.header, \
                                    extra=resampled_nii.extra,   \
                                    file_map=resampled_nii.file_map)
    nib.save(save_nii, preprocessed_nifti_path)
    
    if img_type == 'seg':
        preprocessed_seg_map_path = os.path.join(os.path.dirname(preprocessed_nifti_path), f"{patient_id}_preprocessed_seg_map.nii")
        seg_map = np.zeros_like(cropped_np, dtype=np.int16)
        seg_map = ROI_mapping[cropped_np]
    
        save_nii = type(resampled_nii)(seg_map, \
                                        affine=resampled_nii.affine, \
                                        header=resampled_nii.header, \
                                        extra=resampled_nii.extra,   \
                                        file_map=resampled_nii.file_map)
        nib.save(save_nii, preprocessed_seg_map_path)

    return cropped_np

    
if __name__ == "__main__":
    total_start_time = time.time()
    
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=str, default='~/M2M-Reg/datasets/ADNI_nifti')
    args.add_argument('--out_data_dir', type=str, default='~/M2M-Reg/datasets/ADNI_preprocessed')
    # args.add_argument('--log_path', type=str, default='~/M2M-Reg/datasets/ADNI_preprocessed')
    args.add_argument('--target_shape', type=int, default=128)
    args.add_argument('--target_voxel', type=float, default=1.5)
    
    args = args.parse_args()
    
    data_dir = args.data_dir
    out_data_dir = args.out_data_dir
    os.makedirs(out_data_dir, exist_ok=True)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(f"{out_data_dir}/preprocess_ADNI.log"),
            logging.StreamHandler()
        ]
    )

    target_shape = (args.target_shape, args.target_shape, args.target_shape)
    target_voxel = (args.target_voxel, args.target_voxel, args.target_voxel)

    error_list = []
    value_error_list = []
    for pid in os.listdir(out_data_dir):
        try:
        # if True:
            iteration_start_time = time.time()
            if not os.path.isdir(os.path.join(data_dir, pid)):
                logging.info(f"{pid} is not a directory.")
                continue
            pid_dir = os.path.join(out_data_dir, pid)

            PET_256_path = os.path.join(pid_dir, f"256_PET.nii")
            MR_256_path = os.path.join(pid_dir, f"256_MR.nii")
            seg_256_path = os.path.join(pid_dir, f"seg_MR.nii")
            
            PET_resampled_path = os.path.join(pid_dir, f"resampled_PT.nii")
            MR_resampled_path = os.path.join(pid_dir, f"resampled_MR.nii")
            seg_resampled_path = os.path.join(pid_dir, f"resampled_seg.nii")
            resample_nifti(PET_256_path, PET_resampled_path, spacing=target_voxel)
            resample_with_mask(MR_256_path, seg_256_path, MR_resampled_path, seg_resampled_path, spacing=target_voxel)


            PET_prprocessed_path = os.path.join(pid_dir, f"{pid}_preprocessed_PT.nii")
            MR_prprocessed_path = os.path.join(pid_dir, f"{pid}_preprocessed_MR.nii")
            seg_prprocessed_path = os.path.join(pid_dir, f"{pid}_preprocessed_seg.nii")
            SUVR_csv_path = os.path.join(pid_dir, "results.csv")
            df_suvr = pd.read_csv(SUVR_csv_path)
            pons_average_value = df_suvr.loc[df_suvr["no."] == "ROI_average", "174"].values[0]
            if pons_average_value <= 10:
                value_error_list.append(pid)
            logging.info(f"The value in the '174' column for 'ROI_average' row is: {pons_average_value}")

            crop_with_mask(PET_resampled_path, seg_resampled_path, PET_prprocessed_path, target_shape, pid, img_type='PET', pons=pons_average_value)
            crop_with_mask(MR_resampled_path, seg_resampled_path, MR_prprocessed_path, target_shape, pid, img_type='MR')
            crop_with_mask(seg_resampled_path, seg_resampled_path, seg_prprocessed_path, target_shape, pid, img_type='seg')

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt")
            break
        except Exception as e:
            logging.error(f"Error occurred at {pid}: {e}")
            error_list.append(pid)
            continue
        finally:
            iteration_end_time = time.time()
            iteration_elapsed_time = iteration_end_time - iteration_start_time
            logging.info(f"Processed {pid} in {iteration_elapsed_time:.2f} seconds.")


    total_end_time = time.time()  # 전체 프로세스 종료 시간
    total_elapsed_time = total_end_time - total_start_time
    logging.info(f"All process done. Total time: {total_elapsed_time:.2f} seconds.")
    logging.info(f"Number of errors: {len(error_list)}")
    logging.info(f"Error list: {error_list}")
    logging.info(f"Value Error list: {value_error_list}")