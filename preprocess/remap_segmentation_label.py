import os
import tqdm
import numpy as np
import nibabel as nib

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



data_dir = '~/M2M-Reg/datasets/ADNI_preprocessed'

for pid in tqdm.tqdm(os.listdir(data_dir)):
    pid_dir = os.path.join(data_dir, pid)
    if not os.path.isdir(pid_dir):
        continue
    seg_path = os.path.join(pid_dir, f"{pid}_preprocessed_seg.nii")
    out_seg_path = os.path.join(pid_dir, f"{pid}_preprocessed_seg_map_easy.nii")

    seg_nii = nib.load(seg_path)
    seg_np = seg_nii.get_fdata().astype(np.int16)
    seg_np = np.nan_to_num(seg_np)
    seg_map = np.zeros_like(seg_np, dtype=np.int16)
    seg_map = ROI_mapping[seg_np]

    seg_nii.header['bitpix'] = 16
    seg_nii.header['scl_slope'] = 1
    seg_nii.header['scl_inter'] = 0
    save_nii = type(seg_nii)(seg_map, \
                                    affine=seg_nii.affine, \
                                    header=seg_nii.header, \
                                    extra=seg_nii.extra,   \
                                    file_map=seg_nii.file_map)
    nib.save(save_nii, out_seg_path)