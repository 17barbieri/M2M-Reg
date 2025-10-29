#%%
import os
import pandas as pd

path = '/mnt/c/Users/IMAG2/Documents/MATTEO/Data/'

patient_ids = [p.split('_b0')[0] for p in os.listdir(os.path.join(path, 'b0'))]

print(patient_ids)
dset = pd.DataFrame({'PatientID': patient_ids})

dset['exclude'] = ['']*len(dset)
dset['set'] = ['train']*len(dset)

dset.to_csv('./T2-b0.csv')

# Create csv files
import glob
data_root = '/mnt/c/Users/IMAG2/Documents/MATTEO/Data/'
b0_files = glob.glob(os.path.join(data_root, 'b0', '*b0.nii.gz'))
b0_seg_files = glob.glob(os.path.join(data_root, 'b0_seg_corrected_labels/*b0_seg.nii.gz'))
all_patient_codes = [os.path.basename(p).split('_b0')[0] for p in b0_files]
val_patient_codes = [os.path.basename(p).split('_b0')[0] for p in b0_seg_files]
train_dset_path = './dset_train.csv'
val_dset_path = './dset_val.csv'

train_dset = {'Patient code': [], 'Fixed path': [], 'Moving path': []}
val_dset = {'Patient code': [],
            'Fixed path': [], 'Fixed seg path': [],
            'Moving path': [], 'Moving seg path': []}

for patient_code in all_patient_codes:
    if patient_code in val_patient_codes:
        val_dset['Patient code'].append(patient_code)
        val_dset['Fixed path'].append(glob.glob(os.path.join(data_root, 'b0', f'{patient_code}_b0.nii.gz'))[0])
        val_dset['Fixed seg path'].append(glob.glob(os.path.join(data_root, 'b0_seg_corrected_labels', f'{patient_code}_b0_seg.nii.gz'))[0])
        val_dset['Moving path'].append(glob.glob(os.path.join(data_root, 'T2', f'{patient_code}_coroT2cube.nii.gz'))[0])
        val_dset['Moving seg path'].append(glob.glob(os.path.join(data_root, 'T2_seg', f'{patient_code}_segmentation.nii.gz'))[0])
    else:
        train_dset['Patient code'].append(patient_code)
        train_dset['Fixed path'].append(glob.glob(os.path.join(data_root, 'b0', f'{patient_code}_b0.nii.gz'))[0])
        train_dset['Moving path'].append(glob.glob(os.path.join(data_root, 'T2', f'{patient_code}_coroT2cube.nii.gz'))[0])
os.makedirs(os.path.dirname(train_dset_path), exist_ok = True)
pd.DataFrame(train_dset).to_csv(train_dset_path, index = False)
pd.DataFrame(val_dset).to_csv(val_dset_path, index = False)
#%%