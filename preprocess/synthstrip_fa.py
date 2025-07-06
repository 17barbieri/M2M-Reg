import os

base_path = '/nas/chu/ADNI_FA_paired'

which = 'MR'
for pid in os.listdir(base_path):
    pid_dir = os.path.join(base_path, pid)
    FA_path = os.path.join(pid_dir, f'{pid}_{which}.nii.gz')
    
    strip_path = os.path.join(pid_dir, f'{pid}_striped_{which}.nii.gz')
    mask_path = os.path.join(pid_dir, f'{pid}_mask_{which}.nii.gz')
    inst = f'bash /usr/local/freesurfer/7.4.0/bin/mri_synthstrip -i {FA_path} -o {strip_path} -m {mask_path}'
    print(inst)
    os.system(inst)