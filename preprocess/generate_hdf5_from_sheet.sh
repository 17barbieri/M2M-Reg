#!/bin/bash


cd ~/M2M-Reg/preprocess


# dataset="ADNI_FA"
dataset="ADNI_FBB"
for which_set in "train_val" "test"
do
python -u generate_hdf5_from_sheet.py \
    --sheet_url "https://docs.google.com/spreadsheets/d/1y6kyPNWaAW9uehGHF0nErsn1Q2_rmapK39wRdIBRR8M/edit?usp=sharing" \
    --hdf5_path "~/M2M-Reg/datasets/${dataset}_${which_set}.hdf5" \
    --dataset $dataset \
    --which_set $which_set \
    --data_dir "~/M2M-Reg/datasets/${dataset}_preprocessed" \
    --PT_name "preprocessed_PT.nii" \
    --ST_name "preprocessed_MR.nii" \
    --seg_name "preprocessed_seg_map.nii"
    # --seg_name "preprocessed_seg_map_easy.nii"
done
