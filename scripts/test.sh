#!/bin/bash

cd ~/M2M-Reg

date=$(date +"%y%m%d-%H%M%S")

gpu_ids=0
dataset="ADNI_FBB"
model_path="~/M2M-Reg/results/ADNI/250212-004737-ADNI-cano_1-data_1000-b8-lr0.00005-Linv_1.0-Lcan_0.001/250212-004737-ADNI-cano_1-data_1000-b8-lr0.00005-Linv_1.0-Lcan_0.001/checkpoints/network_weights_e305_iter38250"
model="gradicon"

python -u ~/M2M-Reg/scripts/test.py \
    --out_dir ~/M2M-Reg/results_test_gradicon \
    --data_path ~/M2M-Reg/datasets/${dataset}_preprocessed \
    --model_path $model_path \
    --model $model \
    --dataset $dataset \
    --gpu_ids $gpu_ids \
