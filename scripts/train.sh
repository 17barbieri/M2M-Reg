#!/bin/bash

cd ~/M2M-Reg

date=$(date +"%y%m%d-%H%M%S")

gpu_ids="0"
dataset="ADNI_FBB"

epoch=1600
batch=8
data_num=1000
lr=0.00005

num_cano="-1"
lambda_inv=0.5
lambda_can=0.1

model="gradicon"
exp_name="${model}-${date}-${dataset}-cano_${num_cano}-data_${data_num}-b${batch}-lr${lr}-Linv_${lambda_inv}-Lcan_${lambda_can}"
python -u ~/M2M-Reg/scripts/train_multi.py \
    --exp_name $exp_name \
    --dataset $dataset \
    --epoch $epoch \
    --batch $batch \
    --gpu_ids $gpu_ids \
    --data_num $data_num \
    --num_cano $num_cano \
    --save_period 1000 \
    --eval_period 1000 \
    --lr $lr \
    --lambda_inv $lambda_inv \
    --lambda_can $lambda_can \
    --augment
