#!/bin/bash

cd /nas/chu/miccai

# 경로 설정
DATA_DIR="/nas/chu/ADNI_FA_paired"
OUT_DATA_DIR="/nas/chu/ADNI_FA_preprocessed"
TARGET_SHAPE=128
TARGET_VOXEL=1.5
COREG=True

# 실행 로그 출력
echo "Starting Preprocessing..." | tee -a $LOG_PATH
echo "Data Directory: $DATA_DIR" | tee -a $LOG_PATH
echo "Output Directory: $OUT_DATA_DIR" | tee -a $LOG_PATH

# Python 스크립트 실행
cd $OUT_DATA_DIR/..
python -u /nas/chu/miccai/preprocess_ADNI_FA.py \
    --data_dir "$DATA_DIR" \
    --out_data_dir "$OUT_DATA_DIR" \
    --target_shape $TARGET_SHAPE \
    --target_voxel $TARGET_VOXEL \
    --analysis 0 \
    | tee -a $LOG_PATH

# 종료 로그 출력
echo "Preprocessing Completed." | tee -a $LOG_PATH

cd $OUT_DATA_DIR/..
tar -cvzf ADNI_FA_preprocessed_seg_map.tar.gz $(find ADNI_FA_preprocessed -type f \( -name "*_preprocessed_MR.nii" -o -name "*_preprocessed_PT.nii" -o -name "*_preprocessed_seg.nii" -o -name "*_preprocessed_seg_map.nii" \))
