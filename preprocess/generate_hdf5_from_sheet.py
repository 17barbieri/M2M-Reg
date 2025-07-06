import argparse
import time
import h5py
import os
import numpy as np
import nibabel as nib
import pandas as pd
import logging
import time

def create_hdf5_dataset(args):
    start_d = time.time()

    full_sheet = pd.read_excel(
        args.sheet_url.split("/edit")[0] + "/export?format=xlsx",
        sheet_name=None)
    df = full_sheet[args.dataset]
    df.columns = df.columns.map(str)
    
    backup_sheet_path = os.path.join(os.path.dirname(args.hdf5_path), f"{args.dataset}_data_sheet_{args.which_set}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
    with pd.ExcelWriter(backup_sheet_path) as writer:
        for sheet_name, sheet_data in full_sheet.items():
            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
    logging.info(f"Saved current data_sheet to {backup_sheet_path}")
    
    df = df[~df['exclude'].fillna(0).astype(int).astype(bool)]
    
    if args.which_set == "train_val":
        subject_dir = df[df['set'].fillna('').str.contains(r'train|valid', na=False)]['PatientID'].astype(str).tolist()
    else:
        subject_dir = df[df['set'].fillna('').str.contains(args.which_set, na=False)]['PatientID'].astype(str).tolist()
    logging.info(f"data size: {len(subject_dir)}")
    
    subjects = []
    PT_dataset = []
    ST_dataset = []
    seg_dataset = []

    for idx, pid in enumerate(subject_dir):
        start = time.time()
        pid_dir = os.path.join(args.data_dir, pid)
        if not os.path.isdir(pid_dir):
            continue
        
        logging.info("Volume Nr: {} Processing Data from {}".format(idx, pid))

        subjects.append(pid)

        PT_data = nib.load(os.path.join(pid_dir, f"{pid}_{args.PT_name}"))
        PT_data = np.asanyarray(PT_data.dataobj)
        PT_data = np.nan_to_num(PT_data)

        ST_data = nib.load(os.path.join(pid_dir, f"{pid}_{args.ST_name}"))
        ST_data = np.asanyarray(ST_data.dataobj)
        ST_data = np.nan_to_num(ST_data)

        seg_data = nib.load(os.path.join(pid_dir, f"{pid}_{args.seg_name}"))
        seg_data = np.asanyarray(seg_data.dataobj)
        seg_data = np.nan_to_num(seg_data)
        logging.info(PT_data.dtype)
        logging.info(ST_data.dtype)
        logging.info(seg_data.dtype)
        # 데이터를 리스트에 추가
        PT_dataset.append(PT_data)
        ST_dataset.append(ST_data)
        seg_dataset.append(seg_data)

        end = time.time() - start
        logging.info("Volume: {} Finished Data Reading and Appending in {:.3f} seconds.".format(idx, end))

        if args.debugging and idx == 2:
            break

    # 리스트를 numpy 배열로 변환
    PT_dataset = np.asarray(PT_dataset, dtype=np.float32)
    ST_dataset = np.asarray(ST_dataset, dtype=np.float32)
    seg_dataset = np.asarray(seg_dataset, dtype=np.int16)
    print(PT_dataset.dtype, ST_dataset.dtype, seg_dataset.dtype)

    # HDF5 파일에 저장
    with h5py.File(args.hdf5_path, "w") as hf:
        hf.create_dataset('PT_dataset', data=PT_dataset, dtype=np.float32, compression='gzip')
        hf.create_dataset('ST_dataset', data=ST_dataset, dtype=np.float32, compression='gzip')
        hf.create_dataset('seg_dataset', data=seg_dataset, dtype=np.int16, compression='gzip')
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset("PatientID", data=np.array(subjects, dtype=object), dtype=dt, compression="gzip")

    end_d = time.time() - start_d
    logging.info("Successfully written {} in {:.3f} seconds.".format(args.hdf5_path, end_d))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HDF5-Creation')
    parser.add_argument('--sheet_url', type=str, default="https://docs.google.com/spreadsheets/d/1y6kyPNWaAW9uehGHF0nErsn1Q2_rmapK39wRdIBRR8M/edit?usp=sharing")
    parser.add_argument('--hdf5_path', type=str, default="test.hdf5",
                        help='path and name of hdf5-dataset (default: test.hdf5)')
    parser.add_argument('--dataset', type=str, default="ADNI")
    parser.add_argument('--which_set', type=str)
    parser.add_argument('--data_dir', type=str, default="/testsuite", help="Directory with images to load")
    parser.add_argument('--PT_name', type=str)
    parser.add_argument('--ST_name', type=str)
    parser.add_argument('--seg_name', type=str)
    parser.add_argument('--debugging', action='store_true', help="Limit to first 3 subjects for debugging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(os.path.join(os.path.dirname(args.hdf5_path), f"hdf5_{args.dataset}_{args.which_set}_{time.strftime('%Y%m%d_%H%M%S')}.log")),
            logging.StreamHandler()
        ]
    )
    logging.info(args)
    create_hdf5_dataset(args)
