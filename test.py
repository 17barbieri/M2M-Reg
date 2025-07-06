import os
import time
import argparse
import random
import numpy as np
import logging
import pandas as pd
import h5py
from tqdm import tqdm
import torch
import nibabel as nib
from models import make_network, dice_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "test.log")
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)

def save_nifti(data, reference_nifti_path, save_path, dtype=np.float32):
    ref_nifti = nib.load(reference_nifti_path)
    ref_nifti.header['bitpix'] = 32 if dtype == np.float32 else 16
    ref_nifti.header['scl_slope'] = 1
    ref_nifti.header['scl_inter'] = 0

    save_nii = nib.Nifti1Image(data.astype(dtype), affine=ref_nifti.affine, header=ref_nifti.header)
    nib.save(save_nii, save_path)
    logging.info(f"Saved: {save_path}")

def save_nifti_vector_field(data, reference_nifti_path, save_path):
    """
    3D transformation field (3, D, H, W)를 올바르게 저장하여 ITK-SNAP에서 multi-channel로 표시되도록 함
    """
    ref_nifti = nib.load(reference_nifti_path)
    ref_nifti.header['bitpix'] = 32
    ref_nifti.header['scl_slope'] = 1
    ref_nifti.header['scl_inter'] = 0
    # 새로운 NIfTI 객체 생성
    nii_img = nib.Nifti1Image(data.astype(np.float32), affine=ref_nifti.affine)
    
    # ITK-SNAP에서 벡터 필드로 인식되도록 intent 설정
    nii_img.header.set_intent('vector', (), '')
    
    # 저장
    nib.save(nii_img, save_path)
    logging.info(f"Saved deformation field as vector field to {save_path}")

def minmax_norm(img):
    return (img - img.min()) / (img.max() - img.min())

def main():
    # base options
    parser = argparse.ArgumentParser(description="Registration and DICE score calculation")
    parser.add_argument('--sheet_url', type=str, default="https://docs.google.com/spreadsheets/d/1y6kyPNWaAW9uehGHF0nErsn1Q2_rmapK39wRdIBRR8M/edit?usp=sharing")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model", choices=['initial', 'gradicon', 'transmorph', 'corrmlp'], required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--hdf_path", type=str, required=False, default="~/M2M-Reg/datasets")
    parser.add_argument("--gpu_ids", type=str, default='0')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    args.exp_name = args.model
    if args.model in ["gradicon", "transmorph"]:
        args.exp_name += "_" + args.model_path.split('/')[-3] + "_" + args.model_path.split('_')[-1]
    args.exp_name = "debug_" + args.exp_name if args.debug else args.exp_name
    args.out_dir = os.path.join(args.out_dir, args.dataset, args.exp_name)

    if args.model in ["gradicon", "transmorph"]:
        device = torch.device(f"cuda:{int(args.gpu_ids)}")
    else:
        device = torch.device("cpu")
    
    os.makedirs(args.out_dir, exist_ok=True)

    start_time = time.time()
    setup_logging(args.out_dir)

    full_sheet = pd.read_excel(
        args.sheet_url.split("/edit")[0] + "/export?format=xlsx",
        sheet_name=None)
    df = full_sheet[args.dataset]
    df.columns = df.columns.map(str)
    
    df = df[~df['exclude'].fillna(0).astype(int).astype(bool)]
    
    src_subjects = df[df['set'].str.contains('test', na=False)]['PatientID'].astype(str).tolist()
    tgt_subjects = df[df['set'].str.contains('test', na=False)]['target_pid'].astype(str).tolist()

    data_dict = {}
    if 'ADNI' in args.dataset:
        hdf5_path = os.path.join(args.hdf_path, f"{args.dataset}_test.hdf5")
        with h5py.File(hdf5_path, "r") as hf:
            data_dict['PT'] = torch.tensor(np.array(hf.get('PT_dataset')), dtype=torch.float32)
            data_dict['ST'] = torch.tensor(np.array(hf.get('ST_dataset')), dtype=torch.float32)
            data_dict['seg'] = torch.tensor(np.array(hf.get('seg_dataset')), dtype=torch.long)
            data_dict['PatientID'] = np.array(hf.get('PatientID'))
        
        patient_idx = {pid: idx for idx, pid in enumerate(data_dict['PatientID'])}

    # 평균 loss 저장 변수 추가
    total_similarity_loss = 0.0
    total_transform_magnitude = 0.0
    total_flips = 0.0
    total_dice_score = 0.0
    num_cases = 0
    for src_pid, tgt_pid in tqdm(zip(src_subjects, tgt_subjects)):
        logging.info(f"Processing: {src_pid} -> {tgt_pid}")

        case_out_dir = os.path.join(args.out_dir, f"{src_pid}-{tgt_pid}")
        os.makedirs(case_out_dir, exist_ok=True)

        if 'ADNI' in args.dataset:
            src_idx, tgt_idx = patient_idx[src_pid.encode()], patient_idx[tgt_pid.encode()]
            src_img = minmax_norm(data_dict['PT'][src_idx].unsqueeze(0).unsqueeze(0).to(device))
            tgt_img = minmax_norm(data_dict['ST'][tgt_idx].unsqueeze(0).unsqueeze(0).to(device))
            src_seg = data_dict['seg'][src_idx].unsqueeze(0).unsqueeze(0).float().to(device)
            tgt_seg = data_dict['seg'][tgt_idx].unsqueeze(0).unsqueeze(0).float().to(device)
            
            
        if args.model == 'initial':
            warped_src_img = src_img.cpu().numpy().astype(np.float32)[0, 0]
            warped_src_seg = src_seg.cpu().numpy().astype(np.int16)[0, 0]
            # Dice Score 계산
            dice_score_value = dice_score(src_seg[0], tgt_seg[0], dice_logging=True).item()
            logging.info(f"Dice Score: {dice_score_value:.4f}")

            total_dice_score += dice_score_value
            num_cases += 1
            

        else:
            args.num_cano = '0' # Conventional registration. We don't need M2M-Reg for inference.
            args.lambda_inv = 0.5
            args.lambda_can = 0.1
            if 'ADNI' in args.dataset:
                args.input_shape = (1, 1, 128, 128, 128)
            else:
                args.input_shape = (1, 1, 160, 160, 160)
                
            net = make_network(args, include_last_step=False, use_label=False)
            torch.cuda.set_device(int(args.gpu_ids))
            net.regis_net.load_state_dict(torch.load(args.model_path, map_location="cpu"))
            
            net_par = net.to(device).eval()
            with torch.no_grad():
                test_loss = net_par(src_img, tgt_img, src_img, tgt_img, src_seg, tgt_seg, dice_logging=True)
                logging.info(test_loss)

                warped_src_img = net_par.warped_image_A.cpu().numpy().astype(np.float32)[0, 0]
                warped_src_seg = net_par.warped_label_A.cpu().numpy().astype(np.int16)[0, 0]
                transformation = net_par.phi_AB_vectorfield.cpu().numpy().astype(np.float32)[0]

                # 평균 loss 값 누적
                total_similarity_loss += test_loss.similarity_loss.item()
                total_transform_magnitude += test_loss.transform_magnitude.item()
                total_flips += test_loss.flips.item()
                total_dice_score += test_loss.Dice_score.item()
                num_cases += 1
                print(warped_src_img.shape)
                print(transformation.shape)
                net_par.clean()

        if 'ADNI' in args.dataset:
            # Save warped image, segmentation, and transformation
            save_nifti(warped_src_img, os.path.join(args.data_path, tgt_pid, f"{tgt_pid}_preprocessed_MR.nii"), 
                        os.path.join(case_out_dir, f"{src_pid}->{tgt_pid}_PT.nii"))
            if args.dataset == 'ADNI':
                save_nifti(warped_src_seg, os.path.join(args.data_path, tgt_pid, f"{tgt_pid}_preprocessed_seg_map_easy.nii"), 
                            os.path.join(case_out_dir, f"{src_pid}->{tgt_pid}_seg.nii"), dtype=np.int16)
            else:
                save_nifti(warped_src_seg, os.path.join(args.data_path, tgt_pid, f"{tgt_pid}_preprocessed_seg_map.nii"), 
                            os.path.join(case_out_dir, f"{src_pid}->{tgt_pid}_seg.nii"), dtype=np.int16)
                
            if args.model != 'initial':
                save_nifti_vector_field(transformation, os.path.join(args.data_path, src_pid, f"{src_pid}_preprocessed_PT.nii"), os.path.join(case_out_dir, f"{src_pid}->{tgt_pid}_deformation.nii"))

        logging.info(f"Completed: {src_pid} -> {tgt_pid}")
    
    # 최종 평균 loss 로깅
    if num_cases > 0:
        avg_similarity_loss = total_similarity_loss / num_cases
        avg_transform_magnitude = total_transform_magnitude / num_cases
        avg_flips = total_flips / num_cases
        avg_dice_score = total_dice_score / num_cases

        logging.info("\n===== Final Average Metrics =====")
        if args.model != 'initial':
            if args.model == 'gradicon':
                logging.info(f"Avg Similarity Loss: {avg_similarity_loss:.6f}")
                logging.info(f"Avg Transform Magnitude: {avg_transform_magnitude:.6f}")
            logging.info(f"Avg Flips: {avg_flips:.6f}")
        logging.info(f"Avg Dice Score: {avg_dice_score:.6f}")
        logging.info("=" * 40)
    logging.info(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
