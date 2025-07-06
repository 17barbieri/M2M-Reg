import os
import random
import logging
import numpy as np
import torch
import h5py

class ADNI_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, device="cpu", which_set="train", return_labels=False):
        self.device = device
        self.return_labels = return_labels
        self.which_set = which_set
        
        hdf5_path = os.path.join(args.data_path, f"{args.dataset}_{which_set}.hdf5")
        self.data_dict = {}
        with h5py.File(hdf5_path, "r") as hf:
            self.data_dict['PT'] = torch.tensor(np.array(hf.get('PT_dataset')), dtype=torch.float32)
            self.data_dict['ST'] = torch.tensor(np.array(hf.get('ST_dataset')), dtype=torch.float32)
            self.data_dict['seg'] = torch.tensor(np.array(hf.get('seg_dataset')), dtype=torch.long)
            self.data_dict['PatientID'] = np.array(hf.get('PatientID'))

        self.data_num = args.data_num
        self.subject_num = self.data_dict['PT'].shape[0]
        logging.info(f"subject num: {self.subject_num}")

        if args.num_cano.endswith('%'):
            self.num_cano = int((float(args.num_cano[:-1]) / 100) * self.subject_num)
        else:
            self.num_cano = int(args.num_cano)
        
        if self.num_cano == -1:
            self.cano_idx = list(range(self.subject_num))
        else:
            self.cano_idx = random.sample(range(self.subject_num), self.num_cano)
        logging.info(f"Selected canonical subjects: \n{self.data_dict['PatientID'][self.cano_idx]}")

    def __len__(self):
        return self.data_num

    def minmax_norm(self, img):
        return (img - img.min()) / (img.max() - img.min())
    
    def random_sagittal_flip(self, img, seg, separately=False):
        """Sagittal 방향으로 랜덤 플립 수행"""
        if 'train' in self.which_set:
            if separately:
                if random.random() > 0.5:
                    img = torch.flip(img, dims=[1])
                if random.random() > 0.5:
                    seg = torch.flip(seg, dims=[1])
            else:
                if random.random() > 0.5:
                    img = torch.flip(img, dims=[1])  # img.shape = (1, 128, 128, 128)
                    seg = torch.flip(seg, dims=[1])
        return img, seg
    
    def __getitem__(self, idx):
        if self.num_cano != 0:
            if self.num_cano == -1:
                cano_src_idx = random.choice(range(self.subject_num))
                cano_tgt_idx = random.choice(range(self.subject_num))
                while cano_src_idx == cano_tgt_idx:
                    cano_tgt_idx = random.choice(range(self.subject_num))
            else:
                cano_idx = random.choice(self.cano_idx)
        
        src_idx = random.randint(0, self.subject_num-1)
        tgt_idx = random.randint(0, self.subject_num-1)
        if self.num_cano == -1:
            while self.num_cano != 0 and (src_idx == cano_src_idx or src_idx == cano_tgt_idx or tgt_idx == cano_src_idx or tgt_idx == cano_tgt_idx):
                src_idx = random.randint(0, self.subject_num-1)
                tgt_idx = random.randint(0, self.subject_num-1)
        else:
            while self.num_cano != 0 and (src_idx == cano_idx or tgt_idx == cano_idx):
                src_idx = random.randint(0, self.subject_num-1)
                tgt_idx = random.randint(0, self.subject_num-1)
        
        src_modal, tgt_modal = random.choice([('PT', 'ST'), ('ST', 'PT')])
        src_img = self.data_dict[src_modal][src_idx].unsqueeze(0)
        tgt_img = self.data_dict[tgt_modal][tgt_idx].unsqueeze(0)
        src_seg = self.data_dict['seg'][src_idx].unsqueeze(0).float()
        tgt_seg = self.data_dict['seg'][tgt_idx].unsqueeze(0).float()

        # Min-Max Normalization
        src_img = self.minmax_norm(src_img)
        tgt_img = self.minmax_norm(tgt_img)
        
        # Apply Augmentation if in training set
        if 'train' in self.which_set:
            src_img, src_seg = self.random_sagittal_flip(src_img, src_seg)
            tgt_img, tgt_seg = self.random_sagittal_flip(tgt_img, tgt_seg)

        if self.num_cano != 0:
            if self.num_cano == -1:
                src_cano_img = self.data_dict[src_modal][cano_src_idx].unsqueeze(0)
                tgt_cano_img = self.data_dict[tgt_modal][cano_tgt_idx].unsqueeze(0)
                
                # Min-Max Normalization
                src_cano_img = self.minmax_norm(src_cano_img)
                tgt_cano_img = self.minmax_norm(tgt_cano_img)
                
                if 'train' in self.which_set:
                    src_cano_img, tgt_cano_img = self.random_sagittal_flip(src_cano_img, tgt_cano_img, separately=True)
            else:
                src_cano_img = self.data_dict[src_modal][cano_idx].unsqueeze(0)
                tgt_cano_img = self.data_dict[tgt_modal][cano_idx].unsqueeze(0)
                
                # Min-Max Normalization
                src_cano_img = self.minmax_norm(src_cano_img)
                tgt_cano_img = self.minmax_norm(tgt_cano_img)
                
                if 'train' in self.which_set:
                    src_cano_img, tgt_cano_img = self.random_sagittal_flip(src_cano_img, tgt_cano_img)
            
            return src_img, tgt_img, src_seg, tgt_seg, src_cano_img, tgt_cano_img
        
        return src_img, tgt_img, src_seg, tgt_seg, src_img, tgt_img
