# M2M-Reg: Mono-Modalizing Extremely Heterogeneous Multi-Modal Medical Image Registration

**Accepted at MICCAI 2025**

📄 [arXiv Preprint](https://arxiv.org/abs/2506.15596)

---

## 🔧 Requirements

To set up the environment:

```bash
conda env create -f environment.yml
conda activate m2m_reg
```

---

## 📁 Dataset Preparation

This project uses data from the [ADNI dataset](https://adni.loni.usc.edu/), specifically:

- 18F-FBB PET
- FA (Fractional Anisotropy)  
- T1-weighted MRI  

We perform PET-MRI and FA-MRI registration. For each subject, we assume the presence of:

- A paired image set (PET-MRI or FA-MRI)
- A segmentation mask derived from the MRI

> **Note:** The segmentation mask is **not used during training**, but is used for **validation purposes only**.

---

### 🔄 Preprocessing

We assume that image pairs and segmentation masks are **already co-registered**. This repository includes only the post-registration steps such as resampling and cropping.

> Segmentation (e.g., via FreeSurfer) and registration (e.g., via ANTsPy) steps are not released due to external constraints, but can be replicated using the mentioned tools.

#### Example directory structure (PET-MRI)

```
data_root/
└── SubjectID/
    ├── 256_PET.nii
    ├── 256_MR.nii
    └── seg_MR.nii
```

#### Run preprocessing

- PET-MRI:

```bash
sh preprocess/preprocess_ADNI_FBB.sh
```

- FA-MRI:

```bash
sh preprocess/preprocess_ADNI_FA.sh
```

#### Additional scripts

- FA skull stripping: `preprocess/synthstrip_fa.py`
- Merging DKT-based ROI labels: `preprocess/remap_segmentation_label.py`

---

## 🧬 HDF5 File Generation

Subject metadata must be provided via a [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1y6kyPNWaAW9uehGHF0nErsn1Q2_rmapK39wRdIBRR8M/edit?usp=sharing) with the following **required columns**:

- `PatientID`  
- `set`  
- `exclude`  

You must provide the spreadsheet URL as the `sheet_url` argument when generating the HDF5 file.

#### Expected folder structure after preprocessing (PET-MRI example)

```
~/M2M-Reg/datasets/ADNI_FBB_preprocessed/
└── SubjectID/
    ├── {SubjectID}_preprocessed_PT.nii
    ├── {SubjectID}_preprocessed_MR.nii
    └── {SubjectID}_preprocessed_seg_map.nii
```

#### Generate the HDF5 dataset:

```bash
sh preprocess/generate_hdf5_from_sheet.sh
```

---

## 🏋️ Training

Launch training with:

```bash
sh scripts/train.sh
```

Use the `num_cano` argument to control the training strategy, such as:

- Unsupervised
- Semi-supervised (can control the number of pre-aligned pairs used via `num_cano`)

Refer to the help message for detailed argument usage.

---

## 🧪 Testing

Run testing with:

```bash
sh scripts/test.sh
```

> The `target_pid` column in the subject Spreadsheet is **required** and is used to define subject pairings for evaluation.

---

## 🙏 Acknowledgements

This codebase is based on:

- [ICON](https://github.com/uncbiag/ICON)  
- [MultiGradICON (uniGradICON)](https://github.com/uncbiag/uniGradICON)
