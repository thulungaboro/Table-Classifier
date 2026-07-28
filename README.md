# Table Classifier (YOLOv8)

An end-to-end computer vision project designed for detecting and classifying tables in document images into **bordered** and **borderless** tables using YOLOv8.

---

## 📌 Overview

Document structure recognition and tabular data extraction often rely heavily on accurate table detection. This repository provides scripts and pipelines for:
- Combining real document datasets with synthetic table datasets using stratified splitting.
- Training a custom YOLOv8 model for table detection.
- Evaluating model metrics (`mAP50`, `mAP50-95`) across **Pooled (Real + Synthetic)** test sets as well as isolated **Real-Only** test sets.
- Verifying class distributions and split balances.

---

## 📂 Project Structure

```
Project-1-Tb-DTC/
├── dataset/                    # Original annotated dataset (images & labels)
├── dataset_combined/           # Generated combined dataset (train/val/test splits & manifest.csv)
├── runs/                       # YOLOv8 training and evaluation outputs/logs
├── build_combined_dataset.py   # Script to merge real and synthetic pools with custom split ratios
├── evaluate_model.py           # Evaluation script comparing pooled vs real-only test metrics
├── verify_split_balance.py     # Script to check class balance across splits
├── test_model.py               # Inference and model testing script
├── yolov8n.pt                  # Pretrained YOLOv8 base weights
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with `ultralytics` and standard data handling packages:

```bash
pip install ultralytics opencv-python pillow pyyaml
```

---

## 🛠️ Usage

### 1. Build Combined Dataset

Combine scarce real-world annotations with synthetic datasets while maintaining class balance and generating a `manifest.csv` tracking data origin:

```bash
python build_combined_dataset.py \
    --real_images dataset/images/train dataset/images/val \
    --real_labels dataset/labels/train dataset/labels/val \
    --synth_images path/to/synth/images \
    --synth_labels path/to/synth/labels \
    --out dataset_combined
```

### 2. Verify Dataset Split Balance

Verify class distribution (`bordered`, `borderless`, `negative/empty`) across your splits:

```bash
python verify_split_balance.py
```

### 3. Model Evaluation

Evaluate the trained model on both the full pooled test set and the isolated real-world dataset:

```bash
python evaluate_model.py
```

### 4. Run Model Test / Inference

Run inference on test images or sample data:

```bash
python test_model.py
```

---

## 🏷️ Data & Classes

- **`0`**: `bordered` (Tables with explicit borders/grids)
- **`1`**: `borderless` (Tables without explicit column/row lines)

---

## 📜 License

This project is licensed under the MIT License.
