# Table Classifier (YOLOv8)

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)
![Computer Vision](https://img.shields.io/badge/Task-Table%20Detection%20%26%20Classification-orange.svg)

An end-to-end computer vision repository designed for detecting and classifying tables in document images into **bordered** and **borderless** classes using custom-trained YOLOv8.

---

## 📊 Results & Performance

Evaluated on **100 epochs** of training using pooled (real + synthetic) and real-only document test splits:

| Evaluation Split | Total Images | mAP50 | mAP50-95 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pooled Test Set (Real + Synthetic)** | 263 | **0.973** | **0.858** | 0.962 | 0.961 |
| **Real-Only Test Set** | 110 | **0.965** | **0.842** | 0.958 | 0.954 |

### 📈 Model Evaluation Metrics & Curves
- **Precision-Recall Curve**: `runs/detect/train/BoxPR_curve.png`
- **Confusion Matrix**: `runs/detect/train/confusion_matrix.png`
- **Training Curves**: `runs/detect/train/results.png`

---

## 📂 Project Structure

```
Project-1-Tb-DTC/
├── dataset/                    # Original real annotated document dataset
├── dataset_combined/           # Pooled train/val/test dataset & manifest.csv
├── runs/                       # YOLOv8 training, evaluation plots, and weights
├── build_combined_dataset.py   # Script to merge real + synthetic pools with stratified ratios
├── generate_synthetic_dataset.py # Synthetic document page generator
├── evaluate_model.py           # Evaluates model on pooled vs real-only test splits
├── verify_split_balance.py     # Checks class distributions across splits
├── test_model.py               # Comprehensive inference testing menu
├── demo.py                     # 2-minute quick demo inference script
├── requirements.txt            # Pinned dependencies
└── README.md                   # Documentation
```

---

## 📁 Dataset & Generation

1. **Real Data**: Genuine scanned documents and PDF page renders annotated for `bordered` (Class 0) and `borderless` (Class 1) table structures.
2. **Synthetic Data**: Generated programmatically using [generate_synthetic_dataset.py](file:///d:/Project-1-Tb-DTC/generate_synthetic_dataset.py) to simulate diverse table layouts, fonts, cell densities, and negative (table-free) document pages.
3. **Stratified Splitting**: [build_combined_dataset.py](file:///d:/Project-1-Tb-DTC/build_combined_dataset.py) balances scarce real images (weighted to val/test) with cheap synthetic images (weighted to train), maintaining a `manifest.csv` for data source auditing.

---

## 🚀 Quick Start & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo Inference (2 minutes)
```bash
python demo.py --image table_test.jpg
```

---

## 🛠️ Data Pipeline & Training Workflow

### Generate Synthetic Dataset
```bash
python generate_synthetic_dataset.py --out synthetic_pool --num 500 --neg_ratio 0.15
```

### Merge Real + Synthetic Datasets
```bash
python build_combined_dataset.py \
    --real_images dataset/images/train dataset/images/val \
    --real_labels dataset/labels/train dataset/labels/val \
    --synth_images synthetic_pool/images \
    --synth_labels synthetic_pool/labels \
    --out dataset_combined
```

### Verify Split Balance
```bash
python verify_split_balance.py dataset_combined
```

### Evaluate Model
```bash
python evaluate_model.py
```

---

## 🏷️ Class Definitions

- **`0` (`bordered`)**: Tables with visible lines/borders grid structure.
- **`1` (`borderless`)**: Tables aligned with whitespace without bounding line borders.
