# Table Classifier (YOLOv8)

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Model](https://img.shields.io/badge/model-YOLOv8-orange)
![Task](https://img.shields.io/badge/task-Table%20Detection%20%26%20Classification-green)

An end-to-end computer vision project for detecting and classifying tables in document images into **bordered** and **borderless** tables using YOLOv8.

![Example detection](docs/example_detection.jpg)

---

##  Overview

Document structure recognition and tabular data extraction often rely heavily on accurate table detection. This repository provides scripts and pipelines for:

- Combining real document datasets with synthetic table datasets using stratified splitting.
- Training a custom YOLOv8 model for table detection.
- Evaluating model metrics (`mAP50`, `mAP50-95`) across **Pooled (Real + Synthetic)** test sets as well as isolated **Real-Only** test sets.
- Verifying class distributions and split balances.

---

##  Results

Evaluated on 100 epochs of training across pooled and real-only document test splits:

| Test set | mAP50 | mAP50-95 | Precision | Recall |
| --- | :---: | :---: | :---: | :---: |
| **Pooled (Real + Synthetic)** | **0.9806** | **0.8570** | 0.963 | 0.968 |
| **Real-Only** | **0.9538** | **0.8310** | 0.926 | 0.927 |

![Training curves](docs/training_curves.png)

**Class distribution across splits (`dataset_combined`):**

```
train:  bordered=503  borderless=560  (Total: 1063 annotations across 789 images)
val:    bordered=169  borderless=187  (Total: 356 annotations across 253 images)
test:   bordered=167  borderless=195  (Total: 362 annotations across 263 images)
```

---

##  Project Structure

```
Project-1-Tb-DTC/
├── dataset/                    # Original annotated dataset (images & labels)
├── dataset_combined/           # Generated combined dataset (train/val/test splits & manifest.csv)
├── runs/                       # YOLOv8 training and evaluation outputs/logs
├── docs/                       # README images (example detections, training curves)
├── build_combined_dataset.py   # Merge real and synthetic pools with custom split ratios
├── generate_synthetic_dataset.py # Synthetic document page generator script
├── evaluate_model.py           # Evaluation script comparing pooled vs real-only test metrics
├── verify_split_balance.py     # Script to check class balance across splits
├── test_model.py               # Comprehensive inference testing menu
├── demo.py                     # Quick single-image inference demo
├── requirements.txt            # Pinned dependencies
├── yolov8n.pt                  # Pretrained YOLOv8 base weights
└── README.md                   # Project documentation
```

---

##  Getting Started

### Prerequisites

Python 3.8+ is required.

```bash
git clone https://github.com/thulungaboro/Table-Classifier.git
cd Table-Classifier
pip install -r requirements.txt
```

### Dataset

- **Real images**: Annotated real-world document pages (scanned documents & PDF page renders) with labels for bordered and borderless table regions.
- **Synthetic images**: Generated programmatically via `generate_synthetic_dataset.py` simulating table layouts and text formatting.
- **Combined Split**: Built using `build_combined_dataset.py` to balance real data (skewed to val/test) and synthetic data (skewed to train), tracked via `manifest.csv`.

---

##  Usage

### 1. Generate Synthetic Dataset
```bash
python generate_synthetic_dataset.py --out synth_pool --num 500 --neg_ratio 0.15
```

### 2. Build Combined Dataset

Combine scarce real-world annotations with synthetic datasets while maintaining class balance and generating a `manifest.csv` tracking data origin:

```bash
python build_combined_dataset.py \
    --real_images dataset/images/train dataset/images/val \
    --real_labels dataset/labels/train dataset/labels/val \
    --synth_images synth_pool/images \
    --synth_labels synth_pool/labels \
    --out dataset_combined
```

### 3. Verify Dataset Split Balance

```bash
python verify_split_balance.py dataset_combined
```

### 4. Model Evaluation

```bash
python evaluate_model.py
```

### 5. Quick Demo

Run inference on a single image and save an annotated output:

```bash
python demo.py --image table_test.jpg --weights runs/detect/train/weights/best.pt --out docs/example_detection.jpg
```

---

##  Data & Classes

| ID | Class        | Description                                  |
|----|--------------|-----------------------------------------------|
| 0  | `bordered`   | Tables with explicit borders/grid lines       |
| 1  | `borderless` | Tables without explicit column/row lines      |

---

##  Roadmap

- [ ] Add table structure recognition (rows/columns) on top of detection
- [ ] Export model to ONNX for fast production inference
- [ ] Add a Google Colab notebook for interactive demonstration

---

##  Contributing

Issues and pull requests are welcome. Please open an issue first to discuss any major changes.
