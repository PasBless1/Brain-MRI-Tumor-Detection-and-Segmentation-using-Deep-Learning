
# MRI Brain Tumor Segmentation Using ResUNet Deep Learning Architecture

## Overview

This project implements a **two-stage deep learning pipeline** for brain tumor detection 
and segmentation from MRI scans. The first stage determines whether a tumor is present 
using a ResNet50-based binary classifier. The second stage localizes the tumor at the 
pixel level using a custom **ResUNet** — a U-Net architecture enhanced with residual 
blocks — trained with the Focal Tversky loss function.

This work is clinically motivated: accurate and automated delineation of brain tumors 
from MRI is critical for treatment planning, surgical guidance, and disease monitoring, 
and remains challenging due to high shape variability and irregular tumor boundaries.

---

## Dataset

The project uses the [Brain MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) 
from Kaggle, containing axial MRI slices of low-grade gliomas with corresponding 
expert-annotated binary tumor masks.

- **Total images**: 3,929 MRI slices
- **Tumor-positive slices**: 1,373 (used exclusively for segmentation training)
- **Image format**: RGB, resized to `256 × 256`
- **Labels**: Binary mask — `255` = tumor, `0` = background

---

## Pipeline Architecture

The project is structured as a sequential two-task pipeline:

```
Stage 1: Tumor Classification
   ResNet50 (ImageNet pretrained) → Custom Dense Head → Binary Output (tumor / no tumor)

Stage 2: Tumor Segmentation (tumor-positive cases only)
   ResUNet (Residual U-Net) → Pixel-wise Binary Mask → Tumor Localization
```

---

## Stage 1 — Tumor Classification (ResNet50)

A **transfer learning** approach is used, leveraging a ResNet50 backbone pretrained on 
ImageNet as a feature extractor, with a custom classification head for binary prediction.

### Model Architecture

```
ResNet50 (pretrained, weights frozen)
   └─ AveragePooling2D (pool_size=4×4)
   └─ Flatten
   └─ Dense(256, ReLU) → Dropout(0.3)
   └─ Dense(256, ReLU) → Dropout(0.3)
   └─ Dense(2, Softmax)         ← binary output
```

- **Total parameters**: 25,751,426 (~98.23 MB)
- **Input shape**: `256 × 256 × 3`

### Training Configuration

| Parameter          | Value                          |
|--------------------|--------------------------------|
| Loss               | Categorical Cross-Entropy      |
| Optimizer          | Adam                           |
| Batch size         | 16                             |
| Train / Val / Test | 2,839 / 500 / 590              |
| Callbacks          | EarlyStopping (patience=20), ModelCheckpoint |

---

## Stage 2 — Tumor Segmentation (ResUNet)

For pixel-level tumor localization, a **ResUNet** is constructed from scratch using 
residual blocks within a symmetric encoder-decoder framework. Skip connections preserve 
spatial context across resolution levels, while residual shortcuts mitigate gradient 
degradation in deep networks.

### ResUNet Architecture

```
Input (256×256×3)
│
Encoder:
├─ Stage 1: DoubleConv(16) + ResBlock → MaxPool
├─ Stage 2: ResBlock(32) → MaxPool
├─ Stage 3: ResBlock(64) → MaxPool
├─ Stage 4: ResBlock(128) → MaxPool
└─ Bottleneck: ResBlock(256)

Decoder:
├─ UpSample + Concat(skip4) → ResBlock(128)
├─ UpSample + Concat(skip3) → ResBlock(64)
├─ UpSample + Concat(skip2) → ResBlock(32)
└─ UpSample + Concat(skip1) → ResBlock(16)

Output: Conv2D(1, sigmoid) → Binary mask
```

Each **residual block** implements a main path (`Conv2D → BatchNorm`) and a shortcut 
path (`Conv2D(1×1) → BatchNorm`), with outputs summed before activation — following 
the He et al. (2016) residual learning formulation.

### Custom Loss: Focal Tversky Loss

Standard cross-entropy performs poorly on highly imbalanced segmentation tasks where 
tumor regions occupy a small fraction of the total image. This project employs the 
**Focal Tversky Loss**, which generalizes the F-beta score and focuses training on 
hard, misclassified tumor voxels:

```python
Tversky(y_true, y_pred) = TP / (TP + α·FP + β·FN)
Focal Tversky Loss = (1 - Tversky)^γ
```

This formulation allows direct control over the precision-recall trade-off via `α` 
and `β`, with `γ` modulating focus on difficult examples.

### Training Configuration

| Parameter          | Value                             |
|--------------------|-----------------------------------|
| Loss               | Focal Tversky Loss                |
| Metric             | Tversky Score                     |
| Optimizer          | Adam (lr=0.05, epsilon=0.1)       |
| Input samples      | 1,373 (tumor-positive only)       |
| Callbacks          | EarlyStopping, ModelCheckpoint    |

A custom `DataGenerator` handles image-mask pair loading and batching for the 
segmentation model.

---

## Results

### Classification Performance (ResNet50)

| Metric       | Class 0 (No Tumor) | Class 1 (Tumor) | Weighted Avg |
|--------------|--------------------|-----------------|--------------|
| Precision    | 0.98               | 0.99            | 0.99         |
| Recall       | 0.99               | 0.97            | 0.99         |
| F1-Score     | 0.99               | 0.98            | 0.99         |
| **Accuracy** | —                  | —               | **98.61%**   |

The classifier generalizes well across both classes, achieving near-perfect F1-scores 
on 576 held-out test samples.

### Segmentation Performance (ResUNet)

The ResUNet was trained on tumor-positive MRI slices and evaluated via the Tversky 
score metric. Qualitative inspection of predicted masks showed accurate delineation of 
tumor boundaries, with predictions closely following the annotated ground truth contours.
The Focal Tversky loss effectively guided the model to minimize false negatives, which 
are clinically more costly than false positives in oncological screening.

---

## Visualization

The notebook includes rich visual diagnostics:

- **MRI + mask overlay**: Brain MRI, binary ground truth mask, and red-highlighted 
  tumor region displayed side-by-side for qualitative inspection
- **Confusion matrix**: Seaborn heatmap of true vs. predicted labels for the classifier
- **Segmentation output**: Predicted mask overlaid on original MRI for visual evaluation 
  of boundary accuracy

---

## Dependencies

```bash
pip install tensorflow keras scikit-learn opencv-python matplotlib seaborn nibabel
pip install keras-preprocessing
```

---

## Repository Structure

```
├── MRI-Brain-Tumor-Segmentation-Using-ResUNet-Deep-Learning-Architecture.ipynb
├── utilities.py                    # Custom DataGenerator, Focal Tversky Loss, Tversky metric
├── weights.hdf5                    # Best classifier weights
├── weightsseg.hdf5                 # Best segmentation model weights
├── resnet-50-MRI.json              # Classifier architecture (JSON)
├── ResUNet-MRI.json                # Segmentation model architecture (JSON)
└── data/
    ├── images/                     # MRI scan slices
    └── masks/                      # Corresponding binary tumor masks
```

---

## References

- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.  
  [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- Abraham, N. & Khan, N.M. (2019). *A Novel Focal Tversky Loss Function with Improved 
  Attention U-Net for Lesion Segmentation*. ISBI.
- Dataset: [Brain MRI Segmentation — Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
```
