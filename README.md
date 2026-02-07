
[![GitHub stars](https://img.shields.io/github/stars/Daksh3468/Automated-Bone-Marrow-Cell-Classification?style=social)](https://github.com/Daksh3468/Automated-Bone-Marrow-Cell-Classification)

# Automated Bone Marrow Cell Classification 🧬

## 📌 Overview
This project implements a comprehensive **Computer Vision + Deep Learning pipeline** for automated classification of bone marrow cell images to assist in early leukemia detection.

Multiple **state-of-the-art CNN architectures** were trained and evaluated using **transfer learning**, enabling systematic comparison of performance across models.

---

## 🎯 Objective
Manual analysis of bone marrow cell images is time-consuming and prone to subjectivity.  
This project aims to:
- Automate cell classification using deep learning
- Compare performance across multiple CNN architectures
- Achieve high accuracy on medical imaging datasets
- Build a reproducible and extensible ML pipeline

---

## 🧠 Models Implemented
The following pretrained CNN architectures were used and fine-tuned:

| Model Architecture | Notebook File |
|-------------------|---------------|
| AlexNet | `alexnet-bone-marrow-classification.ipynb` |
| VGG16 | `vgg-16-bone-marrow-classification.ipynb` |
| ResNet50 | `resnet50-bone-marrow-cell-classification.ipynb` |
| ResNet152 | `resnet152-bone-marrow-classification.ipynb` |
| DenseNet121 | `densenet121-bone-marrow-classification.ipynb` |
| InceptionV3 | `inceptionv3-bone-marrow-classification.ipynb` |
| MobileNetV2 | `mobilenetv2-bone-marrow-cell-classification.ipynb` |
| EfficientNet-B5 | `efficient-net-b5-bone-marrow-cell-classification.ipynb` |
| Xception | `xception-bone-marrow-classification.ipynb` |

> Each model was trained using the same preprocessing and evaluation pipeline to ensure fair comparison.

---

## 📊 Results

| Model       | Accuracy |
|-------------|----------|
| ResNet50    | **96%**  |
| VGG16       | 94%      |

> **Note:** Best performance achieved using ResNet50 with data augmentation and preprocessing.

Sample model output predictions and confusion matrices are available in `Final_results.csv`.

---

## 🏗️ Methodology

### 1. Data Preprocessing
- Image resizing and normalization
- Noise reduction
- Dataset splitting (Train / Validation / Test)

### 2. Model Training
- Transfer learning with pretrained ImageNet weights
- Fine-tuning top layers
- Adam optimizer with categorical cross-entropy loss

### 3. Evaluation
- Accuracy
- Confusion Matrix
- Class-wise performance comparison

---


## 🚀 How to Run

### Clone the repo  
```bash
git clone https://github.com/Daksh3468/Automated-Bone-Marrow-Cell-Classification.git
cd Automated-Bone-Marrow-Cell-Classification
```

### Setup environment  
```bash
pip install -r requirements.txt
```


---

## 🛠️ Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** CNNs, Transfer Learning
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** Scikit-learn

---

## 🔑 Key Skills Demonstrated
- Computer Vision for Medical Imaging
- CNN Architecture Comparison
- Transfer Learning & Fine-Tuning
- Data Augmentation & Preprocessing
- Model Evaluation & Benchmarking
- Reproducible Research Pipelines

---

## 👤 Author
**Daksh Bangoria**  
📧 daksh3468@gmail.com  
🔗 https://linkedin.com/in/daksh-bangoria  
🐙 https://github.com/Daksh3468
