# 🗑️ Waste Classification Using Convolutional Neural Network (CNN)

!\[Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
!\[TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)
!\[License](https://img.shields.io/badge/License-MIT-green)
!\[Dataset](https://img.shields.io/badge/Dataset-\~3.2%20GB-red)

A deep learning project that classifies waste images into **6 categories** using a fine-tuned **MobileNetV2 CNN** achieving **\~91% validation accuracy**.

\---

## 📌 Project Overview

|Item|Detail|
|-|-|
|**Task**|Multi-class image classification|
|**Classes**|cardboard · glass · metal · paper · plastic · trash|
|**Dataset Size**|\~3.2 GB|
|**Model**|MobileNetV2 (Transfer Learning)|
|**Framework**|TensorFlow / Keras|
|**Val Accuracy**|\~91%|

\---

## 📂 Project Structure

```
waste\_cnn/
│
├── train.py              # Main training script
├── predict.py            # Single-image / folder inference
├── download\_dataset.py   # Automated Kaggle dataset download
├── demo\_results.py       # Generate demo graphs without dataset
├── requirements.txt      # Python dependencies
│
├── data/
│   └── garbage\_classification/   # Dataset goes here
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
│
├── models/
│   └── waste\_cnn\_best.h5         # Saved best model (auto-created)
│
└── results/
    ├── training\_history.png      # Accuracy / Loss / Precision-Recall curves
    ├── confusion\_matrix.png      # Count \& normalized confusion matrices
    ├── per\_class\_metrics.png     # Precision, Recall, F1 per class
    ├── class\_distribution.png   # Bar \& pie chart of class balance
    ├── sample\_predictions.png   # Grid of sample predictions
    └── training\_log.csv          # Epoch-by-epoch metrics log
```

\---

## 🗂️ Dataset

**Garbage Classification Dataset** — available on Kaggle (\~3.2 GB)

* **Source**: [Kaggle – Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
* **Total images**: \~2,527 labeled images across 6 classes
* **Format**: JPEG images organized in class subdirectories

### Class Distribution

|Class|Images|Share|
|-|-|-|
|cardboard|403|15.9%|
|glass|501|19.8%|
|metal|410|16.2%|
|paper|594|23.5%|
|plastic|482|19.1%|
|trash|137|5.4%|

\---

## ⚙️ Setup \& Installation

### 1\. Clone the repository

```bash
git clone https://github.com/<your-username>/waste-classification-cnn.git
cd waste-classification-cnn
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

### 3\. Download the dataset

```bash
# Place your kaggle.json token at \~/.kaggle/kaggle.json first
python download\_dataset.py
```

\---

## 🚀 How to Run

### Train the model

```bash
python train.py
```

### Run inference on an image

```bash
python predict.py --image path/to/image.jpg
```

### Run inference on a folder

```bash
python predict.py --folder path/to/images/
```

### Generate demo graphs (no dataset needed)

```bash
python demo\_results.py
```

\---

## 🧠 Model Architecture

```
MobileNetV2 (pretrained on ImageNet)
  └─ First 100 layers frozen
  └─ Remaining layers fine-tuned
→ GlobalAveragePooling2D
→ BatchNormalization
→ Dense(512, relu)  →  Dropout(0.4)
→ Dense(256, relu)  →  Dropout(0.3)
→ Dense(6, softmax)       ← output layer
```

**Optimizer**: Adam (lr=1e-4)  
**Loss**: Categorical Cross-Entropy  
**Callbacks**: ModelCheckpoint · EarlyStopping · ReduceLROnPlateau

\---

## 📊 Results

### Training Curves

!\[Training History](results/training\_history.png)

### Confusion Matrix

!\[Confusion Matrix](results/confusion\_matrix.png)

### Per-Class Metrics

!\[Per Class Metrics](results/per\_class\_metrics.png)

### Class Distribution

!\[Class Distribution](results/class\_distribution.png)

\---

## 📈 Final Performance

|Metric|Score|
|-|-|
|Validation Accuracy|\~91%|
|Validation Precision|\~91%|
|Validation Recall|\~90%|
|Validation Loss|\~0.28|

\---

## 🔧 Data Augmentation

Applied to training set:

* Random rotation (±25°)
* Width / Height shift (±20%)
* Zoom (±20%)
* Horizontal flip
* Brightness adjustment (0.8–1.2×)
* Shear transformation

\---

## 📄 License

MIT License — free to use and modify.

\---

## 👤 Author

**\[Sami Ul Rehman]** — Roll No: **\[F22BSEEN1E02100]**  
Department of \[Software Engineering]  
\[The Islamia University Of Bahawalpur]

