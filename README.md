# 🌶️ Chilli Leaf Disease Classification Using Deep Learning (Multi-Model Benchmarking)

## 📌 Project Overview
This project focuses on multi-class classification of chilli leaf diseases using deep learning and transfer learning techniques. The objective is to automatically detect and classify different types of chilli leaf diseases from images to support smart agriculture and early disease diagnosis.

The system benchmarks multiple state-of-the-art Convolutional Neural Network (CNN) architectures on an augmented agricultural image dataset containing six disease categories.

---

## 🧠 Models Implemented
The following deep learning architectures were implemented and compared:

- ConvNeXt Tiny  
- DenseNet121  
- EfficientNet-B4  
- ResNet50  
- MobileNetV2  

All models were trained using transfer learning with pretrained ImageNet weights.

---

## 📊 Dataset
- Dataset: Chilli Leaf Disease Augmented Dataset  
- Number of Classes: 6  
- Classes include:
  - Bacterial Spot  
  - Cercospora Leaf Spot  
  - Curl Virus  
  - Healthy Leaf  
  - Nutrient Deficiency  
  - White Spot  

The dataset was preprocessed and split using a stratified approach.

---

## ⚙️ Methodology
- Data preprocessing and normalization  
- Stratified Train/Validation/Test split (70/15/15)  
- Data augmentation for improved generalization  
- Transfer learning with pretrained CNN models  
- Model benchmarking and comparative analysis  
- GPU-based training pipeline  
- Checkpoint saving for reproducibility  

---

## 📈 Evaluation Metrics
Model performance was evaluated using:
- Accuracy  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  
- ROC-AUC Analysis  
- Precision-Recall Curves  

---

## 🛠️ Tech Stack
- Python  
- PyTorch  
- NumPy  
- Matplotlib  
- scikit-learn  
- Computer Vision (CNNs & Transfer Learning)  

---

## 📂 Project Structure
Chilli-Leaf-Disease-Classification/
│
├── notebooks/
│ ├── convnext_tiny.ipynb
│ ├── densenet121.ipynb
│ ├── efficientnet_b4.ipynb
│ ├── mobilenetv2.ipynb
│ └── resnet50.ipynb
├── README.md
└── requirements.txt


---

## 🚀 Key Features
- Multi-model deep learning benchmarking  
- Reproducible experimental pipeline  
- Stratified dataset splitting  
- Transfer learning for high performance  
- Comparative analysis of modern CNN architectures  

---

## 🔬 Applications
- Smart Agriculture  
- Automated Plant Disease Detection  
- Precision Farming  
- AI in Agriculture & Computer Vision Research  

---

## 📌 Author
**Vuppala Shasank**  
B.Tech CSE (AI), Amrita Vishwa Vidyapeetham  
GitHub: https://github.com/VuppalaShasank
