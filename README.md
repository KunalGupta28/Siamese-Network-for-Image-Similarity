# Siamese-Network-for-Image-Similarity

# 📌 Siamese Network for Image Similarity using PyTorch

## 📖 Project Overview
This project implements a **Siamese Neural Network** using **PyTorch** to learn image similarity through **metric learning**. Instead of classifying images directly, the model learns a meaningful **embedding space** where visually similar images are close together and dissimilar images are far apart.

The system is suitable for tasks such as:
- **Person Re-Identification**
- **Face Verification**
- **Image Retrieval**
- **One-shot / Few-shot Learning**

The project uses a **pretrained EfficientNet backbone** via the `timm` library and is trained using **Triplet Loss**.

---

## 🧠 Key Concept: Siamese Network
A **Siamese Network** consists of two or more identical neural networks that:
- Share the same weights
- Take different inputs
- Learn similarity rather than classification

Instead of predicting class labels, the network outputs **embedding vectors**, and similarity is measured using a distance metric in the embedding space.

---

## 🔁 Complete Project Pipeline

### 1️⃣ Data Preparation
- Images are loaded from the dataset directory  
- Each image belongs to a specific identity/class  
- The dataset is split into **training** and **validation** sets  
- Images are read using `skimage.io`  
- Labels are used to generate **triplets**:
  - **Anchor**: reference image  
  - **Positive**: image from the same class as the anchor  
  - **Negative**: image from a different class  

---

### 2️⃣ Custom Dataset Class
A custom PyTorch `Dataset` is implemented to:
- Load images dynamically
- Generate valid `(anchor, positive, negative)` triplets
- Return tensors suitable for model input

This enables efficient and flexible training for metric learning tasks.

---

### 3️⃣ Model Architecture

#### 🔹 Backbone
- **EfficientNet** (via `timm`)
- Pretrained on **ImageNet**
- Used as a high-quality **feature extractor**

#### 🔹 Embedding Head
- Fully connected layers
- Produces a fixed-size embedding vector
- **L2-normalization** is applied to stabilize distance-based learning

---

### 4️⃣ Loss Function – Triplet Margin Loss
The model is trained using **Triplet Loss**:

This ensures:
- Anchor is closer to the Positive sample
- Anchor is farther from the Negative sample by at least a margin

This loss is ideal for similarity-based learning problems.

---

### 5️⃣ Training Loop
- Forward pass through the Siamese Network
- Compute triplet loss
- Backpropagation using the **Adam optimizer**
- Batch-wise training using `DataLoader`
- Training progress tracked using `tqdm`

---

### 6️⃣ Embedding Extraction
After training:
- All images are passed through the trained model
- Embeddings are extracted and stored
- These embeddings represent images in a learned feature space

---

### 7️⃣ Image Retrieval & Evaluation
Given a query image:
- Its embedding is computed
- Distances to all stored embeddings are calculated
- Closest images are retrieved based on distance
- Results are visualized using `plot_closest_imgs`

---

## 📊 Technologies Used & Justification

| Technology | Purpose |
|---------|--------|
| PyTorch | Core deep learning framework |
| timm | Pretrained EfficientNet models |
| Triplet Loss | Metric learning objective |
| NumPy | Numerical operations |
| Pandas | Dataset handling |
| scikit-learn | Train/validation split |
| scikit-image | Image loading |
| Matplotlib | Visualization |
| tqdm | Training progress tracking |
| Custom Utils | Retrieval visualization |

---

## 📁 Project Structure
├── Deep_Learning_with_PyTorch_Siamese_Network.ipynb
├── utils.py
│ └── plot_closest_imgs()
├── data/
│ ├── train/
│ └── test/
├── requirements.txt
└── README.md


---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt


2️⃣ Open the Notebook
jupyter notebook Deep_Learning_with_PyTorch_Siamese_Network.ipynb

3️⃣ Execute All Cells

Train the Siamese Network
Generate embeddings
Perform image retrieval
