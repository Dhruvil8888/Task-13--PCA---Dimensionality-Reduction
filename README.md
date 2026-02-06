# Task 13: PCA – Dimensionality Reduction

## 📌 Overview
This project demonstrates **Principal Component Analysis (PCA)** for dimensionality reduction using image datasets such as **MNIST** or the **Scikit-learn Digits dataset**.  
The objective is to reduce high-dimensional image data while retaining most of the important information (variance) and to study the trade-off between **model performance and feature compression**.

The project also compares **classification accuracy before and after PCA** using Logistic Regression.

---

## 🛠 Tools & Technologies
- Python  
- Scikit-learn  
- NumPy  
- Matplotlib  

---

## 📊 Dataset
**Primary:** MNIST dataset  
**Alternative:** Sklearn Digits dataset (`load_digits()`)

For demonstration and lightweight execution, the **Sklearn Digits dataset** is used.

- Images: 8×8 grayscale  
- Original features: 64  
- Classes: Digits 0–9  

---

## 📂 Project Structure
task-13-pca-dimensionality-reduction/
│
├── notebooks/
│ └── Task13_PCA_Dimensionality_Reduction.ipynb
│
├── visuals/
│ ├── explained_variance.png
│ └── pca_2d_scatter.png
│
├── data/
│ └── digits_pca_reduced.csv
│
├── README.md
└── requirements.txt


---

## 🔹 Step 1: Load Dataset

```python
from sklearn.datasets import load_digits
import pandas as pd

digits = load_digits()
X = digits.data        # flattened images
y = digits.target

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
🔹 Step 2: Feature Scaling
PCA is sensitive to feature scale, so normalization is required.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
🔹 Step 3: Apply PCA with Different Components
from sklearn.decomposition import PCA

components = [2, 10, 30, 50]
explained_variance = []

for n in components:
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    explained_variance.append(sum(pca.explained_variance_ratio_))
🔹 Step 4: Explained Variance Plot
import matplotlib.pyplot as plt

plt.plot(components, explained_variance, marker='o')
plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs Number of Components")
plt.show()
This plot helps choose the optimal number of components that retain most variance.

🔹 Step 5: Dimensionality Reduction
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_scaled)

print("Reduced feature shape:", X_pca.shape)
🔹 Step 6: Logistic Regression on Reduced Data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
pca_accuracy = accuracy_score(y_test, y_pred)

print("Accuracy with PCA:", pca_accuracy)
🔹 Step 7: Compare with Original Dataset
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model.fit(X_train_o, y_train_o)
y_pred_o = model.predict(X_test_o)

original_accuracy = accuracy_score(y_test_o, y_pred_o)

print("Accuracy without PCA:", original_accuracy)
🔹 Step 8: PCA 2D Visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='tab10', s=10)
plt.colorbar(label="Digit Label")
plt.title("2D PCA Projection of Digits Dataset")
plt.show()
This visualization shows class separation in reduced dimensions.

📈 Sample Results (Typical)
Model	Features	Accuracy
Logistic Regression (Original)	64	~97%
Logistic Regression + PCA	30	~95%
Logistic Regression + PCA	10	~92%
🎯 Final Outcome
After completing this task, the intern:

Understands dimensionality reduction

Learns variance vs accuracy trade-off

Applies PCA correctly with scaling

Visualizes high-dimensional data in 2D

Builds efficient ML pipelines with fewer features

