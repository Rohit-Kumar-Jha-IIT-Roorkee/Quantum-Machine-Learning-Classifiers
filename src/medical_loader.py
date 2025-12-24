
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def process_medical_data(save_path='medical_data.npz'):
    print("--- LOADING MEDICAL DATA (Breast Cancer Wisconsin) ---")
    data = load_breast_cancer()
    X = data.data
    y = data.target # 0: Malignant, 1: Benign (Note: Sklearn default, we might want to flip or keep)
    # In sklearn data: 212 Malignant (0), 357 Benign (1)
    # Usually we want 1 to be the "positive" class (Malignant/Disease).
    # Let's check distribution
    
    unique, counts = np.unique(y, return_counts=True)
    print(f"Original Distribution: {dict(zip(unique, counts))}")
    
    # Flip labels so 1 is Rare/Malignant if needed? 
    # Actually, in this dataset 0 is malignant. Let's flip so 1 = Malignant (Target).
    y = 1 - y 
    print("Labels flipped: 1 = Malignant (Disease), 0 = Benign (Healthy)")
    
    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. PCA -> 4 Components
    pca = PCA(n_components=4)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Variance Preserved: {sum(pca.explained_variance_ratio_):.2f}")
    
    # 4. Save
    np.savez(save_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(f"Saved processed data to {save_path}")
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

if __name__ == "__main__":
    process_medical_data()
