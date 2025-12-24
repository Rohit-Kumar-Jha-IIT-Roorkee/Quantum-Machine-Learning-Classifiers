
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_process_data(filepath='card.csv', test_size=0.2, random_state=42):
    """
    1. Loads dataset
    2. Separates features (X) and target (y)
    3. Normalizes Features to [0, 1]
    4. Applies PCA to reduce to 4 features
    5. Returns Train/Test splits
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Separate Target
    y = df['fraud'].values
    X = df.drop(columns=['fraud'])
    
    # 0. Robust Data Cleaning (Best Practices)
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate rows.")
        
    df.dropna(inplace=True)
    if len(df) < initial_count: # check against previous if dropna did anything
        print(f"Removed {initial_count - len(df) - (initial_count - len(df))} rows with missing values.") # Simplified logic needed here really
        
    # Re-separate after cleaning
    y = df['fraud'].values
    X = df.drop(columns=['fraud'])
    
    print(f"Original shape: {X.shape}")
    
    # 1. Normalization (Critical for PCA and Quantum Embedding)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    
    # 2. PCA: Reduce to 4 features
    print("Applying PCA to reduce dimensions to 4...")
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained Variance Ratio by 4 components: {pca.explained_variance_ratio_}")
    print(f"Total Variance Explained: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_process_data()
    
    # Save processed data for quick loading in later steps
    np.savez("processed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("Processed data saved to processed_data.npz")
