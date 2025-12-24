
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score

def run_classical_robustness(data_path='processed_data.npz'):
    print("--- CLASSICAL ROBUSTNESS ANALYSIS (Input Noise) ---")
    
    # 1. Load Data
    try:
        data = np.load(data_path)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return

    # 2. Train Models (Freshly trained to ensure clean state)
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        "MLP (Small)": MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=500, random_state=42)
    }
    
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        model.fit(X_train, y_train)
        print("Done.")

    # 3. Evaluation Loop over Noise Levels
    # We add Gaussian noise to the normalized inputs (which are roughly -1 to 1 or 0 to 1)
    noise_sigmas = [0.00, 0.05, 0.10, 0.20, 0.50]
    
    print(f"\nEvaluating on {len(X_test)} samples...")
    print(f"{'Model':<20} | {'Noise (std)':<12} | {'Accuracy':<10} | {'Recall (Fraud)':<15}")
    print("-" * 65)
    
    results = {name: [] for name in models}
    
    for sigma in noise_sigmas:
        # Add noise to test set
        if sigma > 0:
            noise = np.random.normal(0, sigma, X_test.shape)
            X_test_noisy = X_test + noise
        else:
            X_test_noisy = X_test
            
        for name, model in models.items():
            y_pred = model.predict(X_test_noisy)
            
            acc = accuracy_score(y_test, y_pred)
            # recall_score for binary classification (pos_label=1 is fraud)
            rec = recall_score(y_test, y_pred, pos_label=1)
            
            print(f"{name:<20} | {sigma:<12.2f} | {acc:<10.4f} | {rec:<15.4f}")
            results[name].append((sigma, acc, rec))
        print("-" * 65)

if __name__ == "__main__":
    run_classical_robustness()
