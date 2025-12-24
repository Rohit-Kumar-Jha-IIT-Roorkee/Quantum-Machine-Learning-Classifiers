
import torch
import numpy as np
import sys
from sklearn.metrics import classification_report, confusion_matrix
from src.quantum_model import HybridModel
from src.train_hybrid import load_data, LAYERS

def evaluate():
    print("--- EVALUATING BEST QUANTUM MODEL ---")
    
    # 1. Load Data
    _, _, X_test, y_test = load_data()
    
    # 2. Load Model
    model = HybridModel(n_layers=LAYERS)
    try:
        model.load_state_dict(torch.load("model_best.pth"))
        print("Loaded 'model_best.pth'")
    except FileNotFoundError:
        print("Error: model_best.pth not found.")
        return

    model.eval()
    
    # 3. Predict
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.sigmoid(logits).round().numpy()
        y_true = y_test.numpy()
        
    # 4. Metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, preds))
    
    print("\nClassification Report:")
    print(classification_report(y_true, preds, target_names=["Non-Fraud", "Fraud"]))

if __name__ == "__main__":
    evaluate()
