
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import json
from src.quantum_model import HybridModel

# --- CONFIGURATION ---
BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 20
LAYERS = 3

def load_data(filepath='processed_data.npz'):
    """Loads and converts numpy data to PyTorch Tensors."""
    try:
        data = np.load(filepath)
        X_train = torch.tensor(data['X_train'], dtype=torch.float32)
        y_train = torch.tensor(data['y_train'], dtype=torch.float32).unsqueeze(1) # Shape (N, 1)
        X_test = torch.tensor(data['X_test'], dtype=torch.float32)
        y_test = torch.tensor(data['y_test'], dtype=torch.float32).unsqueeze(1)
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print("Error: processed_data.npz not found. Run data_loader.py first.")
        sys.exit(1)

def calculate_accuracy(y_true, y_logits):
    """
    y_logits: Raw output from model [-inf, inf] (though technically bound by quantum [-1, 1])
    y_true: 0 or 1
    """
    # Sigmoid to get probability -> Round to get 0 or 1
    y_pred = torch.sigmoid(y_logits).round()
    correct = (y_pred == y_true).float().sum()
    return correct / y_true.shape[0]

def train(data_path='processed_data.npz'):
    print(f"--- HYBRID QML TRAINING (Data: {data_path}) ---")
    
    # 1. Prepare Data
    X_train, y_train, X_test, y_test = load_data(data_path)
    
    # [DEMO] Subsample removed for general training, or kept flexible
    # If using medical data (small), don't subsample
    if 'medical' not in data_path:
        # Subsample ONLY for the massive credit card dataset demo
        # For real training we would use all, but for this interactive session we keep it fast
        X_train = X_train[:500]
        y_train = y_train[:500]
        X_test = X_test[:200]
        y_test = y_test[:200]
    else:
        print("Using full medical dataset (small size).")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    model = HybridModel(n_layers=LAYERS)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Dynamic Weight Calculation
    n_pos = torch.sum(y_train == 1)
    n_neg = torch.sum(y_train == 0)
    # Avoid div by zero
    n_pos = max(n_pos, torch.tensor(1.0))
    pos_weight = n_neg / n_pos
    
    print(f"Class Balance: {n_pos.item()} Pos (1), {n_neg.item()} Neg (0)")
    print(f"Calculated pos_weight: {pos_weight.item():.4f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"Model Initialized. Trainable Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Training Loop
    best_acc = 0.0
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_acc = calculate_accuracy(y_test, test_outputs)
            
        avg_train_loss = total_loss / len(train_loader)
        
        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(test_loss.item())
        history["test_acc"].append(test_acc.item())
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "model_best.pth")
            
    print(f"\nTraining Complete. Best Test Accuracy: {best_acc:.4f}")
    
    with open("training_history.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'processed_data.npz'
    train(data_file)
