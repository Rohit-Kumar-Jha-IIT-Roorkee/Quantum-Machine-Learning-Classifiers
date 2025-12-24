
import torch
import pennylane as qml
from pennylane import numpy as np
from src.quantum_model import HybridModel # To get structure
from src.train_hybrid import load_data, LAYERS

def run_noisy_simulation(data_path='processed_data.npz'):
    print(f"--- NOISY QUANTUM SIMULATION (Data: {data_path}) ---")
    
    # 1. Load Data & Trained Model
    _, _, X_test, y_test = load_data(data_path)
    # Use a subset of test data to speed up simulation (Mixed state sim is slow)
    X_test_sub = X_test[:1000] 
    y_test_sub = y_test[:1000]
    
    clean_model = HybridModel(n_layers=LAYERS)
    try:
        clean_model.load_state_dict(torch.load("model_best.pth"))
        print("Loaded trained weights from 'model_best.pth'")
    except FileNotFoundError:
        print("Error: model_best.pth not found.")
        return

    # Extract learned weights from the Torch layer
    # keys in state_dict: "q_layer.weights"
    learned_weights = clean_model.q_layer.weights.detach().numpy()
    
    # 2. Define Noisy Circuit
    n_qubits = 4
    dev_noisy = qml.device("default.mixed", wires=n_qubits)
    
    def make_noisy_qnode(noise_prob):
        @qml.qnode(dev_noisy)
        def noisy_circuit(inputs, weights):
            # Feature Map
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            # Noise after State Prep
            for w in range(n_qubits):
                qml.DepolarizingChannel(noise_prob, wires=w)
            
            # Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # Noise after Ansatz (simulating gate errors)
            for w in range(n_qubits):
                qml.DepolarizingChannel(noise_prob, wires=w)
                
            return qml.expval(qml.PauliZ(0))
        return noisy_circuit

    # 3. Evaluation Loop over Noise Levels
    noise_levels = [0.00, 0.02, 0.05, 0.10]
    print(f"\nEvaluating on {len(X_test_sub)} samples...")
    print(f"{'Noise (p)':<10} | {'Accuracy':<10} | {'Recall (Fraud)':<15}")
    print("-" * 40)
    
    for p in noise_levels:
        qnode = make_noisy_qnode(p)
        
        preds = []
        for i in range(len(X_test_sub)):
            # Classical Input -> Noisy QNode -> Logic -> Pred
            logits = qnode(X_test_sub[i].numpy(), learned_weights)
            pred = 1 if (torch.sigmoid(torch.tensor(logits)).item() > 0.5) else 0
            preds.append(pred)
            
        preds = np.array(preds)
        y_true = y_test_sub.numpy().flatten()
        
        acc = np.mean(preds == y_true)
        # Calculate Recall manually for Fraud (1)
        fraud_mask = (y_true == 1)
        if np.sum(fraud_mask) > 0:
            recall = np.sum(preds[fraud_mask] == 1) / np.sum(fraud_mask)
        else:
            recall = 0.0
            
        print(f"{p:<10.2f} | {acc:<10.4f} | {recall:<15.4f}")

if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'processed_data.npz'
    run_noisy_simulation(data_file)
