
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history(history_path="training_history.json", save_path="plot_training.png"):
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    plt.plot(epochs, history["test_loss"], 'r--', label='Test Loss')
    plt.title('Training & Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["test_acc"], 'g-', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training plot to {save_path}")
    plt.close()

def plot_robustness(save_path="plot_robustness.png"):
    # Data from our previous experiments
    noise_levels = ['0.0 (Clean)', '0.05', '0.10', '0.20']
    
    # Accuracy values
    # Hybrid Quantum: Stable at 0.6570
    acc_quantum = [0.6570, 0.6570, 0.6570, 0.6570] 
    
    # Logistic Regression: 0.51 -> 0.53 -> 0.56 -> 0.60 (Acc increases but Recall crashes)
    acc_lr = [0.5098, 0.5265, 0.5604, 0.6066]
    
    # Recall (Fraud) values - The critical metric
    # Hybrid Quantum: Stable at 0.6979
    rec_quantum = [0.6979, 0.6979, 0.6979, 0.6979]
    
    # Logistic Regression: 0.94 -> 0.87 -> 0.81 -> 0.70
    rec_lr = [0.9416, 0.8764, 0.8146, 0.7025]

    x = np.arange(len(noise_levels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    
    # Plotting Recall Comparison (Most meaningful for this fraud detection task)
    plt.bar(x - width/2, rec_quantum, width, label='Hybrid Quantum (Recall)', color='purple')
    plt.bar(x + width/2, rec_lr, width, label='Logistic Reg (Recall)', color='orange')
    
    plt.xlabel('Noise Level (std/p)')
    plt.ylabel('Recall (Fraud Detection)')
    plt.title('Robustness: Fraud Recall vs Input/Quantum Noise')
    plt.xticks(x, noise_levels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(save_path)
    print(f"Saved robustness plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_robustness()
    plot_training_history()
