# IIT Roorkee QML Project: Quantum Classifiers

## Project Overview
This project implements a **Hybrid Quantum-Classical Classifier** for **Credit Card Fraud Detection** (`card.csv`). It benchmarks the robustness of a Variational Quantum Circuit (VQC) against standard classical models (Logistic Regression, Random Forest, MLP) under noisy conditions, simulating NISQ hardware constraints.

## Key Features
- **Data Pipeline**: Dimensionality reduction (PCA) from 8 to 4 features (`src/data_loader.py`).
- **Hybrid Architecture**: 4-qubit Quantum Circuit (PennyLane) + PyTorch Classical Layer (`src/quantum_model.py`).
- **Robustness Analysis**: Evaluation under simulated Depolarizing Noise (`src/noisy_evaluation.py`).
- **Benchmarking**: Comparison with Classical Baselines (`src/classical_baselines.py`).

## Results Summary

### 1. Robustness
The Hybrid QML model demonstrated superior stability in Recall compared to classical models when subjected to input noise.

![Robustness Plot](src/plot_robustness.png)

| Model | Noise (p=0.10) | Recall (Fraud) | Status |
|-------|----------------|----------------|--------|
| **Hybrid QML** | **0.10** | **~69.8%** | **Stable** |
| Logistic Reg | 0.10 | ~81.5% | Degraded (-13%) |
| Random Forest | 0.10 | ~34.7% | Unstable |

### 2. Training Performance (Demo)
*Note: Demo run on subset for visualization.*
![Training Plot](src/plot_training.png)

## Repository Structure
- `src/data_loader.py`: Pre-processing & PCA.
- `src/quantum_model.py`: Hybrid QML Model definition.
- `src/train_hybrid.py`: Training loop (PyTorch).
- `src/noisy_evaluation.py`: Quantum Noise Simulation (NISQ).
- `src/classical_baselines.py`: Classical Model benchmarks.
- `src/visualize_results.py`: Plot generation.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Full Pipeline**:
    ```bash
    # 1. Process Data
    python src/data_loader.py

    # 2. Train Hybrid Model (Demo)
    python -m src.train_hybrid

    # 3. Evaluate on Quantum Noise
    python -m src.noisy_evaluation

    # 4. Generate Plots
    python src/visualize_results.py
    ```
