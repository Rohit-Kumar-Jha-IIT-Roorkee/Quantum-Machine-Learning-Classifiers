
import pennylane as qml
from pennylane import numpy as np # PennyLane's wrapper for autograd
import torch
import torch.nn as nn

# --- QUANTUM DEVICE CONFIGURATION ---
# Concept: The "Backend" (Lecture 1.2 "Intro to Quantum Circuits")
# We use a simulator ('default.qubit') for fast prototyping.
# On real hardware, this would be 'qiskit.ibmq', 'honeywell.hqs', etc.
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# --- VARIATIONAL QUANTUM CIRCUIT (VQC) ---
@qml.qnode(dev, interface="torch") # Interface='torch' allows backprop!
def quantum_circuit(inputs, weights):
    """
    The 'Kernel' of our Quantum Model.
    Args:
        inputs: 4-dimensional vector from PCA (Classical Data).
        weights: Trainable parameters for the quantum gates.
    """
    
    # 1. FEATURE MAP (State Preparation)
    # Concept: "Encoding Classical Data into Quantum States" (Lecture 4.1 in your playlist)
    # We use AngleEmbedding to rotate qubits based on input feature values.
    # Rotation='Y' is standard for normalized data [0, 1] scaled to [0, pi].
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    
    # 2. ANSATZ (The "Neural Network" layers)
    # Concept: "Hardware Efficient Ansatze" (Lecture 9.2 in your playlist)
    # StronglyEntanglingLayers applies rotations + CNOT entangling gates.
    # This creates the "interference" patterns the model learns to manipulate.
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # 3. MEASUREMENT
    # Concept: Extracting information back to classical world.
    # We measure the expectation value <Z> of the first qubit.
    # Output range: [-1, 1].
    return qml.expval(qml.PauliZ(0))

# --- HYBRID PYTORCH MODEL ---
class HybridModel(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        
        # Concept: Weight Initialization
        # Determining the shape of weights required by StronglyEntanglingLayers.
        # Shape: (n_layers, n_qubits, 3 parameters per qubit rotation)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Concept: "Quantum Layer" in a Classical Net (Lecture 9.1 "Intro to QNNs")
        # TorchLayer automatically handles the forward/backward pass.
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x):
        # Input x shape: (batch_size, 4)
        
        # Pass through quantum circuit
        x = self.q_layer(x) # Output shape: (batch_size, 1) or (batch_size,)
        
        # Reshape to ensure (batch_size, 1)
        x = x.view(-1, 1)
        
        # Concept: Mapping Quantum Output [-1, 1] to Probability [0, 1]
        # We start with the raw "logits" from the quantum circuit.
        # Since we use BCEWithLogitsLoss during training, we return raw logits here.
        # But conceptually, Sigmoid(x) would be the probability.
        return x
        
if __name__ == "__main__":
    # Quick Test
    model = HybridModel(n_layers=2)
    sample_input = torch.rand(5, 4) # Batch of 5 samples, 4 features
    output = model(sample_input)
    print(f"Input Shape: {sample_input.shape}")
    print(f"Output Shape: {output.shape}")
    print("Example Outputs (Raw Logits):\n", output.detach().numpy())
