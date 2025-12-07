# 🔥 XOR Neural Network — NumPy vs Keras/TensorFlow  
### *A side-by-side exploration of low-level neural network math vs high-level deep learning frameworks.*

---

## ⭐ Overview  
This project demonstrates how a neural network learns the classic **XOR logical function**, using two different approaches:

- **🔧 Pure NumPy** — manual forward pass, backpropagation, and gradient descent  
- **⚡ Keras + TensorFlow** — automatic differentiation, Adam optimizer, and concise model design  

The goal is to understand both how neural networks work internally **and** how deep learning frameworks simplify training.

---

## 🧠 Architecture  
Both implementations use the same neural network structure:

Input Layer: 2 neurons
Hidden Layer: 2 neurons (tanh activation)
Output Layer: 1 neuron (sigmoid activation)
Total trainable parameters: **9**

XOR is **not linearly separable**, so a hidden layer with nonlinearity is required for successful learning.

---

## 🚀 NumPy Implementation (From Scratch)

The NumPy version includes:
- Manual weight initialization  
- Forward propagation across layers  
- Binary cross-entropy loss computation  
- Full backpropagation using the chain rule  
- Gradient descent weight updates  
- Learning rate exploration (0.01 → 1.0)

**Sample Output:**

Predicted probabilities:
[[0.003], [0.998], [0.998], [0.003]]

Rounded predictions:
[0, 1, 1, 0]


This implementation develops deep intuition for how neural networks truly learn.

---

## ⚡ Keras/TensorFlow Implementation

The same architecture is recreated using Keras:
model = Sequential([ Dense(2, activation='tanh', input_dim=2),
    Dense(1, activation='sigmoid')
])
Using the Adam optimizer and binary cross-entropy loss, the model quickly learns XOR with very high confidence.

---

🎨 Decision Boundary Visualization
- This project includes a visualization script that shows:
- XOR regions predicted by the model
- The true XOR data points
- The curved nonlinear boundary learned by the network
This makes the network’s learned function easy to interpret.

---

🎯 Why XOR?

XOR is the classical example that proves the necessity of:
- nonlinear activation functions
- hidden layers
- multi-layer learning
It's a simple yet powerful demonstration of what neural networks can do beyond linear models.

---
