🔥 XOR Neural Network — NumPy vs Keras/TensorFlow  
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

- **Input Layer:** 2 neurons  
- **Hidden Layer:** 2 neurons (tanh activation)  
- **Output Layer:** 1 neuron (sigmoid activation)  

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

### 📊 NumPy XOR Results

| **Inputs** | **True Label** | **Predicted Probability** | **Rounded Prediction** |
|------------|----------------|----------------------------|--------------------------|
| `[0, 0]`   | `0`            | `0.0035`                   | `0`                      |
| `[0, 1]`   | `1`            | `0.9978`                   | `1`                      |
| `[1, 0]`   | `1`            | `0.9978`                   | `1`                      |
| `[1, 1]`   | `0`            | `0.0031`                   | `0`                      |

This implementation develops deep intuition for how neural networks truly learn by exposing every step of the math.

---

## ⚡ Keras/TensorFlow Implementation

The same architecture is recreated using Keras:

model = Sequential([ Dense(2, activation='tanh', input_dim=2), Dense(1, activation='sigmoid') ])

Using the Adam optimizer and binary cross-entropy loss, the model quickly learns XOR with very high confidence.

### 📊 Keras/TensorFlow XOR Results

| **Inputs** | **True Label** | **Predicted Probability** | **Rounded Prediction** |
|------------|----------------|----------------------------|--------------------------|
| `[0, 0]`   | `0`            | `6.37e-06`                 | `0`                      |
| `[0, 1]`   | `1`            | `9.9998e-01`               | `1`                      |
| `[1, 0]`   | `1`            | `9.9999e-01`               | `1`                      |
| `[1, 1]`   | `0`            | `5.37e-06`                 | `0`                      |

---

## 🆚 NumPy vs Keras/TensorFlow — Result Comparison

| **Aspect**               | **NumPy Implementation**                          | **Keras/TensorFlow Implementation**               |
|--------------------------|---------------------------------------------------|---------------------------------------------------|
| **Architecture**         | 2 → 2 (tanh) → 1 (sigmoid)                        | 2 → 2 (tanh) → 1 (sigmoid)                        |
| **Trainable Parameters** | 9                                                 | 9                                                 |
| **Training Logic**       | Manual forward + backprop + gradient descent      | `model.fit()` with Adam optimizer                 |
| **Final Accuracy**       | 100%                                              | 100%                                              |
| **Output Probabilities** | ~0.003 / ~0.998 (confident)                       | ~0.000006 / ~0.99998 (extremely confident)        |
| **Purpose**              | Deep understanding of internal mechanics          | Fast, clean, production-style training            |

Both models learn the **same XOR mapping** and achieve perfect classification, but they represent two different levels of abstraction:

- **NumPy →** full control, full math, hands-on understanding  
- **Keras/TF →** clean, scalable, and ready for real-world applications  

---

## 🎨 Decision Boundary Visualization
This project includes a visualization script that shows:
  
- XOR regions predicted by the model
- The true XOR data points
- The curved nonlinear boundary learned by the network
  
This makes the network’s learned function easy to interpret.

---

## 🎯 Why XOR?

XOR is the classical example that proves the necessity of:

- nonlinear activation functions
- hidden layers
- multi-layer learning
  
It's a simple yet powerful demonstration of what neural networks can do beyond linear models.

---
