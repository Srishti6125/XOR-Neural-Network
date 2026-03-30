# 🔥 XOR Neural Network — NumPy vs Keras/TensorFlow 
### *All frameworks use the same math, but differ in how much control and abstraction they provide.*

---

## ⭐ Overview  
A comparative implementation of a simple neural network to solve the **XOR problem** using three approaches:

- **NumPy (from scratch)** 
- **TensorFlow / Keras**
- **PyTorch**

This project demonstrates how neural networks can learn such patterns using different frameworks and levels of abstraction.

---

## 🎯 Why XOR?

XOR is the classical example that proves the necessity of:

- nonlinear activation functions
- hidden layers
- multi-layer learning
  
It's a simple yet powerful demonstration of what neural networks can do beyond linear models.

---

## 🧠 Model Architecture

Input Layer (2 neurons) --> Hidden Layer (2–4 neurons, Tanh) --> Output Layer (1 neuron, Sigmoid)

---

## 1. NumPy Implementation

- Manual forward propagation  
- Manual backpropagation (chain rule)  
- Gradient descent updates  

Helps in understanding: Chain rule, Gradient flow, Internal working of neural networks

---

## 2. TensorFlow / Keras Implementation

- High-level Sequential API  
- Built-in training loop (`model.fit`)  
- EarlyStopping callback  

Advantages : Minimal code, Faster development, Production-ready features

---

## 3. PyTorch Implementation

- Custom model using `nn.Module`  
- Manual training loop  
- Automatic differentiation (autograd)  
- Custom early stopping logic 

Advantages: Flexible and intuitive, Fine-grained control, Research-friendly

---

## ⚖️ Comparison

| Feature         | NumPy  | TensorFlow | PyTorch   |
|----------------|--------|------------|-----------|
| Control        | Full   | Low        | High      |
| Ease of Use    | Low    | High       | Medium    |
| Backpropagation| Manual | Automatic  | Automatic |
| Flexibility    | High   | Medium     | High      |

---

## 📌 Key Learnings

- Neural networks solve non-linear problems like XOR
- Backpropagation is the core of learning
- All frameworks use the same underlying math
- Difference lies in abstraction and control
  
---

## 🔬 Technical Insights

- XOR cannot be solved using a single linear layer  
- Non-linear activations (Tanh/ReLU) are essential  
- Sigmoid is used for binary classification output  
- Learning rate significantly affects convergence  

---

## 🚀 Future Improvements

- Add visualization of decision boundary 📈
- Extend to larger datasets
- Implement validation-based early stopping
- Compare performance metrics (accuracy, loss curves)

---
