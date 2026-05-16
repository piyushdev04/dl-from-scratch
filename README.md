# DL From Scratch

implementing dl from scratch using first principles.

---

# About

This repository is a collection of deep learning concepts and architectures implemented completely from scratch using:

- Python
- NumPy
- Mathematics
- Matrix operations
- Basic deep learning fundamentals

No high-level training abstractions.  
No black-box frameworks doing everything automatically.

The purpose of this repository is to deeply understand:
- how neural networks actually learn
- how gradients flow
- how backpropagation works
- why transformers work
- how generative models are trained
- how reinforcement learning algorithms optimize behavior

Each chapter is written as an educational notebook with:
- mathematical intuition
- step-by-step implementations
- visual explanations
- forward & backward passes
- training logic
- optimization details

---

# Topics Implemented

## Foundations of Deep Learning

### 1. Perceptron
The basic building block of neural networks.

#### Concepts Covered
- Binary classification
- Linear decision boundaries
- Weighted sums
- Threshold activation

#### What’s Implemented
- perceptron training
- weight updates
- prediction logic
- binary classification examples

---

### 2. Activation Functions
Non-linear transformations used inside neural networks.

#### Concepts Covered
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- Vanishing gradients

#### What’s Implemented
- activation visualizations
- forward computation
- derivative understanding

---

### 3. Loss Functions
Measures how wrong a model’s predictions are.

#### Concepts Covered
- Mean Squared Error
- Cross Entropy Loss
- Binary classification loss
- Multi-class loss

#### What’s Implemented
- loss computation
- gradient intuition
- visualization of loss behavior

---

### 4. Backpropagation
Core algorithm behind neural network learning.

#### Concepts Covered
- Chain rule
- Gradient computation
- Partial derivatives
- Computational graphs

#### What’s Implemented
- manual gradient calculations
- backward propagation
- parameter updates

---

### 5. Feedforward Neural Networks
Multi-layer neural network implementation from scratch.

#### Concepts Covered
- Hidden layers
- Forward propagation
- Neural network architecture

#### What’s Implemented
- dense layers
- forward pass
- backward pass
- training loop

---

### 6. Weight Initialization
Proper initialization for stable training.

#### Concepts Covered
- Xavier Initialization
- He Initialization
- Gradient stability

#### What’s Implemented
- initialization experiments
- training comparisons

---

### 7. Optimizers
Optimization algorithms used during training.

#### Concepts Covered
- SGD
- Momentum
- RMSProp
- Adam

#### What’s Implemented
- optimizer update rules
- parameter optimization
- learning rate behavior

---

### 8. Regularization
Techniques to reduce overfitting.

#### Concepts Covered
- Dropout
- L2 Regularization
- Generalization

#### What’s Implemented
- dropout logic
- penalty terms
- regularized training

---

# Convolutional Neural Networks

### 9. CNN
Convolutional Neural Networks for image understanding.

#### Concepts Covered
- Convolution operation
- Feature extraction
- Filters & kernels

#### What’s Implemented
- convolution operations
- feature maps
- CNN intuition

---

### 10. Pooling
Dimensionality reduction in CNNs.

#### Concepts Covered
- Max Pooling
- Average Pooling
- Spatial reduction

#### What’s Implemented
- pooling operations
- feature reduction
- visualization

---

# Sequence Models

### 11. RNN
Recurrent Neural Networks for sequential data.

#### Concepts Covered
- Hidden states
- Sequence processing
- Temporal dependencies

#### What’s Implemented
- recurrent loops
- sequence prediction
- hidden state updates

---

### 12. LSTM
Long Short-Term Memory networks.

#### Concepts Covered
- Forget gate
- Input gate
- Output gate
- Long-term dependencies

#### What’s Implemented
- gating mechanisms
- memory cell operations
- sequence learning

---

### 13. GRU
Simplified recurrent architecture.

#### Concepts Covered
- Update gate
- Reset gate
- Efficient sequence learning

#### What’s Implemented
- GRU cell mechanics
- hidden state transitions

---

# Transformers & Attention

### 14. Attention
Neural attention mechanisms.

#### Concepts Covered
- Query
- Key
- Value
- Context vectors

#### What’s Implemented
- attention score computation
- weighted representations

---

### 15. Self-Attention
Core transformer mechanism.

#### Concepts Covered
- token relationships
- contextual understanding
- attention matrices

#### What’s Implemented
- self-attention blocks
- scaled dot-product attention

---

### 16. Positional Encoding
Injecting sequence order into transformers.

#### Concepts Covered
- sinusoidal encoding
- positional information

#### What’s Implemented
- positional vector generation
- transformer positional understanding

---

### 17. Embeddings
Dense vector representations of tokens.

#### Concepts Covered
- semantic representation
- embedding space
- vector similarity

#### What’s Implemented
- embedding lookup logic
- token vectorization

---

# Generative Models

### 18. Autoencoder
Neural network for representation learning.

#### Concepts Covered
- Encoder-decoder architecture
- Latent space
- Reconstruction loss

#### What’s Implemented
- encoding
- decoding
- reconstruction training

---

### 19. Variational Autoencoder (VAE)
Probabilistic generative model.

#### Concepts Covered
- Latent distributions
- KL Divergence
- Variational inference

#### What’s Implemented
- probabilistic latent sampling
- VAE loss
- reparameterization trick

---

### 20. GAN
Generative Adversarial Networks.

#### Concepts Covered
- Generator
- Discriminator
- Adversarial training

#### What’s Implemented
- generator training
- discriminator optimization
- adversarial loss

---

### 21. DCGAN
Deep Convolutional GAN implementation.

#### Concepts Covered
- convolutional generators
- image synthesis
- stable GAN training

#### What’s Implemented
- DCGAN architecture
- MNIST experiments
- image generation pipeline

---

# Reinforcement Learning

### 22. Q-Learning
Value-based reinforcement learning.

#### Concepts Covered
- Q-table
- Bellman Equation
- Exploration vs Exploitation

#### What’s Implemented
- reward optimization
- policy updates
- environment interaction

---

### 23. Policy Gradient
Policy-based reinforcement learning.

#### Concepts Covered
- policy optimization
- reward maximization
- gradient-based RL

#### What’s Implemented
- policy learning
- reward-driven updates
- stochastic optimization

---

# Tech Stack

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Project Structure

```bash
dl-from-scratch/
│
├── 01. Perceptron/
├── 02. Activation Functions/
├── 03. Loss Functions/
├── 04. Backpropagation/
├── 05. Feedforward NN/
├── 06. Weight Initialization/
├── 07. Optimizers/
├── 08. Regularization/
├── 09. CNN/
├── 10. Pooling/
├── 11. RNN/
├── 12. LSTM/
├── 13. GRU/
├── 14. Attention/
├── 15. Self-Attention/
├── 16. Positional Encoding/
├── 17. Embeddings/
├── 18. Autoencoder/
├── 19. Variational Autoencoder/
├── 20. GAN/
├── 21. DCGAN/
├── 22. Q-Learning/
└── 23. Policy Gradient/
```

---

# Why This Repository Exists

Most people use deep learning frameworks like this:

```python
model = NeuralNetwork()
model.fit(X, y)
```

without fully understanding:
- how gradients are computed
- how backpropagation works
- why transformers use attention
- how GANs actually learn
- how recurrent networks remember sequences
- how optimization updates parameters

This repository focuses on:
- intuition first
- implementation second
- frameworks later

---

# How to Run

Clone the repository:

```bash
git clone https://github.com/piyushdev04/dl-from-scratch.git
```

Move into the project:

```bash
cd dl-from-scratch
```

Install dependencies:

```bash
pip install numpy matplotlib notebook
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

---

# Connect

If this repository helped you learn something, consider giving it a ⭐

GitHub: https://github.com/piyushdev04/dl-from-scratch