# mikegrad

This project implements backpropagation from scratch with additional enhancements and functionality. It is inspired from Andrej Karpathy's ["Neural Networks: Zero to Hero - Micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0) video.

## Features

- **Autograd engine**: Automatic differentiation for gradient computation
- **Backpropagation**: Implementation from scratch
- **Basic neural network components**: Build your own neural networks
- **Activation functions**:
  - Sigmoid
  - Tanh
  - Softmax
  - ReLU
  - LeakyReLU
- **Loss functions**:
  - CrossEntropyLoss
  - Binary Cross-Entropy Loss (BCELoss)
  - Mean Squared Error (MSELoss)
- **Optimizers**:
  - SGD
  - Adam
  - AdamW
- **Utility functions**:
  - `argmax`: Find the index of the maximum value in a tensor
  - `from_list`: Convert a Python list of numerical values into a list of `Value` class instances
  - `to_list`: Convert a list of `Value` class instances back into a Python list of numerical values
  - `from_numpy`: Convert a NumPy array into a list of `Value` class instances
  - `to_numpy`: Convert a list of `Value` class instances back into a NumPy
  - `one_hot_encode`: Create one-hot encoded vectors from categorical data
- **Example training**:
  - Examples of `Binary Classification`, `Multiclass Classification`, `Regression` for demonstrating training

## TODO

- Refactor the codebase
- Add a regression example with toy dataset
- Training loop function
