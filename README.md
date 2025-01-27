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
- **Loss functions**:
  - CrossEntropyLoss
  - Binary Cross-Entropy Loss (BCELoss)
  - Mean Squared Error (MSELoss)
- **Utility functions**:
  - `argmax`: Find the index of the maximum value in a tensor
  - `from_list`: Convert a Python list of numerical values into a list of `Value` class instances, enabling autograd functionality
  - `to_list`: Convert a list of `Value` class instances back into a Python list of numerical values
  - `one_hot_encode`: Create one-hot encoded vectors from categorical data
- **Example training**:
  - Examples of `Binary Classification`, `Multiclass Classification`, `Regression` for demonstrating training

## TODO

- Refactor the codebase
- More activation functions
- Support common optimizers (SGD, Adam)
- Add examples with toy datasets
- Training loop function
