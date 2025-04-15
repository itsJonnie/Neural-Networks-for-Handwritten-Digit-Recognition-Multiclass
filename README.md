# Neural Networks for Handwritten Digit Recognition (Multiclass)

This repository contains an implementation of a neural network for handwritten digit recognition using the MNIST dataset. The project demonstrates the application of deep learning techniques to classify handwritten digits (0-9) with high accuracy.

## Overview

The notebook in this repository walks through the process of building, training, and evaluating a neural network model that can recognize handwritten digits. This is a classic problem in machine learning and serves as an excellent introduction to neural networks and multi-class classification.

## Features

- Implementation of a neural network from scratch using NumPy
- Forward propagation and backward propagation algorithms
- Gradient descent optimization
- Softmax activation for multi-class classification
- Visualization of training metrics and model performance
- Detailed explanations of the mathematical concepts

## Dataset

The project uses the MNIST dataset, a large collection of handwritten digits that is commonly used for training and testing in the field of machine learning. The dataset contains 70,000 images of handwritten digits (60,000 for training and 10,000 for testing), each normalized and centered in a 28x28 pixel image.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow (for loading the MNIST dataset)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/itsJonnie/Neural-Networks-for-Handwritten-Digit-Recognition-Multiclass.git
```

2. Navigate to the repository directory:
```bash
cd Neural-Networks-for-Handwritten-Digit-Recognition-Multiclass
```

3. Open the Jupyter Notebook:
```bash
jupyter notebook "Neural_Networks_for_Handwritten_Digit_Recognition, Multiclass.ipynb"
```

## Implementation Details

The neural network is implemented with the following architecture:
- Input layer: 784 neurons (28x28 pixels)
- Hidden layer: Configurable number of neurons with ReLU activation
- Output layer: 10 neurons (one for each digit) with softmax activation

Key algorithms implemented:
- Forward propagation
- Backward propagation (gradient computation)
- Parameter updates using gradient descent
- Softmax function for multi-class classification
- Cross-entropy loss function

## Results

The model achieves high accuracy on the MNIST test set, demonstrating the effectiveness of neural networks for image classification tasks.

## Future Improvements

Potential enhancements for the project:
- Implementing additional optimization algorithms (Adam, RMSprop)
- Adding more hidden layers (deep neural network)
- Implementing convolutional layers for improved performance
- Adding regularization techniques to prevent overfitting
- Experimenting with different hyperparameters

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was inspired by the Machine Learning specialization course by Andrew Ng
- Thanks to the creators of the MNIST dataset for providing a valuable resource for machine learning education
