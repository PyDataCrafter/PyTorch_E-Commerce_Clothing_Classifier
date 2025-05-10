# E-Commerce Image Clothing Classifier on PyTorch
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Badges de CI/Cobertura se añadirán después de configurar CI -->

PyTorch-based machine learning model for e-commerce clothing classification. It automates the categorization of clothing items, helping customers find products easily and assisting in inventory management by classifying garments into types like shirts, trousers, shoes, etc. The dataset used for this project is the public FashionMNIST dataset

## Problem and motivation

The goal is to develop a machine learning model capable of classifying images of clothing items into 10 distinct categories (such as T-shirts, pants, dresses, etc.). This is useful for e-commerce applications such as automatic product categorization and improved inventory management.

## Data

- **Source:** [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- **Description:** The dataset consists of 60,000 training images and 10,000 28x28 pixel grayscale test images, distributed across 10 classes.
- **Classes:**
    - 0: T-shirt/top
    - 1: Trouser
    - 2: Pullover
    - 3: Dress
    - 4: Coat
    - 5: Sandal
    - 6: Shirt
    - 7: Sneaker
    - 8: Bag
    - 9: Ankle boot
- **Preprocessing:** Images are normalized using a mean and standard deviation of 0.5.
- **Split:** The standard split provided by the dataset is used (60k training, 10k test).

## Model Architecture

A simple Convolutional Neural Network (CNN) implemented in `cnn_clothes_classifier.ipynb` is used. The architecture consists of:

1. A convolutional layer (`nn.Conv2d`) with 16 3x3 filters, followed by a ReLU activation.
2. A Max Pooling layer (`nn.MaxPool2d`) with a 2x2 kernel.
3. An `nn.Flatten` layer to flatten the output.
4. A fully connected layer (`nn.Linear`) that maps features to the 10 output classes.

## Results

After training the model for 1 epochs with a learning rate of 0.001, the following results were obtained on the test set:

- **Accuracy:** ~0.89
- **Pérdida:** ~0.39

**Métricas por Clase (Precisión y Recall):**

| Clase        | Precisión | Recall |
|--------------|-----------|--------|
| T-shirt/top  | 0.7885    | 0.8949 |
| Trouser      | 0.9867    | 0.9670 |
| Pullover     | 0.8358    | 0.8149 |
| Dress        | 0.8584    | 0.9219 |
| Coat         | 0.7806    | 0.8650 |
| Sandal       | 0.9873    | 0.9359 |
| Shirt        | 0.7858    | 0.5649 |
| Sneaker      | 0.9248    | 0.9599 |
| Bag          | 0.9681    | 0.9720 |
| Ankle boot   | 0.9470    | 0.9649 |
