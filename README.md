# DLFast: Simplifying Deep Learning with Python



**dlfast** is a Python deep learning library that's revolutionizing the way we build neural networks. Designed for both beginners and experienced data scientists, dlfast empowers users to create intricate deep learning models effortlessly. With its intuitive API and high-level abstractions, you can construct complex networks with just a few lines of code. The library offers modularity, efficiency, and extensive documentation, making it a powerful tool for anyone looking to harness the potential of deep learning.



<!-- 
## Overview
**dlfast** is a Python deep learning library that streamlines the creation and training of neural networks. It caters to both beginners and experienced data scientists, offering a straightforward approach to building complex deep learning models with minimal code. This documentation outlines the library's capabilities and provides guidance on usage.

## Function: `ANN` - Create and Train Neural Networks -->


## ANN

The `ANN` function is a versatile tool for building and training neural networks with ease. It provides options for various tasks, data preprocessing, model architecture, optimization, and more. Whether you're working on binary classification, multiclass classification, or regression, `ANN` simplifies the process and offers flexibility in customization.


### Parameters
- `X` (numpy.ndarray): The feature data.
- `y` (numpy.ndarray): The target data.
- `task` (str, optional): Specifies the task type. Options are `'binary'` (default), `'multiclass'`, or `'regression'`.
- `scaler` (str, optional): Specifies the data scaling method. Options are `'standard'` (default), `'robust'`, or `'minmax'`.
- `epochs` (int, optional): The number of training epochs (default: 10).
- `optimizer` (str, optional): Specifies the optimizer. Options are `'Adam'` (default) or `'SGD'`.
- `activation` (str, optional): Specifies the activation functions for input and hidden layers. Format: `'input_activation/hidden_activation'` (default: `'relu'`).
- `layers` (tuple, optional): Defines the neural network architecture as a tuple of layer sizes (default: `(100, 50, 3)`).
- `early_stop` (bool, optional): Enables or disables early stopping (default: `True`).
- `dropout` (float, optional): Dropout rate for regularization (default: `None`).
- `save` (bool, optional): Specifies whether to save the model as an h5 file (default: `False`).

### Returns
- `model` (tensorflow.python.keras.engine.sequential.Sequential): The trained neural network model.
- `evaluation_results` (float): The evaluation results (e.g., accuracy for classification, mean squared error for regression).

### Example Usage

```python
from dlfast import ANN
```


```python
# Assuming you have your X and y data loaded from your dataset
X, y = load_data()
```

**Binary Classification Example:**

```python
model, results = ANN(X, y, task='binary', scaler='standard', epochs=20, optimizer='Adam', activation='relu/sigmoid', early_stop=True, dropout=0.2, save=True)
```

**Multiclass Classification Example:**

```python
model, results = ANN(X, y, task='multiclass', scaler='standard', epochs=20, optimizer='Adam', activation='relu/softmax', early_stop=True, dropout=0.2, save=True)
```


**Regression Example:**
```python
model, results = ANN(X, y, task='regression', scaler='standard', epochs=20, optimizer='Adam', activation='relu/linear', early_stop=True, dropout=0.2, save=True)
```