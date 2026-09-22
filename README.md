# FNN vs CNN: Accuracy and Noise Robustness on MNIST

A head-to-head comparison of a Feedforward Neural Network and a Convolutional Neural Network on MNIST digit classification — both hyperparameter-tuned, then stress-tested against injected noise.

## What's inside

- `1_fnn_mnist.ipynb` — a Dense/FNN classifier (flattened 784-pixel input) tuned via `GridSearchCV` (through `scikeras`) over layer widths, optimizer, and batch size.
- `2_cnn_mnist.ipynb` — a Conv2D/MaxPooling CNN, tuned the same way over filter counts, dense units, and optimizer.
- Both models are evaluated on a clean test set, then again after injecting impulsive (salt-and-pepper-style) noise at increasing intensities, to measure how gracefully each architecture degrades.
- A final comparison of clean vs. noisy accuracy explains *why* the CNN outperforms the FNN: convolution preserves 2D spatial structure that flattening destroys.

## Tech stack

Python, TensorFlow/Keras, scikeras, scikit-learn, Matplotlib, NumPy

## Run it

```bash
pip install tensorflow scikeras scikit-learn matplotlib numpy
jupyter notebook 1_fnn_mnist.ipynb
jupyter notebook 2_cnn_mnist.ipynb
```

MNIST downloads automatically via `keras.datasets.mnist`.
