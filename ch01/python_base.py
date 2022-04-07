import numpy as np


x = np.array([1, 2, 3])
print(x.__class__)
print(x.shape)
print(x.ndim)
print('-------------------')

W = np.array([[1, 2, 3], [4, 5, 6]])
print(W.shape)
print(W.ndim)
print('--------------------')

X = np.array([[0, 1, 2], [3, 4, 5]])
print(W + X)
print(W * X)
print('--------------------')

A = np.array([[1, 2], [3, 4]])
print(A * 10)

b = np.array([10, 20])
print(A * b)
