import numpy as np

c = np.array([1, 0, 0, 0, 0, 0, 0])
W = np.random.rand(7, 3)
h = np.dot(c, W)

print(h)
