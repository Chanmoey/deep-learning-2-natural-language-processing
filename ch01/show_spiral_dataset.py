import numpy as np

from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape)
print('t', t.shape)
mark = ['^', '*', 'x']
mark_idx = np.argwhere(t == 1)[:, 1]

for i in range(len(mark_idx)):
    plt.scatter(x[i][0], x[i][1], marker=mark[mark_idx[i]])
plt.show()


