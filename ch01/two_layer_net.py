import numpy as np

from ch01.forward_net import Affine, Sigmoid


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O, = input_size, hidden_size, output_size

        # 初始化权重和偏置
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)

        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 生成神经网络的层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 将所有权重整理到列表中
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x


if __name__ == '__main__':
    x = np.random.randn(10, 2)
    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)
    print(np.max(s, axis=1))
