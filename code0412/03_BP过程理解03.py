# -*- coding: utf-8 -*-

import numpy as np

# 第一层：隐层具有3个神经元，输入是2个，所以该层的权重系数v形状为:[2,3]
v = np.asarray([
    [0.1, 0.2, 0.3],
    [0.15, 0.25, 0.35]
])
b1 = np.asarray([0.35])
# 第二层：输出层具有2个神经元，输入是3个，所以该层的权重系数w的形状为:[3,2]
w = np.asarray([
    [0.4, 0.45],
    [0.5, 0.55],
    [0.6, 0.65]
])
b2 = np.asarray([0.65])

lr = 0.5


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def training(x, y):
    # 1. 前向过程 - 计算预测结果  + 损失值
    h = sigmoid(np.dot(x, v) + b1)  # shape: [1,2] * [2,3] + [1] -> [1,3]
    o = sigmoid(h @ w + b2)  # shape: [1,3] * [3,2] + [1] -> [1,2]
    loss = 0.5 * np.power(o - y, 2).sum()
    # print(h1, h2, h3, o1, o2, loss)

    # 2. 反向过程 - 基于损失计算梯度 + 基于梯度更新参数 --> TODO: 自行补全

    return (loss, o)


if __name__ == '__main__':
    # 假定就5条样本
    _xs = [
        [5.0, 10.0],
        [2.0, 8.0],
        [3.0, 12.0],
        [3.0, 11.0],
        [16.0, 2.0]
    ]
    _ys = [
        [0.95, 0.12],
        [0.93, 0.01],
        [0.23, 0.77],
        [0.53, 0.45],
        [0.01, 0.99]
    ]
    n = 1
    for epoch in range(n):
        # 假定一个批次就一个样本
        _rs = []
        for i in range(len(_xs)):
            _x = np.asarray(_xs[i]).reshape((1, -1))
            _y = np.asarray(_ys[i]).reshape((1, -1))
            _rs.append(training(_x, _y))  # 使用_x和_y进行参数的更新
        if epoch == 0:
            print("=" * 100)
            print(_rs)
        elif epoch == n - 1:
            print("=" * 100)
            print(_rs)
