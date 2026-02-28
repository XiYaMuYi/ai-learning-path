# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # 第一层：隐层具有3个神经元，输入是2个，所以该层的权重系数v形状为:[2,3]
        self.v = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35]
        ], requires_grad=True)
        self.b1 = torch.tensor([0.35])
        # 第二层：输出层具有2个神经元，输入是3个，所以该层的权重系数w的形状为:[3,2]
        self.w = nn.Parameter(torch.tensor([
            [0.4, 0.45],
            [0.5, 0.55],
            [0.6, 0.65]
        ]))

        self.b2 = torch.tensor([0.65])

        self.lr = torch.tensor(0.01)

    def forward(self, x, y):
        # 1. 前向过程 - 计算预测结果  + 损失值
        h = sigmoid(torch.matmul(x, self.v) + self.b1)  # shape: [1,2] * [2,3] + [1] -> [1,3]
        o = sigmoid(h @ self.w + self.b2)  # shape: [1,3] * [3,2] + [1] -> [1,2]
        loss = 0.5 * torch.pow(o - y, 2).sum()

        return loss, o

    def update(self):
        self.v.data.add_(- self.lr * self.v.grad)
        self.w.data.add_(- self.lr * self.w.grad)


def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


net = Network()


def training(x, y):
    # 1. 前向过程 - 计算预测结果  + 损失值
    loss, o = net(x, y)

    # 2. 反向过程 - 基于损失计算梯度 + 基于梯度更新参数
    loss.backward()  # 触发PyTorch自动计算每个参数w的梯度
    net.update()
    net.zero_grad()

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
    n = 20
    for epoch in range(n):
        # 假定一个批次就一个样本
        _rs = []
        for i in range(len(_xs)):
            _x = torch.tensor(_xs[i]).reshape((1, -1))
            _y = torch.tensor(_ys[i]).reshape((1, -1))
            _rs.append(training(_x, _y))  # 使用_x和_y进行参数的更新
        if epoch == 0:
            print("=" * 100)
            print(_rs)
        elif epoch == n - 1:
            print("=" * 100)
            print(_rs)
