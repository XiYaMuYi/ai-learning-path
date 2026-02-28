# -*- coding: utf-8 -*-

import numpy as np

_w = np.asarray([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
_b = np.asarray([0.35, 0.65])

# 假定就一条样本
_x = np.asarray([5.0, 10.0])
_y = np.asarray([0.01, 0.99])

lr = 0.5


def w(i):
    # i下标完全按照ppt给定
    return _w[i - 1]

def b(i):
    return _b[i - 1]


def x(i):
    return _x[i - 1]


def y(i):
    return _y[i - 1]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def set_w(i, gd):
    _w[i - 1] = _w[i - 1] - lr * gd


def training():
    # 1. 前向过程 - 计算预测结果  + 损失值
    h1 = sigmoid(z=w(1) * x(1) + w(2) * x(2) + b(1))
    h2 = sigmoid(z=w(3) * x(1) + w(4) * x(2) + b(1))
    h3 = sigmoid(z=w(5) * x(1) + w(6) * x(2) + b(1))
    o1 = sigmoid(z=w(7) * h1 + w(9) * h2 + w(11) * h3 + b(2))
    o2 = sigmoid(z=w(8) * h1 + w(10) * h2 + w(12) * h3 + b(2))
    loss = 0.5 * ((y(1) - o1) ** 2 + (y(2) - o2) ** 2)
    print(h1, h2, h3, o1, o2, loss)

    # 2. 反向过程 - 基于损失计算梯度 + 基于梯度更新参数
    t1 = (o1 - y(1)) * o1 * (1 - o1)  # loss对于net_o1的梯度值
    t2 = (o2 - y(2)) * o2 * (1 - o2)  # loss对于net_o2的梯度值

    gds = [
        (t1 * w(7) + t2 * w(8)) * h1 * (1 - h1) * x(1),
        (t1 * w(7) + t2 * w(8)) * h1 * (1 - h1) * x(2),
        (t1 * w(9) + t2 * w(10)) * h2 * (1 - h2) * x(1),
        (t1 * w(9) + t2 * w(10)) * h2 * (1 - h2) * x(2),
        (t1 * w(11) + t2 * w(12)) * h3 * (1 - h3) * x(1),
        (t1 * w(11) + t2 * w(12)) * h3 * (1 - h3) * x(2),
        t1 * h1,
        t2 * h1,
        t1 * h2,
        t2 * h2,
        t1 * h3,
        t2 * h3
    ]
    for _i in range(len(gds)):
        set_w(_i + 1, gds[_i])

    # set_w(1, gd=(t1 * w(7) + t2 * w(8)) * h1 * (1 - h1) * x(1))
    # set_w(2, gd=(t1 * w(7) + t2 * w(8)) * h1 * (1 - h1) * x(2))
    # set_w(3, gd=(t1 * w(9) + t2 * w(10)) * h2 * (1 - h2) * x(1))
    # set_w(4, gd=(t1 * w(9) + t2 * w(10)) * h2 * (1 - h2) * x(2))
    # set_w(5, gd=(t1 * w(11) + t2 * w(12)) * h3 * (1 - h3) * x(1))
    # set_w(6, gd=(t1 * w(11) + t2 * w(12)) * h3 * (1 - h3) * x(2))
    #
    # set_w(7, gd=t1 * h1)
    # set_w(8, gd=t2 * h1)
    # set_w(9, gd=t1 * h2)
    # set_w(10, gd=t2 * h2)
    # set_w(11, gd=t1 * h3)
    # set_w(12, gd=t2 * h3)

    return (loss, o1, o2)


if __name__ == '__main__':
    print(_w)
    _r = training()
    print(_w)
    # for _j in range(1000):
    #     _r = training()
    # print(_r)
    # print(_w)
