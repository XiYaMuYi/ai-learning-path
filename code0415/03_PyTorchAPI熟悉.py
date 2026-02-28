# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


def tt01():
    """
    多维矩阵的运算
    :return:
    """
    a = torch.randn(4, 2)
    b = torch.randn(2, 3)
    c = torch.matmul(a, b)
    print(c.shape)  # [4,3]

    a = torch.randn(4, 5, 2)
    b = torch.randn(4, 2, 3)
    c = torch.matmul(a, b)
    print(c.shape)  # [4,5,3]

    d = []
    for i in range(b.shape[0]):
        ai, bi = a[i], b[i]
        ci = torch.matmul(ai, bi)  # [5,3]
        d.append(ci)
    d = torch.stack(d, dim=0)
    print(d.shape)  # [4,5,3]
    print(torch.max(torch.abs(d - c)))  # 0.0

    a = torch.randn(4, 5, 2)
    b = torch.randn(2, 3)
    c = torch.matmul(a, b)
    print(c.shape)  # [4,5,3]

    d = []
    for i in range(a.shape[0]):
        ai, bi = a[i], b
        ci = torch.matmul(ai, bi)  # [5,3]
        d.append(ci)
    d = torch.stack(d, dim=0)
    print(d.shape)  # [4,5,3]
    print(torch.max(torch.abs(d - c)))  # 0.0

    linear = nn.Linear(2, 3)
    print("linear的参数:")
    for name, param in linear.named_parameters():
        print(name, param.shape)
    a = torch.randn(4, 5, 2)
    c = linear(a)
    print(c.shape)  # [4,5,3]


if __name__ == '__main__':
    tt01()
