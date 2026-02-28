# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F  # 导入函数式API，包含各种激活函数


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # 使用PyTorch的线性层替代手动定义权重和偏置（更规范）
        # 第一层：输入2个特征，输出3个特征
        self.fc1 = nn.Linear(in_features=2, out_features=3)
        # 第二层：输入3个特征，输出2个特征
        self.fc2 = nn.Linear(in_features=3, out_features=2)

        # 初始化权重和偏置，还原你原来的初始值
        with torch.no_grad():  # 初始化时不需要计算梯度
            self.fc1.weight = nn.Parameter(torch.tensor([
                [0.1, 0.15],
                [0.2, 0.25],
                [0.3, 0.35]
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.35, 0.35, 0.35]))
            self.fc2.weight = nn.Parameter(torch.tensor([
                [0.4, 0.5, 0.6],
                [0.45, 0.55, 0.65]
            ]))
            self.fc2.bias = nn.Parameter(torch.tensor([0.65, 0.65]))

        self.lr = 0.01  # 学习率直接用浮点数即可

    def forward(self, x, y=None):
        """
        规范的前向传播实现
        :param x: 输入数据
        :param y: 标签（可选），如果传入则计算损失
        :return: 损失值（如果y存在）、预测输出
        """
        # 使用PyTorch内置的sigmoid函数（两种方式任选）
        # 方式1：函数式API（推荐）
        h = F.sigmoid(self.fc1(x))  # 等价于 torch.sigmoid(self.fc1(x))
        # 方式2：模块式API（需要先实例化）：self.sigmoid = nn.Sigmoid() 然后 h = self.sigmoid(self.fc1(x))

        # 输出层
        o = F.sigmoid(self.fc2(h))

        # 计算损失（如果传入标签）
        loss = None
        if y is not None:
            loss = 0.5 * torch.pow(o - y, 2).sum()

        return loss, o

    def update(self):
        """参数更新：手动梯度下降（实际中推荐用优化器）"""
        with torch.no_grad():  # 更新参数时禁用梯度计算
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= self.lr * param.grad


def training(net, x, y):
    """训练步骤封装"""
    # 1. 清空之前的梯度（必须放在前向传播前）
    net.zero_grad()

    # 2. 前向过程 - 计算预测结果 + 损失值
    loss, o = net(x, y)

    # 3. 反向过程 - 计算梯度
    loss.backward()

    # 4. 更新参数
    net.update()

    return loss.item(), o.detach().numpy()  # 返回numpy值，方便查看


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

    # 转换为tensor（提前转换，避免循环内重复创建）
    xs = [torch.tensor(x).reshape(1, -1).float() for x in _xs]
    ys = [torch.tensor(y).reshape(1, -1).float() for y in _ys]

    # 初始化网络
    net = Network()
    n_epochs = 20

    for epoch in range(n_epochs):
        _rs = []
        for x, y in zip(xs, ys):
            loss_val, pred_val = training(net, x, y)
            _rs.append((loss_val, pred_val))

        # 打印首尾轮次结果
        if epoch == 0:
            print("=" * 100)
            print("第1轮训练结果：")
            for i, (loss, pred) in enumerate(_rs):
                print(f"样本{i + 1} - 损失: {loss:.6f}, 预测值: {pred.flatten()}")
        elif epoch == n_epochs - 1:
            print("=" * 100)
            print(f"第{n_epochs}轮训练结果：")
            for i, (loss, pred) in enumerate(_rs):
                print(f"样本{i + 1} - 损失: {loss:.6f}, 预测值: {pred.flatten()}")