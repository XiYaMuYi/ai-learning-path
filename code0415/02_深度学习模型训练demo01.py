# -*- coding: utf-8 -*-
import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split

from common.network import ClassifyNetworkV0415 as ClassifyNetwork
from common.numpy_dataset import build_dataloader
from common.stop_early import StopEarlyWithBig

warnings.filterwarnings('ignore')

# 1. 数据加载
ds = datasets.load_iris(return_X_y=False)
save_dir = "./output/dl_iris"
# ds = datasets.load_digits(return_X_y=False)  # 手写数字的数据
# save_dir = "./output/dl_digits"
in_features = len(ds.feature_names)
num_classes = len(ds.target_names)
X, Y = ds.data, ds.target  # numpy数组类型
print(f"数据集的特征属性名称列表 {in_features} - {ds.feature_names}")
print(f"数据集的类别名称列表 {num_classes} - {ds.target_names}")
print(f"数据集的shape形状 {type(X)}-{X.shape} {type(Y)}-{Y.shape}")

os.makedirs(save_dir, exist_ok=True)  # 保存的文件夹进行创建

# 2. 数据处理
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=14
)
print(f"训练数据shape {X_train.shape} - {Y_train.shape}")
print(f"测试数据shape {X_test.shape} - {Y_test.shape}")
train_dataloader = build_dataloader(X_train, Y_train, batch_size=8, shuffle=True)
test_dataloader = build_dataloader(X_test, Y_test, batch_size=16, shuffle=False)

# 3. 模型训练
#     3.1 创建 --> 需要人为构造出网络结构、优化器、损失函数
#         模型初始化 ----> 构造网络的执行图(构建图中的各个模块)
#         Loss Function的构造
#         优化器构造
net = ClassifyNetwork(in_features=in_features, num_classes=num_classes)
loss_fn = nn.CrossEntropyLoss()
opt_fn = optim.SGD(net.parameters(), lr=0.01)

#     3.2 训练 --> 需要人为进行数据的遍历以及前向反向过程的代码编写
#         3.2.1 前向过程的执行 ---->
#             属于网络的执行图的构建(模型的执行顺序)
#             loss的获取
#         3.2.2 反向过程的执行 ----> 不需要人为构造（框架会帮我们完成）
#             + 梯度计算 + 参数的更新 + 梯度重置为0
#         NOTE: 训练是一个循环的过程，所以在训练过程中会有模型评估和模型持久化的操作
total_epoch = 1000
early = StopEarlyWithBig(max_step=10)
for epoch in range(total_epoch):
    # 模型训练
    net.train()  # 标记当前网络处于train阶段 原因：PyTorch中存在部分的模块，训练和推理的执行逻辑不同, eg: BN(BatchNorm)、DropOut....
    for batch_idx, (x, y) in enumerate(train_dataloader):
        # 1. 前向过程
        score = net(x)  # 前向执行 获取得到每个样本属于C个类别的置信度
        loss = loss_fn(score, y)

        # 反向过程的执行
        opt_fn.zero_grad()  # 将优化器进行更新的对应参数的梯度重置为0
        loss.backward()  # 求解执行链路中涉及到的所有参数的梯度值
        opt_fn.step()  # 触发参数更新操作 -> 基于梯度进行参数更新

        print(f"{epoch} {batch_idx} Loss {loss.item():.3f}")

    # 4. 模型评估
    #     4.1 需要人为进行数据遍历、模型的推理预测、预测结果的评估
    with torch.no_grad():
        net.eval()  # 标记模型进入了推理阶段
        total_pred_y = []
        for x, _ in test_dataloader:
            score, _ = net(x)  # 前向执行 获取得到每个样本属于C个类别的置信度 [N,C]
            pre_y = torch.argmax(score, dim=1).numpy()  # numpy对象，并且shape形状为 [N,] 每个样本对应的预测类别id
            total_pred_y.extend(list(pre_y))
        print(
            f"Classification report for classifier:\n"
            f"{metrics.classification_report(Y_test, total_pred_y)}\n"
        )
        test_acc = metrics.accuracy_score(Y_test, total_pred_y)
        print(f"Accuracy {test_acc}")
        is_best = early.update(value=test_acc)

        # 5. 模型持久化保存磁盘
        obj = {
            'net': net,
            'opt': opt_fn,
            'feature_names': ds.feature_names,
            'target_names': ds.target_names,
            'acc': test_acc,
            'epoch': epoch
        }
        torch.save(obj, os.path.join(save_dir, "net_last.pkl"))
        if is_best:
            print(f"最优模型持久化 {epoch}")
            torch.save(obj, os.path.join(save_dir, "net_best.pkl"))

    if early.is_stop():
        print(f"提前停止模型训练过程 {epoch}")
        break

print("训练完成!!")
