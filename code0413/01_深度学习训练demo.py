# 深度学习模型训练代码结构：--> 基于PyTorch框架
import os
import joblib
import torch
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
#只要code043这个文件夹/package所在的文件夹路径在sys.path环境变量中，下列的from .. import 代码就是成功的
from code0413.network import ClassifyNetworkV0
#     1. 数据加载
ds = datasets.load_iris(return_X_y=False)
in_features = len(ds.feature_names)
num_classes = len(ds.target_names)
X,Y = ds.data,ds.target
print(f"数据集的特征属性名称列表{in_features} - {ds.feature_names}")
print(f"数据集的类别名称列表{num_classes} - {ds.target_names}")
print(f"数据集的shape形状{type(X)}-{X.shape}  {type(Y)}-{Y.shape}")
#     2. 数据处理
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=14
)
print(f"训练数据shape：{X_train.shape} - {Y_train.shape}")
print(f"测试数据shape：{X_test.shape} - {Y_test.shape}")
#     3. 模型训练, radom_state
#         3.1 创建 --> 需要人为构造出网络结构、优化器、损失函数
#             模型初始化 ----> 构造网络的执行图(构建图中的各个模块)
#             Loss Function的构造
#             优化器构造
net = ClassifyNetworkV0(in_features=in_features,num_classes=num_classes)
loss_fn = nn.CrossEntropyLoss()
opt_fn = optim.SGD(net.parameters(),lr=0.001)
#         3.2 训练 --> 需要人为进行数据的遍历以及前向反向过程的代码编写
#             3.2.1 前向过程的执行 ---->
#                 属于网络的执行图的构建(模型的执行顺序)
#                 loss的获取
#             3.2.2 反向过程的执行 ----> 不需要人为构造（框架会帮我们完成）
#                 + 梯度计算 + 参数的更新 + 梯度重置为0
#             NOTE: 训练是一个循环的过程，所以在训练过程中会有模型评估和模型持久化的操作
batch_size = 8
total_train_samples = len(X_train)
total_batch = total_train_samples // batch_size
for epoch in range(100):
    # 模型训练
    net.train() # 标记当前网络处于train阶段
    shuffle_indexes = np.random.permutation(total_train_samples)
    for batch_idx in range(total_batch):
        batch_indexes = shuffle_indexes[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        batch_x,batch_y  = X_train[batch_indexes],Y_train[batch_indexes]


        #前向执行过程
        x = torch.from_numpy(batch_x.astype('float32'))
        y = torch.from_numpy(batch_y).to(dtype=torch.int64)
        score = net(x)# 向前执行 获取得到每个样本属于C个类别的置信度[N,C]
        loss = loss_fn(score, y)
        # print(loss)

        # 反向过程的执行
        opt_fn.zero_grad()  # 将优化器进行更新的对应参数梯度重置为0
        loss.backward()  # 求解执行链路中涉及道的所有参数的梯度值
        opt_fn.step()  # 触发参数更新操作 -> 基于梯度进行参数更新

        print(f"{epoch} {batch_idx} loss{loss.item():.3f}")


#     4. 模型评估
#         4.1 需要人为进行数据遍历、模型的推理预测、预测结果的评估
with torch.no_grad():
    net.eval() # 标记模型进入了推理阶段
    x = torch.from_numpy(X_test.astype('float32'))
    y = torch.from_numpy(Y_test).to(dtype=torch.int64)
    score = net(x) # 向前执行 获取得到每个样本属于C个类别的置信度[N,C]
    pre_y = torch.argmax(score, dim=1).numpy() #numpy对象，并且shape形状为[N,] 每个样本对应的预测类别id
    print(
        f"Classification report for classifier:\n"
        f"{metrics.classification_report(Y_test, pre_y)}\n"
    )
#     5. 模型持久化保存磁盘
save_dir = "./output/dl"
os.makedirs(save_dir, exist_ok=True) # 保存的文件夹进行创建
joblib.dump(net, os.path.join(save_dir, "net_joblib.pkl"))

# #     5. 模型持久化保存磁盘(优化版)
# save_dir = "./output/dl"
# os.makedirs(save_dir, exist_ok=True)
#
# # 使用 torch.save 保存模型参数字典，后缀通常用 .pth 或 .pt
# model_path = os.path.join(save_dir, "net_params.pth")
# torch.save(net.state_dict(), model_path)
# print(f"模型参数已成功保存至: {model_path}")

# 如果你想连同网络结构一起保存（类似 joblib 的行为），可以这样写：
# torch.save(net, os.path.join(save_dir, "net_full.pth"))



# 深度学习模型推理应用代码结构:
#     1. 加载恢复模型(结构 + 参数，NOTE: 模型持久化的方式和模型恢复的方式必须是一一对应的)
#     2. 和训练采用相同的流程，对待预测的数据进行处理转换
#     3. 调用模型的预测方法(前向过程)获取得到预测结果
#     4. 后处理转换 --> 在模型预测结果的基础上额外的进行一些数据处理的工作
