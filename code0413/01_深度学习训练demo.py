# 深度学习模型训练代码结构：--> 基于PyTorch框架
from sklearn import datasets
from sklearn.model_selection import train_test_split
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
#         3.2 训练 --> 需要人为进行数据的遍历以及前向反向过程的代码编写
#             3.2.1 前向过程的执行 ---->
#                 属于网络的执行图的构建(模型的执行顺序)
#                 loss的获取
#             3.2.2 反向过程的执行 ----> 不需要人为构造（框架会帮我们完成）
#                 + 梯度计算 + 参数的更新 + 梯度重置为0
#             NOTE: 训练是一个循环的过程，所以在训练过程中会有模型评估和模型持久化的操作
#     4. 模型评估
#         4.1 需要人为进行数据遍历、模型的推理预测、预测结果的评估
#     5. 模型持久化保存磁盘
# 深度学习模型推理应用代码结构:
#     1. 加载恢复模型(结构 + 参数，NOTE: 模型持久化的方式和模型恢复的方式必须是一一对应的)
#     2. 和训练采用相同的流程，对待预测的数据进行处理转换
#     3. 调用模型的预测方法(前向过程)获取得到预测结果
#     4. 后处理转换 --> 在模型预测结果的基础上额外的进行一些数据处理的工作
