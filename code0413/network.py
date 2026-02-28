import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
class ClassifyNetworkV0(nn.Module):
    def __init__(self, in_features, num_classes):
        '''
        模型初始化/模型对象构造方法 --> 负责模型涉及到的相关参数/子模块的定义
        '''
        super(ClassifyNetworkV0, self).__init__()
        self.w1 = nn.Parameter(torch.empty(in_features, 8))
        self.b1 = nn.Parameter(torch.empty(8))
        self.w2 = nn.Parameter(torch.empty(8, 8))
        self.b2 = nn.Parameter(torch.empty(8))
        self.w3 = nn.Parameter(torch.empty(8, num_classes))
        self.b3 = nn.Parameter(torch.empty(num_classes))
        # 参数值的初始化操作
        for param in self.parameters():
            nn.init.normal_(param.data)

    def forward(self,x):
        '''
        当前网络的前向执行过程，一般情况下，该方法需要实现的功能是：基于输入得到的模型的推理预测输出
             N: 表示批次大小，也就是样本的数目
        :param x: 输入的特征属性tensor对象，shape为：[N, in_features]
        :return: 输出每个样本对应属于各个类别的置信度[N, num_classes]
        '''
        # 1. 输入层到第一隐层的全连接操作
        n1 = torch.matmul(x, self.w1) + self.b1 #[N,3] * [3,8] + [8] -> [N,8]
        o1 = F.relu(n1)

        # 2. 第一隐层到第二隐层的全连接操作
        n2 = o1 @ self.w2 + self.b2  # [N,8] * [8,8] + [8] -> [N,8]
        o2 = F.relu(n2)

        # 3. 第二隐层到输出层的全连接操作
        score = torch.matmul(o2, self.w3) + self.b3 #[N,8] * [8,3] + [3] ->[N,3]
        return score


def tt01():
    in_features = 4
    net = ClassifyNetworkV0(in_features, num_classes=3)
    print("=" * 100)
    for _name, _param in net.named_parameters():
        print(_name, _param.requires_grad, _param)
    print("=" * 100)
    _x = torch.randn(5, in_features)
    # net(_x) --> 首先调用 net.__call__(_x) --> 再调用 net.  {PyTorch框架的转换} --> 最后调用net.forward
    _s = net(_x)
    print(_s)
    print(_s.shape)

if __name__ == '__main__':
    tt01()

