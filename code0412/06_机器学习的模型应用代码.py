# -*- coding: utf-8 -*-
import json
import os

import joblib
import numpy as np


def tt_invoker():
    # 1. 加载恢复模型(结构 + 参数，NOTE: 模型持久化的方式和模型恢复的方式必须是一一对应的)
    save_dir = "./output/ml"
    poly = joblib.load(os.path.join(save_dir, "poly.pkl"))
    algo = joblib.load(os.path.join(save_dir, "algo.pkl"))

    # 2. 和训练采用相同的流程，对待预测的数据进行处理转换
    x = np.asarray([
        [4.78, 3.4, 1.9, 0.2],
        [5.7, 2.5, 5., 2]
    ])
    x = poly.transform(x)

    # 3. 调用模型的predict方法(预测方法)获取得到预测结果
    score = algo.decision_function(x)  # 得到各个样本属于各个类别的置信度 -> wx+b
    proba = algo.predict_proba(x)  # 得到各个样本属于各个类别的概率值 -> softmax(wx+b)
    r = algo.predict(x)  # 得到各个样本最终所属类别id -> argmax(softmax(wx+b))

    print(score)
    print(proba)
    print(r)
    """
    [[ 11.52941126   4.61847733 -16.14788858]
     [-11.15677034   2.39157527   8.76519507]]
    [[9.99004166e-01 9.95833721e-04 9.53825009e-13]
     [2.22464178e-09 1.70306736e-03 9.98296930e-01]]
    [0 2]
    """
    #     4. 后处理转换 --> 在模型预测结果的基础上额外的进行一些数据处理的工作
    pass


class Predict(object):
    def __init__(self, save_dir):
        super(Predict, self).__init__()
        # 1. 加载恢复模型(结构 + 参数，NOTE: 模型持久化的方式和模型恢复的方式必须是一一对应的)
        self.poly = joblib.load(os.path.join(save_dir, "poly.pkl"))
        self.algo = joblib.load(os.path.join(save_dir, "algo.pkl"))
        # config = joblib.load(os.path.join(save_dir, "config.pkl"))
        with open(os.path.join(save_dir, "config.json"), "r") as reader:
            config = json.load(reader)
        self.class_names = config['target_names']
        self.feature_names = config['feature_names']
        self.feature_numbers = len(self.feature_names)
        print(f"模型恢复成功: {save_dir}")
        print(f"恢复后的模型参数为:\n{self.algo.coef_}")

    def predict(self, x, k=1):
        """
        针对输入的x进行推理预测
        :param x: csv格式的文本数据；样本与样本之间使用";"分割，属性与属性之间使用","分割
        :param k: 针对每个样本获取预测概率最大的前k个结果
        :return:以json(数组&字典)格式返回数据,
            eg:
                [
                    {
                        "x": "....", # 原始样本的特征
                        "predict": [
                            {
                                "id":5, # 预测类别id 第一个预测概率最大的类别id
                                "proba":0.52 # 预测概率
                            },
                            {
                                "id":3, # 预测类别id 第二个预测概率最大的类别id
                                "proba":0.24 # 预测概率
                            },
                            ......
                        ]
                    },
                    {....}, # 第一个样本的预测值
                    ....
                ]
        """
        # 参数解析
        x = [list(map(lambda rx: float(rx.strip()), r.split(","))) for r in x.split(";")]
        x = np.asarray(x)
        if x.shape[1] != self.feature_numbers:
            raise ValueError(f"要求输入的特征必须是{self.feature_numbers}维的，特征名称为:{self.feature_names}")

        # 2. 和训练采用相同的流程，对待预测的数据进行处理转换
        new_x = self.poly.transform(x)

        # 3. 调用模型的predict方法(预测方法)获取得到预测结果
        pred_proba = self.algo.predict_proba(new_x)  # 得到各个样本属于各个类别的概率值 -> softmax(wx+b) [N,C]

        # 4. 后处理转换 --> 在模型预测结果的基础上额外的进行一些数据处理的工作
        result = []
        for i in range(len(x)):
            predict = []
            record = {
                'x': ','.join(map(lambda t: f'{t:.3f}', x[i])),
                'predict': predict
            }
            proba_i = pred_proba[i]
            sorted_indices_i = np.argsort(proba_i)[::-1]
            for idx in sorted_indices_i[:k]:
                idx = int(idx)
                proba = proba_i[idx]
                predict.append({
                    "id": idx,
                    "name": self.class_names[idx],
                    "proba": float(f'{proba:.3}')
                })

            result.append(record)
        return result


def tt_invoke_predict():
    model = Predict(save_dir="./output/ml")
    x = "4.78,3.4,1.9,0.2;5.7,2.5,5.,2"
    for i in range(2):
        r = model.predict(x, k=2)
        print(r)


def tt_invoke_predict_with_command():
    model = Predict(save_dir="./output/ml")
    while True:
        x = input("请输入待预测的样本信息，输入q退出:")
        x = x.lower()
        if x == 'q':
            break
        k = input("输入top k的值（默认k为1）:")
        if k is None or len(k.strip()) == 0:
            k = 1
        else:
            k = int(k)
        r = model.predict(x, k=k)
        print(r)


if __name__ == '__main__':
    # tt_invoker()
    # tt_invoke_predict()
    tt_invoke_predict_with_command()
