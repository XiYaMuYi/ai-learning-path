# -*- coding: utf-8 -*-
import json
import warnings

import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib

warnings.filterwarnings('ignore')

# 1. 数据加载
ds = datasets.load_iris(return_X_y=False)
print(f"当前数据的特征属性列表为:{ds.feature_names} - {ds.target_names}")
X = ds.data
Y = ds.target
print(f"当前数据shape为:{X.shape} - {Y.shape}")
print(f"案例数据:\n{X[:1]}  {Y[:1]}")

# 2. 数据处理
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=True, random_state=14
)
print(f"训练数据shape:{X_train.shape} - {y_train.shape}")
print(f"测试数据shape:{X_test.shape} - {y_test.shape}")
# 临时保存5条测试数据
tmp_test = np.concatenate([X_test[:5], y_test[:5].reshape(-1, 1)], axis=1)
np.savetxt('tmp_data.txt', tmp_test)

# 3. 特征工程
poly = PolynomialFeatures(degree=2)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)
print(f"特征工程后的特征属性shape为:{X_train.shape} - {X_test.shape}")
print(f"多项式扩展:{poly.get_feature_names_out()}")

# 4. 模型训练
#     4.1 创建 --> 创建一个新的算法模型对象即可
algo = LogisticRegression()
# algo = DecisionTreeClassifier()

#     4.2 训练 --> fit方法调用即可
algo.fit(X_train, y_train)

# 5. 模型评估
#     5.1 直接调用封装好的方法即可
pre_y = algo.predict(X_test)
print(
    f"Classification report for classifier {algo}:\n"
    f"{metrics.classification_report(y_test, pre_y)}\n"
)

# 6. 模型持久化保存磁盘
save_dir = "./output/ml"
os.makedirs(save_dir, exist_ok=True)  # 保存的文件夹进行创建
joblib.dump(poly, os.path.join(save_dir, "poly.pkl"))
joblib.dump(algo, os.path.join(save_dir, "algo.pkl"))
config = {
    'feature_names': list(ds.feature_names),
    'target_names': list(ds.target_names),
    'ploy': list(poly.get_feature_names_out()),
    'algo_coef_': list(map(float, algo.coef_.reshape(-1))),
    'algo_intercept_': list(algo.intercept_.reshape(-1))
}
joblib.dump(config, os.path.join(save_dir, "config.pkl"))

# 使用json格式将algo参数保存
with open(os.path.join(save_dir, "config.json"), "w") as writer:
    json.dump(config, writer, indent=2)
