# -*- coding: utf-8 -*-
# pip install scikit-learn==1.5.2

import os
from sklearn import datasets

print(f"当前文件夹路径 {os.path.abspath('.')}")#查询当前文件的绝对路径
#调用加州房屋数据的函数，参数data_home 后面要传数据的存放路径，也就是同等级下的datas文件夹下的housing文件夹
ds = datasets.fetch_california_housing(data_home="../datas/housing")
print(ds)
print(type(ds))
