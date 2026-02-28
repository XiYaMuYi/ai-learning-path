# -*- coding: utf-8 -*-

import cv2 as cv


def tt01():
    print(cv)
    print(cv.__version__)


def tt02():
    # NOTE: OpenCV的图像路径不支持中文
    img_path = r"..\datas\small.png"
    # img shape: [H,W,C] H 图像的高度，W 图像的宽度，C 图像使用几个颜色通道进行特征描述
    img = cv.imread(img_path)  # 默认情况下，针对彩色图像使用BGR的颜色空间进行加载
    print(type(img), img.shape)
    print(img)


if __name__ == '__main__':
    tt02()
