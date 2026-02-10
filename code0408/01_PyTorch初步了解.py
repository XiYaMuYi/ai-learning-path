# -*- coding: utf-8 -*-

import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    a = torch.randn(2, 3).to(device=device)
    #a = torch.randn(2, 3).to(torch.device("cpu"))没有GPU的情况下和上一行意义相同
    b = torch.randn(3, 4).to(device=device)
    c = torch.matmul(a, b)
    d = torch.abs(c)
    print(c)
    print(d)
