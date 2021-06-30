# 71. 単層ニューラルネットワークによる予測

import numpy as np

import torch
from torch import nn

if __name__ == "__main__":

    # Xの読み込み
    X = np.load('../70/train_vector.npy')
    X = torch.from_numpy(X)

    # Wの生成    
    W = torch.randn(300, 4)

    # XとWの内積
    XW = torch.matmul(X, W)

    # 行方向のsoftmaxの演算
    m = nn.Softmax(dim=1)
    output = m(XW)

    print(output)