# 72. 損失と勾配の計算

import numpy as np
import torch
from torch import nn

if __name__ == "__main__":

    # 事例集合x1,x2,x3,x4の読み込み
    X = torch.load("../71/71.pt")

    # 正解データの読み込み
    Y = np.load("../70/train_label.npy")
    Y = torch.from_numpy(Y)

    # lossの計算
    loss = nn.CrossEntropyLoss()
    output = loss(X, Y)

    import ipdb

    ipdb.set_trace()

    print(output)
    # tensor(1.3074, grad_fn=<NllLossBackward>)

    print(X)

    # 勾配の計算
    output.backward()  # Xはrequired_gradがTrueになっているため、outputの計算元であるXに関する勾配を裏で計算している
    print(X.grad)
    # tensor([[-7.1343e-05,  3.2402e-05,  1.8923e-05,  2.0019e-05],
    #     [-7.0948e-05,  1.8294e-05,  2.2431e-05,  3.0223e-05],
    #     [-7.6297e-05,  1.6931e-05,  1.7497e-05,  4.1869e-05],
    #     ...,
    #     [-6.5094e-05,  2.0229e-05,  2.3246e-05,  2.1619e-05],
    #     [-7.1032e-05,  3.0666e-05,  1.9752e-05,  2.0614e-05],
    #     [-7.1319e-05,  1.8884e-05,  1.9923e-05,  3.2512e-05]])
