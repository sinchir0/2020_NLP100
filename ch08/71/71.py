# 71. 単層ニューラルネットワークによる予測

import numpy as np
import torch
from torch import nn

if __name__ == "__main__":

    # Xの読み込み
    X = np.load("../70/train_vector.npy")
    X = torch.tensor(X, requires_grad=True)

    # Wの生成
    W = torch.randn(300, 4)

    # XとWの内積
    XW = torch.matmul(X, W)

    # 行方向のsoftmaxの演算
    m = nn.Softmax(dim=1)
    output = m(XW)

    print(output)
    # tensor([[0.2043, 0.3957, 0.2756, 0.1245],
    #     [0.1173, 0.2283, 0.5908, 0.0636],
    #     [0.0679, 0.4003, 0.4292, 0.1027],
    #     ...,
    #     [0.2458, 0.4192, 0.2750, 0.0599],
    #     [0.0954, 0.6650, 0.1910, 0.0486],
    #     [0.0869, 0.3767, 0.2606, 0.2759]])

    torch.save(output, "71.pt")

    assert torch.equal(
        net.fc(torch.zeros_like(x_train)), torch.zeros(x_train.shape[0], 4)
    )
