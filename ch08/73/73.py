# 73. 確率的勾配降下法による学習

import numpy as np
import torch
from torch import nn

if __name__ == "__main__":

    # 学習データの読み込み
    X = np.load("../70/train_vector.npy")
    X = torch.tensor(X, requires_grad=True)
    # torch.Size([10672, 300])

    # 正解データの読み込み
    y = np.load("../70/train_label.npy")
    y = torch.from_numpy(y)
    # torch.Size([10672])

    # modelの設定
    # y=xA^T + b
    # 今回の例の場合、
    # x:torch.Size([10672, 300])
    # A:torch.Size([4, 300]) これはnet.parameters().__next__().size()によって求まる
    # A^T:torch.Size([300, 4])
    # よって、
    # xA^T:torch.Size([10672, 4])
    net = nn.Linear(300, 4)

    # loss, optimizerの設定
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # parameterの更新
    print("Before")
    print(net.state_dict()["weight"])
    # torch.Size([4, 300])
    # tensor([[ 0.0508, -0.0503,  0.0018,  ..., -0.0246, -0.0344,  0.0418],
    #         [-0.0468,  0.0374, -0.0137,  ...,  0.0034, -0.0196,  0.0232],
    #         [-0.0145,  0.0146, -0.0559,  ..., -0.0218,  0.0179,  0.0148],
    #         [-0.0107, -0.0198,  0.0443,  ...,  0.0069, -0.0069, -0.0463]])
    # それぞれの特徴ベクトルが、どの正解データっぽいかをなんとなく表現
    # 例えば、label0っぽさに対する重み。
    # 0.0508は,xの0番目の特徴ベクトルに対してこの数字をかける。
    # 正解labelが0のとき、0番目の特徴ベクトルはどういう数字になるかを考慮してこの重みは決まる。

    for step in range(100):
        optimizer.zero_grad()

        y_pred = net(X)

        output = loss(y_pred, y)
        output.backward()

        optimizer.step()

    print("After")
    print(net.state_dict()["weight"])
    # tensor([[ 0.0461, -0.0476, -0.0025,  ..., -0.0167, -0.0249,  0.0266],
    #     [-0.0437,  0.0316, -0.0109,  ...,  0.0063, -0.0224,  0.0204],
    #     [-0.0132,  0.0228, -0.0573,  ..., -0.0390,  0.0105,  0.0317],
    #     [-0.0104, -0.0249,  0.0471,  ...,  0.0133, -0.0063, -0.0451]])

    net_path = "73_net.pth"
    torch.save(net.state_dict(), net_path)
