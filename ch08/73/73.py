# 73. 確率的勾配降下法による学習

import numpy as np
import torch
from torch import nn

if __name__ == "__main__":

    # 学習データの読み込み
    X = np.load('../70/train_vector.npy')
    X = torch.tensor(X, requires_grad=True)
    # torch.Size([10672, 300])

    # 正解データの読み込み
    y = np.load('../70/train_label.npy')
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
    print('Before')
    print(net.state_dict()['weight'])
    # torch.Size([4, 300])
    # tensor([[-0.0310,  0.0251, -0.0181,  ..., -0.0304,  0.0563, -0.0150],
    #         [-0.0090,  0.0248, -0.0084,  ..., -0.0283,  0.0510, -0.0543],
    #         [ 0.0043, -0.0266,  0.0139,  ...,  0.0376,  0.0263, -0.0304],
    #         [-0.0034,  0.0544,  0.0378,  ..., -0.0141,  0.0039,  0.0392]])
    # それぞれの特徴ベクトルが、どの正解データっぽいかをなんとなく表現
    # 例えば、label0っぽさに対する重み。
    # 0.0311は,xの0番目の特徴ベクトルに対してこの数字をかける。
    # 正解labelが0のとき、0番目の特徴ベクトルはどういう数字になるかを考慮してこの重みは決まる。
    
    for step in range(100):
        optimizer.zero_grad()

        y_pred = net(X)
        
        output = loss(y_pred, y)
        output.backward()

        optimizer.step()

    print('After')
    print(net.state_dict()['weight'])
    # tensor([[-0.0356,  0.0269, -0.0221,  ..., -0.0217,  0.0655, -0.0303],
    #     [-0.0061,  0.0196, -0.0057,  ..., -0.0258,  0.0486, -0.0571],
    #     [ 0.0057, -0.0178,  0.0123,  ...,  0.0198,  0.0190, -0.0131],
    #     [-0.0031,  0.0490,  0.0407,  ..., -0.0074,  0.0045,  0.0400]])
