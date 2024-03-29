# 81. RNNによる予測
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import pickle

import pandas as pd
import torch
import torch.nn as nn
from ipdb import set_trace as st
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.X)

    def __getitem__(self, idx):  # Dataset[index]で返す値を指定
        # Xをintのlistに変換
        X_list = [int(x) for x in self.X[idx].split()]

        # tensorに変換
        inputs = torch.tensor(
            X_list
        )  # .unsqueeze(0) # unsqueezeはtorch.Size([6]) →　torch.Size([1, 6])
        label = torch.tensor(self.y[idx])

        return inputs, label


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden_size=50, output_size=4):
        super().__init__()

        self.emb = nn.Embedding(
            vocab_size, emb_dim, padding_idx=0  # 0に変換された文字にベクトルを計算しない
        )

        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            bias=True,
        )

        self.fc = nn.Linear(
            in_features=hidden_size, out_features=output_size, bias=True
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h_0=None):
        st()
        x = self.emb(x)
        x, h_t = self.rnn(x, h_0)
        x = x[:, -1, :]  # 一番最後の出力に絞る、やっていいのかこれ？
        x = self.fc(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":

    # データの読み込み
    train = pd.read_pickle("../80/train_title_id.pkl")

    # yの変換
    cat_id_dict = {"b": 0, "t": 1, "e": 2, "m": 3}
    train["CATEGORY"] = train["CATEGORY"].map(cat_id_dict)

    # 辞書の読み込み
    with open("../80/word_id_dict.pkl", "rb") as tf:
        word_id_dict = pickle.load(tf)

    n_letters = len(word_id_dict.keys())
    n_hidden = 50
    n_categories = 4

    # modelの定義
    model = RNN(n_letters, n_hidden, n_categories)

    # datasetの定義
    dataset = TextDataset(train["TITLE"], train["CATEGORY"])

    # 先頭10個の結果を出力
    for i in range(10):
        X = dataset[i][0]
        X = X.unsqueeze(0)
        print(model(x=X))

    # tensor([[0.3996, 0.1701, 0.1483, 0.2821]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.2412, 0.2179, 0.1354, 0.4055]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.1941, 0.3746, 0.0832, 0.3481]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.1472, 0.3155, 0.1035, 0.4338]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.2272, 0.3158, 0.1757, 0.2812]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.2272, 0.3158, 0.1757, 0.2812]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.3958, 0.1784, 0.1419, 0.2839]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.3958, 0.1784, 0.1419, 0.2839]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.1054, 0.5542, 0.0961, 0.2442]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.2882, 0.3606, 0.1956, 0.1556]], grad_fn=<SoftmaxBackward>)
