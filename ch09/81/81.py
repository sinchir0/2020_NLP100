# 81. RNNによる予測
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import pickle

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ipdb import set_trace as st

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self): # len(Dataset)で返す値を指定
        return len(self.X)

    def __getitem__(self, idx): # Dataset[index]で返す値を指定
        # Xをintのlistに変換
        X_list = [int(x) for x in self.X[idx].split()]
        
        # tensorに変換
        inputs = torch.tensor(X_list).unsqueeze(0)
        label = torch.tensor(self.y[idx])
        
        return inputs, label

class RNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim=300,
                 hidden_size=50,
                 output_size=4
                 ):
        super().__init__()

        self.emb = nn.Embedding(
            vocab_size, 
            emb_dim,
            padding_idx=0 # 0に変換された文字にベクトルを計算しない
            )

        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_size, 
            num_layers=1,
            nonlinearity='tanh',
            bias=True
            )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
            bias=True
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h_0=None):
        x = self.emb(x)
        x, h_t = self.rnn(x, h_0)
        x = x[:, -1, :] # 一番最後の出力に絞る、やっていいのかこれ？
        x = self.fc(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':

    # データの読み込み
    train = pd.read_pickle('../80/train_title_id.pkl')

    # yの変換
    cat_id_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    train['CATEGORY'] = train['CATEGORY'].map(cat_id_dict)

    # 辞書の読み込み
    with open("../80/word_id_dict.pkl", "rb") as tf:
        word_id_dict = pickle.load(tf)

    n_letters = len(word_id_dict.keys())
    n_hidden = 50
    n_categories = 4

    # modelの定義
    model = RNN(n_letters, n_hidden, n_categories)

    # datasetの定義
    dataset = TextDataset(train['TITLE'], train['CATEGORY'])

    # 先頭10個の結果を出力
    for i in range(10):
        X = dataset[i][0]
        print(model(x=X))

    # tensor([[0.0973, 0.2198, 0.1984, 0.4845]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.2295, 0.2891, 0.1601, 0.3214]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.1391, 0.3773, 0.2486, 0.2351]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.2070, 0.1848, 0.1685, 0.4398]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.2118, 0.1704, 0.2611, 0.3568]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.2118, 0.1704, 0.2611, 0.3568]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.1008, 0.2200, 0.1648, 0.5144]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.1008, 0.2200, 0.1648, 0.5144]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.3206, 0.2763, 0.2435, 0.1595]], grad_fn=<SoftmaxBackward>)
    # tensor([[0.2695, 0.4305, 0.1578, 0.1422]], grad_fn=<SoftmaxBackward>)