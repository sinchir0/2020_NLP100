# 81. RNNによる予測
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import pickle

import pandas as pd

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden_size=50, output_size=4):
        self.emb = nn.embedding(
            vocab_size, 
            emb_dim
            )
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_size, 
            num_layers=1,
            nonlinearity='tanh'
            bias=True
            )
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
            bias=True
            )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, h_0=0):
        x = self.emb(x)
        x, h_t = self.rnn(x, h_0)
        x = self.linear(x)
        x = self.softmax(x)
        return x, h_t

if __name__ == '__main__':

    # データの読み込み
    train = pd.read_pickle('../80/train_title_id.pkl')

    # 辞書の読み込み
    with open("../80/word_id_dict.pkl", "rb") as tf:
        word_id_dict = pickle.load(tf)

    n_letters = len(word_id_dict.keys())
    n_hidden = 50
    n_categories = 4

    rnn = RNN(n_letters, n_hidden, n_categories)

    import ipdb; ipdb.set_trace()
    