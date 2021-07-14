# 81. RNNによる予測
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import pickle

import pandas as pd

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=300):
        self.emb = nn.embedding(
            vocab_size, 
            emb_dim
            )
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=50, 
            num_layers=1,
            nonlinearity='tanh'
            bias=True
            )
        self.softmax = nn.Softmax()

    def forward(self, x, h_0=0):
        x = self.emb(x)
        x, h_n = self.rnn(x, h_0)

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
    