# 89. 事前学習済み言語モデルからの転移学習
# 事前学習済み言語モデル（例えばBERTなど）を出発点として，ニュース記事見出しをカテゴリに分類するモデルを構築せよ．

# https://qiita.com/yamaru/items/63a342c844cff056a549

import random
import os
from numpy.lib.function_base import kaiser
from tqdm import tqdm

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gensim

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import BertTokenizer, BertModel

from ipdb import set_trace as st

def seed_everything(seed=42, use_torch=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if use_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

class TextDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len, device):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __len__(self): # len(Dataset)で返す値を指定
        return len(self.X)

    def __getitem__(self, idx): # Dataset[index]で返す値を指定
        text = self.X[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            # pad_to_max_length=True
            padding='max_length',
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.LongTensor(ids, device=device),
            'mask': torch.LongTensor(mask, device=device),
            'labels': torch.LongTensor([self.y[idx]], device=device)
        }

class BERTClass(nn.Module):
    def __init__(self, drop_rate, output_size):
        super().__init__()
        # outputs = model(ids, mask)
        # にて、下のエラーで落ちる
        # TypeError: dropout(): argument 'input' (position 1) must be Tensor, not str
        # https://github.com/huggingface/transformers/issues/8879#issuecomment-796328753
        # return_dict=Falseを追加したら解決
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(768, output_size)  # BERTの出力に合わせて768次元を指定
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ids, mask):
        st()
        _, x = self.bert(ids, attention_mask=mask)
        # 引数の一つ目は、(batch_size, seq_length=10, 768)のテンソル、これは生のBERTの出力、多分
        # 引数の二つめは、(batch_size, 768)のテンソル、これは先頭単語[CLS]を取り出して、
        # BertPoolerにて、同じhiddensize→hiddensizeへと全結合層を通して、その後tanhを通して-1~1にしたもの。
        x = self.drop(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

def train_fn(model, dataset, device, optimizer, criterion, BATCH_SIZE) -> float:
    """model, loaderを用いて学習を行い、lossを返す"""

    # dataloaderを生成
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 学習モードに設定
    model.train()

    train_running_loss = 0.0

    for data in loader:
        # デバイスの指定 
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        labels = data['labels'].to(device)

        # labelsの次元を(BATCH_SIZE, 1)から(BATCH_SIZE,)に変更
        labels = labels.squeeze(1)

        optimizer.zero_grad()

        outputs = model(ids, mask)
        # dataloader_xでの損失の計算/
        loss = criterion(outputs, labels)
        # 勾配の計算
        loss.backward()
        optimizer.step()

        # 訓練データでの損失の平均を計算する
        train_running_loss += loss.item() / len(loader)

    return train_running_loss

def calculate_loss_and_accuracy(model, dataset, device=None, criterion=None):
    """損失・正解率を計算"""
    # 評価モードに設定
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  
    loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for data in dataloader:
            # 変数の生成、device送り
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            # 順伝播
            outputs = model(ids, mask)

            # labelsの次元を(BATCH_SIZE, 1)から(BATCH_SIZE,)に変更
            labels = labels.squeeze(1)

            # 損失計算
            loss += criterion(outputs, labels).item()

            # 正解率計算
            pred = torch.argmax(outputs, dim=-1)
            total += len(labels)
            correct += (pred == labels).sum().item()
        
    return loss / len(dataset), correct / total

# def padding(id_seq: str, max_len: int):
#     '''id_seqについて、
#     max_lenより長い場合はmax_lenまでの長さにする。
#     max_lenより短い場合はmax_lenになるように0を追加する。
#     '''
#     id_list = id_seq.split(' ')
#     if len(id_list) > max_len:
#         id_list = id_list[:max_len]
#     else:
#         pad_num = max_len - len(id_list)
#         for _ in range(pad_num):
#             id_list.append('0')
#     return ' '.join(id_list)

def make_graph(value_dict: dict, value_name: str, bn:int, method: str) -> None:
    """value_dictに関するgraphを生成し、保存する。"""
    for phase in ["train", "test"]:
        plt.plot(value_dict[phase], label=phase)
    plt.xlabel("epoch")
    plt.ylabel(value_name)
    plt.title(f"{value_name} per epoch at bn{bn}")
    plt.legend()
    plt.savefig(f"{method}_{value_name}_bn{bn}.png")
    plt.close()


if __name__ == '__main__':

    # seedの固定
    seed_everything(use_torch=True)

    DEBUG =False
    if DEBUG:
        print('DEBUG mode')

    # Colab
    # PATH = '/content/drive/MyDrive/NLP100/ch09'
    # local
    PATH = '..'

    # deviceの指定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Use {device}')

    # データの読み込み
    train = pd.read_csv('../../ch06/50/train.txt', sep='\t', index_col=0)
    test = pd.read_csv('../../ch06/50/test.txt', sep='\t', index_col=0)

    # indexを再設定
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    # 計算の短縮
    if DEBUG:
        train = train.sample(1000).reset_index(drop=True)
        test = test.sample(1000).reset_index(drop=True)

    # 正解データの生成
    cat_id_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = train['CATEGORY'].map(cat_id_dict)
    y_test = test['CATEGORY'].map(cat_id_dict)

    # tokenizerの読み込み
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 10
    
    # datasetの作成
    dataset_train = TextDataset(train['TITLE'], y_train, tokenizer, max_len, device)
    dataset_test = TextDataset(test['TITLE'], y_test, tokenizer, max_len, device)
    # {'input_ids': [101, 2885, 6561, 24514, 2391, 2006, 8169, 2586, 102, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]}
    # BERTでは、変換の過程で元の文の文頭と文末に特殊区切り文字である[CLS]と[SEP]がそれぞれ挿入されるため、それらも101と102として系列に含まれています。0はパディングを表します。


    # train = pd.read_csv(f'{PATH}/80/train_title_id.csv')
    # test = pd.read_csv(f'{PATH}/80/test_title_id.csv')
    # test = test.reset_index(drop=True)

    # # tokenizerの読み込み
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # # paddingの実施
    # max_len = 10
    # train['TITLE'] = train['TITLE'].apply(lambda x : padding(x, max_len))
    # test['TITLE'] = test['TITLE'].apply(lambda x : padding(x, max_len))

    # # yの変換
    # cat_id_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    # train['CATEGORY'] = train['CATEGORY'].map(cat_id_dict)
    # test['CATEGORY'] = test['CATEGORY'].map(cat_id_dict)

    # # 辞書の読み込み
    # with open(f"{PATH}/80/word_id_dict.pkl", "rb") as tf:
    #     word_id_dict = pickle.load(tf)

    # N_LETTERS = len(word_id_dict.keys()) + 1 # pad分をplusする。
    # EMB_SIZE = 300
    # HIDDEN_SIZE = 50
    # N_CATEGORIES = 4

    # modelの定義
    # model = RNN(vocab_size=N_LETTERS,
    #             emb_dim=EMB_SIZE,
    #             hidden_size=HIDDEN_SIZE,
    #             output_size=N_CATEGORIES,
    #             word_id_dict=word_id_dict
    #             ).to(device)

    # パラメータの設定
    DROP_RATE = 0.4
    OUTPUT_SIZE = 4
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    if DEBUG:
        NUM_EPOCHS = 2
    LEARNING_RATE = 2e-5

    # モデルの定義
    model = BERTClass(DROP_RATE, OUTPUT_SIZE)

    # criterion, optimizerの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # datasetの定義
    # dataset_train = TextDataset(train['TITLE'], train['CATEGORY'], device)
    # dataset_test = TextDataset(test['TITLE'], test['CATEGORY'], device)

    # parameterの更新
    # BATCH_SIZE = 32
    # dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in tqdm(range(NUM_EPOCHS)):
        # 学習
        train_running_loss = train_fn(
            model, dataset_train, device, optimizer, criterion, BATCH_SIZE
            )
        print(train_running_loss)

        # 損失と正解率の算出
        train_loss, train_acc = calculate_loss_and_accuracy(model, dataset_train, device, criterion)
        test_loss, test_acc = calculate_loss_and_accuracy(model, dataset_test, device, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 20epoch毎にチェックポイントを生成
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"89_model_epoch{epoch}.pth")
            torch.save(
                optimizer.state_dict(),
                f"89_optimizer_epoch{epoch}.pth",
            )

    # グラフへのプロット
    losses = {"train": train_losses, "test": test_losses}

    accs = {"train": train_accs, "test": test_accs}

    make_graph(losses, "losses", bn=BATCH_SIZE, method='bert')
    make_graph(accs, "accs", bn=BATCH_SIZE, method='bert')

    print(f"train_acc: {train_acc: .4f}")
    print(f"test_acc: {test_acc: .4f}")