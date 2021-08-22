# 91. 機械翻訳モデルの訓練
# 90で準備したデータを用いて，ニューラル機械翻訳のモデルを学習せよ（ニューラルネットワークのモデルはTransformerやLSTMなど適当に選んでよい）

# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# wget https://download.pytorch.org/tutorial/data.zip

from __future__ import unicode_literals, print_function, division
from io import open
import random

from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import math
import time

import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.ticker as ticker
import numpy as np

import MeCab

from ipdb import set_trace as st

#Global変数の定義
MAX_LENGTH = 30

# SOS_token, EOS_tokenとして文章の最初と最後に追加している
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# # [ToDO]日本語用に変更が必要
# # Turn a Unicode string to plain ASCII, thanks to
# # https://stackoverflow.com/a/518232/2809427
# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn'
#     )

# # [ToDO]日本語用に変更が必要
# # Lowercase, trim, and remove non-letter characters
# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s

# [ToDO]今回のデータがこのreturnに合うように変更
def readLangs(lang1: str, lang2: str, reverse=False):
    print("Reading lines...")

    with open(lang1, mode="r") as f:
        lang1_list = f.readlines()

    with open(lang2, mode="r") as f:
        lang2_list = f.readlines()

    # 改行\nを削除する
    lang1_list = [line.replace(' \n','') for line in lang1_list]
    lang2_list = [line.replace('\n','') for line in lang2_list]

    # # Read the file and split into lines
    # lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
    #     read().strip().split('\n')

    # # Split every line into pairs and normalize
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [[lang1_sentence, lang2_sentence] for lang1_sentence, lang2_sentence in zip(lang1_list, lang2_list)]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# def filterPair(p):
#     """pairの両方がMAX_LENGTHより短く、engがeng_prefixesのどれかで始まっていたらTrue"""
#     return len(p[0].split(' ')) < MAX_LENGTH and \
#         len(p[1].split(' ')) < MAX_LENGTH and \
#         p[1].startswith(eng_prefixes)

# def filterPairs(pairs):
#     """filterPairnの適用"""
#     return [pair for pair in pairs if filterPair(pair)]

def filterPair(p):
    """pairの両方がMAX_LENGTHより短いならTrue"""
    return (len(p[0].split(' ')) < MAX_LENGTH) and (len(p[1].split(' ')) < MAX_LENGTH)

def filterPairs(pairs):
    """filterPairnの適用"""
    return [pair for pair in pairs if filterPair(pair)]

# Reading lines...
# Read 135842 sentence pairs
# Trimmed to 10599 sentence pairs
# Counting words...
# Counted words:
# fra 4345
# eng 2803
# ['elles se trouvent juste derriere vous .', 'they re right behind you .']

def prepareData(file_path_ja: str, file_path_en: str, reverse=False):
    """"""
    input_lang, output_lang, pairs = readLangs(file_path_ja, file_path_en, reverse)
    # pairs[1000]: ['je le suppose .', 'i guess so .']
    # 分かち書きされた文章が、要素２のlistで入っている。
    # pairs: list , len(pairs): 135842
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    # len(pairs): 10599
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    # torch.Size([1, 1, 256])

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    # torch.Size([30, 256])

    loss = 0

    for idx in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[idx], encoder_hidden)
        # encoder_output.size() -> torch.Size([1, 1, 256])
        # encoder_hidden.size() -> torch.Size([1, 1, 256])

        encoder_outputs[idx] = encoder_output[0, 0]
        # torch.Size([10, 256])のうち、1行目が埋まる。
        # これをinput_length（見たときは６）分繰り返す

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    # teacher_forcing_ratioの確率で強制的にteacher_forcingを行う
    # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training-the-model
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for idx in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[idx])
            decoder_input = target_tensor[idx]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # decoder_output: torch.Size([1, 2803])
            # この2803はengの語彙数、fraは4345
            # decoder_hidden: torch.Size([1, 1, 256])
            # decoder_attention: torch.Size([1, 10])

            topv, topi = decoder_output.topk(1)
            # decoder_outputはlog(softmax)の値
            # その中で最も大きい値を取るため、つまりは確率が最も大きい語彙を一個選択するのと同じ意味になる。

            # topv: 最も値が大きいdecoder_outputのvalue
            # topi: 最も値が大きいdecoder_outputのindex
            
            decoder_input = topi.squeeze().detach()  # detach from history as input
            # topiを次のinputに使う

            loss += criterion(decoder_output, target_tensor[di])
            # decoder_output：今回の予測結果　と、target_tensorの該当位置diのlossを加える。
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # 75000回、適当にペアを選ぶ。
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                for i in range(n_iters)]
    training_pairs = [tensorsFromPair(pair)
                    for pair in pairs]
    # training_pairsはlist
    # training_pairs[0]はtensorのtuple
    # 多分engとfraのペアのtensor
    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, n_iters + 1)):
        training_pair = training_pairs[iter - 1] # 0から
        
        # training_pairを二つに分ける
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # 後で中身確認
        loss = train(input_tensor, target_tensor, encoder,
                    decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) loss %.4f' % (timeSince(start, iter / n_iters),
                                        iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    save_plot(plot_losses, n_iters)

    # Save Model
    torch.save(encoder.state_dict(), f'encoder_iter{n_iters}.pth')
    torch.save(decoder.state_dict(), f'decoder_iter{n_iters}.pth')

def save_plot(points: list, n_iters: int):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.5)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(f'translation_loss_iter{n_iters}.png')

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def showAttention(input_sentence, output_words, attentions, n_iters: int):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                    ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(f'{input_sentence[0:3]}_attention_iter{n_iters}.png', bbox_inches='tight')

def evaluateAndShowAttention(input_sentence: str, n_iters: int):
    # 分かち書きの実施
    mecab = MeCab.Tagger("-Owakati")
    sentence = mecab.parse(input_sentence)

    # スペース+空白文字の削除
    sentence = sentence.rstrip(' \n')

    # 単語の出力
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, sentence)
    print('input =', sentence)
    print('output =', ' '.join(output_words))
    showAttention(sentence, output_words, attentions, n_iters)

if __name__ == "__main__":

    N_ITERS = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use {device}')

    SOS_token = 0
    EOS_token = 1

    # # [ToDO]日本語用に変更が必要
    # eng_prefixes = (
    #     "i am ", "i m ",
    #     "he is", "he s ",
    #     "she is", "she s ",
    #     "you are", "you re ",
    #     "we are", "we re ",
    #     "they are", "they re "
    # )

    input_lang, output_lang, pairs = prepareData('../90/train.mecab.ja', '../90/train.spacy.en')
    print(random.choice(pairs))

    teacher_forcing_ratio = 0.5

    hidden_size = 256
    #+1は未知語用のidx
    encoder1 = EncoderRNN(input_lang.n_words+1, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, n_iters=N_ITERS, print_every=5, plot_every=5)

    # evaluateRandomly(encoder1, attn_decoder1)

    evaluateAndShowAttention("日本の水墨画を一変させた。", N_ITERS)

    evaluateAndShowAttention("諱は「等楊（とうよう）」、もしくは「拙宗（せっしゅう）」と号した。", N_ITERS)

    evaluateAndShowAttention("備中国に生まれ 、京都・相国寺に入ってから周防国に移る。", N_ITERS)

    evaluateAndShowAttention("その後遣明使に随行して中国（明）に渡って中国の水墨画を学んだ。", N_ITERS)

    # N_ITERS = 10000
    # input = 諱 は 「 等 楊 （ とう よう ） 」 、 もしくは 「 拙 宗 （ せっしゅう ） 」 と 号 し た 。
    # output = " ( , ' ) ' , ' the <EOS>
    # input = 備中 国 に 生まれ 、 京都 ・ 相国寺 に 入っ て から 周防 国 に 移る 。
    # output = On February , , the and in the and and of , and the  <EOS>
    # input = その後 遣 明 使 に 随行 し て 中国 （ 明 ） に 渡っ て 中国 の 水墨 画 を 学ん だ 。
    # output = In , he was a and and and the of and . <EOS>

    # N_ITERS = 50000
    # input = 諱 は 「 等 楊 （ とう よう ） 」 、 もしくは 「 拙 宗 （ せっしゅう ） 」 と 号 し た 。
    # output = " ( , ' ) ' , ' the <EOS>
    # input = 備中 国 に 生まれ 、 京都 ・ 相国寺 に 入っ て から 周防 国 に 移る 。
    # output = On February , , the and in the and and of , and the  <EOS>
    # input = その後 遣 明 使 に 随行 し て 中国 （ 明 ） に 渡っ て 中国 の 水墨 画 を 学ん だ 。
    # output = In , he was a and and and the of and . <EOS>