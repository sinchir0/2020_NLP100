from __future__ import division, print_function, unicode_literals

import random
from io import open

import MeCab
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global変数の定義
MAX_LENGTH = 30
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLangs(lang1: str, lang2: str, reverse=False):

    with open(lang1, mode="r") as f:
        lang1_list = f.readlines()

    with open(lang2, mode="r") as f:
        lang2_list = f.readlines()

    # 改行\nを削除する
    lang1_list = [line.replace(" \n", "") for line in lang1_list]
    lang2_list = [line.replace("\n", "") for line in lang2_list]

    pairs = [
        [lang1_sentence, lang2_sentence]
        for lang1_sentence, lang2_sentence in zip(lang1_list, lang2_list)
    ]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    """pairの両方がMAX_LENGTHより短いならTrue"""
    return (len(p[0].split(" ")) < MAX_LENGTH) and (len(p[1].split(" ")) < MAX_LENGTH)


def filterPairs(pairs):
    """filterPairnの適用"""
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(file_path_ja: str, file_path_en: str, reverse=False):
    """"""
    input_lang, output_lang, pairs = readLangs(file_path_ja, file_path_en, reverse)

    pairs = filterPairs(pairs)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

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
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    # lang.word2index[word]に含まれない場合は適当なidxを返すように変更する。
    indexes = []
    for word in sentence.split(" "):
        try:
            indexes.append(lang.word2index[word])
        except:
            indexes.append(80393)  # 80393は使われていない語彙

    return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def translate(
    encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH
):
    # 分かち書きの実施
    mecab = MeCab.Tagger("-Owakati")
    sentence = mecab.parse(sentence)

    # スペース+空白文字の削除
    sentence = sentence.rstrip(" \n")

    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        translated_sentence = " ".join(decoded_words[:-1])  # '<EOS>を省く'

        return translated_sentence


def translate_api(input_text: str) -> str:
    USE_ITER = 5

    with open("../90/test.mecab.ja", mode="r") as f:
        test_ja = f.readlines()

    # 1行目のInfoboxBuddhistは飛ばす
    test_ja = test_ja[1:]

    # 改行\nを削除する
    test_ja = [line.rstrip(" \n") for line in test_ja]

    input_lang, output_lang, _ = prepareData(
        "../90/train.mecab.ja", "../90/train.spacy.en"
    )

    hidden_size = 256

    encoder1 = EncoderRNN(input_lang.n_words + 1, hidden_size).to(device)  # +1は未知語用のidx
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(
        device
    )

    encoder1.load_state_dict(
        torch.load(
            f"../91/encoder_iter{USE_ITER}.pth", map_location=torch.device("cpu")
        )
    )
    attn_decoder1.load_state_dict(
        torch.load(
            f"../91/decoder_iter{USE_ITER}.pth", map_location=torch.device("cpu")
        )
    )

    return translate(encoder1, attn_decoder1, input_text, input_lang, output_lang)
