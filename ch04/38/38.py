# 38. ヒストグラム
# 単語の出現頻度のヒストグラムを描け．ただし，横軸は出現頻度を表し，1から単語の出現頻度の最大値までの線形目盛とする．
# 縦軸はx軸で示される出現頻度となった単語の異なり数（種類数）である．

import pickle
import collections

import matplotlib.pyplot as plt

from ipdb import set_trace as st

if __name__ == "__main__":

    with open('../35/neko_list', mode="rb") as f:
        neko_list = pickle.load(f)

    word_list = [morphs[0] for morphs in neko_list]
    
    # 単語を頻度順にまとめる
    c_word = collections.Counter(word_list)
    _, word_counts = zip(*c_word.most_common())

    # 保存
    plt.hist(word_counts,bins=100)
    plt.savefig('hist.png')