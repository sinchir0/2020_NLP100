# 39. Zipfの法則
# 単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．

import pickle
import collections

import matplotlib.pyplot as plt

from ipdb import set_trace as st

if __name__ == "__main__":

    with open('../35/neko_list', mode="rb") as f:
        neko_list = pickle.load(f)

    word_list = []
    word_list = [morphs[0] for morphs in neko_list]
    
    # 単語を頻度順にまとめる
    c_word = collections.Counter(word_list)
    _, word_counts = zip(*c_word.most_common())

    # 頻度の頻度を取得する。
    word_counts_c = collections.Counter(word_counts)
    x, y = zip(*word_counts_c.most_common())

    # 保存
    plt.scatter(x, y)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('zipf.png')