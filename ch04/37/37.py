# 37. 「猫」と共起頻度の高い上位10語
# 「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import re
import MeCab

import collections

import matplotlib.pyplot as plt
import japanize_matplotlib

from ipdb import set_trace as st

def ngram(seq: list, n: int):
    return list(zip(*[seq[i:] for i in range(n)]))

if __name__ == "__main__":

    # 読み込み
    with open("../30/neko.txt") as f:
        all_text = [line for line in f.readlines()]

    # MeCabによる分かち書き
    mecab = MeCab.Tagger("-Owakati")
    parse_text = [mecab.parse(line).strip() for line in all_text]

    # 猫が存在する行に絞る
    neko_text = [line for line in parse_text if re.search('猫', line)]

    # 猫と一緒に登場する単語を抽出
    neko_bigram_list = []
    for text in neko_text:
        text_bigram = ngram(text.split(), 2)
        # 猫を要素含むデータのみ抽出
        text_bigram_only_neko = [text_tuple for text_tuple in text_bigram if "猫" in text_tuple]
        # 猫が先頭に来るよう並び替え
        for text_tuple in text_bigram_only_neko:
            if text_tuple[1] == '猫':
                text_tuple = (text_tuple[1], text_tuple[0])
            neko_bigram_list.append(text_tuple)

    c = collections.Counter(neko_bigram_list)

    # 上位10件のlistを取得
    c_top10 = c.most_common(10)

    # 描画用にデータを取得
    col_keys_list = [text_tuple[0][1] for text_tuple in col_top10]
    col_values_list = [text_tuple[1] for text_tuple in col_top10]

    plt.bar(values, counts)
    plt.savefig('37.png')