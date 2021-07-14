# 80. ID番号への変換
# 問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
# 学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，学習データ中で2回以上出現する単語にID番号を付与せよ．
# そして，与えられた単語列に対して，ID番号の列を返す関数を実装せよ．ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ

import collections
import pickle

import pandas as pd

from util import preprocess


def change_word_to_id(input_word: str, word_id_dict: dict) -> str:
    # ID番号へ変換、辞書に存在しないものは0をいれる
    result_list = []
    for word in input_word.split():
        if word in word_id_dict:
            result_list.append(str(word_id_dict[word]))
        else:
            result_list.append('0')

    return ' '.join(result_list)

if __name__ == '__main__':

    # データの読み込み
    train = pd.read_csv('../../ch06/50/train.txt', sep='\t', index_col=0)

    # 前処理
    train['TITLE'] = train[['TITLE']].apply(preprocess)

    # 全文章を一つにまとめたstrを生成
    all_sentence_list = ' '.join(train['TITLE'].tolist()).split(' ')

    # 全文章に含まれる単語の頻度を計算
    all_word_cnt = collections.Counter(all_sentence_list)

    # 出現頻度が2回以上の単語のみを取得
    word_cnt_over2 = [i for i in all_word_cnt.items() if i[1] >= 2]
    word_cnt_over2 = sorted(word_cnt_over2, key=lambda x : x[1], reverse=True)

    # 単語のみ取得
    word_over2 = [i[0] for i in word_cnt_over2]
    # ID番号を取得
    id_list = [i for i in range(1,len(word_over2))]

    # 単語とID番号をdictへとまとめる
    word_id_dict = dict(zip(word_over2, id_list))

    # 出力
    with open("word_id_dict.pkl", "wb") as tf:
        pickle.dump(word_id_dict, tf)

    # trainのTITLEをID番号へと変換
    train['TITLE'] = train['TITLE'].apply(lambda x : change_word_to_id(x, word_id_dict))

    # 出力
    train.to_pickle('train_title_id.pkl')