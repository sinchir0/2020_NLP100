# 注：このコードは動きません

# 64. アナロジーデータでの実験
# 単語アナロジーの評価データをダウンロードし，vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，そのベクトルと類似度が最も高い単語と，その類似度を求めよ．求めた単語と類似度は，各事例の末尾に追記せよ．

# !wget http://download.tensorflow.org/data/questions-words.txt

import time
from typing import Tuple
from multiprocessing import Pool
from multiprocessing import Process, Manager

import gensim

def get_most_similar(input_data) -> str:
    '''questions-wordsのデータから、類似単語と類似度を計算する'''

    # 引数を分ける
    # model, line = input_data
    
    # 最初の単語が:だった場合は\nだけ削除して返す。
    first_word = line.split(' ')[0]
    if first_word == ':':
        return line.replace('\n','')
    
    # 類似度を計算
    second_word = line.split(' ')[1]
    third_word = line.split(' ')[2]

    result = model.most_similar(positive=[second_word, third_word], negative=[first_word], topn=1)

    line_add_result = line[:-1] + ' ' + str(result[0])

    return line_add_result

if __name__ == "__main__":

    start = time.time()

    # データの読み込み
    model = gensim.models.KeyedVectors.load_word2vec_format('../60/GoogleNews-vectors-negative300.bin', binary=True)

    with open('questions-words_dummy.txt') as f:
        questions_words = f.readlines()

    # 引数をまとめる
    input_data_list = [(model,line) for line in questions_words]

    import ipdb; ipdb.set_trace()

    with Pool(4) as p:
        result_list = p.map(get_most_similar, input_data_list)

    join_result = '\n'.join(result_list)

    with open("result_dummy_multiprocess_test.txt", 'w') as f:
        f.write(join_result)

    elapsed_time = time.time() - start
    print (f"elapsed_time: {elapsed_time: .2f}[sec]")
    # 普通にやった場合
    # elapsed_time: 22.35[sec]