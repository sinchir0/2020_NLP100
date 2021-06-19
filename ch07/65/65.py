# 65. アナロジータスクでの正解率Permalink
# 64の実行結果を用い，意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ．

# https://tomowarkar.github.io/blog/posts/nlp100-07/#65-%E3%82%A2%E3%83%8A%E3%83%AD%E3%82%B8%E3%83%BC%E3%82%BF%E3%82%B9%E3%82%AF%E3%81%A7%E3%81%AE%E6%AD%A3%E8%A7%A3%E7%8E%87
# カテゴリ名にgramが含まれるものを文法的アナロジー, そうでないものを意味的アナロジーとします。

import numpy as np
import gensim


if __name__ == "__main__":
    # model = gensim.models.KeyedVectors.load_word2vec_format('../60/GoogleNews-vectors-negative300.bin', binary=True)

    with open('../64/result.txt') as f:
        result = f.readlines()

    syntactic, semantic = [], []

    for line in result:
        if line.startswith(": gram"):
            ctg = "syntactic"
        elif line.startswith(":"):
            ctg = "semantic"
        else:
            true_word = line.split(' ')[3]
            pred_word = line.split(' ')[4][2:-2]
            is_correct = (true_word == pred_word)
            if ctg == "syntactic":
                syntactic.append(is_correct)
            elif ctg == "semantic":
                semantic.append(is_correct)
            else:
                print('No ctg')

    syntactic_acc_rate = np.array(syntactic).mean()
    semantic_acc_rate = np.array(semantic).mean()
    
    print(f'semantic_acc_rate: {semantic_acc_rate: .4f}')
    print(f'syntactic_acc_rate: {syntactic_acc_rate: .4f}')
    # semantic_acc_rate:  0.7309
    # syntactic_acc_rate:  0.7400