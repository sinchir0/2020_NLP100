# 61. 単語の類似度
# “United States”と”U.S.”のコサイン類似度を計算せよ．

import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == "__main__":
    # modelのload
    model = gensim.models.KeyedVectors.load_word2vec_format(
        "../60/GoogleNews-vectors-negative300.bin", binary=True
    )

    print(cos_sim(model["United_States"], model["U.S."]))
    # 0.7310775
