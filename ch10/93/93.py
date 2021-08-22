# 93. BLEUスコアの計測
# 91で学習したニューラル機械翻訳モデルの品質を調べるため，評価データにおけるBLEUスコアを測定せよ．
import pickle
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu
from typing import List

from ipdb import set_trace as st

def evaluate_bleu(pred: List[list], ans: List[list]) -> int:
    pred_ans_dict = defaultdict(list)
    for pred, ans in zip(pred, ans):
        pred_ans_dict[pred].append(ans)

    hypothesis = []
    references = []
    
    for pred_word, ans_word in pred_ans_dict.items():
        hypothesis.append(pred_word)
        references.append(ans_word)

    bleu_score = corpus_bleu(references, hypothesis)
    return bleu_score

if __name__ == "__main__":
    USE_ITER = 5
    # # bleuscoreを正しく計算できていることの確認
    # score1.0
    # pred1 = 'The turmoil is said to be the last and largest peasant uprising in the feudal system of Japan.'
    # ans1 = 'The turmoil is said to be the last and largest peasant uprising in the feudal system of Japan.'
    # pred2 = 'The population was bloating at a fast pace, finally exceeding several hundred million people.'
    # ans2 = 'The population was bloating at a fast pace, finally exceeding several hundred million people.'
    
    # pred = [pred1, pred2]
    # ans = [ans1, ans2]

    # score = evaluate_bleu(pred, ans)
    # print(score)
    # 1.0

    # score0.06
    # pred1 = "The turmoil is said to be the last and largest peasant uprising in the feudal system of Japan."
    # ans1 = "This bread is so good!"
    # pred2 = "The population was bloating at a fast pace, finally exceeding several hundred million people."
    # ans2 = "Today's not a good day for anything."
    
    # pred = [pred1, pred2]
    # ans = [ans1, ans2]

    # score = evaluate_bleu(pred, ans)
    # print(score)

    # 読み込み
    with open(f'../92/filter_test_ja_useiter{USE_ITER}.txt', 'rb') as f:
        filter_test_ja = pickle.load(f)
    with open(f'../92/translated_useiter{USE_ITER}.txt', 'rb') as f:
        translated = pickle.load(f)
    with open(f'../92/filter_test_en.txt', 'rb') as f:
        filter_test_en = pickle.load(f)

    score = evaluate_bleu(translated, filter_test_en)
    print(score)
    # 0.008798193729726685
