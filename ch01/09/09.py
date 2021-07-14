# 09. Typoglycemia
# スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
# ただし，長さが４以下の単語は並び替えないこととする．
# 適当な英語の文（例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）を与え，その実行結果を確認せよ．

import random


def text_randomizer(text: str):
    """長さが４文字より上の単語をランダムに並び替える"""
    return [
        "".join(random.sample(text, len(text))) if len(text) > 4 else text
        for text in center.split(" ")
    ]


if __name__ == "__main__":

    raw_text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

    top = raw_text[:1]
    center = raw_text[1:-1]
    last = raw_text[-1:]

    center_shuffle = " ".join(text_randomizer(center))

    print(top, center_shuffle, last)
