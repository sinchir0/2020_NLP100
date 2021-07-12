# 21. カテゴリ名を含む行を抽出
# 記事中でカテゴリ名を宣言している行を抽出せよ．

# https://github.com/upura/nlp100v2020/blob/master/ch03/ans21.py

import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("../20/jawiki-country.json.gz", lines=True)
    uk_text = df.query('title=="イギリス"')["text"].values[0]
    uk_texts = uk_text.split("\n")
    ans = list(filter(lambda x: "[Category:" in x, uk_texts))
    print(ans)