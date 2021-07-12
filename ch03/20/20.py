# 20. JSONデータの読み込み
# Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．
# 問題21-29では，ここで抽出した記事本文に対して実行せよ．

# wget https://nlp100.github.io/data/jawiki-country.json.gz
# gzip -d jawiki-country.json.gz

# https://github.com/upura/nlp100v2020/blob/master/ch03/ans20.py

import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("jawiki-country.json.gz", lines=True)
    uk_text = df.query('title=="イギリス"')["text"].values[0]
    print(uk_text)