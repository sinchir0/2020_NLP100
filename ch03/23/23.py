# 23. セクション構造
# 記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ．

# レベルの意味が不明だけど、多分
# == セクション名 ==:1
# === セクション名 ===:2
# ==== セクション名 ====:3

import re

import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("../20/jawiki-country.json.gz", lines=True)
    uk_text = df.query('title=="イギリス"')["text"].values[0]
    uk_texts = uk_text.split("\n")

    pattern = re.compile(r"\=\= (.*?)\=\=")

    for txt in uk_texts:
        match_txt = pattern.match(txt).groups()
        print(match_txt)
