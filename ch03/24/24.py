# 24. ファイル参照の抽出
# 記事から参照されているメディアファイルをすべて抜き出せ．

import re

import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("../20/jawiki-country.json.gz", lines=True)
    uk_text = df.query('title=="イギリス"')["text"].values[0]
    uk_texts = uk_text.split("\n")

    pattern = re.compile(r"\[\[ファイル:(.*)")

    for txt in uk_texts:
        m = re.match(pattern, txt)
        if m:
            import ipdb

            ipdb.set_trace()
            match_txt = pattern.match(txt).groups()
            print("".join(match_txt))
