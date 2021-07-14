# 22. カテゴリ名の抽出
# 記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．

# https://github.com/wakamezake/nlp_q100_2020/tree/master/ch03

import re

import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("../20/jawiki-country.json.gz", lines=True)
    uk_text = df.query('title=="イギリス"')["text"].values[0]
    uk_texts = uk_text.split("\n")
    ans = list(filter(lambda x: "[Category:" in x, uk_texts))
    print(ans)
    # import ipdb; ipdb.set_trace()

    # pattern = re.compile(r'\[\[Category:(.*?)\]\]')
    pattern = re.compile(r"\[\[Category:(.*?)(?:\|.*)?\]\]")
    for txt in ans:
        match_txt = pattern.match(txt).groups()
        print(match_txt)
    # ('イギリス',)
    # ('イギリス連邦加盟国',)
    # ('英連邦王国',)
    # ('G8加盟国',)
    # ('欧州連合加盟国',)
    # ('海洋国家',)
    # ('現存する君主国',)
    # ('島国',)
    # ('1801年に成立した国家・領域',)

    # 質問点：正規表現の作り方
    # 今回は他の人の解答見て正規表現コピーした。
