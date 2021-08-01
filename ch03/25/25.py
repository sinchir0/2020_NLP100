# 25. テンプレートの抽出
# 記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，
# 辞書オブジェクトとして格納せよ．

import re

import pandas as pd

if __name__ == "__main__":
    
    df = pd.read_json("../20/jawiki-country.json.gz", lines=True)
    uk_text = df.query('title=="イギリス"')["text"].values[0]
    # import ipdb; ipdb.set_trace()
    uk_texts = uk_text.split("\n")

    pattern = re.compile(r'\|(.+?)\s=\s*(.+)')
    # \| : |という文字を検出
    
    # (.+?) : 
    ### . 任意の１文字
    ### + １回以上の繰り返し

    # \s=\s
    ### \s 任意の空白文字
    
    # *
    ### ０回以上の繰り返し

    # (.+)
    ### . 任意の１文字
    ### + １回以上の繰り返し

    for txt in uk_texts:
        m = re.match(pattern, txt)
        if m:
            match_txt = pattern.match(txt).groups()
            print(''.join(match_txt))
            #import ipdb; ipdb.set_trace()