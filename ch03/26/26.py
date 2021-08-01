# 26. 強調マークアップの除去
# 25の処理時に，テンプレートの値からMediaWikiの強調マークアップ（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ（参考: マークアップ早見表）．

# 正しいか確認していない

import gzip
import json
import re

import pandas as pd
from ipdb import set_trace as st


def remove_stress(dc):
    """dictで取得した各要素について、「'が複数回続く」と言う条件を満たしたvalueを何もなしに置換する"""
    r = re.compile("'+")
    return {k: r.sub("", v) for k, v in dc.items()}


if __name__ == "__main__":

    with gzip.open("../20/jawiki-country.json.gz", mode="rt", encoding="utf-8") as fin:
        dict_list = [json.loads(line) for line in fin]

    # 全要素をstrへ変換
    text_list = [json.dumps(dict_val, ensure_ascii=False) for dict_val in dict_list]

    pattern = re.compile("\|(.+?)\s=\s*(.+)")

    ans = {}
    for line in text_list:
        r = re.search(pattern, line)
        if r:
            ans[r[1]] = r[2]
    print(remove_stress(ans))
