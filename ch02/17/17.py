# 17. １列目の文字列の異なり
# 1列目の文字列の種類（異なる文字列の集合）を求めよ．確認にはcut, sort, uniqコマンドを用いよ．

import sys

if __name__ == "__main__":

    with open("../13/col1_col2_merge.txt", 'rt') as f:
        col1_col2_merge = f.readlines()

    # 元のファイルがbinaryだとrb
    # readlinesは全部読み込むので、実務では使わない

    col1_set = set()

    for col1_col2 in col1_col2_merge:
        col1 = col1_col2.split('\t')[0]
        col1_set.add(col1)

    print('\n'.join((sorted(list(col1_set)))))

    # 別解
    # print((sorted(list({line.strip().split()[0] for line in col1_col2_merge}))))

    # UNIXの例
    # cut -f 1 ../13/col1_col2_merge.txt | sort | uniq
    # Collectionでやり直してみる。