# 16. ファイルをN分割する
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のファイルを行単位でN分割せよ．同様の処理をsplitコマンドで実現せよ．

import sys

if __name__ == "__main__":

    N = int(sys.argv[1])

    with open("../13/col1_col2_merge.txt") as f:
        col1_col2_merge = f.readlines()

    split_point = int(len(col1_col2_merge) / N)

    for num in range(0, split_point, N):
        print("".join(col1_col2_merge[num : num + N]))
        print("---------------------------------")

    # split -l 10 ../13/col1_col2_merge.txt
    # Nはファイル数
