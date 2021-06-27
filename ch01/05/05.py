# 05. n-gram
# 与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．

def ngram(seq: str, n: int):
    return list(zip(*[seq[i:] for i in range(n)]))

# iで分割した二つのlistをzipを使って頭だけ取って来ている。
# ここの理解は下のページが分かりやすい
# https://jackee777.hatenablog.com/entry/2019/05/03/223646

if __name__ == "__main__":
    s = 'I am an NLPer'

    print(ngram(s, n=2))
    print(ngram(s.split(), n=2))