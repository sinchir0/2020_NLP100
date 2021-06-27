# 06. 集合
# “paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ

def ngram(seq: str, n: int):
    return list(zip(*[seq[i:] for i in range(n)]))

if __name__ == "__main__":

    s_1 = "paraparaparadise"
    s_2 = "paragraph"

    X = set(ngram(s_1, n=2))
    Y = set(ngram(s_2, n=2))

    union = X | Y
    interact = X & Y
    diff = X - Y

    print(f'union {union}')
    print(f'interact {interact}')
    print(f'diff {diff}')

    print(('s', 'e') in X)
    print(('s', 'e') in Y)