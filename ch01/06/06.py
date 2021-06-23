# 06. 集合
# “paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ

def bigram(sentence: str, split: str):
    if split == 'word':
        sentence = sentence.split(' ')
    elif split == 'character':
        sentence = [_ for _ in sentence.replace(' ','')]

    return [txt for txt in zip(sentence[0:], sentence[1:])]

if __name__ == "__main__":
    X = set(bigram('paraparaparadise' ,split='character'))
    Y = set(bigram('paragraph' ,split='character'))
    
    interaction = X | Y
    volume = X & Y
    diff = X - Y

    print(X)
    print(Y)
    print(interaction)
    print(volume)
    print(diff)

    print(('s', 'e') in X)
    print(('s', 'e') in Y)