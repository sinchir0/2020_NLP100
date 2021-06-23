# 05. n-gram
# 与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．

def bigram(sentence: str, split: str):
    if split == 'word':
        sentence = sentence.split(' ')
    elif split == 'character':
        sentence = [_ for _ in sentence.replace(' ','')]

    return [txt for txt in zip(sentence[0:], sentence[1:])]

if __name__ == "__main__":

    print(bigram(sentence='I am an NLPer',split='word'))
    print(bigram(sentence='I am an NLPer',split='character'))