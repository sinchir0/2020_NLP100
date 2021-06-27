# 08. 暗号文
# 与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
# 英小文字ならば(219 - 文字コード)の文字に置換
# その他の文字はそのまま出力
# この関数を用い，英語のメッセージを暗号化・復号化せよ．

import re


def cipher(text: str) -> str:
    return [
        chr(219 - ord(char)) if (re.match(r"[a-z]", char)) else char for char in text
    ]


if __name__ == "__main__":
    raw_text = "I am an NLPer"
    encypted = "".join(cipher(raw_text))
    dencypted = "".join(cipher(encypted))

    print(raw_text)
    print(encypted)
    print(dencypted)
