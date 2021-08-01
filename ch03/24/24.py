# 24. ファイル参照の抽出
# 記事から参照されているメディアファイルをすべて抜き出せ．

import re

import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("../20/jawiki-country.json.gz", lines=True)
    uk_text = df.query('title=="イギリス"')["text"].values[0]
    uk_texts = uk_text.split("\n")

    pattern = re.compile(r'\[\[ファイル:(.+?)\|')

    for txt in uk_texts:
        m = re.match(pattern, txt)
        if m:
            match_txt = pattern.match(txt).groups()
            print(''.join(match_txt))
            # Descriptio Prime Tabulae Europae.jpg
            # Lenepveu, Jeanne d'Arc au siège d'Orléans.jpg
            # London.bankofengland.arp.jpg
            # Battle of Waterloo 1815.PNG
            # Uk topo en.jpg
            # BenNevis2005.jpg
            # Population density UK 2011 census.png
            # 2019 Greenwich Peninsula & Canary Wharf.jpg
            # Leeds CBD at night.jpg
            # Palace of Westminster, London - Feb 2007.jpg
            # Scotland Parliament Holyrood.jpg
            # Donald Trump and Theresa May (33998675310) (cropped).jpg
            # Soldiers Trooping the Colour, 16th June 2007.jpg
            # City of London skyline from London City Hall - Oct 2008.jpg
            # Oil platform in the North SeaPros.jpg
            # Eurostar at St Pancras Jan 2008.jpg
            # Heathrow Terminal 5C Iwelumo-1.jpg
            # UKpop.svg
            # Anglospeak.svg
            # Royal Aberdeen Children's Hospital.jpg
            # CHANDOS3.jpg
            # The Fabs.JPG
            # Wembley Stadium, illuminated.jpg