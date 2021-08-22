# https://qiita.com/nymwa/items/867e05a43060d036a174
import re
import spacy

nlp = spacy.load('en_core_web_sm')
for src, dst in [
    ('kftt-data-1.0/data/orig/kyoto-train.en', 'train.spacy.en'),
    ('kftt-data-1.0/data/orig/kyoto-dev.en', 'dev.spacy.en'),
    ('kftt-data-1.0/data/orig/kyoto-test.en', 'test.spacy.en'),
]:
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = nlp.make_doc(x)
            x = ' '.join([doc.text for doc in x])
            print(x, file=g)