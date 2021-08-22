import re
import sentencepiece as spm

spm.SentencePieceTrainer.Train('--input=../90/kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=1.0')

sp = spm.SentencePieceProcessor()
sp.Load('kyoto_ja.model')

for src, dst in [
    ('../90/kftt-data-1.0/data/orig/kyoto-train.ja', 'train.sub.ja'),
    ('../90/kftt-data-1.0/data/orig/kyoto-dev.ja', 'dev.sub.ja'),
    ('../90/kftt-data-1.0/data/orig/kyoto-test.ja', 'test.sub.ja'),
]:
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = sp.encode_as_pieces(x)
            x = ' '.join(x)
            print(x, file=g)