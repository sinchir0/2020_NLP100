subword-nmt learn-bpe -s 16000 < ../90/kftt-data-1.0/data/orig/kyoto-train.en > kyoto_en.codes
subword-nmt apply-bpe -c kyoto_en.codes < ../90/kftt-data-1.0/data/orig/kyoto-train.en > train.sub.en
subword-nmt apply-bpe -c kyoto_en.codes < ../90/kftt-data-1.0/data/orig/kyoto-dev.en > dev.sub.en
subword-nmt apply-bpe -c kyoto_en.codes < ../90/kftt-data-1.0/data/orig/kyoto-test.en > test.sub.en