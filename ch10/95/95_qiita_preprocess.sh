fairseq-preprocess -s ja -t en \
    --trainpref train.sub \
    --validpref dev.sub \
    --destdir data95  \
    --workers 20