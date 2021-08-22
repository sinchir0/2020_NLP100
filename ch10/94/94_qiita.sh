for N in `seq 1 20` ; do
    fairseq-interactive --path ../91/save91/checkpoint_last.pt --beam $N ../91/data91 < ../90/test.spacy.ja | grep '^H' | cut -f3 > 94.$N.out
done

for N in `seq 1 20` ; do
    fairseq-score --sys 94.$N.out --ref ../90/test.spacy.en > 94.$N.score
done