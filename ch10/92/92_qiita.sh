fairseq-interactive --cpu --path ../91/save91/checkpoint_last.pt ../91/data91 < ../90/test.spacy.ja | grep '^H' | cut -f3 > 92.out