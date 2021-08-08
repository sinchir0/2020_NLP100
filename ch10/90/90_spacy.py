import spacy
from tqdm import tqdm

from ipdb import set_trace as st

if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm')

    # input_filename = 'kftt-data-1.0/data/orig/kyoto-train.en'
    # input_filename = 'kftt-data-1.0/data/orig/kyoto-dev.en'
    input_filename = 'kftt-data-1.0/data/orig/kyoto-test.en'

    wakati_line = []

    with open(input_filename, mode="r") as f:
        all_list = f.readlines()

    # span = 30

    # for idx in range(0, len(all_list), span):
    #     part_list = all_list[idx:idx+span]
    #     part_txt = ' '.join(part_list)
    #     part_txt_spacy = nlp(part_txt)
    #     token =  [_ for _ in part_txt_spacy]
    #     st()
    #     # 分かち書きできてない問題

    # 遅い
    for line in tqdm(all_list):
        line_spacy = nlp(line)
        line_list = [str(word) for word in line_spacy]
        wakati_txt = ' '.join(line_list)
        wakati_line.append(wakati_txt)
    
    # output_filename = 'train.spacy.en'
    # output_filename = 'dev.spacy.en'
    output_filename = 'test.spacy.en'

    with open(output_filename, mode='wt') as f:
        f.writelines(wakati_line)