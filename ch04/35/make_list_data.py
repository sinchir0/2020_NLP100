# neko.txt.mecabのデータをlistのデータで持つようにする

import pickle

from ipdb import set_trace as st

if __name__ == "__main__":

    filename = '../30/neko.txt.mecab'

    result_list = []

    with open(filename, mode="r") as f:
        for line in f: # 1行ずつ読込
            fields = line.split('\t')
            if len(fields) != 2 or fields[0] == '':
                continue
            else:
                line = line.replace('\t',',')
                line = line.replace('\n','')
                line = line.replace('\u3000',' ')
                line_list = line.split(',')
                result_list.append(line_list)
                
    with open('neko_list', mode="wb") as f:
        pickle.dump(result_list, f)