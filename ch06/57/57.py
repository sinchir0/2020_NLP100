import  pickle

import numpy as np

with open('../52/LR_with_cnt_feat.pickle', mode='rb') as fp:
    clf = pickle.load(fp)

with open('../51/vec_dict.pkl', 'rb') as f:
    vec_dict = pickle.load(f)

# train_feature = np.loadtxt('../51/train.feature.cntvec.txt')

vec_dict_list = [_ for _ in vec_dict.keys()]

print("Top 10")
for idx, col in enumerate(np.argsort(clf.coef_[0])[::-1]):
    print(f"col_name: {vec_dict_list[col]}, coef: {clf.coef_[0][col] :.2f}")
    if idx == 10:
        break;

print("Last 10")
for idx, col in enumerate(np.argsort(clf.coef_[0])):
    print(f"col_name: {vec_dict_list[col]}, coef: {clf.coef_[0][col] :.2f}")
    if idx == 10:
        break;

# Top 10
# col_name: deflation, coef: 1.65
# col_name: las, coef: 1.62
# col_name: hiddleston, coef: 1.51
# col_name: cent, coef: 1.46
# col_name: flip, coef: 1.33
# col_name: godard, coef: 1.32
# col_name: boys, coef: 1.32
# col_name: explorer, coef: 1.28
# col_name: 2billion, coef: 1.27
# col_name: lively, coef: 1.25
# col_name: praises, coef: 1.24

# Last 10
# col_name: accor, coef: -1.72
# col_name: done, coef: -1.30
# col_name: ceo, coef: -1.30
# col_name: pro, coef: -1.30
# col_name: calm, coef: -1.26
# col_name: cuddle, coef: -1.25
# col_name: iowa, coef: -1.25
# col_name: paula, coef: -1.22
# col_name: jackman, coef: -1.22
# col_name: hogs, coef: -1.21
# col_name: televised, coef: -1.21