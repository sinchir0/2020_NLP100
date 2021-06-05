import pickle

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

import pdb; pdb.set_trace()

train_y = pd.read_table('../50/train.txt', usecols=['CATEGORY'])
# valid_y = pd.read_table('../50/valid.txt',index_col=0)
# test_y = pd.read_table('../50/test.txt',index_col=0)

train_feature = np.loadtxt('../51/train.feature.cntvec.txt')
# valid_feature = pd.read_table('../51/valid.feature.cntvec.txt',index_col=0)
# test_feature = pd.read_table('../51/test.feature.cntvec.txt',index_col=0)

clf = LogisticRegression(solver='liblinear')
clf.fit(train_feature, train_y['CATEGORY'])

with open('LR_with_cnt_feat.pickle', 'wb') as f:
    pickle.dump(clf, f)