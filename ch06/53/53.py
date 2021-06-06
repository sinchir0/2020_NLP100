import pickle

import numpy as np

with open('../52/LR_with_cnt_feat.pickle', mode='rb') as fp:
    clf = pickle.load(fp)

train_feature = np.loadtxt('../51/train.feature.cntvec.txt')
valid_feature = np.loadtxt('../51/valid.feature.cntvec.txt')
test_feature = np.loadtxt('../51/test.feature.cntvec.txt')

train_pred = clf.predict(train_feature)
valid_pred = clf.predict(valid_feature)
test_pred = clf.predict(test_feature)

np.save('../53/train_pred', train_pred)
np.save('../53/valid_pred', valid_pred)
np.save('../53/test_pred', test_pred)