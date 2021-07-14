import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

train_y = pd.read_table("../50/train.txt", usecols=["CATEGORY"])

train_feature = np.loadtxt("../51/train.feature.cntvec.txt")

clf = LogisticRegression(solver="liblinear")
clf.fit(train_feature, train_y["CATEGORY"])

with open("LR_with_cnt_feat.pickle", "wb") as f:
    pickle.dump(clf, f)
