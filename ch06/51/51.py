# n-gram, w2v, tf-idf, fast-text, Universal Encoder? yukiさんがatmaで使っていたやつ
# を試してみたい。
import  pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_table('../50/train.txt',index_col=0)
valid = pd.read_table('../50/valid.txt',index_col=0)
test = pd.read_table('../50/test.txt',index_col=0)

train_idx = train.index
valid_idx = valid.index
test_idx = test.index

all_text = pd.concat([train['TITLE'], valid['TITLE'], test['TITLE']], axis=0)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(all_text)

train_feature = X.toarray()[train_idx]
valid_feature = X.toarray()[valid_idx]
test_feature = X.toarray()[test_idx]

np.savetxt('train.feature.cntvec.txt', train_feature)
np.savetxt('valid.feature.cntvec.txt', valid_feature)
np.savetxt('test.feature.cntvec.txt', test_feature)

with open("vec_dict.pkl","wb") as f:
    pickle.dump(vectorizer.vocabulary_, f)