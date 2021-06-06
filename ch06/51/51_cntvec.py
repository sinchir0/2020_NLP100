# n-gram, w2v, tf-idf, fast-text, Universal Encoder? yukiさんがatmaで使っていたやつ
# を試してみたい。
import sys
sys.path.append('../../src')

import  pickle

import pandas as pd
import numpy as np

from util import load_data
from util import preprocess
from util import TextFeatureFitTransform

METHOD = 'cntvec'
name_list = ['train','valid','test']

# データ読み込み
dfs = load_data()

# 前処理
for name in name_list:
    dfs[name]['TITLE'] = dfs[name][['TITLE']].apply(preprocess)

# 特徴量学習
feat = TextFeatureFitTransform(method=METHOD)
feat.fit(dfs['train']['TITLE'])

# 特徴量生成
result_dfs = {}
for name in name_list:
    result_dfs[name] = feat.transform(dfs[name]['TITLE'])

# 特徴量保存
for name in name_list:
    np.savetxt(f'{name}.feature.{METHOD}.txt', result_dfs[name].toarray())

# 推論時にも使用するため、保存
with open(f"{METHOD}_vec_dict.pkl","wb") as f:
    pickle.dump(feat.vec.vocabulary_, f)