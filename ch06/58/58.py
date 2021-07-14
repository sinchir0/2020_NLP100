import sys

sys.path.append("../../src")

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util import make_predict_by_LR

train_y = pd.read_table("../50/train.txt", usecols=["CATEGORY"])
valid_y = pd.read_table("../50/valid.txt", usecols=["CATEGORY"])
test_y = pd.read_table("../50/test.txt", usecols=["CATEGORY"])

train_feature = np.loadtxt("../51/train.feature.cntvec.txt")
valid_feature = np.loadtxt("../51/valid.feature.cntvec.txt")
test_feature = np.loadtxt("../51/test.feature.cntvec.txt")

result_list = []

for c in [0.1, 0.5, 1, 10, 100]:
    result = make_predict_by_LR(
        train_y, valid_y, test_y, train_feature, valid_feature, test_feature, C=c
    )

    result_list.append(result)

result_df = pd.DataFrame(result_list)
result_df = result_df.set_index(0)
result_df.columns = ["train", "valid", "test"]
result_df.plot()

plt.title("CountVectorizer LR")

plt.title("CountVectorizer LR")
plt.xlabel("c")
plt.ylabel("Accuracy")
plt.savefig("result.png")
