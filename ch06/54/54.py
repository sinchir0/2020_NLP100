import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# ground truth
train_y = pd.read_table("../50/train.txt", usecols=["CATEGORY"])
valid_y = pd.read_table("../50/valid.txt", usecols=["CATEGORY"])
test_y = pd.read_table("../50/test.txt", usecols=["CATEGORY"])

# predict value
train_pred = np.load("../53/train_pred.npy", allow_pickle=True)
valid_pred = np.load("../53/valid_pred.npy", allow_pickle=True)
test_pred = np.load("../53/test_pred.npy", allow_pickle=True)

# calc score
train_score = accuracy_score(train_y, train_pred)
valid_score = accuracy_score(valid_y, valid_pred)
test_score = accuracy_score(test_y, test_pred)

print(f"train_score : {train_score: .4f}")
print(f"valid_score : {valid_score: .4f}")
print(f"test_score : {test_score: .4f}")
