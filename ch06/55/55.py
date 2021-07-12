import sys

sys.path.append("../../src")

import numpy as np
import pandas as pd
from util import plot_confusion_matrix

# ground truth
train_y = pd.read_table("../50/train.txt", usecols=["CATEGORY"])
test_y = pd.read_table("../50/test.txt", usecols=["CATEGORY"])

# predict value
train_pred = np.load("../53/train_pred.npy", allow_pickle=True)
test_pred = np.load("../53/test_pred.npy", allow_pickle=True)

plot_confusion_matrix(train_y, train_pred, title="train_cm", save_fig=True)
plot_confusion_matrix(test_y, test_pred, title="test_cm", save_fig=True)
