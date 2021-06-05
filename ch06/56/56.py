import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

# ground truth
test_y = pd.read_table('../50/test.txt', usecols=['CATEGORY'])

# predict value
test_pred = np.load('../53/test_pred.npy', allow_pickle=True)

print(classification_report(test_y, test_pred))
#               precision    recall  f1-score   support

#            b       0.43      0.48      0.45       573
#            e       0.38      0.45      0.41       529
#            m       0.00      0.00      0.00        83
#            t       0.09      0.03      0.05       149

#     accuracy                           0.39      1334
#    macro avg       0.22      0.24      0.23      1334
# weighted avg       0.34      0.39      0.36      1334