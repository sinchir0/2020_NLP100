from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def train_valid_test_split(data, split_point: Tuple, shuffle=True):
    """pd.DataFrame()をtrain,valid,testに分割する
    Args:
        data: pd.DataFrame()のデータ
        split_point: 分割点
        shuffle: dataをランダムにシャッフルするかどうか
    Returns:
        train,valid,test: ３分割されたdata
    """

    if shuffle:
        data = data.sample(frac=1)

    data_len = len(data)

    first = int(data_len*split_point[0])
    second = int(data_len*split_point[1])
    third = int(data_len*split_point[2])

    train = data[:first]
    valid = data[first:(first+second)]
    test = data[(first+second):(first+second+third)]

    return train, valid, test

def plot_confusion_matrix(y_true,y_pred,
                          normalize=False,
                          title=None,
                          labels=None,
                          figsize=(8,12),
                          cmap=plt.cm.Blues,
                          save_fig=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    
    # labelsを設定している場合はそのlabelのみを使う
    if labels is not None:
        classes = labels
        
    if normalize:
        # 行方向への正規化
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]        

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.figure(figsize=figsize)
    if save_fig:
        fig.savefig(f"{title}.png")
    return ax