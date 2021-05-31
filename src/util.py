from typing import Tuple

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
