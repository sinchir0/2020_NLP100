import sys

sys.path.append("../../src")

from util import train_valid_test_split


newsCorpora = pd.read_table(
    "../../data/NewsAggregatorDataset/newsCorpora.csv", header=None
)
newsCorpora.columns = [
    "ID",
    "TITLE",
    "URL",
    "PUBLISHER",
    "CATEGORY",
    "STORY",
    "HOSTNAME",
    "TIMESTAMP",
]

match_row_index = newsCorpora["PUBLISHER"].isin(
    ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
)
newsCorpora_extract_by_PUBLISHER = newsCorpora[match_row_index]

newsCorpora_extract_by_PUBLISHER = newsCorpora_extract_by_PUBLISHER.reset_index(
    drop=True
)

train, valid, test = train_valid_test_split(
    newsCorpora_extract_by_PUBLISHER, split_point=(0.8, 0.1, 0.1)
)

train.to_csv("train.txt", sep="\t")
valid.to_csv("valid.txt", sep="\t")
test.to_csv("test.txt", sep="\t")

print("train_number_per_category")
print(train["CATEGORY"].value_counts())
print("valid_number_per_category")
print(valid["CATEGORY"].value_counts())
print("test_number_per_category")
print(test["CATEGORY"].value_counts())
# train_number_per_category
# b    4473
# e    4271
# t    1201
# m     727
# Name: CATEGORY, dtype: int64
# valid_number_per_category
# e    569
# b    553
# t    127
# m     85
# Name: CATEGORY, dtype: int64
# test_number_per_category
# b    601
# e    439
# t    196
# m     98
# Name: CATEGORY, dtype: int6
