import pandas
import numpy


df = pandas.read_csv("data/train.csv")
num_records = len(df)

ids = numpy.arange(num_records)
ids = numpy.random.permutation(ids)

train_size = 0.8
partition = int(num_records * train_size)

train_ids, valid_ids = ids[:partition], ids[partition:]

df.loc[train_ids].to_csv("data/processed_train.csv", index=False)
df.loc[valid_ids].to_csv("data/processed_valid.csv", index=False)
