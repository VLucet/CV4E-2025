import pandas as pd

dat_test = pd.read_csv("data/split/dat_test.csv")
dat_train = pd.read_csv("data/split/dat_train.csv")
dat_val = pd.read_csv("data/split/dat_val.csv")

dat_test.head()
dat_train.head()
dat_val.head()