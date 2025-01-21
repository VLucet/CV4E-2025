# Libraries
from skbio import diversity as sd
import pandas as pd
import random
import sys
sys.path.append('.')
from myutils.MDSplit import *

random.seed(77)

print("Splitting data...")

dat_merged = pd.read_csv("data/tabular/all_dat_merged.csv")
dat_merged_summ = pd.read_csv("data/tabular/all_dat_merged_summ.csv")

dat_pivot = dat_merged_summ.pivot_table(index='loc_id', columns='label_group', 
                                   values='size', aggfunc='sum') \
    .reset_index() \
    .fillna(0)
dat_locs = dat_pivot["loc_id"]
counts_table = dat_pivot.drop(["loc_id"], axis=1)
counts_table = counts_table.rename_axis(None, axis=1)

# Compute shannon div
shannon_div = sd.alpha_diversity("shannon", counts_table)
best_loc = dat_locs[shannon_div.argmax()]

# Filter out testloc
dat_test = dat_merged.query(f'loc_id == "{best_loc}"')
dat_test.groupby("label_group").size()
dat_test.to_csv("data/tabular/splits/all/dat_test.csv", index=False)

##################################################################

# Data without test set
dat_tt = dat_merged.query(f'loc_id != "{best_loc}"')
dat_tt.to_csv("data/tabular/splits/all/dat_train_val.csv", index=False)

# Run the split
dat_tt_tab = dat_tt.groupby(by=["label_group", "loc_id"], 
                                     as_index=False, sort=False) \
    .size() \
    .sort_values(['label_group', 'size'], ascending=[True, False]) \
    .pivot_table(index='label_group', columns='loc_id', 
                                   values='size', aggfunc='sum') \
    .reset_index() \
    .set_index('label_group') \
    .rename_axis(None, axis=1) \
    .fillna(0)

dat_tt_tab_dict = dat_tt_tab.to_dict()

the_split = split_locations_into_train_val(dat_tt_tab_dict, 
                               n_random_seeds=10000,
                               target_val_fraction=0.12,
                               category_to_max_allowable_error=None,                                   
                               category_to_error_weight=None,
                               default_max_allowable_error=0.1)

dat_train = dat_merged.query(f'loc_id not in {the_split[0]}')
dat_val = dat_merged.query(f'loc_id in {the_split[0]}')

# print(dat_train.head())
# print(dat_val.head())

dat_train.to_csv("data/tabular/splits/all/dat_train.csv", index=False)
dat_val.to_csv("data/tabular/splits/all/dat_val.csv", index=False)

print("Data is split.")
