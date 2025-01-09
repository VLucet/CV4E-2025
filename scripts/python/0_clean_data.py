# Libraries
import os
import pandas as pd

print('Cleaning data...')

# Read in the annotation data
dat = pd.read_csv("data-raw/tabular/bbox_data_crop_simple_all.csv")
# Extract metadata
dat = dat.assign(meta_split = [x.split("_") for x in 
                               [os.path.basename(x) for x in dat.crop_path]])
dat = dat.assign(dep_id = [x[0] for x in dat.meta_split], 
                 loc_id = [x[1] for x in dat.meta_split],
                 sit_id = [x[2] for x in dat.meta_split],
                 cam_id = [x[3] for x in dat.meta_split])

# Make table of labels
dat_group = dat.groupby(by=["label_spe"], as_index=False, sort=False) \
    .size() \
    .sort_values('size', ascending=False)
# Write out for manual grouping
# dat_group.to_csv("data/tabular/species_groups.csv", index=False)
dat_group_mod = pd.read_csv("data/tabular/species_groups.csv")

# Ordering
dat_group_mod_ord = dat_group_mod.groupby(by=["label_group"], as_index=False, 
                                          sort=False) \
    .sum() \
    .sort_values('size', ascending=False)

# Write out
dat_group_mod_ord.to_csv("data/tabular/species_groups_ord.csv",
                         index=False)

# Merge with grouped data and sort
dat_merged = dat.merge(dat_group_mod.drop('size', axis=1), how='left', 
                       on='label_spe') \
    .query('label_group != "UNID"')

# Write out
dat_merged.to_csv("data/tabular/all_dat_merged.csv",
                         index=False)

print("Data cleaned.")