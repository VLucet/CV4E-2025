## V. Lucet
## Jan 6 2024
## Bar Graphs

# Libraries
import os
import pandas as pd
import plotnine as gg

# Read in the annotation data
dat = pd.read_csv("data/tabular/bbox_data_crop_simple_all.csv")
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

# Re-read it
dat_group_mod = pd.read_csv("data/tabular/species_groups.csv")
dat_group_mod_ord = dat_group_mod.groupby(by=["label_group"], as_index=False, sort=False) \
    .sum() \
    .sort_values('size', ascending=False)

# Merge with grouped data and sort
dat_merged = dat.merge(dat_group_mod.drop('size', axis=1), how='left', on='label_spe')
dat_merged_summ = .groupby(by=["label_group", "loc_id"], as_index=False, sort=False) \
    .size() \
    .sort_values(['label_group', 'size'], ascending=[True, False])
# Make the column categorical for ordered plotting
dat_merged_summ["label_group"] = pd.Categorical(dat_to_plot["label_group"], 
                                            categories=dat_group_mod_ord["label_group"])

# Plot number of bboxes for each group
myplot = gg.ggplot(dat_merged_summ) + gg.aes(x = "label_group", y = "size") + \
    gg.geom_col() + \
    gg.theme_bw() + \
    gg.theme(axis_text_x = gg.element_text(angle = 45, vjust = 1, hjust = 1)) + \
    gg.xlab("Label") + \
    gg.ylab("# of bboxes") + \
    gg.facet_wrap("~loc_id")
# Save plot
# myplot.save(filename="figures/loc_by_class.pdf", height=30, width=30, limitsize=False)
# Add log scale
myplot_log = myplot + gg.scale_y_log10()
# Save plot again
# myplot_log.save(filename="figures/loc_by_class_log.pdf", height=30, width=30, limitsize=False)

# Compute diversity metrics
from skbio import diversity as sd

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