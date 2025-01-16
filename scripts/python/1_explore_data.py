# Libraries
import pandas as pd
import plotnine as gg

print("Exploring data...")

dat_merged = pd.read_csv("data/tabular/all_dat_merged.csv")
dat_group_mod_ord = pd.read_csv("data/tabular/species_groups_ord.csv")

dat_merged_summ = dat_merged.groupby(by=["label_group", "loc_id"], 
                                     as_index=False, sort=False) \
    .size() \
    .sort_values(['label_group', 'size'], ascending=[True, False])
# Make the column categorical for ordered plotting
dat_merged_summ["label_group"] = pd.Categorical(dat_merged_summ["label_group"], 
                                            categories=dat_group_mod_ord["label_group"])

# Write out
dat_merged_summ.to_csv("data/tabular/all_dat_merged_summ.csv",
                       index=False)

# Plot number of bboxes for each group
myplot = gg.ggplot(dat_merged_summ) + gg.aes(x = "label_group", y = "size") + \
    gg.geom_col() + \
    gg.theme_bw() + \
    gg.theme(axis_text_x = gg.element_text(angle = 45, vjust = 1, hjust = 1)) + \
    gg.xlab("Label") + \
    gg.ylab("# of bboxes") + \
    gg.facet_wrap("~loc_id")
# Save plot
myplot.save(filename="figures/loc_by_class.pdf", height=30, width=30, limitsize=False)
# Add log scale
myplot_log = myplot + gg.scale_y_log10()
# Save plot again
myplot_log.save(filename="figures/loc_by_class_log.pdf", height=30, width=30, limitsize=False)

print("Data explored.")