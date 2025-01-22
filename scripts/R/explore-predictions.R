library(readr)
library(ggplot2)
library(dplyr)
library(tidyr)

cats = c("NONE",
         "STAF",
         "CARI",
         "GOOS",
         "DUCK",
         "MOOS",
         "SAND",
         "BIRD",
         "GROU",
         "MAMM",
         "BEAR",
         "MUST")

color_table

preds = read_csv("outputs/predictions.csv") %>%
  select(-`...1`) %>%
  rename(truth = label_group) %>%
  pivot_longer(cols = NONE:MUST, values_to = "score", names_to = "pred_category") %>%
  mutate(pred_category = factor(pred_category, levels = cats))

head(preds)

x <- ggplot(preds) +
  geom_density(aes(x = score, fill=pred_category),
               alpha = 0.2) +
  facet_wrap(~truth) +
  theme_minimal() +
  scale_fill_manual(values = c(c("#E69F00", rep("#56B4E9", 11))))
ggsave("plot.jpg", x)
