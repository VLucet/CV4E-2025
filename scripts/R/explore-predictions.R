library(readr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(sigmoid)

lookup = read_csv("data/tabular/labels_lookup.csv")

preds = read_csv("outputs/predictions.csv") %>%
  rename(truth = label_group, sample = `...1`) %>%
  pivot_longer(cols = NONE:MUST, values_to = "score", names_to = "pred_category") %>%
  group_by(sample) %>%
  mutate(max_score = max(score)) %>%
  mutate(pred = pred_category[score == max_score]) %>%
  ungroup() %>%
  mutate(correct = truth == pred) %>%
  mutate(pred_category = factor(pred_category, levels = lookup$label_group)) %>%
  # pivot_wider(names_from = pred_category, values_from = score) %>%
  select(-max_score)

head(preds)

alpha = 0.5

for (cls in lookup$label_group) {
  x = ggplot() +
    geom_histogram(
      data = preds %>%
        filter(truth == cls,
               pred_category == cls),
      aes(x = score, fill = "POS"), alpha = alpha)+
    geom_histogram(
      data = preds %>%
        filter(truth != cls,
               pred_category == cls),
      aes(x = score, fill = "NEG"), alpha = alpha)+
    theme_minimal() +
    ggtitle(cls)
  plot(x)
}

ggplot() +
  geom_histogram(
    data = preds %>% filter(correct),
    aes(x = logistic(SAND), fill = correct), alpha = alpha)+
  geom_histogram(
    data = preds %>% filter(!correct),
    aes(x = logistic(SAND), fill = correct), alpha = alpha)+
  theme_minimal()
