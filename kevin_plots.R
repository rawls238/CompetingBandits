library(data.table)
library(ggplot2)

dt.data <- data.table(read.csv("~/dev/bandits-rl-project/results/preliminary_plots.csv", header=T, stringsAsFactors=T))


# dt.data[(Algorithm == 'ThompsonSampling' & K==10)]

distribution_names = unique(dt.data[, Distribution])

ggplot(dt.data[(Distribution=="Needle50 - High")], aes(x=t, y=Reward.Mean, color=Algorithm)) +
  # geom_point(alpha=0.1) +
  geom_smooth(method="loess", se=TRUE) +
  scale_x_continuous("Time") +
  scale_y_continuous('Average reward') +
  theme_bw() +
  ggtitle("Average reward over time")