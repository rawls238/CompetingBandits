library("ggplot2")
library("dplyr")
library("reshape")

dat <- read.csv("/Users/garidor/Desktop/bandits-rl-project/results/preliminary_plots_2.csv")
algs <- as.list(unique(dat['Algorithm']))$Algorithm
dists <- as.list(unique(dat['Distribution']))$Distribution

filter_by_dist_and_plot <- function(dist) {
  d <- filter(dat, Distribution == dist)
  title <- paste("Reward Trajectory for", dist)
  q <- ggplot(data=d) + ggtitle(title) + ylab("Mean Reward") + xlab("time") +
    geom_path(aes(x=t, y=Reward.Mean, colour=Algorithm))
  ggsave(width=8, height=8, dpi=300, filename=paste("/Users/garidor/Desktop/bandits-rl-project/results/", title, ".pdf", sep=""), plot=q)
  
  return(d)
}

lapply(dists, filter_by_dist_and_plot)



