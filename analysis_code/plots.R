library("ggplot2")
library("dplyr")
library("reshape")

# Three datasets that currently exist are:
# preliminary_plots.csv includes more distributions but the resulting reward means are only averaged over fifty rounds.
# preliminary_plots_2.csv was run with N=250, T=1000
# preliminary_plots_3.csv was run with N=250, T=5000

WORKING_PATH <- "/Users/garidor/Desktop/bandits-rl-project"

dat <- read.csv(file=paste(WORKING_PATH, "/results/preliminary_plots_3.csv", sep=""))
algs <- as.list(unique(dat['Algorithm']))$Algorithm
filter_algs <- algs
dat <- filter(dat, Algorithm %in% filter_algs)
dists <- as.list(unique(dat['Distribution']))$Distribution

filter_by_dist_and_plot <- function(dist) {
  d <- filter(dat, Distribution == dist)
  title <- paste("Reward Trajectory for", dist, "cur")
  q <- ggplot(data=d, aes(x=t, y=Reward.Mean, colour=Algorithm)) + ggtitle(title) + ylab("Mean Reward") + xlab("time") +
    #geom_path() # just plot the raw trajectory
    geom_smooth(method="loess", se=TRUE) # smooths the trajectory
  ggsave(width=8, height=8, dpi=300, filename=paste(WORKING_PATH, "/results/", title, ".pdf", sep=""), plot=q)
  
  return(d)
}

lapply(dists, filter_by_dist_and_plot)



