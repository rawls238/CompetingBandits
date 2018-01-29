library("ggplot2")
library("dplyr")
library("reshape")

# Three datasets that currently exist are:
# preliminary_plots.csv was run with N=250, T=6000
# preliminary_plots_2.csv includes the Bayesian Dynamic Epsilon Greedy runs

WORKING_PATH <- "/Users/garidor/Desktop/bandits-rl-project"

dat <- read.csv(file=paste(WORKING_PATH, "/results/preliminary_plots_3_arms.csv", sep=""))
dat2 <- read.csv(file=paste(WORKING_PATH, "/results/preliminary_plots_3_arms_2.csv", sep=""))
dat <- rbind(dat, dat2)
algs <- as.list(unique(dat['Algorithm']))$Algorithm
filter_algs <- algs
dat <- filter(dat, Algorithm %in% filter_algs)
dists <- as.list(unique(dat['Distribution']))$Distribution

filter_by_dist_and_plot <- function(dist) {
  d <- filter(dat, Distribution == dist)
  title <- paste("Reward Trajectory for", dist, "inst")
  q <- ggplot(data=d, aes(x=t, y=Instantaneous.Reward.Mean, colour=Algorithm)) + ggtitle(title) + ylab("Instantaneous Mean Reward") + xlab("time") +
    #geom_path() # just plot the raw trajectory
    geom_smooth(method="loess") #+ smooths the trajectory 
    #geom_errorbar(aes(ymin=Instantaneous.Reward.Mean-1.96*Instantaneous.Reward.Std, ymax=Instantaneous.Reward.Mean+1.96*Instantaneous.Reward.Std))
  ggsave(width=8, height=8, dpi=300, filename=paste(WORKING_PATH, "/results/", title, ".pdf", sep=""), plot=q)
  
  return(d)
}

lapply(dists, filter_by_dist_and_plot)

find_average_std_dev <- function() {
  lapply(dists, function(dist) {
    return(lapply(algs, function(alg) {
      d <- filter(dat, Distribution == dist)
      d <- filter(d, Algorithm == alg)
      return(mean(d['Instantaneous.Reward.Std'][[1]]))
    })) 
  })
}

#print(find_average_std_dev())
