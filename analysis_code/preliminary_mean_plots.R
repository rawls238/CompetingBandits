library("ggplot2")
library("dplyr")
library("reshape")

WORKING_PATH <- "/Users/guyaridor/Dropbox/Competing Bandits/Isolation Results"
setwd(WORKING_PATH)

dat <- read.csv(file=paste(WORKING_PATH, "Isolation Results/longer_ws.csv", sep=""))

dat$Algorithm <- replace(as.character(dat$Algorithm), dat$Algorithm == "DynamicEpsilonGreedy", "DynamicEpsilonGreedy, 0.05")
dat$Algorithm <- replace(as.character(dat$Algorithm), dat$Algorithm == "NonBayesianEpsilonGreedy", "NonBayesianEpsilonGreedy, 0.05")
algs <- as.list(unique(dat['Algorithm']))$Algorithm
filter_algs <- c("ThompsonSampling", "UCB1WithConstantOne", "DynamicEpsilonGreedy, 0.05", "DynamicGreedy")
dat <- filter(dat, Algorithm %in% filter_algs)
dists <- as.list(unique(dat['Distribution']))$Distribution

filter_by_dist_and_plot <- function(dist) {
  d <- filter(dat, Distribution == dist)
  title <- paste("Reputation Trajectory for", dist, "10 arms")
  q <- ggplot(data=d, aes(x=t, y=Realized.Reputation, colour=Algorithm)) + ggtitle(title) + ylab("Reputation") + xlab("time") +
    #geom_path() # just plot the raw trajectory
    geom_smooth(method="loess") #+ smooths the trajectory 
    #geom_errorbar(aes(ymin=Instantaneous.Reward.Mean-1.96*Instantaneous.Reward.Std, ymax=Instantaneous.Reward.Mean+1.96*Instantaneous.Reward.Std))
  ggsave(paste(FIGURES_DIR, "persistence_publisher_bz.pdf", sep=""), device="pdf", plot=g, width=8.14, height=7.6)
  
  return(d)
}

lapply(dists, filter_by_dist_and_plot)

find_average_std_dev <- function() {
  lapply(dists, function(dist) {
    return(lapply(algs, function(alg) {
      d <- filter(dat, Distribution == dist)
      d <- filter(d, Algorithm == alg)
      return((d['Instantaneous.Reward.Std'][[1]])[5000])
    })) 
  })
}

#print(find_average_std_dev())
