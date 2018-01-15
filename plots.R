library("ggplot2")
library("dplyr")

dat <- read.csv("/Users/garidor/Desktop/bandits-rl-project/results/preliminary_plots.csv")
algs <- as.list(unique(dat['Algorithm']))$Algorithm
dists <- as.list(unique(dat['Distribution']))$Distribution

filter_by_alg_and_plot <- function(alg) {
  d <- filter(dat, Algorithm == alg)
  filtered_data <- lapply(dists, function(dist) {
    filtered <- filter(d, Distribution == dist)
    print(filtered)
    q <- ggplot(data=filtered, aes(t, Reward.Mean, group=1)) + geom_path()
    print(q)
    
    return(filtered)
  })
}


lapply(algs, filter_by_alg_and_plot)