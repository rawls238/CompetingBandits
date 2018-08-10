library(dplyr)
library(ggplot2)

best_arm <- read.csv("/Users/garidor/Desktop/bandits-rl-project/results/preliminary_raw_results/preliminary_plots_best_arm.csv")

algorithms <- unique(best_arm$Algorithm)
dists <- unique(best_arm$Distribution)
times <- unique(best_arm$t)
dat <- data.frame(matrix(vector(), 0, 4,
                                   dimnames=list(c(), c("Algorithm", "Distribution", "t", "mean_correct"))),
                            stringsAsFactors=F)
count <- 1
for (dist in dists) {
  filtered_dist <- filter(best_arm, Distribution == UQ(dist))
  for (algorithm in algorithms) {
    for (time in times) {
      cur <- filter(filtered_dist, Algorithm == UQ(algorithm) & t == UQ(time))
      avg_correct <- mean(as.numeric(cur$Best.Arm.Identification) - 1)
      item <- c(algorithm, dist, time, avg_correct)
      dat[count, ] <- item
      count <- count + 1
    }
  }
}
dat$t <- as.numeric(dat$t)
dat$mean_correct <- as.numeric(dat$mean_correct)

for (dist in dists ) {
  d <- filter(dat, Distribution == UQ(dist))
  p <- ggplot(d, aes(x=t, y=mean_correct)) + geom_line(aes(col=Algorithm)) + labs(title=dist) + ylab("% that Identified Best Arm")
  print(p)
}
