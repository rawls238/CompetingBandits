library(dplyr)
library(xtable)

WORKING_PATH <- "/Users/garidor/Desktop/bandits-rl-project/results/tournament_raw_results/"
f <- "tournament_experiment_effective_end_of_game_warm_start_raw.csv"
dat <- read.csv(file=paste(WORKING_PATH, f, sep=""))
dat <- filter(dat, Time.Horizon == 2000)
priors <- as.list(unique(dat['Prior']))$Prior
warm_starts <- as.list(unique(dat['Warm.Start']))$Warm.Start
algs <- c("ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy")

for (prior in priors) {
  results <- matrix(nrow=3, ncol=length(warm_starts))
  for (i in 1:length(warm_starts)) {
    warm_start = warm_starts[i]
    for (k in 1:length(algs)) {
      for (l in 1:length(algs)) {
        p1alg = algs[k]
        p2alg = algs[l]
        filtered_dat <- filter(dat, Prior == prior & Warm.Start == warm_start & P1.Alg == p1alg & P2.Alg == p2alg)
        if (nrow(filtered_dat) == 0) { next }
        share <- filtered_dat$Market.Share.for.P1
        test <- t.test(share)
        cin <- signif(test$conf.int[2] - mean(share), digits=1)
        if (p1alg == "ThompsonSampling" && p2alg == "DynamicGreedy") {
          results[1, i] <- mean(filtered_dat$EEOG)
        } else if (p1alg == "ThompsonSampling" && p2alg == "DynamicEpsilonGreedy") {
          results[2, i] <- mean(filtered_dat$EEOG)
        } else {
          results[3, i] <- mean(filtered_dat$EEOG)
        }
      }
    }
  }
  print(results)
}

#WINDOW <- 5000
#for (i in 220:240) {
#  begin <- (i*WINDOW) + 1
#  end_window <- (i + 1) * WINDOW
#  one_sim <- filtered_dat[begin:end_window,]
#  plot(one_sim[,"Time.Horizon"], one_sim[, "Market.Share.for.P1"], main="HMR Market Share over Time", ylab="Market Share", xlab="Time")
#}