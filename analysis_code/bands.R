
#mean and its conf range,
#sample variance and its conf range
#fraction of “degenerate outcomes” (0-1 split).

library(dplyr)
WORKING_PATH <- "/Users/garidor/Desktop/bandits-rl-project"


dat <- read.csv(file=paste(WORKING_PATH, "/results/memory_experiment_raw_results_2.csv", sep=""))
p1_algs <- as.list(unique(dat['P1.Alg']))$P1.Alg
p2_algs <- as.list(unique(dat['P2.Alg']))$P2.Alg
agent_algs <- as.list(unique(dat['Agent.Alg']))$Agent.Alg
time_horizons <-  as.list(unique(dat['Time.Horizon']))$Time.Horizon
priors <- as.list(unique(dat['Prior']))$Prior
EXPERIMENT_1_CV_HIGH <- 156.71410383 #chisquared critical values for 124 degrees of freedom
EXPERIMENT_1_CV_LOW <- 95.07008897

EXPERIMENT_2_CV_HIGH <- 128.42198864
EXPERIMENT_2_CV_LOW <- 73.36108019



ZERO_ONE_CUTOFF <- 0.1
for (prior in priors) {
  for (p1alg in p1_algs) {
    for (p2alg in p2_algs) {
      if (p1alg == p2alg) { next }
      results <- matrix(nrow=length(agent_algs), ncol=length(time_horizons))
      colnames(results) <- c("Time 1000", "Time 3000", "Time 5000")
      rownames(results) <- agent_algs
      for (i in 1:length(agent_algs)) {
        agent_alg <- agent_algs[i]
        for (j in 1:length(time_horizons)) {
          time <- time_horizons[j]
          filtered_dat <- filter(dat, P1.Alg == p1alg & P2.Alg == p2alg & Memory.Size == 100 & Prior == prior & Agent.Alg == agent_alg & Time.Horizon == time)
          share <- filtered_dat$Market.Share.for.P1
          test <- t.test(share)
          cin <- signif(test$conf.int[2] - mean(share), digits=4)
          shares <- sum(share >= (1 - ZERO_ONE_CUTOFF) | share <= ZERO_ONE_CUTOFF, na.rm=TRUE) / nrow(filtered_dat)
          lower_var <- signif((length(share) - 1) * var(share) / EXPERIMENT_1_CV_HIGH, digits=6)
          high_var <- signif((length(share) - 1) * var(share) / EXPERIMENT_1_CV_LOW, digits=6)
          results[i, j] <- paste("Mean Market Share:", signif(mean(share), digits=4), "+/-", cin, "Non-competitive share (as %)", shares, "Var(Market Share):", signif(var(share), digits=6), "95% CI", lower_var, high_var)
        }
      }
      cat("Results for", prior, " - ", p1alg, "vs", p2alg, '\n')
      print(results)
      cat('\n', '\n')
    }
  }
}