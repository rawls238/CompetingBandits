
library(dplyr)
library(knitr)
WORKING_PATH <- "/Users/garidor/Desktop/bandits-rl-project"


dat <- read.csv(file=paste(WORKING_PATH, "/results/free_obs_raw_results/free_obs_experiment_raw_results_2.csv", sep=""))
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

alg_pairs <- c("ThompsonSampling", "DynamicEpsilonGreedy", "ThompsonSampling", "DynamicGreedy", "DynamicGreedy", "DynamicEpsilonGreedy")
library(xtable)
for (prior in priors) {
  for (j in 1:length(time_horizons)) {
    results <- matrix(nrow=3, ncol=3)
    #colnames(results) <- c(paste(alg_pairs[1], alg_pairs[2], sep=", "), paste(alg_pairs[3], alg_pairs[4], sep=", "), paste(alg_pairs[5], alg_pairs[6], sep=", "))    
    #rownames(results) <- agent_algs
    print(agent_algs)
    for (i in 1:length(agent_algs)) {
      agent_alg <- agent_algs[i]
      time <- time_horizons[j]
      for (k in 1:length(alg_pairs)) {
        if (k %% 2 == 0) { next }
        p1alg = alg_pairs[k]
        p2alg = alg_pairs[k+1]
        filtered_dat <- filter(dat, P1.Alg == p1alg & P2.Alg == p2alg & Memory.Size == 100 & Prior == prior & Agent.Alg == agent_alg & Time.Horizon == time)
        if (nrow(filtered_dat) == 0) {
          next
        }
        print((k+1)/2)
        share <- filtered_dat$Market.Share.for.P1
        test <- t.test(share)
        cin <- signif(test$conf.int[2] - mean(share), digits=4)
        shares <- sum(share >= (1 - ZERO_ONE_CUTOFF) | share <= ZERO_ONE_CUTOFF, na.rm=TRUE) / nrow(filtered_dat)
        lower_var <- signif((length(share) - 1) * var(share) / EXPERIMENT_1_CV_HIGH, digits=4)
        high_var <- signif((length(share) - 1) * var(share) / EXPERIMENT_1_CV_LOW, digits=4)
        results[i, ((k+1)/2)] <- paste("<b>",signif(mean(share), digits=4), "</b> +/-", cin, "<br>", signif(var(share), digits=4), paste("(", lower_var, ", ", high_var, ")", sep=""), "<br>", "Share:", shares)
        #print(results)
      }
    }
    tab <-xtable(results, caption=paste("Results for", time, prior))
    #print(tab, type="html")
  }
}
