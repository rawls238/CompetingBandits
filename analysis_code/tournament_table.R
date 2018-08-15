
library(dplyr)
library(knitr)
WORKING_PATH <- "/Users/garidor/Desktop/bandits-rl-project"



dat <- read.csv(file=paste(WORKING_PATH, "/results/free_obs_raw_results/free_obs_experiment_full_sim_no_info_raw.csv", sep=""))
p1_algs <- as.list(unique(dat['P1.Alg']))$P1.Alg
p2_algs <- as.list(unique(dat['P2.Alg']))$P2.Alg
agent_algs <- as.list(unique(dat['Agent.Alg']))$Agent.Alg
agent_algs - c("HardMax")
time_horizons <-  as.list(unique(dat['Time.Horizon']))$Time.Horizon
time_horizons <- c(2000)
priors <- as.list(unique(dat['Prior']))$Prior
priors <- c("Needle In Haystack", "Heavy Tail")
warm_start <- 20

ZERO_ONE_CUTOFF <- 0.1
algs <- c("ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy")
alg_names <- c("TS", "DEG", " DG")
options(xtable.sanitize.text.function=identity)
options(xtable.caption.placement="top")
library(xtable)
for (j in 1:length(time_horizons)) {
  for (prior in priors) {
    for (i in 1:length(agent_algs)) {
      agent_alg <- agent_algs[i]
      print_breaks <- FALSE
      time <- time_horizons[j]
      results <- matrix(nrow=length(algs), ncol=length(algs))
      colnames(results) <- alg_names
      rownames(results) <- alg_names
      for (k in 1:length(algs)) {
        for (l in 1:length(algs)) {
          p1alg = algs[k]
          p2alg = algs[l]
          filtered_dat <- filter(dat, P1.Alg == p1alg & P2.Alg == p2alg & Warm.Start == warm_start & Prior == prior & Agent.Alg == agent_alg & Time.Horizon == time)
          if (nrow(filtered_dat) == 0) {
            if (p1alg == p2alg) { print_breaks <- TRUE }
            next
          }
          share <- filtered_dat$Market.Share.for.P1
          test <- t.test(share)
          cin <- signif(test$conf.int[2] - mean(share), digits=2)
          shares <- sum(share >= (1 - ZERO_ONE_CUTOFF) | share <= ZERO_ONE_CUTOFF, na.rm=TRUE) / nrow(filtered_dat)
          cv_high <- qchisq(.975, df=length(share) - 1)
          cv_low <- qchisq(.025, df=length(share) - 1)
          lower_var <- signif((length(share) - 1) * var(share) / cv_high, digits=2)
          high_var <- signif((length(share) - 1) * var(share) / cv_low, digits=2)
          results[k, l] <- paste("\\makecell{\\textbf{", signif(mean(share), digits=2), "} $\\pm$", cin, "\\\\Variance:", signif(var(share), digits=1), "\\\\", "ES:", 100*signif(shares, digits=2), "\\%}")
        }
      }
      tab <-xtable(results, caption=paste("Information Erased Experiment", prior, "X=200"))
      print(tab, type="latex")
    }
  }
}
