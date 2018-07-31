library(dplyr)
library(knitr)
WORKING_PATH <- "/Users/garidor/Desktop/bandits-rl-project"

#dat <- read.csv(file=paste(WORKING_PATH, "/results/tournament_raw_results/tournament_experiment_reputation_window_raw.csv", sep=""))
dat <- read.csv(file=paste(WORKING_PATH, "/results/free_obs_raw_results/free_obs_experiment_tournament_raw_results.csv", sep=""))
erase_rep_dat <- read.csv(file=paste(WORKING_PATH, "/results/free_obs_raw_results/free_obs_experiment_erase_reputation_raw.csv", sep=""))
erase_info_dat <- read.csv(file=paste(WORKING_PATH, "/results/free_obs_raw_results/free_obs_experiment_erase_information_raw.csv", sep=""))
p1_algs <- as.list(unique(dat['P1.Alg']))$P1.Alg
p2_algs <- as.list(unique(dat['P2.Alg']))$P2.Alg
memory_sizes <- sort(as.list(unique(dat['Memory.Size']))$Memory.Size, decreasing=TRUE)
agent_algs <- as.list(unique(dat['Agent.Alg']))$Agent.Alg
agent_algs <- c("HardMax")
time_horizons <-  as.list(unique(dat['Time.Horizon']))$Time.Horizon
time_horizons <- c(1000)
priors <- as.list(unique(dat['Prior']))$Prior
priors <- c("Heavy Tail")

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
      for (q in 1:3) {
        time <- time_horizons[j]
        results <- matrix(nrow=length(algs), ncol=length(algs))
        colnames(results) <- alg_names
        rownames(results) <- alg_names
        for (k in 1:length(algs)) {
          for (l in 1:length(algs)) {
            p1alg = algs[k]
            p2alg = algs[l]
            filtered_dat <- NULL
            title <- NULL
            if (q == 1) {
              title <- "Info + Rep"
              filtered_dat <- filter(dat, P1.Alg == p1alg & P2.Alg == p2alg & Memory.Size == 100 & Prior == prior & Agent.Alg == agent_alg & Time.Horizon == time)
            } else if (q == 2) {
              title <- "Info"
              filtered_dat <- filter(erase_rep_dat, P1.Alg == p1alg & P2.Alg == p2alg & Memory.Size == 100 & Prior == prior & Agent.Alg == agent_alg & Time.Horizon == time)
            } else if (q == 3) {
              title <- "Rep"
              filtered_dat <- filter(erase_info_dat, P1.Alg == p1alg & P2.Alg == p2alg & Memory.Size == 100 & Prior == prior & Agent.Alg == agent_alg & Time.Horizon == time)
            }
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
            results[k, l] <- paste("\\makecell{ \\textbf{", signif(mean(share), digits=2), "} $\\pm$", cin, "\\\\Var: ", signif(var(share), digits=1), "\\\\", "ES:", 100*signif(shares, digits=2), "\\% }")
          }
        }
        tab <-xtable(results, caption=paste("Results for", agent_alg, "t =", time, prior, title))
        print(tab, type="latex")
        if (print_breaks) { print("<br>") }
      }
    }
  }
}