library(dplyr)
library(knitr)
WORKING_PATH <- "/Users/garidor/Desktop/bandits-rl-project"

#dat <- read.csv(file=paste(WORKING_PATH, "/results/free_obs_raw_results/free_obs_experiment_free_obs_10_arms_raw.csv", sep=""))
#dat_3 <- read.csv(file=paste(WORKING_PATH, "/results/tournament_raw_results/tournament_experiment_large_horizon_3_raw.csv", sep=""))
dat <- read.csv(file=paste(WORKING_PATH, "/results/tournament_raw_results/tournament_experiment_full_ws_many_sim_raw.csv", sep=""))
#dat_2 <- read.csv(file=paste(WORKING_PATH, "/results/tournament_raw_results/tournament_experiment_many_arms_2_raw.csv", sep=""))
#dat_10 <- rbind(dat, dat_2)
#dat <- read.csv(file=paste(WORKING_PATH, "/results/tournament_raw_results/tournament_experiment_ws_small_erase_rep_raw.csv", sep=""))
#dat_3 <- dat


p1_algs <- as.list(unique(dat_3['P1.Alg']))$P1.Alg
p2_algs <- as.list(unique(dat_3['P2.Alg']))$P2.Alg
agent_algs <- as.list(unique(dat_3['Agent.Alg']))$Agent.Alg
agent_algs <- c("HardMax")
time_horizons <-  as.list(unique(dat_3['Time.Horizon']))$Time.Horizon
time_horizons <- c(5000)
priors <- as.list(unique(dat_3['Prior']))$Prior
priors <- c("Heavy Tail", "Needle In Haystack - 0.5")
warm_starts <- c(5)

ZERO_ONE_CUTOFF <- 0.1


concise_alg_rep <- function(alg) {
  if (alg == "ThompsonSampling") {
    return("TS")
  } else if (alg == "DynamicEpsilonGreedy") {
    return("DEG")
  } else if (alg == "DynamicGreedy") {
    return("DG")
  }
}

concise_agent_rep <- function(agent_alg) {
  if (agent_alg == "HardMax") {
    return("HM")
  } else if (agent_alg == "HardMaxWithRandom") {
    return("HMR")
  } else if (agent_alg == "SoftMax") {
    return("SM")
  }
}

get_agent_rep <- function(repre) {
  res <- c()
  for(a in repre) {
    res <- c(res, concise_agent_rep(a))
  }
  return(res)
}

algs <- c("ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy")
options(xtable.sanitize.text.function=identity)
options(xtable.caption.placement="top")

alg_pairs <- c("ThompsonSampling", "DynamicEpsilonGreedy", "ThompsonSampling", "DynamicGreedy", "DynamicGreedy", "DynamicEpsilonGreedy", "ThompsonSampling", "ThompsonSampling", "DynamicEpsilonGreedy", "DynamicEpsilonGreedy", "DynamicGreedy", "DynamicGreedy")
#print("Reported means are for principal 1 which is the first algorithm listed in the pair for each column<br>")
#print("In each cell, first line is mean, second line is variance, third line is % of the time that market share > 0.1 or market share < 0.9")
library(xtable)
for (prior in priors) {
  for (j in 1:length(time_horizons)) {
    results <- matrix(nrow=length(agent_algs), ncol=(length(alg_pairs) / 2))
    colnames(results) <- c(paste(concise_alg_rep(alg_pairs[1]), concise_alg_rep(alg_pairs[2]), sep=" vs "), paste(concise_alg_rep(alg_pairs[3]), concise_alg_rep(alg_pairs[4]), sep=" vs "), paste(concise_alg_rep(alg_pairs[5]), concise_alg_rep(alg_pairs[6]), sep=" vs "), paste(concise_alg_rep(alg_pairs[7]), concise_alg_rep(alg_pairs[8]), sep=" vs "), paste(concise_alg_rep(alg_pairs[9]), concise_alg_rep(alg_pairs[10]), sep=" vs "), paste(concise_alg_rep(alg_pairs[11]), concise_alg_rep(alg_pairs[12]), sep=" vs "))
    rownames(results) <- get_agent_rep(agent_algs)
    for (q in 1:1) {
      for (start in warm_starts) {
        for (i in 1:length(agent_algs)) {
          
          agent_alg <- agent_algs[i]
          time <- time_horizons[j]
          for (k in 1:length(alg_pairs)) {
            if (k %% 2 == 0) { next }
            p1alg = alg_pairs[k]
            p2alg = alg_pairs[k+1]
            K <- 10
            if (q == 1) {
              filtered_dat <- filter(dat, P1.Alg == p1alg & P2.Alg == p2alg & Prior == prior & Agent.Alg == agent_alg & Time.Horizon == time)
            } else {
              filtered_dat <- filter(dat_10, P1.Alg == p1alg & P2.Alg == p2alg & Prior == prior & Agent.Alg == agent_alg & Time.Horizon == time)
            }
            if (nrow(filtered_dat) == 0) {
              next
            }
            share <- filtered_dat$Market.Share.for.P1
            test <- t.test(share)
            cin <- signif(test$conf.int[2] - mean(share), digits=1)
            shares <- sum(share >= (1 - ZERO_ONE_CUTOFF) | share <= ZERO_ONE_CUTOFF, na.rm=TRUE) / nrow(filtered_dat)
            shares_percent <- signif(shares * 100, digits=2)
            cv_high <- qchisq(.975, df=length(share) - 1)
            cv_low <- qchisq(.025, df=length(share) - 1)
            lower_var <- signif((length(share) - 1) * var(share) / cv_high, digits=1)
            high_var <- signif((length(share) - 1) * var(share) / cv_low, digits=1)
            results[i, (k+1)/2] <- paste("\\textbf {",signif(mean(share), digits=2), "} $pm$", cin, "\\\\Var:", signif(var(share), digits=2), "\\\\Share:", shares_percent, "%")
          }
        }
        tab <-xtable(results, caption=paste("Results for", "t=",time, prior, "Warm Start=", start, "K=", K), row.header='Entrant', col.header='Incumbent')
        print(tab, type="latex")
      }
    }
  }
}


