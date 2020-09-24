library(dplyr)
library(xtable)

options(xtable.sanitize.text.function=identity)
options(xtable.caption.placement="top")

concise_alg_rep <- function(alg) {
  if (alg == "ThompsonSampling") {
    return("TS")
  } else if (alg == "DynamicEpsilonGreedy") {
    return("DEG")
  } else if (alg == "DynamicGreedy") {
    return("DG")
  }
}

WORKING_PATH <- "/Users/guyaridor/Dropbox/bandits_results/tournament_raw_results/"
f <- "tournament_experiment_full_welfare_calc_raw.csv"
dat <- read.csv(file=paste(WORKING_PATH, f, sep=""))
dat <- filter(dat, Time.Horizon == 2000)
priors <- as.list(unique(dat['Prior']))$Prior
priors <- c("Needle In Haystack", "Heavy Tail", "Uniform")
warm_starts <- as.list(unique(dat['Warm.Start']))$Warm.Start
warm_starts <- c(20, 250, 500)
algs <- c("ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy")

for (prior in priors) {
  results_market_shares <- matrix(nrow=3, ncol=length(warm_starts))
  results_eeog <- matrix(nrow=3, ncol=length(warm_starts))
  rownames(results_market_shares) <- c("TS vs DG", "TS vs DEG", "DG vs DEG")
  rownames(results_eeog) <- c("TS vs DG", "TS vs DEG", "DG vs DEG")
  colnames(results_market_shares) <- lapply(warm_starts, function (start) { return(paste("$T_0$ =", start))})
  colnames(results_eeog) <- lapply(warm_starts, function (start) { return(paste("$T_0$ =", start))})
  for (i in 1:length(warm_starts)) {
    warm_start = warm_starts[i]
    for (k in 1:length(algs)) {
      for (l in 1:length(algs)) {
        p1alg = algs[k]
        p2alg = algs[l]
        filtered_dat <- filter(dat, Prior == prior & Agent.Alg == "HardMax" & Warm.Start == warm_start & P1.Alg == p1alg & P2.Alg == p2alg)
        if (nrow(filtered_dat) == 0) { next }
        share <- filtered_dat$Market.Share.for.P1
        test <- t.test(share)
        cin <- signif(test$conf.int[2] - mean(share), digits=1)
        eeog_mean <- signif(mean(filtered_dat$EEOG), digits=2)
        eeog_summary <- summary(filtered_dat$EEOG)
        res_text <- paste("\\makecell{\\textbf{",signif(mean(share), digits=2), "} $\\pm$", cin, "}" , sep="")
        res_eeog <- paste(eeog_mean, " (", eeog_summary["Median"], ") ", sep="")
        if (p1alg == "ThompsonSampling" && p2alg == "DynamicGreedy") {
          results_market_shares[1, i] <- res_text
          results_eeog[1, i] <- res_eeog
        } else if (p1alg == "ThompsonSampling" && p2alg == "DynamicEpsilonGreedy") {
          results_market_shares[2, i] <- res_text
          results_eeog[2, i] <- res_eeog
        } else if (p1alg == "DynamicGreedy" && p2alg == "DynamicEpsilonGreedy") {
          results_market_shares[3, i] <- res_text
          results_eeog[3, i] <- res_eeog
        }
      }
    }
  }
  #tab <-xtable(results_market_shares, caption=paste("Duopoly Market Shares Experiment", prior), type="latex")
  #print(tab)
  tab <-xtable(results_eeog, caption=paste("Duopoly EEOG Experiment", prior), type="latex")
  print(tab)
}


