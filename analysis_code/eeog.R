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

WORKING_PATH <- "/Users/garidor/Desktop/CompetingBandits/results/tournament_raw_results/"
f <- "tournament_experiment_longer_ws_raw.csv"
dat <- read.csv(file=paste(WORKING_PATH, f, sep=""))
dat <- filter(dat, Time.Horizon == 2000)
priors <- as.list(unique(dat['Prior']))$Prior
priors <- c("Needle In Haystack", "Heavy Tail", "Uniform")
warm_starts <- as.list(unique(dat['Warm.Start']))$Warm.Start
warm_starts <- c(20, 250, 500)
algs <- c("ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy")

for (prior in priors) {
  results <- matrix(nrow=3, ncol=length(warm_starts))
  rownames(results) <- c("TS vs DG", "TS vs DEG", "DG vs DEG")
  colnames(results) <- lapply(warm_starts, function (start) { return(paste("$T_0$ =", start))})
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
        res_text <- paste("\\makecell{\\textbf{",signif(mean(share), digits=2), "} $\\pm$", cin, "\\\\ eeog \\\\ avg: ", eeog_mean, "\\\\ med: ", eeog_summary["Median"], "}" , sep="")
        if (p1alg == "ThompsonSampling" && p2alg == "DynamicGreedy") {
          results[1, i] <- res_text
        } else if (p1alg == "ThompsonSampling" && p2alg == "DynamicEpsilonGreedy") {
          results[2, i] <- res_text
        } else if (p1alg == "DynamicGreedy" && p2alg == "DynamicEpsilonGreedy") {
          results[3, i] <- res_text
        }
      }
    }
  }
  tab <-xtable(results, caption=paste("Duopoly Experiment", prior), type="latex")
  print(tab)
  
}


