#```{r echo=FALSE, message=FALSE, xtable, results="asis"}

library(dplyr)
library(knitr)
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

WORKING_PATH <- "/Users/garidor/Desktop/bandits-rl-project/results/tournament_raw_results/"
f <- "tournament_experiment_full_ws_many_sim_raw.csv"
dat <- read.csv(file=paste(WORKING_PATH, f, sep=""))
#dat_2 <- read.csv(file=paste(WORKING_PATH, "tournament_experiment_tourn_.5_.7_raw.csv", sep=""))
#dat <- rbind(dat, dat_2)
dat <- filter(dat, Time.Horizon == 2000)
priors <- as.list(unique(dat['Prior']))$Prior
priors <- c("Needle In Haystack", "Heavy Tail")
warm_starts <- as.list(unique(dat['Warm.Start']))$Warm.Start
warm_starts <- c(20, 100, 200)
algs <- c("ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy")

for (prior in priors) {
  results <- matrix(nrow=3, ncol=length(warm_starts))
  rownames(results) <- c("TS vs DG", "TS vs DEG", "DG vs DEG")
  colnames(results) <- lapply(warm_starts, function (start) { return(paste("k =", start))})
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
  tab <-xtable(results, caption=paste("Simultaneous Entry", prior), type="latex")
  print(tab)
  
}


