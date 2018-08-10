library(dplyr)

dat_1 <- read.csv("/Users/garidor/Desktop/bandits-rl-project/results/preliminary_raw_results/preliminary_plots_unified.csv")
dat_2 <- read.csv("/Users/garidor/Desktop/bandits-rl-project/results/tournament_raw_results/tournament_experiment_full_sim_with_realizations_raw.csv")

dist <- "Heavy Tail"
ws_time <- 20
iso_dat <- filter(dat_1, Distribution == dist)
compet_dat <- filter(dat_2, Prior == dist & Warm.Start == 20)

alg1 <- "ThompsonSampling"
alg2 <- "DynamicGreedy"
alg_1 <- filter(iso_dat, Algorithm == alg1)
alg_2 <- filter(iso_dat, Algorithm == alg2)
n_vals <- unique(iso_dat$n)
winning_n <- c()
losing_n <- c()
for (n in n_vals) {
  cur_alg1 <- filter(alg_1,  t == ws_time & n == UQ(n))
  cur_alg2 <- filter(alg_2, t == ws_time & n == UQ(n))
  if (cur_alg1$Realized.Reputation > cur_alg2$Realized.Reputation) {
    winning_n <- c(winning_n, n)
  } else {
    losing_n <- c(losing_n, n)
  }
}

compet_dat_filtered <- filter(compet_dat, Time.Horizon == 2000 & Agent.Alg == "HardMax" & P1.Alg == alg1 & P2.Alg == alg2)
winning_dat <- filter(compet_dat_filtered, row_number() %in% winning_n)
losing_dat <- filter(compet_dat_filtered, row_number() %in% losing_n)
cat("Winning dat", nrow(winning_dat), "Market share", mean(winning_dat$Market.Share.for.P1))
cat("Losing dat", nrow(losing_dat), "Market share", mean(losing_dat$Market.Share.for.P1))
