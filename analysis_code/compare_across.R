library(dplyr)

dat_1 <- read.csv("/Users/garidor/Desktop/CompetingBandits/results/preliminary_raw_results/longer_ws.csv")
#dat_2 <- read.csv("/Users/garidor/Desktop/CompetingBandits/results/tournament_raw_results/tournament_experiment_full_sim_with_realizations_raw.csv")


concise_alg_rep <- function(alg) {
  if (alg == "ThompsonSampling") {
    return("TS")
  } else if (alg == "DynamicEpsilonGreedy") {
    return("DEG")
  } else if (alg == "DynamicGreedy") {
    return("DG")
  }
}

dat_1$Algorithm<- sapply(dat_1$Algorithm, concise_alg_rep)
tmp <- mutate(dat_1, Algorithm = concise_alg_rep(Algorithm))
dist <- ".5/.7 Random Draw"
ws_time <- 20
iso_dat <- filter(dat_1, Distribution == dist)
compet_dat <- filter(dat_2, Prior == dist & Warm.Start == 20)

alg1 <- "TS"
alg2 <- "DG"
alg_1 <- filter(iso_dat, Algorithm == alg1)
alg_2 <- filter(iso_dat, Algorithm == alg2)
n_vals <- unique(iso_dat$n)
winning_n <- c()
losing_n <- c()
for (n in n_vals) {
  cur_alg1 <- filter(alg_1,  t == ws_time & n == UQ(n))
  cur_alg2 <- filter(alg_2, t == ws_time & n == UQ(n))
  if (cur_alg1$Realized.Reputation >= cur_alg2$Realized.Reputation) {
    winning_n <- c(winning_n, n)
  } else {
    losing_n <- c(losing_n, n)
  }
}

compet_dat_filtered <- filter(compet_dat, Time.Horizon == 2000 & Agent.Alg == "HardMax" & P1.Alg == alg1 & P2.Alg == alg2)
winning_dat <- filter(compet_dat_filtered, N %in% winning_n)
losing_dat <- filter(compet_dat_filtered, N %in% losing_n)
cat("Winning dat", nrow(winning_dat), "Market share", mean(winning_dat$Market.Share.for.P1), "\n")
cat("Losing dat", nrow(losing_dat), "Market share", mean(losing_dat$Market.Share.for.P1))


# look at density estimates of reputation

dist <- "Needle In Haystack"
ws_time <- 20
iso_dat <- filter(dat_1, Distribution == dist)
iso_dat_t <- filter(iso_dat, t == 500)
ggplot(iso_dat_t, aes(Realized.Reputation, colour=Algorithm)) +
  geom_density() +
  ggtitle("Reputation Distribution, Needle In Haystack") +
  theme_bw(base_size = 12) +
  xlab("Reputation") + 
  theme(plot.title = element_text(hjust = 0.5))

iso_deg <- filter(iso_dat, t %in% t_vals & Algorithm == "ThompsonSampling") 
iso_dg <- filter(iso_dat, t %in% t_vals & Algorithm == "DynamicGreedy")
iso_deg$rep_diff_dg <- iso_deg$Realized.Reputation - iso_dg$Realized.Reputation

ggplot(iso_deg, aes(rep_diff_dg)) + geom_density() + 
  ggtitle("DEG - DG Reputation Distribution, Heavy Tail") + 
  xlab("Reputation Difference") + 
  theme(plot.title = element_text(hjust = 0.5))

dist <- "Needle In Haystack"
ws_time <- 20
iso_dat <- filter(dat_1, Distribution == dist)
t_vals <- c(500,1000,2000)
alg1 <- "TS"
alg2 <- "DG"

iso_alg1 <- filter(iso_dat, t %in% t_vals & Algorithm == alg1) 
iso_alg2 <- filter(iso_dat, t %in% t_vals & Algorithm == alg2)
iso_alg1$rep_diff <- iso_alg1$Realized.Reputation - iso_alg2$Realized.Reputation

title <- paste(alg1, "-", alg2, "Reputation Distribution,", dist)
ggplot(iso_alg1, aes(rep_diff)) + geom_density(aes(group=t, colour=t)) + 
  ggtitle(title) + 
  xlab("Reputation Difference") +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlim(c(-0.25, 0.25))
