library(dplyr)
library(ggplot2)
library(reshape2)

#dat <- read.csv("/Users/garidor/Desktop/bandits-rl-project/results/preliminary_raw_results/preliminary_plots_unified.csv")

concise_alg_rep <- function(alg) {
  if (alg == "ThompsonSampling") {
    return("TS")
  } else if (alg == "DynamicEpsilonGreedy") {
    return("DEG")
  } else if (alg == "DynamicGreedy") {
    return("DG")
  }
}
print_relative_graphs <- function (dist, alg1, alg2, minComplexity) {
  dist_dat <- filter(dat, Distribution == dist & Realized.Complexity >= minComplexity)
  alg_1 <- filter(dist_dat, Algorithm == alg1)
  alg_2 <- filter(dist_dat, Algorithm == alg2)
  n_vals <- unique(dist_dat$n)
  t_vals <- seq(10, 2000, 10)
  df <- as.data.frame(matrix(nrow=length(t_vals), ncol=3))
  colnames(df) <- c("t", "relative_rep")
  df$t <- t_vals
  
  
  counts <- c()
  ses <- c()
  for (t in t_vals) {
    cur_alg1 <- filter(alg_1,  t == UQ(t))
    cur_alg2 <- filter(alg_2, t == UQ(t))
    count <- sum(cur_alg1$Realized.Reputation - cur_alg2$Realized.Reputation >= 0)
    p <- count / length(n_vals)
    counts <- c(counts, p)
    ses <- c(ses, sqrt(p * (1 - p) / length(n_vals)))
  }
  df$relative_rep <- counts
  df$se <- ses
  plot_title <- paste("Relative Reputation -", dist)
  q <- ggplot(df, aes(x=t, y=relative_rep)) + geom_line() + geom_point() +
    geom_errorbar(aes(ymin=relative_rep-1.96*se, ymax=relative_rep+1.96*se), width=.2) +
    ggtitle(plot_title) + xlab("t") + 
    ylab(paste(concise_alg_rep(alg1), " >= ", concise_alg_rep(alg2), "(reputation)")) +
    theme(plot.title = element_text(hjust = 0.5))
  
  print(q)
}

print_mean_graphs <- function (dist, alg1, alg2, alg3, minComplexity) {
  dist_dat <- filter(dat, Distribution == dist & Realized.Complexity >= minComplexity)
  alg_1 <- filter(dist_dat, Algorithm == alg1)
  alg_2 <- filter(dist_dat, Algorithm == alg2)
  alg_3 <- filter(dist_dat, Algorithm == alg3)
  n_vals <- unique(dat$n)
  t_vals <- seq(10, 2000, 10)
  
  df <- as.data.frame(matrix(nrow=0, ncol=4))
  colnames(df) <- c("t", "mean_rep", "ci", "alg")
  for (t in t_vals) {
    cur_alg1 <- filter(alg_1, t == UQ(t))
    cur_alg2 <- filter(alg_2, t == UQ(t))
    cur_alg3 <- filter(alg_3, t == UQ(t))
    df[nrow(df) + 1,] <-c(t, mean(cur_alg1$Realized.Reputation), 1.96 * sd(cur_alg1$Realized.Reputation) / sqrt(nrow(cur_alg1)), concise_alg_rep(alg1))
    df[nrow(df) + 1,] <- c(t, mean(cur_alg2$Realized.Reputation), 1.96 * sd(cur_alg2$Realized.Reputation) / sqrt(nrow(cur_alg2)), concise_alg_rep(alg2))
    df[nrow(df) + 1,] <-c(t, mean(cur_alg3$Realized.Reputation), 1.96 * sd(cur_alg3$Realized.Reputation) / sqrt(nrow(cur_alg3)), concise_alg_rep(alg3))
  }
  df$t <- as.numeric(df$t)
  df$mean_rep <- as.numeric(df$mean_rep)
  df$ci <- as.numeric(df$ci)

  plot_title <- paste("Mean Reputation -", dist)
  q <- ggplot(data=df, aes(x=t, y=mean_rep, colour=alg)) + geom_line() +
    geom_errorbar(aes(ymin=mean_rep-ci, ymax=mean_rep+ci), width=.2) +
    ggtitle(plot_title) + xlab("t") +
    ylab("Mean Reputation") + theme(plot.title = element_text(hjust = 0.5))
  
  print(q)
}

dists <- unique(dat$Distribution)

for (dist in dists) {
  print_relative_graphs(dist, "ThompsonSampling", "DynamicEpsilonGreedy", 0)
  print_relative_graphs(dist, "ThompsonSampling", "DynamicGreedy", 0)
  print_relative_graphs(dist, "DynamicEpsilonGreedy", "DynamicGreedy", 0)
}
# print mean plots
for (dist in dists) {
  print_mean_graphs(dist, "ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy", 0)
}

tmp_ts <- function(n) {
  return(filter(ts, n == UQ(n)))
}

tmp_dg <- function(n) {
  return(filter(dg, n == UQ(n)))
}

