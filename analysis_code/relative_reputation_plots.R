library(dplyr)
library(ggplot2)
library(reshape2)

#dat <- read.csv("/Users/garidor/Desktop/bandits-rl-project/results/preliminary_raw_results/preliminary_plots_unified.csv")

#dat <- read.csv("/Volumes/Mac/bandits/bandits-rl-project/results/preliminary_raw_results/preliminary_plots_10_arms.csv")


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
  t_vals <- seq(100, 2000, 100)
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
    ylab(paste(concise_alg_rep(alg1), " >= ", concise_alg_rep(alg2), "(reputation)"))
  
  print(q)
}

print_mean_graphs <- function (dist, alg1, alg2, minComplexity) {
  K <- unique(dat$K)[1]
  dist_dat <- filter(dat, Distribution == dist & Realized.Complexity >= minComplexity)
  deg <- filter(dist_dat, Algorithm == alg1)
  dg <- filter(dist_dat, Algorithm == alg2)
  n_vals <- unique(dat$n)
  t_vals <- seq(10, 2000, 10)
  
  means_deg <- c()
  means_dg <- c()
  for (t in t_vals) {
    cur_dg <- filter(dg, t == UQ(t))
    cur_deg <- filter(deg, t == UQ(t))
    means_deg <- c(means_deg, mean(cur_deg$Realized.Reputation))
    means_dg <- c(means_dg, mean(cur_dg$Realized.Reputation))
  }
  plot_title <- paste("Mean Rep -", dist, "K =", K)
  q <- qplot(x=t_vals) +
    geom_point(aes(y=means_deg)) +
    geom_point(aes(y=means_dg)) +
    scale_linetype_discrete(name="Alg", labels=c(alg1, alg2)) +
    ggtitle(plot_title) + xlab("t") +
    ylab("Mean Reputation")
  
  print(q)
}

dists <- unique(dat$Distribution)
#dists <- c("Heavy Tail", "Uniform")

for (dist in dists) {
  print_relative_graphs(dist, "ThompsonSampling", "DynamicEpsilonGreedy", 0)
  print_relative_graphs(dist, "ThompsonSampling", "DynamicGreedy", 0)
  print_relative_graphs(dist, "DynamicEpsilonGreedy", "DynamicGreedy", 0)
  #filtered_dist <- filter(dat, Distribution == dist)
  #cat(dist, "Mean: ", mean(filtered_dist$Realized.Complexity), "Median :", median(filtered_dist$Realized.Complexity), "\n")
}

tmp_ts <- function(n) {
  return(filter(ts, n == UQ(n)))
}

tmp_dg <- function(n) {
  return(filter(dg, n == UQ(n)))
}

