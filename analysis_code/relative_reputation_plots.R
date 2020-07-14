library(dplyr)
library(ggplot2)
library(reshape2)
library(pBrackets)
WORKING_PATH <- "/Users/guyaridor/Dropbox/Competing Bandits/Isolation Results"
setwd(WORKING_PATH)

dat <- read.csv("longer_ws.csv", stringsAsFactors=F)


bracketsGrob <- function(...){
  l <- list(...)
  e <- new.env()
  e$l <- l
  grid:::recordGrob(  {
    do.call(grid.brackets, l)
  }, e)
}

b1 <- bracketsGrob(0.33, 0.15, 0, 0.15, h=0.1, lwd=2, col="red")
b2 <- bracketsGrob(1, 0.15, 0.35, 0.15, h=0.1,  lwd=2, col="red")
concise_alg_rep <- function(alg) {
  alg <- as.character(alg)
  if (alg == "ThompsonSampling") {
    return("TS")
  } else if (alg == "DynamicEpsilonGreedy") {
    return("BEG")
  } else if (alg == "DynamicGreedy") {
    return("BG")
  }
}

dat$alg<- sapply(dat$Algorithm, concise_alg_rep)
print_relative_graphs <- function (dat, dist, alg1, alg2, minComplexity, withAnnotation) {
  dist_dat <- filter(dat, Distribution == dist & Realized.Complexity >= minComplexity)
  alg_1 <- filter(dist_dat, Algorithm == alg1)
  alg_2 <- filter(dist_dat, Algorithm == alg2)
  n_vals <- unique(dist_dat$n)
  t_vals <- seq(10, 1500, 10)
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
  q <- NA
  if (withAnnotation) {
    q <- ggplot(df, aes(x=t, y=relative_rep)) + geom_line() + geom_point() +
      geom_errorbar(aes(ymin=relative_rep-1.96*se, ymax=relative_rep+1.96*se), width=.2) +
      ggtitle(plot_title) + xlab("time") + 
      ylab(paste(concise_alg_rep(alg1), " >= ", concise_alg_rep(alg2), "(reputation)")) +
      annotation_custom(b1) +
      annotate("text", x = 275, y = 0.28, label = "Exploration Disadvantage Period", size=5) +
      theme_bw(base_size = 20) +  ylim(c(0.2, 0.7)) +
      theme(plot.title = element_text(hjust = 0.5))
  } else {
    q <- ggplot(df, aes(x=t, y=relative_rep)) + geom_line() + geom_point() +
      geom_errorbar(aes(ymin=relative_rep-1.96*se, ymax=relative_rep+1.96*se), width=.2) +
      ggtitle(plot_title) + xlab("time") + 
      ylab(paste(concise_alg_rep(alg1), " >= ", concise_alg_rep(alg2), "(reputation)")) +
      theme_bw(base_size = 20) +  ylim(c(0.2, 0.7)) +
      theme(plot.title = element_text(hjust = 0.5))
  }
  ggsave(paste("relative_", dist, "_", alg1, "_", alg2, ".pdf", sep=""), device="pdf", plot=q, width=8.14, height=7.6)
  return(q)
}

print_mean_graphs <- function (dat, dist, alg1, alg2, alg3, minComplexity, key, key_name) {
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
    df[nrow(df) + 1,] <-c(t, mean(cur_alg1[[key]]), 1.96 * sd(cur_alg1[[key]]) / sqrt(nrow(cur_alg1)), concise_alg_rep(alg1))
    df[nrow(df) + 1,] <- c(t, mean(cur_alg2[[key]]), 1.96 * sd(cur_alg2[[key]]) / sqrt(nrow(cur_alg2)), concise_alg_rep(alg2))
    df[nrow(df) + 1,] <-c(t, mean(cur_alg3[[key]]), 1.96 * sd(cur_alg3[[key]]) / sqrt(nrow(cur_alg3)), concise_alg_rep(alg3))
  }
  df$t <- as.numeric(df$t)
  df$mean_rep <- as.numeric(df$mean_rep)
  df$ci <- as.numeric(df$ci)
  df$Algorithm <- df$alg
  plot_title <- paste(key_name, dist, sep=" - ")
  g <- ggplot(data=df, aes(x=t, y=mean_rep, colour=Algorithm)) + geom_line(size=1) +
    geom_errorbar(aes(ymin=mean_rep-ci, ymax=mean_rep+ci), width=.2) +
    ggtitle(plot_title) + xlab("time") +
    ylab(paste("Mean", key_name)) +
    theme_bw(base_size = 20) + 
    theme(plot.title = element_text(hjust = 0.5), legend.position="bottom")
  
  ggsave(paste("mean_", key_name, "_", dist, ".pdf", sep=""), device="pdf", plot=g, width=8.14, height=7.6)
  return(g)
}

difference_over_time <- function(dat, dist, alg1, alg2) {
  dat <- filter(dat, Distribution == dist)
  t_vals <- c(500,1000,2000)
  
  iso_alg1 <- filter(dat, t %in% t_vals & Algorithm == alg1)
  iso_alg2 <- filter(dat, t %in% t_vals & Algorithm == alg2)
  iso_alg1$time <- as.factor(iso_alg1$t)
  iso_alg1$rep_diff <- iso_alg1$Realized.Reputation - iso_alg2$Realized.Reputation
  iso_alg1$Time <- iso_alg1$time
  
  title <- paste(concise_alg_rep(alg1), "-", concise_alg_rep(alg2), "Difference -", dist)
  g <- ggplot(iso_alg1, aes(rep_diff)) + geom_density(aes(group=Time, colour=Time), size=1) + 
    ggtitle(title) + 
    xlab("Reputation Difference") +
    theme_bw(base_size = 20) +
    theme(plot.title = element_text(hjust = 0.5), legend.position="bottom") +
    xlim(c(-0.25, 0.25))
  ggsave(paste("reputation_difference_", dist, ".pdf", sep=""), device="pdf", plot=g, width=8.14, height=7.6)
  print(g)
}

distribution_over_algorithms <- function(dat, dist, time) {
  dat <- dat %>% filter(Distribution == dist & t == time)
  dat <- dat %>% mutate(Algorithm = alg)
  #print(unique(dat$Algorithm))
  g <- ggplot(dat, aes(x=Realized.Reputation)) + geom_density(aes(colour=Algorithm), size=1) + 
    theme_bw(base_size = 20) + ggtitle(paste("Reputation Distribution -", dist)) +
    theme(plot.title = element_text(hjust = 0.5), legend.position="bottom") + xlab("Reputation")
  ggsave(paste("rep_distribution_", dist, ".pdf", sep=""), device="pdf", plot=g, width=8.14, height=7.6)
}

distribution_over_algorithms(dat, "Needle In Haystack", 500)
difference_over_time(dat, "Needle In Haystack", "ThompsonSampling", "DynamicGreedy")
dists <- unique(dat$Distribution)

# print relative reputation plots
print_relative_graphs(dat, "Uniform", "ThompsonSampling", "DynamicGreedy", 0, T)
print_relative_graphs(dat, "Needle In Haystack", "ThompsonSampling", "DynamicGreedy", 0, F)

# print instantaneous reward plots
p1 <- print_mean_graphs(dat, "Needle In Haystack", "ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy", 0, "Instantaneous.Mean.Reward.Mean", "Mean Instantaneous Reward")
p2 <- print_mean_graphs(dat, "Heavy Tail", "ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy", 0, "Instantaneous.Mean.Reward.Mean", "Mean Instantaneous Reward")
p3 <- print_mean_graphs(dat, "Uniform", "ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy", 0, "Instantaneous.Mean.Reward.Mean", "Mean Instantaneous Reward")

# print reputation plots
p1 <- print_mean_graphs(dat, "Needle In Haystack", "ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy", 0, "Realized.Reputation", "Mean Reputation")
p2 <- print_mean_graphs(dat, "Heavy Tail", "ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy", 0, "Realized.Reputation", "Mean Reputation")
p3 <- print_mean_graphs(dat, "Uniform", "ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy", 0, "Realized.Reputation", "Mean Reputation")

dat <- read.csv("ht_3_arm_many_sim.csv",stringsAsFactors=F)
dat$alg<- sapply(dat$Algorithm, concise_alg_rep)
print_mean_graphs(dat, "Heavy Tail", "ThompsonSampling", "DynamicEpsilonGreedy", "DynamicGreedy", 0,  "Realized.Reputation", "Mean Reputation")
print_relative_graphs(dat, "Heavy Tail", "DynamicEpsilonGreedy", "DynamicGreedy", 0, F)

