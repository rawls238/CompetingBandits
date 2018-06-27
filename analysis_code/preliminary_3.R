library(dplyr)
library(ggplot2)
library(reshape2)

dat <- read.csv("/Users/garidor/Desktop/bandits-rl-project/results/preliminary_raw_results/preliminary_plots_10_arms_10_int.csv")
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
K <- unique(dat$K)[1]
dist_dat <- filter(dat, Distribution == dist & Realized.Complexity >= minComplexity)
deg <- filter(dist_dat, Algorithm == alg1)
dg <- filter(dist_dat, Algorithm == alg2)
n_vals <- unique(dist_dat$n)
t_vals <- seq(10, 1500, 10)

counts <- c()
for (t in t_vals) {
  count <- 0
  for (n in n_vals) {
    cur_delayed_t <- t - 200
    cur_dg <- filter(dg, n == UQ(n) & t == UQ(t))
    cur_deg <- filter(deg, n == UQ(n) & t == UQ(t))
    comp <- cur_deg$Realized.Reputation >= cur_dg$Realized.Reputation
    if (comp) {
      count <- count + 1
    }
  }
  counts <- c(counts, (count / length(n_vals)))
}
plot_title <- paste("Re Rep -", dist, "K =", K, "Comp >=", minComplexity)
q <- qplot(t_vals, counts) + 
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
  t_vals <- seq(100, 5000, 100)
  
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

#dists <- unique(dat$Distribution)
dists <- c("Heavy Tail", "Uniform")

for (dist in dists) {
  print_relative_graphs(dist, "ThompsonSampling", "DynamicEpsilonGreedy", 0)
  print_relative_graphs(dist, "ThompsonSampling", "DynamicGreedy", 0)
  print_relative_graphs(dist, "DynamicEpsilonGreedy", "DynamicGreedy", 0)
  #filtered_dist <- filter(dat, Distribution == dist)
  #cat(dist, "Mean: ", mean(filtered_dist$Realized.Complexity), "Median :", median(filtered_dist$Realized.Complexity), "\n")
}

alg1 <- "DynamicGreedy"
alg2 <- "ThompsonSampling"
filtered_dat <- filter(dat, Distribution != "Needle In Haystack - 0.5" & t == 5000)
alg_1_dat <- filter(filtered_dat, Algorithm == alg1)
complexity_vals <- alg_1_dat$Realized.Complexity
alg_1_rep <- alg_1_dat$Realized.Reputation
alg_2_rep <- filter(filtered_dat, Algorithm == alg2)$Realized.Reputation
rep_diff <- alg_2_rep > alg_1_rep
print(qplot(complexity_vals, rep_diff))

