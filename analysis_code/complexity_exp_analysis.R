source("bin_scatter.R")

dat <- read.csv("/Users/garidor/Desktop/bandits-rl-project/results/tournament_raw_results/tournament_experiment_fixed_complexity_raw.csv")
dat <- filter(dat, Time.Horizon == 1000)
alg1 <- "ThompsonSampling"
alg2 <- "DynamicEpsilonGreedy"
ws <- 100
#filtered_dat <- filter(dat, P1.Alg == alg1 & P2.Alg == alg2 & Warm.Start == ws & Instance.Complexity < 10000)
#grouped_dat <- filtered_dat %>% group_by(Instance.Complexity, N) %>% summarize(mshare=mean(Market.Share.for.P1))
#binscatter(formula="mshare ~ Instance.Complexity", key_var = "Instance.Complexity", data=grouped_dat, bins=5, partial=FALSE)

n_vals <- unique(dat$N)
complexities <- c(50, 150, 200, 250, 300, 350, 400)
final_dat <- as.data.frame(matrix(nrow=length(complexities)*length(n_vals), ncol=2))
for (i in 1:length(complexities)) {
  complexity <- complexities[i]
  lower_complex <- complexity - 10
  upper_complex <- complexity + 10
  for (j in 1:length(n_vals)) {
    n = n_vals[j]
    filtered_dat <- filter(dat, P1.Alg == alg1 & P2.Alg == alg2 & Warm.Start == ws & Instance.Complexity <= upper_complex & Instance.Complexity >= lower_complex & N == n)
    final_dat[(i-1)*length(n_vals)+j, 1] <- complexity
    final_dat[(i-1)*length(n_vals)+j, 2] <- mean(filtered_dat$Market.Share.for.P1)
  }
}

colnames(final_dat) <- c("Complexity", "Market.Share")
binscatter(formula="Market.Share ~ Complexity", key_var = "Complexity", data=final_dat, bins=7, partial=FALSE)
