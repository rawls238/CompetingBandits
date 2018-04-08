library(dplyr)

WORKING_PATH <- "/Users/garidor/Desktop/bandits-rl-project"
f <- "tournament_experiment_everything_hmr_raw.csv"
dat <- read.csv(file=paste(WORKING_PATH, "/results/tournament_raw_results/", f, sep=""))
filtered_dat <- filter(dat, Prior == "Heavy Tail")

WINDOW <- 5000
for (i in 220:240) {
  begin <- (i*WINDOW) + 1
  end_window <- (i + 1) * WINDOW
  one_sim <- filtered_dat[begin:end_window,]
  plot(one_sim[,"Time.Horizon"], one_sim[, "Market.Share.for.P1"], main="HMR Market Share over Time", ylab="Market Share", xlab="Time")
}