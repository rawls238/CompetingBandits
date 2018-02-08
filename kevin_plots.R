library(data.table)
library(ggplot2)

dt.data <- data.table(read.csv("~/Desktop/bandits-rl-project/results/preliminary_plots.csv", header=T, stringsAsFactors=T))


# dt.data[(Algorithm == 'ThompsonSampling' & K==10)]

distribution_names = unique(dt.data[, Distribution])

q <- ggplot(dt.data[(Distribution=="Needle50 - High")], aes(x=t, y=Reward.Mean, color=Algorithm)) +
  # geom_point(alpha=0.1) +
  geom_smooth(method="loess", se=TRUE) +
  scale_x_continuous("Time") +
  scale_y_continuous('Average instantaneous reward') +
  theme_bw() +
  ggtitle("Average reward over time")
print(q)


##########

WORKING_PATH <- '/Users/kevinliu/dev/bandits-rl-project'

dat <- data.table(read.csv(file=paste(WORKING_PATH, "/results/preliminary_plots_3_arms.csv", sep="")))
dat2 <- data.table(read.csv(file=paste(WORKING_PATH, "/results/preliminary_plots_3_arms_2.csv", sep="")))
dt.data <- rbind(dat, dat2)
rm(dat)
rm(dat2)

unique(dt.data$SubAlgorithm)

# renaming so that we can get the desired color vs linetype aesthetics for easy visual identification
dt.data[, SubAlgorithm := Algorithm]
dt.data[SubAlgorithm %in% c("NonBayesianEpsilonGreedy, 0.05", 
                            "NonBayesianEpsilonGreedy, T^(-1/3)", 
                            "NonBayesianEpsilonGreedy, (t+1)^(-1/3)"), 
        Algorithm := "NonBayesianEpsilonGreedy"]
dt.data[SubAlgorithm %in% c("DynamicGreedy", 
                            "DynamicEpsilonGreedy, 0.05", 
                            "DynamicEpsilonGreedy, T^(-1/3)", 
                            "DynamicEpsilonGreedy, (t+1)^(-1/3)"), 
        Algorithm := "DynamicEpsilonGreedy"]
dt.data[SubAlgorithm %in% c("UCB1WithConstantOne", 
                            "UCB1WithConstantT"), 
        Algorithm := "UCB1"]

dt.data[(Distribution=="Needle In Haystack Medium" & Algorithm == "NonBayesianEpsilonGreedy")]

ggplot(dt.data[(Distribution=="Needle In Haystack Medium" & Algorithm=="NonBayesianEpsilonGreedy")], aes(x=t, y=Instantaneous.Reward.Mean, color=Algorithm, linetype=SubAlgorithm)) +
  # geom_point(alpha=0.1) +
  geom_smooth(method="loess", se=TRUE) +
  scale_x_continuous("Time") +
  scale_y_continuous('Average instantaneous reward') +
  theme_bw() +
  ggtitle("Average reward over time")
