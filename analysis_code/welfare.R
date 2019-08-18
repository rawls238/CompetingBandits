library(tidyverse)
library(ggplot2)
library(gdata)

simultaneous_welfare <- read.csv("/Volumes/My Passport/bandits/tournament_raw_results/tournament_experiment_full_welfare_calc_raw.csv")
free_obs_welfare <- read.csv("/Volumes/My Passport/bandits/free_obs_raw_results/free_obs_experiment_welfare_calc_raw.csv")
#prelim <- read.csv("/Volumes/Mac/final_bandit_results/results/preliminary_raw_results/longer_ws.csv")
prelim <- read.csv("/Volumes/My Passport/bandits/preliminary_raw_results/regret_calc.csv")
sim_eq_welfare <- filter(simultaneous_welfare, ((Prior == "Heavy Tail" | Prior == "Uniform") & (P1.Alg == "DynamicGreedy" & P2.Alg == "DynamicGreedy")) | (Prior == "Needle In Haystack" & P1.Alg == "ThompsonSampling" & P2.Alg == "ThompsonSampling"))
free_obs_eq_welfare <- filter(free_obs_welfare, ((Prior == "Heavy Tail" | Prior == "Uniform") & (P1.Alg == "DynamicGreedy" & P2.Alg == "ThompsonSampling")))
one_welfare <- combine(sim_eq_welfare, free_obs_eq_welfare)

one_welfare <- one_welfare %>% mutate(label=ifelse(source == "free_obs_eq_welfare", "Temporary Monopoly", "Duopoly"))

prelim <- prelim %>% mutate(Prior = Distribution, Warm.Start = 20, label = "Monopoly", Total.Regret = Cumulative.Regret)
prelim <- prelim %>% mutate(Time.Horizon = t - 20)

ht_greedy <- filter(prelim, Prior == "Heavy Tail" & Algorithm == "DynamicGreedy")
unif_greedy <- filter(prelim, Prior == "Uniform" & Algorithm == "DynamicGreedy")
ht_ts <- filter(prelim, Prior == "Heavy Tail" & Algorithm == "ThompsonSampling")
ht_ts <- ht_ts %>% mutate(label = "TS, No Competition")
unif_ts <- filter(prelim, Prior == "Uniform" & Algorithm == "ThompsonSampling")
unif_ts <- unif_ts %>% mutate(label = "TS, No Competition")

ws_20_ht <- filter(one_welfare, Warm.Start == 20 & Prior == "Heavy Tail")
ws_20_ht <- ws_20_ht %>% mutate(Total.Regret = ifelse(label == "Temporary Monopoly", Market.Regret + WS1.Regret + WS2.Regret + Incum.Regret, Market.Regret + WS1.Regret + WS2.Regret))
ws_20_ht <- bind_rows(ws_20_ht, ht_greedy, ht_ts)
ggplot(ws_20_ht, aes(x=Time.Horizon, y=Total.Regret)) + geom_smooth(aes(colour=label)) + 
  ylab("Cumulative Market Regret") + xlab("Time") + ggtitle("Equilibrium Welfare - Heavy Tail Instance") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5))

ws_20_uniform <- filter(one_welfare, Warm.Start == 20 & Prior == "Uniform")
ws_20_uniform <- ws_20_uniform %>% mutate(Total.Regret = ifelse(label == "Temporary Monopoly", Market.Regret + WS1.Regret + WS2.Regret +Incum.Regret, WS1.Regret + WS2.Regret + Market.Regret))
ws_20_uniform <- bind_rows(ws_20_uniform, unif_greedy, unif_ts)
ggplot(ws_20_uniform, aes(x=Time.Horizon, y=Total.Regret)) + geom_smooth(aes(colour=label)) +
  ylab("Cumulative Market Regret") + xlab("Time") + ggtitle("Equilibrium Welfare - Uniform") + 
  theme_bw() + theme(plot.title = element_text(hjust = 0.5))

many_markets <- read.csv("/Volumes/My Passport/bandits/tournament_raw_results/tournament_experiment_many_markets_raw.csv")
many_markets$NumFirms <- as.factor(many_markets$NumFirms)

unif_competition <- filter(many_markets, Prior == "Uniform" & Warm.Start == 20)
select_firms <- filter(unif_competition, NumFirms == 1 | NumFirms == 5 | NumFirms == 10 | NumFirms == 15 | NumFirms == 19)
ggplot(select_firms, aes(x=Time.Horizon, y=Market.Regret)) + geom_smooth(aes(colour=NumFirms)) +
  xlab("Time") + ylab("Market Regret") + ggtitle("Welfare - Greedy, Uniform") + theme_bw(base_size = 12) + theme(plot.title = element_text(hjust = 0.5))

ht_competition <- filter(many_markets, Prior == "Heavy Tail" & Warm.Start == 20)
select_firms <- filter(ht_competition, NumFirms == 1 | NumFirms == 5 | NumFirms == 10 | NumFirms == 15 | NumFirms == 19)
ggplot(select_firms, aes(x=Time.Horizon, y=Market.Regret)) + geom_smooth(aes(colour=NumFirms)) +
  xlab("Time") + ylab("Market Regret") + ggtitle("Welfare - Greedy, Heavy Tail Instance") + theme_bw(base_size = 12) + theme(plot.title = element_text(hjust = 0.5))


ggplot(ht_competition, aes(x=as.numeric(NumFirms), y=EEOG)) + geom_smooth() + 
  xlab("Number of Firms") + ylab("Effective End of Game") + ggtitle("EEOG vs Number of Firms, Heavy Tail") + 
  theme_bw(base_size = 12) + theme(plot.title = element_text(hjust = 0.5))

ggplot(unif_competition, aes(x=as.numeric(NumFirms), y=EEOG)) + geom_smooth() + 
  xlab("Number of Firms") + ylab("Effective End of Game") + ggtitle("EEOG vs Number of Firms, Uniform") + 
  theme_bw(base_size = 12) + theme(plot.title = element_text(hjust = 0.5))
