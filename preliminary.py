import numpy as np
import csv
import random


from copy import deepcopy


from lib.BanditProblemInstance import BanditProblemInstance

from lib.bandit.StaticGreedy import StaticGreedy
from lib.bandit.DynamicEpsilonGreedy import DynamicEpsilonGreedy
from lib.bandit.NonBayesianEpsilonGreedy import NonBayesianEpsilonGreedy
from lib.bandit.DynamicGreedy import DynamicGreedy
from lib.bandit.UCB import UCB1WithConstantOne, UCB1WithConstantT
from lib.bandit.ThompsonSampling import ThompsonSampling
from lib.bandit.ExploreThenExploit import ExploreThenExploit

from scipy.stats import bernoulli, beta


T = 5000
N = 200
K = 3


DEFAULT_COMMON_PRIOR = [beta(1, 1) for k in xrange(K)]


def sim(alg):
  for t in xrange(T):
    banditAlg.executeStep(t)
  return (banditAlg.realizedRewardHistory, banditAlg.realizedCumulativeRewardHistory, banditAlg.meanRewardHistory, banditAlg.meanCumulativeRewardHistory)

default_mean = 0.5
needle_in_haystack = [bernoulli(default_mean) for i in xrange(K)]

needle_in_haystack_50_high = deepcopy(needle_in_haystack)
needle_in_haystack_50_high[int(K/2)] = bernoulli(default_mean + 0.2)

#needle_in_haystack_50_one_medium_one_high = deepcopy(needle_in_haystack)
#needle_in_haystack_50_one_medium_one_high[int(K/2)] = bernoulli(default_mean + 0.3)
#needle_in_haystack_50_one_medium_one_high[int(K/2) + 1] = bernoulli(default_mean + 0.1)



heavy_tail_prior = beta(0.6, 0.6)
heavy_tailed = [bernoulli(heavy_tail_prior.rvs()) for i in xrange(K)]


ALGS = [NonBayesianEpsilonGreedy, ThompsonSampling, UCB1WithConstantOne, DynamicEpsilonGreedy, DynamicGreedy]
BANDIT_DISTR = {
  'Heavy Tail': heavy_tail_prior,
  'Uniform': None,
  'Needle In Haystack High': needle_in_haystack_50_high
}


# Algorithm, Arms, Prior, t, n, reward

FIELDNAMES = ['Algorithm', 'K', 'Distribution', 't', 'Instantaneous Realized Reward Mean', 'Instantaneous Realized Reward Std', 'Cumulative Realized Reward Mean', 'Reputation', 'Cumulative Realized Reward Std', 'Instantaneous Mean Reward Mean', 'Instantaneous Mean Reward Std', 'Cumulative Mean Reward Mean', 'Cumulative Mean Reward Std', 'Best Arm Mean']
simResults = {}


'''
This runs N iterations of the experiment. In each iteration a set of true distributions is chosen and then for each algorithm we record the reward history. After this is done we aggregate the results and report them.
'''

with open('results/preliminary_raw_results/preliminary_plots_3_arms_reputation.csv', 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
  writer.writeheader()
  for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
    for a in xrange(len(ALGS)):
      simResults[ALGS[a]] = []

    for i in xrange(N):
      ''' get realizations of distributions'''
      if banditDistrName == 'Uniform':
        banditDistr = [bernoulli(random.uniform(0.25, 0.75)) for i in xrange(K)]
      elif banditDistrName == 'Heavy Tail':
        banditDistr = [bernoulli(heavy_tail_prior.rvs()) for i in xrange(K)]

      realizations = {}
      for t in xrange(T):
        realizations[t] = {}
        for j in xrange(len(banditDistr)):
          realizations[t][j] = banditDistr[j].rvs()
      banditProblemInstance = BanditProblemInstance(K, T, banditDistr, realizations)
      bestArmMean = banditProblemInstance.bestArmMean()

      for a in xrange(len(ALGS)):
        alg = ALGS[a]
        banditAlg = alg(banditProblemInstance, DEFAULT_COMMON_PRIOR)
        res = sim(banditAlg)
        simResults[alg].append(res)
      
    for t in xrange(T):
      for (alg, algResult) in simResults.iteritems():
        name = alg.__name__
        cumulative_realized = []
        instantaneous_realized = []
        cumulative_mean = []
        instantaneous_mean = []
        reputation_realized = []
        for j in xrange(len(algResult)):
          reputation_realized.append(np.mean(algResult[j][0][max(0,t-100):t]))
          instantaneous_realized.append(algResult[j][0][t])
          cumulative_realized.append(algResult[j][1][t])
          instantaneous_mean.append(algResult[j][2][t])
          cumulative_mean.append(algResult[j][3][t])
        res = {
          'Algorithm': name,
          'K': str(K),
          'Distribution': banditDistrName,
          't': str(t),
          'Instantaneous Realized Reward Mean': np.mean(instantaneous_realized),
          'Instantaneous Realized Reward Std': np.std(instantaneous_realized),
          'Cumulative Realized Reward Mean': np.mean(cumulative_realized),
          'Cumulative Realized Reward Std': np.std(cumulative_realized),
          'Instantaneous Mean Reward Mean': np.mean(instantaneous_mean),
          'Instantaneous Mean Reward Std': np.std(instantaneous_mean),
          'Cumulative Mean Reward Mean': np.mean(cumulative_mean),
          'Cumulative Mean Reward Std': np.std(cumulative_mean),
          'Best Arm Mean': bestArmMean,
          'Reputation': np.mean(reputation_realized)
        }
        writer.writerow(res)
      

