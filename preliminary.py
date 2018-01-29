import numpy as np
import csv
import random


from copy import deepcopy


from lib.BanditProblemInstance import BanditProblemInstance

from lib.bandit.Bandit
from lib.bandit.StaticGreedy import StaticGreedy
from lib.bandit.DynamicEpsilonGreedy import NonBayesianEpsilonGreedy
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
  return (banditAlg.rewardHistory, banditAlg.cumulativeRewardHistory)



uniform_real_distr = [bernoulli(random.uniform(0.25, 0.75)) for i in xrange(K)]

default_mean = 0.5
needle_in_haystack = [bernoulli(default_mean) for i in xrange(K)]

needle_in_haystack_50_high = deepcopy(needle_in_haystack)
needle_in_haystack_50_high[int(K/2)] = bernoulli(default_mean + 0.2)

needle_in_haystack_50_medium = deepcopy(needle_in_haystack)
needle_in_haystack_50_medium[int(K/2)] = bernoulli(default_mean + 0.05)

needle_in_haystack_50_low = deepcopy(needle_in_haystack)
needle_in_haystack_50_low[int(K/2)] = bernoulli(default_mean + 0.01)

heavy_tail_prior = beta(0.6, 0.6)
heavy_tailed = [bernoulli(heavy_tail_prior.rvs()) for i in xrange(K)]


ALGS = [NonBayesianEpsilonGreedy, NonBayesianEpsilonGreedy, NonBayesianEpsilonGreedy, ThompsonSampling, UCB1WithConstantOne, UCB1WithConstantT, DynamicGreedy]
BANDIT_DISTR = {
  'Uniform': uniform_real_distr, 
  'Heavy Tail' :heavy_tailed,
  'Needle In Haystack High': needle_in_haystack_50_high,
  'Needle In Haystack Medium': needle_in_haystack_50_medium,
  'Needle In Haystack Low': needle_in_haystack_50_medium,
}


# Algorithm, Arms, Prior, t, n, reward

FIELDNAMES = ['Algorithm', 'K', 'Distribution', 't', 'Instantaneous Reward Mean', 'Instantaneous Reward Std', 'Cumulative Reward Mean', 'Cumulative Reward Std', 'Best Arm Mean']
simResults = {}


'''
This runs N iterations of the experiment. In each iteration a set of true distributions is chosen and then for each algorithm we record the reward history. After this is done we aggregate the results and report them.
'''

with open('results/preliminary_plots_3_arms.csv', 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
  writer.writeheader()
  for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
    for a in xrange(len(ALGS)):
      if a <= 2:
        simResults[a] = []
      else:
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
        if alg.shorthand() == 'NBDEG':
          if a == 0:
            banditAlg = NonBayesianEpsilonGreedy(banditProblemInstance, DEFAULT_COMMON_PRIOR)
            res = sim(banditAlg)
            simResults[a].append(res)
          elif a == 1:
            banditAlg = NonBayesianEpsilonGreedy(banditProblemInstance, DEFAULT_COMMON_PRIOR, epsilon=(T**(-1/3)))
            res = sim(banditAlg)
            simResults[a].append(res)
          elif a == 2:
            banditAlg = NonBayesianEpsilonGreedy(banditProblemInstance, DEFAULT_COMMON_PRIOR, dynamicEpsilon=True)
            res = sim(banditAlg)
            simResults[a].append(res)
        else:
          banditAlg = alg(banditProblemInstance, DEFAULT_COMMON_PRIOR)
          res = sim(banditAlg)
          simResults[alg].append(res)
      
    for t in xrange(T):
      for (alg, algResult) in simResults.iteritems():
        # hacky but works for now - makes it so we can test all different epsilon-greedy formulations
        if alg == 0:
          name = 'NonBayesianEpsilonGreedy, 0.05'
        elif alg == 1:
          name = 'NonBayesianEpsilonGreedy, T^(-1/3)'
        elif alg == 2:
          name = 'NonBayesianEpsilonGreedy, (t+1)^(-1/3)'
        else:
          name = alg.__name__
        cumulative = []
        instantaneous = []
        for j in xrange(len(algResult)):
          instantaneous.append(algResult[j][0][t])
          cumulative.append(algResult[j][1][t])
        res = {
          'Algorithm': name,
          'K': str(K),
          'Distribution': banditDistrName,
          't': str(t),
          'Instantaneous Reward Mean': np.mean(instantaneous),
          'Instantaneous Reward Std': np.std(instantaneous),
          'Cumulative Reward Mean': np.mean(cumulative),
          'Cumulative Reward Std': np.std(cumulative),
          'Best Arm Mean': bestArmMean
        }
        writer.writerow(res)
      

