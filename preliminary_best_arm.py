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


T = 1501
N = 100
K = 10


DEFAULT_COMMON_PRIOR = [beta(1, 1) for k in xrange(K)]


def sim(alg):
  for t in xrange(T):
    banditAlg.executeStep(t)
  return (banditAlg.realizedRewardHistory, banditAlg.realizedCumulativeRewardHistory, banditAlg.meanRewardHistory, banditAlg.meanCumulativeRewardHistory)

def sim_best_arm(alg):
  bestArm = banditAlg.banditProblemInstance.getBestArm()
  res = []
  look_at = [100, 200, 250, 500, 1000, 1500]
  for t in xrange(T):
    if t in look_at:
      identifiedBestArm = np.argmax([banditAlg.posteriors[i].mean() for i in xrange(K)])
      res.append((t,identifiedBestArm == bestArm))
    banditAlg.executeStep(t)
  identifiedBestArm = np.argmax([banditAlg.posteriors[i].mean() for i in xrange(K)])
  res.append((T, identifiedBestArm == bestArm))
  return res

default_mean = 0.5
needle_in_haystack = [bernoulli(default_mean) for i in xrange(K)]

needle_in_haystack_50_high = deepcopy(needle_in_haystack)
needle_in_haystack_50_high[int(K/2)] = bernoulli(default_mean + 0.2)

#needle_in_haystack_50_one_medium_one_high = deepcopy(needle_in_haystack)
#needle_in_haystack_50_one_medium_one_high[int(K/2)] = bernoulli(default_mean + 0.3)
#needle_in_haystack_50_one_medium_one_high[int(K/2) + 1] = bernoulli(default_mean + 0.1)



heavy_tail_prior = beta(0.6, 0.6)
heavy_tailed = [bernoulli(heavy_tail_prior.rvs()) for i in xrange(K)]


ALGS = [ThompsonSampling, DynamicEpsilonGreedy, DynamicGreedy]
BANDIT_DISTR = {
  'Needle In Haystack High': needle_in_haystack_50_high,
  'Uniform': None,
  'Heavy Tail': heavy_tail_prior
}


# Algorithm, Arms, Prior, t, n, reward

FIELDNAMES = ['Algorithm', 'K', 'Distribution', 'Iter', 't', 'Best Arm Identification']
simResults = {}

'''
This runs N iterations of the experiment. In each iteration a set of true distributions is chosen and then for each algorithm we record the reward history. After this is done we aggregate the results and report them.
'''

with open('results/preliminary_raw_results/preliminary_plots_best_arm_10_arms.csv', 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
  writer.writeheader()
  for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
    for a in xrange(len(ALGS)):
      simResults[ALGS[a]] = []

    for i in xrange(N):
      if banditDistrName == 'Uniform':
        banditDistr = [bernoulli(random.uniform(0.25, 0.75)) for j in xrange(K)]
      elif banditDistrName == 'Heavy Tail':
        banditDistr = [bernoulli(heavy_tail_prior.rvs()) for j in xrange(K)]

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
        results = sim_best_arm(banditAlg)
        name = alg.__name__
        for (t, res) in results:
          res = {
            'Algorithm': name,
            'Iter': i,
            't': t,
            'K': str(K),
            'Distribution': banditDistrName,
            'Best Arm Identification': res
          }
          writer.writerow(res)

print('all done!')
