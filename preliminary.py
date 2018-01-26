import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import csv
import random


from copy import deepcopy


from lib.BanditProblemInstance import BanditProblemInstance

from lib.bandit.StaticGreedy import StaticGreedy
from lib.bandit.DynamicEpsilonGreedy import DynamicEpsilonGreedy
from lib.bandit.DynamicGreedy import DynamicGreedy
from lib.bandit.UCB import UCB1WithConstantOne, UCB1WithConstantT
from lib.bandit.ThompsonSampling import ThompsonSampling
from lib.bandit.ExploreThenExploit import ExploreThenExploit

from scipy.stats import bernoulli, beta

numCores = multiprocessing.cpu_count()


T = 5000
N = 150
K = 3


DEFAULT_COMMON_PRIOR = [beta(1, 1) for k in xrange(K)]


def sim(alg, banditPrior, banditProblemInstance):
  banditAlg = alg(banditProblemInstance, DEFAULT_COMMON_PRIOR)
  for t in xrange(T):
    banditAlg.executeStep(t)
  return banditAlg.rewardHistory



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


ALGS = [ThompsonSampling, UCB1WithConstantOne, UCB1WithConstantT, DynamicGreedy, DynamicEpsilonGreedy]
BANDIT_DISTR = {
  'Uniform': uniform_real_distr, 
  'Heavy Tail' :heavy_tailed,
  'Needle In Haystack High': needle_in_haystack_50_high
}


# Algorithm, Arms, Prior, t, n, reward

FIELDNAMES = ['Algorithm', 'K', 'Distribution', 't', 'Reward Mean', 'Reward Std', 'Best Arm Mean']

with open('results/preliminary_plots_3_arms_2.csv', 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
  writer.writeheader()
  for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
    if banditDistrName == 'Uniform':
      banditDistr = [bernoulli(random.uniform(0.25, 0.75)) for i in xrange(K)]
    else:
      banditDistr = [bernoulli(heavy_tail_prior.rvs()) for i in xrange(K)]
    realizations = {}
    for t in xrange(T):
      realizations[t] = {}
      for j in xrange(len(banditDistr)):
        realizations[t][j] = banditDistr[j].rvs()
    banditProblemInstance = BanditProblemInstance(K, T, banditDistr, realizations)
    for alg in ALGS:
      simResults = []
      for i in xrange(N):
        result = sim(alg, banditDistr, banditProblemInstance)
        simResults.append(result)
      averageTrajectory = []
      for t in xrange(T):
        cur = []
        for result in simResults:
          cur.append(result[t])
        res = {
          'Algorithm': alg.__name__,
          'K': str(K),
          'Distribution': banditDistrName,
          't': str(t),
          'Reward Mean': np.mean(cur),
          'Reward Std': np.std(cur)
        }
        writer.writerow(res)

