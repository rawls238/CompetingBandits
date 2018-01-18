import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import csv


from copy import deepcopy


from lib.BanditProblemInstance import BanditProblemInstance

from lib.bandit.StaticGreedy import StaticGreedy
from lib.bandit.DynamicEpsilonGreedy import DynamicEpsilonGreedy
from lib.bandit.DynamicGreedy import DynamicGreedy
from lib.bandit.UCB import UCB
from lib.bandit.ThompsonSampling import ThompsonSampling
from lib.bandit.ExploreThenExploit import ExploreThenExploit
from lib.constants import DEFAULT_COMMON_PRIOR, uniform_real_distr

from scipy.stats import bernoulli, beta

numCores = multiprocessing.cpu_count()


T = 5000
N = 250
K = 10


def sim(alg, banditPrior):
  banditProblemInstance = BanditProblemInstance(K, T, banditPrior)
  banditAlg = alg(banditProblemInstance, DEFAULT_COMMON_PRIOR)
  for t in xrange(T):
    banditAlg.executeStep()
  return banditAlg.rewardHistory


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


ALGS = [ThompsonSampling, UCB, ExploreThenExploit, DynamicGreedy, DynamicEpsilonGreedy]
BANDIT_DISTR = {
  'Uniform': uniform_real_distr, 
  'Needle50 - High': needle_in_haystack_50_high, 
  'Needle50 - Medium': needle_in_haystack_50_medium, 
  'Heavy Tail' :heavy_tailed
}


# Algorithm, Arms, Prior, t, n, reward

FIELDNAMES = ['Algorithm', 'K', 'Distribution', 't', 'Reward Mean', 'Reward Std', 'Best Arm Mean']

with open('results/preliminary_plots_3.csv', 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
  writer.writeheader()
  for alg in ALGS:
    for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
      simResults = []
      for i in xrange(N):
        result = sim(alg, banditDistr)
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

