import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import csv


from copy import deepcopy


from BanditProblemInstance import BanditProblemInstance

from StaticGreedy import StaticGreedy
from DynamicEpsilonGreedy import DynamicEpsilonGreedy
from DynamicGreedy import DynamicGreedy
from UCB import UCB
from ThompsonSampling import ThompsonSampling
from ExploreThenExploit import ExploreThenExploit

from constants import DEFAULT_COMMON_PRIOR, uniform_real_distr
from scipy.stats import bernoulli, beta

numCores = multiprocessing.cpu_count()


T = 5000
N = 50
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


default_mean = 0.01
needle_in_haystack = [bernoulli(default_mean) for i in xrange(K)]
needle_in_haystack_01_high = deepcopy(needle_in_haystack)
needle_in_haystack_01_high[int(K/2)] = bernoulli(default_mean + 0.2)

needle_in_haystack_01_medium = deepcopy(needle_in_haystack)
needle_in_haystack_01_medium[int(K/2)] = bernoulli(default_mean + 0.05)

needle_in_haystack_01_low = deepcopy(needle_in_haystack)
needle_in_haystack_01_low[int(K/2)] = bernoulli(default_mean + 0.01)


default_mean = 0.80
needle_in_haystack = [bernoulli(default_mean) for i in xrange(K)]
needle_in_haystack_80_high = deepcopy(needle_in_haystack)
needle_in_haystack_80_high[int(K/2)] = bernoulli(default_mean + 0.2)

needle_in_haystack_80_medium = deepcopy(needle_in_haystack)
needle_in_haystack_80_medium[int(K/2)] = bernoulli(default_mean + 0.05)

needle_in_haystack_80_low = deepcopy(needle_in_haystack)
needle_in_haystack_80_low[int(K/2)] = bernoulli(default_mean + 0.01)

heavy_tail_prior = beta(0.6, 0.6)
heavy_tailed = [bernoulli(heavy_tail_prior.rvs()) for i in xrange(K)]


ALGS = [ThompsonSampling, UCB, ExploreThenExploit, DynamicGreedy, DynamicEpsilonGreedy]
BANDIT_DISTR = {
  'Uniform': uniform_real_distr, 
  'Needle50 - High': needle_in_haystack_50_high, 
  'Needle50 - Medium': needle_in_haystack_50_medium, 
  'Needle50 - Low': needle_in_haystack_50_low,

  'Needle01 - High': needle_in_haystack_01_high, 
  'Needle01 - Medium': needle_in_haystack_01_medium, 
  'Needle01 - Low': needle_in_haystack_01_low,

  'Needle80 - High': needle_in_haystack_80_high, 
  'Needle80 - Medium': needle_in_haystack_80_medium, 
  'Needle80 - Low': needle_in_haystack_80_low,

  'Heavy Tail' :heavy_tailed
}


# Algorithm, Arms, Prior, t, n, reward

FIELDNAMES = ['Algorithm', 'K', 'Distribution', 't', 'Reward Mean', 'Reward Std', 'Best Arm Mean']

with open('results/preliminary_plots.csv', 'w') as csvfile:
  for alg in ALGS:
    for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
      writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
      writer.writeheader()
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
      """      print(averageTrajectory)
      plt.plot(averageTrajectory)
      plt.ylabel('Average Reward')
      plt.xlabel('Time')
      plt.title(alg.__name__)
      plt.show()"""

