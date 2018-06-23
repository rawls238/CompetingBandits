import numpy as np
import csv
import random
import time
from joblib import Parallel, delayed


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


T = 5001
N = 750
K = 3
numCores = 12


DEFAULT_COMMON_PRIOR = [beta(1, 1) for k in xrange(K)]


def sim(alg, banditDistr, realizations, seed):
  seed = int(time.time() * float(seed)) % 2**32
  np.random.seed(seed)
  banditProblemInstance = BanditProblemInstance(K, T, banditDistr, realizations)
  banditAlg = alg(banditProblemInstance, DEFAULT_COMMON_PRIOR)
  for t in xrange(T):
    banditAlg.executeStep(t)
  return (banditAlg.realizedRewardHistory, banditAlg.realizedCumulativeRewardHistory, banditAlg.meanRewardHistory, banditAlg.meanCumulativeRewardHistory, banditProblemInstance.getArmMeans(), banditProblemInstance.getComplexityMetric())

default_mean = 0.5
needle_in_haystack = [bernoulli(default_mean) for i in xrange(K)]

needle_in_haystack_50_high = deepcopy(needle_in_haystack)
needle_in_haystack_50_high[int(K/2)] = bernoulli(default_mean + 0.2)

#needle_in_haystack_50_one_medium_one_high = deepcopy(needle_in_haystack)
#needle_in_haystack_50_one_medium_one_high[int(K/2)] = bernoulli(default_mean + 0.3)
#needle_in_haystack_50_one_medium_one_high[int(K/2) + 1] = bernoulli(default_mean + 0.1)

heavy_tail_prior = beta(0.6, 0.6)
heavy_tailed = [bernoulli(heavy_tail_prior.rvs()) for i in xrange(K)]

def get_needle_in_haystack(starting_mean):
  needle_in_haystack = [bernoulli(starting_mean) for i in xrange(K)]
  needle_in_haystack[int(K/2)] = bernoulli(starting_mean + 0.2)
  return needle_in_haystack

ALGS = [DynamicGreedy, ThompsonSampling, DynamicEpsilonGreedy]
BANDIT_DISTR = {
  'Heavy Tail': heavy_tail_prior,
  'Uniform': None,
  '.5/.7 Random Draw': None,
  'Needle In Haystack - 0.5': get_needle_in_haystack(0.5)
}


WORKING_DIRECTORY = ''
WORKING_DIRECTORY = '/rigel/home/ga2449/bandits-rl-project/'
# Algorithm, Arms, Prior, t, n, reward

RESULTS_DIR = WORKING_DIRECTORY + 'results/preliminary_raw_results/'
FILENAME = 'preliminary_plots_full_path.csv'


FIELDNAMES = ['Realized Complexity', 'n', 'True Mean Reputation', 'Realized Reputation', 'Algorithm', 'K', 'Distribution', 't', 'Instantaneous Realized Reward Mean', 'Instantaneous Mean Reward Mean', 'Arm Means']
simResults = {}

with open(RESULTS_DIR + FILENAME, 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
  writer.writeheader()
  for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
    realizations = {}
    banditDistrs = {}
    for a in xrange(len(ALGS)):
      simResults[ALGS[a]] = []

    for i in xrange(N):
      if banditDistrName == 'Uniform':
        banditDistrs[i] = [bernoulli(np.random.uniform(0.25, 0.75)) for j in xrange(K)]
      elif banditDistrName == 'Heavy Tail':
        banditDistrs[i] = [bernoulli(heavy_tail_prior.rvs()) for j in xrange(K)]
      elif banditDistrName == '.5/.7 Random Draw':
        banditDistrs[i] = [bernoulli(np.random.choice([0.5, 0.7])) for j in xrange(K)]
      else:
        banditDistrs[i] = banditDistr
      
      realizations[i] = {}
      for t in xrange(T):
        realizations[i][t] = {}
        for j in xrange(len(banditDistrs[i])):
          realizations[i][t][j] = banditDistrs[i][j].rvs()

    for a in xrange(len(ALGS)):
      alg = ALGS[a]
      simResults[alg] = Parallel(n_jobs=numCores)(delayed(sim)(alg, banditDistrs[j], realizations[j], j) for j in xrange(N))
    for t in xrange(T-1001, T):
      for (alg, algResult) in simResults.iteritems():
        name = alg.__name__
        for j in xrange(len(algResult)):
          realized_reputation = np.mean(algResult[j][0][max(0,t-100):t])
          true_reputation = np.mean(algResult[j][2][max(0,t-100):t])
          instantaneous_realized = algResult[j][0][t]
          instantaneous_mean = algResult[j][2][t]
          res = {
            'Algorithm': name,
            'K': str(K),
            'n': str(j),
            'Distribution': banditDistrName,
            't': str(t),
            'Instantaneous Realized Reward Mean': instantaneous_realized,
            'Instantaneous Mean Reward Mean': instantaneous_mean,
            'Realized Reputation': realized_reputation,
            'True Mean Reputation': true_reputation,
            'Arm Means': str(algResult[j][3]),
            'Realized Complexity': str(algResult[j][4])
          }
          writer.writerow(res)

print('all done!')
