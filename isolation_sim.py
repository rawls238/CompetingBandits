import numpy as np
import csv
import random
import time
from joblib import Parallel, delayed


from copy import deepcopy

from lib.constants import DEFAULT_FREE_OBS_NUM
from lib.BanditProblemInstance import BanditProblemInstance

from lib.bandit.StaticGreedy import StaticGreedy
from lib.bandit.DynamicEpsilonGreedy import DynamicEpsilonGreedy
from lib.bandit.NonBayesianEpsilonGreedy import NonBayesianEpsilonGreedy
from lib.bandit.DynamicGreedy import DynamicGreedy
from lib.bandit.UCB import UCB1WithConstantOne, UCB1WithConstantT
from lib.bandit.ThompsonSampling import ThompsonSampling
from lib.bandit.ExploreThenExploit import ExploreThenExploit
from simulate import getRealDistributionsFromPrior

from scipy.stats import bernoulli, beta


T = 2001
N = 1000
K = 10
numCores = 10

'''
Currently, realizations are generated and used as follows:
We generate FREE_OBS + MAX_WS + T observations.
The first FREE_OBS observations are not used in the isolation sim (for now)
The next MAX_WS are reserved for the warm start. If a WS < MAX_WS is used then it simply takes WS realizations from this
The observations after FREE_OBS + MAX_WS are the realizations used in competition
'''



MAX_WARM_START_SIZE = 200 # the max warm start size
CUR_WARM_START = 20
numObs = MAX_WARM_START_SIZE + T + DEFAULT_FREE_OBS_NUM


DEFAULT_COMMON_PRIOR = [beta(1, 1) for k in xrange(K)]


def sim(alg, banditDistr, realizations, seed):
  seed = int(seed)
  np.random.seed(seed)

  warmStartRealizations = realizations[DEFAULT_FREE_OBS_NUM:CUR_WARM_START+DEFAULT_FREE_OBS_NUM]
  competitionRealizations = realizations[MAX_WARM_START_SIZE + DEFAULT_FREE_OBS_NUM:]
  totalRealizations = warmStartRealizations + competitionRealizations


  banditProblemInstance = BanditProblemInstance(K, banditDistr, totalRealizations)

  banditAlg = alg(banditProblemInstance, DEFAULT_COMMON_PRIOR)
  for t in xrange(T+CUR_WARM_START):
    banditAlg.executeStep(t)
  return (banditAlg.realizedRewardHistory, banditAlg.realizedCumulativeRewardHistory, banditAlg.meanRewardHistory, banditAlg.meanCumulativeRewardHistory, banditProblemInstance.getArmMeans(), banditProblemInstance.getComplexityMetric())

default_mean = 0.5
needle_in_haystack = [bernoulli(default_mean) for i in xrange(K)]

needle_in_haystack_50_high = deepcopy(needle_in_haystack)
needle_in_haystack_50_high[int(K/2)] = bernoulli(default_mean + 0.2)

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
  'Needle In Haystack': get_needle_in_haystack(0.5),
  '.5/.7 Random Draw': None
}


WORKING_DIRECTORY = ''
WORKING_DIRECTORY = '/rigel/home/ga2449/bandits-rl-project/'
# Algorithm, Arms, Prior, t, n, reward

RESULTS_DIR = WORKING_DIRECTORY + 'results/preliminary_raw_results/'


FILENAME = 'preliminary_plots_unified'
realizations_name = RESULTS_DIR + FILENAME + '_realizations.csv'
dist_name = RESULTS_DIR + FILENAME + '_dist.csv'

FIELDNAMES = ['Realized Complexity', 'n', 'True Mean Reputation', 'Realized Reputation', 'Algorithm', 'K', 'Distribution', 't', 'Instantaneous Realized Reward Mean', 'Instantaneous Mean Reward Mean', 'Arm Means']
simResults = {}

with open(RESULTS_DIR + FILENAME + '.csv', 'w') as csvfile:
  with open(realizations_name, 'w') as realiz:
    with open(dist_name, 'w') as dist:

      writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
      writer.writeheader()
      for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
        for a in xrange(len(ALGS)):
          simResults[ALGS[a]] = []
        realDistributions = {}
        realizations = {}

        dist_writer = csv.writer(dist)
        dist_writer.writerow(['Prior'] + [i for i in xrange(K)])

        realization_writer = csv.writer(realiz)
        realization_writer.writerow(['Prior', 't', 'n'] + [i for i in xrange(K)])

        for q in xrange(N):
          realDistributions[q] = getRealDistributionsFromPrior(banditDistrName, banditDistr, K)
          realizations[q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(numObs)]
          dist_writer.writerow([banditDistrName] + [realDistributions[q][j].mean() for j in xrange(len(realDistributions[q]))])
          realization_writer.writerows([[banditDistrName, k, q] + [z for z in realizations[q][k]] for k in xrange(numObs)])


        for a in xrange(len(ALGS)):
          alg = ALGS[a]
          simResults[alg] = Parallel(n_jobs=numCores)(delayed(sim)(alg, realDistributions[j], realizations[j], j+1) for j in xrange(N))
        for t in range(5, T+CUR_WARM_START, 5):
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
                'Realized Complexity': str(algResult[j][5])
              }
              writer.writerow(res)

print('all done!')
