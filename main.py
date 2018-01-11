
# Import BanditAlgorithm classes
from StaticGreedy import StaticGreedy
from DynamicEpsilonGreedy import DynamicEpsilonGreedy
from DynamicGreedy import DynamicGreedy
from UCB import UCB
from ThompsonSampling import ThompsonSampling
from ExploreThenExploit import ExploreThenExploit
from simulate import simulate, getRealDistributionsFromPrior, COMMON_PRIOR
from constants import NUM_SIMULATIONS

## Import Agent classes
from HardMax import HardMax

# library imports
from scipy.stats import bernoulli, beta, uniform
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
import matplotlib.pyplot as plt
import pickle



# misspecified in the sense that the principal's initial belief (priors) is wrong compared to
# the real distribution
# MISSPECIFIED_PRIOR = [beta(0.45, 0.55) for k in xrange(K)]

# INITIAL_PRINCIPAL_PRIORS = [beta(0.55, 0.45) for k in xrange(K)]

#REAL_DISTRIBUTIONS = [bernoulli(0.45) for k in xrange(K)]
# now, overwrite the first distribution so that it's better than all the others
#REAL_DISTRIBUTIONS[5] = bernoulli(0.55)
#REAL_DISTRIBUTIONS[6] = bernoulli(0.65)


initialResultDict = {
  'marketShare1': [],
  'marketShare2': [],
  'armCounts1': [],
  'armCounts2': [],
  'avgRegret1': [],
  'avgRegret2': [],
  'principalHistory': []
}

numCores = multiprocessing.cpu_count()

AGENT_ALGS = [HardMax]

# valid principal algs are: [StaticGreedy, UCB, DynamicEpsilonGreedy, DynamicGreedy, ExploreThenExploit, ThompsonSampling]
PRINCIPAL1_ALGS = [ThompsonSampling]
PRINCIPAL2_ALGS = [ThompsonSampling]
MEMORY_SIZES = [100, 500]

results = {}
for agentAlg in AGENT_ALGS:
  results[agentAlg] = {}
  for principalAlg1 in PRINCIPAL1_ALGS:
    results[agentAlg][principalAlg1] = {}
    for principalAlg2 in PRINCIPAL2_ALGS:
      results[agentAlg][principalAlg1][principalAlg2] = {}
      for memory in MEMORY_SIZES:
        results[agentAlg][principalAlg1][principalAlg2][memory] = deepcopy(initialResultDict)
        realDistributions = {}
        for i in xrange(NUM_SIMULATIONS):
          realDistributions[i] = getRealDistributionsFromPrior(COMMON_PRIOR)
        print('Running ' + agentAlg.__name__ + ' and principal 1 playing ' + principalAlg1.__name__ + ' and principal 2 playing ' + principalAlg2.__name__ + ' with memory ' + str(memory))
        simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg1, principalAlg2, agentAlg, memory=memory, realDistributions=realDistributions[i]) for i in xrange(NUM_SIMULATIONS))
        for res in simResults:
          for k, v in res.iteritems():
            results[agentAlg][principalAlg1][principalAlg2][memory][k].append(deepcopy(v))

        regrets1 = [x for x in results[agentAlg][principalAlg1][principalAlg2][memory]['avgRegret1'] if x is not None]
        regrets2 = [x for x in results[agentAlg][principalAlg1][principalAlg2][memory]['avgRegret2'] if x is not None]
        print({
          'averageRegret1': np.mean(regrets1),
          'averageRegret2': np.mean(regrets2),
          'stdRegret1': np.std(regrets1),
          'stdRegret2': np.std(regrets2),
          'averageDeltaRegret': np.mean([np.abs(regrets1[i] - regrets2[i]) for i in xrange(len(regrets1))]),
          'averageMarketShare1': np.mean(results[agentAlg][principalAlg1][principalAlg2][memory]['marketShare1']),
          'averageMarketShare2': np.mean(results[agentAlg][principalAlg1][principalAlg2][memory]['marketShare2'])
        })

# save "results" to disk, just for convenience, so i can look at them later
pickle.dump(results, open("bandit_simulations.p", "wb" ))
# later, you can load this by doing: results = pickle.load( open("bandit_simulations.p", "rb" ))
