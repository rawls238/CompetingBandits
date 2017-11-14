
# Import BanditAlgorithm classes
from StaticGreedy import StaticGreedy
from DynamicGreedy import DynamicGreedy
from UCB import UCB

## Import Agent classes
from HardMax import HardMax
from HardMaxWithRandom import HardMaxWithRandom
from SoftMax import SoftMax
from SoftMaxWithRandom import SoftMaxWithRandom


from BanditProblemInstance import BanditProblemInstance

# library imports
from scipy.stats import bernoulli, beta, uniform
import numpy as np
from copy import copy
from joblib import Parallel, delayed
import multiprocessing

K = 2
T = 1000.0

def simulate(principalAlg, agentAlg):
  # true distributions are:
  # arm 1 ~ bernoulli(0.6)   mu_1 = 0.6
  # arm 2 ~ bernoulli(0.4)   mu_2 = 0.4
  realDistributions = [bernoulli(0.6), bernoulli(0.4)]
  #real_distributions = [bernoulli(), bernoulli(P_mean[1].rvs())]

  # Bandit algorithms are given priors:
  # arm 1 ~ bernoulli(0.5)
  # arm 2 ~ bernoulli(0.3)
  principalPriors = [beta(0.5, 0.5), beta(0.3, 0.7)]

  banditProblemInstance = BanditProblemInstance(K, T, realDistributions)

  agentPriors = { 'principal1': beta(0.45, 0.55), 'principal2': beta(0.45, 0.55) }

  # instantiate 2 principals (who are of some subclass of BanditAlgorithm)
  principal1 = principalAlg(banditProblemInstance, principalPriors)
  principal2 = principalAlg(banditProblemInstance, principalPriors)

  principals = { 'principal1': principal1, 'principal2': principal2 }
  agents = agentAlg(principals, agentPriors)

  for t in xrange(int(T)):
    (principalName, principal) = agents.selectPrincipal()
    (reward, arm) = principal.executeStep()
    agents.updateInformationSet(reward, principalName)

  marketShare1 = principal1.n / T
  marketShare2 = principal2.n / T
  return {
    'marketShare1' : marketShare1,
    'marketShare2' : marketShare2,
    'armCounts1' : principal1.armCounts,
    'armCounts2' : principal2.armCounts
  }


initialResultDict = {
  'marketShare1': [],
  'marketShare2': [],
  'armCounts1': [],
  'armCounts2': [],
  'armProbs1': [],
  'armProbs2': []
}

N = 60
numCores = multiprocessing.cpu_count()
PRINCIPAL_ALGS = [StaticGreedy, DynamicGreedy, UCB]
AGENT_ALGS = [HardMax, HardMaxWithRandom, SoftMax, SoftMaxWithRandom]
results = {}
for agentAlg in AGENT_ALGS:
  results[agentAlg] = {}
  for principalAlg in PRINCIPAL_ALGS:
    results[agentAlg][principalAlg] = copy(initialResultDict)
    print('Running ' + agentAlg.__name__ + ' with principal playing ' + principalAlg.__name__)
    simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg, agentAlg) for i in xrange(N))
    for res in simResults:
      for k, v in res.iteritems():
        results[agentAlg][principalAlg][k].append(v)
    print({
      'averageMarketShare1': np.mean(results[agentAlg][principalAlg]['marketShare1']),
      'averageMarketShare2': np.mean(results[agentAlg][principalAlg]['marketShare2'])
    })


