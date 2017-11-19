
# Import BanditAlgorithm classes
from DynamicEpsilonGreedy import DynamicEpsilonGreedy
from DynamicGreedy import DynamicGreedy
from UCB import UCB
from ThompsonSampling import ThompsonSampling

## Import Agent classes
from HardMax import HardMax
from HardMaxWithRandom import HardMaxWithRandom
from SoftMax import SoftMax
from SoftMaxWithRandom import SoftMaxWithRandom
from HeterogenousHardMax import HeterogenousHardMax


from BanditProblemInstance import BanditProblemInstance

# library imports
from scipy.stats import bernoulli, beta, uniform
import numpy as np
from copy import copy
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
import matplotlib.pyplot as plt

K = 2
T = 1000.0

def simulate(principalAlg1, principalAlg2, agentAlg, 
  realDistributions=[bernoulli(0.6), bernoulli(0.4)], 
  principalPriors=[beta(0.1, 0.9), beta(0.9, 0.1)],
  agentPriors={ 'principal1': beta(0.5, 0.5), 'principal2': beta(0.6, 0.4) }):
  # true distributions are:
  # arm 1 ~ bernoulli(0.6)   mu_1 = 0.6
  # arm 2 ~ bernoulli(0.4)   mu_2 = 0.4

  banditProblemInstance = BanditProblemInstance(K, T, realDistributions)


  # instantiate 2 principals (who are of some subclass of BanditAlgorithm)
  principal1 = principalAlg1(banditProblemInstance, principalPriors)
  principal2 = principalAlg2(banditProblemInstance, principalPriors)

  principals = { 'principal1': principal1, 'principal2': principal2 }
  agents = agentAlg(principals, agentPriors)

  principalHistory = []
  for t in xrange(int(T)):
    (principalName, principal) = agents.selectPrincipal()
    principalHistory.append(principalName)
    (reward, arm) = principal.executeStep()
    agents.updateInformationSet(reward, arm, principalName)

  marketShare1 = principal1.n / T
  marketShare2 = principal2.n / T
  return {
    'marketShare1' : marketShare1,
    'marketShare2' : marketShare2,
    'armCounts1' : principal1.armCounts,
    'armCounts2' : principal2.armCounts,
    'principalHistory': principalHistory,
  }


initialResultDict = {
  'marketShare1': [],
  'marketShare2': [],
  'armCounts1': [],
  'armCounts2': [],
  'armProbs1': [],
  'armProbs2': [],
  'principalHistory': []
}


def marketShareOverTime(armHistories, T):
  T = int(T)
  principal1msOverTime = [0.0 for i in xrange(T-1)]
  numArmHistories = len(armHistories)
  for i in xrange(1,T):
    for armHistory in armHistories:
      principal1msOverTime[i-1] += Counter(armHistory[:i])['principal1']

    #average over arm histories and then divide by number of global rounds to get market share
    principal1msOverTime[i-1] = (principal1msOverTime[i-1] / numArmHistories) / i
  return principal1msOverTime

N = 25
numCores = multiprocessing.cpu_count()
PRINCIPAL_ALGS = [ThompsonSampling, DynamicGreedy]
AGENT_ALGS = [HeterogenousHardMax, HardMax, SoftMax]
results = {}
for agentAlg in AGENT_ALGS:
  results[agentAlg] = {}
  for principalAlg in PRINCIPAL_ALGS:
    results[agentAlg][principalAlg] = {}
    for principalAlg2 in PRINCIPAL_ALGS:
      results[agentAlg][principalAlg][principalAlg2] = copy(initialResultDict)
      print('Running ' + agentAlg.__name__ + ' with principal 1 playing ' + principalAlg.__name__ + ' and principal 2 playing ' + principalAlg2.__name__)
      simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg, principalAlg2, agentAlg) for i in xrange(N))
      for res in simResults:
        for k, v in res.iteritems():
          results[agentAlg][principalAlg][principalAlg2][k].append(v)
      print({
        'averageMarketShare1': np.mean(results[agentAlg][principalAlg][principalAlg2]['marketShare1']),
        'averageMarketShare2': np.mean(results[agentAlg][principalAlg][principalAlg2]['marketShare2'])
      })

for agentAlg in AGENT_ALGS:
  for principalAlg in PRINCIPAL_ALGS:
    for principalAlg2 in PRINCIPAL_ALGS:
      ms = marketShareOverTime(results[agentAlg][principalAlg][principalAlg2]['principalHistory'], T)
      plt.plot(ms)
      plt.show()
