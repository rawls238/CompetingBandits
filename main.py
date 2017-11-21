
# Import BanditAlgorithm classes
from StaticGreedy import StaticGreedy
from DynamicEpsilonGreedy import DynamicEpsilonGreedy
from DynamicGreedy import DynamicGreedy
from UCB import UCB
from ThompsonSampling import ThompsonSampling
from ExploreThenExploit import ExploreThenExploit

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
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
import matplotlib.pyplot as plt

K = 2
T = 1000.0
MISSPECIFIED_PRIOR = [beta(0.45, 0.54), beta(0.5, 0.5)]
INITIAL_PRINCIPAL_PRIORS = [beta(0.55, 0.45), beta(0.55, 0.45)]


# realDistributions - the true distribution of the arms
# principalPriors - the priors the principals have over the arms
# agentPriors - the priors the agents have for the distribution of rewards from each principal

def simulate(principalAlg1, principalAlg2, agentAlg, 
  realDistributions=[bernoulli(0.55), bernoulli(0.45)], 
  principalPriors=MISSPECIFIED_PRIOR,
  agentPriors={ 'principal1': beta(0.6, 0.4), 'principal2': beta(0.6, 0.4) }):

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
  numArmHistories = float(len(armHistories))
  for i in xrange(1,T):
    for armHistory in armHistories:
      principal1msOverTime[i-1] += Counter(armHistory[:i])['principal1']

    #average over arm histories and then divide by number of global rounds to get market share
    principal1msOverTime[i-1] = (principal1msOverTime[i-1] / numArmHistories) / i
  return principal1msOverTime

N = 25
numCores = multiprocessing.cpu_count()
PRINCIPAL_ALGS = [StaticGreedy, UCB, DynamicEpsilonGreedy, DynamicGreedy, ExploreThenExploit]
AGENT_ALGS = [HardMax, SoftMax, HardMaxWithRandom, SoftMaxWithRandom]
results = {}
for agentAlg in AGENT_ALGS:
  results[agentAlg] = {}
  for principalAlg in PRINCIPAL_ALGS:
    results[agentAlg][principalAlg] = {}
    for principalAlg2 in PRINCIPAL_ALGS:
      results[agentAlg][principalAlg][principalAlg2] = deepcopy(initialResultDict)
      print('Running ' + agentAlg.__name__ + ' with principal 1 playing ' + principalAlg.__name__ + ' and principal 2 playing ' + principalAlg2.__name__)
      simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg, principalAlg2, agentAlg) for i in xrange(N))
      for res in simResults:
        for k, v in res.iteritems():
          results[agentAlg][principalAlg][principalAlg2][k].append(deepcopy(v))
      print({
        'averageArm0Counts1': np.mean([l[1] for l in results[agentAlg][principalAlg][principalAlg2]['armCounts1']]),
        'averageArm1Counts1': np.mean([l[0] for l in results[agentAlg][principalAlg][principalAlg2]['armCounts1']]),
        'averageArm0Counts2': np.mean([l[1] for l in results[agentAlg][principalAlg][principalAlg2]['armCounts2']]),
        'averageArm1Counts2': np.mean([l[0] for l in results[agentAlg][principalAlg][principalAlg2]['armCounts2']]),
        'averageMarketShare1': np.mean(results[agentAlg][principalAlg][principalAlg2]['marketShare1']),
        'averageMarketShare2': np.mean(results[agentAlg][principalAlg][principalAlg2]['marketShare2'])
      })

i = 0
rows = len(AGENT_ALGS)
cols = len(PRINCIPAL_ALGS) * len(PRINCIPAL_ALGS)
f, axarr = plt.subplots(rows, cols)
for agentAlg in AGENT_ALGS:
  j = 0
  for principalAlg in PRINCIPAL_ALGS:
    for principalAlg2 in PRINCIPAL_ALGS:
      ms = marketShareOverTime(results[agentAlg][principalAlg][principalAlg2]['principalHistory'], T)
      axarr[i, j].plot(ms)
      axarr[i, j].set_title(principalAlg.shorthand() + ' vs ' + principalAlg2.shorthand())
      j += 1
  i += 1
plt.show()
