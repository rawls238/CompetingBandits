
# Import BanditAlgorithm classes
from StaticGreedy import StaticGreedy
from DynamicEpsilonGreedy import DynamicEpsilonGreedy
from DynamicGreedy import DynamicGreedy
from UCB import UCB
from ThompsonSampling import ThompsonSampling
from ExploreThenExploit import ExploreThenExploit

## Import Agent classes
from Uniform import Uniform
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
import pickle


K = 10
T = 5000.0

COMMON_PRIOR = [beta(5, 5) for k in xrange(K)]
PRINCIPAL1PRIORS = [beta(5, 5) for k in xrange(K)]
PRINCIPAL2PRIORS = PRINCIPAL1PRIORS

REAL_DISTRIBUTIONS = [bernoulli(COMMON_PRIOR[i].rvs()) for i in xrange(K)]


# generate real distributions from 
def getRealDistributionsFromPrior(prior):
  return [bernoulli(prior[i].rvs()) for i in xrange(len(prior))]

# realDistributions - the true distribution of the arms
# principalPriors - the priors the principals have over the arms
# agentPriors - the priors the agents have for the distribution of rewards from each principal
def simulate(principalAlg1, principalAlg2, agentAlg,
  memory=50,
  alpha=10,
  realDistributions=deepcopy(REAL_DISTRIBUTIONS),
  principal1Priors=deepcopy(PRINCIPAL1PRIORS),
  principal2Priors=deepcopy(PRINCIPAL2PRIORS),
  agentPriors={ 'principal1': beta(0.5, 0.5), 'principal2': beta(0.5, 0.5) }):

  banditProblemInstance = BanditProblemInstance(K, T, realDistributions)
  bestArmMean = banditProblemInstance.bestArmMean()

  # instantiate 2 principals (who are of some subclass of BanditAlgorithm)
  principal1 = principalAlg1(banditProblemInstance, principal1Priors)
  principal2 = principalAlg2(banditProblemInstance, principal2Priors)

  principals = { 'principal1': principal1, 'principal2': principal2 }
  agents = agentAlg(principals, K, agentPriors, memory=memory)

  principalHistory = []
  for t in xrange(int(T)):
    (principalName, principal) = agents.selectPrincipal()
    principalHistory.append(principalName)
    (reward, arm) = principal.executeStep()
    trueMeanOfArm = banditProblemInstance.getMeanOfArm(arm)
    principal.regret += (bestArmMean - trueMeanOfArm)
    agents.updateInformationSet(reward, arm, principalName)

  marketShare1 = principal1.n / T
  marketShare2 = principal2.n / T
  return {
    'marketShare1' : marketShare1,
    'marketShare2' : marketShare2,
    'armCounts1' : principal1.armCounts,
    'armCounts2' : principal2.armCounts,
    'avgRegret1': principal1.getAverageRegret(),
    'avgRegret2': principal2.getAverageRegret(),
    # 'principalHistory': principalHistory,
  }


initialResultDict = {
  'marketShare1': [],
  'marketShare2': [],
  'armCounts1': [],
  'armCounts2': [],
  'avgRegret1': [],
  'avgRegret2': [],
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

# AGENT_ALGS = [HardMax, SoftMax, HardMaxWithRandom, SoftMaxWithRandom]
AGENT_ALGS = [HardMax]

#greedy algorithms: [StaticGreedy, DynamicGreedy]
#non-adaptive exploration: [DynamicEpsilonGreedy, ExploreThenExploit]
#adaptive exploration: [UCB, ThompsonSampling]

# valid principal algs are: [StaticGreedy, UCB, DynamicEpsilonGreedy, DynamicGreedy, ExploreThenExploit, ThompsonSampling]
PRINCIPAL1_ALGS = [ThompsonSampling]
PRINCIPAL2_ALGS = [DynamicEpsilonGreedy]


results = {}
for agentAlg in AGENT_ALGS:
  results[agentAlg] = {}
  for principalAlg1 in PRINCIPAL1_ALGS:
    results[agentAlg][principalAlg1] = {}
    for principalAlg2 in PRINCIPAL2_ALGS:
      results[agentAlg][principalAlg1][principalAlg2] = {}
      realDistributions = {}
      for i in xrange(N):
        realDistributions[i] = getRealDistributionsFromPrior(COMMON_PRIOR)
      results[agentAlg][principalAlg1][principalAlg2] = deepcopy(initialResultDict)
      print('Running ' + agentAlg.__name__ + ' and principal 1 playing ' + principalAlg1.__name__ + ' and principal 2 playing ' + principalAlg2.__name__)
      simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg1, principalAlg2, agentAlg, realDistributions=realDistributions[i]) for i in xrange(N))
      for res in simResults:
        for k, v in res.iteritems():
          results[agentAlg][principalAlg1][principalAlg2][k].append(deepcopy(v))

      regrets1 = [x for x in results[agentAlg][principalAlg1][principalAlg2]['avgRegret1'] if x is not None]
      regrets2 = [x for x in results[agentAlg][principalAlg1][principalAlg2]['avgRegret2'] if x is not None]

      print({
        # commented this out because now we have K>2 arms
        # 'averageArm0Counts1': np.mean([l[0] for l in results[agentAlg][principalAlg1][principalAlg2]['armCounts1']]),
        # 'averageArm1Counts1': np.mean([l[1] for l in results[agentAlg][principalAlg1][principalAlg2]['armCounts1']]),
        # 'averageArm0Counts2': np.mean([l[0] for l in results[agentAlg][principalAlg1][principalAlg2]['armCounts2']]),
        # 'averageArm1Counts2': np.mean([l[1] for l in results[agentAlg][principalAlg1][principalAlg2]['armCounts2']]),
        'averageRegret1': np.mean(regrets1),
        'averageRegret2': np.mean(regrets2),
        'averageMarketShare1': np.mean(results[agentAlg][principalAlg1][principalAlg2]['marketShare1']),
        'averageMarketShare2': np.mean(results[agentAlg][principalAlg1][principalAlg2]['marketShare2'])
      })

# save "results" to disk, just for convenience, so i can look at them later
pickle.dump(results, open("bandit_simulations.p", "wb" ))
# later, you can load this by doing: results = pickle.load( open("bandit_simulations.p", "rb" ))


