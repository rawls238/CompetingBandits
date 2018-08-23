import numpy as np
import time

from lib.constants import DEFAULT_MEMORY, DEFAULT_DISCOUNT_FACTOR, DEFAULT_ALPHA, DEFAULT_WARM_START_NUM_OBSERVATIONS, RECORD_STATS_AT, DEFAULT_FREE_OBS_NUM
from lib.BanditProblemInstance import BanditProblemInstance

import random
from scipy.stats import bernoulli, beta

def getDefaultPrior(K):
  return [beta(1, 1) for k in xrange(K)]

def getDefaultRealDistributions(K):
  prior = getDefaultPrior(K)
  return [bernoulli(prior[i].rvs()) for i in xrange(K)]

def getRealDistributionsFromPrior(priorName, prior, K, targetComplexityVal=None):
  if priorName == 'Uniform':
    return [bernoulli(random.uniform(0.25, 0.75)) for i in xrange(K)]
  elif priorName == 'Heavy Tail':
    return [bernoulli(prior.rvs()) for i in xrange(K)]
  elif priorName == '.5/.7 Random Draw':
    return [bernoulli(random.choice([0.5, 0.7])) for i in xrange(K)]
  elif priorName == 'Complexity':
    a = [np.random.rand()% 0.5 for i in xrange(int(K/2))]
    b = [(np.random.rand()% 0.5) + 0.5 for i in xrange(int(K/2),K)]
    return a + b
  elif priorName == 'FixedComplexity':
    if targetComplexityVal is None:
      print("Provide target complexity val")
      return
    return genInstanceForComplexityMetric(targetComplexityVal, K)
  return prior

def complexityMetric(means):
  bestArm = max(means)
  metric = 0.0
  for mean in means:
    if mean < bestArm:
      metric += 1.0 / (bestArm - mean)
  return metric

def gen_rand_instance(K):
  a = [np.random.rand()% 0.5 for i in xrange(int(K/2))]
  b = [(np.random.rand()% 0.5) + 0.5 for i in xrange(int(K/2),K)]
  return a + b

def genInstanceForComplexityMetric(targetVal, K):
  while True:
    instance = gen_rand_instance(K)
    complexity = complexityMetric(instance)
    if np.abs(complexity - targetVal) <= 5:
      return [bernoulli(i) for i in instance]

"""
Parameters
  ----------
  principalAlg1 / principalAlg2 : BanditAlgorithm object
    The algorithm played by principal 1 and 2, respectively
  agentAlg : Agent object
    The decision rule utilized by the agents
  K : Int
    The number of arms in the bandit instance
  T : Int
    The time horizon for the competition game
  discountFactor: Float
    Discount factor used in reputation score calculation
  realDistributions: Distributions vector
    The distributions of the K arms
  realizations:
    Realizations for each arm for T+WARM_START + FREE_OBS rounds
  freeObsForP2: Boolean
    Used for the incumbent experiment - does principal 2 (the incumbent) get free agents before the competition game?
  freeObsNum: Int
    For how many rounds is principal 2 the incumbent?
  eraseReputation: Boolean
    Whether to erase the reputation gain of the incumbent
  eraseInformation: Boolean
    Whether to erase the information gain of the incumbent
  principal1Priors / principal2Priors: Vector of Distributions
    Initial beliefs of the principals
  recordStatsAt: Vector of times
    We record the state of the game at these times
  seed: Float
    Used to reinitialize the seed for the random draws - important when using parallelization
  eraseReputation: Boolean
    A boolen flag on whether to erase reputation or not
"""
def simulate(principalAlg1, principalAlg2, agentAlg, maxWarmStart, K, T,
  discountFactor=DEFAULT_DISCOUNT_FACTOR,
  realDistributions=None,
  realizations=None,
  freeObsForP2=False,
  freeObsNum=DEFAULT_FREE_OBS_NUM,
  warmStartNumObservations=DEFAULT_WARM_START_NUM_OBSERVATIONS,
  principal1Priors=None,
  principal2Priors=None,
  eraseReputation=False,
  eraseInformation=False,
  recordStatsAt=RECORD_STATS_AT,
  seed=1.0
):
  seed = int(seed)
  np.random.seed(seed)
  if principal1Priors is None:
    principal1Priors = getDefaultPrior(K)
  if principal2Priors is None:
    principal2Priors = getDefaultPrior(K)
  if realDistributions is None:
    realDistributions = getDefaultRealDistributions(K)

  freeObsRealizations = realizations[:freeObsNum]
  warmStartRealizations = realizations[freeObsNum:freeObsNum + warmStartNumObservations]
  competitionRealizations = realizations[freeObsNum+maxWarmStart:]

  banditProblemInstance = BanditProblemInstance(K, realDistributions, freeObsRealizations)

  # instantiate 2 principals (who are of some subclass of BanditAlgorithm)
  principal1 = principalAlg1(banditProblemInstance, principal1Priors)
  principal2 = principalAlg2(banditProblemInstance, principal2Priors)

  principals = { 'principal1': principal1, 'principal2': principal2 }
  agents = agentAlg(principals, K)

  # first give the "incumbent" free observations if this is the incumbent experiment
  if freeObsForP2:
    for j in range(freeObsNum):
      (reward, arm) = principals['principal2'].executeStep(j)
      agents.updateInformationSet(reward, arm, 'principal2')


  # now, reset the realizations to the warm start realizations so they're the same across firms and give each firm the warm start agents
  for principal in principals.values():
    principal.resetStats()
    if eraseInformation:
      principal.resetPriors()
    principal.setRealizations(warmStartRealizations)

  if eraseReputation:
    agents.resetInformationSet()

  for i in xrange(warmStartNumObservations):
    for (principalName, principal) in principals.iteritems():
      (reward, arm) = principal.executeStep(i)
      agents.updateInformationSet(reward, arm, principalName)

  # Now, reset the bandit instance realizations to the T bandit instance realizations and run the competition game

  for principal in principals.values():
    principal.resetStats()
    principal.setRealizations(competitionRealizations)

  instanceComplexity = banditProblemInstance.getComplexityMetric()
  results = []
  lastPrincipalPicked = None
  effectiveEndOfGame = None
  for t in xrange(int(T)):
    if t in recordStatsAt:
      reputation = agents.getScores()
      results.append({
        'marketShare1' : principal1.n / float(t),
        'marketShare2' : principal2.n / float(t),
        'armCounts1' : principal1.armCounts,
        'armCounts2' : principal2.armCounts,
        'avgRegret1': principal1.getAverageRegret(),
        'avgRegret2': principal2.getAverageRegret(),
        'reputation1': reputation['principal1'],
        'reputation2': reputation['principal2'],
        'time': t,
        'complexity': instanceComplexity
      })
    (principalName, principal) = agents.selectPrincipal()
    if lastPrincipalPicked != principalName:
      lastPrincipalPicked = principalName
      effectiveEndOfGame = t
    (reward, arm) = principal.executeStep(t)
    agents.updateInformationSet(reward, arm, principalName)

  for i in xrange(len(results)):
    results[i]['effectiveEndOfGame'] = effectiveEndOfGame  
  return results


initialResultDict = {
  'marketShare1': [],
  'marketShare2': [],
  'armCounts1': [],
  'armCounts2': [],
  'avgRegret1': [],
  'avgRegret2': [],
  'reputation1': [],
  'reputation2': [],
  'effectiveEndOfGame': [],
  'principalHistory': [],
  'time': []
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
