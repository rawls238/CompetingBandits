import numpy as np

from lib.constants import DEFAULT_MEMORY, DEFAULT_DISCOUNT_FACTOR, DEFAULT_ALPHA, DEFAULT_WARM_START_NUM_OBSERVATIONS, RECORD_STATS_AT
from lib.BanditProblemInstance import BanditProblemInstance

import random
from scipy.stats import bernoulli, beta

def getDefaultPrior(K):
  return [beta(1, 1) for k in xrange(K)]

def getDefaultRealDistributions(K):
  prior = getDefaultPrior(K)
  return [bernoulli(prior[i].rvs()) for i in xrange(K)]

def getRealDistributionsFromPrior(priorName, prior, K):
  if priorName == 'Uniform':
    return [bernoulli(random.uniform(0.25, 0.75)) for i in xrange(K)]
  elif priorName == 'Heavy Tail':
    return [bernoulli(prior.rvs()) for i in xrange(K)]
  return prior

# realDistributions - the true distribution of the arms
# principalPriors - the priors the principals have over the arms
# agentPriors - the priors the agents have for the distribution of rewards from each principal
def simulate(principalAlg1, principalAlg2, agentAlg, K, T,
  memory=DEFAULT_MEMORY,
  alpha=DEFAULT_ALPHA,
  warmStartNumObservations=DEFAULT_WARM_START_NUM_OBSERVATIONS,
  discountFactor=DEFAULT_DISCOUNT_FACTOR,
  realDistributions=None,
  principal1Priors=None,
  principal2Priors=None,
  realizations=None):

  if principal1Priors is None:
    principal1Priors = getDefaultPrior(K)
  if principal2Priors is None:
    principal2Priors = getDefaultPrior(K)
  if realDistributions is None:
    realDistributions = getDefaultRealDistributions(K)

  banditProblemInstance = BanditProblemInstance(K, T, realDistributions)
  bestArmMean = banditProblemInstance.bestArmMean()

  # instantiate 2 principals (who are of some subclass of BanditAlgorithm)
  principal1 = principalAlg1(banditProblemInstance, principal1Priors)
  principal2 = principalAlg2(banditProblemInstance, principal2Priors)

  principals = { 'principal1': principal1, 'principal2': principal2 }
  agents = agentAlg(principals, K, memory=memory)

  # give the agents a few observations
  for i in xrange(warmStartNumObservations):
    for (principalName, principal) in principals.iteritems():
      (reward, arm) = principal.executeStep(i)
      agents.updateInformationSet(reward, arm, principalName)

  for principal in principals.values():
    principal.resetStats()

  #for j in range(200):
  #  (reward, arm) = principals['principal2'].executeStep(i)
  #  agents.updateInformationSet(reward, arm, 'principal2')
  #principals['principal2'].resetStats()


  # we first define the problem instance WITHOUT the realizations so that the "warm start" observations do not draw from the same pre-drawn
  # realization set, but the actual run of competing bandits will pull from it and so we re-instantiate the instance here
  banditProblemInstance = BanditProblemInstance(K, T, realDistributions, realizations)
  for principal in principals.values():
    principal.setBanditInstance(banditProblemInstance)

  principalHistory = []
  results = []
  for t in xrange(int(T)):
    if t in RECORD_STATS_AT:
      results.append({
        'marketShare1' : principal1.n / float(t),
        'marketShare2' : principal2.n / float(t),
        'armCounts1' : principal1.armCounts,
        'armCounts2' : principal2.armCounts,
        'avgRegret1': principal1.getAverageRegret(),
        'avgRegret2': principal2.getAverageRegret(),
        'time': t
      })
    (principalName, principal) = agents.selectPrincipal()
    principalHistory.append(principalName)
    (reward, arm) = principal.executeStep(t)
    trueMeanOfArm = banditProblemInstance.getMeanOfArm(arm)
    principal.regret += (bestArmMean - trueMeanOfArm)
    agents.updateInformationSet(reward, arm, principalName)
  
  return results


initialResultDict = {
  'marketShare1': [],
  'marketShare2': [],
  'armCounts1': [],
  'armCounts2': [],
  'avgRegret1': [],
  'avgRegret2': [],
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
