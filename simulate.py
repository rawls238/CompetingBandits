import numpy as np

from lib.constants import K, T, DEFAULT_PRINCIPAL1PRIORS, DEFAULT_PRINCIPAL2PRIORS, DEFAULT_REAL_DISTRIBUTIONS, \
DEFAULT_MEMORY, DEFAULT_DISCOUNT_FACTOR, DEFAULT_ALPHA, DEFAULT_WARM_START_NUM_OBSERVATIONS
from lib.BanditProblemInstance import BanditProblemInstance


from scipy.stats import bernoulli, beta

def getRealDistributionsFromPrior(prior):
  return [bernoulli(prior[i].rvs()) for i in xrange(K)]

# realDistributions - the true distribution of the arms
# principalPriors - the priors the principals have over the arms
# agentPriors - the priors the agents have for the distribution of rewards from each principal
def simulate(principalAlg1, principalAlg2, agentAlg,
  memory=DEFAULT_MEMORY,
  alpha=DEFAULT_ALPHA,
  warmStartNumObservations=DEFAULT_WARM_START_NUM_OBSERVATIONS,
  discountFactor=DEFAULT_DISCOUNT_FACTOR,
  realDistributions=DEFAULT_REAL_DISTRIBUTIONS,
  principal1Priors=DEFAULT_PRINCIPAL2PRIORS,
  principal2Priors=DEFAULT_PRINCIPAL2PRIORS):
  
  np.random.seed()

  banditProblemInstance = BanditProblemInstance(K, T, realDistributions)
  bestArmMean = banditProblemInstance.bestArmMean()

  # instantiate 2 principals (who are of some subclass of BanditAlgorithm)
  principal1 = principalAlg1(banditProblemInstance, principal1Priors)
  principal2 = principalAlg2(banditProblemInstance, principal2Priors)

  principals = { 'principal1': principal1, 'principal2': principal2 }
  agents = agentAlg(principals, K, memory=memory, discount_factor=discountFactor)

  # give the agents a few observations
  for i in xrange(warmStartNumObservations):
    for (principalName, principal) in principals.iteritems():
      (reward, arm) = principal.executeStep()
      agents.updateInformationSet(reward, arm, principalName)

  for principal in principals.values():
    principal.resetStats()
    principal.resetPriors()

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
