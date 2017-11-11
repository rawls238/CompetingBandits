
# Import BanditAlgorithm classes
from DynamicGreedy import DynamicGreedy
from UCB import UCB

## Import Agent classes
from HardMax import HardMax
from HardMaxWithRandom import HardMaxWithRandom
from SoftMax import SoftMax
from SoftMaxWithRandom import SoftMaxWithRandom


from BanditProblemInstance import BanditProblemInstance

from scipy.stats import bernoulli, beta, uniform

K = 2
T = 1000.0


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
principal1 = UCB(banditProblemInstance, principalPriors)
principal2 = DynamicGreedy(banditProblemInstance, principalPriors)

principals = { 'principal1': principal1, 'principal2': principal2 }
agents = SoftMaxWithRandom(principals, agentPriors)

for t in xrange(1,int(T)):
  (principalName, principal) = agents.selectPrincipal()
  (reward, arm) = principal.executeStep()
  agents.updateInformationSet(reward, principalName)

marketShare1 = principal1.n / T
marketShare2 = principal2.n / T
print('Market Share of Principal 1 ' + str(marketShare1))
print('Principal 1 counts ' + str(principal1.armCounts))
print('Market Share of Principal 2 ' + str(marketShare2))
print('Principal 2 counts ' + str(principal2.armCounts))
