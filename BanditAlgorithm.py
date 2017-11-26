'''
This class contains the principal bandit algorithm that stores the information the principal has about the arms
as well as decision rules for what arms to pull
'''

import abc
from scipy.stats import beta

class BanditAlgorithm:
  __metaclass__ = abc.ABCMeta

  def __init__(self, banditProblemInstance, priors, distr='beta'):
    self.priors = priors
    self.posteriors = priors
    self.distr = distr
    self.banditProblemInstance = banditProblemInstance
    self.resetStats()

  def getArmHistory(self):
    return self.armHistory

  #decide which arm to pick
  @abc.abstractmethod
  def pickAnArm(self):
    return

  def resetStats(self):
    self.n = 0 # how many times have i been picked
    self.armHistory = [] # an array of integers (0..K-1) of the arms i picked. arms are 0-indexed
    self.rewardHistory = []
    self.armCounts = [0.0 for i in range(self.banditProblemInstance.K)]
    self.rewardTotal = [0.0 for i in range(self.banditProblemInstance.K)]
  
  def executeStep(self):
    arm = self.pickAnArm()
    self.n += 1
    self.armHistory.append(arm)

    #get the realization by calling pullArm(k)
    reward = self.banditProblemInstance.pullArm(arm)
    self.rewardHistory.append(reward)
    self.rewardTotal[arm] += reward
    self.armCounts[arm] += 1
    self.updatePosterior(arm, reward)
    return (reward, arm)
  
  # bayesian update your posterior
  def updatePosterior(self, arm, reward):
    distr = self.posteriors[arm]
    if self.distr == 'beta':
      if reward == 1:
        self.posteriors[arm] = beta(distr.args[0] + 1, distr.args[1])
      else:
        self.posteriors[arm] = beta(distr.args[0], distr.args[1] + 1)
