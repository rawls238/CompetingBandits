'''
This class contains the principal bandit algorithm that stores the information the principal has about the arms
as well as decision rules for what arms to pull
'''

import abc
import numpy as np
from scipy.stats import beta
from copy import deepcopy

class BanditAlgorithm:
  __metaclass__ = abc.ABCMeta

  def __init__(self, banditProblemInstance, priors, distr='beta'):
    self.priors = deepcopy(priors)
    self.posteriors = deepcopy(priors)
    self.distr = distr
    self.setBanditInstance(banditProblemInstance)
    self.resetStats()

  def getArmHistory(self):
    return self.armHistory

  #decide which arm to pick
  @abc.abstractmethod
  def pickAnArm(self, t=None):
    return

  def resetStats(self):
    self.n = 0 # how many times have i been picked
    self.armHistory = [] # an array of integers (0..K-1) of the arms i picked. arms are 0-indexed
    self.realizedRewardHistory = []
    self.realizedCumulativeRewardHistory = []
    self.meanRewardHistory = []
    self.meanCumulativeRewardHistory = []
    self.armCounts = [0.0 for i in xrange(self.banditProblemInstance.K)]
    self.rewardTotal = [0.0 for i in xrange(self.banditProblemInstance.K)]
    self.empiricalMeans = [0.0 for i in xrange(self.banditProblemInstance.K)]
    self.regretHistory = []
    self.regret = 0.0

  def resetPriors(self):
    self.posteriors = deepcopy(self.priors)

  def setBanditInstance(self, banditInstance):
    self.banditProblemInstance = banditInstance
    self.bestArmMean = banditInstance.bestArmMean()

  def getAverageRegret(self):
    if self.n == 0:
      return np.nan
    return self.regret / float(self.n)
  
  def executeStep(self, t):
    arm = self.pickAnArm(t)
    meanOfArm = self.banditProblemInstance.getMeanOfArm(arm)
    self.n += 1
    self.armHistory.append(arm)

    # get the realization by calling pullArm(k)
    reward = self.banditProblemInstance.pullArm(arm, t)
    if len(self.realizedCumulativeRewardHistory) == 0:
      self.realizedCumulativeRewardHistory.append(reward)
      self.meanCumulativeRewardHistory.append(meanOfArm)
    else:
      self.realizedCumulativeRewardHistory.append(self.realizedCumulativeRewardHistory[-1] + reward)
      self.meanCumulativeRewardHistory.append(self.meanCumulativeRewardHistory[-1] + meanOfArm)

    self.realizedRewardHistory.append(reward)
    self.meanRewardHistory.append(meanOfArm)
    curRegret = bestArmMean - meanOfArm
    self.regret += curRegret
    self.regretHistory.append(curRegret) 

    self.rewardTotal[arm] += reward
    self.armCounts[arm] += 1
    self.empiricalMeans[arm] = float(self.rewardTotal[arm]) / self.armCounts[arm]
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
