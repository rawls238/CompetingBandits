from BanditAlgorithm import BanditAlgorithm
import numpy as np
import math

class UCBGeneric(BanditAlgorithm):
  def __init__(self, banditProblemInstance, priors, distr='beta', constant='t'):
     super(UCBGeneric, self).__init__(banditProblemInstance, priors, distr)
     self.constant = constant

  @staticmethod
  def shorthand():
    return 'UCB'

  @property
  def constantVal(self):
    if self.constant == 't':
      totalCounts = sum(self.armCounts)
      return (2 * math.log(totalCounts))
    else:
      return self.constant

  def pickAnArm(self, t):
    numArms = len(self.armCounts)
    for i in range(numArms):
      if self.armCounts[i] == 0.0:
        return i

    ucbValues = [0.0 for a in range(numArms)]
    
    for a in range(numArms):
      bonus = math.sqrt(self.constantVal / float(self.armCounts[a]))
      meanReward = self.rewardTotal[a] / float(self.armCounts[a])
      ucbValues[a] = meanReward + bonus
    return np.argmax(ucbValues)

class UCB1WithConstantOne(UCBGeneric):
  def __init__(self, banditProblemInstance, priors, distr='beta'):
    super(UCB1WithConstantOne, self).__init__(banditProblemInstance, priors, distr, 1)

  @staticmethod
  def shorthand():
    return 'UCB, const 1'

class UCB1WithConstantT(UCBGeneric):
  def __init__(self, banditProblemInstance, priors, distr='beta'):
    super(UCB1WithConstantT, self).__init__(banditProblemInstance, priors, distr, 't')

  @staticmethod
  def shorthand():
    return 'UCB, const 2*log(t)'
