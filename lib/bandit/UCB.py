from BanditAlgorithm import BanditAlgorithm
import numpy as np
import math

class UCB(BanditAlgorithm):
  @staticmethod
  def shorthand():
    return 'UCB'

  def pickAnArm(self):
    numArms = len(self.armCounts)
    for i in range(numArms):
      if self.armCounts[i] == 0.0:
        return i

    ucbValues = [0.0 for a in range(numArms)]
    totalCounts = sum(self.armCounts)
    for a in range(numArms):
      bonus = math.sqrt((2 * math.log(totalCounts)) / float(self.armCounts[a]))
      meanReward = self.rewardTotal[a] / float(self.armCounts[a])
      ucbValues[a] = meanReward + bonus
    return np.argmax(ucbValues)
