''' Class to hold information regarding the Agents' beliefs and selection algorithms '''

import abc
from InformationSet import InformationSet

class Agent:
  __metaclass__ = abc.ABCMeta

  def __init__(self, principals, K, priors=None, score='mean', memory=50):
    self.principals = principals
    self.priors = priors
    self.score = score
    self.numRounds = 0
    self.K = K
    self.memory = memory

    # for each principal store the number of times selected and total reward
    self.informationSet = InformationSet(principals, K, priors, memory=memory, score=score)

  @abc.abstractmethod
  def selectPrincipal(self):
    return

  def updateInformationSet(self, reward, arm, principalName):
    self.numRounds += 1
    self.informationSet.updateInformationSet(reward, arm, principalName)

  def printMeanBeliefs(self):
    print(self.informationSet.getScores())
