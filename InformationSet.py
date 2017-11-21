import random as rand
import numpy as np


## TODO: generalize this to K arms
class Info:
  def __init__(self, principal, num_picked=0, total_reward=0.0, sliding_window_size=50):
    self.principal = principal
    self.num_picked = num_picked
    self.total_reward = total_reward
    self.arm_counts = [0.0, 0.0]
    self.arm_history = []
    self.reward_history = []
    self.sliding_window_size = sliding_window_size
    self.moving_average = 0.0

  def getMovingAverageScore(self):
    return self.moving_average

  def getMeanScore(self):
    return self.total_reward / self.num_picked if self.num_picked != 0 else self.total_reward # don't divide by 0

  def updateMovingAverage(self):
    if len(self.reward_history) < self.sliding_window_size:
      self.moving_average = self.total_reward / self.num_picked if self.num_picked != 0 else self.total_reward # don't divide by 0
    else:
      remov = (self.reward_history[len(self.reward_history) - self.sliding_window_size]) / self.sliding_window_size
      add = self.reward_history[-1] / self.sliding_window_size
      self.moving_average = self.moving_average - remov + add

  def getLikelyArm(self):
    numRounds = len(self.arm_history)
    weightedScores = { 0: 0.0, 1: 0.0 }
    for i in range(numRounds):
      curArm = self.arm_history[i]
      weightedScores[curArm] += np.exp(-1 * (numRounds - i))
    return max(weightedScores.iterkeys(), key=(lambda key: weightedScores[key]))

  def getScore(self, t):
    scores = {
      'mean': self.getMeanScore,
      'moving_average': self.getMovingAverageScore
    }
    return scores[t]()

class InformationSet:
  def __init__(self, principals, priors):
     self.infoSet  = { principal: Info(principal, 1, priors[principal].mean()) for (principal, v) in principals.iteritems() }

  def getMaxPrincipalsAndScores(self, typeOfScore='moving_average', infoSet = None):
    if infoSet is None:
      infoSet = self.infoSet
    maxScore = -1
    maxPrincipals = []
    for (k, v) in infoSet.iteritems():
      score = v.getScore(typeOfScore)
      if score > maxScore:
        maxPrincipals = [k]
        maxScore = score
      elif score == maxScore:
        maxPrincipals.append(k)
    return (maxPrincipals, maxScore)

  def getLikelyArm(self, preferredArm):
    preferredPrincipals = {}
    for (principal, info) in self.infoSet.iteritems():
      if preferredArm == info.getLikelyArm():
        preferredPrincipals[principal] = info
    if len(preferredPrincipals) > 0:
      return self.getMaxPrincipalsAndScores(infoSet=preferredPrincipals)
    else:
      return self.getMaxPrincipalsAndScores()

  def getScores(self):
    return dict({ (k, v.getMeanScore()) for (k, v) in self.infoSet.iteritems() })

  def getRandPrincipal(self):
    return rand.choice(self.infoSet.keys())

  def updateInformationSet(self, reward, arm, principal):
    self.infoSet[principal].num_picked += 1
    self.infoSet[principal].arm_counts[arm] += 1
    self.infoSet[principal].arm_history.append(arm)
    self.infoSet[principal].reward_history.append(arm)
    self.infoSet[principal].total_reward += reward
    self.infoSet[principal].updateMovingAverage()
