import random as rand
import numpy as np

# This contains the set of information for an agent about a particular principal
class Info:
  def __init__(self, principal, K, num_picked=0, total_reward=0.0, sliding_window_size=50):
    # note, we actually pass a string here. this variable does not point to a variable of class BanditAlgorithm
    self.principal = principal 
    self.num_picked = num_picked
    self.total_reward = total_reward
    self.K = K
    self.arm_counts = [0.0 for k in xrange(K)]
    self.arm_history = []
    self.reward_history = [total_reward]
    self.sliding_window_size = sliding_window_size
    self.moving_average = self.getMeanScore()

  def getMovingAverageScore(self):
    return self.moving_average

  def getMeanScore(self):
    return self.total_reward / self.num_picked if self.num_picked != 0 else self.total_reward # don't divide by 0

  def updateMovingAverage(self):
    if len(self.reward_history) <= self.sliding_window_size:
      self.moving_average = self.total_reward / self.num_picked if self.num_picked != 0 else self.total_reward
    else:
      remov = (self.reward_history[len(self.reward_history) - self.sliding_window_size - 1]) / float(self.sliding_window_size)
      add = self.reward_history[-1] / float(self.sliding_window_size)
      self.moving_average = self.moving_average - remov + add

  def getLikelyArm(self):
    numRounds = len(self.arm_history)
    weightedScores = { i: 0.0 for i in xrange(self.K) }
    for i in range(numRounds):
      curArm = self.arm_history[i]
      weightedScores[curArm] += np.exp(-1 * (numRounds - i))
    return max(weightedScores.iterkeys(), key=(lambda key: weightedScores[key]))

  # determine what the "score" of the agent is for this principal
  def getScore(self, t):
    scores = {
      'mean': self.getMeanScore,
      'moving_average': self.getMovingAverageScore
    }
    return scores[t]()


# This class contains the set of information that the agents have about all the principals
class InformationSet:

  # We initialize an information class for each principal, initializing each principal as though it has been picked once 
  # and the mean of its prior distribution is considered to be the only observed reward
  def __init__(self, principals, K, priors):
    # note: here, principal is actually a string variable ("principal1") and v is a variable of class BanditAlgorithm
    self.infoSet  = { principal: Info(principal, K, 1, priors[principal].mean()) for (principal, v) in principals.iteritems() }
    self.K = K

  # simply gets the principal with the highest expected reward
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

  # it looks like this returns a (maxPrincipals, maxScore) tuple, not an arm.
  def getLikelyArm(self, preferredArm):
    preferredPrincipals = {}
    for (principal, info) in self.infoSet.iteritems():
      if preferredArm == info.getLikelyArm():
        preferredPrincipals[principal] = info
    if len(preferredPrincipals) > 0:
      return self.getMaxPrincipalsAndScores(infoSet=preferredPrincipals)
    else:
      return self.getMaxPrincipalsAndScores()

  # get the set of scores for all principals
  # return value looks like {'principal1': 0.4, 'principal2': 0.3}
  def getScores(self, typeOfScore='moving_average'):
    return dict({ (k, v.getScore(typeOfScore)) for (k, v) in self.infoSet.iteritems() })

  def getRandPrincipal(self):
    return rand.choice(self.infoSet.keys())

  def updateInformationSet(self, reward, arm, principal):
    self.infoSet[principal].num_picked += 1
    self.infoSet[principal].arm_counts[arm] += 1
    self.infoSet[principal].arm_history.append(arm)
    self.infoSet[principal].reward_history.append(reward)
    self.infoSet[principal].total_reward += reward
    self.infoSet[principal].updateMovingAverage()
