import random as rand

class Info:
  def __init__(self, principal, num_picked=0, total_reward=0.0):
    self.principal = principal
    self.num_picked = num_picked
    self.total_reward = total_reward

  def getMeanScore(self):
    return self.total_reward / self.num_picked if self.num_picked != 0 else self.total_reward # don't divide by 0

class InformationSet:
  def __init__(self, principals):
     self.infoSet  = { principal: Info(principal) for (principal, v) in principals.iteritems() }

  def getMaxPrincipalsAndScores(self):
    maxScore = -1
    maxPrincipals = []
    for (k, v) in self.infoSet.iteritems():
      score = v.getMeanScore()
      if score > maxScore:
        maxPrincipals = [k]
        maxScore = score
      elif score == maxScore:
        maxPrincipals.append(k)
    return (maxPrincipals, maxScore)

  def getScores(self):
    return { (k, v.getMeanScore()) for (k, v) in self.infoSet.iteritems() }

  def getRandPrincipal(self):
    return rand.choice(self.infoSet.keys())

  def updateInformationSet(self, reward, principal):
    self.infoSet[principal].num_picked += 1
    self.infoSet[principal].total_reward += reward
