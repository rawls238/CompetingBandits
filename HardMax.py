import random
from Agent import Agent

class HardMax(Agent):
  def selectPrincipal(self):
    maxPrincipals = []
    maxScore = -1
    for (k, v) in self.informationSet.iteritems():
      score = v[1] / v[0] if v[0] != 0 else v[1] # don't divide by 0
      if score > maxScore:
        maxPrincipals = [k]
        maxScore = score
      elif score == maxScore:
        maxPrincipals.append(k)

    maxPrincipal = self.tieBreak(maxPrincipals)
    return (maxPrincipal, self.principals[maxPrincipal])

  def tieBreak(self, items):
    return random.choice(items)
