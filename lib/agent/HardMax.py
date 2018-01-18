import random as rand
from Agent import Agent

class HardMax(Agent):
  def selectPrincipal(self):
    maxPrincipals = self.informationSet.getMaxPrincipalsAndScores()[0]
    maxPrincipal = self.tieBreak(maxPrincipals)
    return (maxPrincipal, self.principals[maxPrincipal])

  def tieBreak(self, items):
    return rand.choice(items)
