import random as rand
from Agent import Agent

class HeterogenousHardMax(Agent):
  def selectPrincipal(self):
    preferredArm = rand.choice([0, 1])
    maxPrincipals = self.informationSet.getLikelyArm(preferredArm)[0]
    maxPrincipal = self.tieBreak(maxPrincipals)
    return (maxPrincipal, self.principals[maxPrincipal])

  def tieBreak(self, items):
    return rand.choice(items)
