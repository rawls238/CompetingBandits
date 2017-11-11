import random as rand
import numpy as np
from Agent import Agent

class SoftMax(Agent):
  def __init__(self, principals, priors=None, alpha=-3):
    self.alpha = alpha
    super(SoftMax, self).__init__(principals, priors)

  def selectPrincipal(self):
    scores = self.informationSet.getScores()
    for (principal, score) in scores:
      if rand.random() < (1-np.exp(score * self.alpha)): #this seems wrong...
        return (principal, self.principals[maxPrincipal])

  def tieBreak(self, items):
    return rand.choice(items)
