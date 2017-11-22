import random as rand
import numpy as np
from Agent import Agent


DEFAULT_ALPHA = .2
class SoftMax(Agent):
  def __init__(self, principals, numArms, priors=None, alpha=DEFAULT_ALPHA):
    super(SoftMax, self).__init__(principals, numArms, priors)
    self.alpha = alpha

  def selectPrincipal(self):
    scores = self.informationSet.getScores()
    total = sum([np.exp(score * self.alpha) for score in scores.values()])
    probs = { principal: np.exp(score * self.alpha) / total for (principal, score) in scores.iteritems() }
    
    threshold = rand.random()
    cumProb = 0.0
    for (principal, prob) in probs.iteritems():
      cumProb += prob
      if cumProb > threshold:
        return (principal, self.principals[principal])
