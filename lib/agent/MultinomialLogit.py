import random as rand
import numpy as np
from Agent import Agent

class MultinomialLogit(Agent):
  def selectPrincipal(self):
    scores = self.informationSet.getScores()
    total = sum([np.exp(score) for score in scores.values()])
    probs = { principal: np.exp(score) / total for (principal, score) in scores.iteritems() }
    
    threshold = rand.random()
    cumProb = 0.0
    for (principal, prob) in probs.iteritems():
      cumProb += prob
      if threshold <= cumProb
        return (principal, self.principals[principal])
