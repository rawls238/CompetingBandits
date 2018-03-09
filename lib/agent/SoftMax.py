import random as rand
import numpy as np
from Agent import Agent


DEFAULT_ALPHA = 30
class SoftMax(Agent):
  # epsilon: must be in (0, 0.5). baseline probability that each principal gets picked
  def __init__(self, principals, K, priors=None, alpha=DEFAULT_ALPHA, epsilon=0.1, memory=50):
    super(SoftMax, self).__init__(principals, K, priors, memory=memory)
    self.alpha = alpha
    self.epsilon = epsilon

  # def selectPrincipal(self):
  #   scores = self.informationSet.getScores()
  #   total = sum([np.exp(score * self.alpha) for score in scores.values()])
  #   probs = { principal: np.exp(score * self.alpha) / total for (principal, score) in scores.iteritems() }
    
  #   threshold = rand.random()
  #   cumProb = 0.0
  #   for (principal, prob) in probs.iteritems():
  #     cumProb += prob
  #     if cumProb > threshold:
  #       return (principal, self.principals[principal])

  # https://en.wikipedia.org/wiki/Logistic_function
  # alpha corresponds to "k" in the wikipedia article (it controls the steepness of the curve)
  def selectPrincipal(self):
    scores = self.informationSet.getScores()
    x = scores['principal1'] - scores['principal2'] # how much better 1 is than 2

    y = self.epsilon + ((1 - 2 * self.epsilon) / (1 + np.exp(-self.alpha * x)))

    r = rand.random()

    if r < y:
      # then choose principal 1
      return ('principal1', self.principals['principal1'])
    else:
      return ('principal2', self.principals['principal2'])
