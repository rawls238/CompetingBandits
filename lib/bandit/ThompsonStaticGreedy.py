import numpy as np
from BanditAlgorithm import BanditAlgorithm
from deepcopy import deepcopy

class ThompsonDynamicGreedy(BanditAlgorithm):
  @staticmethod
  def shorthand():
    return 'TSDG'

  def __init__(self, banditProblemInstance, priors, distr='beta'):
    super(ExploreThenExploit, self).__init__(banditProblemInstance, priors, distr)
    self.greedy = False

  def switchAlgorithm(self):
    self.greedy = not self.greedy
    self.priors = deepcopy(self.posteriors)
  
  def pickAnArm(self, t):
    if self.greedy:
      return np.argmax([p.mean() for p in self.priors])
    else:
      return np.argmax([p.rvs() for p in self.posteriors])
