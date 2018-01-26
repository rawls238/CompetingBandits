from BanditAlgorithm import BanditAlgorithm

import numpy as np

class StaticGreedy(BanditAlgorithm): 
  @staticmethod
  def shorthand():
    return 'SG'

  def pickAnArm(self, t):
    chosenArm = np.argmax([p.mean() for p in self.priors])
    return chosenArm
