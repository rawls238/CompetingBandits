from BanditAlgorithm import BanditAlgorithm

import numpy as np

class DynamicGreedy(BanditAlgorithm):
  @staticmethod
  def shorthand():
    return 'DG'

  def pickAnArm(self):
    chosenArm = np.argmax([p.mean() for p in self.posteriors])
    return chosenArm
