from BanditAlgorithm import BanditAlgorithm

import numpy as np


class StaticGreedy(BanditAlgorithm):
  def pickAnArm(self):
    chosenArm = np.argmax([p.mean() for p in self.priors]) + 1
    return chosenArm