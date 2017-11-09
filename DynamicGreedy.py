from BanditAlgorithm import BanditAlgorithm

import numpy as np


class DynamicGreedy(BanditAlgorithm):
  def pickAnArm(self):
    chosenArm = np.argmax([p.mean() for p in self.posteriors]) + 1
    return chosenArm