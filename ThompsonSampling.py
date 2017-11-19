import numpy as np
from BanditAlgorithm import BanditAlgorithm

class ThompsonSampling(BanditAlgorithm):
  def pickAnArm(self):
    return np.argmax([p.rvs() for p in self.posteriors])
