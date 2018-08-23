import numpy as np
from SoftMax import SoftMax, DEFAULT_ALPHA

class SoftMaxWithRandom(SoftMax):
  def __init__(self, principals, K, priors=None, alpha=DEFAULT_ALPHA, epsilon=0.05, memory=50):
    super(SoftMaxWithRandom, self).__init__(principals, K, priors, alpha, memory=memory)
    self.epsilon = epsilon

  def selectPrincipal(self):
    if np.random.rand() > self.epsilon:
      return super(SoftMaxWithRandom, self).selectPrincipal()
    else:
      principal = self.informationSet.getRandPrincipal()
      return (principal, self.principals[principal])
