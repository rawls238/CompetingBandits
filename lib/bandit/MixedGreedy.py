import numpy as np
from BanditAlgorithm import BanditAlgorithm
from ThompsonSampling import ThompsonSampling
from DynamicGreedy import DynamicGreedy


# This is a hybrid algorithm that is greedy with probability p
# and plays another algorithm with probability 1 - p
class MixedGreedy(BanditAlgorithm):
  @staticmethod
  def shorthand():
    return 'MG'

  def __init__(self, banditProblemInstance, priors, p=0.2, alg=ThompsonSampling):
    super(MixedGreedy, self).__init__(banditProblemInstance, priors)
    self.alg = alg(banditProblemInstance, priors)
    self.greedy = DynamicGreedy(banditProblemInstance, priors)
  
  def pickAnArm(self, t):

    # ensure that both algorithms have the right posterior
    self.alg.posteriors = self.posteriors
    self.greedy.posteriors = self.posteriors
    if np.random.rand() > self.p:
      return self.alg.pickAnArm(t)
    else:
      return self.greedy.pickAnArm(t)
