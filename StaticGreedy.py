import numpy
import scipy

class StaticGreedy < BanditAlgorithm:
  def pickAnArm():
    # pick the arm with the highest mean and return it
    bestArm = 1
    for k in xrange(1, self.K + 1):
      # get the mean of 