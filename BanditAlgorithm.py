class BanditAlgorithm :
  #prior is the prior distribution
  def __init__(self, K, T, distribution='bernoulli', prior, pullArm):
    self.prior = prior #this is constant
    self.posterior = prior
    self.n = 0 # how many times have i been picked
    self.history = [] # an array of integers (1..K) of the arms i picked

  def getHistory():
    return self.history


  #abstract
  def pickAnArm():
    #decide which arm to pick, k

    #get the realization by calling pullArm(k)

    # bayesian update your posterior

    #return an integer k (which arm you pulled) and the realization

  def updatePosterior():