import abc

class BanditAlgorithm:
  __metaclass__ = abc.ABCMeta

  def __init__(self, banditProblemInstance, priors):
    self.priors = priors #this is constant
    self.posteriors = priors
    self.banditProblemInstance = banditProblemInstance
    self.n = 0 # how many times have i been picked
    self.arm_history = [] # an array of integers (1..K) of the arms i picked
    self.reward_history = []

  def getArmHistory(self):
    return self.arm_history
    

  @abc.abstractmethod
  def pickAnArm(self):
    #decide which arm to pick
    return
  

  def executeStep(self):
    arm = self.pickAnArm()
    self.n += 1
    self.arm_history.append(arm)
    #get the realization by calling pullArm(k)
    reward = self.banditProblemInstance.pullArm(arm)
    self.reward_history.append(reward)
    self.updatePosterior(arm, reward)  
  
  # bayesian update your posterior
  def updatePosterior(self, arm, reward):
    ##**TODO: implement this
    return
