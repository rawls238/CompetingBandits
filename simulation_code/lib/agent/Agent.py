''' Class to hold information regarding the Agents' beliefs and selection algorithms '''
import abc

# https://stackoverflow.com/a/11158224/3889099
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from constants import DEFAULT_MEMORY, DEFAULT_DISCOUNT_FACTOR
from InformationSet import InformationSet

class Agent:
  __metaclass__ = abc.ABCMeta

  def __init__(self, principals, K, priors=None, score='moving_average', memory=DEFAULT_MEMORY, discount_factor=DEFAULT_DISCOUNT_FACTOR):
    self.principals = principals
    self.priors = priors
    self.score = score
    self.K = K
    self.memory = memory
    self.discount_factor = discount_factor

    # for each principal store the number of times selected and total reward
    self.informationSet = InformationSet(principals, K, memory=memory, discount_factor=discount_factor, score=score)

  @abc.abstractmethod
  def selectPrincipal(self):
    return

  def getScores(self):
    return self.informationSet.getScores()

  def updateInformationSet(self, reward, arm, principalName):
    self.informationSet.updateInformationSet(reward, arm, principalName)

  def resetInformationSet(self):
    self.informationSet = InformationSet(self.principals, self.K, discount_factor=self.discount_factor, memory=self.memory, score=self.score)

  def printMeanBeliefs(self):
    print(self.informationSet.getScores())
