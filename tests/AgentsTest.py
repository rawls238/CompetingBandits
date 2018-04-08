import unittest

import sys 
sys.path.append('..')
from lib.agent.HardMax import HardMax
from lib.agent.HardMaxWithRandom import HardMaxWithRandom
from lib.InformationSet import InformationSet

class TestAgents(unittest.TestCase):

  def test_moving_average(self):
    principals = { 'principal1': None, 'principal2': None }
    info = InformationSet(principals, 3, memory=2, score='moving_average', discount_factor=0.99)
    self.assertEqual(info.getScores().values(), [0, 0])
    info.updateInformationSet(5, 1, 'principal1')
    self.assertEqual(info.getMaxPrincipalsAndScores(), (['principal1'], 5.0))
    info.updateInformationSet(3, 1, 'principal1')
    self.assertEqual(info.getMaxPrincipalsAndScores(), (['principal1'], 4.0))
    info.updateInformationSet(1, 1, 'principal1')
    self.assertEqual(info.getMaxPrincipalsAndScores(), (['principal1'], 2.0))
    info.updateInformationSet(-100, 1, 'principal1')
    self.assertEqual(info.getMaxPrincipalsAndScores(), (['principal2'], 0.0))

  def test_hard_max(self):
    principals = { 'principal1': None, 'principal2': None }
    agent = HardMax(principals, 3, memory=2)
    agent.updateInformationSet(5, 1, 'principal1')
    self.assertEqual(agent.selectPrincipal()[0], 'principal1')
    agent.updateInformationSet(3, 1, 'principal1')
    self.assertEqual(agent.selectPrincipal()[0], 'principal1')
    agent.updateInformationSet(-8, 1, 'principal1')
    self.assertEqual(agent.selectPrincipal()[0], 'principal2')
    agent.resetInformationSet()
    agent.updateInformationSet(1, 1, 'principal1')
    self.assertEqual(agent.selectPrincipal()[0], 'principal1')

  def test_hard_max_random(self):
    principals = { 'principal1': None, 'principal2': None }
    agent = HardMaxWithRandom(principals, 3, memory=2)
    agent.updateInformationSet(5, 1, 'principal1')
    N = 1000
    c = 0
    for i in xrange(N):
      if agent.selectPrincipal()[0] == 'principal2':
        c += 1
    self.assertAlmostEqual(c, 25, delta=10)

if __name__ == '__main__':
    unittest.main()
