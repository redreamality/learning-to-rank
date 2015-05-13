'''
Created on 14 jan. 2015

@author: Robert-Jan
'''
import unittest
import lerot.comparison.ProbabilisticMultileave as ml


class Test(unittest.TestCase):
    def testCreditsToOutcome(self):
        pm = ml.ProbabilisticMultileave()
        print(pm.credits_to_outcome([0.0, 0.0, 0.0]))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testCreditsToOutcome']
    unittest.main()
