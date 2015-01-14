'''
Created on 14 jan. 2015

@author: Jos
'''
import unittest

import lerot.comparison.test.script_ProbabilisticMultileave as exp


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def testDataVisualization(self):
        errors = [[i + j * 10 for i in range(100)] for j in range(3)]
        labels = ['a', 'b', 'c']
        exp.visualizeError(errors, labels)


if __name__ == "__main__":
    unittest.main()
