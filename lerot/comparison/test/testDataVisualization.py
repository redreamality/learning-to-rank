'''
Created on 14 jan. 2015

@author: Jos
'''
import unittest

import lerot.comparison.test.evaluateData as ev


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def testDataVisualization(self):
        errors = [[i + j * 10 for i in range(100)] for j in range(3)]
        labels = ['a', 'b', 'c']
        ev.visualizeError(errors, labels)


if __name__ == "__main__":
    unittest.main()
