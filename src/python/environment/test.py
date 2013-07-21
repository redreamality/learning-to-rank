import unittest
import sys
import os
from random import shuffle

sys.path.insert(0, os.path.abspath('..'))

from CascadeUserModel import CascadeUserModel


class TestEnvironment(unittest.TestCase):
    def testPerfectUser(self):
        # initialize toy data (smaller docids are more relevant)
        labels = [4, 3, 2, 2, 1, 1, 0, 0, 0, 0]
        docids = range(len(labels))
        # initialize user model
        um = CascadeUserModel("--p_click 0:.0, 1:1.0, 2:1.0, 3:1.0, 4:1.0"
                              " --p_stop 0:.0, 1:.0, 2:.0, 3:.0, 4:.0")
        # generate result lists and clicks
        for _ in range(5):
            shuffle(docids)
            clicks = um.get_clicks(docids, labels)
            for i, click in enumerate(clicks):
                if not(labels[docids[i]] >= click):
                    self.fail("not perfect")

if __name__ == '__main__':
    unittest.main()
