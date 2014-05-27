# This file is part of Lerot.
#
# Lerot is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Lerot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Lerot.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import sys
import os
from random import shuffle

sys.path.insert(0, os.path.abspath('..'))

from CascadeUserModel import CascadeUserModel
from lerot.document import Document


class TestEnvironment(unittest.TestCase):
    def testPerfectUser(self):
        # initialize toy data (smaller docids are more relevant)
        labels = [4, 3, 2, 2, 1, 1, 0, 0, 0, 0]
        docids = [Document(x) for x in range(len(labels))]
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
