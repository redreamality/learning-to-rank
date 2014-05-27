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
import cStringIO
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

from OneHiddenLayer import OneHiddenLayer
from Linear import Linear


class TestRankers(unittest.TestCase):
    def setUp(self):
        self.feature_count = 50
        self.number_docs = 1000
        self.docs = range(self.number_docs)
        self.features = np.random.rand(self.number_docs, self.feature_count)

        self.linear_model = Linear(self.feature_count)
        self.linear_w = self.linear_model.initialize_weights("random")

        self.hidden_model = OneHiddenLayer(self.feature_count)
        self.hidden_w = self.hidden_model.initialize_weights("random")

    def testLinear(self):
        scores = self.linear_model.score(self.features, self.linear_w)
        docs1 = [d for _, d in sorted(zip(scores, self.docs))]
        scores = self.linear_model.score(self.features, self.linear_w * 200)
        docs2 = [d for _, d in sorted(zip(scores, self.docs))]
        self.assertListEqual(docs1, docs2, "Linear Ranker should be magnitude"
                         "independent")

    def testOneHiddenLayer(self):
        scores = self.hidden_model.score(self.features, self.hidden_w)
        docs1 = [d for _, d in sorted(zip(scores, self.docs))]
        scores = self.hidden_model.score(self.features, self.hidden_w * 10)
        docs2 = [d for _, d in sorted(zip(scores, self.docs))]
        self.assertNotEqual(docs1, docs2, "Hidden Layer Ranker should be "
                            "magnitude dependent")

    def testInitOneHiddenLayer(self):
        orderings = set()
        reps = 1000
        for _ in range(reps):
            w = self.hidden_model.initialize_weights("random")
            scores = self.hidden_model.score(self.features, w)
            ordering = tuple([d for _, d in
                              sorted(zip(scores, self.docs))][:10])
            orderings.add(ordering)
        self.assertEqual(reps, len(orderings))

if __name__ == '__main__':
        unittest.main()
