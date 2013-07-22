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

sys.path.insert(0, os.path.abspath('..'))

from SyntheticComparisonExperiment import SyntheticComparisonExperiment as sce


class TestRankers(unittest.TestCase):

    def setUp(self):
        # self.experiment = SyntheticComparisonExperiment()
        pass

    def testParetoDominates2Docs(self):
        labels = [1, 0]
        a = [0, 1]
        b = [1, 0]
        self.assertEqual(sce._pareto_dominates(a, b, labels), True,
            "a (%s) dominates b (%s)" % (", ".join([str(x) for x in a]),
                                         ", ".join([str(x) for x in b])))
        self.assertEqual(sce._pareto_dominates(b, a, labels), False,
            "b (%s) doesn't dominate a (%s)" % (", ".join([str(x) for x in b]),
                                         ", ".join([str(x) for x in a])))
        b = [0, 1]
        self.assertEqual(sce._pareto_dominates(a, b, labels), False,
            "a (%s) equal to b (%s)" % (", ".join([str(x) for x in a]),
                                        ", ".join([str(x) for x in b])))

    def testParetoDominates3Docs(self):
        labels = [0, 1, 0]
        a = [0, 1, 2]
        b = [1, 0, 2]
        self.assertEqual(sce._pareto_dominates(a, b, labels), False,
            "a (%s) doesn't dominate b (%s)" % (", ".join([str(x) for x in a]),
                                            ", ".join([str(x) for x in b])))
        self.assertEqual(sce._pareto_dominates(b, a, labels), True,
            "b (%s) dominates a (%s)" % (", ".join([str(x) for x in b]),
                                         ", ".join([str(x) for x in a])))
        b = [0, 1, 2]
        self.assertEqual(sce._pareto_dominates(a, b, labels), False,
            "a (%s) equal to b (%s)" % (", ".join([str(x) for x in a]),
                                        ", ".join([str(x) for x in b])))

    def testParetoDominates4Docs(self):
        labels = [0, 1, 0, 0]
        a = [1, 0, 2, 3]
        b = [1, 2, 0, 3]
        self.assertEqual(sce._pareto_dominates(a, b, labels), False,
            "a (%s) equal to b (%s)" % (", ".join([str(x) for x in a]),
                                        ", ".join([str(x) for x in b])))

    def testParetoDominatesSeveralRelevant(self):
        labels = [0, 1, 0, 1]
        a = [0, 1, 2, 3]
        b = [1, 0, 2, 3]
        self.assertEqual(sce._pareto_dominates(a, b, labels), False,
            "a (%s) doesn't dominate b (%s)" % (", ".join([str(x) for x in a]),
                                            ", ".join([str(x) for x in b])))
        self.assertEqual(sce._pareto_dominates(b, a, labels), True,
            "b (%s) dominates a (%s)" % (", ".join([str(x) for x in b]),
                                         ", ".join([str(x) for x in a])))
        b = [0, 1, 2, 3]
        self.assertEqual(sce._pareto_dominates(a, b, labels), False,
            "a (%s) equal to b (%s)" % (", ".join([str(x) for x in a]),
                                        ", ".join([str(x) for x in b])))

    def testParetoDominatesMissingDocs(self):
        labels = [0, 1, 0, 1]
        a = [0, 1, 2]
        b = [1, 0, 3]
        self.assertEqual(sce._pareto_dominates(a, b, labels), False,
            "a (%s) doesn't dominate b (%s)" % (", ".join([str(x) for x in a]),
                                            ", ".join([str(x) for x in b])))
        self.assertEqual(sce._pareto_dominates(b, a, labels), True,
            "b (%s) dominates a (%s)" % (", ".join([str(x) for x in b]),
                                         ", ".join([str(x) for x in a])))
        b = [0, 1, 2]
        self.assertEqual(sce._pareto_dominates(a, b, labels), False,
            "a (%s) equal to b (%s)" % (", ".join([str(x) for x in a]),
                                        ", ".join([str(x) for x in b])))

    def testGenerateSyntheticRankingsRandomlyWithOneRelevant(self):
        length = 2
        docids = range(length)
        labels = [0] * length
        labels[0] = 1
        (better, worse) = sce._generate_synthetic_rankings_randomly(docids,
                                                                    labels)
        self.assertEqual([0, 1], better, "better ranking is [0, 1]: "
            + ", ".join([str(x) for x in better]))
        self.assertEqual([1, 0], worse, "worse ranking is [1, 0]: "
            + ", ".join([str(x) for x in worse]))

        length = 3
        docids = range(length)
        labels = [0] * length
        labels[0] = 1
        (better, worse) = sce._generate_synthetic_rankings_randomly(docids,
                                                                    labels)
        self.assertIn(better, [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 0, 1]],
            "better is valid:" + ", ".join([str(x) for x in better]))
        self.assertIn(worse, [[1, 0, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]],
            "worse is valid:" + ", ".join([str(x) for x in worse]))
        if better in [[1, 0, 2], [2, 0, 1]]:
            self.assertIn(worse, [[1, 2, 0], [2, 1, 0]],
                "better dominates worse:" + ", ".join([str(x) for x in worse]))

        length = 5
        docids = range(length)
        labels = [0] * length
        labels[0] = 1
        (better, worse) = sce._generate_synthetic_rankings_randomly(docids,
                                                                    labels)
        self.assertEqual(True, sce._pareto_dominates(better, worse, labels),
                "better (%s) dominates worse (%s)" %
                (", ".join([str(x) for x in better]),
                 ", ".join([str(x) for x in worse])))

        length = 10
        docids = range(length)
        labels = [0] * length
        labels[0] = 1
        (better, worse) = sce._generate_synthetic_rankings_randomly(docids,
                                                                    labels)
        self.assertEqual(True, sce._pareto_dominates(better, worse, labels),
                "better (%s) dominates worse (%s)" %
                (", ".join([str(x) for x in better]),
                 ", ".join([str(x) for x in worse])))

    def testGenerateSyntheticRankingsRandomlyWithManyRelevant(self):
        length = 3
        docids = range(length)
        labels = [0] * length
        labels[0] = 1
        labels[1] = 1
        (better, worse) = sce._generate_synthetic_rankings_randomly(docids,
                                                                    labels)
        self.assertIn(better, [[0, 1, 2], [1, 0, 2], [0, 2, 1], [1, 2, 0]],
            "better is valid:" + ", ".join([str(x) for x in better]))
        self.assertIn(worse, [[0, 2, 1], [1, 2, 0], [2, 0, 1], [2, 1, 0]],
            "worse is valid:" + ", ".join([str(x) for x in worse]))
        if better in [[0, 2, 1], [1, 2, 0]]:
            self.assertIn(worse, [[2, 0, 1], [2, 1, 0]],
                "better dominates worse:" + ", ".join([str(x) for x in worse]))

        length = 5
        docids = range(length)
        labels = [0] * length
        labels[0] = 1
        labels[2] = 1
        labels[4] = 1
        print better
        print worse
        (better, worse) = sce._generate_synthetic_rankings_randomly(docids,
                                                                    labels)
        self.assertEqual(True, sce._pareto_dominates(better, worse, labels),
                "better (%s) dominates worse (%s)" %
                (", ".join([str(x) for x in better]),
                 ", ".join([str(x) for x in worse])))

        length = 10
        docids = range(length)
        labels = [0] * length
        labels[0] = 1
        labels[1] = 1
        labels[5] = 1
        labels[9] = 1
        print better
        print worse
        (better, worse) = sce._generate_synthetic_rankings_randomly(docids,
                                                                    labels)
        self.assertEqual(True, sce._pareto_dominates(better, worse, labels),
                "better (%s) dominates worse (%s)" %
                (", ".join([str(x) for x in better]),
                 ", ".join([str(x) for x in worse])))

if __name__ == '__main__':
        unittest.main()
