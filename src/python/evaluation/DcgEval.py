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

import math

from AbstractEval import AbstractEval


class DcgEval(AbstractEval):
    """Compute DCG (with gain = 2**rel-1 and log2 discount)."""

    def get_dcg(self, ranked_labels, cutoff=-1):
        if (cutoff == -1):
            cutoff = len(ranked_labels)
        dcg = 0
        for r, label in enumerate(ranked_labels[:cutoff]):
            # Use log2(1 + r), to be consistent with the implementation in the
            # letor 4 evaluation tools (and wikipedia, on 6/27/2012), even
            # though this makes discounting slightly inconsistent (indices are
            # zero-based, so using log2(2 + r) would be more consistent).
            dcg += (2 ** label - 1) / math.log(2 + r, 2)
        return dcg

    def get_value(self, ranking, labels, orientations, cutoff=-1):
        """ Compute the value of the metric

        - ranking contains the list of documents to evaluate
        - labels are the relevance labels for all the documents, even those that are
            not in the ranking; labels[doc.get_id()] is the relevance of doc
        - orientations contains orientation values for the verticals;
            orientations[doc.get_type()] is the orientation value for the doc (from 0 to 1).
        """
        return self.get_dcg([labels[doc.get_id()] for doc in ranking], cutoff)
