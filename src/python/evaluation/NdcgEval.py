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

# KH, 2012/06/20

from numpy import log2
from AbstractEval import AbstractEval


class NdcgEval(AbstractEval):
    """Compute Ndcg (with gain = 2**rel-1 and log2 discount)."""

    def get_dcg(self, labels, cutoff=-1):
        if (cutoff == -1):
            cutoff = len(labels)
        dcg = 0
        # [0:cutoff] returns the labels up to min(len(labels), cutoff)
        for r, label in enumerate(labels[0:cutoff]):
            # use log2(1 + r), to be consistent with the implementation in the
            # letor 4 evaluation tools (and wikipedia, on 6/27/2012), even
            # though this makes discounting slightly inconsistent (indices are
            # zero-based, so using log2(2 + r) would be more consistent)
            dcg += (2 ** label - 1) / log2(2 + r)
        return dcg
