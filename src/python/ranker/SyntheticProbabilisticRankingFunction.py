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

from numpy import array, delete
from ProbabilisticRankingFunction import ProbabilisticRankingFunction


class SyntheticProbabilisticRankingFunction(ProbabilisticRankingFunction):
    """Synthetic ranker for use in this experiment only"""

    def __init__(self, ranker_arg_str, ties="random"):
        self.ranker_type = float(ranker_arg_str)
        self.ties = ties

    def init_ranking(self, synthetic_docids):
        if not synthetic_docids:
            return
        # assume that synthetic_docids are in rank order
        self.docids = synthetic_docids
        ranks = array(range(1, len(self.docids) + 1))
        # determine probabilities based on (reverse) document ranks
        tmp_val = 1. / pow(ranks, self.ranker_type)
        self.probs = tmp_val / sum(tmp_val)

    def _get_doc_pos(self, docid):
        try:
            pos = self.docids.index(docid)
        except:
            pos = [i for i, d in
                   enumerate(self.docids) if d[0] == docid][0]
        return pos

    def get_document_probability(self, docid):
        """get probability of producing doc as the next document drawn"""
        pos = self._get_doc_pos(docid)
        return self.probs[pos]

    def rm_document(self, docid):
        """remove doc from list of available docs, adjust probabilities"""
        # find position of the document
        try:
            pos = self._get_doc_pos(docid)
        except ValueError:
            print "cannot remove", docid,
            print "current document list:", self.docids
            print "qid:", self.qid
        # delete doc and renormalize
        self.docids.pop(pos)
        self.probs = delete(self.probs, pos)
        self.probs = self.probs / sum(self.probs)

    def update_weights(self, new_weights):
        """not required under synthetic data"""
        pass
