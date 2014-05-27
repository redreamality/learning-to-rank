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

from random import randint

from .AbstractRankingFunction import AbstractRankingFunction
from ..utils import rank


class DeterministicRankingFunction(AbstractRankingFunction):

    def init_ranking(self, query):
        self.qid = query.get_qid()
        scores = self.ranking_model.score(query.get_feature_vectors(),
                                          self.w.transpose())
        ranks = rank(scores, reverse=False, ties=self.ties)
        # sort documents by ranks, ties are broken at random by default
        ranked_docids = []
        for pos, docid in enumerate(query.get_docids()):
            ranked_docids.append((ranks[pos], docid))
        ranked_docids.sort(reverse=True)
        self.docids = [docid for (_, docid) in ranked_docids]

    def document_count(self):
        return len(self.docids)

    def next(self):
        """produce the next document"""

        # if there are no more documents
        if len(self.docids) < 1:
            raise Exception("There are no more documents to be selected")
        # otherwise, return highest ranked document
        return self.docids.pop(0)  # pop first element

    def next_det(self):
        return self.next()

    def next_random(self):
        """produce a random next document"""

        # if there are no more documents
        if len(self.docids) < 1:
            raise Exception("There are no more documents to be selected")
        # otherwise, return a random document
        rn = randint(0, len(self.docids) - 1)
        return self.docids.pop(rn)

    def get_document_probability(self, docid):
        """get probability of producing doc as the next document drawn"""
        pos = self.docids.index(docid)
        return 1.0 if pos == 0 else 0.0

    def rm_document(self, docid):
        """remove doc from list of available docs and adjust probabilities"""
        # find position of the document
        pos = self.docids.index(docid)
        # delete doc and renormalize
        self.docids.pop(pos)
