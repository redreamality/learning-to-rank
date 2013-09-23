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

from random import sample
from numpy import mean


class AbstractEval:
    """Abstract base class for computing evaluation metrics for given relevance
    labels."""

    def __init__(self):
        self.prev_solution_w = None
        self.prev_score = None

    def evaluate_all(self, solution, queries, cutoff=-1, ties="random"):
        if self.prev_solution_w != None and (self.prev_solution_w ==
                                             solution.w).all():
            return self.prev_score
        outcomes = []
        for query in queries:
            outcomes.append(self.evaluate_one(solution, query, cutoff, ties))
        score = mean(outcomes)

        self.prev_solution_w = solution.w
        self.prev_score = score

        return score

    def evaluate_one(self, solution, query, cutoff=-1, ties="random"):
        scores = solution.score(query.get_feature_vectors())
        sorted_docs = self._sort_docids_by_score(query.get_docids(), scores,
            ties=ties)
        return self.evaluate_ranking(sorted_docs, query, cutoff)

    def evaluate_ranking(self, ranking, query, cutoff=-1):
        """Compute NDCG for the provided ranking. The ranking is expected
        to contain document ids in rank order."""
        if cutoff == -1 or cutoff > len(ranking):
            cutoff = len(ranking)

        if query.has_ideal():
            ideal_dcg = query.get_ideal()
        else:
            ideal_labels = list(reversed(sorted(query.get_labels())))[:cutoff]
            ideal_dcg = self.get_dcg(ideal_labels, cutoff)
            query.set_ideal(ideal_dcg)

        if ideal_dcg == .0:
            # return 0 when there are no relevant documents. This is consistent
            # with letor evaluation tools; an alternative would be to return
            # 0.5 (e.g., used by the yahoo learning to rank challenge tools)
            return 0.0

        # get labels for the sorted docids
        sorted_labels = [0] * cutoff
        for i in range(cutoff):
            sorted_labels[i] = query.get_label(ranking[i])
        dcg = self.get_dcg(sorted_labels, cutoff)

        return dcg / ideal_dcg

    def _sort_docids_by_score(self, docids, scores, ties="random"):
        n = len(docids)
        if ties == "first":
            scored_docids = zip(scores, reversed(range(n)), docids)
        elif ties == "last":
            scored_docids = zip(scores, range(n), docids)
        elif ties == "random":
            scored_docids = zip(scores, sample(range(n), n), docids)
        else:
            raise Exception("Unknown method for breaking ties: \"%s\"" % ties)
        scored_docids.sort(reverse=True)
        return [docid for _, _, docid in scored_docids]
