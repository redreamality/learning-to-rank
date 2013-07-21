from random import randint
from utils import rank
from AbstractRankingFunction import AbstractRankingFunction


class DeterministicRankingFunction (AbstractRankingFunction):

    def init_ranking(self, query):
        self.qid = query.get_qid()
        scores = self.ranking_model.score(query.get_feature_vectors(),
                                          self.w.transpose())
        ranks = rank(scores, reverse=False, ties=self.ties)
        # sort documents by ranks, ties are broken at random by default
        ranked_docids = []
        for pos, docid in enumerate(query.__docids__):
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
