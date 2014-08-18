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

from ..utils import get_class


class AbstractRankingFunction:
    """Abstract base class for ranking functions."""

    def __init__(self,
                 ranker_arg_str,
                 ties,
                 feature_count,
                 init=None,
                 sample=None):

        self.feature_count = feature_count
        ranking_model_str = "ranker.model.Linear"
        for arg in ranker_arg_str:
            if type(arg) is str and arg.startswith("ranker.model"):
                ranking_model_str = arg
            elif type(arg) is int or type(arg) is float:
                self.ranker_type = float(arg)
        self.ranking_model = get_class(ranking_model_str)(feature_count)

        if sample:
            self.sample = get_class("utils." + sample)

        self.ties = ties
        self.w = self.ranking_model.initialize_weights(init)

    def score(self, features):
        return self.ranking_model.score(features, self.w.transpose())

    def get_candidate_weight(self, delta):
        u = self.sample(self.ranking_model.get_feature_count())
        return self.w + delta * u, u

    def init_ranking(self, query):
        self.dirty = False
        raise NotImplementedError("Derived class needs to implement "
            "init_ranking.")

    def next(self):
        self.dirty = True
        raise NotImplementedError("Derived class needs to implement "
            "next.")

    def next_det(self):
        self.dirty = True
        raise NotImplementedError("Derived class needs to implement "
            "next_det.")

    def next_random(self):
        self.dirty = True
        raise NotImplementedError("Derived class needs to implement "
            "next_random.")

    def get_document_probability(self, docid):
        raise NotImplementedError("Derived class needs to implement "
            "get_document_probability.")


    def getDocs(self, numdocs=None):
        if not hasattr(self, "dirty"):
            raise NotImplementedError("Derived class should (re)set self.dirty")
        if self.dirty:
            raise Exception("Always call init_ranking() before getDocs()!")
        docs = []
        i = 0
        while True:
            if numdocs != None and i >= numdocs:
                break
            try:
                docs.append(self.next())
            except Exception as e:
                break
            i += 1
        return docs

    def rm_document(self, docid):
        raise NotImplementedError("Derived class needs to implement "
            "rm_document.")

    def document_count(self):
        raise NotImplementedError("Derived class needs to implement "
            "document_count.")

    def update_weights(self, w, alpha=None):
        """update weight vector"""
        if alpha == None:
            self.w = w
        else:
            self.w = self.w + alpha * w
