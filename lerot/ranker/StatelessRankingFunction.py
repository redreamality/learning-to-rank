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

from AbstractRankingFunction import AbstractRankingFunction

class StatelessRankingFunction(AbstractRankingFunction):
    def init_ranking(self, query):
        """ Initialize ranking for particular query.

            Since AbstractRankingFunction has a next() function that changes a
            state, we need to have a support for that.
            You need to set self.docs and the only stateful object self.doc_idx
        """
        raise NotImplementedError('Derived class should implement this method')

    def document_count(self):
        return len(self.docs)

    def verticals(self, length=None):
        if length is None:
            length = self.document_count()
        return set(x.get_type() for x in self.docs[:length] \
                   if x.get_type() != 'Web')

    def next(self):
        if self.doc_idx >= self.document_count():
            raise Exception('There are no more documents to be selected')
        else:
            doc = self.docs[self.doc_idx]
            self.doc_idx += 1
            return doc

    def next_det(self):
        return self.next()

    def next_random(self):
        raise Exception('No random stuff. Stateless ranker has to be determenistic')

    def rm_document(self, doc):
        raise Exception('Removing document is not supported')

    def getDocs(self, numdocs=None):
        """ More efficient and less error-prone version of getDocs. """
        if numdocs is None:
            return self.docs
        else:
            return self.docs[:numdocs]
