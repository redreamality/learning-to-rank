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

from StatelessRankingFunction import StatelessRankingFunction

class SyntheticDeterministicRankingFunction(StatelessRankingFunction):
    """ Synthetic deterministic ranker. """
    def __init__(self, synthetic_docs):
        self.docs = synthetic_docs

    def init_ranking(self, query):
        # Nothing needs to be done in this case.
        pass

    def get_document_probability(self, doc):
        """ Get probability of producing doc as the next document drawn. """
        pos = self.docs.index(doc)
        return 1.0 if pos == 0 else 0.0

    def update_weights(self, new_weights):
        # Not required under synthetic data.
        pass
