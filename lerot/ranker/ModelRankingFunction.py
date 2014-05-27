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

class ModelRankingFunction(StatelessRankingFunction):
    def __init__(self):
        self.pages = {}

    def add_doc_for_query(self, query, doc):
        self.pages.setdefault(query, [])
        self.pages[query].append(doc)

    def init_ranking(self, query):
        self.docs = self.pages[query]

    def update_weights(self, new_weights):
        # Not required here.
        pass
