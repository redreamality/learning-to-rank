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

# KH, 2012/06/14
"""
Abstract base class for retrieval system implementations for use in learning
experiments.
"""


class AbstractLearningSystem:
    """An abstract online learning system. New implementations of online
    learning systems should inherit from this class."""

    def get_ranked_list(self, query):
        raise NotImplementedError("Derived class needs to implement "
            "get_ranked_list.")

    def update_solution(self, clicks):
        raise NotImplementedError("Derived class needs to implement "
            "update_solution.")

    def get_solution(self):
        raise NotImplementedError("Derived class needs to implement "
            "get_solution.")
