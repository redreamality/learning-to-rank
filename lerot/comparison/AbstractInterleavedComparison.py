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

# KH, 2012/06/19

"""
Abstract base class for interleaved comparison methods
"""


class AbstractInterleavedComparison:

    def interleave(self, r1, r2, query, length):
        raise NotImplementedError("The derived class needs to implement "
            "interleave.")

    def infer_outcome(self, l, a, c, query):
        raise NotImplementedError("The derived class needs to implement "
            "infer_outcome.")
