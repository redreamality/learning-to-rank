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

import random

from TeamDraft import TeamDraft


class TdiVa(TeamDraft):
    """ Basis class for TdiVa1 and TdiVa2 """

    @staticmethod
    def sampleSmoothly(a, b, maxVal):
        if a > b:
            a, b = b, a
        if a > 0 and b < maxVal:
            randVal = random.randint(a, b + 1)
            if randVal == b + 1:
                return a - 1 if random.randint(0, 1) == 0 else b + 1
            else:
                return randVal
        elif a == 0 and b == maxVal:
            return random.randint(a, b)
        else:   # a > 0 or b < maxVal
            randVal = random.randint(0, 2 * (b - a) + 2)
            if randVal == 2 * (b - a) + 2:
                return (a - 1) if a > 0 else b + 1
            else:
                return a + randVal // 2
