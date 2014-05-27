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

from numpy import array, zeros
from random import gauss

from ...utils import sample_unit_sphere


class AbstractRankingModel(object):

    def __init__(self, feature_count):
        self.feature_count = feature_count

    def get_feature_count(self):
        return self.feature_count

    def initialize_weights(self, method):
        if method == "zero":
            return zeros(self.feature_count)
        elif method == "random":
            return sample_unit_sphere(self.feature_count) * 0.01
        elif method == "fullyrandom":
            v = zeros(self.feature_count)
            for i in range(self.feature_count):
                v[i] = gauss(0, 1)
            return v
        else:
            try:
                weights = array([float(num) for num in method.split(",")])
                if len(weights) != self.feature_count:
                    raise Exception("List of initial weights does not have the"
                        " expected length (%d, expected $d)." %
                        (len(weights, self.feature_count)))
                return weights
            except Exception as ex:
                raise Exception("Could not parse weight initialization method:"
                    " %s. Possible values: zero, random, or a comma-separated "
                    "list of float values that indicate specific weight values"
                    ". Error: %s" % (method, ex))

    def score(self, features, w):
        raise NotImplementedError("Derived class needs to implement "
            "next.")
