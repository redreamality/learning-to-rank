from numpy import array, zeros
from utils import sample_unit_sphere
from random import gauss


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
