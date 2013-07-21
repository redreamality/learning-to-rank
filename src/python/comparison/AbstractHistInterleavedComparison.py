# KH, 2012/08/14
# Abstract base class for interleaved comparison methods with historical data


class AbstractHistInterleavedComparison:

    def infer_outcome(self, l, a, c, target_r1, target_r2, query):
        raise NotImplementedError("The derived class needs to implement "
            "infer_outcome.")
