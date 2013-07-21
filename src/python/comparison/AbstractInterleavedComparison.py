# KH, 2012/06/19
# Abstract base class for interleaved comparison methods


class AbstractInterleavedComparison:

    def interleave(self, r1, r2, query, length):
        raise NotImplementedError("The derived class needs to implement "
            "interleave.")

    def infer_outcome(self, l, a, c, query):
        raise NotImplementedError("The derived class needs to implement "
            "infer_outcome.")
