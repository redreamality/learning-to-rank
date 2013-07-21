# KH, 2012/06/14
# Abstract base class for retrieval system implementations for use in learning
# experiments.


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
