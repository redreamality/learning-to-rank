

class AbstractSampler(object):
    def get_arms(self):
        raise NotImplementedError("Derived class needs to implement "
            "get_arms.")

    def update_scores(self, winner, loser):
        raise NotImplementedError("Derived class needs to implement "
            "update_scores.")

    def get_winner(self):
        raise NotImplementedError("Derived class needs to implement "
            "get_winner.")
