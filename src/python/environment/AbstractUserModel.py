# KH: 07/10/2012


class AbstractUserModel:
    """Defines an abstract base class for user models."""

    def get_clicks(self, result_list, labels):
        raise NotImplementedError("Derived class needs to implement "
            "get_clicks.")
