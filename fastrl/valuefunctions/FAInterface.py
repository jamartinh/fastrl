# interface that express the function approximator for
# Reinforcement Learning Algorithms.


class FARL:
    def __call__(self, s, a=None):
        """ implement here the returned Qvalue of state (s) and action(a)
            e.g. Q.get_value(s,a) is equivalent to Q(s,a)
        """
        if a is None:
            return self.get_value(s)

        return self.get_value(s, a)

    def get_value(self, s, a=None):
        """ Return the Q value of state (s) for action (a)

        """
        raise NotImplementedError

    def update(self, s, a, v):
        """ update action value for action(a)

        """
        raise NotImplementedError

    def update_all(self, s, v):
        """ update action value for action(a)

        """
        raise NotImplementedError

    def has_population(self):
        raise NotImplementedError

    def get_population(self):
        raise NotImplementedError

    def add_trace(self, *args):
        pass
