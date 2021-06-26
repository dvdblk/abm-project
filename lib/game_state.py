

class GameState:
    """Cache for computing mu_t"""

    def __init__(self, history):
        self.history = history

        # Cache
        self.mu_dict = dict()

    def _compute_mu_t(self, m):
        """
        Compute mu_t from current history with a window of memory m.
        """
        binary_history = ((self.history[-m:] + 1) / 2).astype(int)
        return int("".join(map(str, binary_history)), 2)

    def get_mu_t(self, m):
        """
        Return mu_t from cache if it exists otherwise compute it.
        """
        if existing_mu_t := self.mu_dict.get(m, None):
            return existing_mu_t
        else:
            new_mu_t = self._compute_mu_t(m)
            self.mu_dict[m] = new_mu_t
            return new_mu_t

