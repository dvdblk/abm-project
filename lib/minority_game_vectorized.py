import numpy as np
from lib.error import MinorityGameError
from typing import List


class VectorizedMinorityGame:
    """
    Vectorized version of Minority Game in order to run the simulation for a large amount of timesteps.

    This class is used for plotting volatility as a function of alpha and obtaining a peaky global minimum
    due to the large number of time steps.
    """

    def __init__(self, n_agents: int, m=2, max_history=50, rng=None) -> None:
        if n_agents % 2 == 0:
            raise MinorityGameError("Number of agents should be odd")
        self.n_agents = n_agents

        # Random Number Generator
        self.rng = rng or np.random.default_rng()

        # Memory
        self.m = m

        # Number of strategies
        self.S = 2

        # Max history amount
        self.MAX_HISTORY = max_history

        self._reset_game_state()

    def _reset_game_state(self):
        self.strategies = 2*np.random.randint(2, size=(self.n_agents, self.S, 2**self.m))-1

    def simulate_game(self, max_steps=5000):
        """Simulate a Minority Game

        Args:
            max_steps (int): the maximum number of time steps this game is
                             allowed to run for

        Returns:
            times (np.array): 1D array of timesteps, useful for plotting
            attendances (np.array): 1D array of A_t for each time step
            mean_A_t (np.array): 1D array of mean A_t over time
            vol_A_t (np.array): 1D array of the variance of A_t over time
        """
        attendances, times = np.zeros((2, max_steps))

        # Equations from https://arxiv.org/pdf/cond-mat/9909265.pdf
        mu_t = np.random.randint(2**self.m)

        # Eq (3)
        omega_i_mu = (self.strategies[:, 0, :] + self.strategies[:, 1, :]) / 2
        xi_i_mu = (self.strategies[:, 0, :] - self.strategies[:, 1, :]) / 2

        # \Omega = \sum_{i=1} \omega_{\mu}
        omega = np.sum(omega_i_mu, axis=0)
        # Difference between strategy scores
        delta_i = np.zeros(self.n_agents)

        for t in range(max_steps):
            # Eq (9)
            s_i = np.sign(delta_i)

            # Replace zeros signs with a random action
            s_i[np.argwhere(s_i == 0)] = 2*np.random.randint(2)-1

            # Eq (5)
            A_t = omega[mu_t] + np.sum(np.multiply(xi_i_mu[:, mu_t], s_i))

            # Eq (8)
            # *2 has to be here to cancel out the /2 from Eq (3)
            delta_i = delta_i - np.sign(A_t) * xi_i_mu[:, mu_t] * 2

            mu_t = int(np.mod((2*(mu_t+1) + (-np.sign(A_t)-1)/2), 2**self.m)-1)
            # In case mu_t ends up being -1 because of overflow
            if mu_t == -1:
                mu_t = 2**self.m-1

            times[t] = t
            attendances[t] = A_t

        return times, attendances

