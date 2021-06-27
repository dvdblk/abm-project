import numpy as np
import pandas as pd
from lib.error import MinorityGameError
from collections import OrderedDict
from lib.agents.factory_vectorized import VectorizedStrategyUpdatingAgentFactory


class VectorizedMinorityGame:
    """
    Vectorized version of Minority Game. Speeds up the simulation exponentially for a large amount of timesteps.
    """

    def __init__(self, n_agents: int, factory_dict, rng=None) -> None:
        """
        Args:
            n_agents (int): the number of agents `N`, must be odd
            factory_dict (dict): a dictionary that contains the VectorizedAgentFactories,
                                 keys are the fractions of total_agents to produce
                                 from the AgentFactories
            rng (Generator): the random number generator used in the game
        """
        if n_agents % 2 == 0:
            raise MinorityGameError("Number of agents should be odd")
        self.n_agents = n_agents

        # Random Number Generator
        self.rng = rng or np.random.default_rng(0)

        self.factory_dict = factory_dict

        self._reset_game_state()

    def _reset_game_state(self):
        """
        Creates agents and saves their parameters
        (e.g. number of agents in each group based on memory size)
        """
        # Strategies for each group formed by the factories
        strats_groups = []
        # The memory size for each agent in the game
        self.memory_sizes = []
        highest_m = 0

        # Last agent index, used for keeping track of indices of agents in
        # each memory group.
        last_agent_idx = 0

        for i, (agent_frac, factory) in enumerate(self.factory_dict.items()):
            if i == 0 and agent_frac != 1:
                # One of the groups needs to be rounded up in order to have
                # an odd number of total agents.
                # So we take the first group if it's not the only group.
                n_new_agents = np.ceil(self.n_agents * agent_frac).astype(int)
            else:
                n_new_agents = int(self.n_agents * agent_frac)
            strats, m_size, m = factory.create_strategies(n_new_agents, self.rng)

            if m > highest_m:
                highest_m = m

            strats_groups.append(strats)
            self.memory_sizes += m_size.tolist()

            # Save the indices of agents belonging to this factory.
            factory.agent_idxs = np.arange(
                last_agent_idx,
                last_agent_idx + strats.shape[0]
            )

        # Save the largest memory for later.
        self.highest_m = highest_m
        self.memory_sizes = np.array(self.memory_sizes)
        self.strategies = np.zeros((0, 2, 2**highest_m))

        # Each strategy has to be tiled to fit into the largest memory
        # out of all the groups.
        # For example if there would be two groups (80% uniform M=2 and 20% normal distribution)
        # the normal distribution can produce M >> 2 in which case the 80% of strategies would have
        # to be tiled to fit the highest M
        for i, strats in enumerate(strats_groups):
            m = np.log2(strats.shape[2]).astype(int)
            # Adjusted for highest m
            self.strategies = np.concatenate(
                (
                    self.strategies,
                    np.tile(strats, 2**(highest_m-m))
                )
            )

    def simulate_game(self, max_steps=5000, return_individual=False, return_mean=False, return_vol=False):
        """Simulate a Minority Game

        Args:
            max_steps (int): the maximum number of time steps this game is
                             allowed to run for
            return_individual (bool): whether to return the individual volatilities
                                      for each group of agents split by m
            return_mean (bool): whether to return the mean of attendances after
                                each round (time step)
            return_vol (bool): whether to return the variance of attendances
                               after each round (tim step), if return_individual
                               is True then this value is also set to True

        Returns:
            times (np.array): 1D array of timesteps, useful for plotting
            attendances (np.array): 1D array of A_t for each time step
            mean_A_t (np.array): 1D array of mean A_t over time
            vol_A_t (np.array): 1D array of the variance of A_t over time
            n_agents_m (np.array): 2D array of the nr of agents for each memory size group
            m_list (np.array): 1D array of memory sizes of each agent
            vol_A_t_individual_m_df: 2D array of volatility per memory group
        """
        if return_individual:
            return_vol = True

        # Create empty vectors for results and actions of each individual agent
        attendances, times, vol_A_t, mean_A_t = np.zeros((4, max_steps))
        actions_t = np.zeros((max_steps, self.n_agents))

        m_list = self.memory_sizes
        n_agents_m = np.array(np.unique(m_list, return_counts=True)).T
        n_agents_m_df = pd.DataFrame(n_agents_m, columns=["index", "count"])

        # Keep track of indices of agents that belong to each memory group.
        # e.g. agents with m=5 have indices [22, 48, 91, 92]
        agent_idxs_per_m = OrderedDict()
        for i, m in enumerate(n_agents_m[:, 0]):
            agent_idxs_per_m[i] = np.argwhere(m_list == m).squeeze()

        # The number of unique m values
        # in case the memory is uniform for all agents
        # this would be equal to 1
        n_unique_ms = len(agent_idxs_per_m.keys())

        # Equations from https://arxiv.org/pdf/cond-mat/9909265.pdf
        mu_t = self.rng.integers(2**self.highest_m)

        # Eq (3)
        omega_i_mu = (self.strategies[:, 0, :] + self.strategies[:, 1, :]) / 2
        xi_i_mu = (self.strategies[:, 0, :] - self.strategies[:, 1, :]) / 2

        # \Omega = \sum_{i=1} \omega_{\mu}
        omega = np.sum(omega_i_mu, axis=0)

        # Difference between strategy scores
        delta_i = np.zeros(self.n_agents)

        # Check whether factory dictionaries contain a VectorizedStrategyUpdatingAgentFactory
        update_strats = any(
            isinstance(fct, VectorizedStrategyUpdatingAgentFactory) for fct in self.factory_dict.values()
        )

        for t in range(max_steps):
            if update_strats:
                 # Eq (3)
                omega_i_mu = (self.strategies[:, 0, :] + self.strategies[:, 1, :]) / 2
                xi_i_mu = (self.strategies[:, 0, :] - self.strategies[:, 1, :]) / 2

                # \Omega = \sum_{i=1} \omega_{\mu}
                omega = np.sum(omega_i_mu, axis=0)

            # Eq (9)
            s_i = np.sign(delta_i)

            # Replace zeros signs with a random action
            s_i[np.argwhere(s_i == 0)] = 2*self.rng.integers(2)-1

            # Eq (5)
            A_t = omega[mu_t] + np.sum(np.multiply(xi_i_mu[:, mu_t], s_i))

            # Eq (8)
            # *2 has to be here to cancel out the /2 from Eq (3)
            delta_i = delta_i - np.sign(A_t) * xi_i_mu[:, mu_t] * 2

            mu_t = int(np.mod((2*(mu_t+1) + (-np.sign(A_t)-1)/2), 2**self.highest_m)-1)
            # In case mu_t ends up being -1 because of overflow
            if mu_t == -1:
                mu_t = 2**self.highest_m-1

            # Save the results of this round / time step
            times[t] = t
            attendances[t] = A_t
            if return_vol:
                vol_A_t[t] = np.var(attendances[:t+1])
            if return_mean:
                mean_A_t[t] = np.mean(attendances[:t+1])
            actions_t[t] = omega_i_mu[:, mu_t] + xi_i_mu[:, mu_t] * s_i

            # Update agents strategies if needed
            if update_strats:
                for fct in self.factory_dict.values():
                    if isinstance(fct, VectorizedStrategyUpdatingAgentFactory):
                        agent_mask = fct.agent_idxs

                        # random boolean mask for which values will be changed
                        rate_mask = self.rng.choice(
                            [0, 1],
                            size=(self.strategies[agent_mask].shape[0]),
                            p=((1 - fct.update_rate), fct.update_rate)
                        ).astype(np.bool)

                        fraction_mask = self.rng.choice(
                            [0, 1],
                            size=self.strategies[agent_mask][rate_mask].shape,
                            p=((1 - fct.update_fraction), fct.update_fraction)
                        ).astype(np.bool)

                        # random matrix the same shape of your data
                        r = self.rng.choice([-1, 1], size=self.strategies[agent_mask][rate_mask].shape)

                        # use your mask to replace values in your input array
                        self.strategies[agent_mask][rate_mask][fraction_mask] = r[fraction_mask]

        # Build the result
        result = (times, attendances)

        if return_individual:
            # Create the A_t's and volatilities for each group of agents with equal M
            # actions_t_m.shape = (number of unique memory sizes, total game steps)
            actions_t_m = np.zeros((n_unique_ms, max_steps))
            for m, idxs in agent_idxs_per_m.items():

                if actions_t[:, idxs].ndim > 1:
                    # Sum the actions if there are more than 1 agents in this group
                    actions_t_m[m] = np.sum(actions_t[:, idxs], axis=1)
                else:
                    # otherwise just take the vector
                    actions_t_m[m] = actions_t[:, idxs]

            # Compute the volatility
            # Note that the volatility is not computed by averaging the cumulative
            # volatilities since the first game step but only as the volatility over
            # all the steps.
            vol_A_t_individual_m_df = actions_t_m.var(axis=1)

            if return_mean:
                result += (mean_A_t,)

            result += (vol_A_t, n_agents_m_df, m_list, vol_A_t_individual_m_df)
        else:
            if return_mean:
                result += (mean_A_t,)

            if return_vol:
                result += (vol_A_t,)

        return result
