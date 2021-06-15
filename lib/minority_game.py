import numpy as np
from numpy.lib.arraysetops import isin
from lib.error import MinorityGameError
from typing import List


class MinorityGame:
    """Base class for running a minority game (MG) simulation."""

    def __init__(self, n_agents: int, factory_dict, max_history=50, rng=None) -> None:
        """
        Args:
            n_agents (int): the number of agents `N`, must be odd
            factory_dict (dict): a dictionary that contains the AgentFactories,
                                 keys are the fractions of total_agents to produce
                                 from the AgentFactories
            max_history (int): the maximum history that will be kept in the game memory
            rng (Generator): the random number generator used in the game
        """
        if n_agents % 2 == 0:
            raise MinorityGameError("Number of agents should be odd")
        self.n_agents = n_agents

        # Random Number Generator
        self.rng = rng or np.random.default_rng()

        # Agents
        total_agent_frac = np.sum(list(factory_dict.keys()))
        if total_agent_frac != 1:
            raise MinorityGameError(
                f"The sum of agent fractions must be equal to 1 (current value = {total_agent_frac})"
            )

        agents = []
        for agent_frac, factory in factory_dict.items():
            n_new_agents = int(n_agents * agent_frac)
            new_agents = factory.create_agents(n_new_agents, self.rng)
            agents += new_agents
        self.agents = agents

        # Max history amount
        self.MAX_HISTORY = max_history

        self._reset_game_state()

    def _reset_agents_strategies(self):
        """Reset the strategies of each agent."""
        for agent in self.agents:
            agent.reset_strategies()

    def _reset_game_state(self):
        """
        Resets the agents history (current memory of each agent), strategies
        and strategy scores.
        """
        self.history = 2*self.rng.integers(2, size=self.MAX_HISTORY)-1

        self._reset_agents_strategies()

        for agent in self.agents:
            agent.reset_strategy_scores()

    def _update_scores_and_history(self, A_t: int, actions: List[int]):
        """
        Compute the round winner, update history and strategy scores for each
        agent by mu_t

        Args:
            A_t (int): the attendance (sum of actions of each agent)
            mu_t (int): number whose binary representation encodes the current
                        history, used to index the agent's strategy
        """
        # Get the minority (round winner)
        round_winner = -1 if A_t > 0 else 1

        # For each agent update his strategy scores
        for i, agent in enumerate(self.agents):
            agent_action = actions[i]
            agent.update_scores(
                round_winner,
                agent_action,
                self.history
            )

        # Update history
        # Roll the current history to the left by one
        # [1, -1, 1] -> [-1, 1, 1]
        self.history = np.roll(self.history, -1)
        # Replace the last element with the current round winner
        self.history[-1] = round_winner

    def _game_step(self):
        """Runs a simulation of one round of MG."""
        # The actions of each agent during this round
        actions = np.zeros(self.n_agents)

        # Find the best performing strategy of each agent
        # by their strategy score and determine the action
        # they want to take based on the game memory (history)
        for i, agent in enumerate(self.agents):
            a_t = agent.choose_action(self.history)
            actions[i] = a_t

        # Get the attendance / sum of all actions
        A_t = np.sum(actions)

        # Update scores and history accordingly
        self._update_scores_and_history(A_t, actions)

        return A_t

    def volatility(self, A_t):
        """Compute the volatility / variance of A_t

        Args:
            A_t (np.array): a 1D array containing A_t values over
                            multiple repetitions / time
        """
        return A_t.var()

    def total_success_rate(self, agent_cls=None):
        """
        Compute total success rate by taking the mean of success
        rates of each agent.

        Args:
            agent_cls: the filter class to use while computing
                       the total success rate. If None is used
                       (default) the success rate is computed as
                       the average between all agents.
        """
        succes_rates = np.zeros(len(self.agents))
        for i, agent in enumerate(self.agents):
            if agent_cls is not None:
                if not isinstance(agent, agent_cls):
                    continue
            succes_rates[i] = agent.success_rate()
        return succes_rates.mean()

    def simulate_game(self, max_steps=500):
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
        self._reset_game_state()

        times, attendances, mean_A_t, vol_A_t, = np.zeros((4, max_steps))

        # Run the game
        for t in range(max_steps):
            A_t = self._game_step()

            # Save for plotting
            times[t] = t
            attendances[t] = A_t

            # Take only the first t+1 attendances to avoid
            # averaging over the entire max_steps
            attendances_until_now = attendances[:t+1]

            # TODO: Check if the mean / variance is computed properly
            # First paragraph of section 4 in The intro guide mentions
            # that the average should be "time average for long times and
            # an average over possible realizations"
            #   Probably need to add another average every ~50 time steps?
            mean_A_t[t] = attendances_until_now.mean()
            vol_A_t[t] = self.volatility(attendances_until_now)

        return times, attendances, mean_A_t, vol_A_t
