import numpy as np
import matplotlib.pyplot as plt


class MinorityGameError(Exception):
    """Error class for any errors related to MinorityGame"""
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


class MinorityGame:
    """Base class for running a minority game (MG) simulation."""

    def __init__(self, n_agents: int, n_strategies: int, memory_size: int,
                 strategy_update_rate=None, seed=None) -> None:
        """
        Args:
            n_agents (int): the number of agents `N`, must be odd
            memory_size (int): the memory of each agent `m` which is the
                               amount of winning groups each agent can remember
            n_strategies (int): the amount of strategies each agent can have
                                (must be larger than 2)
            strategy_update_rate (float): float in [0, 1) that determines
                                          how often agents update their strategies
            seed (int): the seed used in the random number generator
        """
        if n_agents % 2 == 0:
            raise MinorityGameError("Number of agents should be odd")
        self.n_agents = n_agents

        self.memory_size = memory_size

        if n_strategies < 2:
            raise MinorityGameError("Number of strategies should be >= 2")
        self.n_strategies = n_strategies

        if strategy_update_rate is not None and (strategy_update_rate < 0 or strategy_update_rate >= 1):
            raise MinorityGameError("Strategy update rate should be in the range [0, 1)")
        self.gamma = strategy_update_rate

        # Random Number Generator
        self.rng = np.random.default_rng(seed)

        # Set the initial game state immediately
        self._reset_game_state()

    def _reset_agents_strategies(self):
        """Reset the strategies of each agent."""
        # Create a 3D tensor for strategies of size:
        # (agents, strategies, number of possible encodings of strategy inputs = 2**m)
        strategy_size = (self.n_agents, self.n_strategies, 2**self.memory_size)

        # Reset
        self.strategies = 2*self.rng.integers(2, size=strategy_size)-1

    def _reset_game_state(self):
        """
        Resets the agents history (current memory of each agent), strategies
        and strategy scores.
        """
        self.history = 2*self.rng.integers(2, size=self.memory_size)-1

        self._reset_agents_strategies()

        self.strategy_scores = np.zeros((self.n_agents, self.n_strategies))

    def _update_scores_and_history(self, A_t: int, mu_t: int):
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

        # Get the winning and losing strategies for each agent
        win_strats = self.strategies.transpose(2, 0, 1)[mu_t] == round_winner
        lose_strats = self.strategies.transpose(2, 0, 1)[mu_t] != round_winner

        # Update the indexes of winning and losing strategies
        self.strategy_scores[np.nonzero(win_strats)] += 1
        self.strategy_scores[np.nonzero(lose_strats)] -= 1

        # Update the strategies if needed
        if self.gamma is not None:
            if self.rng.uniform() <= self.gamma:
                # Reset strategies but keep the scores
                self._reset_agents_strategies()

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

        # Compute \mu at time t to properly index a strategy
        # based on the recent history
        binary_history = ((self.history + 1) / 2).astype(int)
        mu_t = int("".join(map(str, binary_history)), 2)

        # Find the best performing strategy of each agent
        # by their strategy score and determine the action
        # they want to take with \mu
        for agent_i in range(self.n_agents):
            # Get agent's strategy scores
            agent_i_strats = self.strategy_scores[agent_i]
            # Find the highest strategy score index
            best_strategy_idx = np.argmax(agent_i_strats)
            # Get the outcome of this strategy and use it as agent's action
            a_t = self.strategies[agent_i][best_strategy_idx][mu_t]
            actions[agent_i] = a_t

        # Get the attendance / sum of all actions
        A_t = np.sum(actions)

        # Update scores and history accordingly
        self._update_scores_and_history(A_t, mu_t)

        return A_t

    def volatility(self, A_t):
        """Compute the volatility / variance of A_t

        Args:
            A_t (np.array): a 1D array containing A_t values over
                            multiple repetitions / time
        """
        return A_t.var()

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


if __name__ == "__main__":
    # Start the game right away
    times, attendances, _, _ = MinorityGame(
        n_agents=301,
        n_strategies=2,
        memory_size=2,
    ).simulate_game()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axhline(y=0, color="k", linestyle="--")
    ax.plot(times, attendances, label=r"$A(t)$")
    ax.set_xlabel("t")
    ax.set_ylabel("A(t)")

    plt.show()
