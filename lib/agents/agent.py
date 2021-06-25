import numpy as np
from lib.error import MinorityGameError


class Agent:
    """Base class for agents"""

    def __init__(self, rng, memory_size=3, strategy_clss=[]):
        """
        Args:
            rng (Generator): the random number generator
            memory_size (int): the memory size given to this agent by some
                               memory generator
            strategy_clss: the classes of strategies that this agent will use
        """
        self.rng = rng
        self.m = memory_size

        # Initialize the strategies
        strategies = []
        for strategy_cls in strategy_clss:
            new_strategy = strategy_cls(rng, memory_size)
            strategies.append(
                new_strategy
            )
        self.strategies = strategies

        self.win_history = []

    def _scores(self):
        scores = np.zeros(len(self.strategies))
        for i, strat in enumerate(self.strategies):
            scores[i] = strat.score
        return scores

    def reset_strategy_scores(self):
        for strat in self.strategies:
            strat.reset_score()

    def reset_strategies(self):
        for strat in self.strategies:
            strat._initialize_strategy_vector()

    def update_scores(self, round_winner, action_taken, game_state):
        """
        Update the scores depending on the round winner and
        history.

        Args:
            round_winner (int): 1 or -1, the last winning action
            action_taken (int): 1 or -1, the action this agent took
                                in last round
            game_state (GameState): GameState object, used to get mu_t
                                    for the chosen strategy
        """
        # Update the scores
        for i, strategy in enumerate(self.strategies):
            mu_t = game_state.get_mu_t(strategy.memory_size)
            if strategy.strategy_vector[mu_t] == round_winner:
                strategy.score += 1
            else:
                strategy.score -= 1

        # Update history based on actions
        self.win_history.append(
            round_winner == action_taken
        )

    def success_rate(self):
        """
        Compute the success rate of this agent based on win history

        Returns:
            float: in range [0, 1]
        """
        return np.array(self.win_history).mean()

    def get_best_strategy(self):
        # Find the highest strategy score index
        scores = self._scores()
        best_strategies = np.argwhere(scores == np.amax(scores)).flatten()
        # Sometimes the strategies might have the same score. If that's the
        # case, rng.choice will choose one randomly. If there is only one strategy
        # rng.choice will have only one choice.
        if len(best_strategies) > 1:
            best_strategy_idx = self.rng.choice(best_strategies)
        else:
            best_strategy_idx = best_strategies[0]
        return self.strategies[best_strategy_idx]


    def choose_action(self, game_state):
        best_strategy = self.get_best_strategy()
        m = best_strategy.memory_size
        best_strategy_mu_t = game_state.get_mu_t(m)

        # Get the outcome of this strategy and use it as agent's action
        a_t = best_strategy.strategy_vector[best_strategy_mu_t]
        return a_t


class StrategyUpdatingAgent(Agent):
    """Agent that can update his strategy"""

    def __init__(self, rng, memory_size=3, strategy_clss=[], strategy_update_rate=0.5, strategy_update_fraction=1):
        """
        Args:
            rng (Generator): the random number generator
            memory_size (int): the memory size given to this agent by some
                               memory generator
            strategy_clss: the classes of strategies that this agent will use
            strategy_update_rate (float): the probability at which an agent
                                          updates all of his strategies at the
                                          end of a round
            strategy_update_fraction (float): the fraction of strategy vector
                                              that will be updated on strategy
                                              updates (defaults to 1 = entire
                                              strategy vector is updated / reset)
        """
        super().__init__(rng, memory_size, strategy_clss)

        if (strategy_update_rate < 0 or strategy_update_rate > 1):
            raise MinorityGameError("Strategy update rate should be in the range [0, 1)")
        self.update_rate = strategy_update_rate

        if (strategy_update_fraction < 0 or strategy_update_fraction > 1):
            raise MinorityGameError("Strategy update fraction should be in the range [0, 1)")
        self.update_fraction = strategy_update_fraction

    def _update_strategies(self):
        """
        Update the strategies by taking into account the fraction
        of strategy vector to update.
        """
        for strategy in self.strategies:
            strategy.update_strategy_vector(self.update_fraction)

    def update_scores(self, round_winner, action_taken, history):
        super().update_scores(round_winner, action_taken, history)

        # Update the strategies if needed
        if self.rng.uniform() <= self.update_rate:
            # Change only some strategies
            self._update_strategies()



