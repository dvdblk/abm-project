import numpy as np
from lib.error import MinorityGameError


class Agent:
    """Base class for agents"""

    def __init__(self, rng=None, memory_size=3, strategy_clss=[]):
        self.rng = rng
        self.m = memory_size
        strategies = []
        for strategy_cls in strategy_clss:
            strategies.append(strategy_cls(rng, memory_size))
        self.strategies = strategies

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

    def _compute_mu_t(self, strategy, history):
        binary_history = ((history[-strategy.memory_size:] + 1) / 2).astype(int)
        return int("".join(map(str, binary_history)), 2)

    def update_scores(self, round_winner, history):
        for i, strategy in enumerate(self.strategies):
            mu_t = self._compute_mu_t(strategy, history)
            if strategy.strategy_vector[mu_t] == round_winner:
                strategy.score += 1
            else:
                strategy.score -= 1

    def choose_action(self, history):
        # Find the highest strategy score index
        scores = self._scores()
        best_strategies = np.argwhere(scores == np.amax(scores)).flatten()
        # Sometimes the strategies might have the same score. If that's the
        # case, rng.choice will choose one randomly. If there is only one strategy
        # rng.choice will have only one choice.
        best_strategy_idx = self.rng.choice(best_strategies)
        best_strategy_mu_t = self._compute_mu_t(self.strategies[best_strategy_idx], history)

        # Get the outcome of this strategy and use it as agent's action
        a_t = self.strategies[best_strategy_idx].strategy_vector[best_strategy_mu_t]
        return a_t


class StrategyUpdatingAgent(Agent):
    """Agent that can update his strategy"""

    def __init__(self, rng=None, memory_size=3, strategy_clss=[], strategy_update_rate=0.5):
        super().__init__(rng, memory_size, strategy_clss)

        if (strategy_update_rate < 0 or strategy_update_rate >= 1):
            raise MinorityGameError("Strategy update rate should be in the range [0, 1)")
        self.gamma = strategy_update_rate

    def update_scores(self, round_winner, history):
        super().update_scores(round_winner, history)

        # Update the strategies if needed
        if self.gamma is not None:
            if self.rng.uniform() <= self.gamma:
                # Reset strategies but keep the scores
                self.reset_strategies()
