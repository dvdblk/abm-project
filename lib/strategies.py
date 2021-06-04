import numpy as np


class BaseStrategy:
    """Base class for strategy implementations"""

    def __init__(self, rng, memory_size):
        self.rng = rng
        self.memory_size = memory_size
        self.strategy_vector = None

        # Set the scores to 0
        self.reset_score()

        # Initialize the strategy vector
        self._initialize_strategy_vector()

    def _initialize_strategy_vector(self):
        """Must be implemented by subclasses"""
        raise NotImplementedError

    def get_strategy_vector(self):
        """Return the strategy vector"""
        return self.strategy_vector

    def reset_score(self):
        """Resets the score of this strategy to 0"""
        self.score = 0


class DefaultStrategy(BaseStrategy):
    """Default (random action) strategy."""

    def _initialize_strategy_vector(self):
        strategy_size = 2**self.memory_size
        self.strategy_vector = 2*self.rng.integers(2, size=strategy_size)-1


class AlwaysOneStrategy(BaseStrategy):
    """Strategy that always results in choosing 1 as the action"""

    def _initialize_strategy_vector(self):
        self.strategy_vector = np.ones(2**self.memory_size)


class FiftyFiftyStrategy(BaseStrategy):
    """50/50 chance to choose 1 or -1 as the action"""

    def _initialize_strategy_vector(self):
        strategy_vector = -np.ones(2**self.memory_size)
        strategy_vector[:2**(self.memory_size-1)] = 1
        self.strategy_vector = strategy_vector
