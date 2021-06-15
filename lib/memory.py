import numpy as np


class BaseMemoryGenerator:
    """
    Abstract class that should be subclassed when implementing
    a new memory generator.
    """
    def __init__(self, m, rng=None):
        """
        Args:
            m (int): the default amount of memory (or the mean memory)
            rng (Generator): the random number generator that should be used
        """
        self.m = m
        self.rng = rng or np.random.default_rng()

    def generate(self, size):
        raise NotImplementedError


class UniformMemoryGenerator(BaseMemoryGenerator):
    """
    Samples memory from a uniform distribution.
    (All memories are of equal length...)
    """

    def generate(self, size):
        return np.ones(size, dtype=np.int32) * self.m
