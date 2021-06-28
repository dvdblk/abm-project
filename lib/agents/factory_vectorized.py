import numpy as np
from lib.memory import BaseMemoryGenerator


class VectorizedAgentFactory:
    """
    Factory class for creating the agents' strategy vectors.
    """

    def __init__(self, memory_generator: BaseMemoryGenerator) -> None:
        """
        Args:
            memory_generator (BaseMemoryGenerator): the memory generator
                                                    that will be used to allocate
                                                    agents' memories
        """
        self.memory_generator = memory_generator

    def create_strategies(self, n_agents: int, rng: np.random.Generator):
        """
        Create 2 strategies for each agent.

        Args:
            n_agents (int): the number of agents that this factory creates
            rng (np.random.Generator): the generator class that is used for randomness
                                       (supplied by the VectorizedMG)
        """
        # Create memory size for each agent
        # (both strategies of an agent have the same memory length)
        memory_sizes = self.memory_generator.generate(n_agents)
        assert (memory_sizes >= 1).all(), "Memory size must not be a negative number."

        # Get the highest memory to create the full strategy vector
        highest_m = memory_sizes.max()
        strategies = np.zeros((n_agents, 2, 2**highest_m))

        # For each agent create two strategies
        n_strats = 2
        for i in range(n_agents):
            m = memory_sizes[i]
            agent_strats = 2*rng.integers(2, size=(n_strats, 2**m))-1

            # Adjusted for highest m
            # Repeat the strategy vector until the length is equal to 2**highest_m
            # This way we can still use the vectorized version of the MG.
            agent_strats_adjusted = np.tile(agent_strats, 2**(highest_m-m))

            strategies[i] = agent_strats_adjusted

        return strategies, memory_sizes, highest_m


class VectorizedStrategyUpdatingAgentFactory(VectorizedAgentFactory):
    """Variation of VectorizedAgentFactory which contains strategy updating params."""

    def __init__(self, memory_generator, strategy_update_rate=0.5, strategy_update_fraction=1) -> None:
        super(VectorizedStrategyUpdatingAgentFactory, self).__init__(memory_generator)
        self.update_rate = strategy_update_rate
        self.update_fraction = strategy_update_fraction