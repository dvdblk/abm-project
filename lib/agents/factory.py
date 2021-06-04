from typing import List, Type
from lib.agents.agent import Agent
from lib.memory import BaseMemoryGenerator
from numpy.random import Generator


class AgentFactory:
    """Factory class for creating agents"""

    def __init__(self, agent_cls: Type[Agent], agent_kwargs: dict, memory_generator: BaseMemoryGenerator) -> None:
        """
        Args:
            agent_cls (Type[Agent]): the class of the agents that will be
                                         created by this factory
            agent_kwargs (dict): the key value arguments that will be passed to
                                 every Agent initializer
            memory_generator (BaseMemoryGenerator): the memory generator used to generate
                                                    memories for each strategy
        """
        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs
        self.memory_generator = memory_generator

    def create_agents(self, n_agents: int, rng: Generator) -> List[Agent]:
        """
        Create n_agents agents of type self.agent_cls

        Args:
            n_agents (int): the number of agents to create
            rng (np.random.Generator): the generator used for randomness
                                       in each agent

        Returns:
            agents (list): list of agents
        """
        agents = []
        # Get memories for each agent
        # (the values can be different if the distribution of memory is
        # not uniform...)
        memories = self.memory_generator.generate(n_agents)

        for i in range(n_agents):
            # Create agent with args
            kwargs = self.agent_kwargs | {"memory_size": memories[i], "rng": rng}
            agents.append(
                self.agent_cls(**kwargs)
            )

        # Return the new agents
        return agents
