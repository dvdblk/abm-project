from lib.minority_game import MinorityGame
from lib.agents.agent import Agent, StrategyUpdatingAgent
from lib.agents.factory import AgentFactory
from lib.strategies import AlwaysOneStrategy, DefaultStrategy, FiftyFiftyStrategy
from lib.memory import UniformMemoryGenerator
from lib.plots import default_plot


def simulate_simple_game():
    """
    Simple function that simulates and plots a default minority game.
    """
    # Start the game right away
    times, attendances, _, _ = MinorityGame(
        n_agents=201,
        factory_dict={
            0.3: AgentFactory(
                Agent,
                agent_kwargs=dict(strategy_clss=[DefaultStrategy, DefaultStrategy]),
                memory_generator=UniformMemoryGenerator(m=3)
            ),
            0.7: AgentFactory(
                StrategyUpdatingAgent,
                agent_kwargs=dict(
                    strategy_clss=[FiftyFiftyStrategy, DefaultStrategy, AlwaysOneStrategy],
                    strategy_update_rate=0.8
                ),
                memory_generator=UniformMemoryGenerator(m=3)
            ),
        }
    ).simulate_game()

    default_plot(times, attendances)
