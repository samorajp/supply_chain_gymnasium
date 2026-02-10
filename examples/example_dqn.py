from gymnasium.wrappers import FilterObservation, FlattenObservation
from rich import print
from stable_baselines3 import DQN

from supply_chain_gymnasium.agents.agent_fixed_order_level import (
    evaluate_fixed_order_level_agent,
)
from supply_chain_gymnasium.envs.gymnasium_warehouse_env import WarehouseEnv

if __name__ == "__main__":
    demand_sequence = [0, 0, 0, 2, 0, 3, 4] * 52
    env = WarehouseEnv(
        daily_demand_sequences=[demand_sequence],
        allowed_order_quantities=[0, 2],
        target_service_level=0.95,
        initial_inventory_level=0,
        lead_time=1,
        past_demands_flag_length=1,
        lowest_inventory_level=-10,
        highest_inventory_level=10,
        render_mode=None,
    )
    for order_level in range(-4, 9):
        total_reward = evaluate_fixed_order_level_agent(env, order_level)
        print(
            f"Order Level: {order_level}, Total Reward: {total_reward}, Per day: {total_reward / len(demand_sequence):.2f}"
        )

    env = FlattenObservation(
        FilterObservation(
            env,
            filter_keys=[
                "days_left",
                "inventory_position",
            ],
        )
    )
    agent = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.01,
        n_steps=500,
        gamma=1,
        verbose=1,
        tensorboard_log="./dqn_warehouse_tensorboard/",
    )
    agent.learn(total_timesteps=100000)
