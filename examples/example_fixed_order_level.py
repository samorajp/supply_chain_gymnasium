from random import randint

from rich import print

from supply_chain_gymnasium.agents.agent_fixed_order_level import (
    evaluate_fixed_order_level_agent,
)
from supply_chain_gymnasium.envs.gymnasium_warehouse_env import WarehouseEnv

if __name__ == "__main__":

    def shift_list(lst, n):
        return lst[n:] + lst[:n]

    demand_sequences = [
        [0, 0, 0, 2, 0, 3, 4] * 52,
        # shift_list([0, 0, 0, 2, 0, 3, 4], 1) * 20,
        # shift_list([0, 0, 0, 2, 0, 3, 4], 2) * 20,
        # shift_list([0, 0, 0, 2, 0, 3, 4], 3) * 20,
        # shift_list([0, 0, 0, 2, 0, 3, 4], 4) * 20,
        # shift_list([0, 0, 0, 2, 0, 3, 4], 5) * 20,
        # shift_list([0, 0, 0, 2, 0, 3, 4], 6) * 20,
    ]
    env = WarehouseEnv(
        daily_demand_sequences=demand_sequences,
        allowed_order_quantities=[0, 2],
        target_service_level=0.95,
        initial_inventory_level=randint(0, 5),
        lead_time=1,
        past_demands_flag_length=1,
        lowest_inventory_level=-10,
        highest_inventory_level=10,
        render_mode="human",
    )

    for order_level in range(-4, 9):
        total_reward = evaluate_fixed_order_level_agent(env, order_level)
        print(
            f"Order Level: {order_level}, Total Reward: {total_reward}, Per day: {total_reward / len(demand_sequences[0]):.2f}"
        )
