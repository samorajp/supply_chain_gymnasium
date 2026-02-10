from collections import deque

import gymnasium as gym
import pygame

FPS = 1


class WarehouseEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        daily_demand_sequences: list[list[int]],
        allowed_order_quantities: list[int],
        target_service_level: float,
        initial_inventory_level: int,
        lead_time: int,
        past_demands_flag_length: int,
        initial_orders_in_transit: list[int] | None = None,
        too_low_service_daily_reward: float = 0,
        too_low_service_end_episode_reward: float = 0,
        holding_cost_per_unit: float = -1,
        backorder_cost_per_unit: float = -19,
        fulfilled_demand_reward_per_unit: float = 0,
        lowest_inventory_level: int = -50,
        highest_inventory_level: int = 50,
        out_of_bounds_inventory_level_reward: float = -1_000_000,
        step_survival_reward: float = 0,
        render_mode=None,
        episode_render_frequency: int = 1000,
    ):
        assert (
            initial_orders_in_transit is None
            or len(initial_orders_in_transit) == lead_time
        )
        assert too_low_service_daily_reward <= 0, (
            "Too low service daily reward should be non-positive."
        )
        assert too_low_service_end_episode_reward <= 0, (
            "Too low service end episode reward should be non-positive."
        )
        assert holding_cost_per_unit < 0, "Holding cost per unit should be negative."
        assert backorder_cost_per_unit <= 0, (
            "Backorder cost per unit should be non-positive."
        )
        if fulfilled_demand_reward_per_unit is not None:
            assert fulfilled_demand_reward_per_unit >= 0, (
                "Fulfilled demand reward per unit should be positive."
            )

        self.daily_demand_sequences = daily_demand_sequences
        self.allowed_order_quantities = allowed_order_quantities
        self.target_service_level = target_service_level
        self.initial_inventory_level = initial_inventory_level
        self.lead_time = lead_time
        self.past_demands_flag_length = past_demands_flag_length
        self.initial_orders_in_transit = initial_orders_in_transit or [0] * lead_time
        self.too_low_service_daily_reward = too_low_service_daily_reward
        self.too_low_service_end_episode_reward = too_low_service_end_episode_reward
        self.holding_cost_per_unit = holding_cost_per_unit
        self.backorder_cost_per_unit = backorder_cost_per_unit
        self.fulfilled_demand_reward_per_unit = fulfilled_demand_reward_per_unit
        self.lowest_inventory_level = lowest_inventory_level
        self.highest_inventory_level = highest_inventory_level
        self.out_of_bounds_inventory_level_reward = out_of_bounds_inventory_level_reward
        self.step_survival_reward = step_survival_reward
        self.action_space = gym.spaces.Discrete(len(self.allowed_order_quantities))
        self.observation_space = gym.spaces.Dict(
            {
                "inventory_level": gym.spaces.Box(
                    low=self.lowest_inventory_level,
                    high=self.highest_inventory_level,
                    shape=(1,),
                    dtype=int,
                ),
                "inventory_level_plus": gym.spaces.Box(
                    low=0,
                    high=self.highest_inventory_level,
                    shape=(1,),
                    dtype=int,
                ),
                "inventory_position": gym.spaces.Box(
                    low=self.lowest_inventory_level,
                    high=self.highest_inventory_level,
                    shape=(1,),
                    dtype=int,
                ),
                "delivery_flag": gym.spaces.Discrete(2),
                "past_demands_flag": gym.spaces.Discrete(2),
                "days_left": gym.spaces.Box(
                    low=0,
                    high=max(len(sequence) for sequence in self.daily_demand_sequences)
                    + 1,
                    shape=(1,),
                    dtype=int,
                ),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 1024  # The size of the PyGame window
        self.window = None
        self.clock = None
        self.episode_count = 0
        self.episode_render_frequency = episode_render_frequency
        self.current_demand_sequence_index = 0
        self.daily_demand_sequence = self.daily_demand_sequences[
            self.current_demand_sequence_index
        ]

    @property
    def service_level(self):
        return (
            self.total_fulfilled_demand / self.total_demand
            if self.total_demand > 0
            else 1.0
        )

    def _get_obs(self):
        return {
            "inventory_level": self.inventory_level,
            "inventory_level_plus": max(self.inventory_level, 0),
            "inventory_position": self.inventory_level + sum(self.orders_in_transit),
            "delivery_flag": int(self.orders_in_transit[0] > 0),
            "past_demands_flag": self.daily_demand_sequence[self.current_day] > 0,
            "days_left": len(self.daily_demand_sequence) - self.current_day - 1,
        }

    def _get_info(self):
        return {
            "end_of_day": self.current_day,
            "fulfilled_demand": self.total_fulfilled_demand,
            "total_demand": self.total_demand,
            "service_level": self.service_level,
            "past_demands": self.daily_demand_sequence[: self.current_day + 1],
            "upcoming_demands": self.daily_demand_sequence[max(0, self.current_day) :],
            "orders_in_transit": list(self.orders_in_transit),
            "inventory_position": self.inventory_level + sum(self.orders_in_transit),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.current_demand_sequence_index = (self.episode_count - 1) % len(
            self.daily_demand_sequences
        )
        self.daily_demand_sequence = self.daily_demand_sequences[
            self.current_demand_sequence_index
        ]
        self.current_day = 0
        self.episode_reward = 0
        self.total_demand = 0
        self.total_fulfilled_demand = 0
        self.orders_in_transit = deque(
            self.initial_orders_in_transit, maxlen=self.lead_time
        )
        self.inventory_level = self.initial_inventory_level

        if (
            self.render_mode == "human"
            and self.episode_count % self.episode_render_frequency == 0
        ):
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_day += 1
        daily_demand = self.daily_demand_sequence[self.current_day]
        self.total_demand += daily_demand

        self.inventory_level += self.orders_in_transit.popleft()
        units_held = max(self.inventory_level, 0)
        fulfilled_demand = min(daily_demand, units_held)
        self.total_fulfilled_demand += fulfilled_demand
        self.inventory_level -= daily_demand

        self.orders_in_transit.append(self.allowed_order_quantities[action])
        terminated = (
            self.current_day == len(self.daily_demand_sequence) - 1
        )  # len(self.daily_demand_sequence) - 1 is last index to read from daily_demand_sequence
        # so last index was read this step

        step_reward = 0
        step_reward += self.holding_cost_per_unit * max(self.inventory_level, 0)
        step_reward += self.backorder_cost_per_unit * max(-self.inventory_level, 0)
        step_reward += self.fulfilled_demand_reward_per_unit * fulfilled_demand

        if terminated:
            if self.service_level < self.target_service_level:
                step_reward += self.too_low_service_end_episode_reward
        elif self.service_level < self.target_service_level:
            step_reward += self.too_low_service_daily_reward

        truncated = False
        if self.inventory_level < self.lowest_inventory_level:
            self.inventory_level = self.lowest_inventory_level
            step_reward += self.out_of_bounds_inventory_level_reward
            truncated = True
        elif self.inventory_level > self.highest_inventory_level:
            self.inventory_level = self.highest_inventory_level
            step_reward += self.out_of_bounds_inventory_level_reward
            truncated = True

        if not truncated:
            step_reward += self.step_survival_reward

        self.episode_reward += step_reward

        if (
            self.render_mode == "human"
            and self.episode_count % self.episode_render_frequency == 0
        ):
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()

        return observation, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw warehouse building
        warehouse_width = 200
        warehouse_height = 300
        warehouse_x = (
            self.window_size // 2 - warehouse_width // 2
        )  # Center horizontally
        warehouse_y = self.window_size - warehouse_height - 50
        pygame.draw.rect(
            canvas,
            (100, 100, 100),
            pygame.Rect(warehouse_x, warehouse_y, warehouse_width, warehouse_height),
            3,
        )

        # Draw warehouse label
        font = pygame.font.Font(None, 24)
        warehouse_text = font.render("Warehouse", True, (0, 0, 0))
        canvas.blit(warehouse_text, (warehouse_x + 45, warehouse_y - 30))

        # Draw inventory level and incoming stock near warehouse
        info_font = pygame.font.Font(None, 20)
        incoming_stock = sum(self.orders_in_transit)
        inventory_info = info_font.render(
            f"Stock: {self.inventory_level}", True, (50, 50, 50)
        )
        incoming_info = info_font.render(
            f"Incoming: {incoming_stock}", True, (70, 130, 180)
        )
        canvas.blit(inventory_info, (warehouse_x + 10, warehouse_y + 10))
        canvas.blit(incoming_info, (warehouse_x + 10, warehouse_y + 30))

        # Draw inventory level as stacked boxes
        box_size = 30
        boxes_per_row = 5
        inventory_to_show = min(max(self.inventory_level, 0), 35)  # Show up to 35 boxes

        # Draw boxes inside the warehouse
        box_start_x = warehouse_x + 20
        box_start_y = warehouse_y + warehouse_height - 40

        num_boxes = inventory_to_show
        for i in range(num_boxes):
            row = i // boxes_per_row
            col = i % boxes_per_row
            box_x = box_start_x + col * (box_size + 5)
            box_y = box_start_y - row * (box_size + 5)

            # Draw box
            pygame.draw.rect(
                canvas,
                (139, 69, 19),  # Brown color for boxes
                pygame.Rect(box_x, box_y, box_size, box_size),
            )
            pygame.draw.rect(
                canvas,
                (101, 50, 15),
                pygame.Rect(box_x, box_y, box_size, box_size),
                2,
            )

        # Draw backorder warning if inventory is negative
        if self.inventory_level < 0:
            backorder_text = font.render(
                f"Backorder: {-self.inventory_level}", True, (255, 0, 0)
            )
            canvas.blit(
                backorder_text, (warehouse_x, warehouse_y + warehouse_height + 10)
            )

        # Draw orders in transit - vertical column above warehouse
        orders_list = list(self.orders_in_transit)
        order_width = 50
        order_height = 40
        order_font = pygame.font.Font(None, 20)

        # Position orders in a vertical line above the warehouse
        delivery_x = warehouse_x + warehouse_width // 2 - order_width // 2

        for i, order_qty in enumerate(orders_list):
            if order_qty > 0:
                # Calculate vertical position - closer to warehouse as lead time decreases
                # i=0 is the next to arrive (closest to warehouse), i=lead_time-1 is furthest away (highest up)
                distance_from_warehouse = (i + 1) * (order_height + 10)
                rect_y = warehouse_y - distance_from_warehouse - 30

                # Draw order box
                pygame.draw.rect(
                    canvas,
                    (70, 130, 180),  # Steel blue color
                    pygame.Rect(delivery_x, rect_y, order_width, order_height),
                )
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),
                    pygame.Rect(delivery_x, rect_y, order_width, order_height),
                    2,
                )

                # Draw order quantity inside the box
                order_text = order_font.render(str(order_qty), True, (255, 255, 255))
                text_rect = order_text.get_rect(
                    center=(delivery_x + order_width // 2, rect_y + order_height // 2)
                )
                canvas.blit(order_text, text_rect)

                # Draw downward arrow pointing to warehouse
                arrow_start = (delivery_x + order_width // 2, rect_y + order_height)
                arrow_end = (delivery_x + order_width // 2, rect_y + order_height + 8)
                pygame.draw.line(canvas, (0, 0, 0), arrow_start, arrow_end, 2)
                pygame.draw.polygon(
                    canvas,
                    (0, 0, 0),
                    [
                        (arrow_end[0], arrow_end[1]),
                        (arrow_end[0] - 5, arrow_end[1] - 5),
                        (arrow_end[0] + 5, arrow_end[1] - 5),
                    ],
                )

                # Draw days until arrival next to the box
                days_text = order_font.render(f"{i + 1}d", True, (0, 0, 0))
                canvas.blit(
                    days_text,
                    (delivery_x + order_width + 5, rect_y + order_height // 2 - 10),
                )

        # Draw incoming demands - horizontal row to the right of warehouse
        demands_to_show = min(7, len(self.daily_demand_sequence) - self.current_day)
        demand_width = 40
        demand_height = 35
        demand_start_x = warehouse_x + warehouse_width + 30
        demand_y = warehouse_y + 20

        # Draw "Incoming Demands" label
        demand_label = font.render("Demands", True, (0, 0, 0))
        canvas.blit(demand_label, (demand_start_x - 10, demand_y - 25))

        for i in range(demands_to_show):
            day_index = self.current_day + i
            if day_index < len(self.daily_demand_sequence):
                demand_qty = self.daily_demand_sequence[day_index]

                rect_x = demand_start_x + i * (demand_width + 5)

                # Only draw demand box if demand is greater than 0
                if demand_qty > 0:
                    # Draw demand box
                    pygame.draw.rect(
                        canvas,
                        (220, 100, 100),  # Reddish color for demands
                        pygame.Rect(rect_x, demand_y, demand_width, demand_height),
                    )
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(rect_x, demand_y, demand_width, demand_height),
                        2,
                    )

                    # Draw demand quantity inside
                    demand_text = order_font.render(
                        str(demand_qty), True, (255, 255, 255)
                    )
                    text_rect = demand_text.get_rect(
                        center=(
                            rect_x + demand_width // 2,
                            demand_y + demand_height // 2,
                        )
                    )
                    canvas.blit(demand_text, text_rect)

                # Always draw day label below (even if demand is 0)
                day_label_text = order_font.render(f"D{i}", True, (0, 0, 0))
                canvas.blit(
                    day_label_text,
                    (rect_x + demand_width // 2 - 10, demand_y + demand_height + 3),
                )

        # Draw past demands - horizontal row going left from warehouse
        past_demands_to_show = min(7, self.current_day)
        past_demand_y = warehouse_y + 120

        if past_demands_to_show > 0:
            # Draw "Past Demands" label to the left of warehouse
            past_demand_label = font.render("Past", True, (100, 100, 100))
            past_demand_start_x = warehouse_x - 10
            canvas.blit(
                past_demand_label, (past_demand_start_x - 40, past_demand_y - 25)
            )

            for i in range(past_demands_to_show):
                # Show most recent past demands first (reversed order)
                # i=0 is most recent (closest to warehouse), higher i values go left
                day_index = self.current_day - 1 - i
                if day_index >= 0 and day_index < len(self.daily_demand_sequence):
                    past_demand_qty = self.daily_demand_sequence[day_index]

                    # Position from right to left (most recent closest to warehouse)
                    rect_x = past_demand_start_x - (i + 1) * (demand_width + 5)

                    # Only draw demand box if demand was greater than 0
                    if past_demand_qty > 0:
                        # Draw past demand box with muted color
                        pygame.draw.rect(
                            canvas,
                            (180, 150, 150),  # Muted/grayed out color for past demands
                            pygame.Rect(
                                rect_x, past_demand_y, demand_width, demand_height
                            ),
                        )
                        pygame.draw.rect(
                            canvas,
                            (100, 100, 100),
                            pygame.Rect(
                                rect_x, past_demand_y, demand_width, demand_height
                            ),
                            2,
                        )

                        # Draw demand quantity inside
                        past_demand_text = order_font.render(
                            str(past_demand_qty), True, (80, 80, 80)
                        )
                        text_rect = past_demand_text.get_rect(
                            center=(
                                rect_x + demand_width // 2,
                                past_demand_y + demand_height // 2,
                            )
                        )
                        canvas.blit(past_demand_text, text_rect)

                    # Always draw day label (negative for past)
                    days_ago = i + 1
                    day_label_text = order_font.render(
                        f"-{days_ago}", True, (100, 100, 100)
                    )
                    canvas.blit(
                        day_label_text,
                        (
                            rect_x + demand_width // 2 - 10,
                            past_demand_y + demand_height + 3,
                        ),
                    )

        # Draw current day
        day_text = font.render(f"Day: {self.current_day}", True, (0, 0, 0))
        canvas.blit(day_text, (10, 10))

        # Draw service level
        if self.total_demand > 0:
            service_level = self.total_fulfilled_demand / self.total_demand
            service_text = font.render(
                f"Service Level: {service_level:.2%}", True, (0, 0, 0)
            )
            canvas.blit(service_text, (10, 40))

        # Draw cumulative reward
        reward_font = pygame.font.Font(None, 24)
        reward_text = reward_font.render(
            f"Cumulative Reward: {self.episode_reward:.2f}", True, (0, 128, 0)
        )
        canvas.blit(reward_text, (10, 120))

        if self.render_mode == "human":
            # Copy canvas to the window and display
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # Limit FPS
            self.clock.tick(FPS)
        else:  # rgb_array
            return pygame.surfarray.array3d(canvas).transpose((1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
