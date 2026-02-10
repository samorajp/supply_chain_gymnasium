import numpy as np
from matplotlib import pyplot as plt
from rich import print


class QLearningAgent:
    def __init__(
        self, env, alpha=0.01, gamma=0.99, epsilon=0.01, logging_interval=1000
    ):
        self.q_table = {}
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.logging_interval = logging_interval
        self.best_episode_reward = -float("inf")

        self.fig = None
        self.ax = None

        observation, info = self.env.reset()
        self.observation_keys = list(observation.keys())
        print(f"Observation keys used for Q-Table: {self.observation_keys}")

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 100)

    def select_greedy_action(self, state):
        q_values = [self.get_q_value(state, a) for a in range(self.env.action_space.n)]
        return int(np.argmax(q_values))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.select_greedy_action(state)

    def update_q_value(self, state, action, reward, next_state, terminated):
        if terminated:
            self.q_table[(state, action)] = reward
            return

        best_next_q = max(
            [self.get_q_value(next_state, a) for a in range(self.env.action_space.n)]
        )
        td_target = reward + self.gamma * best_next_q
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = td_target
        else:
            td_delta = td_target - self.get_q_value(state, action)
            new_q = self.get_q_value(state, action) + self.alpha * td_delta
            self.q_table[(state, action)] = new_q

    def learn(self, num_episodes):
        observation, info = self.env.reset()
        observation_keys = list(observation.keys())
        smoothed_episode_reward = -10000

        for i in range(num_episodes):
            observation, info = self.env.reset()
            observation_keys = list(observation.keys())
            done = False
            episode_reward = 0
            while not done:
                state = tuple(observation[key] for key in observation_keys)
                action = self.select_action(state)
                next_observation, reward, terminated, truncated, info = self.env.step(
                    action
                )
                next_state = tuple(next_observation[key] for key in observation_keys)
                self.update_q_value(state, action, reward, next_state, terminated)
                observation = next_observation
                done = terminated or truncated
                episode_reward += reward

            self.best_episode_reward = max(self.best_episode_reward, episode_reward)
            smoothed_episode_reward += (episode_reward - smoothed_episode_reward) * 0.01

            if i % self.logging_interval == 0:
                greedy_episode_reward = self.greedy_episode()
                print(
                    f"Episode {i}, Smoothed: {smoothed_episode_reward:.2f}, Greedy: {greedy_episode_reward}, Best: {self.best_episode_reward}"
                )
                self.plot()

    def greedy_episode(self):
        observation, info = self.env.reset()
        observation_keys = list(observation.keys())
        done = False
        episode_reward = 0
        while not done:
            state = tuple(observation[key] for key in observation_keys)
            action = self.select_greedy_action(state)
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            observation = next_observation
            done = terminated or truncated
            episode_reward += reward
        return episode_reward

    def plot(self):
        # Extract states and their maximum Q-values
        state_max_q = {}
        for (state, action), q_value in self.q_table.items():
            if state not in state_max_q:
                state_max_q[state] = q_value
            else:
                state_max_q[state] = max(state_max_q[state], q_value)

        num_dimensions = len(self.observation_keys)

        # Compute max Q-value for each dimension
        dim_max_q = [dict() for _ in range(num_dimensions)]

        for state, max_q in state_max_q.items():
            for dim_idx in range(num_dimensions):
                # Max Q-value for this state[dim_idx] value across all other dimensions
                if state[dim_idx] not in dim_max_q[dim_idx]:
                    dim_max_q[dim_idx][state[dim_idx]] = max_q
                else:
                    dim_max_q[dim_idx][state[dim_idx]] = max(
                        dim_max_q[dim_idx][state[dim_idx]], max_q
                    )

        # Sort for plotting
        dim_sorted = [sorted(dim_dict.items()) for dim_dict in dim_max_q]

        # Create figure and axes only once
        if self.fig is None:
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots(
                1, num_dimensions, figsize=(7 * num_dimensions, 5)
            )
            # Handle case where there's only one dimension
            if num_dimensions == 1:
                self.ax = [self.ax]
        else:
            # Clear the existing plots
            for ax in self.ax:
                ax.clear()

        # Plot each dimension
        colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
        for dim_idx in range(num_dimensions):
            if dim_sorted[dim_idx]:
                x, y = zip(*dim_sorted[dim_idx])
                color = colors[dim_idx % len(colors)]
                self.ax[dim_idx].plot(
                    x, y, "o-", linewidth=2, markersize=8, color=color
                )
                self.ax[dim_idx].set_xlabel(f"{self.observation_keys[dim_idx]}")
                self.ax[dim_idx].set_ylabel("Max Q-Value")
                self.ax[dim_idx].set_title(
                    f"Max Q-Value vs {self.observation_keys[dim_idx]}"
                )
                self.ax[dim_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.draw()
        plt.pause(1)
