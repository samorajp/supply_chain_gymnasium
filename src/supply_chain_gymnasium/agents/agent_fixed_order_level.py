class FixedOrderLevelAgent:
    def __init__(self, env, order_level, gamma=0.99):
        self.env = env
        self.order_level = order_level
        self.gamma = gamma
        self.q_table = {}  # Store state-action values

        observation, info = self.env.reset()
        self.observation_keys = list(observation.keys())

    def select_action(self, info):
        if info["inventory_position"] <= self.order_level:
            return 1
        else:
            return 0

    def learn(self, num_episodes=1):
        """
        Evaluate the fixed order level policy by running episodes
        and computing the Q-table (state-action values) empirically.
        """
        if self.env is None:
            raise ValueError("Environment must be provided to learn/evaluate policy")

        episode_rewards = []

        for episode in range(num_episodes):
            observation, info = self.env.reset()
            done = False
            episode_reward = 0
            states_actions_rewards = []

            while not done:
                # Get current state (inventory position)
                state = tuple(observation[key] for key in self.observation_keys)
                # Select action based on fixed policy
                action = self.select_action(info)

                # Take action
                observation, reward, terminated, truncated, next_info = self.env.step(
                    action
                )

                # Store state-action-reward tuple
                states_actions_rewards.append((state, action, reward))

                episode_reward += reward
                info = next_info
                done = terminated or truncated

            episode_rewards.append(episode_reward)

            G = 0
            for state, action, reward in reversed(states_actions_rewards):
                G = reward + G * self.gamma
                if (state, action) not in self.q_table:
                    self.q_table[(state, action)] = []
                self.q_table[(state, action)].append(G)

        # Average Q-values over all episodes
        for key in self.q_table:
            self.q_table[key] = sum(self.q_table[key]) / len(self.q_table[key])

        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print("Q-table learned:")
        for key in sorted(self.q_table.keys()):
            print(
                f"State: {key[0]}, Action: {key[1]}, Q-value: {self.q_table[key]:.2f}"
            )

        return avg_reward


def evaluate_fixed_order_level_agent(env, order_level):
    observation, info = env.reset()
    agent = FixedOrderLevelAgent(env=env, order_level=order_level)
    done = False
    while not done:
        observation, reward, terminated, truncated, info = env.step(
            agent.select_action(info=info)
        )
        done = terminated or truncated
    return env.episode_reward
