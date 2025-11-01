
"""Epsilon Greedy Exploration Strategy."""

import numpy as np
class EpsilonGreedy:
    """Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        """Initialize Epsilon Greedy Exploration Strategy."""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space, accepted_action=None):
        """
        Choose action only from accepted_actions if provided.

        Args:
        - q_table: dict or np.array storing Q-values
        - state: current state
        - action_space: gym-like action space with action_space.n
        - accepted_actions: optional set/list of allowed actions

        Returns:
        - action: selected action
        """
        if accepted_action is not None:
            valid_actions = list(accepted_action)
        else:
            valid_actions = list(range(action_space.n))

        if not valid_actions:
            raise ValueError("No accepted actions provided!")

        if np.random.rand() < self.epsilon:
            # Explore
            action = int(np.random.choice(valid_actions))
        else:
            # Exploit
            q_values = q_table[state]
            valid_q_values = {a: q_values[a] for a in valid_actions}
            action = max(valid_q_values, key=valid_q_values.get)

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

        return action

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon



class QLAgent:
    """Q-learning Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self, rejected_actions=None):
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space, rejected_actions=rejected_actions)
        return self.action

    def learn(self, next_state, reward, final_action=None):
        """Update Q-table with new experience.
        
        final_action: 主程序最终执行的动作（int），如果为 None，使用之前的 self.action。
        """
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state

        # 使用最终执行的动作，默认用之前的动作
        a = final_action if final_action is not None else self.action

        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward
