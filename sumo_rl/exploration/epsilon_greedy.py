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

    def choose(self, q_table, state, action_space, rejected_actions=None):
        """
        Choose action from valid actions only, directly excluding rejected ones.
        
        Returns:
        - action: selected action from valid actions only
        """
        rejected_actions = rejected_actions or set()
        valid_actions = [a for a in range(action_space.n) if a not in rejected_actions]

        if not valid_actions:
            raise ValueError("All actions are rejected! Cannot select a valid action.")

        if np.random.rand() < self.epsilon:
            action = int(np.random.choice(valid_actions))
        else:
            q_values = q_table[state]
            valid_q_values = {a: q_values[a] for a in valid_actions}
            action = max(valid_q_values, key=valid_q_values.get)

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

        return action


        return action, real_action



    # def choose(self, q_table, state, action_space):
    #     """Choose action based on epsilon greedy strategy."""
    #     if np.random.rand() < self.epsilon:
    #         action = int(action_space.sample())
    #     else:
    #         action = np.argmax(q_table[state])

    #     self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
    #     # print(self.epsilon)
    #     return action

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon
