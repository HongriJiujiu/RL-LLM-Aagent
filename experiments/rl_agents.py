import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim


class ACNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(ACNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.pi_head = nn.Linear(hidden_dim, output_dim)  # Actor head
        self.v_head = nn.Linear(hidden_dim, 1)            # Critic head

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy_logits = self.pi_head(x)
        value = self.v_head(x)
        return F.softmax(policy_logits, dim=-1), value


class ACAgent:
    def __init__(self, starting_state, state_dim, action_space, llm_weight=0, lr=0.001, gamma=0.99):
        self.model = ACNetwork(state_dim, action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.llm_weight = llm_weight

        self.state = starting_state
        self.action = None
        self.action_space = action_space
        self.acc_reward = 0

    def act(self, accepted_actions=None):
        """
        根据当前状态选择动作，只在 accepted_actions 中选择。
        
        Args:
            accepted_actions: 一个包含允许选择的动作编号的集合或列表。如果为 None，则使用所有动作。
        """
        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        probs, value = self.model(state_tensor)  # shape: (1, n)
        
        probs = probs.squeeze(0)  # shape -> (n,)
        # # 打印当前状态和网络输出的策略概率
        # print("当前 state:", self.state)
        # print("策略概率 probs:", probs.detach().numpy())
        # 如果指定 accepted_actions，则屏蔽其他动作
        if accepted_actions is not None:
            mask = torch.zeros_like(probs, dtype=torch.bool)
            for a in accepted_actions:
                if 0 <= a < probs.size(0):
                    mask[a] = True
            probs = probs * mask.float()  # 不允许动作概率设为0
            if probs.sum().item() == 0:
                # 如果所有动作都被屏蔽，退化为均匀分布
                probs = torch.ones_like(probs) / probs.size(0)
            else:
                probs = probs / probs.sum()  # 重新归一化

        probs = probs.unsqueeze(0)  # 恢复 shape (1, n)
        dist = Categorical(probs)
        action = dist.sample()

        self._log_prob = dist.log_prob(action)
        self._value = value
        self.action = action.item()
        return self.action

    def learn(self, next_state, reward, final_action=None, llm_score=None):
        """
        学习阶段，使用 TD 目标 + LLM 评估结果进行 critic loss 组合更新。
        final_action: 主程序最终执行的动作（int），如果为 None，使用之前 act 的动作。
        llm_score: LLM 评分。
        """
        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)

        # 重新计算策略分布和状态值
        probs, value = self.model(state_tensor)
        dist = Categorical(probs)

        # 如果传入了最终动作，使用它计算 log_prob，否则用之前 act 的动作
        if final_action is not None:
            action_tensor = torch.tensor([final_action])
            log_prob = dist.log_prob(action_tensor)
        else:
            log_prob = self._log_prob

        # 预测下一状态的值
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        _, next_value = self.model(next_state_tensor)
        next_value = next_value.detach()

        # TD 目标与 Advantage
        td_target = reward + self.gamma * next_value
        advantage = td_target - value

        # Losses
        actor_loss = -log_prob * advantage.detach()
        critic_loss = F.mse_loss(value, td_target)

        # LLM 评分融合
        if llm_score is not None:
            critic_loss = (1 - self.llm_weight) * critic_loss + self.llm_weight * critic_loss * llm_score

        # 总损失反向传播
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新状态和累计奖励
        self.state = next_state
        self.acc_reward += reward


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

    def act(self, accepted_action=None):
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space, accepted_action=accepted_action)
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



def setup_agents(env,args,initial_states,choose_agents,tls_ids):
    if choose_agents == 'QLAgent':
        # 调整参数
        rl_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_spaces(ts),
                alpha=0.4,        # ↑ 学习率提升，加快学习
                gamma=0.95,       # ↓ 稍微降低折扣率
                exploration_strategy=EpsilonGreedy(
                    initial_epsilon=0.4,  # ↑ 增加探索
                    min_epsilon=0.01,     # 稍微提高最低探索概率
                    decay=0.995,          # ↑ 平滑衰减
                ),
            )
            for ts in tls_ids
        }
    elif choose_agents == 'ACAgent':
        # 调整参数
        rl_agents = {
            ts: ACAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_dim=len(env.encode(initial_states[ts], ts)),
                action_space=env.action_spaces(ts),
                llm_weight=0,
                lr=0.001,
                gamma=0.95,
            )
            for ts in tls_ids
        }
    return rl_agents