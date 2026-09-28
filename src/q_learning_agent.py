"""
q_learning_agent.py

Tabular Q-learning agents for the IoV Digital Twin data-transmission decision
problem. The implementation keeps the policy interpretable and dependency-light
for small synthetic or trace-driven experiments.

State discretization: continuous [speed, bandwidth, delay, dt_error] is
binned into a small number of buckets per dimension to build a Q-table.
"""

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_actions=3,
        bins=(6, 6, 6, 8),
        state_ranges=((0, 35), (3, 30), (1, 40), (0, 25)),
        alpha=0.15,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.n_actions = n_actions
        self.bins = bins
        self.state_ranges = state_ranges
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros(bins + (n_actions,), dtype=np.float32)

    def _discretize(self, state):
        idx = []
        for value, (lo, hi), n_bins in zip(state, self.state_ranges, self.bins):
            v = np.clip(value, lo, hi)
            pos = (v - lo) / (hi - lo + 1e-9)
            b = int(pos * n_bins)
            b = min(b, n_bins - 1)
            idx.append(b)
        return tuple(idx)

    def select_action(self, state, greedy=False):
        idx = self._discretize(state)
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table[idx]))

    def update(self, state, action, reward, next_state, done):
        idx = self._discretize(state)
        next_idx = self._discretize(next_state)
        best_next = 0.0 if done else np.max(self.q_table[next_idx])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[idx + (action,)]
        self.q_table[idx + (action,)] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DoubleQLearningAgent(QLearningAgent):
    """Double Q-learning reduces max-action overestimation in noisy traces."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_table_a = np.zeros_like(self.q_table)
        self.q_table_b = np.zeros_like(self.q_table)
        self.q_table = self.q_table_a + self.q_table_b

    def _combined_q(self, idx):
        return self.q_table_a[idx] + self.q_table_b[idx]

    def select_action(self, state, greedy=False):
        idx = self._discretize(state)
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self._combined_q(idx)))

    def update(self, state, action, reward, next_state, done):
        idx = self._discretize(state)
        next_idx = self._discretize(next_state)

        if self.rng.random() < 0.5:
            best_action = int(np.argmax(self.q_table_a[next_idx]))
            best_next = 0.0 if done else self.q_table_b[next_idx + (best_action,)]
            td_target = reward + self.gamma * best_next
            td_error = td_target - self.q_table_a[idx + (action,)]
            self.q_table_a[idx + (action,)] += self.alpha * td_error
        else:
            best_action = int(np.argmax(self.q_table_b[next_idx]))
            best_next = 0.0 if done else self.q_table_a[next_idx + (best_action,)]
            td_target = reward + self.gamma * best_next
            td_error = td_target - self.q_table_b[idx + (action,)]
            self.q_table_b[idx + (action,)] += self.alpha * td_error

        self.q_table = self.q_table_a + self.q_table_b
