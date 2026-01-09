from typing import TypeAlias
import numpy as np

from collections import defaultdict

StateType: TypeAlias = tuple[int, int, bool]


class Agent:
    def __init__(
        self,
        n_actions: int,
        learning_rate: float,
        epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.n_actions = n_actions
        self.q_values = defaultdict(lambda: np.zeros(n_actions))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon

        self.mse = []

    def on_episode_start(self):
        pass

    def get_best_action(self, state: StateType) -> int:
        state_q_values = self.q_values[state]
        return np.argmax(state_q_values)  # type: ignore

    def get_action(self, state: StateType) -> int:
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        return self.get_best_action(state)


class QLearningAgent(Agent):
    def get_value(self, state: StateType) -> float:
        #  optimal value equals to max q_funciton
        state_q_values = self.q_values[state]
        return np.max(state_q_values)  # type: ignore

    def update(
        self,
        state: StateType,
        action: int,
        reward: float,
        terminated: bool,
        next_state: StateType,
    ):
        """Updates the Q-value of an action."""
        next_q_value = (1 - terminated) * self.get_value(next_state)
        y = reward + self.discount_factor * next_q_value
        td_error = y - self.q_values[state][action]

        self.q_values[state][action] += self.lr * td_error
        self.mse.append(td_error**2)
