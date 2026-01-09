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


class SarsaAgent(Agent):
    def __init__(
        self,
        n_actions: int,
        learning_rate: float,
        epsilon: float,
        decay_factor: float,
        discount_factor: float = 0.95,
    ):
        super().__init__(n_actions, learning_rate, epsilon, discount_factor)
        self.decay_factor = decay_factor
        self.on_episode_start()

    def on_episode_start(self):
        self.e_traces = defaultdict(lambda: np.zeros(self.n_actions))

    def get_value(self, state: StateType) -> float:
        action = self.get_action(state)
        return self.q_values[state][action]

    def update(
        self,
        state: StateType,
        action: int,
        reward: float,
        terminated: bool,
        next_state: StateType,
    ):
        next_q_value = (1 - terminated) * self.get_value(next_state)
        y = reward + self.discount_factor * next_q_value
        td_error = y - self.q_values[state]
        self.e_traces[state][action] += 1
        for s in self.e_traces:
            self.q_values[s] += self.lr * self.e_traces[state] * td_error
            self.e_traces[s] *= self.discount_factor * self.decay_factor
        self.mse.append(td_error**2)


class EVSarsaAgent(SarsaAgent):
    def get_value(self, state: StateType) -> float:
        state_q_values = self.q_values[state]
        value = self.epsilon * np.mean(state_q_values) + (1 - self.epsilon) * np.max(state_q_values)
        return value  # type: ignore
