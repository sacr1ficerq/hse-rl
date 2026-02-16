import torch
import torch.nn as nn

import numpy as np
from typing import Any, Tuple

from .utils import conv2d_size_out
from .DQN import GradScaler, GradScalerFunctional

DEBUG = 0


class QuantileDuelingNetwork(nn.Module):
    """
    Implement the Dueling DQN logic.
    """

    def __init__(
        self,
        n_actions: int,
        width: int,
        height: int,
        hidden_size: int,
        n_bins: int,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.width = width
        self.height = height
        self.hidden_size = hidden_size
        self.n_bins = n_bins

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        convolve = lambda size: conv2d_size_out(size, kernel_size=3, stride=2)  # type: ignore

        width = convolve(convolve(convolve(width)))
        height = convolve(convolve(convolve(height)))
        out_features = width * height * 64

        self.value_layer = nn.Sequential(
            nn.Linear(out_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_bins),
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(out_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions * self.n_bins),
        )

        self.grad_scaler = GradScaler(1.0 / (2 ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.shape  # (batch_size, n_images, width, height)
        # When calculating the mean advantage, please, remember, x is a batched input!

        features = self.convolution(x)  # (batch_size, 64, out_features)
        features = features.flatten(start_dim=1)  # (batch_size, out_features)

        # add scaling
        # features = self.grad_scaler(features)

        value = self.value_layer(features).unsqueeze(-1)  # (batch_size, n_bins, 1)
        advantage = self.advantage_layer(features)  # (batch_size, n_actions * n_bins)

        batch_size = advantage.size(0)
        advantage = advantage.view(batch_size, self.n_bins, self.n_actions)

        advantage -= advantage.mean(dim=2, keepdim=True)  # (batch_size, n_bins, n_actions)
        if DEBUG:
            print(f"advantage shape: {advantage.shape}")
            print(f"value shape: {value.shape}")

        return advantage + value  # hopefully works


class QuantileDQNAgent(nn.Module):
    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        n_actions: int,
        n_bins,
        epsilon: float = 0.0,
        hidden_size: int = 32,
        device: torch.device = torch.device("cuda:0"),
    ):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        n_images, width, height = state_shape
        assert n_images == 4  # Network relies on that

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful

        self.network = QuantileDuelingNetwork(n_actions, width, height, hidden_size, n_bins).to(device)

    def forward(self, state_t: torch.Tensor) -> torch.Tensor:
        """
        takes agent's observation (tensor), returns zvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        q_values = self.network(state_t.to(self.device))  # (batch_size, n_bins, n_actions)
        return q_values  # (batch_size, n_bins, n_actions)

    @torch.inference_mode()
    def get_qvalues(self, states: np.ndarray | list) -> np.ndarray:
        """
        like forward, but works on numpy arrays, not tensors
        """
        if isinstance(states, list):
            states = np.array(states)

        tensor = torch.from_numpy(states).to(self.device)
        if DEBUG:
            print(f"Tensor shape: {tensor.shape}")
        with torch.no_grad():
            q_values = self.network(tensor)  # (batch_size, n_bins, n_actions)
            q_values = q_values.mean(dim=1)  # (batch_size, n_actions)
        if DEBUG:
            print(f"q_values shape: {q_values.shape}")
        return q_values.detach().cpu().numpy()

    def sample_actions_by_qvalues(
        self, qvalues: np.ndarray, greedy: bool = False
    ) -> np.ndarray:
        """pick actions given qvalues based on epsilon-greedy exploration strategy."""
        batch_size, n_actions = qvalues.shape
        eps = self.epsilon

        greedy_actions = np.argmax(qvalues, axis=1)

        if greedy:
            return greedy_actions

        rand = np.random.choice(a=n_actions, size=batch_size, replace=True)
        return np.where(np.random.random(batch_size) < eps, rand, greedy_actions)

    def sample_actions(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        if isinstance(states, list):
            states = np.array(states)
        qvalues = self.get_qvalues(states)

        if DEBUG:
            print(f"Q-values shape: {qvalues.shape}")

        return self.sample_actions_by_qvalues(qvalues, greedy)


if __name__ == "__main__":
    BATCH_SIZE = 1
    N_BINS = 5
    N_ACTIONS = 5

    WIDTH = 16
    HEIGHT = 16
    CHANNELS = 4

    device = torch.device("cuda:0")
    module = QuantileDuelingNetwork(n_actions=N_ACTIONS, width=WIDTH, height=HEIGHT, hidden_size=32, n_bins=N_BINS).to(device)

    if DEBUG:
        print(module)

    x = torch.randn(
        size=(BATCH_SIZE, CHANNELS, WIDTH, HEIGHT), device=device
    )  # (batch_size, C, width, height))
    if DEBUG:
        out = module(x)
        print("NN out:", out.shape)

    state_shape = (CHANNELS, WIDTH, HEIGHT)
    print("state_shape", state_shape)
    epsilon = 0.1
    hidden_size = 32
    agent = QuantileDQNAgent(state_shape, N_ACTIONS, epsilon, hidden_size)

    npx = x.detach().cpu().numpy()
    q_values = agent.get_qvalues(npx)
    print("q_values: ", q_values.shape)
