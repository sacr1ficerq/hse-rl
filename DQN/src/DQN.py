import torch
import torch.nn as nn

import numpy as np
from typing import Any, Tuple

from .utils import conv2d_size_out

DEBUG = 0


class DuelingNetwork(nn.Module):
    """
    Implement the Dueling DQN logic.
    """

    def __init__(self, n_actions: int, width: int, height: int, hidden_size: int) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.width = width
        self.height = height
        self.hidden_size = hidden_size

        # <YOUR_CODE>
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
            nn.Linear(out_features, hidden_size),  # hidden_size?
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # hidden_size?
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(out_features, hidden_size),  # hidden_size?
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),  # hidden_size?
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.shape  # (batch_size, n_images, width, height)
        # When calculating the mean advantage, please, remember, x is a batched input!

        # <YOUR_CODE>
        features = self.convolution(x)  # (batch_size, 64, out_features)
        features = features.flatten(start_dim=1)  # (batch_size, out_features)

        value = self.value_layer(features)  # (batch_size)
        advantage = self.advantage_layer(features)

        advantage -= advantage.mean(dim=1, keepdim=True)  # ?
        if DEBUG:
            print(f"advantage shape: {advantage.shape}")
        if DEBUG:
            print(f"value shape: {value.shape}")

        return advantage + value  # repeat value?


class GradScalerFunctional(torch.autograd.Function):
    """
    A torch.autograd.Function works as Identity on forward pass
    and scales the gradient by scale_factor on backward pass.
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, scale_factor: float) -> torch.Tensor:
        ctx.scale_factor = scale_factor
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        scale_factor = ctx.scale_factor
        grad_input = grad_output * scale_factor
        return grad_input, None  # why None?


class GradScaler(nn.Module):
    """
    An nn.Module incapsulating GradScalerFunctional
    """

    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return GradScalerFunctional.apply(x, self.scale_factor)


class DQNAgent(nn.Module):
    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        n_actions: int,
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

        self.network = DuelingNetwork(n_actions, width, height, hidden_size).to(device)

    def forward(self, state_t: torch.Tensor) -> torch.Tensor:
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        q_values = self.network(state_t.to(self.device))
        return q_values

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
            q_values = self.network(tensor)
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
    device = torch.device("cuda:0")
    module = DuelingNetwork(n_actions=4, width=16, height=16, hidden_size=32).to(device)
    if DEBUG:
        print(module)

    x = torch.randn(
        size=(1, 4, 16, 16), device=device
    )  # (batch_size, C, width, height))
    if DEBUG:
        print(module(x))

    n_actions = 5
    state_shape = (4, 16, 16)
    epsilon = 0.1
    hidden_size = 32
    agent = DQNAgent(state_shape, n_actions, epsilon, hidden_size)

    npx = x.detach().cpu().numpy()
    agent.get_qvalues(npx)
