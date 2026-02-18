from tqdm import trange
import numpy as np
import torch
from torch import nn, t

import matplotlib.pyplot as plt
from IPython.display import clear_output

from utils_extra import make_env, play_and_record, evaluate, plot_stats, check_ram

from typing import Tuple, Any

import utils
from src.QRDQN import QuantileDQNAgent, NoisyLinear
from src.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


def train_step(
    agent: QuantileDQNAgent,
    target_network,
    optimizer,
    max_grad_norm,
    loss_fn,
    env,
    state: np.ndarray,
    batch_size: int,
    replay_buffer: ReplayBuffer | PrioritizedReplayBuffer,
    play_n_steps=1,
) -> Tuple[np.ndarray, float, float]:

    # play
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    # TODO: can we make this faster? agent doesnt evolve here
    for _ in range(play_n_steps):
        action = agent.sample_actions(np.array([state]))[0]
        next_s, r, terminated, truncated, _ = env.step(action)
        replay_buffer.add(state, action, r, next_s, terminated)  # add with maximum reward, then update after sampling anyways?
        sum_rewards += r
        state = next_s
        if terminated or truncated:
            state, _ = env.reset()
            agent.reset_noise()

    # train
    device = agent.device
    if isinstance(replay_buffer, PrioritizedReplayBuffer):
        batch, indices, weights = replay_buffer.sample(batch_size, device)
    else:
        batch = replay_buffer.sample(batch_size)
    batch = [torch.from_numpy(x).to(device, non_blocking=True) for x in batch]
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = batch

    losses = loss_fn(
        obs_batch,
        act_batch,
        reward_batch,
        next_obs_batch,
        is_done_batch,
        agent,
        target_network,
        gamma=0.99
    )

    if isinstance(replay_buffer, PrioritizedReplayBuffer):
        loss = (losses * weights).mean()
    else:
        loss = losses.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    optimizer.step()

    if isinstance(replay_buffer, PrioritizedReplayBuffer):
        replay_buffer.update_errors(indices, losses.detach().cpu().numpy())

    return state, loss.data.cpu().item(), grad_norm.cpu().item()


def get_mean_sigma(agent):
    sigmas = []
    for m in agent.modules():
        if isinstance(m, NoisyLinear):
            sigmas.append(m.weight_sigma.data.mean().item())
    return float(np.mean(sigmas))


def train(
    agent: QuantileDQNAgent,
    target_network,
    optimizer,
    max_grad_norm,
    loss_fn,
    env,
    total_steps: int,
    batch_size: int,
    replay_buffer: ReplayBuffer | PrioritizedReplayBuffer,
    # Logging
    loss_freq: int,
    refresh_target_network_freq: int,
    eval_freq: int,
    decay_steps: int,
    # Exploration
    init_epsilon: float,
    final_epsilon: float,
    play_n_steps: int,
):
    mean_rw_history = []
    td_loss_history = []
    initial_state_v_history = []
    grad_norm_history = []
    noise_sigmas_history = []

    state, _ = env.reset()
    # warmup
    for _ in range(batch_size):
        s = state
        action = agent.sample_actions(np.array([s]))[0]
        next_s, r, terminated, truncated, _ = env.step(action)
        replay_buffer.add(s, action, r, next_s, terminated)  # add with maximum reward, then update after sampling anyways?
        s = next_s
        if terminated or truncated:
            s, _ = env.reset()

    for step in trange(total_steps + 1):
        check_ram()
        agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, decay_steps)

        state, loss, grad_norm = train_step(
            agent,
            target_network,
            optimizer,
            max_grad_norm,
            loss_fn,
            env,
            state,
            batch_size,
            replay_buffer,
            play_n_steps,
        )

        if step % loss_freq == 0:
            td_loss_history.append(loss)
            grad_norm_history.append(grad_norm)
            noise_sigmas_history.append(get_mean_sigma(agent))

        if step % refresh_target_network_freq == 0:
            target_network.load_state_dict(agent.state_dict())

        if step % eval_freq == 0:
            mean_rw_history.append(
                evaluate(
                    make_env(clip_rewards=True, seed=step),
                    agent,
                    n_games=3 * 3,
                    greedy=True
                )
            )
            initial_state_q_values = agent.get_qvalues(
                [make_env(seed=step).reset()[0]]
            )
            initial_state_v_history.append(np.max(initial_state_q_values))

            clear_output(True)
            print("buffer size = %i, epsilon = %.5f" % (len(replay_buffer), agent.epsilon))

            plot_stats(mean_rw_history, td_loss_history, initial_state_v_history, grad_norm_history, noise_sigmas_history)


if __name__ == '__main__':
    from src.QRDQN import QuantileDQNAgent
    from src.utils import compute_quantile_loss
    from src.replay_buffer import PrioritizedReplayBuffer
    import random

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0")

    env = make_env(seed=seed)
    state_shape: Tuple[int, int, int] = env.observation_space.shape  # type: ignore
    n_actions = env.action_space.n  # type: ignore

    N_BINS = 5

    agent = QuantileDQNAgent(state_shape, n_actions, n_bins=N_BINS, epsilon=1).to(device)
    target_network = QuantileDQNAgent(state_shape, n_actions, n_bins=N_BINS).to(device)
    target_network.load_state_dict(agent.state_dict())

    timesteps_per_epoch = 1
    batch_size = 32
    total_steps = 3 * 10**6
    decay_steps = 10**6

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

    init_epsilon = 1
    final_epsilon = 0.1

    loss_freq = 50
    refresh_target_network_freq = 5_000
    eval_freq = 5_000
    play_n_steps = 1_000

    max_grad_norm = 50

    n_lives = 5

    replay_buffer = PrioritizedReplayBuffer(100_000)

    histories = train(
        agent=agent,
        target_network=target_network,
        optimizer=optimizer,
        max_grad_norm=max_grad_norm,
        loss_fn=compute_quantile_loss,
        env=env,
        total_steps=total_steps,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        decay_steps=decay_steps,
        init_epsilon=init_epsilon,
        final_epsilon=final_epsilon,
        loss_freq=loss_freq,
        refresh_target_network_freq=refresh_target_network_freq,
        eval_freq=eval_freq,
        play_n_steps=play_n_steps,
    )
