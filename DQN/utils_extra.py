import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

import cv2
import numpy as np
import ale_py


import atari_wrappers

from framebuffer import FrameBuffer
import matplotlib.pyplot as plt
import utils


gym.register_envs(ale_py)


ENV_NAME = "ALE/Breakout-v5"


# ------- Preprocessing -------
class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        super().__init__(env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def _to_gray_scale(self, rgb, channel_weights=[0.8, 0.1, 0.1]):
        return np.dot(rgb[...,:3], channel_weights)[None, :, :]

    def observation(self, observation):
        """what happens to each observation"""
        img = observation

        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (use imresize from any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type
        img = img[31:-17, 7:-8]  # crop
        img = cv2.resize(img, (64, 64))  # resize
        img = self._to_gray_scale(img)  # grayscale
        img = img.astype(np.float32) / 256  # float
        return img


# ------- Wrapping -------
def PrimaryAtariWrap(env, clip_rewards=True):
    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    # This wrapper is yours :)
    env = PreprocessAtariObs(env)
    return env


# ------- Frame Buffer -------
def make_env(clip_rewards=True, seed=None):
    env = gym.make(ENV_NAME, render_mode="rgb_array")  # create raw env
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env


# ------- Dueling DQN -------
def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s, _ = env.reset()
        reward = 0
        for _ in range(t_max):
            action = agent.sample_actions([s], greedy=greedy)[0]
            s, r, terminated, truncated, _ = env.step(action)
            reward += r
            if terminated or truncated:
                break

        rewards.append(reward)
    return np.mean(rewards)


# ------- Experience Replay -------
def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has terminated=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        action = agent.sample_actions([s])[0]
        next_s, r, terminated, truncated, _ = env.step(action)
        exp_replay.add(s, action, r, next_s, terminated)
        sum_rewards += r
        s = next_s
        if terminated or truncated:
            s, _ = env.reset()

    return sum_rewards, s


# ------- Training -------
def check_ram():
    if not utils.is_enough_ram():
        print('less that 100 Mb RAM available, freezing')
        print('make sure everything is ok and make KeyboardInterrupt to continue')
        try:
            while True:
                pass
        except KeyboardInterrupt:
            pass


def plot_stats(
    mean_rw_history,
    td_loss_history,
    initial_state_v_history,
    grad_norm_history,
    noise_sigmas_history
):
    plt.style.use('seaborn-v0_8-muted')
    # plt.style.use('ggplot')

    def ema(series, alpha=0.1):
        """
        Exponential Moving Average.
        High alpha = follows data closely (noisy).
        Low alpha = smoother (more lag).
        """
        series = np.array(series)
        if len(series) == 0:
            return series
        # Using the recursive formula: y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
        smoothed = np.zeros_like(series)
        smoothed[0] = series[0]
        for i in range(1, len(series)):
            smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    plt.figure(figsize=[24, 14])

    plt.subplot(2, 2, 1)
    plt.title("Mean reward per life")
    plt.plot(mean_rw_history, alpha=0.25, color='tab:blue')
    plt.plot(ema(mean_rw_history), color='tab:red', linewidth=2)
    plt.grid()

    assert not np.isnan(td_loss_history[-1])
    plt.subplot(2, 2, 2)
    plt.title("TD loss history (smoothened)")
    plt.plot(utils.smoothen(td_loss_history), color='tab:blue')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.title("Initial state V")
    plt.plot(initial_state_v_history, alpha=0.25, color='tab:blue')
    plt.plot(ema(initial_state_v_history), color='tab:red', linewidth=2)
    plt.grid()

    # plt.subplot(2, 2, 4)
    # plt.title("Grad norm history (smoothened)")
    # plt.plot(utils.smoothen(grad_norm_history), color='tab:blue')
    # plt.grid()
    # plt.show()

    plt.subplot(2, 2, 4)
    plt.title("Noise sigmas history (smoothened)")
    plt.plot(utils.smoothen(noise_sigmas_history), color='tab:blue')
    plt.grid()
    plt.show()
