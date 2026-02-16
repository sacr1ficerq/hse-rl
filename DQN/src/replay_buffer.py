import numpy as np
from typing import Tuple

import torch


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        # <YOUR CODE>
        self._storage = []
        self._idx = 0  # where to put next element
        self._capacity = capacity

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        Make sure, _storage will not exceed _maxsize.
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        """
        data = (obs_t, action, reward, obs_tp1, done)
        # <YOUR CODE>
        # add data to storage
        if len(self._storage) == self._capacity:
            self._storage[self._idx] = data
            self._idx = (self._idx + 1) % self._capacity
        else:
            self._storage.append(data)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        # randomly generate batch_size integers
        # to be used as indexes of samples
        idxs = np.random.choice(len(self._storage), batch_size)

        # collect <s,a,r,s',done> for each index
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in idxs:
            s, a, r, s_, d = self._storage[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)
            dones.append(d)

        # <states>, <actions>, <rewards>, <next_states>, <is_done>
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = []
        self._weights = np.zeros(capacity)
        self._idx = 0  # where to put next element
        self._capacity = capacity
        self.updated = True

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        Make sure, _storage will not exceed _maxsize.
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        """
        error = np.max(self._weights)
        data = (obs_t, action, reward, obs_tp1, done)
        if len(self._storage) == self._capacity:
            self._storage[self._idx] = data
            self._weights[self._idx] = error
            self._idx = (self._idx + 1) % self._capacity
        else:
            self._storage.append(data)
            self._weights[len(self._storage) - 1] = error

    def sample(
        self,
        batch_size,
        device,
        beta=0.4,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, torch.Tensor]:
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if not self.updated:
            print("WARNING: sample() called before errors updated from previous sample!")
        self.updated = False

        n = len(self._storage)
        s = self._weights[:n].sum()
        probs = self._weights[:n] / s if s > 0 else np.ones(n) / n
        idxs = np.random.choice(n, batch_size, replace=False, p=probs)

        # weight = (1/N * 1/P(i))^beta. Then normalize by max weight.
        weights = (n * probs[idxs])**(-beta)
        weights /= weights.max()

        samples = (self._storage[i] for i in idxs)
        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.array(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=bool),
        ), idxs, torch.tensor(weights, dtype=torch.float32, device=device)

    def update_errors(self, idxs: np.ndarray, errors: np.ndarray):
        self.updated = True
        self._weights[idxs] = errors


if __name__ == "__main__":
    exp_replay = ReplayBuffer(10)

    actions = np.random.randint(0, 3, size=30)
    states = np.random.randint(0, 100, size=30)
    new_states = np.random.randint(0, 100, size=30)
    rewards = np.random.randint(-10, 10, size=30)

    for i in range(30):
        exp_replay.add(states[i], actions[i], rewards[i], new_states[i], done=False)

    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = (
        exp_replay.sample(5)
    )
    print(obs_batch.shape)

    assert (
        len(exp_replay) == 10
    ), "experience replay size should be 10 because that's what maximum capacity is"

    prioritized_replay = PrioritizedReplayBuffer(10)
    errors = np.random.uniform(0, 5, size=30)

    for i in range(30):
        prioritized_replay.add(states[i], actions[i], rewards[i], new_states[i], done=False)

    batch, idxs = prioritized_replay.sample(5)
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = batch

    print(obs_batch.shape)

    assert (
        len(prioritized_replay) == 10
    ), "experience replay size should be 10 because that's what maximum capacity is"

    _, _ = prioritized_replay.sample(5)
