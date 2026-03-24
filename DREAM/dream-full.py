import argparse
import copy
import json
import os
import pickle
import socket
import subprocess
import sys
import time
from os.path import join as ospj
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F_torch
from torch.optim import lr_scheduler

from PokerRL.game import Poker, bet_sets
from PokerRL.game._.tree._.nodes import PlayerActionNode
from PokerRL.game.games import DiscretizedNLLeduc, Flop3Holdem, StandardLeduc
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
from PokerRL.game.wrappers import FlatLimitPokerEnvBuilder
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.EvalAgentBase import EvalAgentBase as _EvalAgentBase
from PokerRL.rl.base_cls.TrainingProfileBase import TrainingProfileBase
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase as _ChiefBase
from PokerRL.rl.base_cls.workers.DriverBase import DriverBase
from PokerRL.rl.base_cls.workers.ParameterServerBase import ParameterServerBase
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase
from PokerRL.rl.base_cls.HighLevelAlgoBase import (
    HighLevelAlgoBase as _HighLevelAlgoBase,
)
from PokerRL.rl.errors import UnknownModeError
from PokerRL.rl.neural.AvrgStrategyNet import AvrgNetArgs, AvrgStrategyNet
from PokerRL.rl.neural.CardEmbedding import CardEmbedding
from PokerRL.rl.neural.DuelingQNet import DuelingQArgs, DuelingQNet
from PokerRL.rl.neural.LayerNorm import LayerNorm
from PokerRL.rl.neural.MainPokerModuleFLAT import MPMArgsFLAT
from PokerRL.rl.neural.NetWrapperBase import NetWrapperArgsBase as _NetWrapperArgsBase
from PokerRL.rl.neural.NetWrapperBase import NetWrapperBase as _NetWrapperBase
from PokerRL.util import file_util


# ============================================================
# DREAM code vendored into one file.
# PokerRL stays as a normal import dependency.
# ============================================================


# ============================================================
# MainPokerModuleFLAT_Baseline
# ============================================================

# Copyright (c) 2019 Eric Steinberger


class MainPokerModuleFLAT_Baseline(nn.Module):
    def __init__(
        self,
        env_bldr,
        device,
        mpm_args,
    ):
        super().__init__()

        self._args = mpm_args
        self._env_bldr = env_bldr

        self._device = device

        self._board_start = self._env_bldr.obs_board_idxs[0]
        self._board_stop = self._board_start + len(self._env_bldr.obs_board_idxs)

        self.dropout = nn.Dropout(p=mpm_args.dropout)

        self.card_emb = CardEmbedding(
            env_bldr=env_bldr, dim=mpm_args.dim, device=device
        )

        if mpm_args.deep:
            self.cards_fc_1 = nn.Linear(
                in_features=self.card_emb.out_size * 2, out_features=mpm_args.dim * 3
            )
            self.cards_fc_2 = nn.Linear(
                in_features=mpm_args.dim * 3, out_features=mpm_args.dim * 3
            )
            self.cards_fc_3 = nn.Linear(
                in_features=mpm_args.dim * 3, out_features=mpm_args.dim
            )

            self.history_1 = nn.Linear(
                in_features=self._env_bldr.pub_obs_size - self._env_bldr.obs_size_board,
                out_features=mpm_args.dim,
            )
            self.history_2 = nn.Linear(
                in_features=mpm_args.dim, out_features=mpm_args.dim
            )

            self.comb_1 = nn.Linear(
                in_features=2 * mpm_args.dim, out_features=mpm_args.dim
            )
            self.comb_2 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)

        else:
            self.layer_1 = nn.Linear(
                in_features=self.card_emb.out_size * 2
                + self._env_bldr.pub_obs_size
                - self._env_bldr.obs_size_board,
                out_features=mpm_args.dim,
            )
            self.layer_2 = nn.Linear(
                in_features=mpm_args.dim, out_features=mpm_args.dim
            )
            self.layer_3 = nn.Linear(
                in_features=mpm_args.dim, out_features=mpm_args.dim
            )

        if self._args.normalize:
            self.norm = LayerNorm(mpm_args.dim)

        self.to(device)
        # print("n parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    @property
    def output_units(self):
        return self._args.dim

    @property
    def device(self):
        return self._device

    def forward(self, pub_obses, range_idxs):
        """
        1. do list -> padded
        2. feed through pre-processing fc layers
        3. PackedSequence (sort, pack)
        4. rnn
        5. unpack (unpack re-sort)
        6. cut output to only last entry in sequence

        Args:
            pub_obses (list):                 list of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (LongTensor):        range_idxs (one for each pub_obs) tensor([2, 421, 58, 912, ...])
        """
        if isinstance(pub_obses, list):
            pub_obses = torch.from_numpy(np.array(pub_obses)).to(
                self._device, torch.float32
            )

        hist_o = torch.cat(
            [pub_obses[:, : self._board_start], pub_obses[:, self._board_stop :]],
            dim=-1,
        )

        # """""""""""""""""""""""
        # Card embeddings
        # """""""""""""""""""""""
        range_idxs_0 = (
            range_idxs // 10000
        )  # Big hack! See LearnedBaselineSampler for the reverse opp
        range_idxs_1 = range_idxs % 10000

        card_o_0 = self.card_emb(
            pub_obses=pub_obses,
            range_idxs=torch.where(
                range_idxs_0 == 8888, torch.zeros_like(range_idxs_0), range_idxs_0
            ),
        )

        card_o_0 = torch.where(
            range_idxs_0.unsqueeze(1).expand_as(card_o_0) == 8888,
            torch.full_like(card_o_0, fill_value=-1),
            card_o_0,
        )

        card_o_1 = self.card_emb(
            pub_obses=pub_obses,
            range_idxs=torch.where(
                range_idxs_1 == 8888, torch.zeros_like(range_idxs_1), range_idxs_1
            ),
        )
        card_o_1 = torch.where(
            range_idxs_1.unsqueeze(1).expand_as(card_o_0) == 8888,
            torch.full_like(card_o_1, fill_value=-1),
            card_o_1,
        )
        card_o = torch.cat([card_o_0, card_o_1], dim=-1)

        # """""""""""""""""""""""
        # Network
        # """""""""""""""""""""""
        if self._args.dropout > 0:
            A = lambda x: self.dropout(F.relu(x))
        else:
            A = lambda x: F.relu(x)

        if self._args.deep:
            card_o = A(self.cards_fc_1(card_o))
            card_o = A(self.cards_fc_2(card_o) + card_o)
            card_o = A(self.cards_fc_3(card_o))

            hist_o = A(self.history_1(hist_o))
            hist_o = A(self.history_2(hist_o) + hist_o)

            y = A(self.comb_1(torch.cat([card_o, hist_o], dim=-1)))
            y = A(self.comb_2(y) + y)

        else:
            y = torch.cat([hist_o, card_o], dim=-1)
            y = A(self.layer_1(y))
            y = A(self.layer_2(y) + y)
            y = A(self.layer_3(y) + y)

        # """""""""""""""""""""""
        # Normalize last layer
        # """""""""""""""""""""""
        if self._args.normalize:
            y = self.norm(y)

        return y


class MPMArgsFLAT_Baseline:
    def __init__(
        self,
        deep=True,
        dim=128,
        dropout=0.0,
        normalize=True,
    ):
        self.deep = deep
        self.dim = dim
        self.dropout = dropout
        self.normalize = normalize

    def get_mpm_cls(self):
        return MainPokerModuleFLAT_Baseline


# ============================================================
# AdvWrapper
# ============================================================

# Copyright (c) Eric Steinberger 2020


class AdvWrapper(_NetWrapperBase):
    def __init__(self, env_bldr, adv_training_args, owner, device):
        super().__init__(
            net=DuelingQNet(
                env_bldr=env_bldr, q_args=adv_training_args.adv_net_args, device=device
            ),
            env_bldr=env_bldr,
            args=adv_training_args,
            owner=owner,
            device=device,
        )

    def get_advantages(self, pub_obses, range_idxs, legal_action_mask):
        self._net.eval()
        with torch.no_grad():
            return self._net(
                pub_obses=pub_obses,
                range_idxs=range_idxs,
                legal_action_masks=legal_action_mask,
            )

    def _mini_batch_loop(self, buffer, grad_mngr):
        (
            batch_pub_obs,
            batch_range_idxs,
            batch_legal_action_masks,
            batch_adv,
            batch_loss_weight,
        ) = buffer.sample(device=self.device, batch_size=self._args.batch_size)

        # [batch_size, n_actions]
        adv_pred = self._net(
            pub_obses=batch_pub_obs,
            range_idxs=batch_range_idxs,
            legal_action_masks=batch_legal_action_masks,
        )

        grad_mngr.backprop(
            pred=adv_pred,
            target=batch_adv,
            loss_weights=batch_loss_weight.unsqueeze(-1).expand_as(batch_adv),
        )


class AdvTrainingArgs(_NetWrapperArgsBase):
    def __init__(
        self,
        adv_net_args,
        n_batches_adv_training=1000,
        batch_size=4096,
        optim_str="adam",
        loss_str="weighted_mse",
        lr=0.001,
        grad_norm_clipping=10.0,
        device_training="cpu",
        max_buffer_size=2e6,
        lr_patience=100,
        init_adv_model="last",
    ):
        super().__init__(
            batch_size=batch_size,
            optim_str=optim_str,
            loss_str=loss_str,
            lr=lr,
            grad_norm_clipping=grad_norm_clipping,
            device_training=device_training,
        )
        self.adv_net_args = adv_net_args
        self.n_batches_adv_training = n_batches_adv_training
        self.lr_patience = lr_patience
        self.max_buffer_size = int(max_buffer_size)
        self.init_adv_model = init_adv_model


# ============================================================
# AvrgWrapper
# ============================================================

# Copyright (c) Eric Steinberger 2020


class AvrgWrapper(_NetWrapperBase):
    def __init__(self, owner, env_bldr, avrg_training_args, device):
        super().__init__(
            net=AvrgStrategyNet(
                avrg_net_args=avrg_training_args.avrg_net_args,
                env_bldr=env_bldr,
                device=device,
            ),
            env_bldr=env_bldr,
            args=avrg_training_args,
            owner=owner,
            device=device,
        )
        self._all_range_idxs = torch.arange(
            self._env_bldr.rules.RANGE_SIZE, device=self.device, dtype=torch.long
        )

    def get_a_probs(self, pub_obses, range_idxs, legal_actions_lists):
        """
        Args:
            pub_obses (list):             list of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (np.ndarray):    array of range_idxs (one for each pub_obs) tensor([2, 421, 58, 912, ...])
            legal_actions_lists (list:  list of lists. each 2nd level lists contains ints representing legal actions
        """
        with torch.no_grad():
            masks = rl_util.batch_get_legal_action_mask_torch(
                n_actions=self._env_bldr.N_ACTIONS,
                legal_actions_lists=legal_actions_lists,
                device=self.device,
            )
            masks = masks.view(1, -1)
            return self.get_a_probs2(
                pub_obses=pub_obses, range_idxs=range_idxs, legal_action_masks=masks
            )

    def get_a_probs2(self, pub_obses, range_idxs, legal_action_masks):
        with torch.no_grad():
            pred = self._net(
                pub_obses=pub_obses,
                range_idxs=torch.from_numpy(range_idxs).to(
                    dtype=torch.long, device=self.device
                ),
                legal_action_masks=legal_action_masks,
            )

            return F.softmax(pred, dim=-1).cpu().numpy()

    def get_a_probs_for_each_hand(self, pub_obs, legal_actions_list):
        with torch.no_grad():
            mask = rl_util.get_legal_action_mask_torch(
                n_actions=self._env_bldr.N_ACTIONS,
                legal_actions_list=legal_actions_list,
                device=self.device,
                dtype=torch.uint8,
            )
            mask = mask.unsqueeze(0).expand(self._env_bldr.rules.RANGE_SIZE, -1)

            pred = self._net(
                pub_obses=[pub_obs] * self._env_bldr.rules.RANGE_SIZE,
                range_idxs=self._all_range_idxs,
                legal_action_masks=mask,
            )

            return F.softmax(pred, dim=1).cpu().numpy()

    def _mini_batch_loop(self, buffer, grad_mngr):
        (
            batch_pub_obs,
            batch_range_idxs,
            batch_legal_action_masks,
            batch_a_probs,
            batch_loss_weight,
        ) = buffer.sample(device=self.device, batch_size=self._args.batch_size)

        # [batch_size, n_actions]
        strat_pred = self._net(
            pub_obses=batch_pub_obs,
            range_idxs=batch_range_idxs,
            legal_action_masks=batch_legal_action_masks,
        )
        strat_pred = F.softmax(strat_pred, dim=-1)
        grad_mngr.backprop(
            pred=strat_pred,
            target=batch_a_probs,
            loss_weights=batch_loss_weight.unsqueeze(-1).expand_as(batch_a_probs),
        )


class AvrgTrainingArgs(_NetWrapperArgsBase):
    def __init__(
        self,
        avrg_net_args,
        n_batches_avrg_training=1000,
        batch_size=4096,
        optim_str="adam",
        loss_str="weighted_mse",
        lr=0.001,
        grad_norm_clipping=10.0,
        device_training="cpu",
        max_buffer_size=2e6,
        lr_patience=100,
        init_avrg_model="random",
    ):
        super().__init__(
            batch_size=batch_size,
            optim_str=optim_str,
            loss_str=loss_str,
            lr=lr,
            grad_norm_clipping=grad_norm_clipping,
            device_training=device_training,
        )

        self.avrg_net_args = avrg_net_args
        self.n_batches_avrg_training = n_batches_avrg_training
        self.max_buffer_size = int(max_buffer_size)
        self.lr_patience = lr_patience
        self.init_avrg_model = init_avrg_model


# ============================================================
# LearnedBaselineLearner
# ============================================================

# Copyright (c) Eric Steinberger 2020


class BaselineWrapper(_NetWrapperBase):
    def __init__(self, env_bldr, baseline_args):
        super().__init__(
            net=DuelingQNet(
                env_bldr=env_bldr,
                q_args=baseline_args.q_net_args,
                device=baseline_args.device_training,
            ),
            owner=None,
            env_bldr=env_bldr,
            args=baseline_args,
            device=baseline_args.device_training,
        )

        self._batch_arranged = torch.arange(
            self._args.batch_size, dtype=torch.long, device=self.device
        )
        self._minus_e20 = torch.full(
            (
                self._args.batch_size,
                self._env_bldr.N_ACTIONS,
            ),
            fill_value=-10e20,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

    def get_b(self, pub_obses, range_idxs, legal_actions_lists, to_np=False):
        with torch.no_grad():
            range_idxs = torch.tensor(range_idxs, dtype=torch.long, device=self.device)

            masks = rl_util.batch_get_legal_action_mask_torch(
                n_actions=self._env_bldr.N_ACTIONS,
                legal_actions_lists=legal_actions_lists,
                device=self.device,
                dtype=torch.float32,
            )
            self.eval()
            q = self._net(
                pub_obses=pub_obses, range_idxs=range_idxs, legal_action_masks=masks
            )
            q *= masks

            if to_np:
                q = q.cpu().numpy()

            return q

    def _mini_batch_loop(self, buffer, grad_mngr):
        (
            batch_pub_obs_t,
            batch_range_idx,
            batch_legal_action_mask_t,
            batch_a_t,
            batch_r_t,
            batch_pub_obs_tp1,
            batch_legal_action_mask_tp1,
            batch_done,
            batch_strat_tp1,
        ) = buffer.sample(device=self.device, batch_size=self._args.batch_size)

        # [batch_size, n_actions]
        q1_t = self._net(
            pub_obses=batch_pub_obs_t,
            range_idxs=batch_range_idx,
            legal_action_masks=batch_legal_action_mask_t.to(torch.float32),
        )
        q1_tp1 = self._net(
            pub_obses=batch_pub_obs_tp1,
            range_idxs=batch_range_idx,
            legal_action_masks=batch_legal_action_mask_tp1.to(torch.float32),
        ).detach()

        # ______________________________________________ TD Learning _______________________________________________
        # [batch_size]
        q1_t_of_a_selected = q1_t[self._batch_arranged, batch_a_t]

        # only consider allowed actions for tp1
        q1_tp1 = torch.where(batch_legal_action_mask_tp1, q1_tp1, self._minus_e20)

        # [batch_size]
        q_tp1_of_atp1 = (q1_tp1 * batch_strat_tp1).sum(-1)
        q_tp1_of_atp1 *= 1.0 - batch_done
        target = batch_r_t + q_tp1_of_atp1

        grad_mngr.backprop(pred=q1_t_of_a_selected, target=target)


class BaselineArgs(_NetWrapperArgsBase):
    def __init__(
        self,
        q_net_args,
        max_buffer_size=2e5,
        n_batches_per_iter_baseline=500,
        batch_size=512,
        optim_str="adam",
        loss_str="mse",
        lr=0.001,
        grad_norm_clipping=1.0,
        device_training="cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            optim_str=optim_str,
            loss_str=loss_str,
            lr=lr,
            grad_norm_clipping=grad_norm_clipping,
            device_training=device_training,
        )
        self.q_net_args = q_net_args
        self.max_buffer_size = int(max_buffer_size)
        self.n_batches_per_iter_baseline = n_batches_per_iter_baseline


# ============================================================
# ReservoirBufferBase
# ============================================================

# Copyright (c) 2019 Eric Steinberger


class ReservoirBufferBase:
    def __init__(self, owner, max_size, env_bldr, nn_type, iter_weighting_exponent):
        self._owner = owner
        self._env_bldr = env_bldr
        self.device = torch.device("cpu")

        self._owner = owner
        self._env_bldr = env_bldr
        self._max_size = max_size
        self._nn_type = nn_type
        self.device = torch.device("cpu")
        self.size = 0
        self.n_entries_seen = 0

        if nn_type == "recurrent":
            self._pub_obs_buffer = np.empty(shape=(max_size,), dtype=object)
        elif nn_type == "feedforward":
            self._pub_obs_buffer = torch.zeros(
                (max_size, self._env_bldr.pub_obs_size),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise ValueError(nn_type)

        self._range_idx_buffer = torch.zeros(
            (max_size,), dtype=torch.long, device=self.device
        )
        self._legal_action_mask_buffer = torch.zeros(
            (
                max_size,
                env_bldr.N_ACTIONS,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self._iteration_buffer = torch.zeros(
            (max_size,), dtype=torch.float32, device=self.device
        )
        self._iter_weighting_exponent = iter_weighting_exponent

        self._last_iterationation_seen = None

    def add(self, **kwargs):
        """
        Dont forget to n_entries_seen+=1 !!
        """
        raise NotImplementedError

    def sample(self, batch_size, device):
        raise NotImplementedError

    def _should_add(self):
        return np.random.random() < (float(self._max_size) / float(self.n_entries_seen))

    def _np_to_torch(self, arr):
        return torch.from_numpy(np.copy(arr)).to(self.device)

    def _random_idx(self):
        return np.random.randint(low=0, high=self._max_size)

    def state_dict(self):
        return {
            "owner": self._owner,
            "max_size": self._max_size,
            "nn_type": self._nn_type,
            "size": self.size,
            "n_entries_seen": self.n_entries_seen,
            "iter_weighting_exponent": self._iter_weighting_exponent,
            "pub_obs_buffer": self._pub_obs_buffer,
            "range_idx_buffer": self._range_idx_buffer,
            "legal_action_mask_buffer": self._legal_action_mask_buffer,
            "iteration_buffer": self._iteration_buffer,
        }

    def load_state_dict(self, state):
        assert self._owner == state["owner"]
        assert self._max_size == state["max_size"]
        assert self._nn_type == state["nn_type"]

        self.size = state["size"]
        self.n_entries_seen = state["n_entries_seen"]
        self._iter_weighting_exponent = state["iter_weighting_exponent"]

        if self._nn_type == "recurrent":
            self._pub_obs_buffer = state["pub_obs_buffer"]
            self._range_idx_buffer = state["range_idx_buffer"]
            self._legal_action_mask_buffer = state["legal_action_mask_buffer"]
            self._iteration_buffer = state["iteration_buffer"]

        elif self._nn_type == "feedforward":
            self._pub_obs_buffer = state["pub_obs_buffer"].to(self.device)
            self._range_idx_buffer = state["range_idx_buffer"].to(self.device)
            self._legal_action_mask_buffer = state["legal_action_mask_buffer"].to(
                self.device
            )
            self._iteration_buffer = state["iteration_buffer"].to(self.device)

        else:
            raise ValueError(self._nn_type)


# ============================================================
# AdvReservoirBuffer
# ============================================================

# Copyright (c) Eric Steinberger 2020


class AdvReservoirBuffer(ReservoirBufferBase):
    def __init__(self, owner, nn_type, max_size, env_bldr, iter_weighting_exponent):
        super().__init__(
            owner=owner,
            max_size=max_size,
            env_bldr=env_bldr,
            nn_type=nn_type,
            iter_weighting_exponent=iter_weighting_exponent,
        )

        self._adv_buffer = torch.zeros(
            (max_size, env_bldr.N_ACTIONS), dtype=torch.float32, device=self.device
        )

    def add(self, pub_obs, range_idx, legal_action_mask, adv, iteration):
        if self.size < self._max_size:
            self._add(
                idx=self.size,
                pub_obs=pub_obs,
                range_idx=range_idx,
                legal_action_mask=legal_action_mask,
                adv=adv,
                iteration=iteration,
            )
            self.size += 1

        elif self._should_add():
            self._add(
                idx=self._random_idx(),
                pub_obs=pub_obs,
                range_idx=range_idx,
                legal_action_mask=legal_action_mask,
                adv=adv,
                iteration=iteration,
            )

        self.n_entries_seen += 1

    def sample(self, batch_size, device):
        indices = torch.randint(
            0, self.size, (batch_size,), dtype=torch.long, device=self.device
        )

        if self._nn_type == "recurrent":
            obses = self._pub_obs_buffer[indices.cpu().numpy()]
        elif self._nn_type == "feedforward":
            obses = self._pub_obs_buffer[indices].to(device)
        else:
            raise NotImplementedError

        return (
            obses,
            self._range_idx_buffer[indices].to(device),
            self._legal_action_mask_buffer[indices].to(device),
            self._adv_buffer[indices].to(device),
            self._iteration_buffer[indices].to(device) / self._last_iterationation_seen,
        )

    def _add(self, idx, pub_obs, range_idx, legal_action_mask, adv, iteration):
        if self._nn_type == "feedforward":
            pub_obs = torch.from_numpy(pub_obs)

        self._pub_obs_buffer[idx] = pub_obs
        self._range_idx_buffer[idx] = range_idx
        self._legal_action_mask_buffer[idx] = legal_action_mask
        self._adv_buffer[idx] = adv

        self._iteration_buffer[idx] = float(iteration) ** self._iter_weighting_exponent

        self._last_iterationation_seen = iteration

    def state_dict(self):
        return {
            "base": super().state_dict(),
            "adv": self._adv_buffer,
        }

    def load_state_dict(self, state):
        super().load_state_dict(state["base"])
        self._adv_buffer = state["adv"]


# ============================================================
# AvrgReservoirBuffer
# ============================================================

# Copyright (c) Eric Steinberger 2020


class AvrgReservoirBuffer(ReservoirBufferBase):
    """
    Reservoir buffer to store state+action samples for the average strategy network
    """

    def __init__(self, owner, nn_type, max_size, env_bldr, iter_weighting_exponent):
        super().__init__(
            owner=owner,
            max_size=max_size,
            env_bldr=env_bldr,
            nn_type=nn_type,
            iter_weighting_exponent=iter_weighting_exponent,
        )

        self._a_probs_buffer = torch.zeros(
            (max_size, env_bldr.N_ACTIONS), dtype=torch.float32, device=self.device
        )

    def add(self, pub_obs, range_idx, legal_actions_list, a_probs, iteration):
        if self.size < self._max_size:
            self._add(
                idx=self.size,
                pub_obs=pub_obs,
                range_idx=range_idx,
                legal_action_mask=self._get_mask(legal_actions_list),
                action_probs=a_probs,
                iteration=iteration,
            )
            self.size += 1

        elif self._should_add():
            self._add(
                idx=self._random_idx(),
                pub_obs=pub_obs,
                range_idx=range_idx,
                legal_action_mask=self._get_mask(legal_actions_list),
                action_probs=a_probs,
                iteration=iteration,
            )

        self.n_entries_seen += 1

    def sample(self, batch_size, device):
        indices = torch.randint(
            0, self.size, (batch_size,), dtype=torch.long, device=self.device
        )

        if self._nn_type == "recurrent":
            obses = self._pub_obs_buffer[indices.cpu().numpy()]
        elif self._nn_type == "feedforward":
            obses = self._pub_obs_buffer[indices].to(device)
        else:
            raise NotImplementedError

        return (
            obses,
            self._range_idx_buffer[indices].to(device),
            self._legal_action_mask_buffer[indices].to(device),
            self._a_probs_buffer[indices].to(device),
            self._iteration_buffer[indices].to(device) / self._last_iterationation_seen,
        )

    def _add(self, idx, pub_obs, range_idx, legal_action_mask, action_probs, iteration):
        if self._nn_type == "feedforward":
            pub_obs = torch.from_numpy(pub_obs)

        self._pub_obs_buffer[idx] = pub_obs
        self._range_idx_buffer[idx] = range_idx
        self._legal_action_mask_buffer[idx] = legal_action_mask
        self._a_probs_buffer[idx] = action_probs

        # In "https://arxiv.org/pdf/1811.00164.pdf", Brown et al. weight by floor((t+1)/2), but we assume that
        # this is due to incrementation happening for every alternating update. We count one iteration as an
        # update for both plyrs.
        self._iteration_buffer[idx] = float(iteration) ** self._iter_weighting_exponent
        self._last_iterationation_seen = iteration

    def _get_mask(self, legal_actions_list):
        return rl_util.get_legal_action_mask_torch(
            n_actions=self._env_bldr.N_ACTIONS,
            legal_actions_list=legal_actions_list,
            device=self.device,
            dtype=torch.float32,
        )

    def state_dict(self):
        return {
            "base": super().state_dict(),
            "a_probs": self._a_probs_buffer,
        }

    def load_state_dict(self, state):
        super().load_state_dict(state["base"])
        self._a_probs_buffer = state["a_probs"]


# ============================================================
# CrazyBaselineQCircularBuffer
# ============================================================

# Copyright (c) Eric Steinberger 2020


class CrazyBaselineQCircularBuffer:
    """
    Circular buffer compatible with all NN architectures
    """

    def __init__(self, owner, max_size, env_bldr, nn_type):
        self._owner = owner
        self._env_bldr = env_bldr
        self._max_size = int(max_size)

        self._nn_type = nn_type
        self.device = torch.device("cpu")
        self.size = 0

        if nn_type == "recurrent":
            self._pub_obs_buffer = np.empty(shape=(max_size,), dtype=object)
        elif nn_type == "feedforward":
            self._pub_obs_buffer = torch.zeros(
                (max_size, self._env_bldr.pub_obs_size),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise ValueError(nn_type)

        self._range_idx_buffer = torch.zeros(
            (max_size,), dtype=torch.long, device=self.device
        )
        self._legal_action_mask_buffer = torch.zeros(
            (
                max_size,
                env_bldr.N_ACTIONS,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self._top = None

        self._a_buffer = None
        self._strat_tp1_buffer = None
        self._r_buffer = None
        self._done = None
        self._pub_obs_buffer_tp1 = None
        self._legal_action_mask_buffer_tp1 = None
        self.reset()

    def _np_to_torch(self, arr):
        return torch.from_numpy(np.copy(arr)).to(self.device)

    def _random_idx(self):
        return np.random.randint(low=0, high=self._max_size)

    def add(
        self,
        pub_obs,
        range_idx_crazy_embedded,
        legal_action_mask,
        r,
        a,
        done,
        legal_action_mask_tp1,
        pub_obs_tp1,
        strat_tp1,
    ):
        if self._nn_type == "feedforward":
            pub_obs = torch.from_numpy(pub_obs)
            pub_obs_tp1 = torch.from_numpy(pub_obs_tp1)

        self._pub_obs_buffer[self._top] = pub_obs
        self._pub_obs_buffer_tp1[self._top] = pub_obs_tp1

        self._range_idx_buffer[self._top] = range_idx_crazy_embedded

        self._legal_action_mask_buffer[self._top] = legal_action_mask
        self._legal_action_mask_buffer_tp1[self._top] = legal_action_mask_tp1

        self._r_buffer[self._top] = r
        self._a_buffer[self._top] = a
        self._done[self._top] = float(done)

        self._strat_tp1_buffer[self._top] = strat_tp1
        if self.size < self._max_size:
            self.size += 1

        self._top = (self._top + 1) % self._max_size

    def sample(self, batch_size, device):
        indices = torch.randint(
            0, self.size, (batch_size,), dtype=torch.long, device=self.device
        )

        if self._nn_type == "recurrent":
            obses = self._pub_obs_buffer[indices.cpu().numpy()]
            obses_tp1 = self._pub_obs_buffer_tp1[indices.cpu().numpy()]
        elif self._nn_type == "feedforward":
            obses = self._pub_obs_buffer[indices].to(device)
            obses_tp1 = self._pub_obs_buffer_tp1[indices].to(device)
        else:
            raise NotImplementedError

        return (
            obses,
            self._range_idx_buffer[indices].to(device),
            self._legal_action_mask_buffer[indices].to(device),
            self._a_buffer[indices].to(device),
            self._r_buffer[indices].to(device),
            obses_tp1,
            self._legal_action_mask_buffer_tp1[indices].to(device),
            self._done[indices].to(device),
            self._strat_tp1_buffer[indices].to(device),
        )

    def reset(self):
        self._top = 0
        self.size = 0

        if self._nn_type == "recurrent":
            self._pub_obs_buffer = np.empty(shape=(self._max_size,), dtype=object)
        elif self._nn_type == "feedforward":
            self._pub_obs_buffer = torch.zeros(
                (self._max_size, self._env_bldr.pub_obs_size),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise ValueError(self._nn_type)

        self._range_idx_buffer = torch.zeros(
            (self._max_size,), dtype=torch.long, device=self.device
        )
        self._legal_action_mask_buffer = torch.zeros(
            (
                self._max_size,
                self._env_bldr.N_ACTIONS,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self._a_buffer = torch.zeros(
            (self._max_size,), dtype=torch.long, device=self.device
        )
        self._strat_tp1_buffer = torch.zeros(
            (self._max_size, self._env_bldr.N_ACTIONS),
            dtype=torch.float32,
            device=self.device,
        )
        self._r_buffer = torch.zeros(
            (self._max_size,), dtype=torch.float32, device=self.device
        )
        self._done = torch.zeros(
            (self._max_size,), dtype=torch.float32, device=self.device
        )

        if self._nn_type == "recurrent":
            self._pub_obs_buffer_tp1 = np.empty(shape=(self._max_size,), dtype=object)
        elif self._nn_type == "feedforward":
            self._pub_obs_buffer_tp1 = torch.zeros(
                (self._max_size, self._env_bldr.pub_obs_size),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise ValueError(self._nn_type)

        self._legal_action_mask_buffer_tp1 = torch.zeros(
            (
                self._max_size,
                self._env_bldr.N_ACTIONS,
            ),
            dtype=torch.uint8,
            device=self.device,
        )

    def state_dict(self):
        return {
            "owner": self._owner,
            "max_size": self._max_size,
            "nn_type": self._nn_type,
            "size": self.size,
            "pub_obs_buffer": self._pub_obs_buffer,
            "range_idx_buffer": self._range_idx_buffer,
            "legal_action_mask_buffer": self._legal_action_mask_buffer,
            "a": self._a_buffer,
            "q": self._r_buffer,
            "legal_action_mask_buffer_tp1": self._legal_action_mask_buffer_tp1,
            "pub_obs_buffer_tp1": self._pub_obs_buffer_tp1,
            "done": self._done,
            "strat_tp1": self._strat_tp1_buffer,
        }

    def load_state_dict(self, state):
        assert self._owner == state["owner"]
        assert self._max_size == state["max_size"]
        assert self._nn_type == state["nn_type"]

        self.size = state["size"]

        if self._nn_type == "recurrent":
            self._pub_obs_buffer = state["pub_obs_buffer"]
            self._range_idx_buffer = state["range_idx_buffer"]
            self._legal_action_mask_buffer = state["legal_action_mask_buffer"]

        elif self._nn_type == "feedforward":
            self._pub_obs_buffer = state["pub_obs_buffer"].to(self.device)
            self._range_idx_buffer = state["range_idx_buffer"].to(self.device)
            self._legal_action_mask_buffer = state["legal_action_mask_buffer"].to(
                self.device
            )

        else:
            raise ValueError(self._nn_type)

        self._a_buffer = state["a"]
        self._r_buffer = state["q"]
        self._legal_action_mask_buffer_tp1 = state["legal_action_mask_buffer_tp1"]
        self._done = state["done"]
        self._pub_obs_buffer_tp1 = state["pub_obs_buffer_tp1"]
        self._strat_tp1_buffer = state["strat_tp1"]


# ============================================================
# SamplerBase
# ============================================================

# Copyright (c) Eric Steinberger 2020


class SamplerBase:
    def __init__(
        self,
        env_bldr,
        adv_buffers,
        avrg_buffers=None,
    ):
        self._env_bldr = env_bldr
        self._adv_buffers = adv_buffers
        self._avrg_buffers = avrg_buffers
        self._env_wrapper = self._env_bldr.get_new_wrapper(is_evaluating=False)
        self.total_node_count_traversed = 0

    def _traverser_act(
        self,
        start_state_dict,
        traverser,
        trav_depth,
        plyrs_range_idxs,
        iteration_strats,
        sample_reach,
        cfr_iter,
    ):
        raise NotImplementedError

    def generate(
        self,
        n_traversals,
        traverser,
        iteration_strats,
        cfr_iter,
    ):
        for _ in range(n_traversals):
            self._traverse_once(
                traverser=traverser,
                iteration_strats=iteration_strats,
                cfr_iter=cfr_iter,
            )

    def _traverse_once(
        self,
        traverser,
        iteration_strats,
        cfr_iter,
    ):
        """
        Args:
            traverser (int):                    seat id of the traverser
            iteration_strats (IterationStrategy):
            cfr_iter (int):                  current iteration of Deep CFR
        """
        self._env_wrapper.reset()
        self._recursive_traversal(
            start_state_dict=self._env_wrapper.state_dict(),
            traverser=traverser,
            trav_depth=0,
            plyrs_range_idxs=[
                self._env_wrapper.env.get_range_idx(p_id=p_id)
                for p_id in range(self._env_bldr.N_SEATS)
            ],
            sample_reach=1.0,
            iteration_strats=iteration_strats,
            cfr_iter=cfr_iter,
        )

    def _recursive_traversal(
        self,
        start_state_dict,
        traverser,
        trav_depth,
        plyrs_range_idxs,
        iteration_strats,
        cfr_iter,
        sample_reach=None,
    ):
        """
        assumes passed state_dict is NOT done!
        """

        if start_state_dict["base"]["env"][EnvDictIdxs.current_player] == traverser:
            return self._traverser_act(
                start_state_dict=start_state_dict,
                traverser=traverser,
                trav_depth=trav_depth,
                plyrs_range_idxs=plyrs_range_idxs,
                iteration_strats=iteration_strats,
                sample_reach=sample_reach,
                cfr_iter=cfr_iter,
            )

        return self._any_non_traverser_act(
            start_state_dict=start_state_dict,
            traverser=traverser,
            trav_depth=trav_depth,
            plyrs_range_idxs=plyrs_range_idxs,
            iteration_strats=iteration_strats,
            sample_reach=sample_reach,
            cfr_iter=cfr_iter,
        )

    def _any_non_traverser_act(
        self,
        start_state_dict,
        traverser,
        plyrs_range_idxs,
        trav_depth,
        iteration_strats,
        sample_reach,
        cfr_iter,
    ):
        self._env_wrapper.load_state_dict(start_state_dict)
        p_id_acting = self._env_wrapper.env.current_player.seat_id

        current_pub_obs = self._env_wrapper.get_current_obs()
        range_idx = plyrs_range_idxs[p_id_acting]
        legal_actions_list = self._env_wrapper.env.get_legal_actions()

        # """""""""""""""""""""""""
        # The players strategy
        # """""""""""""""""""""""""
        a_probs = iteration_strats[p_id_acting].get_a_probs(
            pub_obses=[current_pub_obs],
            range_idxs=[range_idx],
            legal_actions_lists=[legal_actions_list],
            to_np=False,
        )[0]

        # """""""""""""""""""""""""
        # Adds to opponent's
        # average buffer if
        # applicable
        # """""""""""""""""""""""""
        if self._avrg_buffers is not None:
            self._avrg_buffers[p_id_acting].add(
                pub_obs=current_pub_obs,
                range_idx=range_idx,
                legal_actions_list=legal_actions_list,
                a_probs=a_probs.to(self._avrg_buffers[p_id_acting].device).squeeze(),
                iteration=cfr_iter + 1,
            )

        # """""""""""""""""""""""""
        # Execute action from strat
        # """""""""""""""""""""""""
        a = torch.multinomial(a_probs.cpu(), num_samples=1).item()
        _obs, _rew_for_all, _done, _info = self._env_wrapper.step(a)
        _rew_traverser = _rew_for_all[traverser]

        # """""""""""""""""""""""""
        # Recurse or Return if done
        # """""""""""""""""""""""""
        if _done:
            self.total_node_count_traversed += 1
            return _rew_traverser
        return _rew_traverser + self._recursive_traversal(
            start_state_dict=self._env_wrapper.state_dict(),
            traverser=traverser,
            trav_depth=trav_depth,
            plyrs_range_idxs=plyrs_range_idxs,
            iteration_strats=iteration_strats,
            sample_reach=sample_reach,
            cfr_iter=cfr_iter,
        )


# ============================================================
# LearnedBaselineSampler
# ============================================================

# Copyright (c) Eric Steinberger 2020


class LearnedBaselineSampler(SamplerBase):
    """
    How to get to next state:
        -   Each time ""traverser"" acts, a number of sub-trees are followed. For each sample, the remaining deck is
            reshuffled to ensure a random future.

        -   When any other player acts, 1 action is chosen w.r.t. their strategy.

        -   When the environment acts, 1 action is chosen according to its natural dynamics. Note that the PokerRL
            environment does this inherently, which is why there is no code for that in this class.


    When what is stored to where:
        -   At every time a player other than ""traverser"" acts, we store their action probability vector to their
            reservoir buffer.

        -   Approximate immediate regrets are stored to ""traverser""'s advantage buffer at every node at which they
            act.
    """

    def __init__(
        self,
        env_bldr,
        adv_buffers,
        baseline_net,
        baseline_buf,
        eps=0.5,
        avrg_buffers=None,
    ):
        super().__init__(
            env_bldr=env_bldr, adv_buffers=adv_buffers, avrg_buffers=avrg_buffers
        )
        self._baseline_net = baseline_net
        self._baseline_buf = baseline_buf

        # self._reg_buf = None

        self._eps = eps
        self._actions_arranged = np.arange(self._env_bldr.N_ACTIONS)

        self.total_node_count_traversed = 0

    def generate(
        self,
        n_traversals,
        traverser,
        iteration_strats,
        cfr_iter,
    ):
        # self._reg_buf = [[] for _ in range(self._env_bldr.rules.N_CARDS_IN_DECK)]

        super().generate(n_traversals, traverser, iteration_strats, cfr_iter)
        # if traverser == 0:
        #     print("STD:  ", np.sum(np.array([np.array(x).std(axis=0) for x in self._reg_buf]), axis=0))
        #     print("Mean: ", np.sum(np.array([np.array(x).mean(axis=0) for x in self._reg_buf]), axis=0))

    def _traverser_act(
        self,
        start_state_dict,
        traverser,
        trav_depth,
        plyrs_range_idxs,
        iteration_strats,
        sample_reach,
        cfr_iter,
    ):
        """
        Last state values are the average, not the sum of all samples of that state since we add
        v~(I) = * p(a) * |A(I)|. Since we sample multiple actions on each traverser node, we have to average over
        their returns like: v~(I) * Sum_a=0_N (v~(I|a) * p(a) * ||A(I)|| / N).
        """
        self.total_node_count_traversed += 1
        self._env_wrapper.load_state_dict(start_state_dict)
        legal_actions_list = self._env_wrapper.env.get_legal_actions()
        legal_action_mask = rl_util.get_legal_action_mask_torch(
            n_actions=self._env_bldr.N_ACTIONS,
            legal_actions_list=legal_actions_list,
            device=self._adv_buffers[traverser].device,
            dtype=torch.float32,
        )
        pub_obs_t = self._env_wrapper.get_current_obs()
        traverser_range_idx = plyrs_range_idxs[traverser]

        # """""""""""""""""""""""""
        # Strategy
        # """""""""""""""""""""""""
        strat_i = iteration_strats[traverser].get_a_probs(
            pub_obses=[pub_obs_t],
            range_idxs=[traverser_range_idx],
            legal_actions_lists=[legal_actions_list],
            to_np=False,
        )[0]

        # """""""""""""""""""""""""
        # Sample action
        # """""""""""""""""""""""""
        n_legal_actions = len(legal_actions_list)
        sample_strat = (1 - self._eps) * strat_i + self._eps * (
            legal_action_mask / n_legal_actions
        )
        a = torch.multinomial(sample_strat, num_samples=1).item()

        # Step
        pub_obs_tp1, rew_for_all, done, _info = self._env_wrapper.step(a)
        legal_action_mask_tp1 = rl_util.get_legal_action_mask_torch(
            n_actions=self._env_bldr.N_ACTIONS,
            legal_actions_list=self._env_wrapper.env.get_legal_actions(),
            device=self._adv_buffers[traverser].device,
            dtype=torch.float32,
        )

        # """""""""""""""""""""""""
        # Recursion
        # """""""""""""""""""""""""
        if done:
            strat_tp1 = torch.zeros_like(strat_i)
        else:
            u_bootstrap, strat_tp1 = self._recursive_traversal(
                start_state_dict=self._env_wrapper.state_dict(),
                traverser=traverser,
                trav_depth=trav_depth + 1,
                plyrs_range_idxs=plyrs_range_idxs,
                iteration_strats=iteration_strats,
                cfr_iter=cfr_iter,
                sample_reach=sample_reach * sample_strat[a] * n_legal_actions,
            )

        # """""""""""""""""""""""""
        # Utility
        # """""""""""""""""""""""""
        utility = self._get_utility(
            traverser=traverser,
            u_bootstrap=rew_for_all[traverser] if done else u_bootstrap,
            range_idx_crazy_embedded=_crazy_embed(plyrs_range_idxs=plyrs_range_idxs),
            pub_obs=pub_obs_t,
            legal_actions_list=legal_actions_list,
            legal_action_mask=legal_action_mask,
            a=a,
            sample_strat=sample_strat,
        )

        # Regret
        aprx_imm_reg = torch.full(
            size=(self._env_bldr.N_ACTIONS,),
            fill_value=-(utility * strat_i).sum(),
            dtype=torch.float32,
            device=self._adv_buffers[traverser].device,
        )
        aprx_imm_reg += utility
        aprx_imm_reg *= legal_action_mask

        # add current datapoint to ADVBuf
        self._adv_buffers[traverser].add(
            pub_obs=pub_obs_t,
            range_idx=traverser_range_idx,
            legal_action_mask=legal_action_mask,
            adv=aprx_imm_reg,
            iteration=(cfr_iter + 1) / sample_reach,
        )

        # add datapoint to baseline net
        self._baseline_buf.add(
            pub_obs=pub_obs_t,
            range_idx_crazy_embedded=_crazy_embed(plyrs_range_idxs=plyrs_range_idxs),
            legal_action_mask=legal_action_mask,
            r=rew_for_all[0],
            a=a,
            done=done,
            pub_obs_tp1=pub_obs_tp1,
            strat_tp1=strat_tp1,
            legal_action_mask_tp1=legal_action_mask_tp1,
        )

        # if trav_depth == 0 and traverser == 0:
        #     self._reg_buf[traverser_range_idx].append(aprx_imm_reg.clone().cpu().numpy())

        return (utility * strat_i).sum(), strat_i

    def _any_non_traverser_act(
        self,
        start_state_dict,
        traverser,
        plyrs_range_idxs,
        trav_depth,
        iteration_strats,
        sample_reach,
        cfr_iter,
    ):
        self.total_node_count_traversed += 1
        self._env_wrapper.load_state_dict(start_state_dict)
        p_id_acting = self._env_wrapper.env.current_player.seat_id

        current_pub_obs = self._env_wrapper.get_current_obs()
        range_idx = plyrs_range_idxs[p_id_acting]
        legal_actions_list = self._env_wrapper.env.get_legal_actions()
        legal_action_mask = rl_util.get_legal_action_mask_torch(
            n_actions=self._env_bldr.N_ACTIONS,
            legal_actions_list=legal_actions_list,
            device=self._adv_buffers[traverser].device,
            dtype=torch.float32,
        )
        # """""""""""""""""""""""""
        # The players strategy
        # """""""""""""""""""""""""
        strat_opp = iteration_strats[p_id_acting].get_a_probs(
            pub_obses=[current_pub_obs],
            range_idxs=[range_idx],
            legal_actions_lists=[legal_actions_list],
            to_np=False,
        )[0]

        # """""""""""""""""""""""""
        # Execute action from strat
        # """""""""""""""""""""""""
        a = torch.multinomial(strat_opp, num_samples=1).item()
        pub_obs_tp1, rew_for_all, done, _info = self._env_wrapper.step(a)
        legal_action_mask_tp1 = rl_util.get_legal_action_mask_torch(
            n_actions=self._env_bldr.N_ACTIONS,
            legal_actions_list=self._env_wrapper.env.get_legal_actions(),
            device=self._adv_buffers[traverser].device,
            dtype=torch.float32,
        )

        # """""""""""""""""""""""""
        # Adds to opponent's
        # average buffer if
        # applicable
        # """""""""""""""""""""""""
        if self._avrg_buffers is not None:
            self._avrg_buffers[p_id_acting].add(
                pub_obs=current_pub_obs,
                range_idx=range_idx,
                legal_actions_list=legal_actions_list,
                a_probs=strat_opp.to(self._avrg_buffers[p_id_acting].device).squeeze(),
                iteration=(cfr_iter + 1) / sample_reach,
            )

        # """""""""""""""""""""""""
        # Recursion
        # """""""""""""""""""""""""
        if done:
            strat_tp1 = torch.zeros_like(strat_opp)
            self.total_node_count_traversed += 1
        else:
            u_bootstrap, strat_tp1 = self._recursive_traversal(
                start_state_dict=self._env_wrapper.state_dict(),
                traverser=traverser,
                trav_depth=trav_depth + 1,
                plyrs_range_idxs=plyrs_range_idxs,
                iteration_strats=iteration_strats,
                cfr_iter=cfr_iter,
                sample_reach=sample_reach,
            )

        # """""""""""""""""""""""""
        # Utility
        # """""""""""""""""""""""""
        utility = self._get_utility(
            traverser=traverser,
            u_bootstrap=rew_for_all[traverser] if done else u_bootstrap,
            pub_obs=current_pub_obs,
            range_idx_crazy_embedded=_crazy_embed(plyrs_range_idxs=plyrs_range_idxs),
            legal_actions_list=legal_actions_list,
            legal_action_mask=legal_action_mask,
            a=a,
            sample_strat=strat_opp,
        )

        # add datapoint to baseline net
        self._baseline_buf.add(
            pub_obs=current_pub_obs,
            range_idx_crazy_embedded=_crazy_embed(plyrs_range_idxs=plyrs_range_idxs),
            legal_action_mask=legal_action_mask,
            r=rew_for_all[0],  # 0 bc we mirror for 1... zero-sum
            a=a,
            done=done,
            pub_obs_tp1=pub_obs_tp1,
            strat_tp1=strat_tp1,
            legal_action_mask_tp1=legal_action_mask_tp1,
        )

        return (utility * strat_opp).sum(), strat_opp

    def _get_utility(
        self,
        traverser,
        pub_obs,
        range_idx_crazy_embedded,
        legal_actions_list,
        legal_action_mask,
        u_bootstrap,
        a,
        sample_strat,
    ):
        ######################
        # Remove variance from
        # action
        ######################
        baselines = self._baseline_net.get_b(
            pub_obses=[pub_obs],
            range_idxs=[range_idx_crazy_embedded],
            legal_actions_lists=[legal_actions_list],
            to_np=False,
        )[0] * (1 if traverser == 0 else -1)

        # print(baselines[a], u_bootstrap, a)
        utility = baselines * legal_action_mask
        utility[a] += (u_bootstrap - utility[a]) / sample_strat[a]

        return utility


# See MPM_Baseline
def _crazy_embed(plyrs_range_idxs):
    return plyrs_range_idxs[0] * 10000 + plyrs_range_idxs[1]


# ============================================================
# IterationStrategy
# ============================================================

# Copyright (c) 2019 Eric Steinberger


class IterationStrategy:
    def __init__(self, t_prof, owner, env_bldr, device, cfr_iter):
        self._t_prof = t_prof
        self._owner = owner
        self._env_bldr = env_bldr
        self._device = device
        self._iteration = cfr_iter

        self._adv_net = None
        self._all_range_idxs = torch.arange(
            self._env_bldr.rules.RANGE_SIZE, device=self._device, dtype=torch.long
        )

    @property
    def owner(self):
        return self._owner

    @property
    def iteration(self):
        return self._iteration

    @property
    def device(self):
        return self._device

    def reset(self):
        self._adv_net = None

    def get_action(self, pub_obses, range_idxs, legal_actions_lists):
        a_probs = self.get_a_probs(
            pub_obses=pub_obses,
            range_idxs=range_idxs,
            legal_actions_lists=legal_actions_lists,
        )

        return torch.multinomial(torch.from_numpy(a_probs), num_samples=1).cpu().numpy()

    def get_a_probs2(self, pub_obses, range_idxs, legal_action_masks, to_np=True):
        """
        Args:
            pub_obses (list):               batch (list) of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (list):              batch (list) of range_idxs (one for each pub_obs) [2, 421, 58, 912, ...]
            legal_action_masks (Torch.tensor)
        """

        with torch.no_grad():
            bs = len(range_idxs)
            if self._iteration == 0:  # at iteration 0
                uniform_even_legal = legal_action_masks / (
                    legal_action_masks.sum(-1)
                    .unsqueeze(-1)
                    .expand_as(legal_action_masks)
                )
                if to_np:
                    return uniform_even_legal.cpu().numpy()
                return uniform_even_legal
            else:
                range_idxs = torch.tensor(
                    range_idxs, dtype=torch.long, device=self._device
                )

                advantages = self._adv_net(
                    pub_obses=pub_obses,
                    range_idxs=range_idxs,
                    legal_action_masks=legal_action_masks,
                )

                # """"""""""""""""""""
                relu_advantages = F.relu(
                    advantages, inplace=False
                )  # Cause the sum of *positive* regret matters in CFR
                sum_pos_adv_expanded = (
                    relu_advantages.sum(1).unsqueeze(-1).expand_as(relu_advantages)
                )

                # """"""""""""""""""""
                # In case all negative
                # """"""""""""""""""""
                best_legal_deterministic = torch.zeros(
                    (
                        bs,
                        self._env_bldr.N_ACTIONS,
                    ),
                    dtype=torch.float32,
                    device=self._device,
                )
                bests = torch.argmax(
                    torch.where(
                        legal_action_masks.byte(),
                        advantages,
                        torch.full_like(advantages, fill_value=-10e20),
                    ),
                    dim=1,
                )
                _batch_arranged = torch.arange(
                    bs, device=self._device, dtype=torch.long
                )
                best_legal_deterministic[_batch_arranged, bests] = 1

                # """"""""""""""""""""
                # Strat
                # """"""""""""""""""""
                strategy = torch.where(
                    sum_pos_adv_expanded > 0,
                    relu_advantages / sum_pos_adv_expanded,
                    best_legal_deterministic,
                )

                if to_np:
                    strategy = strategy.cpu().numpy()
                return strategy

    def get_a_probs(self, pub_obses, range_idxs, legal_actions_lists, to_np=True):
        """
        Args:
            pub_obses (list):               batch (list) of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (list):              batch (list) of range_idxs (one for each pub_obs) [2, 421, 58, 912, ...]
            legal_actions_lists (list):     batch (list) of lists of integers that represent legal actions
        """

        with torch.no_grad():
            masks = rl_util.batch_get_legal_action_mask_torch(
                n_actions=self._env_bldr.N_ACTIONS,
                legal_actions_lists=legal_actions_lists,
                device=self._device,
                dtype=torch.float32,
            )
            return self.get_a_probs2(
                pub_obses=pub_obses,
                range_idxs=range_idxs,
                legal_action_masks=masks,
                to_np=to_np,
            )

    def get_a_probs_for_each_hand(self, pub_obs, legal_actions_list):
        """
        Args:
            pub_obs (np.array(shape=(seq_len, n_features,)))
            legal_actions_list (list):      list of ints representing legal actions
        """

        if self._t_prof.DEBUGGING:
            assert isinstance(pub_obs, np.ndarray)
            assert len(pub_obs.shape) == 2, "all hands have the same public obs"
            assert isinstance(legal_actions_list[0], int), (
                "all hands do the same actions. no need to batch, just parse int"
            )

        return self._get_a_probs_of_hands(
            pub_obs=pub_obs,
            legal_actions_list=legal_actions_list,
            range_idxs_tensor=self._all_range_idxs,
        )

    def get_a_probs_for_each_hand_in_list(
        self, pub_obs, range_idxs, legal_actions_list
    ):
        """
        Args:
            pub_obs (np.array(shape=(seq_len, n_features,)))
            range_idxs (np.ndarray):        list of range_idxs to evaluate in public state ""pub_obs""
            legal_actions_list (list):      list of ints representing legal actions
        """

        if self._t_prof.DEBUGGING:
            assert isinstance(pub_obs, np.ndarray)
            assert isinstance(range_idxs, np.ndarray)
            assert len(pub_obs.shape) == 2, "all hands have the same public obs"
            assert isinstance(legal_actions_list[0], int), (
                "all hands can do the same actions. no need to batch"
            )

        return self._get_a_probs_of_hands(
            pub_obs=pub_obs,
            legal_actions_list=legal_actions_list,
            range_idxs_tensor=torch.from_numpy(range_idxs).to(
                dtype=torch.long, device=self._device
            ),
        )

    def _get_a_probs_of_hands(self, pub_obs, range_idxs_tensor, legal_actions_list):
        with torch.no_grad():
            n_hands = range_idxs_tensor.size(0)

            if self._adv_net is None:  # at iteration 0
                uniform_even_legal = torch.zeros(
                    (self._env_bldr.N_ACTIONS,),
                    dtype=torch.float32,
                    device=self._device,
                )
                uniform_even_legal[legal_actions_list] = 1.0 / len(
                    legal_actions_list
                )  # always >0
                uniform_even_legal = uniform_even_legal.unsqueeze(0).expand(
                    n_hands, self._env_bldr.N_ACTIONS
                )
                return uniform_even_legal.cpu().numpy()

            else:
                legal_action_masks = rl_util.get_legal_action_mask_torch(
                    n_actions=self._env_bldr.N_ACTIONS,
                    legal_actions_list=legal_actions_list,
                    device=self._device,
                    dtype=torch.float32,
                )
                legal_action_masks = legal_action_masks.unsqueeze(0).expand(n_hands, -1)

                advantages = self._adv_net(
                    pub_obses=[pub_obs] * n_hands,
                    range_idxs=range_idxs_tensor,
                    legal_action_masks=legal_action_masks,
                )

                # """"""""""""""""""""
                relu_advantages = F.relu(
                    advantages, inplace=False
                )  # Cause the sum of *positive* regret matters in CFR
                sum_pos_adv_expanded = (
                    relu_advantages.sum(1).unsqueeze(-1).expand_as(relu_advantages)
                )

                # """"""""""""""""""""
                # In case all negative
                # """"""""""""""""""""
                best_legal_deterministic = torch.zeros(
                    (
                        n_hands,
                        self._env_bldr.N_ACTIONS,
                    ),
                    dtype=torch.float32,
                    device=self._device,
                )
                bests = torch.argmax(
                    torch.where(
                        legal_action_masks.byte(),
                        advantages,
                        torch.full_like(advantages, fill_value=-10e20),
                    ),
                    dim=1,
                )

                _batch_arranged = torch.arange(
                    n_hands, device=self._device, dtype=torch.long
                )
                best_legal_deterministic[_batch_arranged, bests] = 1

                # """"""""""""""""""""
                # Strategy
                # """"""""""""""""""""
                strategy = torch.where(
                    sum_pos_adv_expanded > 0,
                    relu_advantages / sum_pos_adv_expanded,
                    best_legal_deterministic,
                )

                return strategy.cpu().numpy()

    def state_dict(self):
        return {
            "owner": self._owner,
            "net": self.net_state_dict(),
            "iter": self._iteration,
        }

    @staticmethod
    def build_from_state_dict(t_prof, env_bldr, device, state):
        s = IterationStrategy(
            t_prof=t_prof,
            env_bldr=env_bldr,
            device=device,
            owner=state["owner"],
            cfr_iter=state["iter"],
        )
        s.load_state_dict(state=state)  # loads net state
        return s

    def load_state_dict(self, state):
        assert self._owner == state["owner"]
        self.load_net_state_dict(state["net"])
        self._iteration = state["iter"]

    def net_state_dict(self):
        """This just wraps the net.state_dict() with the option of returning None if net is None"""
        if self._adv_net is None:
            return None
        return self._adv_net.state_dict()

    def load_net_state_dict(self, state_dict):
        if state_dict is None:
            return  # if this happens (should only for iteration 0), this class will return random actions.
        else:
            self._adv_net = DuelingQNet(
                q_args=self._t_prof.module_args["adv_training"].adv_net_args,
                env_bldr=self._env_bldr,
                device=self._device,
            )
            self._adv_net.load_state_dict(state_dict)
            self._adv_net.to(self._device)

        self._adv_net.eval()
        for param in self._adv_net.parameters():
            param.requires_grad = False

    def get_copy(self, device=None):
        _device = self._device if device is None else device
        return IterationStrategy.build_from_state_dict(
            t_prof=self._t_prof,
            env_bldr=self._env_bldr,
            device=_device,
            state=self.state_dict(),
        )


# ============================================================
# StrategyBuffer
# ============================================================

# Copyright (c) 2019 Eric Steinberger


class StrategyBuffer:
    def __init__(self, t_prof, owner, env_bldr, device, max_size=None):
        self._t_prof = t_prof
        self._env_bldr = env_bldr
        self._owner = owner
        self._device = device

        self._max_size = max_size

        self._strategies = None
        self._weights = None
        self._size = None
        self._last_iteration_seen = None

        self.reset()

    @property
    def owner(self):
        return self._owner

    @property
    def size(self):
        return self._size

    @property
    def device(self):
        return self._device

    @property
    def strategies(self):
        return self._strategies

    @property
    def last_iteration_seen(self):
        return self._last_iteration_seen

    @property
    def max_size(self):
        return self._max_size

    def get(self, i):
        return self._strategies[i]

    def get_strats_and_weights(self):
        return zip(self._strategies, self._weights)

    def sample_strat_weighted(self):
        return self.get(self.sample_strat_idx_weighted())

    def sample_strat_idx_weighted(self):
        if self._size == 0:
            return None

        w = np.array(self._weights)
        s = np.sum(w)
        w = np.full_like(w, fill_value=1 / w.shape[0]) if s == 0 else w / s

        return np.random.choice(
            a=np.arange(start=0, stop=self._size, dtype=np.int32), p=w
        )

    def add(self, iteration_strat):
        if self._max_size is None or (self._size < self._max_size):
            self._strategies.append(iteration_strat.get_copy(device=self._device))
            self._weights.append(iteration_strat.iteration + 1)

            self._size = len(self._strategies)

        elif np.random.random() < (
            float(self._max_size) / float(self._last_iteration_seen)
        ):
            idx = np.random.randint(len(self._strategies))
            self._strategies[idx] = iteration_strat.get_copy(device=self._device)
            self._weights[idx] = iteration_strat.iteration + 1

        self._last_iteration_seen = iteration_strat.iteration

    def state_dict(self):
        return {
            "nets": [(s.net_state_dict(), s.iteration) for s in self._strategies],
            "owner": self.owner,
        }

    def load_state_dict(self, state):
        assert self.owner == state["owner"]

        self._strategies = []
        for net_state_dict, cfr_iter in state["nets"]:
            s = IterationStrategy(
                t_prof=self._t_prof,
                owner=self.owner,
                env_bldr=self._env_bldr,
                device=self._device,
                cfr_iter=cfr_iter,
            )
            s.load_net_state_dict(net_state_dict)
            self._strategies.append(s)
            self._weights.append(cfr_iter)

        self._size = len(self._strategies)

    def reset(self):
        self._strategies = []
        self._weights = []
        self._size = 0
        self._last_iteration_seen = None


# ============================================================
# EvalAgentDeepCFR
# ============================================================

# Copyright (c) 2019 Eric Steinberger


NP_FLOAT_TYPE = np.float64  # Use 64 for extra stability in big games


class EvalAgentDeepCFR(_EvalAgentBase):
    EVAL_MODE_AVRG_NET = "AVRG_NET"
    EVAL_MODE_SINGLE = "SINGLE"
    ALL_MODES = [EVAL_MODE_AVRG_NET, EVAL_MODE_SINGLE]

    def __init__(self, t_prof, mode=None, device=None):
        super().__init__(t_prof=t_prof, mode=mode, device=device)
        self.avrg_args = t_prof.module_args["avrg_training"]

        self._AVRG = (
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self.t_prof.eval_modes_of_algo
        )
        self._SINGLE = (
            EvalAgentDeepCFR.EVAL_MODE_SINGLE in self.t_prof.eval_modes_of_algo
        )

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            self.avrg_net_policies = [
                AvrgWrapper(
                    avrg_training_args=self.avrg_args,
                    owner=p,
                    env_bldr=self.env_bldr,
                    device=self.device,
                )
                for p in range(t_prof.n_seats)
            ]
            for pol in self.avrg_net_policies:
                pol.eval()

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            self._strategy_buffers = [
                StrategyBuffer(
                    t_prof=t_prof,
                    owner=p,
                    env_bldr=self.env_bldr,
                    max_size=self.t_prof.eval_agent_max_strat_buf_size,
                    device=self.device,
                )
                for p in range(t_prof.n_seats)
            ]

            # Iteration whose strategies is used for current episode for each seat.
            # Only applicable if trajectory-sampling SD-CFR is used.
            self._episode_net_idxs = [None for p in range(self.env_bldr.N_SEATS)]

            # Track history
            self._a_history = None

    def can_compute_mode(self):
        """All modes are always computable (i.e. not dependent on iteration etc.)"""
        return True

    # ___________________________ Overrides to track history for reach computation in SD-CFR ___________________________
    def notify_of_reset(self):
        if self._mode == self.EVAL_MODE_SINGLE:
            self._reset_action_history()
            self._sample_new_strategy()
        super().notify_of_reset()

    def reset(self, deck_state_dict=None):
        if self._mode == self.EVAL_MODE_SINGLE:
            self._reset_action_history()
            self._sample_new_strategy()
        super().reset(deck_state_dict=deck_state_dict)

    def set_to_public_tree_node_state(self, node):
        if self._mode == self.EVAL_MODE_SINGLE:
            # """""""""""""""""""""""""""""
            # Set history to correct state
            # """""""""""""""""""""""""""""
            self._reset_action_history()
            relevant_nodes_in_forward_order = []
            _node = node
            while _node is not None:
                if (
                    isinstance(_node, PlayerActionNode)
                    and _node.p_id_acted_last == node.p_id_acting_next
                ):
                    relevant_nodes_in_forward_order.insert(0, _node)
                _node = _node.parent

            for _node in relevant_nodes_in_forward_order:
                super().set_to_public_tree_node_state(node=_node.parent)
                self._add_history_entry(
                    p_id_acting=_node.p_id_acted_last, action_hes_gonna_do=_node.action
                )

        # """""""""""""""""""""""""""""
        # Set env wrapper to correct
        # state.
        # """""""""""""""""""""""""""""
        super().set_to_public_tree_node_state(node=node)

    def get_a_probs_for_each_hand(self):
        """BEFORE CALLING, NOTIFY EVALAGENT OF THE PAST ACTIONS / ACTIONSEQUENCE!!!!!"""
        pub_obs = self._internal_env_wrapper.get_current_obs()
        legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._mode == self.EVAL_MODE_AVRG_NET:
            return self.avrg_net_policies[p_id_acting].get_a_probs_for_each_hand(
                pub_obs=pub_obs, legal_actions_list=legal_actions_list
            )

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        elif self._mode == self.EVAL_MODE_SINGLE:
            unif_rand_legal = np.full(
                shape=self.env_bldr.N_ACTIONS, fill_value=1.0 / len(legal_actions_list)
            ) * rl_util.get_legal_action_mask_np(
                n_actions=self.env_bldr.N_ACTIONS,
                legal_actions_list=legal_actions_list,
                dtype=np.float32,
            )

            n_models = self._strategy_buffers[p_id_acting].size
            if n_models == 0:
                return np.repeat(
                    np.expand_dims(unif_rand_legal, axis=0),
                    repeats=self.env_bldr.rules.RANGE_SIZE,
                    axis=0,
                )
            else:
                # Dim: [model_idx, range_idx]
                reaches = self._get_reach_for_each_model_each_hand(
                    p_id_acting=p_id_acting
                )

                # """"""""""""""""""""""
                # Compute strategy for
                # all infosets with
                # reach >0. Initialize
                # All others stay unif.
                # """"""""""""""""""""""
                contrib_each_model = np.zeros(
                    shape=(
                        n_models,
                        self.env_bldr.rules.RANGE_SIZE,
                        self.env_bldr.N_ACTIONS,
                    ),
                    dtype=NP_FLOAT_TYPE,
                )

                for m_i, (strat, weight) in enumerate(
                    self._strategy_buffers[p_id_acting].get_strats_and_weights()
                ):
                    range_idxs = np.nonzero(reaches[m_i])[0]
                    if range_idxs.shape[0] > 0:
                        a_probs_m = strat.get_a_probs_for_each_hand_in_list(
                            pub_obs=pub_obs,
                            range_idxs=range_idxs,
                            legal_actions_list=legal_actions_list,
                        )
                        contrib_each_model[m_i, range_idxs] = a_probs_m * weight

                # Dim: [range_idx, action_p]
                a_probs = (
                    np.sum(contrib_each_model * np.expand_dims(reaches, axis=2), axis=0)
                ).astype(NP_FLOAT_TYPE)

                # Dim: [range_idx]
                a_probs_sum = np.expand_dims(np.sum(a_probs, axis=1), axis=1)

                # Dim: [range_idx, action_p]
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.where(
                        a_probs_sum == 0,
                        np.repeat(
                            np.expand_dims(unif_rand_legal, axis=0),
                            repeats=self._internal_env_wrapper.env.RANGE_SIZE,
                            axis=0,
                        ),
                        a_probs / a_probs_sum,
                    )

        else:
            raise UnknownModeError(self._mode)

    def get_a_probs(self):
        pub_obs = self._internal_env_wrapper.get_current_obs()
        legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        range_idx = self._internal_env_wrapper.env.get_range_idx(p_id=p_id_acting)

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._mode == self.EVAL_MODE_AVRG_NET:
            return self.avrg_net_policies[p_id_acting].get_a_probs(
                pub_obses=[pub_obs],
                range_idxs=np.array([range_idx], dtype=np.int32),
                legal_actions_lists=[legal_actions_list],
            )[0]

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        elif self._mode == self.EVAL_MODE_SINGLE:
            if self._strategy_buffers[p_id_acting].size == 0:
                unif_rand_legal = np.full(
                    shape=self.env_bldr.N_ACTIONS,
                    fill_value=1.0 / len(legal_actions_list),
                ) * rl_util.get_legal_action_mask_np(
                    n_actions=self.env_bldr.N_ACTIONS,
                    legal_actions_list=legal_actions_list,
                    dtype=np.float32,
                )
                return unif_rand_legal
            else:
                # """""""""""""""""""""
                # Weighted by Iteration
                # """"""""""""""""""""""
                # Dim: [model_idx, action_p]
                a_probs_each_model = np.array(
                    [
                        weight
                        * strat.get_a_probs(
                            pub_obses=[pub_obs],
                            range_idxs=[range_idx],
                            legal_actions_lists=[legal_actions_list],
                        )[0]
                        for strat, weight in self._strategy_buffers[
                            p_id_acting
                        ].get_strats_and_weights()
                    ]
                )

                # """"""""""""""""""""""
                # Weighted by Reach
                # """"""""""""""""""""""
                a_probs_each_model *= np.expand_dims(
                    self._get_reach_for_each_model(
                        p_id_acting=p_id_acting,
                        range_idx=range_idx,
                    ),
                    axis=2,
                )

                # """"""""""""""""""""""
                # Normalize
                # """"""""""""""""""""""
                # Dim: [action_p]
                a_probs = np.sum(a_probs_each_model, axis=0)

                # Dim: []
                a_probs_sum = np.sum(a_probs)

                # Dim: [action_p]
                return a_probs / a_probs_sum

        else:
            raise UnknownModeError(self._mode)

    def get_action(self, step_env=True, need_probs=False):
        """!! BEFORE CALLING, NOTIFY EVALAGENT OF THE PAST ACTIONS / ACTIONSEQUENCE !!"""

        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        range_idx = self._internal_env_wrapper.env.get_range_idx(p_id=p_id_acting)

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._mode == self.EVAL_MODE_AVRG_NET:
            if need_probs:  # only do if necessary
                a_probs_all_hands = self.get_a_probs_for_each_hand()
                a_probs = a_probs_all_hands[range_idx]
            else:
                a_probs_all_hands = None  # not needed

                a_probs = self.avrg_net_policies[p_id_acting].get_a_probs(
                    pub_obses=[self._internal_env_wrapper.get_current_obs()],
                    range_idxs=np.array([range_idx], dtype=np.int32),
                    legal_actions_lists=[
                        self._internal_env_wrapper.env.get_legal_actions()
                    ],
                )[0]

            action = np.random.choice(np.arange(self.env_bldr.N_ACTIONS), p=a_probs)

            if step_env:
                self._internal_env_wrapper.step(action=action)

            return action, a_probs_all_hands

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        elif self._mode == self.EVAL_MODE_SINGLE:
            if need_probs:
                a_probs_all_hands = self.get_a_probs_for_each_hand()
            else:
                a_probs_all_hands = None  # not needed

            legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()

            if self._episode_net_idxs[p_id_acting] is None:  # Iteration 0
                action = legal_actions_list[np.random.randint(len(legal_actions_list))]
            else:  # Iteration > 0
                action = (
                    self._strategy_buffers[p_id_acting]
                    .get(self._episode_net_idxs[p_id_acting])
                    .get_action(
                        pub_obses=[self._internal_env_wrapper.get_current_obs()],
                        range_idxs=[range_idx],
                        legal_actions_lists=[legal_actions_list],
                    )[0]
                    .item()
                )

            if step_env:
                # add to history before modifying env state
                self._add_history_entry(
                    p_id_acting=p_id_acting, action_hes_gonna_do=action
                )

                # make INTERNAL step to keep up with the game state.
                self._internal_env_wrapper.step(action=action)

            return action, a_probs_all_hands
        else:
            raise UnknownModeError(self._mode)

    def get_action_frac_tuple(self, step_env):
        a_idx_raw = self.get_action(step_env=step_env, need_probs=False)[0]

        if self.env_bldr.env_cls.IS_FIXED_LIMIT_GAME:
            return a_idx_raw, -1
        else:
            if a_idx_raw >= 2:
                frac = self.env_bldr.env_args.bet_sizes_list_as_frac_of_pot[
                    a_idx_raw - 2
                ]
                return [Poker.BET_RAISE, frac]
            return [a_idx_raw, -1]

    def update_weights(self, weights_for_eval_agent):
        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            avrg_weights = weights_for_eval_agent[self.EVAL_MODE_AVRG_NET]

            for p in range(self.t_prof.n_seats):
                self.avrg_net_policies[p].load_net_state_dict(
                    self.ray.state_dict_to_torch(avrg_weights[p], device=self.device)
                )
                self.avrg_net_policies[p].eval()

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            list_of_new_iter_strat_state_dicts = copy.deepcopy(
                weights_for_eval_agent[self.EVAL_MODE_SINGLE]
            )

            for p in range(self.t_prof.n_seats):
                for state in list_of_new_iter_strat_state_dicts[p]:
                    state["net"] = self.ray.state_dict_to_torch(
                        state["net"], device=self.device
                    )

                    _iter_strat = IterationStrategy.build_from_state_dict(
                        state=state,
                        t_prof=self.t_prof,
                        env_bldr=self.env_bldr,
                        device=self.device,
                    )

                    self._strategy_buffers[p].add(iteration_strat=_iter_strat)

    def _state_dict(self):
        d = {}

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            d["avrg_nets"] = [pol.net_state_dict() for pol in self.avrg_net_policies]

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            d["strategy_buffers"] = [
                self._strategy_buffers[p].state_dict()
                for p in range(self.t_prof.n_seats)
            ]
            d["curr_net_idxs"] = copy.deepcopy(self._episode_net_idxs)
            d["history"] = copy.deepcopy(self._a_history)

        return d

    def _load_state_dict(self, state):
        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            for i in range(self.t_prof.n_seats):
                self.avrg_net_policies[i].load_net_state_dict(state["avrg_nets"][i])

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            for p in range(self.t_prof.n_seats):
                self._strategy_buffers[p].load_state_dict(
                    state=state["strategy_buffers"][p]
                )
            self._a_history = copy.deepcopy(state["history"])
            self._episode_net_idxs = copy.deepcopy(state["curr_net_idxs"])

    # _____________________________________________ SD-CFR specific _____________________________________________
    def _add_history_entry(self, p_id_acting, action_hes_gonna_do):
        self._a_history[p_id_acting]["pub_obs_batch"].append(
            self._internal_env_wrapper.get_current_obs()
        )
        self._a_history[p_id_acting]["legal_action_list_batch"].append(
            self._internal_env_wrapper.env.get_legal_actions()
        )
        self._a_history[p_id_acting]["a_batch"].append(action_hes_gonna_do)
        self._a_history[p_id_acting]["len"] += 1

    def _get_reach_for_each_model(self, p_id_acting, range_idx):
        models = self._strategy_buffers[p_id_acting].strategies

        H = self._a_history[p_id_acting]
        if H["len"] == 0:
            # Dim: [model_idx]
            return np.ones(shape=(len(models)), dtype=np.float32)

        # """"""""""""""""""""""
        # Batch calls history
        # and computes product
        # of result
        # """"""""""""""""""""""
        # Dim: [model_idx, history_time_step]
        prob_a_each_model_each_timestep = np.array(
            [
                model.get_a_probs(
                    pub_obses=H["pub_obs_batch"],
                    range_idxs=[range_idx] * H["len"],
                    legal_actions_lists=H["legal_action_list_batch"],
                )[np.arange(len(models)), H["a_batch"]]
                for model in models
            ]
        )
        # Dim: [model_idx]
        return np.prod(a=prob_a_each_model_each_timestep, axis=1)

    def _get_reach_for_each_model_each_hand(self, p_id_acting):
        # Probability that each model would perform action a (from history) with each hand
        models = self._strategy_buffers[p_id_acting].strategies

        # Dim: [model_idx, range_idx]
        reaches = np.empty(
            shape=(
                len(models),
                self.env_bldr.rules.RANGE_SIZE,
            ),
            dtype=NP_FLOAT_TYPE,
        )

        H = self._a_history[p_id_acting]

        for m_i, model in enumerate(models):
            non_zero_hands = list(range(self.env_bldr.rules.RANGE_SIZE))

            # """"""""""""""""""""""
            # Batch calls hands but
            # not history timesteps.
            # """"""""""""""""""""""
            reach_hist = np.zeros(
                shape=(H["len"], self.env_bldr.rules.RANGE_SIZE), dtype=NP_FLOAT_TYPE
            )
            for hist_idx in range(H["len"]):
                if len(non_zero_hands) == 0:
                    break

                # Dim: [model_idx, RANGE_SIZE]
                p_m_a = model.get_a_probs_for_each_hand_in_list(
                    pub_obs=H["pub_obs_batch"][hist_idx],
                    legal_actions_list=H["legal_action_list_batch"][hist_idx],
                    range_idxs=np.array(non_zero_hands),
                )[:, H["a_batch"][hist_idx]]

                reach_hist[hist_idx, non_zero_hands] = p_m_a * len(
                    H["legal_action_list_batch"][hist_idx]
                )
                # collect zeros to avoid unnecessary future queries
                for h_idx in reversed(range(len(non_zero_hands))):
                    if p_m_a[h_idx] == 0:
                        del non_zero_hands[h_idx]

            reaches[m_i] = np.prod(reach_hist, axis=0)

        # Dim: [model_idx, RANGE_SIZE]
        return reaches

    def _sample_new_strategy(self):
        """
        Sample one current strategy from the buffer to play by this episode
        """
        self._episode_net_idxs = [
            self._strategy_buffers[p].sample_strat_idx_weighted()
            for p in range(self.env_bldr.N_SEATS)
        ]

    def _reset_action_history(self):
        self._a_history = {
            p_id: {
                "pub_obs_batch": [],
                "legal_action_list_batch": [],
                "a_batch": [],
                "len": 0,
            }
            for p_id in range(self.env_bldr.N_SEATS)
        }


# ============================================================
# TrainingProfile
# ============================================================

# Copyright (c) 2019 Eric Steinberger


class TrainingProfile(TrainingProfileBase):
    def __init__(
        self,
        # ------ General
        name="",
        log_verbose=False,
        log_memory=False,
        log_export_freq=1,
        checkpoint_freq=99999999,
        eval_agent_export_freq=999999999,
        n_learner_actor_workers=8,
        max_n_las_sync_simultaneously=10,
        nn_type="feedforward",  # "recurrent" or "feedforward"
        # ------ Computing
        path_data=None,
        local_crayon_server_docker_address="localhost",
        device_inference="cpu",
        device_training="cpu",
        device_parameter_server="cpu",
        DISTRIBUTED=False,
        CLUSTER=False,
        DEBUGGING=False,
        # ------ Env
        game_cls=DiscretizedNLLeduc,
        n_seats=2,
        agent_bet_set=bet_sets.B_2,
        start_chips=None,
        chip_randomness=(0, 0),
        uniform_action_interpolation=False,
        use_simplified_headsup_obs=True,
        # ------ Evaluation
        eval_modes_of_algo=(EvalAgentDeepCFR.EVAL_MODE_SINGLE,),
        eval_stack_sizes=None,
        # ------ General Deep CFR params
        n_traversals_per_iter=30000,
        iter_weighting_exponent=1.0,
        n_actions_traverser_samples=3,
        sampler="mo",
        turn_off_baseline=False,  # Only for VR-OS
        os_eps=1,
        periodic_restart=1,
        # --- Baseline Hyperparameters
        max_buffer_size_baseline=2e5,
        batch_size_baseline=512,
        n_batches_per_iter_baseline=300,
        dim_baseline=64,
        deep_baseline=True,
        normalize_last_layer_FLAT_baseline=True,
        # --- Adv Hyperparameters
        n_batches_adv_training=5000,
        init_adv_model="random",
        mini_batch_size_adv=2048,
        dim_adv=64,
        deep_adv=True,
        optimizer_adv="adam",
        loss_adv="weighted_mse",
        lr_adv=0.001,
        grad_norm_clipping_adv=1.0,
        lr_patience_adv=999999999,
        normalize_last_layer_FLAT_adv=True,
        max_buffer_size_adv=2e6,
        # ------ SPECIFIC TO AVRG NET
        n_batches_avrg_training=15000,
        init_avrg_model="random",
        dim_avrg=64,
        deep_avrg=True,
        mini_batch_size_avrg=2048,
        loss_avrg="weighted_mse",
        optimizer_avrg="adam",
        lr_avrg=0.001,
        grad_norm_clipping_avrg=1.0,
        lr_patience_avrg=999999999,
        normalize_last_layer_FLAT_avrg=True,
        max_buffer_size_avrg=2e6,
        # ------ SPECIFIC TO SINGLE
        export_each_net=False,
        eval_agent_max_strat_buf_size=None,
        # ------ Optional
        lbr_args=None,
        rlbr_args=None,
        h2h_args=None,
    ):
        if nn_type == "feedforward":
            env_bldr_cls = FlatLimitPokerEnvBuilder

            mpm_args_adv = MPMArgsFLAT(
                deep=deep_adv, dim=dim_adv, normalize=normalize_last_layer_FLAT_adv
            )
            mpm_args_baseline = MPMArgsFLAT_Baseline(
                deep=deep_baseline,
                dim=dim_baseline,
                normalize=normalize_last_layer_FLAT_baseline,
            )
            mpm_args_avrg = MPMArgsFLAT(
                deep=deep_avrg, dim=dim_avrg, normalize=normalize_last_layer_FLAT_avrg
            )

        else:
            raise ValueError(nn_type)

        super().__init__(
            name=name,
            log_verbose=log_verbose,
            log_export_freq=log_export_freq,
            checkpoint_freq=checkpoint_freq,
            eval_agent_export_freq=eval_agent_export_freq,
            path_data=path_data,
            game_cls=game_cls,
            env_bldr_cls=env_bldr_cls,
            start_chips=start_chips,
            eval_modes_of_algo=eval_modes_of_algo,
            eval_stack_sizes=eval_stack_sizes,
            DEBUGGING=DEBUGGING,
            DISTRIBUTED=DISTRIBUTED,
            CLUSTER=CLUSTER,
            device_inference=device_inference,
            local_crayon_server_docker_address=local_crayon_server_docker_address,
            module_args={
                "adv_training": AdvTrainingArgs(
                    adv_net_args=DuelingQArgs(
                        mpm_args=mpm_args_adv, n_units_final=dim_adv
                    ),
                    n_batches_adv_training=n_batches_adv_training,
                    init_adv_model=init_adv_model,
                    batch_size=mini_batch_size_adv,
                    optim_str=optimizer_adv,
                    loss_str=loss_adv,
                    lr=lr_adv,
                    grad_norm_clipping=grad_norm_clipping_adv,
                    device_training=device_training,
                    max_buffer_size=max_buffer_size_adv,
                    lr_patience=lr_patience_adv,
                ),
                "avrg_training": AvrgTrainingArgs(
                    avrg_net_args=AvrgNetArgs(
                        mpm_args=mpm_args_avrg,
                        n_units_final=dim_avrg,
                    ),
                    n_batches_avrg_training=n_batches_avrg_training,
                    init_avrg_model=init_avrg_model,
                    batch_size=mini_batch_size_avrg,
                    loss_str=loss_avrg,
                    optim_str=optimizer_avrg,
                    lr=lr_avrg,
                    grad_norm_clipping=grad_norm_clipping_avrg,
                    device_training=device_training,
                    max_buffer_size=max_buffer_size_avrg,
                    lr_patience=lr_patience_avrg,
                ),
                "env": game_cls.ARGS_CLS(
                    n_seats=n_seats,
                    starting_stack_sizes_list=[start_chips for _ in range(n_seats)],
                    bet_sizes_list_as_frac_of_pot=copy.deepcopy(agent_bet_set),
                    stack_randomization_range=chip_randomness,
                    use_simplified_headsup_obs=use_simplified_headsup_obs,
                    uniform_action_interpolation=uniform_action_interpolation,
                ),
                "mccfr_baseline": BaselineArgs(
                    q_net_args=DuelingQArgs(
                        mpm_args=mpm_args_baseline,
                        n_units_final=dim_baseline,
                    ),
                    max_buffer_size=max_buffer_size_baseline,
                    batch_size=batch_size_baseline,
                    n_batches_per_iter_baseline=n_batches_per_iter_baseline,
                ),
                "lbr": lbr_args,
                "rlbr": rlbr_args,
                "h2h": h2h_args,
            },
            log_memory=log_memory,
        )

        self.nn_type = nn_type
        self.n_traversals_per_iter = int(n_traversals_per_iter)
        self.iter_weighting_exponent = iter_weighting_exponent
        self.sampler = sampler
        self.os_eps = os_eps
        self.periodic_restart = periodic_restart
        self.turn_off_baseline = turn_off_baseline
        self.n_actions_traverser_samples = n_actions_traverser_samples

        # SINGLE
        self.export_each_net = export_each_net
        self.eval_agent_max_strat_buf_size = eval_agent_max_strat_buf_size

        # Different for dist and local
        if DISTRIBUTED or CLUSTER:
            print("Running with ", n_learner_actor_workers, "LearnerActor Workers.")
            self.n_learner_actors = n_learner_actor_workers
        else:
            self.n_learner_actors = 1
        self.max_n_las_sync_simultaneously = max_n_las_sync_simultaneously

        assert isinstance(device_parameter_server, str), (
            "Please pass a string (either 'cpu' or 'cuda')!"
        )
        self.device_parameter_server = torch.device(device_parameter_server)


# ============================================================
# Chief
# ============================================================

# Copyright Eric Steinberger 2020
# Copyright (c) Eric Steinberger 2020


class Chief(_ChiefBase):
    def __init__(self, t_prof):
        super().__init__(t_prof=t_prof)
        self._ps_handles = None
        self._la_handles = None
        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)

        self._SINGLE = (
            EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo
        )
        self._AVRG = (
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        )

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            self._strategy_buffers = [
                StrategyBuffer(
                    t_prof=t_prof,
                    owner=p,
                    env_bldr=self._env_bldr,
                    max_size=None,
                    device=self._t_prof.device_inference,
                )
                for p in range(t_prof.n_seats)
            ]

            if self._t_prof.log_memory:
                self._exp_mem_usage = self.create_experiment(
                    self._t_prof.name + " Chief_Memory_Usage"
                )

            self._last_iter_receiver_has = {}

    def set_la_handles(self, *la_handles):
        self._la_handles = list(la_handles)

    def set_ps_handle(self, *ps_handles):
        self._ps_handles = list(ps_handles)

    def update_alive_las(self, alive_la_handles):
        self._la_handles = alive_la_handles

    # ____________________________________________________ Strategy ____________________________________________________
    def pull_current_eval_strategy(self, receiver_name):
        """
        Args:
            last_iteration_receiver_has (list):     None or int for each player
        """
        d = {}

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            d[EvalAgentDeepCFR.EVAL_MODE_AVRG_NET] = self._pull_avrg_net_eval_strat()

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            d[EvalAgentDeepCFR.EVAL_MODE_SINGLE] = self._pull_single_eval_strat(
                receiver_name=receiver_name
            )

        return d

    def _pull_avrg_net_eval_strat(self):
        return [
            self._ray.get(self._ray.remote(ps.get_avrg_weights))
            for ps in self._ps_handles
        ]

    def _pull_single_eval_strat(self, receiver_name):
        """
        Args:
            last_iteration_receiver_has (list):     None or int for each player
        """
        if receiver_name in self._last_iter_receiver_has:
            last_iteration_receiver_has = self._last_iter_receiver_has[receiver_name]
        else:
            last_iteration_receiver_has = None

        buf_sizes = [
            self._strategy_buffers[_p_id].size for _p_id in range(self._t_prof.n_seats)
        ]
        assert buf_sizes[0] == buf_sizes[1]

        _first_iteration_to_get = (
            0 if last_iteration_receiver_has is None else last_iteration_receiver_has
        )

        def _to_torch(cum_strat_state_dict):
            cum_strat_state_dict["net"] = self._ray.state_dict_to_numpy(
                cum_strat_state_dict["net"]
            )
            return cum_strat_state_dict

        state_dicts = [
            [
                _to_torch(self._strategy_buffers[_p_id].get(i).state_dict())
                for i in range(
                    _first_iteration_to_get, self._strategy_buffers[_p_id].size
                )
            ]
            for _p_id in range(self._t_prof.n_seats)
        ]

        self._last_iter_receiver_has[receiver_name] = buf_sizes[0]

        return state_dicts

    # Only applicable to SINGLE
    def add_new_iteration_strategy_model(self, owner, adv_net_state_dict, cfr_iter):
        iter_strat = IterationStrategy(
            t_prof=self._t_prof,
            env_bldr=self._env_bldr,
            owner=owner,
            device=self._t_prof.device_inference,
            cfr_iter=cfr_iter,
        )

        iter_strat.load_net_state_dict(
            self._ray.state_dict_to_torch(
                adv_net_state_dict, device=self._t_prof.device_inference
            )
        )
        self._strategy_buffers[iter_strat.owner].add(iteration_strat=iter_strat)

        #  Store to disk
        if self._t_prof.export_each_net:
            path = ospj(self._t_prof.path_strategy_nets, self._t_prof.name)
            file_util.create_dir_if_not_exist(path)
            file_util.do_pickle(
                obj=iter_strat.state_dict(),
                path=path,
                file_name=str(iter_strat.iteration)
                + "_P"
                + str(iter_strat.owner)
                + ".pkl",
            )

        if self._t_prof.log_memory:
            if owner == 1:
                # Logs
                process = psutil.Process(os.getpid())
                self.add_scalar(
                    self._exp_mem_usage,
                    "Debug/Memory Usage/Chief",
                    cfr_iter,
                    process.memory_info().rss,
                )

    # ________________________________ Store a pickled API class to play against the AI ________________________________
    def export_agent(self, step):
        _dir = ospj(
            self._t_prof.path_agent_export_storage, str(self._t_prof.name), str(step)
        )
        file_util.create_dir_if_not_exist(_dir)

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            MODE = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET

            t_prof = copy.deepcopy(self._t_prof)
            t_prof.eval_modes_of_algo = [MODE]

            eval_agent = EvalAgentDeepCFR(t_prof=t_prof)
            eval_agent.reset()

            w = {EvalAgentDeepCFR.EVAL_MODE_AVRG_NET: self._pull_avrg_net_eval_strat()}
            eval_agent.update_weights(w)
            eval_agent.set_mode(mode=MODE)
            eval_agent.store_to_disk(path=_dir, file_name="eval_agent" + MODE)

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            MODE = EvalAgentDeepCFR.EVAL_MODE_SINGLE
            t_prof = copy.deepcopy(self._t_prof)
            t_prof.eval_modes_of_algo = [MODE]

            eval_agent = EvalAgentDeepCFR(t_prof=t_prof)
            eval_agent.reset()

            eval_agent._strategy_buffers = (
                self._strategy_buffers
            )  # could copy - it's just for the export, so it's ok
            eval_agent.set_mode(mode=MODE)
            eval_agent.store_to_disk(path=_dir, file_name="eval_agent" + MODE)


# ============================================================
# ParameterServer
# ============================================================

# Copyright (c) Eric Steinberger 2020


class ParameterServer(ParameterServerBase):
    def __init__(self, t_prof, owner, chief_handle):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle)

        self.owner = owner
        self._adv_args = t_prof.module_args["adv_training"]

        self._adv_net = self._get_new_adv_net()
        self._adv_optim, self._adv_lr_scheduler = self._get_new_adv_optim()

        if self._t_prof.log_memory:
            self._exp_mem_usage = self._ray.get(
                self._ray.remote(
                    self._chief_handle.create_experiment,
                    self._t_prof.name + "_PS" + str(owner) + "_Memory_Usage",
                )
            )

        self._AVRG = (
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        )
        self._SINGLE = (
            EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo
        )

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]
            self._avrg_net = self._get_new_avrg_net()
            self._avrg_optim, self._avrg_lr_scheduler = self._get_new_avrg_optim()

        # """"""""""""""""""""""""""""
        # Baseline
        # """"""""""""""""""""""""""""
        self._BASELINE = self._t_prof.sampler == "learned_baseline"
        if self._BASELINE and owner == 0:
            self._baseline_args = t_prof.module_args["mccfr_baseline"]
            self._baseline_net = self._get_new_baseline_net()
            self._baseline_optim = self._get_new_baseline_optim()

    # ______________________________________________ API to pull from PS _______________________________________________

    def get_adv_weights(self):
        self._adv_net.zero_grad()
        return self._ray.state_dict_to_numpy(self._adv_net.state_dict())

    def get_avrg_weights(self):
        self._avrg_net.zero_grad()
        return self._ray.state_dict_to_numpy(self._avrg_net.state_dict())

    def get_baseline_weights(self):
        self._baseline_net.zero_grad()
        return self._ray.state_dict_to_numpy(self._baseline_net.state_dict())

    # ____________________________________________ API to make PS compute ______________________________________________
    def apply_grads_adv(self, list_of_grads):
        self._apply_grads(
            list_of_grads=list_of_grads,
            optimizer=self._adv_optim,
            net=self._adv_net,
            grad_norm_clip=self._adv_args.grad_norm_clipping,
        )

    def apply_grads_avrg(self, list_of_grads):
        self._apply_grads(
            list_of_grads=list_of_grads,
            optimizer=self._avrg_optim,
            net=self._avrg_net,
            grad_norm_clip=self._avrg_args.grad_norm_clipping,
        )

    def apply_grads_baseline(self, list_of_grads):
        self._apply_grads(
            list_of_grads=list_of_grads,
            optimizer=self._baseline_optim,
            net=self._baseline_net,
            grad_norm_clip=self._baseline_args.grad_norm_clipping,
        )

    def reset_adv_net(self, cfr_iter):
        if self._adv_args.init_adv_model == "last":
            self._adv_net.zero_grad()
            if not self._t_prof.online:
                self._adv_optim, self._adv_lr_scheduler = self._get_new_adv_optim()
        elif self._adv_args.init_adv_model == "random":
            self._adv_net = self._get_new_adv_net()
            self._adv_optim, self._adv_lr_scheduler = self._get_new_adv_optim()
        else:
            raise ValueError(self._adv_args.init_adv_model)

        if self._t_prof.log_memory and (cfr_iter % 3 == 0):
            # Logs
            process = psutil.Process(os.getpid())
            self._ray.remote(
                self._chief_handle.add_scalar,
                self._exp_mem_usage,
                "Debug/MemoryUsage/PS",
                cfr_iter,
                process.memory_info().rss,
            )

    def reset_avrg_net(self):
        if self._avrg_args.init_avrg_model == "last":
            self._avrg_net.zero_grad()
            if not self._t_prof.online:
                self._avrg_optim, self._avrg_lr_scheduler = self._get_new_avrg_optim()

        elif self._avrg_args.init_avrg_model == "random":
            self._avrg_net = self._get_new_avrg_net()
            self._avrg_optim, self._avrg_lr_scheduler = self._get_new_avrg_optim()

        else:
            raise ValueError(self._avrg_args.init_avrg_model)

    def step_scheduler_adv(self, loss):
        self._adv_lr_scheduler.step(loss)

    def step_scheduler_avrg(self, loss):
        self._avrg_lr_scheduler.step(loss)

    # ______________________________________________ API for checkpointing _____________________________________________
    def checkpoint(self, curr_step):
        state = {
            "adv_net": self._adv_net.state_dict(),
            "adv_optim": self._adv_optim.state_dict(),
            "adv_lr_sched": self._adv_lr_scheduler.state_dict(),
            "seat_id": self.owner,
        }
        if self._AVRG:
            state["avrg_net"] = self._avrg_net.state_dict()
            state["avrg_optim"] = self._avrg_optim.state_dict()
            state["avrg_lr_sched"] = self._avrg_lr_scheduler.state_dict()

        if self._BASELINE:
            state["baseline_net"] = self._baseline_net.state_dict()
            state["baseline_optim"] = self._baseline_optim.state_dict()

        with open(
            self._get_checkpoint_file_path(
                name=self._t_prof.name,
                step=curr_step,
                cls=self.__class__,
                worker_id="P" + str(self.owner),
            ),
            "wb",
        ) as pkl_file:
            pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, name_to_load, step):
        with open(
            self._get_checkpoint_file_path(
                name=name_to_load,
                step=step,
                cls=self.__class__,
                worker_id="P" + str(self.owner),
            ),
            "rb",
        ) as pkl_file:
            state = pickle.load(pkl_file)

            assert self.owner == state["seat_id"]

        self._adv_net.load_state_dict(state["adv_net"])
        self._adv_optim.load_state_dict(state["adv_optim"])
        self._adv_lr_scheduler.load_state_dict(state["adv_lr_sched"])

        if self._AVRG:
            self._avrg_net.load_state_dict(state["avrg_net"])
            self._avrg_optim.load_state_dict(state["avrg_optim"])
            self._avrg_lr_scheduler.load_state_dict(state["avrg_lr_sched"])

        if self._BASELINE:
            self._baseline_net.load_state_dict(state["baseline_net"])
            self._baseline_optim.load_state_dict(state["baseline_optim"])

    # __________________________________________________________________________________________________________________
    def _get_new_adv_net(self):
        return DuelingQNet(
            q_args=self._adv_args.adv_net_args,
            env_bldr=self._env_bldr,
            device=self._device,
        )

    def _get_new_avrg_net(self):
        return AvrgStrategyNet(
            avrg_net_args=self._avrg_args.avrg_net_args,
            env_bldr=self._env_bldr,
            device=self._device,
        )

    def _get_new_baseline_net(self):
        return DuelingQNet(
            q_args=self._baseline_args.q_net_args,
            env_bldr=self._env_bldr,
            device=self._device,
        )

    def _get_new_adv_optim(self):
        opt = rl_util.str_to_optim_cls(self._adv_args.optim_str)(
            self._adv_net.parameters(), lr=self._adv_args.lr
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            threshold=0.001,
            factor=0.5,
            patience=self._adv_args.lr_patience,
            min_lr=0.00002,
        )
        return opt, scheduler

    def _get_new_avrg_optim(self):
        opt = rl_util.str_to_optim_cls(self._avrg_args.optim_str)(
            self._avrg_net.parameters(), lr=self._avrg_args.lr
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            threshold=0.0001,
            factor=0.5,
            patience=self._avrg_args.lr_patience,
            min_lr=0.00002,
        )
        return opt, scheduler

    def _get_new_baseline_optim(self):
        opt = rl_util.str_to_optim_cls(self._baseline_args.optim_str)(
            self._baseline_net.parameters(), lr=self._baseline_args.lr
        )
        return opt


# ============================================================
# LearnerActor
# ============================================================

# Copyright (c) Eric Steinberger 2020


class LearnerActor(WorkerBase):
    def __init__(self, t_prof, worker_id, chief_handle):
        super().__init__(t_prof=t_prof)

        self._adv_args = t_prof.module_args["adv_training"]

        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)
        self._id = worker_id
        self._chief_handle = chief_handle

        self._adv_buffers = [
            AdvReservoirBuffer(
                owner=p,
                env_bldr=self._env_bldr,
                max_size=self._adv_args.max_buffer_size,
                nn_type=t_prof.nn_type,
                iter_weighting_exponent=self._t_prof.iter_weighting_exponent,
            )
            for p in range(self._t_prof.n_seats)
        ]

        self._adv_wrappers = [
            AdvWrapper(
                owner=p,
                env_bldr=self._env_bldr,
                adv_training_args=self._adv_args,
                device=self._adv_args.device_training,
            )
            for p in range(self._t_prof.n_seats)
        ]

        self._AVRG = (
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        )
        self._SINGLE = (
            EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo
        )

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]

            self._avrg_buffers = [
                AvrgReservoirBuffer(
                    owner=p,
                    env_bldr=self._env_bldr,
                    max_size=self._avrg_args.max_buffer_size,
                    nn_type=t_prof.nn_type,
                    iter_weighting_exponent=self._t_prof.iter_weighting_exponent,
                )
                for p in range(self._t_prof.n_seats)
            ]

            self._avrg_wrappers = [
                AvrgWrapper(
                    owner=p,
                    env_bldr=self._env_bldr,
                    avrg_training_args=self._avrg_args,
                    device=self._avrg_args.device_training,
                )
                for p in range(self._t_prof.n_seats)
            ]

            if self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=self._avrg_buffers,
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples,
                )

            elif self._t_prof.sampler.lower() == "learned_baseline":
                assert t_prof.module_args["mccfr_baseline"] is not None, (
                    "Please give 'baseline_args' for VR Sampler."
                )
                self._baseline_args = t_prof.module_args["mccfr_baseline"]
                self._baseline_wrapper = BaselineWrapper(
                    env_bldr=self._env_bldr, baseline_args=self._baseline_args
                )

                self._baseline_buf = CrazyBaselineQCircularBuffer(
                    owner=None,
                    env_bldr=self._env_bldr,
                    max_size=self._baseline_args.max_buffer_size,
                    nn_type=t_prof.nn_type,
                )

                self._data_sampler = LearnedBaselineSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    eps=self._t_prof.os_eps,
                    baseline_net=self._baseline_wrapper,
                    baseline_buf=self._baseline_buf,
                    avrg_buffers=self._avrg_buffers,
                )
            else:
                raise ValueError(
                    "Currently we don't support",
                    self._t_prof.sampler.lower(),
                    "sampling.",
                )
        else:
            if self._t_prof.sampler.lower() == "learned_baseline":
                assert t_prof.module_args["mccfr_baseline"] is not None, (
                    "Please give 'baseline_args' for VR Sampler."
                )
                self._baseline_args = t_prof.module_args["mccfr_baseline"]
                self._baseline_wrapper = BaselineWrapper(
                    env_bldr=self._env_bldr, baseline_args=self._baseline_args
                )

                self._baseline_buf = CrazyBaselineQCircularBuffer(
                    owner=None,
                    env_bldr=self._env_bldr,
                    max_size=self._baseline_args.max_buffer_size,
                    nn_type=t_prof.nn_type,
                )

                self._data_sampler = LearnedBaselineSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    eps=self._t_prof.os_eps,
                    baseline_net=self._baseline_wrapper,
                    baseline_buf=self._baseline_buf,
                )

            elif self._t_prof.sampler.lower() == "es":
                self._data_sampler = ExternalSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=None,
                )

            elif self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=None,
                    eps=self._t_prof.os_eps,
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples,
                )
            else:
                raise ValueError(
                    "Currently we don't support",
                    self._t_prof.sampler.lower(),
                    "sampling.",
                )

        if self._t_prof.log_verbose:
            self._exp_mem_usage = self._ray.get(
                self._ray.remote(
                    self._chief_handle.create_experiment,
                    self._t_prof.name + "_LA" + str(worker_id) + "_Memory_Usage",
                )
            )
            self._exps_adv_buffer_size = self._ray.get(
                [
                    self._ray.remote(
                        self._chief_handle.create_experiment,
                        self._t_prof.name
                        + "_LA"
                        + str(worker_id)
                        + "_P"
                        + str(p)
                        + "_ADV_BufSize",
                    )
                    for p in range(self._t_prof.n_seats)
                ]
            )
            if self._AVRG:
                self._exps_avrg_buffer_size = self._ray.get(
                    [
                        self._ray.remote(
                            self._chief_handle.create_experiment,
                            self._t_prof.name
                            + "_LA"
                            + str(worker_id)
                            + "_P"
                            + str(p)
                            + "_AVRG_BufSize",
                        )
                        for p in range(self._t_prof.n_seats)
                    ]
                )

    def generate_data(self, traverser, cfr_iter):
        iteration_strats = [
            IterationStrategy(
                t_prof=self._t_prof,
                env_bldr=self._env_bldr,
                owner=p,
                device=self._t_prof.device_inference,
                cfr_iter=cfr_iter,
            )
            for p in range(self._t_prof.n_seats)
        ]
        for s in iteration_strats:
            s.load_net_state_dict(
                state_dict=self._adv_wrappers[s.owner].net_state_dict()
            )

        self._data_sampler.generate(
            n_traversals=self._t_prof.n_traversals_per_iter,
            traverser=traverser,
            iteration_strats=iteration_strats,
            cfr_iter=cfr_iter,
        )

        # Log after both players generated data
        if self._t_prof.log_verbose and traverser == 1 and (cfr_iter % 3 == 0):
            for p in range(self._t_prof.n_seats):
                self._ray.remote(
                    self._chief_handle.add_scalar,
                    self._exps_adv_buffer_size[p],
                    "Debug/BufferSize",
                    cfr_iter,
                    self._adv_buffers[p].size,
                )
                if self._AVRG:
                    self._ray.remote(
                        self._chief_handle.add_scalar,
                        self._exps_avrg_buffer_size[p],
                        "Debug/BufferSize",
                        cfr_iter,
                        self._avrg_buffers[p].size,
                    )

            process = psutil.Process(os.getpid())
            self._ray.remote(
                self._chief_handle.add_scalar,
                self._exp_mem_usage,
                "Debug/MemoryUsage/LA",
                cfr_iter,
                process.memory_info().rss,
            )

        return self._data_sampler.total_node_count_traversed

    def update(
        self, adv_state_dicts=None, avrg_state_dicts=None, baseline_state_dict=None
    ):
        """
        Args:
            adv_state_dicts (list):         Optional. if not None:
                                                        expects a list of neural net state dicts or None for each player
                                                        in order of their seat_ids. This allows updating only some
                                                        players.

            avrg_state_dicts (list):         Optional. if not None:
                                                        expects a list of neural net state dicts or None for each player
                                                        in order of their seat_ids. This allows updating only some
                                                        players.
        """
        baseline_state_dict = baseline_state_dict[0]  # wrapped bc of object id stuff
        if baseline_state_dict is not None:
            self._baseline_wrapper.load_net_state_dict(
                state_dict=self._ray.state_dict_to_torch(
                    self._ray.get(baseline_state_dict),
                    device=self._baseline_wrapper.device,
                )
            )

        for p_id in range(self._t_prof.n_seats):
            if adv_state_dicts[p_id] is not None:
                self._adv_wrappers[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(
                        self._ray.get(adv_state_dicts[p_id]),
                        device=self._adv_wrappers[p_id].device,
                    )
                )

            if avrg_state_dicts[p_id] is not None:
                self._avrg_wrappers[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(
                        self._ray.get(avrg_state_dicts[p_id]),
                        device=self._avrg_wrappers[p_id].device,
                    )
                )

    def get_loss_last_batch_adv(self, p_id):
        return self._adv_wrappers[p_id].loss_last_batch

    def get_loss_last_batch_avrg(self, p_id):
        return self._avrg_wrappers[p_id].loss_last_batch

    def get_loss_last_batch_baseline(self):
        return self._baseline_wrapper.loss_last_batch

    def get_adv_grads(self, p_id):
        return self._ray.grads_to_numpy(
            self._adv_wrappers[p_id].get_grads_one_batch_from_buffer(
                buffer=self._adv_buffers[p_id]
            )
        )

    def get_avrg_grads(self, p_id):
        return self._ray.grads_to_numpy(
            self._avrg_wrappers[p_id].get_grads_one_batch_from_buffer(
                buffer=self._avrg_buffers[p_id]
            )
        )

    def get_baseline_grads(self):
        return self._ray.grads_to_numpy(
            self._baseline_wrapper.get_grads_one_batch_from_buffer(
                buffer=self._baseline_buf
            )
        )


# ============================================================
# HighLevelAlgo
# ============================================================

# Copyright (c) Eric Steinberger 2020


class HighLevelAlgo(_HighLevelAlgoBase):
    def __init__(self, t_prof, la_handles, ps_handles, chief_handle):
        super().__init__(
            t_prof=t_prof, chief_handle=chief_handle, la_handles=la_handles
        )
        self._ps_handles = ps_handles
        self._all_p_aranged = list(range(self._t_prof.n_seats))

        self._AVRG = (
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        )
        self._SINGLE = (
            EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo
        )
        self._BASELINE = t_prof.sampler == "learned_baseline"
        if self._BASELINE:
            self._baseline_args = t_prof.module_args["mccfr_baseline"]

        self._adv_args = t_prof.module_args["adv_training"]
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]

        self._exp_states_traversed = self._ray.get(
            self._ray.remote(
                self._chief_handle.create_experiment,
                self._t_prof.name + "_States_traversed",
            )
        )

    def init(self):
        # """"""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""
        if self._AVRG:
            self._update_leaner_actors(
                update_adv_for_plyrs=self._all_p_aranged,
                update_avrg_for_plyrs=self._all_p_aranged,
            )

        # """"""""""""""""""""""
        # NOT Deep CFR
        # """"""""""""""""""""""
        else:
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)

    def run_one_iter_alternating_update(self, cfr_iter):
        t_generating_data = 0.0
        t_computation_adv = 0.0
        t_syncing_adv = 0.0
        for p_learning in range(self._t_prof.n_seats):
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)
            print("Generating Data...")
            _t_generating_data = self._generate_traversals(
                p_id=p_learning, cfr_iter=cfr_iter
            )
            t_generating_data += _t_generating_data

            print("Training Advantage Net...")
            _t_computation_adv, _t_syncing_adv = self._train_adv(
                p_id=p_learning, cfr_iter=cfr_iter
            )
            t_computation_adv += _t_computation_adv
            t_syncing_adv += _t_syncing_adv

            if self._SINGLE:
                print("Pushing new net to chief...")
                self._push_newest_adv_net_to_chief(p_id=p_learning, cfr_iter=cfr_iter)

        if self._BASELINE:
            t_computation_baseline, t_syncing_baseline = self._train_baseline(
                n_updates=self._baseline_args.n_batches_per_iter_baseline
            )

        print("Synchronizing...")
        self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)

        ret = {
            "t_generating_data": t_generating_data,
            "t_computation_adv": t_computation_adv,
            "t_syncing_adv": t_syncing_adv,
        }
        if self._BASELINE:
            ret["t_computation_baseline"] = t_computation_baseline
            ret["t_syncing_baseline"] = t_syncing_baseline

        return ret

    def train_average_nets(self, cfr_iter):
        print("Training Average Nets...")
        t_computation_avrg = 0.0
        t_syncing_avrg = 0.0
        for p in range(self._t_prof.n_seats):
            _c, _s = self._train_avrg(p_id=p, cfr_iter=cfr_iter)
            t_computation_avrg += _c
            t_syncing_avrg += _s

        return {
            "t_computation_avrg": t_computation_avrg,
            "t_syncing_avrg": t_syncing_avrg,
        }

    def _train_adv(self, p_id, cfr_iter):
        t_computation = 0.0
        t_syncing = 0.0

        # For logging the loss to see convergence in Tensorboard
        if self._t_prof.log_verbose:
            exp_loss_each_p = [
                self._ray.remote(
                    self._chief_handle.create_experiment,
                    self._t_prof.name + "_ADV_Loss_P" + str(p) + "_I" + str(cfr_iter),
                )
                for p in range(self._t_prof.n_seats)
            ]

        if cfr_iter % self._t_prof.periodic_restart == 0:
            self._ray.wait(
                [self._ray.remote(self._ps_handles[p_id].reset_adv_net, cfr_iter)]
            )
            NB = self._adv_args.n_batches_adv_training
        else:
            NB = int(self._adv_args.n_batches_adv_training / 5)

        self._update_leaner_actors(update_adv_for_plyrs=[p_id])

        SMOOTHING = 200
        accumulated_averaged_loss = 0.0
        for epoch_nr in range(NB):
            t0 = time.time()

            # Compute gradients
            grads_from_all_las, _averaged_loss = self._get_adv_gradients(p_id=p_id)
            accumulated_averaged_loss += _averaged_loss

            t_computation += time.time() - t0

            # Applying gradients
            t0 = time.time()
            self._ray.wait(
                [
                    self._ray.remote(
                        self._ps_handles[p_id].apply_grads_adv, grads_from_all_las
                    )
                ]
            )

            # Step LR scheduler
            self._ray.wait(
                [
                    self._ray.remote(
                        self._ps_handles[p_id].step_scheduler_adv, _averaged_loss
                    )
                ]
            )

            # update ADV on all las
            self._update_leaner_actors(update_adv_for_plyrs=[p_id])

            # log current loss
            if self._t_prof.log_verbose and ((epoch_nr + 1) % SMOOTHING == 0):
                self._ray.wait(
                    [
                        self._ray.remote(
                            self._chief_handle.add_scalar,
                            exp_loss_each_p[p_id],
                            "DCFR_NN_Losses/Advantage",
                            epoch_nr,
                            accumulated_averaged_loss / SMOOTHING,
                        )
                    ]
                )
                accumulated_averaged_loss = 0.0

            t_syncing += time.time() - t0

        return t_computation, t_syncing

    def _get_adv_gradients(self, p_id):
        grads = [self._ray.remote(la.get_adv_grads, p_id) for la in self._la_handles]
        self._ray.wait(grads)

        losses = self._ray.get(
            [
                self._ray.remote(la.get_loss_last_batch_adv, p_id)
                for la in self._la_handles
            ]
        )

        losses = [loss for loss in losses if loss is not None]

        n = len(losses)
        averaged_loss = sum(losses) / float(n) if n > 0 else -1

        return grads, averaged_loss

    def _generate_traversals(self, p_id, cfr_iter):
        t_gen = time.time()
        states_seen = self._ray.get(
            [
                self._ray.remote(la.generate_data, p_id, cfr_iter)
                for la in self._la_handles
            ]
        )
        t_gen = time.time() - t_gen

        if p_id == 1:
            self._ray.wait(
                [
                    self._ray.remote(
                        self._chief_handle.add_scalar,
                        self._exp_states_traversed,
                        "States Seen",
                        cfr_iter,
                        sum(states_seen),
                    )
                ]
            )

        return t_gen

    def _update_leaner_actors(
        self,
        update_adv_for_plyrs=None,
        update_avrg_for_plyrs=None,
        update_baseline=None,
    ):
        """

        Args:
            update_adv_for_plyrs (list):         list of player_ids to update adv for
            update_avrg_for_plyrs (list):        list of player_ids to update avrg for
        """

        assert isinstance(update_adv_for_plyrs, list) or update_adv_for_plyrs is None
        assert isinstance(update_avrg_for_plyrs, list) or update_avrg_for_plyrs is None
        assert isinstance(update_baseline, bool) or update_baseline is None

        _update_adv_per_p = [
            True
            if (update_adv_for_plyrs is not None) and (p in update_adv_for_plyrs)
            else False
            for p in range(self._t_prof.n_seats)
        ]

        _update_avrg_per_p = [
            True
            if (update_avrg_for_plyrs is not None) and (p in update_avrg_for_plyrs)
            else False
            for p in range(self._t_prof.n_seats)
        ]

        la_batches = []
        n = len(self._la_handles)
        c = 0
        while n > c:
            s = min(n, c + self._t_prof.max_n_las_sync_simultaneously)
            la_batches.append(self._la_handles[c:s])
            if type(la_batches[-1]) is not list:
                la_batches[-1] = [la_batches[-1]]
            c = s

        w_adv = [None for _ in range(self._t_prof.n_seats)]
        w_avrg = [None for _ in range(self._t_prof.n_seats)]

        w_baseline = [
            None
            if not update_baseline
            else self._ray.remote(self._ps_handles[0].get_baseline_weights)
        ]

        for p_id in range(self._t_prof.n_seats):
            w_adv[p_id] = (
                None
                if not _update_adv_per_p[p_id]
                else self._ray.remote(self._ps_handles[p_id].get_adv_weights)
            )

            w_avrg[p_id] = (
                None
                if not _update_avrg_per_p[p_id]
                else self._ray.remote(self._ps_handles[p_id].get_avrg_weights)
            )

        for batch in la_batches:
            self._ray.wait(
                [
                    self._ray.remote(
                        la.update,
                        w_adv,
                        w_avrg,
                        w_baseline,
                    )
                    for la in batch
                ]
            )

    # ____________ Baseline
    def _train_baseline(self, n_updates):
        t_computation = 0.0
        t_syncing = 0.0
        self._update_leaner_actors(update_baseline=True)

        for epoch_nr in range(n_updates):
            t0 = time.time()

            # Compute gradients
            grads_from_all_las = [
                self._ray.remote(
                    la.get_baseline_grads,
                )
                for la in self._la_handles
            ]
            self._ray.wait(grads_from_all_las)
            t_computation += time.time() - t0

            # Applying gradients
            t0 = time.time()
            self._ray.wait(
                [
                    self._ray.remote(
                        self._ps_handles[0].apply_grads_baseline, grads_from_all_las
                    )
                ]
            )

            # update Baseline on all las
            self._update_leaner_actors(update_baseline=True)

            t_syncing += time.time() - t0

        return t_computation, t_syncing

    # ____________ SINGLE only
    def _push_newest_adv_net_to_chief(self, p_id, cfr_iter):
        self._ray.wait(
            [
                self._ray.remote(
                    self._chief_handle.add_new_iteration_strategy_model,
                    p_id,
                    self._ray.remote(self._ps_handles[p_id].get_adv_weights),
                    cfr_iter,
                )
            ]
        )

    # ____________ AVRG only
    def _get_avrg_gradients(self, p_id):
        grads = [self._ray.remote(la.get_avrg_grads, p_id) for la in self._la_handles]
        self._ray.wait(grads)

        losses = self._ray.get(
            [
                self._ray.remote(la.get_loss_last_batch_avrg, p_id)
                for la in self._la_handles
            ]
        )

        losses = [loss for loss in losses if loss is not None]

        n = len(losses)
        averaged_loss = sum(losses) / float(n) if n > 0 else -1

        return grads, averaged_loss

    def _train_avrg(self, p_id, cfr_iter):
        t_computation = 0.0
        t_syncing = 0.0

        # For logging the loss to see convergence in Tensorboard
        if self._t_prof.log_verbose:
            exp_loss_each_p = [
                self._ray.remote(
                    self._chief_handle.create_experiment,
                    self._t_prof.name
                    + "_AverageNet_Loss_P"
                    + str(p)
                    + "_I"
                    + str(cfr_iter),
                )
                for p in range(self._t_prof.n_seats)
            ]

        self._ray.wait([self._ray.remote(self._ps_handles[p_id].reset_avrg_net)])
        self._update_leaner_actors(update_avrg_for_plyrs=[p_id])

        SMOOTHING = 200
        accumulated_averaged_loss = 0.0

        if cfr_iter > 0:
            for epoch_nr in range(self._avrg_args.n_batches_avrg_training):
                t0 = time.time()

                # Compute gradients
                grads_from_all_las, _averaged_loss = self._get_avrg_gradients(p_id=p_id)
                accumulated_averaged_loss += _averaged_loss

                t_computation += time.time() - t0

                # Applying gradients
                t0 = time.time()
                self._ray.wait(
                    [
                        self._ray.remote(
                            self._ps_handles[p_id].apply_grads_avrg, grads_from_all_las
                        )
                    ]
                )

                # Step LR scheduler
                self._ray.wait(
                    [
                        self._ray.remote(
                            self._ps_handles[p_id].step_scheduler_avrg, _averaged_loss
                        )
                    ]
                )

                # update AvrgStrategyNet on all las
                self._update_leaner_actors(update_avrg_for_plyrs=[p_id])

                # log current loss
                if self._t_prof.log_verbose and ((epoch_nr + 1) % SMOOTHING == 0):
                    self._ray.wait(
                        [
                            self._ray.remote(
                                self._chief_handle.add_scalar,
                                exp_loss_each_p[p_id],
                                "DCFR_NN_Losses/Average",
                                epoch_nr,
                                accumulated_averaged_loss / SMOOTHING,
                            )
                        ]
                    )
                    accumulated_averaged_loss = 0.0

                t_syncing += time.time() - t0

        return t_computation, t_syncing


# ============================================================
# Driver
# ============================================================

# Copyright (c) Eric Steinberger 2020


class Driver(DriverBase):
    def __init__(
        self,
        t_prof,
        eval_methods,
        n_iterations=None,
        iteration_to_import=None,
        name_to_import=None,
    ):
        if t_prof.DISTRIBUTED:
            raise NotImplementedError(
                "`dream-full.py` only supports DISTRIBUTED=False; "
                "distributed worker wrappers were not vendored."
            )

        chief_cls = Chief
        learner_actor_cls = LearnerActor
        parameter_server_cls = ParameterServer

        super().__init__(
            t_prof=t_prof,
            eval_methods=eval_methods,
            n_iterations=n_iterations,
            iteration_to_import=iteration_to_import,
            name_to_import=name_to_import,
            chief_cls=chief_cls,
            eval_agent_cls=EvalAgentDeepCFR,
        )

        if "h2h" in list(eval_methods.keys()):
            assert EvalAgentDeepCFR.EVAL_MODE_SINGLE in t_prof.eval_modes_of_algo
            assert EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in t_prof.eval_modes_of_algo
            self._ray.remote(
                self.eval_masters["h2h"][0].set_modes,
                [
                    EvalAgentDeepCFR.EVAL_MODE_SINGLE,
                    EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,
                ],
            )

        print("Creating LAs...")
        self.la_handles = [
            self._ray.create_worker(learner_actor_cls, t_prof, i, self.chief_handle)
            for i in range(t_prof.n_learner_actors)
        ]

        print("Creating Parameter Servers...")
        self.ps_handles = [
            self._ray.create_worker(parameter_server_cls, t_prof, p, self.chief_handle)
            for p in range(t_prof.n_seats)
        ]

        self._ray.wait(
            [
                self._ray.remote(self.chief_handle.set_ps_handle, *self.ps_handles),
                self._ray.remote(self.chief_handle.set_la_handles, *self.la_handles),
            ]
        )

        print("Created and initialized Workers")

        self.algo = HighLevelAlgo(
            t_prof=t_prof,
            la_handles=self.la_handles,
            ps_handles=self.ps_handles,
            chief_handle=self.chief_handle,
        )

        self._AVRG = (
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        )
        self._SINGLE = (
            EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo
        )

        self._BASELINE = t_prof.sampler == "learned_baseline"

        self._maybe_load_checkpoint_init()

    def run(self):
        print("Setting stuff up...")

        # """"""""""""""""
        # Init globally
        # """"""""""""""""
        self.algo.init()

        print("Starting Training...")
        for _iter_nr in range(
            10000000 if self.n_iterations is None else self.n_iterations
        ):
            print("Iteration: ", self._iteration)

            # """"""""""""""""
            # Maybe train AVRG
            # """"""""""""""""
            avrg_times = None
            if self._AVRG and self._any_eval_needs_avrg_net():
                avrg_times = self.algo.train_average_nets(cfr_iter=_iter_nr)

            # """"""""""""""""
            # Eval
            # """"""""""""""""
            # Evaluate. Sync & Lock, then train while evaluating on other workers
            self.evaluate()

            # """"""""""""""""
            # Log
            # """"""""""""""""
            if self._iteration % self._t_prof.log_export_freq == 0:
                self.save_logs()
            self.periodically_export_eval_agent()

            # """"""""""""""""
            # Iteration
            # """"""""""""""""
            iter_times = self.algo.run_one_iter_alternating_update(
                cfr_iter=self._iteration
            )

            print(
                "Generating Data: ",
                str(iter_times["t_generating_data"]) + "s.",
                "  ||  Trained ADV",
                str(iter_times["t_computation_adv"]) + "s.",
                "  ||  Synced ADV",
                str(iter_times["t_syncing_adv"]) + "s.",
            )
            if self._BASELINE:
                print(
                    "Trained Baseline",
                    str(iter_times["t_computation_baseline"]) + "s.",
                    "  ||  Synced Baseline",
                    str(iter_times["t_syncing_baseline"]) + "s.",
                    "\n",
                )

            if self._AVRG and avrg_times:
                print(
                    "Trained AVRG",
                    str(avrg_times["t_computation_avrg"]) + "s.",
                    "  ||  Synced AVRG",
                    str(avrg_times["t_syncing_avrg"]) + "s.",
                    "\n",
                )

            self._iteration += 1

            # """"""""""""""""
            # Checkpoint
            # """"""""""""""""
            self.periodically_checkpoint()

    def _any_eval_needs_avrg_net(self):
        for e in list(self.eval_masters.values()):
            if self._iteration % e[1] == 0:
                return True
        return False

    def checkpoint(self, **kwargs):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]:
            self._ray.wait([self._ray.remote(w.checkpoint, self._iteration)])

        # Delete past checkpoints
        s = [self._iteration]
        if self._iteration > self._t_prof.checkpoint_freq + 1:
            s.append(self._iteration - self._t_prof.checkpoint_freq)

        self._delete_past_checkpoints(steps_not_to_delete=s)

    def load_checkpoint(self, step, name_to_load):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]:
            self._ray.wait([self._ray.remote(w.load_checkpoint, name_to_load, step)])


# ============================================================
# Runner helpers
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "poker_ai_data"
CRAYON_HOST = "localhost"
CRAYON_PORT = 8889
CRAYON_IMAGE = "alband/crayon"
CRAYON_CONTAINER = "crayon"
GAME_REGISTRY = {
    "leduc": StandardLeduc,
    "fhp": Flop3Holdem,
}


def port_is_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_crayon_server():
    if port_is_open(CRAYON_HOST, CRAYON_PORT):
        return

    existing = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={CRAYON_CONTAINER}",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
    )
    if existing.returncode != 0:
        raise RuntimeError("Docker is required because the repo logs through Crayon.")

    if CRAYON_CONTAINER in existing.stdout.splitlines():
        command = ["docker", "start", CRAYON_CONTAINER]
    else:
        command = [
            "docker",
            "run",
            "-d",
            "-p",
            "8888:8888",
            "-p",
            "8889:8889",
            "--name",
            CRAYON_CONTAINER,
            CRAYON_IMAGE,
        ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or "Failed to start Crayon."
        )

    for _ in range(30):
        if port_is_open(CRAYON_HOST, CRAYON_PORT):
            return
        time.sleep(1)
    raise RuntimeError("Crayon started, but localhost:8889 never became reachable.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-file DREAM runner without local DREAM imports."
    )
    parser.add_argument("--game", choices=sorted(GAME_REGISTRY), default="leduc")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--traversals", type=int, default=100)
    parser.add_argument("--adv-batches", type=int, default=100)
    parser.add_argument("--baseline-batches", type=int, default=50)
    parser.add_argument("--avrg-batches", type=int, default=50)
    parser.add_argument("--adv-batch-size", type=int, default=512)
    parser.add_argument("--baseline-batch-size", type=int, default=512)
    parser.add_argument("--avrg-batch-size", type=int, default=512)
    parser.add_argument("--n-learner-actors", type=int, default=1)
    parser.add_argument("--max-sync-lars", type=int, default=1)
    parser.add_argument("--os-eps", type=float, default=0.5)
    parser.add_argument("--training-device", default="cpu")
    parser.add_argument("--parameter-server-device", default="cpu")
    parser.add_argument("--inference-device", default="cpu")
    parser.add_argument("--log-verbose", action="store_true")
    return parser.parse_args()


def build_run_name(args):
    if args.run_name is not None:
        return args.run_name
    return f"DREAM_{args.game.upper()}_{int(time.time())}"


def build_training_profile(args, run_name):
    game_cls = GAME_REGISTRY[args.game]
    return TrainingProfile(
        name=run_name,
        path_data=str(DATA_ROOT),
        nn_type="feedforward",
        game_cls=game_cls,
        sampler="learned_baseline",
        os_eps=args.os_eps,
        n_traversals_per_iter=args.traversals,
        n_batches_adv_training=args.adv_batches,
        n_batches_per_iter_baseline=args.baseline_batches,
        n_batches_avrg_training=args.avrg_batches,
        mini_batch_size_adv=args.adv_batch_size,
        mini_batch_size_avrg=args.avrg_batch_size,
        batch_size_baseline=args.baseline_batch_size,
        n_learner_actor_workers=args.n_learner_actors,
        max_n_las_sync_simultaneously=args.max_sync_lars,
        device_inference=args.inference_device,
        device_training=args.training_device,
        device_parameter_server=args.parameter_server_device,
        log_export_freq=1,
        eval_agent_export_freq=1,
        checkpoint_freq=10**9,
        eval_modes_of_algo=(
            EvalAgentDeepCFR.EVAL_MODE_SINGLE,
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,
        ),
        DISTRIBUTED=False,
        log_verbose=args.log_verbose,
    )


def load_latest_metrics(run_name):
    run_dir = DATA_ROOT / "logs" / run_name
    step_dirs = [
        path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()
    ]
    latest_step_dir = max(step_dirs, key=lambda path: int(path.name))
    with open(
        latest_step_dir / "as_json" / "logs.json", "r", encoding="utf-8"
    ) as file_handle:
        return int(latest_step_dir.name), json.load(file_handle)


def print_final_report(run_name):
    latest_step, metrics = load_latest_metrics(run_name)
    print("\nFinal exported metrics")
    print(f"- run: {run_name}")
    print(f"- exported step: {latest_step}")
    for experiment_name, scalar_groups in sorted(metrics.items()):
        print(f"- experiment: {experiment_name}")
        for metric_name, series in sorted(scalar_groups.items()):
            if not series:
                continue
            last_point = series[-1]
            iter_idx, value = next(iter(last_point.items()))
            print(f"  - {metric_name}: iter {iter_idx} -> {value}")
    print("\nArtifacts")
    print(f"- logs: {DATA_ROOT / 'logs' / run_name}")
    print(f"- eval agents: {DATA_ROOT / 'eval_agent' / run_name}")
    print(f"- training profile: {DATA_ROOT / 'TrainingProfiles' / (run_name + '.pkl')}")
    print("- dashboard: http://localhost:8888")


def main():
    args = parse_args()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_crayon_server()
    run_name = build_run_name(args)
    print("Preparing training profile...")
    profile = build_training_profile(args, run_name)
    print("Starting training...")
    driver = Driver(
        t_prof=profile,
        eval_methods={"br": max(args.eval_every, 1)},
        n_iterations=args.iterations,
    )
    driver.run()
    print_final_report(run_name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
