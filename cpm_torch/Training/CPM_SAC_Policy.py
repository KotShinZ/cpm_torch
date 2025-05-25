import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from torch import nn
from typing import Optional, Union, Any, Type, Dict, List, Tuple # Tuple を追加
import cpm_torch.Training.U_Net as U_Net
from cpm_torch.CPM_Map import extract_patches_manual_padding_with_offset_batch

from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp, # MultiAgentContinuousCritic で使用
    get_actor_critic_arch, # MultiAgentSACPolicy で使用
)
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs
from stable_baselines3.common.utils import get_schedule_fn # 学習率スケジュール用
# SACPolicy が内部で使用するコンポーネント
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim

import torch

LOG_STD_MAX = 2
LOG_STD_MIN = -20

from cpm_torch.Training.ThroughExtractor import ThroughExtractor

class Actor_Net(nn.Module):
    """
    CPM用のActorネットワーク。
    (B, 256, 256, C)の観測を受け取り、(B, N * D_act)の行動を出力する。
    """
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box):
        super().__init__()
        self.fc1 = nn.Linear(observation_space.shape[2] * 9, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, 4)
        self.fc_log_std = nn.Linear(128, 4)

    def forward(self, x: th.Tensor) -> th.Tensor:
        B = x.shape[0]  # バッチサイズ
        patched = self.preprocess_obs(x)  # (B*7396, 9*C)
        x = th.relu(self.fc1(patched))  # (B*7396, 256)
        x = th.relu(self.fc2(x))  # (B*7396, 128)
        x = self.fc3(x) # (B*7396, 4)
        mean = self.fc_mean(x)  # (B*7396, 4)
        log_std = self.fc_log_std(x)  # (B*7396, 4)
        return  mean.reshape(B, -1), log_std.reshape(B, -1)
    
    
    def vectorized_row_unique_rank_by_appearance(self, x: torch.Tensor) -> torch.Tensor:
        """ベクトル化された手法で各行の出現順ランクを計算する関数"""
        B, N = x.shape
        device = x.device

        if B == 0:  # B=0 の場合のガード処理
            return torch.empty((0, N), dtype=torch.int64, device=device)
        if N == 0:  # N=0 の場合のガード処理 (B > 0)
            return torch.empty_like(x, dtype=torch.int64)

        # ステージ1: 各要素の値がその行で最初に出現する列インデックスを計算 (mci)
        cols_broadcast = torch.arange(N, device=device).view(1, 1, N).expand(B, N, N)
        eq_mask = x.unsqueeze(2) == x.unsqueeze(1)

        sentinel = N
        masked_cols = torch.where(eq_mask, cols_broadcast, sentinel)
        mci = masked_cols.min(dim=2).values

        # ステージ2: mci テンソルの各行に対してデンスランクを計算
        A = mci
        S_values, S_indices = A.sort(dim=1)

        R_sorted = torch.zeros_like(A)
        R_sorted[:, 1:] = (S_values[:, 1:] != S_values[:, :-1]).cumsum(dim=1)

        R_final = torch.empty_like(A)
        R_final.scatter_(dim=1, index=S_indices, src=R_sorted)

        return R_final

        
    
    def patch_unique(self, map_patched: torch.Tensor):
        """
        パッチごとにユニークなIDを割り当てる。
        
        Args:
            map_patched (torch.Tensor): パッチ化されたマップテンソル (B*7396, 9, C)
        Returns:
            torch.Tensor: ユニークなIDが割り当てられたマップテンソル (B*7396, 9, C)
        """
        map_patched = map_patched.reshape(
            -1, map_patched.shape[2], map_patched.shape[3]
        )  # (B*7396, 9, C)

        _in = map_patched[:, :, 0]  # (B*7396, 9)

        uni = self.vectorized_row_unique_rank_by_appearance(_in) + 1  # (B*7396, 9)
        map_patched[:, :, 0] = _in.where(_in == 0, uni)
        return map_patched  # (B*7396, 9, C)

    def preprocess_obs(self, obs):
        """
        状態の前処理を行い、パッチ化されたマップテンソルを返す。
        Args:
            obs (torch.Tensor): 観測テンソル (B, 256, 256, C)
        Returns:
            torch.Tensor: パッチ化されたマップテンソル (B*7396, 9*C)
        """
        iter_in_mcs = int(obs[0, 0, 0, 0] - int(obs[0, 0, 0, 0] / 10**2) * 10**2)
        obs[0, 0, 0, 0] = (obs[0, 0, 0, 0] - iter_in_mcs) / 10**2
    
        # パッチに分割 (B, 256, 256, C) -> (B, 7396, 9, C)
        map_patched = extract_patches_manual_padding_with_offset_batch(
            obs, 3, 3, iter_in_mcs % 3, (iter_in_mcs // 3) % 3
        )

        # 相対的なIDに変換
        map_patched = self.patch_unique(map_patched) # (B*7396, 9, C)

        map_patched = map_patched.reshape(map_patched.shape[0], -1)  # (B*7396, 9*C)
        
        return map_patched  # (B*7396, 9*C)

class CPM_Actor(Actor):
    """
    CPM用のActorクラス。
    (B, 256, 256, C)の観測を受け取り、(B, N * D_act)の行動を出力する。
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )
        self.latent_pi = Actor_Net(observation_space, action_space)  # (B, 7396*4)
        
    def get_action_dist_params(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, self.features_extractor)
        mean_actions, log_std = self.latent_pi(features) # (B, 7396*4), (B, 7396*4)

        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        #print(f"mean_actions shape: {mean_actions.shape}, log_std shape: {log_std.shape}")
        return mean_actions, log_std, {}

class Critic_Net(nn.Module):
    """
    CPM用のCriticネットワーク。
    (B, 256, 256, C)の観測と(B, N * D_act)の行動を受け取り、(B, 1)のQ値を出力する。
    """
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box):
        super().__init__()
        self.u_net1 = U_Net.UNet(
            in_channels=observation_space.shape[2],
            out_channels=1,
            features=[64, 128, 256],
        )  # (B, 1, 256, 256)
        self.obs1 = nn.Linear(observation_space.shape[0] * observation_space.shape[1], 256)  # (B, 256)
        self.ac1 = nn.Linear(action_space.shape[0], 256)  # (B, 256)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x: th.Tensor, actions: th.Tensor) -> th.Tensor:
        x = self.u_net1(x.permute(0, 3, 1, 2))  # (B, 1, 256, 256)
        x = x.flatten(start_dim=1)  # (B, 1*256*256)
        x = th.relu(self.obs1(x))  # (B, 256)
        actions = th.relu(self.ac1(actions.reshape(x.shape[0], -1)))  # (B, 256)
        x = th.relu(self.fc1(th.cat([x, actions], dim=1)))  # (B, 512)
        x = th.relu(self.fc2(x))  # (B, 256)
        x = self.fc3(x)  # (B, 1)
        return x

class CPM_Critic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net = Critic_Net(observation_space, action_space)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        
        return tuple(q_net(features, actions) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](features, actions)


class CPM_SAC_Policy(SACPolicy):
    """
    CPMのための SAC (Soft Actor-Critic) ポリシー。
    CPM用のActorとCriticを使用し、マルチエージェント環境に対応。
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space, # アクターが直接扱う単一エージェントの行動空間
            lr_schedule=lr_schedule,
            features_extractor_class=ThroughExtractor,
            **kwargs,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CPM_Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CPM_Critic(**critic_kwargs).to(self.device)
