import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from functools import partial
import numpy as np
import cpm_torch.Training.U_Net as U_Net
from cpm_torch.CPM_Map import extract_patches_manual_padding_with_offset_batch


class CPMPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.SiLU,
        log_std_init=0,
        *args,
        **kwargs
    ):
        super(CPMPolicy, self).__init__(
            observation_space,  # (256, 256, C)
            action_space,  # (N, 4)
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )
        # デフォルトのネットワーク構造を設定
        if net_arch is None:
            self.net_arch = dict(pi=[128, 128], vf=[128, 128])

        self.log_std_init = log_std_init  # 初期ログ分散（std=0.5に対応）
        self._build(lr_schedule)

    def _build(self, lr_schedule):
        # MLP抽出器を構築

        # 行動ネットワーク
        # (B*7396, 9*C) -> (B*7396, 4)
        self.policy_net = nn.Sequential(
            nn.Linear(9 * self.observation_space.shape[2], 128),  # (B*7396, 128)
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
        )  # (B*7396, 4)

        # クリティック（価値関数）ネットワーク
        # (B, C, 256, 256) -> (B, 1)
        self.value_net = nn.Sequential(
            U_Net.UNet(
                in_channels=self.observation_space.shape[2],
                out_channels=1,
                features=[64, 128, 256],
            ),  # (B, 1, 256, 256)
            nn.Flatten(),  # (B, 1*256*256)
            nn.Linear(self.observation_space.shape[0] * self.observation_space.shape[1], 32),  # (B, 128)
            nn.SiLU(),
            nn.Linear(32, 1),  # (B, 1)
        )
        # (B, 1)
        # アクション分布のためのネットワークを構築
        _, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=self.action_space.shape[0], log_std_init=self.log_std_init
        )
        del _

        # オプティマイザを設定
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

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
        map_patched = map_patched.reshape(
            -1, map_patched.shape[2], map_patched.shape[3]
        )  # (B*7396, 9, C)

        _in = map_patched[:, :, 0]  # (B*7396, 9)

        uni = self.vectorized_row_unique_rank_by_appearance(_in) + 1  # (B*7396, 9)
        map_patched[:, :, 0] = _in.where(_in == 0, uni)
        return map_patched  # (B*7396, 9, C)

    def preprocess_obs(self, obs):
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

    def forward(self, obs, deterministic=False):
        # iter_in_mcsは
        map_patched = self.preprocess_obs(obs)  # (B*7396, 9, C)
        policy_features = self.policy_net(map_patched)  # (B*7396, 4)
        policy_features = policy_features.reshape(obs.shape[0], -1)  # (B, 7396*4)

        # 価値関数の計算 (B, 256, 256, C) -> (B, 1)
        values = self.predict_values(obs)  # (B, 1)

        # 正規分布からlog確率を取得
        distribution = self._get_action_dist_from_latent(policy_features)
        actions = distribution.get_actions(deterministic=deterministic)  # (B, 7396*4)
        log_prob = distribution.log_prob(actions)  # (B, 7396*4)

        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi):
        # mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(latent_pi, self.log_std)

    def evaluate_actions(self, obs, actions):
        map_patched = self.preprocess_obs(obs)
        policy_features = self.policy_net(map_patched)  # (B*7396, 4)
        policy_features = policy_features.reshape(obs.shape[0], -1)  # (B, 7396*4)
        values = self.predict_values(obs)  # (B, 1)

        distribution = self._get_action_dist_from_latent(policy_features)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def extract_features(self, obs):
        return obs

    def predict_values(self, obs):
        return self.value_net(obs.permute(0, 3, 1, 2))  # (B, 1)
