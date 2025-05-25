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
        """
        ベクトル化された手法で各行の出現順ランクを計算する関数。
        ランクは0から始まる整数です。同じ値は同じランクになります。
        例: x = torch.tensor([[10, 20, 10, 0, 20]])
            ranks = self.vectorized_row_unique_rank_by_appearance(x)
            # ranks は [[0, 1, 0, 2, 1]] となる (値10がランク0, 値20がランク1, 値0がランク2)
        """
        B, N = x.shape
        device = x.device

        if B == 0:
            return torch.empty((0, N), dtype=torch.int64, device=device)
        if N == 0:
            return torch.empty((B, 0), dtype=torch.int64, device=device)

        cols_broadcast = torch.arange(N, device=device).view(1, 1, N).expand(B, N, N)
        eq_mask = x.unsqueeze(2) == x.unsqueeze(1)

        sentinel = N 
        masked_cols = torch.where(eq_mask, cols_broadcast, sentinel)
        mci = masked_cols.min(dim=2).values

        A = mci
        S_values, S_indices = A.sort(dim=1)

        R_sorted = torch.zeros_like(A, dtype=torch.int64)
        if N > 1:
            R_sorted[:, 1:] = (S_values[:, 1:] != S_values[:, :-1]).cumsum(dim=1)

        R_final = torch.empty_like(A, dtype=torch.int64)
        R_final.scatter_(dim=1, index=S_indices, src=R_sorted)
        return R_final

    def patch_unique(self, map_patched: torch.Tensor) -> torch.Tensor:
        """
        パッチごとに新しいルールでユニークなIDを割り当てる。
        ルールは前回の説明通り（同じ値には同じID、中央値優先など）。
        """
        original_shape = map_patched.shape
        original_ndim = map_patched.ndim
        
        n_elements_in_patch = 0 

        if original_ndim == 4: 
            num_items = original_shape[0] * original_shape[1]
            n_elements_in_patch = original_shape[2]
            num_channels = original_shape[3]
            reshaped_map_patched = map_patched.reshape(num_items, n_elements_in_patch, num_channels)
        elif original_ndim == 3: 
            num_items = original_shape[0]
            n_elements_in_patch = original_shape[1]
            num_channels = original_shape[2]
            reshaped_map_patched = map_patched
        elif original_ndim == 2 and map_patched.is_contiguous(): 
            num_items = original_shape[0]
            n_elements_in_patch = original_shape[1]
            num_channels = 1 
            reshaped_map_patched = map_patched.unsqueeze(-1)
        else:
            if torch.numel(map_patched) == 0 : 
                return map_patched 
            raise ValueError(
                f"Input tensor map_patched has an unsupported shape: {original_shape}. "
                "Expected 2D (ITEMS, N), 3D (ITEMS, N, C), or 4D (B, P, N, C)."
            )

        _in_data = reshaped_map_patched[:, :, 0].clone() 
        
        B, N = _in_data.shape 
        device = _in_data.device
        
        if B == 0 or N == 0:
            if original_ndim == 2 and num_channels == 1: 
                return reshaped_map_patched.squeeze(-1)
            return reshaped_map_patched 

        R_all_values_rank = self.vectorized_row_unique_rank_by_appearance(_in_data)

        final_ids = torch.zeros_like(_in_data, dtype=torch.int64, device=device)
        center_idx = N // 2
        
        center_value_per_row = _in_data[:, center_idx].unsqueeze(1) 
        is_center_value_nonzero = (center_value_per_row != 0)      
        next_base_id_for_others = torch.where(is_center_value_nonzero,
                                           torch.tensor(2, dtype=torch.int64, device=device),
                                           torch.tensor(1, dtype=torch.int64, device=device)) 

        rank_of_center_value = R_all_values_rank[:, center_idx].unsqueeze(1)
        
        placeholder_rank_for_missing_zero = R_all_values_rank.max().item() + 1 
        
        R_masked_for_zero_val_rank = torch.where(_in_data == 0, R_all_values_rank, placeholder_rank_for_missing_zero)
        rank_of_zero_value, _ = R_masked_for_zero_val_rank.min(dim=1, keepdim=True) 
        zero_value_exists = (rank_of_zero_value < placeholder_rank_for_missing_zero) 

        shift_amount = torch.zeros_like(R_all_values_rank, dtype=torch.int64) 
        
        shift_due_to_zero = (R_all_values_rank > rank_of_zero_value) & zero_value_exists
        shift_amount += shift_due_to_zero.long()
        
        shift_due_to_center_value = (R_all_values_rank > rank_of_center_value) & is_center_value_nonzero
        shift_amount += shift_due_to_center_value.long()

        adjusted_rank_for_others = R_all_values_rank - shift_amount
        ids_candidate_for_others = adjusted_rank_for_others + next_base_id_for_others

        final_ids = ids_candidate_for_others
        
        mask_value_is_center_value = (_in_data == center_value_per_row) 
        final_ids = torch.where(mask_value_is_center_value & is_center_value_nonzero,
                                torch.tensor(1, dtype=torch.int64, device=device),
                                final_ids)
        
        final_ids[_in_data == 0] = 0 
        
        reshaped_map_patched[:, :, 0] = final_ids.to(reshaped_map_patched.dtype)
        
        if original_ndim == 2 and num_channels == 1 and map_patched.is_contiguous():
             return reshaped_map_patched.squeeze(-1) 
        
        return reshaped_map_patched

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
