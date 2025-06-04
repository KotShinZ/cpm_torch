from cpm_torch.CPM import *
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Tuple, List, Optional
from cpm_torch.CPM_Image import *
from cpm_torch.CPMEnv import CPMEnv
from cpm_torch.Diffusion import *


# --- RL Environment ---
class CPMDiffusionEnv(CPMEnv):
    def __init__(self, cpm_config: CPM_config, device: str = "cuda"):
        super().__init__(cpm_config, device)

        self.action_space = spaces.Box(
            low=-self.action_bound,
            high=self.action_bound,
            shape=(
                self.bathch_size,
                (4 + cpm_config.other_channels),
            ),  # (7396, 4)
            dtype=np.float32,
        )
        
    def step(self, action):
        # action:(7396, 4)
        # 予測したニューラルハミルトニアンを使って、1step進める
        if action is not torch.Tensor:
            action = torch.tensor(action, dtype=torch.float32, device=self.device)

        action_tensor = torch.zeros(
            action.shape[0], 9, self.config.other_channels, device=self.device
        )
        action_tensor[:, 4, :] = action[:, 4:]  # (486, 9, 1)

        # 分子の生成
        self.cpm_model.map_tensor[
            :, :, self.config.diffusion_channels
        ] += reconstruct_image_from_patches(
            action_tensor,
            (
                self.observation_space.shape[0],
                self.observation_space.shape[1],
                self.config.other_channels,
            ),
            3,
            3,
            self.iter_in_mcs % 3,
            (self.iter_in_mcs // 3) % 3,
        )

        # 分子の数の制限
        self.cpm_model.map_tensor[:, :, self.config.diffusion_channels] = torch.clip(
            self.cpm_model.map_tensor[:, :, self.config.diffusion_channels], 0, 1
        )

        # 分子の拡散
        for _ in range(100):
            self.cpm_model.map_tensor = diffusion_step(
                self.cpm_model.map_tensor,
                params=self.config.diffusion_channels,
                dts=self.config.diffusion_D,
                percent=self.config.diffusion_percent,
            )

        return super().step(action[:, :4])

    def reset(self, seed=None, options=None):
        # CPMEnvのresetを呼び出して、マップを初期化
        # ランダムなターゲット位置
        self.target_pos = torch.randint(
            0, self.cpm_model.map_tensor.shape[0], (2,), device=self.device
        )
        observation, info = super().reset(seed=seed, options=options)

        return observation, info

    def get_reward_target(self) -> float:
        """
        ターゲット位置に近いほど大きい報酬を返す。
        ターゲット位置との距離に基づいて報酬を計算する。
        """
        ids = self.cpm_model.map_tensor[:, :, 0]
        rows, cols = ids.shape

        target_row, target_col = self.target_pos
        # ターゲット位置からの距離を計算
        distance_map = torch.sqrt(
            (torch.arange(rows, device=ids.device).unsqueeze(1) - target_row) ** 2
            + (torch.arange(cols, device=ids.device).unsqueeze(0) - target_col) ** 2
        )
        # 距離を0~1の範囲に正規化
        max_distance = torch.sqrt((rows - 1) ** 2 + (cols - 1) ** 2)
        normalized_distance = distance_map / max_distance
        imshow_map_area(normalized_distance.unsqueeze(2), target_channel=0, _max=1)
        # 距離が小さいほど報酬が大きくなるように計算
        reward = 1.0 - normalized_distance[target_row, target_col]
        # 5. 元のコードのスケール調整を維持
        reward = float(reward) / 1000.0

        return reward

    def render(self, mode="ansi", channel=[]):
        for c in channel:
            imshow_map_area_autoRange(
                self.cpm_model.map_tensor,
                target_channel=c,
            )

        super().render(mode)
