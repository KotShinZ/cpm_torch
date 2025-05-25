from cpm_torch.CPM import *
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Tuple, List, Optional
from cpm_torch.CPM_Image import *

# --- RL Environment ---
class CPMEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, cpm_config: CPM_config, device: str = "cuda"):
        super().__init__()
        self.config = cpm_config
        self.device = torch.device(device)
        self.cpm_model = CPM(cpm_config, self.device)

        action_bound = 50.0  # Bound for dH adjustments (can be tuned)
        bathch_size = int((self.cpm_model.map_tensor.shape[0] // 3 + 1)* (self.cpm_model.map_tensor.shape[0] // 3 + 1))
        self.action_space = spaces.Box(
            low=-action_bound,
            high=action_bound,
            shape=(bathch_size * 4,), # (7396, 4)
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=self.cpm_model.map_tensor.shape, # (256, 256, C)
            dtype=np.float32,
        )
        self.current_target_pixel_coords: Optional[Tuple[int, int]] = None
        self.current_step = 0
        self.current_episode = 0
        
        self.iter_in_mcs = 0 # MCSのチェッカーボードのイテレーション数

    def get_obs(self):
        # (256, 256, C)のテンソルを返す
        obs = self.cpm_model.map_tensor.clone()
        obs[0, 0, 0] = obs[0, 0, 0] * 10**2 + self.iter_in_mcs % 9
        return obs.to("cpu")

    def get_reward_direction(self) -> float:
        """
        右にいるほど大きい報酬を返す。
        0~1の範囲で、右端にいると1、左端にいると0の報酬を返す。
        """
        ids = self.cpm_model.map_tensor[:, :, 0]
        rows, cols = ids.shape

        row_weights = torch.arange(rows, 0, -1, dtype=torch.float32, device=ids.device) / rows 
        
        weights_per_row = row_weights.unsqueeze(1).expand(-1, cols)
        one_tensor = torch.clip(ids, 0, 1)

        weighted_map = one_tensor * weights_per_row
        
        # imshow_map_area(weighted_map.unsqueeze(2), target_channel=0, _max=1) # デバッグ用に表示する場合は適宜実装してください

        reward = torch.sum(weighted_map)

        # 5. 元のコードのスケール調整を維持
        return float(reward) / 1000
    
    def get_reward_half(self) -> float:
        """
        上半分にいると1, 下半分にいると0の報酬を返す。
        """
        ids = self.cpm_model.map_tensor[:, :, 0]
        rows, cols = ids.shape

        # 上半分の行数
        half_rows = rows // 2

        # 上半分のIDを取得
        upper_half_ids = ids[:half_rows, :]

        return torch.sum((upper_half_ids > 0).to(torch.float32)) / (half_rows * cols)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cpm_model.reset()
        center = self.cpm_model.map_tensor.shape[0] // 2
        for x in range(0, 1):
            for y in range(0, 1):
                self.cpm_model.add_cell(x * 5 + center, y * 5 + center)
        self.current_step = 0
        return self.get_obs(), {}

    def step(self, action: np.ndarray):
        # action:(7396, 4)
        # 予測したニューラルハミルトニアンを使って、1step進める
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.cpm_model.cpm_checkerboard_step(self.iter_in_mcs % 3, (self.iter_in_mcs // 3) % 3, dH_NN = action.reshape(-1, 4))
        self.iter_in_mcs += 1
        
        observation = self.get_obs()  # (256, 256, C)
        reward = self.get_reward_half()
        terminated = self.current_step == 2000
        truncated = False  # Not using truncation based on time limit separately here
        
        self.current_step += 1
        
        if terminated:
            self.current_episode += 1
            if self.current_episode % 3 == 0:
                self.render()

        return observation, reward, terminated, truncated, {"iter_in_mcs": self.iter_in_mcs}

    def render(self, mode="ansi"):
        imshow_map(self.cpm_model.map_tensor)

    def close(self):
        pass  # Cleanup if needed
