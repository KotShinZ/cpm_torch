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

    def __init__(self, cpm_config: CPM_config, device_str: str = "cpu"):
        super().__init__()
        self.config = cpm_config
        self.device = torch.device(device_str)
        self.cpm_model = CPM(cpm_config, self.device)

        action_bound = 10.0  # Bound for dH adjustments (can be tuned)
        self.action_space = spaces.Box(
            low=-action_bound,
            high=action_bound,
            shape=(4,),
            dtype=np.float32,
        )
        bathch_size = int((self.cpm_model.map_tensor.shape[0] // 3 + 1)* (self.cpm_model.map_tensor.shape[0] // 3 + 1))
        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(bathch_size ,9, self.cpm_model.map_tensor.shape[2]),
            dtype=np.float32,
        )
        self.current_target_pixel_coords: Optional[Tuple[int, int]] = None
        self.current_step = 0
        
        self.iter_in_mcs = 0 # MCSのチェッカーボードのイテレーション数

    def get_obs(self) -> np.ndarray:
        # (batch_size, 9, 3)のテンソルを返す
        return self.cpm_model.get_map_patched(self.iter_in_mcs % 3,  (self.iter_in_mcs // 3) % 3)

    def get_reward(self) -> float:
        # 報酬は定数
        target = torch.zeros_like(self.cpm_model.map_tensor)
        half = target.shape[0] // 2
        target[:half, :] = 1.0
        reward = torch.sum(self.cpm_model.map_tensor * target)
        return float(reward) / 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cpm_model.reset()
        for x in range(10):
            for y in range(10):
                self.cpm_model.add_cell(x + 122, y + 122)
        self.current_step = 0
        return self.get_obs(), {}

    def step(self, action: np.ndarray):
        # actionは(batch_size, 4)のテンソル
        # 予測したニューラルハミルトニアンを使って、1step進める
        self.cpm_model.cpm_checkerboard_step(self.iter_in_mcs % 3, self.iter_in_mcs // 3, dH_NN = action)
        self.iter_in_mcs += 1
        
        observation = self.get_obs()
        reward = self.get_reward()
        terminated = self.current_step >= 1000
        truncated = False  # Not using truncation based on time limit separately here
        
        self.current_step += 1

        return observation, reward, terminated, truncated, {}

    def render(self, mode="ansi"):
        imshow_map(self.cpm_model.map_tensor)

    def close(self):
        pass  # Cleanup if needed
