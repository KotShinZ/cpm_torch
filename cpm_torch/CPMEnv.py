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

        action_bound = 30.0  # Bound for dH adjustments (can be tuned)
        bathch_size = int((self.cpm_model.map_tensor.shape[0] // 3 + 1)* (self.cpm_model.map_tensor.shape[0] // 3 + 1))
        self.action_space = spaces.Box(
            low=-action_bound,
            high=action_bound,
            shape=(bathch_size * 4,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=self.cpm_model.map_tensor.shape,
            dtype=np.float32,
        )
        self.current_target_pixel_coords: Optional[Tuple[int, int]] = None
        self.current_step = 0
        
        self.iter_in_mcs = 0 # MCSのチェッカーボードのイテレーション数

    def get_obs(self):
        # (256, 256, C)のテンソルを返す
        obs = self.cpm_model.map_tensor.clone()
        obs[0, 0, 0] = obs[0, 0, 0] * 10**2 + self.iter_in_mcs % 9
        return obs

    def get_reward(self) -> float:
        # (1)
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
        # action:(7396, 4)
        # 予測したニューラルハミルトニアンを使って、1step進める
        self.cpm_model.cpm_checkerboard_step(self.iter_in_mcs % 3, (self.iter_in_mcs // 3) % 3, dH_NN = action.reshape(-1, 4))
        self.iter_in_mcs += 1
        
        observation = self.get_obs()
        reward = self.get_reward()
        terminated = self.current_step >= 1000
        truncated = False  # Not using truncation based on time limit separately here
        
        self.current_step += 1
        
        if terminated:
            self.render()

        return observation, reward, terminated, truncated, {"iter_in_mcs": self.iter_in_mcs}

    def render(self, mode="ansi"):
        imshow_map(self.cpm_model.map_tensor)

    def close(self):
        pass  # Cleanup if needed
