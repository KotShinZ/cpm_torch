from cpm_torch.CPM import *
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Tuple, List, Optional
from cpm_torch.CPM_Image import *

import torchvision
import torchvision.transforms as transforms


# --- RL Environment ---
class CPMEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, cpm_config: CPM_config, device: str = "cuda"):
        super().__init__()
        self.config = cpm_config
        self.device = torch.device(device)
        self.cpm_model = CPM(cpm_config, self.device)

        self.W, self.H, self.C = self.cpm_model.map_tensor.shape  # (256, 256, C)

        self.action_bound = 50.0  # dHの最大値
        self.bathch_size = int((self.W // 3 + 1) * (self.H // 3 + 1))
        self.action_space = spaces.Box(
            low=-self.action_bound,
            high=self.action_bound,
            shape=(
                self.bathch_size,
                4,
            ),  # (7396, 4)
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=self.cpm_model.map_tensor.shape,  # (256, 256, C)
            dtype=np.float32,
        )
        self.current_target_pixel_coords: Optional[Tuple[int, int]] = None
        self.current_step = 0
        self.current_episode = 0

        self.iter_in_mcs = 0  # MCSのチェッカーボードのイテレーション数

        self.reward_func = self.get_reward_image  # デフォルトの報酬関数を設定

        self.image_tensor = self.get_image_tensor_line()  # 報酬画像を取得

    # region　画像取得関数

    def get_image_tensor_line(self):
        image = torch.zeros((self.W, self.H), dtype=torch.float32, device=self.device)
        image[:, self.H // 2] = 1.0  # 中央の行に1.0を設定
        imshow_map_area(
            image.unsqueeze(2), target_channel=0, _max=1
        )  # デバッグ用に表示
        return image

    def get_image_tensor_mnist(self):
        """MNISTの画像テンソルを取得するメソッド。

        Returns:
            torch.Tensor: 64x64の画像テンソル。
        """
        transform = transforms.Compose(
            [
                transforms.Resize((self.W, self.H)),  # 画像を64x64にリサイズ
                transforms.ToTensor(),  # PILImageまたはnumpy.ndarrayをテンソルに変換
            ]
        )
        trainset = torchvision.datasets.MNIST(
            root="../data", train=True, download=True, transform=transform
        )
        image_of_2 = None
        for image, label in trainset:
            if label == 2:
                image_of_2 = image
                break  # 最初の「2」の画像が見つかったらループを終了
        image_tensor = image_of_2[0].to(self.device)  # (64, 64)のテンソル
        imshow_map_area(
            image_tensor.unsqueeze(2), target_channel=0, _max=1
        )  # デバッグ用に表示
        return image_tensor

    # endregion

    # region 報酬関数

    def get_reward_image(self) -> float:
        """image_tensorに近いほど大きい報酬を返す。"""
        ids = self.cpm_model.map_tensor[:, :, 0]  # (64, 64)
        rows, cols = ids.shape
        return torch.sum((ids > 0).to(torch.float32) * self.image_tensor) / (
            rows * cols
        )

    def get_reward_direction(self) -> float:
        """
        右にいるほど大きい報酬を返す。
        0~1の範囲で、右端にいると1、左端にいると0の報酬を返す。
        """
        ids = self.cpm_model.map_tensor[:, :, 0]
        rows, cols = ids.shape

        row_weights = (
            torch.arange(rows, 0, -1, dtype=torch.float32, device=ids.device) / rows
        )

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

    # endregion

    def get_obs(self):
        # (256, 256, C)のテンソルを返す
        obs = self.cpm_model.map_tensor.clone()
        obs[0, 0, 0] = obs[0, 0, 0] * 10**2 + self.iter_in_mcs % 9
        if self.image_tensor is not None:
            obs[:, :, 1] = self.image_tensor #　画像テンソルを状態に入れる
        return obs.to("cpu")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cpm_model.reset()
        center = self.W // 2
        for x in range(-2, 3):
            for y in range(-2, 3):
                self.cpm_model.add_cell(x * 15 + center, y * 15 + center)
        self.current_step = 0
        return self.get_obs(), {}

    def step(self, action):
        # action:(7396, 4)
        # 予測したニューラルハミルトニアンを使って、1step進める
        if action is not torch.Tensor:
            action = torch.tensor(action, dtype=torch.float32, device=self.device)

        self.cpm_model.cpm_checkerboard_step(
            self.iter_in_mcs % 3, (self.iter_in_mcs // 3) % 3, dH_NN=action
        )
        self.iter_in_mcs += 1

        observation = self.get_obs()  # (256, 256, C)
        reward = self.reward_func()
        terminated = self.current_step >= 500
        truncated = False  # Not using truncation based on time limit separately here

        self.current_step += 1

        if terminated:
            self.current_step = 0
            self.current_episode += 1
            if self.current_episode % 3 == 0:
                self.render()

        return (
            observation,
            reward,
            terminated,
            truncated,
            {"iter_in_mcs": self.iter_in_mcs},
        )

    def render(self, mode="ansi"):
        imshow_map(self.cpm_model.map_tensor)

    def close(self):
        pass  # Cleanup if needed
