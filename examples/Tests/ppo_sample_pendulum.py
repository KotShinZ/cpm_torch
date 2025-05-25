import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from functools import partial
import numpy as np

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh, log_std_init=-0.693, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            observation_space,
            action_space,
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
        
        self.policy_net = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )

        # クリティック（価値関数）ネットワーク
        self.value_net = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # アクション分布のためのネットワークを構築
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=128, log_std_init=self.log_std_init
        )

        # オプティマイザを設定
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs, deterministic=False):
        # 特徴抽出

        # 価値関数の計算
        policy_features = self.policy_net(obs)
        values = self.value_net(obs)

        # アクション分布の生成
        distribution = self._get_action_dist_from_latent(policy_features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def evaluate_actions(self, obs, actions):
        policy_features = self.policy_net(obs)
        values = self.value_net(obs)
        
        distribution = self._get_action_dist_from_latent(policy_features)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def extract_features(self, obs):
        return obs

    def predict_values(self, obs):
        return self.value_net(obs)

if __name__ == "__main__":
    from stable_baselines3 import PPO
    import gymnasium as gym

    # 環境の作成
    env = gym.make("Pendulum-v1")

    # PPOモデルの作成
    model = PPO(
        policy=CustomPolicy,
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )
    print(model.policy)
    # 学習の実行
    model.learn(total_timesteps=1000000)