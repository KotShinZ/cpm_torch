import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ThroughExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, observation_space.shape[0] * observation_space.shape[1] * observation_space.shape[2])

    def forward(self, observations):
        return observations