"""Gym-compatible wrapper for RobomimicWrapper.

Adapts our RobomimicWrapper to the gym.Env interface required by
AsyncVectorEnv. Returns observation dict from reset() and step()
matching Chi's obs format.

Observation dict keys (matching Chi's shape_meta):
  agentview_image:          (3, 84, 84) float32 [0, 1]
  robot0_eye_in_hand_image: (3, 84, 84) float32 [0, 1]
  robot0_eef_pos:           (3,) float32
  robot0_eef_quat:          (4,) float32
  robot0_gripper_qpos:      (2,) float32
"""

import gym
from gym import spaces
import numpy as np

from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper


class RobomimicGymWrapper(gym.Env):
    """Gym-compatible wrapper around RobomimicWrapper.

    Args:
        task:       Task name (e.g. 'lift').
        abs_action: Use absolute actions.
        image_size: Render size for images.
        seed:       Initial seed (optional).
    """

    def __init__(self, task="lift", abs_action=True, image_size=84, seed=None):
        super().__init__()
        self._env = RobomimicWrapper(
            task=task, abs_action=abs_action, image_size=image_size, seed=seed,
        )
        self._image_size = image_size

        # Define gym spaces matching Chi's shape_meta
        self.observation_space = spaces.Dict({
            "agentview_image": spaces.Box(0, 1, (3, image_size, image_size), dtype=np.float32),
            "robot0_eye_in_hand_image": spaces.Box(0, 1, (3, image_size, image_size), dtype=np.float32),
            "robot0_eef_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            "robot0_eef_quat": spaces.Box(-1, 1, (4,), dtype=np.float32),
            "robot0_gripper_qpos": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
        })
        # 7D absolute action: pos(3) + axis_angle(3) + gripper(1)
        self.action_space = spaces.Box(-10, 10, (7,), dtype=np.float32)

    def _make_obs(self):
        """Convert RobomimicWrapper's outputs to flat obs dict."""
        # Images: (1, 4, 3, H, W) → extract slots 0 and 3
        images = self._env.get_multiview_images()  # (1, 4, 3, H, W) float32 [0,1]
        # Proprio: (1, 9) → split into eef_pos, eef_quat, gripper_qpos
        proprio = self._env.get_proprio()  # (1, 9) float32

        return {
            "agentview_image": images[0, 0].astype(np.float32),           # (3, H, W)
            "robot0_eye_in_hand_image": images[0, 3].astype(np.float32),  # (3, H, W)
            "robot0_eef_pos": proprio[0, :3].astype(np.float32),          # (3,)
            "robot0_eef_quat": proprio[0, 3:7].astype(np.float32),        # (4,)
            "robot0_gripper_qpos": proprio[0, 7:9].astype(np.float32),    # (2,)
        }

    def seed(self, seed):
        self._env.seed(seed)

    def reset(self):
        self._env.reset()
        return self._make_obs()

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return self._make_obs(), reward, done, info

    def close(self):
        self._env.close()

    def is_success(self):
        return self._env._env._check_success()
