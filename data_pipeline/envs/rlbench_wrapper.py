"""RLBench environment wrapper for evaluation.

Uses JointPosition (absolute) + Discrete gripper control.
Actions are 8D: [joint_targets(7), gripper(1)].
Gripper: 0=closed, 1=open (raw RLBench Discrete format).

Proprio is [joint_positions(7), gripper_centered(1)] = 8D.
Gripper centered: {0,1} → {-1,+1} for training symmetry.

Validated via GT replay: 100% success on open_drawer with
joint_position_action from stepjam RLBench demos.
"""

import numpy as np
from PIL import Image

from data_pipeline.envs.base_env import BaseManipulationEnv

_CAMERA_ATTRS = {
    0: "front_rgb",
    1: "left_shoulder_rgb",
    2: "right_shoulder_rgb",
    3: "wrist_rgb",
}

TASK_CLASS_MAP = {
    "close_jar": "CloseJar",
    "open_drawer": "OpenDrawer",
    "slide_block_to_color_target": "SlideBlockToColorTarget",
    "put_item_in_drawer": "PutItemInDrawer",
    "stack_cups": "StackCups",
    "place_shape_in_shape_sorter": "PlaceShapeInShapeSorter",
    "meat_off_grill": "MeatOffGrill",
    "turn_tap": "TurnTap",
    "push_buttons": "PushButtons",
    "reach_and_drag": "ReachAndDrag",
    "place_wine_at_rack_location": "PlaceWineAtRackLocation",
    "sweep_to_dustpan_of_size": "SweepToDustpanOfSize",
}


def _process_image(img_hwc: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize uint8 HWC → float32 CHW [0,1]."""
    if img_hwc.shape[0] != target_size or img_hwc.shape[1] != target_size:
        img_hwc = np.array(
            Image.fromarray(img_hwc).resize(
                (target_size, target_size), Image.LANCZOS
            )
        )
    return np.moveaxis(img_hwc.astype(np.float32) / 255.0, -1, -3)


class RLBenchWrapper(BaseManipulationEnv):
    """RLBench eval wrapper using JointPosition control."""

    def __init__(self, task_name: str, image_size: int = 224,
                 headless: bool = True, cameras: bool = True):
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointPosition
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig

        self._task_name = task_name
        self._image_size = image_size

        if task_name not in TASK_CLASS_MAP:
            raise ValueError(f"Unknown task: {task_name}. "
                             f"Supported: {list(TASK_CLASS_MAP.keys())}")

        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(cameras)

        self._env = Environment(
            action_mode=MoveArmThenGripper(JointPosition(), Discrete()),
            obs_config=obs_config,
            headless=headless,
        )
        self._env.launch()

        import rlbench.tasks
        task_cls = getattr(rlbench.tasks, TASK_CLASS_MAP[task_name])
        self._task = self._env.get_task(task_cls)
        self._last_obs = None

    def seed(self, seed: int) -> None:
        np.random.seed(seed)

    def reset(self) -> dict:
        descriptions, obs = self._task.reset()
        self._last_obs = obs
        return {"descriptions": descriptions}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Execute one step. Action: [joint_targets(7), gripper(1)]."""
        joint_targets = action[:7].astype(np.float64)
        gripper = 1.0 if float(action[7]) > 0.5 else 0.0
        rlbench_action = np.concatenate([joint_targets, [gripper]])

        try:
            obs, reward, terminate = self._task.step(rlbench_action)
        except Exception as e:
            return {}, 0.0, True, {"success": False, "error": str(e)}
        self._last_obs = obs
        return {}, float(reward), bool(terminate), {"success": reward > 0}

    def get_multiview_images(self) -> np.ndarray:
        result = np.zeros(
            (1, 4, 3, self._image_size, self._image_size), dtype=np.float32
        )
        for slot, cam_attr in _CAMERA_ATTRS.items():
            img = getattr(self._last_obs, cam_attr, None)
            if img is not None:
                result[0, slot] = _process_image(img, self._image_size)
        return result

    def get_proprio(self) -> np.ndarray:
        """Return [1, 8]: joint_positions(7) + gripper_centered(1)."""
        obs = self._last_obs
        proprio = np.concatenate([
            obs.joint_positions.astype(np.float32),
            [float(obs.gripper_open) * 2 - 1],
        ])
        return proprio.reshape(1, -1)

    def get_view_present(self) -> np.ndarray:
        return np.array([True, True, True, True])

    def close(self) -> None:
        self._env.shutdown()

    @property
    def proprio_dim(self) -> int:
        return 8

    @property
    def num_cameras(self) -> int:
        return 4
