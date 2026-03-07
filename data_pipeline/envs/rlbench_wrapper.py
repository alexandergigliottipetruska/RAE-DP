"""RLBench environment wrapper for evaluation.

Wraps an RLBench Environment to implement the BaseManipulationEnv interface.
Extracts 4 camera views (front, left_shoulder, right_shoulder, wrist),
resizes to 224x224, applies ImageNet normalization.

The policy outputs delta-EE actions, but RLBench expects absolute EE poses.
This wrapper maintains internal EE state and accumulates deltas each step.

Gripper is thresholded at 0.5 to binary {0.0, 1.0}.

Requires CoppeliaSim (WSL2 or remote Linux).
"""

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from data_pipeline.envs.base_env import BaseManipulationEnv

# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Camera mapping: RLBench uses all 4 slots
# slot 0: front, slot 1: left_shoulder, slot 2: right_shoulder, slot 3: wrist
_CAMERA_ATTRS = {
    0: "front_rgb",
    1: "left_shoulder_rgb",
    2: "right_shoulder_rgb",
    3: "wrist_rgb",
}

# Task name -> RLBench task class name
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
}


def _process_image(img_hwc: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize + ImageNet normalize + HWC->CHW.

    Args:
        img_hwc: uint8 [H, W, 3] image from RLBench.

    Returns:
        float32 [3, H, W] ImageNet-normalized.
    """
    if img_hwc.shape[0] != target_size or img_hwc.shape[1] != target_size:
        img_hwc = np.array(
            Image.fromarray(img_hwc).resize(
                (target_size, target_size), Image.LANCZOS
            )
        )
    img_float = img_hwc.astype(np.float32) / 255.0
    normalized = (img_float - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.moveaxis(normalized, -1, -3)  # [3, H, W]


class RLBenchWrapper(BaseManipulationEnv):
    """Wraps an RLBench environment for policy evaluation.

    Accumulates delta-EE actions into absolute poses for the sim.

    Args:
        task_name: Task name (must be in TASK_CLASS_MAP).
        image_size: Target image size (default 224).
        headless: Run CoppeliaSim headless (default True).
    """

    def __init__(
        self,
        task_name: str,
        image_size: int = 224,
        headless: bool = True,
    ):
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig

        self._task_name = task_name
        self._image_size = image_size

        if task_name not in TASK_CLASS_MAP:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Supported: {list(TASK_CLASS_MAP.keys())}"
            )

        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(True)

        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete(),
        )

        self._env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=headless,
        )
        self._env.launch()

        import rlbench.tasks
        task_cls = getattr(rlbench.tasks, TASK_CLASS_MAP[task_name])
        self._task = self._env.get_task(task_cls)

        self._last_obs = None
        # Internal EE state for delta->absolute accumulation
        self._current_pos = None      # [3]
        self._current_rot = None      # Rotation object

    def reset(self) -> dict:
        descriptions, obs = self._task.reset()
        self._last_obs = obs

        # Initialize EE state from sim
        self._current_pos = np.array(obs.gripper_pose[:3], dtype=np.float64)
        self._current_rot = R.from_quat(obs.gripper_pose[3:])  # xyzw

        return {"descriptions": descriptions}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Execute one delta-EE action by accumulating to absolute pose.

        Args:
            action: [7] float32 — [delta_pos(3), delta_rotvec(3), gripper(1)]
        """
        delta_pos = action[:3].astype(np.float64)
        delta_rotvec = action[3:6].astype(np.float64)
        gripper_raw = float(action[6])

        # Accumulate deltas to absolute pose
        self._current_pos = self._current_pos + delta_pos
        delta_rot = R.from_rotvec(delta_rotvec)
        self._current_rot = delta_rot * self._current_rot  # world-frame

        # Threshold gripper to binary
        gripper = 1.0 if gripper_raw > 0.5 else 0.0

        # Build absolute action: [x, y, z, qx, qy, qz, qw, gripper]
        abs_quat = self._current_rot.as_quat()  # xyzw
        abs_action = np.concatenate([
            self._current_pos, abs_quat, [gripper]
        ])

        obs, reward, terminate = self._task.step(abs_action)
        self._last_obs = obs

        success = reward == 1.0
        return {}, float(reward), bool(terminate), {"success": success}

    def get_multiview_images(self) -> np.ndarray:
        """Return [1, 4, 3, 224, 224] with all 4 camera slots filled."""
        result = np.zeros(
            (1, 4, 3, self._image_size, self._image_size), dtype=np.float32
        )
        for slot, cam_attr in _CAMERA_ATTRS.items():
            img = getattr(self._last_obs, cam_attr, None)
            if img is not None:
                result[0, slot] = _process_image(img, self._image_size)
        return result

    def get_proprio(self) -> np.ndarray:
        """Return [1, 8] proprio: joint_positions(7) + gripper_open(1)."""
        obs = self._last_obs
        proprio = np.concatenate([
            np.array(obs.joint_positions, dtype=np.float32),  # [7]
            [float(obs.gripper_open)],                         # [1]
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
