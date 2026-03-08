"""Custom RLBench arm action mode with OMPL motion planning.

Matches the approach used by RVT-2 (Goyal et al.) and Chain-of-Action
(Zhang et al.): OMPL-based path planning for absolute EE pose targets.

RVT-2 adds workspace bounds clipping to prevent out-of-bounds errors.
CoA adds BiTRRT planner with retries and success checking during path
execution.  We combine both ideas into a single clean action mode.

Why OMPL planning (not Jacobian IK):
- OMPL is a global planner that handles large pose jumps
- Jacobian IK is local and fails for targets far from current pose
- All major RLBench papers (PerAct, RVT, RVT-2, CoA, Act3D) use
  EndEffectorPoseViaPlanning for evaluation
"""

import numpy as np
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning


class EndEffectorPoseViaPlanning2(EndEffectorPoseViaPlanning):
    """OMPL-based EE pose planning with workspace clipping.

    Extends RLBench's stock EndEffectorPoseViaPlanning to clip target
    positions to workspace bounds (from RVT-2). Uses absolute mode so
    the action IS the target EE pose, not a delta.

    Action shape: (7,) — [x, y, z, qx, qy, qz, qw]
    """

    def __init__(self):
        super().__init__(absolute_mode=True)

    def action(self, scene, action: np.ndarray, ignore_collisions: bool = True):
        # Clip target position to workspace bounds (from RVT-2)
        action[:3] = np.clip(
            action[:3],
            np.array([
                scene._workspace_minx,
                scene._workspace_miny,
                scene._workspace_minz,
            ]) + 1e-7,
            np.array([
                scene._workspace_maxx,
                scene._workspace_maxy,
                scene._workspace_maxz,
            ]) - 1e-7,
        )
        super().action(scene, action, ignore_collisions)
