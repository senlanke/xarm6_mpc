import ctypes
import os
import sys

# Avoid pulling ROS site-packages (can shadow pinocchio/crocoddyl from conda env).
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

# Preload conda libstdc++ to satisfy crocoddyl CXXABI requirements.
conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    libstdcpp = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
    if os.path.exists(libstdcpp):
        ctypes.CDLL(libstdcpp, mode=ctypes.RTLD_GLOBAL)

import crocoddyl
import numpy as np
import pinocchio as pin

from mpc.utils.action import DAM_fwd_example, DAM_fwd_exampleT


class solver_ddp_reach_xarm6:
    def __init__(
        self,
        urdf_path,
        ee_frame_name="link6",
        T=50,
        DT=0.01,
        w_goal_running=1e4,
        w_goal_terminal=1e7,
        w_state=1e-1,
        w_ctrl=1e-5,
    ):
        self.robot_model = pin.buildModelFromUrdf(urdf_path)
        self.state = crocoddyl.StateMultibody(self.robot_model)
        self.T = T
        self.DT = DT

        self.ee_frame_id = self.robot_model.getFrameId(ee_frame_name)
        if self.ee_frame_id >= len(self.robot_model.frames):
            raise ValueError(f"End-effector frame not found: {ee_frame_name}")

        self.w_goal_running = w_goal_running
        self.w_goal_terminal = w_goal_terminal
        self.w_state = w_state
        self.w_ctrl = w_ctrl

    def generate_ddp(self, x0, x_ref, max_iter=100):
        x_ref = np.asarray(x_ref).reshape(3)

        frame_translation_residual = crocoddyl.ResidualModelFrameTranslation(
            self.state, self.ee_frame_id, x_ref
        )
        goal_tracking_cost = crocoddyl.CostModelResidual(
            self.state, frame_translation_residual
        )
        x_reg_cost = crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelState(self.state)
        )
        u_reg_cost = crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelControl(self.state)
        )

        running_cost_model = crocoddyl.CostModelSum(self.state)
        terminal_cost_model = crocoddyl.CostModelSum(self.state)

        running_cost_model.addCost("ee_track", goal_tracking_cost, self.w_goal_running)
        running_cost_model.addCost("state_reg", x_reg_cost, self.w_state)
        running_cost_model.addCost("ctrl_reg", u_reg_cost, self.w_ctrl)

        terminal_cost_model.addCost("ee_track", goal_tracking_cost, self.w_goal_terminal)
        terminal_cost_model.addCost("state_reg", x_reg_cost, self.w_state)
        terminal_cost_model.addCost("ctrl_reg", u_reg_cost, self.w_ctrl)

        running_model = crocoddyl.IntegratedActionModelEuler(
            DAM_fwd_example(self.state, running_cost_model), self.DT
        )
        terminal_model = crocoddyl.IntegratedActionModelEuler(
            DAM_fwd_exampleT(self.state, terminal_cost_model), 0.0
        )

        problem = crocoddyl.ShootingProblem(x0, [running_model] * self.T, terminal_model)
        ddp = crocoddyl.SolverDDP(problem)
        solved = ddp.solve([], [], max_iter)
        return ddp.xs, ddp.us, solved
