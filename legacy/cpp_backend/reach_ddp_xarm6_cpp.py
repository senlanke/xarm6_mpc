import crocoddyl
import numpy as np
import pinocchio as pin


class SolverDDPReachXarm6Cpp:
    """DDP reach solver backed by Crocoddyl built-in C++ dynamics models."""

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
        self.actuation = crocoddyl.ActuationModelFull(self.state)
        self.T = T
        self.DT = DT

        self.ee_frame_id = self.robot_model.getFrameId(ee_frame_name)
        if self.ee_frame_id >= len(self.robot_model.frames):
            raise ValueError(f"End-effector frame not found: {ee_frame_name}")

        self.goal_residual = crocoddyl.ResidualModelFrameTranslation(
            self.state, self.ee_frame_id, np.zeros(3)
        )
        self.goal_tracking_cost = crocoddyl.CostModelResidual(
            self.state, self.goal_residual
        )
        self.x_reg_cost = crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelState(self.state)
        )
        self.u_reg_cost = crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelControl(self.state)
        )

        running_cost_model = crocoddyl.CostModelSum(self.state)
        terminal_cost_model = crocoddyl.CostModelSum(self.state)

        running_cost_model.addCost("ee_track", self.goal_tracking_cost, w_goal_running)
        running_cost_model.addCost("state_reg", self.x_reg_cost, w_state)
        running_cost_model.addCost("ctrl_reg", self.u_reg_cost, w_ctrl)

        terminal_cost_model.addCost("ee_track", self.goal_tracking_cost, w_goal_terminal)
        terminal_cost_model.addCost("state_reg", self.x_reg_cost, w_state)
        terminal_cost_model.addCost("ctrl_reg", self.u_reg_cost, w_ctrl)

        running_dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, running_cost_model
        )
        terminal_dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, terminal_cost_model
        )

        running_model = crocoddyl.IntegratedActionModelEuler(running_dam, self.DT)
        terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_dam, 0.0)

        x0 = np.zeros(self.state.nx)
        self.problem = crocoddyl.ShootingProblem(
            x0, [running_model] * self.T, terminal_model
        )
        self.ddp = crocoddyl.SolverDDP(self.problem)

        self._xs_init = None
        self._us_init = None

    def _default_warm_start(self, x0):
        xs = [x0.copy() for _ in range(self.T + 1)]
        us = [np.zeros(self.actuation.nu) for _ in range(self.T)]
        return xs, us

    def generate_ddp(self, x0, x_ref, max_iter=100):
        x0 = np.asarray(x0).reshape(self.state.nx)
        x_ref = np.asarray(x_ref).reshape(3)

        self.goal_residual.reference = x_ref
        self.problem.x0 = x0

        if self._xs_init is None or self._us_init is None:
            xs_init, us_init = self._default_warm_start(x0)
        else:
            xs_init = [x.copy() for x in self._xs_init]
            us_init = [u.copy() for u in self._us_init]
            xs_init[0] = x0.copy()

        solved = self.ddp.solve(xs_init, us_init, max_iter)

        xs = [x.copy() for x in self.ddp.xs]
        us = [u.copy() for u in self.ddp.us]

        self._xs_init = xs
        self._us_init = us

        return xs, us, solved
