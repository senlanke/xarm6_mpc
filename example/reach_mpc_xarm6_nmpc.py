import os
import sys
import time
import ctypes

def _prepare_runtime_env():
    # Avoid mixing ROS Python packages with conda packages (pinocchio/crocoddyl mismatch).
    sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

    # Prefer conda libstdc++ for crocoddyl shared library symbols (CXXABI_1.3.15+).
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        libstdcpp = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
        if os.path.exists(libstdcpp):
            ctypes.CDLL(libstdcpp, mode=ctypes.RTLD_GLOBAL)


_prepare_runtime_env()

import mujoco
import mujoco_viewer
import numpy as np
import pinocchio as pin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mpc.solver.reach_ddp_xarm6 import solver_ddp_reach_xarm6
from utils import robot_utils
from utils import visualizer


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF_PATH = os.path.join(ROOT_DIR, "ufactory_xarm6", "xarm6_robot.urdf")
MJCF_PATH = os.path.join(ROOT_DIR, "ufactory_xarm6", "xarm6_nohand_motor.xml")
EE_FRAME_NAME = "link6"

# NMPC settings (tuned on xArm6 reachable workspace within joint limits)
T = 40
DT = 0.01
H = 2
K = 6
SIM_TIME = 20.0

# Joint-space tracking gains (same control style as reach_mpc.py)
P = 1000.0
D = 50.0

# Cartesian goal for the end-effector frame (from feasible mid-range joint posture)
X_GOAL = np.array([0.3871, 0.0, 0.4278])


# Build Pinocchio model for dynamics / kinematics.
pin_model = pin.buildModelFromUrdf(URDF_PATH)
pin_data = pin_model.createData()
eeid_pin = pin_model.getFrameId(EE_FRAME_NAME)
if eeid_pin >= len(pin_model.frames):
    raise ValueError(f"Frame not found in URDF: {EE_FRAME_NAME}")

# Build MuJoCo model for simulation (motor/general torque actuators).
mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.002

if pin_model.nq != mj_model.nq or pin_model.nv != mj_model.nv:
    raise RuntimeError(
        f"Dimension mismatch: pin(nq={pin_model.nq}, nv={pin_model.nv}) vs "
        f"mujoco(nq={mj_model.nq}, nv={mj_model.nv})"
    )
if mj_model.nu != pin_model.nv:
    raise RuntimeError(
        f"Actuator dimension mismatch: nu={mj_model.nu}, expected {pin_model.nv}"
    )

viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data)

solver = solver_ddp_reach_xarm6(
    urdf_path=URDF_PATH,
    ee_frame_name=EE_FRAME_NAME,
    T=T,
    DT=DT,
)

nq = pin_model.nq
i = 0
xs, us = None, None
xf = np.concatenate([mj_data.qpos.copy(), mj_data.qvel.copy()])

xee = np.zeros((3, T + 1))
plan_period = H * K

print(f"Running xArm6 NMPC torque control with goal {X_GOAL}.")
print(f"Model dims: nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")

while mj_data.time < SIM_TIME:
    q = mj_data.qpos.copy()
    v = mj_data.qvel.copy()
    x = np.concatenate([q, v])

    # Re-plan every H*K simulation steps (reach_mpc.py style).
    if xs is None or i % plan_period == 0:
        t0 = time.time()
        xs, us, solved = solver.generate_ddp(x, X_GOAL, max_iter=50)
        t1 = time.time()
        print(
            f"[t={mj_data.time:.3f}] DDP solved={solved}, "
            f"iters={len(us)}, solve_time={t1 - t0:.3f}s"
        )

        for k in range(T + 1):
            qk = xs[k][:nq]
            xee[:, k] = robot_utils.forward_kinematics(
                pin_model, pin_data, eeid_pin, qk
            )[:3, 3]
        visualizer.visualize_trajectory(viewer, xee, (0, 0, 1, 1))
        i = 0

    if i % K == 0:
        j = int(i / K)
        xf = xs[min(j, len(xs) - 1)]
    q_des, dq_des = xf[:nq], xf[nq:]
    tau = P * (q_des - q) + D * (dq_des - v) + pin.rnea(
        pin_model, pin_data, q, np.zeros(nq), np.zeros(nq)
    )

    mj_data.ctrl = tau
    mujoco.mj_step(mj_model, mj_data)

    # Render current ee frame and goal marker.
    T_ee = robot_utils.forward_kinematics(pin_model, pin_data, eeid_pin, mj_data.qpos)
    visualizer.visualize_frame(viewer, T_ee, size=0.08)
    visualizer.add_marker(viewer, X_GOAL, size=0.012, color=(1, 0, 0, 1))
    viewer.render()

    i += 1

print("xArm6 NMPC run finished.")
