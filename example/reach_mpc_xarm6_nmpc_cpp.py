import argparse
import ctypes
import os
import sys


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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
CPP_BACKEND_DIR = os.path.join(ROOT_DIR, "cpp_backend")
if CPP_BACKEND_DIR not in sys.path:
    sys.path.insert(0, CPP_BACKEND_DIR)

import numpy as np

URDF_PATH = os.path.join(ROOT_DIR, "ufactory_xarm6", "xarm6_robot.urdf")
MJCF_PATH = os.path.join(ROOT_DIR, "ufactory_xarm6", "xarm6_nohand_motor.xml")
EE_FRAME_NAME = "link6"

# NMPC settings
T = 40
DT = 0.01
H = 2
K = 6
SIM_TIME = 20.0

# Joint-space tracking gains
P = 1000.0
D = 50.0

X_GOAL = np.array([0.3871, 0.0, 0.4278])


def run_render(sim_time: float, max_iter: int):
    import mujoco
    import mujoco_viewer

    try:
        import nmpc_native
    except ImportError as exc:
        raise RuntimeError(
            "nmpc_native module not found. Build it first:\n"
            "PYTHON_BIN=/home/kesl/miniconda3/envs/ke/bin/python bash /home/kesl/ke/xarm6_mpc/build_cpp_all.sh"
        ) from exc

    print("[init] loading MuJoCo model...", flush=True)
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_model.opt.timestep = 0.002
    ee_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    if mj_model.nu != mj_model.nv:
        raise RuntimeError(f"Actuator dimension mismatch: nu={mj_model.nu}, expected {mj_model.nv}")

    print("[init] creating viewer...", flush=True)
    viewer = mujoco_viewer.MujocoViewer(
        mj_model,
        mj_data,
        width=1280,
        height=720,
    )
    print("[init] viewer ready.", flush=True)

    print("[init] creating RenderNmpcRunner...", flush=True)
    runner = nmpc_native.RenderNmpcRunner(
        urdf_path=URDF_PATH,
        ee_frame_name=EE_FRAME_NAME,
        T=T,
        DT=DT,
        H=H,
        K=K,
        P=P,
        D=D,
    )
    print("[init] runner ready.", flush=True)

    nq = mj_model.nq
    nv = mj_model.nv

    print(f"Running xArm6 NMPC (render mode, C++ backend) with goal {X_GOAL}.", flush=True)
    print(f"Model dims: nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}", flush=True)
    # Render one frame first so window appears even if first replan is slow.
    viewer.render()

    while mj_data.time < sim_time:
        q = mj_data.qpos[:nq]
        v = mj_data.qvel[:nv]
        will_replan = (runner.step_index % runner.plan_period) == 0
        if will_replan:
            print(f"[t={mj_data.time:.3f}] planning...", flush=True)
        step_out = runner.step(
            viewer=viewer,
            q=q,
            v=v,
            x_goal=X_GOAL,
            max_iter=max_iter,
            force_replan=False,
            traj_size=0.02,
            ee_frame_size=0.08,
            goal_marker_size=0.012,
        )

        if bool(step_out["replanned"]):
            solved = bool(step_out["solved"])
            solve_time = float(step_out["solve_time"])
            print(
                f"[t={mj_data.time:.3f}] DDP solved={solved}, "
                f"iters={max_iter}, solve_time={solve_time:.3f}s"
            )

        tau = np.asarray(step_out["tau"], dtype=np.float64)

        mj_data.ctrl = tau
        mujoco.mj_step(mj_model, mj_data)

        viewer.render()

    final_ee = np.asarray(runner.ee_position(mj_data.qpos), dtype=np.float64)
    final_link6_err = np.linalg.norm(final_ee - X_GOAL)
    print(f"final_link6_error={final_link6_err:.6f}m")
    if ee_site_id >= 0:
        final_ee_site_err = np.linalg.norm(mj_data.site_xpos[ee_site_id] - X_GOAL)
        print(f"final_ee_site_error={final_ee_site_err:.6f}m")
    print("xArm6 NMPC render run finished.")


def run_native(sim_time: float, max_iter: int):
    try:
        import nmpc_native
    except ImportError as exc:
        raise RuntimeError(
            "nmpc_native module not found. Build it first:\n"
            "PYTHON_BIN=/home/kesl/miniconda3/envs/ke/bin/python bash /home/kesl/ke/xarm6_mpc/build_cpp_all.sh"
        ) from exc

    print(f"Running xArm6 NMPC (native C++ loop) with goal {X_GOAL}.")
    result = nmpc_native.run_nmpc(
        urdf_path=URDF_PATH,
        mjcf_path=MJCF_PATH,
        ee_frame_name=EE_FRAME_NAME,
        x_goal=X_GOAL,
        T=T,
        DT=DT,
        H=H,
        K=K,
        sim_time=sim_time,
        P=P,
        D=D,
        max_iter=max_iter,
    )
    print("xArm6 NMPC native run finished.")
    print(
        f"steps={result['steps']}, replans={result['replans']}, "
        f"solve_success={result['solve_success_count']}/{result['replans']}"
    )
    print(
        f"mean_solve_time={result['mean_solve_time']:.6f}s, "
        f"final_error={result['final_error']:.6f}m, min_error={result['min_error']:.6f}m"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["render", "native"],
        default="render",
        help="render: default with mujoco_viewer; native: full C++ loop without viewer",
    )
    parser.add_argument("--sim-time", type=float, default=SIM_TIME)
    parser.add_argument("--max-iter", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.backend == "render":
        run_render(sim_time=args.sim_time, max_iter=args.max_iter)
    else:
        run_native(sim_time=args.sim_time, max_iter=args.max_iter)
