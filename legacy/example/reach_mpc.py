import ctypes
import os
import sys

# Avoid importing ROS Python packages when running in conda.
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

# Ensure conda libstdc++ is available for crocoddyl.
conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    libstdcpp = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
    if os.path.exists(libstdcpp):
        ctypes.CDLL(libstdcpp, mode=ctypes.RTLD_GLOBAL)

import mujoco
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pinocchio as pin
import numpy as np 
import mujoco_viewer
from utils import robot_utils
from utils import visualizer 
from mpc.solver.reach_ddp import solver_ddp_reach
import glfw


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF_PATH = os.path.join(ROOT_DIR, "xarm_description", "robots", "xarm7.urdf")
MJCF_PATH = os.path.join(ROOT_DIR, "xarm_description", "xarm_mj", "xarm7_nohand.xml")

pin_model = pin.buildModelFromUrdf(URDF_PATH)
pin_data = pin_model.createData()
urdf_path = URDF_PATH

mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.002
eeid_mj = 8 
eeid_pin = pin_model.getFrameId("link7")
viewer = mujoco_viewer.MujocoViewer(mj_model,mj_data)
# viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
nq = pin_model.nq

# mode = "mpc_test"
# mode = "traj_opt_test"
mode = "line_track_test"   # 新增直线轨迹跟踪模式

T = 50
DT = 0.01
H = 3
K = 5 


P = 1000
D = 50
i = 0

xee = np.zeros((3,T+1))



print(mj_data.mocap_pos)

T = 50
DT = 0.01
nq = pin_model.nq

if mode == "line_track_test":
    import time

    # 定义直线轨迹起点和终点 (笛卡尔空间)
    start = np.array([10.2, -0.3, 0.4])
    end   = np.array([0.6, -0.3, 0.4])
    line_traj = np.linspace(start, end, T+1)

    # 初始状态
    q, v = mj_data.qpos, mj_data.qvel
    x = np.concatenate([q, v])

    # 使用 DDP 求解器生成控制序列
    solver = solver_ddp_reach(urdf_path, T, DT)
    et = time.time()
    xs, us = solver.generate_ddp(x, line_traj[-1])  # 以终点作为整体目标
    st = time.time()
    print("time required to solve", st-et)

    # 可视化直线轨迹
    xee = np.zeros((3, T+1))
    for i in range(T+1):
        qs = xs[i][:nq]
        xee[:, i] = robot_utils.forward_kinematics(pin_model, pin_data, eeid_pin, qs)[:3, 3]
    visualizer.visualize_trajectory(viewer, xee, (0, 0, 1, 1))

    # 执行轨迹跟踪
    n_col = len(us)
    for i in range(n_col):
        xi = xs[i]
        q = xi[:nq]
        dq = xi[nq:]
        u = us[i]

        mj_data.qpos, mj_data.qvel, mj_data.ctrl = q, dq, u
        mujoco.mj_step(mj_model, mj_data)

        # 渲染仿真
        viewer.render()

    print("直线轨迹跟踪完成")

if mode == "mpc_test":
    while mj_data.time < 20:
        
        t = mj_data.time
        q, v = mj_data.qpos, mj_data.qvel 
        x_goal = np.array([0.4,-0.4,0.5])
        x_goal = mj_data.mocap_pos[0]
        x = np.concatenate([q,v])
        if i%(H*K) == 0:
            solver = solver_ddp_reach(urdf_path, T, DT)
            xs, us = solver.generate_ddp(x,x_goal)
            i = 0 
            for i in range(T+1):
                qs = xs[i][:nq]
                xee[:,i] = robot_utils.forward_kinematics(pin_model,pin_data, eeid_pin,qs)[:3,3]

            
            visualizer.visualize_trajectory(viewer, xee, (0,0,1,1))
        if i%(K) == 0:
            j = int(i/K)
            xf = xs[j]
        q_des, dq_des = xf[:nq], xf[nq:]
        tau = P*(q_des-q)+D*(dq_des-v)+pin.rnea(pin_model,pin_data,q, np.zeros(nq), np.zeros(nq))
        i+=1

        mj_data.ctrl = tau 
        mujoco.mj_step(mj_model,mj_data)
        viewer.render()
        # viewer.sync()




if mode == "solver_eval_test":
    for i in range(30):
        q, v = np.random.rand(nq), np.random.rand(nq)
        x_goal = np.array([0.4, -0.4, 0.4])
        x = np.concatenate([q,v])
        solver = solver_ddp_reach(urdf_path,T,DT)
        xs, us = solver.generate_ddp(x,x_goal)
        n_col = len(us)
        xf = xs[-1]
        qf = xf[:nq]
        x_f = robot_utils.forward_kinematics(pin_model, pin_data, eeid_pin, qf)[:3,3]
        print("EE error", np.linalg.norm(x_f-x_goal), "m")


if mode == "traj_opt_test":
    import time 
    q, v = mj_data.qpos, mj_data.qvel
    x_goal = np.array([0.4, -0.4, 0.4])
    x = np.concatenate([q,v])
    solver = solver_ddp_reach(urdf_path,T,DT)
    et = time.time()
    xs, us = solver.generate_ddp(x,x_goal)
    st = time.time()
    print("time required to solve", st-et)
    n_col = len(us)

    xf = xs[-1]
    qf = xf[:nq]

    print("reached EE position", robot_utils.forward_kinematics(pin_model, pin_data, eeid_pin, qf)[:3,3])
    for i in range(n_col):
        xi = xs[i]
        q = xi[:nq]
        dq = xi[nq:]
        u = us[i]
        mj_data.qpos, mj_data.qvel, mj_data.ctrl = q,dq,u
        mujoco.mj_step(mj_model,mj_data)

        viewer.render()
        # viewer.sync()

# viewer.close()
