from reach_mpc_xarm6_nmpc_cpp import run_native, SIM_TIME


if __name__ == "__main__":
    run_native(sim_time=SIM_TIME, max_iter=50)
