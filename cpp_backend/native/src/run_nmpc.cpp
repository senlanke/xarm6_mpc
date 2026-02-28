#include "nmpc_native/run_nmpc.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <mujoco/mujoco.h>

#include <pybind11/stl.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "nmpc_native/ddp_reach_solver.hpp"

namespace nmpc_native {

namespace {

using MjModelPtr = std::unique_ptr<mjModel, decltype(&mj_deleteModel)>;
using MjDataPtr = std::unique_ptr<mjData, decltype(&mj_deleteData)>;

Eigen::VectorXd concat_qv(const mjData* d, int nq, int nv) {
  Eigen::VectorXd x(nq + nv);
  x.head(nq) = Eigen::Map<const Eigen::VectorXd>(d->qpos, nq);
  x.tail(nv) = Eigen::Map<const Eigen::VectorXd>(d->qvel, nv);
  return x;
}

}  // namespace

py::dict run_nmpc(
    const std::string& urdf_path, const std::string& mjcf_path,
    const std::string& ee_frame_name,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_goal_in,
    int T, double DT, int H, int K, double sim_time, double P, double D,
    int max_iter, bool verbose) {
  if (x_goal_in.size() != 3) {
    throw std::runtime_error("x_goal must contain exactly 3 values.");
  }
  if (T <= 0 || DT <= 0.0 || H <= 0 || K <= 0 || sim_time <= 0.0 ||
      max_iter <= 0) {
    throw std::runtime_error(
        "T/DT/H/K/sim_time/max_iter must all be positive.");
  }

  const auto x_goal_buf = x_goal_in.unchecked<1>();
  Eigen::Vector3d x_goal(x_goal_buf(0), x_goal_buf(1), x_goal_buf(2));

  pinocchio::Model pin_model;
  pinocchio::urdf::buildModel(urdf_path, pin_model);
  pinocchio::Data pin_data(pin_model);
  const pinocchio::FrameIndex eeid_pin = pin_model.getFrameId(ee_frame_name);
  if (eeid_pin >= pin_model.frames.size()) {
    throw std::runtime_error("End-effector frame not found: " + ee_frame_name);
  }

  char errbuf[1024];
  mjModel* raw_model = mj_loadXML(mjcf_path.c_str(), nullptr, errbuf, 1024);
  if (raw_model == nullptr) {
    throw std::runtime_error(std::string("mj_loadXML failed: ") + errbuf);
  }
  MjModelPtr mj_model(raw_model, mj_deleteModel);
  MjDataPtr mj_data(mj_makeData(mj_model.get()), mj_deleteData);
  if (mj_data == nullptr) {
    throw std::runtime_error("mj_makeData failed.");
  }
  if (mj_model->nkey > 0) {
    mj_resetDataKeyframe(mj_model.get(), mj_data.get(), 0);
  }
  mj_model->opt.timestep = 0.002;

  if (pin_model.nq != mj_model->nq || pin_model.nv != mj_model->nv) {
    throw std::runtime_error("Dimension mismatch between Pinocchio and MuJoCo.");
  }
  if (mj_model->nu != pin_model.nv) {
    throw std::runtime_error("Actuator dimension mismatch (nu != nv).");
  }

  const int nq = pin_model.nq;
  const int nv = pin_model.nv;
  const int nu = mj_model->nu;
  const int plan_period = H * K;

  DDPReachSolver solver(urdf_path, ee_frame_name, T, DT, 1e4, 1e7, 1e-1, 1e-5);
  if (solver.nq() != nq || solver.nv() != nv || solver.nu() != nu) {
    throw std::runtime_error("DDP solver dimensions mismatch.");
  }

  Eigen::VectorXd x0 = concat_qv(mj_data.get(), nq, nv);
  std::vector<Eigen::VectorXd> xs;

  Eigen::VectorXd xf = x0;
  Eigen::VectorXd zeros = Eigen::VectorXd::Zero(nv);

  std::vector<double> solve_times;
  std::vector<double> error_history;
  std::size_t replan_count = 0;
  std::size_t solve_success_count = 0;
  std::size_t step_count = 0;

  {
    py::gil_scoped_release release;

    int i = 0;
    while (mj_data->time < sim_time) {
      Eigen::VectorXd q = Eigen::Map<const Eigen::VectorXd>(mj_data->qpos, nq);
      Eigen::VectorXd v = Eigen::Map<const Eigen::VectorXd>(mj_data->qvel, nv);
      Eigen::VectorXd x(nq + nv);
      x << q, v;

      if (xs.empty() || i % plan_period == 0) {
        double dt_s = 0.0;
        const bool solved = solver.solve_once(x, x_goal, max_iter, &dt_s);
        solve_times.push_back(dt_s);
        ++replan_count;
        if (solved) {
          ++solve_success_count;
        }

        xs = solver.xs();
        i = 0;
      }

      if (i % K == 0) {
        int j = i / K;
        j = std::min(j, static_cast<int>(xs.size()) - 1);
        xf = xs[static_cast<std::size_t>(j)];
      }

      const Eigen::VectorXd q_des = xf.head(nq);
      const Eigen::VectorXd dq_des = xf.tail(nv);
      const Eigen::VectorXd tau =
          P * (q_des - q) + D * (dq_des - v) +
          pinocchio::rnea(pin_model, pin_data, q, zeros, zeros);

      for (int a = 0; a < nu; ++a) {
        mj_data->ctrl[a] = tau[a];
      }

      mj_step(mj_model.get(), mj_data.get());

      const Eigen::VectorXd q_now =
          Eigen::Map<const Eigen::VectorXd>(mj_data->qpos, nq);
      pinocchio::forwardKinematics(pin_model, pin_data, q_now);
      pinocchio::updateFramePlacements(pin_model, pin_data);
      const Eigen::Vector3d ee_pos = pin_data.oMf[eeid_pin].translation();
      error_history.push_back((ee_pos - x_goal).norm());

      ++i;
      ++step_count;
    }
  }

  const double mean_solve_time =
      solve_times.empty()
          ? 0.0
          : std::accumulate(solve_times.begin(), solve_times.end(), 0.0) /
                static_cast<double>(solve_times.size());

  const double final_error = error_history.empty() ? 0.0 : error_history.back();
  const double min_error =
      error_history.empty()
          ? 0.0
          : *std::min_element(error_history.begin(), error_history.end());

  py::dict out;
  out["steps"] = py::int_(step_count);
  out["replans"] = py::int_(replan_count);
  out["solve_success_count"] = py::int_(solve_success_count);
  out["solve_times"] = solve_times;
  out["mean_solve_time"] = py::float_(mean_solve_time);
  out["final_error"] = py::float_(final_error);
  out["min_error"] = py::float_(min_error);
  out["error_history"] = error_history;
  out["final_time"] = py::float_(mj_data->time);
  out["verbose"] = py::bool_(verbose);
  return out;
}

}  // namespace nmpc_native
