#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nmpc_native {

class DDPReachSolver {
 public:
  DDPReachSolver(const std::string& urdf_path,
                 const std::string& ee_frame_name = "link6", int T = 40,
                 double DT = 0.01, double w_goal_running = 1e4,
                 double w_goal_terminal = 1e7, double w_state = 1e-1,
                 double w_ctrl = 1e-5);
  ~DDPReachSolver();

  DDPReachSolver(const DDPReachSolver&) = delete;
  DDPReachSolver& operator=(const DDPReachSolver&) = delete;
  DDPReachSolver(DDPReachSolver&&) noexcept;
  DDPReachSolver& operator=(DDPReachSolver&&) noexcept;

  int T() const;
  double DT() const;
  int nq() const;
  int nv() const;
  int nx() const;
  int nu() const;

  void reset_warm_start();

  bool solve_once(const Eigen::VectorXd& x0, const Eigen::Vector3d& x_goal,
                  int max_iter, double* solve_time = nullptr);

  const std::vector<Eigen::VectorXd>& xs() const;
  const std::vector<Eigen::VectorXd>& us() const;

  py::dict solve(
      py::array_t<double, py::array::c_style | py::array::forcecast> x0_in,
      py::array_t<double, py::array::c_style | py::array::forcecast> x_goal_in,
      int max_iter = 50);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nmpc_native
