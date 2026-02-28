#pragma once

#include <memory>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nmpc_native {

class RenderNmpcRunner {
 public:
  RenderNmpcRunner(const std::string& urdf_path,
                   const std::string& ee_frame_name = "link6", int T = 40,
                   double DT = 0.01, int H = 2, int K = 6, double P = 1000.0,
                   double D = 50.0, double w_goal_running = 1e4,
                   double w_goal_terminal = 1e7, double w_state = 1e-1,
                   double w_ctrl = 1e-5);
  ~RenderNmpcRunner();

  RenderNmpcRunner(const RenderNmpcRunner&) = delete;
  RenderNmpcRunner& operator=(const RenderNmpcRunner&) = delete;
  RenderNmpcRunner(RenderNmpcRunner&&) noexcept;
  RenderNmpcRunner& operator=(RenderNmpcRunner&&) noexcept;

  int nq() const;
  int nv() const;
  int nu() const;
  int step_index() const;
  int plan_period() const;

  void reset();

  py::dict step(
      py::object viewer,
      py::array_t<double, py::array::c_style | py::array::forcecast> q_in,
      py::array_t<double, py::array::c_style | py::array::forcecast> v_in,
      py::array_t<double, py::array::c_style | py::array::forcecast> x_goal_in,
      int max_iter = 50, bool force_replan = false, double traj_size = 0.02,
      double ee_frame_size = 0.08, double goal_marker_size = 0.012);

  py::array_t<double> ee_position(
      py::array_t<double, py::array::c_style | py::array::forcecast> q_in);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nmpc_native

