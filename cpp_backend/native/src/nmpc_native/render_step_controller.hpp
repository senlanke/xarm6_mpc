#pragma once

#include <memory>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nmpc_native {

class RenderStepController {
 public:
  RenderStepController(const std::string& urdf_path, int K = 6,
                       double P = 1000.0, double D = 50.0);
  ~RenderStepController();

  RenderStepController(const RenderStepController&) = delete;
  RenderStepController& operator=(const RenderStepController&) = delete;
  RenderStepController(RenderStepController&&) noexcept;
  RenderStepController& operator=(RenderStepController&&) noexcept;

  int nq() const;
  int nv() const;
  int nx() const;
  int step_index() const;
  int plan_index() const;

  void set_gains(double P, double D);
  void set_downsample(int K);
  void reset_phase();

  void reset_plan(
      py::array_t<double, py::array::c_style | py::array::forcecast> xs_in);

  py::array_t<double> compute_tau(
      py::array_t<double, py::array::c_style | py::array::forcecast> q_in,
      py::array_t<double, py::array::c_style | py::array::forcecast> v_in);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nmpc_native
