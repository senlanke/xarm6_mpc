#pragma once

#include <memory>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nmpc_native {

class RenderTools {
 public:
  RenderTools(const std::string& urdf_path, const std::string& ee_frame_name);
  ~RenderTools();

  RenderTools(const RenderTools&) = delete;
  RenderTools& operator=(const RenderTools&) = delete;
  RenderTools(RenderTools&&) noexcept;
  RenderTools& operator=(RenderTools&&) noexcept;

  int nq() const;

  py::array_t<double> batch_trajectory_fk(
      py::array_t<double, py::array::c_style | py::array::forcecast> xs_in);

  py::array_t<double> ee_position(
      py::array_t<double, py::array::c_style | py::array::forcecast> q_in);

  void add_marker(py::object viewer,
                  py::array_t<double, py::array::c_style | py::array::forcecast>
                      pos_in,
                  double size = 0.02,
                  py::tuple color = py::make_tuple(1.0, 0.0, 0.0, 1.0));

  void add_markers(
      py::object viewer,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          positions_in,
      double size = 0.02,
      py::tuple color = py::make_tuple(1.0, 0.0, 0.0, 1.0));

  void draw_trajectory(
      py::object viewer,
      py::array_t<double, py::array::c_style | py::array::forcecast> traj_in,
      py::tuple color = py::make_tuple(0.0, 0.0, 1.0, 1.0),
      double size = 0.02);

  void draw_ee_frame(
      py::object viewer,
      py::array_t<double, py::array::c_style | py::array::forcecast> q_in,
      double size = 0.08);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nmpc_native
