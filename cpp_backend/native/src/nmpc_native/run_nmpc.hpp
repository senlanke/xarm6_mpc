#pragma once

#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nmpc_native {

py::dict run_nmpc(
    const std::string& urdf_path, const std::string& mjcf_path,
    const std::string& ee_frame_name,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_goal_in,
    int T = 40, double DT = 0.01, int H = 2, int K = 6, double sim_time = 20.0,
    double P = 1000.0, double D = 50.0, int max_iter = 50,
    bool verbose = true);

}  // namespace nmpc_native
