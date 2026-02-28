#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nmpc_native/ddp_reach_solver.hpp"
#include "nmpc_native/render_step_controller.hpp"
#include "nmpc_native/render_tools.hpp"
#include "nmpc_native/run_nmpc.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nmpc_native, m) {
  m.doc() = "Native C++ NMPC loop and render helpers for xArm6";

  py::class_<nmpc_native::DDPReachSolver>(m, "DDPReachSolver")
      .def(py::init<const std::string&, const std::string&, int, double, double,
                    double, double, double>(),
           py::arg("urdf_path"), py::arg("ee_frame_name") = "link6",
           py::arg("T") = 40, py::arg("DT") = 0.01,
           py::arg("w_goal_running") = 1e4, py::arg("w_goal_terminal") = 1e7,
           py::arg("w_state") = 1e-1, py::arg("w_ctrl") = 1e-5)
      .def("solve", &nmpc_native::DDPReachSolver::solve, py::arg("x0"),
           py::arg("x_goal"), py::arg("max_iter") = 50)
      .def("reset_warm_start", &nmpc_native::DDPReachSolver::reset_warm_start)
      .def_property_readonly("T", &nmpc_native::DDPReachSolver::T)
      .def_property_readonly("DT", &nmpc_native::DDPReachSolver::DT)
      .def_property_readonly("nq", &nmpc_native::DDPReachSolver::nq)
      .def_property_readonly("nv", &nmpc_native::DDPReachSolver::nv)
      .def_property_readonly("nx", &nmpc_native::DDPReachSolver::nx)
      .def_property_readonly("nu", &nmpc_native::DDPReachSolver::nu);

  py::class_<nmpc_native::RenderStepController>(m, "RenderStepController")
      .def(py::init<const std::string&, int, double, double>(),
           py::arg("urdf_path"), py::arg("K") = 6, py::arg("P") = 1000.0,
           py::arg("D") = 50.0)
      .def("reset_plan", &nmpc_native::RenderStepController::reset_plan,
           py::arg("xs"))
      .def("compute_tau", &nmpc_native::RenderStepController::compute_tau,
           py::arg("q"), py::arg("v"))
      .def("set_gains", &nmpc_native::RenderStepController::set_gains,
           py::arg("P"), py::arg("D"))
      .def("set_downsample", &nmpc_native::RenderStepController::set_downsample,
           py::arg("K"))
      .def("reset_phase", &nmpc_native::RenderStepController::reset_phase)
      .def_property_readonly("nq", &nmpc_native::RenderStepController::nq)
      .def_property_readonly("nv", &nmpc_native::RenderStepController::nv)
      .def_property_readonly("nx", &nmpc_native::RenderStepController::nx)
      .def_property_readonly("step_index",
                             &nmpc_native::RenderStepController::step_index)
      .def_property_readonly("plan_index",
                             &nmpc_native::RenderStepController::plan_index);

  py::class_<nmpc_native::RenderTools>(m, "RenderTools")
      .def(py::init<const std::string&, const std::string&>(),
           py::arg("urdf_path"), py::arg("ee_frame_name") = "link6")
      .def("batch_trajectory_fk", &nmpc_native::RenderTools::batch_trajectory_fk,
           py::arg("xs"))
      .def("ee_position", &nmpc_native::RenderTools::ee_position,
           py::arg("q"))
      .def("add_marker", &nmpc_native::RenderTools::add_marker,
           py::arg("viewer"), py::arg("pos"), py::arg("size") = 0.02,
           py::arg("color") = py::make_tuple(1.0, 0.0, 0.0, 1.0))
      .def("add_markers", &nmpc_native::RenderTools::add_markers,
           py::arg("viewer"), py::arg("positions"), py::arg("size") = 0.02,
           py::arg("color") = py::make_tuple(1.0, 0.0, 0.0, 1.0))
      .def("draw_trajectory", &nmpc_native::RenderTools::draw_trajectory,
           py::arg("viewer"), py::arg("trajectory"),
           py::arg("color") = py::make_tuple(0.0, 0.0, 1.0, 1.0),
           py::arg("size") = 0.02)
      .def("draw_ee_frame", &nmpc_native::RenderTools::draw_ee_frame,
           py::arg("viewer"), py::arg("q"), py::arg("size") = 0.08)
      .def_property_readonly("nq", &nmpc_native::RenderTools::nq);

  m.def("run_nmpc", &nmpc_native::run_nmpc, py::arg("urdf_path"),
        py::arg("mjcf_path"), py::arg("ee_frame_name"), py::arg("x_goal"),
        py::arg("T") = 40, py::arg("DT") = 0.01, py::arg("H") = 2,
        py::arg("K") = 6, py::arg("sim_time") = 20.0, py::arg("P") = 1000.0,
        py::arg("D") = 50.0, py::arg("max_iter") = 50,
        py::arg("verbose") = true);
}
