#include "nmpc_native/render_nmpc_runner.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "nmpc_native/ddp_reach_solver.hpp"
#include "nmpc_native/render_step_controller.hpp"
#include "nmpc_native/render_tools.hpp"

namespace nmpc_native {

namespace {

Eigen::VectorXd parse_vector(
    py::array_t<double, py::array::c_style | py::array::forcecast> in,
    int expected, const char* name) {
  const auto x = in.unchecked<1>();
  if (x.shape(0) != expected) {
    throw std::runtime_error(std::string(name) + " shape mismatch.");
  }
  Eigen::VectorXd out(expected);
  for (int i = 0; i < expected; ++i) {
    out[i] = x(i);
  }
  return out;
}

py::array_t<double> vector_to_numpy(const Eigen::VectorXd& x) {
  py::array_t<double> out(x.size());
  auto m = out.mutable_unchecked<1>();
  for (int i = 0; i < x.size(); ++i) {
    m(i) = x[i];
  }
  return out;
}

py::array_t<double> vectors_to_numpy_2d(const std::vector<Eigen::VectorXd>& vecs,
                                        int cols) {
  const py::ssize_t rows = static_cast<py::ssize_t>(vecs.size());
  py::array out_raw(py::dtype::of<double>(), {rows, py::ssize_t(cols)});
  py::array_t<double> out = out_raw.cast<py::array_t<double>>();
  auto out_m = out.mutable_unchecked<2>();
  for (py::ssize_t r = 0; r < rows; ++r) {
    const auto& v = vecs[static_cast<std::size_t>(r)];
    if (v.size() != cols) {
      throw std::runtime_error("vector width mismatch while exporting plan.");
    }
    for (int c = 0; c < cols; ++c) {
      out_m(r, c) = v[c];
    }
  }
  return out;
}

}  // namespace

struct RenderNmpcRunner::Impl {
  DDPReachSolver solver;
  RenderStepController controller;
  RenderTools render_tools;
  int nq{0};
  int nv{0};
  int nu{0};
  int nx{0};
  int H{2};
  int K{6};
  int plan_period{12};
  int i_plan{0};
  bool has_plan{false};

  Impl(const std::string& urdf_path, const std::string& ee_frame_name, int T,
       double DT, int H_in, int K_in, double P, double D,
       double w_goal_running, double w_goal_terminal, double w_state,
       double w_ctrl)
      : solver(urdf_path, ee_frame_name, T, DT, w_goal_running, w_goal_terminal,
               w_state, w_ctrl),
        controller(urdf_path, K_in, P, D),
        render_tools(urdf_path, ee_frame_name),
        H(H_in),
        K(K_in) {
    if (H <= 0 || K <= 0) {
      throw std::runtime_error("H and K must be positive.");
    }
    nq = solver.nq();
    nv = solver.nv();
    nu = solver.nu();
    nx = solver.nx();
    plan_period = H * K;
    if (controller.nq() != nq || controller.nv() != nv ||
        render_tools.nq() != nq) {
      throw std::runtime_error("Runner dimensions mismatch.");
    }
  }
};

RenderNmpcRunner::RenderNmpcRunner(const std::string& urdf_path,
                                   const std::string& ee_frame_name, int T,
                                   double DT, int H, int K, double P, double D,
                                   double w_goal_running,
                                   double w_goal_terminal, double w_state,
                                   double w_ctrl)
    : impl_(std::make_unique<Impl>(
          urdf_path, ee_frame_name, T, DT, H, K, P, D, w_goal_running,
          w_goal_terminal, w_state, w_ctrl)) {}

RenderNmpcRunner::~RenderNmpcRunner() = default;
RenderNmpcRunner::RenderNmpcRunner(RenderNmpcRunner&&) noexcept = default;
RenderNmpcRunner& RenderNmpcRunner::operator=(RenderNmpcRunner&&) noexcept =
    default;

int RenderNmpcRunner::nq() const { return impl_->nq; }
int RenderNmpcRunner::nv() const { return impl_->nv; }
int RenderNmpcRunner::nu() const { return impl_->nu; }
int RenderNmpcRunner::step_index() const { return impl_->i_plan; }
int RenderNmpcRunner::plan_period() const { return impl_->plan_period; }

void RenderNmpcRunner::reset() {
  impl_->solver.reset_warm_start();
  impl_->controller.reset_phase();
  impl_->i_plan = 0;
  impl_->has_plan = false;
}

py::dict RenderNmpcRunner::step(
    py::object viewer,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> v_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_goal_in,
    int max_iter, bool force_replan, double traj_size, double ee_frame_size,
    double goal_marker_size) {
  if (max_iter <= 0) {
    throw std::runtime_error("max_iter must be positive.");
  }

  const Eigen::VectorXd q = parse_vector(q_in, impl_->nq, "q");
  const Eigen::VectorXd v = parse_vector(v_in, impl_->nv, "v");
  const Eigen::VectorXd x_goal_v = parse_vector(x_goal_in, 3, "x_goal");
  const Eigen::Vector3d x_goal = x_goal_v.head<3>();

  Eigen::VectorXd x(impl_->nx);
  x << q, v;

  bool replanned = false;
  bool solved = false;
  double solve_time = 0.0;
  py::array_t<double> xs_np;
  py::array_t<double> xee_np;

  if (!impl_->has_plan || force_replan ||
      (impl_->i_plan % impl_->plan_period == 0)) {
    {
      py::gil_scoped_release release;
      solved = impl_->solver.solve_once(x, x_goal, max_iter, &solve_time);
    }
    replanned = true;
    impl_->has_plan = true;
    impl_->i_plan = 0;

    xs_np = vectors_to_numpy_2d(impl_->solver.xs(), impl_->nx);
    impl_->controller.reset_plan(xs_np);
    xee_np = impl_->render_tools.batch_trajectory_fk(xs_np);
    impl_->render_tools.draw_trajectory(
        viewer, xee_np, py::make_tuple(0.0, 0.0, 1.0, 1.0), traj_size);
  }

  const py::array_t<double> q_np = vector_to_numpy(q);
  const py::array_t<double> v_np = vector_to_numpy(v);
  const py::array_t<double> tau =
      impl_->controller.compute_tau(q_np, v_np);

  impl_->render_tools.draw_ee_frame(viewer, q_np, ee_frame_size);
  impl_->render_tools.add_marker(
      viewer, x_goal_in, goal_marker_size, py::make_tuple(1.0, 0.0, 0.0, 1.0));

  ++impl_->i_plan;

  py::dict out;
  out["tau"] = tau;
  out["replanned"] = py::bool_(replanned);
  out["solved"] = py::bool_(solved);
  out["solve_time"] = py::float_(solve_time);
  return out;
}

py::array_t<double> RenderNmpcRunner::ee_position(
    py::array_t<double, py::array::c_style | py::array::forcecast> q_in) {
  return impl_->render_tools.ee_position(q_in);
}

}  // namespace nmpc_native

