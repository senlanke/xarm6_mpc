#include "nmpc_native/ddp_reach_solver.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/solvers/ddp.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/residuals/frame-translation.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace nmpc_native {

namespace {

using Clock = std::chrono::steady_clock;

std::vector<Eigen::VectorXd> make_xs_warm(const Eigen::VectorXd& x0, int T) {
  return std::vector<Eigen::VectorXd>(static_cast<std::size_t>(T + 1), x0);
}

std::vector<Eigen::VectorXd> make_us_warm(int T, int nu) {
  std::vector<Eigen::VectorXd> us(static_cast<std::size_t>(T));
  for (int i = 0; i < T; ++i) {
    us[static_cast<std::size_t>(i)] = Eigen::VectorXd::Zero(nu);
  }
  return us;
}

Eigen::VectorXd parse_vector(
    py::array_t<double, py::array::c_style | py::array::forcecast> x_in,
    int expected_size, const char* name) {
  const auto x = x_in.unchecked<1>();
  if (x.shape(0) != expected_size) {
    throw std::runtime_error(std::string(name) + " shape mismatch.");
  }
  Eigen::VectorXd out(expected_size);
  for (int i = 0; i < expected_size; ++i) {
    out[i] = x(i);
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
      throw std::runtime_error("vector width mismatch while exporting to numpy.");
    }
    for (int c = 0; c < cols; ++c) {
      out_m(r, c) = v[c];
    }
  }
  return out;
}

}  // namespace

struct DDPReachSolver::Impl {
  pinocchio::Model pin_model;
  std::shared_ptr<pinocchio::Model> pin_model_ptr;
  pinocchio::FrameIndex eeid{0};

  std::shared_ptr<crocoddyl::StateMultibody> state;
  std::shared_ptr<crocoddyl::ActuationModelFull> actuation;
  std::shared_ptr<crocoddyl::ResidualModelFrameTranslation> goal_residual;
  std::shared_ptr<crocoddyl::ShootingProblem> problem;
  std::unique_ptr<crocoddyl::SolverDDP> ddp;

  std::vector<Eigen::VectorXd> xs_warm;
  std::vector<Eigen::VectorXd> us_warm;

  int T{40};
  double DT{0.01};
  int nq{0};
  int nv{0};
  int nx{0};
  int nu{0};
  bool warm_started{false};
};

DDPReachSolver::DDPReachSolver(const std::string& urdf_path,
                               const std::string& ee_frame_name, int T,
                               double DT, double w_goal_running,
                               double w_goal_terminal, double w_state,
                               double w_ctrl)
    : impl_(std::make_unique<Impl>()) {
  if (T <= 0 || DT <= 0.0) {
    throw std::runtime_error("T and DT must be positive.");
  }

  impl_->T = T;
  impl_->DT = DT;

  pinocchio::urdf::buildModel(urdf_path, impl_->pin_model);
  impl_->pin_model_ptr = std::make_shared<pinocchio::Model>(impl_->pin_model);
  impl_->eeid = impl_->pin_model.getFrameId(ee_frame_name);
  if (impl_->eeid >= impl_->pin_model.frames.size()) {
    throw std::runtime_error("End-effector frame not found: " + ee_frame_name);
  }

  impl_->state = std::make_shared<crocoddyl::StateMultibody>(impl_->pin_model_ptr);
  impl_->actuation = std::make_shared<crocoddyl::ActuationModelFull>(impl_->state);

  impl_->goal_residual =
      std::make_shared<crocoddyl::ResidualModelFrameTranslation>(
          impl_->state, impl_->eeid, Eigen::Vector3d::Zero());
  auto goal_cost =
      std::make_shared<crocoddyl::CostModelResidual>(impl_->state, impl_->goal_residual);
  auto state_cost = std::make_shared<crocoddyl::CostModelResidual>(
      impl_->state, std::make_shared<crocoddyl::ResidualModelState>(impl_->state));
  auto ctrl_cost = std::make_shared<crocoddyl::CostModelResidual>(
      impl_->state, std::make_shared<crocoddyl::ResidualModelControl>(impl_->state));

  auto running_cost_sum = std::make_shared<crocoddyl::CostModelSum>(impl_->state);
  auto terminal_cost_sum = std::make_shared<crocoddyl::CostModelSum>(impl_->state);

  running_cost_sum->addCost("ee_track", goal_cost, w_goal_running);
  running_cost_sum->addCost("state_reg", state_cost, w_state);
  running_cost_sum->addCost("ctrl_reg", ctrl_cost, w_ctrl);

  terminal_cost_sum->addCost("ee_track", goal_cost, w_goal_terminal);
  terminal_cost_sum->addCost("state_reg", state_cost, w_state);
  terminal_cost_sum->addCost("ctrl_reg", ctrl_cost, w_ctrl);

  auto running_dam =
      std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
          impl_->state, impl_->actuation, running_cost_sum);
  auto terminal_dam =
      std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
          impl_->state, impl_->actuation, terminal_cost_sum);

  auto running_model =
      std::make_shared<crocoddyl::IntegratedActionModelEuler>(running_dam, impl_->DT);
  auto terminal_model =
      std::make_shared<crocoddyl::IntegratedActionModelEuler>(terminal_dam, 0.0);

  impl_->nx = impl_->state->get_nx();
  impl_->nu = impl_->actuation->get_nu();
  impl_->nq = impl_->pin_model.nq;
  impl_->nv = impl_->pin_model.nv;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(impl_->nx);
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
      static_cast<std::size_t>(impl_->T), running_model);
  impl_->problem = std::make_shared<crocoddyl::ShootingProblem>(
      x0, running_models, terminal_model);
  impl_->ddp = std::make_unique<crocoddyl::SolverDDP>(impl_->problem);
}

DDPReachSolver::~DDPReachSolver() = default;
DDPReachSolver::DDPReachSolver(DDPReachSolver&&) noexcept = default;
DDPReachSolver& DDPReachSolver::operator=(DDPReachSolver&&) noexcept = default;

int DDPReachSolver::T() const { return impl_->T; }
double DDPReachSolver::DT() const { return impl_->DT; }
int DDPReachSolver::nq() const { return impl_->nq; }
int DDPReachSolver::nv() const { return impl_->nv; }
int DDPReachSolver::nx() const { return impl_->nx; }
int DDPReachSolver::nu() const { return impl_->nu; }

void DDPReachSolver::reset_warm_start() {
  impl_->xs_warm.clear();
  impl_->us_warm.clear();
  impl_->warm_started = false;
}

bool DDPReachSolver::solve_once(const Eigen::VectorXd& x0,
                                const Eigen::Vector3d& x_goal, int max_iter,
                                double* solve_time) {
  if (max_iter <= 0) {
    throw std::runtime_error("max_iter must be positive.");
  }
  if (x0.size() != impl_->nx) {
    throw std::runtime_error("x0 shape mismatch.");
  }

  impl_->problem->set_x0(x0);
  impl_->goal_residual->set_reference(x_goal);

  if (!impl_->warm_started) {
    impl_->xs_warm = make_xs_warm(x0, impl_->T);
    impl_->us_warm = make_us_warm(impl_->T, impl_->nu);
    impl_->warm_started = true;
  } else {
    if (impl_->xs_warm.size() != static_cast<std::size_t>(impl_->T + 1) ||
        impl_->us_warm.size() != static_cast<std::size_t>(impl_->T)) {
      impl_->xs_warm = make_xs_warm(x0, impl_->T);
      impl_->us_warm = make_us_warm(impl_->T, impl_->nu);
    } else {
      impl_->xs_warm.front() = x0;
    }
  }

  const auto t0 = Clock::now();
  const bool solved = impl_->ddp->solve(impl_->xs_warm, impl_->us_warm,
                                        static_cast<std::size_t>(max_iter));
  const auto t1 = Clock::now();

  if (solve_time != nullptr) {
    *solve_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count();
  }

  impl_->xs_warm = impl_->ddp->get_xs();
  impl_->us_warm = impl_->ddp->get_us();
  return solved;
}

const std::vector<Eigen::VectorXd>& DDPReachSolver::xs() const {
  return impl_->xs_warm;
}

const std::vector<Eigen::VectorXd>& DDPReachSolver::us() const {
  return impl_->us_warm;
}

py::dict DDPReachSolver::solve(
    py::array_t<double, py::array::c_style | py::array::forcecast> x0_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_goal_in,
    int max_iter) {
  const Eigen::VectorXd x0 = parse_vector(x0_in, impl_->nx, "x0");
  const Eigen::VectorXd x_goal_v = parse_vector(x_goal_in, 3, "x_goal");
  const Eigen::Vector3d x_goal = x_goal_v.head<3>();

  double solve_time = 0.0;
  bool solved = false;
  {
    py::gil_scoped_release release;
    solved = solve_once(x0, x_goal, max_iter, &solve_time);
  }

  py::dict out;
  out["solved"] = py::bool_(solved);
  out["solve_time"] = py::float_(solve_time);
  out["xs"] = vectors_to_numpy_2d(impl_->xs_warm, impl_->nx);
  out["us"] = vectors_to_numpy_2d(impl_->us_warm, impl_->nu);
  return out;
}

}  // namespace nmpc_native
