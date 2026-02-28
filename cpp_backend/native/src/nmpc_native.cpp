#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <mujoco/mujoco.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

using Clock = std::chrono::steady_clock;
using MjModelPtr = std::unique_ptr<mjModel, decltype(&mj_deleteModel)>;
using MjDataPtr = std::unique_ptr<mjData, decltype(&mj_deleteData)>;

Eigen::VectorXd concat_qv(const mjData* d, int nq, int nv) {
  Eigen::VectorXd x(nq + nv);
  x.head(nq) = Eigen::Map<const Eigen::VectorXd>(d->qpos, nq);
  x.tail(nv) = Eigen::Map<const Eigen::VectorXd>(d->qvel, nv);
  return x;
}

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

class DDPReachSolver {
 public:
  DDPReachSolver(const std::string& urdf_path,
                 const std::string& ee_frame_name = "link6", int T = 40,
                 double DT = 0.01, double w_goal_running = 1e4,
                 double w_goal_terminal = 1e7, double w_state = 1e-1,
                 double w_ctrl = 1e-5)
      : T_(T), DT_(DT) {
    if (T_ <= 0 || DT_ <= 0.0) {
      throw std::runtime_error("T and DT must be positive.");
    }

    pinocchio::urdf::buildModel(urdf_path, pin_model_);
    pin_model_ptr_ = std::make_shared<pinocchio::Model>(pin_model_);
    eeid_ = pin_model_.getFrameId(ee_frame_name);
    if (eeid_ >= pin_model_.frames.size()) {
      throw std::runtime_error("End-effector frame not found: " + ee_frame_name);
    }

    state_ = std::make_shared<crocoddyl::StateMultibody>(pin_model_ptr_);
    actuation_ = std::make_shared<crocoddyl::ActuationModelFull>(state_);

    goal_residual_ = std::make_shared<crocoddyl::ResidualModelFrameTranslation>(
        state_, eeid_, Eigen::Vector3d::Zero());
    auto goal_cost =
        std::make_shared<crocoddyl::CostModelResidual>(state_, goal_residual_);
    auto state_cost = std::make_shared<crocoddyl::CostModelResidual>(
        state_, std::make_shared<crocoddyl::ResidualModelState>(state_));
    auto ctrl_cost = std::make_shared<crocoddyl::CostModelResidual>(
        state_, std::make_shared<crocoddyl::ResidualModelControl>(state_));

    auto running_cost_sum = std::make_shared<crocoddyl::CostModelSum>(state_);
    auto terminal_cost_sum = std::make_shared<crocoddyl::CostModelSum>(state_);

    running_cost_sum->addCost("ee_track", goal_cost, w_goal_running);
    running_cost_sum->addCost("state_reg", state_cost, w_state);
    running_cost_sum->addCost("ctrl_reg", ctrl_cost, w_ctrl);

    terminal_cost_sum->addCost("ee_track", goal_cost, w_goal_terminal);
    terminal_cost_sum->addCost("state_reg", state_cost, w_state);
    terminal_cost_sum->addCost("ctrl_reg", ctrl_cost, w_ctrl);

    auto running_dam =
        std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
            state_, actuation_, running_cost_sum);
    auto terminal_dam =
        std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
            state_, actuation_, terminal_cost_sum);

    auto running_model =
        std::make_shared<crocoddyl::IntegratedActionModelEuler>(running_dam, DT_);
    auto terminal_model =
        std::make_shared<crocoddyl::IntegratedActionModelEuler>(terminal_dam, 0.0);

    nx_ = state_->get_nx();
    nu_ = actuation_->get_nu();
    nq_ = pin_model_.nq;
    nv_ = pin_model_.nv;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx_);
    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
        static_cast<std::size_t>(T_), running_model);
    problem_ = std::make_shared<crocoddyl::ShootingProblem>(x0, running_models,
                                                             terminal_model);
    ddp_ = std::make_unique<crocoddyl::SolverDDP>(problem_);
  }

  int T() const { return T_; }
  double DT() const { return DT_; }
  int nq() const { return nq_; }
  int nv() const { return nv_; }
  int nx() const { return nx_; }
  int nu() const { return nu_; }

  void reset_warm_start() {
    xs_warm_.clear();
    us_warm_.clear();
    warm_started_ = false;
  }

  bool solve_once(const Eigen::VectorXd& x0, const Eigen::Vector3d& x_goal,
                  int max_iter, double* solve_time = nullptr) {
    if (max_iter <= 0) {
      throw std::runtime_error("max_iter must be positive.");
    }
    if (x0.size() != nx_) {
      throw std::runtime_error("x0 shape mismatch.");
    }

    problem_->set_x0(x0);
    goal_residual_->set_reference(x_goal);

    if (!warm_started_) {
      xs_warm_ = make_xs_warm(x0, T_);
      us_warm_ = make_us_warm(T_, nu_);
      warm_started_ = true;
    } else {
      if (xs_warm_.size() != static_cast<std::size_t>(T_ + 1) ||
          us_warm_.size() != static_cast<std::size_t>(T_)) {
        xs_warm_ = make_xs_warm(x0, T_);
        us_warm_ = make_us_warm(T_, nu_);
      } else {
        xs_warm_.front() = x0;
      }
    }

    const auto t0 = Clock::now();
    const bool solved =
        ddp_->solve(xs_warm_, us_warm_, static_cast<std::size_t>(max_iter));
    const auto t1 = Clock::now();

    if (solve_time != nullptr) {
      *solve_time =
          std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
              .count();
    }

    xs_warm_ = ddp_->get_xs();
    us_warm_ = ddp_->get_us();
    return solved;
  }

  const std::vector<Eigen::VectorXd>& xs() const { return xs_warm_; }
  const std::vector<Eigen::VectorXd>& us() const { return us_warm_; }

  py::dict solve(
      py::array_t<double, py::array::c_style | py::array::forcecast> x0_in,
      py::array_t<double, py::array::c_style | py::array::forcecast> x_goal_in,
      int max_iter = 50) {
    const Eigen::VectorXd x0 = parse_vector(x0_in, nx_, "x0");
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
    out["xs"] = vectors_to_numpy_2d(xs_warm_, nx_);
    out["us"] = vectors_to_numpy_2d(us_warm_, nu_);
    return out;
  }

 private:
  pinocchio::Model pin_model_;
  std::shared_ptr<pinocchio::Model> pin_model_ptr_;
  pinocchio::FrameIndex eeid_{0};

  std::shared_ptr<crocoddyl::StateMultibody> state_;
  std::shared_ptr<crocoddyl::ActuationModelFull> actuation_;
  std::shared_ptr<crocoddyl::ResidualModelFrameTranslation> goal_residual_;
  std::shared_ptr<crocoddyl::ShootingProblem> problem_;
  std::unique_ptr<crocoddyl::SolverDDP> ddp_;

  std::vector<Eigen::VectorXd> xs_warm_;
  std::vector<Eigen::VectorXd> us_warm_;

  int T_{40};
  double DT_{0.01};
  int nq_{0};
  int nv_{0};
  int nx_{0};
  int nu_{0};
  bool warm_started_{false};
};

class RenderStepController {
 public:
  RenderStepController(const std::string& urdf_path, int K = 6,
                       double P = 1000.0, double D = 50.0)
      : K_(K), P_(P), D_(D) {
    if (K_ <= 0) {
      throw std::runtime_error("K must be positive.");
    }
    pinocchio::urdf::buildModel(urdf_path, pin_model_);
    pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);
    nq_ = pin_model_.nq;
    nv_ = pin_model_.nv;
    zeros_ = Eigen::VectorXd::Zero(nv_);
    xf_ = Eigen::VectorXd::Zero(nq_ + nv_);
  }

  int nq() const { return nq_; }
  int nv() const { return nv_; }
  int nx() const { return nq_ + nv_; }
  int step_index() const { return i_; }
  int plan_index() const { return plan_index_; }

  void set_gains(double P, double D) {
    P_ = P;
    D_ = D;
  }

  void set_downsample(int K) {
    if (K <= 0) {
      throw std::runtime_error("K must be positive.");
    }
    K_ = K;
  }

  void reset_phase() { i_ = 0; }

  void reset_plan(
      py::array_t<double, py::array::c_style | py::array::forcecast> xs_in) {
    const auto xs = xs_in.unchecked<2>();
    if (xs.shape(1) != nx()) {
      throw std::runtime_error("xs shape mismatch: second dim must be nq+nv.");
    }
    if (xs.shape(0) <= 0) {
      throw std::runtime_error("xs must have at least one row.");
    }

    xs_plan_.resize(xs.shape(0), nx());
    for (ssize_t r = 0; r < xs.shape(0); ++r) {
      for (int c = 0; c < nx(); ++c) {
        xs_plan_(r, c) = xs(r, c);
      }
    }

    i_ = 0;
    plan_index_ = 0;
    xf_ = xs_plan_.row(0).transpose();
  }

  py::array_t<double> compute_tau(
      py::array_t<double, py::array::c_style | py::array::forcecast> q_in,
      py::array_t<double, py::array::c_style | py::array::forcecast> v_in) {
    if (xs_plan_.rows() == 0) {
      throw std::runtime_error("Plan not initialized. Call reset_plan(xs) first.");
    }

    const auto q = q_in.unchecked<1>();
    const auto v = v_in.unchecked<1>();
    if (q.shape(0) != nq_ || v.shape(0) != nv_) {
      throw std::runtime_error("q/v shape mismatch.");
    }

    if (i_ % K_ == 0) {
      int j = i_ / K_;
      j = std::min(j, static_cast<int>(xs_plan_.rows()) - 1);
      plan_index_ = j;
      xf_ = xs_plan_.row(j).transpose();
    }

    Eigen::VectorXd q_vec(nq_);
    Eigen::VectorXd v_vec(nv_);
    for (int k = 0; k < nq_; ++k) {
      q_vec[k] = q(k);
    }
    for (int k = 0; k < nv_; ++k) {
      v_vec[k] = v(k);
    }

    const Eigen::VectorXd q_des = xf_.head(nq_);
    const Eigen::VectorXd dq_des = xf_.tail(nv_);
    const Eigen::VectorXd tau =
        P_ * (q_des - q_vec) + D_ * (dq_des - v_vec) +
        pinocchio::rnea(pin_model_, *pin_data_, q_vec, zeros_, zeros_);

    ++i_;

    py::array_t<double> tau_out(nv_);
    auto tau_out_m = tau_out.mutable_unchecked<1>();
    for (int k = 0; k < nv_; ++k) {
      tau_out_m(k) = tau[k];
    }
    return tau_out;
  }

 private:
  pinocchio::Model pin_model_;
  std::unique_ptr<pinocchio::Data> pin_data_;
  Eigen::VectorXd zeros_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      xs_plan_;
  Eigen::VectorXd xf_;
  int nq_{0};
  int nv_{0};
  int K_{6};
  int i_{0};
  int plan_index_{0};
  double P_{1000.0};
  double D_{50.0};
};

class RenderTools {
 public:
  RenderTools(const std::string& urdf_path, const std::string& ee_frame_name) {
    pinocchio::urdf::buildModel(urdf_path, pin_model_);
    pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);
    eeid_ = pin_model_.getFrameId(ee_frame_name);
    if (eeid_ >= pin_model_.frames.size()) {
      throw std::runtime_error("End-effector frame not found: " + ee_frame_name);
    }
    nq_ = pin_model_.nq;

    py::module_ mujoco = py::module_::import("mujoco");
    auto mjt_geom = mujoco.attr("mjtGeom");
    geom_sphere_ = mjt_geom.attr("mjGEOM_SPHERE");
    geom_arrow_ = mjt_geom.attr("mjGEOM_ARROW");
  }

  int nq() const { return nq_; }

  py::array_t<double> batch_trajectory_fk(
      py::array_t<double, py::array::c_style | py::array::forcecast> xs_in) {
    const auto xs = xs_in.unchecked<2>();
    if (xs.shape(0) <= 0) {
      throw std::runtime_error("xs must have at least one row.");
    }
    if (xs.shape(1) < nq_) {
      throw std::runtime_error("xs second dim must be >= nq.");
    }

    const ssize_t n = xs.shape(0);
    py::array out_raw(py::dtype::of<double>(), {py::ssize_t(3), n});
    py::array_t<double> out = out_raw.cast<py::array_t<double>>();
    auto out_m = out.mutable_unchecked<2>();

    Eigen::VectorXd q(nq_);
    for (ssize_t i = 0; i < n; ++i) {
      for (int k = 0; k < nq_; ++k) {
        q[k] = xs(i, k);
      }
      pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
      pinocchio::updateFramePlacements(pin_model_, *pin_data_);
      const Eigen::Vector3d p = pin_data_->oMf[eeid_].translation();
      out_m(0, i) = p[0];
      out_m(1, i) = p[1];
      out_m(2, i) = p[2];
    }
    return out;
  }

  py::array_t<double> ee_position(
      py::array_t<double, py::array::c_style | py::array::forcecast> q_in) {
    Eigen::VectorXd q = parse_q(q_in);
    pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
    pinocchio::updateFramePlacements(pin_model_, *pin_data_);
    const Eigen::Vector3d p = pin_data_->oMf[eeid_].translation();

    py::array_t<double> out(3);
    auto out_m = out.mutable_unchecked<1>();
    out_m(0) = p[0];
    out_m(1) = p[1];
    out_m(2) = p[2];
    return out;
  }

  void add_marker(py::object viewer,
                  py::array_t<double, py::array::c_style | py::array::forcecast>
                      pos_in,
                  double size = 0.02,
                  py::tuple color = py::make_tuple(1.0, 0.0, 0.0, 1.0)) {
    const auto pos = parse_vec3(pos_in);
    const auto rgba = parse_rgba(color);
    py::list markers;
    markers.append(make_sphere_marker(pos, size, rgba));
    submit_markers(viewer, markers);
  }

  void add_markers(
      py::object viewer,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          positions_in,
      double size = 0.02,
      py::tuple color = py::make_tuple(1.0, 0.0, 0.0, 1.0)) {
    const auto rgba = parse_rgba(color);
    py::list markers = make_sphere_markers(positions_in, size, rgba);
    submit_markers(viewer, markers);
  }

  void draw_trajectory(
      py::object viewer,
      py::array_t<double, py::array::c_style | py::array::forcecast> traj_in,
      py::tuple color = py::make_tuple(0.0, 0.0, 1.0, 1.0),
      double size = 0.02) {
    add_markers(viewer, traj_in, size, color);
  }

  void draw_ee_frame(
      py::object viewer,
      py::array_t<double, py::array::c_style | py::array::forcecast> q_in,
      double size = 0.08) {
    Eigen::VectorXd q = parse_q(q_in);
    pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
    pinocchio::updateFramePlacements(pin_model_, *pin_data_);
    const auto& M = pin_data_->oMf[eeid_];
    const Eigen::Vector3d pos = M.translation();
    const Eigen::Matrix3d R = M.rotation();

    const double axis_radius = std::max(0.002, 0.125 * size);
    const double axis_length = std::max(0.02, 2.5 * size);

    const Eigen::Matrix3d mx =
        R * (Eigen::Matrix3d() << 0, 0, 1, 0, 1, 0, -1, 0, 0).finished();
    const Eigen::Matrix3d my =
        R * (Eigen::Matrix3d() << 1, 0, 0, 0, 0, 1, 0, -1, 0).finished();
    const Eigen::Matrix3d mz = R;

    py::list markers;
    markers.append(make_arrow_marker(pos, mx, {1.0, 0.0, 0.0, 1.0}, axis_radius,
                                     axis_length));
    markers.append(make_arrow_marker(pos, my, {0.0, 1.0, 0.0, 1.0}, axis_radius,
                                     axis_length));
    markers.append(make_arrow_marker(pos, mz, {0.0, 0.0, 1.0, 1.0}, axis_radius,
                                     axis_length));
    submit_markers(viewer, markers);
  }

 private:
  Eigen::VectorXd parse_q(
      py::array_t<double, py::array::c_style | py::array::forcecast> q_in) const {
    const auto q = q_in.unchecked<1>();
    if (q.shape(0) < nq_) {
      throw std::runtime_error("q length must be >= nq.");
    }
    Eigen::VectorXd out(nq_);
    for (int i = 0; i < nq_; ++i) {
      out[i] = q(i);
    }
    return out;
  }

  std::array<double, 3> parse_vec3(
      py::array_t<double, py::array::c_style | py::array::forcecast> v_in) const {
    const auto v = v_in.unchecked<1>();
    if (v.shape(0) != 3) {
      throw std::runtime_error("pos must contain exactly 3 values.");
    }
    return {v(0), v(1), v(2)};
  }

  std::array<double, 4> parse_rgba(py::tuple rgba_in) const {
    if (py::len(rgba_in) != 4) {
      throw std::runtime_error("color must have 4 values (r,g,b,a).");
    }
    return {rgba_in[0].cast<double>(), rgba_in[1].cast<double>(),
            rgba_in[2].cast<double>(), rgba_in[3].cast<double>()};
  }

  py::array_t<double> mat_to_numpy(const Eigen::Matrix3d& M) const {
    py::array out_raw(py::dtype::of<double>(), {py::ssize_t(3), py::ssize_t(3)});
    py::array_t<double> out = out_raw.cast<py::array_t<double>>();
    auto out_m = out.mutable_unchecked<2>();
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        out_m(r, c) = M(r, c);
      }
    }
    return out;
  }

  py::dict make_sphere_marker(const std::array<double, 3>& pos, double size,
                              const std::array<double, 4>& rgba) const {
    py::dict marker;
    marker["pos"] = py::make_tuple(pos[0], pos[1], pos[2]);
    marker["size"] = py::make_tuple(size, size, size);
    marker["rgba"] = py::make_tuple(rgba[0], rgba[1], rgba[2], rgba[3]);
    marker["type"] = geom_sphere_;
    return marker;
  }

  py::list make_sphere_markers(
      py::array_t<double, py::array::c_style | py::array::forcecast> points_in,
      double size, const std::array<double, 4>& rgba) const {
    const auto pts = points_in.unchecked<2>();
    bool shape_3_n = false;
    if (pts.shape(0) == 3) {
      shape_3_n = true;
    } else if (pts.shape(1) == 3) {
      shape_3_n = false;
    } else {
      throw std::runtime_error("positions must be shape (3,N) or (N,3).");
    }

    const ssize_t n = shape_3_n ? pts.shape(1) : pts.shape(0);
    py::list markers;
    for (ssize_t i = 0; i < n; ++i) {
      const std::array<double, 3> pos = {
          shape_3_n ? pts(0, i) : pts(i, 0), shape_3_n ? pts(1, i) : pts(i, 1),
          shape_3_n ? pts(2, i) : pts(i, 2)};
      markers.append(make_sphere_marker(pos, size, rgba));
    }
    return markers;
  }

  py::dict make_arrow_marker(const Eigen::Vector3d& pos, const Eigen::Matrix3d& mat,
                             const std::array<double, 4>& rgba,
                             double axis_radius, double axis_length) const {
    py::dict marker;
    marker["pos"] = py::make_tuple(pos[0], pos[1], pos[2]);
    marker["size"] = py::make_tuple(axis_radius, axis_radius, axis_length);
    marker["rgba"] = py::make_tuple(rgba[0], rgba[1], rgba[2], rgba[3]);
    marker["type"] = geom_arrow_;
    marker["mat"] = mat_to_numpy(mat);
    return marker;
  }

  void submit_markers(py::object viewer, const py::list& markers) const {
    if (py::hasattr(viewer, "add_markers")) {
      viewer.attr("add_markers")(markers);
      return;
    }

    for (py::handle marker_h : markers) {
      py::dict marker = py::reinterpret_borrow<py::dict>(marker_h);
      if (marker.contains("mat")) {
        viewer.attr("add_marker")(
            py::arg("pos") = marker["pos"], py::arg("size") = marker["size"],
            py::arg("rgba") = marker["rgba"], py::arg("type") = marker["type"],
            py::arg("mat") = marker["mat"]);
      } else {
        viewer.attr("add_marker")(
            py::arg("pos") = marker["pos"], py::arg("size") = marker["size"],
            py::arg("rgba") = marker["rgba"], py::arg("type") = marker["type"]);
      }
    }
  }

  pinocchio::Model pin_model_;
  std::unique_ptr<pinocchio::Data> pin_data_;
  pinocchio::FrameIndex eeid_{0};
  int nq_{0};
  py::object geom_sphere_;
  py::object geom_arrow_;
};

}  // namespace

py::dict run_nmpc(
    const std::string& urdf_path, const std::string& mjcf_path,
    const std::string& ee_frame_name,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_goal_in,
    int T = 40, double DT = 0.01, int H = 2, int K = 6, double sim_time = 20.0,
    double P = 1000.0, double D = 50.0, int max_iter = 50,
    bool verbose = true) {
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

PYBIND11_MODULE(nmpc_native, m) {
  m.doc() = "Native C++ NMPC loop and render helpers for xArm6";

  py::class_<DDPReachSolver>(m, "DDPReachSolver")
      .def(py::init<const std::string&, const std::string&, int, double, double,
                    double, double, double>(),
           py::arg("urdf_path"), py::arg("ee_frame_name") = "link6",
           py::arg("T") = 40, py::arg("DT") = 0.01,
           py::arg("w_goal_running") = 1e4, py::arg("w_goal_terminal") = 1e7,
           py::arg("w_state") = 1e-1, py::arg("w_ctrl") = 1e-5)
      .def("solve", &DDPReachSolver::solve, py::arg("x0"), py::arg("x_goal"),
           py::arg("max_iter") = 50)
      .def("reset_warm_start", &DDPReachSolver::reset_warm_start)
      .def_property_readonly("T", &DDPReachSolver::T)
      .def_property_readonly("DT", &DDPReachSolver::DT)
      .def_property_readonly("nq", &DDPReachSolver::nq)
      .def_property_readonly("nv", &DDPReachSolver::nv)
      .def_property_readonly("nx", &DDPReachSolver::nx)
      .def_property_readonly("nu", &DDPReachSolver::nu);

  py::class_<RenderStepController>(m, "RenderStepController")
      .def(py::init<const std::string&, int, double, double>(),
           py::arg("urdf_path"), py::arg("K") = 6, py::arg("P") = 1000.0,
           py::arg("D") = 50.0)
      .def("reset_plan", &RenderStepController::reset_plan, py::arg("xs"))
      .def("compute_tau", &RenderStepController::compute_tau, py::arg("q"),
           py::arg("v"))
      .def("set_gains", &RenderStepController::set_gains, py::arg("P"),
           py::arg("D"))
      .def("set_downsample", &RenderStepController::set_downsample,
           py::arg("K"))
      .def("reset_phase", &RenderStepController::reset_phase)
      .def_property_readonly("nq", &RenderStepController::nq)
      .def_property_readonly("nv", &RenderStepController::nv)
      .def_property_readonly("nx", &RenderStepController::nx)
      .def_property_readonly("step_index", &RenderStepController::step_index)
      .def_property_readonly("plan_index", &RenderStepController::plan_index);

  py::class_<RenderTools>(m, "RenderTools")
      .def(py::init<const std::string&, const std::string&>(),
           py::arg("urdf_path"), py::arg("ee_frame_name") = "link6")
      .def("batch_trajectory_fk", &RenderTools::batch_trajectory_fk,
           py::arg("xs"))
      .def("ee_position", &RenderTools::ee_position, py::arg("q"))
      .def("add_marker", &RenderTools::add_marker, py::arg("viewer"),
           py::arg("pos"), py::arg("size") = 0.02,
           py::arg("color") = py::make_tuple(1.0, 0.0, 0.0, 1.0))
      .def("add_markers", &RenderTools::add_markers, py::arg("viewer"),
           py::arg("positions"), py::arg("size") = 0.02,
           py::arg("color") = py::make_tuple(1.0, 0.0, 0.0, 1.0))
      .def("draw_trajectory", &RenderTools::draw_trajectory,
           py::arg("viewer"), py::arg("trajectory"),
           py::arg("color") = py::make_tuple(0.0, 0.0, 1.0, 1.0),
           py::arg("size") = 0.02)
      .def("draw_ee_frame", &RenderTools::draw_ee_frame, py::arg("viewer"),
           py::arg("q"), py::arg("size") = 0.08)
      .def_property_readonly("nq", &RenderTools::nq);

  m.def(
      "run_nmpc", &run_nmpc, py::arg("urdf_path"), py::arg("mjcf_path"),
      py::arg("ee_frame_name"), py::arg("x_goal"), py::arg("T") = 40,
      py::arg("DT") = 0.01, py::arg("H") = 2, py::arg("K") = 6,
      py::arg("sim_time") = 20.0, py::arg("P") = 1000.0,
      py::arg("D") = 50.0, py::arg("max_iter") = 50,
      py::arg("verbose") = true);
}
