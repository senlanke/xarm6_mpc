#include "nmpc_native/render_step_controller.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace nmpc_native {

struct RenderStepController::Impl {
  pinocchio::Model pin_model;
  std::unique_ptr<pinocchio::Data> pin_data;
  Eigen::VectorXd zeros;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xs_plan;
  Eigen::VectorXd xf;
  int nq{0};
  int nv{0};
  int K{6};
  int i{0};
  int plan_index{0};
  double P{1000.0};
  double D{50.0};
};

RenderStepController::RenderStepController(const std::string& urdf_path, int K,
                                           double P, double D)
    : impl_(std::make_unique<Impl>()) {
  if (K <= 0) {
    throw std::runtime_error("K must be positive.");
  }
  impl_->K = K;
  impl_->P = P;
  impl_->D = D;

  pinocchio::urdf::buildModel(urdf_path, impl_->pin_model);
  impl_->pin_data = std::make_unique<pinocchio::Data>(impl_->pin_model);
  impl_->nq = impl_->pin_model.nq;
  impl_->nv = impl_->pin_model.nv;
  impl_->zeros = Eigen::VectorXd::Zero(impl_->nv);
  impl_->xf = Eigen::VectorXd::Zero(impl_->nq + impl_->nv);
}

RenderStepController::~RenderStepController() = default;
RenderStepController::RenderStepController(RenderStepController&&) noexcept =
    default;
RenderStepController& RenderStepController::operator=(
    RenderStepController&&) noexcept = default;

int RenderStepController::nq() const { return impl_->nq; }
int RenderStepController::nv() const { return impl_->nv; }
int RenderStepController::nx() const { return impl_->nq + impl_->nv; }
int RenderStepController::step_index() const { return impl_->i; }
int RenderStepController::plan_index() const { return impl_->plan_index; }

void RenderStepController::set_gains(double P, double D) {
  impl_->P = P;
  impl_->D = D;
}

void RenderStepController::set_downsample(int K) {
  if (K <= 0) {
    throw std::runtime_error("K must be positive.");
  }
  impl_->K = K;
}

void RenderStepController::reset_phase() { impl_->i = 0; }

void RenderStepController::reset_plan(
    py::array_t<double, py::array::c_style | py::array::forcecast> xs_in) {
  const auto xs = xs_in.unchecked<2>();
  if (xs.shape(1) != nx()) {
    throw std::runtime_error("xs shape mismatch: second dim must be nq+nv.");
  }
  if (xs.shape(0) <= 0) {
    throw std::runtime_error("xs must have at least one row.");
  }

  impl_->xs_plan.resize(xs.shape(0), nx());
  for (ssize_t r = 0; r < xs.shape(0); ++r) {
    for (int c = 0; c < nx(); ++c) {
      impl_->xs_plan(r, c) = xs(r, c);
    }
  }

  impl_->i = 0;
  impl_->plan_index = 0;
  impl_->xf = impl_->xs_plan.row(0).transpose();
}

py::array_t<double> RenderStepController::compute_tau(
    py::array_t<double, py::array::c_style | py::array::forcecast> q_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> v_in) {
  if (impl_->xs_plan.rows() == 0) {
    throw std::runtime_error("Plan not initialized. Call reset_plan(xs) first.");
  }

  const auto q = q_in.unchecked<1>();
  const auto v = v_in.unchecked<1>();
  if (q.shape(0) != impl_->nq || v.shape(0) != impl_->nv) {
    throw std::runtime_error("q/v shape mismatch.");
  }

  if (impl_->i % impl_->K == 0) {
    int j = impl_->i / impl_->K;
    j = std::min(j, static_cast<int>(impl_->xs_plan.rows()) - 1);
    impl_->plan_index = j;
    impl_->xf = impl_->xs_plan.row(j).transpose();
  }

  Eigen::VectorXd q_vec(impl_->nq);
  Eigen::VectorXd v_vec(impl_->nv);
  for (int k = 0; k < impl_->nq; ++k) {
    q_vec[k] = q(k);
  }
  for (int k = 0; k < impl_->nv; ++k) {
    v_vec[k] = v(k);
  }

  const Eigen::VectorXd q_des = impl_->xf.head(impl_->nq);
  const Eigen::VectorXd dq_des = impl_->xf.tail(impl_->nv);
  const Eigen::VectorXd tau = impl_->P * (q_des - q_vec) +
                              impl_->D * (dq_des - v_vec) +
                              pinocchio::rnea(impl_->pin_model, *impl_->pin_data,
                                              q_vec, impl_->zeros, impl_->zeros);

  ++impl_->i;

  py::array_t<double> tau_out(impl_->nv);
  auto tau_out_m = tau_out.mutable_unchecked<1>();
  for (int k = 0; k < impl_->nv; ++k) {
    tau_out_m(k) = tau[k];
  }
  return tau_out;
}

}  // namespace nmpc_native
