#include "nmpc_native/render_tools.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace nmpc_native {

struct RenderTools::Impl {
  pinocchio::Model pin_model;
  std::unique_ptr<pinocchio::Data> pin_data;
  pinocchio::FrameIndex eeid{0};
  int nq{0};
  py::object geom_sphere;
  py::object geom_arrow;
};

namespace {

Eigen::VectorXd parse_q(
    int nq,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_in) {
  const auto q = q_in.unchecked<1>();
  if (q.shape(0) < nq) {
    throw std::runtime_error("q length must be >= nq.");
  }
  Eigen::VectorXd out(nq);
  for (int i = 0; i < nq; ++i) {
    out[i] = q(i);
  }
  return out;
}

std::array<double, 3> parse_vec3(
    py::array_t<double, py::array::c_style | py::array::forcecast> v_in) {
  const auto v = v_in.unchecked<1>();
  if (v.shape(0) != 3) {
    throw std::runtime_error("pos must contain exactly 3 values.");
  }
  return {v(0), v(1), v(2)};
}

std::array<double, 4> parse_rgba(py::tuple rgba_in) {
  if (py::len(rgba_in) != 4) {
    throw std::runtime_error("color must have 4 values (r,g,b,a).");
  }
  return {rgba_in[0].cast<double>(), rgba_in[1].cast<double>(),
          rgba_in[2].cast<double>(), rgba_in[3].cast<double>()};
}

py::array_t<double> mat_to_numpy(const Eigen::Matrix3d& M) {
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

py::dict make_sphere_marker(py::object geom_sphere,
                            const std::array<double, 3>& pos, double size,
                            const std::array<double, 4>& rgba) {
  py::dict marker;
  marker["pos"] = py::make_tuple(pos[0], pos[1], pos[2]);
  marker["size"] = py::make_tuple(size, size, size);
  marker["rgba"] = py::make_tuple(rgba[0], rgba[1], rgba[2], rgba[3]);
  marker["type"] = geom_sphere;
  return marker;
}

py::list make_sphere_markers(
    py::object geom_sphere,
    py::array_t<double, py::array::c_style | py::array::forcecast> points_in,
    double size, const std::array<double, 4>& rgba) {
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
        shape_3_n ? pts(0, i) : pts(i, 0),
        shape_3_n ? pts(1, i) : pts(i, 1),
        shape_3_n ? pts(2, i) : pts(i, 2)};
    markers.append(make_sphere_marker(geom_sphere, pos, size, rgba));
  }
  return markers;
}

py::dict make_arrow_marker(py::object geom_arrow,
                           const Eigen::Vector3d& pos,
                           const Eigen::Matrix3d& mat,
                           const std::array<double, 4>& rgba,
                           double axis_radius, double axis_length) {
  py::dict marker;
  marker["pos"] = py::make_tuple(pos[0], pos[1], pos[2]);
  marker["size"] = py::make_tuple(axis_radius, axis_radius, axis_length);
  marker["rgba"] = py::make_tuple(rgba[0], rgba[1], rgba[2], rgba[3]);
  marker["type"] = geom_arrow;
  marker["mat"] = mat_to_numpy(mat);
  return marker;
}

void submit_markers(py::object viewer, const py::list& markers) {
  if (py::hasattr(viewer, "add_markers")) {
    viewer.attr("add_markers")(markers);
    return;
  }

  for (py::handle marker_h : markers) {
    py::dict marker = py::reinterpret_borrow<py::dict>(marker_h);
    if (marker.contains("mat")) {
      viewer.attr("add_marker")(py::arg("pos") = marker["pos"],
                                py::arg("size") = marker["size"],
                                py::arg("rgba") = marker["rgba"],
                                py::arg("type") = marker["type"],
                                py::arg("mat") = marker["mat"]);
    } else {
      viewer.attr("add_marker")(py::arg("pos") = marker["pos"],
                                py::arg("size") = marker["size"],
                                py::arg("rgba") = marker["rgba"],
                                py::arg("type") = marker["type"]);
    }
  }
}

}  // namespace

RenderTools::RenderTools(const std::string& urdf_path,
                         const std::string& ee_frame_name)
    : impl_(std::make_unique<Impl>()) {
  pinocchio::urdf::buildModel(urdf_path, impl_->pin_model);
  impl_->pin_data = std::make_unique<pinocchio::Data>(impl_->pin_model);
  impl_->eeid = impl_->pin_model.getFrameId(ee_frame_name);
  if (impl_->eeid >= impl_->pin_model.frames.size()) {
    throw std::runtime_error("End-effector frame not found: " + ee_frame_name);
  }
  impl_->nq = impl_->pin_model.nq;

  py::module_ mujoco = py::module_::import("mujoco");
  auto mjt_geom = mujoco.attr("mjtGeom");
  impl_->geom_sphere = mjt_geom.attr("mjGEOM_SPHERE");
  impl_->geom_arrow = mjt_geom.attr("mjGEOM_ARROW");
}

RenderTools::~RenderTools() = default;
RenderTools::RenderTools(RenderTools&&) noexcept = default;
RenderTools& RenderTools::operator=(RenderTools&&) noexcept = default;

int RenderTools::nq() const { return impl_->nq; }

py::array_t<double> RenderTools::batch_trajectory_fk(
    py::array_t<double, py::array::c_style | py::array::forcecast> xs_in) {
  const auto xs = xs_in.unchecked<2>();
  if (xs.shape(0) <= 0) {
    throw std::runtime_error("xs must have at least one row.");
  }
  if (xs.shape(1) < impl_->nq) {
    throw std::runtime_error("xs second dim must be >= nq.");
  }

  const ssize_t n = xs.shape(0);
  py::array out_raw(py::dtype::of<double>(), {py::ssize_t(3), n});
  py::array_t<double> out = out_raw.cast<py::array_t<double>>();
  auto out_m = out.mutable_unchecked<2>();

  Eigen::VectorXd q(impl_->nq);
  for (ssize_t i = 0; i < n; ++i) {
    for (int k = 0; k < impl_->nq; ++k) {
      q[k] = xs(i, k);
    }
    pinocchio::forwardKinematics(impl_->pin_model, *impl_->pin_data, q);
    pinocchio::updateFramePlacements(impl_->pin_model, *impl_->pin_data);
    const Eigen::Vector3d p = impl_->pin_data->oMf[impl_->eeid].translation();
    out_m(0, i) = p[0];
    out_m(1, i) = p[1];
    out_m(2, i) = p[2];
  }
  return out;
}

py::array_t<double> RenderTools::ee_position(
    py::array_t<double, py::array::c_style | py::array::forcecast> q_in) {
  Eigen::VectorXd q = parse_q(impl_->nq, q_in);
  pinocchio::forwardKinematics(impl_->pin_model, *impl_->pin_data, q);
  pinocchio::updateFramePlacements(impl_->pin_model, *impl_->pin_data);
  const Eigen::Vector3d p = impl_->pin_data->oMf[impl_->eeid].translation();

  py::array_t<double> out(3);
  auto out_m = out.mutable_unchecked<1>();
  out_m(0) = p[0];
  out_m(1) = p[1];
  out_m(2) = p[2];
  return out;
}

void RenderTools::add_marker(
    py::object viewer,
    py::array_t<double, py::array::c_style | py::array::forcecast> pos_in,
    double size, py::tuple color) {
  const auto pos = parse_vec3(pos_in);
  const auto rgba = parse_rgba(color);
  py::list markers;
  markers.append(make_sphere_marker(impl_->geom_sphere, pos, size, rgba));
  submit_markers(viewer, markers);
}

void RenderTools::add_markers(
    py::object viewer,
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_in,
    double size, py::tuple color) {
  const auto rgba = parse_rgba(color);
  py::list markers =
      make_sphere_markers(impl_->geom_sphere, positions_in, size, rgba);
  submit_markers(viewer, markers);
}

void RenderTools::draw_trajectory(
    py::object viewer,
    py::array_t<double, py::array::c_style | py::array::forcecast> traj_in,
    py::tuple color, double size) {
  add_markers(viewer, traj_in, size, color);
}

void RenderTools::draw_ee_frame(
    py::object viewer,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_in,
    double size) {
  Eigen::VectorXd q = parse_q(impl_->nq, q_in);
  pinocchio::forwardKinematics(impl_->pin_model, *impl_->pin_data, q);
  pinocchio::updateFramePlacements(impl_->pin_model, *impl_->pin_data);
  const auto& M = impl_->pin_data->oMf[impl_->eeid];
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
  markers.append(
      make_arrow_marker(impl_->geom_arrow, pos, mx, {1.0, 0.0, 0.0, 1.0},
                        axis_radius,
                        axis_length));
  markers.append(
      make_arrow_marker(impl_->geom_arrow, pos, my, {0.0, 1.0, 0.0, 1.0},
                        axis_radius,
                        axis_length));
  markers.append(
      make_arrow_marker(impl_->geom_arrow, pos, mz, {0.0, 0.0, 1.0, 1.0},
                        axis_radius,
                        axis_length));
  submit_markers(viewer, markers);
}

}  // namespace nmpc_native
