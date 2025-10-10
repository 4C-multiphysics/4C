// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_interpolation.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_determinant.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_mat_fourier.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_solid_3D_ele_surface.hpp"

FOUR_C_NAMESPACE_OPEN


double Discret::Elements::SolidSurface::estimate_nitsche_trace_max_eigenvalue(
    const std::vector<double>& parent_disp)
{
  // call the implementation that is dependent on scalars with an empty scalar vector
  return estimate_nitsche_trace_max_eigenvalue(parent_disp, {});
}

double Discret::Elements::SolidSurface::estimate_nitsche_trace_max_eigenvalue(
    const std::vector<double>& parent_disp, const std::vector<double>& parent_scalar)
{
  switch (parent_element()->shape())
  {
    case Core::FE::CellType::hex8:
      if (shape() == Core::FE::CellType::quad4)
      {
        return estimate_nitsche_trace_max_eigenvalue<Core::FE::CellType::hex8,
            Core::FE::CellType::quad4>(parent_disp, parent_scalar);
      }
      else
      {
        FOUR_C_THROW("how can an hex8 element have a surface that is not quad4?");
      }
    case Core::FE::CellType::hex27:
      return estimate_nitsche_trace_max_eigenvalue<Core::FE::CellType::hex27,
          Core::FE::CellType::quad9>(parent_disp, parent_scalar);
    case Core::FE::CellType::tet4:
      return estimate_nitsche_trace_max_eigenvalue<Core::FE::CellType::tet4,
          Core::FE::CellType::tri3>(parent_disp, parent_scalar);
    case Core::FE::CellType::nurbs27:
      return estimate_nitsche_trace_max_eigenvalue<Core::FE::CellType::nurbs27,
          Core::FE::CellType::nurbs9>(parent_disp, parent_scalar);
    default:
      FOUR_C_THROW("parent shape not implemented");
  }
}


template <Core::FE::CellType dt_vol, Core::FE::CellType dt_surf>
double Discret::Elements::SolidSurface::estimate_nitsche_trace_max_eigenvalue(
    const std::vector<double>& parent_disp, const std::vector<double>& parent_scalar)
{
  const int dim = Core::FE::dim<dt_vol>;
  const int num_dof = Core::FE::num_nodes(dt_vol) * Core::FE::dim<dt_vol>;
  const int dim_image = Core::FE::num_nodes(dt_vol) * Core::FE::dim<dt_vol> -
                        Core::FE::dim<dt_vol> * (Core::FE::dim<dt_vol> + 1) / 2;

  Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3> xrefe, xcurr;

  for (int i = 0; i < parent_element()->num_node(); ++i)
  {
    for (int d = 0; d < dim; ++d)
    {
      xrefe(i, d) = parent_element()->nodes()[i]->x()[d];
      xcurr(i, d) = xrefe(i, d) + parent_disp[i * dim + d];
    }
  }

  Core::LinAlg::Matrix<num_dof, num_dof> vol, surf;

  trace_estimate_vol_matrix<dt_vol>(xrefe, xcurr, parent_scalar, vol);
  trace_estimate_surf_matrix<dt_vol, dt_surf>(xrefe, xcurr, parent_scalar, surf);

  Core::LinAlg::Matrix<num_dof, dim_image> proj, tmp;
  subspace_projector<dt_vol>(xcurr, proj);

  Core::LinAlg::Matrix<dim_image, dim_image> vol_red, surf_red;

  tmp.multiply(vol, proj);
  vol_red.multiply_tn(proj, tmp);
  tmp.multiply(surf, proj);
  surf_red.multiply_tn(proj, tmp);

  Core::LinAlg::SerialDenseMatrix vol_red_sd(
      Teuchos::View, vol_red.data(), dim_image, dim_image, dim_image);
  Core::LinAlg::SerialDenseMatrix surf_red_sd(
      Teuchos::View, surf_red.data(), dim_image, dim_image, dim_image);

  return Core::LinAlg::generalized_eigen_max_real_eigenvalue(surf_red_sd, vol_red_sd);
}


template <Core::FE::CellType dt_vol>
void Discret::Elements::SolidSurface::trace_estimate_vol_matrix(
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xrefe,
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xcurr,
    const std::vector<double>& parent_scalar,
    Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol) * 3, Core::FE::num_nodes(dt_vol) * 3>& vol)
{
  const int dim = Core::FE::dim<dt_vol>;

  double jac;
  Core::LinAlg::Tensor<double, 3, 3> defgrd;
  Core::LinAlg::SymmetricTensor<double, 3, 3> rcg;
  Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain;
  Core::LinAlg::Matrix<6, Core::FE::num_nodes(dt_vol) * 3> bop;
  Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol) * 3, 6> bc;
  Core::LinAlg::Matrix<dim, Core::FE::num_nodes(dt_vol)> N_XYZ;

  Core::FE::IntPointsAndWeights<dim> ip(Discret::Elements::DisTypeToOptGaussRule<dt_vol>::rule);

  for (int gp = 0; gp < ip.ip().nquad; ++gp)
  {
    const Core::LinAlg::Matrix<3, 1> xi(ip.ip().qxg[gp], false);
    strains<dt_vol>(xrefe, xcurr, xi, jac, defgrd, glstrain, rcg, bop, N_XYZ);

    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat{};
    Core::LinAlg::SymmetricTensor<double, 3, 3> stress{};
    Teuchos::ParameterList params;
    if (not parent_scalar.empty())
    {
      // as long as we need the parameter list to pass the information, we need to wrap it into a
      // std::shared_ptr<> as the values of a Teuchos::ParameterList have to be printable
      auto scalar_values_at_xi = std::make_shared<std::vector<double>>();
      *scalar_values_at_xi = Core::FE::interpolate_to_xi<dt_vol>(xi, parent_scalar);
      params.set("scalars", scalar_values_at_xi);
    }

    Core::LinAlg::Tensor<double, 3> xi_t = {{xi(0), xi(1), xi(2)}};
    Mat::EvaluationContext context{.total_time = nullptr,  // Do not have the time here
        .time_step_size = nullptr,                         // Do not have the time-step here
        .xi = &xi_t,
        .ref_coords = nullptr};
    std::dynamic_pointer_cast<Mat::So3Material>(parent_element()->material())
        ->evaluate(&defgrd, glstrain, params, context, stress, cmat, gp, parent_element()->id());
    bc.multiply_tn(bop, Core::LinAlg::make_stress_like_voigt_view(cmat));
    vol.multiply(ip.ip().qwgt[gp] * jac, bc, bop, 1.);
  }
}


template <Core::FE::CellType dt_vol, Core::FE::CellType dt_surf>
void Discret::Elements::SolidSurface::trace_estimate_surf_matrix(
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xrefe,
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xcurr,
    const std::vector<double>& parent_scalar,
    Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol) * 3, Core::FE::num_nodes(dt_vol) * 3>& surf)
{
  const int dim = Core::FE::dim<dt_vol>;

  Core::LinAlg::Matrix<6, 6> id4;
  for (int i = 0; i < 3; ++i) id4(i, i) = 1.0;
  for (int i = 3; i < 6; ++i) id4(i, i) = 2.0;

  Core::LinAlg::SerialDenseMatrix xrefe_surf(Core::FE::num_nodes(dt_surf), dim);
  material_configuration(xrefe_surf);

  std::vector<double> n(3);
  Core::LinAlg::Matrix<3, 1> n_v(n.data(), true);
  double detA, jac;
  Core::LinAlg::Tensor<double, 3, 3> defgrd;
  Core::LinAlg::Matrix<3, 3> nn;
  Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain, rcg;

  Core::LinAlg::Matrix<6, Core::FE::num_nodes(dt_vol) * 3> bop;
  Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol) * 3, 6> bc;
  Core::LinAlg::Matrix<dim, Core::FE::num_nodes(dt_vol)> N_XYZ;

  Core::FE::IntPointsAndWeights<dim - 1> ip(
      Discret::Elements::DisTypeToOptGaussRule<dt_surf>::rule);
  Core::LinAlg::SerialDenseMatrix deriv_surf(2, Core::FE::num_nodes(dt_surf));

  for (int gp = 0; gp < ip.ip().nquad; ++gp)
  {
    Core::FE::CollectedGaussPoints intpoints =
        Core::FE::CollectedGaussPoints(1);  // reserve just for 1 entry ...
    intpoints.append(ip.ip().qxg[gp][0], ip.ip().qxg[gp][1], 0.0, ip.ip().qwgt[gp]);

    // get coordinates of gauss point w.r.t. local parent coordinate system
    Core::LinAlg::SerialDenseMatrix pqxg(1, 3);
    Core::LinAlg::Matrix<3, 3> derivtrafo;

    Core::FE::boundary_gp_to_parent_gp<3>(
        pqxg, derivtrafo, intpoints, parent_element()->shape(), shape(), face_parent_number());

    Core::LinAlg::Matrix<3, 1> xi;
    for (int i = 0; i < 3; ++i) xi(i) = pqxg(0, i);
    strains<dt_vol>(xrefe, xcurr, xi, jac, defgrd, glstrain, rcg, bop, N_XYZ);

    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat{};
    Core::LinAlg::SymmetricTensor<double, 3, 3> stress{};
    Teuchos::ParameterList params;
    if (not parent_scalar.empty())
    {
      // as long as we need the parameter list to pass the information, we need to wrap it into a
      // std::shared_ptr<> as the values of a Teuchos::ParameterList have to be printable
      auto scalar_values_at_xi = std::make_shared<std::vector<double>>();
      *scalar_values_at_xi = Core::FE::interpolate_to_xi<dt_vol>(xi, parent_scalar);
      params.set("scalars", scalar_values_at_xi);
    }

    Core::LinAlg::Tensor<double, 3> xi_t = {{xi(0), xi(1), xi(2)}};
    Mat::EvaluationContext context{.total_time = nullptr,  // Do not have the time here
        .time_step_size = nullptr,                         // Do not have the time-step here
        .xi = &xi_t,
        .ref_coords = nullptr};
    std::dynamic_pointer_cast<Mat::So3Material>(parent_element()->material())
        ->evaluate(&defgrd, glstrain, params, context, stress, cmat, gp, parent_element()->id());

    double normalfac = 1.0;
    if (shape() == Core::FE::CellType::nurbs9)
    {
      std::vector<Core::LinAlg::SerialDenseVector> parentknots(dim);
      std::vector<Core::LinAlg::SerialDenseVector> boundaryknots(dim - 1);
      dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(
          Global::Problem::instance()->get_dis("structure").get())
          ->get_knot_vector()
          ->get_boundary_ele_and_parent_knots(
              parentknots, boundaryknots, normalfac, parent_element()->id(), face_parent_number());

      Core::LinAlg::Matrix<Core::FE::num_nodes(dt_surf), 1> weights, shapefcn;
      for (int i = 0; i < Core::FE::num_nodes(dt_surf); ++i)
        weights(i) = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[i])->w();

      Core::LinAlg::Matrix<2, 1> xi_surf;
      xi_surf(0) = ip.ip().qxg[gp][0];
      xi_surf(1) = ip.ip().qxg[gp][1];
      Core::FE::Nurbs::nurbs_get_2d_funct_deriv(
          shapefcn, deriv_surf, xi_surf, boundaryknots, weights, dt_surf);
    }
    else
    {
      Core::FE::shape_function_2d_deriv1(
          deriv_surf, ip.ip().qxg[gp][0], ip.ip().qxg[gp][1], shape());
    }

    surface_integration(detA, n, xrefe_surf, deriv_surf);
    n_v.scale(normalfac);
    n_v.scale(1.0 / n_v.norm2());
    nn.multiply_nt(n_v, n_v);

    Core::LinAlg::Matrix<6, 6> cn;
    Core::LinAlg::FourTensorOperations::add_symmetric_holzapfel_product(
        cn, Core::LinAlg::make_matrix<3, 3>(Core::LinAlg::get_full(rcg)), nn, 0.25);

    Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);
    Core::LinAlg::Matrix<6, 6> tmp1, tmp2;
    tmp1.multiply(cmat_view, id4);
    tmp2.multiply(tmp1, cn);
    tmp1.multiply(tmp2, id4);
    tmp2.multiply(tmp1, cmat_view);

    Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol) * 3, 6> tmp3;
    tmp3.multiply_tn(bop, tmp2);

    surf.multiply(detA * ip.ip().qwgt[gp], tmp3, bop, 1.0);
  }
}


template <Core::FE::CellType dt_vol>
void Discret::Elements::SolidSurface::strains(
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xrefe,
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xcurr,
    const Core::LinAlg::Matrix<3, 1>& xi, double& jac, Core::LinAlg::Tensor<double, 3, 3>& defgrd,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
    Core::LinAlg::Matrix<6, Core::FE::num_nodes(dt_vol) * 3>& bop,
    Core::LinAlg::Matrix<3, Core::FE::num_nodes(dt_vol)>& N_XYZ)
{
  const int dim = Core::FE::dim<dt_vol>;
  const int num_node = Core::FE::num_nodes(dt_vol);
  Core::LinAlg::Matrix<dim, num_node> deriv;

  if (dt_vol == Core::FE::CellType::nurbs27)
  {
    std::vector<Core::LinAlg::SerialDenseVector> knots;
    dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(
        Global::Problem::instance()->get_dis("structure").get())
        ->get_knot_vector()
        ->get_ele_knots(knots, parent_element_id());

    Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 1> weights, shapefcn;

    for (int i = 0; i < Core::FE::num_nodes(dt_vol); ++i)
      weights(i) = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(parent_element()->nodes()[i])->w();

    Core::FE::Nurbs::nurbs_get_3d_funct_deriv(shapefcn, deriv, xi, knots, weights, dt_vol);
  }
  else
  {
    Core::FE::shape_function_deriv1<dt_vol>(xi, deriv);
  }

  Core::LinAlg::Matrix<dim, dim> invJ;
  Core::LinAlg::Matrix<dim, dim> F_mat_view = Core::LinAlg::make_matrix_view(defgrd);
  invJ.multiply(deriv, xrefe);
  jac = invJ.invert();
  N_XYZ.multiply(invJ, deriv);
  F_mat_view.multiply_tt(xcurr, N_XYZ);

  rcg = Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(defgrd) * defgrd);
  glstrain = 0.5 * (rcg - Core::LinAlg::TensorGenerators::identity<double, 3, 3>);

  for (int i = 0; i < num_node; ++i)
  {
    bop(0, dim * i + 0) = defgrd(0, 0) * N_XYZ(0, i);
    bop(0, dim * i + 1) = defgrd(1, 0) * N_XYZ(0, i);
    bop(0, dim * i + 2) = defgrd(2, 0) * N_XYZ(0, i);
    bop(1, dim * i + 0) = defgrd(0, 1) * N_XYZ(1, i);
    bop(1, dim * i + 1) = defgrd(1, 1) * N_XYZ(1, i);
    bop(1, dim * i + 2) = defgrd(2, 1) * N_XYZ(1, i);
    bop(2, dim * i + 0) = defgrd(0, 2) * N_XYZ(2, i);
    bop(2, dim * i + 1) = defgrd(1, 2) * N_XYZ(2, i);
    bop(2, dim * i + 2) = defgrd(2, 2) * N_XYZ(2, i);
    /* ~~~ */
    bop(3, dim * i + 0) = defgrd(0, 0) * N_XYZ(1, i) + defgrd(0, 1) * N_XYZ(0, i);
    bop(3, dim * i + 1) = defgrd(1, 0) * N_XYZ(1, i) + defgrd(1, 1) * N_XYZ(0, i);
    bop(3, dim * i + 2) = defgrd(2, 0) * N_XYZ(1, i) + defgrd(2, 1) * N_XYZ(0, i);
    bop(4, dim * i + 0) = defgrd(0, 1) * N_XYZ(2, i) + defgrd(0, 2) * N_XYZ(1, i);
    bop(4, dim * i + 1) = defgrd(1, 1) * N_XYZ(2, i) + defgrd(1, 2) * N_XYZ(1, i);
    bop(4, dim * i + 2) = defgrd(2, 1) * N_XYZ(2, i) + defgrd(2, 2) * N_XYZ(1, i);
    bop(5, dim * i + 0) = defgrd(0, 2) * N_XYZ(0, i) + defgrd(0, 0) * N_XYZ(2, i);
    bop(5, dim * i + 1) = defgrd(1, 2) * N_XYZ(0, i) + defgrd(1, 0) * N_XYZ(2, i);
    bop(5, dim * i + 2) = defgrd(2, 2) * N_XYZ(0, i) + defgrd(2, 0) * N_XYZ(2, i);
  }
}


template <Core::FE::CellType dt_vol>
void Discret::Elements::SolidSurface::subspace_projector(
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xcurr,
    Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol) * Core::FE::dim<dt_vol>,
        Core::FE::num_nodes(dt_vol) * Core::FE::dim<dt_vol> -
            Core::FE::dim<dt_vol>*(Core::FE::dim<dt_vol> + 1) / 2>& proj)
{
  const int dim = Core::FE::dim<dt_vol>;
  const int num_node = Core::FE::num_nodes(dt_vol);
  if (dim != 3) FOUR_C_THROW("this should be 3D");

  Core::LinAlg::Matrix<3, 1> c;
  for (int r = 0; r < (int)xcurr.num_rows(); ++r)
    for (int d = 0; d < (int)xcurr.num_cols(); ++d) c(d) += xcurr(r, d);
  c.scale(1. / xcurr.num_rows());

  Core::LinAlg::Matrix<dim, 1> r[3];
  for (int i = 0; i < 3; ++i) r[i](i) = 1.;

  // basis, where the first six entries are the rigid body modes and the
  // remaining are constructed to be orthogonal to the rigid body modes
  Core::LinAlg::Matrix<dim * num_node, 1> basis[dim * num_node];

  // rigid body translations
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < num_node; ++j) basis[i](j * dim + i) = 1.;

  // rigid body rotations
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < num_node; ++j)
    {
      Core::LinAlg::Matrix<3, 1> x;
      for (int d = 0; d < 3; ++d) x(d) = xcurr(j, d);
      x.update(-1., c, 1.);
      Core::LinAlg::Matrix<3, 1> cross;
      cross.cross_product(r[i], x);
      for (int k = 0; k < 3; ++k) basis[i + 3](j * 3 + k) = cross(k);
    }
  for (int i = 0; i < 6; ++i) basis[i].scale(1. / basis[i].norm2());

  // build the remaining basis vectors by generalized cross products
  for (int i = 6; i < dim * num_node; ++i)
  {
    double sign = +1.;
    int off = 0;
    bool new_basis_found = false;
    for (off = 0; (off < dim * num_node - i) && !new_basis_found; ++off)
    {
      for (int j = 0; j < i + 1; ++j)
      {
        Core::LinAlg::SerialDenseMatrix det(i, i, true);
        for (int c = 0; c < i; ++c)
        {
          for (int k = 0; k < j; ++k) det(k, c) = basis[c](k + off);
          for (int k = j; k < i; ++k) det(k, c) = basis[c](k + 1 + off);
        }
        basis[i](j + off) = Core::LinAlg::determinant_lu(det) * sign;
        sign *= -1.;
      }
      if (basis[i].norm2() > 1.e-6)
      {
        basis[i].scale(1. / basis[i].norm2());
        new_basis_found = true;
      }
    }
    if (!new_basis_found) FOUR_C_THROW("no new basis vector found");
  }

  // at this point basis should already contain an ONB.
  // due to cut-off errors we do another sweep of Gram-Schmidt
  for (int i = 0; i < dim * num_node; ++i)
  {
    const Core::LinAlg::Matrix<dim * num_node, 1> tmp(basis[i]);
    for (int j = 0; j < i; ++j) basis[i].update(-tmp.dot(basis[j]), basis[j], 1.);

    basis[i].scale(1. / basis[i].norm2());
  }

  // hand out the projection matrix, i.e. the ONB not containing rigid body modes
  for (int i = 0; i < dim * num_node; ++i)
    for (int j = 6; j < dim * num_node; ++j) proj(i, j - 6) = basis[j](i);
}

double Discret::Elements::SolidSurface::estimate_nitsche_trace_max_eigenvalue_tsi(
    std::vector<double>& parent_disp)
{
  switch (parent_element()->shape())
  {
    case Core::FE::CellType::hex8:
      if (shape() == Core::FE::CellType::quad4)
        return estimate_nitsche_trace_max_eigenvalue_tsi<Core::FE::CellType::hex8,
            Core::FE::CellType::quad4>(parent_disp);
      else
        FOUR_C_THROW("how can an hex8 element have a surface that is not quad4 ???");
    case Core::FE::CellType::hex27:
      return estimate_nitsche_trace_max_eigenvalue_tsi<Core::FE::CellType::hex27,
          Core::FE::CellType::quad9>(parent_disp);
    case Core::FE::CellType::tet4:
      return estimate_nitsche_trace_max_eigenvalue_tsi<Core::FE::CellType::tet4,
          Core::FE::CellType::tri3>(parent_disp);
    case Core::FE::CellType::nurbs27:
      return estimate_nitsche_trace_max_eigenvalue_tsi<Core::FE::CellType::nurbs27,
          Core::FE::CellType::nurbs9>(parent_disp);
    default:
      FOUR_C_THROW("parent shape not implemented");
  }

  return 0;
}

template <Core::FE::CellType dt_vol, Core::FE::CellType dt_surf>
double Discret::Elements::SolidSurface::estimate_nitsche_trace_max_eigenvalue_tsi(
    std::vector<double>& parent_disp)
{
  const int dim = Core::FE::dim<dt_vol>;
  const int num_dof = Core::FE::num_nodes(dt_vol);
  const int dim_image = Core::FE::num_nodes(dt_vol) - 1;

  Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3> xrefe;
  Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3> xcurr;

  for (int i = 0; i < parent_element()->num_node(); ++i)
    for (int d = 0; d < dim; ++d)
    {
      xrefe(i, d) = parent_element()->nodes()[i]->x()[d];
      xcurr(i, d) = xrefe(i, d) + parent_disp[i * dim + d];
    }

  Core::LinAlg::Matrix<num_dof, num_dof> vol, surf;

  trace_estimate_vol_matrix_tsi<dt_vol>(xrefe, xcurr, vol);
  trace_estimate_surf_matrix_tsi<dt_vol, dt_surf>(xrefe, xcurr, surf);


  Core::LinAlg::Matrix<num_dof, dim_image> proj, tmp;
  subspace_projector_scalar<dt_vol>(proj);

  Core::LinAlg::Matrix<dim_image, dim_image> vol_red, surf_red;

  tmp.multiply(vol, proj);
  vol_red.multiply_tn(proj, tmp);
  tmp.multiply(surf, proj);
  surf_red.multiply_tn(proj, tmp);

  Core::LinAlg::SerialDenseMatrix vol_red_sd(
      Teuchos::View, vol_red.data(), dim_image, dim_image, dim_image);
  Core::LinAlg::SerialDenseMatrix surf_red_sd(
      Teuchos::View, surf_red.data(), dim_image, dim_image, dim_image);

  return Core::LinAlg::generalized_eigen_max_real_eigenvalue(surf_red_sd, vol_red_sd);
}

template <Core::FE::CellType dt_vol>
void Discret::Elements::SolidSurface::trace_estimate_vol_matrix_tsi(
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xrefe,
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xcurr,
    Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), Core::FE::num_nodes(dt_vol)>& vol)
{
  const int dim = Core::FE::dim<dt_vol>;
  const int num_node = Core::FE::num_nodes(dt_vol);

  double jac;
  Core::LinAlg::Tensor<double, 3, 3> defgrd;
  Core::LinAlg::SymmetricTensor<double, 3, 3> rcg;
  Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain;
  Core::LinAlg::Matrix<6, Core::FE::num_nodes(dt_vol) * 3> bop;
  Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol) * 3, 6> bc;
  Core::LinAlg::Matrix<dim, num_node> N_XYZ, iC_N_XYZ;

  Core::FE::IntPointsAndWeights<dim> ip(Discret::Elements::DisTypeToOptGaussRule<dt_vol>::rule);

  if (parent_element()->num_material() < 2) FOUR_C_THROW("where's my second material");
  std::shared_ptr<Mat::Fourier> mat_thermo =
      std::dynamic_pointer_cast<Mat::Fourier>(parent_element()->material(1));

  std::vector<double> k = mat_thermo->conductivity(parent_element_id());
  FOUR_C_ASSERT(k.size() != 1, "Conductivity value is a vector quantity, but has to be a scalar.");

  for (int gp = 0; gp < ip.ip().nquad; ++gp)
  {
    const Core::LinAlg::Matrix<3, 1> xi(ip.ip().qxg[gp], false);
    strains<dt_vol>(xrefe, xcurr, xi, jac, defgrd, glstrain, rcg, bop, N_XYZ);

    Core::LinAlg::Matrix<3, 3> iC;
    Core::LinAlg::Matrix<3, 3> F_view = Core::LinAlg::make_matrix_view(defgrd);
    iC.multiply_tn(F_view, F_view);
    iC.invert();

    iC_N_XYZ.multiply(iC, N_XYZ);
    iC_N_XYZ.scale(k[0]);

    vol.multiply_tn(ip.ip().qwgt[gp] * jac, N_XYZ, iC_N_XYZ, 1.);
  }
}


template <Core::FE::CellType dt_vol, Core::FE::CellType dt_surf>
void Discret::Elements::SolidSurface::trace_estimate_surf_matrix_tsi(
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xrefe,
    const Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), 3>& xcurr,
    Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), Core::FE::num_nodes(dt_vol)>& surf)
{
  const int dim = Core::FE::dim<dt_vol>;
  const int num_node = Core::FE::num_nodes(dt_vol);

  double jac;
  Core::LinAlg::Tensor<double, 3, 3> defgrd;
  Core::LinAlg::SymmetricTensor<double, 3, 3> rcg;
  Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain;
  Core::LinAlg::Matrix<6, Core::FE::num_nodes(dt_vol) * 3> bop;
  Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol) * 3, 6> bc;
  Core::LinAlg::Matrix<dim, num_node> N_XYZ;
  Core::LinAlg::Matrix<1, num_node> iCn_N_XYZ;

  Core::LinAlg::SerialDenseMatrix xrefe_surf(Core::FE::num_nodes(dt_surf), dim);
  material_configuration(xrefe_surf);

  std::vector<double> n(3);
  Core::LinAlg::Matrix<3, 1> n_v(n.data(), true), iCn;
  double detA;

  Core::FE::IntPointsAndWeights<dim - 1> ip(
      Discret::Elements::DisTypeToOptGaussRule<dt_surf>::rule);
  Core::LinAlg::SerialDenseMatrix deriv_surf(2, Core::FE::num_nodes(dt_surf));

  if (parent_element()->num_material() < 2) FOUR_C_THROW("where's my second material");
  std::shared_ptr<Mat::Fourier> mat_thermo =
      std::dynamic_pointer_cast<Mat::Fourier>(parent_element()->material(1));

  std::vector<double> k = mat_thermo->conductivity(parent_element_id());
  FOUR_C_ASSERT(k.size() != 1, "Conductivity value is a vector quantity, but has to be a scalar.");

  for (int gp = 0; gp < ip.ip().nquad; ++gp)
  {
    Core::FE::shape_function_2d_deriv1(deriv_surf, ip.ip().qxg[gp][0], ip.ip().qxg[gp][1], shape());
    surface_integration(detA, n, xrefe_surf, deriv_surf);
    n_v.scale(1. / n_v.norm2());

    Core::FE::CollectedGaussPoints intpoints =
        Core::FE::CollectedGaussPoints(1);  // reserve just for 1 entry ...
    intpoints.append(ip.ip().qxg[gp][0], ip.ip().qxg[gp][1], 0.0, ip.ip().qwgt[gp]);

    // get coordinates of gauss point w.r.t. local parent coordinate system
    Core::LinAlg::SerialDenseMatrix pqxg(1, 3);
    Core::LinAlg::Matrix<3, 3> derivtrafo;

    Core::FE::boundary_gp_to_parent_gp<3>(
        pqxg, derivtrafo, intpoints, parent_element()->shape(), shape(), face_parent_number());

    Core::LinAlg::Matrix<3, 1> xi;
    for (int i = 0; i < 3; ++i) xi(i) = pqxg(0, i);

    strains<dt_vol>(xrefe, xcurr, xi, jac, defgrd, glstrain, rcg, bop, N_XYZ);

    Core::LinAlg::Matrix<3, 3> iC;
    Core::LinAlg::Matrix<3, 3> F_view = Core::LinAlg::make_matrix_view(defgrd);
    iC.multiply_tn(F_view, F_view);
    iC.invert();
    iCn.multiply(iC, n_v);

    iCn_N_XYZ.multiply_tn(iCn, N_XYZ);
    iCn_N_XYZ.scale(k[0]);

    surf.multiply_tn(detA * ip.ip().qwgt[gp], iCn_N_XYZ, iCn_N_XYZ, 1.);
  }
}



template <Core::FE::CellType dt_vol>
void Discret::Elements::SolidSurface::subspace_projector_scalar(
    Core::LinAlg::Matrix<Core::FE::num_nodes(dt_vol), Core::FE::num_nodes(dt_vol) - 1>& proj)
{
  const int num_node = Core::FE::num_nodes(dt_vol);
  Core::LinAlg::Matrix<num_node, 1> basis[num_node];

  for (int i = 0; i < num_node; ++i) basis[0](i) = 1.;

  for (int i = 1; i < num_node; ++i)
  {
    double sign = +1.;
    for (int j = 0; j < i + 1; ++j)
    {
      Core::LinAlg::SerialDenseMatrix det(i, i, true);
      for (int c = 0; c < i; ++c)
      {
        for (int k = 0; k < j; ++k) det(k, c) = basis[c](k);
        for (int k = j; k < i; ++k) det(k, c) = basis[c](k + 1);
      }
      basis[i](j) = Core::LinAlg::determinant_lu(det) * sign;
      sign *= -1.;
    }
    basis[i].scale(1. / basis[i].norm2());
  }

  // hand out the projection matrix, i.e. the ONB not containing rigid body modes
  for (int i = 0; i < num_node; ++i)
    for (int j = 1; j < num_node; ++j) proj(i, j - 1) = basis[j](i);
}

FOUR_C_NAMESPACE_CLOSE
