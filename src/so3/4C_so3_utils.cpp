// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_utils.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_fiber_node.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_utils_densematrix_svd.hpp"
#include "4C_so3_prestress.hpp"

#include <Teuchos_ParameterList.hpp>

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

template <Core::FE::CellType distype>
void Discret::Elements::Utils::calc_r(const Core::Elements::Element* ele,
    const std::vector<double>& disp,
    Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::dim<distype>>& R)
{
  // number of nodes per element
  const int nen = Core::FE::num_nodes<distype>;

  // spatial dimension
  const int nsd = Core::FE::dim<distype>;

  if (disp.size() != nsd * nen) FOUR_C_THROW("mismatch in dimensions");

  Core::LinAlg::Matrix<nsd, 1> xi_ele_center =
      Core::FE::get_local_center_position<nsd>(distype);  // depending on distype

  Core::LinAlg::Matrix<nen, nsd> xrefe;  // X, material coord. of element
  Core::LinAlg::Matrix<nen, nsd> xcurr;  // x, current  coord. of element
  for (int i = 0; i < nen; ++i)
  {
    for (int d = 0; d < nsd; ++d)
    {
      xrefe(i, d) = ele->nodes()[i]->x()[d];
      xcurr(i, d) = ele->nodes()[i]->x()[d] + disp[i * nsd + d];
    }
  }
  Core::LinAlg::Matrix<nsd, nen> deriv;
  Core::FE::shape_function_deriv1<distype>(xi_ele_center, deriv);

  Core::LinAlg::Matrix<nsd, nsd> jac;
  Core::LinAlg::Matrix<nsd, nsd> defgrd;
  Core::LinAlg::Matrix<nsd, nen> deriv_xyz;
  jac.multiply(deriv, xrefe);
  jac.invert();
  deriv_xyz.multiply(jac, deriv);
  defgrd.multiply_tt(xcurr, deriv_xyz);

  // Calculate rotcurr from defgrd
  Core::LinAlg::Matrix<nsd, nsd> Q(true);
  Core::LinAlg::Matrix<nsd, nsd> S(true);
  Core::LinAlg::Matrix<nsd, nsd> VT(true);
  Core::LinAlg::svd<nsd, nsd>(defgrd, Q, S, VT);
  R.multiply_nn(Q, VT);
}

template <Core::FE::CellType distype>
void Discret::Elements::Utils::get_temperature_for_structural_material(
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype>,
        1>& shapefctsGP,            // shape function of current Gauss-point
    Teuchos::ParameterList& params  // special material parameter e.g. scalartemp
)
{
  // initialise the temperature
  Teuchos::RCP<std::vector<double>> temperature_vector =
      params.get<Teuchos::RCP<std::vector<double>>>("nodal_tempnp", Teuchos::null);

  // current temperature vector is available
  if (temperature_vector != Teuchos::null)
  {
    double scalartemp = 0.0;
    for (int i = 0; i < Core::FE::num_nodes<distype>; ++i)
    {
      scalartemp += shapefctsGP(i) * (*temperature_vector)[i];
    }

    // insert current element temperature T_{n+1} into parameter list
    params.set<double>("temperature", scalartemp);
  }
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::Utils::compute_deformation_gradient(
    Core::LinAlg::Matrix<probdim, probdim>& defgrd, Core::Nodes::Node** nodes,
    const Core::LinAlg::Matrix<probdim, 1>& xsi,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, probdim>& xdisp)
{
  static Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, probdim> xrefe, xcurr;

  evaluate_nodal_coordinates<distype, probdim>(nodes, xrefe);
  evaluate_current_nodal_coordinates<distype, probdim>(xrefe, xdisp, xcurr);

  Core::LinAlg::Matrix<probdim, Core::FE::num_nodes<distype>> N_rst(true);
  Core::FE::shape_function_deriv1<distype>(xsi, N_rst);

  static Core::LinAlg::Matrix<probdim, probdim> inv_detFJ;
  inv_detFJ.multiply(N_rst, xrefe);
  inv_detFJ.invert();

  compute_deformation_gradient_standard<distype, probdim>(defgrd, xcurr, N_rst, inv_detFJ);
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::Utils::compute_deformation_gradient(
    Core::LinAlg::Matrix<probdim, probdim>& defgrd, Core::Nodes::Node** nodes,
    const Core::LinAlg::Matrix<probdim, 1>& xsi, const std::vector<double>& displacement)
{
  static Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, probdim> xdisp;
  evaluate_nodal_displacements<distype, probdim>(displacement, xdisp);

  compute_deformation_gradient<distype, probdim>(defgrd, nodes, xsi, xdisp);
}

template <Core::FE::CellType distype>
void Discret::Elements::Utils::compute_deformation_gradient(
    Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::dim<distype>>& defgrd,
    const Inpar::Solid::KinemType kinemType,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, Core::FE::dim<distype>>& xdisp,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, Core::FE::dim<distype>>& xcurr,
    const Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::dim<distype>>& inverseJacobian,
    const Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>>& derivs,
    const Inpar::Solid::PreStress prestressType, Discret::Elements::PreStress& mulfHistory,
    const int gp)
{
  if (kinemType == Inpar::Solid::KinemType::linear)
  {
    defgrd.clear();
    for (auto i = 0; i < Core::FE::dim<distype>; ++i)
    {
      defgrd(i, i) = 1.0;
    }
    return;
  }

  if (prestressType == Inpar::Solid::PreStress::mulf)
  {
    compute_deformation_gradient_mulf<distype>(defgrd, xdisp, derivs, mulfHistory, gp);
    return;
  }

  compute_deformation_gradient_standard<distype, Core::FE::dim<distype>>(
      defgrd, xcurr, derivs, inverseJacobian);
}

template <Core::FE::CellType distype>
void Discret::Elements::Utils::compute_deformation_gradient_mulf(
    Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::dim<distype>>& defgrd,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, Core::FE::dim<distype>>& xdisp,
    const Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>>& derivs,
    Discret::Elements::PreStress& mulfHistory, const int gp)
{
  // get Jacobian mapping wrt to the stored configuration
  Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::dim<distype>> invJdef;
  mulfHistory.storageto_matrix(gp, invJdef, mulfHistory.j_history());

  // get derivatives wrt to last spatial configuration
  Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>> N_xyz;
  N_xyz.multiply(invJdef, derivs);

  // build multiplicative incremental defgrd
  Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::dim<distype>> Finc;
  Finc.multiply_tt(xdisp, N_xyz);
  for (auto i = 0; i < Core::FE::dim<distype>; ++i)
  {
    defgrd(i, i) += 1.0;
  }

  // get stored old incremental F
  Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::dim<distype>> Fhist;
  mulfHistory.storageto_matrix(gp, Fhist, mulfHistory.f_history());

  // build total defgrd = delta F * F_old
  defgrd.multiply(Finc, Fhist);
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::Utils::compute_deformation_gradient_standard(
    Core::LinAlg::Matrix<probdim, probdim>& defgrd,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, probdim>& xcurr,
    const Core::LinAlg::Matrix<probdim, Core::FE::num_nodes<distype>>& derivs,
    const Core::LinAlg::Matrix<probdim, probdim>& inverseJacobian)
{
  Core::LinAlg::Matrix<probdim, Core::FE::num_nodes<distype>> N_XYZ(false);
  N_XYZ.multiply(inverseJacobian, derivs);

  defgrd.multiply_tt(xcurr, N_XYZ);
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::Utils::evaluate_nodal_coordinates(
    Core::Nodes::Node** nodes, Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, probdim>& xrefe)
{
  for (auto i = 0; i < Core::FE::num_nodes<distype>; ++i)
  {
    const auto& x = nodes[i]->x();
    for (auto dim = 0; dim < probdim; ++dim) xrefe(i, dim) = x[dim];
  }
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::Utils::evaluate_nodal_displacements(const std::vector<double>& disp,
    Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, probdim>& xdisp)
{
  for (auto i = 0; i < Core::FE::num_nodes<distype>; ++i)
  {
    for (auto dim = 0; dim < probdim; ++dim) xdisp(i, dim) = disp[i * probdim + dim];
  }
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::Utils::evaluate_current_nodal_coordinates(
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, probdim>& xrefe,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, probdim>& xdisp,
    Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, probdim>& xcurr)
{
  xcurr.update(1.0, xrefe, 1.0, xdisp);
}

template <Core::FE::CellType distype>
void Discret::Elements::Utils::evaluate_inverse_jacobian(
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, Core::FE::dim<distype>>& xrefe,
    const Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>>& derivs,
    Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::dim<distype>>& inverseJacobian)
{
  inverseJacobian.multiply(1.0, derivs, xrefe, 0.0);
  inverseJacobian.invert();
}

void Discret::Elements::Utils::throw_error_fd_material_tangent(
    const Teuchos::ParameterList& sdyn, const std::string& eletype)
{
  if (sdyn.get<std::string>("MATERIALTANGENT") != "analytical")
  {
    FOUR_C_THROW(
        "Approximation of material tangent by finite differences not implemented by %s elements. "
        "Set parameter MATERIALTANGENT to analytical.",
        eletype.c_str());
  }
}

template void Discret::Elements::Utils::calc_r<Core::FE::CellType::tet10>(
    const Core::Elements::Element*, const std::vector<double>&, Core::LinAlg::Matrix<3, 3>&);

template void
Discret::Elements::Utils::get_temperature_for_structural_material<Core::FE::CellType::tet4>(
    const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tet4>, 1>& shapefctsGP,
    Teuchos::ParameterList& params);

template void
Discret::Elements::Utils::get_temperature_for_structural_material<Core::FE::CellType::hex27>(
    const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::hex27>, 1>& shapefctsGP,
    Teuchos::ParameterList& params);

template void
Discret::Elements::Utils::get_temperature_for_structural_material<Core::FE::CellType::hex8>(
    const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::hex8>, 1>& shapefctsGP,
    Teuchos::ParameterList& params);

template void
Discret::Elements::Utils::get_temperature_for_structural_material<Core::FE::CellType::nurbs27>(
    const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::nurbs27>, 1>& shapefctsGP,
    Teuchos::ParameterList& params);

template void
Discret::Elements::Utils::get_temperature_for_structural_material<Core::FE::CellType::tet10>(
    const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tet10>, 1>& shapefctsGP,
    Teuchos::ParameterList& params);

template void
Discret::Elements::Utils::get_temperature_for_structural_material<Core::FE::CellType::hex20>(
    const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::hex20>, 1>& shapefctsGP,
    Teuchos::ParameterList& params);

template void Discret::Elements::Utils::compute_deformation_gradient<Core::FE::CellType::hex8, 3>(
    Core::LinAlg::Matrix<3, 3>& defgrd, Core::Nodes::Node** nodes,
    const Core::LinAlg::Matrix<3, 1>& xsi, const Core::LinAlg::Matrix<8, 3>& xdisp);
template void Discret::Elements::Utils::compute_deformation_gradient<Core::FE::CellType::tet4, 3>(
    Core::LinAlg::Matrix<3, 3>& defgrd, Core::Nodes::Node** nodes,
    const Core::LinAlg::Matrix<3, 1>& xsi, const Core::LinAlg::Matrix<4, 3>& xdisp);

template void Discret::Elements::Utils::compute_deformation_gradient<Core::FE::CellType::hex8, 3>(
    Core::LinAlg::Matrix<3, 3>& defgrd, Core::Nodes::Node** nodes,
    const Core::LinAlg::Matrix<3, 1>& xsi, const std::vector<double>& displacement);
template void Discret::Elements::Utils::compute_deformation_gradient<Core::FE::CellType::tet4, 3>(
    Core::LinAlg::Matrix<3, 3>& defgrd, Core::Nodes::Node** nodes,
    const Core::LinAlg::Matrix<3, 1>& xsi, const std::vector<double>& displacement);

template void Discret::Elements::Utils::compute_deformation_gradient<Core::FE::CellType::hex8>(
    Core::LinAlg::Matrix<3, 3>& defgrd, const Inpar::Solid::KinemType kinemType,
    const Core::LinAlg::Matrix<8, 3>& xdisp, const Core::LinAlg::Matrix<8, 3>& xcurr,
    const Core::LinAlg::Matrix<3, 3>& inverseJacobian, const Core::LinAlg::Matrix<3, 8>& derivs,
    const Inpar::Solid::PreStress prestressType, Discret::Elements::PreStress& mulfHistory,
    const int gp);
template void Discret::Elements::Utils::compute_deformation_gradient<Core::FE::CellType::tet4>(
    Core::LinAlg::Matrix<3, 3>& defgrd, const Inpar::Solid::KinemType kinemType,
    const Core::LinAlg::Matrix<4, 3>& xdisp, const Core::LinAlg::Matrix<4, 3>& xcurr,
    const Core::LinAlg::Matrix<3, 3>& inverseJacobian, const Core::LinAlg::Matrix<3, 4>& derivs,
    const Inpar::Solid::PreStress prestressType, Discret::Elements::PreStress& mulfHistory,
    const int gp);
template void Discret::Elements::Utils::compute_deformation_gradient<Core::FE::CellType::tet10>(
    Core::LinAlg::Matrix<3, 3>& defgrd, const Inpar::Solid::KinemType kinemType,
    const Core::LinAlg::Matrix<10, 3>& xdisp, const Core::LinAlg::Matrix<10, 3>& xcurr,
    const Core::LinAlg::Matrix<3, 3>& inverseJacobian, const Core::LinAlg::Matrix<3, 10>& derivs,
    const Inpar::Solid::PreStress prestressType, Discret::Elements::PreStress& mulfHistory,
    const int gp);

template void Discret::Elements::Utils::compute_deformation_gradient_mulf<Core::FE::CellType::hex8>(
    Core::LinAlg::Matrix<3, 3>& defgrd, const Core::LinAlg::Matrix<8, 3>& xdisp,
    const Core::LinAlg::Matrix<3, 8>& derivs, Discret::Elements::PreStress& mulfHistory,
    const int gp);
template void Discret::Elements::Utils::compute_deformation_gradient_mulf<Core::FE::CellType::tet4>(
    Core::LinAlg::Matrix<3, 3>& defgrd, const Core::LinAlg::Matrix<4, 3>& xdisp,
    const Core::LinAlg::Matrix<3, 4>& derivs, Discret::Elements::PreStress& mulfHistory,
    const int gp);
template void
Discret::Elements::Utils::compute_deformation_gradient_mulf<Core::FE::CellType::tet10>(
    Core::LinAlg::Matrix<3, 3>& defgrd, const Core::LinAlg::Matrix<10, 3>& xdisp,
    const Core::LinAlg::Matrix<3, 10>& derivs, Discret::Elements::PreStress& mulfHistory,
    const int gp);

template void
Discret::Elements::Utils::compute_deformation_gradient_standard<Core::FE::CellType::hex8, 3>(
    Core::LinAlg::Matrix<3, 3>& defgrd, const Core::LinAlg::Matrix<8, 3>& xcurr,
    const Core::LinAlg::Matrix<3, 8>& derivs, const Core::LinAlg::Matrix<3, 3>& inverseJacobian);
template void
Discret::Elements::Utils::compute_deformation_gradient_standard<Core::FE::CellType::tet4, 3>(
    Core::LinAlg::Matrix<3, 3>& defgrd, const Core::LinAlg::Matrix<4, 3>& xcurr,
    const Core::LinAlg::Matrix<3, 4>& derivs, const Core::LinAlg::Matrix<3, 3>& inverseJacobian);
template void
Discret::Elements::Utils::compute_deformation_gradient_standard<Core::FE::CellType::tet10, 3>(
    Core::LinAlg::Matrix<3, 3>& defgrd, const Core::LinAlg::Matrix<10, 3>& xcurr,
    const Core::LinAlg::Matrix<3, 10>& derivs, const Core::LinAlg::Matrix<3, 3>& inverseJacobian);

template void Discret::Elements::Utils::evaluate_nodal_coordinates<Core::FE::CellType::hex8, 3>(
    Core::Nodes::Node** nodes, Core::LinAlg::Matrix<8, 3>& xrefe);
template void Discret::Elements::Utils::evaluate_nodal_coordinates<Core::FE::CellType::tet4, 3>(
    Core::Nodes::Node** nodes, Core::LinAlg::Matrix<4, 3>& xrefe);
template void Discret::Elements::Utils::evaluate_nodal_coordinates<Core::FE::CellType::tet10, 3>(
    Core::Nodes::Node** nodes, Core::LinAlg::Matrix<10, 3>& xrefe);
template void Discret::Elements::Utils::evaluate_nodal_coordinates<Core::FE::CellType::quad4, 3>(
    Core::Nodes::Node** nodes, Core::LinAlg::Matrix<4, 3>& xrefe);
template void Discret::Elements::Utils::evaluate_nodal_coordinates<Core::FE::CellType::tri3, 3>(
    Core::Nodes::Node** nodes, Core::LinAlg::Matrix<3, 3>& xrefe);

template void Discret::Elements::Utils::evaluate_nodal_displacements<Core::FE::CellType::hex8, 3>(
    const std::vector<double>&, Core::LinAlg::Matrix<8, 3>& xrefe);
template void Discret::Elements::Utils::evaluate_nodal_displacements<Core::FE::CellType::tet4, 3>(
    const std::vector<double>&, Core::LinAlg::Matrix<4, 3>& xrefe);
template void Discret::Elements::Utils::evaluate_nodal_displacements<Core::FE::CellType::tet10, 3>(
    const std::vector<double>&, Core::LinAlg::Matrix<10, 3>& xrefe);

template void Discret::Elements::Utils::evaluate_current_nodal_coordinates<Core::FE::CellType::hex8,
    3>(const Core::LinAlg::Matrix<8, 3>& xrefe, const Core::LinAlg::Matrix<8, 3>& xdisp,
    Core::LinAlg::Matrix<8, 3>& xcurr);
template void Discret::Elements::Utils::evaluate_current_nodal_coordinates<Core::FE::CellType::tet4,
    3>(const Core::LinAlg::Matrix<4, 3>& xrefe, const Core::LinAlg::Matrix<4, 3>& xdisp,
    Core::LinAlg::Matrix<4, 3>& xcurr);
template void
Discret::Elements::Utils::evaluate_current_nodal_coordinates<Core::FE::CellType::tet10, 3>(
    const Core::LinAlg::Matrix<10, 3>& xrefe, const Core::LinAlg::Matrix<10, 3>& xdisp,
    Core::LinAlg::Matrix<10, 3>& xcurr);

template void Discret::Elements::Utils::evaluate_inverse_jacobian<Core::FE::CellType::tet4>(
    const Core::LinAlg::Matrix<4, 3>& xrefe, const Core::LinAlg::Matrix<3, 4>& derivs,
    Core::LinAlg::Matrix<3, 3>& inverseJacobian);

FOUR_C_NAMESPACE_CLOSE
