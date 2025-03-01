// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_poro_3D_ele_calc_pressure_based.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_lib_integration.hpp"
#include "4C_solid_poro_3D_ele_calc_lib.hpp"

#include <optional>

FOUR_C_NAMESPACE_OPEN

template <Core::FE::CellType celltype>
Discret::Elements::SolidPoroPressureBasedEleCalc<celltype>::SolidPoroPressureBasedEleCalc()
    : gauss_integration_(Core::FE::create_gauss_integration<celltype>(
          get_gauss_rule_stiffness_matrix_poro<celltype>()))
{
}

template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureBasedEleCalc<celltype>::poro_setup(
    Mat::StructPoro& porostructmat, const Core::IO::InputParameterContainer& container)
{
  // attention: Make sure to use the same gauss integration rule as in the solid elements in case
  // you use a material, in which the fluid terms are dependent on solid history terms
  porostructmat.poro_setup(gauss_integration_.num_points(), container);
}

template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureBasedEleCalc<celltype>::evaluate_nonlinear_force_stiffness(
    const Core::Elements::Element& ele, Mat::StructPoro& porostructmat,
    Mat::FluidPoroMultiPhase& porofluidmat, const Inpar::Solid::KinemType& kinematictype,
    const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector* force_vector,
    Core::LinAlg::SerialDenseMatrix* stiffness_matrix)
{
  // Create views to SerialDenseMatrices
  std::optional<Core::LinAlg::Matrix<num_dim_ * num_nodes_, num_dim_ * num_nodes_>> stiff = {};
  std::optional<Core::LinAlg::Matrix<num_dim_ * num_nodes_, 1>> force = {};
  if (stiffness_matrix != nullptr) stiff.emplace(*stiffness_matrix, true);
  if (force_vector != nullptr) force.emplace(*force_vector, true);

  // get primary variables of multiphase porous medium flow
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
      discretization.get_state(1, "porofluid");
  const std::vector<double> fluidmultiphase_ephi =
      Core::FE::extract_values(*matrix_state, la[1].lm_);


  // Initialize variables of multiphase porous medium flow
  const int nummultifluiddofpernode = porofluidmat.num_mat();
  const int numfluidphases = porofluidmat.num_fluid_phases();
  const int numvolfrac = porofluidmat.num_vol_frac();
  const bool hasvolfracs = (nummultifluiddofpernode > numfluidphases);

  // get nodal coordinates current and reference
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, la[0].lm_);

  // Check for negative Jacobian determinants
  ensure_positive_jacobian_determinant_at_element_nodes(nodal_coordinates);

  // Loop over all Gauss points
  for_each_gauss_point(nodal_coordinates, gauss_integration_,
      [&](const Core::LinAlg::Matrix<num_dim_, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        const SpatialMaterialMapping<celltype> spatial_material_mapping =
            evaluate_spatial_material_mapping<celltype>(
                jacobian_mapping, nodal_coordinates, 1.0, kinematictype);

        const CauchyGreenAndInverse<celltype> cauchygreen =
            evaluate_cauchy_green_and_inverse(spatial_material_mapping);

        Core::LinAlg::Matrix<num_str_, num_dof_per_ele_> Bop =
            evaluate_strain_gradient(jacobian_mapping, spatial_material_mapping);

        Core::LinAlg::Matrix<num_str_, num_dof_per_ele_> dInverseRightCauchyGreen_dDisp =
            evaluate_inverse_cauchy_green_linearization(
                cauchygreen, jacobian_mapping, spatial_material_mapping);

        const double volchange = compute_volume_change<celltype>(nodal_coordinates.displacements,
            spatial_material_mapping, jacobian_mapping, ele, kinematictype);

        Core::LinAlg::Matrix<1, num_dof_per_ele_> dDetDefGrad_dDisp =
            compute_linearization_of_detdefgrad_wrt_disp<celltype>(
                spatial_material_mapping, jacobian_mapping, kinematictype);

        const Core::LinAlg::Matrix<1, num_dof_per_ele_> dVolchange_dDisp =
            compute_linearization_of_volchange_wrt_disp<celltype>(
                dDetDefGrad_dDisp, jacobian_mapping, kinematictype);

        std::vector<double> fluidmultiphase_phiAtGP =
            compute_fluid_multiphase_primary_variables_at_gp<celltype>(
                fluidmultiphase_ephi, nummultifluiddofpernode, shape_functions);

        double solidpressure = compute_sol_pressure_at_gp<celltype>(
            numfluidphases, fluidmultiphase_phiAtGP, porofluidmat);
        // derivative of press w.r.t. displacements (only in case of volfracs)
        Core::LinAlg::Matrix<1, num_dof_per_ele_> dSolidpressure_dDisp(true);

        if (hasvolfracs)
        {
          Core::LinAlg::Matrix<1, num_dof_per_ele_> dPorosity_dDisp;
          double porosity = 0.0;

          compute_porosity_and_linearization<celltype>(porostructmat, params, solidpressure, gp,
              volchange, porosity, dDetDefGrad_dDisp, dPorosity_dDisp);

          // save the pressure coming from the fluid S_i*p_i (old solidpressure, without accounting
          // for volfracs)
          const double fluidpress = solidpressure;

          solidpressure = recalculate_sol_pressure_at_gp(fluidpress, porosity,
              nummultifluiddofpernode, numfluidphases, numvolfrac, fluidmultiphase_phiAtGP);

          recalculate_linearization_of_solpress_wrt_disp<celltype>(fluidpress, porosity,
              nummultifluiddofpernode, numfluidphases, numvolfrac, fluidmultiphase_phiAtGP,
              dPorosity_dDisp, dSolidpressure_dDisp);
        }

        // inverse Right Cauchy-Green tensor as vector in voigt notation
        Core::LinAlg::Matrix<num_str_, 1> C_inv_vec(false);
        Core::LinAlg::Voigt::Stresses::matrix_to_vector(
            cauchygreen.inverse_right_cauchy_green_, C_inv_vec);

        // B^T . C^-1
        Core::LinAlg::Matrix<num_dof_per_ele_, 1> BopCinv(true);
        BopCinv.multiply_tn(Bop, C_inv_vec);

        // update internal force vector
        if (force.has_value())
        {
          update_internal_forcevector_with_fluidstressterm<celltype>(integration_factor,
              solidpressure, spatial_material_mapping.determinant_deformation_gradient_, BopCinv,
              *force);
        }

        // update stiffness matrix
        if (stiff.has_value())
        {
          update_elastic_stiffness_matrix<celltype>(integration_factor, solidpressure,
              spatial_material_mapping.determinant_deformation_gradient_, BopCinv, Bop,
              dDetDefGrad_dDisp, dSolidpressure_dDisp, dInverseRightCauchyGreen_dDisp, *stiff);

          // factor for `geometric' stiffness matrix
          Core::LinAlg::Matrix<num_str_, 1> sfac(C_inv_vec);  // auxiliary integrated stress

          // scale
          sfac.scale((-integration_factor * solidpressure *
                      spatial_material_mapping.determinant_deformation_gradient_));

          update_geometric_stiffness_matrix<celltype>(sfac, jacobian_mapping.N_XYZ_, *stiff);
        }
      });
}

template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureBasedEleCalc<
    celltype>::evaluate_nonlinear_force_stiffness_od(const Core::Elements::Element& ele,
    Mat::StructPoro& porostructmat, Mat::FluidPoroMultiPhase& porofluidmat,
    const Inpar::Solid::KinemType& kinematictype, const Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseMatrix& stiffness_matrix)
{
  // get primary variables of multiphase porous medium flow
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
      discretization.get_state(1, "porofluid");
  std::vector<double> fluidmultiphase_ephi = Core::FE::extract_values(*matrix_state, la[1].lm_);

  // Initialize variables of multiphase porous medium flow
  const int nummultifluiddofpernode = porofluidmat.num_mat();
  const int numfluidphases = porofluidmat.num_fluid_phases();
  const int numvolfrac = porofluidmat.num_vol_frac();
  const bool hasvolfracs = (nummultifluiddofpernode > numfluidphases);

  // get nodal coordinates current and reference
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, la[0].lm_);

  // Loop over all Gauss points
  for_each_gauss_point(nodal_coordinates, gauss_integration_,
      [&](const Core::LinAlg::Matrix<num_dim_, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp

      )
      {
        const SpatialMaterialMapping<celltype> spatial_material_mapping =
            evaluate_spatial_material_mapping<celltype>(
                jacobian_mapping, nodal_coordinates, 1.0, kinematictype);

        const CauchyGreenAndInverse<celltype> cauchygreen =
            evaluate_cauchy_green_and_inverse(spatial_material_mapping);

        Core::LinAlg::Matrix<num_str_, num_dof_per_ele_> Bop =
            evaluate_strain_gradient(jacobian_mapping, spatial_material_mapping);

        // volume change (used for porosity law). Same as J in nonlinear theory.
        const double volchange = compute_volume_change<celltype>(nodal_coordinates.displacements,
            spatial_material_mapping, jacobian_mapping, ele, kinematictype);

        std::vector<double> fluidmultiphase_phiAtGP =
            compute_fluid_multiphase_primary_variables_at_gp<celltype>(
                fluidmultiphase_ephi, nummultifluiddofpernode, shape_functions);

        std::vector<double> solidpressurederiv = compute_solid_pressure_deriv<celltype>(
            porofluidmat, fluidmultiphase_phiAtGP, numfluidphases);

        if (hasvolfracs)
        {
          double solidpressure = compute_sol_pressure_at_gp<celltype>(
              numfluidphases, fluidmultiphase_phiAtGP, porofluidmat);

          double porosity =
              compute_porosity<celltype>(porostructmat, params, solidpressure, volchange, gp);

          recalculate_sol_pressure_deriv(fluidmultiphase_phiAtGP, nummultifluiddofpernode,
              numfluidphases, numvolfrac, solidpressure, porosity, solidpressurederiv);
        }

        const double detJ_w = jacobian_mapping.determinant_ * gauss_integration_.weight(gp);

        // inverse Right Cauchy-Green tensor as vector in voigt notation
        Core::LinAlg::Matrix<num_str_, 1> C_inv_vec(false);
        Core::LinAlg::Voigt::Stresses::matrix_to_vector(
            cauchygreen.inverse_right_cauchy_green_, C_inv_vec);

        // B^T . C^-1
        Core::LinAlg::Matrix<num_dof_per_ele_, 1> BopCinv(true);
        BopCinv.multiply_tn(Bop, C_inv_vec);

        update_stiffness_matrix_coupling_multiphase_pressurebased<celltype>(detJ_w,
            solidpressurederiv, BopCinv, shape_functions,
            spatial_material_mapping.determinant_deformation_gradient_, nummultifluiddofpernode,
            stiffness_matrix);
      });
}

template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureBasedEleCalc<celltype>::coupling_stress(
    const Core::Elements::Element& ele, const Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Teuchos::ParameterList& params)
{
  auto iocouplingstress =
      params.get<Inpar::Solid::StressType>("iocouplstress", Inpar::Solid::stress_none);

  // check for output of coupling stress
  if (iocouplingstress == Inpar::Solid::stress_none)
  {
    // nothing to do for calculation of effective stress
    return;
  }
  else
  {
    FOUR_C_THROW("coupling stress poroelast not yet implemented for pressure-based variant");
  }
}

// template classes
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::hex8>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::hex27>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::tet4>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::tet10>;

FOUR_C_NAMESPACE_CLOSE
