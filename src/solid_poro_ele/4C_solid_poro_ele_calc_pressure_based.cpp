// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_poro_ele_calc_pressure_based.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_solid_ele_calc_lib.hpp"
#include "4C_solid_ele_calc_lib_integration.hpp"
#include "4C_solid_poro_ele_calc_lib.hpp"

#include <optional>

FOUR_C_NAMESPACE_OPEN

template <Core::FE::CellType celltype>
Discret::Elements::SolidPoroPressureBasedEleCalc<celltype>::SolidPoroPressureBasedEleCalc()
  requires(Core::FE::dim<celltype> == 3)
    : gauss_integration_(Core::FE::create_gauss_integration<celltype>(
          get_gauss_rule_stiffness_matrix_poro<celltype>()))

{
}

template <Core::FE::CellType celltype>
Discret::Elements::SolidPoroPressureBasedEleCalc<celltype>::SolidPoroPressureBasedEleCalc(
    const double reference_thickness, const Discret::Elements::PlaneAssumption plane_assumption)
  requires(Core::FE::dim<celltype> == 2)
    : gauss_integration_(Core::FE::create_gauss_integration<celltype>(
          get_gauss_rule_stiffness_matrix_poro<celltype>())),
      element_properties_(
          {.reference_thickness = reference_thickness, .plane_assumption = plane_assumption})

{
}

template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureBasedEleCalc<celltype>::poro_setup(
    Mat::StructPoro& porostructmat, const Core::IO::InputParameterContainer& container)
{
  // attention: Make sure to use the same gauss integration rule as in the solid elements in case
  // you use a material, in which the fluid terms are dependent on solid history terms
  porostructmat.poro_setup(
      gauss_integration_.num_points(), read_fibers(container), read_coordinate_system(container));
}

template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureBasedEleCalc<celltype>::
    add_solidpressure_contribution_to_nonlinear_force_stiffness(const Core::Elements::Element& ele,
        Mat::StructPoro& porostructmat, Mat::FluidPoroMultiPhase& porofluidmat,
        const Inpar::Solid::KinemType& kinematictype,
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
  const SolidPoroFluidProperties solidporo_fluid_properties =
      evaluate_porofluid_properties(porofluidmat);
  const bool hasvolfracs =
      (solidporo_fluid_properties.number_of_fluid_dofs_per_node_ >
          solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_);

  if (hasvolfracs)
  {
    // safety check
    // this is only possible if we have an incompressible solid because the porosity of the
    // structure is calculated with the solidpressure which is calculated without
    // contributions from the additional porous network
    FOUR_C_ASSERT_ALWAYS(
        porostructmat.poro_law_type() == Core::Materials::m_poro_law_incompr_skeleton ||
            porostructmat.poro_law_type() == Core::Materials::m_poro_law_constant ||
            (porostructmat.poro_law_type() == Core::Materials::m_poro_law_density_dependent &&
                porostructmat.inv_bulk_modulus() < 1.0e-10),
        "Additional porous network is currently only possible with an incompressible "
        "skeleton. Fix "
        "your StructPoro material!");
  }

  const Mat::PAR::PoroFluidPressureBased::ClosingRelation type_volfrac_closingrelation =
      get_volfrac_closing_relation_type(solidporo_fluid_properties);

  // get nodal coordinates current and reference
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, la[0].lm_);

  // Check for negative Jacobian determinants
  ensure_positive_jacobian_determinant_at_element_nodes(nodal_coordinates);

  // Loop over all Gauss points
  for_each_gauss_point(nodal_coordinates, element_properties_, gauss_integration_,
      [&](const Core::LinAlg::Tensor<double, num_dim_>& xi,
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

        const double volchange = compute_volume_change<celltype>(nodal_coordinates.displacements,
            spatial_material_mapping, jacobian_mapping, ele, kinematictype);

        std::vector<double> fluid_phase_phi_at_gp =
            compute_fluid_phase_primary_variables_at_gp<celltype>(fluidmultiphase_ephi,
                solidporo_fluid_properties.number_of_fluid_dofs_per_node_, shape_functions);

        SolidPoroFluidAdditionalPorousNetworkVariables additional_porespace_variables_at_gp =
            evaluate_fluid_additional_porous_network_variables(solidporo_fluid_properties,
                fluid_phase_phi_at_gp, type_volfrac_closingrelation,
                spatial_material_mapping.determinant_deformation_gradient_, porofluidmat);

        double solidpressure = compute_sol_pressure_at_gp<celltype>(
            solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
            fluid_phase_phi_at_gp, porofluidmat);

        // derivative of press w.r.t. displacements (only in case of volfracs)
        double dSolidpressure_ddetJ = 0.0;

        if (hasvolfracs)
        {
          if (type_volfrac_closingrelation == Mat::PAR::PoroFluidPressureBased::ClosingRelation::
                                                  evolutionequation_homogenized_vasculature_tumor)
          {
            double porosity = 0.0;
            double dPorosity_dJ = 0.0;

            compute_porosity_and_linearization<celltype>(
                porostructmat, params, solidpressure, gp, volchange, porosity, dPorosity_dJ);

            // save the pressure coming from the fluid S_i*p_i (old solidpressure, without
            // accounting for volfracs)
            const double fluidpress = solidpressure;

            solidpressure = recalculate_solidpressure_at_gp(fluidpress, porosity,
                solidporo_fluid_properties.number_of_volfracs_,
                additional_porespace_variables_at_gp);

            dSolidpressure_ddetJ = recalculate_linearization_of_solpress_wrt_det_def_grad<celltype>(
                fluidpress, porosity, solidporo_fluid_properties.number_of_fluid_dofs_per_node_,
                solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
                solidporo_fluid_properties.number_of_volfracs_, fluid_phase_phi_at_gp,
                dPorosity_dJ);
          }
          else if (type_volfrac_closingrelation ==
                   Mat::PAR::PoroFluidPressureBased::ClosingRelation::evolutionequation_blood_lung)
          {
            double dPorosity_dJ = 0.0;
            double porosity = 0.0;

            compute_porosity_and_linearization<celltype>(
                porostructmat, params, solidpressure, gp, volchange, porosity, dPorosity_dJ);

            // save the pressure coming from the fluid S_i*p_i (old solidpressure, without
            // accounting for volfracs)
            const double fluidpress = solidpressure;

            solidpressure = recalculate_solidpressure_at_gp(fluidpress, porosity,
                solidporo_fluid_properties.number_of_volfracs_,
                additional_porespace_variables_at_gp);

            dSolidpressure_ddetJ = recalculate_linearization_of_solpress_wrt_det_def_grad<celltype>(
                fluidpress, porosity,
                solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
                fluid_phase_phi_at_gp, spatial_material_mapping.determinant_deformation_gradient_,
                porofluidmat, dPorosity_dJ);
          }
          else
          {
            FOUR_C_THROW("Internal error!");
          }
        }

        const double J = spatial_material_mapping.determinant_deformation_gradient_;
        const auto& C_inv = cauchygreen.inverse_right_cauchy_green_;

        Stress<celltype> stress{};
        stress.pk2_ = -solidpressure * J * C_inv;

        // update internal force vector
        if (force.has_value())
        {
          // Add fluid stress term to internal force vector
          add_internal_force_vector(jacobian_mapping,
              spatial_material_mapping.deformation_gradient_, stress.pk2_, integration_factor,
              *force);
        }

        // update stiffness matrix
        if (stiff.has_value())
        {
          stress.cmat_ =
              -(dSolidpressure_ddetJ * J + solidpressure) * J * Core::LinAlg::dyadic(C_inv, C_inv) +
              solidpressure * J *
                  (Core::LinAlg::einsum_sym<"ik", "jl">(C_inv, C_inv) +
                      Core::LinAlg::einsum_sym<"il", "jk">(C_inv, C_inv));

          add_stiffness_matrix(jacobian_mapping, spatial_material_mapping.deformation_gradient_,
              stress, integration_factor, *stiff);
        }
      });
}

template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureBasedEleCalc<celltype>::
    add_bodyforce_contribution_to_nonlinear_force_stiffness(const Core::Elements::Element& ele,
        Mat::StructPoro& porostructmat, Mat::FluidPoroMultiPhase& porofluidmat,
        const Inpar::Solid::KinemType& kinematictype,
        const Core::Utils::FunctionOfSpaceTime* bodyforce_contribution_function, const double time,
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
  const SolidPoroFluidProperties solidporo_fluid_properties =
      evaluate_porofluid_properties(porofluidmat);

  if (solidporo_fluid_properties.number_of_volfracs_)
  {
    // safety check
    // this is only possible if we have an incompressible solid because the porosity of the
    // structure is calculated with the solidpressure which is calculated without
    // contributions from the additional porous network
    FOUR_C_ASSERT_ALWAYS(
        porostructmat.poro_law_type() == Core::Materials::m_poro_law_incompr_skeleton ||
            porostructmat.poro_law_type() == Core::Materials::m_poro_law_constant ||
            (porostructmat.poro_law_type() == Core::Materials::m_poro_law_density_dependent &&
                porostructmat.inv_bulk_modulus() < 1.0e-10),
        "Additional porous network is currently only possible with an incompressible "
        "skeleton. Fix your StructPoro material!");
  }

  const Mat::PAR::PoroFluidPressureBased::ClosingRelation type_volfrac_closingrelation =
      get_volfrac_closing_relation_type(solidporo_fluid_properties);

  // get nodal coordinates current and reference
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, la[0].lm_);

  // Check for negative Jacobian determinants
  ensure_positive_jacobian_determinant_at_element_nodes(nodal_coordinates);

  // Loop over all Gauss points
  for_each_gauss_point(nodal_coordinates, element_properties_, gauss_integration_,
      [&](const Core::LinAlg::Tensor<double, num_dim_>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        // get fluid phase primary variables
        std::vector<double> fluid_phase_phi_at_gp =
            compute_fluid_phase_primary_variables_at_gp<celltype>(fluidmultiphase_ephi,
                solidporo_fluid_properties.number_of_fluid_dofs_per_node_, shape_functions);

        const SpatialMaterialMapping<celltype> spatial_material_mapping =
            evaluate_spatial_material_mapping<celltype>(
                jacobian_mapping, nodal_coordinates, 1.0, kinematictype);

        SolidPoroFluidAdditionalPorousNetworkVariables additional_porespace_variables_at_gp =
            evaluate_fluid_additional_porous_network_variables(solidporo_fluid_properties,
                fluid_phase_phi_at_gp, type_volfrac_closingrelation,
                spatial_material_mapping.determinant_deformation_gradient_, porofluidmat);


        double solidpressure = compute_sol_pressure_at_gp<celltype>(
            solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
            fluid_phase_phi_at_gp, porofluidmat);

        // needed for computation of porosity and linearization
        const double volchange = compute_volume_change<celltype>(nodal_coordinates.displacements,
            spatial_material_mapping, jacobian_mapping, ele, kinematictype);

        Core::LinAlg::Matrix<1, num_dof_per_ele_> dDetDefGrad_dDisp =
            compute_linearization_of_detdefgrad_wrt_disp<celltype>(
                spatial_material_mapping, jacobian_mapping, kinematictype);

        double dPorosity_dDetDefGrad = 0.0;
        double porosity = 0.0;

        compute_porosity_and_linearization<celltype>(
            porostructmat, params, solidpressure, gp, volchange, porosity, dPorosity_dDetDefGrad);


        const double volfrac_solid = 1.0 - porosity;
        double pre_factor_force_contribution_solid =
            volfrac_solid * porostructmat.density_solid_phase();

        // get phase densities
        const std::vector<double> phase_densities = porofluidmat.get_phase_densities();

        // bodyforce contribution from additional porous network / volfracs
        double pre_factor_force_contribution_volfrac{};
        for (int i_volfrac = 0; i_volfrac < solidporo_fluid_properties.number_of_volfracs_;
            ++i_volfrac)
        {
          pre_factor_force_contribution_volfrac +=
              phase_densities.at(
                  solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_ +
                  i_volfrac) *
              additional_porespace_variables_at_gp.volfrac.at(i_volfrac);
        }

        // bodyforce contribution from multiphase porespace
        double pre_factor_force_contribution_multifluid{};
        if (solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_)
        {
          // get fluid phase saturations
          std::vector<double> saturation =
              compute_porofluid_saturation_in_multiphase_porespace<celltype>(
                  solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
                  fluid_phase_phi_at_gp, porofluidmat);
          for (int i_phase = 0;
              i_phase < solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_;
              ++i_phase)
          {
            pre_factor_force_contribution_multifluid +=

                phase_densities.at(i_phase) *
                (porosity - additional_porespace_variables_at_gp.sumaddvolfrac) *
                saturation[i_phase];
          }
        }

        // calculate the coordinates of the gauss point
        std::array<double, num_dim_> gp_coordinate{};
        for (int idim = 0; idim < num_dim_; idim++)
        {
          for (int inode = 0; inode < num_nodes_; inode++)
          {
            gp_coordinate.at(idim) += shape_functions.shapefunctions_(inode) *
                                      nodal_coordinates.current_coordinates(idim, inode);
          }
        }

        // evaluate bodyforce function
        Core::LinAlg::Tensor<double, num_dim_> bodyforce_contribution{};
        for (int idim = 0; idim < num_dim_; idim++)
          bodyforce_contribution(idim) =
              bodyforce_contribution_function->evaluate(gp_coordinate, time, idim);

        Core::LinAlg::Matrix<num_dof_per_ele_, 1> bodyforce(Core::LinAlg::Initialization::zero);
        for (int j = 0; j < num_dim_; ++j)
        {
          for (int i_node = 0; i_node < num_nodes_; ++i_node)
          {
            bodyforce(i_node * num_dim_ + j, 0) +=
                shape_functions.shapefunctions_(i_node) * bodyforce_contribution(j);
          }
        }

        // update force vector
        if (force.has_value())
        {
          double pre_factor = pre_factor_force_contribution_solid +
                              pre_factor_force_contribution_volfrac +
                              pre_factor_force_contribution_multifluid;

          force->update(-1.0 * pre_factor * integration_factor, bodyforce, 1.0);
        }

        // update stiffness matrix dvolfrac/dJ dJ/du
        if (stiff.has_value())
        {
          // (dprefactor_soliddJ + dprefactor_volfracdJ + dprefactor_fluid_multiphase_dJ) * dJddisp
          Core::LinAlg::Matrix<num_dof_per_ele_, num_dof_per_ele_> dBodyforcedDisp;

          double dpre_factor_force_contribution_solid_dDetDefGrad =
              -1.0 * dPorosity_dDetDefGrad * porostructmat.density_solid_phase();

          double dprefactor_volfrac_dDetDefGrad{};
          if (solidporo_fluid_properties.number_of_volfracs_ &&
              get_volfrac_closing_relation_type(solidporo_fluid_properties) ==
                  Mat::PAR::PoroFluidPressureBased::ClosingRelation::evolutionequation_blood_lung)
          {
            // only one volfrac possible with this closing relation
            int i_volfrac = 0;
            dprefactor_volfrac_dDetDefGrad +=
                compute_linearization_of_volfrac_blood_lung_wrt_det_def_grad(
                    solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
                    fluid_phase_phi_at_gp,
                    spatial_material_mapping.determinant_deformation_gradient_, porofluidmat) *
                phase_densities.at(
                    solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_ +
                    i_volfrac);
          }

          // bodyforce contribution from multiphase porespace
          double dpre_factor_force_contribution_multifluid_dDetDefGrad{};
          if (solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_)
          {
            // get fluid phase saturations
            std::vector<double> saturation =
                compute_porofluid_saturation_in_multiphase_porespace<celltype>(
                    solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
                    fluid_phase_phi_at_gp, porofluidmat);
            for (int i_phase = 0;
                i_phase <
                solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_;
                ++i_phase)
            {
              dpre_factor_force_contribution_multifluid_dDetDefGrad +=
                  phase_densities.at(i_phase) * saturation[i_phase] * dPorosity_dDetDefGrad;

              if (solidporo_fluid_properties.number_of_volfracs_ &&
                  get_volfrac_closing_relation_type(solidporo_fluid_properties) ==
                      Mat::PAR::PoroFluidPressureBased::ClosingRelation::
                          evolutionequation_blood_lung)
              {
                dpre_factor_force_contribution_multifluid_dDetDefGrad -=
                    phase_densities.at(i_phase) * saturation[i_phase] *
                    compute_linearization_of_volfrac_blood_lung_wrt_det_def_grad(
                        solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
                        fluid_phase_phi_at_gp,
                        spatial_material_mapping.determinant_deformation_gradient_, porofluidmat);
              }
            }
          }


          double pre_factor_linearization_wrt_DetDefGrad =
              dpre_factor_force_contribution_solid_dDetDefGrad + dprefactor_volfrac_dDetDefGrad +
              dpre_factor_force_contribution_multifluid_dDetDefGrad;

          dBodyforcedDisp.multiply(integration_factor * pre_factor_linearization_wrt_DetDefGrad,
              bodyforce, dDetDefGrad_dDisp);

          stiff->update(1.0, dBodyforcedDisp, 1.0);
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
  const SolidPoroFluidProperties solidporo_fluid_properties =
      evaluate_porofluid_properties(porofluidmat);
  const bool hasvolfracs =
      (solidporo_fluid_properties.number_of_fluid_dofs_per_node_ >
          solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_);

  const Mat::PAR::PoroFluidPressureBased::ClosingRelation type_volfrac_closingrelation =
      get_volfrac_closing_relation_type(solidporo_fluid_properties);

  // get nodal coordinates current and reference
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, la[0].lm_);

  // Loop over all Gauss points
  for_each_gauss_point(nodal_coordinates, element_properties_, gauss_integration_,
      [&](const Core::LinAlg::Tensor<double, num_dim_>& xi,
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

        std::vector<double> fluidmultiphase_phi_at_gp =
            compute_fluid_phase_primary_variables_at_gp<celltype>(fluidmultiphase_ephi,
                solidporo_fluid_properties.number_of_fluid_dofs_per_node_, shape_functions);

        std::vector<double> solidpressurederiv =
            compute_solid_pressure_deriv<celltype>(porofluidmat, fluidmultiphase_phi_at_gp,
                solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_);

        if (hasvolfracs)
        {
          double solidpressure = compute_sol_pressure_at_gp<celltype>(
              solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
              fluidmultiphase_phi_at_gp, porofluidmat);
          double porosity =
              compute_porosity<celltype>(porostructmat, params, solidpressure, volchange, gp);


          if (type_volfrac_closingrelation == Mat::PAR::PoroFluidPressureBased::ClosingRelation::
                                                  evolutionequation_homogenized_vasculature_tumor)
          {
            recalculate_sol_pressure_deriv(fluidmultiphase_phi_at_gp,
                solidporo_fluid_properties.number_of_fluid_dofs_per_node_,
                solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
                solidporo_fluid_properties.number_of_volfracs_, solidpressure, porosity,
                solidpressurederiv);
          }
          else if (type_volfrac_closingrelation ==
                   Mat::PAR::PoroFluidPressureBased::ClosingRelation::evolutionequation_blood_lung)
          {
            recalculate_sol_pressure_deriv(fluidmultiphase_phi_at_gp,
                solidporo_fluid_properties.number_of_fluid_dofs_per_node_,
                solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_,
                solidporo_fluid_properties.number_of_volfracs_, solidpressure, porosity,
                solidpressurederiv, porofluidmat,
                spatial_material_mapping.determinant_deformation_gradient_);
          }
          else
          {
            FOUR_C_THROW("Internal error!");
          }
        }

        const double detJ_w = jacobian_mapping.determinant_ * gauss_integration_.weight(gp);

        // B^T . C^-1
        Core::LinAlg::Matrix<num_dof_per_ele_, 1> BopCinv(Core::LinAlg::Initialization::zero);
        BopCinv.multiply_tn(Bop,
            Core::LinAlg::make_stress_like_voigt_view(cauchygreen.inverse_right_cauchy_green_));

        update_stiffness_matrix_coupling_multiphase_pressurebased<celltype>(detJ_w,
            solidpressurederiv, BopCinv, shape_functions,
            spatial_material_mapping.determinant_deformation_gradient_,
            solidporo_fluid_properties.number_of_fluid_dofs_per_node_, stiffness_matrix);
      });

  // linearization of bodyforce term not yet included
}


// template classes
// 2D
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::quad4>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::quad8>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::quad9>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::nurbs9>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::tri3>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::tri6>;

// 3D
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::hex8>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::hex27>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::tet4>;
template class Discret::Elements::SolidPoroPressureBasedEleCalc<Core::FE::CellType::tet10>;

FOUR_C_NAMESPACE_CLOSE
