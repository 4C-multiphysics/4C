// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_poro_3D_ele_calc_pressure_velocity_based.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_lib_integration.hpp"
#include "4C_solid_3D_ele_calc_lib_io.hpp"
#include "4C_solid_3D_ele_utils.hpp"
#include "4C_solid_poro_3D_ele_calc_lib.hpp"

#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace
{
  /*!
   * @brief Calculate Derivative of transposed deformation gradient w.r.t. displacements
   * @tparam celltype : Cell type
   * @param jacobian_mapping (in) :An object holding quantities of the jacobian mapping
   * (inverse Jacobian, determinant, derivatives of the shape functions w.r.t. XYZ)
   * @param spatial_material_mapping (in) :An object holding quantities of the spatial material
   * mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient)
   * @param kinematictype (in) : kinematic type of element
   * @return dInverseDeformationGradientTransposed_dDisp
   */
  template <Core::FE::CellType celltype>
  Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<celltype> *
                           Discret::Elements::Internal::num_dim<celltype>,
      Discret::Elements::Internal::num_dim<celltype> *
          Discret::Elements::Internal::num_nodes<celltype>>
  compute_linearization_of_deformation_gradient_transposed_wrt_disp(
      const Discret::Elements::JacobianMapping<celltype>& jacobian_mapping,
      const Discret::Elements::SpatialMaterialMapping<celltype>& spatial_material_mapping,
      const Inpar::Solid::KinemType& kinematictype = Inpar::Solid::KinemType::nonlinearTotLag)
    requires(Discret::Elements::Internal::num_dim<celltype> == 3)
  {
    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<celltype> *
                             Discret::Elements::Internal::num_dim<celltype>,
        Discret::Elements::Internal::num_dim<celltype> *
            Discret::Elements::Internal::num_nodes<celltype>>
        dInverseDeformationGradientTransposed_dDisp(true);
    if (kinematictype != Inpar::Solid::KinemType::linear)
    {
      // dF^-T/dus
      for (int i = 0; i < Discret::Elements::Internal::num_dim<celltype>; i++)
      {
        for (int n = 0; n < Discret::Elements::Internal::num_nodes<celltype>; n++)
        {
          for (int j = 0; j < Discret::Elements::Internal::num_dim<celltype>; j++)
          {
            const int gid = Discret::Elements::Internal::num_dim<celltype> * n + j;
            for (int k = 0; k < Discret::Elements::Internal::num_dim<celltype>; k++)
              for (int l = 0; l < Discret::Elements::Internal::num_dim<celltype>; l++)
                dInverseDeformationGradientTransposed_dDisp(
                    i * Discret::Elements::Internal::num_dim<celltype> + l, gid) +=
                    -spatial_material_mapping.inverse_deformation_gradient_(l, j) *
                    jacobian_mapping.N_XYZ_(k, n) *
                    spatial_material_mapping.inverse_deformation_gradient_(k, i);
          }
        }
      }
    }
    return dInverseDeformationGradientTransposed_dDisp;
  }

  /*!
   * @brief Calculate Derivative of inverse deformation gradient w.r.t. displacements and
   * multiplication with material gradient of fluid pressure
   * @tparam celltype : Cell type
   * @param d_inverse_deformationgradient_transposed_ddisp (in) : Derivative of inverse deformation
   * gradient w.r.t. displacements times material gradient of fluid pressure
   * @param Gradp (in) : material gradient of fluid pressure
   * @param kinematictype (in) : kinematic type of element
   * @return dInverseDeformationGradient_dDisp_Gradp : Derivative of inverse deformation gradient
   * w.r.t. displacements times material gradient of fluid pressure
   */
  template <Core::FE::CellType celltype>
  Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<celltype> *
                           Discret::Elements::Internal::num_dim<celltype>,
      Discret::Elements::Internal::num_dim<celltype> *
          Discret::Elements::Internal::num_nodes<celltype>>
  evaluate_inverse_deformation_gradient_linearization_multiplication(
      const Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<celltype> *
                                     Discret::Elements::Internal::num_dim<celltype>,
          Discret::Elements::Internal::num_dim<celltype> *
              Discret::Elements::Internal::num_nodes<celltype>>&
          d_inverse_deformationgradient_transposed_ddisp,
      const Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<celltype>, 1>& Gradp,
      const Inpar::Solid::KinemType& kinematictype = Inpar::Solid::KinemType::nonlinearTotLag)
    requires(Discret::Elements::Internal::num_dim<celltype> == 3)
  {
    // dF^-T/dus * Grad p
    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<celltype> *
                             Discret::Elements::Internal::num_dim<celltype>,
        Discret::Elements::Internal::num_dim<celltype> *
            Discret::Elements::Internal::num_nodes<celltype>>
        dInverseDeformationGradient_dDisp_Gradp(true);
    if (kinematictype != Inpar::Solid::KinemType::linear)
    {
      for (int i = 0; i < Discret::Elements::Internal::num_dim<celltype>; i++)
      {
        for (int n = 0; n < Discret::Elements::Internal::num_nodes<celltype>; n++)
        {
          for (int j = 0; j < Discret::Elements::Internal::num_dim<celltype>; j++)
          {
            const int gid = Discret::Elements::Internal::num_dim<celltype> * n + j;
            for (int l = 0; l < Discret::Elements::Internal::num_dim<celltype>; l++)
              dInverseDeformationGradient_dDisp_Gradp(i, gid) +=
                  d_inverse_deformationgradient_transposed_ddisp(
                      i * Discret::Elements::Internal::num_dim<celltype> + l, gid) *
                  Gradp(l);
          }
        }
      }
    }
    return dInverseDeformationGradient_dDisp_Gradp;
  }
}  // namespace

template <Core::FE::CellType celltype>
Discret::Elements::SolidPoroPressureVelocityBasedEleCalc<
    celltype>::SolidPoroPressureVelocityBasedEleCalc()
    : gauss_integration_(Core::FE::create_gauss_integration<celltype>(
          get_gauss_rule_stiffness_matrix_poro<celltype>()))
{
}

template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureVelocityBasedEleCalc<celltype>::poro_setup(
    Mat::StructPoro& porostructmat, const Core::IO::InputParameterContainer& container)
{
  // attention: Make sure to use the same gauss integration rule as in the solid elements in case
  // you use a material, in which the fluid terms are dependent on solid history terms
  porostructmat.poro_setup(gauss_integration_.num_points(), container);
}

template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureVelocityBasedEleCalc<
    celltype>::evaluate_nonlinear_force_stiffness(const Core::Elements::Element& ele,
    Mat::StructPoro& porostructmat, Mat::FluidPoro& porofluidmat,
    AnisotropyProperties anisotropy_properties, const Inpar::Solid::KinemType& kinematictype,
    const Core::FE::Discretization& discretization,
    const SolidPoroPrimaryVariables& primary_variables, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector* force_vector,
    Core::LinAlg::SerialDenseMatrix* stiffness_matrix,
    Core::LinAlg::SerialDenseMatrix* reactive_matrix)
{
  // Create views to SerialDenseMatrices
  std::optional<Core::LinAlg::Matrix<num_dim_ * num_nodes_, num_dim_ * num_nodes_>> stiff = {};
  std::optional<Core::LinAlg::Matrix<num_dim_ * num_nodes_, num_dim_ * num_nodes_>> react = {};
  std::optional<Core::LinAlg::Matrix<num_dim_ * num_nodes_, num_dim_ * num_nodes_>> react_matrix =
      {};

  if (reactive_matrix) react_matrix.emplace(reactive_matrix->values(), true);
  std::optional<Core::LinAlg::Matrix<num_dim_ * num_nodes_, 1>> force = {};
  enum Inpar::Solid::DampKind damping =
      params.get<enum Inpar::Solid::DampKind>("damping", Inpar::Solid::damp_none);
  if (stiffness_matrix != nullptr) stiff.emplace(*stiffness_matrix, true);
  if (react_matrix && react_matrix.value().is_initialized() &&
      damping == Inpar::Solid::damp_material)
    react.emplace(*reactive_matrix, true);
  if (force_vector != nullptr) force.emplace(*force_vector, true);

  // get primary variables of porous medium flow
  FluidVariables<celltype> fluid_variables = get_fluid_variable_views<celltype>(primary_variables);

  // get primary variables from structure field
  SolidVariables<celltype> solid_variables =
      get_solid_variable_views<celltype>(discretization, primary_variables);


  // get nodal coordinates current and reference
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, primary_variables.solid_displacements);

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

        Core::LinAlg::Matrix<num_dim_ * num_dim_, num_dim_ * num_nodes_>
            dInverseDeformationGradientTransposed_dDisp =
                compute_linearization_of_deformation_gradient_transposed_wrt_disp<celltype>(
                    jacobian_mapping, spatial_material_mapping, kinematictype);

        const double volchange = compute_volume_change<celltype>(nodal_coordinates.displacements,
            spatial_material_mapping, jacobian_mapping, ele, kinematictype);

        Core::LinAlg::Matrix<1, num_dof_per_ele_> dDetDefGrad_dDisp =
            compute_linearization_of_detdefgrad_wrt_disp<celltype>(
                spatial_material_mapping, jacobian_mapping, kinematictype);

        const Core::LinAlg::Matrix<1, num_dof_per_ele_> dVolchange_dDisp =
            compute_linearization_of_volchange_wrt_disp<celltype>(
                dDetDefGrad_dDisp, jacobian_mapping, kinematictype);

        // pressure at integration point
        double fluid_press = shape_functions.shapefunctions_.dot(fluid_variables.fluidpress_nodal);

        // structure velocity at integration point
        Core::LinAlg::Matrix<num_dim_, 1> disp_velocity(true);
        disp_velocity.multiply(solid_variables.solidvel_nodal, shape_functions.shapefunctions_);

        // fluid velocity at integration point
        Core::LinAlg::Matrix<num_dim_, 1> fluid_velocity(true);
        fluid_velocity.multiply(fluid_variables.fluidvel_nodal, shape_functions.shapefunctions_);

        // material fluid velocity gradient at integration point
        Core::LinAlg::Matrix<num_dim_, num_dim_> fvelder(true);
        fvelder.multiply_nt(fluid_variables.fluidvel_nodal, jacobian_mapping.N_XYZ_);

        // pressure gradient at integration point
        Core::LinAlg::Matrix<num_dim_, 1> Gradp(true);
        Gradp.multiply(jacobian_mapping.N_XYZ_, fluid_variables.fluidpress_nodal);

        // F^-1 * Grad p
        Core::LinAlg::Matrix<num_dim_, 1> FinvGradp(true);
        FinvGradp.multiply_tn(spatial_material_mapping.inverse_deformation_gradient_, Gradp);

        Core::LinAlg::Matrix<num_dim_ * num_dim_, num_dim_ * num_nodes_>
            dInverseDeformationGradient_dDisp_Gradp =
                evaluate_inverse_deformation_gradient_linearization_multiplication<celltype>(
                    dInverseDeformationGradientTransposed_dDisp, Gradp, kinematictype);

        Core::LinAlg::Matrix<1, num_dof_per_ele_> dPorosity_dDisp;
        double porosity = 0.0;

        compute_porosity_and_linearization<celltype>(porostructmat, params, fluid_press, gp,
            volchange, porosity, dDetDefGrad_dDisp, dPorosity_dDisp);

        // inverse Right Cauchy-Green tensor as vector in voigt notation
        Core::LinAlg::Matrix<num_str_, 1> C_inv_vec(false);
        Core::LinAlg::Voigt::Stresses::matrix_to_vector(
            cauchygreen.inverse_right_cauchy_green_, C_inv_vec);

        // B^T . C^-1
        Core::LinAlg::Matrix<num_dof_per_ele_, 1> BopCinv(true);
        BopCinv.multiply_tn(Bop, C_inv_vec);

        Core::LinAlg::Matrix<num_str_, 1> fstress(true);


        // update internal force vector
        if (force.has_value())
        {
          // update internal force vector with darcy-brinkman additions
          if (porofluidmat.type() == Mat::PAR::darcy_brinkman)
          {
            update_internal_force_vector_for_brinkman_flow<celltype>(integration_factor,
                porofluidmat.viscosity(),
                spatial_material_mapping.determinant_deformation_gradient_, porosity, fvelder,
                spatial_material_mapping.inverse_deformation_gradient_, Bop,
                cauchygreen.inverse_right_cauchy_green_, fstress, *force);
          }

          update_internal_forcevector_with_structure_fluid_coupling_and_reactive_darcy_terms<
              celltype>(integration_factor, shape_functions.shapefunctions_, porofluidmat,
              anisotropy_properties, spatial_material_mapping, porosity, disp_velocity,
              fluid_velocity, FinvGradp, *force);

          update_internal_forcevector_with_fluidstressterm<celltype>(integration_factor,
              fluid_press, spatial_material_mapping.determinant_deformation_gradient_, BopCinv,
              *force);
        }

        // update stiffness matrix
        if (stiff.has_value())
        {
          // update stiffness matrix with darcy-brinkman additions
          if (porofluidmat.type() == Mat::PAR::darcy_brinkman)
          {
            update_stiffness_matrix_for_brinkman_flow<celltype>(integration_factor,
                porofluidmat.viscosity(),
                spatial_material_mapping.determinant_deformation_gradient_, porosity, fvelder,
                spatial_material_mapping.inverse_deformation_gradient_, Bop,
                cauchygreen.inverse_right_cauchy_green_, dPorosity_dDisp, dDetDefGrad_dDisp,
                dInverseRightCauchyGreen_dDisp, dInverseDeformationGradientTransposed_dDisp,
                fstress, *stiff);
          }

          // initialize element matrizes and vectors
          Core::LinAlg::Matrix<num_dof_per_ele_, num_dof_per_ele_> erea_v(true);

          update_stiffness_matrix_with_structure_fluid_coupling_and_reactive_darcy_terms<celltype>(
              integration_factor, shape_functions.shapefunctions_, porofluidmat,
              anisotropy_properties, spatial_material_mapping, porosity, disp_velocity,
              fluid_velocity, FinvGradp, dDetDefGrad_dDisp, dInverseDeformationGradient_dDisp_Gradp,
              dPorosity_dDisp, dInverseDeformationGradientTransposed_dDisp, erea_v, *stiff);

          // derivative of press w.r.t. displacements zero here
          Core::LinAlg::Matrix<1, num_dof_per_ele_> dSolidpressure_dDisp(true);

          update_elastic_stiffness_matrix<celltype>(integration_factor, fluid_press,
              spatial_material_mapping.determinant_deformation_gradient_, BopCinv, Bop,
              dDetDefGrad_dDisp, dSolidpressure_dDisp, dInverseRightCauchyGreen_dDisp, *stiff);

          // factor for `geometric' stiffness matrix
          Core::LinAlg::Matrix<num_str_, 1> sfac(C_inv_vec);  // auxiliary integrated stress

          // scale and add viscous stress
          sfac.update(integration_factor, fstress,
              (-1.0) * integration_factor * fluid_press *
                  spatial_material_mapping.determinant_deformation_gradient_);

          update_geometric_stiffness_matrix<celltype>(sfac, jacobian_mapping.N_XYZ_, *stiff);

          if (react.has_value())
          {
            /* additional "reactive darcy-term"
            detJ * w(gp) * ( J * reacoeff * phi^2  ) * D(v_s)
  */
            react->update(1.0, erea_v, 1.0);
          }
        }
      });
}

template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureVelocityBasedEleCalc<
    celltype>::evaluate_nonlinear_force_stiffness_od(const Core::Elements::Element& ele,
    Mat::StructPoro& porostructmat, Mat::FluidPoro& porofluidmat,
    AnisotropyProperties anisotropy_properties, const Inpar::Solid::KinemType& kinematictype,
    const Core::FE::Discretization& discretization,
    const SolidPoroPrimaryVariables& primary_variables, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseMatrix* stiffness_matrix)
{
  // Create views to SerialDenseMatrices
  std::optional<Core::LinAlg::Matrix<num_dim_ * num_nodes_, (num_dim_ + 1) * num_nodes_>> stiff =
      {};
  if (stiffness_matrix != nullptr) stiff.emplace(*stiffness_matrix, true);

  if (stiff.has_value())
  {
    // get primary variables of porous medium flow
    FluidVariables<celltype> fluid_variables =
        get_fluid_variable_views<celltype>(primary_variables);

    // get primary variables from structure field
    SolidVariables<celltype> solid_variables =
        get_solid_variable_views<celltype>(discretization, primary_variables);

    // get nodal coordinates current and reference
    const ElementNodes<celltype> nodal_coordinates =
        evaluate_element_nodes<celltype>(ele, primary_variables.solid_displacements);

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

          const double volchange = compute_volume_change<celltype>(nodal_coordinates.displacements,
              spatial_material_mapping, jacobian_mapping, ele, kinematictype);

          // pressure at integration point
          double fluid_press =
              shape_functions.shapefunctions_.dot(fluid_variables.fluidpress_nodal);

          // structure velocity at integration point
          Core::LinAlg::Matrix<num_dim_, 1> disp_velocity = interpolate_nodal_value_to_gp<celltype>(
              solid_variables.solidvel_nodal, shape_functions);

          // fluid velocity at integration point
          Core::LinAlg::Matrix<num_dim_, 1> fluid_velocity =
              interpolate_nodal_value_to_gp<celltype>(
                  fluid_variables.fluidvel_nodal, shape_functions);

          // material fluid velocity gradient at integration point
          Core::LinAlg::Matrix<num_dim_, num_dim_> fluid_velocity_gradient(true);
          fluid_velocity_gradient.multiply_nt(
              fluid_variables.fluidvel_nodal, jacobian_mapping.N_XYZ_);

          // pressure gradient at integration point
          Core::LinAlg::Matrix<num_dim_, 1> pressure_gradient(true);
          pressure_gradient.multiply(jacobian_mapping.N_XYZ_, fluid_variables.fluidpress_nodal);

          const PorosityAndLinearizationOD porosity_and_linearization_od =
              compute_porosity_and_linearization_od<celltype>(
                  porostructmat, params, fluid_press, volchange, gp);

          // inverse Right Cauchy-Green tensor as vector in voigt notation
          Core::LinAlg::Matrix<num_str_, 1> C_inv_vec(false);
          Core::LinAlg::Voigt::Stresses::matrix_to_vector(
              cauchygreen.inverse_right_cauchy_green_, C_inv_vec);

          // B^T . C^-1
          Core::LinAlg::Matrix<num_dof_per_ele_, 1> BopCinv(true);
          BopCinv.multiply_tn(Bop, C_inv_vec);

          // F^-T * grad p
          Core::LinAlg::Matrix<num_dim_, 1> Finvgradp;
          Finvgradp.multiply_tn(
              spatial_material_mapping.inverse_deformation_gradient_, pressure_gradient);

          // F^-T * N_XYZ
          Core::LinAlg::Matrix<num_dim_, num_nodes_> FinvNXYZ;
          FinvNXYZ.multiply_tn(
              spatial_material_mapping.inverse_deformation_gradient_, jacobian_mapping.N_XYZ_);

          Core::LinAlg::Matrix<num_dim_, num_dim_> reatensor(true);
          Core::LinAlg::Matrix<num_dim_, num_dim_> linreac_dporosity(
              true);  // Derivative of the material reaction tensor w.r.t. the porosity
          Core::LinAlg::Matrix<num_dim_, 1> rea_fluid_vel(true);
          Core::LinAlg::Matrix<num_dim_, 1> rea_disp_vel(true);

          compute_linearization_of_reaction_tensor_od<celltype>(porofluidmat,
              shape_functions.shapefunctions_, spatial_material_mapping,
              porosity_and_linearization_od.porosity, disp_velocity, fluid_velocity,
              anisotropy_properties, reatensor, linreac_dporosity, rea_fluid_vel, rea_disp_vel);

          update_stiffness_matrix_od<celltype>(integration_factor, shape_functions.shapefunctions_,
              spatial_material_mapping, porosity_and_linearization_od.porosity,
              porosity_and_linearization_od.d_porosity_d_pressure, BopCinv, Finvgradp, FinvNXYZ,
              porofluidmat, disp_velocity, fluid_velocity, reatensor, linreac_dporosity,
              rea_fluid_vel, rea_disp_vel, *stiff);

          if (porofluidmat.type() == Mat::PAR::darcy_brinkman)
          {
            update_stiffness_brinkman_flow_od<celltype>(integration_factor,
                porofluidmat.viscosity(), porosity_and_linearization_od.porosity,
                porosity_and_linearization_od.d_porosity_d_pressure,
                shape_functions.shapefunctions_, jacobian_mapping, spatial_material_mapping,
                cauchygreen.inverse_right_cauchy_green_, fluid_velocity_gradient, Bop, *stiff);
          }
        });
  }
}


template <Core::FE::CellType celltype>
void Discret::Elements::SolidPoroPressureVelocityBasedEleCalc<celltype>::coupling_stress_poroelast(
    const Core::Elements::Element& ele, Mat::StructPoro& porostructmat,
    const Inpar::Solid::KinemType& kinematictype, const CouplStressIO& couplingstressIO,
    const Core::FE::Discretization& discretization,
    const SolidPoroPrimaryVariables& primary_variables, Teuchos::ParameterList& params)
{
  // initialize the coupling stress
  std::vector<char>& serialized_stress_data = couplingstressIO.mutable_data;
  Core::LinAlg::SerialDenseMatrix couplstress_data(gauss_integration_.num_points(), num_str_);

  if (couplingstressIO.type == Inpar::Solid::stress_none)
  {
    return;
  }

  // get primary variables of porous medium flow
  FluidVariables<celltype> fluid_variables = get_fluid_variable_views<celltype>(primary_variables);

  // get nodal coordinates current and reference
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, primary_variables.solid_displacements);

  // Loop over all Gauss points
  for_each_gauss_point(nodal_coordinates, gauss_integration_,
      [&](const Core::LinAlg::Matrix<num_dim_, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        const SpatialMaterialMapping<celltype> spatial_material_mapping =
            evaluate_spatial_material_mapping<celltype>(
                jacobian_mapping, nodal_coordinates, 1.0, kinematictype);

        // pressure at integration point
        double fluid_press = shape_functions.shapefunctions_.dot(fluid_variables.fluidpress_nodal);

        // coupling part of homogenized 2 Piola-Kirchhoff stress (3D)
        Core::LinAlg::Matrix<num_str_, 1> pk2_couplstress_gp(true);

        porostructmat.coupl_stress(
            spatial_material_mapping.deformation_gradient_, fluid_press, pk2_couplstress_gp);

        // return gp stresses
        switch (couplingstressIO.type)
        {
          case Inpar::Solid::stress_2pk:
          {
            for (int i = 0; i < num_str_; ++i) (couplstress_data)(gp, i) = pk2_couplstress_gp(i);
          }
          break;
          case Inpar::Solid::stress_cauchy:
          {
            // push forward of material stress to the spatial configuration
            Core::LinAlg::Matrix<num_str_, 1> cauchycouplstressvector;
            Solid::Utils::pk2_to_cauchy(pk2_couplstress_gp,
                spatial_material_mapping.deformation_gradient_, cauchycouplstressvector);
            Internal::assemble_vector_to_matrix_row(cauchycouplstressvector, couplstress_data, gp);
          }
          break;
          case Inpar::Solid::stress_none:
            break;

          default:
            FOUR_C_THROW("requested stress type not available");
            break;
        }
      });

  // pack the data for postprocessing
  serialize(couplstress_data, serialized_stress_data);
}



// template classes
template class Discret::Elements::SolidPoroPressureVelocityBasedEleCalc<Core::FE::CellType::hex8>;
template class Discret::Elements::SolidPoroPressureVelocityBasedEleCalc<Core::FE::CellType::hex27>;
template class Discret::Elements::SolidPoroPressureVelocityBasedEleCalc<Core::FE::CellType::tet4>;
template class Discret::Elements::SolidPoroPressureVelocityBasedEleCalc<Core::FE::CellType::tet10>;


FOUR_C_NAMESPACE_CLOSE