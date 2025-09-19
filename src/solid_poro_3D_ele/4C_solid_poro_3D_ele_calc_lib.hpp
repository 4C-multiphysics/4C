// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_PORO_3D_ELE_CALC_LIB_HPP
#define FOUR_C_SOLID_PORO_3D_ELE_CALC_LIB_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_element_dof_matrix.hpp"
#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_porofluid_pressure_based_ele_calc_utils.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_poro_3D_ele_properties.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <ranges>

FOUR_C_NAMESPACE_OPEN

namespace
{
  /**
   * Calculates the volume fraction based on the closing relation (given in material)
   * @param fluidmultiphase_phi_at_gp (in): fluid in multiphase porespace primary variables at gauss
   * point
   * @param volfracpressure (in): pressure of additional porous network
   * @param determinant_deformation_gradient (in): determinant of deformation gradient
   * @param porofluidmat (in): fluid material
   * @param numfluidphases (in): number of fluid phases in multiphase porespace
   * @return: volume fraction of additional porous network
   */
  double calculate_volfrac_from_closing_relation_blood_lung(
      const std::vector<double>& fluidmultiphase_phi_at_gp, const double volfracpressure,
      const double determinant_deformation_gradient, const Mat::FluidPoroMultiPhase& porofluidmat,
      const int numfluidphases)
  {
    // pressure of air (always first phase in multiphase porespace)
    if (fluidmultiphase_phi_at_gp[0] / volfracpressure <= 1.0)
    {
      return PoroPressureBased::ElementUtils::
                 get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                     porofluidmat, numfluidphases)
                     .initial_volfrac() *
             pow(determinant_deformation_gradient,
                 PoroPressureBased::ElementUtils::
                     get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                         porofluidmat, numfluidphases)
                         .scaling_parameter_deformation());
    }
    else
    {
      return PoroPressureBased::ElementUtils::
                 get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                     porofluidmat, numfluidphases)
                     .initial_volfrac() *
             pow(determinant_deformation_gradient,
                 PoroPressureBased::ElementUtils::
                     get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                         porofluidmat, numfluidphases)
                         .scaling_parameter_deformation()) *
             pow((fluidmultiphase_phi_at_gp[0] / volfracpressure),
                 PoroPressureBased::ElementUtils::
                     get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                         porofluidmat, numfluidphases)
                         .scaling_parameter_pressure());
    }
  }
}  // namespace

namespace Discret::Elements::Internal
{
  template <Core::FE::CellType celltype>
  inline static constexpr int num_dof_per_node = num_dim<celltype>;

  template <Core::FE::CellType celltype>
  inline void calculate_viscous_stress(const double integration_fac, const double viscosity,
      const double det_defgrd, const double porosity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>& fvelder,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          defgrd_inv,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>& C_inv,
      Core::LinAlg::Matrix<Internal::num_str<celltype>, 1>& fstress,
      Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>& CinvFvel)
  {
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> visctress1;
    CinvFvel.multiply(C_inv, fvelder);
    visctress1.multiply_nt(CinvFvel, defgrd_inv);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> visctress2(
        visctress1);
    visctress1.update_t(1.0, visctress2, 1.0);
    fstress(0) = visctress1(0, 0);
    fstress(1) = visctress1(1, 1);
    fstress(2) = visctress1(2, 2);
    fstress(3) = visctress1(0, 1);
    fstress(4) = visctress1(1, 2);
    fstress(5) = visctress1(2, 0);
    fstress.scale(integration_fac * viscosity * det_defgrd * porosity);
  }

}  // namespace Discret::Elements::Internal


namespace Discret::Elements
{
  enum class PorosityFormulation : std::uint8_t
  {
    from_material_law,
    as_primary_variable
  };

  template <PorosityFormulation porosity_formulation>
  struct SolidPoroPrimaryVariables
  {
  };

  template <>
  struct SolidPoroPrimaryVariables<PorosityFormulation::from_material_law>
  {
    const std::vector<int> solid_location_array;
    std::vector<double> solid_displacements{};

    const std::vector<int> fluid_location_array;
    std::vector<double> fluid_velocities{};
    std::vector<double> fluid_pressures{};
  };

  template <>
  struct SolidPoroPrimaryVariables<PorosityFormulation::as_primary_variable>
  {
    const std::vector<int> solid_location_array;
    std::vector<double> solid_displacements{};

    const std::vector<double>& initial_porosity;
    std::vector<double> porosity{};

    const std::vector<int> fluid_location_array;
    std::vector<double> fluid_velocities{};
    std::vector<double> fluid_pressures{};
  };

  /*!
   * @brief A group of block matrices for the off-diagonal coupling between solid dofs
   * (displacements + (optional) porosity) and fluid dofs (velocity + pressure)
   *
   * @tparam porosity_formulation
   */
  template <PorosityFormulation porosity_formulation>
  struct SolidPoroOffDiagonalBlockMatrices
  {
  };

  template <>
  struct SolidPoroOffDiagonalBlockMatrices<PorosityFormulation::from_material_law>
  {
    Core::LinAlg::SerialDenseMatrix* K_displacement_fluid_dofs = nullptr;
  };

  template <>
  struct SolidPoroOffDiagonalBlockMatrices<PorosityFormulation::as_primary_variable>
  {
    Core::LinAlg::SerialDenseMatrix* K_displacement_fluid_dofs = nullptr;
    Core::LinAlg::SerialDenseMatrix* K_porosity_pressure = nullptr;
  };

  /*!
   * @brief A group of block matrices for all solid dofs (displacements + (optional) porosity)
   *
   * @tparam porosity_formulation
   */
  template <PorosityFormulation porosity_formulation>
  struct SolidPoroDiagonalBlockMatrices
  {
  };

  template <>
  struct SolidPoroDiagonalBlockMatrices<PorosityFormulation::from_material_law>
  {
    Core::LinAlg::SerialDenseVector* force_vector = nullptr;
    Core::LinAlg::SerialDenseMatrix* K_displacement_displacement = nullptr;
  };

  template <>
  struct SolidPoroDiagonalBlockMatrices<PorosityFormulation::as_primary_variable>
  {
    Core::LinAlg::SerialDenseVector* force_vector = nullptr;
    Core::LinAlg::SerialDenseVector* porosity_force_vector = nullptr;
    Core::LinAlg::SerialDenseMatrix* K_displacement_displacement = nullptr;
    Core::LinAlg::SerialDenseMatrix* K_displacement_porosity = nullptr;
    Core::LinAlg::SerialDenseMatrix* K_porosity_displacement = nullptr;
    Core::LinAlg::SerialDenseMatrix* K_porosity_porosity = nullptr;
  };

  /*!
   * @brief Fluid phases general properties
   *
   */
  struct SolidPoroFluidProperties
  {
    int number_of_fluid_dofs_per_node_{};
    int number_of_fluid_phases_in_multiphase_porespace_{};
    int number_of_volfracs_{};
  };


  /*!
   * @brief Set fluid phases general properties from material
   *
   */
  inline SolidPoroFluidProperties evaluate_porofluid_properties(
      const Mat::FluidPoroMultiPhase& porofluidmat)
  {
    SolidPoroFluidProperties solidporo_fluid_properties;
    solidporo_fluid_properties.number_of_fluid_dofs_per_node_ = porofluidmat.num_mat();
    solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_ =
        porofluidmat.num_fluid_phases();
    solidporo_fluid_properties.number_of_volfracs_ = porofluidmat.num_vol_frac();
    return solidporo_fluid_properties;
  }

  /*!
   * @brief Check what closing relation is set in the volfrac material dependent on the number of
   * dofs, fluidphases and volfracs
   *
   */
  inline Mat::PAR::PoroFluidPressureBased::ClosingRelation get_volfrac_closing_relation_type(
      const SolidPoroFluidProperties solidporo_fluid_properties)
  {
    if (solidporo_fluid_properties.number_of_fluid_dofs_per_node_ ==
        solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_ +
            solidporo_fluid_properties.number_of_volfracs_)
      return Mat::PAR::PoroFluidPressureBased::ClosingRelation::evolutionequation_blood_lung;
    else if (solidporo_fluid_properties.number_of_fluid_dofs_per_node_ ==
             solidporo_fluid_properties.number_of_fluid_phases_in_multiphase_porespace_ +
                 solidporo_fluid_properties.number_of_volfracs_ * 2)
      return Mat::PAR::PoroFluidPressureBased::ClosingRelation::
          evolutionequation_homogenized_vasculature_tumor;
    else
    {
      FOUR_C_THROW("The Number of volume fractions is not correct in the solid evaluator!");
    }
  }

  inline SolidPoroPrimaryVariables<PorosityFormulation::from_material_law>
  extract_solid_poro_primary_variables(const Core::FE::Discretization& discretization,
      const Core::Elements::LocationArray& la, Core::FE::CellType celltype)
  {
    const int num_nodes = Core::FE::get_number_of_element_nodes(celltype);
    const int num_dim = Core::FE::get_dimension(celltype);

    std::vector<double> fluid_vel(num_nodes * num_dim, 0.0);
    std::vector<double> fluid_pres(num_nodes, 0.0);

    if (discretization.has_state(1, "fluidvel"))
    {
      const Core::LinAlg::Vector<double>& fluidvel = *discretization.get_state(1, "fluidvel");

      auto velocity_dofs = [i = -1, num_dim](const auto&) mutable
      {
        i = (i + 1) % (num_dim + 1);
        return i < num_dim;
      };
      auto pressure_dofs = [i = -1, num_dim](const auto&) mutable
      {
        i = (i + 1) % (num_dim + 1);
        return i == num_dim;
      };

      fluid_vel = Core::FE::extract_values(
          fluidvel, la[1].lm_ | std::views::filter(velocity_dofs), num_nodes * num_dim);
      fluid_pres = Core::FE::extract_values(
          fluidvel, la[1].lm_ | std::views::filter(pressure_dofs), num_nodes);
    }

    return {
        .solid_location_array = la[0].lm_,
        .solid_displacements =
            Core::FE::extract_values(*discretization.get_state(0, "displacement"), la[0].lm_),
        .fluid_location_array = la[1].lm_,
        .fluid_velocities = fluid_vel,

        .fluid_pressures = fluid_pres,
    };
  }

  inline SolidPoroPrimaryVariables<PorosityFormulation::as_primary_variable>
  extract_solid_poro_primary_variables(const Core::FE::Discretization& discretization,
      const Core::Elements::LocationArray& la, Core::FE::CellType celltype,
      const std::vector<double>& initial_porosity)
  {
    const int num_nodes = Core::FE::get_number_of_element_nodes(celltype);
    const int num_dim = Core::FE::get_dimension(celltype);

    std::vector<double> fluid_vel(num_nodes * num_dim, 0.0);
    std::vector<double> fluid_pres(num_nodes, 0.0);

    if (discretization.has_state(1, "fluidvel"))
    {
      const Core::LinAlg::Vector<double>& fluidvel = *discretization.get_state(1, "fluidvel");

      auto velocity_dofs = [i = -1, num_dim](const auto&) mutable
      {
        i = (i + 1) % (num_dim + 1);
        return i < num_dim;
      };
      auto pressure_dofs = [i = -1, num_dim](const auto&) mutable
      {
        i = (i + 1) % (num_dim + 1);
        return i == num_dim;
      };

      fluid_vel = Core::FE::extract_values(
          fluidvel, la[1].lm_ | std::views::filter(velocity_dofs), num_nodes * num_dim);
      fluid_pres = Core::FE::extract_values(
          fluidvel, la[1].lm_ | std::views::filter(pressure_dofs), num_nodes);
    }

    // for explicit porosity (solid-poro-p1), the porosity is stored additionally in the solid
    // field (d_x^1, d_y^1, d_z^1, porosity^1, ...)
    // We need to split the location array into a displacement and a porosity part
    const Core::LinAlg::Vector<double>& displacements =
        *discretization.get_state(0, "displacement");

    auto displacement_dofs = [i = -1, num_dim](const auto&) mutable
    {
      i = (i + 1) % (num_dim + 1);
      return i < num_dim;
    };
    auto porosity_dofs = [i = -1, num_dim](const auto&) mutable
    {
      i = (i + 1) % (num_dim + 1);
      return i == num_dim;
    };

    std::vector<int> displacement_location_array;
    displacement_location_array.reserve(num_nodes * num_dim);
    std::ranges::copy(la[0].lm_ | std::views::filter(displacement_dofs),
        std::back_inserter(displacement_location_array));


    return {
        .solid_location_array = displacement_location_array,
        .solid_displacements = Core::FE::extract_values(displacements, displacement_location_array),

        .initial_porosity = initial_porosity,
        .porosity = Core::FE::extract_values(
            displacements, la[0].lm_ | std::views::filter(porosity_dofs), num_nodes),

        .fluid_location_array = la[1].lm_,
        .fluid_velocities = fluid_vel,
        .fluid_pressures = fluid_pres,
    };
  }

  template <Core::FE::CellType celltype>
  constexpr auto get_gauss_rule_stiffness_matrix_poro()
  {
    return Discret::Elements::DisTypeToOptGaussRule<celltype>::rule;
  }

  /*!
   * @brief Calculate volume change
   *
   * @tparam celltype: Cell type
   * @param displacements (in) : An object holding the displacements of the element nodes
   * @param spatial_material_mapping (in) : An object holding quantities of the spatial material
   * mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient)
   * @param jacobian_mapping (in) : An object holding quantities of the jacobian mapping
   * (inverse Jacobian, determinant, derivatives of the shape functions w.r.t. XYZ)
   * @param ele (in) : Element
   * @param discretization (in) : discretization
   * @param lm (in) : Location vector of the element, i.e., global dof numbers of elemental dofs
   * @param kinematictype (in): kinematic type of element
   * @return volchange: volume change
   */
  template <Core::FE::CellType celltype>
  inline double compute_volume_change(
      const Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::num_nodes(celltype)>&
          displacements,
      const SpatialMaterialMapping<celltype>& spatial_material_mapping,
      const JacobianMapping<celltype>& jacobian_mapping, const Core::Elements::Element& ele,
      const Inpar::Solid::KinemType& kinematictype)
  {
    if (kinematictype == Inpar::Solid::KinemType::linear)
    {
      Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> dispgrad(
          Core::LinAlg::Initialization::zero);
      // gradient of displacements
      dispgrad.multiply_nt(displacements, Core::LinAlg::make_matrix_view(jacobian_mapping.N_XYZ));

      double volchange = 1.0;
      // volchange = 1 + trace of the linearized strains (= trace of displacement gradient)
      for (int i = 0; i < Internal::num_dim<celltype>; ++i) volchange += dispgrad(i, i);

      return volchange;
    }
    else
    {
      return spatial_material_mapping.determinant_deformation_gradient_;
    }
  }

  /*!
   * @brief Calculate derivative of determinant of deformation gradient w.r.t.
   * the displacements
   *
   * @tparam celltype: Cell type
   * @param spatial_material_mapping (in) : An object holding quantities of the spatial material
   * mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient)
   * @param jacobian_mapping (in) : An object holding quantities of the jacobian mapping
   * (inverse Jacobian, determinant, derivatives of the shape functions w.r.t. XYZ)
   * @param kinematictype (in): kinematic type of element
   * @return ddet_defgrd_ddisp: derivative of determinant of deformation gradient w.r.t.
   * the displacements
   */
  template <Core::FE::CellType celltype>
  inline Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>
  compute_linearization_of_detdefgrad_wrt_disp(
      const SpatialMaterialMapping<celltype> spatial_material_mapping,
      const JacobianMapping<celltype> jacobian_mapping,
      const Inpar::Solid::KinemType& kinematictype)
    requires(Internal::num_dim<celltype> == 3)
  {
    Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>> ddet_defgrd_ddisp;

    if (kinematictype == Inpar::Solid::KinemType::linear)
    {
      ddet_defgrd_ddisp.clear();
      return ddet_defgrd_ddisp;
    }
    else
    {
      Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_dim<celltype>, 1>
          defgrd_inv_vec;
      defgrd_inv_vec(0) = spatial_material_mapping.inverse_deformation_gradient_(0, 0);
      defgrd_inv_vec(1) = spatial_material_mapping.inverse_deformation_gradient_(0, 1);
      defgrd_inv_vec(2) = spatial_material_mapping.inverse_deformation_gradient_(0, 2);
      defgrd_inv_vec(3) = spatial_material_mapping.inverse_deformation_gradient_(1, 0);
      defgrd_inv_vec(4) = spatial_material_mapping.inverse_deformation_gradient_(1, 1);
      defgrd_inv_vec(5) = spatial_material_mapping.inverse_deformation_gradient_(1, 2);
      defgrd_inv_vec(6) = spatial_material_mapping.inverse_deformation_gradient_(2, 0);
      defgrd_inv_vec(7) = spatial_material_mapping.inverse_deformation_gradient_(2, 1);
      defgrd_inv_vec(8) = spatial_material_mapping.inverse_deformation_gradient_(2, 2);

      // build N_X operator (w.r.t. material configuration)
      Core::LinAlg::Matrix<9, Internal::num_dof_per_ele<celltype>> N_X(
          Core::LinAlg::Initialization::zero);  // set to zero
      for (int i = 0; i < Internal::num_nodes<celltype>; ++i)
      {
        N_X(0, 3 * i + 0) = jacobian_mapping.N_XYZ[i](0);
        N_X(1, 3 * i + 1) = jacobian_mapping.N_XYZ[i](0);
        N_X(2, 3 * i + 2) = jacobian_mapping.N_XYZ[i](0);

        N_X(3, 3 * i + 0) = jacobian_mapping.N_XYZ[i](1);
        N_X(4, 3 * i + 1) = jacobian_mapping.N_XYZ[i](1);
        N_X(5, 3 * i + 2) = jacobian_mapping.N_XYZ[i](1);

        N_X(6, 3 * i + 0) = jacobian_mapping.N_XYZ[i](2);
        N_X(7, 3 * i + 1) = jacobian_mapping.N_XYZ[i](2);
        N_X(8, 3 * i + 2) = jacobian_mapping.N_XYZ[i](2);
      }

      // linearization of determinant of deformation gradient detF w.r.t. structural displacement
      // u_s dDetF/du_s = dDetF/dF : dF/du_s = DetF * F^-T * N,X
      ddet_defgrd_ddisp.multiply_tn(
          spatial_material_mapping.determinant_deformation_gradient_, defgrd_inv_vec, N_X);
      return ddet_defgrd_ddisp;
    }
  }

  /*!
   * @brief Calculate derivative of volume change w.r.t. the displacements
   *
   * @tparam celltype: Cell type
   * @param ddet_defgrd_ddisp (in) : derivative of determinant of deformation gradient w.r.t.
   * the displacements
   * @param jacobian_mapping (in) : An object holding quantities of the jacobian mapping
   * (inverse Jacobian, determinant, derivatives of the shape functions w.r.t. XYZ)
   * @param kinematictype (in): kinematic type of element
   * @return dVolchange_dDisp: derivative of volume change w.r.t. the displacements
   */
  template <Core::FE::CellType celltype>
  inline Core::LinAlg::Matrix<1, Internal::num_dim<celltype> * Internal::num_nodes<celltype>>
  compute_linearization_of_volchange_wrt_disp(
      const Core::LinAlg::Matrix<1, Internal::num_dim<celltype> * Internal::num_nodes<celltype>>
          ddet_defgrd_ddisp,
      const JacobianMapping<celltype>& jacobian_mapping,
      const Inpar::Solid::KinemType& kinematictype)
    requires(Internal::num_dim<celltype> == 3)
  {
    if (kinematictype == Inpar::Solid::KinemType::linear)
    {
      Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>> dVolchange_dDisp;

      for (int i = 0; i < Internal::num_dim<celltype>; ++i)
        for (int j = 0; j < Internal::num_nodes<celltype>; ++j)
          dVolchange_dDisp(Internal::num_dim<celltype> * j + i) = jacobian_mapping.N_XYZ[j](i);

      return dVolchange_dDisp;
    }
    else
    {
      return ddet_defgrd_ddisp;
    }
  }

  /*!
   * @brief Calculate porosity depending on poro law given in input file and derivatives of
   * multiphase fluid primary variables w.r.t. the displacements
   *
   * @tparam celltype: Cell type
   * @param porostructmat (in) : material of skeleton (solid phase of porous domain)
   * @param params (in) : List of additional parameter to pass quantities from the time integrator
   * to the material
   * @param solidpressure (in): solid pressure
   * @param gp (in): Gauss point
   * @param volchange (in): volume change
   * @param porosity (in/out): porosity (volfrac of multiphase porspace + volfracs of additional
   *  @param ddet_defgrd_ddisp (in): derivative of determinant of deformation gradient w.r.t. the
   * displacements
   * @param dPorosity_dDisp (in/out): derivative of porosity w.r.t. the displacements
   */
  template <Core::FE::CellType celltype>
  inline void compute_porosity_and_linearization(Mat::StructPoro& porostructmat,
      Teuchos::ParameterList& params, const double solidpressure, const int gp,
      const double volchange, double& porosity,
      const Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& ddet_defgrd_ddisp,
      Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& dPorosity_dDisp)
  {
    double dphi_dJ = 0.0;

    porostructmat.compute_porosity(params, solidpressure, volchange, gp, porosity,
        nullptr,  // dphi_dp not needed
        &dphi_dJ,
        nullptr,  // dphi_dJdp not needed
        nullptr,  // dphi_dJJ not needed
        nullptr   // dphi_dpp not needed
    );

    dPorosity_dDisp.update(dphi_dJ, ddet_defgrd_ddisp);
  }

  /*!
   * @brief Calculate porosity depending on poro law given in input file and derivatives of
   * multiphase fluid primary variables w.r.t. the displacements
   *
   * @tparam celltype: Cell type
   * @param porostructmat (in) : material of skeleton (solid phase of porous domain)
   * @param params (in) : List of additional parameter to pass quantities from the time integrator
   * to the material
   * @param solidpressure (in): solid pressure
   * @param gp (in): Gauss point
   * @param volchange (in): volume change
   * @param porosity (in/out): porosity (volfrac of multiphase porspace + volfracs of additional
   *  @param ddet_defgrd_ddisp (in): derivative of determinant of deformation gradient w.r.t. the
   * displacements
   * @param dPorosity_dDisp (in/out): derivative of porosity w.r.t. the displacements
   * * @param dPorosity_dDetDefGrad (in/out): derivative of porosity w.r.t. the determinant of the
   * deformation gradient
   */
  template <Core::FE::CellType celltype>
  inline void compute_porosity_and_linearization(Mat::StructPoro& porostructmat,
      Teuchos::ParameterList& params, const double solidpressure, const int gp,
      const double volchange, double& porosity,
      const Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& ddet_defgrd_ddisp,
      Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& dPorosity_dDisp,
      double& dPorosity_dDetDefGrad)
  {
    porostructmat.compute_porosity(params, solidpressure, volchange, gp, porosity,
        nullptr,  // dphi_dp not needed
        &dPorosity_dDetDefGrad,
        nullptr,  // dphi_dJdp not needed
        nullptr,  // dphi_dJJ not needed
        nullptr   // dphi_dpp not needed
    );

    dPorosity_dDisp.update(dPorosity_dDetDefGrad, ddet_defgrd_ddisp);
  }



  // solid nodal displacement and velocity values
  template <Core::FE::CellType celltype, PorosityFormulation porosity_formulation>
  struct SolidVariables
  {
  };

  template <Core::FE::CellType celltype>
  struct SolidVariables<celltype, PorosityFormulation::from_material_law>
  {
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_nodes<celltype>>
        soliddisp_nodal{};
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_nodes<celltype>>
        solidvel_nodal{};
  };

  template <Core::FE::CellType celltype>
  struct SolidVariables<celltype, PorosityFormulation::as_primary_variable>
  {
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_nodes<celltype>>
        soliddisp_nodal{};
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_nodes<celltype>>
        solidvel_nodal{};
    Core::LinAlg::Matrix<Internal::num_nodes<celltype>, 1> solid_porosity_nodal{};
  };

  /*!
   * @brief Get nodal primary solid variables from structure field
   *
   * @tparam celltype: Cell type
   * @param discretization (in) : discretization
   * @param la (in): LocationArray of this element inside discretization
   * @return solid_variables:  nodal primary solid variables (displacement and velocity)
   */
  template <Core::FE::CellType celltype, PorosityFormulation porosity_formulation>
  inline SolidVariables<celltype, porosity_formulation> get_solid_variable_views(
      const Core::FE::Discretization& dis,
      const SolidPoroPrimaryVariables<porosity_formulation>& primary_variables)
  {
    std::vector<double> solid_velocity(
        Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>, 0.0);

    if (dis.has_state(0, "velocity"))
    {
      solid_velocity = Core::FE::extract_values(
          *dis.get_state(0, "velocity"), primary_variables.solid_location_array);
    }


    if constexpr (porosity_formulation == PorosityFormulation::from_material_law)
    {
      return SolidVariables<celltype, porosity_formulation>{
          .soliddisp_nodal =
              Core::FE::get_element_dof_matrix_view<celltype, Core::FE::dim<celltype>>(
                  primary_variables.solid_displacements),

          // Note: solid velocity is not a view since it is directly extracted from the global
          // variables
          .solidvel_nodal =
              Core::FE::get_element_dof_matrix<celltype, Core::FE::dim<celltype>>(solid_velocity),
      };
    }
    else
    {
      return SolidVariables<celltype, porosity_formulation>{
          .soliddisp_nodal =
              Core::FE::get_element_dof_matrix_view<celltype, Core::FE::dim<celltype>>(
                  primary_variables.solid_displacements),

          // Note: solid velocity is not a view since it is directly extracted from the global
          // variables
          .solidvel_nodal =
              Core::FE::get_element_dof_matrix<celltype, Core::FE::dim<celltype>>(solid_velocity),

          .solid_porosity_nodal = Core::LinAlg::Matrix<Core::FE::num_nodes(celltype), 1>(
              primary_variables.porosity.data(), true),
      };
    }
  }

  template <Core::FE::CellType celltype, PorosityFormulation porosity_formulation>
  inline double compute_porosity_and_linearization(Mat::StructPoro& porostructmat,
      Teuchos::ParameterList& params,
      const SolidVariables<celltype, porosity_formulation>& solid_variables,
      const ShapeFunctionsAndDerivatives<celltype> shape_functions, const double solidpressure,
      const int gp, const double volchange,
      const Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& ddet_defgrd_ddisp,
      Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& dPorosity_dDisp)
  {
    if constexpr (porosity_formulation == PorosityFormulation::as_primary_variable)
    {
      dPorosity_dDisp.put_scalar(0.0);

      return shape_functions.shapefunctions_.dot(solid_variables.solid_porosity_nodal);
    }
    else
    {
      // Porosity is implicitly given by the solid-poro material
      double dphi_dJ = 0.0;
      double porosity = 0.0;

      porostructmat.compute_porosity(params, solidpressure, volchange, gp, porosity,
          nullptr,  // dphi_dp not needed
          &dphi_dJ,
          nullptr,  // dphi_dJdp not needed
          nullptr,  // dphi_dJJ not needed
          nullptr   // dphi_dpp not needed
      );

      dPorosity_dDisp.update(dphi_dJ, ddet_defgrd_ddisp);

      return porosity;
    }
  };

  template <Core::FE::CellType celltype, PorosityFormulation porosity_formulation>
  inline double compute_porosity(Mat::StructPoro& porostructmat, Teuchos::ParameterList& params,
      const SolidVariables<celltype, porosity_formulation>& solid_variables,
      const ShapeFunctionsAndDerivatives<celltype> shape_functions, const double solidpressure,
      const int gp, const double volchange)
  {
    if constexpr (porosity_formulation == PorosityFormulation::as_primary_variable)
    {
      return shape_functions.shapefunctions_.dot(solid_variables.solid_porosity_nodal);
    }
    else
    {
      // Porosity is implicitly given by the solid-poro material
      double porosity = 0.0;

      porostructmat.compute_porosity(params, solidpressure, volchange, gp, porosity,
          nullptr,  // dphi_dp not needed
          nullptr,
          nullptr,  // dphi_dJdp not needed
          nullptr,  // dphi_dJJ not needed
          nullptr   // dphi_dpp not needed
      );

      return porosity;
    }
  };

  // porosity and derivative of porosity w.r.t. the pressure at gauss point
  struct PorosityAndLinearizationOD
  {
    double porosity = 0.0;
    double d_porosity_d_pressure = 0.0;
  };


  /*!
   * @brief Calculate porosity depending on poro law given in input file and derivative w.r.t. fluid
   * pressure
   *
   * @tparam celltype: Cell type
   * @param porostructmat (in) : material of skeleton (solid phase of porous domain)
   * @param params (in) : List of additional parameter to pass quantities from the time integrator
   * to the material
   * @param solidpressure (in): solid pressure
   * @param volchange (in): volume change
   * @param gp (in): Gauss point
   */
  template <Core::FE::CellType celltype, PorosityFormulation porosity_formulation>
  inline PorosityAndLinearizationOD compute_porosity_and_linearization_od(
      Mat::StructPoro& porostructmat, Teuchos::ParameterList& params,
      const SolidVariables<celltype, porosity_formulation>& solid_variables,
      const ShapeFunctionsAndDerivatives<celltype> shape_functions, const double solidpressure,
      const double volchange, const int gp)
  {
    if constexpr (porosity_formulation == PorosityFormulation::as_primary_variable)
    {
      return {
          .porosity = shape_functions.shapefunctions_.dot(solid_variables.solid_porosity_nodal),
          .d_porosity_d_pressure = 0.0,
      };
    }

    PorosityAndLinearizationOD porosity_and_linearization_od{};
    porostructmat.compute_porosity(
        params, solidpressure, volchange, gp, porosity_and_linearization_od.porosity,
        &porosity_and_linearization_od.d_porosity_d_pressure,  // first derivative of porosity
                                                               // w.r.t. pressure at gauss point
        nullptr,                                               // dphi_dJ not needed
        nullptr,                                               // dphi_dJdp not needed
        nullptr,                                               // dphi_dJJ not needed
        nullptr                                                // dphi_dpp not needed
    );

    return porosity_and_linearization_od;
  }

  // fluid nodal pressure and velocity values
  template <Core::FE::CellType celltype>
  struct FluidVariables
  {
    Core::LinAlg::Matrix<Internal::num_nodes<celltype>, 1> fluidpress_nodal{};
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_nodes<celltype>>
        fluidvel_nodal{};
  };

  /*!
   * @brief Get nodal primary fluid variables from fluid field
   *
   * @tparam celltype: Cell type
   * @param ele (in) : element
   * @param discretization (in) : discretization
   * @param la (in): LocationArray of this element inside discretization
   * @return fluid_variables:  nodal primary fluid variables (pressure and velocity)
   */
  template <Core::FE::CellType celltype, PorosityFormulation porosity_formulation>
  inline FluidVariables<celltype> get_fluid_variable_views(
      const SolidPoroPrimaryVariables<porosity_formulation>& primary_variables)
  {
    FluidVariables<celltype> fluid_variables{
        .fluidpress_nodal = Core::LinAlg::Matrix<Core::FE::num_nodes(celltype), 1>(
            primary_variables.fluid_pressures.data(), true),
        .fluidvel_nodal = Core::FE::get_element_dof_matrix_view<celltype, Core::FE::dim<celltype>>(
            primary_variables.fluid_velocities)};

    return fluid_variables;
  }

  // get values at integration point
  template <Core::FE::CellType celltype>
  inline Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1> interpolate_nodal_value_to_gp(
      Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_nodes<celltype>> node_values,
      const ShapeFunctionsAndDerivatives<celltype> shapefunctions)
  {
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1> value_at_gp(
        Core::LinAlg::Initialization::zero);
    value_at_gp.multiply(node_values, shapefunctions.shapefunctions_);
    return value_at_gp;
  }


  /*!
   * @brief Calculate porosity depending on poro law given in input file
   *
   * @tparam celltype: Cell type
   * @param porostructmat (in) : material of skeleton (solid phase of porous domain)
   * @param params (in) : List of additional parameter to pass quantities from the time
   * integrator to the material
   * @param solidpressure (in): solid pressure
   * @param volchange (in): volume change
   * @param gp (in): Gauss point
   * @return porosity (volfrac of multiphase porspace + volfracs of additional porous networks)
   */
  template <Core::FE::CellType celltype>
  inline double compute_porosity(Mat::StructPoro& porostructmat, Teuchos::ParameterList& params,
      const double solidpressure, const double volchange, const int gp)
  {
    double porosity = 0.0;
    porostructmat.compute_porosity(params, solidpressure, volchange, gp, porosity, nullptr,
        nullptr,  // dphi_dJ not needed
        nullptr,  // dphi_dJdp not needed
        nullptr,  // dphi_dJJ not needed
        nullptr   // dphi_dpp not needed
    );
    return porosity;
  }

  /*!
   * @brief Recalculate derivative of solidpressure w.r.t. the displacements in case of volfracs
   *
   * @tparam celltype: Cell type
   * @param fluidpress (in) : solid pressure contribution coming from the multiphase fluid S_i*p_i
   * @param porosity (in) : porosity = volumefraction in multiphase porespace + volfracs from
   * additional porous networks
   * @param nummultifluiddofpernode (in): number of fluid multiphase dofs per node
   * @param numfluidphases (in): number of fluidphases in multiphase porespace
   * @param numvolfrac (in): number of volfracs
   * @param fluidmultiphase_phi_at_gp (in): fluid multiphase primary variables at GP
   * @param dPorosity_dDisp (in): derivative of porosity w.r.t. the displacements
   * @param dsolidpressure_ddisp (in/out): derivative of solidpressure w.r.t. the displacements
   */
  template <Core::FE::CellType celltype>
  inline void recalculate_linearization_of_solpress_wrt_disp(const double fluidpress,
      const double porosity, const int nummultifluiddofpernode, const int numfluidphases,
      const int numvolfrac, const std::vector<double>& fluidmultiphase_phi_at_gp,
      const Core::LinAlg::Matrix<1, Internal::num_dim<celltype> * Internal::num_nodes<celltype>>&
          dPorosity_dDisp,
      Core::LinAlg::Matrix<1, Internal::num_dim<celltype> * Internal::num_nodes<celltype>>&
          dsolidpressure_ddisp)
  {
    // get volume fraction primary variables
    std::vector<double> volfracphi(fluidmultiphase_phi_at_gp.data() + numfluidphases,
        fluidmultiphase_phi_at_gp.data() + numfluidphases + numvolfrac);
    double sumaddvolfrac = 0.0;
    for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++) sumaddvolfrac += volfracphi[ivolfrac];

    // get volume fraction pressure at [numfluidphases+numvolfrac...nummultifluiddofpernode-1]
    std::vector<double> volfracpressure(
        fluidmultiphase_phi_at_gp.data() + numfluidphases + numvolfrac,
        fluidmultiphase_phi_at_gp.data() + nummultifluiddofpernode);

    // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
    //       + 1.0 / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
    // d (p_s) / d porosity = + sumaddvolfrac/porosity/porosity * fluidpress
    double dps_dphi = sumaddvolfrac / (porosity * porosity) * fluidpress;

    // ... + 1.0 / porosity / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
    for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
      dps_dphi -= volfracphi[ivolfrac] * volfracpressure[ivolfrac] / (porosity * porosity);

    // d (p_s) / d u_s = d (p_s) / d porosity * d porosity / d u_s
    dsolidpressure_ddisp.update(dps_dphi, dPorosity_dDisp);
  }

  /*!
   * @brief Recalculate derivative of solidpressure w.r.t. the displacements in case of volfracs
   *
   * @tparam celltype: Cell type
   * @param fluidpress (in) : solid pressure contribution coming from the multiphase fluid S_i*p_i
   * @param porosity (in) : porosity = volumefraction in multiphase porespace + volfracs from
   * additional porous networks
   * @param numfluidphases (in): number of fluidphases in multiphase porespace
   * @param fluidmultiphase_phi_at_gp (in): fluid multiphase primary variables at GP
   * @param dsolidpressure_ddisp (in/out): derivative of solidpressure w.r.t. the displacements
   * @param determinant_deformation_gradient (in): determinant of deformation gradient
   * @param porofluidmat (in): fluid material
   * @param dPorosity_dDetDefGrad (in): derivative of porosity w.r.t. determinant of derformation
   * gradient
   * @param dDetDefGrad_dDisp (in): derivative of determinant of derformation w.r.t. displacements
   */
  template <Core::FE::CellType celltype>
  inline void recalculate_linearization_of_solpress_wrt_disp(const double fluidpress,
      const double porosity, const int numfluidphases,
      const std::vector<double>& fluidmultiphase_phi_at_gp,
      Core::LinAlg::Matrix<1, Internal::num_dim<celltype> * Internal::num_nodes<celltype>>&
          dsolidpressure_ddisp,
      const double determinant_deformation_gradient, const Mat::FluidPoroMultiPhase& porofluidmat,
      const double dPorosity_dDetDefGrad,
      const Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& dDetDefGrad_dDisp)
  {
    // get volfrac pressures at [numfluidphases] so far only one volfrac pressure with closing
    // relation for blood lung is possible
    const double volfracpressure = fluidmultiphase_phi_at_gp[numfluidphases];

    // so far only one volfrac with closing relation for blood lung is possible
    const double volfrac =
        calculate_volfrac_from_closing_relation_blood_lung(fluidmultiphase_phi_at_gp,
            volfracpressure, determinant_deformation_gradient, porofluidmat, numfluidphases);

    // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
    //       + 1.0 / porosity * volfrac *  volfracpressure
    // d (p_s) / d porosity = + volfrac/porosity/porosity * fluidpress

    const double dvolfrac_dDetDefGrad = std::invoke(
        [&]()
        {
          if ((fluidmultiphase_phi_at_gp[0] / volfracpressure) <= 1.0)
          {
            return (PoroPressureBased::ElementUtils::
                        get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                            porofluidmat, numfluidphases)
                            .scaling_parameter_deformation() *
                    PoroPressureBased::ElementUtils::
                        get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                            porofluidmat, numfluidphases)
                            .initial_volfrac() *
                    pow(determinant_deformation_gradient,
                        PoroPressureBased::ElementUtils::
                                get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                                    porofluidmat, numfluidphases)
                                    .scaling_parameter_deformation() -
                            1.0));
          }
          else
          {
            return (PoroPressureBased::ElementUtils::
                           get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                               porofluidmat, numfluidphases)
                               .scaling_parameter_deformation() *
                       PoroPressureBased::ElementUtils::
                           get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                               porofluidmat, numfluidphases)
                               .initial_volfrac() *
                       pow(determinant_deformation_gradient,
                           PoroPressureBased::ElementUtils::
                                   get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                                       porofluidmat, numfluidphases)
                                       .scaling_parameter_deformation() -
                               1.0)) *
                   pow(fluidmultiphase_phi_at_gp[0] / volfracpressure,
                       PoroPressureBased::ElementUtils::
                           get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                               porofluidmat, numfluidphases)
                               .scaling_parameter_pressure());
          }
        });

    double dvolfrac_multiphaseporespace_dDetDefGrad = dPorosity_dDetDefGrad - dvolfrac_dDetDefGrad;

    double dps_dDetDefGrad =
        dvolfrac_multiphaseporespace_dDetDefGrad * 1.0 / porosity * fluidpress +
        (porosity - volfrac) * (-1.0 * 1.0 / (porosity * porosity) * dPorosity_dDetDefGrad) *
            fluidpress +
        dvolfrac_dDetDefGrad * volfracpressure * 1.0 / porosity +
        volfrac * (1.0 * 1.0 / (porosity * porosity) * dPorosity_dDetDefGrad) * volfracpressure;

    // d (p_s) / d u_s = d (p_s) / d J * d J / d u_s
    dsolidpressure_ddisp.update(dps_dDetDefGrad, dDetDefGrad_dDisp);
  }



  /*!
   * @brief Recalculate solidpressure in case of volfracs
   *
   * @param press (in) : current solid pressure
   * @param porosity (in) : porosity = volumefraction in multiphase porespace + volfracs from
   * additional porous networks
   * @param nummultifluiddofpernode (in): number of fluid multiphase dofs per node
   * @param numfluidphases (in): number of fluidphases in multiphase porespace
   * @param numvolfrac (in): number of volfracs
   * @param fluidmultiphase_phi_at_gp (in): fluid multiphase primary variables at GP
   * @return solid pressure
   */
  inline double recalculate_sol_pressure_at_gp(double press, const double porosity,
      const int nummultifluiddofpernode, const int numfluidphases, const int numvolfrac,
      const std::vector<double>& fluidmultiphase_phi_at_gp)
  {
    // get volume fraction primary variables at [numfluidphases-1...numfluidphase-1+numvolfrac]
    std::vector<double> volfracphi(fluidmultiphase_phi_at_gp.data() + numfluidphases,
        fluidmultiphase_phi_at_gp.data() + numfluidphases + numvolfrac);
    double sumaddvolfrac = 0.0;
    for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++) sumaddvolfrac += volfracphi[ivolfrac];

    // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
    //      + 1.0 / porosity * sum_i=1^numvolfrac (volfrac_i*pressure_i)
    // first part
    press *= (porosity - sumaddvolfrac) / porosity;

    // get volfrac pressures at [numfluidphases+numvolfrac...nummultifluiddofpernode-1]
    std::vector<double> volfracpressure(
        fluidmultiphase_phi_at_gp.data() + numfluidphases + numvolfrac,
        fluidmultiphase_phi_at_gp.data() + nummultifluiddofpernode);

    // second part
    for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
      press += volfracphi[ivolfrac] / porosity * volfracpressure[ivolfrac];

    // note: in recalculate_solid_pressure in porofluid_phasemanager calculation is performed a bit
    //       differently since we already pass porosity = porosity - sumaddvolfrac, but result is
    //       equivalent

    return press;
  }

  /*!
   * @brief Recalculate solidpressure in case of volfracs
   *
   * @param press (in) : current solid pressure
   * @param porosity (in) : porosity = volumefraction in multiphase porespace + volfracs from
   * additional porous networks
   * @param numfluidphases (in): number of fluidphases in multiphase porespace
   * @param fluidmultiphase_phi_at_gp (in): fluid multiphase primary variables at GP
   * @param determinant_deformation_gradient (in): determinant of deformation gradient
   * @param porofluidmat (in): fluid material
   * @return solid pressure
   */
  inline double recalculate_sol_pressure_at_gp(double press, const double porosity,
      const int numfluidphases, const std::vector<double>& fluidmultiphase_phi_at_gp,
      const double determinant_deformation_gradient, const Mat::FluidPoroMultiPhase& porofluidmat)
  {
    // get volfrac pressures at [numfluidphases] so far only one volfrac pressure with closing
    // relation for blood lung is possible
    double volfracpressure = fluidmultiphase_phi_at_gp[numfluidphases];


    // so far only one volfrac with closing relation for blood lung is possible
    const double volfrac =
        calculate_volfrac_from_closing_relation_blood_lung(fluidmultiphase_phi_at_gp,
            volfracpressure, determinant_deformation_gradient, porofluidmat, numfluidphases);

    // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
    //      + 1.0 / porosity * volfrac * volfrac_pressure
    // first part
    press *= (porosity - volfrac) / porosity;

    // second part
    press += volfrac / porosity * volfracpressure;

    // note: in recalculate_solid_pressure in porofluid_phasemanager calculation is performed a
    // bit
    //       differently since we already pass porosity = porosity - sumaddvolfrac, but result
    //       is equivalent

    return press;
  }

  /*!
   * @brief Update the internal force vector with poroelasticity contribution of one Gauss point
   *
   * @tparam celltype: Cell type
   * @param detJ_w (in) : integration factor (Gauss point weight times the determinant of
   * the jacobian)
   * @param solidpressure (in) : solid pressure
   * @param det_defgrd (in) : determinant of deformation gradient
   * * @param bopCinv (in) : B^T . C^-1
   * @param force_vector (in/out) : Force vector where the local contribution is added to
   */
  template <Core::FE::CellType celltype>
  inline void update_internal_forcevector_with_fluidstressterm(const double detJ_w,
      const double solidpressure, const double det_defgrd,
      const Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, 1>& bopCinv,
      Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_nodes<celltype>, 1>&
          force_vector)
  {
    // additional fluid stress- stiffness term RHS -(B^T .  C^-1  * J * p^f * detJ * w(gp))
    double factor = -detJ_w * solidpressure * det_defgrd;
    force_vector.update(factor, bopCinv, 1.0);
  }


  /*!
   * @brief Compute the anisotropic permeability coefficients at the Gauss point
   *
   * @tparam celltype: Cell type
   * @param shapefct (in) : Shape functions at Gauss point
   * @param anisotropic_permeability_nodal_coeffs_ (in) : anisotropic permeability coefficients at
   * nodes of ele
   * @param anisotropic_permeability_coeffs (out) : Force vector where the local contribution is
   * added to
   */
  template <Core::FE::CellType celltype>
  std::vector<double> compute_anisotropic_permeability_coeffs_at_gp(
      const Core::LinAlg::Matrix<Internal::num_nodes<celltype>, 1>& shapefct,
      const std::vector<std::vector<double>>& anisotropic_permeability_nodal_coeffs)
  {
    std::vector<double> anisotropic_permeability_coeffs(Internal::num_dim<celltype>, 0.0);

    FOUR_C_ASSERT(std::all_of(anisotropic_permeability_nodal_coeffs.begin(),
                      anisotropic_permeability_nodal_coeffs.end(), [](const auto& nodal_coeffs)
                      { return nodal_coeffs.size() == Core::FE::num_nodes(celltype); }),
        "Given permeability coefficients do not match the number of nodes. Expecting {}.",
        Core::FE::num_nodes(celltype));

    for (int node = 0; node < Internal::num_nodes<celltype>; ++node)
    {
      const double shape_val = shapefct(node);
      for (int dim = 0; dim < Internal::num_dim<celltype>; ++dim)
      {
        anisotropic_permeability_coeffs[dim] +=
            shape_val * anisotropic_permeability_nodal_coeffs[dim][node];
      }
    }

    return anisotropic_permeability_coeffs;
  }

  /*!
   * @brief Update the internal force vector with structure-fluid coupling and reactive darcy terms
   *
   * @tparam celltype: Cell type
   * @param shapefunctions (in) : Shape functions
   * @param porofluidmat (in) : porofluid material
   * @param anisotropy_properties (in): anisotropic properties (nodal coefficients and directions)
   * @param spatial_material_mapping (in) :An object holding quantities of the spatial material
   * mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient)
   * @param porosity (in) : porosity
   * @param disp_velocity (in) : solid velocity
   * @param fluid_velocity (in) : fluid velocity
   * @param FinvGradp (in) : Inverse deformation gradient times gradient fluid pressure
   * @param force_vector (in/out) : Force vector where the local contribution is added to
   */
  template <Core::FE::CellType celltype>
  inline void update_internal_forcevector_with_structure_fluid_coupling_and_reactive_darcy_terms(
      const double detJ_w, Core::LinAlg::Matrix<Internal::num_nodes<celltype>, 1> shapefunctions,
      const Mat::FluidPoro& porofluidmat,
      const Discret::Elements::AnisotropyProperties& anisotropy_properties,
      const SpatialMaterialMapping<celltype>& spatial_material_mapping, const double porosity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& disp_velocity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& fluid_velocity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& FinvGradp,
      Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_nodes<celltype>, 1>&
          force_vector)
  {
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> matreatensor(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> reatensor(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> linreac_dphi(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> linreac_dJ(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1> rea_fluid_vel(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1> rea_disp_vel(
        Core::LinAlg::Initialization::zero);

    static Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> temp(
        Core::LinAlg::Initialization::zero);
    std::vector<double> anisotropic_permeability_coeffs =
        compute_anisotropic_permeability_coeffs_at_gp<celltype>(
            shapefunctions, anisotropy_properties.nodal_coeffs_);
    porofluidmat.compute_reaction_tensor(matreatensor,
        spatial_material_mapping.determinant_deformation_gradient_, porosity,
        anisotropy_properties.directions_, anisotropic_permeability_coeffs);
    porofluidmat.compute_lin_mat_reaction_tensor(linreac_dphi, linreac_dJ,
        spatial_material_mapping.determinant_deformation_gradient_, porosity);
    temp.multiply(1.0, matreatensor,
        Core::LinAlg::make_matrix_view(spatial_material_mapping.inverse_deformation_gradient_));
    reatensor.multiply_tn(
        Core::LinAlg::make_matrix_view(spatial_material_mapping.inverse_deformation_gradient_),
        temp);
    rea_disp_vel.multiply(reatensor, disp_velocity);
    rea_fluid_vel.multiply(reatensor, fluid_velocity);


    for (int idim = 0; idim < Internal::num_dim<celltype>; idim++)
    {
      const double reafvel_idim = rea_fluid_vel(idim);
      const double reac_vel_idim = rea_disp_vel(idim);
      const double Finvgradp_idim = FinvGradp(idim);

      for (int inode = 0; inode < Internal::num_nodes<celltype>; inode++)
      {
        const double fac = detJ_w * shapefunctions(inode);
        const double v = fac * porosity * porosity *
                         spatial_material_mapping.determinant_deformation_gradient_ *
                         spatial_material_mapping.determinant_deformation_gradient_;
        const int fk = Internal::num_dim<celltype> * inode;

        /*-------structure- fluid velocity coupling:  RHS "darcy-terms" - reacoeff * J^2 *  phi^2 *
         * v^f */
        (force_vector)(fk + idim) += -v * reafvel_idim;

        /* "reactive darcy-terms" reacoeff * J^2 *  phi^2 *  v^s */
        (force_vector)(fk + idim) += v * reac_vel_idim;

        /*-------structure- fluid pressure coupling: RHS * "pressure gradient terms" - J *  F^-T *
         * Grad(p) * phi */
        (force_vector)(fk + idim) += fac *
                                     spatial_material_mapping.determinant_deformation_gradient_ *
                                     Finvgradp_idim * (-porosity);
      }
    }
  }

  /*!
   * @brief Compute off-diagonal linearization of reaction tensor
   *
   * @tparam celltype: Cell type
   * @param porofluidmat (in) : porofluid material
   * @param shapefunctions (in) : Shape functions
   * @param spatial_material_mapping (in) :An object holding quantities of the spatial material
   * mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient)
   * @param porosity (in) : porosity
   * @param disp_velocity (in) : solid velocity
   * @param fluid_velocity (in) : fluid velocity
   * @param anisotropy_properties (in) : anisotropic properties (nodal coefficients and directions)
   * @param reatensor (in) : reaction tensor
   * @param linreac_dporosity (in/out) : Derivative of the material reaction tensor w.r.t. the
   * porosity
   * @param rea_fluid_vel (in/out) : reactive fluid velocity
   * @param rea_disp_vel (in/out) : reactive solid velocity
   */
  template <Core::FE::CellType celltype>
  inline void compute_linearization_of_reaction_tensor_od(const Mat::FluidPoro& porofluidmat,
      const Core::LinAlg::Matrix<Internal::num_nodes<celltype>, 1>& shapefunctions,
      const SpatialMaterialMapping<celltype>& spatial_material_mapping, double porosity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& disp_velocity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& fluid_velocity,
      const AnisotropyProperties& anisotropy_properties,
      Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>& reatensor,
      Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          linreac_dporosity,
      Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& rea_fluid_vel,
      Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& rea_disp_vel)
  {
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> matreatensor(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>
        linreac_ddet_defgrd(
            Core::LinAlg::Initialization::zero);  // Derivative of the material reaction tensor
                                                  // w.r.t. the determinant of the

    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> temp(
        Core::LinAlg::Initialization::zero);

    std::vector<double> anisotropic_permeability_coeffs =
        compute_anisotropic_permeability_coeffs_at_gp<celltype>(
            shapefunctions, anisotropy_properties.nodal_coeffs_);

    porofluidmat.compute_reaction_tensor(matreatensor,
        spatial_material_mapping.determinant_deformation_gradient_, porosity,
        anisotropy_properties.directions_, anisotropic_permeability_coeffs);

    porofluidmat.compute_lin_mat_reaction_tensor(linreac_dporosity, linreac_ddet_defgrd,
        spatial_material_mapping.determinant_deformation_gradient_, porosity);

    temp.multiply(1.0, matreatensor,
        Core::LinAlg::make_matrix_view(spatial_material_mapping.inverse_deformation_gradient_));
    reatensor.multiply_tn(
        Core::LinAlg::make_matrix_view(spatial_material_mapping.inverse_deformation_gradient_),
        temp);
    rea_disp_vel.multiply(reatensor, disp_velocity);
    rea_fluid_vel.multiply(reatensor, fluid_velocity);
  }

  /*!
   * @brief Update stiffness matrix with off-diogonal brinkmann flow contribution
   *
   * @tparam celltype: Cell type
   * @param integration_fac (in) : Integration factor (Gauss point weight times the determinant of
   * the jacobian)
   * @param viscosity (in) : viscosity of fluid phase
   * @param dporosity_dpressure (in) : Derivative of porosity w.r.t. fluid pressure
   * @param shapefunctions (in) : Shape functions
   * @param jacobian_mapping (in): An object holding quantities of the jacobian mapping
   * (inverse Jacobian, determinant, derivatives of the shape functions w.r.t. XYZ)
   * @param spatial_material_mapping (in) :An object holding quantities of the spatial material
   * mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient)
   * @param inverse_right_cauchy_green (in) : inverse right cauchygreen trensor
   * @param fvelder (in) : material fluid velocity gradient at integration point
   * @param bop (in) : Strain gradient (B-Operator)
   * @param rea_fluid_vel (in/out) : reactive fluid velocity
   * @param stiffness_matrix (in/out) : stiffness matrix where the local contribution is added to
   */
  template <Core::FE::CellType celltype>
  inline void update_stiffness_brinkman_flow_od(const double integration_fac,
      const double viscosity, const double porosity, const double dporosity_dpressure,
      const Core::LinAlg::Matrix<Internal::num_nodes<celltype>, 1>& shapefunctions,
      const JacobianMapping<celltype>& jacobian_mapping,
      const SpatialMaterialMapping<celltype>& spatial_material_mapping,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          inverse_right_cauchy_green,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>& fvelder,
      const Core::LinAlg::Matrix<Internal::num_str<celltype>, Internal::num_dof_per_ele<celltype>>&
          bop,
      Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_nodes<celltype>,
          (Internal::num_dim<celltype> + 1) * Internal::num_nodes<celltype>>& stiffness_matrix)
  {
    Core::LinAlg::Matrix<Internal::num_str<celltype>, 1> fstress;

    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> CinvFvel;
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> tmp;
    CinvFvel.multiply(inverse_right_cauchy_green, fvelder);
    tmp.multiply_nt(CinvFvel,
        Core::LinAlg::make_matrix_view(spatial_material_mapping.inverse_deformation_gradient_));
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> tmp2(tmp);
    tmp.update_t(1.0, tmp2, 1.0);

    fstress(0) = tmp(0, 0);
    fstress(1) = tmp(1, 1);
    fstress(2) = tmp(2, 2);
    fstress(3) = tmp(0, 1);
    fstress(4) = tmp(1, 2);
    fstress(5) = tmp(2, 0);

    // B^T . \sigma
    static Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, 1> fstressb;
    fstressb.multiply_tn(bop, fstress);
    static Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_nodes<celltype>>
        N_XYZ_Finv;
    N_XYZ_Finv.multiply(
        Core::LinAlg::make_matrix_view(spatial_material_mapping.inverse_deformation_gradient_),
        Core::LinAlg::make_matrix_view(jacobian_mapping.N_XYZ));

    // dfstress/dv^f
    Core::LinAlg::Matrix<Internal::num_str<celltype>, Internal::num_dof_per_ele<celltype>>
        dfstressb_dv;
    for (int j = 0; j < Internal::num_dim<celltype>; j++)
    {
      const double C_inv_0_j = inverse_right_cauchy_green(0, j);
      const double C_inv_1_j = inverse_right_cauchy_green(1, j);
      const double C_inv_2_j = inverse_right_cauchy_green(2, j);

      for (int i = 0; i < Internal::num_nodes<celltype>; i++)
      {
        const int k = Internal::num_dim<celltype> * i + j;
        const double N_XYZ_Finv_0_i = N_XYZ_Finv(0, i);
        const double N_XYZ_Finv_1_i = N_XYZ_Finv(1, i);
        const double N_XYZ_Finv_2_i = N_XYZ_Finv(2, i);

        dfstressb_dv(0, k) = 2 * N_XYZ_Finv_0_i * C_inv_0_j;
        dfstressb_dv(1, k) = 2 * N_XYZ_Finv_1_i * C_inv_1_j;
        dfstressb_dv(2, k) = 2 * N_XYZ_Finv_2_i * C_inv_2_j;
        //**********************************
        dfstressb_dv(3, k) = N_XYZ_Finv_0_i * C_inv_1_j + N_XYZ_Finv_1_i * C_inv_0_j;
        dfstressb_dv(4, k) = N_XYZ_Finv_1_i * C_inv_2_j + N_XYZ_Finv_2_i * C_inv_1_j;
        dfstressb_dv(5, k) = N_XYZ_Finv_2_i * C_inv_0_j + N_XYZ_Finv_0_i * C_inv_2_j;
      }
    }

    // B^T . dfstress/dv^f
    Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, Internal::num_dof_per_ele<celltype>>
        dfstressb_dv_bop(Core::LinAlg::Initialization::zero);
    dfstressb_dv_bop.multiply_tn(bop, dfstressb_dv);

    for (int i = 0; i < Internal::num_nodes<celltype>; i++)
    {
      const int fi = Internal::num_dof_per_node<celltype> * i;
      for (int j = 0; j < Internal::num_dim<celltype>; j++)
      {
        const double fstressb_i_j = fstressb(fi + j);

        for (int k = 0; k < Internal::num_nodes<celltype>; k++)
        {
          const int fk = Internal::num_dof_per_node<celltype> * k;
          const int fkp1 = (Internal::num_dim<celltype> + 1) * k;

          /*-------structure- fluid pressure coupling: "darcy-brinkman stress terms" B^T . ( \mu*J -
           * d(phi)/(dp) * fstress ) * Dp */
          (stiffness_matrix)(fi + j, fkp1 + Internal::num_dim<celltype>) +=
              integration_fac * fstressb_i_j * dporosity_dpressure * viscosity *
              spatial_material_mapping.determinant_deformation_gradient_ * shapefunctions(k);
          for (int l = 0; l < Internal::num_dof_per_node<celltype>; l++)
          {
            /*-------structure- fluid velocity coupling: "darcy-brinkman stress terms" B^T . ( \mu*J
             * - phi * dfstress/dv^f ) * Dp */
            (stiffness_matrix)(fi + j, fkp1 + l) +=
                integration_fac * viscosity *
                spatial_material_mapping.determinant_deformation_gradient_ * porosity *
                dfstressb_dv_bop(fi + j, fk + l);
          }
        }
      }
    }
  }


  /*!
   * @brief Add off-diagonal contribution of one Gauss point to stiffness matrix
   *
   * @tparam celltype : Cell type
   * @param integration_fac (in) : Integration factor (Gauss point weight times the determinant of
   * the jacobian)
   * * @param shapefunctions (in) : Shape functions
   * @param spatial_material_mapping (in) :An object holding quantities of the spatial material
   * mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient)
   * @param porosity (in) : porosity
   * @param dPorosity_dPressure (in) : Derivative of porosity w.r.t. fluid pressure
   * @param bopCinv (in) :B^T . C^-1
   * @param Finvgradp (in) : F^-T * grad p
   * @param FinvNXYZ (in) :  F^-T * N_XYZ   * @param porofluidmat (in) : porofluid material
   * @param disp_velocity (in) :  solid velocity
   * @param fluid_velocity (in) :  fluid velocity
   * * @param reatensor (in) : reaction tensor
   * @param linreac_dporosity (in) :  Derivative of the material reaction tensor w.r.t. the porosity
   * @param rea_fluid_vel (in) : reactive fluid velocity
   * @param rea_disp_vel (in) :  reactive solid velocity
   * @param stiffness_matrix (in/out) : stiffness matrix where the local contribution is added to
   */
  template <Core::FE::CellType celltype>
  inline void update_stiffness_matrix_od(const double integration_fac,
      const Core::LinAlg::Matrix<Internal::num_nodes<celltype>, 1>& shapefunctions,
      const SpatialMaterialMapping<celltype>& spatial_material_mapping, const double porosity,
      const double dPorosity_dPressure,
      const Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, 1>& bopCinv,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& Finvgradp,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_nodes<celltype>>&
          FinvNXYZ,
      const Mat::FluidPoro& porofluidmat,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& disp_velocity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& fluid_velocity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          reatensor,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          linreac_dporosity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& rea_fluid_vel,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& rea_disp_vel,
      Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_nodes<celltype>,
          (Internal::num_dim<celltype> + 1) * Internal::num_nodes<celltype>>& stiffness_matrix)
  {
    const double numdim_ = Internal::num_dim<celltype>;
    const double numnod_ = Internal::num_nodes<celltype>;

    {
      const double fac = integration_fac *
                         spatial_material_mapping.determinant_deformation_gradient_ *
                         spatial_material_mapping.determinant_deformation_gradient_ * 2 * porosity *
                         dPorosity_dPressure;
      for (int idim = 0; idim < numdim_; idim++)
      {
        const double reafvel_idim = rea_fluid_vel(idim);
        const double reac_vel_idim = rea_disp_vel(idim);

        for (int jnode = 0; jnode < numnod_; jnode++)
        {
          const int fkp1 = (numdim_ + 1) * jnode;

          const double val = fac * shapefunctions(jnode) * (reac_vel_idim - reafvel_idim);
          for (int inode = 0; inode < numnod_; inode++)
          {
            /*-------structure- fluid pressure coupling:  "dracy-terms" + "reactive darcy-terms" - 2
             * * reacoeff * J * v^f * phi * d(phi)/dp  Dp  + 2 * reacoeff * J * v^s * phi *
             * d(phi)/dp  Dp */
            (stiffness_matrix)(numdim_ * inode + idim, fkp1 + numdim_) +=
                shapefunctions(inode) * val;
          }
        }
      }
    }

    {
      for (int idim = 0; idim < numdim_; idim++)
      {
        const double Finvgradp_idim = Finvgradp(idim);
        for (int jnode = 0; jnode < numnod_; jnode++)
        {
          const int fkp1 = (numdim_ + 1) * jnode;

          const double val1 = integration_fac * (-1.0) *
                              spatial_material_mapping.determinant_deformation_gradient_ *
                              shapefunctions(jnode);
          const double val2 = -1.0 * integration_fac *
                              spatial_material_mapping.determinant_deformation_gradient_ *
                              (Finvgradp_idim * dPorosity_dPressure * shapefunctions(jnode) +
                                  porosity * FinvNXYZ(idim, jnode));

          for (int inode = 0; inode < numnod_; inode++)
          {
            /*-------structure- fluid pressure coupling: "stress terms" + "pressure gradient terms"
             * -B^T . ( -1*J*C^-1 ) * Dp - J * F^-T * dphi/dp * Dp - J * F^-T * d(Grad((p))/(dp) *
             * phi * Dp */
            (stiffness_matrix)(numdim_ * inode + idim, fkp1 + numdim_) +=
                val1 * bopCinv(numdim_ * inode + idim) + val2 * shapefunctions(inode);
          }
        }
      }
    }

    // check if derivatives of reaction tensor are zero --> significant speed up
    if (porofluidmat.permeability_function() != Mat::PAR::constant)
    {
      const double fac = integration_fac *
                         spatial_material_mapping.determinant_deformation_gradient_ *
                         spatial_material_mapping.determinant_deformation_gradient_ * porosity *
                         porosity * dPorosity_dPressure;
      for (int idim = 0; idim < numdim_; idim++)
      {
        for (int jnode = 0; jnode < numnod_; jnode++)
        {
          const int fkp1 = (numdim_ + 1) * jnode;
          const double shapefct_jnode = shapefunctions(jnode);

          for (int inode = 0; inode < numnod_; inode++)
          {
            double val = 0.0;
            for (int p = 0; p < numdim_; ++p)
            {
              const double velint_fvelint_p = disp_velocity(p) - fluid_velocity(p);
              for (int n = 0; n < numdim_; ++n)
              {
                const double defgrd_inv_n_p =
                    spatial_material_mapping.inverse_deformation_gradient_(n, p);
                for (int m = 0; m < numdim_; ++m)
                {
                  val += fac * spatial_material_mapping.inverse_deformation_gradient_(m, idim) *
                         linreac_dporosity(m, n) * defgrd_inv_n_p * velint_fvelint_p;
                }
              }
            }
            val *= shapefct_jnode;

            /*-------structure- fluid pressure coupling:   "reactive darcy-terms" + J * J * phi *
             * phi * defgrd_^-T * d(mat_reacoeff)/d(phi) * defgrd_^-1 * (v^s-v^f) * d(phi)/dp Dp */
            (stiffness_matrix)(numdim_ * inode + idim, fkp1 + numdim_) +=
                shapefunctions(inode) * val;
          }
        }
      }
    }

    {
      const double fac =
          integration_fac * spatial_material_mapping.determinant_deformation_gradient_ *
          spatial_material_mapping.determinant_deformation_gradient_ * porosity * porosity;
      for (int idim = 0; idim < numdim_; idim++)
      {
        for (int jdim = 0; jdim < numdim_; jdim++)
        {
          const double reatensor_idim_jdim = reatensor(idim, jdim);
          for (int jnode = 0; jnode < numnod_; jnode++)
          {
            const double val = -1.0 * fac * shapefunctions(jnode) * reatensor_idim_jdim;

            /*-------structure- fluid velocity coupling:  "darcy-terms"
              -reacoeff * J * J *  phi^2 *  Dv^f
            */
            for (int inode = 0; inode < numnod_; inode++)
              (stiffness_matrix)(numdim_ * inode + idim, (numdim_ + 1) * jnode + jdim) +=
                  val * shapefunctions(inode);
          }
        }
      }
    }
  }

  template <Core::FE::CellType celltype>
  inline void update_stiffness_matrix_with_structure_fluid_coupling_and_reactive_darcy_terms(
      const double detJ_w,
      const Core::LinAlg::Matrix<Internal::num_nodes<celltype>, 1>& shapefunctions,
      const Mat::FluidPoro& porofluidmat,
      const Discret::Elements::AnisotropyProperties& anisotropy_properties,
      const SpatialMaterialMapping<celltype>& spatial_material_mapping, const double porosity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& disp_velocity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& fluid_velocity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& FinvGradp,
      const Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& ddet_defgrd_ddisp,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_dim<celltype>,
          Internal::num_dim<celltype> * Internal::num_nodes<celltype>>& dinverse_defgrd_ddisp_gradp,
      const Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& dPorosity_dDisp,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_dim<celltype>,
          Internal::num_dof_per_ele<celltype>>& dInverseDeformationGradientTransposed_dDisp,
      Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>,
          Internal::num_dof_per_ele<celltype>>& erea_v,
      Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_nodes<celltype>,
          Internal::num_dim<celltype> * Internal::num_nodes<celltype>>& stiffness_matrix)
  {
    const double numdim_ = Internal::num_dim<celltype>;
    const double numnod_ = Internal::num_nodes<celltype>;
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> matreatensor(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> reatensor(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> linreac_dphi(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> linreac_dJ(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1> reafvel(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1> reavel(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> temp(
        Core::LinAlg::Initialization::zero);
    std::vector<double> anisotropic_permeability_coeffs =
        compute_anisotropic_permeability_coeffs_at_gp<celltype>(
            shapefunctions, anisotropy_properties.nodal_coeffs_);
    porofluidmat.compute_reaction_tensor(matreatensor,
        spatial_material_mapping.determinant_deformation_gradient_, porosity,
        anisotropy_properties.directions_, anisotropic_permeability_coeffs);
    porofluidmat.compute_lin_mat_reaction_tensor(linreac_dphi, linreac_dJ,
        spatial_material_mapping.determinant_deformation_gradient_, porosity);
    temp.multiply(1.0, matreatensor,
        Core::LinAlg::make_matrix_view(spatial_material_mapping.inverse_deformation_gradient_));
    reatensor.multiply_tn(
        Core::LinAlg::make_matrix_view(spatial_material_mapping.inverse_deformation_gradient_),
        temp);
    reavel.multiply(reatensor, disp_velocity);
    reafvel.multiply(reatensor, fluid_velocity);
    for (int idim = 0; idim < numdim_; idim++)
    {
      for (int jdim = 0; jdim < numdim_; jdim++)
      {
        const double reatensor_i_j = reatensor(idim, jdim);
        for (int inode = 0; inode < numnod_; inode++)
        {
          const int fk = numdim_ * inode;
          const double v = detJ_w * shapefunctions(inode) * porosity * porosity *
                           spatial_material_mapping.determinant_deformation_gradient_ *
                           spatial_material_mapping.determinant_deformation_gradient_;
          for (int jnode = 0; jnode < numnod_; jnode++)
          {
            const int fi = numdim_ * jnode;

            /* additional "reactive darcy-term" detJ * w(gp) * ( J^2 * reacoeff * phi^2  ) * D(v_s)
             */
            erea_v(fk + idim, fi + jdim) += v * reatensor_i_j * shapefunctions(jnode);
          }
        }
      }
    }

    for (int idim = 0; idim < numdim_; idim++)
    {
      const double Finvgradp_j = FinvGradp(idim);

      for (int jdim = 0; jdim < numdim_; jdim++)
      {
        for (int jnode = 0; jnode < numnod_; jnode++)
        {
          const int fi = numdim_ * jnode;

          const double val =
              detJ_w *
              (-porosity * ddet_defgrd_ddisp(fi + jdim) * Finvgradp_j -
                  porosity * spatial_material_mapping.determinant_deformation_gradient_ *
                      dinverse_defgrd_ddisp_gradp(idim, fi + jdim) -
                  dPorosity_dDisp(fi + jdim) *
                      spatial_material_mapping.determinant_deformation_gradient_ * Finvgradp_j);

          for (int inode = 0; inode < numnod_; inode++)
          {
            /* additional "pressure gradient term"
              -  detJ * w(gp) * phi *  ( dJ/d(us) * F^-T * Grad(p) - J * d(F^-T)/d(us) *Grad(p) ) *
                          D(us)
                      - detJ * w(gp) * d(phi)/d(us) * J * F^-T * Grad(p) * D(us)
              */
            (stiffness_matrix)(numdim_ * inode + idim, fi + jdim) += shapefunctions(inode) * val;
          }
        }
      }
    }

    for (int idim = 0; idim < numdim_; idim++)
    {
      const double reac_vel_j = reavel(idim);
      const double reafvel_j = reafvel(idim);
      for (int jdim = 0; jdim < numdim_; jdim++)
      {
        for (int jnode = 0; jnode < numnod_; jnode++)
        {
          const int fi = numdim_ * jnode;
          const double val = detJ_w * spatial_material_mapping.determinant_deformation_gradient_ *
                             porosity * 2 * (reac_vel_j - reafvel_j) *
                             (porosity * ddet_defgrd_ddisp(fi + jdim) +
                                 spatial_material_mapping.determinant_deformation_gradient_ *
                                     dPorosity_dDisp(fi + jdim));

          for (int inode = 0; inode < numnod_; inode++)
          {
            /* additional "reactive darcy-term detJ * w(gp) * 2 * ( dJ/d(us) * vs * reacoeff * phi^2
             * + J * reacoeff * phi * d(phi)/d(us) * vs ) * D(us) - detJ * w(gp) *  2 * ( J *
             * dJ/d(us) * v^f * reacoeff * phi^2 + J * reacoeff * phi * d(phi)/d(us) * v^f ) * D(us)
             */
            (stiffness_matrix)(numdim_ * inode + idim, fi + jdim) += shapefunctions(inode) * val;
          }
        }
      }
    }

    // check if derivatives of reaction tensor are zero --> significant speed up
    if (porofluidmat.permeability_function() == Mat::PAR::constant)
    {
      const double fac = detJ_w * porosity * porosity *
                         spatial_material_mapping.determinant_deformation_gradient_ *
                         spatial_material_mapping.determinant_deformation_gradient_;
      for (int idim = 0; idim < numdim_; idim++)
      {
        for (int jdim = 0; jdim < numdim_; jdim++)
        {
          for (int jnode = 0; jnode < numnod_; jnode++)
          {
            const int fi = numdim_ * jnode;

            for (int inode = 0; inode < numnod_; inode++)
            {
              double val = 0.0;
              for (int p = 0; p < numdim_; ++p)
              {
                const double velint_p = disp_velocity(p);
                const double fvelint_p = fluid_velocity(p);
                for (int n = 0; n < numdim_; ++n)
                {
                  const double defgrd_inv_n_p =
                      spatial_material_mapping.inverse_deformation_gradient_(n, p);
                  const double dFinvTdus_n_p =
                      dInverseDeformationGradientTransposed_dDisp(p * numdim_ + n, fi + jdim);
                  for (int m = 0; m < numdim_; ++m)
                  {
                    val += fac * (velint_p - fvelint_p) *
                           (dInverseDeformationGradientTransposed_dDisp(
                                idim * numdim_ + m, fi + jdim) *
                                   matreatensor(m, n) * defgrd_inv_n_p +
                               spatial_material_mapping.inverse_deformation_gradient_(m, idim) *
                                   matreatensor(m, n) * dFinvTdus_n_p);
                  }
                }
              }
              (stiffness_matrix)(numdim_ * inode + idim, fi + jdim) += shapefunctions(inode) * val;
            }
          }
        }
      }
    }
    else
    {
      const double fac = detJ_w * porosity * porosity *
                         spatial_material_mapping.determinant_deformation_gradient_ *
                         spatial_material_mapping.determinant_deformation_gradient_;
      for (int idim = 0; idim < numdim_; idim++)
      {
        for (int jdim = 0; jdim < numdim_; jdim++)
        {
          for (int jnode = 0; jnode < numnod_; jnode++)
          {
            const int fi = numdim_ * jnode;
            const double dphi_dus_fi_l = dPorosity_dDisp(fi + jdim);
            const double dJ_dus_fi_l = ddet_defgrd_ddisp(fi + jdim);
            for (int inode = 0; inode < numnod_; inode++)
            {
              double val = 0.0;
              for (int m = 0; m < numdim_; ++m)
              {
                const double dFinvTdus_idim_m_fi_jdim =
                    dInverseDeformationGradientTransposed_dDisp(idim * numdim_ + m, fi + jdim);
                const double defgrd_inv_m_idim =
                    spatial_material_mapping.inverse_deformation_gradient_(m, idim);
                for (int n = 0; n < numdim_; ++n)
                {
                  const double matreatensor_m_n = matreatensor(m, n);
                  const double linreac_dphi_m_n = linreac_dphi(m, n);
                  const double linreac_dJ_m_n = linreac_dJ(m, n);
                  for (int p = 0; p < numdim_; ++p)
                  {
                    val +=
                        fac * (disp_velocity(p) - fluid_velocity(p)) *
                        (dFinvTdus_idim_m_fi_jdim * matreatensor_m_n *
                                spatial_material_mapping.inverse_deformation_gradient_(n, p) +
                            defgrd_inv_m_idim * matreatensor_m_n *
                                dInverseDeformationGradientTransposed_dDisp(
                                    p * numdim_ + n, fi + jdim) +
                            defgrd_inv_m_idim *
                                (linreac_dphi_m_n * dphi_dus_fi_l + linreac_dJ_m_n * dJ_dus_fi_l) *
                                spatial_material_mapping.inverse_deformation_gradient_(n, p));
                  }
                }
              }
              (stiffness_matrix)(numdim_ * inode + idim, fi + jdim) += val * shapefunctions(inode);
            }
          }
        }
      }
    }
  }

  /*!
   * @brief Update elastic stiffness matrix with poroelasticity contribution of one Gauss point
   *
   * @tparam celltype: Cell type
   * @param detJ_w (in) : integration factor (Gauss point weight times the determinant of
   * the jacobian)
   * @param solidpressure (in) : solidpressure
   * @param det_defgrd (in) : determinant of deformation gradient
   * @param bopCinv (in) : B^T . C^-1
   * @param bop (in) : Strain gradient (B-Operator)
   * @param ddet_defgrd_ddisp (in) : derivative of determinant of derformation gradient w.r.t. the
   * displacements
   * @param dsolidpressure_ddisp (in) : derivative of solidpressure w.r.t. the
   * displacements
   * @param dinverserightcauchygreen_ddisp (in) : derivatives of inverse right cauchy green tensor
   * w.r.t. the displacements
   * @param stiffness_matrix (in/out) : stiffness matrix where the local contribution is added to
   */
  template <Core::FE::CellType celltype>
  inline void update_elastic_stiffness_matrix(const double detJ_w, const double solidpressure,
      const double det_defgrd,
      const Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, 1>& bopCinv,
      const Core::LinAlg::Matrix<Internal::num_str<celltype>, Internal::num_dof_per_ele<celltype>>&
          bop,
      const Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& ddet_defgrd_ddisp,
      const Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& dsolidpressure_ddisp,
      const Core::LinAlg::Matrix<Internal::num_str<celltype>, Internal::num_dof_per_ele<celltype>>&
          dinverserightcauchygreen_ddisp,
      Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_nodes<celltype>,
          Internal::num_dim<celltype> * Internal::num_nodes<celltype>>& stiffness_matrix)
  {
    Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, Internal::num_dof_per_ele<celltype>>
        tmp;

    // additional fluid stress- stiffness term -(B^T . C^-1 . dJ/d(us) * p^f * detJ * w(gp))
    tmp.multiply((-detJ_w * solidpressure), bopCinv, ddet_defgrd_ddisp);
    stiffness_matrix.update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term -(B^T .  dC^-1/d(us) * J * p^f * detJ * w(gp))
    tmp.multiply_tn((-detJ_w * solidpressure * det_defgrd), bop, dinverserightcauchygreen_ddisp);
    stiffness_matrix.update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term -(B^T .  dC^-1 * J * dp^s/d(us) * detJ * w(gp))
    tmp.multiply((-detJ_w * det_defgrd), bopCinv, dsolidpressure_ddisp);
    stiffness_matrix.update(1.0, tmp, 1.0);
  }

  /*!
   * @brief Update geometric stiffness matrix with poroelasticity contribution of one Gauss point
   *
   * @param sfac (in) : scale factor
   * @param N_XYZ (in) : derivatives of the shape functions w.r.t. XYZ
   * @param stiffness_matrix (in/out) : stiffness matrix where the local contribution is added to
   */
  template <Core::FE::CellType celltype>
  void update_geometric_stiffness_matrix(
      const Core::LinAlg::Matrix<Internal::num_str<celltype>, 1>& sfac,
      const std::array<Core::LinAlg::Tensor<double, Internal::num_dim<celltype>>,
          Internal::num_nodes<celltype>>& N_XYZ,
      Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_nodes<celltype>,
          Internal::num_dim<celltype> * Internal::num_nodes<celltype>>& stiffness_matrix)
  {
    std::vector<double> SmB_L(3);  // intermediate Sm.B_L
    // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj
    for (int inod = 0; inod < Internal::num_nodes<celltype>; ++inod)
    {
      SmB_L[0] = sfac(0) * N_XYZ[inod](0) + sfac(3) * N_XYZ[inod](1) + sfac(5) * N_XYZ[inod](2);
      SmB_L[1] = sfac(3) * N_XYZ[inod](0) + sfac(1) * N_XYZ[inod](1) + sfac(4) * N_XYZ[inod](2);
      SmB_L[2] = sfac(5) * N_XYZ[inod](0) + sfac(4) * N_XYZ[inod](1) + sfac(2) * N_XYZ[inod](2);
      for (int jnod = 0; jnod < Internal::num_nodes<celltype>; ++jnod)
      {
        double bopstrbop = 0.0;  // intermediate value
        for (int idim = 0; idim < Internal::num_dim<celltype>; ++idim)
          bopstrbop += N_XYZ[jnod](idim) * SmB_L[idim];
        stiffness_matrix(Internal::num_dim<celltype> * inod + 0,
            Internal::num_dim<celltype> * jnod + 0) += bopstrbop;
        stiffness_matrix(Internal::num_dim<celltype> * inod + 1,
            Internal::num_dim<celltype> * jnod + 1) += bopstrbop;
        stiffness_matrix(Internal::num_dim<celltype> * inod + 2,
            Internal::num_dim<celltype> * jnod + 2) += bopstrbop;
      }
    }
  }

  /*!
   * @brief Add coupling contribution (poroelasticity OD entries) of one Gauss point to stiffness
   * matrix
   *
   * @tparam celltype: Cell type
   * @param detJ_w (in) : integration factor (Gauss point weight times the determinant of
   * the jacobian)
   * @param solidpressurederiv (in) : derivative of solidpressure w.r.t. fluid multiphase
   * @param bopCinv (in) : B^T * C^-1
   * @param shape_functions (in) : Shape function
   * * @param det_defgrd (in) : determinant of deformation gradient
   * @param nummultifluiddofpernode (in) : number of fluid multiphase dofs per node
   * @param stiffness_matrix (in/out) : stiffness matrix where the local contribution is added to
   */
  template <Core::FE::CellType celltype>
  inline void update_stiffness_matrix_coupling_multiphase_pressurebased(const double detJ_w,
      const std::vector<double>& solidpressurederiv,
      const Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, 1>& bopCinv,
      const ShapeFunctionsAndDerivatives<celltype>& shape_functions, const double det_defgrd,
      const int nummultifluiddofpernode, Core::LinAlg::SerialDenseMatrix& stiffness_matrix)
  {
    for (int i = 0; i < Internal::num_nodes<celltype>; i++)
    {
      const int fi = Internal::num_dim<celltype> * i;

      for (int j = 0; j < Internal::num_dim<celltype>; j++)
      {
        for (int k = 0; k < Internal::num_nodes<celltype>; k++)
        {
          for (int iphase = 0; iphase < nummultifluiddofpernode; iphase++)
          {
            int fk_press = k * nummultifluiddofpernode + iphase;

            /*-------structure- fluid pressure coupling: "stress term"
             -B^T . ( -1*J*C^-1 ) * Dp
             */
            stiffness_matrix(fi + j, fk_press) += detJ_w * bopCinv(fi + j) * (-1.0) * det_defgrd *
                                                  shape_functions.shapefunctions_(k) *
                                                  solidpressurederiv[iphase];
          }
        }
      }
    }
  }

  /*!
   * @brief Update force vector with brinkmann flow  contribution
   *
   * @tparam celltype: Cell type
   * @param integration_fac (in) : Integration factor (Gauss point weight times the determinant of
   * the jacobian)
   *  @param det_defgrd (in) : determinant of deformation gradient
   *  @param porosity (in) : porosity
   * @param fvelder (in) :  material fluid velocity gradient at integration point
   * @param defgrd_inv (in) : inverse deformationgradient
   * @param bop (in) : Strain gradient (B-Operator)
   * @param C_inv (in) : inverse right cachygreen tensor
   * @param fstress (in) : viscous stress
   * @param force_vector (in/out) : Force vector where the local contribution is added to
   */
  template <Core::FE::CellType celltype>
  inline void update_internal_force_vector_for_brinkman_flow(const double integration_fac,
      const double viscosity, const double det_defgrd, const double porosity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>& fvelder,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          defgrd_inv,
      const Core::LinAlg::Matrix<Internal::num_str<celltype>, Internal::num_dof_per_ele<celltype>>&
          bop,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>& C_inv,
      Core::LinAlg::Matrix<Internal::num_str<celltype>, 1>& fstress,
      Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_nodes<celltype>, 1>&
          force_vector)
  {
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> CinvFvel;
    Discret::Elements::Internal::calculate_viscous_stress<celltype>(integration_fac, viscosity,
        det_defgrd, porosity, fvelder, defgrd_inv, C_inv, fstress, CinvFvel);
    // B^T . C^-1
    static Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, 1> fstressb(
        Core::LinAlg::Initialization::zero);
    fstressb.multiply_tn(bop, fstress);
    force_vector.update(1.0, fstressb, 1.0);
  }


  /*!
   * @brief Update stiffness matrix with brinkmann flow  contribution
   *
   * @tparam celltype: Cell type
   * @param integration_fac (in) : Integration factor (Gauss point weight times the determinant of
   * the jacobian)
   *  @param det_defgrd (in) : determinant of deformation gradient
   *  @param porosity (in) : porosity
   * @param fvelder (in) :  material fluid velocity gradient at integration point
   * @param defgrd_inv (in) : inverse deformationgradient
   * @param bop (in) : Strain gradient (B-Operator)
   * @param C_inv (in) : inverse right cachygreen tensor
   * @param dporosity_dus (in) : derivative of porosity w.r.t. displacements
   * @param dJ_dus (in) : derivative of determinante of deformationgradient w.r.t. displacements
   * @param dCinv_dus (in) :  derivative of right cauchy greeen tensor w.r.t. displacements
   * @param dFinvTdus (in) : derivative of inverse transposed deformation gradient w.r.t.
   * displacements
   * @param fstress (in) : viscous stress
   * @param stiffness_matrix (in/out) : stiffness matrix where the local contribution is added to
   */
  template <Core::FE::CellType celltype>
  inline void update_stiffness_matrix_for_brinkman_flow(const double integration_fac,
      const double viscosity, const double det_defgrd, const double porosity,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>& fvelder,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          defgrd_inv,
      const Core::LinAlg::Matrix<Internal::num_str<celltype>, Internal::num_dof_per_ele<celltype>>&
          bop,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>& C_inv,
      const Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& dporosity_dus,
      const Core::LinAlg::Matrix<1, Internal::num_dof_per_ele<celltype>>& dJ_dus,
      const Core::LinAlg::Matrix<Internal::num_str<celltype>, Internal::num_dof_per_ele<celltype>>&
          dCinv_dus,
      const Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_dim<celltype>,
          Internal::num_dof_per_ele<celltype>>& dFinvTdus,
      Core::LinAlg::Matrix<Internal::num_str<celltype>, 1>& fstress,
      Core::LinAlg::Matrix<Internal::num_dim<celltype> * Internal::num_nodes<celltype>,
          Internal::num_dim<celltype> * Internal::num_nodes<celltype>>& stiffness_matrix)
  {
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> CinvFvel;
    Discret::Elements::Internal::calculate_viscous_stress<celltype>(integration_fac, viscosity,
        det_defgrd, porosity, fvelder, defgrd_inv, C_inv, fstress, CinvFvel);
    // B^T . C^-1
    static Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, 1> fstressb(
        Core::LinAlg::Initialization::zero);
    fstressb.multiply_tn(bop, fstress);

    // evaluate viscous terms (for darcy-brinkman flow only)
    {
      static Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>> tmp;
      tmp.multiply_nt(fvelder, defgrd_inv);
      double fac = integration_fac * viscosity;
      Core::LinAlg::Matrix<Internal::num_str<celltype>, Internal::num_dof_per_ele<celltype>>
          fstress_dus(Core::LinAlg::Initialization::zero);

      {
        for (int n = 0; n < Internal::num_nodes<celltype>; ++n)
        {
          for (int k = 0; k < Internal::num_dim<celltype>; ++k)
          {
            const int gid = n * Internal::num_dim<celltype> + k;
            fstress_dus(0, gid) +=
                2 * (dCinv_dus(0, gid) * tmp(0, 0) + dCinv_dus(3, gid) * tmp(1, 0) +
                        dCinv_dus(5, gid) * tmp(2, 0));
            fstress_dus(1, gid) +=
                2 * (dCinv_dus(3, gid) * tmp(0, 1) + dCinv_dus(1, gid) * tmp(1, 1) +
                        dCinv_dus(4, gid) * tmp(2, 1));
            fstress_dus(2, gid) +=
                2 * (dCinv_dus(5, gid) * tmp(0, 2) + dCinv_dus(4, gid) * tmp(1, 2) +
                        dCinv_dus(2, gid) * tmp(2, 2));
            /* ~~~ */
            fstress_dus(3, gid) += +dCinv_dus(0, gid) * tmp(0, 1) + dCinv_dus(3, gid) * tmp(1, 1) +
                                   dCinv_dus(5, gid) * tmp(2, 1) + dCinv_dus(3, gid) * tmp(0, 0) +
                                   dCinv_dus(1, gid) * tmp(1, 0) + dCinv_dus(4, gid) * tmp(2, 0);
            fstress_dus(4, gid) += +dCinv_dus(3, gid) * tmp(0, 2) + dCinv_dus(1, gid) * tmp(1, 2) +
                                   dCinv_dus(4, gid) * tmp(2, 2) + dCinv_dus(5, gid) * tmp(0, 1) +
                                   dCinv_dus(4, gid) * tmp(1, 1) + dCinv_dus(2, gid) * tmp(2, 1);
            fstress_dus(5, gid) += +dCinv_dus(5, gid) * tmp(0, 0) + dCinv_dus(4, gid) * tmp(1, 0) +
                                   dCinv_dus(2, gid) * tmp(2, 0) + dCinv_dus(0, gid) * tmp(0, 2) +
                                   dCinv_dus(3, gid) * tmp(1, 2) + dCinv_dus(5, gid) * tmp(2, 2);
            fstress_dus(0, gid) +=
                2 * CinvFvel(0, 0) * dFinvTdus(0 * Internal::num_dim<celltype>, gid) +
                2 * CinvFvel(0, 1) * dFinvTdus(1 * Internal::num_dim<celltype>, gid) +
                2 * CinvFvel(0, 2) * dFinvTdus(2 * Internal::num_dim<celltype>, gid);
            fstress_dus(1, gid) +=
                2 * CinvFvel(1, 0) * dFinvTdus(0 * Internal::num_dim<celltype> + 1, gid) +
                2 * CinvFvel(1, 1) * dFinvTdus(1 * Internal::num_dim<celltype> + 1, gid) +
                2 * CinvFvel(1, 2) * dFinvTdus(2 * Internal::num_dim<celltype> + 1, gid);
            fstress_dus(2, gid) +=
                2 * CinvFvel(2, 0) * dFinvTdus(0 * Internal::num_dim<celltype> + 2, gid) +
                2 * CinvFvel(2, 1) * dFinvTdus(1 * Internal::num_dim<celltype> + 2, gid) +
                2 * CinvFvel(2, 2) * dFinvTdus(2 * Internal::num_dim<celltype> + 2, gid);
            /* ~~~ */
            fstress_dus(3, gid) +=
                CinvFvel(0, 0) * dFinvTdus(0 * Internal::num_dim<celltype> + 1, gid) +
                CinvFvel(1, 0) * dFinvTdus(0 * Internal::num_dim<celltype>, gid) +
                CinvFvel(0, 1) * dFinvTdus(1 * Internal::num_dim<celltype> + 1, gid) +
                CinvFvel(1, 1) * dFinvTdus(1 * Internal::num_dim<celltype>, gid) +
                CinvFvel(0, 2) * dFinvTdus(2 * Internal::num_dim<celltype> + 1, gid) +
                CinvFvel(1, 2) * dFinvTdus(2 * Internal::num_dim<celltype>, gid);
            fstress_dus(4, gid) +=
                CinvFvel(1, 0) * dFinvTdus(0 * Internal::num_dim<celltype> + 2, gid) +
                CinvFvel(2, 0) * dFinvTdus(0 * Internal::num_dim<celltype> + 1, gid) +
                CinvFvel(1, 1) * dFinvTdus(1 * Internal::num_dim<celltype> + 2, gid) +
                CinvFvel(2, 1) * dFinvTdus(1 * Internal::num_dim<celltype> + 1, gid) +
                CinvFvel(1, 2) * dFinvTdus(2 * Internal::num_dim<celltype> + 2, gid) +
                CinvFvel(2, 2) * dFinvTdus(2 * Internal::num_dim<celltype> + 1, gid);
            fstress_dus(5, gid) +=
                CinvFvel(2, 0) * dFinvTdus(0 * Internal::num_dim<celltype>, gid) +
                CinvFvel(0, 0) * dFinvTdus(0 * Internal::num_dim<celltype> + 2, gid) +
                CinvFvel(2, 1) * dFinvTdus(1 * Internal::num_dim<celltype>, gid) +
                CinvFvel(0, 1) * dFinvTdus(1 * Internal::num_dim<celltype> + 2, gid) +
                CinvFvel(2, 2) * dFinvTdus(2 * Internal::num_dim<celltype>, gid) +
                CinvFvel(0, 2) * dFinvTdus(2 * Internal::num_dim<celltype> + 2, gid);
          }
        }
      }
      static Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>,
          Internal::num_dof_per_ele<celltype>>
          fluidstress_part;

      /* additional viscous fluid stress- stiffness term (B^T . fstress . dJ/d(us) * porosity * detJ
       * * w(gp)) */
      fluidstress_part.multiply(fac * porosity, fstressb, dJ_dus);
      stiffness_matrix.update(1.0, fluidstress_part, 1.0);
      // additional fluid stress- stiffness term (B^T .  d\phi/d(us) . fstress  * J * w(gp))
      fluidstress_part.multiply(fac * det_defgrd, fstressb, dporosity_dus);
      stiffness_matrix.update(1.0, fluidstress_part, 1.0);
      // additional fluid stress- stiffness term (B^T .  phi . dfstress/d(us)  * J * w(gp))
      fluidstress_part.multiply_tn(
          integration_fac * viscosity * det_defgrd * porosity, bop, fstress_dus);
      stiffness_matrix.update(1.0, fluidstress_part, 1.0);
    }
  }

  /*!
   * @brief Calculates fluid mulltiphase primary variables at GP
   *
   * @tparam celltype: Cell type
   * @param fluidmultiphase_ephi (in) : primary variables of multiphase porous medium flow
   * @param nummultifluiddofpernode (in) : number of fluid multiphase dofs per node
   * @param: shape_functions (in): Shape functions
   * @returns: fluidmultiphase_phi_at_gp: fluid multiphase primary variables at GP
   */
  template <Core::FE::CellType celltype>
  inline std::vector<double> compute_fluid_multiphase_primary_variables_at_gp(
      const std::vector<double>& fluidmultiphase_ephi, const int nummultifluiddofpernode,
      const ShapeFunctionsAndDerivatives<celltype>& shape_functions)
  {
    std::vector<double> fluidmultiphase_phi_at_gp(nummultifluiddofpernode);
    // compute phi at GP = phi * shapefunction
    for (int i = 0; i < Internal::num_nodes<celltype>; i++)
    {
      for (int j = 0; j < nummultifluiddofpernode; j++)
      {
        fluidmultiphase_phi_at_gp[j] += shape_functions.shapefunctions_(i) *
                                        fluidmultiphase_ephi[i * nummultifluiddofpernode + j];
      }
    }

    return fluidmultiphase_phi_at_gp;
  }

  /*!
   * @brief Calculates solid pressure derivatives w.r.t. primary variables of fluid phases in
   * multiphase porespace
   *
   * @tparam celltype: Cell type
   * @param porofluidmat (in) : material of multiphase fluid
   * @param fluidmultiphase_phi_at_gp (in) : fluid multiphase primary variables at GP
   * @param numfluidphases (in): number of fluidphases in multiphase porespace
   * @returns solidpressurederiv: derivative of solidpressure w.r.t. fluid multiphase
   * primary variables
   */
  template <Core::FE::CellType celltype>
  inline std::vector<double> compute_solid_pressure_deriv(Mat::FluidPoroMultiPhase& porofluidmat,
      const std::vector<double>& fluidmultiphase_phi_at_gp, const int numfluidphases)
  {
    // zero out everything
    std::vector<double> solidpressurederiv(fluidmultiphase_phi_at_gp.size());

    // initialize auxiliary variables
    std::vector<double> genpress(numfluidphases);
    std::vector<double> press(numfluidphases);
    std::vector<double> sat(numfluidphases);
    Core::LinAlg::SerialDenseMatrix helpderiv(numfluidphases, numfluidphases, true);
    Core::LinAlg::SerialDenseMatrix satderiv(numfluidphases, numfluidphases, true);
    Core::LinAlg::SerialDenseMatrix pressderiv(numfluidphases, numfluidphases, true);
    std::vector<double> fluidphi(
        fluidmultiphase_phi_at_gp.data(), fluidmultiphase_phi_at_gp.data() + numfluidphases);

    // evaluate the pressures
    porofluidmat.evaluate_gen_pressure(genpress, fluidphi);

    // transform generalized pressures to true pressure values
    porofluidmat.transform_gen_pres_to_true_pres(genpress, press);

    // explicit evaluation of saturation
    porofluidmat.evaluate_saturation(sat, fluidphi, press);

    // calculate the derivative of the pressure (actually first its inverse)
    porofluidmat.evaluate_deriv_of_dof_wrt_pressure(pressderiv, fluidphi);

    // now invert the derivatives of the dofs w.r.t. pressure to get the derivatives
    // of the pressure w.r.t. the dofs
    {
      Teuchos::SerialDenseSolver<int, double> inverse;

      inverse.setMatrix(Teuchos::rcpFromRef(pressderiv.base()));
      int err = inverse.invert();
      if (err != 0)
        FOUR_C_THROW("Inversion of matrix for pressure derivative failed with error code {}.", err);
    }

    // calculate derivatives of saturation w.r.t. pressure
    porofluidmat.evaluate_deriv_of_saturation_wrt_pressure(helpderiv, press);

    // chain rule: the derivative of saturation w.r.t. dof =
    // (derivative of saturation w.r.t. pressure) * (derivative of pressure w.r.t. dof)
    Core::LinAlg::multiply(satderiv, helpderiv, pressderiv);
    // compute derivative of solid pressure w.r.t. dofs with product rule
    // standard derivative: no volume fractions present
    for (int iphase = 0; iphase < numfluidphases; iphase++)
    {
      for (int jphase = 0; jphase < numfluidphases; jphase++)
        solidpressurederiv[iphase] +=
            pressderiv(jphase, iphase) * sat[jphase] + satderiv(jphase, iphase) * press[jphase];
    }

    return solidpressurederiv;
  }

  /*!
   * @brief Calculates solidpressure
   *
   * @tparam celltype: Cell type
   * @param nummultifluiddofpernode (in) : number of fluid multiphase dofs per node
   * @param numfluidphases (in) : number of fluidphases in multiphase porespace
   * @param: fluidmultiphase_phi_at_gp (in): fluid multiphase primary variables at GP
   * @param: porofluidmat (in): material of multiphase fluid
   * @return solidpressure
   */
  template <Core::FE::CellType celltype>
  inline double compute_sol_pressure_at_gp(const int numfluidphases,
      const std::vector<double>& fluidmultiphase_phi_at_gp, Mat::FluidPoroMultiPhase& porofluidmat)
  {
    // initialize auxiliary variables
    std::vector<double> genpress(numfluidphases, 0.0);
    std::vector<double> sat(numfluidphases, 0.0);
    std::vector<double> press(numfluidphases, 0.0);
    std::vector<double> fluidphi(
        fluidmultiphase_phi_at_gp.data(), fluidmultiphase_phi_at_gp.data() + numfluidphases);

    // evaluate the pressures
    porofluidmat.evaluate_gen_pressure(genpress, fluidphi);

    //! transform generalized pressures to true pressure values
    porofluidmat.transform_gen_pres_to_true_pres(genpress, press);

    // explicit evaluation of saturation
    porofluidmat.evaluate_saturation(sat, fluidphi, press);

    // solid pressure = sum (S_i*p_i)
    const double solidpressure = std::inner_product(sat.begin(), sat.end(), press.begin(), 0.0);

    return solidpressure;
  }

  /*!
   * @brief Recalculates solid pressure derivative in case of volfracs
   *
   * @param fluidmultiphase_phi_at_gp (in) : fluid multiphase primary variables at GP
   * @param nummultifluiddofpernode (in) : number of fluid multiphase dofs per node
   * @param: numfluidphases (in): number of fluidphases in multiphase porespace
   * @param: numvolfrac (in): number of volfracs from additionalporous networks
   * @param: solidpressure (in): solidpressue
   * @param: porosity (in): porosity = volumefraction in multiphase porespace + volfracs from
   * additional porous networks
   * @param: solidpressurederiv (in/out): derivative of solidpressure w.r.t. fluid multiphase
   * primary variables and volfracs
   */
  inline void recalculate_sol_pressure_deriv(const std::vector<double>& fluidmultiphase_phi_at_gp,
      const int nummultifluiddofpernode, const int numfluidphases, const int numvolfrac,
      const double solidpressure, const double porosity, std::vector<double>& solidpressurederiv)
  {
    // get volume fraction primary variables
    std::vector<double> volfracphi(fluidmultiphase_phi_at_gp.data() + numfluidphases,
        fluidmultiphase_phi_at_gp.data() + numfluidphases + numvolfrac);
    double sumaddvolfrac = 0.0;
    for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++) sumaddvolfrac += volfracphi[ivolfrac];

    // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
    //      + 1.0 / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
    const double scale = (porosity - sumaddvolfrac) / porosity;

    // scale original fluid press deriv with (porosity - sumaddvolfrac)/porosity
    for (int iphase = 0; iphase < numfluidphases; iphase++) solidpressurederiv[iphase] *= scale;

    // get volfrac pressures at [numfluidphases+numvolfrac...nummultifluiddofpernode-1]
    std::vector<double> volfracpressure(
        fluidmultiphase_phi_at_gp.data() + numfluidphases + numvolfrac,
        fluidmultiphase_phi_at_gp.data() + nummultifluiddofpernode);

    for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
    {
      // d p_s / d volfrac = - fluidpress/porosity + volfracpressure/porosity
      solidpressurederiv[ivolfrac + numfluidphases] =
          -1.0 / porosity * solidpressure + 1.0 / porosity * volfracpressure[ivolfrac];
      // d p_s / d volfracpress = + volfracphi/porosity
      solidpressurederiv[ivolfrac + numfluidphases + numvolfrac] = volfracphi[ivolfrac] / porosity;
    }
  }

  /*!
   * @brief Recalculates solid pressure derivative in case of volfracs
   *
   * @param fluidmultiphase_phi_at_gp (in) : fluid multiphase primary variables at GP
   * @param nummultifluiddofpernode (in) : number of fluid multiphase dofs per node
   * @param: numfluidphases (in): number of fluidphases in multiphase porespace
   * @param: numvolfrac (in): number of volfracs from additionalporous networks
   * @param: solidpressure (in): solidpressue
   * @param: porosity (in): porosity = volumefraction in multiphase porespace + volfracs from
   * additional porous networks
   * @param: solidpressurederiv (in/out): derivative of solidpressure w.r.t. fluid multiphase
   * primary variables and volfracs
   */
  inline void recalculate_sol_pressure_deriv(const std::vector<double>& fluidmultiphase_phi_at_gp,
      const int nummultifluiddofpernode, const int numfluidphases, const int numvolfrac,
      const double solidpressure, const double porosity, std::vector<double>& solidpressurederiv,
      const Mat::FluidPoroMultiPhase& porofluidmat, const double determinant_deformation_gradient)
  {
    // get volfrac pressures (only one possible in closing relation blood lung)
    double volfracpressure = fluidmultiphase_phi_at_gp[numfluidphases];

    // get volume fraction (only one possible in closing relation blood lung)
    const double volfrac =
        calculate_volfrac_from_closing_relation_blood_lung(fluidmultiphase_phi_at_gp,
            volfracpressure, determinant_deformation_gradient, porofluidmat, numfluidphases);

    // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
    //      + 1.0 / porosity volfrac * volfracpressure
    const double scale = (porosity - volfrac) / porosity;

    // scale original fluid press deriv with (porosity - sumaddvolfrac)/porosity
    for (int iphase = 0; iphase < numfluidphases; iphase++) solidpressurederiv[iphase] *= scale;


    if (fluidmultiphase_phi_at_gp[0] < volfracpressure)
    {
      // d p_s / d volfracpress = + volfracphi/porosity
      solidpressurederiv[numfluidphases] = volfrac / porosity;
    }
    else
    {
      // initialize auxiliary variables
      std::vector<double> genpress(numfluidphases, 0.0);
      std::vector<double> saturation(numfluidphases, 0.0);
      std::vector<double> press(numfluidphases, 0.0);
      std::vector<double> fluidphi(
          &fluidmultiphase_phi_at_gp[0], &fluidmultiphase_phi_at_gp[numfluidphases]);

      // evaluate the pressures
      porofluidmat.evaluate_gen_pressure(genpress, fluidphi);
      // transform generalized pressures to true pressure values
      porofluidmat.transform_gen_pres_to_true_pres(genpress, press);
      // explicit evaluation of saturation
      porofluidmat.evaluate_saturation(saturation, fluidphi, press);


      // recalculate derivative for phase air ( air is always the first phse in multiphase porespace
      double dvolfrac_dpA =
          PoroPressureBased::ElementUtils::
              get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                  porofluidmat, numfluidphases)
                  .initial_volfrac() *
          pow(determinant_deformation_gradient,
              PoroPressureBased::ElementUtils::
                  get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                      porofluidmat, numfluidphases)
                      .scaling_parameter_deformation()) *
          (PoroPressureBased::ElementUtils::
                  get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                      porofluidmat, numfluidphases)
                      .scaling_parameter_pressure()) *
          pow(fluidmultiphase_phi_at_gp[0] / volfracpressure,
              PoroPressureBased::ElementUtils::
                      get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                          porofluidmat, numfluidphases)
                          .scaling_parameter_pressure() -
                  1.0) *
          1.0 / volfracpressure;

      double dinverseporosity_dpA = (-1.0) * 1.0 / (porosity * porosity) * dvolfrac_dpA;
      solidpressurederiv[0] +=
          (saturation[0] * (porosity - volfrac) * fluidmultiphase_phi_at_gp[0] +
              volfracpressure * volfrac) *
              dinverseporosity_dpA +
          volfracpressure * 1.0 / porosity * dvolfrac_dpA;
      // solid pressure with fluid phases multiphase porespace = sum (S_i*p_i)
      const double solidpressure_in_multiphase_porespace =
          std::inner_product(saturation.begin(), saturation.end(), press.begin(), 0.0);
      double dvolfrac_dvolfracpressure =
          PoroPressureBased::ElementUtils::
              get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                  porofluidmat, numfluidphases)
                  .initial_volfrac() *
          pow(determinant_deformation_gradient,
              PoroPressureBased::ElementUtils::
                  get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                      porofluidmat, numfluidphases)
                      .scaling_parameter_deformation()) *
          (-1.0) *
          (PoroPressureBased::ElementUtils::
                  get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                      porofluidmat, numfluidphases)
                      .scaling_parameter_pressure()) *
          pow(fluidmultiphase_phi_at_gp[0] / volfracpressure,
              PoroPressureBased::ElementUtils::
                      get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
                          porofluidmat, numfluidphases)
                          .scaling_parameter_pressure() -
                  1.0) *
          fluidmultiphase_phi_at_gp[0] * 1.0 / (volfracpressure * volfracpressure);

      // d p_s / d volfracpress = + volfracphi/porosity
      solidpressurederiv[numfluidphases] =
          (-1.0) * (porosity - volfrac) * 1.0 / (porosity * porosity) * dvolfrac_dvolfracpressure *
              solidpressure_in_multiphase_porespace +
          volfrac / porosity + 1.0 / porosity * dvolfrac_dvolfracpressure -
          volfrac * 1.0 / (porosity * porosity) * dvolfrac_dvolfracpressure;
    }
  }



  template <Core::FE::CellType celltype>
  struct CauchyGreenAndInverse
  {
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>
        right_cauchy_green_;
    Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>
        inverse_right_cauchy_green_;
  };

  /*!
   * @brief Evaluates right Cauchy-Green deformation tensor and its inverse
   *
   * @tparam celltype: Cell type
   * @param spatial_material_mapping (in) : An object holding quantities of the spatial material
   * mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient)
   * @return CauchyGreenAndInverse<celltype> : An object holding the right Cauchy-Green deformation
   * tensor and its inverse
   */
  template <Core::FE::CellType celltype>
  CauchyGreenAndInverse<celltype> evaluate_cauchy_green_and_inverse(
      const SpatialMaterialMapping<celltype>& spatial_material_mapping)
    requires(Internal::num_dim<celltype> == 3)
  {
    CauchyGreenAndInverse<celltype> cauchygreen;

    cauchygreen.right_cauchy_green_ = Core::LinAlg::make_matrix(
        Core::LinAlg::get_full(Discret::Elements::evaluate_cauchy_green(spatial_material_mapping)));
    cauchygreen.inverse_right_cauchy_green_.invert(cauchygreen.right_cauchy_green_);

    return cauchygreen;
  }

  /*!
   * @brief Evaluates the derivative of the inverse right Cauchy-Green deformation tensor w.r.t. the
   * displacements
   *
   * @tparam celltype: Cell type
   * @param cauchygreen (in) : An object holding the right Cauchy-Green deformation tensor and its
   * inverse
   * @param jacobian_mapping (in) : n object holding quantities of the jacobian mapping
   * (inverse Jacobian, determinant, derivatives of the shape functions w.r.t. XYZ)
   * @param: spatial_material_mapping (in): An object holding quantities of the spatial material
   * mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient)
   * @return dInverseCauchyGreen_dDisp : derivative of the inverse right Cauchy-Green deformation
   * tensor w.r.t. the displacements
   */
  template <Core::FE::CellType celltype>
  Core::LinAlg::Matrix<Internal::num_str<celltype>,
      Internal::num_dim<celltype> * Internal::num_nodes<celltype>>
  evaluate_inverse_cauchy_green_linearization(const CauchyGreenAndInverse<celltype>& cauchygreen,
      const JacobianMapping<celltype>& jacobian_mapping,
      const SpatialMaterialMapping<celltype>& spatial_material_mapping)
    requires(Internal::num_dim<celltype> == 3)
  {
    // dC^-1/dDisp
    Core::LinAlg::Matrix<Internal::num_str<celltype>,
        Internal::num_dim<celltype> * Internal::num_nodes<celltype>>
        dInverseCauchyGreen_dDisp(Core::LinAlg::Initialization::zero);

    for (int n = 0; n < Internal::num_nodes<celltype>; ++n)
    {
      for (int k = 0; k < Internal::num_dim<celltype>; ++k)
      {
        const int gid = n * Internal::num_dim<celltype> + k;
        for (int i = 0; i < Internal::num_dim<celltype>; ++i)
        {
          dInverseCauchyGreen_dDisp(0, gid) +=
              -2 * cauchygreen.inverse_right_cauchy_green_(0, i) * jacobian_mapping.N_XYZ[n](i) *
              spatial_material_mapping.inverse_deformation_gradient_(0, k);
          dInverseCauchyGreen_dDisp(1, gid) +=
              -2 * cauchygreen.inverse_right_cauchy_green_(1, i) * jacobian_mapping.N_XYZ[n](i) *
              spatial_material_mapping.inverse_deformation_gradient_(1, k);
          dInverseCauchyGreen_dDisp(2, gid) +=
              -2 * cauchygreen.inverse_right_cauchy_green_(2, i) * jacobian_mapping.N_XYZ[n](i) *
              spatial_material_mapping.inverse_deformation_gradient_(2, k);
          /* ~~~ */
          dInverseCauchyGreen_dDisp(3, gid) +=
              -cauchygreen.inverse_right_cauchy_green_(0, i) * jacobian_mapping.N_XYZ[n](i) *
                  spatial_material_mapping.inverse_deformation_gradient_(1, k) -
              spatial_material_mapping.inverse_deformation_gradient_(0, k) *
                  jacobian_mapping.N_XYZ[n](i) * cauchygreen.inverse_right_cauchy_green_(1, i);
          dInverseCauchyGreen_dDisp(4, gid) +=
              -cauchygreen.inverse_right_cauchy_green_(1, i) * jacobian_mapping.N_XYZ[n](i) *
                  spatial_material_mapping.inverse_deformation_gradient_(2, k) -
              spatial_material_mapping.inverse_deformation_gradient_(1, k) *
                  jacobian_mapping.N_XYZ[n](i) * cauchygreen.inverse_right_cauchy_green_(2, i);
          dInverseCauchyGreen_dDisp(5, gid) +=
              -cauchygreen.inverse_right_cauchy_green_(2, i) * jacobian_mapping.N_XYZ[n](i) *
                  spatial_material_mapping.inverse_deformation_gradient_(0, k) -
              spatial_material_mapping.inverse_deformation_gradient_(2, k) *
                  jacobian_mapping.N_XYZ[n](i) * cauchygreen.inverse_right_cauchy_green_(0, i);
        }
      }
    }
    return dInverseCauchyGreen_dDisp;
  }


}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif