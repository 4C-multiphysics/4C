// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_3D_ele_factory.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_solid_3D_ele_calc_displacement_based.hpp"
#include "4C_solid_3D_ele_calc_eas.hpp"
#include "4C_solid_3D_ele_calc_eas_helpers.hpp"
#include "4C_solid_3D_ele_calc_fbar.hpp"
#include "4C_solid_3D_ele_calc_mulf.hpp"
#include "4C_solid_3D_ele_calc_mulf_fbar.hpp"
#include "4C_solid_3D_ele_calc_shell_ans.hpp"
#include "4C_solid_3D_ele_properties.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <magic_enum/magic_enum_switch.hpp>

#include <type_traits>

FOUR_C_NAMESPACE_OPEN

namespace
{
  /*!
   * @brief A template class that is taking different element formulation switches as template
   * parameter. If implemented, the struct defines the type of the solid evaluation.
   *
   * @tparam celltype
   * @tparam kinem : Kinematic type (linear, nonliner)
   * @tparam ele_tech : Element technology (none, FBar, EAS mild and full)
   * @tparam prestress_technology : Prestress technology (none or mulf)
   * @tparam Enable : A dummy parameter for enabling a subset of switches.
   */
  template <Core::FE::CellType celltype, Inpar::Solid::KinemType kinem,
      Discret::Elements::ElementTechnology ele_tech,
      Discret::Elements::PrestressTechnology prestress_technology>
  struct SolidCalculationFormulation
  {
  };

  /*!
   * @brief Standard nonlinear displacement based total lagrangian formulation valid for all
   * celltypes
   */
  template <Core::FE::CellType celltype>
  struct SolidCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::none, Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::DisplacementBasedSolidIntegrator<celltype>;
  };

  /*!
   * @brief Displacement based formulation valid for all celltypes with linear kinematics (i.e.
   * small displacements)
   */
  template <Core::FE::CellType celltype>
  struct SolidCalculationFormulation<celltype, Inpar::Solid::KinemType::linear,
      Discret::Elements::ElementTechnology::none, Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::DisplacementBasedLinearKinematicsSolidIntegrator<celltype>;
  };

  /*!
   * @brief Nonlinear total lagrangian formulation with mild EAS for hex8
   */
  template <>
  struct SolidCalculationFormulation<Core::FE::CellType::hex8,
      Inpar::Solid::KinemType::nonlinearTotLag, Discret::Elements::ElementTechnology::eas_mild,
      Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::EASSolidIntegrator<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_h8_9, Inpar::Solid::KinemType::nonlinearTotLag>;
  };

  /*!
   * @brief Small displacements formulation with mild EAS for hex8
   */
  template <>
  struct SolidCalculationFormulation<Core::FE::CellType::hex8, Inpar::Solid::KinemType::linear,
      Discret::Elements::ElementTechnology::eas_mild, Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::EASSolidIntegrator<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_h8_9, Inpar::Solid::KinemType::linear>;
  };

  /*!
   * @brief Nonlinear total lagrangian formulation with full EAS for hex8
   */
  template <>
  struct SolidCalculationFormulation<Core::FE::CellType::hex8,
      Inpar::Solid::KinemType::nonlinearTotLag, Discret::Elements::ElementTechnology::eas_full,
      Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::EASSolidIntegrator<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_h8_21, Inpar::Solid::KinemType::nonlinearTotLag>;
  };

  /*!
   * @brief Small displacements formulation with full EAS for hex8
   */
  template <>
  struct SolidCalculationFormulation<Core::FE::CellType::hex8, Inpar::Solid::KinemType::linear,
      Discret::Elements::ElementTechnology::eas_full, Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::EASSolidIntegrator<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_h8_21, Inpar::Solid::KinemType::linear>;
  };

  /*!
   * @brief Nonlinear total lagrangian formulation with shell EAS for hex8
   */
  template <>
  struct SolidCalculationFormulation<Core::FE::CellType::hex8,
      Inpar::Solid::KinemType::nonlinearTotLag, Discret::Elements::ElementTechnology::shell_eas,
      Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::EASSolidIntegrator<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_sh8_7, Inpar::Solid::KinemType::nonlinearTotLag>;
  };

  /*!
   * @brief Nonlinear total lagrangian formulation with F-Bar for hex8 and pyramid 5
   */
  template <Core::FE::CellType celltype>
    requires(celltype == Core::FE::CellType::hex8 || celltype == Core::FE::CellType::pyramid5)
  struct SolidCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::fbar, Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::FBarSolidIntegrator<celltype>;
  };

  /*!
   * @brief Nonlinear total lagrangian solid-shell formulation with ANS
   */
  template <Core::FE::CellType celltype>
    requires(celltype == Core::FE::CellType::hex8 || celltype == Core::FE::CellType::wedge6)
  struct SolidCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::shell_ans, Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::ANSSolidShellIntegrator<celltype>;
  };

  /*!
   * @brief Nonlinear total lagrangian hex8 solid-shell formulation with EAS and ANS
   */
  template <Core::FE::CellType celltype>
    requires(celltype == Core::FE::CellType::hex8)
  struct SolidCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::shell_eas_ans,
      Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::EasAnsSolidShellIntegrator<celltype,
        Discret::Elements::EasType::eastype_sh8_7>;
  };

  /*!
   * @brief Nonlinear total lagrangian wedge6 solid-shell formulation with EAS and ANS
   */
  template <Core::FE::CellType celltype>
    requires(celltype == Core::FE::CellType::wedge6)
  struct SolidCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::shell_eas_ans,
      Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::EasAnsSolidShellIntegrator<celltype,
        Discret::Elements::EasType::eastype_sw6_1>;
  };

  /*!
   * @brief Nonlinear formulation with F-Bar and MULF prestressing for hex8 and pyramid5
   */
  template <Core::FE::CellType celltype>
    requires(celltype == Core::FE::CellType::hex8 || celltype == Core::FE::CellType::pyramid5)
  struct SolidCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::fbar, Discret::Elements::PrestressTechnology::mulf>
  {
    using type = Discret::Elements::MulfFBarSolidIntegrator<celltype>;
  };

  /*!
   * @brief Nonlinear modified updated lagrangian prestressing for all celltypes
   */
  template <Core::FE::CellType celltype>
  struct SolidCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::none, Discret::Elements::PrestressTechnology::mulf>
  {
    using type =
        Discret::Elements::SolidEleCalc<celltype, Discret::Elements::MulfFormulation<celltype>>;
  };
}  // namespace

Discret::Elements::SolidCalcVariant Discret::Elements::create_solid_calculation_interface(
    Core::FE::CellType celltype,
    const Discret::Elements::SolidElementProperties& element_properties)
{
  // We have 4 different element properties and each combination results in a different element
  // formulation.
  return Core::FE::cell_type_switch<Internal::ImplementedSolidCellTypes>(celltype,
      [&](auto celltype_t)
      {
        return magic_enum::enum_switch(
            [&](auto kinemtype_t)
            {
              return magic_enum::enum_switch(
                  [&](auto eletech_t)
                  {
                    return magic_enum::enum_switch(
                        [&](auto prestress_tech_t) -> SolidCalcVariant
                        {
                          constexpr Core::FE::CellType celltype_c = celltype_t();
                          constexpr Inpar::Solid::KinemType kinemtype_c = kinemtype_t();
                          constexpr ElementTechnology eletech_c = eletech_t();
                          constexpr PrestressTechnology prestress_tech_c = prestress_tech_t();
                          if constexpr (is_valid_type<SolidCalculationFormulation<celltype_c,
                                            kinemtype_c, eletech_c, prestress_tech_c>>)
                          {
                            return typename SolidCalculationFormulation<celltype_c, kinemtype_c,
                                eletech_c, prestress_tech_c>::type();
                          }

                          FOUR_C_THROW(
                              "Your element formulation with cell type {}, kinematic type {},"
                              " element technology {} and prestress type {} does not exist ",
                              Core::FE::celltype_string<celltype_t()>,
                              Inpar::Solid::kinem_type_string(element_properties.kintype).c_str(),
                              element_technology_string(element_properties.element_technology)
                                  .c_str(),
                              element_properties.prestress_technology);
                        },
                        element_properties.prestress_technology);
                  },
                  element_properties.element_technology);
            },
            element_properties.kintype);
      });
}

FOUR_C_NAMESPACE_CLOSE
