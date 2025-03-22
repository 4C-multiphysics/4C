// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_scatra_3D_ele_factory.hpp"

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_solid_3D_ele_factory_lib.hpp"
#include "4C_solid_3D_ele_properties.hpp"
#include "4C_utils_enum.hpp"

#include <magic_enum/magic_enum_switch.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{
  /*!
   * @brief A template class that is taking different element formulation switches as template
   * parameter. If implemented, the struct defines the type of the solid formulation in the
   * solid-scatra element evaluation.
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
  struct SolidScatraCalculationFormulation
  {
  };

  /*!
   * @brief Standard nonlinear displacement based total lagrangian formulation valid for all
   * celltypes
   */
  template <Core::FE::CellType celltype>
  struct SolidScatraCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::none, Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::Internal::DisplacementBasedSolidScatraIntegrator<celltype>;
  };

  /*!
   * @brief Standard nonlinear displacement based total lagrangian formulation valid for all
   * celltypes
   */
  template <Core::FE::CellType celltype>
  struct SolidScatraCalculationFormulation<celltype, Inpar::Solid::KinemType::linear,
      Discret::Elements::ElementTechnology::none, Discret::Elements::PrestressTechnology::none>
  {
    using type =
        Discret::Elements::Internal::DisplacementBasedLinearKinematicsSolidScatraIntegrator<
            celltype>;
  };

  /*!
   * @brief Nonlinear total lagrangian formulation with F-Bar for hex8 and pyramid 5
   */
  template <Core::FE::CellType celltype>
    requires(celltype == Core::FE::CellType::hex8)
  struct SolidScatraCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::fbar, Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::Internal::FBarSolidScatraIntegrator<celltype>;
  };

  /*!
   * @brief EAS full Formulation for hex8
   */
  template <Core::FE::CellType celltype>
    requires(celltype == Core::FE::CellType::hex8)
  struct SolidScatraCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::eas_full, Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::Internal::EASSolidScatraIntegrator<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_h8_21, Inpar::Solid::KinemType::nonlinearTotLag>;
  };

  /*!
   * @brief EAS mild Formulation for hex8
   */
  template <Core::FE::CellType celltype>
    requires(celltype == Core::FE::CellType::hex8)
  struct SolidScatraCalculationFormulation<celltype, Inpar::Solid::KinemType::nonlinearTotLag,
      Discret::Elements::ElementTechnology::eas_mild, Discret::Elements::PrestressTechnology::none>
  {
    using type = Discret::Elements::Internal::EASSolidScatraIntegrator<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_h8_9, Inpar::Solid::KinemType::nonlinearTotLag>;
  };
}  // namespace

void Discret::Elements::add_to_pack(Core::Communication::PackBuffer& data,
    const Discret::Elements::SolidScatraElementProperties& properties)
{
  add_to_pack(data, properties.impltype);

  Discret::Elements::add_to_pack(data, properties.solid);
}

void Discret::Elements::extract_from_pack(Core::Communication::UnpackBuffer& buffer,
    Discret::Elements::SolidScatraElementProperties& properties)
{
  extract_from_pack(buffer, properties.impltype);

  Discret::Elements::extract_from_pack(buffer, properties.solid);
}

Discret::Elements::SolidScatraCalcVariant
Discret::Elements::create_solid_scatra_calculation_interface(Core::FE::CellType celltype,
    const Discret::Elements::SolidElementProperties& element_properties)
{
  // We have 4 different element properties and each combination results in a different element
  // formulation.
  return Core::FE::cell_type_switch<Internal::ImplementedSolidScatraCellTypes>(celltype,
      [&](auto celltype_t)
      {
        return magic_enum::enum_switch(
            [&](auto kinemtype_t)
            {
              return magic_enum::enum_switch(
                  [&](auto eletech_t)
                  {
                    return magic_enum::enum_switch(
                        [&](auto prestress_tech_t) -> SolidScatraCalcVariant
                        {
                          constexpr Core::FE::CellType celltype_c = celltype_t();
                          constexpr Inpar::Solid::KinemType kinemtype_c = kinemtype_t();
                          constexpr ElementTechnology eletech_c = eletech_t();
                          constexpr PrestressTechnology prestress_tech_c = prestress_tech_t();
                          if constexpr (is_valid_type<SolidScatraCalculationFormulation<celltype_c,
                                            kinemtype_c, eletech_c, prestress_tech_c>>)
                          {
                            return typename SolidScatraCalculationFormulation<celltype_c,
                                kinemtype_c, eletech_c, prestress_tech_c>::type();
                          }

                          FOUR_C_THROW(
                              "Your element formulation with cell type {}, kinematic type {},"
                              " element technology {} and prestress type {} does not exist in the "
                              "solid-scatra context.",
                              Core::FE::celltype_string<celltype_t()>, element_properties.kintype,
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