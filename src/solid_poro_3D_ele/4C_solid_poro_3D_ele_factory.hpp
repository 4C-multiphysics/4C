// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_PORO_3D_ELE_FACTORY_HPP
#define FOUR_C_SOLID_PORO_3D_ELE_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_solid_3D_ele_factory.hpp"
#include "4C_solid_3D_ele_factory_lib.hpp"
#include "4C_solid_poro_3D_ele_calc_pressure_based.hpp"
#include "4C_solid_poro_3D_ele_calc_pressure_velocity_based.hpp"
#include "4C_solid_scatra_3D_ele_factory.hpp"

#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{

  namespace Internal
  {
    using ImplementedSolidPoroCellTypes = Core::FE::CelltypeSequence<Core::FE::CellType::hex8,
        Core::FE::CellType::hex27, Core::FE::CellType::tet4, Core::FE::CellType::tet10>;

    using PoroPressureBasedEvaluators =
        Core::FE::apply_celltype_sequence<Discret::Elements::SolidPoroPressureBasedEleCalc,
            ImplementedSolidPoroCellTypes>;

    using SolidPoroPressureBasedEvaluators = Core::FE::Join<PoroPressureBasedEvaluators>;

    using ImplementedSolidPoroPressureVelocityBasedCellTypes =
        Core::FE::CelltypeSequence<Core::FE::CellType::hex8, Core::FE::CellType::hex27,
            Core::FE::CellType::tet4, Core::FE::CellType::tet10>;

    template <Core::FE::CellType celltype>
    using SolidPoroDefaultPressureVelocityBasedEleCalc =
        SolidPoroPressureVelocityBasedEleCalc<celltype,
            Discret::Elements::PorosityFormulation::implicit_porosity>;

    using PoroPressureVelocityBasedEvaluators =
        Core::FE::apply_celltype_sequence<SolidPoroDefaultPressureVelocityBasedEleCalc,
            ImplementedSolidPoroPressureVelocityBasedCellTypes>;


    template <Core::FE::CellType celltype>
    using SolidPoroPressureVelocityBasedP1EleCalc = SolidPoroPressureVelocityBasedEleCalc<celltype,
        Discret::Elements::PorosityFormulation::explicit_porosity>;

    using PoroPressureVelocityBasedP1Evaluators =
        Core::FE::apply_celltype_sequence<SolidPoroPressureVelocityBasedP1EleCalc,
            ImplementedSolidPoroPressureVelocityBasedCellTypes>;


    using SolidPoroPressureVelocityBasedEvaluators =
        Core::FE::Join<PoroPressureVelocityBasedEvaluators>;

    using SolidPoroPressureVelocityBasedP1Evaluators =
        Core::FE::Join<PoroPressureVelocityBasedP1Evaluators>;

    // Solid-Poro simulations might also carry a scalar. The solid-interfance can, therefore, be a
    // Solid-Scalar or a pure Solid.
    template <class... Args>
    struct VariantUnionHelper;

    template <class... Args1, class... Args2>
    struct VariantUnionHelper<std::variant<Args1...>, std::variant<Args2...>>
    {
      using type = std::variant<Args1..., Args2...>;
    };
  }  // namespace Internal


  using SolidAndSolidScatraCalcVariant =
      Internal::VariantUnionHelper<SolidCalcVariant, SolidScatraCalcVariant>::type;

  inline SolidAndSolidScatraCalcVariant create_solid_or_solid_scatra_calculation_interface(
      Core::FE::CellType celltype,
      const Discret::Elements::SolidElementProperties& element_properties, bool with_scatra)
  {
    if (with_scatra)
    {
      SolidScatraCalcVariant solid_scatra_item =
          create_solid_scatra_calculation_interface(celltype, element_properties);
      return std::visit([](auto& interface) -> SolidAndSolidScatraCalcVariant { return interface; },
          solid_scatra_item);
    }

    SolidCalcVariant solid_item = create_solid_calculation_interface(celltype, element_properties);
    return std::visit(
        [](auto& interface) -> SolidAndSolidScatraCalcVariant { return interface; }, solid_item);
  };

  using SolidPoroPressureBasedCalcVariant =
      CreateVariantType<Internal::SolidPoroPressureBasedEvaluators>;

  SolidPoroPressureBasedCalcVariant create_solid_poro_pressure_based_calculation_interface(
      Core::FE::CellType celltype);

  template <Core::FE::CellType celltype>
  SolidPoroPressureBasedCalcVariant create_solid_poro_pressure_based_calculation_interface();

  using SolidPoroPressureVelocityBasedCalcVariant =
      CreateVariantType<Internal::SolidPoroPressureVelocityBasedEvaluators>;

  using SolidPoroPressureVelocityBasedP1CalcVariant =
      CreateVariantType<Internal::SolidPoroPressureVelocityBasedP1Evaluators>;


  SolidPoroPressureVelocityBasedCalcVariant
  create_solid_poro_pressure_velocity_based_calculation_interface(Core::FE::CellType celltype);

  template <Core::FE::CellType celltype>
  SolidPoroPressureVelocityBasedCalcVariant
  create_solid_poro_pressure_velocity_based_calculation_interface();


  SolidPoroPressureVelocityBasedP1CalcVariant
  create_solid_poro_pressure_velocity_based_p1_calculation_interface(Core::FE::CellType celltype);

  template <Core::FE::CellType celltype>
  SolidPoroPressureVelocityBasedP1CalcVariant
  create_solid_poro_pressure_velocity_based_p1_calculation_interface();

}  // namespace Discret::Elements


FOUR_C_NAMESPACE_CLOSE

#endif