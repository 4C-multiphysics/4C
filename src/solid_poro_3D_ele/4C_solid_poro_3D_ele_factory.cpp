// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_poro_3D_ele_factory.hpp"

#include "4C_solid_poro_3D_ele_calc_lib.hpp"

FOUR_C_NAMESPACE_OPEN



Discret::Elements::SolidPoroPressureBasedCalcVariant
Discret::Elements::create_solid_poro_pressure_based_calculation_interface(
    Core::FE::CellType celltype)
{
  return Core::FE::cell_type_switch<Discret::Elements::Internal::ImplementedSolidPoroCellTypes>(
      celltype, [&](auto celltype_t)
      { return create_solid_poro_pressure_based_calculation_interface<celltype_t()>(); });
}

template <Core::FE::CellType celltype>
Discret::Elements::SolidPoroPressureBasedCalcVariant
Discret::Elements::create_solid_poro_pressure_based_calculation_interface()
{
  return SolidPoroPressureBasedEleCalc<celltype>();
}

Discret::Elements::SolidPoroPressureVelocityBasedCalcVariant
Discret::Elements::create_solid_poro_pressure_velocity_based_calculation_interface(
    Core::FE::CellType celltype)
{
  return Core::FE::cell_type_switch<
      Discret::Elements::Internal::ImplementedSolidPoroPressureVelocityBasedCellTypes>(celltype,
      [&](auto celltype_t)
      { return create_solid_poro_pressure_velocity_based_calculation_interface<celltype_t()>(); });
}


template <Core::FE::CellType celltype>
Discret::Elements::SolidPoroPressureVelocityBasedCalcVariant
Discret::Elements::create_solid_poro_pressure_velocity_based_calculation_interface()
{
  return SolidPoroPressureVelocityBasedEleCalc<celltype, PorosityFormulation::implicit_porosity>();
}

Discret::Elements::SolidPoroPressureVelocityBasedP1CalcVariant
Discret::Elements::create_solid_poro_pressure_velocity_based_p1_calculation_interface(
    Core::FE::CellType celltype)
{
  return Core::FE::cell_type_switch<
      Discret::Elements::Internal::ImplementedSolidPoroPressureVelocityBasedCellTypes>(celltype,
      [&](auto celltype_t)
      {
        return create_solid_poro_pressure_velocity_based_p1_calculation_interface<celltype_t()>();
      });
}


template <Core::FE::CellType celltype>
Discret::Elements::SolidPoroPressureVelocityBasedP1CalcVariant
Discret::Elements::create_solid_poro_pressure_velocity_based_p1_calculation_interface()
{
  return SolidPoroPressureVelocityBasedEleCalc<celltype, PorosityFormulation::explicit_porosity>();
}



FOUR_C_NAMESPACE_CLOSE
