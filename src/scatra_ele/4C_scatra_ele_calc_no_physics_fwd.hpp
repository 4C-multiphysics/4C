// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_NO_PHYSICS_FWD_HPP
#define FOUR_C_SCATRA_ELE_CALC_NO_PHYSICS_FWD_HPP

/*----------------------------------------------------------------------*/
/*! \file

\brief forward declarations for scatra_ele_calc_no_physics classes

\level 2


*/

FOUR_C_NAMESPACE_OPEN

// 1D elements
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::line2, 1>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::line3, 1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::tri3, 2>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::tri6, 2>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::quad4, 3>;
// template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::hex8, 3>;
// template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::tet10, 3>;
// template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::pyramid5, 3>;
// template class Discret::Elements::ScaTraEleCalcNoPhysics<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE

#endif
