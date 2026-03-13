// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_ele_factory.hpp"

#include "4C_porofluid_pressure_based_ele_calc.hpp"

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------------*
 | provide the implementation of evaluation class      (public) vuong 08/16 |
 *--------------------------------------------------------------------------*/
Discret::Elements::PoroFluidMultiPhaseEleInterface*
Discret::Elements::PoroFluidMultiPhaseFactory::provide_impl(Core::FE::CellType distype,
    int spatial_dimension, const int numdofpernode, const std::string& disname)
{
  switch (distype)
  {
    case Core::FE::CellType::quad4:
    {
      if (spatial_dimension == 2)
        return define_problem_type<Core::FE::CellType::quad4>(numdofpernode, disname);
      else
        FOUR_C_THROW("invalid problem dimension for quad4 porofluidmultiphase element!");
      break;
    }
      //  case Core::FE::CellType::quad8:
      //  {
      //    if(ndim==2)
      //      return
      //      define_problem_type<Core::FE::CellType::quad8>(numdofpernode,disname);
      //    else
      //      FOUR_C_THROW("invalid problem dimension for quad8 porofluidmultiphase element!");
      //    break;
      //  }
    case Core::FE::CellType::quad9:
    {
      if (spatial_dimension == 2)
        return define_problem_type<Core::FE::CellType::quad9>(numdofpernode, disname);
      else
        FOUR_C_THROW("invalid problem dimension for quad9 porofluidmultiphase element!");
      break;
    }
    case Core::FE::CellType::tri3:
    {
      if (spatial_dimension == 2)
        return define_problem_type<Core::FE::CellType::tri3>(numdofpernode, disname);
      else
        FOUR_C_THROW("invalid problem dimension for tri3 porofluidmultiphase element!");
      break;
    }
      //  case Core::FE::CellType::tri6:
      //  {
      //    if(ndim==2)
      //      return
      //      define_problem_type<Core::FE::CellType::tri6>(numdofpernode,disname);
      //    else
      //      FOUR_C_THROW("invalid problem dimension for tri6 porofluidmultiphase element!");
      //    break;
      //  }
    case Core::FE::CellType::line2:
    {
      if (spatial_dimension == 1)
        return define_problem_type<Core::FE::CellType::line2>(numdofpernode, disname);
      else
        FOUR_C_THROW("invalid problem dimension for line2 porofluidmultiphase element!");
      break;
    }
    case Core::FE::CellType::line3:
    {
      if (spatial_dimension == 1)
        return define_problem_type<Core::FE::CellType::line3>(numdofpernode, disname);
      else
        FOUR_C_THROW("invalid problem dimension for line3 porofluidmultiphase element!");
      break;
    }
    case Core::FE::CellType::hex8:
    {
      if (spatial_dimension == 3)
        return define_problem_type<Core::FE::CellType::hex8>(numdofpernode, disname);
      else
        FOUR_C_THROW("invalid problem dimension for hex8 porofluidmultiphase element!");
      break;
    }
    case Core::FE::CellType::hex27:
    {
      if (spatial_dimension == 3)
        return define_problem_type<Core::FE::CellType::hex27>(numdofpernode, disname);
      else
        FOUR_C_THROW("invalid problem dimension for hex27 porofluidmultiphase element!");
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (spatial_dimension == 3)
        return define_problem_type<Core::FE::CellType::tet4>(numdofpernode, disname);
      else
        FOUR_C_THROW("invalid problem dimension for tet4 porofluidmultiphase element!");
      break;
    }
    case Core::FE::CellType::tet10:
    {
      if (spatial_dimension == 3)
        return define_problem_type<Core::FE::CellType::tet10>(numdofpernode, disname);
      else
        FOUR_C_THROW("invalid problem dimension for tet10 porofluidmultiphase element!");
      break;
    }
    default:
      FOUR_C_THROW(
          "Element shape {} not activated. Just do it.", Core::FE::cell_type_to_string(distype));
      break;
  }
  return nullptr;
}

/*--------------------------------------------------------------------------*
 | provide the implementation of evaluation class      (public) vuong 08/16 |
 *--------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::PoroFluidMultiPhaseEleInterface*
Discret::Elements::PoroFluidMultiPhaseFactory::define_problem_type(
    const int numdofpernode, const std::string& disname)
{
  return Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::instance(numdofpernode, disname);
}

FOUR_C_NAMESPACE_CLOSE
