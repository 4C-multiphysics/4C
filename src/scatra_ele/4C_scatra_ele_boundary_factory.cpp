// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_boundary_factory.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_elchmat.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_boundary_calc_elch_diffcond.hpp"
#include "4C_scatra_ele_boundary_calc_elch_electrode.hpp"
#include "4C_scatra_ele_boundary_calc_elch_electrode_growth.hpp"
#include "4C_scatra_ele_boundary_calc_elch_electrode_sti_thermo.hpp"
#include "4C_scatra_ele_boundary_calc_elch_NP.hpp"
#include "4C_scatra_ele_boundary_calc_loma.hpp"
#include "4C_scatra_ele_boundary_calc_poro.hpp"
#include "4C_scatra_ele_boundary_calc_std.hpp"
#include "4C_scatra_ele_boundary_calc_sti_electrode.hpp"
#include "4C_scatra_ele_boundary_interface.hpp"
#include "4C_scatra_ele_calc.hpp"
#include "4C_scatra_input.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN


/*--------------------------------------------------------------------------*
 |                                                 (public) rasthofer 11/13 |
 *--------------------------------------------------------------------------*/
Discret::Elements::ScaTraBoundaryInterface* Discret::Elements::ScaTraBoundaryFactory::provide_impl(
    const Core::Elements::Element* ele, const ScaTra::ImplType impltype, const int numdofpernode,
    const int numscal, const std::string& disname)
{
  // number of space dimensions
  const int ndim = disname != "scatra_micro" ? Global::Problem::instance()->n_dim() : 1;

  switch (ele->shape())
  {
    case Core::FE::CellType::quad4:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::quad4, 3>(
            impltype, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension!");
    }
    case Core::FE::CellType::quad8:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::quad8, 3>(
            impltype, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension!");
    }
    case Core::FE::CellType::quad9:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::quad9, 3>(
            impltype, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension!");
    }
    case Core::FE::CellType::tri3:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::tri3, 3>(
            impltype, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension!");
    }
    case Core::FE::CellType::tri6:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::tri6, 3>(
            impltype, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension!");
    }
    case Core::FE::CellType::line2:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::line2, 2>(
            impltype, numdofpernode, numscal, disname);
      else if (ndim == 3)
        return define_problem_type<Core::FE::CellType::line2, 3>(
            impltype, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension!");
    }
    case Core::FE::CellType::line3:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::line3, 2>(
            impltype, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension!");
    }
    case Core::FE::CellType::nurbs3:  // 1D nurbs boundary element
    {
      return define_problem_type<Core::FE::CellType::nurbs3, 2>(
          impltype, numdofpernode, numscal, disname);
    }
    case Core::FE::CellType::nurbs9:  // 2D nurbs boundary element
    {
      return define_problem_type<Core::FE::CellType::nurbs9, 3>(
          impltype, numdofpernode, numscal, disname);
    }
    default:
    {
      FOUR_C_THROW(
          "Element shape {} ({} nodes) not activated. Just do it.", ele->shape(), ele->num_node());
      break;
    }
  }

  return nullptr;
}


/*-------------------------------------------------------------------------------------------*
 | return instance of element evaluation class depending on implementation type   fang 02/15 |
 *-------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraBoundaryInterface*
Discret::Elements::ScaTraBoundaryFactory::define_problem_type(const ScaTra::ImplType impltype,
    const int numdofpernode, const int numscal, const std::string& disname)
{
  switch (impltype)
  {
    case ScaTra::impltype_advreac:
    case ScaTra::impltype_aniso:
    case ScaTra::impltype_cardiac_monodomain:
    case ScaTra::impltype_gr:
    case ScaTra::impltype_nl_stimulus:
    case ScaTra::impltype_chemo:
    case ScaTra::impltype_chemoreac:
    case ScaTra::impltype_levelset:
    case ScaTra::impltype_std:
    case ScaTra::impltype_thermo_elch_diffcond:
    case ScaTra::impltype_multipororeac:
    {
      return Discret::Elements::ScaTraEleBoundaryCalcStd<distype, probdim>::instance(
          numdofpernode, numscal, disname);
      break;
    }
    case ScaTra::impltype_loma:
    {
      return Discret::Elements::ScaTraEleBoundaryCalcLoma<distype, probdim>::instance(
          numdofpernode, numscal, disname);
      break;
    }
    case ScaTra::impltype_elch_electrode:
    {
      return Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype, probdim>::instance(
          numdofpernode, numscal, disname);
      break;
    }
    case ScaTra::impltype_elch_electrode_thermo:
    {
      return Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype,
          probdim>::instance(numdofpernode, numscal, disname);
      break;
    }
    case ScaTra::impltype_elch_diffcond:
    case ScaTra::impltype_elch_diffcond_thermo:
    {
      return Discret::Elements::ScaTraEleBoundaryCalcElchDiffCond<distype, probdim>::instance(
          numdofpernode, numscal, disname);
      break;
    }
    case ScaTra::impltype_elch_NP:
    {
      return Discret::Elements::ScaTraEleBoundaryCalcElchNP<distype, probdim>::instance(
          numdofpernode, numscal, disname);
      break;
    }
    case ScaTra::impltype_poro:
    case ScaTra::impltype_pororeac:
    {
      return Discret::Elements::ScaTraEleBoundaryCalcPoro<distype, probdim>::instance(
          numdofpernode, numscal, disname);
      break;
    }
    case ScaTra::impltype_thermo_elch_electrode:
    {
      return Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype, probdim>::instance(
          numdofpernode, numscal, disname);
      break;
    }
    case ScaTra::impltype_elch_electrode_growth:
    {
      return Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeGrowth<distype,
          probdim>::instance(numdofpernode, numscal, disname);
      break;
    }
    default:
    {
      FOUR_C_THROW("Defined implementation type does not exist!");
      break;
    }
  }  // switch(impltype)

  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE
