// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_factory.hpp"

#include "4C_global_data.hpp"
#include "4C_scatra_ele_calc_advanced_reaction.hpp"
#include "4C_scatra_ele_calc_aniso.hpp"
#include "4C_scatra_ele_calc_artery.hpp"
#include "4C_scatra_ele_calc_cardiac_monodomain.hpp"
#include "4C_scatra_ele_calc_cardiac_monodomain_hdg.hpp"
#include "4C_scatra_ele_calc_chemo.hpp"
#include "4C_scatra_ele_calc_chemo_reac.hpp"
#include "4C_scatra_ele_calc_elch_diffcond.hpp"
#include "4C_scatra_ele_calc_elch_diffcond_multiscale.hpp"
#include "4C_scatra_ele_calc_elch_diffcond_sti_thermo.hpp"
#include "4C_scatra_ele_calc_elch_electrode.hpp"
#include "4C_scatra_ele_calc_elch_electrode_sti_thermo.hpp"
#include "4C_scatra_ele_calc_elch_NP.hpp"
#include "4C_scatra_ele_calc_elch_scl.hpp"
#include "4C_scatra_ele_calc_growth_remodel.hpp"
#include "4C_scatra_ele_calc_hdg.hpp"
#include "4C_scatra_ele_calc_loma.hpp"
#include "4C_scatra_ele_calc_ls.hpp"
#include "4C_scatra_ele_calc_lsreinit.hpp"
#include "4C_scatra_ele_calc_no_physics.hpp"
#include "4C_scatra_ele_calc_nonlocal_stimulus.hpp"
#include "4C_scatra_ele_calc_poro.hpp"
#include "4C_scatra_ele_calc_poro_reac.hpp"
#include "4C_scatra_ele_calc_poro_reac_ECM.hpp"
#include "4C_scatra_ele_calc_porofluid_pressure_based.hpp"
#include "4C_scatra_ele_calc_std.hpp"
#include "4C_scatra_ele_calc_sti_diffcond.hpp"
#include "4C_scatra_ele_calc_sti_electrode.hpp"
#include "4C_scatra_ele_calc_utils.hpp"
#include "4C_scatra_ele_parameter_std.hpp"

FOUR_C_NAMESPACE_OPEN


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
Discret::Elements::ScaTraEleInterface* Discret::Elements::ScaTraFactory::provide_impl(
    Core::FE::CellType distype, ScaTra::ImplType problem, const int numdofpernode,
    const int numscal, const std::string& disname)
{
  // number of space dimensions
  const int ndim =
      disname != "scatra_micro"
          ? Global::Problem::instance(
                Discret::Elements::ScaTraEleParameterStd::instance(disname)->prob_num())
                ->n_dim()
          : 1;

  switch (distype)
  {
    case Core::FE::CellType::hex8:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::hex8, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for HEX8 transport element!");
      break;
    }
    case Core::FE::CellType::hex27:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::hex27, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for HEX27 transport element!");
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::tet4, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for TET4 transport element!");
      break;
    }
    case Core::FE::CellType::tet10:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::tet10, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for TET10 transport element!");
      break;
    }
    case Core::FE::CellType::pyramid5:
    {
      if (ndim == 3)
      {
        return define_problem_type<Core::FE::CellType::pyramid5, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for PYRAMID5 transport element!");
      break;
    }
    case Core::FE::CellType::quad4:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::quad4, 2>(
            problem, numdofpernode, numscal, disname);
      else if (ndim == 3)
        return define_problem_type<Core::FE::CellType::quad4, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for quad4 transport element!");
      break;
    }
    case Core::FE::CellType::quad9:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::quad9, 2>(
            problem, numdofpernode, numscal, disname);
      else
      {
        FOUR_C_THROW(
            "QUAD9 transport element not implemented as part of {}-dimensional problem. Just do it",
            ndim);
      }
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::nurbs9, 2>(
            problem, numdofpernode, numscal, disname);
      else
      {
        FOUR_C_THROW(
            "NURBS9 transport element not implemented as part of {}-dimensional problem. Just do "
            "it",
            ndim);
      }
      break;
    }
    case Core::FE::CellType::tri3:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::tri3, 2>(
            problem, numdofpernode, numscal, disname);
      else if (ndim == 3)
        return define_problem_type<Core::FE::CellType::tri3, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for tri3 transport element!");
      break;
    }
    case Core::FE::CellType::tri6:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::tri6, 2>(
            problem, numdofpernode, numscal, disname);
      else
      {
        FOUR_C_THROW(
            "TRI6 transport element not implemented as part of {}-dimensional problem. Just do it",
            ndim);
      }
      break;
    }
    case Core::FE::CellType::line2:
    {
      if (ndim == 1)
        return define_problem_type<Core::FE::CellType::line2, 1>(
            problem, numdofpernode, numscal, disname);
      else if (ndim == 2)
        return define_problem_type<Core::FE::CellType::line2, 2>(
            problem, numdofpernode, numscal, disname);
      else if (ndim == 3)
        return define_problem_type<Core::FE::CellType::line2, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for LINE2 transport element!");
      break;
    }
    case Core::FE::CellType::line3:
    {
      if (ndim != 1)
      {
        FOUR_C_THROW(
            "LINE3 transport element not implemented as part of {}-dimensional problem. Just do it",
            ndim);
      }
      else
        return define_problem_type<Core::FE::CellType::line3, 1>(
            problem, numdofpernode, numscal, disname);
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
 *--------------------------------------------------------------------------*/
Discret::Elements::ScaTraEleInterface* Discret::Elements::ScaTraFactory::provide_impl_hdg(
    Core::FE::CellType distype, ScaTra::ImplType problem, const int numdofpernode,
    const int numscal, const std::string& disname)
{
  // -------------------------------------- number of degrees of freedom
  // number of degrees of freedom
  static const int ndim = Global::Problem::instance()->n_dim();

  switch (distype)
  {
    case Core::FE::CellType::hex8:
    {
      if (ndim == 3)
      {
        return define_problem_type_hdg<Core::FE::CellType::hex8, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for HEX8 transport element!");
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (ndim == 3)
      {
        return define_problem_type_hdg<Core::FE::CellType::tet4, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for TET4 transport element!");
      break;
    }
    case Core::FE::CellType::tet10:
    {
      if (ndim == 3)
      {
        return define_problem_type_hdg<Core::FE::CellType::tet10, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for TET10 transport element!");
      break;
    }
    case Core::FE::CellType::pyramid5:
    {
      if (ndim == 3)
      {
        return define_problem_type_hdg<Core::FE::CellType::pyramid5, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for PYRAMID5 transport element!");
      break;
    }
    case Core::FE::CellType::quad4:
    {
      if (ndim == 2)
      {
        return define_problem_type_hdg<Core::FE::CellType::quad4, 2>(
            problem, numdofpernode, numscal, disname);
      }
      else if (ndim == 3)
      {
        return define_problem_type_hdg<Core::FE::CellType::quad4, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for quad4 transport element!");
      break;
    }
    case Core::FE::CellType::tri3:
    {
      return define_problem_type_hdg<Core::FE::CellType::tri3, 2>(
          problem, numdofpernode, numscal, disname);
    }
    default:
      FOUR_C_THROW(
          "Element shape {} not activated. Just do it.", Core::FE::cell_type_to_string(distype));
      break;
  }
  return nullptr;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleInterface* Discret::Elements::ScaTraFactory::define_problem_type(
    ScaTra::ImplType problem, const int numdofpernode, const int numscal,
    const std::string& disname)
{
  if ((probdim - Core::FE::dim<distype>) == 1)
  {
    if (problem != ScaTra::impltype_std and problem != ScaTra::impltype_cardiac_monodomain and
        problem != ScaTra::impltype_gr and problem != ScaTra::impltype_advreac and
        problem != ScaTra::impltype_lsreinit and problem != ScaTra::impltype_one_d_artery and
        problem != ScaTra::impltype_no_physics and problem != ScaTra::impltype_elch_electrode and
        problem != ScaTra::impltype_elch_diffcond)
      FOUR_C_THROW("ImplType '{}' not implemented for transport on manifolds!",
          ScaTra::impl_type_to_string(problem));
  }

  switch (problem)
  {
    case ScaTra::impltype_std:
    {
      return Discret::Elements::ScaTraEleCalcStd<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_thermo_elch_electrode:
    {
      return Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_thermo_elch_diffcond:
    {
      return Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_levelset:
    {
      return Discret::Elements::ScaTraEleCalcLS<distype>::instance(numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_lsreinit:
    {
      return Discret::Elements::ScaTraEleCalcLsReinit<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_loma:
    {
      return Discret::Elements::ScaTraEleCalcLoma<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_elch_NP:
    {
      return Discret::Elements::ScaTraEleCalcElchNP<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_elch_electrode:
    case ScaTra::impltype_elch_electrode_growth:
    {
      return Discret::Elements::ScaTraEleCalcElchElectrode<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_elch_electrode_thermo:
    {
      return Discret::Elements::ScaTraEleCalcElchElectrodeSTIThermo<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_elch_diffcond:
    {
      return Discret::Elements::ScaTraEleCalcElchDiffCond<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_elch_diffcond_multiscale:
    {
      return Discret::Elements::ScaTraEleCalcElchDiffCondMultiScale<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_elch_diffcond_thermo:
    {
      return Discret::Elements::ScaTraEleCalcElchDiffCondSTIThermo<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_elch_scl:
    {
      return Discret::Elements::ScaTraEleCalcElchScl<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_poro:
    {
      return Discret::Elements::ScaTraEleCalcPoro<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_advreac:
    {
      return Discret::Elements::ScaTraEleCalcAdvReac<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_chemo:
    {
      return Discret::Elements::ScaTraEleCalcChemo<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_chemoreac:
    {
      return Discret::Elements::ScaTraEleCalcChemoReac<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_multipororeac:
    {
      return Discret::Elements::ScaTraEleCalcPorofluidPressureBased<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_pororeac:
    {
      return Discret::Elements::ScaTraEleCalcPoroReac<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_pororeacECM:
    {
      return Discret::Elements::ScaTraEleCalcPoroReacECM<distype>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_aniso:
    {
      return Discret::Elements::ScaTraEleCalcAniso<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_cardiac_monodomain:
    {
      return Discret::Elements::ScaTraEleCalcCardiacMonodomain<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_gr:
    {
      return Discret::Elements::ScaTraEleCalcGrowthRemodel<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_nl_stimulus:
    {
      return Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_one_d_artery:
    {
      return Discret::Elements::ScaTraEleCalcArtery<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_no_physics:
      return Discret::Elements::ScaTraEleCalcNoPhysics<distype, probdim>::instance(
          numdofpernode, numscal, disname);

    default:
    {
      FOUR_C_THROW("Defined problem type does not exist!!");
      break;
    }
  }

  return nullptr;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleInterface* Discret::Elements::ScaTraFactory::define_problem_type_hdg(
    ScaTra::ImplType problem, const int numdofpernode, const int numscal,
    const std::string& disname)
{
  if (Core::FE::dim<distype> != probdim)
  {
    if (problem != ScaTra::impltype_std and problem != ScaTra::impltype_cardiac_monodomain)
      FOUR_C_THROW("ImplType '{}' not implemented for transport on manifolds!",
          ScaTra::impl_type_to_string(problem));
  }

  switch (problem)
  {
    case ScaTra::impltype_std_hdg:
    {
      return Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    case ScaTra::impltype_cardiac_monodomain_hdg:
    {
      return Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::instance(
          numdofpernode, numscal, disname);
    }
    default:
    {
      FOUR_C_THROW("Defined problem type does not exist!!");
      break;
    }
  }

  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE
