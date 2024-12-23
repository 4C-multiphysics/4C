// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale_dyn.hpp"
#include "4C_art_net_dyn_drt.hpp"
#include "4C_ehl_dyn.hpp"
#include "4C_elch_dyn.hpp"
#include "4C_elemag_dyn.hpp"
#include "4C_fluid_dyn_nln_drt.hpp"
#include "4C_fpsi_dyn.hpp"
#include "4C_fs3i_dyn.hpp"
#include "4C_fsi_dyn.hpp"
#include "4C_global_data.hpp"
#include "4C_immersed_problem_dyn.hpp"
#include "4C_levelset_dyn.hpp"
#include "4C_loma_dyn.hpp"
#include "4C_lubrication_dyn.hpp"
#include "4C_particle_algorithm_sim.hpp"
#include "4C_pasi_dyn.hpp"
#include "4C_poroelast_dyn.hpp"
#include "4C_poroelast_scatra_dyn.hpp"
#include "4C_porofluidmultiphase_dyn.hpp"
#include "4C_poromultiphase_dyn.hpp"
#include "4C_poromultiphase_scatra_dyn.hpp"
#include "4C_red_airways_dyn_drt.hpp"
#include "4C_scatra_cardiac_monodomain_dyn.hpp"
#include "4C_scatra_dyn.hpp"
#include "4C_ssi_dyn.hpp"
#include "4C_ssti_dyn.hpp"
#include "4C_sti_dyn.hpp"
#include "4C_stru_multi_microstatic_npsupport.hpp"
#include "4C_structure_dyn_nln_drt.hpp"
#include "4C_thermo_dyn.hpp"
#include "4C_tsi_dyn.hpp"

/*----------------------------------------------------------------------*
 |  routine to control execution phase                   m.gee 6/01     |
 *----------------------------------------------------------------------*/
void ntacal()
{
  using namespace FourC;

  int restart = Global::Problem::instance()->restart();

  // choose the entry-routine depending on the problem type
  switch (Global::Problem::instance()->get_problem_type())
  {
    case Core::ProblemType::structure:
    case Core::ProblemType::polymernetwork:
      caldyn_drt();
      break;
    case Core::ProblemType::fluid:
    case Core::ProblemType::fluid_redmodels:
      dyn_fluid_drt(restart);
      break;
    case Core::ProblemType::lubrication:
      lubrication_dyn(restart);
      break;
    case Core::ProblemType::ehl:
      ehl_dyn();
      break;
    case Core::ProblemType::scatra:
      scatra_dyn(restart);
      break;
    case Core::ProblemType::cardiac_monodomain:
      scatra_cardiac_monodomain_dyn(restart);
      break;
    case Core::ProblemType::sti:
      sti_dyn(restart);
      break;
    case Core::ProblemType::fluid_xfem:
      fluid_xfem_drt();
      break;
      break;
    case Core::ProblemType::fluid_ale:
      fluid_ale_drt();
      break;

    case Core::ProblemType::fsi:
    case Core::ProblemType::fsi_redmodels:
      fsi_ale_drt();
      break;
    case Core::ProblemType::fsi_xfem:
      xfsi_drt();
      break;
    case Core::ProblemType::fpsi_xfem:
      xfpsi_drt();
      break;
    case Core::ProblemType::gas_fsi:
    case Core::ProblemType::biofilm_fsi:
    case Core::ProblemType::thermo_fsi:
    case Core::ProblemType::fps3i:
      fs3i_dyn();
      break;
    case Core::ProblemType::fbi:
      fsi_immersed_drt();
      break;

    case Core::ProblemType::ale:
      dyn_ale_drt();
      break;

    case Core::ProblemType::thermo:
      thermo_dyn_drt();
      break;

    case Core::ProblemType::tsi:
      tsi_dyn_drt();
      break;

    case Core::ProblemType::loma:
      loma_dyn(restart);
      break;

    case Core::ProblemType::elch:
      elch_dyn(restart);
      break;

    case Core::ProblemType::art_net:
      dyn_art_net_drt();
      break;

    case Core::ProblemType::red_airways:
      dyn_red_airways_drt();
      break;

    case Core::ProblemType::immersed_fsi:
      immersed_problem_drt();
      break;

    case Core::ProblemType::poroelast:
      poroelast_drt();
      break;
    case Core::ProblemType::poroscatra:
      poro_scatra_drt();
      break;
    case Core::ProblemType::porofluidmultiphase:
      porofluidmultiphase_dyn(restart);
      break;
    case Core::ProblemType::poromultiphase:
      poromultiphase_dyn(restart);
      break;
    case Core::ProblemType::poromultiphasescatra:
      poromultiphasescatra_dyn(restart);
      break;
    case Core::ProblemType::fpsi:
      fpsi_drt();
      break;
    case Core::ProblemType::ssi:
      ssi_drt();
      break;
    case Core::ProblemType::ssti:
      ssti_drt();
      break;
    case Core::ProblemType::redairways_tissue:
      redairway_tissue_dyn();
      break;

    case Core::ProblemType::particle:
      particle_drt();
      break;

    case Core::ProblemType::pasi:
      pasi_dyn();
      break;

    case Core::ProblemType::level_set:
      levelset_dyn(restart);
      break;

    case Core::ProblemType::np_support:
      MultiScale::np_support_drt();
      break;

    case Core::ProblemType::elemag:
      electromagnetics_drt();
      break;

    default:
      FOUR_C_THROW("solution of unknown problemtype %d requested",
          Global::Problem::instance()->get_problem_type());
      break;
  }
}
