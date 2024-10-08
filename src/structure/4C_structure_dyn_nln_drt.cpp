/*---------------------------------------------------------------------*/
/*! \file
\brief Control routine for structural dynamics (outsourced to adapter layer)

\level 1

*/
/*---------------------------------------------------------------------*/

#include "4C_structure_dyn_nln_drt.hpp"

#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_resulttest.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void caldyn_drt()
{
  // get input lists
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  // major switch to different time integrators
  switch (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdyn, "DYNAMICTYP"))
  {
    case Inpar::Solid::dyna_statics:
    case Inpar::Solid::dyna_genalpha:
    case Inpar::Solid::dyna_genalpha_liegroup:
    case Inpar::Solid::dyna_onesteptheta:
    case Inpar::Solid::dyna_expleuler:
    case Inpar::Solid::dyna_centrdiff:
    case Inpar::Solid::dyna_ab2:
    case Inpar::Solid::dyna_ab4:
      dyn_nlnstructural_drt();
      break;
    default:
      FOUR_C_THROW(
          "unknown time integration scheme '%s'", sdyn.get<std::string>("DYNAMICTYP").c_str());
      break;
  }

  return;
}


/*----------------------------------------------------------------------*
 | structural nonlinear dynamics                                        |
 *----------------------------------------------------------------------*/
void dyn_nlnstructural_drt()
{
  // get input lists
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  // access the structural discretization
  Teuchos::RCP<Core::FE::Discretization> structdis =
      Global::Problem::instance()->get_dis("structure");

  // connect degrees of freedom for periodic boundary conditions
  {
    Core::Conditions::PeriodicBoundaryConditions pbc_struct(structdis);

    if (pbc_struct.has_pbc())
    {
      pbc_struct.update_dofs_for_periodic_boundary_conditions();
    }
  }

  // create an adapterbase and adapter
  Teuchos::RCP<Adapter::Structure> structadapter = Teuchos::null;
  // FixMe The following switch is just a temporal hack, such we can jump between the new and the
  // old structure implementation. Has to be deleted after the clean-up has been finished!
  const auto intstrat =
      Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(sdyn, "INT_STRATEGY");
  switch (intstrat)
  {
    // -------------------------------------------------------------------
    // old implementation
    // -------------------------------------------------------------------
    case Inpar::Solid::int_old:
    {
      Teuchos::RCP<Adapter::StructureBaseAlgorithm> adapterbase_old_ptr =
          Teuchos::RCP(new Adapter::StructureBaseAlgorithm(
              sdyn, const_cast<Teuchos::ParameterList&>(sdyn), structdis));
      structadapter = adapterbase_old_ptr->structure_field();
      structadapter->setup();
      break;
    }
    // -------------------------------------------------------------------
    // new implementation
    // -------------------------------------------------------------------
    default:
    {
      Teuchos::RCP<Adapter::StructureBaseAlgorithmNew> adapterbase_ptr =
          Adapter::build_structure_algorithm(sdyn);
      adapterbase_ptr->init(sdyn, const_cast<Teuchos::ParameterList&>(sdyn), structdis);
      adapterbase_ptr->setup();
      structadapter = adapterbase_ptr->structure_field();
      break;
    }
  }

  const bool write_initial_state =
      Global::Problem::instance()->io_params().get<bool>("WRITE_INITIAL_STATE");
  const bool write_final_state =
      Global::Problem::instance()->io_params().get<bool>("WRITE_FINAL_STATE");

  // do restart
  const int restart = Global::Problem::instance()->restart();
  if (restart)
  {
    structadapter->read_restart(restart);
  }
  // write output at beginnning of calc
  else
  {
    if (write_initial_state)
    {
      constexpr bool force_prepare = true;
      structadapter->prepare_output(force_prepare);
      structadapter->output();
      structadapter->post_output();
    }
  }

  // run time integration
  structadapter->integrate();

  if (write_final_state && !structadapter->has_final_state_been_written())
  {
    constexpr bool forceWriteRestart = true;
    constexpr bool force_prepare = true;
    structadapter->prepare_output(force_prepare);
    structadapter->output(forceWriteRestart);
    structadapter->post_output();
  }

  // test results
  Global::Problem::instance()->add_field_test(structadapter->create_field_test());
  Global::Problem::instance()->test_all(structadapter->dof_row_map()->Comm());

  // print monitoring of time consumption
  Teuchos::RCP<const Teuchos::Comm<int>> TeuchosComm =
      Core::Communication::to_teuchos_comm<int>(structdis->get_comm());
  Teuchos::TimeMonitor::summarize(TeuchosComm.ptr(), std::cout, false, true, true);

  // time to go home...
  return;

}  // end of dyn_nlnstructural_drt()

FOUR_C_NAMESPACE_CLOSE
