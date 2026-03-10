// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_partitioned_1wc.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_adapter_str_wrapper.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_scatra_timint_cardiac_monodomain.hpp"
#include "4C_scatra_timint_implicit.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

SSI::SSIPart1WC::SSIPart1WC(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart(comm, globaltimeparams), isscatrafromfile_(false)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call he setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Setup this class                                         rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart1WC::init(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
    const std::string& struct_disname, const std::string& scatra_disname, bool isAle)
{
  // call setup of base class
  SSI::SSIPart::init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIPart1WC::do_struct_step()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n***********************\n STRUCTURE SOLVER \n***********************\n";
  }

  // Newton-Raphson iteration
  structure_field()->solve();
  // calculate stresses, strains, energies
  constexpr bool force_prepare = false;
  structure_field()->prepare_output(force_prepare);
  // update all single field solvers
  structure_field()->update();
  // write output to files
  structure_field()->output();
  // write output to screen
  structure_field()->print_step();
  // clean up
  structure_field()->discretization()->clear_state(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIPart1WC::do_scatra_step()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n***********************\n TRANSPORT SOLVER \n***********************\n";
  }

  // -------------------------------------------------------------------
  //      load solution from previously performed scatra simulation
  // -------------------------------------------------------------------
  if (isscatrafromfile_)
  {
    int diffsteps = structure_field()->dt() / scatra_field()->dt();
    if (scatra_field()->step() % diffsteps == 0)
    {
      Core::IO::DiscretizationReader reader(*scatra_field()->discretization(),
          Global::Problem::instance()->input_control_file(), scatra_field()->step());

      // check if this is a cardiac monodomain problem
      std::shared_ptr<ScaTra::TimIntCardiacMonodomain> cardmono =
          std::dynamic_pointer_cast<ScaTra::TimIntCardiacMonodomain>(scatra_field());

      if (cardmono == nullptr)
      {
        // read phinp from restart file
        std::shared_ptr<Core::LinAlg::MultiVector<double>> phinptemp = reader.read_vector("phinp");

        // update phinp
        scatra_field()->phinp()->update(1.0, *phinptemp, 0.0);
      }
      else
      {
        // create vector with noderowmap from previously performed scatra calculation
        std::shared_ptr<Core::LinAlg::Vector<double>> phinptemp =
            std::make_shared<Core::LinAlg::Vector<double>>(
                *cardmono->discretization()->node_row_map());

        // read phinp from restart file
        reader.read_vector(phinptemp, "phinp");

        // update phinp
        scatra_field()->phinp()->update(1.0, *phinptemp, 0.0);
      }
    }
  }
  // -------------------------------------------------------------------
  //                  solve nonlinear / linear equation
  // -------------------------------------------------------------------
  else
    scatra_field()->solve();


  // -------------------------------------------------------------------
  //                         update solution
  //        current solution becomes old solution of next timestep
  // -------------------------------------------------------------------
  scatra_field()->update();

  // -------------------------------------------------------------------
  // evaluate error for problems with analytical solution
  // -------------------------------------------------------------------
  scatra_field()->evaluate_error_compared_to_analytical_sol();

  // -------------------------------------------------------------------
  //                         output of solution
  // -------------------------------------------------------------------
  scatra_field()->check_and_write_output_and_restart();

  // cleanup
  scatra_field()->discretization()->clear_state();
}

/*----------------------------------------------------------------------*/
// prepare time step
/*----------------------------------------------------------------------*/
void SSI::SSIPart1WCSolidToScatra::prepare_time_step(bool printheader)
{
  increment_time_and_step();

  if (printheader) print_header();

  // if adaptive time stepping: calculate time step in scatra (prepare_time_step() of Scatra) and
  // pass to structure
  if (scatra_field()->time_step_adapted()) set_dt_from_scatra_to_structure();

  structure_field()->prepare_time_step();

  const int diffsteps = scatra_field()->dt() / structure_field()->dt();

  if (structure_field()->step() % diffsteps == 0)
  {
    if (is_s2i_kinetics_with_pseudo_contact()) structure_field()->determine_stress_strain();
    set_struct_solution(*structure_field()->dispn(), structure_field()->veln(),
        is_s2i_kinetics_with_pseudo_contact());
    scatra_field()->prepare_time_step();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
SSI::SSIPart1WCSolidToScatra::SSIPart1WCSolidToScatra(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart1WC(comm, globaltimeparams)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call he setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Setup this class                                         rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart1WCSolidToScatra::init(MPI_Comm comm,
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& scatraparams,
    const Teuchos::ParameterList& structparams, const std::string& struct_disname,
    const std::string& scatra_disname, bool isAle)
{
  // call setup of base class
  SSI::SSIPart1WC::init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIPart1WCSolidToScatra::timeloop()
{
  // safety checks
  check_is_init();
  check_is_setup();

  if (structure_field()->dt() > scatra_field()->dt())
  {
    FOUR_C_THROW(
        "Timestepsize of scatra should be equal or bigger than solid timestep in solid to scatra "
        "interaction");
  }

  const int diffsteps = scatra_field()->dt() / structure_field()->dt();

  while (not_finished())
  {
    prepare_time_step(false);
    do_struct_step();  // It has its own time and timestep variables, and it increments them by
                       // itself.
    if (structure_field()->step() % diffsteps == 0)
    {
      if (is_s2i_kinetics_with_pseudo_contact()) structure_field()->determine_stress_strain();
      set_struct_solution(*structure_field()->dispnp(), structure_field()->velnp(),
          is_s2i_kinetics_with_pseudo_contact());
      do_scatra_step();  // It has its own time and timestep variables, and it increments them by
                         // itself.
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
SSI::SSIPart1WCScatraToSolid::SSIPart1WCScatraToSolid(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart1WC(comm, globaltimeparams)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call he setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Setup this class                                         rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart1WCScatraToSolid::init(MPI_Comm comm,
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& scatraparams,
    const Teuchos::ParameterList& structparams, const std::string& struct_disname,
    const std::string& scatra_disname, bool isAle)
{
  // call setup of base class
  SSI::SSIPart1WC::init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);

  // Flag for reading scatra result from restart file instead of computing it
  isscatrafromfile_ =
      Global::Problem::instance()->ssi_control_params().get<bool>("SCATRA_FROM_RESTART_FILE");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIPart1WCScatraToSolid::timeloop()
{
  if (structure_field()->dt() < scatra_field()->dt())
  {
    FOUR_C_THROW(
        "Timestepsize of solid should be equal or bigger than scatra timestep in scatra to solid "
        "interaction");
  }

  // set zero velocity and displacement field for scatra
  auto zeros_structure =
      std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(), true);
  set_struct_solution(*zeros_structure, zeros_structure, false);

  scatra_field()->prepare_time_loop();

  const int diffsteps = structure_field()->dt() / scatra_field()->dt();
  while (!finished())
  {
    prepare_time_step();
    do_scatra_step();  // It has its own time and timestep variables, and it increments them by
                       // itself.
    if (scatra_field()->step() % diffsteps == 0)
    {
      set_scatra_solution(scatra_field()->phinp());

      // set micro scale value (projected to macro scale) to structure field
      if (macro_scale()) set_micro_scatra_solution(scatra_field()->phinp_micro());

      // evaluate temperature from function and set to structural discretization
      evaluate_and_set_temperature_field();

      // prepare_time_step() is called after solving the scalar transport, because then the
      // predictor will include the new scalar solution
      structure_field()->prepare_time_step();
      do_struct_step();  // It has its own time and timestep variables, and it increments them by
                         // itself.
    }
  }
}

/*----------------------------------------------------------------------*/
// prepare time step
/*----------------------------------------------------------------------*/
void SSI::SSIPart1WCScatraToSolid::prepare_time_step(bool printheader)
{
  increment_time_and_step();
  print_header();

  scatra_field()->prepare_time_step();
  // prepare_time_step of structure field is called later

  // copy time step to SSI problem, in case it was modified in ScaTra
  if (scatra_field()->time_step_adapted()) set_dt_from_scatra_to_ssi();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool SSI::SSIPart1WCScatraToSolid::finished() const
{
  if (diff_time_step_size())
    return !not_finished();
  else
    return !(
        not_finished() and scatra_field()->not_finished() and structure_field()->not_finished());
}

FOUR_C_NAMESPACE_CLOSE
