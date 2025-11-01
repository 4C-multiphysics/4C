// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_elch_moving_boundary_algorithm.hpp"

#include "4C_elch_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_scatra_timint_elch.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ElCh::MovingBoundaryAlgorithm::MovingBoundaryAlgorithm(MPI_Comm comm,
    const Teuchos::ParameterList& elchcontrol, const Teuchos::ParameterList& scatradyn,
    const Teuchos::ParameterList& solverparams)
    : ScaTraFluidAleCouplingAlgorithm(comm, scatradyn, "FSICoupling", solverparams),
      pseudotransient_(false),
      molarvolume_(elchcontrol.get<double>("MOLARVOLUME")),
      idispn_(nullptr),
      idispnp_(nullptr),
      iveln_(nullptr),
      itmax_(elchcontrol.get<int>("MOVBOUNDARYITEMAX")),
      ittol_(elchcontrol.get<double>("MOVBOUNDARYCONVTOL")),
      theta_(elchcontrol.get<double>("MOVBOUNDARYTHETA")),
      elch_params_(elchcontrol)
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ElCh::MovingBoundaryAlgorithm::init()
{
  // call setup in base class
  Adapter::ScaTraFluidAleCouplingAlgorithm::init();

  // safety check
  if (!scatra_field()->discretization()->has_condition("ScaTraFluxCalc"))
  {
    FOUR_C_THROW(
        "Scalar transport discretization must have boundary condition for flux calculation at FSI "
        "interface!");
  }

  pseudotransient_ = (Teuchos::getIntegralValue<ElCh::ElchMovingBoundary>(
                          elch_params_, "MOVINGBOUNDARY") == ElCh::elch_mov_bndry_pseudo_transient);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ElCh::MovingBoundaryAlgorithm::setup()
{
  // call init in base class
  Adapter::ScaTraFluidAleCouplingAlgorithm::setup();

  // set pointers
  idispn_ = fluid_field()->extract_interface_veln();
  idispnp_ = fluid_field()->extract_interface_veln();
  iveln_ = fluid_field()->extract_interface_veln();

  idispn_->put_scalar(0.0);
  idispnp_->put_scalar(0.0);
  iveln_->put_scalar(0.0);

  // calculate normal flux vector field only at FSICoupling boundaries (no output to file)
  if (pseudotransient_ or (theta_ < 0.999))
  {
    solve_scatra();  // set-up trueresidual_
  }

  // transfer moving mesh data
  scatra_field()->apply_mesh_movement(*ale_field()->dispnp());

  // initialize the multivector for all possible cases
  fluxn_ = scatra_field()->calc_flux_at_boundary(false);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ElCh::MovingBoundaryAlgorithm::time_loop()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // provide information about initial field (do not do for restarts!)
  if (step() == 0)
  {
    fluid_field()->statistics_and_output();
    if (algo_parameters().get<int>("RESTARTEVERY") != 0)
      fluid_field()->disc_writer()->write_vector("idispn", idispnp_);
    ale_field()->output();
  }

  // prepare scatra field
  scatra_field()->prepare_time_loop();

  if (not pseudotransient_)
  {
    // transfer convective velocity = fluid velocity - grid velocity
    scatra_field()->set_convective_velocity(*fluid_field()->convective_vel());
    scatra_field()->set_velocity_field(*fluid_field()->convective_vel());
    scatra_field()->set_acceleration_field(*fluid_field()->hist());
  }

  // transfer moving mesh data
  scatra_field()->apply_mesh_movement(*ale_field()->dispnp());

  // time loop
  while (not_finished())
  {
    // prepare next time step
    prepare_time_step();

    auto incr = fluid_field()->extract_interface_veln();
    incr->put_scalar(0.0);
    double incnorm = 0.0;
    int iter = 0;
    bool stopiter = false;

    // ToDo
    // improve this convergence test
    // (better check increment of ivel ????, test relative value etc.)
    while (!stopiter)  // do at least one step
    {
      iter++;

      /// compute interface displacement and velocity
      compute_interface_vectors(*idispnp_, *iveln_);

      // save guessed value before solve
      incr->update(1.0, *idispnp_, 0.0);

      // solve nonlinear Navier-Stokes system on a deforming mesh
      solve_fluid_ale();

      // solve transport equations for ion concentrations and electric potential
      solve_scatra();

      /// compute interface displacement and velocity
      compute_interface_vectors(*idispnp_, *iveln_);

      // compare with value after solving
      incr->update(-1.0, *idispnp_, 1.0);

      // compute L2 norm of increment
      incr->norm_2(&incnorm);

      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        std::cout << "After outer iteration " << iter << " of " << itmax_
                  << ":  ||idispnpinc|| = " << incnorm << std::endl;
      }
      if (incnorm < ittol_)
      {
        stopiter = true;
        if (Core::Communication::my_mpi_rank(get_comm()) == 0)
          std::cout << "   || Outer iteration loop converged! ||\n\n\n";
      }
      if (iter == itmax_)
      {
        stopiter = true;
        if (Core::Communication::my_mpi_rank(get_comm()) == 0)
          std::cout << "   || Maximum number of iterations reached: " << itmax_ << " ||\n\n\n";
      }
    }

    double normidsinp;
    idispnp_->norm_2(&normidsinp);
    std::cout << "norm of isdispnp = " << normidsinp << std::endl;

    // update all single field solvers
    update();

    // compute error for problems with analytical solution
    scatra_field()->evaluate_error_compared_to_analytical_sol();

    // write output to screen and files
    output();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ElCh::MovingBoundaryAlgorithm::prepare_time_step()
{
  increment_time_and_step();

  // screen output
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << std::endl;
    std::cout << "*************************************************************************"
              << std::endl;
    std::cout << "  MOVING-BOUNDARY ALGORITHM FOR ELECTROCHEMISTRY  ---  STEP = " << std::setw(4)
              << step() << "/" << std::setw(4) << n_step() << std::endl;
    std::cout << "*************************************************************************"
              << std::endl
              << std::endl;
  }

  fluid_field()->prepare_time_step();
  ale_field()->prepare_time_step();

  // prepare time step
  /* remark: initial velocity field has been transferred to scalar transport field in constructor of
   * ScaTraFluidCouplingMovingBoundaryAlgorithm (initialvelset_ == true). Time integration schemes,
   * such as the one-step-theta scheme, are thus initialized correctly.
   */
  scatra_field()->prepare_time_step();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ElCh::MovingBoundaryAlgorithm::solve_fluid_ale()
{
  // screen output
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << std::endl;
    std::cout << "*********************" << std::endl;
    std::cout << "  FLUID-ALE SOLVER   " << std::endl;
    std::cout << "*********************" << std::endl;
  }

  // solve nonlinear Navier-Stokes system on a moving mesh
  fluid_ale_nonlinear_solve(idispnp_, iveln_, pseudotransient_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ElCh::MovingBoundaryAlgorithm::solve_scatra()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << '\n';
    std::cout << "************************" << '\n';
    std::cout << "       ELCH SOLVER      " << '\n';
    std::cout << "************************" << '\n';
  }

  switch (fluid_field()->tim_int_scheme())
  {
    case Inpar::FLUID::timeint_npgenalpha:
    case Inpar::FLUID::timeint_afgenalpha:
      FOUR_C_THROW("ConvectiveVel() not implemented for Gen.Alpha versions");
    case Inpar::FLUID::timeint_one_step_theta:
    case Inpar::FLUID::timeint_bdf2:
    {
      if (not pseudotransient_)
      {
        // transfer convective velocity = fluid velocity - grid velocity
        scatra_field()->set_convective_velocity(*fluid_field()->convective_vel());
        scatra_field()->set_velocity_field(*fluid_field()->convective_vel());
        scatra_field()->set_acceleration_field(*fluid_field()->hist());
      }
    }
    break;
    default:
      FOUR_C_THROW("Time integration scheme not supported");
  }

  // transfer moving mesh data
  scatra_field()->apply_mesh_movement(*ale_field()->dispnp());

  // solve coupled electrochemistry equations
  scatra_field()->solve();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ElCh::MovingBoundaryAlgorithm::update()
{
  fluid_field()->update();
  ale_field()->update();
  scatra_field()->update();

  // perform time shift of interface displacement
  idispn_->update(1.0, *idispnp_, 0.0);
  // perform time shift of interface mass flux vectors
  fluxn_->update(1.0, *fluxnp_, 0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ElCh::MovingBoundaryAlgorithm::output()
{
  // Note: The order is important here! In here control file entries are
  // written. And these entries define the order in which the filters handle
  // the discretizations, which in turn defines the dof number ordering of the
  // discretizations.
  fluid_field()->statistics_and_output();
  // additional vector needed for restarts:
  int uprestart = algo_parameters().get<int>("RESTARTEVERY");
  if ((uprestart != 0) && (fluid_field()->step() % uprestart == 0))
  {
    fluid_field()->disc_writer()->write_vector("idispn", idispnp_);
  }

  // now the other physical fields
  scatra_field()->check_and_write_output_and_restart();
  ale_field()->output();
}


void ElCh::MovingBoundaryAlgorithm::compute_interface_vectors(
    Core::LinAlg::Vector<double>& idispnp, Core::LinAlg::Vector<double>& iveln)
{
  // calculate normal flux vector field at FSI boundaries (no output to file)
  fluxnp_ = scatra_field()->calc_flux_at_boundary(false);

  // access discretizations
  std::shared_ptr<Core::FE::Discretization> fluiddis = fluid_field()->discretization();
  std::shared_ptr<Core::FE::Discretization> scatradis = scatra_field()->discretization();

  // no support for multiple reactions at the interface !
  // id of the reacting species
  int reactingspeciesid = 0;

  const Core::LinAlg::Map& ivelmap = iveln.get_map();

  // loop over all local nodes of fluid discretization
  for (int lnodeid = 0; lnodeid < fluiddis->num_my_row_nodes(); lnodeid++)
  {
    // Here we rely on the fact that the scatra discretization
    // is a clone of the fluid mesh. => a scatra node has the same
    // local (and global) ID as its corresponding fluid node!

    // get the processor's local fluid node with the same lnodeid
    Core::Nodes::Node* fluidlnode = fluiddis->l_row_node(lnodeid);
    // get the degrees of freedom associated with this fluid node
    std::vector<int> fluidnodedofs = fluiddis->dof(0, fluidlnode);

    if (ivelmap.my_gid(fluidnodedofs[0]))  // is this GID (implies: node) relevant for iveln_?
    {
      // determine number of space dimensions (numdof - pressure dof)
      const int numdim = ((int)fluidnodedofs.size()) - 1;
      // number of dof per node in ScaTra
      int numscatradof = scatradis->num_dof(0, scatradis->l_row_node(lnodeid));

      std::vector<double> Values(numdim);
      for (int index = 0; index < numdim; ++index)
      {
        const int pos = lnodeid * numscatradof + reactingspeciesid;
        // interface growth has opposite direction of metal ion mass flow -> minus sign !!
        Values[index] = (-molarvolume_) *
                        (theta_ * (((*fluxnp_)(index)).local_values_as_span()[pos]) +
                            (1.0 - theta_) * (((*fluxn_)(index)).local_values_as_span()[pos]));
      }

      // now insert only the first numdim entries (pressure dof is not inserted!)
      iveln_->replace_global_values(numdim, Values.data(), fluidnodedofs.data());
    }
  }

  // have to compute an approximate displacement from given interface velocity
  // id^{n+1} = id^{n} + \delta t vel_i
  idispnp.update(1.0, *idispn_, 0.0);
  idispnp.update(dt(), *iveln_, 1.0);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ElCh::MovingBoundaryAlgorithm::read_restart(int step)
{
  ScaTraFluidCouplingAlgorithm::read_restart(step);

  ale_field()->read_restart(step);  // add reading of ALE restart data

  // finally read isdispn which was written to the fluid restart data
  Core::IO::DiscretizationReader reader(
      *fluid_field()->discretization(), Global::Problem::instance()->input_control_file(), step);
  reader.read_vector(idispn_, "idispn");
  // read same result into vector isdispnp_ as a 'good guess'
  reader.read_vector(idispnp_, "idispn");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ElCh::MovingBoundaryAlgorithm::test_results()
{
  auto* problem = Global::Problem::instance();
  problem->add_field_test(fluid_field()->create_field_test());
  problem->add_field_test(ale_field()->create_field_test());
  problem->add_field_test(scatra_field()->create_scatra_field_test());
  problem->test_all(scatra_field()->discretization()->get_comm());
}
FOUR_C_NAMESPACE_CLOSE
