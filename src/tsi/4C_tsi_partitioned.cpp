// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_tsi_partitioned.hpp"

#include "4C_adapter_str_structure.hpp"
#include "4C_contact_abstract_strategy.hpp"
#include "4C_contact_lagrange_strategy_tsi.hpp"
#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_thermo_adapter.hpp"
#include "4C_tsi_input.hpp"
#include "4C_tsi_utils.hpp"

FOUR_C_NAMESPACE_OPEN

//! Note: The order of calling the two BasePartitioned-constructors is
//! important here! In here control file entries are written. And these entries
//! define the order in which the filters handle the Discretizations, which in
//! turn defines the dof number ordering of the Discretizations.
/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 12/09 |
 *----------------------------------------------------------------------*/
TSI::Partitioned::Partitioned(MPI_Comm comm)
    : Algorithm(comm),
      tempincnp_(nullptr),
      dispincnp_(nullptr),
      disp_(nullptr),
      vel_(nullptr),
      temp_(nullptr),
      del_(nullptr),
      delhist_(nullptr),
      mu_(0.0)
{
  // call the TSI parameter list
  const Teuchos::ParameterList& tsidyn = Global::Problem::instance()->tsi_dynamic_params();
  // call the TSI parameter list
  const Teuchos::ParameterList& tsidynpart =
      Global::Problem::instance()->tsi_dynamic_params().sublist("PARTITIONED");

  // get the parameters for the convergence_check
  itmax_ = tsidyn.get<int>("ITEMAX");          // default: =1
  ittol_ = tsidynpart.get<double>("CONVTOL");  // default: =1e-6
  normtypeinc_ = Teuchos::getIntegralValue<TSI::ConvNorm>(tsidyn, "NORM_INC");

  // decide which coupling scheme is applied (e.g. one-way or full coupling)
  coupling_ = Teuchos::getIntegralValue<TSI::SolutionSchemeOverFields>(tsidyn, "COUPALGO");

  // coupling variable
  displacementcoupling_ = tsidynpart.get<std::string>("COUPVARIABLE") == "Displacement";
  if (displacementcoupling_)
    std::cout << "Coupling variable: displacement" << std::endl;
  else
    std::cout << "Coupling variable: temperature" << std::endl;

  // if structure field is quasi-static --> calc_velocity
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  // major switch to different time integrators
  quasistatic_ = (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdyn, "DYNAMICTYPE") ==
                  Inpar::Solid::DynamicType::Statics);

  // initialise internal variables with values
  tempincnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*(thermo_field()->tempnp()));
  dispincnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*(structure_field()->dispnp()));
  disp_ = std::make_shared<Core::LinAlg::Vector<double>>(*(structure_field()->dispn()));
  temp_ = std::make_shared<Core::LinAlg::Vector<double>>(*(thermo_field()->tempn()));
  // set internal variables to pointer of current velocities
  vel_ = structure_field()->write_access_velnp();

  // structural and thermal contact
  prepare_contact_strategy();
}



/*----------------------------------------------------------------------*
 | read restart information for given time step (public)     dano 12/09 |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::read_restart(int step)
{
  thermo_field()->read_restart(step);
  structure_field()->read_restart(step);
  set_time_step(thermo_field()->time_old(), step);

  // Material pointers to other field were deleted during read_restart().
  // They need to be reset.
  TSI::Utils::set_material_pointers_matching_grid(
      *structure_field()->discretization(), *thermo_field()->discretization());

  // structural and thermal contact
  prepare_contact_strategy();
}  // read_restart()


/*----------------------------------------------------------------------*
 | prepare time step (public)                                dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::prepare_time_step()
{
  // counter and print header
  increment_time_and_step();
  print_header();

}  // prepare_time_step()


/*----------------------------------------------------------------------*
 | initialise internal variables needed as guess             dano 02/12 |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::setup_system() {}  // SetupSystem()


/*----------------------------------------------------------------------*
 | non-linear solve, i.e. (multiple) corrector (public)      dano 12/09 |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::solve()
{
  // switch TSI algorithm (synchronous)
  // only the one field knows the 2nd field, that contains the coupling
  // variable
  // choose algorithm depending on solution type
  switch (coupling_)
  {
    case TSI::SolutionSchemeOverFields::OneWay:
    {
      time_loop_one_way();
      break;
    }
    // complete volume coupling system
    // sequential staggered scheme
    case TSI::SolutionSchemeOverFields::SequStagg:
    {
      time_loop_sequ_stagg();
      break;
    }
    // iterative staggered scheme
    case TSI::SolutionSchemeOverFields::IterStagg:
    case TSI::SolutionSchemeOverFields::IterStaggAitken:
    case TSI::SolutionSchemeOverFields::IterStaggAitkenIrons:
    case TSI::SolutionSchemeOverFields::IterStaggFixedRel:
    {
      time_loop_full();
      break;
    }
    default:
      FOUR_C_THROW("desired type of thermo-structure interaction algorithm not supported");
      break;
  }  // end switch

  return;
}  // Solve()


/*----------------------------------------------------------------------*
 | time loop                                                 dano 12/09 |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::time_loop()
{
  // tsi with or without contact
  // only tsi
  // time loop
  while (not_finished())
  {
    // counter and print header
    prepare_time_step();

    // normally prepare_time_step() should be called outside the nonlinear loop,
    // but at this point the coupling variables are not known, yet!
    // call prepare_time_step() after ApplyCouplingState()/ApplyStructVariables()

    // integrate time step
    solve();

    // calculate stresses, strains, energies
    prepare_output();

    // update all single field solvers
    update();

    // write output to screen and files
    output();

  }  // time loop

  // ==================================================================

}  // TSI::Partitioned::TimeLoop()


/*----------------------------------------------------------------------*
 | One-way coupling (only one field knows its counterpart)   dano 06/10 |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::time_loop_one_way()
{
  if (displacementcoupling_)  // (temperature change due to deformation)
  {
    // -------------------------------------------------- structure field

    // predict the structure field without influence of temperatures
    structure_field()->prepare_time_step();

    /// solve structural system
    // get current displacement due to solve structure step
    do_structure_step();

    // now extract the current displacements/velocities
    disp_ = structure_field()->dispnp();

    // major switch to different time integrators
    if (quasistatic_)
    {
      // the displacement -> velocity conversion
      // use this kind of calculation for quasi-static structure calculation
      vel_ = calc_velocity(*disp_);
    }
    //  else use vel_ from WriteAccessVelnp()

    // ----------------------------------------------------- thermo field

    // use the structural solution calculated in do_structure_step

    // call the current displacements and velocities and pass it to Thermo
    apply_struct_coupling_state(disp_, vel_);

    // call the predictor here, because the displacement field is now also
    // available
    thermo_field()->prepare_time_step();

    /// solve temperature system
    // do the solve for the time step. All boundary conditions have been set
    do_thermo_step();

  }  // displacement coupling
  else  // temperature coupling (deformation due to heating)
  {
    // ----------------------------------------------------- thermo field

    // predict the thermal field without influence of structure
    thermo_field()->prepare_time_step();

    /// solve temperature system
    // get current temperatures due to Solve Thermo Step
    do_thermo_step();

    // now extract the current temperatures and pass it to the structure
    temp_ = thermo_field()->tempnp();

    // -------------------------------------------------- structure field

    // pass the current temperatures to the structural field
    apply_thermo_coupling_state(temp_);

    // call the predictor here, because the temperature is now also available
    structure_field()->prepare_time_step();

    /// solve structure system
    // do the nonlinear solve for the time step. All boundary conditions have
    // been set.
    do_structure_step();

  }  // temperature coupling

}  // TSI::Partitioned::time_loop_one_way()


/*----------------------------------------------------------------------*
 | One-way coupling between the fields                       dano 09/10 |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::time_loop_sequ_stagg()
{
  // now extract the current temperatures and pass it to the structure
  temp_ = thermo_field()->tempnp();
  // extract final displacements,
  // update is called afterwards, so we extract the newest solution
  disp_ = structure_field()->dispnp();

  // like implicit-implicit staggered scheme, compare Farhat & Park, 1991
  if (displacementcoupling_)  // (temperature change due to deformation)
  {
    // begin nonlinear solver ************************************************

    // predict the displacement field (structural predictor for TSI)

    // get structure variables of old time step (d_n, v_n) use variables according to
    //  constructor

    // ----------------------------------------------------- thermo field

    // solve thermo field with predicted structural values (Apply d_n,v_n to
    // Thermo, solve coupled equation, extract new temperatures T_n+1)

    // pass the current displacements and velocities to the thermo field
    apply_struct_coupling_state(disp_, vel_);

    // prepare time step with coupled variables
    thermo_field()->prepare_time_step();

    // solve thermal system

    // do the solve for the time step. All boundary conditions have been set.
    do_thermo_step();

    // -------------------------------------------------- structure field

    // pass the current temperatures to the structure field
    apply_thermo_coupling_state(temp_);

    // prepare time step with coupled variables
    structure_field()->prepare_time_step();

    // solve coupled equation
    do_structure_step();

    // end nonlinear solver **************************************************

    // get the velocities needed as predictor for the thermo field for the next
    // time step
    if (quasistatic_) vel_ = calc_velocity(*disp_);
    // else vel_ = structure_field()->WriteAccessVelnp();

  }  // end displacement coupling

  // temperature is coupling variable
  else  // (deformation due to heating)
  {
    // begin nonlinear solver *************************************************

    // predict the temperature field (thermal predictor for TSI)

    // get temperature solution of old time step (is set in constructor)

    // -------------------------------------------------- structure field

    // get current displacement due to solve structure step

    // pass the current temperatures to the structure field
    apply_thermo_coupling_state(temp_);

    // prepare time step with coupled variables
    structure_field()->prepare_time_step();

    // solve structural coupled equation
    do_structure_step();

    // extract the velocities of the current solution
    // quasistatic to exclude oszillation, use displacements to compute velocities
    if (quasistatic_) vel_ = calc_velocity(*disp_);
    // else vel_ = WriteAccessVelnp_

    // ----------------------------------------------------- thermo field

    // pass the current displacements and velocities to the thermo field
    apply_struct_coupling_state(disp_, vel_);

    // prepare time step with coupled variables
    thermo_field()->prepare_time_step();

    /// solve temperature system
    /// do the nonlinear solve for the time step. All boundary conditions have
    /// been set.
    do_thermo_step();

    // end nonlinear solver **************************************************

  }  // temperature coupling

}  // TSI::Partitioned::TimeLoopSequStagg()


/*----------------------------------------------------------------------*
 | Full coupling (iterative staggered scheme)                dano 06/10 |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::time_loop_full()
{
  // extract final displacement and velocity
  // update is called afterwards, so we extract the newest solution
  disp_ = structure_field()->dispnp();
  temp_ = thermo_field()->tempnp();

  // outer iteration loop
  outer_iteration_loop();

}  // TSI::Partitioned::TimeLoopFull()


/*----------------------------------------------------------------------*
 | Outer Loop with convergence check                         dano 10/10 |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::outer_iteration_loop()
{
  // iterate between the two fields
  int itnum = 0;
  bool stopnonliniter = false;

  // outer iteration loop starts
  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
      (step() % print_screen_every() == 0))
  {
    std::cout << "\n";
    std::cout << "**************************************************************\n";
    std::cout << "      OUTER ITERATION LOOP \n";
    printf("      Time Step %3d/%3d \n", thermo_field()->step_old(), thermo_field()->num_step());
    std::cout << "**************************************************************\n";
  }

  // call the TSI parameter lists
  const Teuchos::ParameterList& tsidyn = Global::Problem::instance()->tsi_dynamic_params();
  const Teuchos::ParameterList& tsidynpart =
      Global::Problem::instance()->tsi_dynamic_params().sublist("PARTITIONED");

  // decide if one-way coupling or full coupling
  auto coupling = Teuchos::getIntegralValue<TSI::SolutionSchemeOverFields>(tsidyn, "COUPALGO");

  // Pure iterative staggered algorithms
  // iterative staggered TSI withOUT Aitken relaxation
  if (coupling == TSI::SolutionSchemeOverFields::IterStagg)
  {
    // structural predictor
    if (displacementcoupling_)  // (temperature change due to deformation)
    {
      // mechanical predictor for the coupling iteration outside the loop
      // get structure variables of old time step (d_n, v_n)
      // d^p_n+1 = d_n, v^p_n+1 = v_n
      // initialise new time step n+1 with values of old time step n
      std::shared_ptr<Core::LinAlg::Vector<double>> dispnp =
          Core::LinAlg::create_vector(*(structure_field()->dof_row_map(0)), true);
      if (step() == 1)
      {
        dispnp->update(1.0, *(structure_field()->dispn()), 0.0);
        vel_ = structure_field()->veln();
      }
      // else: use the velocity of the last converged step

      // start OUTER ITERATION
      while (stopnonliniter == false)
      {
        itnum++;

        // kind of mechanical predictor
        // 1st iteration: get structure variables of old time step (d_n, v_n)
        if (itnum == 1) dispnp->update(1.0, *(structure_field()->dispn()), 0.0);
        // else (itnum>1) use the current solution dispnp of old iteration step

        // store temperature from first solution for convergence check (like in
        // elch_algorithm: use current values)
        tempincnp_->update(1.0, *thermo_field()->tempnp(), 0.0);
        dispincnp_->update(1.0, *structure_field()->dispnp(), 0.0);

        // begin nonlinear solver / outer iteration ***************************

        // ------------------------------------------------- thermo field

        // pass the current displacements and velocities to the thermo field
        apply_struct_coupling_state(dispnp, vel_);

        // prepare time step with coupled variables
        if (itnum == 1) thermo_field()->prepare_time_step();
        // within the nonlinear loop, e.g. itnum>1 call only prepare_step
        else if (itnum != 1)
          thermo_field()->prepare_step();

        // do the nonlinear solve for the time step. All boundary conditions
        // have been set.
        do_thermo_step();

        // ---------------------------------------------- structure field

        // pass the current temperatures to the structure field
        // todo
        // ApplyCouplingState() also calls prepare_step() for prediction
        // with just calculated incremental solutions
        apply_thermo_coupling_state(temp_);

        // prepare time step with coupled variables
        if (itnum == 1) structure_field()->prepare_time_step();

        // solve coupled structural equation
        do_structure_step();

        // extract current displacements
        dispnp->update(1.0, *(structure_field()->dispnp()), 0.0);

        if (quasistatic_)
          vel_ = calc_velocity(*dispnp);
        else
          vel_ = structure_field()->velnp();

        // check convergence of both field for "partitioned scheme"
        stopnonliniter = convergence_check(itnum, itmax_, ittol_);

        // end nonlinear solver / outer iteration ******************************

      }  // end OUTER ITERATION
    }  // end displacement predictor

    // temperature is predictor
    else  // (deformation due to heating)
    {
      // thermal predictor for the coupling iteration outside the loop
      // get temperature of old time step (T_n)
      // T^p_n+1 = T_n
      temp_ = thermo_field()->tempn();

      // start OUTER ITERATION
      while (stopnonliniter == false)
      {
        itnum++;

        // get current temperatures due to solve thermo step, like predictor in FSI
        if (itnum != 1) temp_ = thermo_field()->tempnp();

        // begin nonlinear solver / outer iteration ***************************

        // ---------------------------------------------- structure field

        // pass the current temperatures to the structure field
        // todo
        // ApplyCouplingState() also calls prepare_step() for prediction
        // with just calculated incremental solutions
        apply_thermo_coupling_state(temp_);

        // prepare time step with coupled variables
        if (itnum == 1) structure_field()->prepare_time_step();

        // solve coupled structural equation
        do_structure_step();

        // extract the velocities of the current solution

        // quasistatic to exclude oszillation, use displacements to compute velocities
        if (quasistatic_) vel_ = calc_velocity(*disp_);
        // else use Velnp()

        // ------------------------------------------------- thermo field

        // pass the current displacements and velocities to the thermo field
        apply_struct_coupling_state(disp_, vel_);

        // prepare time step with coupled variables
        if (itnum == 1) thermo_field()->prepare_time_step();
        // within the nonlinear loop, e.g. itnum>1 call only prepare_step
        else if (itnum != 1)
          thermo_field()->prepare_step();

        /// solve coupled thermal system
        /// do the solve for the time step. All boundary conditions have been set.
        do_thermo_step();

        // end nonlinear solver / outer iteration ****************************

        // check convergence of both field for "partitioned scheme"
        stopnonliniter = convergence_check(itnum, itmax_, ittol_);

      }  // end OUTER ITERATION

    }  // temperature predictor

  }  // iterstagg WITHOUT relaxation

  // notation according to Aitken relaxation of FSI Mok, Uli
  // coupling==TSI::SolutionSchemeOverFields::IterStaggAitken
  // 1. calculate the Aitken factor mu
  // 2. calculate the relaxation factor omega = 1-mu
  // 3. T^{i+1} = T^i + omega^{i+1} * ( T^{i+1} - T^i )
  //            = T^{i+1} - mu_^{i+1} * ( T^{i+1} - T^i )
  // 4. limit Aitken factor mu_ for next time step with 1.0
  //
  // another notation of relaxation according to Paper by Irons & Tuck (1969)
  // coupling==TSI::SolutionSchemeOverFields::IterStaggAitkenIrons
  // 1. calculate an relaxation factor mu
  // 2. T^{i+1} = (1 - mu^{i+1}) T^{i+1} + mu^{i+1} T^i
  // 3. limit Aitken factor mu for next time step with maximal value MAXOMEGA
  //
  // coupling==TSI::SolutionSchemeOverFields::IterStaggFixedRel
  // 1. relaxation factor omega = FIXEDOMEGA = const
  // 2. T^{i+1} = omega^{i+1} . T^{i+1} + (1- omega^{i+1}) T^i
  //            = T^i + omega^{i+1} * ( T^{i+1} - T^i )
  else if ((coupling == TSI::SolutionSchemeOverFields::IterStaggAitken) or
           (coupling == TSI::SolutionSchemeOverFields::IterStaggAitkenIrons) or
           (coupling == TSI::SolutionSchemeOverFields::IterStaggFixedRel))
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      std::cout << "Iterative staggered TSI with relaxation" << std::endl;
      std::cout << "Have you set MAXOMEGA to the wished value?" << std::endl;
    }

    // structural predictor
    if (displacementcoupling_)  // (temperature change due to deformation)
    {
      // mechanical predictor for the coupling iteration outside the loop
      // get structure variables of old time step (d_n, v_n)
      // d^p_n+1 = d_n, v^p_n+1 = v_n
      // initialise new time step n+1 with values of old time step n
      std::shared_ptr<Core::LinAlg::Vector<double>> dispnp =
          Core::LinAlg::create_vector(*(structure_field()->dof_row_map(0)), true);
      if (step() == 1)
      {
        dispnp->update(1.0, *(structure_field()->dispn()), 0.0);
      }
      // else: use the velocity of the last converged step

      if ((coupling == TSI::SolutionSchemeOverFields::IterStaggAitken) or
          (coupling == TSI::SolutionSchemeOverFields::IterStaggAitkenIrons))
      {
        // constrain the Aitken factor in the 1st relaxation step of new time
        // step n+1 to maximal value maxomega
        double maxomega = tsidynpart.get<double>("MAXOMEGA");
        // omega_{n+1} = min( omega_n, maxomega ) with omega = 1-mu
        if ((maxomega > 0.0) and (maxomega < (1 - mu_))) mu_ = 1 - maxomega;

        // reset in every new time step
        if (del_ != nullptr)
        {
          del_->put_scalar(1.0e20);
        }
      }

      // start OUTER ITERATION
      while (stopnonliniter == false)
      {
        itnum++;

        // kind of mechanical predictor
        // 1st iteration: get structure variables of old time step (d_n, v_n)
        if (itnum == 1) dispnp->update(1.0, *(structure_field()->dispn()), 0.0);
        // else (itnum>1) use the current solution dispnp of old iteration step

        // store temperature from first solution for convergence check (like in
        // elch_algorithm: use current values)
        // n: time indices, i: Newton iteration
        // calculate increment Delta T^{i+1}_{n+1}
        // 1. iteration step (i=0): Delta T^{n+1}_{1} = T^{n},
        //                          so far no solving has occurred: T^{n+1} = T^{n}
        // i+1. iteration step:     Inc T^{i+1}_{n+1} = T^{i+1}_{n+1} - T^{i}_{n+1}
        //                     fill Inc T^{i+1}_{n+1} = T^{i}_{n+1}
        tempincnp_->update(1.0, *thermo_field()->tempnp(), 0.0);
        dispincnp_->update(1.0, *structure_field()->dispnp(), 0.0);

        // begin nonlinear solver / outer iteration ***************************

        // ------------------------------------------------- thermo field

        // pass the current displacements and velocities to the thermo field
        apply_struct_coupling_state(dispnp, vel_);

        // prepare time step with coupled variables
        if (itnum == 1) thermo_field()->prepare_time_step();
        // within the nonlinear loop, e.g. itnum>1 call only prepare_step
        else if (itnum != 1)
          thermo_field()->prepare_step();

        // do the nonlinear solve for the time step. All boundary conditions
        // have been set.
        do_thermo_step();

        // ---------------------------------------------- structure field

        // pass the current temperatures to the structure field
        // ApplyCouplingState() also calls prepare_step() for prediction
        // with just calculated incremental solutions
        apply_thermo_coupling_state(temp_);

        // prepare time step with coupled variables
        if (itnum == 1) structure_field()->prepare_time_step();

        if (coupling == TSI::SolutionSchemeOverFields::IterStaggFixedRel)
        {
          // get the displacements of the old iteration step d^i_{n+1}
          dispnp->update(1.0, *(structure_field()->dispnp()), 0.0);
        }

        // solve coupled structural equation
        do_structure_step();

        // end nonlinear solver / outer iteration *****************************

        // check convergence of both field for "partitioned scheme"
        stopnonliniter = convergence_check(itnum, itmax_, ittol_);

        // ------------------------------------------------------------
        // if r_{i+1} not converged in convergence_check()
        // --> apply relaxation to displacements

        if (coupling == TSI::SolutionSchemeOverFields::IterStaggFixedRel)
        {
          // ------------------------------------ relax the displacements

          // get fixed relaxation parameter
          double fixedomega = tsidynpart.get<double>("FIXEDOMEGA");
          // fixed relaxation can be applied even in the 1st iteration
          // d^{i+1} = omega^{i+1} . d^{i+1} + (1- omega^{i+1}) d^i
          //         = d^i + omega^{i+1} * ( d^{i+1} - d^i )
          dispnp->update(fixedomega, *dispincnp_, 1.0);

          // ------------------------------------------ end of relaxation
        }
        // dynamic relaxation
        else
        {
          // Aitken factor
          // relaxation can start at i=2, i.e. two vectors inc_{i-1}, inc_i are
          // available (also stated in Diss Uli, p. 82ff)
          // we need vectors from iteration step 1,2 to calculate relaxed step
          // with nu^i_0 = 0.0
          // Irons & Tuck:
          // increment r := old - new

          // 4C:
          // increment inc := new - old
          // nu^{i+1}_{n+1} = nu^i_{n+1} +
          //
          //                     ( r^{i+1}_{n+1} - r^i_{n+1} )^T . ( - r^{i+1}_{n+1} )
          // + (nu^i_{n+1} - 1) . -------------------------------------------------
          //                              | r^{i+1}_{n+1} - r^i_{n+1} |^2

          // initialise increment vector with solution of last iteration (i)
          // update del_ with current residual vector
          // difference of last two solutions
          if (del_ == nullptr)  // first iteration, itnum==1
          {
            del_ = Core::LinAlg::create_vector(*(thermo_field()->dof_row_map(0)), true);
            delhist_ = Core::LinAlg::create_vector(*(thermo_field()->dof_row_map(0)), true);
            del_->put_scalar(1.0e20);
            delhist_->put_scalar(0.0);
          }

          // calculate difference of current (i+1) and old (i) residual vector
          // delhist = ( r^{i+1}_{n+1} - r^i_{n+1} )
          // update history vector old increment r^i_{n+1}
          delhist_->update(1.0, *del_, 0.0);  // r^i_{n+1}
          // den = |r^{i+1} - r^{i}|^2
          double den = 0.0;
          delhist_->norm_2(&den);
          // calculate dot product
          // dot = delhist_ . del_ = ( r^{i+1}_{n+1} - r^i_{n+1} )^T . r^{i+1}_{n+1}
          double top = 0.0;
          delhist_->dot(*del_, &top);

          // Aikten factor
          // nu^{i+1} = nu^i + (nu^i -1) . (r^{i+1} - r^i)^T . (-r^{i+1}) / |r^{i+1} - r^{i}|^2
          // top = ( r^{i+1} - r^i )^T . r^{i+1} --> use -top
          // Uli's implementation: mu_ = mu_ + (mu_ - 1.0) * top / (den*den). with '-' included in
          // top
          mu_ = mu_ + (mu_ - 1) * (-top) / (den * den);

          // relaxation parameter
          // omega^{i+1} = 1- mu^{i+1}
          double omega = 1.0 - mu_;

          // two previous residuals are required (initial guess and result of 1st
          // iteration)
          // --> start relaxation process if two iterative residuals are available
          if (itnum == 1)
          {
            dispnp->update(1.0, *(structure_field()->dispnp()), 0.0);

            // calculate the velocities with the updated/relaxed displacements
            if (quasistatic_) vel_ = calc_velocity(*dispnp);
            // else use Velnp
          }
          else  // (itnum > 1)
          {
            if (coupling == TSI::SolutionSchemeOverFields::IterStaggAitken)
            {
              // relax temperature solution for next iteration step
              // overwrite temp_ with relaxed solution vector
              // d^{i+1} = omega^{i+1} . d^{i+1} + (1- omega^{i+1}) d^i
              //         = d^i + omega^{i+1} * ( d^{i+1} - d^i )
              dispnp->update(omega, *dispincnp_, 1.0);
            }
            // another notation of relaxation according to Paper by Irons & Tuck (1969)
            // 1. calculate an Aitken factor mu == relaxation factor
            // 2. d^{i+1} = (1 - mu^{i+1}) d^{i+1} + mu^{i+1} d^i
            // 3. limit Aitken factor mu for next time step with 0.0
            else if (coupling == TSI::SolutionSchemeOverFields::IterStaggAitkenIrons)
            {
              // relax displacement solution for next iteration step
              // overwrite dispnp with relaxed solution vector
              // d^{i+1} = d^{i+1} - mu^{i+1} * r^{i+1}
              //         = d^{i+1} - mu^{i+1} * ( d^{i+1} - d^i )
              //         = (1 - mu^{i+1}) d^{i+1} + mu^{i+1} d^i
              dispnp->update(1.0, *(structure_field()->dispnp()), (-mu_), *dispincnp_, 0.0);
            }
          }  // itnum>1
        }  // dynamic relaxation

        // velocity has to fit the corresponding displacements
        // --> update velocities according to relaxed displacements
        vel_ = calc_velocity(*dispnp);

        // ----------------------------------------------- end relaxation
      }  // end OUTER ITERATION
    }  // end relax displacement

    // relax the temperatures
    else  // (!displacementcoupling_)
    {
      if (step() == 1)
      {
        // thermal predictor for the coupling iteration outside the loop
        // get temperature of old time step (T_n)
        // T^p_n+1 = T_n
        temp_->update(1.0, *(thermo_field()->tempn()), 0.0);
      }

      if ((coupling == TSI::SolutionSchemeOverFields::IterStaggAitken) or
          (coupling == TSI::SolutionSchemeOverFields::IterStaggAitkenIrons))
      {
        // initial guess for next time step n+1
        // use minimum between omega_n and maxomega as start value for mu_{n+1}^{i=0}
        // in case of doubt use 1.0, meaning that direction of new solution vector
        // tempnp better than old one

        // constrain the Aitken factor in the 1st relaxation step of new time step
        // n+1 to maximal value maxomega
        double maxomega = tsidynpart.get<double>("MAXOMEGA");
        // omega_{n+1} = min( omega_n, maxomega )
        if ((maxomega > 0.0) and (maxomega < 1 - mu_)) mu_ = 1 - maxomega;

        // reset in every new time step
        if (del_ != nullptr)
        {
          del_->put_scalar(1.0e20);
        }
      }

      // start OUTER ITERATION
      while (stopnonliniter == false)
      {
        itnum++;

        // get current temperatures due to solve thermo step, like predictor in FSI
        // 1. iteration: get newest temperatures, i.e. T_{n+1} == T_n
        if (itnum == 1)
        {
          temp_->update(1.0, *(thermo_field()->tempn()), 0.0);
        }
        // else: use solution vector temp_ of old iteration step i

        // store temperature from first solution for convergence check (like in
        // elch_algorithm: use current values)
        // n: time indices, i: Newton iteration
        // calculate increment Delta T^{i+1}_{n+1}
        // 1. iteration step (i=0): Delta T^{n+1}_{1} = T^{n},
        //                          so far no solving has occurred: T^{n+1} = T^{n}
        // i+1. iteration step:     Inc T^{i+1}_{n+1} = T^{i+1}_{n+1} - T^{i}_{n+1}
        //                     fill Inc T^{i+1}_{n+1} = T^{i}_{n+1}
        tempincnp_->update(1.0, *thermo_field()->tempnp(), 0.0);
        dispincnp_->update(1.0, *structure_field()->dispnp(), 0.0);

        // begin nonlinear solver / outer iteration ***************************

        // ---------------------------------------------- structure field

        // pass the current temperatures to the structure field
        // todo
        // ApplyCouplingState() also calls prepare_step() for prediction
        // with just calculated incremental solutions
        // structure_field()->discretization()->set_state(1,"temperature",temp_);
        apply_thermo_coupling_state(temp_);

        // prepare time step with coupled variables
        if (itnum == 1) structure_field()->prepare_time_step();

        // solve coupled structural equation
        do_structure_step();

        // extract the velocities of the current solution
        // the displacement -> velocity conversion
        if (quasistatic_) vel_ = calc_velocity(*disp_);
        // else use vel_

        // ------------------------------------------------- thermo field

        // pass the current displacements and velocities to the thermo field
        apply_struct_coupling_state(disp_, vel_);

        // prepare time step with coupled variables
        if (itnum == 1) thermo_field()->prepare_time_step();
        // within the nonlinear loop, e.g. itnum>1 call only prepare_step
        else if (itnum != 1)
          thermo_field()->prepare_step();

        if (coupling == TSI::SolutionSchemeOverFields::IterStaggFixedRel)
        {
          // get temperature solution of old iteration step T_{n+1}^i
          temp_->update(1.0, *(thermo_field()->tempnp()), 0.0);
        }

        /// solve coupled thermal system
        /// do the solve for the time step. All boundary conditions have been set.
        do_thermo_step();

        // end nonlinear solver / outer iteration *****************************

        // check convergence of both field for "partitioned scheme"
        stopnonliniter = convergence_check(itnum, itmax_, ittol_);
        // --> update of tempincnp_ is done in convergence_check()

        // --------------------------------------------- start relaxation

        if (coupling == TSI::SolutionSchemeOverFields::IterStaggFixedRel)
        {
          // ------------------------------------- relax the temperatures

          // get fixed relaxation parameter
          double fixedomega = tsidynpart.get<double>("FIXEDOMEGA");
          // fixed relaxation can be applied even in the 1st iteration
          // T^{i+1} = omega^{i+1} . T^{i+1} + (1- omega^{i+1}) T^i
          //         = T^i + omega^{i+1} . ( T^{i+1} - T^i )
          temp_->update(fixedomega, *tempincnp_, 1.0);

          // ------------------------------------------ end of relaxation
        }
        // dynamic relaxation
        else
        {
          // ------------------------------------------------------------
          // if r_{i+1} not converged in convergence_check()
          // --> apply Aitken relaxation to temperatures
          // as implemented in FSI by Irons and Tuck (1969)

          // Aitken factor
          // relaxation can start at i=2, i.e. two vectors inc_{i-1}, inc_i are
          // available (also stated in Diss Uli, p. 82ff)
          // we need vectors from iteration step 1,2 to calculate relaxed step
          // with mu^i_0 = 0.0
          // Irons & Tuck:
          // increment r := old - new

          // 4C:
          // increment inc := new - old
          // mu^{i+1}_{n+1} = mu^i_{n+1} +
          //
          //                     ( r^{i+1}_{n+1} - r^i_{n+1} )^T . ( - r^{i+1}_{n+1} )
          // + (mu^i_{n+1} - 1) . -------------------------------------------------
          //                              | r^{i+1}_{n+1} - r^i_{n+1} |^2

          // initialise increment vector with solution of last iteration (i)
          // update del_ with current residual vector
          // difference of last two solutions
          if (del_ == nullptr)  // first iteration, itnum==1
          {
            del_ = Core::LinAlg::create_vector(*(thermo_field()->dof_row_map(0)), true);
            delhist_ = Core::LinAlg::create_vector(*(thermo_field()->dof_row_map(0)), true);
            del_->put_scalar(1.0e20);
            delhist_->put_scalar(0.0);
          }

          // calculate difference of current (i+1) and old (i) residual vector
          // delhist = ( r^{i+1}_{n+1} - r^i_{n+1} )
          // update history vector old increment r^i_{n+1}
          delhist_->update(1.0, *del_, 0.0);           // r^i_{n+1}
          delhist_->update(1.0, *tempincnp_, (-1.0));  // update r^{i+1}_{n+1}

          // del_ = r^{i+1}_{n+1} = T^{i+1}_{n+1} - T^{i}_{n+1}
          del_->update(1.0, *tempincnp_, 0.0);
          // den = |r^{i+1} - r^{i}|^2
          double den = 0.0;
          delhist_->norm_2(&den);
          // calculate dot product
          // dot = delhist_ . del_ = ( r^{i+1}_{n+1} - r^i_{n+1} )^T . r^{i+1}_{n+1}
          double top = 0.0;
          delhist_->dot(*del_, &top);

          // mu_: Aikten factor in Mok's version
          // mu_: relaxation parameter in Irons & Tuck
          // mu^{i+1} = mu^i + (mu^i -1) . (r^{i+1} - r^i)^T . (-r^{i+1}) / |r^{i+1} - r^{i}|^2
          // top = ( r^{i+1} - r^i )^T . r^{i+1} --> use -top
          // Uli's implementation: mu_ = mu_ + (mu_ - 1.0) * top / (den*den). with '-' included in
          // top
          mu_ = mu_ + (mu_ - 1) * (-top) / (den * den);

          // two previous residuals are required (initial guess and result of 1st
          // iteration)
          // --> start relaxation process if two iterative residuals are available
          if (itnum == 1)
            // get temperature T_{n+1}^{i=1}
            temp_->update(1.0, *(thermo_field()->tempnp()), 0.0);
          else  // (itnum > 1)
          {
            if (coupling == TSI::SolutionSchemeOverFields::IterStaggAitken)
            {
              // relaxation parameter
              // omega^{i+1} = 1- mu^{i+1}
              double omega = 1.0 - mu_;

              // relax temperature solution for next iteration step
              // overwrite temp_ with relaxed solution vector
              // T^{i+1} = omega^{i+1} . T^{i+1} + (1- omega^{i+1}) T^i
              //         = T^i + omega^{i+1} * ( T^{i+1} - T^i )
              temp_->update(omega, *tempincnp_, 1.0);
            }
            else if (coupling == TSI::SolutionSchemeOverFields::IterStaggAitkenIrons)
            {
              // relax displacement solution for next iteration step
              // overwrite tempnp with relaxed solution vector
              // T^{i+1} = T^{i+1} - mu^{i+1} * r^{i+1}
              //         = T^{i+1} - mu^{i+1} * ( T^{i+1} - T^i )
              //         = (1 - mu^{i+1}) T^{i+1} + mu^{i+1} T^i
              // --> Irons T^{i+1} = T^{i+1} + mu^{i+1} * DEL^{i+1} with DEL = T^i - T^{i+1}
              temp_->update(1.0, *(thermo_field()->tempnp()), (-mu_), *tempincnp_, 0.0);
            }
          }  // itnum > 1
        }  // dynamic relaxation

        // ----------------------------------------------- end relaxation

      }  // end OUTER ITERATION
    }  // relax temperatures
  }  // iterative staggered TSI with relaxation
  return;

}  // outer_iteration_loop()


/*----------------------------------------------------------------------*
 | solve the structural system (protected)                    dano 12/10 |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::do_structure_step()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout << "************************\n";
    std::cout << "    STRUCTURE SOLVER    \n";
    std::cout << "************************\n";
  }

  /// solve structural system
  // do the nonlinear solve for the time step. All boundary conditions have
  // been set.
  structure_field()->solve();

  return;

}  // do_structure_step()


/*----------------------------------------------------------------------*
 | solve the thermal system (protected)                       dano 12/10 |
 | coupling of displacements to thermo field and extract current        |
 | displacements needed for coupling to thermo field                    |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::do_thermo_step()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout << "*********************\n";
    std::cout << "    THERMO SOLVER    \n";
    std::cout << "*********************\n";
  }

  /// solve thermal system
  // do the solve for the time step. All boundary conditions have
  // been set.
  thermo_field()->solve();

  return;

}  // do_thermo_step()


/*----------------------------------------------------------------------*
 | convergence check for both fields (thermo & structure)    dano 02/10 |
 | originally by vg 01/09                                               |
 *----------------------------------------------------------------------*/
bool TSI::Partitioned::convergence_check(int itnum, const int itmax, const double ittol)
{
  // convergence check based on the temperature increment
  bool stopnonliniter = false;

  //    | temperature increment |_2
  //  -------------------------------- < Tolerance
  //     | temperature_n+1 |_2

  // variables to save different L2 - Norms
  // define L2-norm of incremental temperature and temperature
  // here: only the temperature field is checked for convergence!!!
  double tempincnorm_L2(0.0);
  double tempnorm_L2(0.0);
  double dispincnorm_L2(0.0);
  double dispnorm_L2(0.0);

  // build the current temperature increment Inc T^{i+1} with Newton iteration index i
  // \f Delta T^{i+1} = Inc T^{i+1} = T^{i+1} - T^{i}  \f
  tempincnp_->update(1.0, *(thermo_field()->tempnp()), -1.0);
  dispincnp_->update(1.0, *(structure_field()->dispnp()), -1.0);

  // for convergence test choose the last converged solution vector T_n/D_n,
  // be careful to check the convergence with the current, NOT yet converged values n+1
  // if a solution n+1 is highly difficult to find, the norm can oscillate

  // build the L2-norm of the increments and the old solution vectors
  tempincnp_->norm_2(&tempincnorm_L2);
  thermo_field()->tempn()->norm_2(&tempnorm_L2);
  dispincnp_->norm_2(&dispincnorm_L2);
  structure_field()->dispn()->norm_2(&dispnorm_L2);

  // care for the case that there is (almost) zero temperature
  // (usually not required for temperature)
  if (tempnorm_L2 < 1e-6) tempnorm_L2 = 1.0;
  if (dispnorm_L2 < 1e-6) dispnorm_L2 = 1.0;

  // converged
  // check convergence of the outer loop with a relative norm
  // norm of the increment with respect to the norm of the variables itself: use
  // the last converged solution
  // residual increments
  switch (normtypeinc_)
  {
    // default check:
    case TSI::convnorm_abs:
    {
      // print the incremental based convergence check to the screen
      // test here increment
      if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
          (step() % print_screen_every() == 0))
      {
        std::cout << "\n";
        std::cout << "*****************************************************************************"
                     "******\n";
        std::cout << "    OUTER ITERATION STEP    \n";
        std::cout << "*****************************************************************************"
                     "******\n";
        printf(
            "+--------------+------------------------+--------------------+--------------------+"
            "\n");
        printf(
            "|-  step/max  -|-  tol      [norm]     -|--  temp-inc      --|--  disp-inc      "
            "--|\n");
        printf("|   %3d/%3d    |  %10.3E[L_2 ]      | %10.3E         | %10.3E         |", itnum,
            itmax, ittol, tempincnorm_L2, dispincnorm_L2);
        printf("\n");
        printf(
            "+--------------+------------------------+--------------------+--------------------+"
            "\n");
      }

      // converged
      // check convergence of the outer loop with a relative norm
      // norm of the increment with respect to the norm of the variables itself: use the last
      // converged solution
      stopnonliniter = ((tempincnorm_L2 <= ittol) and (dispincnorm_L2 <= ittol));
      if ((stopnonliniter == true) and print_screen_every() and
          (Core::Communication::my_mpi_rank(get_comm()) == 0) and
          (step() % print_screen_every() == 0))
      {
        printf("\n");
        printf(
            "|  Outer Iteration loop converged after iteration %3d/%3d !                       |\n",
            itnum, itmax);
        printf(
            "+--------------+------------------------+--------------------+--------------------+"
            "\n");
      }

      // warn if itemax is reached without convergence, but proceed to next
      // timestep
      if ((itnum == itmax) and ((tempincnorm_L2 > ittol) or (dispincnorm_L2 > ittol)))
      {
        stopnonliniter = true;
        if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
            (step() % print_screen_every() == 0))
        {
          printf(
              "|     >>>>>> not converged in itemax steps!                                       "
              "|\n");
          printf(
              "+--------------+------------------------+--------------------+--------------------+"
              "\n");
          printf("\n");
          printf("\n");
        }
      }
    }  // TSI::convnorm_abs
    break;

    case TSI::convnorm_rel:
    {
      // print the incremental based convergence check to the screen
      // test here increment/variable
      if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
          (step() % print_screen_every() == 0))
      {
        std::cout << "\n";
        std::cout << "*****************************************************************************"
                     "******\n";
        std::cout << "    OUTER ITERATION STEP    \n";
        std::cout << "*****************************************************************************"
                     "******\n";
        printf(
            "+--------------+------------------------+--------------------+--------------------+"
            "\n");
        printf(
            "|-  step/max  -|-  tol      [norm]     -|--  temp-inc/temp --|--  disp-inc/disp "
            "--|\n");
        printf("|   %3d/%3d    |  %10.3E[L_2 ]      | %10.3E         | %10.3E         |", itnum,
            itmax, ittol, tempincnorm_L2 / tempnorm_L2, dispincnorm_L2 / dispnorm_L2);
        printf("\n");
        printf(
            "+--------------+------------------------+--------------------+--------------------+"
            "\n");
      }

      stopnonliniter =
          ((tempincnorm_L2 / tempnorm_L2 <= ittol) and (dispincnorm_L2 / dispnorm_L2 <= ittol));
      if ((stopnonliniter == true) and (Core::Communication::my_mpi_rank(get_comm()) == 0) and
          print_screen_every() and (step() % print_screen_every() == 0))
      {
        printf("\n");
        printf(
            "|  Outer Iteration loop converged after iteration %3d/%3d !                       |\n",
            itnum, itmax);
        printf(
            "+--------------+------------------------+--------------------+--------------------+"
            "\n");
      }

      // warn if itemax is reached without convergence, but proceed to next
      // timestep
      if ((itnum == itmax) and
          ((tempincnorm_L2 / tempnorm_L2 > ittol) or (dispincnorm_L2 / dispnorm_L2 > ittol)))
      {
        stopnonliniter = true;
        if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
            (step() % print_screen_every() == 0))
        {
          printf(
              "|     >>>>>> not converged in itemax steps!                                       "
              "|\n");
          printf(
              "+--------------+------------------------+--------------------+--------------------+"
              "\n");
          printf("\n");
          printf("\n");
        }
      }
    }  // TSI::convnorm_rel
    break;

    case TSI::convnorm_mix:
    default:
      FOUR_C_THROW("Cannot check for convergence of residual values!");
      break;
  }

  return stopnonliniter;

}  // convergence_check()


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::prepare_output()
{
  // prepare output (i.e. calculate stresses, strains, energies)
  constexpr bool force_prepare = false;
  structure_field()->prepare_output(force_prepare);

  // reset states
  structure_field()->discretization()->clear_state(true);

}  // prepare_output()


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::prepare_contact_strategy()
{
  TSI::Algorithm::prepare_contact_strategy();
}  // prepare_contact_strategy()


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Partitioned::update()
{
  structure_field()->update();
  thermo_field()->update();
  if (contact_strategy_lagrange_ != nullptr)
    contact_strategy_lagrange_->update((structure_field()->dispnp()));
}

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
