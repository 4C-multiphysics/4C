// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_stru_multi_microstatic.hpp"

#include "4C_comm_utils.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_material.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_structure_aux.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  ctor (public)|
 *----------------------------------------------------------------------*/
MultiScale::MicroStatic::MicroStatic(const int microdisnum, const double V0)
    : microdisnum_(microdisnum), V0_(V0)
{
  // -------------------------------------------------------------------
  // access the discretization
  // -------------------------------------------------------------------
  discret_ = Global::Problem::instance(microdisnum_)->get_dis("structure");

  // set degrees of freedom in the discretization
  if (!discret_->filled()) discret_->fill_complete();

  // -------------------------------------------------------------------
  // set some pointers and variables
  // -------------------------------------------------------------------
  // time step size etc. need to be consistent in both input files, we
  // choose to use the ones defined in the macroscale input file
  // while other parameters (like output options, convergence checks)
  // can be used individually from the microscale input file
  const Teuchos::ParameterList& sdyn_micro =
      Global::Problem::instance(microdisnum_)->structural_dynamic_params();
  const Teuchos::ParameterList& sdyn_macro =
      Global::Problem::instance()->structural_dynamic_params();

  // i/o options should be read from the corresponding micro-file
  const Teuchos::ParameterList& ioflags = Global::Problem::instance(microdisnum_)->io_params();

  // -------------------------------------------------------------------
  // create a solver
  // -------------------------------------------------------------------
  // get the solver number used for structural solver
  const int linsolvernumber = sdyn_micro.get<int>("LINEAR_SOLVER");
  // check if the structural solver has a valid solver number
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for structural field. Please set LINEAR_SOLVER in STRUCTURAL "
        "DYNAMIC to a valid number!");

  solver_ = std::make_shared<Core::LinAlg::Solver>(
      Global::Problem::instance(microdisnum_)->solver_params(linsolvernumber), discret_->get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));
  discret_->compute_null_space_if_necessary(solver_->params());

  auto pred = Teuchos::getIntegralValue<Inpar::Solid::PredEnum>(sdyn_micro, "PREDICT");
  pred_ = pred;
  combdisifres_ =
      Teuchos::getIntegralValue<Inpar::Solid::BinaryOp>(sdyn_micro, "NORMCOMBI_RESFDISP");
  normtypedisi_ = Teuchos::getIntegralValue<Inpar::Solid::ConvNorm>(sdyn_micro, "NORM_DISP");
  normtypefres_ = Teuchos::getIntegralValue<Inpar::Solid::ConvNorm>(sdyn_micro, "NORM_RESF");
  auto iternorm = Teuchos::getIntegralValue<Inpar::Solid::VectorNorm>(sdyn_micro, "ITERNORM");
  iternorm_ = iternorm;

  dt_ = sdyn_macro.get<double>("TIMESTEP");
  // broadcast important data that must be consistent on macro and micro scale (master and
  // supporting procs)
  Core::Communication::broadcast(&dt_, 1, 0, discret_->get_comm());
  time_ = 0.0;
  timen_ = time_ + dt_;
  step_ = 0;
  stepn_ = step_ + 1;
  numstep_ = sdyn_macro.get<int>("NUMSTEP");
  maxiter_ = sdyn_micro.get<int>("MAXITER");
  numiter_ = -1;

  tolfres_ = sdyn_micro.get<double>("TOLRES");
  toldisi_ = sdyn_micro.get<double>("TOLDISP");
  printscreen_ = (ioflags.get<int>("STDOUTEVERY"));


  restart_ = Global::Problem::instance()->restart();
  restartevry_ = sdyn_macro.get<int>("RESTARTEVERY");
  iodisp_ = ioflags.get<bool>("STRUCT_DISP");
  resevrydisp_ = sdyn_micro.get<int>("RESULTSEVERY");
  auto iostress = Teuchos::getIntegralValue<Inpar::Solid::StressType>(ioflags, "STRUCT_STRESS");
  iostress_ = iostress;
  resevrystrs_ = sdyn_micro.get<int>("RESULTSEVERY");
  auto iostrain = Teuchos::getIntegralValue<Inpar::Solid::StrainType>(ioflags, "STRUCT_STRAIN");
  iostrain_ = iostrain;
  auto ioplstrain =
      Teuchos::getIntegralValue<Inpar::Solid::StrainType>(ioflags, "STRUCT_PLASTIC_STRAIN");
  ioplstrain_ = ioplstrain;
  iosurfactant_ = ioflags.get<bool>("STRUCT_SURFACTANT");

  isadapttol_ = (sdyn_micro.get<bool>("ADAPTCONV"));
  adaptolbetter_ = sdyn_micro.get<double>("ADAPTCONV_BETTER");

  // broadcast important data that must be consistent on macro and micro scale (master and
  // supporting procs)
  Core::Communication::broadcast(&numstep_, 1, 0, discret_->get_comm());
  Core::Communication::broadcast(&restart_, 1, 0, discret_->get_comm());
  Core::Communication::broadcast(&restartevry_, 1, 0, discret_->get_comm());

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  // -------------------------------------------------------------------
  if (!discret_->filled()) discret_->fill_complete();
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
  myrank_ = Core::Communication::my_mpi_rank(discret_->get_comm());

  // -------------------------------------------------------------------
  // create empty matrices
  // -------------------------------------------------------------------
  stiff_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 81, true, true);

  // -------------------------------------------------------------------
  // create empty vectors
  // -------------------------------------------------------------------
  // a zero vector of full length
  zeros_ = Core::LinAlg::create_vector(*dofrowmap, true);
  // vector of full length; for each component
  //                /  1   i-th DOF is supported, ie Dirichlet BC
  //    vector_i =  <
  //                \  0   i-th DOF is free
  dirichtoggle_ = Core::LinAlg::create_vector(*dofrowmap, true);
  // opposite of dirichtoggle vector, ie for each component
  //                /  0   i-th DOF is supported, ie Dirichlet BC
  //    vector_i =  <
  //                \  1   i-th DOF is free
  invtoggle_ = Core::LinAlg::create_vector(*dofrowmap, false);

  // displacements D_{n+1} at new time
  disn_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // displacements D_{n+1} at old time
  dis_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // iterative displacement increments IncD_{n+1}
  // also known as residual displacements
  disi_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // internal force vector F_int
  fintn_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // dynamic force residual
  // also known as out-of-balance-force
  fresn_ = Core::LinAlg::create_vector(*dofrowmap, false);

  // -------------------------------------------------------------------
  // create "empty" EAS history map
  //
  // -------------------------------------------------------------------
  {
    lastalpha_ =
        std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();
    oldalpha_ = std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();
    oldfeas_ = std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();
    oldKaainv_ =
        std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();
    oldKda_ = std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();
  }

  // -------------------------------------------------------------------
  // call elements to calculate stiffness and mass
  // -------------------------------------------------------------------
  {
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    // action for elements
    p.set("action", "calc_struct_nlnstiff");
    // other parameters that might be needed by the elements
    p.set("total time", timen_);
    p.set("delta time", dt_);
    // set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("residual displacement", *zeros_);
    discret_->set_state("displacement", *dis_);

    discret_->evaluate(p, stiff_, nullptr, fintn_, nullptr, nullptr);
    discret_->clear_state();
  }

  // Determine dirichtoggle_ and its inverse since boundary conditions for
  // microscale simulations are due to the MicroBoundary condition
  // (and not Dirichlet BC)

  MultiScale::MicroStatic::determine_toggle();
  MultiScale::MicroStatic::set_up_homogenization();

  // reaction force vector at different times
  freactn_ = Core::LinAlg::create_vector(*pdof_, true);

  //----------------------- compute an inverse of the dirichtoggle vector
  invtoggle_->put_scalar(1.0);
  invtoggle_->update(-1.0, *dirichtoggle_, 1.0);

  // ------------------------- Calculate initial volume and density of microstructure
  // the macroscopic density has to be averaged over the entire
  // microstructural reference volume

  double my_micro_discretization_volume = 0.0;
  double my_micro_discretization_density_integration = 0.0;

  // create the parameters for the discretization
  discret_->set_state("displacement", *dis_);
  Core::Elements::LocationArray la(discret_->num_dof_sets());
  for (const auto* ele : discret_->my_row_element_range())
  {
    ele->location_vector(*discret_, la);

    const auto* solid_ele = dynamic_cast<const Discret::Elements::Solid*>(ele);
    FOUR_C_ASSERT_ALWAYS(solid_ele,
        "Multiscale simulations are currently only possible with the new solid elements");

    solid_ele->for_each_gauss_point(*discret_, la[0].lm_,
        [&](Mat::So3Material& solid_material, double integration_factor, int gp)
        {
          // integrate volume
          my_micro_discretization_volume += integration_factor;

          // integrate density
          my_micro_discretization_density_integration +=
              solid_material.density(gp) * integration_factor;
        });
  }
  discret_->clear_state();

  // compute volume of all elements
  Core::Communication::sum_all(&my_micro_discretization_volume, &V0_, 1, discret_->get_comm());

  // compute density of all elements
  double micro_discretization_density_integration = 0.0;
  Core::Communication::sum_all(&my_micro_discretization_density_integration,
      &micro_discretization_density_integration, 1, discret_->get_comm());

  density_ = micro_discretization_density_integration / V0_;

  FOUR_C_ASSERT_ALWAYS(
      density_ > 0, "Density determined from homogenization procedure must be larger than zero!");
}  // MultiScale::MicroStatic::MicroStatic


void MultiScale::MicroStatic::predictor(Core::LinAlg::Matrix<3, 3>* defgrd)
{
  if (pred_ == Inpar::Solid::pred_constdis)
    predict_const_dis(defgrd);
  else if (pred_ == Inpar::Solid::pred_tangdis)
    predict_tang_dis(defgrd);
  else
    FOUR_C_THROW("requested predictor not implemented on the micro-scale");
  return;
}


/*----------------------------------------------------------------------*
 |  do predictor step (public)                               mwgee 03/07|
 *----------------------------------------------------------------------*/
void MultiScale::MicroStatic::predict_const_dis(Core::LinAlg::Matrix<3, 3>* defgrd)
{
  // apply new displacements at DBCs -> this has to be done with the
  // mid-displacements since the given macroscopic deformation
  // gradient is evaluated at the mid-point!
  {
    // disn then also holds prescribed new dirichlet displacements
    evaluate_micro_bc(defgrd, *disn_);
    discret_->clear_state();
  }

  //--------------------------------- set EAS internal data if necessary

  // this has to be done only once since the elements will remember
  // their EAS data until the end of the microscale simulation
  // (end of macroscopic iteration step)
  set_eas_data();

  //------------- eval fint at interpolated state, eval stiffness matrix
  {
    // zero out stiffness
    stiff_->zero();
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    // action for elements
    p.set("action", "calc_struct_nlnstiff");
    // other parameters that might be needed by the elements
    p.set("total time", timen_);
    p.set("delta time", dt_);
    // set vector values needed by elements
    discret_->clear_state();
    disi_->put_scalar(0.0);
    discret_->set_state("residual displacement", *disi_);
    discret_->set_state("displacement", *disn_);
    fintn_->put_scalar(0.0);  // initialise internal force vector

    discret_->evaluate(p, stiff_, nullptr, fintn_, nullptr, nullptr);
    discret_->clear_state();

    // complete stiffness matrix
    stiff_->complete();

    // set norm of displacement increments
    normdisi_ = 1.0e6;
  }

  //-------------------------------------------- compute residual forces
  // add static mid-balance
  fresn_->update(-1.0, *fintn_, 0.0);

  // extract reaction forces
  int err = freactn_->import(*fresn_, *importp_, Insert);
  if (err)
    FOUR_C_THROW(
        "Importing reaction forces of prescribed dofs using importer returned err={}", err);

  // blank residual at DOFs on Dirichlet BC
  Core::LinAlg::Vector<double> fresncopy(*fresn_);
  fresn_->multiply(1.0, *invtoggle_, fresncopy, 0.0);

  // store norm of residual
  normfres_ = Solid::calculate_vector_norm(iternorm_, *fresn_);

  return;
}  // MultiScale::MicroStatic::Predictor()


/*----------------------------------------------------------------------*
 |  do predictor step (public)                                  lw 01/09|
 *----------------------------------------------------------------------*/
void MultiScale::MicroStatic::predict_tang_dis(Core::LinAlg::Matrix<3, 3>* defgrd)
{
  // for displacement increments on Dirichlet boundary
  std::shared_ptr<Core::LinAlg::Vector<double>> dbcinc =
      Core::LinAlg::create_vector(*(discret_->dof_row_map()), true);

  // copy last converged displacements
  dbcinc->update(1.0, *disn_, 0.0);

  // apply new displacements at DBCs -> this has to be done with the
  // mid-displacements since the given macroscopic deformation
  // gradient is evaluated at the mid-point!
  {
    // dbcinc then also holds prescribed new dirichlet displacements
    evaluate_micro_bc(defgrd, *dbcinc);
    discret_->clear_state();
  }

  // subtract the displacements of the last converged step
  // DBC-DOFs hold increments of current step
  // free-DOFs hold zeros
  dbcinc->update(-1.0, *disn_, 1.0);

  //--------------------------------- set EAS internal data if necessary

  // this has to be done only once since the elements will remember
  // their EAS data until the end of the microscale simulation
  // (end of macroscopic iteration step)
  set_eas_data();

  //------------- eval fint at interpolated state, eval stiffness matrix
  {
    // zero out stiffness
    stiff_->zero();
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    // action for elements
    p.set("action", "calc_struct_nlnstiff");
    // other parameters that might be needed by the elements
    p.set("total time", timen_);
    p.set("delta time", dt_);
    // set vector values needed by elements
    discret_->clear_state();
    disi_->put_scalar(0.0);
    discret_->set_state("residual displacement", *disi_);
    discret_->set_state("displacement", *disn_);
    fintn_->put_scalar(0.0);  // initialise internal force vector

    discret_->evaluate(p, stiff_, nullptr, fintn_, nullptr, nullptr);
    discret_->clear_state();
  }

  stiff_->complete();

  //-------------------------------------------- compute residual forces
  // add static mid-balance
  fresn_->update(-1.0, *fintn_, 0.0);

  // add linear reaction forces to residual
  {
    // linear reactions
    std::shared_ptr<Core::LinAlg::Vector<double>> freact =
        Core::LinAlg::create_vector(*(discret_->dof_row_map()), true);
    stiff_->multiply(false, *dbcinc, *freact);

    // add linear reaction forces due to prescribed Dirichlet BCs
    fresn_->update(-1.0, *freact, 1.0);
  }

  // blank residual at DOFs on Dirichlet BC
  {
    Core::LinAlg::Vector<double> fresncopy(*fresn_);
    fresn_->multiply(1.0, *invtoggle_, fresncopy, 0.0);
  }

  // apply Dirichlet BCs to system of equations
  disi_->put_scalar(0.0);
  stiff_->complete();
  Core::LinAlg::apply_dirichlet_to_system(*stiff_, *disi_, *fresn_, *zeros_, *dirichtoggle_);

  // solve for disi_
  // Solve K_Teffdyn . IncD = -R  ===>  IncD_{n+1}
  solver_->reset();
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->solve(stiff_->epetra_operator(), disi_, fresn_, solver_params);
  solver_->reset();

  // store norm of displacement increments
  normdisi_ = Solid::calculate_vector_norm(iternorm_, *disi_);

  //---------------------------------- update mid configuration values
  // set Dirichlet increments in displacement increments
  disi_->update(1.0, *dbcinc, 1.0);

  // displacements
  disn_->update(1.0, *disi_, 1.0);

  // reset anything that needs to be reset at the element level

  // strictly speaking, this (as well as the resetting of disi) is not
  // mandatory here, we do it just to be in line with the classical
  // time intgrator sti. there tangdis is assumed to be a predictor only, no
  // update of EAS parameters etc is desired. perhaps this might be
  // changed when speed should be optimized later on.
  {
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    p.set("action", "calc_struct_reset_istep");
    // go to elements
    discret_->evaluate(p, nullptr, nullptr, nullptr, nullptr, nullptr);
    discret_->clear_state();
  }

  //------------- eval fint at interpolated state, eval stiffness matrix
  {
    // zero out stiffness
    stiff_->zero();
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    // action for elements
    p.set("action", "calc_struct_nlnstiff");
    // other parameters that might be needed by the elements
    p.set("total time", timen_);
    p.set("delta time", dt_);
    // set vector values needed by elements
    discret_->clear_state();
    disi_->put_scalar(0.0);
    discret_->set_state("residual displacement", *disi_);
    discret_->set_state("displacement", *disn_);
    fintn_->put_scalar(0.0);  // initialise internal force vector

    discret_->evaluate(p, stiff_, nullptr, fintn_, nullptr, nullptr);
    discret_->clear_state();
  }

  //-------------------------------------------- compute residual forces
  // add static mid-balance
  fresn_->update(-1.0, *fintn_, 0.0);

  // extract reaction forces
  int err = freactn_->import(*fresn_, *importp_, Insert);
  if (err)
    FOUR_C_THROW(
        "Importing reaction forces of prescribed dofs using importer returned err={}", err);

  // blank residual at DOFs on Dirichlet BC
  Core::LinAlg::Vector<double> fresncopy(*fresn_);
  fresn_->multiply(1.0, *invtoggle_, fresncopy, 0.0);

  // store norm of residual
  normfres_ = Solid::calculate_vector_norm(iternorm_, *fresn_);

  return;
}

/*----------------------------------------------------------------------*
 |  do Newton iteration (public)                             mwgee 03/07|
 *----------------------------------------------------------------------*/
void MultiScale::MicroStatic::full_newton()
{
  //=================================================== equilibrium loop
  numiter_ = 0;

  // if TangDis-Predictor is employed, the number of iterations needs
  // to be increased by one, since it involves already one solution of
  // the non-linear system!
  if (pred_ == Inpar::Solid::pred_tangdis) numiter_++;

  // store norms of old displacements and maximum of norms of
  // internal, external and inertial forces (needed for relative convergence
  // check)
  calc_ref_norms();

  Teuchos::Time timer("", true);
  timer.reset();

  while (!converged() && numiter_ <= maxiter_)
  {
    //----------------------- apply dirichlet BCs to system of equations
    disi_->put_scalar(0.0);  // Useful? depends on solver and more

    Core::LinAlg::apply_dirichlet_to_system(*stiff_, *disi_, *fresn_, *zeros_, *dirichtoggle_);

    //--------------------------------------------------- solve for disi
    // Solve K_Teffdyn . IncD = -R  ===>  IncD_{n+1}
    Core::LinAlg::SolverParams solver_params;
    if (isadapttol_ && numiter_)
    {
      solver_params.nonlin_tolerance = tolfres_;
      solver_params.nonlin_residual = normfres_;
      solver_params.lin_tol_better = adaptolbetter_;
    }
    solver_params.refactor = true;
    solver_params.reset = numiter_ == 0;
    solver_->solve(stiff_->epetra_operator(), disi_, fresn_, solver_params);
    solver_->reset_tolerance();

    //---------------------------------- update mid configuration values
    // displacements
    disn_->update(1.0, *disi_, 1.0);

    //---------------------------- compute internal forces and stiffness
    {
      // zero out stiffness
      stiff_->zero();
      // create the parameters for the discretization
      Teuchos::ParameterList p;
      // action for elements
      p.set("action", "calc_struct_nlnstiff");
      // other parameters that might be needed by the elements
      p.set("total time", timen_);
      p.set("delta time", dt_);
      // set vector values needed by elements
      discret_->clear_state();
      // we do not need to scale disi_ here with 1-alphaf (cf. strugenalpha), since
      // everything on the microscale "lives" at the pseudo generalized midpoint
      // -> we solve our quasi-static problem there and only update data to the "end"
      // of the time step after having finished a macroscopic dt
      discret_->set_state("residual displacement", *disi_);
      discret_->set_state("displacement", *disn_);
      fintn_->put_scalar(0.0);  // initialise internal force vector

      discret_->evaluate(p, stiff_, nullptr, fintn_, nullptr, nullptr);
      discret_->clear_state();
    }

    // complete stiffness matrix
    stiff_->complete();

    //------------------------------------------ compute residual forces
    // add static mid-balance
    fresn_->update(-1.0, *fintn_, 0.0);

    // extract reaction forces
    int err = freactn_->import(*fresn_, *importp_, Insert);
    if (err)
      FOUR_C_THROW(
          "Importing reaction forces of prescribed dofs using importer returned err={}", err);

    // blank residual DOFs which are on Dirichlet BC
    Core::LinAlg::Vector<double> fresncopy(*fresn_);
    fresn_->multiply(1.0, *invtoggle_, fresncopy, 0.0);

    //---------------------------------------------- build residual norm
    normdisi_ = Solid::calculate_vector_norm(iternorm_, *disi_);

    normfres_ = Solid::calculate_vector_norm(iternorm_, *fresn_);

    //--------------------------------- increment equilibrium loop index
    ++numiter_;
  }
  //============================================= end equilibrium loop

  //-------------------------------- test whether max iterations was hit
  if (numiter_ >= maxiter_)
  {
    FOUR_C_THROW("Newton unconverged in {} iterations", numiter_);
  }

  return;
}  // MultiScale::MicroStatic::FullNewton()


/*----------------------------------------------------------------------*
 |  "prepare" output (public)                                   ly 09/11|
 *----------------------------------------------------------------------*/
void MultiScale::MicroStatic::prepare_output()
{
  if (resevrystrs_ and !(stepn_ % resevrystrs_) and iostress_ != Inpar::Solid::stress_none)
  {
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    // action for elements
    p.set("action", "calc_struct_stress");
    // other parameters that might be needed by the elements
    p.set("total time", timen_);
    p.set("delta time", dt_);
    p.set("stress", stress_);
    p.set("strain", strain_);
    p.set("plstrain", plstrain_);
    p.set<Inpar::Solid::StressType>("iostress", iostress_);
    p.set<Inpar::Solid::StrainType>("iostrain", iostrain_);
    p.set<Inpar::Solid::StrainType>("ioplstrain", ioplstrain_);
    // set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("residual displacement", *zeros_);
    discret_->set_state("displacement", *disn_);
    discret_->evaluate(p, nullptr, nullptr, nullptr, nullptr, nullptr);
    discret_->clear_state();
  }
}


/*----------------------------------------------------------------------*
 |  write output (public)                                       lw 02/08|
 *----------------------------------------------------------------------*/
void MultiScale::MicroStatic::output(
    Core::IO::DiscretizationWriter& output, const double time, const int step, const double dt)
{
  bool isdatawritten = false;

  //----------------------------------------------------- output results
  if (iodisp_ && resevrydisp_ && step % resevrydisp_ == 0 && !isdatawritten)
  {
    output.new_step(step, time);
    output.write_vector("displacement", dis_);
    isdatawritten = true;
  }

  //------------------------------------- stress/strain output
  if (resevrystrs_ and !(step % resevrystrs_) and iostress_ != Inpar::Solid::stress_none)
  {
    if (!isdatawritten) output.new_step(step, time);
    isdatawritten = true;

    if (stress_ == nullptr or strain_ == nullptr or plstrain_ == nullptr)
      FOUR_C_THROW("Missing stresses and strains in micro-structural time integrator");

    switch (iostress_)
    {
      case Inpar::Solid::stress_cauchy:
        output.write_vector("gauss_cauchy_stresses_xyz", *stress_, *discret_->element_row_map());
        break;
      case Inpar::Solid::stress_2pk:
        output.write_vector("gauss_2PK_stresses_xyz", *stress_, *discret_->element_row_map());
        break;
      case Inpar::Solid::stress_none:
        break;
      default:
        FOUR_C_THROW("requested stress type not supported");
    }

    switch (iostrain_)
    {
      case Inpar::Solid::strain_ea:
        output.write_vector("gauss_EA_strains_xyz", *strain_, *discret_->element_row_map());
        break;
      case Inpar::Solid::strain_gl:
        output.write_vector("gauss_GL_strains_xyz", *strain_, *discret_->element_row_map());
        break;
      case Inpar::Solid::strain_none:
        break;
      default:
        FOUR_C_THROW("requested strain type not supported");
    }

    switch (ioplstrain_)
    {
      case Inpar::Solid::strain_ea:
        output.write_vector("gauss_pl_EA_strains_xyz", *plstrain_, *discret_->element_row_map());
        break;
      case Inpar::Solid::strain_gl:
        output.write_vector("gauss_pl_GL_strains_xyz", *plstrain_, *discret_->element_row_map());
        break;
      case Inpar::Solid::strain_none:
        break;
      default:
        FOUR_C_THROW("requested plastic strain type not supported");
    }
  }
}  // MultiScale::MicroStatic::output()

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MultiScale::MicroStatic::write_restart(std::shared_ptr<Core::IO::DiscretizationWriter> output,
    const double time, const int step, const double dt)
{
  output->write_mesh(step, time);
  output->new_step(step, time);
  output->write_vector("displacement", dis_);

  std::shared_ptr<Core::LinAlg::SerialDenseMatrix> emptyalpha(
      new Core::LinAlg::SerialDenseMatrix(1, 1));

  Core::Communication::PackBuffer data;

  for (int i = 0; i < discret_->element_col_map()->num_my_elements(); ++i)
  {
    if ((*lastalpha_)[i] != nullptr)
    {
      add_to_pack(data, *(*lastalpha_)[i]);
    }
    else
    {
      add_to_pack(data, *emptyalpha);
    }
  }
  output->write_vector("alpha", data(), *discret_->element_col_map());
}

/*----------------------------------------------------------------------*
 |  read restart (public)                                       lw 03/08|
 *----------------------------------------------------------------------*/
void MultiScale::MicroStatic::read_restart(int step,
    std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> lastalpha,
    std::string name)
{
  std::shared_ptr<Core::IO::InputControl> inputcontrol =
      std::make_shared<Core::IO::InputControl>(name, true);
  Core::IO::DiscretizationReader reader(discret_, inputcontrol, step);
  double time = reader.read_double("time");
  int rstep = reader.read_int("step");
  if (rstep != step) FOUR_C_THROW("Time step on file not equal to given step");

  reader.read_vector(dis, "displacement");
  // It does not make any sense to read the mesh and corresponding
  // element based data because we surely have different element based
  // data at every Gauss point
  // reader.read_mesh(step);

  // Override current time and step with values from file
  time_ = time;
  timen_ = time_ + dt_;
  step_ = rstep;
  stepn_ = step_ + 1;

  reader.read_serial_dense_matrix(*lastalpha, "alpha");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double MultiScale::MicroStatic::get_time_to_step(int step, std::string name)
{
  std::shared_ptr<Core::IO::InputControl> inputcontrol(new Core::IO::InputControl(name, true));
  Core::IO::DiscretizationReader reader(discret_, inputcontrol, step);
  return reader.read_double("time");
}

void MultiScale::MicroStatic::evaluate_micro_bc(
    Core::LinAlg::Matrix<3, 3>* defgrd, Core::LinAlg::Vector<double>& disp)
{
  std::vector<const Core::Conditions::Condition*> conds;
  discret_->get_condition("MicroBoundary", conds);
  for (auto& cond : conds)
  {
    const auto nodeids = *cond->get_nodes();

    for (int nodeid : nodeids)
    {
      // do only nodes in my row map
      if (!discret_->node_row_map()->my_gid(nodeid)) continue;
      Core::Nodes::Node* actnode = discret_->g_node(nodeid);
      if (!actnode) FOUR_C_THROW("Cannot find global node {}", nodeid);

      // nodal coordinates
      const auto& x = actnode->x();

      // boundary displacements are prescribed via the macroscopic
      // deformation gradient
      double disp_prescribed[3];
      Core::LinAlg::Matrix<3, 3> Du(defgrd->data(), false);
      Core::LinAlg::Matrix<3, 3> I(Core::LinAlg::Initialization::zero);
      I(0, 0) = -1.0;
      I(1, 1) = -1.0;
      I(2, 2) = -1.0;
      Du += I;

      for (int k = 0; k < 3; k++)
      {
        double dis = 0.;

        for (int l = 0; l < 3; l++)
        {
          dis += Du(k, l) * x[l];
        }

        disp_prescribed[k] = dis;
      }

      std::vector<int> dofs = discret_->dof(actnode);

      for (int l = 0; l < 3; ++l)
      {
        const int gid = dofs[l];

        const int lid = disp.get_map().lid(gid);
        if (lid < 0) FOUR_C_THROW("Global id {} not on this proc in system vector", gid);
        (disp)[lid] = disp_prescribed[l];
      }
    }
  }
}

void MultiScale::MicroStatic::set_state(std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<Core::LinAlg::Vector<double>> disn, std::shared_ptr<std::vector<char>> stress,
    std::shared_ptr<std::vector<char>> strain, std::shared_ptr<std::vector<char>> plstrain,
    std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> lastalpha,
    std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> oldalpha,
    std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> oldfeas,
    std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> oldKaainv,
    std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> oldKda)
{
  dis_ = dis;
  disn_ = disn;

  stress_ = stress;
  strain_ = strain;
  plstrain_ = plstrain;

  // using std::shared_ptr's here means we do not need to return EAS data explicitly
  lastalpha_ = lastalpha;
  oldalpha_ = oldalpha;
  oldfeas_ = oldfeas;
  oldKaainv_ = oldKaainv;
  oldKda_ = oldKda;
}

void MultiScale::MicroStatic::set_time(
    const double time, const double timen, const double dt, const int step, const int stepn)
{
  time_ = time;
  timen_ = timen;
  dt_ = dt;
  step_ = step;
  stepn_ = stepn;
}

// std::shared_ptr<Core::LinAlg::Vector<double>> MultiScale::MicroStatic::ReturnNewDism() { return
// Teuchos::rcp(new Core::LinAlg::Vector<double>(*dism_)); }

void MultiScale::MicroStatic::clear_state()
{
  dis_ = nullptr;
  disn_ = nullptr;
}

void MultiScale::MicroStatic::set_eas_data() {}



void MultiScale::MicroStatic::static_homogenization(Core::LinAlg::Matrix<6, 1>* stress,
    Core::LinAlg::Matrix<6, 6>* cmat, Core::LinAlg::Matrix<3, 3>* defgrd, const bool mod_newton,
    bool& build_stiff)
{
  // determine macroscopic parameters via averaging (homogenization) of
  // microscopic features according to Kouznetsova, Miehe etc.
  // this was implemented against the background of serial usage
  // -> if a parallel version of microscale simulations is EVER wanted,
  // carefully check if/what/where things have to change

  // split microscale stiffness into parts corresponding to prescribed
  // and free dofs -> see thesis of Kouznetsova (Computational
  // homogenization for the multi-scale analysis of multi-phase
  // materials, Eindhoven, 2002)

  // for calculating the stresses, we need to choose the
  // right three components of freactm_ corresponding to a single node and
  // take the inner product with the material coordinates of this
  // node. The sum over all boundary nodes delivers the first
  // Piola-Kirchhoff macroscopic stress which has to be transformed
  // into the second Piola-Kirchhoff counterpart.
  // All these complicated conversions are necessary since only for
  // the energy-conjugated pair of first Piola-Kirchhoff and
  // deformation gradient the averaging integrals can be transformed
  // into integrals over the boundaries only in case of negligible
  // inertial forces (which simplifies matters significantly) whereas
  // the calling macroscopic material routine demands a second
  // Piola-Kirchhoff stress tensor.

  // IMPORTANT: the RVE has to be centered around (0,0,0), otherwise
  // modifications of this approach are necessary.

  freactn_->scale(-1.0);

  Core::LinAlg::Matrix<3, 3> P(Core::LinAlg::Initialization::zero);

  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int n = 0; n < np_ / 3; ++n)
      {
        P(i, j) += (*freactn_)[n * 3 + i] * (*Xp_)[n * 3 + j];
      }
      P(i, j) /= V0_;
      // sum P(i,j) over the microdis
      double sum = 0.0;
      Core::Communication::sum_all(&(P(i, j)), &sum, 1, discret_->get_comm());
      P(i, j) = sum;
    }
  }

  // determine inverse of deformation gradient

  Core::LinAlg::Matrix<3, 3> F_inv(defgrd->data(), false);
  F_inv.invert();

  // convert to second Piola-Kirchhoff stresses and store them in
  // vector format
  // assembly of stresses (cf Solid3 Hex8): S11,S22,S33,S12,S23,S13

  stress->put_scalar(0.0);

  for (int i = 0; i < 3; ++i)
  {
    (*stress)(0) += F_inv(0, i) * P(i, 0);  // S11
    (*stress)(1) += F_inv(1, i) * P(i, 1);  // S22
    (*stress)(2) += F_inv(2, i) * P(i, 2);  // S33
    (*stress)(3) += F_inv(0, i) * P(i, 1);  // S12
    (*stress)(4) += F_inv(1, i) * P(i, 2);  // S23
    (*stress)(5) += F_inv(0, i) * P(i, 2);  // S13
  }

  if (build_stiff)
  {
    // The calculation of the consistent macroscopic constitutive tensor
    // follows
    //
    // C. Miehe, Computational micro-to-macro transitions for
    // discretized micro-structures of heterogeneous materials at finite
    // strains based on a minimization of averaged incremental energy.
    // Computer Methods in Applied Mechanics and Engineering 192: 559-591, 2003.

    const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
    Core::LinAlg::MultiVector<double> cmatpf(D_->get_map(), 9);

    // make a copy
    stiff_dirich_ = std::make_shared<Core::LinAlg::SparseMatrix>(*stiff_);

    stiff_->apply_dirichlet(*dirichtoggle_);

    // use solver blocks for structure
    // get the solver number used for structural solver
    const int linsolvernumber = 9;

    // TODO: insert input parameter from input file for solver block Belos

    // get solver parameter list of linear solver
    const Teuchos::ParameterList& solverparams =
        Global::Problem::instance(microdisnum_)->solver_params(linsolvernumber);

    const auto solvertype =
        Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(solverparams, "SOLVER");

    // create solver
    Core::LinAlg::Solver solver(solverparams, discret_->get_comm(),
        Global::Problem::instance()->solver_params_callback(),
        Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
            Global::Problem::instance()->io_params(), "VERBOSITY"));

    // prescribe rigid body modes
    discret_->compute_null_space_if_necessary(solver.params());

    std::shared_ptr<Core::LinAlg::MultiVector<double>> iterinc =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*dofrowmap, 9);
    iterinc->PutScalar(0.0);

    switch (solvertype)
    {
      case Core::LinearSolver::SolverType::belos:
      {
        // solve for 9 rhs at the same time --> thanks to Belos
        Core::LinAlg::SolverParams solver_params;
        solver_params.refactor = true;
        solver_params.reset = true;
        solver.solve_with_multi_vector(stiff_->epetra_operator(), iterinc, rhs_, solver_params);
        break;
      }
      case Core::LinearSolver::SolverType::superlu:
      case Core::LinearSolver::SolverType::umfpack:
      {
        // solve for 9 rhs iteratively
        for (int i = 0; i < rhs_->NumVectors(); i++)
        {
          Core::LinAlg::SolverParams solver_params;
          solver_params.refactor = true;
          solver_params.reset = true;
          solver.solve_with_multi_vector(stiff_->epetra_operator(),
              Core::Utils::shared_ptr_from_ref((*iterinc)(i).as_multi_vector()),
              Core::Utils::shared_ptr_from_ref((*rhs_)(i).as_multi_vector()), solver_params);
        }
        break;
      }
      default:
      {
        FOUR_C_THROW("You have to choose either belos, superlu or umfpack for micro structures!");
      }
    }

    Core::LinAlg::MultiVector<double> temp(*dofrowmap, 9);
    stiff_dirich_->multiply(false, *iterinc, temp);

    Core::LinAlg::MultiVector<double> fexp(*pdof_, 9);
    int err = fexp.Import(temp, importp_->get_epetra_import(), Insert);
    if (err) FOUR_C_THROW("Export of boundary 'forces' failed with err={}", err);

    // multiply manually D_ and fexp because D_ is not distributed as usual
    // Core::LinAlg::MultiVector<double>s and, hence, standard Multiply functions do not apply.
    // NOTE: D_ has the same row GIDs (0-8), but different col IDs on different procs (corresponding
    // to pdof_). fexp is distributed normally with 9 vectors (=cols) and np_ rows. result is saved
    // as a std::vector<double> to ease subsequent communication
    std::vector<double> val(81, 0.0);
    int D_rows = D_->NumVectors();
    for (int i = 0; i < D_rows; i++)
    {
      for (int j = 0; j < fexp.NumVectors(); j++)
      {
        for (int k = 0; k < D_->MyLength(); k++)
        {
          val[i * D_rows + j] += (((*D_)(i))[k]) * ((fexp(j))[k]);
        }
      }
    }

    // sum result of matrix-matrix product over procs
    std::vector<double> sum(81, 0.0);
    Core::Communication::sum_all(val.data(), sum.data(), 81, discret_->get_comm());

    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    {
      // write as a 9x9 matrix
      for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++) ((cmatpf(j)))[i] = sum[i * 9 + j];

      // scale with inverse of RVE volume
      cmatpf.Scale(1.0 / V0_);

      // We now have to transform the calculated constitutive tensor
      // relating first Piola-Kirchhoff stresses to the deformation
      // gradient into a constitutive tensor relating second
      // Piola-Kirchhoff stresses to Green-Lagrange strains.

      convert_mat(cmatpf, F_inv, *stress, *cmat);
    }

    // after having constructed the stiffness matrix, this need not be
    // done in case of modified Newton as nonlinear solver of the
    // macroscale until the next update of macroscopic time step, when
    // build_stiff is set to true in the micromaterialgp again!

    if (mod_newton) build_stiff = false;
  }
}


void MultiScale::stop_np_multiscale()
{
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();
  int task[2] = {
      static_cast<int>(MultiScale::MicromaterialNestedParallelismAction::stop_multiscale), -1};
  Core::Communication::broadcast(task, 2, 0, subcomm);
}


void MultiScale::MicroStaticParObject::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, unique_par_object_id());

  const auto* micro_data = MultiScale::MicroStaticParObject::get_micro_static_data_ptr();
  add_to_pack(data, micro_data->gp_);
  add_to_pack(data, micro_data->eleowner_);
  add_to_pack(data, micro_data->microdisnum_);
  add_to_pack(data, micro_data->V0_);
  add_to_pack(data, micro_data->defgrd_);
  add_to_pack(data, micro_data->stress_);
  add_to_pack(data, micro_data->cmat_);
}

void MultiScale::MicroStaticParObject::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  MultiScale::MicroStaticParObject::MicroStaticData micro_data{};
  extract_from_pack(buffer, micro_data.gp_);
  extract_from_pack(buffer, micro_data.eleowner_);
  extract_from_pack(buffer, micro_data.microdisnum_);
  extract_from_pack(buffer, micro_data.V0_);
  extract_from_pack(buffer, micro_data.defgrd_);
  extract_from_pack(buffer, micro_data.stress_);
  extract_from_pack(buffer, micro_data.cmat_);
  set_micro_static_data(micro_data);
}

MultiScale::MicroStaticParObjectType MultiScale::MicroStaticParObjectType::instance_;

Core::Communication::ParObject* MultiScale::MicroStaticParObjectType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* micro = new MultiScale::MicroStaticParObject();
  micro->unpack(buffer);
  return micro;
}

FOUR_C_NAMESPACE_CLOSE
