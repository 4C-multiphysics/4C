// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_algorithm.hpp"

#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_general_l2_projection.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_gmsh.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_print.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_porofluid_pressure_based_ele.hpp"
#include "4C_porofluid_pressure_based_ele_action.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"
#include "4C_porofluid_pressure_based_meshtying_strategy_artery.hpp"
#include "4C_porofluid_pressure_based_resulttest.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


PoroPressureBased::PorofluidAlgorithm::PorofluidAlgorithm(
    std::shared_ptr<Core::FE::Discretization> actdis, const int linsolvernumber,
    const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& poroparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    :  // call constructor for "nontrivial" objects
      solver_(nullptr),
      linsolvernumber_(linsolvernumber),
      params_(probparams),
      poroparams_(poroparams),
      myrank_(Core::Communication::my_mpi_rank(actdis->get_comm())),
      nsd_(Global::Problem::instance()->n_dim()),
      isale_(false),
      skipinitder_(poroparams_.get<bool>("skip_initial_time_derivative")),
      output_satpress_(poroparams_.sublist("output").get<bool>("saturation_and_pressure")),
      output_solidpress_(poroparams_.sublist("output").get<bool>("solid_pressure")),
      output_porosity_(poroparams_.sublist("output").get<bool>("porosity")),
      output_volfrac_blood_lung_(poroparams_.sublist("output").get<bool>("volfrac_blood_lung")),
      output_det_def_grad_(
          poroparams_.sublist("output").get<bool>("determinant_of_deformation_gradient")),
      output_phase_velocities_(poroparams_.sublist("output").get<bool>("phase_velocities")),
      output_bloodvesselvolfrac_(false),
      stab_biot_(poroparams_.sublist("biot_stabilization").get<bool>("active")),
      domainint_funct_(std::vector<int>()),
      num_domainint_funct_(0),
      calcerr_(poroparams_.sublist("calculate_error_to_analytical_solution").get<bool>("active")),
      flux_reconstruction_active_(poroparams_.sublist("flux_reconstruction").get<bool>("active")),
      fluxreconsolvernum_(poroparams_.sublist("flux_reconstruction").get<int>("solver_id")),
      divcontype_(Teuchos::getIntegralValue<PoroPressureBased::DivergenceAction>(
          poroparams_, "divergence_action")),
      fdcheck_(poroparams_.sublist("fd_check").get<bool>("active")),
      fdcheckeps_(poroparams_.sublist("fd_check").get<double>("epsilon")),
      fdchecktol_(poroparams_.sublist("fd_check").get<double>("tolerance")),
      stab_biot_scaling_(poroparams_.sublist("biot_stabilization").get<double>("scaling_factor")),
      time_(0.0),
      maxtime_(params_.get<double>("total_simulation_time")),
      step_(0),
      stepmax_(params_.sublist("time_integration").get<int>("number_of_time_steps")),
      dt_(params_.sublist("time_integration").get<double>("time_step_size")),
      dtele_(0.0),
      dtsolve_(0.0),
      iternum_(0),
      itemax_(poroparams_.sublist("nonlinear_solver").get<int>("maximum_number_of_iterations")),
      upres_(params_.sublist("output").get<int>("result_data_every")),
      uprestart_(params_.sublist("output").get<int>("restart_data_every")),
      vectornormfres_(Teuchos::getIntegralValue<PoroPressureBased::VectorNorm>(
          poroparams_.sublist("nonlinear_solver").sublist("residual"), "vector_norm")),
      vectornorminc_(Teuchos::getIntegralValue<PoroPressureBased::VectorNorm>(
          poroparams_.sublist("nonlinear_solver").sublist("increment"), "vector_norm")),
      ittolres_(
          poroparams_.sublist("nonlinear_solver").sublist("residual").get<double>("tolerance")),
      ittolinc_(
          poroparams_.sublist("nonlinear_solver").sublist("increment").get<double>("tolerance")),
      artery_coupling_active_(params_.get<bool>("artery_coupling_active")),
      // Initialization of degrees of freedom variables
      phin_(nullptr),
      phinp_(nullptr),
      phidtn_(nullptr),
      phidtnp_(nullptr),
      hist_(nullptr),
      pressure_(nullptr),
      saturation_(nullptr),
      solidpressure_(nullptr),
      valid_volfracpress_dofs_(nullptr),
      valid_volfracspec_dofs_(nullptr),
      flux_(nullptr),
      nds_disp_(-1),
      nds_vel_(-1),
      nds_solidpressure_(-1),
      nds_scatra_(-1),
      discret_(actdis),
      output_(output),
      sysmat_(nullptr),
      zeros_(nullptr),
      dbcmaps_(nullptr),
      dbcmaps_with_volfracpress_(nullptr),
      dbcmaps_starting_condition_(nullptr),
      neumann_loads_(nullptr),
      residual_(nullptr),
      trueresidual_(nullptr),
      increment_(nullptr),
      starting_dbc_time_end_(poroparams_.sublist("starting_DBC").get<double>("time_end")),
      starting_dbc_onoff_(std::vector<bool>()),
      starting_dbc_funct_(std::vector<int>()),
      theta_(params_.sublist("time_integration").get<double>("theta")),
      visualization_writer_(nullptr)
{
  const int restart_step = Global::Problem::instance()->restart();
  if (restart_step > 0)
  {
    Core::IO::DiscretizationReader reader(
        *discret_, Global::Problem::instance()->input_control_file(), restart_step);

    time_ = reader.read_double("time");
  }

  visualization_writer_ = std::make_unique<Core::IO::DiscretizationVisualizationWriterMesh>(
      actdis, Core::IO::visualization_parameters_factory(
                  Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                  *Global::Problem::instance()->output_control_file(), time_));
}


void PoroPressureBased::PorofluidAlgorithm::init(bool isale, int nds_disp, int nds_vel,
    int nds_solidpressure, int nds_scalar, const std::map<int, std::set<int>>* nearby_ele_pairs)
{
  // set flags
  isale_ = isale;
  nds_disp_ = nds_disp;
  nds_vel_ = nds_vel;
  nds_solidpressure_ = nds_solidpressure;
  nds_scatra_ = nds_scalar;

  // make sure the values make sense
  // -1 is the default value, meaning that there is no coupling
  if (nds_disp_ != -1)
    if (nds_disp_ < 0 or nds_disp_ > discret_->num_dof_sets() - 1)
      FOUR_C_THROW("invalid number of dofset for mesh displacements!");

  // make sure the values make sense
  // -1 is the default value, meaning that there is no coupling
  if (nds_vel_ != -1)
    if (nds_vel_ < 0 or nds_vel_ > discret_->num_dof_sets() - 1)
      FOUR_C_THROW("invalid number of dofset for mesh velocities!");

  // make sure the values make sense
  // there has to be a valid number for the solid pressure in all cases
  if (nds_solidpressure_ < 0 or nds_solidpressure_ > discret_->num_dof_sets() - 1)
    FOUR_C_THROW("invalid number of dofset for solid pressure!");

  // ensure that degrees of freedom in the discretization have been set
  if ((not discret_->filled()) or (not discret_->have_dofs())) discret_->fill_complete();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices: local <-> global dof numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // -------------------------------------------------------------------
  // create empty system matrix (27 adjacent nodes as 'good' guess)
  // -------------------------------------------------------------------
  sysmat_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*(discret_->dof_row_map()), 27, false, true);

  // -------------------------------------------------------------------
  // create vectors containing problem variables
  // -------------------------------------------------------------------
  // solutions at time n+1
  phinp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  // solutions at time n
  phin_ = Core::LinAlg::create_vector(*dofrowmap, true);
  // time derivative of solutions at time n
  phidtn_ = Core::LinAlg::create_vector(*dofrowmap, true);
  // time derivative of solutions at time n+1
  phidtnp_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // history vector
  hist_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // valid (physically meaningful) volume fraction dofs
  valid_volfracpress_dofs_ = Core::LinAlg::create_vector(*dofrowmap, true);
  valid_volfracspec_dofs_ = Core::LinAlg::create_vector(*dofrowmap, true);
  if (output_satpress_)
  {
    // pressure at time n+1
    pressure_ = Core::LinAlg::create_vector(*dofrowmap, true);
    // saturation at time n+1
    saturation_ = Core::LinAlg::create_vector(*dofrowmap, true);
  }
  // solid pressure at time n+1
  if (output_solidpress_)
    solidpressure_ = Core::LinAlg::create_vector(*discret_->dof_row_map(nds_solidpressure_), true);
  // porosity at time n+1 (lives on same dofset as solid pressure)
  if (output_porosity_)
    porosity_ = Core::LinAlg::create_vector(*discret_->dof_row_map(nds_solidpressure_), true);

  //! volfrac blood lung at time n+1 (lives on same dofset as solid pressure)
  if (output_volfrac_blood_lung_)
    volfrac_blood_lung_ =
        Core::LinAlg::create_vector(*discret_->dof_row_map(nds_solidpressure_), true);

  //! determinant of derformation gradient at time n+1 (lives on same dofset as solid pressure)
  if (output_det_def_grad_)
    det_def_grad_ = Core::LinAlg::create_vector(*discret_->dof_row_map(nds_solidpressure_), true);

  if (output_phase_velocities_)
  {
    const int num_poro_dof = discret_->num_dof(0, discret_->l_row_node(0));
    const int num_rows = num_poro_dof * nsd_;
    phase_velocities_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
        *discret_->element_row_map(), num_rows, true);
  }

  // a vector of zeros to be used to enforce zero dirichlet boundary conditions
  zeros_ = Core::LinAlg::create_vector(*dofrowmap, true);

  int stream;
  std::istringstream stream_dbc_onoff(
      Teuchos::getNumericStringParameter(poroparams_.sublist("starting_DBC"), "active"));
  while (stream_dbc_onoff >> stream) starting_dbc_onoff_.push_back(static_cast<bool>(stream));

  std::istringstream stream_dbc_funct(
      Teuchos::getNumericStringParameter(poroparams_.sublist("starting_DBC"), "function_ids"));
  while (stream_dbc_funct >> stream) starting_dbc_funct_.push_back(static_cast<int>(stream));

  // object holds maps/subsets for DOFs subjected to Dirichlet BCs and otherwise
  dbcmaps_ = std::make_shared<Core::LinAlg::MapExtractor>();
  dbcmaps_with_volfracpress_ = std::make_shared<Core::LinAlg::MapExtractor>();
  dbcmaps_starting_condition_ = std::make_shared<Core::LinAlg::MapExtractor>();
  {
    Teuchos::ParameterList eleparams;
    // other parameters needed by the elements
    eleparams.set("total time", time_);
    eleparams.set<const Core::Utils::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());
    discret_->evaluate_dirichlet(eleparams, zeros_, nullptr, nullptr, nullptr, dbcmaps_);
    discret_->evaluate_dirichlet(
        eleparams, zeros_, nullptr, nullptr, nullptr, dbcmaps_with_volfracpress_);
    discret_->evaluate_dirichlet(
        eleparams, zeros_, nullptr, nullptr, nullptr, dbcmaps_starting_condition_);
    zeros_->put_scalar(0.0);  // just in case of change
  }

  // the vector containing body and surface forces
  neumann_loads_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // the residual vector --- more or less the rhs
  residual_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // residual vector containing the normal boundary fluxes
  trueresidual_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // incremental solution vector
  increment_ = Core::LinAlg::create_vector(*dofrowmap, true);

  set_initial_field(Teuchos::getIntegralValue<PoroPressureBased::InitialField>(
                        poroparams_.sublist("initial_condition"), "type"),
      poroparams_.sublist("initial_condition").get<int>("function_id"));

  int word1;
  std::istringstream coupled_art_dof_stream(
      Teuchos::getNumericStringParameter(poroparams_, "domain_integrals_function_ids"));
  while (coupled_art_dof_stream >> word1) domainint_funct_.push_back((int)(word1));
  // no domain integration function selected by user
  if (domainint_funct_.size() == 1 and domainint_funct_[0] < 0) domainint_funct_.resize(0);
  num_domainint_funct_ = domainint_funct_.size();

  // the values of the integrals
  domain_integrals_ = std::make_shared<Core::LinAlg::SerialDenseVector>(num_domainint_funct_);

  if (num_domainint_funct_ > 0)
  {
    // setup csv writer for domain integrals
    runtime_csvwriter_domain_integrals_.emplace(
        myrank_, *Global::Problem::instance()->output_control_file(), "domain_integrals");
    for (int funct_i = 0; funct_i < num_domainint_funct_; funct_i++)
    {
      runtime_csvwriter_domain_integrals_->register_data_vector(
          "Function ID " + std::to_string(domainint_funct_[funct_i]), 1, 14);
    }
  }

  set_element_general_parameters();

  // build mesh tying strategy
  if (artery_coupling_active_)
  {
    output_bloodvesselvolfrac_ =
        poroparams_.sublist("artery_coupling").get<bool>("output_blood_vessel_volume_fraction");
    meshtying_ = std::make_shared<PoroPressureBased::MeshtyingArtery>(this, params_, poroparams_);
    meshtying_->check_initial_fields(phinp_);
    meshtying_->set_nearby_ele_pairs(nearby_ele_pairs);
    meshtying_->setup();
  }

  solver_ = std::make_shared<Core::LinAlg::Solver>(
      Global::Problem::instance()->solver_params(linsolvernumber_), discret_->get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));
  if (artery_coupling_active_)
  {
    meshtying_->initialize_linear_solver(*solver_);
  }
  compute_null_space_if_necessary(*discretization(), solver_->params());
}


void PoroPressureBased::PorofluidAlgorithm::set_element_general_parameters() const
{
  Teuchos::ParameterList eleparams;

  eleparams.set<bool>("isale", isale_);
  eleparams.set<int>("nds_disp", nds_disp_);
  eleparams.set<int>("nds_vel", nds_vel_);
  eleparams.set<int>("nds_solidpressure", nds_solidpressure_);
  eleparams.set<int>("nds_scalar", nds_scatra_);
  eleparams.set<bool>("stab_biot", stab_biot_);

  eleparams.set<bool>("using generalized-alpha time integration", false);
  eleparams.set<bool>("using stationary formulation", false);
  eleparams.set<double>("alpha_F", 1.0);

  eleparams.set<int>("num_domainint_funct", num_domainint_funct_);
  for (int ifunct = 0; ifunct < num_domainint_funct_; ifunct++)
    eleparams.set<int>("domainint_funct_" + std::to_string(ifunct), domainint_funct_[ifunct]);

  Discret::Elements::PoroFluidMultiPhaseEleParameter::instance(discret_->name())
      ->set_general_parameters(eleparams);
}


void PoroPressureBased::PorofluidAlgorithm::prepare_time_loop()
{
  // compute pressure and saturations
  reconstruct_pressures_and_saturations();

  // compute velocities
  reconstruct_flux();

  // provide information about initial field (do not do for restarts!)
  if (step_ == 0)
  {
    // write out initial state
    output();

    // compute error for problems with analytical solution (initial field!)
    evaluate_error_compared_to_analytical_sol();
  }

  if (artery_coupling_active_)
  {
    meshtying_->prepare_time_loop();
  }
}


void PoroPressureBased::PorofluidAlgorithm::prepare_time_step()
{
  // time measurement: prepare time step
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:    + prepare time step");

  // initialization
  if (step_ == 0) prepare_first_time_step();

  // set time dependent parameters
  // note the order of the following three functions is important
  increment_time_and_step();

  // set part of the rhs vector belonging to the old timestep
  set_old_part_of_righthandside();
  // reset every parameter that potentially changes for every time step
  set_element_time_step_parameter();

  // evaluate Dirichlet and Neumann boundary conditions
  // TODO: Dirichlet auch im Fall von genalpha prenp
  // Neumann(n + alpha_f)
  apply_dirichlet_bc(time_, phinp_, nullptr);
  apply_neumann_bc(*neumann_loads_);

  // volume fraction pressure specific stuff
  evaluate_valid_volume_frac_press_and_spec();
  apply_additional_dbc_for_vol_frac_press();

  if (time_ <= starting_dbc_time_end_)
  {
    apply_starting_dbc();
  }

  if (artery_coupling_active_)
  {
    meshtying_->prepare_time_step();
  }
}


void PoroPressureBased::PorofluidAlgorithm::set_element_time_step_parameter() const
{
  Teuchos::ParameterList eleparams;

  // the total time definitely changes
  eleparams.set<double>("total time", time_);
  // we set the time step and related, just in case we want adaptive time stepping
  eleparams.set<double>("time-step length", dt_);
  eleparams.set<double>("time factor", theta_ * dt_);

  Discret::Elements::PoroFluidMultiPhaseEleParameter::instance(discret_->name())
      ->set_time_step_parameters(eleparams);
}


void PoroPressureBased::PorofluidAlgorithm::set_time_for_neumann_evaluation(
    Teuchos::ParameterList& params)
{
  params.set("total time", time_);
}


void PoroPressureBased::PorofluidAlgorithm::set_old_part_of_righthandside()
{
  // hist_ = phin_ + dt*(1-Theta)*phidtn_
  hist_->update(1.0, *phin_, dt_ * (1.0 - theta_), *phidtn_, 0.0);
}


void PoroPressureBased::PorofluidAlgorithm::add_neumann_to_residual()
{
  residual_->update(theta_ * dt_, *neumann_loads_, 1.0);
}


void PoroPressureBased::PorofluidAlgorithm::explicit_predictor()
{
  phinp_->update(dt_, *phidtn_, 1.0);
}


void PoroPressureBased::PorofluidAlgorithm::add_time_integration_specific_vectors()
{
  discret_->set_state("hist", *hist_);
  discret_->set_state("phinp_fluid", *phinp_);
  discret_->set_state("phin_fluid", *phin_);
  discret_->set_state("phidtnp", *phidtnp_);
}


void PoroPressureBased::PorofluidAlgorithm::compute_time_derivative()
{
  // time derivative of phi:
  // *phidt(n+1) = (phi(n+1)-phi(n)) / (theta*dt) + (1-(1/theta))*phidt(n)
  const double fact1 = 1.0 / (theta_ * dt_);
  const double fact2 = 1.0 - (1.0 / theta_);
  phidtnp_->update(fact2, *phidtn_, 0.0);
  phidtnp_->update(fact1, *phinp_, -fact1, *phin_, 1.0);
}


void PoroPressureBased::PorofluidAlgorithm::prepare_first_time_step()
{
  if (not skipinitder_)
  {
    if (nds_vel_ != -1 || !isale_)  // if some velocity field has been set
    {
      // TODO: Restructure enforcement of Dirichlet boundary conditions on phin_
      apply_dirichlet_bc(time_, phin_, nullptr);
      calc_initial_time_derivative();
    }
    // if initial velocity field has not been set here, the initial time derivative of phi will be
    // calculated wrongly for some time integration schemes
    else
      FOUR_C_THROW("Initial velocity field has not been set!");
  }
}


void PoroPressureBased::PorofluidAlgorithm::time_loop()
{
  // time measurement: time loop
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:  + time loop");

  // prepare time loop
  prepare_time_loop();

  while ((step_ < stepmax_) and ((time_ + 1e-12) < maxtime_))
  {
    prepare_time_step();

    solve();

    // update solution: current solution becomes old solution of next timestep
    update();

    if (calcerr_) evaluate_error_compared_to_analytical_sol();

    output();
  }
}


void PoroPressureBased::PorofluidAlgorithm::solve()
{
  nonlinear_solve();

  // reconstruct pressures and saturations
  reconstruct_pressures_and_saturations();

  // reconstruct velocities
  reconstruct_flux();
}


void PoroPressureBased::PorofluidAlgorithm::update()
{
  if (artery_coupling_active_)
  {
    meshtying_->update();
  }

  // compute time derivative at time n+1
  compute_time_derivative();

  // solution of this step becomes most recent solution of the last step
  phin_->update(1.0, *phinp_, 0.0);

  // time deriv. of this step becomes most recent time derivative of
  // last step
  phidtn_->update(1.0, *phidtnp_, 0.0);
}


void PoroPressureBased::PorofluidAlgorithm::apply_mesh_movement(
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp)
{
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE: apply mesh movement");

  if (nds_disp_ == -1)
  {
    FOUR_C_THROW("Dof set number of displacement related dofs has not been set!");
  }

  // check existence of displacement vector
  if (dispnp == nullptr) FOUR_C_THROW("Got null pointer for displacements!");

  // provide discretization with displacement field
  set_state(nds_disp_, "dispnp", dispnp);

  if (artery_coupling_active_)
  {
    meshtying_->apply_mesh_movement();
  }
}


inline void PoroPressureBased::PorofluidAlgorithm::print_time_step_info()
{
  if (myrank_ == 0)
  {
    std::cout << "| TIME: " << std::setw(11) << std::setprecision(4) << std::scientific << time_
              << "/" << std::setw(11) << std::setprecision(4) << std::scientific << maxtime_
              << "  DT = " << std::setw(11) << std::setprecision(4) << std::scientific << dt_
              << "  " << "One-Step-Theta (theta = " << std::setw(3) << std::setprecision(2)
              << theta_ << ") STEP = " << std::setw(4) << step_ << "/" << std::setw(4) << stepmax_
              << "            |" << '\n';
  }
}


void PoroPressureBased::PorofluidAlgorithm::collect_runtime_output_data()
{
  // write domain decomposition for visualization (only once at step 0!)
  if (step_ == 0)
  {
    visualization_writer_->append_element_owner("Owner");

    // write output of blood vessel volume fraction
    if (artery_coupling_active_ && output_bloodvesselvolfrac_)
    {
      visualization_writer_->append_result_data_vector_with_context(
          *meshtying_->blood_vessel_volume_fraction(), Core::IO::OutputEntity::element,
          {"bloodvesselvolfrac"});
    }
  }

  // write artery output if active
  if (artery_coupling_active_)
  {
    meshtying_->output();
  }

  const int numdof = discret_->num_dof(0, discret_->l_row_node(0));
  std::vector<std::optional<std::string>> context(numdof);
  {
    for (int i = 0; i < numdof; ++i)
    {
      context[i] = "phi_" + std::to_string(i + 1);
    }

    visualization_writer_->append_result_data_vector_with_context(
        *phinp_, Core::IO::OutputEntity::dof, context);
  }

  if (output_satpress_)
  {
    // collect pressure
    {
      for (int i = 0; i < numdof; ++i)
      {
        context[i] = "pressure_" + std::to_string(i + 1);
      }

      visualization_writer_->append_result_data_vector_with_context(
          *pressure_, Core::IO::OutputEntity::dof, context);
    }

    // collect saturation
    {
      for (int i = 0; i < numdof; ++i)
      {
        context[i] = "saturation_" + std::to_string(i + 1);
      }
      visualization_writer_->append_result_data_vector_with_context(
          *saturation_, Core::IO::OutputEntity::dof, context);
    }
  }

  // solid pressure
  if (output_solidpress_)
  {
    // convert dof-based vector into node-based multi-vector for postprocessing
    auto solidpressure_multi = Core::IO::convert_dof_vector_to_node_based_multi_vector(
        *discret_, *solidpressure_, nds_solidpressure_, 1);

    visualization_writer_->append_result_data_vector_with_context(
        *solidpressure_multi, Core::IO::OutputEntity::node, {"solidpressure"});
  }

  // displacement field
  if (isale_)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
        discret_->get_state(nds_disp_, "dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot extract displacement field from discretization");

    // convert dof-based vector into node-based multi-vector for postprocessing
    auto dispnp_multi = Core::IO::convert_dof_vector_to_node_based_multi_vector(
        *discret_, *dispnp, nds_disp_, nsd_);

    std::vector<std::optional<std::string>> context(nsd_, "ale-displacement");
    visualization_writer_->append_result_data_vector_with_context(
        *dispnp_multi, Core::IO::OutputEntity::node, context);
  }

  // fluxes
  if (flux_ != nullptr)
  {
    const int dim = Global::Problem::instance()->n_dim();
    const int numdof = discret_->num_dof(0, discret_->l_row_node(0));
    // get the noderowmap
    const Core::LinAlg::Map* noderowmap = discret_->node_row_map();
    for (int k = 0; k < numdof; k++)
    {
      Core::LinAlg::MultiVector<double> flux_k(*noderowmap, 3, true);

      std::string name = "flux_" + std::to_string(k + 1);
      for (int i = 0; i < flux_k.local_length(); ++i)
      {
        // get value for each component of flux vector
        for (int idim = 0; idim < dim; idim++)
        {
          double value = (*flux_)(k * dim + idim).local_values_as_span()[i];
          flux_k.replace_local_value(i, idim, value);
        }
      }
      std::vector<std::optional<std::string>> context(flux_k.num_vectors(), name);
      visualization_writer_->append_result_data_vector_with_context(
          flux_k, Core::IO::OutputEntity::node, context);
    }
  }

  if (output_phase_velocities_)
  {
    const int num_dim = Global::Problem::instance()->n_dim();
    const int num_poro_dof = discret_->num_dof(0, discret_->l_row_node(0));

    const Core::LinAlg::Map* element_row_map = discret_->element_row_map();

    for (int k = 0; k < num_poro_dof; k++)
    {
      Core::LinAlg::MultiVector<double> velocity_k(*element_row_map, num_dim, true);

      for (int i = 0; i < velocity_k.local_length(); ++i)
      {
        for (int idim = 0; idim < num_dim; idim++)
        {
          double value = (*phase_velocities_)(k * num_dim + idim).local_values_as_span()[i];
          velocity_k.replace_local_value(i, idim, value);
        }
      }

      std::string output_name = "velocity_" + std::to_string(k + 1);
      std::vector<std::optional<std::string>> context(velocity_k.num_vectors(), output_name);
      visualization_writer_->append_result_data_vector_with_context(
          velocity_k, Core::IO::OutputEntity::element, context);
    }
  }

  // porosity
  if (output_porosity_)
  {
    // convert dof-based vector into node-based multi-vector for postprocessing
    auto porosity_multi = Core::IO::convert_dof_vector_to_node_based_multi_vector(
        *discret_, *porosity_, nds_solidpressure_, 1);

    visualization_writer_->append_result_data_vector_with_context(
        *porosity_multi, Core::IO::OutputEntity::node, {"porosity"});
  }

  //! determinant of deformation gradient
  if (output_det_def_grad_)
  {
    //! convert dof-based vector into node-based multi-vector for postprocessing
    auto det_def_grad_multi = Core::IO::convert_dof_vector_to_node_based_multi_vector(
        *discret_, *det_def_grad_, nds_solidpressure_, 1);

    visualization_writer_->append_result_data_vector_with_context(
        *det_def_grad_multi, Core::IO::OutputEntity::node, {"det_def_grad"});
  }

  //! volfrac blood lung
  if (output_volfrac_blood_lung_)
  {
    //! convert dof-based vector into node-based multi-vector for postprocessing
    auto volfrac_blood_lung_multi = Core::IO::convert_dof_vector_to_node_based_multi_vector(
        *discret_, *volfrac_blood_lung_, nds_solidpressure_, 1);

    visualization_writer_->append_result_data_vector_with_context(
        *volfrac_blood_lung_multi, Core::IO::OutputEntity::node, {"volfrac_blood_lung"});
  }
}


void PoroPressureBased::PorofluidAlgorithm::output()
{
  // time measurement: output of solution
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:    + output of solution");

  // solution output and potentially restart data and/or flux data
  if (do_output())
  {
    // reconstruct porosity for output; porosity is only needed for output and does not have to be
    // transferred between fields
    if (output_porosity_) reconstruct_porosity();

    //! reconstruct volfrac of additional porous network with closing relation blood lung for
    //! output; it is only needed for output and does not have to be transferred between fields
    if (output_volfrac_blood_lung_) reconstruct_volfrac_blood_lung();

    //! reconstruct determinant of deformation gradient
    if (output_det_def_grad_) reconstruct_determinant_of_derformation_gradient();

    if (output_phase_velocities_) calculate_phase_velocities();

    // evaluate domain integrals
    if (num_domainint_funct_ > 0) evaluate_domain_integrals();

    // do the runtime output
    {
      visualization_writer_->reset();

      collect_runtime_output_data();

      visualization_writer_->write_to_disk(time_, step_);
    }

    // add restart data
    if (step_ % uprestart_ == 0 and step_ != 0) output_restart();
  }
}


void PoroPressureBased::PorofluidAlgorithm::output_restart()
{
  // step number and time (only after that data output is possible)
  output_->new_step(step_, time_);

  // solution
  output_->write_vector("phinp_fluid", phinp_);

  // additional state vectors that are needed for One-Step-Theta restart
  output_->write_vector("phidtn_fluid", phidtn_);
  output_->write_vector("phin_fluid", phin_);
}


std::shared_ptr<const Core::LinAlg::Map> PoroPressureBased::PorofluidAlgorithm::dof_row_map(
    unsigned nds) const
{
  return Core::Utils::shared_ptr_from_ref(*discret_->dof_row_map(nds));
}


std::shared_ptr<const Core::LinAlg::Map> PoroPressureBased::PorofluidAlgorithm::artery_dof_row_map()
    const
{
  FOUR_C_ASSERT(artery_coupling_active_,
      "artery_dof_row_map can only be called when artery coupling is active.");
  return meshtying_->artery_dof_row_map();
}


std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
PoroPressureBased::PorofluidAlgorithm::artery_porofluid_sysmat() const
{
  FOUR_C_ASSERT(artery_coupling_active_,
      "artery_porofluid_sysmat can only be called when artery coupling is active.");

  return meshtying_->artery_porofluid_sysmat();
}


std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::PorofluidAlgorithm::artery_porofluid_rhs() const
{
  FOUR_C_ASSERT(artery_coupling_active_,
      "artery_porofluid_rhs can only be called when artery coupling is active.");
  return meshtying_->artery_porofluid_rhs();
}


void PoroPressureBased::PorofluidAlgorithm::apply_dirichlet_bc(const double time,
    std::shared_ptr<Core::LinAlg::Vector<double>> prenp,
    std::shared_ptr<Core::LinAlg::Vector<double>> predt)
{
  // time measurement: apply Dirichlet conditions
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:      + apply dirich cond.");

  // needed parameters
  Teuchos::ParameterList p;
  p.set("total time", time);  // actual time t_{n+1}
  p.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // predicted Dirichlet values
  // \c  prenp then also holds prescribed new Dirichlet values
  discret_->clear_state();
  discret_->evaluate_dirichlet(p, prenp, predt, nullptr, nullptr, dbcmaps_);
  discret_->clear_state();
}


void PoroPressureBased::PorofluidAlgorithm::scaling_and_neumann()
{
  // scaling to get true residual vector for all time integration schemes
  trueresidual_->update(residual_scaling(), *residual_, 0.0);

  // add Neumann b.c. scaled with a factor due to time discretization
  add_neumann_to_residual();
}


void PoroPressureBased::PorofluidAlgorithm::apply_neumann_bc(
    Core::LinAlg::Vector<double>& neumann_loads  //!< Neumann loads
)
{
  // prepare load vector
  neumann_loads.put_scalar(0.0);

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  condparams.set<PoroPressureBased::BoundaryAction>("action", PoroPressureBased::bd_calc_Neumann);

  // set time for evaluation of point Neumann conditions as parameter depending on time integration
  // scheme line/surface/volume Neumann conditions use the time stored in the time parameter class
  set_time_for_neumann_evaluation(condparams);

  // evaluate Neumann boundary conditions at time t_{n+alpha_F} (generalized alpha) or time t_{n+1}
  // (otherwise)
  discret_->evaluate_neumann(condparams, neumann_loads);
  discret_->clear_state();
}


void PoroPressureBased::PorofluidAlgorithm::assemble_mat_and_rhs()
{
  // time measurement: element calls
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:       + element calls");

  // get cpu time
  const double tcpuele = Teuchos::Time::wallTime();

  // zero out matrix entries
  sysmat_->zero();

  // reset the residual vector
  residual_->put_scalar(0.0);

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::calc_mat_and_rhs);

  // clean up, just in case ...
  discret_->clear_state();

  // set vector values needed by elements
  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // call loop over elements
  discret_->evaluate(eleparams, sysmat_, nullptr, residual_, nullptr, nullptr);

  // clean up
  discret_->clear_state();

  // potential residual scaling and potential addition of Neumann terms
  scaling_and_neumann();

  // finalize assembly of system matrix
  sysmat_->complete();

  // end time measurement for element
  double mydtele = Teuchos::Time::wallTime() - tcpuele;
  dtele_ = Core::Communication::max_all(mydtele, discret_->get_comm());
}


void PoroPressureBased::PorofluidAlgorithm::evaluate_valid_volume_frac_press_and_spec()
{
  // time measurement: element calls
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:       + element calls");

  // reset valid_volfracpress_dofs and _spec_dofs-vector
  valid_volfracpress_dofs_->put_scalar(0.0);
  valid_volfracspec_dofs_->put_scalar(0.0);

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::calc_valid_dofs);

  // clean up, just in case ...
  discret_->clear_state();

  // set vector values needed by elements
  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // call loop over elements (with valid volume fraction pressure DOFs)
  discret_->evaluate(
      eleparams, nullptr, nullptr, nullptr, valid_volfracpress_dofs_, valid_volfracspec_dofs_);

  // clean up
  discret_->clear_state();
}


void PoroPressureBased::PorofluidAlgorithm::apply_additional_dbc_for_vol_frac_press()
{
  const Core::LinAlg::Map* elecolmap = discret_->element_col_map();
  std::vector<int> mydirichdofs;

  // we identify the volume fraction pressure dofs which do not have a physical meaning and set
  // a DBC on them
  for (int i = 0; i < elecolmap->num_my_elements(); ++i)
  {
    // dynamic_cast necessary because virtual inheritance needs runtime information
    auto* myele = dynamic_cast<Discret::Elements::PoroFluidMultiPhase*>(
        discret_->g_element(elecolmap->gid(i)));

    const Core::Mat::Material& material = *(myele->material());

    // check the material
    if (material.material_type() != Core::Materials::m_fluidporo_multiphase and
        material.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
      FOUR_C_THROW("only poro multiphase and poro multiphase reactions material valid");

    const auto& multiphasemat = static_cast<const Mat::FluidPoroMultiPhase&>(material);

    const int numfluidphases = multiphasemat.num_fluid_phases();
    const int numvolfrac = multiphasemat.num_vol_frac();
    const int nummat = multiphasemat.num_mat();

    // this is only necessary if we have volume fractions present
    // TODO: this works only if we have the same number of phases in every element
    if (nummat == numfluidphases) return;

    Core::Nodes::Node** nodes = myele->nodes();
    for (int inode = 0; inode < (myele->num_node()); inode++)
    {
      if (nodes[inode]->owner() == myrank_)
      {
        std::vector<int> dofs = discret_->dof(0, nodes[inode]);

        for (int idof = numfluidphases + numvolfrac; idof < nummat; ++idof)
        {
          // if not already in original dirich map     &&   if it is not a valid volume fraction
          // pressure dof identified with < 1
          if (dbcmaps_->cond_map()->lid(dofs[idof]) == -1 &&
              (int)valid_volfracpress_dofs_
                      ->local_values_as_span()[discret_->dof_row_map()->lid(dofs[idof])] < 1)
            if (not(std::find(mydirichdofs.begin(), mydirichdofs.end(), dofs[idof]) !=
                    mydirichdofs.end()))
            {
              mydirichdofs.push_back(dofs[idof]);
              phinp_->replace_global_value(dofs[idof], 0.0);
            }
        }
      }
    }
  }

  // build map
  int nummydirichvals = mydirichdofs.size();
  std::shared_ptr<Core::LinAlg::Map> dirichmap = std::make_shared<Core::LinAlg::Map>(
      -1, nummydirichvals, mydirichdofs.data(), 0, discret_->get_comm());

  // build vector of maps
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> condmaps;
  condmaps.push_back(dirichmap);
  condmaps.push_back(dbcmaps_->cond_map());

  // combined map
  std::shared_ptr<Core::LinAlg::Map> condmerged =
      Core::LinAlg::MultiMapExtractor::merge_maps(condmaps);
  *dbcmaps_with_volfracpress_ = Core::LinAlg::MapExtractor(*(discret_->dof_row_map()), condmerged);
}


void PoroPressureBased::PorofluidAlgorithm::apply_starting_dbc()
{
  const auto& elecolmap = *discret_->element_col_map();
  std::vector<int> dirichlet_dofs;
  const int num_poro_dofs = discret_->num_dof(0, discret_->l_row_node(0));

  for (int ele_idx = 0; ele_idx < elecolmap.num_my_elements(); ++ele_idx)
  {
    const auto& current_element = *discret_->g_element(elecolmap.gid(ele_idx));
    const auto* const nodes = current_element.nodes();

    for (int node_idx = 0; node_idx < (current_element.num_node()); node_idx++)
    {
      const auto* const current_node = nodes[node_idx];
      if (current_node->owner() == myrank_)
      {
        const std::vector<int> gid_node_dofs = discret_->dof(0, current_node);

        for (int dof_idx = 0; dof_idx < num_poro_dofs; ++dof_idx)
        {
          if (starting_dbc_onoff_[dof_idx])
          {
            auto const gid = gid_node_dofs[dof_idx];
            if (std::find(dirichlet_dofs.begin(), dirichlet_dofs.end(), gid) ==
                dirichlet_dofs.end())
            {
              // LID returns -1 if not found in this map/on this processor
              if (dbcmaps_with_volfracpress_->cond_map()->lid(gid) == -1)
              {
                dirichlet_dofs.push_back(gid);
              }
              const double dbc_value = Global::Problem::instance()
                                           ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                                               starting_dbc_funct_[dof_idx])
                                           .evaluate(current_node->x(), time_, 0);
              phinp_->replace_global_value(gid, dbc_value);
            }
          }
        }
      }
    }
  }

  // build combined DBC map
  std::shared_ptr<Core::LinAlg::Map> additional_map = std::make_shared<Core::LinAlg::Map>(
      -1, dirichlet_dofs.size(), dirichlet_dofs.data(), 0, discret_->get_comm());

  std::vector<std::shared_ptr<const Core::LinAlg::Map>> condition_maps;
  condition_maps.emplace_back(additional_map);
  condition_maps.push_back(dbcmaps_with_volfracpress_->cond_map());

  std::shared_ptr<Core::LinAlg::Map> combined_map =
      Core::LinAlg::MultiMapExtractor::merge_maps(condition_maps);
  *dbcmaps_starting_condition_ =
      Core::LinAlg::MapExtractor(*(discret_->dof_row_map()), combined_map);
}


void PoroPressureBased::PorofluidAlgorithm::assemble_fluid_struct_coupling_mat(
    std::shared_ptr<Core::LinAlg::SparseOperator> k_fs)
{
  // time measurement: element calls
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:       + element calls");

  // get cpu time
  const double tcpuele = Teuchos::Time::wallTime();

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<PoroPressureBased::Action>(
      "action", PoroPressureBased::calc_fluid_struct_coupl_mat);

  // clean up, just in case ...
  discret_->clear_state();

  // set vector values needed by elements
  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // create strategy for assembly of solid pressure
  Core::FE::AssembleStrategy fluidstrategy(0,  // fluiddofset for row
      1,                                       // structdofset for column
      k_fs,                                    // fluid-mechanical matrix
      nullptr,                                 // no other matrix or vectors
      nullptr, nullptr, nullptr);

  // Evaluate coupling matrix
  discret_->evaluate(eleparams, fluidstrategy);

  // clean up
  discret_->clear_state();

  // end time measurement for element
  dtele_ = Teuchos::Time::wallTime() - tcpuele;
}


void PoroPressureBased::PorofluidAlgorithm::assemble_fluid_scatra_coupling_mat(
    std::shared_ptr<Core::LinAlg::SparseOperator> k_pfs)
{
  // time measurement: element calls
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:       + element calls");

  // get cpu time
  const double tcpuele = Teuchos::Time::wallTime();

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<PoroPressureBased::Action>(
      "action", PoroPressureBased::calc_fluid_scatra_coupl_mat);

  // clean up, just in case ...
  discret_->clear_state();

  // set vector values needed by elements
  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // create strategy for assembly of solid pressure
  Core::FE::AssembleStrategy fluidstrategy(0,  // fluiddofset for row
      3,                                       // scatradofset for column
      k_pfs,                                   // fluid-scatra matrix
      nullptr,                                 // no other matrix or vectors
      nullptr, nullptr, nullptr);

  // Evaluate coupling matrix
  discret_->evaluate(eleparams, fluidstrategy);

  // clean up
  discret_->clear_state();

  // end time measurement for element
  dtele_ = Teuchos::Time::wallTime() - tcpuele;
}


void PoroPressureBased::PorofluidAlgorithm::nonlinear_solve()
{
  // time measurement: nonlinear iteration
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:   + nonlin. iteration/lin. solve");

  // out to screen
  print_header();
  print_time_step_info();

  // print header of convergence table to screen
  print_convergence_header();

  // turn adaptive solver tolerance on/off
  const bool isadapttol = poroparams_.sublist("nonlinear_solver")
                              .sublist("convergence_criteria_adaptivity")
                              .get<bool>("active");
  const double adaptolbetter = poroparams_.sublist("nonlinear_solver")
                                   .sublist("convergence_criteria_adaptivity")
                                   .get<double>("nonlinear_to_linear_tolerance_ratio");

  const double abstolres =
      poroparams_.sublist("nonlinear_solver").get<double>("absolute_tolerance_residual");
  double actresidual(0.0);

  // prepare Newton-Raphson iteration
  iternum_ = 0;

  // start Newton-Raphson iteration
  while (true)
  {
    iternum_++;

    // call elements to calculate system matrix and rhs and assemble
    // note: DBC is applied herein
    evaluate();

    // abort nonlinear iteration if desired
    if (abort_nonlin_iter(iternum_, itemax_, abstolres, actresidual)) break;

    // initialize increment vector
    increment_->put_scalar(0.0);

    linear_solve(isadapttol, actresidual, adaptolbetter);

    //------------------------------------------------ update solution vector
    if (artery_coupling_active_)
    {
      update_iter(meshtying_->combined_increment());
    }
    else
    {
      update_iter(increment_);
    }
  }
}


void PoroPressureBased::PorofluidAlgorithm::linear_solve(
    bool isadapttol, double actresidual, double adaptolbetter)
{
  // get cpu time
  const double tcpusolve = Teuchos::Time::wallTime();

  // time measurement: call linear solver
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:       + call linear solver");

  // do adaptive linear solver tolerance (not in first solve)
  Core::LinAlg::SolverParams solver_params;
  if (isadapttol && iternum_ > 1)
  {
    solver_params.nonlin_tolerance = ittolres_;
    solver_params.nonlin_residual = actresidual;
    solver_params.lin_tol_better = adaptolbetter;
  }

  if (artery_coupling_active_)
  {
    meshtying_->linear_solve(solver_, solver_params);
  }
  else
  {
    solver_params.refactor = true;
    solver_params.reset = true;
    solver_->solve(sysmat_, increment_, residual_, solver_params);
  }

  solver_->reset_tolerance();

  // end time measurement for solver
  double mydtsolve = Teuchos::Time::wallTime() - tcpusolve;
  dtsolve_ = Core::Communication::max_all(mydtsolve, discret_->get_comm());
}


bool PoroPressureBased::PorofluidAlgorithm::abort_nonlin_iter(
    const int itnum, const int itemax, const double abstolres, double& actresidual)
{
  //----------------------------------------------------- compute norms
  std::vector<double> preresnorm;
  std::vector<double> incprenorm;
  std::vector<double> prenorm;
  if (artery_coupling_active_)
  {
    meshtying_->calculate_norms(preresnorm, incprenorm, prenorm);
  }
  else
  {
    preresnorm.resize(1);
    incprenorm.resize(1);
    prenorm.resize(1);

    preresnorm[0] = calculate_vector_norm(vectornormfres_, *residual_);
    incprenorm[0] = calculate_vector_norm(vectornorminc_, *increment_);
    prenorm[0] = calculate_vector_norm(vectornorminc_, *phinp_);
  }

  std::vector<double> relinc(prenorm.size());

  for (std::size_t i = 0; i < prenorm.size(); ++i)
  {
    // care for the case that nothing really happens in the pressure
    if (prenorm[i] < 1.0e-6) prenorm[i] = 1.0;
    relinc[i] = incprenorm[i] / prenorm[i];
  }

  const double maxres = *std::max_element(preresnorm.begin(), preresnorm.end());
  const double maxrelinc = *std::max_element(relinc.begin(), relinc.end());

  //-------------------------------------------------- output to screen
  // special case of very first iteration step: solution increment is not yet available
  if (itnum == 1)
  {
    // print first line of convergence table to screen
    print_convergence_values_first_iter(itnum, itemax, ittolinc_, preresnorm);
    // we have to solve at least once --> return false
    return false;
  }

  // ordinary case later iteration steps: solution increment can be printed and convergence check
  // should be done
  else
  {
    // print current line of convergence table to screen
    print_convergence_values(itnum, itemax, ittolinc_, preresnorm, incprenorm, prenorm);

    // convergence check
    if (maxres <= ittolres_ and maxrelinc <= ittolinc_)
    {
      // print finish line of convergence table to screen
      print_convergence_finish_line();

      return true;
    }
  }

  // abort iteration, when there's nothing more to do! -> more robustness
  // absolute tolerance for deciding if residual is (already) zero
  // prevents additional solver calls that will not improve the residual anymore
  if ((maxres < abstolres))
  {
    // print finish line of convergence table to screen
    print_convergence_finish_line();

    return true;
  }


  if ((itnum == itemax))
  {
    switch (divcontype_)
    {
      case DivergenceAction::continue_anyway:
      {
        // warn if itemax is reached without convergence, but proceed to
        // next timestep...
        if (myrank_ == 0)
        {
          std::cout << "+---------------------------------------------------------------+"
                    << std::endl;
          std::cout << "|            >>>>>> not converged in itemax steps!              |"
                    << std::endl;
          std::cout << "+---------------------------------------------------------------+"
                    << std::endl
                    << std::endl;
        }
        break;
      }
      case DivergenceAction::stop:
      {
        FOUR_C_THROW("Porofluid multiphase solver not converged in itemax steps!");
        break;
      }
      default:
        FOUR_C_THROW("unknown divercont action!");
        break;
    }
    // yes, we stop the iteration
    return true;
  }

  // return the maximum residual value -> used for adaptivity of linear solver tolerance
  actresidual = std::max(maxres, maxrelinc);

  // check for INF's and NaN's before going on...
  for (std::size_t i = 0; i < prenorm.size(); ++i)
    if (std::isnan(incprenorm[i]) or std::isnan(prenorm[i]) or std::isnan(preresnorm[i]))
      FOUR_C_THROW("calculated vector norm is NaN.");
  for (std::size_t i = 0; i < prenorm.size(); ++i)
    if (std::isinf(incprenorm[i]) or std::isinf(prenorm[i]) or std::isinf(preresnorm[i]))
      FOUR_C_THROW("calculated vector norm is INF.");

  return false;
}


void PoroPressureBased::PorofluidAlgorithm::print_header()
{
  if (myrank_ == 0)
  {
    std::cout << "\n";
    std::cout << "+--------------------------------------------------------------------------------"
                 "--------------------------------+\n";
    std::cout << "| PORO MULTIPHASE FLUID SOLVER                                                   "
                 "                                |\n";
  }
}


void PoroPressureBased::PorofluidAlgorithm::reconstruct_pressures_and_saturations()
{
  if (output_satpress_)
  {
    // reset
    pressure_->put_scalar(0.0);
    saturation_->put_scalar(0.0);

    // create parameter list for elements
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::calc_pres_and_sat);

    // set vector values needed by elements
    discret_->clear_state();

    // add state vectors according to time-integration scheme
    add_time_integration_specific_vectors();

    // initialize counter vector (will store how many times the node has been evaluated)
    std::shared_ptr<Core::LinAlg::Vector<double>> counter =
        Core::LinAlg::create_vector(*discret_->dof_row_map(), true);

    // call loop over elements
    discret_->evaluate(eleparams, nullptr, nullptr, pressure_, saturation_, counter);

    discret_->clear_state();

    // dummy way: the values have been assembled too many times -> just divide by number of
    // evaluations
    for (int i = 0; i < discret_->dof_row_map()->num_my_elements(); i++)
    {
      pressure_->get_values()[i] *= 1.0 / counter->local_values_as_span()[i];
      saturation_->get_values()[i] *= 1.0 / counter->local_values_as_span()[i];
    }
  }

  // reconstruct also the solid pressures
  if (output_solidpress_) reconstruct_solid_pressures();
}


void PoroPressureBased::PorofluidAlgorithm::reconstruct_solid_pressures()
{
  // reset
  solidpressure_->put_scalar(0.0);

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::calc_solidpressure);

  // set vector values needed by elements
  discret_->clear_state();

  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // initialize counter vector (will store how many times the node has been evaluated)
  std::shared_ptr<Core::LinAlg::Vector<double>> counter =
      Core::LinAlg::create_vector(*discret_->dof_row_map(nds_solidpressure_), true);

  // create strategy for assembly of solid pressure
  Core::FE::AssembleStrategy strategysolidpressure(
      nds_solidpressure_, 0, nullptr, nullptr, solidpressure_, counter, nullptr);

  // call loop over elements
  discret_->evaluate(eleparams, strategysolidpressure);

  discret_->clear_state();

  // dummy way: the values have been assembled too many times -> just divide by number of
  // evaluations
  for (int i = 0; i < discret_->dof_row_map(nds_solidpressure_)->num_my_elements(); i++)
  {
    (*solidpressure_).get_values()[i] *= 1.0 / counter->local_values_as_span()[i];
  }
}


void PoroPressureBased::PorofluidAlgorithm::reconstruct_flux()
{
  if (flux_reconstruction_active_)
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::recon_flux_at_nodes);

    const int dim = Global::Problem::instance()->n_dim();
    // we assume same number of dofs per node in the whole dis here
    const int totalnumdof = discret_->num_dof(0, discret_->l_row_node(0));
    const int numvec = totalnumdof * dim;

    // add state vectors according to time-integration scheme
    add_time_integration_specific_vectors();
    const auto& solverparams = Global::Problem::instance()->solver_params(fluxreconsolvernum_);
    flux_ = Core::FE::compute_nodal_l2_projection(*discret_, "phinp_fluid", numvec, eleparams,
        solverparams, Global::Problem::instance()->solver_params_callback());
  }
}


void PoroPressureBased::PorofluidAlgorithm::calculate_phase_velocities()
{
  phase_velocities_->put_scalar(0.0);

  Teuchos::ParameterList eleparams;
  eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::calc_phase_velocities);

  discret_->clear_state();

  add_time_integration_specific_vectors();

  discret_->evaluate_scalars(eleparams, *phase_velocities_);
}


void PoroPressureBased::PorofluidAlgorithm::reconstruct_porosity()
{
  // time measurement: reconstruction of porosity
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:    + reconstruct porosity");

  // reset
  porosity_->put_scalar(0.0);

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::calc_porosity);

  // set vector values needed by elements
  discret_->clear_state();

  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // initialize counter vector (will store how many times the node has been evaluated)
  std::shared_ptr<Core::LinAlg::Vector<double>> counter =
      Core::LinAlg::create_vector(*discret_->dof_row_map(nds_solidpressure_), true);

  // create strategy for assembly of porosity
  Core::FE::AssembleStrategy strategyporosity(
      nds_solidpressure_, 0, nullptr, nullptr, porosity_, counter, nullptr);

  // call loop over elements
  discret_->evaluate(eleparams, strategyporosity);

  discret_->clear_state();

  // dummy way: the values have been assembled too many times -> just divide by number of
  // evaluations
  for (int i = 0; i < discret_->dof_row_map(nds_solidpressure_)->num_my_elements(); i++)
  {
    porosity_->get_values()[i] *= 1.0 / counter->local_values_as_span()[i];
  }
}

void PoroPressureBased::PorofluidAlgorithm::reconstruct_volfrac_blood_lung()
{
  // time measurement: reconstruction of volfrac
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:    + reconstruct volfrac blood lung");

  // reset
  volfrac_blood_lung_->put_scalar(0.0);

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::calc_volfrac_blood_lung);

  // set vector values needed by elements
  discret_->clear_state();

  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // initialize counter vector (will store how many times the node has been evaluated)
  std::shared_ptr<Core::LinAlg::Vector<double>> counter =
      Core::LinAlg::create_vector(*discret_->dof_row_map(nds_solidpressure_), true);

  // create strategy for assembly of volfrac
  Core::FE::AssembleStrategy strategyvolfrac(
      nds_solidpressure_, 0, nullptr, nullptr, volfrac_blood_lung_, counter, nullptr);

  // call loop over elements
  discret_->evaluate(eleparams, strategyvolfrac);

  discret_->clear_state();

  // dummy way: the values have been assembled too many times -> just divide by number of
  // evaluations
  for (int i = 0; i < discret_->dof_row_map(nds_solidpressure_)->num_my_elements(); i++)
  {
    volfrac_blood_lung_->get_values()[i] *= 1.0 / counter->local_values_as_span()[i];
  }
}

void PoroPressureBased::PorofluidAlgorithm::reconstruct_determinant_of_derformation_gradient()
{
  // time measurement: reconstruction of volfrac
  TEUCHOS_FUNC_TIME_MONITOR(
      "POROFLUIDMULTIPHASE:    + reconstruct determinant of deformation gradient");

  // reset
  det_def_grad_->put_scalar(0.0);

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<PoroPressureBased::Action>(
      "action", PoroPressureBased::calc_determinant_of_deformationgradient);

  // set vector values needed by elements
  discret_->clear_state();

  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // initialize counter vector (will store how many times the node has been evaluated)
  std::shared_ptr<Core::LinAlg::Vector<double>> counter =
      Core::LinAlg::create_vector(*discret_->dof_row_map(nds_solidpressure_), true);

  // create strategy for assembly of volfrac
  Core::FE::AssembleStrategy strategydetdefgrad(
      nds_solidpressure_, 0, nullptr, nullptr, det_def_grad_, counter, nullptr);

  // call loop over elements
  discret_->evaluate(eleparams, strategydetdefgrad);

  discret_->clear_state();

  // dummy way: the values have been assembled too many times -> just divide by number of
  // evaluations
  for (int i = 0; i < discret_->dof_row_map(nds_solidpressure_)->num_my_elements(); i++)
  {
    det_def_grad_->get_values()[i] *= 1.0 / counter->local_values_as_span()[i];
  }
}


void PoroPressureBased::PorofluidAlgorithm::evaluate_domain_integrals()
{
  // time measurement: evaluation of domain integrals
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:    + evaluate domain integrals");

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::calc_domain_integrals);

  // set vector values needed by elements
  discret_->clear_state();

  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // evaluate
  discret_->evaluate_scalars(eleparams, *domain_integrals_);

  // check for csv writer
  FOUR_C_ASSERT(runtime_csvwriter_domain_integrals_.has_value(),
      "internal error: runtime csv writer not created.");

  // fill csv-output with functions
  std::map<std::string, std::vector<double>> output_data;
  for (int i = 0; i < num_domainint_funct_; i++)
  {
    output_data["Function ID " + std::to_string(domainint_funct_[i])] = {(*domain_integrals_)[i]};
  }
  runtime_csvwriter_domain_integrals_->write_data_to_file(time(), step(), output_data);

  discret_->clear_state();
}


inline void PoroPressureBased::PorofluidAlgorithm::print_convergence_header()
{
  if (myrank_ == 0)
  {
    if (artery_coupling_active_)
    {
      std::cout
          << "+------------+-------------------+--------------+--------------+-------------------+-"
             "-------------+--------------+\n"
          << "|- step/max -|- tol-res   [norm]-|-- pre-res ---|--- 1D-res ---|- "
             "tol-relinc[norm]-|-- pre-inc ---|--- 1D-inc ---|"
          << std::endl;
    }
    else
    {
      std::cout
          << "+------------+-------------------+--------------+-------------------+--------------+-"
             "--"
             "--------------------------+\n"
          << "|- step/max -|- tol-res   [norm]-|-- pre-res ---|- tol-relinc[norm]-|-- pre-inc ---|"
          << std::endl;
    }
  }
}


inline void PoroPressureBased::PorofluidAlgorithm::print_convergence_values_first_iter(
    const int& itnum,                      //!< current Newton-Raphson iteration step
    const int& itemax,                     //!< maximum number of Newton-Raphson iteration steps
    const double& ittol,                   //!< relative tolerance for Newton-Raphson scheme
    const std::vector<double>& preresnorm  //!< L2 norm of pressure residual
)
{
  if (myrank_ == 0)
  {
    using EnumTools::operator<<;
    std::cout << "|  " << std::setw(3) << itnum << "/" << std::setw(3) << itemax << "   | "
              << std::setw(10) << std::setprecision(3) << std::scientific << ittolres_ << " ["
              << std::setw(3) << vectornormfres_ << "]  | ";

    for (std::size_t i = 0; i < preresnorm.size(); ++i)
      std::cout << std::setw(10) << std::setprecision(3) << std::scientific << preresnorm[i]
                << "   | ";

    std::cout << std::setw(10) << std::setprecision(3) << std::scientific << ittolinc_ << " ["
              << std::setw(3) << vectornorminc_ << "]  |";

    for (std::size_t i = 0; i < preresnorm.size(); ++i) std::cout << "      --      |";
    std::cout << " (    --   ,te=" << std::setw(10) << std::setprecision(3) << std::scientific
              << dtele_ << ")" << std::endl;
  }
}


inline void PoroPressureBased::PorofluidAlgorithm::print_convergence_values(
    const int& itnum,                       //!< current Newton-Raphson iteration step
    const int& itemax,                      //!< maximum number of Newton-Raphson iteration steps
    const double& ittol,                    //!< relative tolerance for Newton-Raphson scheme
    const std::vector<double>& preresnorm,  //!< norm of pressure residual
    const std::vector<double>& incprenorm,  //!< norm of pressure increment
    const std::vector<double>& prenorm      //!< norm of pressure state vector
)
{
  if (myrank_ == 0)
  {
    using EnumTools::operator<<;
    std::cout << "|  " << std::setw(3) << itnum << "/" << std::setw(3) << itemax << "   | "
              << std::setw(10) << std::setprecision(3) << std::scientific << ittolres_ << " ["
              << std::setw(3) << vectornormfres_ << "]  | ";
    for (std::size_t i = 0; i < preresnorm.size(); ++i)
      std::cout << std::setw(10) << std::setprecision(3) << std::scientific << preresnorm[i]
                << "   | ";
    std::cout << std::setw(10) << std::setprecision(3) << std::scientific << ittolres_ << " ["
              << std::setw(3) << vectornorminc_ << "]  | ";
    for (std::size_t i = 0; i < preresnorm.size(); ++i)
      std::cout << std::setw(10) << std::setprecision(3) << std::scientific
                << incprenorm[i] / prenorm[i] << "   | ";
    std::cout << std::setw(10) << std::setprecision(3) << std::scientific << dtsolve_
              << ",te=" << std::setw(10) << std::setprecision(3) << std::scientific << dtele_ << ")"
              << std::endl;
  }
}


inline void PoroPressureBased::PorofluidAlgorithm::print_convergence_finish_line()
{
  if (myrank_ == 0)
  {
    if (artery_coupling_active_)
    {
      std::cout << "+------------+-------------------+--------------+--------------+---------------"
                   "----+--------------+--------------+"
                << std::endl;
    }
    else
    {
      std::cout
          << "+------------+-------------------+--------------+-------------------+--------------+"
          << std::endl;
    }
  }
}


inline void PoroPressureBased::PorofluidAlgorithm::increment_time_and_step()
{
  step_ += 1;
  time_ += dt_;
}


void PoroPressureBased::PorofluidAlgorithm::evaluate_error_compared_to_analytical_sol()
{
  if (!calcerr_) return;

  // create the parameters for the error calculation
  Teuchos::ParameterList eleparams;
  eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::calc_error);
  eleparams.set("total time", time_);
  eleparams.set<bool>("calcerrorflag", calcerr_);

  if (calcerr_)
  {
    const int errorfunctnumber = poroparams_.get<int>("CALCERRORNO");
    if (errorfunctnumber < 1)
      FOUR_C_THROW("invalid value of parameter CALCERRORNO for error function evaluation!");

    eleparams.set<int>("error function number", errorfunctnumber);
  }

  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state("phinp_fluid", *phinp_);

  // get (squared) error values
  Core::LinAlg::SerialDenseVector errors(4);
  discret_->evaluate_scalars(eleparams, errors);
  discret_->clear_state();

  // std::vector containing
  // [0]: relative L2 pressure error
  // [1]: relative H1 pressure error
  std::vector<double> relerror(2);

  if (std::abs((errors)[2]) > 1e-14)
    (relerror)[0] = sqrt((errors)[0]) / sqrt((errors)[2]);
  else
    (relerror)[0] = sqrt((errors)[0]);
  if (std::abs((errors)[2]) > 1e-14)
    (relerror)[1] = sqrt((errors)[1]) / sqrt((errors)[3]);
  else
    (relerror)[1] = sqrt((errors)[1]);

  if (myrank_ == 0)
  {
    // print last error in a separate file

    const std::string simulation = Global::Problem::instance()->output_control_file()->file_name();
    const std::string fname = simulation + "_pressure_time.relerror";

    if (step_ == 0)
    {
      std::ofstream f;
      f.open(fname.c_str());
      f << "#| Step | Time | rel. L2-error  | rel. H1-error  |\n";
      f << std::setprecision(10) << step_ << " " << std::setw(1) << std::setprecision(5) << time_
        << std::setw(1) << std::setprecision(6) << " " << (relerror)[0] << std::setw(1)
        << std::setprecision(6) << " " << (relerror)[1] << "\n";

      f.flush();
      f.close();
    }
    else
    {
      std::ofstream f;
      f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
      f << std::setprecision(10) << step_ << " " << std::setw(3) << std::setprecision(5) << time_
        << std::setw(1) << std::setprecision(6) << " " << (relerror)[0] << std::setw(1)
        << std::setprecision(6) << " " << (relerror)[1] << "\n";

      f.flush();
      f.close();
    }
  }
}


void PoroPressureBased::PorofluidAlgorithm::evaluate()
{
  // call elements to calculate system matrix and rhs and assemble
  assemble_mat_and_rhs();

  // perform finite difference check on time integrator level
  if (fdcheck_) fd_check();

  // Apply Dirichlet Boundary Condition
  prepare_system_for_newton_solve();

  if (artery_coupling_active_)
  {
    meshtying_->evaluate();
  }
}


void PoroPressureBased::PorofluidAlgorithm::prepare_system_for_newton_solve()
{
  // Apply Dirichlet boundary conditions to system of equations
  // residual values are supposed to be zero at Dirichlet boundaries
  {
    // time measurement: application of DBC to system
    TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:       + apply DBC to system");

    if (time_ <= starting_dbc_time_end_)
    {
      Core::LinAlg::apply_dirichlet_to_system(
          *sysmat_, *increment_, *residual_, *zeros_, *(dbcmaps_starting_condition_->cond_map()));
    }
    else
    {
      Core::LinAlg::apply_dirichlet_to_system(
          *sysmat_, *increment_, *residual_, *zeros_, *(dbcmaps_with_volfracpress_->cond_map()));
    }
  }
}


void PoroPressureBased::PorofluidAlgorithm::update_iter(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> inc)
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> extractedinc;
  if (artery_coupling_active_)
  {
    extractedinc = meshtying_->extract_and_update_iter(inc);
  }
  else
  {
    extractedinc = inc;
  }

  // store incremental vector to be available for convergence check
  // if incremental vector is received from outside for coupled problem
  increment_->update(1.0, *extractedinc, 0.0);

  // update scalar values by adding increments
  phinp_->update(1.0, *extractedinc, 1.0);

  // compute time derivative at time n+1
  compute_time_derivative();
}


void PoroPressureBased::PorofluidAlgorithm::set_velocity_field(
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel  //!< velocity vector
)
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE: set convective velocity field");

  //---------------------------------------------------------------------------
  // preliminaries
  //---------------------------------------------------------------------------
  if (nds_vel_ == -1)
    FOUR_C_THROW("Dof set for velocity degreess of freedom has not been assigned!");

  if (vel == nullptr) FOUR_C_THROW("Velocity state is nullptr");

  if (nds_vel_ >= discret_->num_dof_sets())
    FOUR_C_THROW("Too few dofsets on poro fluid discretization!");

  if (not vel->get_map().same_as(*discret_->dof_row_map(nds_vel_)))
    FOUR_C_THROW(
        "Map of given velocity and associated dof row map in poro fluid discretization"
        " do not match!");

  // provide discretization with velocity
  set_state(nds_vel_, "velocity field", vel);
}


void PoroPressureBased::PorofluidAlgorithm::set_state(unsigned nds, const std::string& name,
    std::shared_ptr<const Core::LinAlg::Vector<double>> state)
{
  // provide discretization with velocity
  discret_->set_state(nds, name, *state);
}


void PoroPressureBased::PorofluidAlgorithm::set_initial_field(
    const PoroPressureBased::InitialField init, const int startfuncno)
{
  switch (init)
  {
    case InitialField::zero:
    {
      phin_->put_scalar(0.0);
      phinp_->put_scalar(0.0);
      break;
    }
    case InitialField::by_function:
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      for (auto lnode : discret_->my_row_node_range())
      {
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        int numdofs = nodedofset.size();
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->lid(dofgid);
          // evaluate component k of spatial function
          double initialval = Global::Problem::instance()
                                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(startfuncno)
                                  .evaluate(lnode.x(), time_, k);
          phin_->replace_local_value(doflid, initialval);
          ;
        }
      }

      // initialize also the solution vector. These values are a pretty good guess for the
      // solution after the first time step (much better than starting with a zero vector)
      phinp_->update(1.0, *phin_, 0.0);

      break;
    }
    case InitialField::by_condition:
    {
      // set initial field for ALL existing scatra fields in condition
      const std::string field = "PoroMultiFluid";

      const int numdof = discret_->num_dof(0, discret_->l_row_node(0));

      // get initial field conditions
      std::vector<const Core::Conditions::Condition*> initfieldconditions;
      discret_->get_condition("Initfield", initfieldconditions);

      if (not initfieldconditions.size())
        FOUR_C_THROW(
            "Tried to evaluate initial field by condition without a corresponding condition "
            "defined on the PoroMultiFluid discretization!");
      std::set<int> numdofpernode;
      for (unsigned icond = 0; icond < initfieldconditions.size(); icond++)
      {
        const int condmaxnumdofpernode = numdof;

        if (condmaxnumdofpernode != 0) numdofpernode.insert(condmaxnumdofpernode);
      }

      if (numdofpernode.empty()) FOUR_C_THROW("No DOFs defined on initial field condition!");

      const int maxnumdofpernode = *(numdofpernode.rbegin());

      std::vector<int> localdofs(maxnumdofpernode);
      for (int i = 0; i < maxnumdofpernode; i++)
      {
        localdofs[i] = i;
      }
      discret_->evaluate_initial_field(
          Global::Problem::instance()->function_manager(), field, *phin_, localdofs);

      // initialize also the solution vector. These values are a pretty good guess for the
      // solution after the first time step (much better than starting with a zero vector)
      phinp_->update(1.0, *phin_, 0.0);

      break;
    }
    default:
      FOUR_C_THROW("Unknown option for initial field: {}", init);
      break;
  }
}


std::shared_ptr<Core::Utils::ResultTest> PoroPressureBased::PorofluidAlgorithm::create_field_test()
{
  if (artery_coupling_active_)
  {
    meshtying_->create_result_test();
  }
  return std::make_shared<PoroPressureBased::ResultTest>(*this);
}


void PoroPressureBased::PorofluidAlgorithm::read_restart(const int step)
{
  if (artery_coupling_active_)
  {
    meshtying_->read_restart(step);
  }

  std::shared_ptr<Core::IO::DiscretizationReader> reader(nullptr);
  reader = std::make_shared<Core::IO::DiscretizationReader>(
      *discret_, Global::Problem::instance()->input_control_file(), step);

  time_ = reader->read_double("time");
  step_ = reader->read_int("step");

  if (myrank_ == 0)
    std::cout << "Reading POROFLUIDMULTIPHASE restart data (time=" << time_ << " ; step=" << step_
              << ")" << '\n';

  // read state vectors that are needed for One-Step-Theta restart
  reader->read_vector(phinp_, "phinp_fluid");
  reader->read_vector(phin_, "phin_fluid");
  reader->read_vector(phidtn_, "phidtn_fluid");
}


void PoroPressureBased::PorofluidAlgorithm::calc_initial_time_derivative()
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("POROFLUIDMULTIPHASE:       + calculate initial time derivative");

  // initial screen output
  if (myrank_ == 0)
    std::cout << "POROFLUIDMULTIPHASE: calculating initial time derivative of state variables on "
                 "discretization \""
              << discret_->name().c_str() << "\" (step " << step() << ", time " << time()
              << ") ... ... ";

  // standard general element parameter without stabilization
  set_element_general_parameters();

  // we also have to modify the time-parameter list (incremental solve)
  // actually we do not need a time integration scheme for calculating the initial time derivatives,
  // but the rhs of the standard element routine is used as starting point for this special system
  // of equations. Therefore, the rhs vector has to be scaled correctly.
  set_element_time_step_parameter();

  // reset global system matrix
  sysmat_->zero();

  // reset the residual vector
  residual_->put_scalar(0.0);

  // evaluate Dirichlet and Neumann boundary conditions at time t = 0 to ensure consistent
  // computation of initial time derivative vector Dirichlet boundary conditions should be
  // consistent with initial field
  apply_dirichlet_bc(time_, phinp_, nullptr);
  apply_neumann_bc(*neumann_loads_);

  // create and fill parameter list for elements
  Teuchos::ParameterList eleparams;
  eleparams.set<PoroPressureBased::Action>("action", PoroPressureBased::calc_initial_time_deriv);

  // add state vectors according to time integration scheme
  discret_->clear_state();
  add_time_integration_specific_vectors();

  // We evaluate the discretization such that
  // mp * dphidt + msp * dphidt + msat * dphidt = - rhsfac (sdivvel + diff + reac)
  // (      only    matrix                    )   (          only rhs              )
  // later we will also have to scale the system matrix with rhsfac
  discret_->evaluate(eleparams, sysmat_, residual_);
  discret_->clear_state();

  // potential residual scaling and potential addition of Neumann terms
  scaling_and_neumann();

  // We have to Scale the system matrix consistently
  // TODO: this is kind of a hack, does it work for other schemes than one-step theta??
  // sysmat_->Scale(1.0/residual_scaling());
  residual_->scale(residual_scaling());

  // finalize assembly of system matrix
  sysmat_->complete();

  // solve global system of equations for initial time derivative of state variables
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->solve(sysmat_, phidtnp_, residual_, solver_params);

  // copy solution
  phidtn_->update(1.0, *phidtnp_, 0.0);

  // reset global system matrix and its graph, since we solved a very special problem with a special
  // sparsity pattern
  sysmat_->reset();

  // reset solver
  solver_->reset();

  // reset true residual vector computed during assembly of the standard global system of equations,
  // since not yet needed
  trueresidual_->put_scalar(0.0);

  double maxval = 0.0;
  phidtnp_->max_value(&maxval);
  // final screen output
  if (myrank_ == 0)
  {
    std::cout << "done!" << std::endl;
    std::cout << "MAX value: " << maxval << std::endl;
  }

  // and finally undo our temporary settings
  set_element_general_parameters();
  set_element_time_step_parameter();
}


void PoroPressureBased::PorofluidAlgorithm::fd_check()
{
  // initial screen output
  if (myrank_ == 0)
    std::cout << std::endl << "FINITE DIFFERENCE CHECK FOR SYSTEM MATRIX" << std::endl;

  // make a copy of state variables to undo perturbations later
  Core::LinAlg::Vector<double> phinp_original(*phinp_);

  std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_original = nullptr;
  if (std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_) != nullptr)
    sysmat_original = std::make_shared<Core::LinAlg::SparseMatrix>(
        *(std::static_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_)));

  else if (std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat_) != nullptr)
    sysmat_original = std::make_shared<Core::LinAlg::SparseMatrix>(
        *(std::static_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat_)->merge()));
  else
    FOUR_C_THROW("Type of system matrix unknown!");
  sysmat_original->complete();

  // make a copy of system right-hand side vector
  Core::LinAlg::Vector<double> rhs_original(*residual_);

  // initialize counter for system matrix entries with failing finite difference check
  int counter(0);

  // initialize tracking variable for maximum absolute and relative errors
  double maxabserr(0.);
  double maxrelerr(0.);

  for (int colgid = 0; colgid <= sysmat_original->col_map().max_all_gid(); ++colgid)
  {
    // check whether current column index is a valid global column index and continue loop if not
    int collid(sysmat_original->col_map().lid(colgid));
    int maxcollid(-1);
    maxcollid = Core::Communication::max_all(collid, discret_->get_comm());
    if (maxcollid < 0) continue;

    // fill state vector with original state variables
    phinp_->update(1., phinp_original, 0.);

    // impose perturbation
    if (phinp_->get_map().my_gid(colgid)) phinp_->sum_into_global_value(colgid, fdcheckeps_);

    compute_time_derivative();

    // calculate element right-hand side vector for perturbed state
    assemble_mat_and_rhs();

    // Now we compare the difference between the current entries in the system matrix
    // and their finite difference approximations according to
    // entries ?= (residual_perturbed - residual_original) / epsilon

    // Note that the residual_ vector actually denotes the right-hand side of the linear
    // system of equations, i.e., the negative system residual.
    // To account for errors due to numerical cancellation, we additionally consider
    // entries + residual_original / epsilon ?= residual_perturbed / epsilon

    // Note that we still need to evaluate the first comparison as well. For small entries in the
    // system matrix, the second comparison might yield good agreement in spite of the entries being
    // wrong!
    for (int rowlid = 0; rowlid < discret_->dof_row_map()->num_my_elements(); ++rowlid)
    {
      // get global index of current matrix row
      const int rowgid = sysmat_original->row_map().gid(rowlid);
      if (rowgid < 0) FOUR_C_THROW("Invalid global ID of matrix row!");

      // get relevant entry in current row of original system matrix
      double entry(0.);
      int length = sysmat_original->num_my_entries(rowlid);
      int numentries;
      std::vector<double> values(length);
      std::vector<int> indices(length);
      sysmat_original->extract_my_row_copy(
          rowlid, length, numentries, values.data(), indices.data());
      for (int ientry = 0; ientry < length; ++ientry)
      {
        if (sysmat_original->col_map().gid(indices[ientry]) == colgid)
        {
          entry = values[ientry];
          break;
        }
      }

      // finite difference suggestion (first divide by epsilon and then add for better conditioning)
      const double fdval = -residual_->local_values_as_span()[rowlid] / fdcheckeps_ +
                           rhs_original.local_values_as_span()[rowlid] / fdcheckeps_;

      // confirm accuracy of first comparison
      if (abs(fdval) > 1.e-17 and abs(fdval) < 1.e-15)
        FOUR_C_THROW("Finite difference check involves values too close to numerical zero!");

      // absolute and relative errors in first comparison
      const double abserr1 = entry - fdval;
      if (abs(abserr1) > maxabserr) maxabserr = abs(abserr1);
      double relerr1(0.);
      if (abs(entry) > 1.e-17)
        relerr1 = abserr1 / abs(entry);
      else if (abs(fdval) > 1.e-17)
        relerr1 = abserr1 / abs(fdval);
      if (abs(relerr1) > maxrelerr) maxrelerr = abs(relerr1);

      // evaluate first comparison
      if (abs(relerr1) > fdchecktol_)
      {
        std::cout << "sysmat[" << rowgid << "," << colgid << "]:  " << entry << "   ";
        std::cout << "finite difference suggestion:  " << fdval << "   ";
        std::cout << "absolute error:  " << abserr1 << "   ";
        std::cout << "relative error:  " << relerr1 << std::endl;

        counter++;
      }

      // first comparison OK
      else
      {
        // left-hand side in second comparison
        const double left = entry - rhs_original.local_values_as_span()[rowlid] / fdcheckeps_;

        // right-hand side in second comparison
        const double right = -residual_->local_values_as_span()[rowlid] / fdcheckeps_;

        // confirm accuracy of second comparison
        if (abs(right) > 1.e-17 and abs(right) < 1.e-15)
          FOUR_C_THROW("Finite difference check involves values too close to numerical zero!");

        // absolute and relative errors in second comparison
        const double abserr2 = left - right;
        if (abs(abserr2) > maxabserr) maxabserr = abs(abserr2);
        double relerr2(0.);
        if (abs(left) > 1.e-17)
          relerr2 = abserr2 / abs(left);
        else if (abs(right) > 1.e-17)
          relerr2 = abserr2 / abs(right);
        if (abs(relerr2) > maxrelerr) maxrelerr = abs(relerr2);

        // evaluate second comparison
        if (abs(relerr2) > fdchecktol_)
        {
          std::cout << "sysmat[" << rowgid << "," << colgid << "]-rhs[" << rowgid
                    << "]/eps:  " << left << "   ";
          std::cout << "-rhs_perturbed[" << rowgid << "]/eps:  " << right << "   ";
          std::cout << "absolute error:  " << abserr2 << "   ";
          std::cout << "relative error:  " << relerr2 << std::endl;

          counter++;
        }
      }
    }
  }

  // communicate tracking variables
  int counterglobal(0);
  counterglobal = Core::Communication::sum_all(counter, discret_->get_comm());
  double maxabserrglobal(0.);
  maxabserrglobal = Core::Communication::max_all(maxabserr, discret_->get_comm());
  double maxrelerrglobal(0.);
  maxrelerrglobal = Core::Communication::max_all(maxrelerr, discret_->get_comm());

  // final screen output
  if (myrank_ == 0)
  {
    if (counterglobal)
    {
      printf(
          "--> FAILED AS LISTED ABOVE WITH %d CRITICAL MATRIX ENTRIES IN TOTAL\n\n", counterglobal);
      FOUR_C_THROW("Finite difference check failed for scalar transport system matrix!");
    }
    else
      printf(
          "--> PASSED WITH MAXIMUM ABSOLUTE ERROR %+12.5e AND MAXIMUM RELATIVE ERROR %+12.5e\n\n",
          maxabserrglobal, maxrelerrglobal);
  }

  // undo perturbations of state variables
  phinp_->update(1., phinp_original, 0.);
  compute_time_derivative();

  // recompute system matrix and right-hand side vector based on original state variables
  assemble_mat_and_rhs();
}


std::shared_ptr<Adapter::ArtNet> PoroPressureBased::PorofluidAlgorithm::art_net_tim_int()
{
  FOUR_C_ASSERT(artery_coupling_active_,
      "art_net_tim_int can only be called when artery coupling is active.");
  return meshtying_->artery_algorithm();
}

FOUR_C_NAMESPACE_CLOSE
