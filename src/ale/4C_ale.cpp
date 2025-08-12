// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale.hpp"

#include "4C_ale_input.hpp"
#include "4C_ale_meshsliding.hpp"
#include "4C_ale_resulttest.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
ALE::Ale::Ale(std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : discret_(actdis),
      solver_(solver),
      params_(params),
      output_(output),
      step_(0),
      numstep_(params_->get<int>("NUMSTEP")),
      time_(0.0),
      maxtime_(params_->get<double>("MAXTIME")),
      dt_(params_->get<double>("TIMESTEP")),
      writerestartevery_(params->get<int>("RESTARTEVERY")),
      writeresultsevery_(params->get<int>("RESULTSEVERY")),
      sysmat_(nullptr),
      residual_(nullptr),
      rhs_(nullptr),
      dispnp_(nullptr),
      dispn_(nullptr),
      disi_(nullptr),
      zeros_(nullptr),
      eledetjac_(nullptr),
      elequality_(nullptr),
      elequalityyesno_(params->get<bool>("ASSESSMESHQUALITY")),
      aletype_(Teuchos::getIntegralValue<ALE::AleDynamic>(*params, "ALE_TYPE")),
      maxiter_(params->get<int>("MAXITER")),
      tolres_(params->get<double>("TOLRES")),
      toldisp_(params->get<double>("TOLDISP")),
      divercont_(Teuchos::getIntegralValue<ALE::DivContAct>(*params, "DIVERCONT")),
      msht_(Teuchos::getIntegralValue<ALE::MeshTying>(*params, "MESHTYING")),
      initialdisp_(Teuchos::getIntegralValue<ALE::InitialDisp>(*params, "INITIALDISP")),
      startfuncno_(params->get<int>("STARTFUNCNO"))
{
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  dispn_ = Core::LinAlg::create_vector(*dofrowmap, true);
  dispnp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  disi_ = Core::LinAlg::create_vector(*dofrowmap, true);
  residual_ = Core::LinAlg::create_vector(*dofrowmap, true);
  rhs_ = Core::LinAlg::create_vector(*dofrowmap, true);
  zeros_ = Core::LinAlg::create_vector(*dofrowmap, true);

  eledetjac_ = Core::LinAlg::create_vector(*discretization()->element_row_map(), true);
  elequality_ = Core::LinAlg::create_vector(*discretization()->element_row_map(), true);

  // -------------------------------------------------------------------
  // set initial displacement
  // -------------------------------------------------------------------
  set_initial_displacement(initialdisp_, startfuncno_);

  setup_dbc_map_ex();

  // ensure that the ALE string was removed from conditions
  FOUR_C_ASSERT_ALWAYS(!discret_->has_condition("ALEDirichlet"),
      "Found a ALE Dirichlet condition. Remove ALE string!");

  if (msht_ == ALE::meshsliding)
  {
    meshtying_ = std::make_shared<Meshsliding>(
        discret_, *solver_, msht_, Global::Problem::instance()->n_dim(), nullptr);
  }
  else if (msht_ == ALE::meshtying)
  {
    meshtying_ = std::make_shared<Meshtying>(
        discret_, *solver_, msht_, Global::Problem::instance()->n_dim(), nullptr);
  }

  // ---------------------------------------------------------------------
  // Create LocSysManager, if needed (used for LocSys-Dirichlet BCs)
  // ---------------------------------------------------------------------
  {
    std::vector<const Core::Conditions::Condition*> locsysconditions;
    discret_->get_condition("Locsys", locsysconditions);
    if (locsysconditions.size())
    {
      // Initialize locsys manager
      locsysman_ = std::make_shared<Core::Conditions::LocsysManager>(
          *discret_, Global::Problem::instance()->n_dim());
    }
  }

  create_system_matrix();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ALE::Ale::set_initial_displacement(const ALE::InitialDisp init, const int startfuncno)
{
  switch (init)
  {
    case ALE::initdisp_zero_disp:
    {
      dispn_->put_scalar(0.0);
      dispnp_->put_scalar(0.0);

      break;
    }
    case ALE::initdisp_disp_by_function:
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      // loop all nodes on the processor
      for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
      {
        // get the processor local node
        Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);

        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        // loop nodal dofs
        for (unsigned int d = 0; d < nodedofset.size(); ++d)
        {
          const int dofgid = nodedofset[d];
          int doflid = dofrowmap->lid(dofgid);

          // evaluate component d of function
          double initialval = Global::Problem::instance()
                                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(startfuncno)
                                  .evaluate(lnode->x().data(), 0, d);

          int err = dispn_->replace_local_value(doflid, initialval);
          if (err != 0) FOUR_C_THROW("dof not on proc");
        }
      }

      // initialize also the solution vector
      dispnp_->update(1.0, *dispn_, 0.0);

      break;
    }
    default:
      FOUR_C_THROW("Unknown option for initial displacement: {}", init);
      break;
  }

  return;
}
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::create_system_matrix(std::shared_ptr<const ALE::Utils::MapExtractor> interface)
{
  if (msht_ != ALE::no_meshtying)
  {
    std::vector<int> coupleddof(Global::Problem::instance()->n_dim(), 1);
    sysmat_ = meshtying_->setup(coupleddof, dispnp_);
    meshtying_->dirichlet_on_master(dbcmaps_[ALE::Utils::MapExtractor::dbc_set_std]->cond_map());

    if (interface != nullptr)
    {
      meshtying_->is_multifield(*interface, true);
    }
  }
  else if (interface == nullptr)
  {
    const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
    sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 81, false, true);
  }
  else
  {
    sysmat_ =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            *interface, *interface, 81, false, true);
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> stepinc,
    ALE::Utils::MapExtractor::AleDBCSetType dbc_type)
{
  // We save the current solution here. This will not change the
  // result of our element call, but the next time somebody asks us we
  // know the displacements.
  //
  // Note: What we get here is the sum of all increments in this time
  // step, not just the latest increment.

  disi_->put_scalar(0.0);

  if (stepinc != nullptr)
  {
    dispnp_->update(1.0, *stepinc, 1.0, *dispn_, 0.0);
  }

  if (msht_ != ALE::no_meshtying)
  {
    meshtying_->msht_split(sysmat_);
  }

  evaluate_elements();
  evaluate_element_quality();

  // prepare meshtying system
  if (msht_ != ALE::no_meshtying)
  {
    meshtying_->prepare_meshtying_system(sysmat_, residual_, dispnp_);
    meshtying_->multifield_split(sysmat_);
  }

  // dispnp_ has zeros at the Dirichlet-entries, so we maintain zeros there.
  if (locsys_manager() != nullptr)
  {
    // Transform system matrix and rhs to local coordinate systems
    locsys_manager()->rotate_global_to_local(
        std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_), *residual_);

    // When using local systems, a rotated dispnp_ vector needs to be used as dbcval for
    // apply_dirichlet_to_system
    Core::LinAlg::Vector<double> dispnp_local(*(zeros_));
    locsys_manager()->rotate_global_to_local(dispnp_local);

    if (get_loc_sys_trafo() != nullptr)
    {
      Core::LinAlg::apply_dirichlet_to_system(
          *Core::LinAlg::cast_to_sparse_matrix_and_check_success(sysmat_), *disi_, *residual_,
          *get_loc_sys_trafo(), dispnp_local, *(dbcmaps_[dbc_type]->cond_map()));
    }
    else
    {
      Core::LinAlg::apply_dirichlet_to_system(
          *sysmat_, *disi_, *residual_, dispnp_local, *(dbcmaps_[dbc_type]->cond_map()));
    }
  }
  else
  {
    Core::LinAlg::apply_dirichlet_to_system(
        *sysmat_, *disi_, *residual_, *zeros_, *(dbcmaps_[dbc_type]->cond_map()));
  }

  /* residual_ contains the most recent "mechanical" residual including DBCs.
   * We make this negative and store it in rhs_ for use in Newton-type methods.
   */
  rhs_->update(-1.0, *residual_, 0.0);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int ALE::Ale::solve()
{
  // We need the negative residual here as right hand side of the linear problem
  std::shared_ptr<Core::LinAlg::Vector<double>> rhs =
      std::make_shared<Core::LinAlg::Vector<double>>(*residual_);
  rhs->scale(-1.0);

  // ToDo (mayr) Why can't we use rhs_ instead of local variable rhs???
  int errorcode = 0;
  if (msht_ == ALE::no_meshtying)
  {
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    errorcode = solver_->solve(sysmat_, disi_, rhs, solver_params);
  }
  else
    errorcode = meshtying_->solve_meshtying(*solver_, sysmat_, disi_, rhs, dispnp_);
  // calc norm
  disi_->norm_2(&normdisi_);
  normdisi_ /= sqrt(disi_->global_length());

  return errorcode;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::update_iter() { dispnp_->update(1.0, *disi_, 1.0); }

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::update() { dispn_->update(1.0, *dispnp_, 0.0); }

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
bool ALE::Ale::converged(const int iter)
{
  if (iter == 0) normdisi_ = 0.0;

  bool converged = false;
  // determine norms
  double res_norm;
  residual_->norm_2(&res_norm);
  res_norm /= sqrt(residual_->global_length());
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    std::cout << "ITER: " << iter << "  RES NORM: " << res_norm << " DISP NORM: " << normdisi_
              << std::endl;

  if (res_norm < tolres_ && normdisi_ < toldisp_)
  {
    converged = true;
  }
  return converged;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::evaluate_elements()
{
  sysmat_->zero();

  // zero out residual
  residual_->put_scalar(0.0);

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // set vector values needed by elements
  discret_->clear_state();

  // action for elements
  eleparams.set<std::string>("action", element_action_string(aletype_));
  eleparams.set<bool>("use spatial configuration", update_sys_mat_every_step());

  discret_->set_state("dispnp", *dispnp_);

  discret_->evaluate(eleparams, sysmat_, residual_);
  discret_->clear_state();

  sysmat_->complete();
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::string ALE::Ale::element_action_string(const enum ALE::AleDynamic name)
{
  switch (name)
  {
    case ALE::solid:
      return "calc_ale_solid";
      break;
    case ALE::solid_linear:
      return "calc_ale_solid_linear";
      break;
    case ALE::laplace_material:
      return "calc_ale_laplace_material";
      break;
    case ALE::laplace_spatial:
      return "calc_ale_laplace_spatial";
      break;
    case ALE::springs_material:
      return "calc_ale_springs_material";
      break;
    case ALE::springs_spatial:
      return "calc_ale_springs_spatial";
      break;
    default:
      FOUR_C_THROW("Cannot make std::string for ALE type {}", name);
      return "";
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> ALE::Ale::dof_row_map() const
{
  return Core::Utils::shared_ptr_from_ref(*discret_->dof_row_map());
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> ALE::Ale::system_matrix()
{
  return std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> ALE::Ale::block_system_matrix()
{
  return std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat_);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int ALE::Ale::integrate()
{
  // add eps to prevent stopping one step too early due to memory trash on last digits
  const double eps = 1.0e-12;
  while (step_ < numstep_ and time_ <= maxtime_ + eps)
  {
    prepare_time_step();
    time_step();
    update();
    output();
  }

  return 0;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::output()
{
  /*  We need ALE output only in case of pure ALE problems. If fluid is present,
   *  the fluid field writes its own displacement field as output.
   *
   *  Though, we might need restart data.
   */

  // Has any output data been written?
  bool datawritten = false;

  // write restart data if necessary
  if (writerestartevery_ != 0 and step_ % writerestartevery_ == 0)
  {
    output_restart(datawritten);
  }

  // write output data if necessary
  if (not datawritten and writeresultsevery_ != 0 and step_ % writeresultsevery_ == 0)
  {
    output_state(datawritten);
  }

  // write domain decomposition for visualization
  if ((step_ == writeresultsevery_ or step_ == 0)) output_->write_element_data(true);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::output_state(bool& datawritten)
{
  // write output data
  output_->new_step(step_, time_);
  output_->write_vector("dispnp", dispnp_);

  if (elequalityyesno_)
  {
    output_->write_vector("det_j", eledetjac_, Core::IO::elementvector);
    output_->write_vector("element_quality", elequality_, Core::IO::elementvector);
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::output_restart(bool& datawritten)
{
  // write restart data
  output_->new_step(step_, time_);
  output_->write_vector("dispnp", dispnp_);
  output_->write_vector("dispn", dispn_);

  // restart/output data has been written
  datawritten = true;

  // info dedicated to user's eyes staring at standard out
  // Print restart info only in case of pure ALE problem. Coupled problems
  // print their own restart info.
  if (Global::Problem::instance()->get_problem_type() == Core::ProblemType::ale)
  {
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
      Core::IO::cout << "====== Restart written in step " << step_ << Core::IO::endl;
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::read_restart(const int step)
{
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step);
  time_ = reader.read_double("time");
  step_ = reader.read_int("step");

  reader.read_vector(dispnp_, "dispnp");
  reader.read_vector(dispn_, "dispn");
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::prepare_time_step()
{
  step_ += 1;
  time_ += dt_;

  // Print time step header only in case of pure ALE problem. Coupled problems
  // print their own time step header.
  if (Global::Problem::instance()->get_problem_type() == Core::ProblemType::ale)
    print_time_step_header();

  // Update local coordinate systems (which may be time dependent)
  if (locsysman_ != nullptr)
  {
    discret_->clear_state();
    discret_->set_state("dispnp", *dispnp_);
    locsysman_->update(time_, {}, Global::Problem::instance()->function_manager());
    discret_->clear_state();
  }

  Teuchos::ParameterList eleparams;
  eleparams.set("total time", time_);
  eleparams.set("delta time", dt_);
  eleparams.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // Apply Dirichlet boundary conditions on provided state vector
  ALE::Ale::apply_dirichlet_bc(eleparams, dispnp_, nullptr, nullptr, false);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::time_step(ALE::Utils::MapExtractor::AleDBCSetType dbc_type)
{
  bool is_converged = false;
  int iter = 0;

  // Newton loop to deal with possible non-linearities
  while (!is_converged && iter < maxiter_)
  {
    evaluate(nullptr, dbc_type);

    if (converged(iter))
    {
      is_converged = true;
      continue;
    }
    solve();
    update_iter();
    ++iter;
  }

  if (!is_converged)
  {
    switch (divercont_)
    {
      case ALE::divcont_stop:
        FOUR_C_THROW("ALE newton not converged in {} iterations. Abort! ", maxiter_);
        break;
      case ALE::divcont_continue:
        if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
        {
          Core::IO::cout << "ALE newton not converged in " << maxiter_ << " iterations. Continue"
                         << Core::IO::endl;
        }
        break;
      default:
        FOUR_C_THROW("Unknown divercont action! ");
        break;
    }
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::print_time_step_header() const
{
  Core::IO::cout << "TIME: " << time_ << "/" << maxtime_ << "  DT = " << dt_ << "  STEP = " << step_
                 << "/" << numstep_ << Core::IO::endl;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::setup_dbc_map_ex(ALE::Utils::MapExtractor::AleDBCSetType dbc_type,
    std::shared_ptr<const ALE::Utils::MapExtractor> interface,
    std::shared_ptr<const ALE::Utils::XFluidFluidMapExtractor> xff_interface)
{
  // set fixed nodes (conditions != 0 are not supported right now). hahn: Why?!
  Teuchos::ParameterList eleparams;
  eleparams.set("total time", time_);
  eleparams.set("delta time", dt_);
  eleparams.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // some consistency checks
  if (interface == nullptr && dbc_type != ALE::Utils::MapExtractor::dbc_set_std &&
      dbc_type != ALE::Utils::MapExtractor::dbc_set_x_ff)
    FOUR_C_THROW(
        "For non-standard use of SetupDBCMapEx, please provide a valid ALE::Utils::MapExtractor.");

  if (xff_interface == nullptr && dbc_type == ALE::Utils::MapExtractor::dbc_set_x_ff)
    FOUR_C_THROW(
        "For non-standard use of SetupDBCMapEx with fluid-fluid coupling, please provide a "
        "x_fluid_fluid_map_extractor.");
  // REMARK: for all applications, setup of the standard Dirichlet sets is done in the ctor of this
  // class

  switch (dbc_type)
  {
    case ALE::Utils::MapExtractor::dbc_set_std:
      dbcmaps_[ALE::Utils::MapExtractor::dbc_set_std] =
          std::make_shared<Core::LinAlg::MapExtractor>();
      apply_dirichlet_bc(eleparams, dispnp_, nullptr, nullptr, true);
      break;
    case ALE::Utils::MapExtractor::dbc_set_x_ff:
    {
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> condmaps;
      condmaps.push_back(xff_interface->xfluid_fluid_cond_map());
      condmaps.push_back(dbcmaps_[ALE::Utils::MapExtractor::dbc_set_std]->cond_map());
      std::shared_ptr<Core::LinAlg::Map> condmerged =
          Core::LinAlg::MultiMapExtractor::merge_maps(condmaps);

      dbcmaps_[ALE::Utils::MapExtractor::dbc_set_x_ff] =
          std::make_shared<Core::LinAlg::MapExtractor>(*(discret_->dof_row_map()), condmerged);
      break;
    }
    case ALE::Utils::MapExtractor::dbc_set_x_fsi:
    case ALE::Utils::MapExtractor::dbc_set_biofilm:
    case ALE::Utils::MapExtractor::dbc_set_part_fsi:
    {
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> condmaps;
      condmaps.push_back(interface->fsi_cond_map());
      condmaps.push_back(dbcmaps_[ALE::Utils::MapExtractor::dbc_set_std]->cond_map());
      std::shared_ptr<Core::LinAlg::Map> condmerged =
          Core::LinAlg::MultiMapExtractor::merge_maps(condmaps);

      dbcmaps_[dbc_type] =
          std::make_shared<Core::LinAlg::MapExtractor>(*(discret_->dof_row_map()), condmerged);
      break;
    }
    case ALE::Utils::MapExtractor::dbc_set_wear:
    {
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> condmaps;
      condmaps.push_back(interface->ale_wear_cond_map());
      condmaps.push_back(dbcmaps_[ALE::Utils::MapExtractor::dbc_set_std]->cond_map());

      std::shared_ptr<Core::LinAlg::Map> condmerged =
          Core::LinAlg::MultiMapExtractor::merge_maps(condmaps);
      dbcmaps_[ALE::Utils::MapExtractor::dbc_set_wear] =
          std::make_shared<Core::LinAlg::MapExtractor>(*(discret_->dof_row_map()), condmerged);
      break;
    }
    default:
      FOUR_C_THROW("Undefined type of ALE Dirichlet sets.");
      break;
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> ALE::Ale::create_field_test()
{
  return std::make_shared<ALE::AleResultTest>(*this);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::apply_dirichlet_bc(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvectord,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvectordd, bool recreatemap)
{
  // In the case of local coordinate systems, we have to rotate forward ...
  // ---------------------------------------------------------------------------
  if (locsysman_ != nullptr)
  {
    if (systemvector != nullptr) locsysman_->rotate_global_to_local(*systemvector);
    if (systemvectord != nullptr) locsysman_->rotate_global_to_local(*systemvectord);
    if (systemvectordd != nullptr) locsysman_->rotate_global_to_local(*systemvectordd);
  }

  // Apply DBCs
  // ---------------------------------------------------------------------------
  discret_->clear_state();
  if (recreatemap)
  {
    discret_->evaluate_dirichlet(params, systemvector, systemvectord, systemvectordd, nullptr,
        dbcmaps_[ALE::Utils::MapExtractor::dbc_set_std]);
  }
  else
  {
    discret_->evaluate_dirichlet(
        params, systemvector, systemvectord, systemvectordd, nullptr, nullptr);
  }
  discret_->clear_state();

  // In the case of local coordinate systems, we have to rotate back into global Cartesian frame
  // ---------------------------------------------------------------------------
  if (locsysman_ != nullptr)
  {
    if (systemvector != nullptr) locsysman_->rotate_local_to_global(*systemvector);
    if (systemvectord != nullptr) locsysman_->rotate_local_to_global(*systemvectord);
    if (systemvectordd != nullptr) locsysman_->rotate_local_to_global(*systemvectordd);
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::reset()
{
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  dispnp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  dispn_ = Core::LinAlg::create_vector(*dofrowmap, true);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::reset_step()
{
  dispnp_->update(1.0, *dispn_, 0.0);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::reset_time(const double dtold)
{
  time_ = time_ - dtold;
  step_ = step_ - 1;

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::set_dt(const double dtnew)
{
  dt_ = dtnew;

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::SparseMatrix> ALE::Ale::get_loc_sys_trafo() const
{
  if (locsysman_ != nullptr) return locsysman_->trafo();

  return nullptr;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Ale::update_slave_dof(std::shared_ptr<Core::LinAlg::Vector<double>>& a)
{
  if (msht_ != ALE::no_meshtying)
  {
    meshtying_->update_slave_dof(a, dispnp_);
    meshtying_->recover(a);
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
bool ALE::Ale::evaluate_element_quality()
{
  if (elequalityyesno_)
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    discret_->clear_state();
    discret_->set_state("dispnp", *dispnp_);

    for (int i = 0; i < discretization()->num_my_row_elements(); ++i)
    {
      Core::Elements::Element* actele;
      actele = discretization()->l_row_element(i);

      // list to define routines at elementlevel
      Teuchos::ParameterList eleparams;
      eleparams.set("action", "calc_jacobian_determinant");

      // initialize element vectors
      Core::Elements::LocationArray la(discretization()->num_dof_sets());
      actele->location_vector(*discretization(), la);

      // only two entries per element necessary (detJ and quality measure)
      Core::LinAlg::SerialDenseMatrix elematrix1;
      Core::LinAlg::SerialDenseMatrix elematrix2;
      Core::LinAlg::SerialDenseVector elevector1(2);
      Core::LinAlg::SerialDenseVector elevector2;
      Core::LinAlg::SerialDenseVector elevector3;

      actele->evaluate(
          eleparams, *discret_, la, elematrix1, elematrix2, elevector1, elevector2, elevector3);

      eledetjac_->replace_local_value(i, elevector1[0]);
      elequality_->replace_local_value(i, elevector1[1]);

    }  // loop elements

    discret_->clear_state();

    // check for non-valid elements
    bool validshapes = true;
    double negdetjac = 0.0;
    eledetjac_->min_value(&negdetjac);
    if (negdetjac <= 0)
    {
      validshapes = false;
      FOUR_C_THROW("Negative determinant {} in time step {}", negdetjac, step_);
    }

    return validshapes;
  }
  else
  {
    // no assessment of mesh quality. Return true to assume that everything is fine.
    return true;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// class AleLinear ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*----------------------------------------------------------------------------*/
ALE::AleLinear::AleLinear(std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params_in,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : Ale(actdis, solver, params_in, output), validsysmat_(false), updateeverystep_(false)
{
  updateeverystep_ = params().get<bool>("UPDATEMATRIX");
}

/*----------------------------------------------------------------------------*/
void ALE::AleLinear::prepare_time_step()
{
  Ale::prepare_time_step();

  if (updateeverystep_) validsysmat_ = false;

  return;
}

/*----------------------------------------------------------------------------*/
void ALE::AleLinear::time_step(ALE::Utils::MapExtractor::AleDBCSetType dbc_type)
{
  evaluate(nullptr, dbc_type);
  solve();
  update_iter();

  return;
}

/*----------------------------------------------------------------------------*/
void ALE::AleLinear::evaluate_elements()
{
  if (not validsysmat_)
  {
    Ale::evaluate_elements();

    validsysmat_ = true;
  }
  else if (system_matrix())
    system_matrix()->Apply(*dispnp(), *write_access_residual());
  else if (block_system_matrix())
    block_system_matrix()->Apply(*dispnp(), *write_access_residual());
  else
    FOUR_C_THROW("Can't compute residual for linear ALE.");

  return;
}

FOUR_C_NAMESPACE_CLOSE
