// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_basedataglobalstate.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beaminteraction_str_model_evaluator.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_meshtying_abstract_strategy.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_group_prepostoperator.hpp"
#include "4C_structure_new_factory.hpp"
#include "4C_structure_new_integrator.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_model_evaluator_meshtying.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"
#include "4C_structure_new_utils.hpp"
#include "4C_utils_enum.hpp"

#include <NOX_Epetra_Vector.H>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataGlobalState::BaseDataGlobalState()
    : isinit_(false),
      issetup_(false),
      datasdyn_(nullptr),
      dim_(Global::Problem::instance()->n_dim()),
      discret_(nullptr),
      comm_(MPI_COMM_NULL),
      my_rank_(-1),
      timenp_(0.0),
      timen_(nullptr),
      dt_(nullptr),
      stepn_(0),
      stepnp_(0),
      restartstep_(0),
      ispredict_(false),
      dis_(nullptr),
      vel_(nullptr),
      acc_(nullptr),
      disnp_(nullptr),
      velnp_(nullptr),
      accnp_(nullptr),
      fintnp_(nullptr),
      fextnp_(nullptr),
      freactn_(nullptr),
      freactnp_(nullptr),
      finertialn_(nullptr),
      finertialnp_(nullptr),
      fviscon_(nullptr),
      fvisconp_(nullptr),
      fstructold_(nullptr),
      jac_(nullptr),
      stiff_(nullptr),
      mass_(nullptr),
      damp_(nullptr),
      timer_(nullptr),
      dtsolve_(0.0),
      dtele_(0.0),
      max_block_num_(0),
      gproblem_map_ptr_(nullptr),
      pressextractor_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataGlobalState& Solid::TimeInt::BaseDataGlobalState::operator=(
    const Solid::TimeInt::BaseDataGlobalState& source)
{
  this->datasdyn_ = source.datasdyn_;

  this->discret_ = source.discret_;
  this->comm_ = source.comm_;
  this->my_rank_ = source.my_rank_;

  this->timen_ = source.timen_;
  this->dt_ = source.dt_;

  this->timenp_ = source.timenp_;
  this->stepnp_ = source.stepnp_;

  this->isinit_ = source.isinit_;

  // the setup information is not copied --> set boolean to false
  this->issetup_ = false;

  return *this;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataGlobalState::init(
    const std::shared_ptr<Core::FE::Discretization> discret,
    const Teuchos::ParameterList& sdynparams, const std::shared_ptr<const BaseDataSDyn> datasdyn)
{
  // We have to call setup() after init()
  issetup_ = false;

  // ----------------------------------------------------------
  // const pointer to the sDynData container
  // ----------------------------------------------------------
  datasdyn_ = datasdyn;

  // ----------------------------------------------------------
  // general purpose algorithm members
  // ----------------------------------------------------------
  {
    discret_ = discret;
    comm_ = discret_->get_comm();
    my_rank_ = Core::Communication::my_mpi_rank(comm_);
  }

  // --------------------------------------
  // control parameters
  // --------------------------------------
  {
    timen_ = std::make_shared<TimeStepping::TimIntMStep<double>>(
        0, 0, sdynparams.get<double>("TIMEINIT"));
    dt_ = std::make_shared<TimeStepping::TimIntMStep<double>>(
        0, 0, sdynparams.get<double>("TIMESTEP"));

    // initialize target time to initial time plus step size
    timenp_ = (*timen_)[0] + (*dt_)[0];
    stepnp_ = stepn_ + 1;

    // initialize restart step
    restartstep_ = Global::Problem::instance()->restart();
    if (restartstep_ < 0) FOUR_C_THROW("The restart step is expected to be positive.");
  }

  // end of initialization
  isinit_ = true;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataGlobalState::setup()
{
  // safety check
  check_init();

  // --------------------------------------
  // control parameters
  // --------------------------------------
  timer_ = std::make_shared<Teuchos::Time>("", true);

  // --------------------------------------
  // vectors
  // --------------------------------------
  // displacements D_{n}
  dis_ = std::make_shared<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, dof_row_map_view(), true);
  // velocities V_{n}
  vel_ = std::make_shared<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, dof_row_map_view(), true);
  // accelerations A_{n}
  acc_ = std::make_shared<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, dof_row_map_view(), true);

  // displacements D_{n+1} at t_{n+1}
  disnp_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  // velocities V_{n+1} at t_{n+1}
  velnp_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  // accelerations A_{n+1} at t_{n+1}
  accnp_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);

  fintn_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  fintnp_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);

  fextn_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  fextnp_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);

  freactn_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  freactnp_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);

  finertialn_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  finertialnp_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);

  fviscon_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  fvisconp_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);

  fstructold_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);

  // --------------------------------------
  // sparse operators
  // --------------------------------------
  mass_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map_view(), 81, true, true);
  if (datasdyn_->get_damping_type() != Inpar::Solid::damp_none)
  {
    if (datasdyn_->get_mass_lin_type() == Inpar::Solid::MassLin::ml_none)
    {
      damp_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map_view(), 81, true, true);
    }
    else
    {
      /* Since our element evaluate routine is only designed for two input matrices
       * (stiffness and damping or stiffness and mass) its not possible, to have nonlinear
       * inertia forces AND material damping. */
      FOUR_C_THROW("So far it is not possible to model nonlinear inertia forces and damping!");
    }
  }

  if (datasdyn_->get_dynamic_type() == Inpar::Solid::DynamicType::Statics and
      datasdyn_->get_mass_lin_type() != Inpar::Solid::MassLin::ml_none)
    FOUR_C_THROW(
        "Do not set parameter MASSLIN in static simulations as this leads to undesired"
        " evaluation of mass matrix on element level!");

  set_initial_fields();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataGlobalState::set_initial_fields()
{
  // set initial velocity field if existing
  const std::string field = "Velocity";
  std::vector<int> localdofs;
  localdofs.push_back(0);
  localdofs.push_back(1);
  localdofs.push_back(2);

  Core::FE::evaluate_initial_field(
      Global::Problem::instance()->function_manager(), *discret_, field, *velnp_, localdofs);

  // set initial porosity field if existing
  const std::string porosityfield = "Porosity";
  std::vector<int> porositylocaldofs;
  porositylocaldofs.push_back(Global::Problem::instance()->n_dim());

  Core::FE::evaluate_initial_field(Global::Problem::instance()->function_manager(), *discret_,
      porosityfield, *(*dis_)(0), porositylocaldofs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<::NOX::Epetra::Vector> Solid::TimeInt::BaseDataGlobalState::create_global_vector()
    const
{
  return create_global_vector(VecInitType::zero, nullptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::BaseDataGlobalState::setup_block_information(
    const Solid::ModelEvaluator::Generic& me, const Inpar::Solid::ModelType& mt)
{
  check_init();
  Global::Problem* problem = Global::Problem::instance();
  std::shared_ptr<const Core::LinAlg::Map> me_map_ptr = me.get_block_dof_row_map_ptr();

  model_maps_[mt] = me_map_ptr;

  switch (mt)
  {
    case Inpar::Solid::model_structure:
    {
      // always called first, so we can use it to reset things
      gproblem_map_ptr_ = nullptr;
      model_block_id_[mt] = 0;
      max_block_num_ = 1;
      break;
    }
    case Inpar::Solid::model_contact:
    {
      auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(
          problem->contact_dynamic_params(), "SYSTEM");

      auto soltype = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(
          problem->contact_dynamic_params(), "STRATEGY");

      // systems without additional dofs
      if (soltype == CONTACT::SolvingStrategy::nitsche ||
          soltype == CONTACT::SolvingStrategy::penalty ||
          soltype == CONTACT::SolvingStrategy::uzawa ||
          soltype == CONTACT::SolvingStrategy::multiscale)
      {
        model_block_id_[mt] = 0;
      }
      // --- saddle-point system
      else if (systype == CONTACT::SystemType::saddlepoint)
      {
        model_block_id_[mt] = max_block_num_;
        ++max_block_num_;
      }
      // --- condensed system
      else
      {
        model_block_id_[mt] = 0;
      }
      break;
    }
    case Inpar::Solid::model_meshtying:
    {
      const Solid::ModelEvaluator::Meshtying& mt_me =
          dynamic_cast<const Solid::ModelEvaluator::Meshtying&>(me);

      enum CONTACT::SystemType systype = mt_me.strategy().system_type();

      auto soltype = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(
          mt_me.strategy().params(), "STRATEGY");

      // systems without additional dofs
      if (soltype == CONTACT::SolvingStrategy::nitsche ||
          soltype == CONTACT::SolvingStrategy::penalty ||
          soltype == CONTACT::SolvingStrategy::uzawa ||
          soltype == CONTACT::SolvingStrategy::multiscale)
      {
        model_block_id_[mt] = 0;
      }
      // --- saddle-point system
      else if (systype == CONTACT::SystemType::saddlepoint)
      {
        model_block_id_[mt] = max_block_num_;
        ++max_block_num_;
      }
      // --- condensed system
      else if (systype == CONTACT::SystemType::condensed)
      {
        model_block_id_[mt] = 0;
      }
      else
        FOUR_C_THROW("I don't know what to do");
      break;
    }
    case Inpar::Solid::model_cardiovascular0d:
    {
      // --- 2x2 block system
      model_block_id_[mt] = max_block_num_;
      ++max_block_num_;
      break;
    }
    case Inpar::Solid::model_lag_pen_constraint:
    {
      // ----------------------------------------------------------------------
      // check type of constraint conditions (Lagrange multiplier vs. penalty)
      // ----------------------------------------------------------------------
      bool have_lag_constraint = false;
      std::vector<const Core::Conditions::Condition*> lagcond_volconstr3d;
      std::vector<const Core::Conditions::Condition*> lagcond_areaconstr3d;
      std::vector<const Core::Conditions::Condition*> lagcond_areaconstr2d;
      std::vector<const Core::Conditions::Condition*> lagcond_mpconline2d;
      std::vector<const Core::Conditions::Condition*> lagcond_mpconplane3d;
      std::vector<const Core::Conditions::Condition*> lagcond_mpcnormcomp3d;
      discret_->get_condition("VolumeConstraint_3D", lagcond_volconstr3d);
      discret_->get_condition("AreaConstraint_3D", lagcond_areaconstr3d);
      discret_->get_condition("AreaConstraint_2D", lagcond_areaconstr2d);
      discret_->get_condition("MPC_NodeOnLine_2D", lagcond_mpconline2d);
      discret_->get_condition("MPC_NodeOnPlane_3D", lagcond_mpconplane3d);
      discret_->get_condition("MPC_NormalComponent_3D", lagcond_mpcnormcomp3d);
      if (lagcond_volconstr3d.size() or lagcond_areaconstr3d.size() or
          lagcond_areaconstr2d.size() or lagcond_mpconline2d.size() or
          lagcond_mpconplane3d.size() or lagcond_mpcnormcomp3d.size())
        have_lag_constraint = true;

      // --- 2x2 block system (saddle-point structure)
      if (have_lag_constraint)
      {
        model_block_id_[mt] = max_block_num_;
        ++max_block_num_;
      }
      // --- standard system
      else
      {
        model_block_id_[mt] = 0;
      }
      break;
    }
    case Inpar::Solid::model_springdashpot:
    case Inpar::Solid::model_beam_interaction_old:
    case Inpar::Solid::model_browniandyn:
    case Inpar::Solid::model_constraints:
    {
      model_block_id_[mt] = 0;
      break;
    }
    case Inpar::Solid::model_basic_coupling:
    case Inpar::Solid::model_monolithic_coupling:
    case Inpar::Solid::model_partitioned_coupling:
    {
      // do nothing
      break;
    }
    case Inpar::Solid::model_beaminteraction:
    {
      const Solid::ModelEvaluator::BeamInteraction& beaminteraction_evaluator =
          dynamic_cast<const Solid::ModelEvaluator::BeamInteraction&>(me);
      if (beaminteraction_evaluator.have_lagrange_dofs())
      {
        model_block_id_[mt] = max_block_num_;
        ++max_block_num_;
      }
      break;
    }
    case Inpar::Solid::model_multiscale:
    {
      // do nothing
      break;
    }
    default:
    {
      // FixMe please
      FOUR_C_THROW("Augment this function for your model type!");
      break;
    }
  }
  // create a global problem map
  gproblem_map_ptr_ = Core::LinAlg::merge_map(gproblem_map_ptr_, me_map_ptr);

  return gproblem_map_ptr_->max_all_gid();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataGlobalState::setup_multi_map_extractor()
{
  check_init();
  /* copy the std::map into a std::vector and keep the numbering of the model-id
   * map */
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps_vec(max_block_number(), nullptr);
  // Make sure, that the block ids and the vector entry ids coincide!
  std::map<Inpar::Solid::ModelType, int>::const_iterator ci;
  for (ci = model_block_id_.begin(); ci != model_block_id_.end(); ++ci)
  {
    enum Inpar::Solid::ModelType mt = ci->first;
    int bid = ci->second;
    maps_vec[bid] = model_maps_.at(mt);
  }
  blockextractor_.setup(*gproblem_map_ptr_, maps_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataGlobalState::setup_element_technology_map_extractors()
{
  check_init();

  // loop all active element technologies
  const std::set<enum Inpar::Solid::EleTech>& ele_techs = datasdyn_->get_element_technologies();
  for (const enum Inpar::Solid::EleTech et : ele_techs)
  {
    // mapextractor for element technology
    Core::LinAlg::MultiMapExtractor mapext;

    switch (et)
    {
      case (Inpar::Solid::EleTech::rotvec):
      {
        setup_rot_vec_map_extractor(mapext);
        break;
      }
      // element technology doesn't require a map extractor: skip
      default:
        continue;
    }

    // sanity check
    mapext.check_for_valid_map_extractor();

    // insert into map
    const auto check = mapextractors_.insert(
        std::pair<Inpar::Solid::EleTech, Core::LinAlg::MultiMapExtractor>(et, mapext));

    if (not check.second) FOUR_C_THROW("Insert failed!");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::MultiMapExtractor&
Solid::TimeInt::BaseDataGlobalState::get_element_technology_map_extractor(
    const enum Inpar::Solid::EleTech etech) const
{
  if (mapextractors_.find(etech) == mapextractors_.end())
    FOUR_C_THROW("Could not find element technology \"{}\" in map extractors.", etech);

  return mapextractors_.at(etech);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataGlobalState::setup_rot_vec_map_extractor(
    Core::LinAlg::MultiMapExtractor& multimapext)
{
  check_init();

  /* all additive DoFs, i.e. members of real value vector spaces
   * such as translational displacements, tangent vector displacements,
   * 1D rotation angles, ... */
  std::set<int> additdofset;
  /* DoFs which are non-additive and therefore e.g. can not be updated in usual
   * incremental manner, need special treatment in time integration ...
   * (currently only rotation pseudo-vector DoFs of beam elements) */
  std::set<int> rotvecdofset;

  for (int i = 0; i < discret_->num_my_row_nodes(); ++i)
  {
    Core::Nodes::Node* nodeptr = discret_->l_row_node(i);

    const Discret::Elements::Beam3Base* beameleptr =
        dynamic_cast<const Discret::Elements::Beam3Base*>(nodeptr->elements()[0]);

    std::vector<int> nodaladditdofs;
    std::vector<int> nodalrotvecdofs;

    // so far we only expect DoFs of beam elements for the rotvecdofset
    if (beameleptr == nullptr)
    {
      nodaladditdofs = discret_->dof(0, nodeptr);
    }
    else
    {
      std::shared_ptr<Core::FE::Discretization> discret = discret_;
      nodaladditdofs = beameleptr->get_additive_dof_gids(*discret, *nodeptr);
      nodalrotvecdofs = beameleptr->get_rot_vec_dof_gids(*discret, *nodeptr);

      if (nodaladditdofs.size() + nodalrotvecdofs.size() !=
          (unsigned)beameleptr->num_dof_per_node(*nodeptr))
        FOUR_C_THROW("Expected {} DoFs for node with GID {} but collected {} DoFs",
            beameleptr->num_dof_per_node(*nodeptr), discret_->node_row_map()->gid(i),
            nodaladditdofs.size() + nodalrotvecdofs.size());
    }

    // add the DoFs of this node to the total set
    for (unsigned j = 0; j < nodaladditdofs.size(); ++j) additdofset.insert(nodaladditdofs[j]);

    for (unsigned j = 0; j < nodalrotvecdofs.size(); ++j) rotvecdofset.insert(nodalrotvecdofs[j]);

  }  // loop over row nodes

  // create the required Epetra maps
  std::vector<int> additdofmapvec;
  additdofmapvec.reserve(additdofset.size());
  additdofmapvec.assign(additdofset.begin(), additdofset.end());
  additdofset.clear();
  std::shared_ptr<Core::LinAlg::Map> additdofmap = std::make_shared<Core::LinAlg::Map>(
      -1, additdofmapvec.size(), additdofmapvec.data(), 0, discret_->get_comm());
  additdofmapvec.clear();

  std::vector<int> rotvecdofmapvec;
  rotvecdofmapvec.reserve(rotvecdofset.size());
  rotvecdofmapvec.assign(rotvecdofset.begin(), rotvecdofset.end());
  rotvecdofset.clear();
  std::shared_ptr<Core::LinAlg::Map> rotvecdofmap = std::make_shared<Core::LinAlg::Map>(
      -1, rotvecdofmapvec.size(), rotvecdofmapvec.data(), 0, discret_->get_comm());
  rotvecdofmapvec.clear();

  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps(2);
  maps[0] = additdofmap;
  maps[1] = rotvecdofmap;

  multimapext.setup(*dof_row_map_view(), maps);
}

Core::LinAlg::Map Solid::TimeInt::BaseDataGlobalState::block_map(
    const Inpar::Solid::ModelType& mt) const
{
  if (model_maps_.find(mt) == model_maps_.end())
    FOUR_C_THROW(
        "There is no block map for the given "
        "modeltype \"{}\".",
        mt);

  return *(model_maps_.at(mt));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::MultiMapExtractor& Solid::TimeInt::BaseDataGlobalState::block_extractor() const
{
  // sanity check
  blockextractor_.check_for_valid_map_extractor();
  return blockextractor_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<::NOX::Epetra::Vector> Solid::TimeInt::BaseDataGlobalState::create_global_vector(
    const enum VecInitType& vecinittype,
    const std::shared_ptr<const Solid::ModelEvaluatorManager>& modeleval) const
{
  check_init();
  Core::LinAlg::Vector<double> xvec_ptr(global_problem_map(), true);

  // switch between the different vector initialization options
  switch (vecinittype)
  {
    /* use the last converged state to construct a new solution vector */
    case VecInitType::last_time_step:
    {
      FOUR_C_ASSERT(modeleval, "We need access to the Solid::ModelEvaluatorManager object!");

      std::map<Inpar::Solid::ModelType, int>::const_iterator ci;
      for (ci = model_block_id_.begin(); ci != model_block_id_.end(); ++ci)
      {
        // get the partial solution vector of the last time step
        std::shared_ptr<const Core::LinAlg::Vector<double>> model_sol_ptr =
            modeleval->evaluator(ci->first).get_last_time_step_solution_ptr();
        // if there is a partial solution, we insert it into the full vector
        if (model_sol_ptr) block_extractor().insert_vector(*model_sol_ptr, ci->second, xvec_ptr);
        model_sol_ptr = nullptr;
      }
      break;
    }
    /* use the current global state to construct a new solution vector */
    case VecInitType::init_current_state:
    {
      FOUR_C_ASSERT(modeleval, "We need access to the Solid::ModelEvaluatorManager object!");

      std::map<Inpar::Solid::ModelType, int>::const_iterator ci;
      for (ci = model_block_id_.begin(); ci != model_block_id_.end(); ++ci)
      {
        // get the partial solution vector of the current state
        std::shared_ptr<const Core::LinAlg::Vector<double>> model_sol_ptr =
            modeleval->evaluator(ci->first).get_current_solution_ptr();
        // if there is a partial solution, we insert it into the full vector
        if (model_sol_ptr) block_extractor().insert_vector(*model_sol_ptr, ci->second, xvec_ptr);
      }
      break;
    }
    /* construct a new solution vector filled with zeros */
    case VecInitType::zero:
    default:
    {
      // nothing to do.
      break;
    }
  }  // end of the switch-case statement

  // Copy the content of our vector into a vector that NOX can use
  return std::make_shared<::NOX::Epetra::Vector>(
      Teuchos::make_rcp<Epetra_Vector>(xvec_ptr.get_ref_of_epetra_vector()),
      ::NOX::Epetra::Vector::CreateView);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::SparseOperator*
Solid::TimeInt::BaseDataGlobalState::create_structural_stiffness_matrix_block()
{
  stiff_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map_view(), 81, true, true);

  return stiff_.get();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseOperator>&
Solid::TimeInt::BaseDataGlobalState::create_jacobian()
{
  check_init();
  jac_ = nullptr;

  if (max_block_num_ > 1)
  {
    jac_ =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            block_extractor(), block_extractor(), 81, true, true);
  }
  else
  {
    // pure structural case
    jac_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map_view(), 81, true, true);
  }

  return jac_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseOperator>
Solid::TimeInt::BaseDataGlobalState::create_aux_jacobian() const
{
  check_init();
  std::shared_ptr<Core::LinAlg::SparseOperator> jac = nullptr;

  if (max_block_num_ > 1)
  {
    jac =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            block_extractor(), block_extractor(), 81, true, true);
  }
  else
  {
    // pure structural case
    jac = std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map_view(), 81, true, true);
  }

  return jac;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> Solid::TimeInt::BaseDataGlobalState::dof_row_map() const
{
  check_init();
  const Core::LinAlg::Map* dofrowmap_ptr = discret_->dof_row_map();
  // since it's const, we do not need to copy the map
  return Core::Utils::shared_ptr_from_ref(*dofrowmap_ptr);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> Solid::TimeInt::BaseDataGlobalState::dof_row_map(
    unsigned nds) const
{
  check_init();
  const Core::LinAlg::Map* dofrowmap_ptr = discret_->dof_row_map(nds);
  // since it's const, we do not need to copy the map
  return Core::Utils::shared_ptr_from_ref(*dofrowmap_ptr);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Map* Solid::TimeInt::BaseDataGlobalState::dof_row_map_view() const
{
  check_init();
  return discret_->dof_row_map();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Map* Solid::TimeInt::BaseDataGlobalState::additive_dof_row_map_view() const
{
  check_init();
  return get_element_technology_map_extractor(Inpar::Solid::EleTech::rotvec).map(0).get();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Map* Solid::TimeInt::BaseDataGlobalState::rot_vec_dof_row_map_view() const
{
  check_init();
  return get_element_technology_map_extractor(Inpar::Solid::EleTech::rotvec).map(1).get();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Solid::TimeInt::BaseDataGlobalState::extract_displ_entries(
    const Core::LinAlg::Vector<double>& source) const
{
  return extract_model_entries(Inpar::Solid::model_structure, source);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Solid::TimeInt::BaseDataGlobalState::extract_model_entries(
    const Inpar::Solid::ModelType& mt, const Core::LinAlg::Vector<double>& source) const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> model_ptr = nullptr;
  // extract from the full state vector
  if (source.get_map().num_global_elements() == block_extractor().full_map()->num_global_elements())
  {
    model_ptr = block_extractor().extract_vector(source, model_block_id_.at(mt));
  }
  // copy the vector
  else if (source.get_map().num_global_elements() == model_maps_.at(mt)->num_global_elements())
  {
    model_ptr = std::make_shared<Core::LinAlg::Vector<double>>(source);
  }
  // otherwise do a standard export
  else
  {
    model_ptr = std::make_shared<Core::LinAlg::Vector<double>>(*model_maps_.at(mt));
    Core::LinAlg::export_to(source, *model_ptr);
  }


  return model_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Solid::TimeInt::BaseDataGlobalState::extract_rot_vec_entries(
    const Core::LinAlg::Vector<double>& source) const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> addit_ptr =
      get_element_technology_map_extractor(Inpar::Solid::EleTech::rotvec).extract_vector(source, 1);

  return addit_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataGlobalState::assign_model_block(Core::LinAlg::SparseOperator& jac,
    const Core::LinAlg::SparseMatrix& matrix, const Inpar::Solid::ModelType& mt,
    const MatBlockType& bt, const Core::LinAlg::DataAccess& access) const
{
  Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>* blockmat_ptr =
      dynamic_cast<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>*>(
          &jac);
  if (blockmat_ptr != nullptr)
  {
    if (max_block_number() < 2)
      FOUR_C_THROW(
          "The jacobian is a Core::LinAlg::BlockSparseMatrix but has less than"
          " two blocks! Seems wrong.");

    const int& b_id = model_block_id_.at(mt);
    switch (bt)
    {
      case MatBlockType::displ_displ:
      {
        blockmat_ptr->matrix(0, 0).assign(access, matrix);
        break;
      }
      case MatBlockType::displ_lm:
      {
        blockmat_ptr->matrix(0, b_id).assign(access, matrix);
        break;
      }
      case MatBlockType::lm_displ:
      {
        blockmat_ptr->matrix(b_id, 0).assign(access, matrix);
        break;
      }
      case MatBlockType::lm_lm:
      {
        blockmat_ptr->matrix(b_id, b_id).assign(access, matrix);
        break;
      }
      default:
      {
        FOUR_C_THROW("model block {} is not supported", mat_block_type_to_string(bt));
        break;
      }
    }
    return;
  }

  // sanity check
  if (model_block_id_.find(mt) == model_block_id_.end() or bt != MatBlockType::displ_displ)
    FOUR_C_THROW(
        "It seems as you are trying to access a matrix block which has "
        "not been created.");

  Core::LinAlg::SparseMatrix* stiff_ptr = dynamic_cast<Core::LinAlg::SparseMatrix*>(&jac);
  if (stiff_ptr != nullptr)
  {
    stiff_ptr->assign(access, matrix);
    return;
  }

  FOUR_C_THROW(
      "The jacobian has the wrong type! (no Core::LinAlg::SparseMatrix "
      "and no Core::LinAlg::BlockSparseMatrix)");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
Solid::TimeInt::BaseDataGlobalState::extract_model_block(Core::LinAlg::SparseOperator& jac,
    const Inpar::Solid::ModelType& mt, const MatBlockType& bt) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> block = nullptr;
  Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>* blockmat_ptr =
      dynamic_cast<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>*>(
          &jac);
  if (blockmat_ptr != nullptr)
  {
    if (max_block_number() < 2)
      FOUR_C_THROW(
          "The jacobian is a Core::LinAlg::BlockSparseMatrix but has less than"
          " two blocks! Seems wrong.");
    const int& b_id = model_block_id_.at(mt);
    switch (bt)
    {
      case MatBlockType::displ_displ:
      {
        block = Core::Utils::shared_ptr_from_ref(blockmat_ptr->matrix(0, 0));
        break;
      }
      case MatBlockType::displ_lm:
      {
        block = Core::Utils::shared_ptr_from_ref(blockmat_ptr->matrix(0, b_id));
        break;
      }
      case MatBlockType::lm_displ:
      {
        block = Core::Utils::shared_ptr_from_ref(blockmat_ptr->matrix(b_id, 0));
        break;
      }
      case MatBlockType::lm_lm:
      {
        block = Core::Utils::shared_ptr_from_ref(blockmat_ptr->matrix(b_id, b_id));
        break;
      }
      default:
      {
        FOUR_C_THROW("model block {} is not supported", mat_block_type_to_string(bt));
        break;
      }
    }
    return block;
  }

  // sanity check
  if (model_block_id_.find(mt) == model_block_id_.end() or bt != MatBlockType::displ_displ)
    FOUR_C_THROW(
        "It seems as you are trying to access a matrix block which has "
        "not been created.");

  Core::LinAlg::SparseMatrix* stiff_ptr = dynamic_cast<Core::LinAlg::SparseMatrix*>(&jac);
  if (stiff_ptr != nullptr)
  {
    block = Core::Utils::shared_ptr_from_ref(*stiff_ptr);
    return block;
  }

  FOUR_C_THROW(
      "The jacobian has the wrong type! (no Core::LinAlg::SparseMatrix "
      "and no Core::LinAlg::BlockSparseMatrix)");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<std::vector<Core::LinAlg::SparseMatrix*>>
Solid::TimeInt::BaseDataGlobalState::extract_displ_row_of_blocks(
    Core::LinAlg::SparseOperator& jac) const
{
  return extract_row_of_blocks(jac, Inpar::Solid::model_structure);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<std::vector<Core::LinAlg::SparseMatrix*>>
Solid::TimeInt::BaseDataGlobalState::extract_row_of_blocks(
    Core::LinAlg::SparseOperator& jac, const Inpar::Solid::ModelType& mt) const
{
  std::shared_ptr<std::vector<Core::LinAlg::SparseMatrix*>> rowofblocks = nullptr;

  Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>* blockmat_ptr =
      dynamic_cast<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>*>(
          &jac);
  if (blockmat_ptr != nullptr)
  {
    if (max_block_number() < 2)
      FOUR_C_THROW(
          "The jacobian is a Core::LinAlg::BlockSparseMatrix but has less than"
          " two blocks! Seems wrong.");
    const int& b_id = model_block_id_.at(mt);

    const int num_cols = blockmat_ptr->cols();
    rowofblocks = std::make_shared<std::vector<Core::LinAlg::SparseMatrix*>>(num_cols, nullptr);

    for (int i = 0; i < num_cols; ++i) (*rowofblocks)[i] = &(blockmat_ptr->matrix(b_id, i));

    return rowofblocks;
  }

  // sanity check
  if (model_block_id_.find(mt) == model_block_id_.end())
    FOUR_C_THROW(
        "It seems as you are trying to access a matrix block row which has "
        "not been created.");

  Core::LinAlg::SparseMatrix* stiff_ptr = dynamic_cast<Core::LinAlg::SparseMatrix*>(&jac);
  if (stiff_ptr != nullptr)
  {
    rowofblocks = std::make_shared<std::vector<Core::LinAlg::SparseMatrix*>>(1, nullptr);
    (*rowofblocks)[0] = stiff_ptr;
    return rowofblocks;
  }

  FOUR_C_THROW(
      "The jacobian has the wrong type! (no Core::LinAlg::SparseMatrix "
      "and no Core::LinAlg::BlockSparseMatrix)");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
Solid::TimeInt::BaseDataGlobalState::extract_displ_block(Core::LinAlg::SparseOperator& jac) const
{
  return extract_model_block(jac, Inpar::Solid::model_structure, MatBlockType::displ_displ);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::SparseMatrix>
Solid::TimeInt::BaseDataGlobalState::get_jacobian_displ_block() const
{
  FOUR_C_ASSERT(jac_, "The jacobian is not initialized!");
  return extract_displ_block(*jac_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
Solid::TimeInt::BaseDataGlobalState::jacobian_displ_block()
{
  FOUR_C_ASSERT(jac_, "The jacobian is not initialized!");
  return extract_displ_block(*jac_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::SparseMatrix>
Solid::TimeInt::BaseDataGlobalState::get_jacobian_block(
    const Inpar::Solid::ModelType mt, const MatBlockType bt) const
{
  FOUR_C_ASSERT(jac_, "The jacobian is not initialized!");

  return extract_model_block(*jac_, mt, bt);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::BaseDataGlobalState::get_last_lin_iteration_number(const unsigned step) const
{
  check_init_setup();
  if (step < 1) FOUR_C_THROW("The given step number must be larger than 1. (step={})", step);

  auto linsolvers = datasdyn_->get_lin_solvers();
  int iter = -1;

  for (auto& linsolver : linsolvers)
  {
    switch (linsolver.first)
    {
      // has only one field solver per default
      case Inpar::Solid::model_structure:
      case Inpar::Solid::model_springdashpot:
      case Inpar::Solid::model_browniandyn:
      case Inpar::Solid::model_beaminteraction:
      case Inpar::Solid::model_basic_coupling:
      case Inpar::Solid::model_monolithic_coupling:
      case Inpar::Solid::model_partitioned_coupling:
      case Inpar::Solid::model_beam_interaction_old:
      {
        iter = linsolvers[linsolver.first]->get_num_iters();
        break;
      }
      default:
        FOUR_C_THROW(
            "The given model type '{}' is not supported for linear iteration output right now.",
            Inpar::Solid::model_structure);
    }
  }

  return iter;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::BaseDataGlobalState::get_nln_iteration_number(const unsigned step) const
{
  check_init_setup();
  if (step < 1) FOUR_C_THROW("The given step number must be larger than 1. (step={})", step);

  auto cit = nln_iter_numbers_.begin();
  while (cit != nln_iter_numbers_.end())
  {
    if (cit->first == static_cast<int>(step)) return cit->second;
    ++cit;
  }

  FOUR_C_THROW("There is no nonlinear iteration number for the given step {}.", step);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataGlobalState::set_nln_iteration_number(const int nln_iter)
{
  check_init_setup();

  auto cit = nln_iter_numbers_.cbegin();
  while (cit != nln_iter_numbers_.end())
  {
    if (cit->first == stepn_)
    {
      if (cit->second != nln_iter)
        FOUR_C_THROW(
            "There is already a different nonlinear iteration number "
            "for step {}.",
            stepn_);
      else
        return;
    }
    ++cit;
  }
  nln_iter_numbers_.push_back(std::make_pair(stepn_, nln_iter));
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::GROUP::PrePostOp::TimeInt::RotVecUpdater::RotVecUpdater(
    const std::shared_ptr<const Solid::TimeInt::BaseDataGlobalState>& gstate_ptr)
    : gstate_ptr_(gstate_ptr)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::GROUP::PrePostOp::TimeInt::RotVecUpdater::run_pre_compute_x(
    const NOX::Nln::Group& input_grp, const Core::LinAlg::Vector<double>& dir, const double& step,
    const NOX::Nln::Group& curr_grp)
{
  const auto& xold = dynamic_cast<const ::NOX::Epetra::Vector&>(input_grp.getX()).getEpetraVector();

  // cast the const away so that the new x vector can be set after the update
  NOX::Nln::Group& curr_grp_mutable = const_cast<NOX::Nln::Group&>(curr_grp);

  std::shared_ptr<Core::LinAlg::Vector<double>> xnew =
      std::make_shared<Core::LinAlg::Vector<double>>(xold.Map(), true);

  /* we do the multiplicative update only for those entries which belong to
   * rotation (pseudo-)vectors */
  Core::LinAlg::Vector<double> x_rotvec =
      *gstate_ptr_->extract_rot_vec_entries(Core::LinAlg::Vector<double>(xold));
  Core::LinAlg::Vector<double> dir_rotvec = *gstate_ptr_->extract_rot_vec_entries(dir);

  Core::LinAlg::Matrix<4, 1> Qold;
  Core::LinAlg::Matrix<4, 1> deltaQ;
  Core::LinAlg::Matrix<4, 1> Qnew;

  /* since parallel distribution is node-wise, the three entries belonging to
   * a rotation vector should be stored on the same processor: safety-check */
  if (x_rotvec.get_map().num_my_elements() % 3 != 0 or
      dir_rotvec.get_map().num_my_elements() % 3 != 0)
    FOUR_C_THROW(
        "fatal error: apparently, the three DOFs of a nodal rotation vector are"
        " not stored on this processor. Can't apply multiplicative update!");

  // rotation vectors always consist of three consecutive DoFs
  for (int i = 0; i < x_rotvec.get_map().num_my_elements(); i = i + 3)
  {
    // create a Core::LinAlg::Matrix from reference to three x vector entries
    Core::LinAlg::Matrix<3, 1> theta(&x_rotvec[i], true);
    Core::LargeRotations::angletoquaternion(theta, Qold);

    // same for relative rotation angle deltatheta
    Core::LinAlg::Matrix<3, 1> deltatheta(&dir_rotvec[i], true);
    deltatheta.scale(step);

    Core::LargeRotations::angletoquaternion(deltatheta, deltaQ);
    Core::LargeRotations::quaternionproduct(Qold, deltaQ, Qnew);
    Core::LargeRotations::quaterniontoangle(Qnew, theta);
  }

  // first update entire x vector in an additive manner
  xnew->update(1.0, xold, step, dir, 0.0);

  // now replace the rotvec entries by the correct value computed before
  Core::LinAlg::assemble_my_vector(0.0, *xnew, 1.0, x_rotvec);
  curr_grp_mutable.setX(Teuchos::rcpFromRef(xnew->get_ref_of_epetra_vector()));

  /* tell the NOX::Nln::Group that the x vector has already been updated in
   * this preComputeX operator call */
  curr_grp_mutable.set_skip_update_x(true);
}

FOUR_C_NAMESPACE_CLOSE
