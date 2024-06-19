/*-----------------------------------------------------------*/
/*! \file

\brief Global state data container for the structural (time) integration


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_timint_basedataglobalstate.hpp"

#include "4C_beam3_base.hpp"
#include "4C_contact_meshtying_abstract_strategy.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_group_prepostoperator.hpp"
#include "4C_structure_new_model_evaluator.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"
#include "4C_structure_new_model_evaluator_meshtying.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"
#include "4C_structure_new_utils.hpp"

#include <Epetra_Vector.h>
#include <NOX_Epetra_Vector.H>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::TimeInt::BaseDataGlobalState::BaseDataGlobalState()
    : isinit_(false),
      issetup_(false),
      datasdyn_(Teuchos::null),
      discret_(Teuchos::null),
      comm_(Teuchos::null),
      my_rank_(-1),
      timenp_(0.0),
      timen_(Teuchos::null),
      dt_(Teuchos::null),
      stepn_(0),
      stepnp_(0),
      restartstep_(0),
      ispredict_(false),
      dis_(Teuchos::null),
      vel_(Teuchos::null),
      acc_(Teuchos::null),
      disnp_(Teuchos::null),
      velnp_(Teuchos::null),
      accnp_(Teuchos::null),
      fintnp_(Teuchos::null),
      fextnp_(Teuchos::null),
      freactn_(Teuchos::null),
      freactnp_(Teuchos::null),
      finertialn_(Teuchos::null),
      finertialnp_(Teuchos::null),
      fviscon_(Teuchos::null),
      fvisconp_(Teuchos::null),
      fstructold_(Teuchos::null),
      jac_(Teuchos::null),
      stiff_(Teuchos::null),
      mass_(Teuchos::null),
      damp_(Teuchos::null),
      timer_(Teuchos::null),
      dtsolve_(0.0),
      dtele_(0.0),
      max_block_num_(0),
      gproblem_map_ptr_(Teuchos::null),
      pressextractor_(Teuchos::null)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::TimeInt::BaseDataGlobalState& STR::TimeInt::BaseDataGlobalState::operator=(
    const STR::TimeInt::BaseDataGlobalState& source)
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
void STR::TimeInt::BaseDataGlobalState::init(const Teuchos::RCP<Core::FE::Discretization> discret,
    const Teuchos::ParameterList& sdynparams, const Teuchos::RCP<const BaseDataSDyn> datasdyn)
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
    comm_ = Teuchos::rcpFromRef(discret_->Comm());
    my_rank_ = comm_->MyPID();
  }

  // --------------------------------------
  // control parameters
  // --------------------------------------
  {
    timen_ = Teuchos::rcp(
        new TimeStepping::TimIntMStep<double>(0, 0, sdynparams.get<double>("TIMEINIT")));
    dt_ = Teuchos::rcp(
        new TimeStepping::TimIntMStep<double>(0, 0, sdynparams.get<double>("TIMESTEP")));

    // initialize target time to initial time plus step size
    timenp_ = (*timen_)[0] + (*dt_)[0];
    stepnp_ = stepn_ + 1;

    // initialize restart step
    restartstep_ = Global::Problem::Instance()->Restart();
    if (restartstep_ < 0) FOUR_C_THROW("The restart step is expected to be positive.");
  }

  // end of initialization
  isinit_ = true;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::setup()
{
  // safety check
  check_init();

  // --------------------------------------
  // control parameters
  // --------------------------------------
  timer_ = Teuchos::rcp(new Teuchos::Time("", true));

  // --------------------------------------
  // vectors
  // --------------------------------------
  // displacements D_{n}
  dis_ = Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));
  // velocities V_{n}
  vel_ = Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));
  // accelerations A_{n}
  acc_ = Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));

  // displacements D_{n+1} at t_{n+1}
  disnp_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  // velocities V_{n+1} at t_{n+1}
  velnp_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  // accelerations A_{n+1} at t_{n+1}
  accnp_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

  fintn_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  fintnp_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

  fextn_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  fextnp_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

  freactn_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  freactnp_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

  finertialn_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  finertialnp_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

  fviscon_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  fvisconp_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

  fstructold_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

  // --------------------------------------
  // sparse operators
  // --------------------------------------
  mass_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dof_row_map_view(), 81, true, true));
  if (datasdyn_->get_damping_type() != Inpar::STR::damp_none)
  {
    if (datasdyn_->GetMassLinType() == Inpar::STR::ml_none)
    {
      damp_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dof_row_map_view(), 81, true, true));
    }
    else
    {
      /* Since our element evaluate routine is only designed for two input matrices
       * (stiffness and damping or stiffness and mass) its not possible, to have nonlinear
       * inertia forces AND material damping. */
      FOUR_C_THROW("So far it is not possible to model nonlinear inertia forces and damping!");
    }
  }

  if (datasdyn_->get_dynamic_type() == Inpar::STR::dyna_statics and
      datasdyn_->GetMassLinType() != Inpar::STR::ml_none)
    FOUR_C_THROW(
        "Do not set parameter MASSLIN in static simulations as this leads to undesired"
        " evaluation of mass matrix on element level!");

  set_initial_fields();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::set_initial_fields()
{
  // set initial velocity field if existing
  const std::string field = "Velocity";
  std::vector<int> localdofs;
  localdofs.push_back(0);
  localdofs.push_back(1);
  localdofs.push_back(2);

  Core::FE::UTILS::evaluate_initial_field(
      Global::Problem::Instance()->FunctionManager(), *discret_, field, velnp_, localdofs);

  // set initial porosity field if existing
  const std::string porosityfield = "Porosity";
  std::vector<int> porositylocaldofs;
  porositylocaldofs.push_back(Global::Problem::Instance()->NDim());

  Core::FE::UTILS::evaluate_initial_field(Global::Problem::Instance()->FunctionManager(), *discret_,
      porosityfield, (*dis_)(0), porositylocaldofs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Epetra::Vector> STR::TimeInt::BaseDataGlobalState::create_global_vector() const
{
  return create_global_vector(VecInitType::zero, Teuchos::null);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int STR::TimeInt::BaseDataGlobalState::setup_block_information(
    const STR::MODELEVALUATOR::Generic& me, const Inpar::STR::ModelType& mt)
{
  check_init();
  Global::Problem* problem = Global::Problem::Instance();
  Teuchos::RCP<const Epetra_Map> me_map_ptr = me.get_block_dof_row_map_ptr();

  model_maps_[mt] = me_map_ptr;

  switch (mt)
  {
    case Inpar::STR::model_structure:
    {
      // always called first, so we can use it to reset things
      gproblem_map_ptr_ = Teuchos::null;
      model_block_id_[mt] = 0;
      max_block_num_ = 1;
      break;
    }
    case Inpar::STR::model_contact:
    {
      enum Inpar::CONTACT::SystemType systype =
          Core::UTILS::IntegralValue<Inpar::CONTACT::SystemType>(
              problem->contact_dynamic_params(), "SYSTEM");

      enum Inpar::CONTACT::SolvingStrategy soltype =
          Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(
              problem->contact_dynamic_params(), "STRATEGY");

      // systems without additional dofs
      if (soltype == Inpar::CONTACT::solution_nitsche ||
          soltype == Inpar::CONTACT::solution_penalty ||
          soltype == Inpar::CONTACT::solution_uzawa ||
          soltype == Inpar::CONTACT::solution_multiscale)
      {
        model_block_id_[mt] = 0;
      }
      // --- saddle-point system
      else if (systype == Inpar::CONTACT::system_saddlepoint)
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
    case Inpar::STR::model_meshtying:
    {
      const STR::MODELEVALUATOR::Meshtying& mt_me =
          dynamic_cast<const STR::MODELEVALUATOR::Meshtying&>(me);

      enum Inpar::CONTACT::SystemType systype = mt_me.Strategy().SystemType();

      enum Inpar::CONTACT::SolvingStrategy soltype =
          Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(
              mt_me.Strategy().Params(), "STRATEGY");

      // systems without additional dofs
      if (soltype == Inpar::CONTACT::solution_nitsche ||
          soltype == Inpar::CONTACT::solution_penalty ||
          soltype == Inpar::CONTACT::solution_uzawa ||
          soltype == Inpar::CONTACT::solution_multiscale)
      {
        model_block_id_[mt] = 0;
      }
      // --- saddle-point system
      else if (systype == Inpar::CONTACT::system_saddlepoint)
      {
        model_block_id_[mt] = max_block_num_;
        ++max_block_num_;
      }
      // --- condensed system
      else if (systype == Inpar::CONTACT::system_condensed)
      {
        model_block_id_[mt] = 0;
      }
      else
        FOUR_C_THROW("I don't know what to do");
      break;
    }
    case Inpar::STR::model_cardiovascular0d:
    {
      // --- 2x2 block system
      model_block_id_[mt] = max_block_num_;
      ++max_block_num_;
      break;
    }
    case Inpar::STR::model_lag_pen_constraint:
    {
      // ----------------------------------------------------------------------
      // check type of constraint conditions (Lagrange multiplier vs. penalty)
      // ----------------------------------------------------------------------
      bool have_lag_constraint = false;
      std::vector<Core::Conditions::Condition*> lagcond_volconstr3d(0);
      std::vector<Core::Conditions::Condition*> lagcond_areaconstr3d(0);
      std::vector<Core::Conditions::Condition*> lagcond_areaconstr2d(0);
      std::vector<Core::Conditions::Condition*> lagcond_mpconline2d(0);
      std::vector<Core::Conditions::Condition*> lagcond_mpconplane3d(0);
      std::vector<Core::Conditions::Condition*> lagcond_mpcnormcomp3d(0);
      discret_->GetCondition("VolumeConstraint_3D", lagcond_volconstr3d);
      discret_->GetCondition("AreaConstraint_3D", lagcond_areaconstr3d);
      discret_->GetCondition("AreaConstraint_2D", lagcond_areaconstr2d);
      discret_->GetCondition("MPC_NodeOnLine_2D", lagcond_mpconline2d);
      discret_->GetCondition("MPC_NodeOnPlane_3D", lagcond_mpconplane3d);
      discret_->GetCondition("MPC_NormalComponent_3D", lagcond_mpcnormcomp3d);
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
    case Inpar::STR::model_springdashpot:
    case Inpar::STR::model_beam_interaction_old:
    case Inpar::STR::model_browniandyn:
    case Inpar::STR::model_beaminteraction:
    {
      // structural block
      model_block_id_[mt] = 0;
      break;
    }
    case Inpar::STR::model_basic_coupling:
    case Inpar::STR::model_monolithic_coupling:
    case Inpar::STR::model_partitioned_coupling:
    {
      // do nothing
      break;
    }
    case Inpar::STR::model_constraints:
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
  gproblem_map_ptr_ = Core::LinAlg::MergeMap(gproblem_map_ptr_, me_map_ptr);

  return gproblem_map_ptr_->MaxAllGID();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::setup_multi_map_extractor()
{
  check_init();
  /* copy the std::map into a std::vector and keep the numbering of the model-id
   * map */
  std::vector<Teuchos::RCP<const Epetra_Map>> maps_vec(max_block_number(), Teuchos::null);
  // Make sure, that the block ids and the vector entry ids coincide!
  std::map<Inpar::STR::ModelType, int>::const_iterator ci;
  for (ci = model_block_id_.begin(); ci != model_block_id_.end(); ++ci)
  {
    enum Inpar::STR::ModelType mt = ci->first;
    int bid = ci->second;
    maps_vec[bid] = model_maps_.at(mt);
  }
  blockextractor_.setup(*gproblem_map_ptr_, maps_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::setup_element_technology_map_extractors()
{
  check_init();

  // loop all active element technologies
  const std::set<enum Inpar::STR::EleTech>& ele_techs = datasdyn_->get_element_technologies();
  for (const enum Inpar::STR::EleTech et : ele_techs)
  {
    // mapextractor for element technology
    Core::LinAlg::MultiMapExtractor mapext;

    switch (et)
    {
      case (Inpar::STR::EleTech::rotvec):
      {
        setup_rot_vec_map_extractor(mapext);
        break;
      }
      case (Inpar::STR::EleTech::pressure):
      {
        setup_press_extractor(mapext);
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
        std::pair<Inpar::STR::EleTech, Core::LinAlg::MultiMapExtractor>(et, mapext));

    if (not check.second) FOUR_C_THROW("Insert failed!");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::MultiMapExtractor&
STR::TimeInt::BaseDataGlobalState::get_element_technology_map_extractor(
    const enum Inpar::STR::EleTech etech) const
{
  if (mapextractors_.find(etech) == mapextractors_.end())
    FOUR_C_THROW("Could not find element technology \"%s\" in map extractors.",
        Inpar::STR::EleTechString(etech).c_str());

  return mapextractors_.at(etech);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::setup_rot_vec_map_extractor(
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

  for (int i = 0; i < discret_->NumMyRowNodes(); ++i)
  {
    Core::Nodes::Node* nodeptr = discret_->lRowNode(i);

    const Discret::ELEMENTS::Beam3Base* beameleptr =
        dynamic_cast<const Discret::ELEMENTS::Beam3Base*>(nodeptr->Elements()[0]);

    std::vector<int> nodaladditdofs;
    std::vector<int> nodalrotvecdofs;

    // so far we only expect DoFs of beam elements for the rotvecdofset
    if (beameleptr == nullptr)
    {
      nodaladditdofs = discret_->Dof(0, nodeptr);
    }
    else
    {
      Teuchos::RCP<Core::FE::Discretization> discret = discret_;
      nodaladditdofs = beameleptr->GetAdditiveDofGIDs(*discret, *nodeptr);
      nodalrotvecdofs = beameleptr->GetRotVecDofGIDs(*discret, *nodeptr);

      if (nodaladditdofs.size() + nodalrotvecdofs.size() !=
          (unsigned)beameleptr->NumDofPerNode(*nodeptr))
        FOUR_C_THROW("Expected %d DoFs for node with GID %d but collected %d DoFs",
            beameleptr->NumDofPerNode(*nodeptr), discret_->NodeRowMap()->GID(i),
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
  Teuchos::RCP<Epetra_Map> additdofmap = Teuchos::rcp(
      new Epetra_Map(-1, additdofmapvec.size(), additdofmapvec.data(), 0, discret_->Comm()));
  additdofmapvec.clear();

  std::vector<int> rotvecdofmapvec;
  rotvecdofmapvec.reserve(rotvecdofset.size());
  rotvecdofmapvec.assign(rotvecdofset.begin(), rotvecdofset.end());
  rotvecdofset.clear();
  Teuchos::RCP<Epetra_Map> rotvecdofmap = Teuchos::rcp(
      new Epetra_Map(-1, rotvecdofmapvec.size(), rotvecdofmapvec.data(), 0, discret_->Comm()));
  rotvecdofmapvec.clear();

  std::vector<Teuchos::RCP<const Epetra_Map>> maps(2);
  maps[0] = additdofmap;
  maps[1] = rotvecdofmap;

  multimapext.setup(*dof_row_map_view(), maps);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::setup_press_extractor(
    Core::LinAlg::MultiMapExtractor& multimapext)
{
  check_init();

  // identify pressure DOFs
  Core::LinAlg::CreateMapExtractorFromDiscretization(*discret_, 3, multimapext);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::MultiMapExtractor& STR::TimeInt::BaseDataGlobalState::block_extractor() const
{
  // sanity check
  blockextractor_.check_for_valid_map_extractor();
  return blockextractor_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Epetra::Vector> STR::TimeInt::BaseDataGlobalState::create_global_vector(
    const enum VecInitType& vecinittype,
    const Teuchos::RCP<const STR::ModelEvaluator>& modeleval) const
{
  check_init();
  Teuchos::RCP<Epetra_Vector> xvec_ptr =
      Teuchos::rcp(new Epetra_Vector(global_problem_map(), true));

  // switch between the different vector initialization options
  switch (vecinittype)
  {
    /* use the last converged state to construct a new solution vector */
    case VecInitType::last_time_step:
    {
      FOUR_C_ASSERT(!modeleval.is_null(), "We need access to the STR::ModelEvaluator object!");

      std::map<Inpar::STR::ModelType, int>::const_iterator ci;
      for (ci = model_block_id_.begin(); ci != model_block_id_.end(); ++ci)
      {
        // get the partial solution vector of the last time step
        Teuchos::RCP<const Epetra_Vector> model_sol_ptr =
            modeleval->evaluator(ci->first).get_last_time_step_solution_ptr();
        // if there is a partial solution, we insert it into the full vector
        if (not model_sol_ptr.is_null())
          block_extractor().InsertVector(model_sol_ptr, ci->second, xvec_ptr);
        model_sol_ptr = Teuchos::null;
      }
      break;
    }
    /* use the current global state to construct a new solution vector */
    case VecInitType::init_current_state:
    {
      FOUR_C_ASSERT(!modeleval.is_null(), "We need access to the STR::ModelEvaluator object!");

      std::map<Inpar::STR::ModelType, int>::const_iterator ci;
      for (ci = model_block_id_.begin(); ci != model_block_id_.end(); ++ci)
      {
        // get the partial solution vector of the current state
        Teuchos::RCP<const Epetra_Vector> model_sol_ptr =
            modeleval->evaluator(ci->first).get_current_solution_ptr();
        // if there is a partial solution, we insert it into the full vector
        if (not model_sol_ptr.is_null())
          block_extractor().InsertVector(model_sol_ptr, ci->second, xvec_ptr);
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

  // wrap and return
  return Teuchos::rcp(new ::NOX::Epetra::Vector(xvec_ptr, ::NOX::Epetra::Vector::CreateView));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::SparseOperator*
STR::TimeInt::BaseDataGlobalState::create_structural_stiffness_matrix_block()
{
  stiff_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dof_row_map_view(), 81, true, true));

  return stiff_.get();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseOperator>& STR::TimeInt::BaseDataGlobalState::create_jacobian()
{
  check_init();
  jac_ = Teuchos::null;

  if (max_block_num_ > 1)
  {
    jac_ =
        Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
            block_extractor(), block_extractor(), 81, true, true));
  }
  else
  {
    // pure structural case
    jac_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dof_row_map_view(), 81, true, true));
  }

  return jac_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseOperator> STR::TimeInt::BaseDataGlobalState::create_aux_jacobian()
    const
{
  check_init();
  Teuchos::RCP<Core::LinAlg::SparseOperator> jac = Teuchos::null;

  if (max_block_num_ > 1)
  {
    jac =
        Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
            block_extractor(), block_extractor(), 81, true, true));
  }
  else
  {
    // pure structural case
    jac = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dof_row_map_view(), 81, true, true));
  }

  return jac;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> STR::TimeInt::BaseDataGlobalState::dof_row_map() const
{
  check_init();
  const Epetra_Map* dofrowmap_ptr = discret_->dof_row_map();
  // since it's const, we do not need to copy the map
  return Teuchos::rcp(dofrowmap_ptr, false);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> STR::TimeInt::BaseDataGlobalState::dof_row_map(unsigned nds) const
{
  check_init();
  const Epetra_Map* dofrowmap_ptr = discret_->dof_row_map(nds);
  // since it's const, we do not need to copy the map
  return Teuchos::rcp(dofrowmap_ptr, false);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Epetra_Map* STR::TimeInt::BaseDataGlobalState::dof_row_map_view() const
{
  check_init();
  return discret_->dof_row_map();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Epetra_Map* STR::TimeInt::BaseDataGlobalState::additive_dof_row_map_view() const
{
  check_init();
  return get_element_technology_map_extractor(Inpar::STR::EleTech::rotvec).Map(0).get();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Epetra_Map* STR::TimeInt::BaseDataGlobalState::rot_vec_dof_row_map_view() const
{
  check_init();
  return get_element_technology_map_extractor(Inpar::STR::EleTech::rotvec).Map(1).get();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> STR::TimeInt::BaseDataGlobalState::extract_displ_entries(
    const Epetra_Vector& source) const
{
  return extract_model_entries(Inpar::STR::model_structure, source);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> STR::TimeInt::BaseDataGlobalState::extract_model_entries(
    const Inpar::STR::ModelType& mt, const Epetra_Vector& source) const
{
  Teuchos::RCP<Epetra_Vector> model_ptr = Teuchos::null;
  // extract from the full state vector
  if (source.Map().NumGlobalElements() == block_extractor().FullMap()->NumGlobalElements())
  {
    model_ptr = block_extractor().ExtractVector(source, model_block_id_.at(mt));
  }
  // copy the vector
  else if (source.Map().NumGlobalElements() == model_maps_.at(mt)->NumGlobalElements())
  {
    model_ptr = Teuchos::rcp(new Epetra_Vector(source));
  }
  // otherwise do a standard export
  else
  {
    model_ptr = Teuchos::rcp(new Epetra_Vector(*model_maps_.at(mt)));
    Core::LinAlg::Export(source, *model_ptr);
  }


  return model_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::remove_element_technologies(
    Teuchos::RCP<Epetra_Vector>& rhs_ptr) const
{
  // loop all active element technologies
  const std::set<enum Inpar::STR::EleTech> ele_techs = datasdyn_->get_element_technologies();

  for (const Inpar::STR::EleTech et : ele_techs)
  {
    switch (et)
    {
      case (Inpar::STR::EleTech::pressure):
      {
        rhs_ptr = get_element_technology_map_extractor(et).ExtractVector(rhs_ptr, 0);
        break;
      }
      // element technology doesn't use extra DOFs: skip
      default:
        continue;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::extract_element_technologies(
    const NOX::Nln::StatusTest::QuantityType checkquantity,
    Teuchos::RCP<Epetra_Vector>& rhs_ptr) const
{
  // convert the given quantity type to an element technology
  enum Inpar::STR::EleTech eletech = STR::Nln::ConvertQuantityType2EleTech(checkquantity);
  switch (eletech)
  {
    case Inpar::STR::EleTech::pressure:
    {
      rhs_ptr = get_element_technology_map_extractor(eletech).ExtractVector(rhs_ptr, 1);
      break;
    }
    default:
    {
      FOUR_C_THROW("Element technology doesn't use any extra DOFs.");
      exit(EXIT_FAILURE);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::apply_element_technology_to_acceleration_system(
    Core::LinAlg::SparseOperator& mass, Epetra_Vector& rhs) const
{
  // loop all active element technologies
  const std::set<enum Inpar::STR::EleTech>& ele_techs = datasdyn_->get_element_technologies();

  for (const enum Inpar::STR::EleTech et : ele_techs)
  {
    switch (et)
    {
      case Inpar::STR::EleTech::pressure:
      {
        // get map extractor
        const Core::LinAlg::MultiMapExtractor& mapext = get_element_technology_map_extractor(et);

        // set 1 on pressure DOFs in mass matrix
        mass.ApplyDirichlet(*mapext.Map(1));

        // set 0 on pressure DOFs in rhs
        const Epetra_Vector zeros(*mapext.Map(1), true);
        mapext.InsertVector(zeros, 1, rhs);

        break;
      }
      // element technology doesn't use extra DOFs: skip
      default:
        break;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> STR::TimeInt::BaseDataGlobalState::extract_additive_entries(
    const Epetra_Vector& source) const
{
  Teuchos::RCP<Epetra_Vector> addit_ptr =
      get_element_technology_map_extractor(Inpar::STR::EleTech::rotvec).ExtractVector(source, 0);

  return addit_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> STR::TimeInt::BaseDataGlobalState::extract_rot_vec_entries(
    const Epetra_Vector& source) const
{
  Teuchos::RCP<Epetra_Vector> addit_ptr =
      get_element_technology_map_extractor(Inpar::STR::EleTech::rotvec).ExtractVector(source, 1);

  return addit_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::assign_model_block(Core::LinAlg::SparseOperator& jac,
    const Core::LinAlg::SparseMatrix& matrix, const Inpar::STR::ModelType& mt,
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
        blockmat_ptr->Matrix(0, 0).Assign(access, matrix);
        break;
      }
      case MatBlockType::displ_lm:
      {
        blockmat_ptr->Matrix(0, b_id).Assign(access, matrix);
        break;
      }
      case MatBlockType::lm_displ:
      {
        blockmat_ptr->Matrix(b_id, 0).Assign(access, matrix);
        break;
      }
      case MatBlockType::lm_lm:
      {
        blockmat_ptr->Matrix(b_id, b_id).Assign(access, matrix);
        break;
      }
      default:
      {
        FOUR_C_THROW("model block %s is not supported", MatBlockType2String(bt).c_str());
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
    stiff_ptr->Assign(access, matrix);
    return;
  }

  FOUR_C_THROW(
      "The jacobian has the wrong type! (no Core::LinAlg::SparseMatrix "
      "and no Core::LinAlg::BlockSparseMatrix)");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseMatrix> STR::TimeInt::BaseDataGlobalState::extract_model_block(
    Core::LinAlg::SparseOperator& jac, const Inpar::STR::ModelType& mt,
    const MatBlockType& bt) const
{
  Teuchos::RCP<Core::LinAlg::SparseMatrix> block = Teuchos::null;
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
        block = Teuchos::rcpFromRef(blockmat_ptr->Matrix(0, 0));
        break;
      }
      case MatBlockType::displ_lm:
      {
        block = Teuchos::rcpFromRef(blockmat_ptr->Matrix(0, b_id));
        break;
      }
      case MatBlockType::lm_displ:
      {
        block = Teuchos::rcpFromRef(blockmat_ptr->Matrix(b_id, 0));
        break;
      }
      case MatBlockType::lm_lm:
      {
        block = Teuchos::rcpFromRef(blockmat_ptr->Matrix(b_id, b_id));
        break;
      }
      default:
      {
        FOUR_C_THROW("model block %s is not supported", MatBlockType2String(bt).c_str());
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
    block = Teuchos::rcpFromRef(*stiff_ptr);
    return block;
  }

  FOUR_C_THROW(
      "The jacobian has the wrong type! (no Core::LinAlg::SparseMatrix "
      "and no Core::LinAlg::BlockSparseMatrix)");
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<Core::LinAlg::SparseMatrix*>>
STR::TimeInt::BaseDataGlobalState::extract_displ_row_of_blocks(
    Core::LinAlg::SparseOperator& jac) const
{
  return extract_row_of_blocks(jac, Inpar::STR::model_structure);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<Core::LinAlg::SparseMatrix*>>
STR::TimeInt::BaseDataGlobalState::extract_row_of_blocks(
    Core::LinAlg::SparseOperator& jac, const Inpar::STR::ModelType& mt) const
{
  Teuchos::RCP<std::vector<Core::LinAlg::SparseMatrix*>> rowofblocks = Teuchos::null;

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

    const int num_cols = blockmat_ptr->Cols();
    rowofblocks = Teuchos::rcp(new std::vector<Core::LinAlg::SparseMatrix*>(num_cols, nullptr));

    for (int i = 0; i < num_cols; ++i) (*rowofblocks)[i] = &(blockmat_ptr->Matrix(b_id, i));

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
    rowofblocks = Teuchos::rcp(new std::vector<Core::LinAlg::SparseMatrix*>(1, nullptr));
    (*rowofblocks)[0] = stiff_ptr;
    return rowofblocks;
  }

  FOUR_C_THROW(
      "The jacobian has the wrong type! (no Core::LinAlg::SparseMatrix "
      "and no Core::LinAlg::BlockSparseMatrix)");
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseMatrix> STR::TimeInt::BaseDataGlobalState::extract_displ_block(
    Core::LinAlg::SparseOperator& jac) const
{
  return extract_model_block(jac, Inpar::STR::model_structure, MatBlockType::displ_displ);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Core::LinAlg::SparseMatrix>
STR::TimeInt::BaseDataGlobalState::get_jacobian_displ_block() const
{
  FOUR_C_ASSERT(!jac_.is_null(), "The jacobian is not initialized!");
  return extract_displ_block(*jac_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseMatrix> STR::TimeInt::BaseDataGlobalState::jacobian_displ_block()
{
  FOUR_C_ASSERT(!jac_.is_null(), "The jacobian is not initialized!");
  return extract_displ_block(*jac_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Core::LinAlg::SparseMatrix>
STR::TimeInt::BaseDataGlobalState::get_jacobian_block(
    const Inpar::STR::ModelType mt, const MatBlockType bt) const
{
  FOUR_C_ASSERT(!jac_.is_null(), "The jacobian is not initialized!");

  return extract_model_block(*jac_, mt, bt);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int STR::TimeInt::BaseDataGlobalState::get_last_lin_iteration_number(const unsigned step) const
{
  check_init_setup();
  if (step < 1) FOUR_C_THROW("The given step number must be larger than 1. (step=%d)", step);

  auto linsolvers = datasdyn_->GetLinSolvers();
  int iter = -1;

  for (auto& linsolver : linsolvers)
  {
    switch (linsolver.first)
    {
      // has only one field solver per default
      case Inpar::STR::model_structure:
      case Inpar::STR::model_springdashpot:
      case Inpar::STR::model_browniandyn:
      case Inpar::STR::model_beaminteraction:
      case Inpar::STR::model_basic_coupling:
      case Inpar::STR::model_monolithic_coupling:
      case Inpar::STR::model_partitioned_coupling:
      case Inpar::STR::model_beam_interaction_old:
      {
        iter = linsolvers[linsolver.first]->getNumIters();
        break;
      }
      default:
        FOUR_C_THROW(
            "The given model type '%s' is not supported for linear iteration output right now.",
            Inpar::STR::model_structure);
    }
  }

  return iter;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int STR::TimeInt::BaseDataGlobalState::get_nln_iteration_number(const unsigned step) const
{
  check_init_setup();
  if (step < 1) FOUR_C_THROW("The given step number must be larger than 1. (step=%d)", step);

  auto cit = nln_iter_numbers_.begin();
  while (cit != nln_iter_numbers_.end())
  {
    if (cit->first == static_cast<int>(step)) return cit->second;
    ++cit;
  }

  FOUR_C_THROW("There is no nonlinear iteration number for the given step %d.", step);
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TimeInt::BaseDataGlobalState::set_nln_iteration_number(const int nln_iter)
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
            "for step %d.",
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
    const Teuchos::RCP<const STR::TimeInt::BaseDataGlobalState>& gstate_ptr)
    : gstate_ptr_(gstate_ptr)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::GROUP::PrePostOp::TimeInt::RotVecUpdater::runPreComputeX(
    const NOX::Nln::Group& input_grp, const Epetra_Vector& dir, const double& step,
    const NOX::Nln::Group& curr_grp)
{
  const Epetra_Vector& xold =
      dynamic_cast<const ::NOX::Epetra::Vector&>(input_grp.getX()).getEpetraVector();

  // cast the const away so that the new x vector can be set after the update
  NOX::Nln::Group& curr_grp_mutable = const_cast<NOX::Nln::Group&>(curr_grp);

  Teuchos::RCP<Epetra_Vector> xnew = Teuchos::rcp(new Epetra_Vector(xold.Map(), true));

  /* we do the multiplicative update only for those entries which belong to
   * rotation (pseudo-)vectors */
  Epetra_Vector x_rotvec = *gstate_ptr_->extract_rot_vec_entries(xold);
  Epetra_Vector dir_rotvec = *gstate_ptr_->extract_rot_vec_entries(dir);

  Core::LinAlg::Matrix<4, 1> Qold;
  Core::LinAlg::Matrix<4, 1> deltaQ;
  Core::LinAlg::Matrix<4, 1> Qnew;

  /* since parallel distribution is node-wise, the three entries belonging to
   * a rotation vector should be stored on the same processor: safety-check */
  if (x_rotvec.Map().NumMyElements() % 3 != 0 or dir_rotvec.Map().NumMyElements() % 3 != 0)
    FOUR_C_THROW(
        "fatal error: apparently, the three DOFs of a nodal rotation vector are"
        " not stored on this processor. Can't apply multiplicative update!");

  // rotation vectors always consist of three consecutive DoFs
  for (int i = 0; i < x_rotvec.Map().NumMyElements(); i = i + 3)
  {
    // create a Core::LinAlg::Matrix from reference to three x vector entries
    Core::LinAlg::Matrix<3, 1> theta(&x_rotvec[i], true);
    Core::LargeRotations::angletoquaternion(theta, Qold);

    // same for relative rotation angle deltatheta
    Core::LinAlg::Matrix<3, 1> deltatheta(&dir_rotvec[i], true);
    deltatheta.Scale(step);

    Core::LargeRotations::angletoquaternion(deltatheta, deltaQ);
    Core::LargeRotations::quaternionproduct(Qold, deltaQ, Qnew);
    Core::LargeRotations::quaterniontoangle(Qnew, theta);
  }

  // first update entire x vector in an additive manner
  xnew->Update(1.0, xold, step, dir, 0.0);

  // now replace the rotvec entries by the correct value computed before
  Core::LinAlg::AssembleMyVector(0.0, *xnew, 1.0, x_rotvec);
  curr_grp_mutable.setX(xnew);

  /* tell the NOX::Nln::Group that the x vector has already been updated in
   * this preComputeX operator call */
  curr_grp_mutable.setSkipUpdateX(true);
}

FOUR_C_NAMESPACE_CLOSE
