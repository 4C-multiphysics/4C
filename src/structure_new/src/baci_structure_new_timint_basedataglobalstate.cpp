/*-----------------------------------------------------------*/
/*! \file

\brief Global state data container for the structural (time) integration


\level 3

*/
/*-----------------------------------------------------------*/

#include "baci_structure_new_timint_basedataglobalstate.hpp"

#include "baci_beam3_base.hpp"
#include "baci_contact_meshtying_abstract_strategy.hpp"
#include "baci_discretization_fem_general_largerotations.hpp"
#include "baci_global_data.hpp"
#include "baci_inpar_contact.hpp"
#include "baci_lib_utils_discret.hpp"
#include "baci_linalg_sparsematrix.hpp"
#include "baci_linalg_utils_sparse_algebra_assemble.hpp"
#include "baci_linalg_utils_sparse_algebra_create.hpp"
#include "baci_linalg_utils_sparse_algebra_manipulation.hpp"
#include "baci_linear_solver_method_linalg.hpp"
#include "baci_solver_nonlin_nox_group.hpp"
#include "baci_solver_nonlin_nox_group_prepostoperator.hpp"
#include "baci_structure_new_model_evaluator.hpp"
#include "baci_structure_new_model_evaluator_generic.hpp"
#include "baci_structure_new_model_evaluator_meshtying.hpp"
#include "baci_structure_new_timint_basedatasdyn.hpp"
#include "baci_structure_new_utils.hpp"

#include <Epetra_Vector.h>
#include <NOX_Epetra_Vector.H>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::TIMINT::BaseDataGlobalState::BaseDataGlobalState()
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
STR::TIMINT::BaseDataGlobalState& STR::TIMINT::BaseDataGlobalState::operator=(
    const STR::TIMINT::BaseDataGlobalState& source)
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
void STR::TIMINT::BaseDataGlobalState::Init(const Teuchos::RCP<DRT::Discretization> discret,
    const Teuchos::ParameterList& sdynparams, const Teuchos::RCP<const BaseDataSDyn> datasdyn)
{
  // We have to call Setup() after Init()
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
        new TIMESTEPPING::TimIntMStep<double>(0, 0, sdynparams.get<double>("TIMEINIT")));
    dt_ = Teuchos::rcp(
        new TIMESTEPPING::TimIntMStep<double>(0, 0, sdynparams.get<double>("TIMESTEP")));

    // initialize target time to initial time plus step size
    timenp_ = (*timen_)[0] + (*dt_)[0];
    stepnp_ = stepn_ + 1;

    // initialize restart step
    restartstep_ = GLOBAL::Problem::Instance()->Restart();
    if (restartstep_ < 0) FOUR_C_THROW("The restart step is expected to be positive.");
  }

  // end of initialization
  isinit_ = true;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataGlobalState::Setup()
{
  // safety check
  CheckInit();

  // --------------------------------------
  // control parameters
  // --------------------------------------
  timer_ = Teuchos::rcp(new Teuchos::Time("", true));

  // --------------------------------------
  // vectors
  // --------------------------------------
  // displacements D_{n}
  dis_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));
  // velocities V_{n}
  vel_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));
  // accelerations A_{n}
  acc_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));

  // displacements D_{n+1} at t_{n+1}
  disnp_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // velocities V_{n+1} at t_{n+1}
  velnp_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // accelerations A_{n+1} at t_{n+1}
  accnp_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  fintn_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  fintnp_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  fextn_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  fextnp_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  freactn_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  freactnp_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  finertialn_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  finertialnp_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  fviscon_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  fvisconp_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  fstructold_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  // --------------------------------------
  // sparse operators
  // --------------------------------------
  mass_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*DofRowMapView(), 81, true, true));
  if (datasdyn_->GetDampingType() != INPAR::STR::damp_none)
  {
    if (datasdyn_->GetMassLinType() == INPAR::STR::ml_none)
    {
      damp_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*DofRowMapView(), 81, true, true));
    }
    else
    {
      /* Since our element evaluate routine is only designed for two input matrices
       * (stiffness and damping or stiffness and mass) its not possible, to have nonlinear
       * inertia forces AND material damping. */
      FOUR_C_THROW("So far it is not possible to model nonlinear inertia forces and damping!");
    }
  }

  if (datasdyn_->GetDynamicType() == INPAR::STR::dyna_statics and
      datasdyn_->GetMassLinType() != INPAR::STR::ml_none)
    FOUR_C_THROW(
        "Do not set parameter MASSLIN in static simulations as this leads to undesired"
        " evaluation of mass matrix on element level!");

  SetInitialFields();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataGlobalState::SetInitialFields()
{
  // set initial velocity field if existing
  const std::string field = "Velocity";
  std::vector<int> localdofs;
  localdofs.push_back(0);
  localdofs.push_back(1);
  localdofs.push_back(2);
  DRT::UTILS::EvaluateInitialField(*discret_, field, velnp_, localdofs);

  // set initial porosity field if existing
  const std::string porosityfield = "Porosity";
  std::vector<int> porositylocaldofs;
  porositylocaldofs.push_back(GLOBAL::Problem::Instance()->NDim());
  DRT::UTILS::EvaluateInitialField(*discret_, porosityfield, (*dis_)(0), porositylocaldofs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Epetra::Vector> STR::TIMINT::BaseDataGlobalState::CreateGlobalVector() const
{
  return CreateGlobalVector(VecInitType::zero, Teuchos::null);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int STR::TIMINT::BaseDataGlobalState::SetupBlockInformation(
    const STR::MODELEVALUATOR::Generic& me, const INPAR::STR::ModelType& mt)
{
  CheckInit();
  GLOBAL::Problem* problem = GLOBAL::Problem::Instance();
  Teuchos::RCP<const Epetra_Map> me_map_ptr = me.GetBlockDofRowMapPtr();

  model_maps_[mt] = me_map_ptr;

  switch (mt)
  {
    case INPAR::STR::model_structure:
    {
      // always called first, so we can use it to reset things
      gproblem_map_ptr_ = Teuchos::null;
      model_block_id_[mt] = 0;
      max_block_num_ = 1;
      break;
    }
    case INPAR::STR::model_contact:
    {
      enum INPAR::CONTACT::SystemType systype =
          CORE::UTILS::IntegralValue<INPAR::CONTACT::SystemType>(
              problem->ContactDynamicParams(), "SYSTEM");

      enum INPAR::CONTACT::SolvingStrategy soltype =
          CORE::UTILS::IntegralValue<INPAR::CONTACT::SolvingStrategy>(
              problem->ContactDynamicParams(), "STRATEGY");

      // systems without additional dofs
      if (soltype == INPAR::CONTACT::solution_nitsche ||
          soltype == INPAR::CONTACT::solution_penalty ||
          soltype == INPAR::CONTACT::solution_uzawa ||
          soltype == INPAR::CONTACT::solution_multiscale)
      {
        model_block_id_[mt] = 0;
      }
      // --- saddle-point system
      else if (systype == INPAR::CONTACT::system_saddlepoint)
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
    case INPAR::STR::model_meshtying:
    {
      const STR::MODELEVALUATOR::Meshtying& mt_me =
          dynamic_cast<const STR::MODELEVALUATOR::Meshtying&>(me);

      enum INPAR::CONTACT::SystemType systype = mt_me.Strategy().SystemType();

      enum INPAR::CONTACT::SolvingStrategy soltype =
          CORE::UTILS::IntegralValue<INPAR::CONTACT::SolvingStrategy>(
              mt_me.Strategy().Params(), "STRATEGY");

      // systems without additional dofs
      if (soltype == INPAR::CONTACT::solution_nitsche ||
          soltype == INPAR::CONTACT::solution_penalty ||
          soltype == INPAR::CONTACT::solution_uzawa ||
          soltype == INPAR::CONTACT::solution_multiscale)
      {
        model_block_id_[mt] = 0;
      }
      // --- saddle-point system
      else if (systype == INPAR::CONTACT::system_saddlepoint)
      {
        model_block_id_[mt] = max_block_num_;
        ++max_block_num_;
      }
      // --- condensed system
      else if (systype == INPAR::CONTACT::system_condensed)
      {
        model_block_id_[mt] = 0;
      }
      else
        FOUR_C_THROW("I don't know what to do");
      break;
    }
    case INPAR::STR::model_cardiovascular0d:
    {
      // --- 2x2 block system
      model_block_id_[mt] = max_block_num_;
      ++max_block_num_;
      break;
    }
    case INPAR::STR::model_lag_pen_constraint:
    {
      // ----------------------------------------------------------------------
      // check type of constraint conditions (Lagrange multiplier vs. penalty)
      // ----------------------------------------------------------------------
      bool have_lag_constraint = false;
      std::vector<DRT::Condition*> lagcond_volconstr3d(0);
      std::vector<DRT::Condition*> lagcond_areaconstr3d(0);
      std::vector<DRT::Condition*> lagcond_areaconstr2d(0);
      std::vector<DRT::Condition*> lagcond_mpconline2d(0);
      std::vector<DRT::Condition*> lagcond_mpconplane3d(0);
      std::vector<DRT::Condition*> lagcond_mpcnormcomp3d(0);
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
    case INPAR::STR::model_springdashpot:
    case INPAR::STR::model_beam_interaction_old:
    case INPAR::STR::model_browniandyn:
    case INPAR::STR::model_beaminteraction:
    {
      // structural block
      model_block_id_[mt] = 0;
      break;
    }
    case INPAR::STR::model_basic_coupling:
    case INPAR::STR::model_monolithic_coupling:
    case INPAR::STR::model_partitioned_coupling:
    {
      // do nothing
      break;
    }
    case INPAR::STR::model_constraints:
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
  gproblem_map_ptr_ = CORE::LINALG::MergeMap(gproblem_map_ptr_, me_map_ptr);

  return gproblem_map_ptr_->MaxAllGID();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataGlobalState::SetupMultiMapExtractor()
{
  CheckInit();
  /* copy the std::map into a std::vector and keep the numbering of the model-id
   * map */
  std::vector<Teuchos::RCP<const Epetra_Map>> maps_vec(MaxBlockNumber(), Teuchos::null);
  // Make sure, that the block ids and the vector entry ids coincide!
  std::map<INPAR::STR::ModelType, int>::const_iterator ci;
  for (ci = model_block_id_.begin(); ci != model_block_id_.end(); ++ci)
  {
    enum INPAR::STR::ModelType mt = ci->first;
    int bid = ci->second;
    maps_vec[bid] = model_maps_.at(mt);
  }
  blockextractor_.Setup(*gproblem_map_ptr_, maps_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataGlobalState::SetupElementTechnologyMapExtractors()
{
  CheckInit();

  // loop all active element technologies
  const std::set<enum INPAR::STR::EleTech>& ele_techs = datasdyn_->GetElementTechnologies();
  for (const enum INPAR::STR::EleTech et : ele_techs)
  {
    // mapextractor for element technology
    CORE::LINALG::MultiMapExtractor mapext;

    switch (et)
    {
      case (INPAR::STR::EleTech::rotvec):
      {
        SetupRotVecMapExtractor(mapext);
        break;
      }
      case (INPAR::STR::EleTech::pressure):
      {
        SetupPressExtractor(mapext);
        break;
      }
      // element technology doesn't require a map extractor: skip
      default:
        continue;
    }

    // sanity check
    mapext.CheckForValidMapExtractor();

    // insert into map
    const auto check = mapextractors_.insert(
        std::pair<INPAR::STR::EleTech, CORE::LINALG::MultiMapExtractor>(et, mapext));

    if (not check.second) FOUR_C_THROW("Insert failed!");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const CORE::LINALG::MultiMapExtractor&
STR::TIMINT::BaseDataGlobalState::GetElementTechnologyMapExtractor(
    const enum INPAR::STR::EleTech etech) const
{
  if (mapextractors_.find(etech) == mapextractors_.end())
    FOUR_C_THROW("Could not find element technology \"%s\" in map extractors.",
        INPAR::STR::EleTechString(etech).c_str());

  return mapextractors_.at(etech);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataGlobalState::SetupRotVecMapExtractor(
    CORE::LINALG::MultiMapExtractor& multimapext)
{
  CheckInit();

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
    DRT::Node* nodeptr = discret_->lRowNode(i);

    const DRT::ELEMENTS::Beam3Base* beameleptr =
        dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(nodeptr->Elements()[0]);

    std::vector<int> nodaladditdofs;
    std::vector<int> nodalrotvecdofs;

    // so far we only expect DoFs of beam elements for the rotvecdofset
    if (beameleptr == nullptr)
    {
      nodaladditdofs = discret_->Dof(0, nodeptr);
    }
    else
    {
      Teuchos::RCP<DRT::Discretization> discret = discret_;
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

  multimapext.Setup(*DofRowMapView(), maps);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataGlobalState::SetupPressExtractor(
    CORE::LINALG::MultiMapExtractor& multimapext)
{
  CheckInit();

  // identify pressure DOFs
  CORE::LINALG::CreateMapExtractorFromDiscretization(*discret_, 3, multimapext);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const CORE::LINALG::MultiMapExtractor& STR::TIMINT::BaseDataGlobalState::BlockExtractor() const
{
  // sanity check
  blockextractor_.CheckForValidMapExtractor();
  return blockextractor_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Epetra::Vector> STR::TIMINT::BaseDataGlobalState::CreateGlobalVector(
    const enum VecInitType& vecinittype,
    const Teuchos::RCP<const STR::ModelEvaluator>& modeleval_ptr) const
{
  CheckInit();
  Teuchos::RCP<Epetra_Vector> xvec_ptr = Teuchos::rcp(new Epetra_Vector(GlobalProblemMap(), true));

  // switch between the different vector initialization options
  switch (vecinittype)
  {
    /* use the last converged state to construct a new solution vector */
    case VecInitType::last_time_step:
    {
      FOUR_C_ASSERT(!modeleval_ptr.is_null(), "We need access to the STR::ModelEvaluator object!");

      std::map<INPAR::STR::ModelType, int>::const_iterator ci;
      for (ci = model_block_id_.begin(); ci != model_block_id_.end(); ++ci)
      {
        // get the partial solution vector of the last time step
        Teuchos::RCP<const Epetra_Vector> model_sol_ptr =
            modeleval_ptr->Evaluator(ci->first).GetLastTimeStepSolutionPtr();
        // if there is a partial solution, we insert it into the full vector
        if (not model_sol_ptr.is_null())
          BlockExtractor().InsertVector(model_sol_ptr, ci->second, xvec_ptr);
        model_sol_ptr = Teuchos::null;
      }
      break;
    }
    /* use the current global state to construct a new solution vector */
    case VecInitType::init_current_state:
    {
      FOUR_C_ASSERT(!modeleval_ptr.is_null(), "We need access to the STR::ModelEvaluator object!");

      std::map<INPAR::STR::ModelType, int>::const_iterator ci;
      for (ci = model_block_id_.begin(); ci != model_block_id_.end(); ++ci)
      {
        // get the partial solution vector of the current state
        Teuchos::RCP<const Epetra_Vector> model_sol_ptr =
            modeleval_ptr->Evaluator(ci->first).GetCurrentSolutionPtr();
        // if there is a partial solution, we insert it into the full vector
        if (not model_sol_ptr.is_null())
          BlockExtractor().InsertVector(model_sol_ptr, ci->second, xvec_ptr);
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
CORE::LINALG::SparseOperator*
STR::TIMINT::BaseDataGlobalState::CreateStructuralStiffnessMatrixBlock()
{
  stiff_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*DofRowMapView(), 81, true, true));

  return stiff_.get();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::SparseOperator>& STR::TIMINT::BaseDataGlobalState::CreateJacobian()
{
  CheckInit();
  jac_ = Teuchos::null;

  if (max_block_num_ > 1)
  {
    jac_ =
        Teuchos::rcp(new CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>(
            BlockExtractor(), BlockExtractor(), 81, true, true));
  }
  else
  {
    // pure structural case
    jac_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*DofRowMapView(), 81, true, true));
  }

  return jac_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::SparseOperator> STR::TIMINT::BaseDataGlobalState::CreateAuxJacobian()
    const
{
  CheckInit();
  Teuchos::RCP<CORE::LINALG::SparseOperator> jac = Teuchos::null;

  if (max_block_num_ > 1)
  {
    jac =
        Teuchos::rcp(new CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>(
            BlockExtractor(), BlockExtractor(), 81, true, true));
  }
  else
  {
    // pure structural case
    jac = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*DofRowMapView(), 81, true, true));
  }

  return jac;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> STR::TIMINT::BaseDataGlobalState::DofRowMap() const
{
  CheckInit();
  const Epetra_Map* dofrowmap_ptr = discret_->DofRowMap();
  // since it's const, we do not need to copy the map
  return Teuchos::rcp(dofrowmap_ptr, false);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> STR::TIMINT::BaseDataGlobalState::DofRowMap(unsigned nds) const
{
  CheckInit();
  const Epetra_Map* dofrowmap_ptr = discret_->DofRowMap(nds);
  // since it's const, we do not need to copy the map
  return Teuchos::rcp(dofrowmap_ptr, false);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Epetra_Map* STR::TIMINT::BaseDataGlobalState::DofRowMapView() const
{
  CheckInit();
  return discret_->DofRowMap();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Epetra_Map* STR::TIMINT::BaseDataGlobalState::AdditiveDofRowMapView() const
{
  CheckInit();
  return GetElementTechnologyMapExtractor(INPAR::STR::EleTech::rotvec).Map(0).get();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Epetra_Map* STR::TIMINT::BaseDataGlobalState::RotVecDofRowMapView() const
{
  CheckInit();
  return GetElementTechnologyMapExtractor(INPAR::STR::EleTech::rotvec).Map(1).get();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> STR::TIMINT::BaseDataGlobalState::ExtractDisplEntries(
    const Epetra_Vector& source) const
{
  return ExtractModelEntries(INPAR::STR::model_structure, source);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> STR::TIMINT::BaseDataGlobalState::ExtractModelEntries(
    const INPAR::STR::ModelType& mt, const Epetra_Vector& source) const
{
  Teuchos::RCP<Epetra_Vector> model_ptr = Teuchos::null;
  // extract from the full state vector
  if (source.Map().NumGlobalElements() == BlockExtractor().FullMap()->NumGlobalElements())
  {
    model_ptr = BlockExtractor().ExtractVector(source, model_block_id_.at(mt));
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
    CORE::LINALG::Export(source, *model_ptr);
  }


  return model_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataGlobalState::RemoveElementTechnologies(
    Teuchos::RCP<Epetra_Vector>& rhs_ptr) const
{
  // loop all active element technologies
  const std::set<enum INPAR::STR::EleTech> ele_techs = datasdyn_->GetElementTechnologies();

  for (const INPAR::STR::EleTech et : ele_techs)
  {
    switch (et)
    {
      case (INPAR::STR::EleTech::pressure):
      {
        rhs_ptr = GetElementTechnologyMapExtractor(et).ExtractVector(rhs_ptr, 0);
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
void STR::TIMINT::BaseDataGlobalState::ExtractElementTechnologies(
    const NOX::NLN::StatusTest::QuantityType checkquantity,
    Teuchos::RCP<Epetra_Vector>& rhs_ptr) const
{
  // convert the given quantity type to an element technology
  enum INPAR::STR::EleTech eletech = STR::NLN::ConvertQuantityType2EleTech(checkquantity);
  switch (eletech)
  {
    case INPAR::STR::EleTech::pressure:
    {
      rhs_ptr = GetElementTechnologyMapExtractor(eletech).ExtractVector(rhs_ptr, 1);
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
void STR::TIMINT::BaseDataGlobalState::ApplyElementTechnologyToAccelerationSystem(
    CORE::LINALG::SparseOperator& mass, Epetra_Vector& rhs) const
{
  // loop all active element technologies
  const std::set<enum INPAR::STR::EleTech>& ele_techs = datasdyn_->GetElementTechnologies();

  for (const enum INPAR::STR::EleTech et : ele_techs)
  {
    switch (et)
    {
      case INPAR::STR::EleTech::pressure:
      {
        // get map extractor
        const CORE::LINALG::MultiMapExtractor& mapext = GetElementTechnologyMapExtractor(et);

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
Teuchos::RCP<Epetra_Vector> STR::TIMINT::BaseDataGlobalState::ExtractAdditiveEntries(
    const Epetra_Vector& source) const
{
  Teuchos::RCP<Epetra_Vector> addit_ptr =
      GetElementTechnologyMapExtractor(INPAR::STR::EleTech::rotvec).ExtractVector(source, 0);

  return addit_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> STR::TIMINT::BaseDataGlobalState::ExtractRotVecEntries(
    const Epetra_Vector& source) const
{
  Teuchos::RCP<Epetra_Vector> addit_ptr =
      GetElementTechnologyMapExtractor(INPAR::STR::EleTech::rotvec).ExtractVector(source, 1);

  return addit_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataGlobalState::AssignModelBlock(CORE::LINALG::SparseOperator& jac,
    const CORE::LINALG::SparseMatrix& matrix, const INPAR::STR::ModelType& mt,
    const MatBlockType& bt, const CORE::LINALG::DataAccess& access) const
{
  CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>* blockmat_ptr =
      dynamic_cast<CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>*>(
          &jac);
  if (blockmat_ptr != nullptr)
  {
    if (MaxBlockNumber() < 2)
      FOUR_C_THROW(
          "The jacobian is a CORE::LINALG::BlockSparseMatrix but has less than"
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

  CORE::LINALG::SparseMatrix* stiff_ptr = dynamic_cast<CORE::LINALG::SparseMatrix*>(&jac);
  if (stiff_ptr != nullptr)
  {
    stiff_ptr->Assign(access, matrix);
    return;
  }

  FOUR_C_THROW(
      "The jacobian has the wrong type! (no CORE::LINALG::SparseMatrix "
      "and no CORE::LINALG::BlockSparseMatrix)");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::SparseMatrix> STR::TIMINT::BaseDataGlobalState::ExtractModelBlock(
    CORE::LINALG::SparseOperator& jac, const INPAR::STR::ModelType& mt,
    const MatBlockType& bt) const
{
  Teuchos::RCP<CORE::LINALG::SparseMatrix> block = Teuchos::null;
  CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>* blockmat_ptr =
      dynamic_cast<CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>*>(
          &jac);
  if (blockmat_ptr != nullptr)
  {
    if (MaxBlockNumber() < 2)
      FOUR_C_THROW(
          "The jacobian is a CORE::LINALG::BlockSparseMatrix but has less than"
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

  CORE::LINALG::SparseMatrix* stiff_ptr = dynamic_cast<CORE::LINALG::SparseMatrix*>(&jac);
  if (stiff_ptr != nullptr)
  {
    block = Teuchos::rcpFromRef(*stiff_ptr);
    return block;
  }

  FOUR_C_THROW(
      "The jacobian has the wrong type! (no CORE::LINALG::SparseMatrix "
      "and no CORE::LINALG::BlockSparseMatrix)");
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<CORE::LINALG::SparseMatrix*>>
STR::TIMINT::BaseDataGlobalState::ExtractDisplRowOfBlocks(CORE::LINALG::SparseOperator& jac) const
{
  return ExtractRowOfBlocks(jac, INPAR::STR::model_structure);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<CORE::LINALG::SparseMatrix*>>
STR::TIMINT::BaseDataGlobalState::ExtractRowOfBlocks(
    CORE::LINALG::SparseOperator& jac, const INPAR::STR::ModelType& mt) const
{
  Teuchos::RCP<std::vector<CORE::LINALG::SparseMatrix*>> rowofblocks = Teuchos::null;

  CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>* blockmat_ptr =
      dynamic_cast<CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>*>(
          &jac);
  if (blockmat_ptr != nullptr)
  {
    if (MaxBlockNumber() < 2)
      FOUR_C_THROW(
          "The jacobian is a CORE::LINALG::BlockSparseMatrix but has less than"
          " two blocks! Seems wrong.");
    const int& b_id = model_block_id_.at(mt);

    const int num_cols = blockmat_ptr->Cols();
    rowofblocks = Teuchos::rcp(new std::vector<CORE::LINALG::SparseMatrix*>(num_cols, nullptr));

    for (int i = 0; i < num_cols; ++i) (*rowofblocks)[i] = &(blockmat_ptr->Matrix(b_id, i));

    return rowofblocks;
  }

  // sanity check
  if (model_block_id_.find(mt) == model_block_id_.end())
    FOUR_C_THROW(
        "It seems as you are trying to access a matrix block row which has "
        "not been created.");

  CORE::LINALG::SparseMatrix* stiff_ptr = dynamic_cast<CORE::LINALG::SparseMatrix*>(&jac);
  if (stiff_ptr != nullptr)
  {
    rowofblocks = Teuchos::rcp(new std::vector<CORE::LINALG::SparseMatrix*>(1, nullptr));
    (*rowofblocks)[0] = stiff_ptr;
    return rowofblocks;
  }

  FOUR_C_THROW(
      "The jacobian has the wrong type! (no CORE::LINALG::SparseMatrix "
      "and no CORE::LINALG::BlockSparseMatrix)");
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::SparseMatrix> STR::TIMINT::BaseDataGlobalState::ExtractDisplBlock(
    CORE::LINALG::SparseOperator& jac) const
{
  return ExtractModelBlock(jac, INPAR::STR::model_structure, MatBlockType::displ_displ);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const CORE::LINALG::SparseMatrix>
STR::TIMINT::BaseDataGlobalState::GetJacobianDisplBlock() const
{
  FOUR_C_ASSERT(!jac_.is_null(), "The jacobian is not initialized!");
  return ExtractDisplBlock(*jac_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::SparseMatrix> STR::TIMINT::BaseDataGlobalState::JacobianDisplBlock()
{
  FOUR_C_ASSERT(!jac_.is_null(), "The jacobian is not initialized!");
  return ExtractDisplBlock(*jac_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const CORE::LINALG::SparseMatrix> STR::TIMINT::BaseDataGlobalState::GetJacobianBlock(
    const INPAR::STR::ModelType mt, const MatBlockType bt) const
{
  FOUR_C_ASSERT(!jac_.is_null(), "The jacobian is not initialized!");

  return ExtractModelBlock(*jac_, mt, bt);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int STR::TIMINT::BaseDataGlobalState::GetLastLinIterationNumber(const unsigned step) const
{
  CheckInitSetup();
  if (step < 1) FOUR_C_THROW("The given step number must be larger than 1. (step=%d)", step);

  auto linsolvers = datasdyn_->GetLinSolvers();
  int iter = -1;

  for (auto& linsolver : linsolvers)
  {
    switch (linsolver.first)
    {
      // has only one field solver per default
      case INPAR::STR::model_structure:
      case INPAR::STR::model_springdashpot:
      case INPAR::STR::model_browniandyn:
      case INPAR::STR::model_beaminteraction:
      case INPAR::STR::model_basic_coupling:
      case INPAR::STR::model_monolithic_coupling:
      case INPAR::STR::model_partitioned_coupling:
      case INPAR::STR::model_beam_interaction_old:
      {
        iter = linsolvers[linsolver.first]->getNumIters();
        break;
      }
      default:
        FOUR_C_THROW(
            "The given model type '%s' is not supported for linear iteration output right now.",
            INPAR::STR::model_structure);
    }
  }

  return iter;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int STR::TIMINT::BaseDataGlobalState::GetNlnIterationNumber(const unsigned step) const
{
  CheckInitSetup();
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
void STR::TIMINT::BaseDataGlobalState::SetNlnIterationNumber(const int nln_iter)
{
  CheckInitSetup();

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
NOX::NLN::GROUP::PrePostOp::TIMINT::RotVecUpdater::RotVecUpdater(
    const Teuchos::RCP<const STR::TIMINT::BaseDataGlobalState>& gstate_ptr)
    : gstate_ptr_(gstate_ptr)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::GROUP::PrePostOp::TIMINT::RotVecUpdater::runPreComputeX(
    const NOX::NLN::Group& input_grp, const Epetra_Vector& dir, const double& step,
    const NOX::NLN::Group& curr_grp)
{
  const Epetra_Vector& xold =
      dynamic_cast<const ::NOX::Epetra::Vector&>(input_grp.getX()).getEpetraVector();

  // cast the const away so that the new x vector can be set after the update
  NOX::NLN::Group& curr_grp_mutable = const_cast<NOX::NLN::Group&>(curr_grp);

  Teuchos::RCP<Epetra_Vector> xnew = Teuchos::rcp(new Epetra_Vector(xold.Map(), true));

  /* we do the multiplicative update only for those entries which belong to
   * rotation (pseudo-)vectors */
  Epetra_Vector x_rotvec = *gstate_ptr_->ExtractRotVecEntries(xold);
  Epetra_Vector dir_rotvec = *gstate_ptr_->ExtractRotVecEntries(dir);

  CORE::LINALG::Matrix<4, 1> Qold;
  CORE::LINALG::Matrix<4, 1> deltaQ;
  CORE::LINALG::Matrix<4, 1> Qnew;

  /* since parallel distribution is node-wise, the three entries belonging to
   * a rotation vector should be stored on the same processor: safety-check */
  if (x_rotvec.Map().NumMyElements() % 3 != 0 or dir_rotvec.Map().NumMyElements() % 3 != 0)
    FOUR_C_THROW(
        "fatal error: apparently, the three DOFs of a nodal rotation vector are"
        " not stored on this processor. Can't apply multiplicative update!");

  // rotation vectors always consist of three consecutive DoFs
  for (int i = 0; i < x_rotvec.Map().NumMyElements(); i = i + 3)
  {
    // create a CORE::LINALG::Matrix from reference to three x vector entries
    CORE::LINALG::Matrix<3, 1> theta(&x_rotvec[i], true);
    CORE::LARGEROTATIONS::angletoquaternion(theta, Qold);

    // same for relative rotation angle deltatheta
    CORE::LINALG::Matrix<3, 1> deltatheta(&dir_rotvec[i], true);
    deltatheta.Scale(step);

    CORE::LARGEROTATIONS::angletoquaternion(deltatheta, deltaQ);
    CORE::LARGEROTATIONS::quaternionproduct(Qold, deltaQ, Qnew);
    CORE::LARGEROTATIONS::quaterniontoangle(Qnew, theta);
  }

  // first update entire x vector in an additive manner
  xnew->Update(1.0, xold, step, dir, 0.0);

  // now replace the rotvec entries by the correct value computed before
  CORE::LINALG::AssembleMyVector(0.0, *xnew, 1.0, x_rotvec);
  curr_grp_mutable.setX(xnew);

  /* tell the NOX::NLN::Group that the x vector has already been updated in
   * this preComputeX operator call */
  curr_grp_mutable.setSkipUpdateX(true);
}

FOUR_C_NAMESPACE_CLOSE
