/*----------------------------------------------------------------------*/
/*! \file

\brief Class controlling constraints and containing the necessary data, code originally by Thomas
Kloeppel


\level 2


*----------------------------------------------------------------------*/


#include "4C_constraint_manager.hpp"

#include "4C_constraint.hpp"
#include "4C_constraint_dofset.hpp"
#include "4C_constraint_monitor.hpp"
#include "4C_constraint_multipointconstraint2.hpp"
#include "4C_constraint_multipointconstraint3.hpp"
#include "4C_constraint_multipointconstraint3penalty.hpp"
#include "4C_constraint_penalty.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONSTRAINTS::ConstrManager::ConstrManager()
    : offset_id_(-1),
      max_constr_id_(0),
      num_constr_id_(-1),
      num_monitor_id_(-1),
      min_monitor_id_(-1),
      haveconstraint_(false),
      havelagrconstr_(false),
      havepenaconstr_(false),
      havemonitor_(false),
      uzawaparam_(0.0),
      issetup_(false),
      isinit_(false)

{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::init(
    Teuchos::RCP<Core::FE::Discretization> discr, const Teuchos::ParameterList& params)
{
  set_is_setup(false);

  // set pointer to discretization
  actdisc_ = discr;

  //----------------------------------------------------------------------------
  //---------------------------------------------------------Constraint Conditions!

  // constructors of constraints increment number of constraints defined and the minimum
  // ConditionID read so far.
  num_constr_id_ = 0;
  offset_id_ = 10000;
  // Check, what kind of constraining boundary conditions there are
  volconstr3d_ =
      Teuchos::rcp(new Constraint(actdisc_, "VolumeConstraint_3D", offset_id_, max_constr_id_));
  areaconstr3d_ =
      Teuchos::rcp(new Constraint(actdisc_, "AreaConstraint_3D", offset_id_, max_constr_id_));
  areaconstr2d_ =
      Teuchos::rcp(new Constraint(actdisc_, "AreaConstraint_2D", offset_id_, max_constr_id_));
  mpconline2d_ =
      Teuchos::rcp(new MPConstraint2(actdisc_, "MPC_NodeOnLine_2D", offset_id_, max_constr_id_));
  mpconplane3d_ =
      Teuchos::rcp(new MPConstraint3(actdisc_, "MPC_NodeOnPlane_3D", offset_id_, max_constr_id_));
  mpcnormcomp3d_ = Teuchos::rcp(
      new MPConstraint3(actdisc_, "MPC_NormalComponent_3D", offset_id_, max_constr_id_));

  volconstr3dpen_ = Teuchos::rcp(new ConstraintPenalty(actdisc_, "VolumeConstraint_3D_Pen"));
  areaconstr3dpen_ = Teuchos::rcp(new ConstraintPenalty(actdisc_, "AreaConstraint_3D_Pen"));
  mpcnormcomp3dpen_ =
      Teuchos::rcp(new MPConstraint3Penalty(actdisc_, "MPC_NormalComponent_3D_Pen"));

  havepenaconstr_ = (mpcnormcomp3dpen_->have_constraint()) or
                    (volconstr3dpen_->have_constraint()) or (areaconstr3dpen_->have_constraint());

  //----------------------------------------------------
  //-----------include possible further constraints here
  //----------------------------------------------------
  havelagrconstr_ = (areaconstr3d_->have_constraint()) or (volconstr3d_->have_constraint()) or
                    (areaconstr2d_->have_constraint()) or (mpconplane3d_->have_constraint()) or
                    (mpcnormcomp3d_->have_constraint()) or (mpconline2d_->have_constraint());
  haveconstraint_ = havepenaconstr_ or havelagrconstr_;


  set_is_init(true);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::setup(
    Teuchos::RCP<const Epetra_Vector> disp, Teuchos::ParameterList params)
{
  check_is_init();

  if (haveconstraint_)
  {
    num_constr_id_ = std::max(max_constr_id_ - offset_id_ + 1, 0);
    constrdofset_ = Teuchos::rcp(new ConstraintDofSet());
    constrdofset_->assign_degrees_of_freedom(actdisc_, num_constr_id_, 0);
    offset_id_ -= constrdofset_->first_gid();
    Teuchos::ParameterList p;
    uzawaparam_ = params.get<double>("uzawa parameter", 1);
    double time = params.get<double>("total time", 0.0);
    const Epetra_Map* dofrowmap = actdisc_->dof_row_map();
    // initialize constrMatrix
    constr_matrix_ =
        Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dofrowmap, num_constr_id_, false, true));
    // build Epetra_Map used as domainmap for constrMatrix and rowmap for result vectors
    constrmap_ = Teuchos::rcp(new Epetra_Map(*(constrdofset_->dof_row_map())));
    // build an all reduced version of the constraintmap, since sometimes all processors
    // have to know all values of the constraints and Lagrange multipliers
    redconstrmap_ = Core::LinAlg::allreduce_e_map(*constrmap_);
    // importer
    conimpo_ = Teuchos::rcp(new Epetra_Export(*redconstrmap_, *constrmap_));
    // sum up initial values
    refbasevalues_ = Teuchos::rcp(new Epetra_Vector(*constrmap_));
    Teuchos::RCP<Epetra_Vector> refbaseredundant = Teuchos::rcp(new Epetra_Vector(*redconstrmap_));
    // Compute initial values and assemble them to the completely redundant vector
    // We will always use the third systemvector for this purpose
    p.set("OffsetID", offset_id_);
    p.set("total time", time);
    actdisc_->set_state("displacement", disp);
    volconstr3d_->initialize(p, refbaseredundant);
    areaconstr3d_->initialize(p, refbaseredundant);
    areaconstr2d_->initialize(p, refbaseredundant);
    volconstr3dpen_->initialize(p);
    areaconstr3dpen_->initialize(p);

    mpconline2d_->set_constr_state("displacement", disp);
    mpconline2d_->initialize(p, refbaseredundant);
    mpconplane3d_->set_constr_state("displacement", disp);
    mpconplane3d_->initialize(p, refbaseredundant);
    mpcnormcomp3d_->set_constr_state("displacement", disp);
    mpcnormcomp3d_->initialize(p, refbaseredundant);
    mpcnormcomp3dpen_->set_constr_state("displacement", disp);
    mpcnormcomp3dpen_->initialize(p);

    // Export redundant vector into distributed one
    refbasevalues_->Export(*refbaseredundant, *conimpo_, Add);

    // Initialize Lagrange Multipliers, reference values and errors
    actdisc_->clear_state();
    referencevalues_ = Teuchos::rcp(new Epetra_Vector(*constrmap_));
    actvalues_ = Teuchos::rcp(new Epetra_Vector(*constrmap_, true));
    constrainterr_ = Teuchos::rcp(new Epetra_Vector(*constrmap_));
    lagr_mult_vec_ = Teuchos::rcp(new Epetra_Vector(*constrmap_, true));
    lagr_mult_vec_old_ = Teuchos::rcp(new Epetra_Vector(*constrmap_, true));
    fact_ = Teuchos::rcp(new Epetra_Vector(*constrmap_));
  }
  //----------------------------------------------------------------------------
  //---------------------------------------------------------Monitor Conditions!
  actdisc_->set_state("displacement", disp);
  min_monitor_id_ = 10000;
  int maxMonitorID = 0;
  volmonitor3d_ =
      Teuchos::rcp(new Monitor(actdisc_, "VolumeMonitor_3D", min_monitor_id_, maxMonitorID));
  areamonitor3d_ =
      Teuchos::rcp(new Monitor(actdisc_, "AreaMonitor_3D", min_monitor_id_, maxMonitorID));
  areamonitor2d_ =
      Teuchos::rcp(new Monitor(actdisc_, "AreaMonitor_2D", min_monitor_id_, maxMonitorID));
  //----------------------------------------------------
  //--------------include possible further monitors here
  //----------------------------------------------------
  num_monitor_id_ = std::max(maxMonitorID - min_monitor_id_ + 1, 0);
  havemonitor_ = (areamonitor3d_->have_monitor()) || (volmonitor3d_->have_monitor()) ||
                 (areamonitor2d_->have_monitor());
  if (havemonitor_)
  {
    Teuchos::ParameterList p1;
    // monitor values are only stored on processor zero since they are only needed for output
    int nummyele = 0;
    if (!actdisc_->get_comm().MyPID())
    {
      nummyele = num_monitor_id_;
    }
    // initialize maps and importer
    monitormap_ = Teuchos::rcp(new Epetra_Map(num_monitor_id_, nummyele, 0, actdisc_->get_comm()));
    redmonmap_ = Core::LinAlg::allreduce_e_map(*monitormap_);
    monimpo_ = Teuchos::rcp(new Epetra_Export(*redmonmap_, *monitormap_));
    monitorvalues_ = Teuchos::rcp(new Epetra_Vector(*monitormap_));
    initialmonvalues_ = Teuchos::rcp(new Epetra_Vector(*monitormap_));

    Teuchos::RCP<Epetra_Vector> initialmonredundant = Teuchos::rcp(new Epetra_Vector(*redmonmap_));
    p1.set("OffsetID", min_monitor_id_);
    volmonitor3d_->evaluate(p1, initialmonredundant);
    areamonitor3d_->evaluate(p1, initialmonredundant);
    areamonitor2d_->evaluate(p1, initialmonredundant);

    // Export redundant vector into distributed one
    initialmonvalues_->Export(*initialmonredundant, *monimpo_, Add);
    monitortypes_ = Teuchos::rcp(new Epetra_Vector(*redmonmap_));
    build_moni_type();
  }

  set_is_setup(true);
}


/*----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::evaluate_force_stiff(const double time,
    Teuchos::RCP<const Epetra_Vector> displast, Teuchos::RCP<const Epetra_Vector> disp,
    Teuchos::RCP<Epetra_Vector> fint, Teuchos::RCP<Core::LinAlg::SparseOperator> stiff,
    Teuchos::ParameterList scalelist)
{
  check_is_init();
  check_is_setup();

  double scStiff = scalelist.get("scaleStiffEntries", 1.0);
  double scConMat = scalelist.get("scaleConstrMat", 1.0);

  // create the parameters for the discretization
  Teuchos::ParameterList p;
  std::vector<Core::Conditions::Condition*> constrcond(0);
  const Epetra_Map* dofrowmap = actdisc_->dof_row_map();
  constr_matrix_->reset();  //=Teuchos::rcp(new
                            // Core::LinAlg::SparseMatrix(*dofrowmap,numConstrID_,false,true));

  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("OffsetID", offset_id_);
  p.set("NumberofID", num_constr_id_);
  p.set("old disp", displast);
  p.set("new disp", disp);
  p.set("scaleStiffEntries", scStiff);
  p.set("scaleConstrMat", scConMat);
  p.set("vector curve factors", fact_);
  // Convert Epetra_Vector containing lagrange multipliers to an completely
  // redundant Epetra_vector since every element with the constraint condition needs them
  Teuchos::RCP<Epetra_Vector> lagrMultVecDense = Teuchos::rcp(new Epetra_Vector(*redconstrmap_));
  Core::LinAlg::export_to(*lagr_mult_vec_, *lagrMultVecDense);
  p.set("LagrMultVector", lagrMultVecDense);
  // Construct a redundant time curve factor and put it into parameter list
  Teuchos::RCP<Epetra_Vector> factredundant = Teuchos::rcp(new Epetra_Vector(*redconstrmap_));
  p.set("vector curve factors", factredundant);

  Teuchos::RCP<Epetra_Vector> actredundant = Teuchos::rcp(new Epetra_Vector(*redconstrmap_));
  Teuchos::RCP<Epetra_Vector> refbaseredundant = Teuchos::rcp(new Epetra_Vector(*redconstrmap_));

  actdisc_->clear_state();
  actdisc_->set_state("displacement", disp);
  volconstr3d_->evaluate(p, stiff, constr_matrix_, fint, refbaseredundant, actredundant);
  areaconstr3d_->evaluate(p, stiff, constr_matrix_, fint, refbaseredundant, actredundant);
  areaconstr2d_->evaluate(p, stiff, constr_matrix_, fint, refbaseredundant, actredundant);
  volconstr3dpen_->evaluate(p, stiff, Teuchos::null, fint, Teuchos::null, Teuchos::null);
  areaconstr3dpen_->evaluate(p, stiff, Teuchos::null, fint, Teuchos::null, Teuchos::null);

  mpconplane3d_->set_constr_state("displacement", disp);
  mpconplane3d_->evaluate(p, stiff, constr_matrix_, fint, refbaseredundant, actredundant);
  mpcnormcomp3d_->set_constr_state("displacement", disp);
  mpcnormcomp3d_->evaluate(p, stiff, constr_matrix_, fint, refbaseredundant, actredundant);
  mpcnormcomp3dpen_->set_constr_state("displacement", disp);
  mpcnormcomp3dpen_->evaluate(p, stiff, Teuchos::null, fint, Teuchos::null, Teuchos::null);
  mpconline2d_->set_constr_state("displacement", disp);
  mpconline2d_->evaluate(p, stiff, constr_matrix_, fint, refbaseredundant, actredundant);
  // Export redundant vectors into distributed ones
  actvalues_->PutScalar(0.0);
  actvalues_->Export(*actredundant, *conimpo_, Add);
  Teuchos::RCP<Epetra_Vector> addrefbase = Teuchos::rcp(new Epetra_Vector(*constrmap_));
  addrefbase->Export(*refbaseredundant, *conimpo_, Add);
  refbasevalues_->Update(1.0, *addrefbase, 1.0);
  fact_->PutScalar(0.0);
  fact_->Export(*factredundant, *conimpo_, AbsMax);
  // ----------------------------------------------------
  // -----------include possible further constraints here
  // ----------------------------------------------------
  // Compute current reference volumes as elemetwise product of timecurvefactor and initialvalues
  referencevalues_->Multiply(1.0, *fact_, *refbasevalues_, 0.0);
  constrainterr_->Update(scConMat, *referencevalues_, -1.0 * scConMat, *actvalues_, 0.0);
  actdisc_->clear_state();
  // finalize the constraint matrix
  std::string label(constr_matrix_->Label());
  if (label == "Core::LinAlg::BlockSparseMatrixBase")
    constr_matrix_->complete();
  else
    constr_matrix_->complete(*constrmap_, *dofrowmap);
}

/*----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::compute_error(double time, Teuchos::RCP<Epetra_Vector> disp)
{
  check_is_init();
  check_is_setup();

  std::vector<Core::Conditions::Condition*> constrcond(0);
  Teuchos::ParameterList p;
  p.set("total time", time);
  actdisc_->set_state("displacement", disp);

  Teuchos::RCP<Epetra_Vector> actredundant = Teuchos::rcp(new Epetra_Vector(*redconstrmap_));
  Core::LinAlg::export_to(*actvalues_, *actredundant);
  // Compute current values and assemble them to the completely redundant vector
  // We will always use the third systemvector for this purpose
  p.set("OffsetID", offset_id_);
  volconstr3d_->evaluate(
      p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, actredundant);
  areaconstr3d_->evaluate(
      p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, actredundant);
  areaconstr2d_->evaluate(
      p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, actredundant);

  mpconplane3d_->evaluate(
      p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, actredundant);
  mpconplane3d_->evaluate(
      p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, actredundant);

  mpcnormcomp3d_->evaluate(
      p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, actredundant);
  mpcnormcomp3d_->evaluate(
      p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, actredundant);

  // Export redundant vectors into distributed ones
  actvalues_->PutScalar(0.0);
  actvalues_->Export(*actredundant, *conimpo_, Add);

  constrainterr_->Update(1.0, *referencevalues_, -1.0, *actvalues_, 0.0);
}


/*----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::read_restart(
    Core::IO::DiscretizationReader& reader, const double& time)
{
  //  double uzawatemp = reader.ReadDouble("uzawaparameter");
  //  consolv_->SetUzawaParameter(uzawatemp);
  Teuchos::RCP<Epetra_Map> constrmap = get_constraint_map();
  Teuchos::RCP<Epetra_Vector> tempvec = Core::LinAlg::create_vector(*constrmap, true);
  reader.read_vector(tempvec, "lagrmultiplier");
  set_lagr_mult_vector(tempvec);
  reader.read_vector(tempvec, "refconval");
  set_ref_base_values(tempvec, time);
}

/*----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::set_ref_base_values(
    Teuchos::RCP<Epetra_Vector> newrefval, const double& time)
{
  volconstr3d_->initialize(time);
  areaconstr3d_->initialize(time);
  areaconstr2d_->initialize(time);
  mpconplane3d_->initialize(time);
  mpcnormcomp3d_->initialize(time);
  mpconline2d_->initialize(time);

  refbasevalues_->Update(1.0, *newrefval, 0.0);
}

/*----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::update_lagr_mult(double factor)
{
  lagr_mult_vec_->Update(factor, *constrainterr_, 1.0);
  if (volconstr3d_->have_constraint())
  {
    std::vector<int> volconID = volconstr3d_->get_active_cond_id();
    for (unsigned int i = 0; i < volconID.size(); i++)
    {
      if (constrmap_->LID(int(i - offset_id_)) != -1)
      {
        std::cout << "Multiplier for Volume Constraint: " << volconID.at(i) << ":  "
                  << (*lagr_mult_vec_)[constrmap_->LID(int(i - offset_id_))] << '\n';
      }
    }
  }
}

void CONSTRAINTS::ConstrManager::update() { lagr_mult_vec_old_->Update(1.0, *lagr_mult_vec_, 0.0); }

/*----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::update_lagr_mult(Teuchos::RCP<Epetra_Vector> vect)
{
  lagr_mult_vec_->Update(1.0, *vect, 1.0);
}

void CONSTRAINTS::ConstrManager::update_tot_lagr_mult(Teuchos::RCP<Epetra_Vector> vect)
{
  lagr_mult_vec_->Update(1.0, *vect, 1.0, *lagr_mult_vec_old_, 0.0);
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::compute_monitor_values(Teuchos::RCP<Epetra_Vector> disp)
{
  std::vector<Core::Conditions::Condition*> monitcond(0);
  monitorvalues_->PutScalar(0.0);
  Teuchos::ParameterList p;
  actdisc_->set_state("displacement", disp);

  Teuchos::RCP<Epetra_Vector> actmonredundant = Teuchos::rcp(new Epetra_Vector(*redmonmap_));
  p.set("OffsetID", min_monitor_id_);

  volmonitor3d_->evaluate(p, actmonredundant);
  areamonitor3d_->evaluate(p, actmonredundant);
  areamonitor2d_->evaluate(p, actmonredundant);

  Epetra_Import monimpo(*monitormap_, *redmonmap_);
  monitorvalues_->Export(*actmonredundant, *monimpo_, Add);
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::compute_monitor_values(Teuchos::RCP<const Epetra_Vector> disp)
{
  std::vector<Core::Conditions::Condition*> monitcond(0);
  monitorvalues_->PutScalar(0.0);
  Teuchos::ParameterList p;
  if (not actdisc_->dof_row_map()->SameAs(disp->Map()))
  {
    // build merged dof row map
    Teuchos::RCP<Epetra_Map> largemap =
        Core::LinAlg::merge_map(*actdisc_->dof_row_map(), *constrmap_, false);

    Core::LinAlg::MapExtractor conmerger;
    conmerger.setup(*largemap, Teuchos::rcp(actdisc_->dof_row_map(), false), constrmap_);
    actdisc_->set_state("displacement", conmerger.extract_cond_vector(disp));
  }
  else
    actdisc_->set_state("displacement", disp);

  Teuchos::RCP<Epetra_Vector> actmonredundant = Teuchos::rcp(new Epetra_Vector(*redmonmap_));
  p.set("OffsetID", min_monitor_id_);

  volmonitor3d_->evaluate(p, actmonredundant);
  areamonitor3d_->evaluate(p, actmonredundant);
  areamonitor2d_->evaluate(p, actmonredundant);

  Epetra_Import monimpo(*monitormap_, *redmonmap_);
  monitorvalues_->Export(*actmonredundant, *monimpo_, Add);
}

/*----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::print_monitor_values() const
{
  if (num_monitor_id_ == 1)
    printf("Monitor value:\n");
  else if (num_monitor_id_ > 1)
    printf("Monitor values:\n");

  for (int i = 0; i < num_monitor_id_; ++i)
  {
    if ((*monitortypes_)[i] == 1.0)
    {
      printf("%2d (volume): %10.5e (%5.2f%% of initial value)\n", i + min_monitor_id_,
          abs((*monitorvalues_)[i]), ((*monitorvalues_)[i]) * 100 / ((*initialmonvalues_)[i]));
    }
    else if ((*monitortypes_)[i] == 2.0)
    {
      printf("%2d   (area): %10.5e (%5.2f%% of initial value)\n", i + min_monitor_id_,
          abs((*monitorvalues_)[i]), ((*monitorvalues_)[i]) * 100 / ((*initialmonvalues_)[i]));
    }
  }
}

void CONSTRAINTS::ConstrManager::build_moni_type()
{
  Teuchos::ParameterList p1;
  // build distributed and redundant dummy monitor vector
  Teuchos::RCP<Epetra_Vector> dummymonredundant = Teuchos::rcp(new Epetra_Vector(*redmonmap_));
  Teuchos::RCP<Epetra_Vector> dummymondist = Teuchos::rcp(new Epetra_Vector(*monitormap_));
  p1.set("OffsetID", min_monitor_id_);

  // do the volumes
  volmonitor3d_->evaluate(p1, dummymonredundant);
  // Export redundant vector into distributed one
  dummymondist->Export(*dummymonredundant, *monimpo_, Add);
  // Now export back
  Core::LinAlg::export_to(*dummymondist, *dummymonredundant);
  for (int i = 0; i < dummymonredundant->MyLength(); i++)
  {
    if ((*dummymonredundant)[i] != 0.0) (*monitortypes_)[i] = 1.0;
  }

  // do the area in 3D
  dummymonredundant->PutScalar(0.0);
  dummymondist->PutScalar(0.0);
  areamonitor3d_->evaluate(p1, dummymonredundant);
  // Export redundant vector into distributed one
  dummymondist->Export(*dummymonredundant, *monimpo_, Add);
  // Now export back
  Core::LinAlg::export_to(*dummymondist, *dummymonredundant);
  for (int i = 0; i < dummymonredundant->MyLength(); i++)
  {
    if ((*dummymonredundant)[i] != 0.0) (*monitortypes_)[i] = 2.0;
  }

  // do the area in 2D
  dummymonredundant->PutScalar(0.0);
  dummymondist->PutScalar(0.0);
  areamonitor2d_->evaluate(p1, dummymonredundant);
  // Export redundant vector into distributed one
  dummymondist->Export(*dummymonredundant, *monimpo_, Add);
  // Now export back
  Core::LinAlg::export_to(*dummymondist, *dummymonredundant);
  for (int i = 0; i < dummymonredundant->MyLength(); i++)
  {
    if ((*dummymonredundant)[i] != 0.0) (*monitortypes_)[i] = 3.0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONSTRAINTS::ConstrManager::use_block_matrix(
    Teuchos::RCP<const Core::LinAlg::MultiMapExtractor> domainmaps,
    Teuchos::RCP<const Core::LinAlg::MultiMapExtractor> rangemaps)
{
  // (re)allocate system matrix
  constr_matrix_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *domainmaps, *rangemaps, 81, false, true));
}

FOUR_C_NAMESPACE_CLOSE
