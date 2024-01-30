/*---------------------------------------------------------------------*/
/*! \file
\brief Penalty contact solving strategy: The contact constrains are enforced
       by a penalty formulation.

\level 2


*/
/*---------------------------------------------------------------------*/

#include "baci_contact_penalty_strategy.H"

#include "baci_contact_constitutivelaw_cubic_contactconstitutivelaw.H"
#include "baci_contact_defines.H"
#include "baci_contact_element.H"
#include "baci_contact_interface.H"
#include "baci_contact_node.H"
#include "baci_contact_paramsinterface.H"
#include "baci_global_data.H"
#include "baci_inpar_contact.H"
#include "baci_lib_utils.H"
#include "baci_linalg_multiply.H"
#include "baci_linalg_sparsematrix.H"
#include "baci_linalg_utils_sparse_algebra_create.H"
#include "baci_linalg_utils_sparse_algebra_manipulation.H"
#include "baci_mortar_defines.H"
#include "baci_mortar_utils.H"

#include <Epetra_CrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Epetra_Operator.h>

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                              popp 05/09|
 *----------------------------------------------------------------------*/
CONTACT::PenaltyStrategy::PenaltyStrategy(const Epetra_Map* DofRowMap, const Epetra_Map* NodeRowMap,
    Teuchos::ParameterList params, std::vector<Teuchos::RCP<CONTACT::Interface>> interface,
    const int spatialDim, const Teuchos::RCP<const Epetra_Comm>& comm, const double alphaf,
    const int maxdof)
    : AbstractStrategy(Teuchos::rcp(new CONTACT::AbstractStratDataContainer()), DofRowMap,
          NodeRowMap, params, spatialDim, comm, alphaf, maxdof),
      interface_(interface),
      constrnorm_(0.0),
      constrnormtan_(0.0),
      initialpenalty_(Params().get<double>("PENALTYPARAM")),
      initialpenaltytan_(Params().get<double>("PENALTYPARAMTAN"))
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::PenaltyStrategy::PenaltyStrategy(
    const Teuchos::RCP<CONTACT::AbstractStratDataContainer>& data_ptr, const Epetra_Map* DofRowMap,
    const Epetra_Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<Teuchos::RCP<CONTACT::Interface>> interface, const int spatialDim,
    const Teuchos::RCP<const Epetra_Comm>& comm, const double alphaf, const int maxdof)
    : AbstractStrategy(data_ptr, DofRowMap, NodeRowMap, params, spatialDim, comm, alphaf, maxdof),
      interface_(interface),
      constrnorm_(0.0),
      constrnormtan_(0.0),
      initialpenalty_(Params().get<double>("PENALTYPARAM")),
      initialpenaltytan_(Params().get<double>("PENALTYPARAMTAN"))
{
  // empty constructor
}


/*----------------------------------------------------------------------*
 |  save the gap-scaling kappa from reference config          popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::SaveReferenceState(Teuchos::RCP<const Epetra_Vector> dis)
{
  // initialize the displacement field
  SetState(MORTAR::state_new_displacement, *dis);

  // kappa will be the shape function integral on the slave sides
  // (1) build the nodal information
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // interface needs to be complete
    if (!interface_[i]->Filled() && Comm().MyPID() == 0)
      dserror("FillComplete() not called on interface %", i);

    // do the computation of nodal shape function integral
    // (for convenience, the results will be stored in nodal gap)

    // loop over proc's slave elements of the interface for integration
    // use standard column map to include processor's ghosted elements
    for (int j = 0; j < interface_[i]->SlaveColElements()->NumMyElements(); ++j)
    {
      int gid1 = interface_[i]->SlaveColElements()->GID(j);
      DRT::Element* ele1 = interface_[i]->Discret().gElement(gid1);
      if (!ele1) dserror("Cannot find slave element with gid %", gid1);
      Element* selement = dynamic_cast<Element*>(ele1);

      interface_[i]->IntegrateKappaPenalty(*selement);
    }

    // loop over all slave row nodes on the current interface
    for (int j = 0; j < interface_[i]->SlaveRowNodes()->NumMyElements(); ++j)
    {
      int gid = interface_[i]->SlaveRowNodes()->GID(j);
      DRT::Node* node = interface_[i]->Discret().gNode(gid);
      if (!node) dserror("Cannot find node with gid %", gid);
      Node* cnode = dynamic_cast<Node*>(node);

      // get nodal weighted gap
      // (this is where we stored the shape function integrals)
      double gap = cnode->Data().Getg();

      // store kappa as the inverse of gap
      // (this removes the scaling introduced by weighting the gap!!!)
      cnode->Data().Kappa() = 1.0 / gap;

      // std::cout << "S-NODE #" << gid << " kappa=" << cnode->Data().Kappa() << std::endl;
    }
  }
}

/*----------------------------------------------------------------------*
 | evaluate relative movement in predictor step               popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::EvaluateRelMovPredict()
{
  // only for frictional contact
  if (friction_ == false) return;

  // call evaluation method of base class
  EvaluateRelMov();

  return;
}

/*----------------------------------------------------------------------*
 | initialize global contact variables for next Newton step   popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::Initialize()
{
  // (re)setup global matrices containing fc derivatives
  // must use FE_MATRIX type here, as we will do non-local assembly!
  lindmatrix_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(
      *gsdofrowmap_, 100, true, false, CORE::LINALG::SparseMatrix::FE_MATRIX));
  linmmatrix_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(
      *gmdofrowmap_, 100, true, false, CORE::LINALG::SparseMatrix::FE_MATRIX));

  // (re)setup global vector containing lagrange multipliers
  z_ = CORE::LINALG::CreateVector(*gsdofrowmap_, true);

  // (re)setup global matrix containing lagrange multiplier derivatives
  linzmatrix_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*gsdofrowmap_, 100));

  return;
}

/*----------------------------------------------------------------------*
 | evaluate contact and create linear system                  popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::EvaluateContact(
    Teuchos::RCP<CORE::LINALG::SparseOperator>& kteff, Teuchos::RCP<Epetra_Vector>& feff)
{
  // in the beginning of this function, the regularized contact forces
  // in normal and tangential direction are evaluated from geometric
  // measures (gap and relative tangential velocity). Here, also active and
  // slip nodes are detected. Then, the insertion of the according stiffness
  // blocks takes place.

  bool isincontact = false;
  bool activesetchange = false;

  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    bool localisincontact = false;
    bool localactivesetchange = false;

    // evaluate lagrange multipliers (regularized forces) in normal direction
    // and nodal derivz matrix values, store them in nodes
    interface_[i]->AssembleRegNormalForces(localisincontact, localactivesetchange);

    // evaluate lagrange multipliers (regularized forces) in tangential direction
    INPAR::CONTACT::SolvingStrategy soltype =
        INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(Params(), "STRATEGY");

    if (friction_ and (soltype == INPAR::CONTACT::solution_penalty or
                          soltype == INPAR::CONTACT::solution_multiscale))
      interface_[i]->AssembleRegTangentForcesPenalty();

    if (friction_ and soltype == INPAR::CONTACT::solution_uzawa)
      interface_[i]->AssembleRegTangentForcesUzawa();

    isincontact = isincontact || localisincontact;
    activesetchange = activesetchange || localactivesetchange;
  }

  // broadcast contact status & active set change
  int globalcontact, globalchange = 0;
  int localcontact = isincontact;
  int localchange = activesetchange;

  Comm().SumAll(&localcontact, &globalcontact, 1);
  Comm().SumAll(&localchange, &globalchange, 1);

  if (globalcontact >= 1)
  {
    isincontact_ = true;
    wasincontact_ = true;
  }
  else
    isincontact_ = false;

  if ((Comm().MyPID() == 0) && (globalchange >= 1))
    std::cout << "ACTIVE CONTACT SET HAS CHANGED..." << std::endl;

  // (re)setup active global Epetra_Maps
  // the map of global active nodes is needed for the penalty case, too.
  // this is due to the fact that we want to monitor the constraint norm
  // of the active nodes
  gactivenodes_ = Teuchos::null;
  gslipnodes_ = Teuchos::null;
  gactivedofs_ = Teuchos::null;

  // update active sets of all interfaces
  // (these maps are NOT allowed to be overlapping !!!)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->BuildActiveSet();
    gactivenodes_ = CORE::LINALG::MergeMap(gactivenodes_, interface_[i]->ActiveNodes(), false);
    gactivedofs_ = CORE::LINALG::MergeMap(gactivedofs_, interface_[i]->ActiveDofs(), false);
    gslipnodes_ = CORE::LINALG::MergeMap(gslipnodes_, interface_[i]->SlipNodes(), false);
  }

  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!IsInContact() && !WasInContact() && !WasInContactLastTimeStep()) return;

  // since we will modify the graph of kteff by adding additional
  // meshtyong stiffness entries, we have to uncomplete it
  kteff->UnComplete();

  // assemble contact quantities on all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // assemble global lagrangian multiplier vector
    interface_[i]->AssembleLM(*z_);
    // assemble global derivatives of lagrangian multipliers
    interface_[i]->AssembleLinZ(*linzmatrix_);
    // assemble global derivatives of mortar D and M matrices
    interface_[i]->AssembleLinDM(*lindmatrix_, *linmmatrix_);
  }

  // FillComplete() global matrices LinD, LinM, LinZ
  lindmatrix_->Complete(*gsmdofrowmap_, *gsdofrowmap_);
  linmmatrix_->Complete(*gsmdofrowmap_, *gmdofrowmap_);
  linzmatrix_->Complete(*gsmdofrowmap_, *gsdofrowmap_);

  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // LinD      ---->   T^(-T) * LinD
  // D         ---->   D * T^(-1)
  //----------------------------------------------------------------------
  if (Dualquadslavetrafo())
  {
    // modify lindmatrix_ and dmatrix_
    Teuchos::RCP<CORE::LINALG::SparseMatrix> temp1 =
        CORE::LINALG::MLMultiply(*invtrafo_, true, *lindmatrix_, false, false, false, true);
    Teuchos::RCP<CORE::LINALG::SparseMatrix> temp2 =
        CORE::LINALG::MLMultiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
    lindmatrix_ = temp1;
    dmatrix_ = temp2;
  }

#ifdef CONTACTFDPENALTYTRAC
  INPAR::CONTACT::FrictionType ftype =
      INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(Params(), "FRICTION");

  // check derivatives of penalty traction
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    if (IsInContact())
    {
      if (ftype == INPAR::CONTACT::friction_coulomb)
      {
        std::cout << "LINZMATRIX" << *linzmatrix_ << std::endl;
        interface_[i]->FDCheckPenaltyTracFric();
      }
      else if (ftype == INPAR::CONTACT::friction_none)
      {
        std::cout << "-- CONTACTFDDERIVZ --------------------" << std::endl;
        interface_[i]->FDCheckPenaltyTracNor();
        std::cout << "-- CONTACTFDDERIVZ --------------------" << std::endl;
      }
      else
        dserror("Error: FD Check for this friction type not implemented!");
    }
  }
#endif

  // **********************************************************************
  // Build Contact Stiffness #1
  // **********************************************************************
  // involving contributions of derivatives of D and M:
  //  Kc,1 = delta[ 0 -M(transpose) D] * LM

  // transform if necessary
  if (ParRedist())
  {
    lindmatrix_ = MORTAR::MatrixRowTransform(lindmatrix_, pgsdofrowmap_);
    linmmatrix_ = MORTAR::MatrixRowTransform(linmmatrix_, pgmdofrowmap_);
  }

  // add to kteff
  kteff->Add(*lindmatrix_, false, 1.0 - alphaf_, 1.0);
  kteff->Add(*linmmatrix_, false, 1.0 - alphaf_, 1.0);

  // **********************************************************************
  // Build Contact Stiffness #2
  // **********************************************************************
  // involving contributions of derivatives of lagrange multipliers:
  //  Kc,2= [ 0 -M(transpose) D] * deltaLM

  // multiply Mortar matrices D and M with LinZ
  Teuchos::RCP<CORE::LINALG::SparseMatrix> dtilde =
      CORE::LINALG::MLMultiply(*dmatrix_, true, *linzmatrix_, false, false, false, true);
  Teuchos::RCP<CORE::LINALG::SparseMatrix> mtilde =
      CORE::LINALG::MLMultiply(*mmatrix_, true, *linzmatrix_, false, false, false, true);

  // transform if necessary
  if (ParRedist())
  {
    dtilde = MORTAR::MatrixRowTransform(dtilde, pgsdofrowmap_);
    mtilde = MORTAR::MatrixRowTransform(mtilde, pgmdofrowmap_);
  }

  // add to kteff
  kteff->Add(*dtilde, false, 1.0 - alphaf_, 1.0);
  kteff->Add(*mtilde, false, -(1.0 - alphaf_), 1.0);

  // **********************************************************************
  // Build RHS
  // **********************************************************************
  // feff += -alphaf * fc,n - (1-alphaf) * fc,n+1,k

  {
    // we initialize fcmdold with dold-rowmap instead of gsdofrowmap
    // (this way, possible self contact is automatically included)

    Teuchos::RCP<Epetra_Vector> fcmdold = Teuchos::rcp(new Epetra_Vector(dold_->RowMap()));
    dold_->Multiply(true, *zold_, *fcmdold);
    Teuchos::RCP<Epetra_Vector> fcmdoldtemp = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
    CORE::LINALG::Export(*fcmdold, *fcmdoldtemp);
    feff->Update(-alphaf_, *fcmdoldtemp, 1.0);
  }

  {
    // we initialize fcmmold with mold-domainmap instead of gmdofrowmap
    // (this way, possible self contact is automatically included)

    Teuchos::RCP<Epetra_Vector> fcmmold = Teuchos::rcp(new Epetra_Vector(mold_->DomainMap()));
    mold_->Multiply(true, *zold_, *fcmmold);
    Teuchos::RCP<Epetra_Vector> fcmmoldtemp = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
    CORE::LINALG::Export(*fcmmold, *fcmmoldtemp);
    feff->Update(alphaf_, *fcmmoldtemp, 1.0);
  }

  {
    Teuchos::RCP<Epetra_Vector> fcmd = Teuchos::rcp(new Epetra_Vector(*gsdofrowmap_));
    dmatrix_->Multiply(true, *z_, *fcmd);
    Teuchos::RCP<Epetra_Vector> fcmdtemp = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
    CORE::LINALG::Export(*fcmd, *fcmdtemp);
    feff->Update(-(1 - alphaf_), *fcmdtemp, 1.0);
  }

  {
    Teuchos::RCP<Epetra_Vector> fcmm = CORE::LINALG::CreateVector(*gmdofrowmap_, true);
    mmatrix_->Multiply(true, *z_, *fcmm);
    Teuchos::RCP<Epetra_Vector> fcmmtemp = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
    CORE::LINALG::Export(*fcmm, *fcmmtemp);
    feff->Update(1 - alphaf_, *fcmmtemp, 1.0);
  }

#ifdef CONTACTFDGAP
  // FD check of weighted gap g derivatives (non-penetr. condition)

  std::cout << "-- CONTACTFDGAP -----------------------------" << std::endl;
  interface_[0]->FDCheckGapDeriv();
  std::cout << "-- CONTACTFDGAP -----------------------------" << std::endl;

#endif  // #ifdef CONTACTFDGAP

  return;
}

/*----------------------------------------------------------------------*
 | evaluate frictional contact and create linear system gitterle   10/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::EvaluateFriction(
    Teuchos::RCP<CORE::LINALG::SparseOperator>& kteff, Teuchos::RCP<Epetra_Vector>& feff)
{
  // this is almost the same as in the frictionless contact
  // whereas we chose the EvaluateContact routine with
  // one difference

  // check if friction should be applied
  INPAR::CONTACT::FrictionType ftype =
      INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(Params(), "FRICTION");

  // coulomb friction case
  if (ftype == INPAR::CONTACT::friction_coulomb || ftype == INPAR::CONTACT::friction_stick)
  {
    EvaluateContact(kteff, feff);
  }
  else if (ftype == INPAR::CONTACT::friction_tresca)
  {
    dserror(
        "Error in AbstractStrategy::Evaluate: Penalty Strategy for"
        " Tresca friction not yet implemented");
  }
  else
    dserror("Error in AbstractStrategy::Evaluate: Unknown friction type");

  return;
}

/*----------------------------------------------------------------------*
 | reset penalty parameter to intial value                    popp 08/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::ResetPenalty()
{
  // reset penalty parameter in strategy
  Params().set<double>("PENALTYPARAM", InitialPenalty());
  Params().set<double>("PENALTYPARAMTAN", InitialPenaltyTan());

  // reset penalty parameter in all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->InterfaceParams().set<double>("PENALTYPARAM", InitialPenalty());
    interface_[i]->InterfaceParams().set<double>("PENALTYPARAMTAN", InitialPenaltyTan());
  }

  return;
}

/*----------------------------------------------------------------------*
 | modify penalty parameter to intial value                    mhv 03/16|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::ModifyPenalty()
{
  // generate random number between 0.95 and 1.05
  double randnum = ((double)rand() / (double)RAND_MAX) * 0.1 + 0.95;
  double pennew = randnum * InitialPenalty();

  // modify penalty parameter in strategy
  Params().set<double>("PENALTYPARAM", pennew);
  Params().set<double>("PENALTYPARAMTAN", pennew);

  // modify penalty parameter in all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->InterfaceParams().set<double>("PENALTYPARAM", pennew);
    interface_[i]->InterfaceParams().set<double>("PENALTYPARAMTAN", pennew);
  }

  return;
}

/*----------------------------------------------------------------------*
 | intialize second, third,... Uzawa step                     popp 01/10|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::InitializeUzawa(
    Teuchos::RCP<CORE::LINALG::SparseOperator>& kteff, Teuchos::RCP<Epetra_Vector>& feff)
{
  // remove old stiffness terms
  // (FIXME: redundant code to EvaluateContact(), expect for minus sign)

  // since we will modify the graph of kteff by adding additional
  // meshtying stiffness entries, we have to uncomplete it
  kteff->UnComplete();

  // remove contact stiffness #1 from kteff
  kteff->Add(*lindmatrix_, false, -(1.0 - alphaf_), 1.0);
  kteff->Add(*linmmatrix_, false, -(1.0 - alphaf_), 1.0);

  // multiply Mortar matrices D and M with LinZ
  Teuchos::RCP<CORE::LINALG::SparseMatrix> dtilde =
      CORE::LINALG::MLMultiply(*dmatrix_, true, *linzmatrix_, false, false, false, true);
  Teuchos::RCP<CORE::LINALG::SparseMatrix> mtilde =
      CORE::LINALG::MLMultiply(*mmatrix_, true, *linzmatrix_, false, false, false, true);

  // transform if necessary
  if (ParRedist())
  {
    dtilde = MORTAR::MatrixRowTransform(dtilde, pgsdofrowmap_);
    mtilde = MORTAR::MatrixRowTransform(mtilde, pgmdofrowmap_);
  }

  // remove contact stiffness #2 from kteff
  kteff->Add(*dtilde, false, -(1.0 - alphaf_), 1.0);
  kteff->Add(*mtilde, false, (1.0 - alphaf_), 1.0);

  // remove old force terms
  // (FIXME: redundant code to EvaluateContact(), expect for minus sign)

  Teuchos::RCP<Epetra_Vector> fcmdold = Teuchos::rcp(new Epetra_Vector(dold_->RowMap()));
  dold_->Multiply(true, *zold_, *fcmdold);
  Teuchos::RCP<Epetra_Vector> fcmdoldtemp = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
  CORE::LINALG::Export(*fcmdold, *fcmdoldtemp);
  feff->Update(alphaf_, *fcmdoldtemp, 1.0);

  Teuchos::RCP<Epetra_Vector> fcmmold = Teuchos::rcp(new Epetra_Vector(mold_->DomainMap()));
  mold_->Multiply(true, *zold_, *fcmmold);
  Teuchos::RCP<Epetra_Vector> fcmmoldtemp = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
  CORE::LINALG::Export(*fcmmold, *fcmmoldtemp);
  feff->Update(-alphaf_, *fcmmoldtemp, 1.0);

  Teuchos::RCP<Epetra_Vector> fcmd = Teuchos::rcp(new Epetra_Vector(*gsdofrowmap_));
  dmatrix_->Multiply(true, *z_, *fcmd);
  Teuchos::RCP<Epetra_Vector> fcmdtemp = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
  CORE::LINALG::Export(*fcmd, *fcmdtemp);
  feff->Update(1 - alphaf_, *fcmdtemp, 1.0);

  Teuchos::RCP<Epetra_Vector> fcmm = CORE::LINALG::CreateVector(*gmdofrowmap_, true);
  mmatrix_->Multiply(true, *z_, *fcmm);
  Teuchos::RCP<Epetra_Vector> fcmmtemp = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
  CORE::LINALG::Export(*fcmm, *fcmmtemp);
  feff->Update(-(1 - alphaf_), *fcmmtemp, 1.0);

  // reset some matrices
  // must use FE_MATRIX type here, as we will do non-local assembly!
  lindmatrix_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(
      *gsdofrowmap_, 100, true, false, CORE::LINALG::SparseMatrix::FE_MATRIX));
  linmmatrix_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(
      *gmdofrowmap_, 100, true, false, CORE::LINALG::SparseMatrix::FE_MATRIX));

  // reset nodal derivZ values
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    for (int j = 0; j < interface_[i]->SlaveColNodesBound()->NumMyElements(); ++j)
    {
      int gid = interface_[i]->SlaveColNodesBound()->GID(i);
      DRT::Node* node = interface_[i]->Discret().gNode(gid);
      if (!node) dserror("Cannot find node with gid %", gid);
      Node* cnode = dynamic_cast<Node*>(node);

      for (int k = 0; k < (int)((cnode->Data().GetDerivZ()).size()); ++k)
        (cnode->Data().GetDerivZ())[k].clear();
      (cnode->Data().GetDerivZ()).resize(0);
    }
  }

  // now redo Initialize()
  Initialize();

  // and finally redo Evaluate()
  Teuchos::RCP<Epetra_Vector> nullvec = Teuchos::null;
  Evaluate(kteff, feff, nullvec);

  // complete stiffness matrix
  kteff->Complete();

  return;
}

/*----------------------------------------------------------------------*
 | evaluate L2-norm of active constraints                     popp 08/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::UpdateConstraintNorm(int uzawaiter)
{
  // initialize parameters
  double cnorm = 0.0;
  double cnormtan = 0.0;
  bool updatepenalty = false;
  bool updatepenaltytan = false;
  double ppcurr = Params().get<double>("PENALTYPARAM");
  double ppcurrtan = Params().get<double>("PENALTYPARAMTAN");

  // gactivenodes_ is undefined
  if (gactivenodes_ == Teuchos::null)
  {
    constrnorm_ = 0;
    constrnormtan_ = 0;
  }

  // gactivenodes_ has no elements
  else if (gactivenodes_->NumGlobalElements() == 0)
  {
    constrnorm_ = 0;
    constrnormtan_ = 0;
  }

  // gactivenodes_ has at least one element
  else
  {
    // export weighted gap vector to gactiveN-map
    Teuchos::RCP<Epetra_Vector> gact;
    if (constr_direction_ == INPAR::CONTACT::constr_xyz)
    {
      gact = CORE::LINALG::CreateVector(*gactivedofs_, true);
      CORE::LINALG::Export(*g_, *gact);
    }
    else
    {
      gact = CORE::LINALG::CreateVector(*gactivenodes_, true);
      if (gact->GlobalLength()) CORE::LINALG::Export(*g_, *gact);
    }

    // compute constraint norm
    gact->Norm2(&cnorm);

    // Evaluate norm in tangential direction for frictional contact
    if (friction_)
    {
      for (int i = 0; i < (int)interface_.size(); ++i) interface_[i]->EvaluateTangentNorm(cnormtan);

      cnormtan = sqrt(cnormtan);
    }

    //********************************************************************
    // adaptive update of penalty parameter
    // (only for Uzawa Augmented Lagrange strategy)
    //********************************************************************
    INPAR::CONTACT::SolvingStrategy soltype =
        INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(Params(), "STRATEGY");

    if (soltype == INPAR::CONTACT::solution_uzawa)
    {
      // check convergence of cnorm and update penalty parameter
      // only do this for second, third, ... Uzawa iteration
      // cf. Wriggers, Computational Contact Mechanics, 2nd edition (2006), p. 340
      if ((uzawaiter >= 2) && (cnorm > 0.25 * ConstraintNorm()))
      {
        updatepenalty = true;

        // update penalty parameter in strategy
        Params().set<double>("PENALTYPARAM", 10 * ppcurr);

        // update penalty parameter in all interfaces
        for (int i = 0; i < (int)interface_.size(); ++i)
        {
          double ippcurr = interface_[i]->InterfaceParams().get<double>("PENALTYPARAM");
          if (ippcurr != ppcurr) dserror("Something wrong with penalty parameter");
          interface_[i]->InterfaceParams().set<double>("PENALTYPARAM", 10 * ippcurr);
        }
        // in the case of frictional contact, the tangential penalty
        // parameter is also dated up when this is done for the normal one
        if (friction_)
        {
          updatepenaltytan = true;

          // update penalty parameter in strategy
          Params().set<double>("PENALTYPARAMTAN", 10 * ppcurrtan);

          // update penalty parameter in all interfaces
          for (int i = 0; i < (int)interface_.size(); ++i)
          {
            double ippcurrtan = interface_[i]->InterfaceParams().get<double>("PENALTYPARAMTAN");
            if (ippcurrtan != ppcurrtan) dserror("Something wrong with penalty parameter");
            interface_[i]->InterfaceParams().set<double>("PENALTYPARAMTAN", 10 * ippcurrtan);
          }
        }
      }
    }
    //********************************************************************

    // update constraint norm
    constrnorm_ = cnorm;
    constrnormtan_ = cnormtan;
  }

  // output to screen
  if (Comm().MyPID() == 0)
  {
    std::cout << "********************************************\n";
    std::cout << "Normal Constraint Norm: " << cnorm << "\n";
    if (friction_) std::cout << "Tangential Constraint Norm: " << cnormtan << "\n";
    if (updatepenalty)
      std::cout << "Updated normal penalty parameter: " << ppcurr << " -> "
                << Params().get<double>("PENALTYPARAM") << "\n";
    if (updatepenaltytan == true && friction_)
      std::cout << "Updated tangential penalty parameter: " << ppcurrtan << " -> "
                << Params().get<double>("PENALTYPARAMTAN") << "\n";
    std::cout << "********************************************\n";
  }

  return;
}

/*----------------------------------------------------------------------*
 | store Lagrange multipliers for next Uzawa step             popp 08/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::UpdateUzawaAugmentedLagrange()
{
  // store current LM into Uzawa LM
  // (note that this is also done after the last Uzawa step of one
  // time step and thus also gives the guess for the initial
  // Lagrange multiplier lambda_0 of the next time step)
  zuzawa_ = Teuchos::rcp(new Epetra_Vector(*z_));
  StoreNodalQuantities(MORTAR::StrategyBase::lmuzawa);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::EvalForce(CONTACT::ParamsInterface& cparams)
{
  //---------------------------------------------------------------
  // For selfcontact the master/slave sets are updated within the -
  // contact search, see SelfBinaryTree.                          -
  // Therefore, we have to initialize the mortar matrices after   -
  // interface evaluations.                                       -
  //---------------------------------------------------------------
  if (IsSelfContact())
  {
    InitEvalInterface();  // evaluate mortar terms (integrate...)
    InitMortar();         // initialize mortar matrices and vectors
    AssembleMortar();     // assemble mortar terms into global matrices
  }
  else
  {
    InitMortar();         // initialize mortar matrices and vectors
    InitEvalInterface();  // evaluate mortar terms (integrate...)
    AssembleMortar();     // assemble mortar terms into global matrices
  }

  // evaluate relative movement for friction
  if (cparams.IsPredictor())
    EvaluateRelMovPredict();
  else
    EvaluateRelMov();

  // update active set
  UpdateActiveSetSemiSmooth();

  // apply contact forces and stiffness
  Initialize();  // init lin-matrices

  // assemble force and stiffness
  Assemble();

  // bye bye
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::Assemble()
{
  fc_ = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
  kc_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*ProblemDofs(), 100, true, true));

  // in the beginning of this function, the regularized contact forces
  // in normal and tangential direction are evaluated from geometric
  // measures (gap and relative tangential velocity). Here, also active and
  // slip nodes are detected. Then, the insertion of the according stiffness
  // blocks takes place.

  bool isincontact = false;
  bool activesetchange = false;

  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    bool localisincontact = false;
    bool localactivesetchange = false;

    // evaluate lagrange multipliers (regularized forces) in normal direction
    // and nodal derivz matrix values, store them in nodes
    interface_[i]->AssembleRegNormalForces(localisincontact, localactivesetchange);

    // evaluate lagrange multipliers (regularized forces) in tangential direction
    INPAR::CONTACT::SolvingStrategy soltype =
        INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(Params(), "STRATEGY");

    if (friction_ and (soltype == INPAR::CONTACT::solution_penalty or
                          soltype == INPAR::CONTACT::solution_multiscale))
      interface_[i]->AssembleRegTangentForcesPenalty();

    if (friction_ and soltype == INPAR::CONTACT::solution_uzawa)
      interface_[i]->AssembleRegTangentForcesUzawa();

    isincontact = isincontact || localisincontact;
    activesetchange = activesetchange || localactivesetchange;
  }

  // broadcast contact status & active set change
  int globalcontact, globalchange = 0;
  int localcontact = isincontact;
  int localchange = activesetchange;

  Comm().SumAll(&localcontact, &globalcontact, 1);
  Comm().SumAll(&localchange, &globalchange, 1);

  if (globalcontact >= 1)
  {
    isincontact_ = true;
    wasincontact_ = true;
  }
  else
    isincontact_ = false;

  if ((Comm().MyPID() == 0) && (globalchange >= 1))
    std::cout << "ACTIVE CONTACT SET HAS CHANGED..." << std::endl;

  // (re)setup active global Epetra_Maps
  // the map of global active nodes is needed for the penalty case, too.
  // this is due to the fact that we want to monitor the constraint norm
  // of the active nodes
  gactivenodes_ = Teuchos::null;
  gslipnodes_ = Teuchos::null;
  gactivedofs_ = Teuchos::null;

  // update active sets of all interfaces
  // (these maps are NOT allowed to be overlapping !!!)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->BuildActiveSet();
    gactivenodes_ = CORE::LINALG::MergeMap(gactivenodes_, interface_[i]->ActiveNodes(), false);
    gactivedofs_ = CORE::LINALG::MergeMap(gactivedofs_, interface_[i]->ActiveDofs(), false);
    gslipnodes_ = CORE::LINALG::MergeMap(gslipnodes_, interface_[i]->SlipNodes(), false);
  }

  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!IsInContact() && !WasInContact() && !WasInContactLastTimeStep()) return;

  // assemble contact quantities on all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // assemble global lagrangian multiplier vector
    interface_[i]->AssembleLM(*z_);
    // assemble global derivatives of lagrangian multipliers
    interface_[i]->AssembleLinZ(*linzmatrix_);
    // assemble global derivatives of mortar D and M matrices
    interface_[i]->AssembleLinDM(*lindmatrix_, *linmmatrix_);
  }

  // FillComplete() global matrices LinD, LinM, LinZ
  lindmatrix_->Complete(*gsmdofrowmap_, *gsdofrowmap_);
  linmmatrix_->Complete(*gsmdofrowmap_, *gmdofrowmap_);
  linzmatrix_->Complete(*gsmdofrowmap_, *gsdofrowmap_);

  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // LinD      ---->   T^(-T) * LinD
  // D         ---->   D * T^(-1)
  //----------------------------------------------------------------------
  if (Dualquadslavetrafo())
  {
    // modify lindmatrix_ and dmatrix_
    Teuchos::RCP<CORE::LINALG::SparseMatrix> temp1 =
        CORE::LINALG::MLMultiply(*invtrafo_, true, *lindmatrix_, false, false, false, true);
    Teuchos::RCP<CORE::LINALG::SparseMatrix> temp2 =
        CORE::LINALG::MLMultiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
    lindmatrix_ = temp1;
    dmatrix_ = temp2;
  }

  // **********************************************************************
  // Build Contact Stiffness #1
  // **********************************************************************
  // involving contributions of derivatives of D and M:
  //  Kc,1 = delta[ 0 -M(transpose) D] * LM

  // transform if necessary
  if (ParRedist())
  {
    lindmatrix_ = MORTAR::MatrixRowTransform(lindmatrix_, pgsdofrowmap_);
    linmmatrix_ = MORTAR::MatrixRowTransform(linmmatrix_, pgmdofrowmap_);
  }

  // add to kteff
  kc_->Add(*lindmatrix_, false, 1.0, 1.0);
  kc_->Add(*linmmatrix_, false, 1.0, 1.0);

  // **********************************************************************
  // Build Contact Stiffness #2
  // **********************************************************************
  // involving contributions of derivatives of lagrange multipliers:
  //  Kc,2= [ 0 -M(transpose) D] * deltaLM

  // multiply Mortar matrices D and M with LinZ
  Teuchos::RCP<CORE::LINALG::SparseMatrix> dtilde =
      CORE::LINALG::MLMultiply(*dmatrix_, true, *linzmatrix_, false, false, false, true);
  Teuchos::RCP<CORE::LINALG::SparseMatrix> mtilde =
      CORE::LINALG::MLMultiply(*mmatrix_, true, *linzmatrix_, false, false, false, true);

  // transform if necessary
  if (ParRedist())
  {
    dtilde = MORTAR::MatrixRowTransform(dtilde, pgsdofrowmap_);
    mtilde = MORTAR::MatrixRowTransform(mtilde, pgmdofrowmap_);
  }

  // add to kteff
  kc_->Add(*dtilde, false, 1.0, 1.0);
  kc_->Add(*mtilde, false, -(1.0), 1.0);

  // **********************************************************************
  // Build RHS
  // **********************************************************************
  // feff += -alphaf * fc,n - (1-alphaf) * fc,n+1,k
  {
    Teuchos::RCP<Epetra_Vector> fcmd = Teuchos::rcp(new Epetra_Vector(*gsdofrowmap_));
    dmatrix_->Multiply(true, *z_, *fcmd);
    Teuchos::RCP<Epetra_Vector> fcmdtemp = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
    CORE::LINALG::Export(*fcmd, *fcmdtemp);
    fc_->Update(-(1.), *fcmdtemp, 1.0);
  }

  {
    Teuchos::RCP<Epetra_Vector> fcmm = CORE::LINALG::CreateVector(*gmdofrowmap_, true);
    mmatrix_->Multiply(true, *z_, *fcmm);
    Teuchos::RCP<Epetra_Vector> fcmmtemp = Teuchos::rcp(new Epetra_Vector(*ProblemDofs()));
    CORE::LINALG::Export(*fcmm, *fcmmtemp);
    fc_->Update(1, *fcmmtemp, 1.0);
  }

  fc_->Scale(-1.);
  kc_->Complete();


  return;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> CONTACT::PenaltyStrategy::GetRhsBlockPtr(
    const enum CONTACT::VecBlockType& bt) const
{
  // if there are no active contact contributions
  if (!IsInContact() && !WasInContact() && !WasInContactLastTimeStep()) return Teuchos::null;

  Teuchos::RCP<const Epetra_Vector> vec_ptr = Teuchos::null;
  switch (bt)
  {
    case CONTACT::VecBlockType::displ:
    {
      vec_ptr = fc_;
      break;
    }
    case CONTACT::VecBlockType::constraint:
      return Teuchos::null;
      break;
    default:
    {
      dserror("Unknown STR::VecBlockType!");
      break;
    }
  }

  return vec_ptr;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::EvalForceStiff(CONTACT::ParamsInterface& cparams)
{
  // call the evaluate force routine if not done before
  if (!evalForceCalled_) EvalForce(cparams);

  // bye bye
  return;
}

/*----------------------------------------------------------------------*
 | set force evaluation flag before evaluation step          farah 08/16|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::PreEvaluate(CONTACT::ParamsInterface& cparams)
{
  const enum MORTAR::ActionType& act = cparams.GetActionType();

  switch (act)
  {
      // -------------------------------------------------------------------
      // reset force evaluation flag for predictor step
      // -------------------------------------------------------------------
    case MORTAR::eval_force_stiff:
    {
      if (cparams.IsPredictor()) evalForceCalled_ = false;
      break;
    }
    // -------------------------------------------------------------------
    // default
    // -------------------------------------------------------------------
    default:
    {
      // do nothing
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | set force evaluation flag after evaluation                farah 08/16|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::PostEvaluate(CONTACT::ParamsInterface& cparams)
{
  const enum MORTAR::ActionType& act = cparams.GetActionType();

  switch (act)
  {
    // -------------------------------------------------------------------
    // set flag to false after force stiff evaluation
    // -------------------------------------------------------------------
    case MORTAR::eval_force_stiff:
    {
      evalForceCalled_ = false;
      break;
    }
    // -------------------------------------------------------------------
    // set flag for force evaluation to true
    // -------------------------------------------------------------------
    case MORTAR::eval_force:
    {
      evalForceCalled_ = true;
      break;
    }
    // -------------------------------------------------------------------
    // default
    // -------------------------------------------------------------------
    default:
    {
      // do nothing
      break;
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::SparseMatrix> CONTACT::PenaltyStrategy::GetMatrixBlockPtr(
    const enum CONTACT::MatBlockType& bt, const CONTACT::ParamsInterface* cparams) const
{
  // if there are no active contact contributions
  if (!IsInContact() && !WasInContact() && !WasInContactLastTimeStep()) return Teuchos::null;

  Teuchos::RCP<CORE::LINALG::SparseMatrix> mat_ptr = Teuchos::null;
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_displ:
    {
      mat_ptr = kc_;
      break;
    }
    default:
    {
      dserror("Unknown STR::MatBlockType!");
      break;
    }
  }

  return mat_ptr;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> CONTACT::PenaltyStrategy::GetLagrMultN(const bool& redist) const
{
  if (GLOBAL::Problem::Instance()->StructuralDynamicParams().get<std::string>("INT_STRATEGY") ==
      "Old")
    return CONTACT::AbstractStrategy::GetLagrMultN(redist);
  else
    return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> CONTACT::PenaltyStrategy::GetLagrMultNp(const bool& redist) const
{
  if (GLOBAL::Problem::Instance()->StructuralDynamicParams().get<std::string>("INT_STRATEGY") ==
      "Old")
    return CONTACT::AbstractStrategy::GetLagrMultNp(redist);
  else
    return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> CONTACT::PenaltyStrategy::LagrMultOld()
{
  if (GLOBAL::Problem::Instance()->StructuralDynamicParams().get<std::string>("INT_STRATEGY") ==
      "Old")
    return CONTACT::AbstractStrategy::LagrMultOld();
  else
    return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> CONTACT::PenaltyStrategy::LMDoFRowMapPtr(const bool& redist) const
{
  if (GLOBAL::Problem::Instance()->StructuralDynamicParams().get<std::string>("INT_STRATEGY") ==
      "Old")
    return CONTACT::AbstractStrategy::LMDoFRowMapPtr(redist);
  else
    return Teuchos::null;
}

BACI_NAMESPACE_CLOSE
