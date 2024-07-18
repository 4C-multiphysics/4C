/*---------------------------------------------------------------------*/
/*! \file
\brief Penalty contact solving strategy: The contact constrains are enforced
       by a penalty formulation.

\level 2


*/
/*---------------------------------------------------------------------*/

#include "4C_contact_penalty_strategy.hpp"

#include "4C_contact_constitutivelaw_cubic_contactconstitutivelaw.hpp"
#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_node.hpp"
#include "4C_contact_paramsinterface.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_linalg_multiply.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_utils.hpp"

#include <Epetra_FEVector.h>
#include <Epetra_Operator.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                              popp 05/09|
 *----------------------------------------------------------------------*/
CONTACT::PenaltyStrategy::PenaltyStrategy(const Epetra_Map* dof_row_map,
    const Epetra_Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<Teuchos::RCP<CONTACT::Interface>> interface, const int spatialDim,
    const Teuchos::RCP<const Epetra_Comm>& comm, const double alphaf, const int maxdof)
    : AbstractStrategy(Teuchos::rcp(new CONTACT::AbstractStratDataContainer()), dof_row_map,
          NodeRowMap, params, spatialDim, comm, alphaf, maxdof),
      interface_(interface),
      constrnorm_(0.0),
      constrnormtan_(0.0),
      initialpenalty_(PenaltyStrategy::params().get<double>("PENALTYPARAM")),
      initialpenaltytan_(PenaltyStrategy::params().get<double>("PENALTYPARAMTAN"))
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::PenaltyStrategy::PenaltyStrategy(
    const Teuchos::RCP<CONTACT::AbstractStratDataContainer>& data_ptr,
    const Epetra_Map* dof_row_map, const Epetra_Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<Teuchos::RCP<CONTACT::Interface>> interface, const int spatialDim,
    const Teuchos::RCP<const Epetra_Comm>& comm, const double alphaf, const int maxdof)
    : AbstractStrategy(data_ptr, dof_row_map, NodeRowMap, params, spatialDim, comm, alphaf, maxdof),
      interface_(interface),
      constrnorm_(0.0),
      constrnormtan_(0.0),
      initialpenalty_(PenaltyStrategy::params().get<double>("PENALTYPARAM")),
      initialpenaltytan_(PenaltyStrategy::params().get<double>("PENALTYPARAMTAN"))
{
  // empty constructor
}


/*----------------------------------------------------------------------*
 |  save the gap-scaling kappa from reference config          popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::save_reference_state(Teuchos::RCP<const Epetra_Vector> dis)
{
  // initialize the displacement field
  set_state(Mortar::state_new_displacement, *dis);

  // kappa will be the shape function integral on the slave sides
  // (1) build the nodal information
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // interface needs to be complete
    if (!interface_[i]->filled() && get_comm().MyPID() == 0)
      FOUR_C_THROW("fill_complete() not called on interface %", i);

    // do the computation of nodal shape function integral
    // (for convenience, the results will be stored in nodal gap)

    // loop over proc's slave elements of the interface for integration
    // use standard column map to include processor's ghosted elements
    for (int j = 0; j < interface_[i]->slave_col_elements()->NumMyElements(); ++j)
    {
      int gid1 = interface_[i]->slave_col_elements()->GID(j);
      Core::Elements::Element* ele1 = interface_[i]->discret().g_element(gid1);
      if (!ele1) FOUR_C_THROW("Cannot find slave element with gid %", gid1);
      Element* selement = dynamic_cast<Element*>(ele1);

      interface_[i]->integrate_kappa_penalty(*selement);
    }

    // loop over all slave row nodes on the current interface
    for (int j = 0; j < interface_[i]->slave_row_nodes()->NumMyElements(); ++j)
    {
      int gid = interface_[i]->slave_row_nodes()->GID(j);
      Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      Node* cnode = dynamic_cast<Node*>(node);

      // get nodal weighted gap
      // (this is where we stored the shape function integrals)
      double gap = cnode->data().getg();

      // store kappa as the inverse of gap
      // (this removes the scaling introduced by weighting the gap!!!)
      cnode->data().kappa() = 1.0 / gap;

      // std::cout << "S-NODE #" << gid << " kappa=" << cnode->Data().Kappa() << std::endl;
    }
  }
}

/*----------------------------------------------------------------------*
 | evaluate relative movement in predictor step               popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::predict_relative_movement()
{
  // only for frictional contact
  if (friction_ == false) return;

  // call evaluation method of base class
  evaluate_relative_movement();

  return;
}

/*----------------------------------------------------------------------*
 | initialize global contact variables for next Newton step   popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::initialize()
{
  // (re)setup global matrices containing fc derivatives
  // must use FE_MATRIX type here, as we will do non-local assembly!
  lindmatrix_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
      *gsdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));
  linmmatrix_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));

  // (re)setup global vector containing lagrange multipliers
  z_ = Core::LinAlg::CreateVector(*gsdofrowmap_, true);

  // (re)setup global matrix containing lagrange multiplier derivatives
  linzmatrix_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*gsdofrowmap_, 100));

  return;
}

/*----------------------------------------------------------------------*
 | evaluate contact and create linear system                  popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::evaluate_contact(
    Teuchos::RCP<Core::LinAlg::SparseOperator>& kteff, Teuchos::RCP<Epetra_Vector>& feff)
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
    interface_[i]->assemble_reg_normal_forces(localisincontact, localactivesetchange);

    // evaluate lagrange multipliers (regularized forces) in tangential direction
    Inpar::CONTACT::SolvingStrategy soltype =
        Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(params(), "STRATEGY");

    if (friction_ and (soltype == Inpar::CONTACT::solution_penalty or
                          soltype == Inpar::CONTACT::solution_multiscale))
      interface_[i]->assemble_reg_tangent_forces_penalty();

    if (friction_ and soltype == Inpar::CONTACT::solution_uzawa)
      interface_[i]->assemble_reg_tangent_forces_uzawa();

    isincontact = isincontact || localisincontact;
    activesetchange = activesetchange || localactivesetchange;
  }

  // broadcast contact status & active set change
  int globalcontact, globalchange = 0;
  int localcontact = isincontact;
  int localchange = activesetchange;

  get_comm().SumAll(&localcontact, &globalcontact, 1);
  get_comm().SumAll(&localchange, &globalchange, 1);

  if (globalcontact >= 1)
  {
    isincontact_ = true;
    wasincontact_ = true;
  }
  else
    isincontact_ = false;

  if ((get_comm().MyPID() == 0) && (globalchange >= 1))
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
    interface_[i]->build_active_set();
    gactivenodes_ = Core::LinAlg::MergeMap(gactivenodes_, interface_[i]->active_nodes(), false);
    gactivedofs_ = Core::LinAlg::MergeMap(gactivedofs_, interface_[i]->active_dofs(), false);
    gslipnodes_ = Core::LinAlg::MergeMap(gslipnodes_, interface_[i]->slip_nodes(), false);
  }

  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  // since we will modify the graph of kteff by adding additional
  // meshtyong stiffness entries, we have to uncomplete it
  kteff->un_complete();

  // assemble contact quantities on all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // assemble global lagrangian multiplier vector
    interface_[i]->assemble_lm(*z_);
    // assemble global derivatives of lagrangian multipliers
    interface_[i]->assemble_lin_z(*linzmatrix_);
    // assemble global derivatives of mortar D and M matrices
    interface_[i]->assemble_lin_dm(*lindmatrix_, *linmmatrix_);
  }

  // fill_complete() global matrices LinD, LinM, LinZ
  lindmatrix_->complete(*gsmdofrowmap_, *gsdofrowmap_);
  linmmatrix_->complete(*gsmdofrowmap_, *gmdofrowmap_);
  linzmatrix_->complete(*gsmdofrowmap_, *gsdofrowmap_);

  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // LinD      ---->   T^(-T) * LinD
  // D         ---->   D * T^(-1)
  //----------------------------------------------------------------------
  if (is_dual_quad_slave_trafo())
  {
    // modify lindmatrix_ and dmatrix_
    Teuchos::RCP<Core::LinAlg::SparseMatrix> temp1 =
        Core::LinAlg::MLMultiply(*invtrafo_, true, *lindmatrix_, false, false, false, true);
    Teuchos::RCP<Core::LinAlg::SparseMatrix> temp2 =
        Core::LinAlg::MLMultiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
    lindmatrix_ = temp1;
    dmatrix_ = temp2;
  }

#ifdef CONTACTFDPENALTYTRAC
  Inpar::CONTACT::FrictionType ftype =
      Core::UTILS::IntegralValue<Inpar::CONTACT::FrictionType>(Params(), "FRICTION");

  // check derivatives of penalty traction
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    if (IsInContact())
    {
      if (ftype == Inpar::CONTACT::friction_coulomb)
      {
        std::cout << "LINZMATRIX" << *linzmatrix_ << std::endl;
        interface_[i]->fd_check_penalty_trac_fric();
      }
      else if (ftype == Inpar::CONTACT::friction_none)
      {
        std::cout << "-- CONTACTFDDERIVZ --------------------" << std::endl;
        interface_[i]->fd_check_penalty_trac_nor();
        std::cout << "-- CONTACTFDDERIVZ --------------------" << std::endl;
      }
      else
        FOUR_C_THROW("Error: FD Check for this friction type not implemented!");
    }
  }
#endif

  // **********************************************************************
  // Build Contact Stiffness #1
  // **********************************************************************
  // involving contributions of derivatives of D and M:
  //  Kc,1 = delta[ 0 -M(transpose) D] * LM

  // transform if necessary
  if (parallel_redistribution_status())
  {
    lindmatrix_ = Mortar::MatrixRowTransform(lindmatrix_, pgsdofrowmap_);
    linmmatrix_ = Mortar::MatrixRowTransform(linmmatrix_, pgmdofrowmap_);
  }

  // add to kteff
  kteff->add(*lindmatrix_, false, 1.0 - alphaf_, 1.0);
  kteff->add(*linmmatrix_, false, 1.0 - alphaf_, 1.0);

  // **********************************************************************
  // Build Contact Stiffness #2
  // **********************************************************************
  // involving contributions of derivatives of lagrange multipliers:
  //  Kc,2= [ 0 -M(transpose) D] * deltaLM

  // multiply Mortar matrices D and M with LinZ
  Teuchos::RCP<Core::LinAlg::SparseMatrix> dtilde =
      Core::LinAlg::MLMultiply(*dmatrix_, true, *linzmatrix_, false, false, false, true);
  Teuchos::RCP<Core::LinAlg::SparseMatrix> mtilde =
      Core::LinAlg::MLMultiply(*mmatrix_, true, *linzmatrix_, false, false, false, true);

  // transform if necessary
  if (parallel_redistribution_status())
  {
    dtilde = Mortar::MatrixRowTransform(dtilde, pgsdofrowmap_);
    mtilde = Mortar::MatrixRowTransform(mtilde, pgmdofrowmap_);
  }

  // add to kteff
  kteff->add(*dtilde, false, 1.0 - alphaf_, 1.0);
  kteff->add(*mtilde, false, -(1.0 - alphaf_), 1.0);

  // **********************************************************************
  // Build RHS
  // **********************************************************************
  // feff += -alphaf * fc,n - (1-alphaf) * fc,n+1,k

  {
    // we initialize fcmdold with dold-rowmap instead of gsdofrowmap
    // (this way, possible self contact is automatically included)

    Teuchos::RCP<Epetra_Vector> fcmdold = Teuchos::rcp(new Epetra_Vector(dold_->row_map()));
    dold_->multiply(true, *zold_, *fcmdold);
    Teuchos::RCP<Epetra_Vector> fcmdoldtemp = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
    Core::LinAlg::export_to(*fcmdold, *fcmdoldtemp);
    feff->Update(-alphaf_, *fcmdoldtemp, 1.0);
  }

  {
    // we initialize fcmmold with mold-domainmap instead of gmdofrowmap
    // (this way, possible self contact is automatically included)

    Teuchos::RCP<Epetra_Vector> fcmmold = Teuchos::rcp(new Epetra_Vector(mold_->domain_map()));
    mold_->multiply(true, *zold_, *fcmmold);
    Teuchos::RCP<Epetra_Vector> fcmmoldtemp = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
    Core::LinAlg::export_to(*fcmmold, *fcmmoldtemp);
    feff->Update(alphaf_, *fcmmoldtemp, 1.0);
  }

  {
    Teuchos::RCP<Epetra_Vector> fcmd = Teuchos::rcp(new Epetra_Vector(*gsdofrowmap_));
    dmatrix_->multiply(true, *z_, *fcmd);
    Teuchos::RCP<Epetra_Vector> fcmdtemp = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
    Core::LinAlg::export_to(*fcmd, *fcmdtemp);
    feff->Update(-(1 - alphaf_), *fcmdtemp, 1.0);
  }

  {
    Teuchos::RCP<Epetra_Vector> fcmm = Core::LinAlg::CreateVector(*gmdofrowmap_, true);
    mmatrix_->multiply(true, *z_, *fcmm);
    Teuchos::RCP<Epetra_Vector> fcmmtemp = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
    Core::LinAlg::export_to(*fcmm, *fcmmtemp);
    feff->Update(1 - alphaf_, *fcmmtemp, 1.0);
  }

#ifdef CONTACTFDGAP
  // FD check of weighted gap g derivatives (non-penetr. condition)

  std::cout << "-- CONTACTFDGAP -----------------------------" << std::endl;
  interface_[0]->FDCheckGapDeriv();
  std::cout << "-- CONTACTFDGAP -----------------------------" << std::endl;

#endif

  return;
}

/*----------------------------------------------------------------------*
 | evaluate frictional contact and create linear system gitterle   10/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::evaluate_friction(
    Teuchos::RCP<Core::LinAlg::SparseOperator>& kteff, Teuchos::RCP<Epetra_Vector>& feff)
{
  // this is almost the same as in the frictionless contact
  // whereas we chose the evaluate_contact routine with
  // one difference

  // check if friction should be applied
  Inpar::CONTACT::FrictionType ftype =
      Core::UTILS::IntegralValue<Inpar::CONTACT::FrictionType>(params(), "FRICTION");

  // coulomb friction case
  if (ftype == Inpar::CONTACT::friction_coulomb || ftype == Inpar::CONTACT::friction_stick)
  {
    evaluate_contact(kteff, feff);
  }
  else if (ftype == Inpar::CONTACT::friction_tresca)
  {
    FOUR_C_THROW(
        "Error in AbstractStrategy::Evaluate: Penalty Strategy for"
        " Tresca friction not yet implemented");
  }
  else
    FOUR_C_THROW("Error in AbstractStrategy::Evaluate: Unknown friction type");

  return;
}

/*----------------------------------------------------------------------*
 | reset penalty parameter to intial value                    popp 08/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::reset_penalty()
{
  // reset penalty parameter in strategy
  params().set<double>("PENALTYPARAM", initial_penalty());
  params().set<double>("PENALTYPARAMTAN", initial_penalty_tan());

  // reset penalty parameter in all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->interface_params().set<double>("PENALTYPARAM", initial_penalty());
    interface_[i]->interface_params().set<double>("PENALTYPARAMTAN", initial_penalty_tan());
  }

  return;
}

/*----------------------------------------------------------------------*
 | modify penalty parameter to intial value                    mhv 03/16|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::modify_penalty()
{
  // generate random number between 0.95 and 1.05
  double randnum = ((double)rand() / (double)RAND_MAX) * 0.1 + 0.95;
  double pennew = randnum * initial_penalty();

  // modify penalty parameter in strategy
  params().set<double>("PENALTYPARAM", pennew);
  params().set<double>("PENALTYPARAMTAN", pennew);

  // modify penalty parameter in all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->interface_params().set<double>("PENALTYPARAM", pennew);
    interface_[i]->interface_params().set<double>("PENALTYPARAMTAN", pennew);
  }

  return;
}

/*----------------------------------------------------------------------*
 | intialize second, third,... Uzawa step                     popp 01/10|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::initialize_uzawa(
    Teuchos::RCP<Core::LinAlg::SparseOperator>& kteff, Teuchos::RCP<Epetra_Vector>& feff)
{
  // remove old stiffness terms
  // (FIXME: redundant code to evaluate_contact(), expect for minus sign)

  // since we will modify the graph of kteff by adding additional
  // meshtying stiffness entries, we have to uncomplete it
  kteff->un_complete();

  // remove contact stiffness #1 from kteff
  kteff->add(*lindmatrix_, false, -(1.0 - alphaf_), 1.0);
  kteff->add(*linmmatrix_, false, -(1.0 - alphaf_), 1.0);

  // multiply Mortar matrices D and M with LinZ
  Teuchos::RCP<Core::LinAlg::SparseMatrix> dtilde =
      Core::LinAlg::MLMultiply(*dmatrix_, true, *linzmatrix_, false, false, false, true);
  Teuchos::RCP<Core::LinAlg::SparseMatrix> mtilde =
      Core::LinAlg::MLMultiply(*mmatrix_, true, *linzmatrix_, false, false, false, true);

  // transform if necessary
  if (parallel_redistribution_status())
  {
    dtilde = Mortar::MatrixRowTransform(dtilde, pgsdofrowmap_);
    mtilde = Mortar::MatrixRowTransform(mtilde, pgmdofrowmap_);
  }

  // remove contact stiffness #2 from kteff
  kteff->add(*dtilde, false, -(1.0 - alphaf_), 1.0);
  kteff->add(*mtilde, false, (1.0 - alphaf_), 1.0);

  // remove old force terms
  // (FIXME: redundant code to evaluate_contact(), expect for minus sign)

  Teuchos::RCP<Epetra_Vector> fcmdold = Teuchos::rcp(new Epetra_Vector(dold_->row_map()));
  dold_->multiply(true, *zold_, *fcmdold);
  Teuchos::RCP<Epetra_Vector> fcmdoldtemp = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
  Core::LinAlg::export_to(*fcmdold, *fcmdoldtemp);
  feff->Update(alphaf_, *fcmdoldtemp, 1.0);

  Teuchos::RCP<Epetra_Vector> fcmmold = Teuchos::rcp(new Epetra_Vector(mold_->domain_map()));
  mold_->multiply(true, *zold_, *fcmmold);
  Teuchos::RCP<Epetra_Vector> fcmmoldtemp = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
  Core::LinAlg::export_to(*fcmmold, *fcmmoldtemp);
  feff->Update(-alphaf_, *fcmmoldtemp, 1.0);

  Teuchos::RCP<Epetra_Vector> fcmd = Teuchos::rcp(new Epetra_Vector(*gsdofrowmap_));
  dmatrix_->multiply(true, *z_, *fcmd);
  Teuchos::RCP<Epetra_Vector> fcmdtemp = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
  Core::LinAlg::export_to(*fcmd, *fcmdtemp);
  feff->Update(1 - alphaf_, *fcmdtemp, 1.0);

  Teuchos::RCP<Epetra_Vector> fcmm = Core::LinAlg::CreateVector(*gmdofrowmap_, true);
  mmatrix_->multiply(true, *z_, *fcmm);
  Teuchos::RCP<Epetra_Vector> fcmmtemp = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
  Core::LinAlg::export_to(*fcmm, *fcmmtemp);
  feff->Update(-(1 - alphaf_), *fcmmtemp, 1.0);

  // reset some matrices
  // must use FE_MATRIX type here, as we will do non-local assembly!
  lindmatrix_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
      *gsdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));
  linmmatrix_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));

  // reset nodal derivZ values
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    for (int j = 0; j < interface_[i]->slave_col_nodes_bound()->NumMyElements(); ++j)
    {
      int gid = interface_[i]->slave_col_nodes_bound()->GID(j);
      Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      Node* cnode = dynamic_cast<Node*>(node);

      for (int k = 0; k < (int)((cnode->data().get_deriv_z()).size()); ++k)
        (cnode->data().get_deriv_z())[k].clear();
      (cnode->data().get_deriv_z()).resize(0);
    }
  }

  // now redo initialize()
  initialize();

  // and finally redo evaluate()
  Teuchos::RCP<Epetra_Vector> nullvec = Teuchos::null;
  evaluate(kteff, feff, nullvec);

  // complete stiffness matrix
  kteff->complete();

  return;
}

/*----------------------------------------------------------------------*
 | evaluate L2-norm of active constraints                     popp 08/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::update_constraint_norm(int uzawaiter)
{
  // initialize parameters
  double cnorm = 0.0;
  double cnormtan = 0.0;
  bool updatepenalty = false;
  bool updatepenaltytan = false;
  double ppcurr = params().get<double>("PENALTYPARAM");
  double ppcurrtan = params().get<double>("PENALTYPARAMTAN");

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
    if (constr_direction_ == Inpar::CONTACT::constr_xyz)
    {
      gact = Core::LinAlg::CreateVector(*gactivedofs_, true);
      Core::LinAlg::export_to(*wgap_, *gact);
    }
    else
    {
      gact = Core::LinAlg::CreateVector(*gactivenodes_, true);
      if (gact->GlobalLength()) Core::LinAlg::export_to(*wgap_, *gact);
    }

    // compute constraint norm
    gact->Norm2(&cnorm);

    // Evaluate norm in tangential direction for frictional contact
    if (friction_)
    {
      for (int i = 0; i < (int)interface_.size(); ++i)
        interface_[i]->evaluate_tangent_norm(cnormtan);

      cnormtan = sqrt(cnormtan);
    }

    //********************************************************************
    // adaptive update of penalty parameter
    // (only for Uzawa Augmented Lagrange strategy)
    //********************************************************************
    Inpar::CONTACT::SolvingStrategy soltype =
        Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(params(), "STRATEGY");

    if (soltype == Inpar::CONTACT::solution_uzawa)
    {
      // check convergence of cnorm and update penalty parameter
      // only do this for second, third, ... Uzawa iteration
      // cf. Wriggers, Computational Contact Mechanics, 2nd edition (2006), p. 340
      if ((uzawaiter >= 2) && (cnorm > 0.25 * constraint_norm()))
      {
        updatepenalty = true;

        // update penalty parameter in strategy
        params().set<double>("PENALTYPARAM", 10 * ppcurr);

        // update penalty parameter in all interfaces
        for (int i = 0; i < (int)interface_.size(); ++i)
        {
          double ippcurr = interface_[i]->interface_params().get<double>("PENALTYPARAM");
          if (ippcurr != ppcurr) FOUR_C_THROW("Something wrong with penalty parameter");
          interface_[i]->interface_params().set<double>("PENALTYPARAM", 10 * ippcurr);
        }
        // in the case of frictional contact, the tangential penalty
        // parameter is also dated up when this is done for the normal one
        if (friction_)
        {
          updatepenaltytan = true;

          // update penalty parameter in strategy
          params().set<double>("PENALTYPARAMTAN", 10 * ppcurrtan);

          // update penalty parameter in all interfaces
          for (int i = 0; i < (int)interface_.size(); ++i)
          {
            double ippcurrtan = interface_[i]->interface_params().get<double>("PENALTYPARAMTAN");
            if (ippcurrtan != ppcurrtan) FOUR_C_THROW("Something wrong with penalty parameter");
            interface_[i]->interface_params().set<double>("PENALTYPARAMTAN", 10 * ippcurrtan);
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
  if (get_comm().MyPID() == 0)
  {
    std::cout << "********************************************\n";
    std::cout << "Normal Constraint Norm: " << cnorm << "\n";
    if (friction_) std::cout << "Tangential Constraint Norm: " << cnormtan << "\n";
    if (updatepenalty)
      std::cout << "Updated normal penalty parameter: " << ppcurr << " -> "
                << params().get<double>("PENALTYPARAM") << "\n";
    if (updatepenaltytan == true && friction_)
      std::cout << "Updated tangential penalty parameter: " << ppcurrtan << " -> "
                << params().get<double>("PENALTYPARAMTAN") << "\n";
    std::cout << "********************************************\n";
  }

  return;
}

/*----------------------------------------------------------------------*
 | store Lagrange multipliers for next Uzawa step             popp 08/09|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::update_uzawa_augmented_lagrange()
{
  // store current LM into Uzawa LM
  // (note that this is also done after the last Uzawa step of one
  // time step and thus also gives the guess for the initial
  // Lagrange multiplier lambda_0 of the next time step)
  zuzawa_ = Teuchos::rcp(new Epetra_Vector(*z_));
  store_nodal_quantities(Mortar::StrategyBase::lmuzawa);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::evaluate_force(CONTACT::ParamsInterface& cparams)
{
  //---------------------------------------------------------------
  // For selfcontact the master/slave sets are updated within the -
  // contact search, see SelfBinaryTree.                          -
  // Therefore, we have to initialize the mortar matrices after   -
  // interface evaluations.                                       -
  //---------------------------------------------------------------
  if (is_self_contact())
  {
    initialize_and_evaluate_interface();  // evaluate mortar terms (integrate...)
    initialize_mortar();                  // initialize mortar matrices and vectors
    assemble_mortar();                    // assemble mortar terms into global matrices
  }
  else
  {
    initialize_mortar();                  // initialize mortar matrices and vectors
    initialize_and_evaluate_interface();  // evaluate mortar terms (integrate...)
    assemble_mortar();                    // assemble mortar terms into global matrices
  }

  // evaluate relative movement for friction
  if (cparams.is_predictor())
    predict_relative_movement();
  else
    evaluate_relative_movement();

  // update active set
  update_active_set_semi_smooth();

  // apply contact forces and stiffness
  initialize();  // init lin-matrices

  // assemble force and stiffness
  assemble();

  // bye bye
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::assemble()
{
  fc_ = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
  kc_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*problem_dofs(), 100, true, true));

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
    interface_[i]->assemble_reg_normal_forces(localisincontact, localactivesetchange);

    // evaluate lagrange multipliers (regularized forces) in tangential direction
    Inpar::CONTACT::SolvingStrategy soltype =
        Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(params(), "STRATEGY");

    if (friction_ and (soltype == Inpar::CONTACT::solution_penalty or
                          soltype == Inpar::CONTACT::solution_multiscale))
      interface_[i]->assemble_reg_tangent_forces_penalty();

    if (friction_ and soltype == Inpar::CONTACT::solution_uzawa)
      interface_[i]->assemble_reg_tangent_forces_uzawa();

    isincontact = isincontact || localisincontact;
    activesetchange = activesetchange || localactivesetchange;
  }

  // broadcast contact status & active set change
  int globalcontact, globalchange = 0;
  int localcontact = isincontact;
  int localchange = activesetchange;

  get_comm().SumAll(&localcontact, &globalcontact, 1);
  get_comm().SumAll(&localchange, &globalchange, 1);

  if (globalcontact >= 1)
  {
    isincontact_ = true;
    wasincontact_ = true;
  }
  else
    isincontact_ = false;

  if ((get_comm().MyPID() == 0) && (globalchange >= 1))
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
    interface_[i]->build_active_set();
    gactivenodes_ = Core::LinAlg::MergeMap(gactivenodes_, interface_[i]->active_nodes(), false);
    gactivedofs_ = Core::LinAlg::MergeMap(gactivedofs_, interface_[i]->active_dofs(), false);
    gslipnodes_ = Core::LinAlg::MergeMap(gslipnodes_, interface_[i]->slip_nodes(), false);
  }

  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  // assemble contact quantities on all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // assemble global lagrangian multiplier vector
    interface_[i]->assemble_lm(*z_);
    // assemble global derivatives of lagrangian multipliers
    interface_[i]->assemble_lin_z(*linzmatrix_);
    // assemble global derivatives of mortar D and M matrices
    interface_[i]->assemble_lin_dm(*lindmatrix_, *linmmatrix_);
  }

  // fill_complete() global matrices LinD, LinM, LinZ
  lindmatrix_->complete(*gsmdofrowmap_, *gsdofrowmap_);
  linmmatrix_->complete(*gsmdofrowmap_, *gmdofrowmap_);
  linzmatrix_->complete(*gsmdofrowmap_, *gsdofrowmap_);

  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // LinD      ---->   T^(-T) * LinD
  // D         ---->   D * T^(-1)
  //----------------------------------------------------------------------
  if (is_dual_quad_slave_trafo())
  {
    // modify lindmatrix_ and dmatrix_
    Teuchos::RCP<Core::LinAlg::SparseMatrix> temp1 =
        Core::LinAlg::MLMultiply(*invtrafo_, true, *lindmatrix_, false, false, false, true);
    Teuchos::RCP<Core::LinAlg::SparseMatrix> temp2 =
        Core::LinAlg::MLMultiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
    lindmatrix_ = temp1;
    dmatrix_ = temp2;
  }

  // **********************************************************************
  // Build Contact Stiffness #1
  // **********************************************************************
  // involving contributions of derivatives of D and M:
  //  Kc,1 = delta[ 0 -M(transpose) D] * LM

  // transform if necessary
  if (parallel_redistribution_status())
  {
    lindmatrix_ = Mortar::MatrixRowTransform(lindmatrix_, pgsdofrowmap_);
    linmmatrix_ = Mortar::MatrixRowTransform(linmmatrix_, pgmdofrowmap_);
  }

  // add to kteff
  kc_->add(*lindmatrix_, false, 1.0, 1.0);
  kc_->add(*linmmatrix_, false, 1.0, 1.0);

  // **********************************************************************
  // Build Contact Stiffness #2
  // **********************************************************************
  // involving contributions of derivatives of lagrange multipliers:
  //  Kc,2= [ 0 -M(transpose) D] * deltaLM

  // multiply Mortar matrices D and M with LinZ
  Teuchos::RCP<Core::LinAlg::SparseMatrix> dtilde =
      Core::LinAlg::MLMultiply(*dmatrix_, true, *linzmatrix_, false, false, false, true);
  Teuchos::RCP<Core::LinAlg::SparseMatrix> mtilde =
      Core::LinAlg::MLMultiply(*mmatrix_, true, *linzmatrix_, false, false, false, true);

  // transform if necessary
  if (parallel_redistribution_status())
  {
    dtilde = Mortar::MatrixRowTransform(dtilde, pgsdofrowmap_);
    mtilde = Mortar::MatrixRowTransform(mtilde, pgmdofrowmap_);
  }

  // add to kteff
  kc_->add(*dtilde, false, 1.0, 1.0);
  kc_->add(*mtilde, false, -(1.0), 1.0);

  // **********************************************************************
  // Build RHS
  // **********************************************************************
  // feff += -alphaf * fc,n - (1-alphaf) * fc,n+1,k
  {
    Teuchos::RCP<Epetra_Vector> fcmd = Teuchos::rcp(new Epetra_Vector(*gsdofrowmap_));
    dmatrix_->multiply(true, *z_, *fcmd);
    Teuchos::RCP<Epetra_Vector> fcmdtemp = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
    Core::LinAlg::export_to(*fcmd, *fcmdtemp);
    fc_->Update(-(1.), *fcmdtemp, 1.0);
  }

  {
    Teuchos::RCP<Epetra_Vector> fcmm = Core::LinAlg::CreateVector(*gmdofrowmap_, true);
    mmatrix_->multiply(true, *z_, *fcmm);
    Teuchos::RCP<Epetra_Vector> fcmmtemp = Teuchos::rcp(new Epetra_Vector(*problem_dofs()));
    Core::LinAlg::export_to(*fcmm, *fcmmtemp);
    fc_->Update(1, *fcmmtemp, 1.0);
  }

  fc_->Scale(-1.);
  kc_->complete();


  return;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> CONTACT::PenaltyStrategy::get_rhs_block_ptr(
    const enum CONTACT::VecBlockType& bt) const
{
  // if there are no active contact contributions
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step())
    return Teuchos::null;

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
      FOUR_C_THROW("Unknown Solid::VecBlockType!");
      break;
    }
  }

  return vec_ptr;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::evaluate_force_stiff(CONTACT::ParamsInterface& cparams)
{
  // call the evaluate force routine if not done before
  if (!evalForceCalled_) evaluate_force(cparams);

  // bye bye
  return;
}

/*----------------------------------------------------------------------*
 | set force evaluation flag before evaluation step          farah 08/16|
 *----------------------------------------------------------------------*/
void CONTACT::PenaltyStrategy::pre_evaluate(CONTACT::ParamsInterface& cparams)
{
  const enum Mortar::ActionType& act = cparams.get_action_type();

  switch (act)
  {
      // -------------------------------------------------------------------
      // reset force evaluation flag for predictor step
      // -------------------------------------------------------------------
    case Mortar::eval_force_stiff:
    {
      if (cparams.is_predictor()) evalForceCalled_ = false;
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
void CONTACT::PenaltyStrategy::post_evaluate(CONTACT::ParamsInterface& cparams)
{
  const enum Mortar::ActionType& act = cparams.get_action_type();

  switch (act)
  {
    // -------------------------------------------------------------------
    // set flag to false after force stiff evaluation
    // -------------------------------------------------------------------
    case Mortar::eval_force_stiff:
    {
      evalForceCalled_ = false;
      break;
    }
    // -------------------------------------------------------------------
    // set flag for force evaluation to true
    // -------------------------------------------------------------------
    case Mortar::eval_force:
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
Teuchos::RCP<Core::LinAlg::SparseMatrix> CONTACT::PenaltyStrategy::get_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt, const CONTACT::ParamsInterface* cparams) const
{
  // if there are no active contact contributions
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step())
    return Teuchos::null;

  Teuchos::RCP<Core::LinAlg::SparseMatrix> mat_ptr = Teuchos::null;
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_displ:
    {
      mat_ptr = kc_;
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown Solid::MatBlockType!");
      break;
    }
  }

  return mat_ptr;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> CONTACT::PenaltyStrategy::lagrange_multiplier_n(
    const bool& redist) const
{
  if (Global::Problem::instance()->structural_dynamic_params().get<std::string>("INT_STRATEGY") ==
      "Old")
    return CONTACT::AbstractStrategy::lagrange_multiplier_n(redist);
  else
    return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> CONTACT::PenaltyStrategy::lagrange_multiplier_np(
    const bool& redist) const
{
  if (Global::Problem::instance()->structural_dynamic_params().get<std::string>("INT_STRATEGY") ==
      "Old")
    return CONTACT::AbstractStrategy::lagrange_multiplier_np(redist);
  else
    return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> CONTACT::PenaltyStrategy::lagrange_multiplier_old()
{
  if (Global::Problem::instance()->structural_dynamic_params().get<std::string>("INT_STRATEGY") ==
      "Old")
    return CONTACT::AbstractStrategy::lagrange_multiplier_old();
  else
    return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> CONTACT::PenaltyStrategy::lm_dof_row_map_ptr(
    const bool& redist) const
{
  if (Global::Problem::instance()->structural_dynamic_params().get<std::string>("INT_STRATEGY") ==
      "Old")
    return CONTACT::AbstractStrategy::lm_dof_row_map_ptr(redist);
  else
    return Teuchos::null;
}

FOUR_C_NAMESPACE_CLOSE
