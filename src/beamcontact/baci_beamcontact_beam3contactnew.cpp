/*----------------------------------------------------------------------*/
/*! \file

\brief contact element for contact between two 3D beam elements

\level 3

*/
/*----------------------------------------------------------------------*/

#include "baci_beamcontact_beam3contactnew.H"

#include "baci_beam3_euler_bernoulli.H"
#include "baci_beam3_reissner.H"
#include "baci_beaminteraction_beam_to_beam_contact_defines.H"
#include "baci_beaminteraction_beam_to_beam_contact_tangentsmoothing.H"
#include "baci_beaminteraction_beam_to_beam_contact_utils.H"
#include "baci_discretization_fem_general_utils_fem_shapefunctions.H"
#include "baci_inpar_beamcontact.H"
#include "baci_inpar_contact.H"
#include "baci_lib_discret.H"
#include "baci_lib_globalproblem.H"
#include "baci_linalg_utils_sparse_algebra_assemble.H"
#include "baci_structure_timint_impl.H"
#include "baci_utils_exceptions.H"

#include <Teuchos_TimeMonitor.hpp>

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  constructor (public)                                     meier 01/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
CONTACT::Beam3contactnew<numnodes, numnodalvalues>::Beam3contactnew(
    const DRT::Discretization& pdiscret, const DRT::Discretization& cdiscret,
    const std::map<int, int>& dofoffsetmap, DRT::Element* element1, DRT::Element* element2,
    Teuchos::ParameterList& beamcontactparams)
    : pdiscret_(pdiscret),
      cdiscret_(cdiscret),
      dofoffsetmap_(dofoffsetmap),
      element1_(element1),
      element2_(element2),
      bcparams_(beamcontactparams),
      sgn_(1.0),
      firstcallofstep_(true),
      firsttimestep_(true),
      gap_(0.0),
      gap_original_(0.0),
      contactflag_(false),
      dampingcontactflag_(false),
      oldcontactflag_(false),
      currentlyincontact_(false),
      elementscolinear_(false),
      elementscrossing_(false),
      shiftnodalvalues_(false),
      xi1_(0.0),
      xi2_(0.0),
      xi1_old_(0.0),
      xi2_old_(0.0),
      pp_(0.0),
      fp_(0.0),
      dfp_(0.0),
      fd_(0.0),
      dfd_(0.0),
      d_(0.0),
      dd_(0.0),
      iter_(0),
      numstep_(0),
      dt_(0.0),
      beamendcontactopened_(false),
      beamsalmostparallel_(false),
      cppunconverged_(false),
      oldcppunconverged_(false),
      ele1length_(0.0),
      ele2length_(0.0),
      neighbornormalrequired_(false),
      tangentproduct_(0.0),
      radius1_(0.0),
      radius2_(0.0)
{
  for (int i = 0; i < 3; i++)
  {
    r1_(i) = 0.0;
    r2_(i) = 0.0;
    r1_old_(i) = 0.0;
    r2_old_(i) = 0.0;
    r1_xi_ = 0.0;
    r2_xi_ = 0.0;
    r1_xi_old_ = 0.0;
    r2_xi_old_ = 0.0;
    normal_(i) = 0.0;
    normal_old_(i) = 0.0;
  }
  for (int i = 0; i < 3 * numnodes * numnodalvalues; i++)
  {
    ele1pos_(i) = 0.0;
    ele2pos_(i) = 0.0;
    ele1pos_old_(i) = 0.0;
    ele2pos_old_(i) = 0.0;
    ele1pos_lastiter_(i) = 0.0;
    ele2pos_lastiter_(i) = 0.0;
  }
  for (int i = 0; i < 3 * numnodes; i++)
  {
    nodaltangentssmooth1_(i) = 0.0;
    nodaltangentssmooth2_(i) = 0.0;
  }

  smoothing_ = DRT::INPUT::IntegralValue<INPAR::BEAMCONTACT::Smoothing>(
      beamcontactparams, "BEAMS_SMOOTHING");

  const DRT::ElementType& eot1 = element1_->ElementType();

  if (smoothing_ == INPAR::BEAMCONTACT::bsm_cpp and eot1 != DRT::ELEMENTS::Beam3rType::Instance())
    dserror("Tangent smoothing only implemented for beams of type beam3r!");

  // For both elements the 2 direct neighbor elements are determined and saved in the
  // B3CNeighbor-Class variables neighbors1_ and neighbors2_. The neighbors are not only necessary
  // for tangent smoothing but also in order to determine the vector normalold_ of the neighbor,
  // which is needed to perform sliding contact (with changing active pairs) for slender beams.
  {
    neighbors1_ = BEAMINTERACTION::B3TANGENTSMOOTHING::DetermineNeigbors(element1);
    neighbors2_ = BEAMINTERACTION::B3TANGENTSMOOTHING::DetermineNeigbors(element2);
  }

  // Calculate initial length of beam elements (approximation for initially curved elements!)
  CORE::LINALG::Matrix<3, 1, double> lvec1(true);
  CORE::LINALG::Matrix<3, 1, double> lvec2(true);
  for (int i = 0; i < 3; i++)
  {
    lvec1(i) = (element1_->Nodes())[0]->X()[i] - (element1_->Nodes())[1]->X()[i];
    lvec2(i) = (element2_->Nodes())[0]->X()[i] - (element2_->Nodes())[1]->X()[i];
  }
  ele1length_ = lvec1.Norm2();
  ele2length_ = lvec2.Norm2();

  if (element1->ElementType() != element2->ElementType())
    dserror("The class beam3contact only works for contact pairs of the same beam element type!");

  if (element1->Id() >= element2->Id())
    dserror("Element 1 has to have the smaller element-ID. Adapt your contact search!");


  // get radius of elements
  const DRT::ELEMENTS::Beam3Base* beamele1 =
      static_cast<const DRT::ELEMENTS::Beam3Base*>(element1_);

  if (beamele1 == nullptr) dserror("cast to beam base failed!");

  radius1_ = MANIPULATERADIUS * beamele1->GetCircularCrossSectionRadiusForInteractions();

  const DRT::ELEMENTS::Beam3Base* beamele2 =
      static_cast<const DRT::ELEMENTS::Beam3Base*>(element2_);

  if (beamele2 == nullptr) dserror("cast to beam base failed!");

  radius2_ = MANIPULATERADIUS * beamele2->GetCircularCrossSectionRadiusForInteractions();


  if (DRT::INPUT::IntegralValue<INPAR::BEAMCONTACT::OctreeType>(
          beamcontactparams, "BEAMS_OCTREE") != INPAR::BEAMCONTACT::boct_none)
  {
    // TODO: Here we need a warning in case we have no additive bounding box extrusion value!
  }

  searchboxinc_ = BEAMINTERACTION::DetermineSearchboxInc(beamcontactparams);

  if (searchboxinc_ < 0.0)
    dserror("Choose a positive value for the searchbox extrusion factor BEAMS_EXTVAL!");

  if (DRT::INPUT::IntegralValue<int>(bcparams_, "BEAMS_NEWGAP") and
      !DRT::INPUT::IntegralValue<int>(beamcontactparams, "BEAMS_ADDITEXT"))
    dserror("New gap function only possible when the flag BEAMS_ADDITEXT is set true!");

  int penaltylaw = DRT::INPUT::IntegralValue<INPAR::BEAMCONTACT::PenaltyLaw>(
      beamcontactparams, "BEAMS_PENALTYLAW");
  if (penaltylaw != INPAR::BEAMCONTACT::pl_lp and penaltylaw != INPAR::BEAMCONTACT::pl_qp)
  {
    if (beamcontactparams.get<double>("BEAMS_PENREGPARAM_F0", -1.0) == -1.0 or
        beamcontactparams.get<double>("BEAMS_PENREGPARAM_G0", -1.0) == -1.0 or
        beamcontactparams.get<double>("BEAMS_PENREGPARAM_C0", -1.0) == -1.0)
      dserror("Regularized penalty law chosen, but not all regularization parameters are set!");
  }

  if (DRT::INPUT::IntegralValue<INPAR::BEAMCONTACT::Damping>(beamcontactparams, "BEAMS_DAMPING") !=
      INPAR::BEAMCONTACT::bd_no)
  {
    if (beamcontactparams.get<double>("BEAMS_DAMPINGPARAM", -1.0) == -1.0 or
        beamcontactparams.get<double>("BEAMS_DAMPREGPARAM1", -1.0) == -1.0 or
        beamcontactparams.get<double>("BEAMS_DAMPREGPARAM2", -1.0) == -1.0)
      dserror("Damping force chosen in input-file, but no damping (regularization) parameter!");
  }

  if (beamcontactparams.get<double>("BEAMS_GAPSHIFTPARAM", 0.0) != 0.0)
    dserror(
        "BEAMS_GAPSHIFTPARAM not implemented for beam3contactnew (input parameter "
        "BEAMS_SEGCON==No)!");

  return;
}
/*----------------------------------------------------------------------*
 |  end: constructor
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Evaluate the element (public)                             meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
bool CONTACT::Beam3contactnew<numnodes, numnodalvalues>::Evaluate(
    CORE::LINALG::SparseMatrix& stiffmatrix, Epetra_Vector& fint, const double& pp,
    std::map<std::pair<int, int>, Teuchos::RCP<Beam3contactinterface>>& contactpairmap,
    Teuchos::ParameterList& timeintparams, bool fdcheck)
{
  //**********************************************************************
  // Evaluation of contact forces and stiffness
  //**********************************************************************
  // (1) Closest Point Projection (CPP)
  //     -> find closest point where contact forces are evaluated
  // (2) Compute some auxiliary quantities
  //     -> normal vector, gap, shape functions, contact flag,
  //     -> linearizations of all geometric quantities
  // (3) Compute contact forces and stiffness
  //     -> stiffness terms are directly assembled to global matrix
  //     -> contact forces are only returned as global vector
  // (4) Perform some finite difference checks
  //     -> only if the flag BEAMCONTACTFDCHECKS is defined
  //***************Get some parameters in the beginning*******************

  // All updates that have to be done in every iteration have do be done here,
  // since most of the elements leave directly after the closest point projection!
  SetClassVariables(pp, timeintparams);

  //**********************************************************************
  // (1) Closest Point Projection (CPP)
  //**********************************************************************
  ClosestPointProjection();

// If the contact opens once at a boundary element, the contactflag_ is set to false for the whole
// Newton iteration
#ifdef CHECKBOUNDARYCONTACT
  CheckBoundaryContact();
#endif

  //**********************************************************************
  // (2) Compute some auxiliary quantities
  //**********************************************************************

  // vectors for shape functions and their derivatives
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N1(true);       // = N1
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N2(true);       // = N2
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N1_xi(true);    // = N1,xi
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N2_xi(true);    // = N2,eta
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N1_xixi(true);  // = N1,xixi
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N2_xixi(true);  // = N2,etaeta

  // coords and derivatives of the two contacting points
  CORE::LINALG::Matrix<3, 1, TYPE> r1(true);       // = r1
  CORE::LINALG::Matrix<3, 1, TYPE> r2(true);       // = r2
  CORE::LINALG::Matrix<3, 1, TYPE> r1_xi(true);    // = r1,xi
  CORE::LINALG::Matrix<3, 1, TYPE> r2_xi(true);    // = r2,eta
  CORE::LINALG::Matrix<3, 1, TYPE> r1_xixi(true);  // = r1,xixi
  CORE::LINALG::Matrix<3, 1, TYPE> r2_xixi(true);  // = r2,etaeta
  CORE::LINALG::Matrix<3, 1, TYPE> delta_r(true);  // = r1-r2
  TYPE norm_delta_r = 0.0;                         // = g

  // Check if the CPP found for this contact pair is really on the considered element, i.e. xi \in
  // [-1;1]
  if (CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(xi1_)) < (1.0 + XIETATOL) &&
      CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(xi2_)) < (1.0 + XIETATOL))
  {
    // update shape functions and their derivatives
    GetShapeFunctions(N1, N2, N1_xi, N2_xi, N1_xixi, N2_xixi, xi1_, xi2_);
    // update coordinates and derivatives of contact points
    ComputeCoordsAndDerivs(
        r1, r2, r1_xi, r2_xi, r1_xixi, r2_xixi, N1, N2, N1_xi, N2_xi, N1_xixi, N2_xixi);

    // update coordinates and derivatives of current contact points at last time step
    ComputeOldCoordsAndDerivs(r1_old_, r2_old_, r1_xi_old_, r2_xi_old_, N1, N2, N1_xi, N2_xi);

    tangentproduct_ = CORE::FADUTILS::Norm(CORE::FADUTILS::ScalarProduct(r1_xi, r2_xi)) /
                      (CORE::FADUTILS::VectorNorm<3>(r1_xi) * CORE::FADUTILS::VectorNorm<3>(r2_xi));

    // In Case, the contact happend on the neighbor element pair in the last time step, we have not
    // calculated normal_old for this element in the last time step. In this case, we take
    // normal_old from the neighbor element!
    if (fabs(xi1_old_) > 1.0 + XIETATOL or fabs(xi2_old_) > 1.0 + XIETATOL)
    {
      GetNeighborNormalOld(contactpairmap);
    }

    // call function to compute scaled normal and gap of contact point and store this quantities in
    // the corresponding class variables. The auxiliary variables delta_r and norm_delta_r might be
    // useful for later applications.
    ComputeNormal(delta_r, norm_delta_r, contactpairmap);

    // call function to evaluate contact status
    CheckContactStatus(pp);

    if (contactflag_ or dampingcontactflag_)
    {
      if (tangentproduct_ > PARALLEL_DEACTIVATION_VAL)
      {
        // For very small tangent angles the contact is not evaluated. This would lead to a bad
        // conditioned problem. Therefore, the problem of almost parallel beams have to be modeled
        // by an alternative contact approach
        beamsalmostparallel_ = true;
      }
      if (tangentproduct_ < PARALLEL_ACTIVATION_VAL)
      {
        // In order to avoid an oscillation between the status "beamsalmostparallel_ = true" and the
        // status "beamsalmostparallel_ = false" during the Newton iterations we have introduced a
        // gap between the value that deactivates contact and the value that activates contact.
        beamsalmostparallel_ = false;
      }
    }

    // If one beam has passed the end of the second beam the contact has to be opened
    if (beamendcontactopened_ or beamsalmostparallel_)
    {
      contactflag_ = false;
      dampingcontactflag_ = false;
    }
  }
  else
  {
    contactflag_ = false;
    dampingcontactflag_ = false;
    // Iterative Update of class variables
    UpdateClassVariablesIter();

    return (false);
  }

  //**********************************************************************
  // (3) Compute contact forces and stiffness
  //**********************************************************************

  // Set class variables fp_ and dfp_ for scalar penalty force and linearization
  CalcPenaltyLaw();

  // Set class variables fd_, dfd_, d_ and dd_ for scalar damping force = d_*fd_ and linearization
  CalcDampingLaw();

  //  //Debugging output
  //  std::cout << "Pair: " << element1_->Id() << " / " << element2_->Id() << std::endl;
  //  std::cout << "xi1: " << xi1_ << "xi2: " << xi2_ << std::endl;
  //  std::cout << "normal_: " << normal_ << std::endl;
  //  std::cout << "normal_old_: " << normal_old_ << std::endl;
  //  std::cout << "gap_: " << gap_ << std::endl;
  //  std::cout << "gap_original_: " << gap_original_ << std::endl;
  //  std::cout << "beamendcontactopened_: " << beamendcontactopened_ << std::endl;
  //  std::cout << "radius1_: " << radius1_ << std::endl;
  //  std::cout << "radius2_: " << radius2_ << std::endl;
  //  std::cout << "iter_: " << iter_ << std::endl;

  // call function to evaluate and assemble contact forces
  EvaluateFcContact(pp, &fint, N1, N2);
  // call function to evaluate and assemble contact stiffness
  EvaluateStiffcContact(pp, norm_delta_r, delta_r, stiffmatrix, r1, r2, r1_xi, r2_xi, r1_xixi,
      r2_xixi, N1, N2, N1_xi, N2_xi, N1_xixi, N2_xixi);

// Apply algorithmic contact forces and stiffnesses that may improve the convergence behavior but
// will not change the physical results!
#if defined(ALGORITHMICDAMP) or defined(BEAMCONTACTPTC) or defined(BASICSTIFFWEIGHT)
  EvaluateAlgorithmicForce(pp, &fint, N1, N2);
  EvaluateAlgorithmicStiff(pp, norm_delta_r, delta_r, stiffmatrix, r1, r2, r1_xi, r2_xi, r1_xixi,
      r2_xixi, N1, N2, N1_xi, N2_xi, N1_xixi, N2_xixi);
#endif

  // Iterative Update of class variables
  UpdateClassVariablesIter();

  return (true);
}
/*----------------------------------------------------------------------*
 |  end: Evaluate the element
  *---------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Compute contact forces                                   meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::EvaluateFcContact(const double& pp,
    Epetra_Vector* fint, const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
    CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, TYPE>* fc1_FAD,
    CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, TYPE>* fc2_FAD)
{
  // get dimensions for vectors fc1 and fc2
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodes * numnodalvalues;

  // temporary vectors for contact forces, DOF-GIDs and owning procs
  CORE::LINALG::Matrix<dim1, 1, TYPE> fc1(true);
  CORE::LINALG::Matrix<dim2, 1, TYPE> fc2(true);
  CORE::LINALG::SerialDenseVector fcontact1(dim1);
  CORE::LINALG::SerialDenseVector fcontact2(dim2);
  std::vector<int> lm1(dim1);
  std::vector<int> lm2(dim2);
  std::vector<int> lmowner1(dim1);
  std::vector<int> lmowner2(dim2);

  // flag indicating assembly
  bool DoNotAssemble = true;

  // node ids of both elements
  const int* node_ids1 = element1_->NodeIds();
  const int* node_ids2 = element2_->NodeIds();

  for (int i = 0; i < numnodes; ++i)
  {
    // get node pointer and dof ids
    DRT::Node* node = ContactDiscret().gNode(node_ids1[i]);
    std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

    // compute force vector Fc1 and prepare assembly
    for (int j = 0; j < 3 * numnodalvalues; ++j)
    {
      lm1[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
      lmowner1[3 * numnodalvalues * i + j] = node->Owner();
    }
  }

  for (int i = 0; i < numnodes; ++i)
  {
    // get node pointer and dof ids
    DRT::Node* node = ContactDiscret().gNode(node_ids2[i]);
    std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

    // compute force vector Fc1 and prepare assembly
    for (int j = 0; j < 3 * numnodalvalues; ++j)
    {
      lm2[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
      lmowner2[3 * numnodalvalues * i + j] = node->Owner();
    }
  }

  //**********************************************************************
  // evaluate contact forces for active pairs
  //**********************************************************************
  if (contactflag_)
  {
    DoNotAssemble = false;
    //********************************************************************
    // Compute Fc1 (force acting on first element)
    //********************************************************************
    for (int i = 0; i < dim1; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        fc1(i) += sgn_ * N1(j, i) * normal_(j) * fp_;
      }
    }

    //********************************************************************
    // Compute Fc2 (force acting on second element)
    //********************************************************************
    for (int i = 0; i < dim2; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        fc2(i) += -sgn_ * N2(j, i) * normal_(j) * fp_;
      }
    }
  }

  //**********************************************************************
  // evaluate damping forces for active pairs
  //**********************************************************************
  if (DRT::INPUT::IntegralValue<int>(bcparams_, "BEAMS_DAMPING") != INPAR::BEAMCONTACT::bd_no and
      dampingcontactflag_)
  {
    DoNotAssemble = false;
    //********************************************************************
    // Compute Fd1 (force acting on first element)
    //********************************************************************
    for (int i = 0; i < dim1; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        fc1(i) += N1(j, i) * normal_(j) * d_ * fd_;
      }
    }
    //********************************************************************
    // Compute Fd2 (force acting on second element)
    //********************************************************************
    for (int i = 0; i < dim2; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        fc2(i) += -N2(j, i) * normal_(j) * d_ * fd_;
      }
    }
  }

// Quantities necessary for automatic differentiation
#ifdef AUTOMATICDIFF
  if (fc1_FAD != nullptr and fc2_FAD != nullptr)
  {
    for (int i = 0; i < dim1; ++i)
    {
      (*fc1_FAD)(i) = fc1(i);
    }
    for (int i = 0; i < dim2; ++i)
    {
      (*fc2_FAD)(i) = fc2(i);
    }
  }
#endif

  //**********************************************************************
  // assemble contact forces
  //**********************************************************************
  if (!DoNotAssemble and fint != nullptr)
  {
    for (int i = 0; i < dim1; ++i)
    {
      fcontact1[i] = CORE::FADUTILS::CastToDouble(fc1(i));
    }
    for (int i = 0; i < dim2; ++i)
    {
      fcontact2[i] = CORE::FADUTILS::CastToDouble(fc2(i));
    }
    // assemble fc1 and fc2 into global contact force vector
    CORE::LINALG::Assemble(*fint, fcontact1, lm1, lmowner1);
    CORE::LINALG::Assemble(*fint, fcontact2, lm2, lmowner2);
  }

  return;
}
/*----------------------------------------------------------------------*
 |  end: Compute contact forces
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Evaluate contact stiffness                               meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::EvaluateStiffcContact(const double& pp,
    const TYPE& norm_delta_r, const CORE::LINALG::Matrix<3, 1, TYPE>& delta_r,
    CORE::LINALG::SparseMatrix& stiffmatrix, const CORE::LINALG::Matrix<3, 1, TYPE>& r1,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xixi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xixi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xixi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xixi)
{
  // get dimensions for vectors fc1 and fc2
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodes * numnodalvalues;

  // temporary matrices for stiffness and vectors for DOF-GIDs and owning procs
  CORE::LINALG::Matrix<dim1, dim1 + dim2, TYPE> stiffc1(true);
  CORE::LINALG::Matrix<dim2, dim1 + dim2, TYPE> stiffc2(true);
  CORE::LINALG::Matrix<dim1, dim1 + dim2, TYPE> stiffc1_FAD(true);
  CORE::LINALG::Matrix<dim2, dim1 + dim2, TYPE> stiffc2_FAD(true);
  CORE::LINALG::SerialDenseMatrix stiffcontact1(dim1, dim1 + dim2);
  CORE::LINALG::SerialDenseMatrix stiffcontact2(dim2, dim1 + dim2);
  std::vector<int> lmrow1(dim1);
  std::vector<int> lmrow2(dim2);
  std::vector<int> lmrowowner1(dim1);
  std::vector<int> lmrowowner2(dim2);
  std::vector<int> lmcol1(dim1 + dim2);
  std::vector<int> lmcol2(dim1 + dim2);

  // flag indicating assembly
  bool DoNotAssemble = true;

  // if the bool inactivestiff is true, the contact stiffness will always be applied in the first
  // Newton steps for pair which have been active in the last time step (even when they are
  // currentely not active) -> This makes the algorithm more robust.
  bool inactivestiff = DRT::INPUT::IntegralValue<int>(bcparams_, "BEAMS_INACTIVESTIFF");

  // In order to accelerate convergence, we only apply the basic stiffness part in case of very
  // large gaps!
  double basicstiffgap = bcparams_.get<double>("BEAMS_BASICSTIFFGAP", -1.0);
  bool completestiff = true;
  if (basicstiffgap != -1.0)
  {
    if (basicstiffgap < 0.0)
      dserror("The parameter BEAMS_BASICSTIFFGAP has to be positive!");
    else if (gap_ < -1.0 * basicstiffgap)
    {
      completestiff = false;
    }
  }

  // Apply additional weighting of the basic stiffness term e.g. in the first iterations or when
  // the Newton scheme oscillates (no convergence after a certain number of iterations)
  double basicstiffweightfac = 1.0;
#ifdef BASICSTIFFWEIGHT
  if (iter_ < 5)
  {
    basicstiffweightfac = BASICSTIFFWEIGHT;
  }
#endif

  //**********************************************************************
  // evaluate contact stiffness for active pairs
  //**********************************************************************
  if (contactflag_ or
      (iter_ == 0 and inactivestiff and oldcontactflag_ and !beamendcontactopened_ and
          !beamsalmostparallel_) or
      dampingcontactflag_)
  {
    // node ids of both elements
    const int* node_ids1 = element1_->NodeIds();
    const int* node_ids2 = element2_->NodeIds();

    // initialize storage for linearizations
    CORE::LINALG::Matrix<dim1 + dim2, 1, TYPE> delta_xi(true);
    CORE::LINALG::Matrix<dim1 + dim2, 1, TYPE> delta_eta(true);
    CORE::LINALG::Matrix<dim1 + dim2, 1, TYPE> delta_gap(true);
    CORE::LINALG::Matrix<dim1 + dim2, 1, TYPE> delta_gap_t(true);
    CORE::LINALG::Matrix<3, dim1 + dim2, TYPE> delta_x1_minus_x2(true);
    CORE::LINALG::Matrix<3, dim1 + dim2, TYPE> delta_n(true);

    //********************************************************************
    // evaluate linearizations and distance
    //********************************************************************
    // linearization of contact point
    ComputeLinXiAndLinEta(
        delta_xi, delta_eta, delta_r, r1_xi, r2_xi, r1_xixi, r2_xixi, N1, N2, N1_xi, N2_xi);

#ifdef FADCHECKS
    //    std::cout << "delta_xi: " << std::endl;
    //      for (int i=0;i<dim1+dim2;i++)
    //        std::cout << delta_xi(i).val() << "  ";
    //      std::cout << std::endl << "delta_eta: " << std::endl;
    //      for (int i=0;i<dim1+dim2;i++)
    //        std::cout << delta_eta(i).val() << "  ";
    //      std::cout << std::endl;
    //      FADCheckLinXiAndLinEta(delta_r,r1_xi,r2_xi,r1_xixi,r2_xixi,N1,N2,N1_xi,N2_xi);
#endif

    // linearization of gap function which is equal to delta d
    ComputeLinGap(delta_gap, delta_xi, delta_eta, delta_r, norm_delta_r, r1_xi, r2_xi, N1, N2);

    // linearization of normal vector
    ComputeLinNormal(delta_n, delta_xi, delta_eta, norm_delta_r, r1_xi, r2_xi, N1, N2);

    // linearization of gap function which is equal to delta d
    ComputeLinGapt(delta_gap_t, delta_xi, delta_eta, delta_n, r1_xi, r2_xi, N1, N2, r1_old_,
        r2_old_, r1_xi_old_, r2_xi_old_);

    //********************************************************************
    // prepare assembly
    //********************************************************************
    // fill lmrow1 and lmrowowner1
    for (int i = 0; i < numnodes; ++i)
    {
      // get pointer and dof ids
      DRT::Node* node = ContactDiscret().gNode(node_ids1[i]);
      std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

      for (int j = 0; j < 3 * numnodalvalues; ++j)
      {
        lmrow1[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
        lmrowowner1[3 * numnodalvalues * i + j] = node->Owner();
      }
    }

    // fill lmrow2 and lmrowowner2
    for (int i = 0; i < numnodes; ++i)
    {
      // get pointer and node ids
      DRT::Node* node = ContactDiscret().gNode(node_ids2[i]);
      std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

      for (int j = 0; j < 3 * numnodalvalues; ++j)
      {
        lmrow2[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
        lmrowowner2[3 * numnodalvalues * i + j] = node->Owner();
      }
    }

    // fill lmcol1 and lmcol2
    for (int i = 0; i < numnodes; ++i)
    {
      // get pointer and node ids
      DRT::Node* node = ContactDiscret().gNode(node_ids1[i]);
      std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

      for (int j = 0; j < 3 * numnodalvalues; ++j)
      {
        lmcol1[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
        lmcol2[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
      }
    }

    // fill lmcol1 and lmcol2
    for (int i = 0; i < numnodes; ++i)
    {
      // get pointer and node ids
      DRT::Node* node = ContactDiscret().gNode(node_ids2[i]);
      std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

      for (int j = 0; j < 3 * numnodalvalues; ++j)
      {
        lmcol1[3 * numnodalvalues * numnodes + 3 * numnodalvalues * i + j] = NodeDofGIDs[j];
        lmcol2[3 * numnodalvalues * numnodes + 3 * numnodalvalues * i + j] = NodeDofGIDs[j];
      }
    }

    //*************Begin of standard linearization of penalty contact forces**************
    // The full contact stiffness is only applied if the contact flag is true
    // and gap_ > -BEAMS_BASICSTIFFGAP. If gap_ < -BEAMS_BASICSTIFFGAP, only the basic
    // stiffness is applied. If incactivestiff==true, the basic stiffness part is also
    // applied to inactive pairs in the first Newton step of a time step, if these
    // pairs were active in the converged configuration of the last time step. This
    // makes the contact more robust.
    if (contactflag_ or (iter_ == 0 and inactivestiff and oldcontactflag_ and
                            !beamendcontactopened_ and !beamsalmostparallel_))
    {
      DoNotAssemble = false;

      //********************************************************************
      // evaluate contact stiffness
      // (1) stiffc1 of first element
      //********************************************************************

      //********************************************************************
      // part I - basic stiffness
      //********************************************************************
      CORE::LINALG::Matrix<dim1, 1, TYPE> N1T_normal(true);
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < dim1; j++)
        {
          N1T_normal(j) += N1(i, j) * normal_(i);
        }
      }
      for (int i = 0; i < dim1; i++)
      {
        for (int j = 0; j < dim1 + dim2; j++)
        {
          stiffc1(i, j) += basicstiffweightfac * sgn_ * dfp_ * N1T_normal(i) * delta_gap(j);
        }
      }

      // The geoemtric part is only applied for gap_ < -BEAMS_BASICSTIFFGAP and when the contact is
      // really active (not for the "inactivestiff"-case)
      if (completestiff and contactflag_)
      {
        //********************************************************************
        // part II - geometric stiffness 1
        //********************************************************************
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < dim1; j++)
          {
            for (int k = 0; k < dim1 + dim2; k++)
            {
              stiffc1(j, k) += sgn_ * fp_ * N1(i, j) * delta_n(i, k);
            }
          }
        }
        //********************************************************************
        // part III - geometric stiffness 2
        //********************************************************************
        CORE::LINALG::Matrix<dim1, 1, TYPE> N1xiT_normal(true);
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < dim1; j++)
          {
            N1xiT_normal(j) += N1_xi(i, j) * normal_(i);
          }
        }

        for (int i = 0; i < dim1; i++)
        {
          for (int j = 0; j < dim1 + dim2; j++)
          {
            stiffc1(i, j) += sgn_ * fp_ * N1xiT_normal(i) * delta_xi(j);
          }
        }
      }
      //********************************************************************
      // evaluate contact stiffness
      // (2) stiffc2 of second element
      //********************************************************************

      //********************************************************************
      // part I
      //********************************************************************
      CORE::LINALG::Matrix<dim2, 1, TYPE> N2T_normal(true);
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < dim2; j++)
        {
          N2T_normal(j) += N2(i, j) * normal_(i);
        }
      }
      for (int i = 0; i < dim2; i++)
      {
        for (int j = 0; j < dim1 + dim2; j++)
        {
          stiffc2(i, j) += -basicstiffweightfac * sgn_ * dfp_ * N2T_normal(i) * delta_gap(j);
        }
      }

      if (completestiff and contactflag_)
      {
        //********************************************************************
        // part II
        //********************************************************************
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < dim2; j++)
          {
            for (int k = 0; k < dim1 + dim2; k++)
            {
              stiffc2(j, k) += -sgn_ * fp_ * N2(i, j) * delta_n(i, k);
            }
          }
        }
        //********************************************************************
        // part III
        //********************************************************************
        CORE::LINALG::Matrix<dim1, 1, TYPE> N2xiT_normal(true);
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < dim2; j++)
          {
            N2xiT_normal(j) += N2_xi(i, j) * normal_(i);
          }
        }

        for (int i = 0; i < dim2; i++)
        {
          for (int j = 0; j < dim1 + dim2; j++)
          {
            stiffc2(i, j) += -sgn_ * fp_ * N2xiT_normal(i) * delta_eta(j);
          }
        }
      }
    }
    //*************End of standard linearization of penalty contact forces****************

    //*************Begin of standard linearization of damping contact forces**************
    if (DRT::INPUT::IntegralValue<int>(bcparams_, "BEAMS_DAMPING") != INPAR::BEAMCONTACT::bd_no and
        dampingcontactflag_)
    {
      //*************Begin of standard linearization**************************
      DoNotAssemble = false;

      //********************************************************************
      // evaluate damping stiffness
      // (1) stiffc1 of first element
      //********************************************************************

      //********************************************************************
      // part I
      //********************************************************************
      CORE::LINALG::Matrix<dim1, 1, TYPE> N1T_normal(true);
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < dim1; j++)
        {
          N1T_normal(j) += N1(i, j) * normal_(i);
        }
      }
      for (int i = 0; i < dim1; i++)
      {
        for (int j = 0; j < dim1 + dim2; j++)
        {
          stiffc1(i, j) += N1T_normal(i) * (d_ * dfd_ * delta_gap_t(j) + dd_ * fd_ * delta_gap(j));
        }
      }

      if (completestiff)
      {
        //********************************************************************
        // part II
        //********************************************************************
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < dim1; j++)
          {
            for (int k = 0; k < dim1 + dim2; k++)
            {
              stiffc1(j, k) += d_ * fd_ * N1(i, j) * delta_n(i, k);
            }
          }
        }
        //********************************************************************
        // part III
        //********************************************************************
        CORE::LINALG::Matrix<dim1, 1, TYPE> N1xiT_normal(true);
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < dim1; j++)
          {
            N1xiT_normal(j) += N1_xi(i, j) * normal_(i);
          }
        }
        for (int i = 0; i < dim1; i++)
        {
          for (int j = 0; j < dim1 + dim2; j++)
          {
            stiffc1(i, j) += d_ * fd_ * N1xiT_normal(i) * delta_xi(j);
          }
        }
      }
      //********************************************************************
      // evaluate damping stiffness
      // (2) stiffc2 of second element
      //********************************************************************

      //********************************************************************
      // part I
      //********************************************************************
      CORE::LINALG::Matrix<dim2, 1, TYPE> N2T_normal(true);
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < dim2; j++)
        {
          N2T_normal(j) += N2(i, j) * normal_(i);
        }
      }
      for (int i = 0; i < dim1; i++)
      {
        for (int j = 0; j < dim1 + dim2; j++)
        {
          stiffc2(i, j) += -N2T_normal(i) * (d_ * dfd_ * delta_gap_t(j) + dd_ * fd_ * delta_gap(j));
        }
      }

      if (completestiff)
      {
        //********************************************************************
        // part II
        //********************************************************************
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < dim2; j++)
          {
            for (int k = 0; k < dim1 + dim2; k++)
            {
              stiffc2(j, k) += -d_ * fd_ * N2(i, j) * delta_n(i, k);
            }
          }
        }
        //********************************************************************
        // part III
        //********************************************************************
        CORE::LINALG::Matrix<dim1, 1, TYPE> N2xiT_normal(true);
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < dim2; j++)
          {
            N2xiT_normal(j) += N2_xi(i, j) * normal_(i);
          }
        }
        for (int i = 0; i < dim2; i++)
        {
          for (int j = 0; j < dim1 + dim2; j++)
          {
            stiffc2(i, j) += -d_ * fd_ * N2xiT_normal(i) * delta_eta(j);
          }
        }
      }
    }
//***************End of standard linearization of damping contact forces**************

// automatic differentiation for debugging
#ifdef AUTOMATICDIFF
    CORE::LINALG::Matrix<dim1, 1, TYPE> fc1_FAD(true);
    CORE::LINALG::Matrix<dim2, 1, TYPE> fc2_FAD(true);

    EvaluateFcContact(pp, nullptr, N1, N2, &fc1_FAD, &fc2_FAD);
    for (int j = 0; j < dim1 + dim2; j++)
    {
      for (int i = 0; i < dim1; i++)
        stiffc1_FAD(i, j) = (fc1_FAD(i).dx(j) + fc1_FAD(i).dx(dim1 + dim2) * delta_xi(j) +
                             fc1_FAD(i).dx(dim1 + dim2 + 1) * delta_eta(j));
      for (int i = 0; i < dim2; i++)
        stiffc2_FAD(i, j) = (fc2_FAD(i).dx(j) + fc2_FAD(i).dx(dim1 + dim2) * delta_xi(j) +
                             fc2_FAD(i).dx(dim1 + dim2 + 1) * delta_eta(j));
    }

    //      std::cout << "BTB Contact Pair: " << element1_->Id() << " / " << element2_->Id() <<
    //      std::endl;
    //
    //      std::cout << "stiffc1: " << std::endl;
    //      for (int i=0;i<dim1;i++)
    //      {
    //        for (int j=0;j<dim1+dim2;j++)
    //        {
    //          std::cout << std::setw(14) << stiffc1(i,j).val() << " ";
    //        }
    //        std::cout << std::endl;
    //      }
    //      std::cout << std::endl;
    //      std::cout << "stiffc1_FAD: " << std::endl;
    //      for (int i=0;i<dim1;i++)
    //      {
    //        for (int j=0;j<dim1+dim2;j++)
    //        {
    //          std::cout << std::setw(14) << stiffc1_FAD(i,j).val() << " ";
    //        }
    //        std::cout << std::endl;
    //      }
    //      std::cout << std::endl;
    ////
    ////      std::cout << "d fc1_/ d xi: " << std::endl;
    ////      for (int i=0;i<dim1;i++)
    ////      {
    ////        std::cout << std::setw(14) << fc1_FAD(i).dx(dim1+dim2);
    ////      }
    ////      std::cout << std::endl;
    //
    //      std::cout << "stiffc2: " << std::endl;
    //      for (int i=0;i<dim2;i++)
    //      {
    //        for (int j=0;j<dim1+dim2;j++)
    //        {
    //          std::cout << std::setw(14) << stiffc2(i,j).val() << " ";
    //        }
    //        std::cout << std::endl;
    //      }
    //      std::cout << std::endl;
    //      std::cout << "stiffc2_FAD: " << std::endl;
    //      for (int i=0;i<dim2;i++)
    //      {
    //        for (int j=0;j<dim1+dim2;j++)
    //        {
    //          std::cout << std::setw(14) << stiffc2_FAD(i,j).val() << " ";
    //        }
    //        std::cout << std::endl;
    //      }
    //      std::cout << std::endl;
    ////
    ////      std::cout << "d fc2_/ d xi: " << std::endl;
    ////      for (int i=0;i<dim2;i++)
    ////      {
    ////        std::cout << std::setw(14) << fc2_FAD(i).dx(dim1+dim2);
    ////      }
    ////      std::cout << std::endl;
#endif
  }  // if (contactflag_ or (iter_==0 and inactivestiff and oldcontactflag_ and
     // !beamendcontactopened_ and !beamsalmostparallel_) or dampingcontactflag_)

  //**********************************************************************
  // assemble contact stiffness
  //**********************************************************************
  // change sign of stiffc1 and stiffc2 due to time integration.
  // according to analytical derivation there is no minus sign, but for
  // our time integration methods the negative stiffness must be assembled.

  // now finally assemble stiffc1 and stiffc2
  if (!DoNotAssemble)
  {
#ifndef AUTOMATICDIFF
    for (int j = 0; j < dim1 + dim2; j++)
    {
      for (int i = 0; i < dim1; i++)
        stiffcontact1(i, j) = -CORE::FADUTILS::CastToDouble(stiffc1(i, j));
      for (int i = 0; i < dim2; i++)
        stiffcontact2(i, j) = -CORE::FADUTILS::CastToDouble(stiffc2(i, j));
    }
#else
    for (int j = 0; j < dim1 + dim2; j++)
    {
      for (int i = 0; i < dim1; i++)
        stiffcontact1(i, j) = -CORE::FADUTILS::CastToDouble(stiffc1_FAD(i, j));
      for (int i = 0; i < dim2; i++)
        stiffcontact2(i, j) = -CORE::FADUTILS::CastToDouble(stiffc2_FAD(i, j));
    }
#endif

    stiffmatrix.Assemble(0, stiffcontact1, lmrow1, lmrowowner1, lmcol1);
    stiffmatrix.Assemble(0, stiffcontact2, lmrow2, lmrowowner2, lmcol2);
  }

  return;
}
/*----------------------------------------------------------------------*
 |  end: Evaluate contact stiffness
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Compute algorithmic forces                               meier 08/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::EvaluateAlgorithmicForce(const double& pp,
    Epetra_Vector* fint, const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
    CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, TYPE>* fc1_FAD,
    CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, TYPE>* fc2_FAD)
{
#ifdef ALGORITHMICDAMP
  CORE::LINALG::Matrix<3, 1, TYPE> r1_lastiter(true);
  CORE::LINALG::Matrix<3, 1, TYPE> r2_lastiter(true);
  CORE::LINALG::Matrix<3, 1, TYPE> vc1(true);
  CORE::LINALG::Matrix<3, 1, TYPE> vc2(true);
  TYPE fd = 0.0;
  TYPE d = 0.0;

  // compute output variable
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3 * numnodes * numnodalvalues; j++)
    {
      r1_lastiter(i) += N1(i, j) * ele1pos_lastiter_(j);
      r2_lastiter(i) += N2(i, j) * ele2pos_lastiter_(j);
    }
  }

  if (firstcallofstep_)
  {
    // In the first time step the pair was found, we have no history information and can't calculate
    // velocities. No damping forces can be applied then. However, it is assumed that in the first
    // time step the pair is found by the contact search, it will not be active immediately and thus
    // no damping force is needed. If this happens anyway, an error is thrown in the method
    // CheckContactStatus().
    for (int i = 0; i < 3; i++)
    {
      vc1(i) = 0.0;
      vc2(i) = 0.0;
    }
  }
  else
  {
    // Attention: The velocities vc1 and vc2 are not the total contact point velocities, since they
    // do not contain the velocity contribution due to the change in xi and eta. However, since
    // these contributions are perpendicular on the normal_ vector, they are not needed in order to
    // calculate g_t (this case is similar to the calculation of the gap variation).
    for (int i = 0; i < 3; i++)
    {
      vc1(i) = (r1_(i) - r1_lastiter(i)) / dt_;
      vc2(i) = (r2_(i) - r2_lastiter(i)) / dt_;
    }
  }

  double d0 = ALGORITHMICDAMP;
  double gd1 = ALGDAMPREGFAC1;
  double gd2 = ALGDAMPREGFAC2;

  if (currentlyincontact_)
  {
    TYPE g_t =
        CORE::FADUTILS::ScalarProduct(normal_, vc1) - CORE::FADUTILS::ScalarProduct(normal_, vc2);
    fd = -g_t;

    if (fabs(gd1 - gd2) < DAMPTOL)
    {
      if (gap_ > gd1)
      {
        d = 0.0;
      }
      else
      {
        d = d0;
      }
    }
    else
    {
      if (gap_ > gd1)
      {
        d = 0.0;
      }
      else if (gap_ > gd2)
      {
        d = d0 / 2 * (1 - cos((gap_ - gd1) / (gd2 - gd1) * M_PI));
      }
      else
      {
        d = d0;
      }
    }
  }
  else
  {
    fd = 0.0;
    d = 0.0;
  }
#endif

  // get dimensions for vectors fc1 and fc2
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodes * numnodalvalues;

  // temporary vectors for contact forces, DOF-GIDs and owning procs
  CORE::LINALG::Matrix<dim1, 1, TYPE> fc1(true);
  CORE::LINALG::Matrix<dim2, 1, TYPE> fc2(true);
  CORE::LINALG::SerialDenseVector fcontact1(dim1);
  CORE::LINALG::SerialDenseVector fcontact2(dim2);
  std::vector<int> lm1(dim1);
  std::vector<int> lm2(dim2);
  std::vector<int> lmowner1(dim1);
  std::vector<int> lmowner2(dim2);

  // flag indicating assembly
  bool DoNotAssemble = false;

  // node ids of both elements
  const int* node_ids1 = element1_->NodeIds();
  const int* node_ids2 = element2_->NodeIds();

  for (int i = 0; i < numnodes; ++i)
  {
    // get node pointer and dof ids
    DRT::Node* node = ContactDiscret().gNode(node_ids1[i]);
    std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

    // compute force vector Fc1 and prepare assembly
    for (int j = 0; j < 3 * numnodalvalues; ++j)
    {
      lm1[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
      lmowner1[3 * numnodalvalues * i + j] = node->Owner();
    }
  }

  for (int i = 0; i < numnodes; ++i)
  {
    // get node pointer and dof ids
    DRT::Node* node = ContactDiscret().gNode(node_ids2[i]);
    std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

    // compute force vector Fc1 and prepare assembly
    for (int j = 0; j < 3 * numnodalvalues; ++j)
    {
      lm2[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
      lmowner2[3 * numnodalvalues * i + j] = node->Owner();
    }
  }

  //**********************************************************************
  // evaluate contact forces for active pairs
  //**********************************************************************
  if (currentlyincontact_ and iter_ > ITERMAX)
  {
#ifdef ALGORITHMICDAMP
    //********************************************************************
    // Compute Fd1 (force acting on first element)
    //********************************************************************
    for (int i = 0; i < dim1; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        fc1(i) += N1(j, i) * normal_(j) * d * fd;
      }
    }
    //********************************************************************
    // Compute Fd2 (force acting on second element)
    //********************************************************************
    for (int i = 0; i < dim2; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        fc2(i) += -N2(j, i) * normal_(j) * d * fd;
      }
    }
#endif
  }

#ifdef AUTOMATICDIFF
  if (fc1_FAD != nullptr and fc2_FAD != nullptr)
  {
    for (int i = 0; i < dim1; ++i)
    {
      (*fc1_FAD)(i) = fc1(i);
    }
    for (int i = 0; i < dim2; ++i)
    {
      (*fc2_FAD)(i) = fc2(i);
    }
  }
#endif

  //**********************************************************************
  // assemble contact forces
  //**********************************************************************
  if (!DoNotAssemble and fint != nullptr)
  {
    for (int i = 0; i < dim1; ++i)
    {
      fcontact1[i] = CORE::FADUTILS::CastToDouble(fc1(i));
    }
    for (int i = 0; i < dim2; ++i)
    {
      fcontact2[i] = CORE::FADUTILS::CastToDouble(fc2(i));
    }
    // assemble fc1 and fc2 into global contact force vector
    CORE::LINALG::Assemble(*fint, fcontact1, lm1, lmowner1);
    CORE::LINALG::Assemble(*fint, fcontact2, lm2, lmowner2);
  }

  return;
}
/*----------------------------------------------------------------------*
 |  end: Compute algorithmic forces
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Evaluate algorithmic stiffness                           meier 08/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::EvaluateAlgorithmicStiff(const double& pp,
    const TYPE& norm_delta_r, const CORE::LINALG::Matrix<3, 1, TYPE>& delta_r,
    CORE::LINALG::SparseMatrix& stiffmatrix, const CORE::LINALG::Matrix<3, 1, TYPE>& r1,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xixi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xixi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xixi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xixi)
{
  CORE::LINALG::Matrix<3, 1, TYPE> r1_lastiter(true);
  CORE::LINALG::Matrix<3, 1, TYPE> r2_lastiter(true);
  CORE::LINALG::Matrix<3, 1, TYPE> r1_xi_lastiter(true);
  CORE::LINALG::Matrix<3, 1, TYPE> r2_xi_lastiter(true);
  CORE::LINALG::Matrix<3, 1, TYPE> vc1(true);
  CORE::LINALG::Matrix<3, 1, TYPE> vc2(true);
  CORE::LINALG::Matrix<3, 1, TYPE> vc1_xi(true);
  CORE::LINALG::Matrix<3, 1, TYPE> vc2_xi(true);
  TYPE fd = 0.0;
  TYPE dfd = 0.0;
  TYPE d = 0.0;
  TYPE dd = 0.0;
  double algdampbasicstifffac = 1.0;

#ifdef ALGORITHMICDAMP
  // compute output variable
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3 * numnodes * numnodalvalues; j++)
    {
      r1_lastiter(i) += N1(i, j) * ele1pos_lastiter_(j);
      r2_lastiter(i) += N2(i, j) * ele2pos_lastiter_(j);
      r1_xi_lastiter(i) += N1_xi(i, j) * ele1pos_lastiter_(j);
      r2_xi_lastiter(i) += N2_xi(i, j) * ele2pos_lastiter_(j);
    }
  }

  if (firstcallofstep_)
  {
    // In the first time step the pair was found, we have no history information and can't calculate
    // velocities. No damping forces can be applied then. However, it is assumed that in the first
    // time step the pair is found by the contact search, it will not be active immediately and thus
    // no damping force is needed. If this happens anyway, an error is thrown in the method
    // CheckContactStatus().
    for (int i = 0; i < 3; i++)
    {
      vc1(i) = 0.0;
      vc2(i) = 0.0;
    }
  }
  else
  {
    // Attention: The velocities vc1 and vc2 are not the total contact point velocities, since they
    // do not contain the velocity contribution due to the change in xi and eta. However, since
    // these contributions are perpendicular on the normal_ vector, they are not needed in order to
    // calculate g_t (this case is similar to the calculation of the gap variation).
    for (int i = 0; i < 3; i++)
    {
      vc1(i) = (r1_(i) - r1_lastiter(i)) / dt_;
      vc2(i) = (r2_(i) - r2_lastiter(i)) / dt_;
    }
  }

  double d0 = ALGORITHMICDAMP;
  double gd1 = ALGDAMPREGFAC1;
  double gd2 = ALGDAMPREGFAC2;

  if (currentlyincontact_)
  {
    TYPE g_t =
        CORE::FADUTILS::ScalarProduct(normal_, vc1) - CORE::FADUTILS::ScalarProduct(normal_, vc2);
    fd = -g_t;
    dfd = -1.0;

    if (fabs(gd1 - gd2) < DAMPTOL)
    {
      if (gap_ > gd1)
      {
        d = 0.0;
        dd = 0.0;
      }
      else
      {
        d = d0;
        dd = 0.0;
      }
    }
    else
    {
      if (gap_ > gd1)
      {
        d = 0.0;
        dd = 0.0;
      }
      else if (gap_ > gd2)
      {
        d = d0 / 2 * (1 - cos((gap_ - gd1) / (gd2 - gd1) * M_PI));
        dd = d0 * M_PI / (2 * (gd2 - gd1)) * sin((gap_ - gd1) / (gd2 - gd1) * M_PI);
      }
      else
      {
        d = d0;
        dd = 0.0;
      }
    }
  }
  else
  {
    fd = 0.0;
    dfd = 0.0;
    d = 0.0;
    dd = 0.0;
  }
  algdampbasicstifffac = ALGDAMPBASICSTIFFFAC;
#endif

  // get dimensions for vectors fc1 and fc2
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodes * numnodalvalues;

  // temporary matrices for stiffness and vectors for DOF-GIDs and owning procs
  CORE::LINALG::Matrix<dim1, dim1 + dim2, TYPE> stiffc1(true);
  CORE::LINALG::Matrix<dim2, dim1 + dim2, TYPE> stiffc2(true);
  CORE::LINALG::Matrix<dim1, dim1 + dim2, TYPE> stiffc1_FAD(true);
  CORE::LINALG::Matrix<dim2, dim1 + dim2, TYPE> stiffc2_FAD(true);
  CORE::LINALG::SerialDenseMatrix stiffcontact1(dim1, dim1 + dim2);
  CORE::LINALG::SerialDenseMatrix stiffcontact2(dim2, dim1 + dim2);
  std::vector<int> lmrow1(dim1);
  std::vector<int> lmrow2(dim2);
  std::vector<int> lmrowowner1(dim1);
  std::vector<int> lmrowowner2(dim2);
  std::vector<int> lmcol1(dim1 + dim2);
  std::vector<int> lmcol2(dim1 + dim2);

  // flag indicating assembly
  bool DoNotAssemble = true;

  // Only apply complete stiffness of algorithmic damping when flag ALGCOMPLETESTIFF is set
  bool completestiff = false;
#ifdef ALGCOMPLETESTIFF
  completestiff = true;
  // In order to accelerate convergence, we only apply the basic stiffness part in case of very
  // large gaps!
  double basicstiffgap = bcparams_.get<double>("BEAMS_BASICSTIFFGAP", -1.0);
  if (basicstiffgap != -1.0)
  {
    if (basicstiffgap < 0.0)
      dserror("The parameter BEAMS_BASICSTIFFGAP has to be positive!");
    else if (gap_ < -1.0 * basicstiffgap)
    {
      completestiff = false;
    }
  }
#endif

  //**********************************************************************
  // evaluate contact stiffness for active pairs
  //**********************************************************************
  if (currentlyincontact_ and iter_ > ITERMAX)
  {
    DoNotAssemble = false;
    // node ids of both elements
    const int* node_ids1 = element1_->NodeIds();
    const int* node_ids2 = element2_->NodeIds();

    // initialize storage for linearizations
    CORE::LINALG::Matrix<dim1 + dim2, 1, TYPE> delta_xi(true);
    CORE::LINALG::Matrix<dim1 + dim2, 1, TYPE> delta_eta(true);
    CORE::LINALG::Matrix<dim1 + dim2, 1, TYPE> delta_gap(true);
    CORE::LINALG::Matrix<dim1 + dim2, 1, TYPE> delta_gap_t(true);
    CORE::LINALG::Matrix<3, dim1 + dim2, TYPE> delta_x1_minus_x2(true);
    CORE::LINALG::Matrix<3, dim1 + dim2, TYPE> delta_n(true);

    //********************************************************************
    // evaluate linearizations and distance
    //********************************************************************
    // linearization of contact point
    ComputeLinXiAndLinEta(
        delta_xi, delta_eta, delta_r, r1_xi, r2_xi, r1_xixi, r2_xixi, N1, N2, N1_xi, N2_xi);

    // linearization of gap function which is equal to delta d
    ComputeLinGap(delta_gap, delta_xi, delta_eta, delta_r, norm_delta_r, r1_xi, r2_xi, N1, N2);

    // linearization of normal vector
    ComputeLinNormal(delta_n, delta_xi, delta_eta, norm_delta_r, r1_xi, r2_xi, N1, N2);

    // linearization of gap function which is equal to delta d
    ComputeLinGapt(delta_gap_t, delta_xi, delta_eta, delta_n, r1_xi, r2_xi, N1, N2, r1_lastiter,
        r2_lastiter, r1_xi_lastiter, r2_xi_lastiter);

    //********************************************************************
    // prepare assembly
    //********************************************************************
    // fill lmrow1 and lmrowowner1
    for (int i = 0; i < numnodes; ++i)
    {
      // get pointer and dof ids
      DRT::Node* node = ContactDiscret().gNode(node_ids1[i]);
      std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

      for (int j = 0; j < 3 * numnodalvalues; ++j)
      {
        lmrow1[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
        lmrowowner1[3 * numnodalvalues * i + j] = node->Owner();
      }
    }

    // fill lmrow2 and lmrowowner2
    for (int i = 0; i < numnodes; ++i)
    {
      // get pointer and node ids
      DRT::Node* node = ContactDiscret().gNode(node_ids2[i]);
      std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

      for (int j = 0; j < 3 * numnodalvalues; ++j)
      {
        lmrow2[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
        lmrowowner2[3 * numnodalvalues * i + j] = node->Owner();
      }
    }

    // fill lmcol1 and lmcol2
    for (int i = 0; i < numnodes; ++i)
    {
      // get pointer and node ids
      DRT::Node* node = ContactDiscret().gNode(node_ids1[i]);
      std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

      for (int j = 0; j < 3 * numnodalvalues; ++j)
      {
        lmcol1[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
        lmcol2[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
      }
    }

    // fill lmcol1 and lmcol2
    for (int i = 0; i < numnodes; ++i)
    {
      // get pointer and node ids
      DRT::Node* node = ContactDiscret().gNode(node_ids2[i]);
      std::vector<int> NodeDofGIDs = GetGlobalDofs(node);

      for (int j = 0; j < 3 * numnodalvalues; ++j)
      {
        lmcol1[3 * numnodalvalues * numnodes + 3 * numnodalvalues * i + j] = NodeDofGIDs[j];
        lmcol2[3 * numnodalvalues * numnodes + 3 * numnodalvalues * i + j] = NodeDofGIDs[j];
      }
    }

    //*************Begin of standard linearization of damping contact forces**************
    //********************************************************************
    // evaluate contact stiffness
    // (1) stiffc1 of first element
    //********************************************************************

    //********************************************************************
    // part I
    //********************************************************************

    CORE::LINALG::Matrix<dim1, 1, TYPE> N1T_normal(true);
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < dim1; j++)
      {
        N1T_normal(j) += N1(i, j) * normal_(i);
      }
    }
    for (int i = 0; i < dim1; i++)
    {
      for (int j = 0; j < dim1 + dim2; j++)
      {
        stiffc1(i, j) += algdampbasicstifffac * N1T_normal(i) *
                         (d * dfd * delta_gap_t(j) + dd * fd * delta_gap(j));
      }
    }

    if (completestiff)
    {
      //********************************************************************
      // part II
      //********************************************************************
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < dim1; j++)
        {
          for (int k = 0; k < dim1 + dim2; k++)
          {
            stiffc1(j, k) += d * fd * N1(i, j) * delta_n(i, k);
          }
        }
      }
      //********************************************************************
      // part III
      //********************************************************************
      CORE::LINALG::Matrix<dim1, 1, TYPE> N1xiT_normal(true);
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < dim1; j++)
        {
          N1xiT_normal(j) += N1_xi(i, j) * normal_(i);
        }
      }
      for (int i = 0; i < dim1; i++)
      {
        for (int j = 0; j < dim1 + dim2; j++)
        {
          stiffc1(i, j) += d * fd * N1xiT_normal(i) * delta_xi(j);
        }
      }
    }
    //********************************************************************
    // evaluate contact stiffness
    // (2) stiffc2 of second element
    //********************************************************************

    //********************************************************************
    // part I
    //********************************************************************
    CORE::LINALG::Matrix<dim2, 1, TYPE> N2T_normal(true);
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < dim2; j++)
      {
        N2T_normal(j) += N2(i, j) * normal_(i);
      }
    }
    for (int i = 0; i < dim1; i++)
    {
      for (int j = 0; j < dim1 + dim2; j++)
      {
        stiffc2(i, j) += -algdampbasicstifffac * N2T_normal(i) *
                         (d * dfd * delta_gap_t(j) + dd * fd * delta_gap(j));
      }
    }

    if (completestiff)
    {
      //********************************************************************
      // part II
      //********************************************************************
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < dim2; j++)
        {
          for (int k = 0; k < dim1 + dim2; k++)
          {
            stiffc2(j, k) += -d * fd * N2(i, j) * delta_n(i, k);
          }
        }
      }
      //********************************************************************
      // part III
      //********************************************************************
      CORE::LINALG::Matrix<dim1, 1, TYPE> N2xiT_normal(true);
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < dim2; j++)
        {
          N2xiT_normal(j) += N2_xi(i, j) * normal_(i);
        }
      }
      for (int i = 0; i < dim2; i++)
      {
        for (int j = 0; j < dim1 + dim2; j++)
        {
          stiffc2(i, j) += -d * fd * N2xiT_normal(i) * delta_eta(j);
        }
      }
    }
//***************End of standard linearization of damping contact forces**************

// automatic differentiation for debugging
#ifdef AUTOMATICDIFF
    CORE::LINALG::Matrix<dim1, 1, TYPE> fc1_FAD(true);
    CORE::LINALG::Matrix<dim2, 1, TYPE> fc2_FAD(true);

    EvaluateFcContact(pp, nullptr, N1, N2, &fc1_FAD, &fc2_FAD);
    for (int j = 0; j < dim1 + dim2; j++)
    {
      for (int i = 0; i < dim1; i++)
        stiffc1_FAD(i, j) = (fc1_FAD(i).dx(j) + fc1_FAD(i).dx(dim1 + dim2) * delta_xi(j) +
                             fc1_FAD(i).dx(dim1 + dim2 + 1) * delta_eta(j));
      for (int i = 0; i < dim2; i++)
        stiffc2_FAD(i, j) = (fc2_FAD(i).dx(j) + fc2_FAD(i).dx(dim1 + dim2) * delta_xi(j) +
                             fc2_FAD(i).dx(dim1 + dim2 + 1) * delta_eta(j));
    }

    std::cout << "Pair: " << element1_->Id() << " / " << element2_->Id() << std::endl;

    std::cout << "stiffc1: " << std::endl;
    for (int i = 0; i < dim1; i++)
    {
      for (int j = 0; j < dim1 + dim2; j++)
      {
        std::cout << stiffc1(i, j).val() << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "stiffc1_FAD: " << std::endl;
    for (int i = 0; i < dim1; i++)
    {
      for (int j = 0; j < dim1 + dim2; j++)
      {
        std::cout << stiffc1_FAD(i, j).val() << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "stiffc2: " << std::endl;
    for (int i = 0; i < dim1; i++)
    {
      for (int j = 0; j < dim1 + dim2; j++)
      {
        std::cout << stiffc2(i, j).val() << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "stiffc2_FAD: " << std::endl;
    for (int i = 0; i < dim1; i++)
    {
      for (int j = 0; j < dim1 + dim2; j++)
      {
        std::cout << stiffc2_FAD(i, j).val() << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

// Additional application of first stiffness contribution with arbitrary scaling (also possible in
// the inactive case)
#ifdef BEAMCONTACTPTC
    // Choose/uncomment one of the following two alternatives
    if (currentlyincontact_ and iter_ > ITERMAX)
    {
      DoNotAssemble = false;
      double ptc = BEAMCONTACTPTC;

      // 1)*******************additional basic stiffness*********************
      //********************************************************************
      // PTC for element 1 (similar to part I of original stiffness)
      //********************************************************************
      CORE::LINALG::Matrix<dim1, 1, TYPE> N1T_normal(true);
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < dim1; j++)
        {
          N1T_normal(j) += N1(i, j) * normal_(i);
        }
      }
      for (int i = 0; i < dim1; i++)
      {
        for (int j = 0; j < dim1 + dim2; j++)
        {
          stiffc1(i, j) += sgn_ * ptc * N1T_normal(i) * delta_gap(j);
        }
      }
      //********************************************************************
      // PTC for element 2 (similar to part I of original stiffness)
      //********************************************************************
      CORE::LINALG::Matrix<dim2, 1, TYPE> N2T_normal(true);
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < dim2; j++)
        {
          N2T_normal(j) += N2(i, j) * normal_(i);
        }
      }
      for (int i = 0; i < dim2; i++)
      {
        for (int j = 0; j < dim1 + dim2; j++)
        {
          stiffc2(i, j) += -sgn_ * ptc * N2T_normal(i) * delta_gap(j);
        }
      }
      // 1)*******************additional basic stiffness*********************

      // 2)*******************standard ptc diagonal terms********************
      //        //********************************************************************
      //        // standard PTC for element 1
      //        //********************************************************************
      //        double jacobi1 = GetJacobi(element1_);
      //        for (int j=0;j<numnodes;j++)
      //        {
      //          for (int i=0;i<3;i++)
      //          {
      //            stiffc1(i+j*3*numnodalvalues,i+j*3*numnodalvalues) += 0.5 * BEAMCONTACTPTC *
      //            jacobi1; stiffc1(i+j*3*numnodalvalues+3,i+j*3*numnodalvalues+3) += 0.5 *
      //            BEAMCONTACTPTCROT * jacobi1;
      //          }
      //        }
      //        //********************************************************************
      //        // standard PTC for element 2
      //        //********************************************************************
      //        double jacobi2 = GetJacobi(element2_);
      //        for (int j=0;j<numnodes;j++)
      //        {
      //          for (int i=0;i<3;i++)
      //          {
      //            stiffc2(i+j*3*numnodalvalues,dim1+i+j*3*numnodalvalues) += 0.5 * BEAMCONTACTPTC
      //            * jacobi2; stiffc2(i+j*3*numnodalvalues+3,dim1+i+j*3*numnodalvalues+3) += 0.5 *
      //            BEAMCONTACTPTCROT * jacobi2;
      //          }
      //        }
      // 2)*******************standard ptc diagonal terms********************

    }  // if(currentlyincontact_)
#endif

  }  // if (currentlyincontact_ and iter_ > ITERMAX)

  //**********************************************************************
  // assemble contact stiffness
  //**********************************************************************
  // change sign of stiffc1 and stiffc2 due to time integration.
  // according to analytical derivation there is no minus sign, but for
  // our time integration methods the negative stiffness must be assembled.

  // now finally assemble stiffc1 and stiffc2
  if (!DoNotAssemble)
  {
#ifndef AUTOMATICDIFF
    for (int j = 0; j < dim1 + dim2; j++)
    {
      for (int i = 0; i < dim1; i++)
        stiffcontact1(i, j) = -CORE::FADUTILS::CastToDouble(stiffc1(i, j));
      for (int i = 0; i < dim2; i++)
        stiffcontact2(i, j) = -CORE::FADUTILS::CastToDouble(stiffc2(i, j));
    }
#else
    for (int j = 0; j < dim1 + dim2; j++)
    {
      for (int i = 0; i < dim1; i++)
        stiffcontact1(i, j) = -CORE::FADUTILS::CastToDouble(stiffc1_FAD(i, j));
      for (int i = 0; i < dim2; i++)
        stiffcontact2(i, j) = -CORE::FADUTILS::CastToDouble(stiffc2_FAD(i, j));
    }
#endif

    stiffmatrix.Assemble(0, stiffcontact1, lmrow1, lmrowowner1, lmcol1);
    stiffmatrix.Assemble(0, stiffcontact2, lmrow2, lmrowowner2, lmcol2);
  }

  return;
}
/*----------------------------------------------------------------------*
 |  end: Evaluate algorithmic stiffness
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Linearizations of contact point                          meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::ComputeLinXiAndLinEta(
    CORE::LINALG::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi,
    CORE::LINALG::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_eta,
    const CORE::LINALG::Matrix<3, 1, TYPE>& delta_r, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xixi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xixi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi)
{
  //**********************************************************************
  // we have to solve the following system of equations:
  //  _              _       _      _       _              _      _       _
  // | L(1,1)  L(1,2) |    | Lin_Xi  |    |  B(1,1)  B(1,2) |   | Lin_d1 |
  // |                | *  |         | =  |                 | * |        |
  // |_L(2,1)  L(2,2)_|    |_Lin_Eta_|    |_B(2,1)  B(2,2)_ |   |_Lin_d2_|
  //
  // this can be done easily because it is a linear 2x2-system.
  // we obtain the solution by inverting matrix L:
  //
  // [Lin_Xi; Lin_Eta] = L^-1 * B * [Lin_d1; Lin_d2] = D * [Lin_d1; Lin_d2]
  //
  //**********************************************************************

  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodes * numnodalvalues;

  // matrices to compute Lin_Xi and Lin_Eta
  CORE::LINALG::Matrix<2, 2, TYPE> L(true);
  CORE::LINALG::Matrix<2, 2, TYPE> L_inv(true);
  CORE::LINALG::Matrix<2, dim1 + dim2, TYPE> B(true);
  CORE::LINALG::Matrix<2, dim1 + dim2, TYPE> D(true);

  // compute L elementwise
  L(0, 0) =
      CORE::FADUTILS::ScalarProduct(r1_xi, r1_xi) + CORE::FADUTILS::ScalarProduct(delta_r, r1_xixi);
  L(1, 1) = -CORE::FADUTILS::ScalarProduct(r2_xi, r2_xi) +
            CORE::FADUTILS::ScalarProduct(delta_r, r2_xixi);
  L(0, 1) = -CORE::FADUTILS::ScalarProduct(r2_xi, r1_xi);
  L(1, 0) = -L(0, 1);

  // invert L by hand
  TYPE det_L = L(0, 0) * L(1, 1) - L(0, 1) * L(1, 0);
  if (CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(det_L)) < DETERMINANTTOL)
    dserror("ERROR: Determinant of L = 0");
  L_inv(0, 0) = L(1, 1) / det_L;
  L_inv(0, 1) = -L(0, 1) / det_L;
  L_inv(1, 0) = -L(1, 0) / det_L;
  L_inv(1, 1) = L(0, 0) / det_L;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1; j++)
    {
      B(0, j) += -delta_r(i) * N1_xi(i, j) - r1_xi(i) * N1(i, j);
      B(1, j) += -r2_xi(i) * N1(i, j);
    }
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim2; j++)
    {
      B(0, j + dim1) += r1_xi(i) * N2(i, j);
      B(1, j + dim1) += -delta_r(i) * N2_xi(i, j) + r2_xi(i) * N2(i, j);
    }
  }

  // compute D = L^-1 * B
  D.Multiply(L_inv, B);

  // finally the linearizations / directional derivatives
  for (int i = 0; i < dim1 + dim2; i++)
  {
    delta_xi(i) = D(0, i);
    delta_eta(i) = D(1, i);
  }

  return;
}
/*----------------------------------------------------------------------*
 |  end: Linearizations of contact point
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Compute linearization of gap                              meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::ComputeLinGap(
    CORE::LINALG::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_gap,
    const CORE::LINALG::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi,
    const CORE::LINALG::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_eta,
    const CORE::LINALG::Matrix<3, 1, TYPE>& delta_r, const TYPE& norm_delta_r,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi, const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodes * numnodalvalues;

  // delta g := delta_r/||delta_r||*auxiliary_matri1 delta d, with auxiliary_matri1 =
  // (r1_xi*delta_xi-r2_xi*delta_eta + (N1, -N2))

  CORE::LINALG::Matrix<3, dim1 + dim2, TYPE> auxiliary_matrix1(true);

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++)
    {
      auxiliary_matrix1(i, j) += r1_xi(i) * delta_xi(j) - r2_xi(i) * delta_eta(j);
    }
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1; j++)
    {
      auxiliary_matrix1(i, j) += N1(i, j);
    }
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim2; j++)
    {
      auxiliary_matrix1(i, j + dim1) += -N2(i, j);
    }
  }

  // compute linearization of gap
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++)
    {
      delta_gap(j) += sgn_ * delta_r(i) * auxiliary_matrix1(i, j) / norm_delta_r;
    }
  }

  return;
}
/*----------------------------------------------------------------------*
 | end: Compute linearization of gap
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Compute linearization of time derivative of gap           meier 07/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::ComputeLinGapt(
    CORE::LINALG::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_gap_t,
    const CORE::LINALG::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi,
    const CORE::LINALG::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_eta,
    const CORE::LINALG::Matrix<3, 2 * 3 * numnodes * numnodalvalues, TYPE>& delta_n,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi, const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r1_old, const CORE::LINALG::Matrix<3, 1, TYPE>& r2_old,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi_old,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi_old)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodes * numnodalvalues;

  CORE::LINALG::Matrix<3, 2 * 3 * numnodes * numnodalvalues, TYPE> delta_vc1(true);
  CORE::LINALG::Matrix<3, 2 * 3 * numnodes * numnodalvalues, TYPE> delta_vc2(true);

  CORE::LINALG::Matrix<3, 1, TYPE> vc1;
  CORE::LINALG::Matrix<3, 1, TYPE> vc2;
  CORE::LINALG::Matrix<3, 1, TYPE> vc1_xi;
  CORE::LINALG::Matrix<3, 1, TYPE> vc2_xi;
  // Calculate velocities
  if (firsttimestep_)
  {
    // In the first time step the pair was found, we have no history information and can't calculate
    // velocities. No damping forces can be applied then. However, it is assumed that in the first
    // time step the pair is found by the contact search, it will not be active immediately and thus
    // no damping force is needed. If this happens anyway, an error is thrown in the method
    // CheckContactStatus().
    for (int i = 0; i < 3; i++)
    {
      vc1(i) = 0.0;
      vc2(i) = 0.0;
      vc1_xi(i) = 0.0;
      vc2_xi(i) = 0.0;
    }
  }
  else
  {
    // Attention: The velocities vc1 and vc2 are not the total contact point velocities, since they
    // do not contain the velocity contribution due to the change in xi and eta. However, since
    // these contributions are perpendicular on the normal_ vector, they are not needed in order to
    // calculate g_t (this case is similar to the calculation of the gap variation).
    for (int i = 0; i < 3; i++)
    {
      vc1(i) = (r1_(i) - r1_old(i)) / dt_;
      vc2(i) = (r2_(i) - r2_old(i)) / dt_;
      vc1_xi(i) = (r1_xi_(i) - r1_xi_old(i)) / dt_;
      vc2_xi(i) = (r2_xi_(i) - r2_xi_old(i)) / dt_;
    }
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++)
    {
      delta_vc1(i, j) += vc1_xi(i) * delta_xi(j, 0);
      delta_vc2(i, j) += vc2_xi(i) * delta_eta(j, 0);
    }
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1; j++)
    {
      delta_vc1(i, j) += N1(i, j) / dt_;
    }
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim2; j++)
    {
      delta_vc2(i, j + dim1) += N2(i, j) / dt_;
    }
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++)
    {
      delta_gap_t(j) +=
          (vc1(i) - vc2(i)) * delta_n(i, j) + normal_(i) * (delta_vc1(i, j) - delta_vc2(i, j));
    }
  }

  return;
}
/*----------------------------------------------------------------------*
 | end: Compute linearization of time derivative of gap
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Compute linearization of normal vector                    meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::ComputeLinNormal(
    CORE::LINALG::Matrix<3, 2 * 3 * numnodes * numnodalvalues, TYPE>& delta_normal,
    const CORE::LINALG::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi,
    const CORE::LINALG::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_eta,
    const TYPE& norm_delta_r, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodes * numnodalvalues;

  // delta n := auxiliary_matri2*auxiliary_matrix1* delta d, with auxiliary_matri2 =
  // (I-nxn)/||r1-r2|| and auxiliary_matri1 = (r1_xi*delta_xi-r2_xi*delta_eta + (N1, -N2))

  CORE::LINALG::Matrix<3, dim1 + dim2, TYPE> auxiliary_matrix1(true);
  CORE::LINALG::Matrix<3, 3, TYPE> auxiliary_matrix2(true);

  // compute auxiliary_matrix1
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++)
    {
      auxiliary_matrix1(i, j) += r1_xi(i) * delta_xi(j) - r2_xi(i) * delta_eta(j);
    }
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1; j++)
    {
      auxiliary_matrix1(i, j) += N1(i, j);
    }
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim2; j++)
    {
      auxiliary_matrix1(i, j + dim1) += -N2(i, j);
    }
  }

  // compute auxiliary_matrix2
  for (int i = 0; i < 3; i++)
  {
    auxiliary_matrix2(i, i) += 1.0 / norm_delta_r;
    for (int j = 0; j < 3; j++)
    {
      auxiliary_matrix2(i, j) += -normal_(i) * normal_(j) / norm_delta_r;
    }
  }

  // compute linearization of normal vector
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < dim1 + dim2; k++)
        delta_normal(i, k) += auxiliary_matrix2(i, j) * auxiliary_matrix1(j, k);

  return;
}
/*----------------------------------------------------------------------*
 | end: Compute linearization of normal vector
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Closest point projection                                  meier 01/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::ClosestPointProjection()
{
  TYPE eta1 = 0.0;
  TYPE eta2 = 0.0;

  // Calculate initial values for eta1 and eta2. This initial guess is based on an assumed linear
  // node interpolation The definitions of b_1, b_2, t_1 and t_2 are according to the paper "ON
  // CONTACT BETWEEN THREE-DIMENSIONAL BEAMS UNDERGOING LARGE DEFLECTIONS" of Wriggers and Zavarise
  // (1997)
  CORE::LINALG::Matrix<3, 1, TYPE> b_1(true);
  CORE::LINALG::Matrix<3, 1, TYPE> b_2(true);
  CORE::LINALG::Matrix<3, 1, TYPE> t_1(true);
  CORE::LINALG::Matrix<3, 1, TYPE> t_2(true);

  // This procedure also works for higher order Reissner beams, since the boundary node still
  // has the ID=2 and takes the second place in the ele1pos_/ele2pos_ vectors
  for (int i = 0; i < 3; i++)
  {
    b_1(i) = ele1pos_(i) + ele1pos_(3 * numnodalvalues + i);
    b_2(i) = ele2pos_(i) + ele2pos_(3 * numnodalvalues + i);
    t_1(i) = -ele1pos_(i) + ele1pos_(3 * numnodalvalues + i);
    t_2(i) = -ele2pos_(i) + ele2pos_(3 * numnodalvalues + i);
  }

  TYPE denom =
      ((CORE::FADUTILS::ScalarProduct(t_2, t_2) * CORE::FADUTILS::ScalarProduct(t_1, t_1) -
           CORE::FADUTILS::ScalarProduct(t_2, t_1) * CORE::FADUTILS::ScalarProduct(t_2, t_1)) /
          (CORE::FADUTILS::ScalarProduct(t_2, t_2) * CORE::FADUTILS::ScalarProduct(t_1, t_1)));

  if (denom > PARALLELTOL)
  {
    // local variables for element coordinates
    TYPE aux1 = CORE::FADUTILS::ScalarProduct(CORE::FADUTILS::DiffVector(b_1, b_2), t_2);
    aux1 = aux1 * CORE::FADUTILS::ScalarProduct(t_1, t_2);
    TYPE aux2 = CORE::FADUTILS::ScalarProduct(CORE::FADUTILS::DiffVector(b_2, b_1), t_1);
    aux2 = aux2 * CORE::FADUTILS::ScalarProduct(t_2, t_2);
    eta1 = (aux1 + aux2) /
           (CORE::FADUTILS::ScalarProduct(t_2, t_2) * CORE::FADUTILS::ScalarProduct(t_1, t_1) -
               CORE::FADUTILS::ScalarProduct(t_2, t_1) * CORE::FADUTILS::ScalarProduct(t_2, t_1));

    aux1 = CORE::FADUTILS::ScalarProduct(CORE::FADUTILS::DiffVector(b_2, b_1), t_1);
    aux1 = aux1 * CORE::FADUTILS::ScalarProduct(t_1, t_2);
    aux2 = CORE::FADUTILS::ScalarProduct(CORE::FADUTILS::DiffVector(b_1, b_2), t_2);
    aux2 = aux2 * CORE::FADUTILS::ScalarProduct(t_1, t_1);
    eta2 = (aux1 + aux2) /
           (CORE::FADUTILS::ScalarProduct(t_2, t_2) * CORE::FADUTILS::ScalarProduct(t_1, t_1) -
               CORE::FADUTILS::ScalarProduct(t_2, t_1) * CORE::FADUTILS::ScalarProduct(t_2, t_1));
  }

  // vectors for shape functions and their derivatives
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N1(true);       // = N1
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N2(true);       // = N2
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N1_xi(true);    // = N1,xi
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N2_xi(true);    // = N2,eta
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N1_xixi(true);  // = N1,xixi
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE> N2_xixi(true);  // = N2,etaeta

  // coords and derivatives of the two contacting points
  CORE::LINALG::Matrix<3, 1, TYPE> r1(true);       // = r1
  CORE::LINALG::Matrix<3, 1, TYPE> r2(true);       // = r2
  CORE::LINALG::Matrix<3, 1, TYPE> r1_xi(true);    // = r1,xi
  CORE::LINALG::Matrix<3, 1, TYPE> r2_xi(true);    // = r2,eta
  CORE::LINALG::Matrix<3, 1, TYPE> r1_xixi(true);  // = r1,xixi
  CORE::LINALG::Matrix<3, 1, TYPE> r2_xixi(true);  // = r2,etaeta
  CORE::LINALG::Matrix<3, 1, TYPE> delta_r(true);  // = r1-r2

  // Tangent and derivatives for tangent field smoothing (only for Reissner beams)
  CORE::LINALG::Matrix<3, 1, TYPE> t1(true);
  CORE::LINALG::Matrix<3, 1, TYPE> t1_xi(true);
  CORE::LINALG::Matrix<3, 1, TYPE> t2(true);
  CORE::LINALG::Matrix<3, 1, TYPE> t2_xi(true);

  // initialize function f and Jacobian df for Newton iteration
  CORE::LINALG::Matrix<2, 1, TYPE> f(true);
  CORE::LINALG::Matrix<2, 2, TYPE> df(true);
  CORE::LINALG::Matrix<2, 2, TYPE> dfinv(true);

  // initial scalar residual (L2-norm of f)
  TYPE residual = 0.0;
  TYPE lastresidual = 0.0;
  TYPE residual0 = 0.0;

  int iter = 0;

  // set these excluding criteria to false in the default case
  elementscrossing_ = false;
  shiftnodalvalues_ = false;

  //**********************************************************************
  // local Newton iteration
  //**********************************************************************
  for (int i = 0; i < BEAMCONTACTMAXITER; ++i)
  {
    lastresidual = residual;

    iter++;
    // reset shape function variables to zero
    N1.Clear();
    N2.Clear();
    N1_xi.Clear();
    N2_xi.Clear();
    N1_xixi.Clear();
    N2_xixi.Clear();

    // update shape functions and their derivatives
    GetShapeFunctions(N1, N2, N1_xi, N2_xi, N1_xixi, N2_xixi, eta1, eta2);
    // update coordinates and derivatives of contact points
    ComputeCoordsAndDerivs(
        r1, r2, r1_xi, r2_xi, r1_xixi, r2_xixi, N1, N2, N1_xi, N2_xi, N1_xixi, N2_xixi);
    // compute delta_r=r1-r2
    for (int j = 0; j < 3; j++) delta_r(j) = r1(j) - r2(j);

    // compute norm of difference vector to scale the equations
    // (this yields better conditioning)
    // Note: Even if automatic differentiation via FAD is applied, norm_delta_r has to be of type
    // double since this factor is needed for a pure scaling of the nonlinear CCP and has not to be
    // linearized!
    double norm_delta_r = CORE::FADUTILS::CastToDouble(CORE::FADUTILS::VectorNorm<3>(delta_r));

    // The closer the beams get, the smaller is norm_delta_r, but
    // norm_delta_r is not allowed to be too small, else numerical problems occur.
    // It can happen quite often that the centerlines of two beam elements of the same physical beam
    // cross in one point and norm_delta_r = 0. Since in this case |eta1|>1 and |eta2|>1 they will
    // be sorted out later anyways.
    // std::cout << "norm_delta_r: " << norm_delta_r << std::endl;
    if (norm_delta_r < NORMTOL)
    {
      // this exludes pairs with IDs i and i+2, i.e. contact with the next but one element
      if (CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(eta1)) +
              CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(eta2)) <
          NEIGHBORTOL)
      {
        std::cout << "Warning! pair " << element1_->Id() << " / " << element2_->Id()
                  << ": Nodal Values shifted! " << std::endl;

        // Shift nodal values of contact pair by a small pre-defined value in order to enable
        // contact evaluation for contact pairs with r1=r2
        // TODO: May this can be done in a more beautiful way in the future
        // It is checked by the method pairs_[i]->GetShiftStatus() in the function void
        // CONTACT::Beam3cmanager::UpdateConstrNorm(double* cnorm) that all active contact pairs
        // fulfill shifnodalvalues_ = false in the converged configuration!!!
        ShiftNodalPositions();
        shiftnodalvalues_ = true;
        continue;
      }
      else
      {
        elementscrossing_ = true;
        break;
      }
    }

    // Evaluate nodal tangents in each case. However, they are used only if
    // smoothing_=INPAR::BEAMCONTACT::bsm_cpp
    BEAMINTERACTION::B3TANGENTSMOOTHING::ComputeTangentsAndDerivs<numnodes, numnodalvalues>(
        t1, t1_xi, nodaltangentssmooth1_, N1, N1_xi);
    BEAMINTERACTION::B3TANGENTSMOOTHING::ComputeTangentsAndDerivs<numnodes, numnodalvalues>(
        t2, t2_xi, nodaltangentssmooth2_, N2, N2_xi);

    // evaluate f at current eta1, eta2
    EvaluateOrthogonalityCondition(f, delta_r, norm_delta_r, r1_xi, r2_xi, t1, t2);

    double jacobi1 = 1.0;
    double jacobi2 = 1.0;
    jacobi1 = GetJacobi(element1_);
    jacobi2 = GetJacobi(element2_);

    // compute the scalar residuum
    // The residual is scaled with 1/element_length since an absolute
    // residual norm is used as local CPP convergence criteria and r_xi scales with the
    // element_length
    residual = f(0) * f(0) / (jacobi1 * jacobi1) + f(1) * f(1) / (jacobi2 * jacobi2);

    if (iter == 1) residual0 = residual;

    // check if Newton iteration has converged
    if (CORE::FADUTILS::CastToDouble(residual) < BEAMCONTACTTOL) break;

    // evaluate Jacobian of f at current eta1, eta2
    // Note: Parallel elements can not be handled with this beam contact formulation; such pairs
    // are sorted out within ComputeCoordsAndDerivs(...) and the local Newton loop is terminated!
    EvaluateLinOrthogonalityCondition(
        df, dfinv, delta_r, norm_delta_r, r1_xi, r2_xi, r1_xixi, r2_xixi, t1, t2, t1_xi, t2_xi);

    if (elementscolinear_) break;

    //    #ifdef FADCHECKS
    //    std::cout << "df: " << df << std::endl;
    //      BEAMCONTACT::SetFADParCoordDofs<numnodes, numnodalvalues>(eta1, eta2);
    //      FADCheckLinOrthogonalityCondition(delta_r,r1_xi,r2_xi);
    //    #endif

    // update element coordinates of contact point
    eta1 += -dfinv(0, 0) * f(0) - dfinv(0, 1) * f(1);
    eta2 += -dfinv(1, 0) * f(0) - dfinv(1, 1) * f(1);
  }
  //**********************************************************************

  // Newton iteration unconverged after BEAMCONTACTMAXITER
  if (residual > BEAMCONTACTTOL)
  {
    if (residual / residual0 < 1.0e-08 and residual0 > 10 * BEAMCONTACTTOL and
        fabs(eta1) < 1.0 + XIETATOL and fabs(eta2) < 1.0 + XIETATOL)
    {
      std::cout << "iter: " << iter << std::endl;
      std::cout << "residual0: " << residual0 << std::endl;
      std::cout << "lastresidual: " << lastresidual << std::endl;
      std::cout << "residual: " << residual << std::endl;
      std::cout << "eta1: " << eta1 << std::endl;
      std::cout << "eta2: " << eta2 << std::endl;
      dserror(
          "Relative CPP residual norm is smaller than 1.0e-08 but Newton is not converged. Adapt "
          "your absolut CPP residual norm!");
    }


    eta1 = 1e+12;
    eta2 = 1e+12;
    cppunconverged_ = true;
  }
  else
  {
    cppunconverged_ = false;
  }


  // store and return final result
  xi1_ = eta1;
  xi2_ = eta2;

// Set xi1_ and xi2_ as primary variables for automatic differentiation
// The dependence between the infinitesimal changes delta xi1_ and delta xi2_ and the
// the increments of the primary displacement variables delta disp have to be given explicitly,
// since no explicit relation between the finite quantities xi1_, xi2_ and disp exists. The latter
// would have been necessary if the full linearization had to be computed directly with Sacado!!!
#ifdef AUTOMATICDIFF
  BEAMCONTACT::SetFADParCoordDofs<numnodes, numnodalvalues>(xi1_, xi2_);
#endif

  return;
}
/*----------------------------------------------------------------------*
|  end: Closest point projection
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Calculate scalar contact force                           meier 08/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::CalcPenaltyLaw()
{
  // if the bool inactivestiff is true, the contact stiffness will always be applied in the first
  // Newton steps for pair which have been active in the last time step (even when they are
  // currentely not active) -> This makes the algorithm more robust.
  bool inactivestiff = DRT::INPUT::IntegralValue<int>(bcparams_, "BEAMS_INACTIVESTIFF");

  if (contactflag_ or (iter_ == 0 and inactivestiff and oldcontactflag_))
  {
    // First parameter for contact force regularization
    double g0 = bcparams_.get<double>("BEAMS_PENREGPARAM_G0", -1.0);

    switch (
        DRT::INPUT::IntegralValue<INPAR::BEAMCONTACT::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW"))
    {
      case INPAR::BEAMCONTACT::pl_lp:  // linear penalty force law
      {
        fp_ = -pp_ * gap_;
        dfp_ = -pp_;

        break;
      }
      case INPAR::BEAMCONTACT::pl_qp:  // quadratic penalty force law
      {
        fp_ = pp_ * gap_ * gap_;
        dfp_ = 2 * pp_ * gap_;

        break;
      }
      case INPAR::BEAMCONTACT::pl_lnqp:  // quadratic regularization for negative gaps
      {
        if (g0 == -1.0) dserror("Invalid value of regularization parameter BEAMS_PENREGPARAM_G0!");

        if (gap_ > -g0)
        {
          // std::cout << "Regularized Penalty!" << std::endl;
          fp_ = pp_ / (2.0 * g0) * gap_ * gap_;
          dfp_ = pp_ / (g0)*gap_;
        }
        else
        {
          // std::cout << "Linear Penalty!" << std::endl;
          fp_ = -pp_ * (gap_ + g0 / 2.0);
          dfp_ = -pp_;
        }

        break;
      }
      case INPAR::BEAMCONTACT::pl_lpqp:  // quadratic regularization for positiv gaps
      {
        if (g0 == -1.0) dserror("Invalid value of regularization parameter BEAMS_PENREGPARAM_G0!");

        double f0 = g0 * pp_ / 2.0;
        double factor_a = pp_ / (g0)-f0 / (g0 * g0);
        double factor_b = -pp_;
        double factor_c = f0;
        if (gap_ > 0)
        {
          // std::cout << "Regularized Penalty!" << std::endl;
          fp_ = factor_a * gap_ * gap_ + factor_b * gap_ + factor_c;
          dfp_ = 2 * factor_a * gap_ + factor_b;
        }
        else
        {
          // std::cout << "Linear Penalty!" << std::endl;
          fp_ = f0 - pp_ * gap_;
          dfp_ = -pp_;
        }

        break;
      }
      case INPAR::BEAMCONTACT::pl_lpcp:  // cubic regularization for positive gaps
      {
        if (g0 == -1.0) dserror("Invalid value of regularization parameter BEAMS_PENREGPARAM_G0!");

        // Third parameter for contact force regularization
        double c0 = bcparams_.get<double>("BEAMS_PENREGPARAM_C0", -1.0);
        if (c0 == -1.0) dserror("Invalid value of regularization parameter BEAMS_PENREGPARAM_C0!");

        // k \in ~[1;3] delivers sensible results representing a parable without turning point
        // k \in ~[3;6] delivers a parable with turning point and consequentely also small negative
        // contact forces ~0.1*f0 k=2.0 is  identical to the quadratic regularization for positive
        // gaps!
        double k = c0;
        double f0 = pp_ * g0 / k;
        double factor_a = -pp_ / (g0 * g0) + 2 * f0 / (g0 * g0 * g0);
        double factor_b = 2 * pp_ / (g0)-3 * f0 / (g0 * g0);
        double factor_c = -pp_;
        double factor_d = f0;
        if (gap_ > 0.0)
        {
          // std::cout << "Regularized Penalty!" << std::endl;
          fp_ = factor_a * gap_ * gap_ * gap_ + factor_b * gap_ * gap_ + factor_c * gap_ + factor_d;
          dfp_ = 3 * factor_a * gap_ * gap_ + 2 * factor_b * gap_ + factor_c;
        }
        else
        {
          // std::cout << "Linear Penalty!" << std::endl;
          fp_ = f0 - pp_ * gap_;
          dfp_ = -pp_;
        }

        break;
      }
      case INPAR::BEAMCONTACT::pl_lpdqp:  // double quadratic regularization for positiv gaps
      {
        if (g0 == -1.0) dserror("Invalid value of regularization parameter BEAMS_PENREGPARAM_G0!");

        // Third parameter for contact force regularization
        double c0 = bcparams_.get<double>("BEAMS_PENREGPARAM_C0", -1.0);
        if (c0 == -1.0) dserror("Invalid value of regularization parameter BEAMS_PENREGPARAM_C0!");

        // Second parameter for contact force regularization
        double f0 = bcparams_.get<double>("BEAMS_PENREGPARAM_F0", -1.0);
        if (f0 == -1.0) dserror("Invalid value of regularization parameter BEAMS_PENREGPARAM_F0!");

        double k =
            c0;  // transition between first and second quadratic regularization part: k \in [0;2.0]
        double g1 = k * f0 / pp_;
        double c_tilde = f0;
        double b_tilde = -pp_;
        double a_bar = (2 * f0 - pp_ * g1) / (2 * g0 * (g0 - g1));
        double b_bar = -2 * g0 * a_bar;
        double c_bar = -g0 * g0 * a_bar - g0 * b_bar;
        double a_tilde = (2 * g1 * a_bar + b_bar - b_tilde) / (2 * g1);

        if (gap_ > g1)
        {
          // std::cout << "Regularized Penalty: g1 < gap < g0!" << std::endl;
          fp_ = a_bar * gap_ * gap_ + b_bar * gap_ + c_bar;
          dfp_ = 2 * a_bar * gap_ + b_bar;
        }
        else if (gap_ > 0)
        {
          // std::cout << "Regularized Penalty: 0 < gap < g1!" << std::endl;
          fp_ = a_tilde * gap_ * gap_ + b_tilde * gap_ + c_tilde;
          dfp_ = 2 * a_tilde * gap_ + b_tilde;
        }
        else
        {
          // std::cout << "Linear Penalty!" << std::endl;
          fp_ = f0 - pp_ * gap_;
          dfp_ = -pp_;
        }

        break;
      }
      case INPAR::BEAMCONTACT::pl_lpep:  // exponential regularization for positiv gaps. Here g0
                                         // represents the cut off radius!
      {
        if (g0 == -1.0) dserror("Invalid value of regularization parameter BEAMS_PENREGPARAM_G0!");

        // Second parameter for contact force regularization
        double f0 = bcparams_.get<double>("BEAMS_PENREGPARAM_F0", -1.0);
        if (f0 == -1.0) dserror("Invalid value of regularization parameter BEAMS_PENREGPARAM_F0!");

        if (gap_ > 0)
        {
          // std::cout << "Regularized Penalty: 0 < gap < g1!" << std::endl;
          fp_ = f0 * exp(-pp_ * gap_ / f0);
          dfp_ = -pp_ * exp(-pp_ * gap_ / f0);
          if (f0 * exp(-pp_ * g0 / f0) > 0.01 * f0)
          {
            std::cout << "Warning - g0: " << g0
                      << " f0*exp(-pp*g0/f0): " << f0 * exp(-pp_ * g0 / f0)
                      << "-> Choose higher cut-off radius g0!" << std::endl;
          }
        }
        else
        {
          // std::cout << "Linear Penalty!" << std::endl;
          fp_ = f0 - pp_ * gap_;
          dfp_ = -pp_;
        }

        break;
      }
    }
  }  // if contactflag_
  else
  {
    fp_ = 0.0;
    dfp_ = 0.0;
  }

#ifdef MAXFORCE
  // If a maximum penalty force is defined, we regularize the penalty force and apply the original /
  // a secant penalty parameter
  if (fp_ > MAXFORCE)
  {
    std::cout << "Maximal force reached: penalty force has been regularized!" << std::endl;
    fp_ = MAXFORCE;
    // Uncomment one of the followin two options:
    // 1)
    dfp_ = -pp_;  // original penalty parameter
    // 2)
    // dfp_ = -MAXFORCE / CORE::FADUTILS::Norm(gap_);     //secant penalty parameter
  }
#endif

  return;
}
/*----------------------------------------------------------------------*
 |  end: Calculate scalar contact force
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Calculate scalar damping force                           meier 08/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::CalcDampingLaw()
{
  if (DRT::INPUT::IntegralValue<int>(bcparams_, "BEAMS_DAMPING") == INPAR::BEAMCONTACT::bd_no)
    return;

  // Damping force parameter
  double d0 = bcparams_.get<double>("BEAMS_DAMPINGPARAM", -1000.0);
  // First parameter for damping force regularization
  double gd1 = bcparams_.get<double>("BEAMS_DAMPREGPARAM1", -1000.0);
  // Second parameter for damping force regularization
  double gd2 = bcparams_.get<double>("BEAMS_DAMPREGPARAM2", -1000.0);

  if (d0 == -1000.0 or gd1 == -1000.0 or gd2 == -1000.0)
    dserror(
        "Damping parameter BEAMS_DAMPINGPARAM, BEAMS_DAMPREGPARAM1 and BEAMS_DAMPREGPARAM2 have to "
        "be chosen!");

  if (gd1 < gd2) dserror("BEAMS_DAMPREGPARAM1 has to be larger or equal to BEAMS_DAMPREGPARAM2!");

  if (dampingcontactflag_)
  {
    CORE::LINALG::Matrix<3, 1, TYPE> vc1;
    CORE::LINALG::Matrix<3, 1, TYPE> vc2;
    // Calculate velocities
    if (firsttimestep_)
    {
      // In the first time step the pair was found, we have no history information and can't
      // calculate velocities. No damping forces can be applied then. However, it is assumed that in
      // the first time step the pair is found by the contact search, it will not be active
      // immediately and thus no damping force is needed. If this happens anyway, an error is thrown
      // in the method CheckContactStatus().
      for (int i = 0; i < 3; i++)
      {
        vc1(i) = 0.0;
        vc2(i) = 0.0;
      }
    }
    else
    {
      // Attention: The velocities vc1 and vc2 are not the total contact point velocities, since
      // they do not contain the velocity contribution due to the change in xi and eta. However,
      // since these contributions are perpendicular on the normal_ vector, they are not needed in
      // order to calculate g_t (this case is similar to the calculation of the gap variation).
      for (int i = 0; i < 3; i++)
      {
        vc1(i) = (r1_(i) - r1_old_(i)) / dt_;
        vc2(i) = (r2_(i) - r2_old_(i)) / dt_;
      }
    }
    TYPE g_t =
        CORE::FADUTILS::ScalarProduct(normal_, vc1) - CORE::FADUTILS::ScalarProduct(normal_, vc2);
    fd_ = -g_t;
    dfd_ = -1.0;

    if (fabs(gd1 - gd2) < DAMPTOL)
    {
      if (gap_ > gd1)
      {
        d_ = 0.0;
        dd_ = 0.0;
      }
      else
      {
        d_ = d0;
        dd_ = 0.0;
      }
    }
    else
    {
      if (gap_ > gd1)
      {
        d_ = 0.0;
        dd_ = 0.0;
      }
      else if (gap_ > gd2)
      {
        d_ = d0 / 2 * (1 - cos((gap_ - gd1) / (gd2 - gd1) * M_PI));
        dd_ = d0 * M_PI / (2 * (gd2 - gd1)) * sin((gap_ - gd1) / (gd2 - gd1) * M_PI);
      }
      else
      {
        d_ = d0;
        dd_ = 0.0;
      }
    }
  }  // if dampingcontactflag_
  else
  {
    fd_ = 0.0;
    dfd_ = 0.0;
    d_ = 0.0;
    dd_ = 0.0;
  }

  return;
}
/*----------------------------------------------------------------------*
 |  Calculate scalar damping force
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  evaluate shape functions and derivatives                 meier 01/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::GetShapeFunctions(
    CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
    CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
    CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi,
    CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xixi,
    CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xixi, const TYPE& eta1,
    const TYPE& eta2)
{
  // get both discretization types
  const CORE::FE::CellType distype1 = element1_->Shape();
  const CORE::FE::CellType distype2 = element2_->Shape();

  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, TYPE> N1_i(true);
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, TYPE> N1_i_xi(true);
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, TYPE> N1_i_xixi(true);
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, TYPE> N2_i(true);
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, TYPE> N2_i_xi(true);
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, TYPE> N2_i_xixi(true);

  if (numnodalvalues == 1)
  {
    // get values and derivatives of shape functions
    CORE::DRT::UTILS::shape_function_1D(N1_i, eta1, distype1);
    CORE::DRT::UTILS::shape_function_1D(N2_i, eta2, distype2);
    CORE::DRT::UTILS::shape_function_1D_deriv1(N1_i_xi, eta1, distype1);
    CORE::DRT::UTILS::shape_function_1D_deriv1(N2_i_xi, eta2, distype2);
    CORE::DRT::UTILS::shape_function_1D_deriv2(N1_i_xixi, eta1, distype1);
    CORE::DRT::UTILS::shape_function_1D_deriv2(N2_i_xixi, eta2, distype2);
  }
  else if (numnodalvalues == 2)
  {
    if (element1_->ElementType() != DRT::ELEMENTS::Beam3ebType::Instance())
      dserror("Only elements of type Beam3eb are valid for the case numnodalvalues=2!");

    if (element2_->ElementType() != DRT::ELEMENTS::Beam3ebType::Instance())
      dserror("Only elements of type Beam3eb are valid for the case numnodalvalues=2!");

    double length1 = 2 * (static_cast<DRT::ELEMENTS::Beam3eb*>(element1_))->jacobi();
    double length2 = 2 * (static_cast<DRT::ELEMENTS::Beam3eb*>(element2_))->jacobi();

    // get values and derivatives of shape functions
    CORE::DRT::UTILS::shape_function_hermite_1D(N1_i, eta1, length1, distype1);
    CORE::DRT::UTILS::shape_function_hermite_1D(N2_i, eta2, length2, distype2);
    CORE::DRT::UTILS::shape_function_hermite_1D_deriv1(N1_i_xi, eta1, length1, distype1);
    CORE::DRT::UTILS::shape_function_hermite_1D_deriv1(N2_i_xi, eta2, length2, distype2);
    CORE::DRT::UTILS::shape_function_hermite_1D_deriv2(N1_i_xixi, eta1, length1, distype1);
    CORE::DRT::UTILS::shape_function_hermite_1D_deriv2(N2_i_xixi, eta2, length2, distype2);
  }
  else
    dserror(
        "Only beam elements with one (nodal positions) or two (nodal positions + nodal tangents) "
        "values are valid!");

  // Assemble the individual shape functions in matrices, such that: r1=N1*d1, r1_xi=N1_xi*d1,
  // r1_xixi=N1_xixi*d1, r2=N2*d2, r2_xi=N2_xi*d2, r2_xixi=N2_xixi*d2
  AssembleShapefunctions(N1_i, N1_i_xi, N1_i_xixi, N1, N1_xi, N1_xixi);
  AssembleShapefunctions(N2_i, N2_i_xi, N2_i_xixi, N2, N2_xi, N2_xixi);

  return;
}
/*----------------------------------------------------------------------*
 |  end: evaluate shape functions and derivatives
 *----------------------------------------------------------------------*/


/*-----------------------------------------------------------------------------------------------------------*
 |  Assemble all shape functions meier 01/14|
 *-----------------------------------------------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::AssembleShapefunctions(
    const CORE::LINALG::Matrix<1, numnodes * numnodalvalues, TYPE>& N_i,
    const CORE::LINALG::Matrix<1, numnodes * numnodalvalues, TYPE>& N_i_xi,
    const CORE::LINALG::Matrix<1, numnodes * numnodalvalues, TYPE>& N_i_xixi,
    CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N,
    CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N_xi,
    CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N_xixi)
{
  // assembly_N is just an array to help assemble the matrices of the shape functions
  // it determines, which shape function is used in which column of N

  std::array<std::array<unsigned int, 3 * numnodes * numnodalvalues>, 3> assembly_N{};

  /*
  Set number of shape functions for each 3*3 block:
  e.g. second order Reissner beam (numnodes=3, numnodalvalues=1)
  int assembly_N[3][9]=  { {1,0,0,2,0,0,3,0,0},
                           {0,1,0,0,2,0,0,3,0},
                           {0,0,1,0,0,2,0,0,3}};

  e.g. Kirchhoff beam (numnodes=2, numnodalvalues=2)
  int assembly_N[3][12]=  {{1,0,0,2,0,0,3,0,0,4,0,0},
                           {0,1,0,0,2,0,0,3,0,0,4,0},
                           {0,0,1,0,0,2,0,0,3,0,0,4}};
  */

  for (int i = 0; i < numnodes * numnodalvalues; i++)
  {
    assembly_N[0][3 * i] = i + 1;
    assembly_N[1][3 * i + 1] = i + 1;
    assembly_N[2][3 * i + 2] = i + 1;
  }

  // Assemble the matrices of the shape functions
  for (int i = 0; i < 3 * numnodes * numnodalvalues; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      if (assembly_N[j][i] == 0)
      {
        N(j, i) = 0;
        N_xi(j, i) = 0;
        N_xixi(j, i) = 0;
      }
      else
      {
        N(j, i) = N_i(assembly_N[j][i] - 1);
        N_xi(j, i) = N_i_xi(assembly_N[j][i] - 1);
        N_xixi(j, i) = N_i_xixi(assembly_N[j][i] - 1);
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | compute contact point coordinates and their derivatives   meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::ComputeCoordsAndDerivs(
    CORE::LINALG::Matrix<3, 1, TYPE>& r1, CORE::LINALG::Matrix<3, 1, TYPE>& r2,
    CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi, CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi,
    CORE::LINALG::Matrix<3, 1, TYPE>& r1_xixi, CORE::LINALG::Matrix<3, 1, TYPE>& r2_xixi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xixi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xixi)
{
  r1.Clear();
  r2.Clear();
  r1_xi.Clear();
  r2_xi.Clear();
  r1_xixi.Clear();
  r2_xixi.Clear();

#ifdef AUTOMATICDIFF
  BEAMCONTACT::SetFADDispDofs<numnodes, numnodalvalues>(ele1pos_, ele2pos_);
#endif

  // compute output variable
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3 * numnodes * numnodalvalues; j++)
    {
      r1(i) += N1(i, j) * ele1pos_(j);
      r2(i) += N2(i, j) * ele2pos_(j);
      r1_xi(i) += N1_xi(i, j) * ele1pos_(j);
      r2_xi(i) += N2_xi(i, j) * ele2pos_(j);
      r1_xixi(i) += N1_xixi(i, j) * ele1pos_(j);
      r2_xixi(i) += N2_xixi(i, j) * ele2pos_(j);
    }
  }

  // store coordinates of contact point into class variables
  for (int i = 0; i < 3; i++)
  {
    r1_(i) = r1(i);
    r2_(i) = r2(i);
    r1_xi_(i) = r1_xi(i);
    r2_xi_(i) = r2_xi(i);
  }

  return;
}
/*----------------------------------------------------------------------*
 | end: compute contact point coordinates and their derivatives         |
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | compute contact point coordinates at last time step       meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::ComputeOldCoordsAndDerivs(
    CORE::LINALG::Matrix<3, 1, TYPE>& r1_old, CORE::LINALG::Matrix<3, 1, TYPE>& r2_old,
    CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi_old, CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi_old,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi)
{
  r1_old.Clear();
  r2_old.Clear();
  r1_xi_old.Clear();
  r2_xi_old.Clear();

  // compute old position vectors
  // important: in order to compute the derivatives correctely, the current parameter coordinates xi
  // and eta of the contact points have to be applied!
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3 * numnodes * numnodalvalues; j++)
    {
      r1_old(i) += N1(i, j) * ele1pos_old_(j);
      r2_old(i) += N2(i, j) * ele2pos_old_(j);
      r1_xi_old(i) += N1_xi(i, j) * ele1pos_old_(j);
      r2_xi_old(i) += N2_xi(i, j) * ele2pos_old_(j);
    }
  }

  return;
}
/*----------------------------------------------------------------------*
 | end: compute contact point coordinates at last time step             |
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Evaluate function f in CPP                               meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::EvaluateOrthogonalityCondition(
    CORE::LINALG::Matrix<2, 1, TYPE>& f, const CORE::LINALG::Matrix<3, 1, TYPE>& delta_r,
    const double norm_delta_r, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi, const CORE::LINALG::Matrix<3, 1, TYPE>& t1,
    const CORE::LINALG::Matrix<3, 1, TYPE>& t2)
{
  // reset f
  f.Clear();

  // evaluate f
  // see Wriggers, Computational Contact Mechanics, equation (12.5)
  if (smoothing_ == INPAR::BEAMCONTACT::bsm_none)  // non-smoothed
  {
    for (int i = 0; i < 3; i++)
    {
      f(0) += delta_r(i) * r1_xi(i) / norm_delta_r;
      f(1) += -delta_r(i) * r2_xi(i) / norm_delta_r;
    }
  }
  else  // smoothed
  {
    std::cout
        << "Warning: The smoothing procedure is not consistent linearized so far! Thereto, the "
           "quantities lin_xi and "
           "lin_eta have to be calculated consistent to the smoothed orthogonality condition below!"
        << std::endl;
    for (int i = 0; i < 3; i++)
    {
      f(0) += delta_r(i) * t1(i) / norm_delta_r;
      f(1) += -delta_r(i) * t2(i) / norm_delta_r;
    }
  }


  return;
}
/*----------------------------------------------------------------------*
 |  end: Evaluate function f in CPP
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Evaluate Jacobian df in CPP                              meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::EvaluateLinOrthogonalityCondition(
    CORE::LINALG::Matrix<2, 2, TYPE>& df, CORE::LINALG::Matrix<2, 2, TYPE>& dfinv,
    const CORE::LINALG::Matrix<3, 1, TYPE>& delta_r, const double norm_delta_r,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi, const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xixi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xixi, const CORE::LINALG::Matrix<3, 1, TYPE>& t1,
    const CORE::LINALG::Matrix<3, 1, TYPE>& t2, const CORE::LINALG::Matrix<3, 1, TYPE>& t1_xi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& t2_xi)

{
  // reset df and dfinv
  df.Clear();
  dfinv.Clear();

  // evaluate df
  // see Wriggers, Computational Contact Mechanics, equation (12.7)

  if (smoothing_ == INPAR::BEAMCONTACT::bsm_none)  // non-smoothed
  {
    for (int i = 0; i < 3; i++)
    {
      df(0, 0) += (r1_xi(i) * r1_xi(i) + delta_r(i) * r1_xixi(i)) / norm_delta_r;
      df(0, 1) += -r1_xi(i) * r2_xi(i) / norm_delta_r;
      df(1, 0) += -r2_xi(i) * r1_xi(i) / norm_delta_r;
      df(1, 1) += (r2_xi(i) * r2_xi(i) - delta_r(i) * r2_xixi(i)) / norm_delta_r;
    }
  }
  else  // smoothed
  {
    for (int i = 0; i < 3; i++)
    {
      df(0, 0) += (r1_xi(i) * t1(i) + delta_r(i) * t1_xi(i)) / norm_delta_r;
      df(0, 1) += -t1(i) * r2_xi(i) / norm_delta_r;
      df(1, 0) += -t2(i) * t1_xi(i) / norm_delta_r;
      df(1, 1) += (r2_xi(i) * t2(i) - delta_r(i) * t2_xi(i)) / norm_delta_r;
    }
  }

  // Inverting (2x2) matrix df by hard coded formula, so that it is
  // possible to handle colinear vectors, because they lead to det(df) =0
  TYPE det_df = df(0, 0) * df(1, 1) - df(1, 0) * df(0, 1);

  //********************************************************************
  // ASSUMPTION:
  // If det_df=0 we assume, that the two elements have an identical
  // neutral axis. These contact objects will be rejected. The outcome
  // of this physically rare phenomenon is that handling of line contact
  // is not possible with this approach.
  //********************************************************************

  // singular df
  if (CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(det_df)) < COLINEARTOL)
  {
    // sort out
    elementscolinear_ = true;
  }
  // regular df (inversion possible)
  else
  {
    // do not sort out
    elementscolinear_ = false;

    // invert df
    dfinv(0, 0) = df(1, 1) / det_df;
    dfinv(0, 1) = -df(0, 1) / det_df;
    dfinv(1, 0) = -df(1, 0) / det_df;
    dfinv(1, 1) = df(0, 0) / det_df;
  }

  return;
}
/*----------------------------------------------------------------------*
 |  end: Evaluate Jacobian df in CPP
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Compute normal vector in contact point                   meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::ComputeNormal(
    CORE::LINALG::Matrix<3, 1, TYPE>& delta_r, TYPE& norm_delta_r,
    std::map<std::pair<int, int>, Teuchos::RCP<Beam3contactinterface>>& contactpairmap)
{
  // compute non-unit normal
  for (int i = 0; i < 3; i++) delta_r(i) = r1_(i) - r2_(i);

  // compute length of normal
  norm_delta_r = CORE::FADUTILS::VectorNorm<3>(delta_r);

  if (CORE::FADUTILS::CastToDouble(norm_delta_r) < NORMTOL)
    dserror("ERROR: Normal of length zero! --> change time step!");

  // compute unit normal
  for (int i = 0; i < 3; i++)
  {
    normal_(i) = delta_r(i) / norm_delta_r;
  }

  // Initialize normal_old_ in the first step with valid closest point projection (in this case the
  // vector normal_old_ is zero, since no valid normal vector was available in the last time step).
  // In case of "sliding contact", i.e. when normal_old_ has already been calculated out of the
  // neighbor element (GetNeighborNormalOld), this is not allowed. Otherwise we would overwrite the
  // vector normal_old_ which has already been calcualted via the neighbor element. (For this reason
  // we check with the norm of normal_old_ and not with the variable firstcall_).
  if (CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(
          CORE::FADUTILS::ScalarProduct(normal_old_, normal_old_))) < NORMALTOL)
  {
    for (int i = 0; i < 3; i++) normal_old_(i) = normal_(i);
  }

  TYPE gap = 0.0;
  sgn_ = 1.0;

  if (DRT::INPUT::IntegralValue<int>(bcparams_, "BEAMS_NEWGAP"))
  {
    if (CORE::FADUTILS::CastToDouble(
            CORE::FADUTILS::Norm(CORE::FADUTILS::ScalarProduct(normal_, normal_old_))) < NORMALTOL)
      dserror("ERROR: Rotation too large! --> Choose smaller Time step!");

    gap =
        CORE::FADUTILS::Signum(CORE::FADUTILS::ScalarProduct(normal_, normal_old_)) * norm_delta_r -
        radius1_ - radius2_;
    sgn_ = CORE::FADUTILS::CastToDouble(
        CORE::FADUTILS::Signum(CORE::FADUTILS::ScalarProduct(normal_, normal_old_)));
  }
  else
  {
    gap = norm_delta_r - radius1_ - radius2_;
  }

  // also set class variable
  gap_ = gap;

  // for comparison reasons we calculate in each case additionally the original gap function
  // definition, thus gap_original==gap_ if the origianl gap function definition is applied
  gap_original_ = norm_delta_r - radius1_ - radius2_;

  return;
}
/*----------------------------------------------------------------------*
 |  end: Compute normal vector in contact point
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Check if contact is active or inactive                    meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::CheckContactStatus(const double& pp)
{
  // First parameter for contact force regularization
  double g0 = bcparams_.get<double>("BEAMS_PENREGPARAM_G0", -1.0);

  int penaltylaw =
      DRT::INPUT::IntegralValue<INPAR::BEAMCONTACT::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW");

  if (penaltylaw == INPAR::BEAMCONTACT::pl_lp)
  {
    // linear penalty force law
    if (gap_ < 0)
    {
      contactflag_ = true;
      currentlyincontact_ = true;
    }
    else
      contactflag_ = false;
  }

  if (penaltylaw == INPAR::BEAMCONTACT::pl_qp)
  {
    // quadratic penalty force law
    if (gap_ < 0)
    {
      contactflag_ = true;
      currentlyincontact_ = true;
    }
    else
      contactflag_ = false;
  }

  if (penaltylaw == INPAR::BEAMCONTACT::pl_lpqp or penaltylaw == INPAR::BEAMCONTACT::pl_lpcp or
      penaltylaw == INPAR::BEAMCONTACT::pl_lpdqp or penaltylaw == INPAR::BEAMCONTACT::pl_lpep)
  {
    // penalty laws with regularization for positive gaps
    if (g0 == -1.0) dserror("Invalid value of regularization parameter BEAMS_PENREGPARAM_G0!");

    if (gap_ < g0)
    {
      contactflag_ = true;
      currentlyincontact_ = true;
    }
    else
      contactflag_ = false;
  }

  if (penaltylaw == INPAR::BEAMCONTACT::pl_lnqp)
  {
    // penalty law with quadratic regularization for negative gaps
    if (gap_ < 0)
    {
      contactflag_ = true;
      currentlyincontact_ = true;
    }
    else
      contactflag_ = false;
  }

  if (DRT::INPUT::IntegralValue<int>(bcparams_, "BEAMS_DAMPING") != INPAR::BEAMCONTACT::bd_no)
  {
    // First parameter for contact force regularization
    double gd1 = bcparams_.get<double>("BEAMS_DAMPREGPARAM1", -1000.0);
    if (gd1 == -1000.0)
      dserror(
          "Damping parameter BEAMS_DAMPINGPARAM, BEAMS_DAMPREGPARAM1 and BEAMS_DAMPREGPARAM2 have "
          "to be chosen!");

    if (gap_ < gd1)
    {
      dampingcontactflag_ = true;
    }
    else
    {
      dampingcontactflag_ = false;
    }
  }

  // Contact is not allowed to happen in the first time step a pair was found by the contact search.
  // An exception is the first time step of a simulation, where no history is available.
  if ((contactflag_ or dampingcontactflag_) and firsttimestep_ and numstep_ > 1)
    dserror(
        "Contact is not allowed to happen in the first time step a pair was found by the contact "
        "search! Choose larger search radius or smaller time step!");

  return;
}
/*----------------------------------------------------------------------*
 |  end: Check if conact is active or inactive
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Get global dofs of a node                                 meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
std::vector<int> CONTACT::Beam3contactnew<numnodes, numnodalvalues>::GetGlobalDofs(
    const DRT::Node* node)
{
  // get dofs in beam contact discretization
  std::vector<int> cdofs = ContactDiscret().Dof(node);

  // get dofs in problem discretization via offset
  std::vector<int> pdofs((int)(cdofs.size()));
  for (int k = 0; k < (int)(cdofs.size()); ++k)
  {
    pdofs[k] = (dofoffsetmap_.find(cdofs[k]))->second;
  }

  return pdofs;
}
/*----------------------------------------------------------------------*
 |  end: Get global dofs of a node
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Change the sign of the normal vector                   meier 02/2014|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::InvertNormal()
{
  for (int i = 0; i < 3; i++) normal_(i) = -normal_(i);
}
/*----------------------------------------------------------------------*
 |  end: Change the sign of the old normal vector                       |
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Update all class variables at the end of time step     meier 02/2014|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::UpdateClassVariablesStep()
{
  // This method is called at the end of the time step for all element pairs found by the contact
  // search, therefore all updates are done here!


  // First we check, that no pair surpasses the maximal displacement of MAXDELTADFAC*searchboxinc_
  // per time step:
  double ele1_delta_pos1 = 0.0;
  double ele1_delta_pos2 = 0.0;
  double ele2_delta_pos1 = 0.0;
  double ele2_delta_pos2 = 0.0;
  for (int i = 0; i < 3; i++)
  {
    ele1_delta_pos1 += (ele1pos_old_(i) - CORE::FADUTILS::CastToDouble(ele1pos_(i))) *
                       (ele1pos_old_(i) - CORE::FADUTILS::CastToDouble(ele1pos_(i)));
    ele1_delta_pos2 += (ele1pos_old_(3 * numnodalvalues + i) -
                           CORE::FADUTILS::CastToDouble(ele1pos_(3 * numnodalvalues + i))) *
                       (ele1pos_old_(3 * numnodalvalues + i) -
                           CORE::FADUTILS::CastToDouble(ele1pos_(3 * numnodalvalues + i)));
    ele2_delta_pos1 += (ele2pos_old_(i) - CORE::FADUTILS::CastToDouble(ele2pos_(i))) *
                       (ele2pos_old_(i) - CORE::FADUTILS::CastToDouble(ele2pos_(i)));
    ele2_delta_pos2 += (ele2pos_old_(3 * numnodalvalues + i) -
                           CORE::FADUTILS::CastToDouble(ele2pos_(3 * numnodalvalues + i))) *
                       (ele2pos_old_(3 * numnodalvalues + i) -
                           CORE::FADUTILS::CastToDouble(ele2pos_(3 * numnodalvalues + i)));
  }
  ele1_delta_pos1 = sqrt(ele1_delta_pos1);
  ele1_delta_pos2 = sqrt(ele1_delta_pos2);
  ele2_delta_pos1 = sqrt(ele2_delta_pos1);
  ele2_delta_pos2 = sqrt(ele2_delta_pos2);

  // Change of nodal positions is not allowed to be larger than MAXDELTADFAC*searchboxinc_ (except
  // for the first time step where elepos_old_=0)
  // TODO: Also Check element midpoint!
  if ((ele1_delta_pos1 > MAXDELTADFAC * searchboxinc_ or
          ele1_delta_pos2 > MAXDELTADFAC * searchboxinc_ or
          ele2_delta_pos1 > MAXDELTADFAC * searchboxinc_ or
          ele2_delta_pos2 > MAXDELTADFAC * searchboxinc_) and
      !firsttimestep_)
  {
    std::cout << "ele1_delta_pos1: " << ele1_delta_pos1 << std::endl;
    std::cout << "ele1_delta_pos2: " << ele1_delta_pos2 << std::endl;
    std::cout << "ele2_delta_pos1: " << ele2_delta_pos1 << std::endl;
    std::cout << "ele2_delta_pos2: " << ele2_delta_pos2 << std::endl;
    std::cout << "MAXDELTADFAC*searchboxinc_: " << MAXDELTADFAC * searchboxinc_ << std::endl;
    std::cout << "ele1pos_: " << ele1pos_ << std::endl;
    std::cout << "ele1pos_old_: " << ele1pos_old_ << std::endl;
    std::cout << "ele2pos_: " << ele2pos_ << std::endl;
    std::cout << "ele2pos_old_: " << ele2pos_old_ << std::endl;
    dserror(
        "Change in nodal positions per time step is larger than prescribed maximum "
        "MAXDELTADFAC*searchboxinc_! Choose smaller time step or larger search radius!");
  }

  // No contact should happen in the first time step an element has been found by the search
  // algorithm (since we need history variables like normal_old_ in order to detect contact).
  // However, there is the following dilemma when this criterion has to be checked: On the one hand,
  // the vector normal_old is needed in order to decide, if we have contact in the first step or
  // not. On the other hand, the vector normal_old_ can only be calculated correctly, when we
  // guarantee, that no contact has happened in the first step. Thus, we choose a heuristic
  // criterion: when at the end of the first time step the distance between the two elements is
  // larger than 2*MAXDELTADFAC*searchboxinc_ no contact has happened. The reason for this choice is
  // as follows: If the elements had crossed in the current time step (contact active) and the
  // distance at the end of the time step is 2*MAXDELTADFAC*searchboxinc_, the distance in the
  // beginning of this time step (=distance at the end of last time step) would have been zero in
  // the worst case. In this case, the contact pair would have already been found in the last time
  // step (2*searchboxinc_ > 2*MAXDELTADFAC*searchboxinc_). These assumption do not hold for the
  // first time step in the simulation, where pairs can be found that already penetrate each other!
  if (firsttimestep_ and numstep_ != 0)
  {
    CORE::LINALG::Matrix<3, 1> midpos1(true);
    CORE::LINALG::Matrix<3, 1> midpos2(true);
    CORE::LINALG::Matrix<3, 1> nodedistance1(true);
    CORE::LINALG::Matrix<3, 1> nodedistance2(true);
    CORE::LINALG::Matrix<3, 1> diffvector(true);
    for (int i = 0; i < 3; i++)
    {
      midpos1(i) =
          0.5 * CORE::FADUTILS::CastToDouble(ele1pos_(i) + ele1pos_(3 * numnodalvalues + i));
      midpos2(i) =
          0.5 * CORE::FADUTILS::CastToDouble(ele2pos_(i) + ele2pos_(3 * numnodalvalues + i));
      nodedistance1(i) =
          CORE::FADUTILS::CastToDouble(ele1pos_(i) - ele1pos_(3 * numnodalvalues + i));
      nodedistance2(i) =
          CORE::FADUTILS::CastToDouble(ele2pos_(i) - ele2pos_(3 * numnodalvalues + i));
      diffvector(i) = midpos1(i) - midpos2(i);
    }
  }


  // Update all history variables!
  for (int j = 0; j < 3; j++) normal_old_(j) = normal_(j);

  xi1_old_ = CORE::FADUTILS::CastToDouble(xi1_);
  xi2_old_ = CORE::FADUTILS::CastToDouble(xi2_);

  for (int i = 0; i < 3 * numnodes * numnodalvalues; i++)
  {
    ele1pos_old_(i) = CORE::FADUTILS::CastToDouble(ele1pos_(i));
    ele2pos_old_(i) = CORE::FADUTILS::CastToDouble(ele2pos_(i));
  }

  // Reset of class variables
  beamendcontactopened_ = false;
  beamsalmostparallel_ = false;
  oldcontactflag_ = contactflag_;
  currentlyincontact_ = false;
  firstcallofstep_ = true;
  firsttimestep_ = false;
  oldcppunconverged_ = cppunconverged_;
}
/*----------------------------------------------------------------------*
 |  end: Update all class variables at the end of time step
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Iterative update of class variables                    meier 08/2014|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::UpdateClassVariablesIter()
{
  for (int i = 0; i < 3 * numnodes * numnodalvalues; i++)
  {
    ele1pos_lastiter_(i) = CORE::FADUTILS::CastToDouble(ele1pos_(i));
    ele2pos_lastiter_(i) = CORE::FADUTILS::CastToDouble(ele2pos_(i));
  }
}
/*----------------------------------------------------------------------*
 |  end: Iterative update of class variables
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Set all class variables                                meier 08/2014|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::SetClassVariables(
    const double& pp, Teuchos::ParameterList timeintparams)
{
  pp_ = pp;
  iter_ = timeintparams.get<int>("iter", -10);
  dt_ = timeintparams.get<double>("dt", -10);
  numstep_ = timeintparams.get<int>("numstep", -10);
  //  std::cout << "iter_: " << iter_ << std::endl;
  //  std::cout << "dt_: " << dt_ << std::endl;
  //  std::cout << "numstep_: " << numstep_ << std::endl;
  if (iter_ == -10.0 or dt_ == -10.0 or numstep_ == -10.0)
    dserror("Invalid time integration parameter!");

  cppunconverged_ = true;
  sgn_ = 1.0;
  gap_ = 0.0;
  gap_original_ = 0.0;
  contactflag_ = false;
  dampingcontactflag_ = false;
  elementscolinear_ = false;
  elementscrossing_ = false;
  shiftnodalvalues_ = false;
  for (int i = 0; i < 3; i++)
  {
    r1_(i) = 0.0;
    r2_(i) = 0.0;
    r1_xi_(i) = 0.0;
    r2_xi_(i) = 0.0;
    r1_old_(i) = 0.0;
    r2_old_(i) = 0.0;
    r1_xi_old_(i) = 0.0;
    r2_xi_old_(i) = 0.0;
    normal_(i) = 0.0;
  }
  fp_ = 0.0;
  dfp_ = 0.0;
  fd_ = 0.0;
  dfd_ = 0.0;
  d_ = 0.0;
  dd_ = 0.0;
  neighbornormalrequired_ = false;
  tangentproduct_ = 0.0;

  // initialize positions of last time step (needed for damping)
  // This means that the velocities are set to zero for the complete first time step!
  if (firsttimestep_)
  {
    for (int i = 0; i < 3 * numnodes * numnodalvalues; i++)
    {
      ele1pos_old_(i) = CORE::FADUTILS::CastToDouble(ele1pos_(i));
      ele2pos_old_(i) = CORE::FADUTILS::CastToDouble(ele2pos_(i));
    }
  }

  // initialize positions of last iteration (needed for algortihmic damping)
  if (firstcallofstep_)
  {
    for (int i = 0; i < 3 * numnodes * numnodalvalues; i++)
    {
      ele1pos_lastiter_(i) = CORE::FADUTILS::CastToDouble(ele1pos_(i));
      ele2pos_lastiter_(i) = CORE::FADUTILS::CastToDouble(ele2pos_(i));
    }
    firstcallofstep_ = false;
  }
}
/*----------------------------------------------------------------------*
 |  end: Set all class variables
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Check if there is a difference of old and new gap      meier 02/2014|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
bool CONTACT::Beam3contactnew<numnodes, numnodalvalues>::GetNewGapStatus()
{
  TYPE gap_diff = gap_ - gap_original_;

  if (CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(gap_diff)) < GAPTOL)
    return false;

  else
    return true;
}
/*----------------------------------------------------------------------*
 |  end: Check if there is a difference of old and new gap               |
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Update nodal coordinates (public)                        meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::UpdateElePos(
    CORE::LINALG::SerialDenseMatrix& newele1pos, CORE::LINALG::SerialDenseMatrix& newele2pos)
{
  for (int i = 0; i < 3 * numnodalvalues; i++)
  {
    for (int j = 0; j < numnodes; j++)
    {
      ele1pos_(3 * numnodalvalues * j + i) = newele1pos(i, j);
      ele2pos_(3 * numnodalvalues * j + i) = newele2pos(i, j);
    }
  }

  return;
}
/*----------------------------------------------------------------------*
 |  end: Update nodal coordinates (public)
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Update nodal tangents for tangent smoothing (public)      meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::UpdateEleSmoothTangents(
    std::map<int, CORE::LINALG::Matrix<3, 1>>& currentpositions)
{
  // Tangent smoothing is only possible for Reissner beam elements --> dserror() otherwise
  if (numnodalvalues > 1)
    dserror("Tangent smoothing only possible for Reissner beam elements (numnodalvalues=1)!!!");

  CORE::LINALG::Matrix<3 * numnodes, 1> elepos_aux(true);
  // Tangent smoothing only possible with data type double (not with Sacado FAD)
  for (int i = 0; i < 3 * numnodes; i++) elepos_aux(i) = CORE::FADUTILS::CastToDouble(ele1pos_(i));

  nodaltangentssmooth1_ = BEAMINTERACTION::B3TANGENTSMOOTHING::CalculateNodalTangents<numnodes>(
      currentpositions, elepos_aux, element1_, neighbors1_);

  elepos_aux.Clear();
  // Tangent smoothing only possible with data type double (not with Sacado FAD)
  for (int i = 0; i < 3 * numnodes; i++) elepos_aux(i) = CORE::FADUTILS::CastToDouble(ele2pos_(i));

  nodaltangentssmooth2_ = BEAMINTERACTION::B3TANGENTSMOOTHING::CalculateNodalTangents<numnodes>(
      currentpositions, elepos_aux, element2_, neighbors2_);
}
/*----------------------------------------------------------------------*
 |  end: Update nodal coordinates (public)
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  Shift Nodal positions in case of crossing (public)       meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::ShiftNodalPositions()
{
  // Reissner beams
  if (numnodalvalues == 1)
  {
    for (int i = 0; i < numnodes; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        ele1pos_(3 * i + j) = ele1pos_(3 * i + j) + SHIFTVALUE * normal_old_(j);
      }
    }
  }
  // Kirchhoff beams
  else if (numnodalvalues == 2)
  {
    if (numnodes == 2)
    {
      for (int j = 0; j < 3; j++)
      {
        ele1pos_(j) = ele1pos_(j) + SHIFTVALUE * normal_old_(j);
        ele1pos_(6 + j) = ele1pos_(6 + j) + SHIFTVALUE * normal_old_(j);
      }
    }
    else
    {
      dserror("Only numnodes = 2 possible for Kirchhoff beams!!!");
    }
  }
  else
  {
    dserror("The parameter numnodalvalues can only have the values 1 or 2!!!");
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Get normalold_ from neighbor element pair (public)       meier 03/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::GetNeighborNormalOld(
    std::map<std::pair<int, int>, Teuchos::RCP<Beam3contactinterface>>& contactpairmap)
{
  // In this method we calculate an approximation for the vector normal_old_ based on the neighbor
  // element pair.
  int id1 = -1.0;
  int id2 = -1.0;

  CORE::LINALG::Matrix<3, 1> delta_r(true);

  // compute non-unit normal
  for (int i = 0; i < 3; i++) delta_r(i) = CORE::FADUTILS::CastToDouble(r1_(i) - r2_(i));

  bool beamsclose = false;

  if (delta_r.Norm2() < (radius1_ + radius2_ + 2 * MAXDELTADFAC * searchboxinc_)) beamsclose = true;

  // bool indicating that this method has been called
  neighbornormalrequired_ = true;

  // bool indicating if the vector normal_old_ will be set within this method or not
  bool normaloldset = false;

  // If the considered pair had no valid closest point pair in the last time step (see
  // ClosestPointProjection: eta1=eta2=1e+12), we can not find the correct neighbor element pair! If
  // the coordinate values of xi1 and xi2 are larger than NEIGHBORNORMALTOL, the normal of the
  // direct neighbor does not provide a good approximation for the own normal_old_ vector. In both
  // cases, we exit here and calculate no approximation for normal_old_;
  if (fabs(xi1_old_) < NEIGHBORNORMALTOL and fabs(xi2_old_) < NEIGHBORNORMALTOL and beamsclose and
      tangentproduct_ < PARALLEL_DEACTIVATION_VAL)
  {
    // This method is based on the assumption, that in each pair the element with the lower global
    // ID is element1_
    if (xi1_old_ < -1.0)
    {
      if (neighbors1_->GetLeftNeighbor() != nullptr)
      {
        id1 = neighbors1_->GetLeftNeighbor()->Id();
      }
    }
    else if (xi1_old_ > 1.0)
    {
      if (neighbors1_->GetRightNeighbor() != nullptr) id1 = neighbors1_->GetRightNeighbor()->Id();
    }
    else
    {
      id1 = element1_->Id();
    }

    if (xi2_old_ < -1.0)
    {
      if (neighbors2_->GetLeftNeighbor() != nullptr)
      {
        id2 = neighbors2_->GetLeftNeighbor()->Id();
      }
    }
    else if (xi2_old_ > 1.0)
    {
      if (neighbors2_->GetRightNeighbor() != nullptr) id2 = neighbors2_->GetRightNeighbor()->Id();
    }
    else
    {
      id2 = element2_->Id();
    }

    // One ID being -1 means, that no neighbor element has been found. Consequently, the considered
    // beam element is a boundary element and no sensible information for normal_old is
    // available/needed.
    if (id1 == -1 or id2 == -1)
    {
    }
    else if (id1 < id2)
    {
      if (contactpairmap.find(std::make_pair(id1, id2)) != contactpairmap.end())
      {
        // It is already tested in GetNormalOld() that the corresponding pair had a valid closest
        // point solution in the last time step if &normal_old_!=nullptr!
        if (contactpairmap[std::make_pair(id1, id2)]->GetNormalOld() != nullptr)
        {
          normal_old_ = *(contactpairmap[std::make_pair(id1, id2)]->GetNormalOld());

          if (contactpairmap[std::make_pair(id1, id2)]->FirstTimeStep() == true and
              beamsclose == true)
            dserror(
                "Vector normal_old_ requested but not available in the first time step the pair "
                "has been found: Choose larger search radius!!!");

          normaloldset = true;

          // If the neighbor pair had a valid cpp solution at last time step the value
          // ||normal_old_=0|| should not be possible!
          if (CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(
                  CORE::FADUTILS::ScalarProduct(normal_old_, normal_old_))) < NORMALTOL and
              beamsclose)
          {
            std::cout << "pair: " << element1_->Id() << " / " << element2_->Id() << ":"
                      << std::endl;
            std::cout << "neighbor pair: " << id1 << " / " << id2 << ":" << std::endl;
            dserror(
                "The vector normal_old_ is not allowed to be zero when taken from neighbor element "
                "pair!");
          }
        }
        else
        {
          std::cout << "Warning: No valid vector normal_old_ of neighbor pair " << id1 << " / "
                    << id2 << " available in order to calculate normal_old_ for pair "
                    << element1_->Id() << " / " << element2_->Id() << "!" << std::endl;
        }
      }
      else
      {
        if (beamsclose)
        {
          std::cout << "Warning: Neighbor pair " << id1 << " / " << id2
                    << " not found in order to calculate normal_old_ for pair " << element1_->Id()
                    << " / " << element2_->Id() << "! Choose larger search radius!" << std::endl;
          dserror("Stopped due to Warning above!");
        }
      }
    }
    else if (id1 > id2)
    {
      if (contactpairmap.find(std::make_pair(id2, id1)) != contactpairmap.end())
      {
        // It is already tested in GetNormalOld() that the corresponding pair had a valid closest
        // point solution in the last time step if &normal_old_!=nullptr!
        if (contactpairmap[std::make_pair(id2, id1)]->GetNormalOld() != nullptr)
        {
          normal_old_ = *(contactpairmap[std::make_pair(id2, id1)]->GetNormalOld());

          if (contactpairmap[std::make_pair(id2, id1)]->FirstTimeStep() == true and
              beamsclose == true)
            dserror(
                "Vector normal_old_requested but not available in the first time step the pair has "
                "been found: Choose larger search radius!!!");

          normaloldset = true;

          // If the neighbor pair had a valid cpp solution at last time step the value
          // ||normal_old_=0|| should not be possible!
          if (CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(
                  CORE::FADUTILS::ScalarProduct(normal_old_, normal_old_))) < NORMALTOL and
              beamsclose)
          {
            std::cout << "pair: " << element1_->Id() << " / " << element2_->Id() << ":"
                      << std::endl;
            std::cout << "neighbor pair: " << id2 << " / " << id1 << ":" << std::endl;
            dserror(
                "The vector normal_old_ is not allowed to be zero when taken from neighbor element "
                "pair!");
          }
        }
        else
        {
          std::cout << "Warning: No valid vector normal_old_ of neighbor pair " << id2 << " / "
                    << id1 << " available in order to calculate normal_old_ for pair "
                    << element1_->Id() << " / " << element2_->Id() << "!" << std::endl;
        }
      }
      else
      {
        if (beamsclose)
        {
          std::cout << "Warning: Neighbor pair " << id2 << " / " << id1
                    << " not found in order to calculate normal_old_ for pair " << element1_->Id()
                    << " / " << element2_->Id() << "! Choose larger search radius!" << std::endl;
          dserror("Stopped due to Warning above!");
        }
      }
    }
    else if (id1 == id2)
      dserror("Selfcontact not possible!!!");
  }

  // If no valid vector normal_old_ has been delivered from the neighbor element pair we set it to
  // zero. In this case normal_old_ is initialized with normal_ later on in the method
  // ComputeNormal().
  if (normaloldset == false)
    for (int i = 0; i < 3; i++) normal_old_ = 0.0;

  if (beamsclose and
      (CORE::FADUTILS::Norm(xi1_ - xi1_old_) > MAXDELTAXIETA or
          CORE::FADUTILS::Norm(xi2_ - xi2_old_) > MAXDELTAXIETA) and
      tangentproduct_ < PARALLEL_DEACTIVATION_VAL)
  {
    std::cout << "Pair: " << element1_->Id() << " / " << element2_->Id() << std::endl;
    std::cout << "xi1: " << xi1_ << "xi2: " << xi2_ << std::endl;
    std::cout << "xi1_old_: " << xi1_old_ << "xi2_old_: " << xi2_old_ << std::endl;
    std::cout << "delta_r.Norm2(): " << delta_r.Norm2() << std::endl;
    std::cout << "tangentproduct_: " << tangentproduct_ << std::endl;
    std::cout
        << "Warning: Neighbor normal required for an element with |xi1_-xi1_old_|>MAXDELTAXIETA or "
           "|xi2_-xi2_old_|>MAXDELTAXIETA. Choose smaller time step or larger element size!"
        << std::endl;
  }

  return;
}

template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::CheckBoundaryContact()
{
  // If the considered element has no neighbor (-> boundary element) and the corresponding
  // parameter coordinate has exceeded the end of the physical beam, the contact is deactivated
  // for this pair for the complete time step!
  if (neighbors1_->GetLeftNeighbor() == nullptr and xi1_ < -1.0 and !cppunconverged_)
  {
    beamendcontactopened_ = true;
  }
  if (neighbors1_->GetRightNeighbor() == nullptr and xi1_ > 1.0 and !cppunconverged_)
  {
    beamendcontactopened_ = true;
  }
  if (neighbors2_->GetLeftNeighbor() == nullptr and xi2_ < -1.0 and !cppunconverged_)
  {
    beamendcontactopened_ = true;
  }
  if (neighbors2_->GetRightNeighbor() == nullptr and xi2_ > 1.0 and !cppunconverged_)
  {
    beamendcontactopened_ = true;
  }

  return;
}

template <const int numnodes, const int numnodalvalues>
double CONTACT::Beam3contactnew<numnodes, numnodalvalues>::GetJacobi(DRT::Element* element1)
{
  double jacobi = 1.0;
  const DRT::ElementType& eot1 = element1->ElementType();

  // The jacobi factor is only needed in order to scale the CPP condition. Therefore, we only use
  // the jacobi_ factor corresponding to the first gauss point of the beam element
  if (eot1 == DRT::ELEMENTS::Beam3ebType::Instance())
  {
    jacobi = (static_cast<DRT::ELEMENTS::Beam3eb*>(element1))->GetJacobi();
  }
  else if (eot1 == DRT::ELEMENTS::Beam3rType::Instance())
  {
    jacobi = (static_cast<DRT::ELEMENTS::Beam3r*>(element1))->GetJacobi();
  }
  else
  {
    std::cout << "Warning: No valid jacobi weight in CPP supported by applied beam element!!!"
              << std::endl;
  }

  return jacobi;
}

#ifdef FADCHECKS
/*----------------------------------------------------------------------*
 |  FAD-Check for Linearizations of contact point            meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::FADCheckLinXiAndLinEta(
    const CORE::LINALG::Matrix<3, 1, TYPE>& delta_r, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xixi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xixi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi)
{
  CORE::LINALG::Matrix<2, 1, TYPE> f(true);

  // compute norm of difference vector to scale the equations
  // (this yields better conditioning)
  // Note: Even if automatic differentiation via FAD is applied, norm_delta_r has to be of type
  // double since this factor is needed for a pure scaling of the nonlinear CCP and has not to be
  // linearized!
  double norm_delta_r = CORE::FADUTILS::CastToDouble(CORE::FADUTILS::VectorNorm<3>(delta_r));

  // evaluate f of CCP condition
  // see Wriggers, Computational Contact Mechanics, equation (12.5)
  for (int i = 0; i < 3; i++)
  {
    f(0) += delta_r(i) * r1_xi(i) / norm_delta_r;
    f(1) += -delta_r(i) * r2_xi(i) / norm_delta_r;
  }

  //**********************************************************************
  // we have to solve the following system of equations:
  //  _              _       _      _       _              _      _       _
  // | L(1,1)  L(1,2) |    | Lin_Xi  |    |  B(1,1)  B(1,2) |   | Lin_d1 |
  // |                | *  |         | =  |                 | * |        |
  // |_L(2,1)  L(2,2)_|    |_Lin_Eta_|    |_B(2,1)  B(2,2)_ |   |_Lin_d2_|
  //
  // this can be done easily because it is a linear 2x2-system.
  // we obtain the solution by inverting matrix L:
  //
  // [Lin_Xi; Lin_Eta] = L^-1 * B * [Lin_d1; Lin_d2] = D * [Lin_d1; Lin_d2]
  //
  //**********************************************************************

  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodes * numnodalvalues;

  // matrices to compute Lin_Xi and Lin_Eta
  CORE::LINALG::Matrix<2, 2, TYPE> L(true);
  CORE::LINALG::Matrix<2, 2, TYPE> L_inv(true);
  CORE::LINALG::Matrix<2, dim1 + dim2, TYPE> B(true);
  CORE::LINALG::Matrix<2, dim1 + dim2, TYPE> D(true);

  // compute L elementwise
  L(0, 0) = f(0).dx(2 * 3 * numnodes * numnodalvalues);
  L(0, 1) = f(0).dx(2 * 3 * numnodes * numnodalvalues + 1);
  L(1, 0) = f(1).dx(2 * 3 * numnodes * numnodalvalues);
  L(1, 1) = f(1).dx(2 * 3 * numnodes * numnodalvalues + 1);

  // invert L by hand
  TYPE det_L = L(0, 0) * L(1, 1) - L(0, 1) * L(1, 0);
  if (CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(det_L)) < DETERMINANTTOL)
    dserror("ERROR: Determinant of L = 0");
  L_inv(0, 0) = L(1, 1) / det_L;
  L_inv(0, 1) = -L(0, 1) / det_L;
  L_inv(1, 0) = -L(1, 0) / det_L;
  L_inv(1, 1) = L(0, 0) / det_L;

  for (int j = 0; j < dim1 + dim2; j++)
  {
    B(0, j) = -f(0).dx(j);
    B(1, j) = -f(1).dx(j);
  }

  // compute D = L^-1 * B
  D.Multiply(L_inv, B);

  std::cout << "linxi and lineta: " << std::endl;

  std::cout << D << std::endl;

  return;
}
/*----------------------------------------------------------------------*
 |  End: FAD-Check for Linearizations of contact point
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  FAD-Check for Linearizations of CCP                      meier 02/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
void CONTACT::Beam3contactnew<numnodes, numnodalvalues>::FADCheckLinOrthogonalityCondition(
    const CORE::LINALG::Matrix<3, 1, TYPE>& delta_r, const CORE::LINALG::Matrix<3, 1, TYPE>& r1_xi,
    const CORE::LINALG::Matrix<3, 1, TYPE>& r2_xi)
{
  CORE::LINALG::Matrix<2, 1, TYPE> f(true);

  // compute norm of difference vector to scale the equations
  // (this yields better conditioning)
  // Note: Even if automatic differentiation via FAD is applied, norm_delta_r has to be of type
  // double since this factor is needed for a pure scaling of the nonlinear CCP and has not to be
  // linearized!
  double norm_delta_r = CORE::FADUTILS::CastToDouble(CORE::FADUTILS::VectorNorm<3>(delta_r));

  // evaluate f of CCP condition
  // see Wriggers, Computational Contact Mechanics, equation (12.5)
  for (int i = 0; i < 3; i++)
  {
    f(0) += delta_r(i) * r1_xi(i) / norm_delta_r;
    f(1) += -delta_r(i) * r2_xi(i) / norm_delta_r;
  }

  CORE::LINALG::Matrix<2, 2, TYPE> df(true);

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      df(i, j) = f(i).dx(2 * 3 * numnodes * numnodalvalues + j);
    }
  }

  std::cout << "df: " << std::endl;

  std::cout << df << std::endl;

  return;
}
/*----------------------------------------------------------------------*
 |  End: FAD-Check for Linearizations of CPP
 *----------------------------------------------------------------------*/
#endif  // #ifdef FADCHECKS

// Possible template cases: this is necessary for the compiler
template class CONTACT::Beam3contactnew<2, 1>;
template class CONTACT::Beam3contactnew<3, 1>;
template class CONTACT::Beam3contactnew<4, 1>;
template class CONTACT::Beam3contactnew<5, 1>;
template class CONTACT::Beam3contactnew<2, 2>;

BACI_NAMESPACE_CLOSE
