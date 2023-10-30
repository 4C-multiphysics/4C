/*----------------------------------------------------------------------*/
/*! \file

\brief evaluation of scatra boundary terms at integration points

\level 1

 */
/*----------------------------------------------------------------------*/
#include "baci_scatra_ele_boundary_calc.H"

#include "baci_discretization_fem_general_utils_boundary_integration.H"
#include "baci_fluid_rotsym_periodicbc.H"
#include "baci_lib_function.H"
#include "baci_lib_globalproblem.H"
#include "baci_mat_fourieriso.H"
#include "baci_mat_list.H"
#include "baci_mat_scatra_mat.H"
#include "baci_mat_thermostvenantkirchhoff.H"
#include "baci_nurbs_discret_nurbs_utils.H"
#include "baci_scatra_ele_parameter_boundary.H"
#include "baci_scatra_ele_parameter_std.H"
#include "baci_scatra_ele_parameter_timint.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::ScaTraEleBoundaryCalc(
    const int numdofpernode, const int numscal, const std::string& disname)
    : scatraparamstimint_(DRT::ELEMENTS::ScaTraEleParameterTimInt::Instance(disname)),
      scatraparams_(DRT::ELEMENTS::ScaTraEleParameterStd::Instance(disname)),
      scatraparamsboundary_(DRT::ELEMENTS::ScaTraEleParameterBoundary::Instance("scatra")),
      numdofpernode_(numdofpernode),
      numscal_(numscal),
      xyze_(true),  // initialize to zero
      weights_(true),
      myknots_(nsd_ele_),
      mypknots_(nsd_),
      normalfac_(1.0),
      ephinp_(numdofpernode_, CORE::LINALG::Matrix<nen_, 1>(true)),
      edispnp_(true),
      diffus_(numscal_, 0),
      shcacp_(0.0),
      xsi_(true),
      funct_(true),
      deriv_(true),
      derxy_(true),
      normal_(true),
      velint_(true),
      metrictensor_(true),
      rotsymmpbc_(Teuchos::rcp(new FLD::RotationallySymmetricPeriodicBC<distype, nsd_ + 1,
          DRT::ELEMENTS::Fluid::none>()))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
int DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::SetupCalc(
    DRT::FaceElement* ele, Teuchos::ParameterList& params, DRT::Discretization& discretization)
{
  // get node coordinates
  CORE::GEO::fillInitialPositionArray<distype, nsd_, CORE::LINALG::Matrix<nsd_, nen_>>(ele, xyze_);

  // Now do the nurbs specific stuff (for isogeometric elements)
  if (DRT::NURBS::IsNurbs(distype))
  {
    // for isogeometric elements --- get knotvectors for parent
    // element and boundary element, get weights
    bool zero_size =
        DRT::NURBS::GetKnotVectorAndWeightsForNurbsBoundary(ele, ele->FaceParentNumber(),
            ele->ParentElement()->Id(), discretization, mypknots_, myknots_, weights_, normalfac_);

    // if we have a zero sized element due to a interpolated point -> exit here
    if (zero_size) return -1;
  }  // Nurbs specific stuff

  // rotationally symmetric periodic bc's: do setup for current element
  rotsymmpbc_->Setup(ele);

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
int DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::Evaluate(DRT::FaceElement* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization,
    DRT::Element::LocationArray& la, CORE::LINALG::SerialDenseMatrix& elemat1_epetra,
    CORE::LINALG::SerialDenseMatrix& elemat2_epetra,
    CORE::LINALG::SerialDenseVector& elevec1_epetra,
    CORE::LINALG::SerialDenseVector& elevec2_epetra,
    CORE::LINALG::SerialDenseVector& elevec3_epetra)
{
  //--------------------------------------------------------------------------------
  // preparations for element
  //--------------------------------------------------------------------------------
  if (SetupCalc(ele, params, discretization) == -1) return 0;

  ExtractDisplacementValues(ele, discretization, la);

  // check for the action parameter
  const auto action = Teuchos::getIntegralValue<SCATRA::BoundaryAction>(params, "action");
  // evaluate action
  EvaluateAction(ele, params, discretization, action, la, elemat1_epetra, elemat2_epetra,
      elevec1_epetra, elevec2_epetra, elevec3_epetra);

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::ExtractDisplacementValues(
    DRT::FaceElement* ele, const DRT::Discretization& discretization,
    DRT::Element::LocationArray& la)
{
  switch (ele->ParentElement()->Shape())
  {
    case CORE::FE::CellType::hex8:
    {
      ExtractDisplacementValues<CORE::FE::CellType::hex8>(ele, discretization, la);
      break;
    }
    case CORE::FE::CellType::hex27:
    {
      ExtractDisplacementValues<CORE::FE::CellType::hex27>(ele, discretization, la);
      break;
    }
    case CORE::FE::CellType::tet4:
    {
      ExtractDisplacementValues<CORE::FE::CellType::tet4>(ele, discretization, la);
      break;
    }
    case CORE::FE::CellType::quad4:
    {
      ExtractDisplacementValues<CORE::FE::CellType::quad4>(ele, discretization, la);
      break;
    }
    case CORE::FE::CellType::tri6:
    {
      ExtractDisplacementValues<CORE::FE::CellType::tri6>(ele, discretization, la);
      break;
    }
    case CORE::FE::CellType::tri3:
    {
      ExtractDisplacementValues<CORE::FE::CellType::tri3>(ele, discretization, la);
      break;
    }
    case CORE::FE::CellType::nurbs9:
    {
      ExtractDisplacementValues<CORE::FE::CellType::nurbs9>(ele, discretization, la);
      break;
    }
    default:
      dserror("Not implemented for discretization type: %i!", ele->ParentElement()->Shape());
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
template <CORE::FE::CellType parentdistype>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::ExtractDisplacementValues(
    DRT::FaceElement* ele, const DRT::Discretization& discretization,
    DRT::Element::LocationArray& la)
{
  // get additional state vector for ALE case: grid displacement
  if (scatraparams_->IsAle())
  {
    // get number of dof-set associated with displacement related dofs
    const int ndsdisp = scatraparams_->NdsDisp();

    Teuchos::RCP<const Epetra_Vector> dispnp = discretization.GetState(ndsdisp, "dispnp");
    dsassert(dispnp != Teuchos::null, "Cannot get state vector 'dispnp'");

    // determine number of displacement related dofs per node
    const int numdispdofpernode = la[ndsdisp].lm_.size() / nen_;

    // construct location vector for displacement related dofs
    std::vector<int> lmdisp(nsd_ * nen_, -1);
    for (int inode = 0; inode < nen_; ++inode)
      for (int idim = 0; idim < nsd_; ++idim)
        lmdisp[inode * nsd_ + idim] = la[ndsdisp].lm_[inode * numdispdofpernode + idim];

    // extract local values of displacement field from global state vector
    DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nsd_, nen_>>(*dispnp, edispnp_, lmdisp);

    // add nodal displacements to point coordinates
    UpdateNodeCoordinates();

    // determine location array information of parent element
    DRT::Element::LocationArray parent_la(discretization.NumDofSets());
    ele->ParentElement()->LocationVector(discretization, parent_la, false);

    const int num_node_parent_ele =
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<parentdistype>::numNodePerElement;

    // determine number of the displacement related dofs per node
    const int parent_numdispdofpernode = parent_la[ndsdisp].lm_.size() / num_node_parent_ele;

    std::vector<int> parent_lmdisp(nsd_ * num_node_parent_ele, -1);
    for (int inode = 0; inode < num_node_parent_ele; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        parent_lmdisp[inode * nsd_ + idim] =
            parent_la[ndsdisp].lm_[inode * parent_numdispdofpernode + idim];
      }
    }

    // extract local values of displacement field from global state vector
    CORE::LINALG::Matrix<nsd_, num_node_parent_ele> parentdispnp;
    DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nsd_, num_node_parent_ele>>(
        *dispnp, parentdispnp, parent_lmdisp);

    eparentdispnp_.resize(num_node_parent_ele * nsd_);
    for (int i = 0; i < num_node_parent_ele; ++i)
      for (int idim = 0; idim < nsd_; ++idim)
        eparentdispnp_[i * nsd_ + idim] = parentdispnp(idim, i);
  }
  else
  {
    edispnp_.Clear();
    eparentdispnp_.clear();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
int DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvaluateAction(DRT::FaceElement* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization,
    SCATRA::BoundaryAction action, DRT::Element::LocationArray& la,
    CORE::LINALG::SerialDenseMatrix& elemat1_epetra,
    CORE::LINALG::SerialDenseMatrix& elemat2_epetra,
    CORE::LINALG::SerialDenseVector& elevec1_epetra,
    CORE::LINALG::SerialDenseVector& elevec2_epetra,
    CORE::LINALG::SerialDenseVector& elevec3_epetra)
{
  std::vector<int>& lm = la[0].lm_;

  switch (action)
  {
    case SCATRA::BoundaryAction::calc_normal_vectors:
    {
      CalcNormalVectors(params, ele);
      break;
    }

    case SCATRA::BoundaryAction::integrate_shape_functions:
    {
      // NOTE: add area value only for elements which are NOT ghosted!
      const bool addarea = (ele->Owner() == discretization.Comm().MyPID());
      IntegrateShapeFunctions(ele, params, elevec1_epetra, addarea);

      break;
    }

    case SCATRA::BoundaryAction::calc_mass_matrix:
    {
      CalcMatMass(ele, elemat1_epetra);

      break;
    }

    case SCATRA::BoundaryAction::calc_Neumann:
    {
      DRT::Condition* condition = params.get<DRT::Condition*>("condition");
      if (condition == nullptr) dserror("Cannot access Neumann boundary condition!");

      EvaluateNeumann(ele, params, discretization, *condition, la, elevec1_epetra, 1.);

      break;
    }

    case SCATRA::BoundaryAction::calc_Neumann_inflow:
    {
      NeumannInflow(ele, params, discretization, la, elemat1_epetra, elevec1_epetra);

      break;
    }

    case SCATRA::BoundaryAction::calc_convective_heat_transfer:
    {
      // get the parent element including its material
      DRT::Element* parentele = ele->ParentElement();
      Teuchos::RCP<MAT::Material> mat = parentele->Material();

      // get values of scalar
      Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
      if (phinp == Teuchos::null) dserror("Cannot get state vector 'phinp'");

      // extract local values from global vector
      std::vector<CORE::LINALG::Matrix<nen_, 1>> ephinp(
          numdofpernode_, CORE::LINALG::Matrix<nen_, 1>(true));
      DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nen_, 1>>(*phinp, ephinp, lm);

      // get condition
      Teuchos::RCP<DRT::Condition> cond = params.get<Teuchos::RCP<DRT::Condition>>("condition");
      if (cond == Teuchos::null) dserror("Cannot access condition 'TransportThermoConvections'!");

      // get heat transfer coefficient and surrounding temperature
      const double heatranscoeff = cond->GetDouble("coeff");
      const double surtemp = cond->GetDouble("surtemp");

      ConvectiveHeatTransfer(
          ele, mat, ephinp, elemat1_epetra, elevec1_epetra, heatranscoeff, surtemp);

      break;
    }

    case SCATRA::BoundaryAction::calc_weak_Dirichlet:
    {
      // get the parent element including its material
      DRT::Element* parentele = ele->ParentElement();
      Teuchos::RCP<MAT::Material> mat = parentele->Material();

      if (numscal_ > 1) dserror("not yet implemented for more than one scalar\n");

      switch (distype)
      {
        // 2D:
        case CORE::FE::CellType::line2:
        {
          if (ele->ParentElement()->Shape() == CORE::FE::CellType::quad4)
          {
            WeakDirichlet<CORE::FE::CellType::line2, CORE::FE::CellType::quad4>(
                ele, params, discretization, mat, elemat1_epetra, elevec1_epetra);
          }
          else
          {
            dserror("expected combination quad4/hex8 or line2/quad4 for surface/parent pair");
          }
          break;
        }

        // 3D:
        case CORE::FE::CellType::quad4:
        {
          if (ele->ParentElement()->Shape() == CORE::FE::CellType::hex8)
          {
            WeakDirichlet<CORE::FE::CellType::quad4, CORE::FE::CellType::hex8>(
                ele, params, discretization, mat, elemat1_epetra, elevec1_epetra);
          }
          else
            dserror("expected combination quad4/hex8 or line2/quad4 for surface/parent pair");

          break;
        }

        default:
        {
          dserror("not implemented yet\n");
          break;
        }
      }

      break;
    }

    case SCATRA::BoundaryAction::calc_fs3i_surface_permeability:
    {
      EvaluateSurfacePermeability(ele, params, discretization, la, elemat1_epetra, elevec1_epetra);

      break;
    }

    case SCATRA::BoundaryAction::calc_fps3i_surface_permeability:
    {
      EvaluateKedemKatchalsky(ele, params, discretization, la, elemat1_epetra, elevec1_epetra);

      break;
    }

    case SCATRA::BoundaryAction::add_convective_mass_flux:
    {
      // calculate integral of convective mass/heat flux
      // NOTE: since results are added to a global vector via normal assembly
      //       it would be wrong to suppress results for a ghosted boundary!

      // get actual values of transported scalars
      Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
      if (phinp == Teuchos::null) dserror("Cannot get state vector 'phinp'");

      // extract local values from the global vector
      std::vector<CORE::LINALG::Matrix<nen_, 1>> ephinp(
          numdofpernode_, CORE::LINALG::Matrix<nen_, 1>(true));
      DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nen_, 1>>(*phinp, ephinp, lm);

      // get number of dofset associated with velocity related dofs
      const int ndsvel = scatraparams_->NdsVel();

      // get convective (velocity - mesh displacement) velocity at nodes
      Teuchos::RCP<const Epetra_Vector> convel =
          discretization.GetState(ndsvel, "convective velocity field");
      if (convel == Teuchos::null) dserror("Cannot get state vector convective velocity");

      // determine number of velocity related dofs per node
      const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

      // construct location vector for velocity related dofs
      std::vector<int> lmvel(nsd_ * nen_, -1);
      for (int inode = 0; inode < nen_; ++inode)
        for (int idim = 0; idim < nsd_; ++idim)
          lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

      // we deal with a nsd_-dimensional flow field
      CORE::LINALG::Matrix<nsd_, nen_> econvel(true);

      // extract local values of convective velocity field from global state vector
      DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nsd_, nen_>>(*convel, econvel, lmvel);

      // rotate the vector field in the case of rotationally symmetric boundary conditions
      rotsymmpbc_->RotateMyValuesIfNecessary(econvel);

      // for the moment we ignore the return values of this method
      CalcConvectiveFlux(ele, ephinp, econvel, elevec1_epetra);

      break;
    }

    case SCATRA::BoundaryAction::calc_s2icoupling:
    {
      EvaluateS2ICoupling(
          ele, params, discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra);

      break;
    }

    case SCATRA::BoundaryAction::calc_s2icoupling_capacitance:
    {
      EvaluateS2ICouplingCapacitance(
          discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra, elevec2_epetra);

      break;
    }

    case SCATRA::BoundaryAction::calc_s2icoupling_od:
    {
      EvaluateS2ICouplingOD(ele, params, discretization, la, elemat1_epetra);
      break;
    }

    case SCATRA::BoundaryAction::calc_s2icoupling_capacitance_od:
    {
      EvaluateS2ICouplingCapacitanceOD(params, discretization, la, elemat1_epetra, elemat2_epetra);
      break;
    }

    case SCATRA::BoundaryAction::calc_boundary_integral:
    {
      CalcBoundaryIntegral(ele, elevec1_epetra);
      break;
    }
    case SCATRA::BoundaryAction::calc_nodal_size:
    {
      EvaluateNodalSize(ele, params, discretization, la, elevec1_epetra);
      break;
    }
    case SCATRA::BoundaryAction::calc_Robin:
    {
      CalcRobinBoundary(ele, params, discretization, la, elemat1_epetra, elevec1_epetra, 1.);
      break;
    }
    case SCATRA::BoundaryAction::calc_s2icoupling_flux:
    {
      CalcS2ICouplingFlux(ele, params, discretization, la, elevec1_epetra);
      break;
    }
    default:
    {
      dserror("Not acting on this boundary action. Forgot implementation?");
      break;
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
int DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvaluateNeumann(DRT::FaceElement* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization, DRT::Condition& condition,
    DRT::Element::LocationArray& la, CORE::LINALG::SerialDenseVector& elevec1, const double scalar)
{
  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // find out whether we will use a time curve
  const double time = scatraparamstimint_->Time();

  // get values, switches and spatial functions from the condition
  // (assumed to be constant on element boundary)
  const int numdof = condition.GetInt("numdof");
  const auto* onoff = condition.Get<std::vector<int>>("onoff");
  const auto* val = condition.Get<std::vector<double>>("val");
  const auto* func = condition.Get<std::vector<int>>("funct");

  if (numdofpernode_ != numdof)
  {
    dserror(
        "The NUMDOF you have entered in your TRANSPORT NEUMANN CONDITION does not equal the number "
        "of scalars.");
  }

  // integration loop
  for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
  {
    double fac = EvalShapeFuncAndIntFac(intpoints, iquad);

    // factor given by spatial function
    double functfac = 1.0;

    // determine global coordinates of current Gauss point
    CORE::LINALG::Matrix<nsd_, 1> coordgp;  // coordinate has always to be given in 3D!
    coordgp.MultiplyNN(xyze_, funct_);

    int functnum = -1;
    const double* coordgpref = &coordgp(0);  // needed for function evaluation

    for (int dof = 0; dof < numdofpernode_; ++dof)
    {
      if ((*onoff)[dof])  // is this dof activated?
      {
        // factor given by spatial function
        if (func) functnum = (*func)[dof];

        if (functnum > 0)
        {
          // evaluate function at current Gauss point (provide always 3D coordinates!)
          functfac = DRT::Problem::Instance()
                         ->FunctionById<DRT::UTILS::FunctionOfSpaceTime>(functnum - 1)
                         .Evaluate(coordgpref, time, dof);
        }
        else
          functfac = 1.;

        const double val_fac_funct_fac = (*val)[dof] * fac * functfac;

        for (int node = 0; node < nen_; ++node)
          // TODO: with or without eps_
          elevec1[node * numdofpernode_ + dof] += scalar * funct_(node) * val_fac_funct_fac;
      }  // if ((*onoff)[dof])
    }    // loop over dofs
  }      // loop over integration points

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::CalcNormalVectors(
    Teuchos::ParameterList& params, DRT::FaceElement* ele)
{
  // access the global vector
  const auto normals =
      params.get<Teuchos::RCP<Epetra_MultiVector>>("normal vectors", Teuchos::null);
  if (normals == Teuchos::null) dserror("Could not access vector 'normal vectors'");

  // determine constant outer normal to this element
  if constexpr (nsd_ == 3 and nsd_ele_ == 1)
  {
    // get first 3 nodes in parent element
    auto* p_ele = ele->ParentElement();
    dsassert(p_ele->NumNode() >= 3, "Parent element must at least have 3 nodes.");
    CORE::LINALG::Matrix<nsd_, 3> xyz_parent_ele;

    for (int i_node = 0; i_node < 3; ++i_node)
    {
      const auto* coords = p_ele->Nodes()[i_node]->X();
      for (int dim = 0; dim < nsd_; ++dim) xyz_parent_ele(dim, i_node) = coords[dim];
    }

    normal_ = GetConstNormal(xyze_, xyz_parent_ele);
  }
  else if constexpr (nsd_ - nsd_ele_ == 1)
  {
    normal_ = GetConstNormal(xyze_);
  }
  else
    dserror("This combination of space dimension and element dimension makes no sense.");

  for (int j = 0; j < nen_; j++)
  {
    const int nodegid = (ele->Nodes()[j])->Id();
    if (normals->Map().MyGID(nodegid))
    {
      // scaling to a unit vector is performed on the global level after
      // assembly of nodal contributions since we have no reliable information
      // about the number of boundary elements adjacent to a node
      for (int dim = 0; dim < nsd_; dim++)
      {
        normals->SumIntoGlobalValue(nodegid, dim, normal_(dim));
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::NeumannInflow(
    const DRT::FaceElement* ele, Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la,
    CORE::LINALG::SerialDenseMatrix& emat, CORE::LINALG::SerialDenseVector& erhs)
{
  // get location vector associated with primary dofset
  std::vector<int>& lm = la[0].lm_;

  // get parent element
  DRT::Element* parentele = ele->ParentElement();

  // get material of parent element
  Teuchos::RCP<MAT::Material> material = parentele->Material();

  // we don't know the parent element's lm vector; so we have to build it here
  const int nenparent = parentele->NumNode();
  std::vector<int> lmparent(nenparent);
  std::vector<int> lmparentowner;
  std::vector<int> lmparentstride;
  parentele->LocationVector(discretization, lmparent, lmparentowner, lmparentstride);

  // get values of scalar
  Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
  if (phinp == Teuchos::null) dserror("Cannot get state vector 'phinp'");

  // extract local values from global vector
  std::vector<CORE::LINALG::Matrix<nen_, 1>> ephinp(
      numdofpernode_, CORE::LINALG::Matrix<nen_, 1>(true));
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nen_, 1>>(*phinp, ephinp, lm);

  // get number of dofset associated with velocity related dofs
  const int ndsvel = scatraparams_->NdsVel();

  // get convective (velocity - mesh displacement) velocity at nodes
  Teuchos::RCP<const Epetra_Vector> convel =
      discretization.GetState(ndsvel, "convective velocity field");
  if (convel == Teuchos::null) dserror("Cannot get state vector convective velocity");

  // determine number of velocity related dofs per node
  const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

  // construct location vector for velocity related dofs
  std::vector<int> lmvel(nsd_ * nen_, -1);
  for (int inode = 0; inode < nen_; ++inode)
    for (int idim = 0; idim < nsd_; ++idim)
      lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

  // we deal with a nsd_-dimensional flow field
  CORE::LINALG::Matrix<nsd_, nen_> econvel(true);

  // extract local values of convective velocity field from global state vector
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nsd_, nen_>>(*convel, econvel, lmvel);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  rotsymmpbc_->RotateMyValuesIfNecessary(econvel);

  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over all scalars
  for (int k = 0; k < numdofpernode_; ++k)
  {
    // loop over all integration points
    for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
    {
      const double fac = EvalShapeFuncAndIntFac(intpoints, iquad, &normal_);

      // get velocity at integration point
      velint_.Multiply(econvel, funct_);

      // normal velocity
      const double normvel = velint_.Dot(normal_);

      if (normvel < -0.0001)
      {
        // set density to 1.0
        double dens = GetDensity(material, ephinp, k);

        // integration factor for left-hand side
        const double lhsfac = dens * normvel * scatraparamstimint_->TimeFac() * fac;

        // integration factor for right-hand side
        double rhsfac = 0.0;
        if (scatraparamstimint_->IsIncremental() and scatraparamstimint_->IsGenAlpha())
          rhsfac = lhsfac / scatraparamstimint_->AlphaF();
        else if (not scatraparamstimint_->IsIncremental() and scatraparamstimint_->IsGenAlpha())
          rhsfac = lhsfac * (1.0 - scatraparamstimint_->AlphaF()) / scatraparamstimint_->AlphaF();
        else if (scatraparamstimint_->IsIncremental() and not scatraparamstimint_->IsGenAlpha())
          rhsfac = lhsfac;

        // matrix
        for (int vi = 0; vi < nen_; ++vi)
        {
          const double vlhs = lhsfac * funct_(vi);

          const int fvi = vi * numdofpernode_ + k;

          for (int ui = 0; ui < nen_; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            emat(fvi, fui) -= vlhs * funct_(ui);
          }
        }

        // scalar at integration point
        const double phi = funct_.Dot(ephinp[k]);

        // rhs
        const double vrhs = rhsfac * phi;
        for (int vi = 0; vi < nen_; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          erhs[fvi] += vrhs * funct_(vi);
        }
      }
    }
  }
}  // DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype>::NeumannInflow

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
double DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::GetDensity(
    Teuchos::RCP<const MAT::Material> material,
    const std::vector<CORE::LINALG::Matrix<nen_, 1>>& ephinp, const int k)
{
  // initialization
  double density(0.);

  // get density depending on material
  switch (material->MaterialType())
  {
    case INPAR::MAT::m_matlist:
    {
      const auto* actmat = static_cast<const MAT::MatList*>(material.get());

      const int matid = actmat->MatID(0);

      if (actmat->MaterialById(matid)->MaterialType() == INPAR::MAT::m_scatra)
      {
        // set density to unity
        density = 1.;
      }
      else
        dserror("type of material found in material list is not supported");

      break;
    }

    case INPAR::MAT::m_matlist_reactions:
    case INPAR::MAT::m_scatra:
    {
      // set density to unity
      density = 1.;

      break;
    }

    default:
    {
      dserror("Invalid material type!");
      break;
    }
  }

  return density;
}  // DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype>::GetDensity

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
std::vector<double> DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::CalcConvectiveFlux(
    const DRT::FaceElement* ele, const std::vector<CORE::LINALG::Matrix<nen_, 1>>& ephinp,
    const CORE::LINALG::Matrix<nsd_, nen_>& evelnp, CORE::LINALG::SerialDenseVector& erhs)
{
  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  std::vector<double> integralflux(numscal_);

  // loop over all scalars
  for (int k = 0; k < numscal_; ++k)
  {
    integralflux[k] = 0.0;

    // loop over all integration points
    for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
    {
      const double fac = EvalShapeFuncAndIntFac(intpoints, iquad, &normal_);

      // get velocity at integration point
      velint_.Multiply(evelnp, funct_);

      // normal velocity (note: normal_ is already a unit(!) normal)
      const double normvel = velint_.Dot(normal_);

      // scalar at integration point
      const double phi = funct_.Dot(ephinp[k]);

      const double val = phi * normvel * fac;
      integralflux[k] += val;
      // add contribution to provided vector (distribute over nodes using shape fct.)
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * numdofpernode_ + k;
        erhs[fvi] += val * funct_(vi);
      }
    }
  }

  return integralflux;

}  // ScaTraEleBoundaryCalc<distype>::ConvectiveFlux

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::ConvectiveHeatTransfer(
    const DRT::FaceElement* ele, Teuchos::RCP<const MAT::Material> material,
    const std::vector<CORE::LINALG::Matrix<nen_, 1>>& ephinp, CORE::LINALG::SerialDenseMatrix& emat,
    CORE::LINALG::SerialDenseVector& erhs, const double heatranscoeff, const double surtemp)
{
  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over all scalars
  for (int k = 0; k < numdofpernode_; ++k)
  {
    // loop over all integration points
    for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
    {
      const double fac = EvalShapeFuncAndIntFac(intpoints, iquad, &normal_);

      // get specific heat capacity at constant volume
      double shc = 0.0;
      if (material->MaterialType() == INPAR::MAT::m_th_fourier_iso)
      {
        const auto* actmat = static_cast<const MAT::FourierIso*>(material.get());

        shc = actmat->Capacity();
      }
      else if (material->MaterialType() == INPAR::MAT::m_thermostvenant)
      {
        const auto* actmat = static_cast<const MAT::ThermoStVenantKirchhoff*>(material.get());

        shc = actmat->Capacity();
      }
      else
        dserror("Material type is not supported for convective heat transfer!");

      // integration factor for left-hand side
      const double lhsfac = heatranscoeff * scatraparamstimint_->TimeFac() * fac / shc;

      // integration factor for right-hand side
      double rhsfac = 0.0;
      if (scatraparamstimint_->IsIncremental() and scatraparamstimint_->IsGenAlpha())
        rhsfac = lhsfac / scatraparamstimint_->AlphaF();
      else if (not scatraparamstimint_->IsIncremental() and scatraparamstimint_->IsGenAlpha())
        rhsfac = lhsfac * (1.0 - scatraparamstimint_->AlphaF()) / scatraparamstimint_->AlphaF();
      else if (scatraparamstimint_->IsIncremental() and not scatraparamstimint_->IsGenAlpha())
        rhsfac = lhsfac;

      // matrix
      for (int vi = 0; vi < nen_; ++vi)
      {
        const double vlhs = lhsfac * funct_(vi);

        const int fvi = vi * numdofpernode_ + k;

        for (int ui = 0; ui < nen_; ++ui)
        {
          const int fui = ui * numdofpernode_ + k;

          emat(fvi, fui) -= vlhs * funct_(ui);
        }
      }

      // scalar at integration point
      const double phi = funct_.Dot(ephinp[k]);

      // rhs
      const double vrhs = rhsfac * (phi - surtemp);
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * numdofpernode_ + k;

        erhs[fvi] += vrhs * funct_(vi);
      }
    }
  }
}  // DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype>::ConvectiveHeatTransfer

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::
    EvaluateSpatialDerivativeOfAreaIntegrationFactor(
        const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_>& intpoints, const int iquad,
        CORE::LINALG::Matrix<nsd_, nen_>& dsqrtdetg_dd)
{
  // safety check
  if (nsd_ele_ != 2)
    dserror("Computation of shape derivatives only implemented for 2D interfaces!");

  EvaluateShapeFuncAndDerivativeAtIntPoint(intpoints, iquad);

  // compute derivatives of spatial coordinates w.r.t. reference coordinates
  static CORE::LINALG::Matrix<nsd_ele_, nsd_> dxyz_drs;
  dxyz_drs.MultiplyNT(deriv_, xyze_);

  // compute basic components of shape derivatives
  const double xr(dxyz_drs(0, 0)), xs(dxyz_drs(1, 0)), yr(dxyz_drs(0, 1)), ys(dxyz_drs(1, 1)),
      zr(dxyz_drs(0, 2)), zs(dxyz_drs(1, 2));
  const double denominator_inv =
      1.0 / std::sqrt(xr * xr * ys * ys + xr * xr * zs * zs - 2.0 * xr * xs * yr * ys -
                      2.0 * xr * xs * zr * zs + xs * xs * yr * yr + xs * xs * zr * zr +
                      yr * yr * zs * zs - 2.0 * yr * ys * zr * zs + ys * ys * zr * zr);
  const double numerator_xr = xr * ys * ys + xr * zs * zs - xs * yr * ys - xs * zr * zs;
  const double numerator_xs = -(xr * yr * ys + xr * zr * zs - xs * yr * yr - xs * zr * zr);
  const double numerator_yr = -(xr * xs * ys - xs * xs * yr - yr * zs * zs + ys * zr * zs);
  const double numerator_ys = xr * xr * ys - xr * xs * yr - yr * zr * zs + ys * zr * zr;
  const double numerator_zr = -(xr * xs * zs - xs * xs * zr + yr * ys * zs - ys * ys * zr);
  const double numerator_zs = xr * xr * zs - xr * xs * zr + yr * yr * zs - yr * ys * zr;

  // compute shape derivatives
  for (int ui = 0; ui < nen_; ++ui)
  {
    dsqrtdetg_dd(0, ui) =
        denominator_inv * (numerator_xr * deriv_(0, ui) + numerator_xs * deriv_(1, ui));
    dsqrtdetg_dd(1, ui) =
        denominator_inv * (numerator_yr * deriv_(0, ui) + numerator_ys * deriv_(1, ui));
    dsqrtdetg_dd(2, ui) =
        denominator_inv * (numerator_zr * deriv_(0, ui) + numerator_zs * deriv_(1, ui));
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
double DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvalShapeFuncAndIntFac(
    const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_>& intpoints, const int iquad,
    CORE::LINALG::Matrix<nsd_, 1>* normalvec)
{
  EvaluateShapeFuncAndDerivativeAtIntPoint(intpoints, iquad);

  // the metric tensor and the area of an infinitesimal surface/line element
  // optional: get normal at integration point as well
  double drs(0.0);
  CORE::DRT::UTILS::ComputeMetricTensorForBoundaryEle<distype, probdim>(
      xyze_, deriv_, metrictensor_, drs, true, normalvec);

  // for nurbs elements the normal vector must be scaled with a special orientation factor!!
  if (DRT::NURBS::IsNurbs(distype))
  {
    if (normalvec != nullptr) normal_.Scale(normalfac_);
  }

  // return the integration factor
  return intpoints.IP().qwgt[iquad] * drs;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::
    EvaluateShapeFuncAndDerivativeAtIntPoint(
        const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_>& intpoints, const int iquad)
{
  // coordinates of the current integration point
  const double* gpcoord = (intpoints.IP().qxg)[iquad];
  for (int idim = 0; idim < nsd_ele_; idim++)
  {
    xsi_(idim) = gpcoord[idim];
  }

  if (not DRT::NURBS::IsNurbs(distype))
  {
    // shape functions and their first derivatives
    CORE::DRT::UTILS::shape_function<distype>(xsi_, funct_);
    CORE::DRT::UTILS::shape_function_deriv1<distype>(xsi_, deriv_);
  }
  else  // nurbs elements are always somewhat special...
  {
    CORE::DRT::NURBS::UTILS::nurbs_get_funct_deriv(
        funct_, deriv_, xsi_, myknots_, weights_, distype);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
CORE::LINALG::Matrix<3, 1> DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::GetConstNormal(
    const CORE::LINALG::Matrix<3, nen_>& xyze)
{
  if (DRT::NURBS::IsNurbs(distype)) dserror("Element normal not implemented for NURBS");

  CORE::LINALG::Matrix<3, 1> normal(true), dist1(true), dist2(true);
  for (int i = 0; i < 3; i++)
  {
    dist1(i) = xyze(i, 1) - xyze(i, 0);
    dist2(i) = xyze(i, 2) - xyze(i, 0);
  }

  normal.CrossProduct(dist1, dist2);

  const double length = normal.Norm2();
  if (length < 1.0e-16) dserror("Zero length for element normal");

  normal.Scale(1.0 / length);

  return normal;
}  // ScaTraEleBoundaryCalc<distype>::GetConstNormal

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
CORE::LINALG::Matrix<2, 1> DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::GetConstNormal(
    const CORE::LINALG::Matrix<2, nen_>& xyze)
{
  if (DRT::NURBS::IsNurbs(distype)) dserror("Element normal not implemented for NURBS");

  CORE::LINALG::Matrix<2, 1> normal(true);

  normal(0) = xyze(1, 1) - xyze(1, 0);
  normal(1) = (-1.0) * (xyze(0, 1) - xyze(0, 0));

  const double length = normal.Norm2();
  if (length < 1.0e-16) dserror("Zero length for element normal");

  normal.Scale(1.0 / length);

  return normal;
}  // ScaTraEleBoundaryCalc<distype>::GetConstNormal

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
CORE::LINALG::Matrix<3, 1> DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::GetConstNormal(
    const CORE::LINALG::Matrix<3, nen_>& xyze, const CORE::LINALG::Matrix<3, 3>& nodes_parent_ele)
{
  if (DRT::NURBS::IsNurbs(distype)) dserror("Element normal not implemented for NURBS");

  CORE::LINALG::Matrix<3, 1> normal(true), normal_parent_ele(true), boundary_ele(true),
      parent_ele_v1(true), parent_ele_v2(true);

  for (int dim = 0; dim < 3; ++dim)
  {
    boundary_ele(dim, 0) = xyze(dim, 0) - xyze(dim, 1);
    parent_ele_v1(dim, 0) = nodes_parent_ele(dim, 0) - nodes_parent_ele(dim, 1);
    parent_ele_v2(dim, 0) = nodes_parent_ele(dim, 0) - nodes_parent_ele(dim, 2);
  }

  normal_parent_ele.CrossProduct(parent_ele_v1, parent_ele_v2);
  normal.CrossProduct(normal_parent_ele, boundary_ele);

  // compute inward vector and check if its scalar product with the normal vector is negative.
  // Otherwise, change the sign of the normal vector
  CORE::LINALG::Matrix<3, 1> distance(true), inward_vector(true);
  // find node on parent element, that has non-zero distance to all boundary nodes
  for (int i_parent_node = 0; i_parent_node < 3; ++i_parent_node)
  {
    bool is_boundary_node = false;
    for (int i_boundary_node = 0; i_boundary_node < nen_; ++i_boundary_node)
    {
      for (int dim = 0; dim < 3; ++dim)
        distance(dim, 0) = nodes_parent_ele(dim, i_parent_node) - xyze(dim, i_boundary_node);

      // if the distance of the parent element to one boundary node is zero, it cannot be a
      // non-boundary node
      if (distance.Norm2() < 1.0e-10)
      {
        is_boundary_node = true;
        break;
      }
    }
    if (!is_boundary_node)
    {
      inward_vector.Update(1.0, distance, 0.0);
      break;
    }
  }
  if (inward_vector.Dot(normal) >= 0.0) normal.Scale(-1.0);

  const double length = normal.Norm2();
  if (length < 1.0e-16) dserror("Zero length for element normal");

  normal.Scale(1.0 / length);

  return normal;
}  // ScaTraEleBoundaryCalc<distype>::GetConstNormal

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvaluateS2ICoupling(
    const DRT::FaceElement* ele, Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la,
    CORE::LINALG::SerialDenseMatrix& eslavematrix, CORE::LINALG::SerialDenseMatrix& emastermatrix,
    CORE::LINALG::SerialDenseVector& eslaveresidual)
{
  // extract local nodal values on present and opposite sides of scatra-scatra interface
  ExtractNodeValues(discretization, la);
  std::vector<CORE::LINALG::Matrix<nen_, 1>> emasterphinp(numscal_);
  ExtractNodeValues(emasterphinp, discretization, la, "imasterphinp");

  // dummy element matrix and vector
  CORE::LINALG::SerialDenseMatrix dummymatrix;
  CORE::LINALG::SerialDenseVector dummyvector;

  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  CORE::LINALG::Matrix<nsd_, 1> normal;

  // element slave mechanical stress tensor
  const bool is_pseudo_contact = scatraparamsboundary_->IsPseudoContact();
  std::vector<CORE::LINALG::Matrix<nen_, 1>> eslavestress_vector(
      6, CORE::LINALG::Matrix<nen_, 1>(true));
  if (is_pseudo_contact)
    ExtractNodeValues(eslavestress_vector, discretization, la, "mechanicalStressState",
        scatraparams_->NdsTwoTensorQuantity());

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.IP().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac =
        DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvalShapeFuncAndIntFac(
            intpoints, gpid, &normal);

    // evaluate overall integration factors
    const double timefacfac = scatraparamstimint_->TimeFac() * fac;
    const double timefacrhsfac = scatraparamstimint_->TimeFacRhs() * fac;
    if (timefacfac < 0. or timefacrhsfac < 0.) dserror("Integration factor is negative!");

    const double pseudo_contact_fac =
        CalculatePseudoContactFactor(is_pseudo_contact, eslavestress_vector, normal, funct_);

    EvaluateS2ICouplingAtIntegrationPoint<distype>(ephinp_, emasterphinp, pseudo_contact_fac,
        funct_, funct_, funct_, funct_, numscal_, scatraparamsboundary_, timefacfac, timefacrhsfac,
        eslavematrix, emastermatrix, dummymatrix, dummymatrix, eslaveresidual, dummyvector);
  }  // end of loop over integration points
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
template <CORE::FE::CellType distype_master>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvaluateS2ICouplingAtIntegrationPoint(
    const std::vector<CORE::LINALG::Matrix<nen_, 1>>& eslavephinp,
    const std::vector<CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<distype_master>::numNodePerElement, 1>>&
        emasterphinp,
    const double pseudo_contact_fac, const CORE::LINALG::Matrix<nen_, 1>& funct_slave,
    const CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<distype_master>::numNodePerElement, 1>&
        funct_master,
    const CORE::LINALG::Matrix<nen_, 1>& test_slave,
    const CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<distype_master>::numNodePerElement, 1>&
        test_master,
    const int numscal,
    const DRT::ELEMENTS::ScaTraEleParameterBoundary* const scatra_parameter_boundary,
    const double timefacfac, const double timefacrhsfac, CORE::LINALG::SerialDenseMatrix& k_ss,
    CORE::LINALG::SerialDenseMatrix& k_sm, CORE::LINALG::SerialDenseMatrix& k_ms,
    CORE::LINALG::SerialDenseMatrix& k_mm, CORE::LINALG::SerialDenseVector& r_s,
    CORE::LINALG::SerialDenseVector& r_m)
{
  // get condition specific parameters
  const int kineticmodel = scatra_parameter_boundary->KineticModel();
  const std::vector<double>* permeabilities = scatra_parameter_boundary->Permeabilities();

  // number of nodes of master-side mortar element
  const int nen_master =
      CORE::DRT::UTILS::DisTypeToNumNodePerEle<distype_master>::numNodePerElement;

  // loop over scalars
  for (int k = 0; k < numscal; ++k)
  {
    // evaluate dof values at current integration point on slave and master sides of scatra-scatra
    // interface
    const double slavephiint = funct_slave.Dot(eslavephinp[k]);
    const double masterphiint = funct_master.Dot(emasterphinp[k]);

    // compute matrix and vector contributions according to kinetic model for current scatra-scatra
    // interface coupling condition
    switch (kineticmodel)
    {
      // constant permeability model
      case INPAR::S2I::kinetics_constperm:
      {
        if (permeabilities == nullptr)
          dserror("Cannot access vector of permeabilities for scatra-scatra interface coupling!");
        if (permeabilities->size() != (unsigned)numscal)
          dserror("Number of permeabilities does not match number of scalars!");

        // core residual
        const double N_timefacrhsfac = pseudo_contact_fac * timefacrhsfac * (*permeabilities)[k] *
                                       (slavephiint - masterphiint);

        // core linearizations
        const double dN_dc_slave_timefacfac =
            pseudo_contact_fac * timefacfac * (*permeabilities)[k];
        const double dN_dc_master_timefacfac = -dN_dc_slave_timefacfac;

        if (k_ss.numRows() and k_sm.numRows() and r_s.length())
        {
          for (int vi = 0; vi < nen_; ++vi)
          {
            const int fvi = vi * numscal + k;

            for (int ui = 0; ui < nen_; ++ui)
              k_ss(fvi, ui * numscal + k) +=
                  test_slave(vi) * dN_dc_slave_timefacfac * funct_slave(ui);

            for (int ui = 0; ui < nen_master; ++ui)
              k_sm(fvi, ui * numscal + k) +=
                  test_slave(vi) * dN_dc_master_timefacfac * funct_master(ui);

            r_s[fvi] -= test_slave(vi) * N_timefacrhsfac;
          }
        }
        else if (k_ss.numRows() or k_sm.numRows() or r_s.length())
          dserror("Must provide both slave-side matrices and slave-side vector or none of them!");

        if (k_ms.numRows() and k_mm.numRows() and r_m.length())
        {
          for (int vi = 0; vi < nen_master; ++vi)
          {
            const int fvi = vi * numscal + k;

            for (int ui = 0; ui < nen_; ++ui)
              k_ms(fvi, ui * numscal + k) -=
                  test_master(vi) * dN_dc_slave_timefacfac * funct_slave(ui);

            for (int ui = 0; ui < nen_master; ++ui)
              k_mm(fvi, ui * numscal + k) -=
                  test_master(vi) * dN_dc_master_timefacfac * funct_master(ui);

            r_m[fvi] += test_master(vi) * N_timefacrhsfac;
          }
        }
        else if (k_ms.numRows() or k_mm.numRows() or r_m.length())
          dserror("Must provide both master-side matrices and master-side vector or none of them!");

        break;
      }

      default:
      {
        dserror("Kinetic model for scatra-scatra interface coupling not yet implemented!");
        break;
      }
    }
  }  // end of loop over scalars
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
double DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::CalculatePseudoContactFactor(
    const bool is_pseudo_contact,
    const std::vector<CORE::LINALG::Matrix<nen_, 1>>& eslavestress_vector,
    const CORE::LINALG::Matrix<nsd_, 1>& gp_normal,
    const CORE::LINALG::Matrix<nen_, 1>& funct_slave)
{
  if (is_pseudo_contact)
  {
    static CORE::LINALG::Matrix<1, 1> normal_stress_comp_gp;
    static CORE::LINALG::Matrix<nsd_, nsd_> current_gp_stresses;
    static CORE::LINALG::Matrix<nsd_, 1> tmp;
    current_gp_stresses(0, 0) = funct_slave.Dot(eslavestress_vector[0]);
    current_gp_stresses(1, 1) = funct_slave.Dot(eslavestress_vector[1]);
    current_gp_stresses(2, 2) = funct_slave.Dot(eslavestress_vector[2]);
    current_gp_stresses(0, 1) = current_gp_stresses(1, 0) = funct_slave.Dot(eslavestress_vector[3]);
    current_gp_stresses(1, 2) = current_gp_stresses(2, 1) = funct_slave.Dot(eslavestress_vector[4]);
    current_gp_stresses(0, 2) = current_gp_stresses(2, 0) = funct_slave.Dot(eslavestress_vector[5]);

    tmp.MultiplyNN(1.0, current_gp_stresses, gp_normal, 0.0);
    normal_stress_comp_gp.MultiplyTN(1.0, gp_normal, tmp, 0.0);

    // if tensile stress, i.e. normal stress component > 0 return 0.0, otherwise return 1.0
    return normal_stress_comp_gp(0) > 0.0 ? 0.0 : 1.0;
  }
  else
    return 1.0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
double DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::CalculateDetFOfParentElement(
    const DRT::FaceElement* faceele, const double* faceele_xsi)
{
  if (scatraparams_->IsAle())
  {
    switch (faceele->ParentElement()->Shape())
    {
      case CORE::FE::CellType::hex8:
      {
        return CalculateDetFOfParentElement<CORE::FE::CellType::hex8>(faceele, faceele_xsi);
      }
      case CORE::FE::CellType::tet4:
      {
        return CalculateDetFOfParentElement<CORE::FE::CellType::tet4>(faceele, faceele_xsi);
      }
      default:
      {
        dserror("Not implemented for discretization type: %i!", faceele->ParentElement()->Shape());
        break;
      }
    }
  }

  return 1.0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
template <CORE::FE::CellType parentdistype>
double DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::CalculateDetFOfParentElement(
    const DRT::FaceElement* faceele, const double* faceele_xi)
{
  const int parent_ele_dim = CORE::DRT::UTILS::DisTypeToDim<parentdistype>::dim;
  const int parent_ele_num_nodes =
      CORE::DRT::UTILS::DisTypeToNumNodePerEle<parentdistype>::numNodePerElement;

  auto parent_xi =
      CORE::DRT::UTILS::CalculateParentGPFromFaceElementData<parent_ele_dim>(faceele_xi, faceele);
  static CORE::LINALG::Matrix<probdim, probdim> defgrd;

  static CORE::LINALG::Matrix<parent_ele_num_nodes, probdim> xdisp, xrefe, xcurr;

  for (auto i = 0; i < parent_ele_num_nodes; ++i)
  {
    const double* x = faceele->ParentElement()->Nodes()[i]->X();
    for (auto dim = 0; dim < probdim; ++dim)
    {
      xdisp(i, dim) = eparentdispnp_.at(i * probdim + dim);
      xrefe(i, dim) = x[dim];
    }
  }

  CORE::LINALG::Matrix<probdim, parent_ele_num_nodes> deriv_parent(true);
  CORE::DRT::UTILS::shape_function_deriv1<parentdistype>(parent_xi, deriv_parent);

  static CORE::LINALG::Matrix<probdim, probdim> inv_detF;
  inv_detF.Multiply(deriv_parent, xrefe);
  inv_detF.Invert();

  static CORE::LINALG::Matrix<probdim, parent_ele_num_nodes> N_XYZ;
  xcurr.Update(1.0, xrefe, 1.0, xdisp);
  N_XYZ.Multiply(inv_detF, deriv_parent);
  defgrd.MultiplyTT(xcurr, N_XYZ);

  return defgrd.Determinant();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvaluateS2ICouplingOD(
    const DRT::FaceElement* ele, Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la,
    CORE::LINALG::SerialDenseMatrix& eslavematrix)
{
  // extract local nodal values on present and opposite side of scatra-scatra interface
  ExtractNodeValues(discretization, la);
  std::vector<CORE::LINALG::Matrix<nen_, 1>> emasterphinp(
      numscal_, CORE::LINALG::Matrix<nen_, 1>(true));
  ExtractNodeValues(emasterphinp, discretization, la, "imasterphinp");

  CORE::LINALG::Matrix<nsd_, 1> normal;

  // element slave mechanical stress tensor
  const bool is_pseudo_contact = scatraparamsboundary_->IsPseudoContact();
  std::vector<CORE::LINALG::Matrix<nen_, 1>> eslavestress_vector(
      6, CORE::LINALG::Matrix<nen_, 1>(true));
  if (is_pseudo_contact)
    ExtractNodeValues(eslavestress_vector, discretization, la, "mechanicalStressState",
        scatraparams_->NdsTwoTensorQuantity());

  // get current scatra-scatra interface coupling condition
  Teuchos::RCP<DRT::Condition> s2icondition = params.get<Teuchos::RCP<DRT::Condition>>("condition");
  if (s2icondition == Teuchos::null)
    dserror("Cannot access scatra-scatra interface coupling condition!");

  // get primary variable to derive the linearization
  const auto differentiationtype =
      Teuchos::getIntegralValue<SCATRA::DifferentiationType>(params, "differentiationtype");

  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.IP().nquad; ++gpid)
  {
    // evaluate values of shape functions at current integration point
    EvalShapeFuncAndIntFac(intpoints, gpid, &normal);

    const double pseudo_contact_fac =
        CalculatePseudoContactFactor(is_pseudo_contact, eslavestress_vector, normal, funct_);

    // evaluate shape derivatives
    static CORE::LINALG::Matrix<nsd_, nen_> dsqrtdetg_dd;
    if (differentiationtype == SCATRA::DifferentiationType::disp)
      EvaluateSpatialDerivativeOfAreaIntegrationFactor(intpoints, gpid, dsqrtdetg_dd);

    // evaluate overall integration factor
    const double timefacwgt = scatraparamstimint_->TimeFac() * intpoints.IP().qwgt[gpid];
    if (timefacwgt < 0.) dserror("Integration factor is negative!");

    // loop over scalars
    for (int k = 0; k < numscal_; ++k)
    {
      // evaluate dof values at current integration point on slave and master sides of scatra-scatra
      // interface
      const double slavephiint = funct_.Dot(ephinp_[k]);
      const double masterphiint = funct_.Dot(emasterphinp[k]);

      // compute matrix contributions according to kinetic model for current scatra-scatra interface
      // coupling condition
      switch (scatraparamsboundary_->KineticModel())
      {
        // constant permeability model
        case INPAR::S2I::kinetics_constperm:
        {
          // dervivative of interface flux w.r.t. displacement
          switch (differentiationtype)
          {
            case SCATRA::DifferentiationType::disp:
            {
              // access real vector of constant permeabilities associated with current condition
              const std::vector<double>* permeabilities = scatraparamsboundary_->Permeabilities();
              if (permeabilities == nullptr)
                dserror(
                    "Cannot access vector of permeabilities for scatra-scatra interface coupling!");
              if (permeabilities->size() != static_cast<unsigned>(numscal_))
                dserror("Number of permeabilities does not match number of scalars!");

              // core linearization
              const double dN_dsqrtdetg_timefacwgt = pseudo_contact_fac * timefacwgt *
                                                     (*permeabilities)[k] *
                                                     (slavephiint - masterphiint);

              // loop over matrix columns
              for (int ui = 0; ui < nen_; ++ui)
              {
                const int fui = ui * 3;

                // loop over matrix rows
                for (int vi = 0; vi < nen_; ++vi)
                {
                  const int fvi = vi * numscal_ + k;
                  const double vi_dN_dsqrtdetg = funct_(vi) * dN_dsqrtdetg_timefacwgt;

                  // loop over spatial dimensions
                  for (int dim = 0; dim < 3; ++dim)
                    // compute linearizations w.r.t. slave-side structural displacements
                    eslavematrix(fvi, fui + dim) += vi_dN_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                }
              }
              break;
            }
            default:
            {
              dserror("Unknown primary quantity to calculate derivative");
              break;
            }
          }

          break;
        }

        default:
        {
          dserror("Kinetic model for scatra-scatra interface coupling not yet implemented!");
          break;
        }
      }  // selection of kinetic model
    }    // loop over scalars
  }      // loop over integration points
}  // DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype>::EvaluateS2ICouplingOD

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::ExtractNodeValues(
    const DRT::Discretization& discretization, DRT::Element::LocationArray& la)
{
  // extract nodal state variables associated with time t_{n+1} or t_{n+alpha_f}
  ExtractNodeValues(ephinp_, discretization, la);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::ExtractNodeValues(
    CORE::LINALG::Matrix<nen_, 1>& estate, const DRT::Discretization& discretization,
    DRT::Element::LocationArray& la, const std::string& statename, const int& nds) const
{
  // initialize matrix vector
  std::vector<CORE::LINALG::Matrix<nen_, 1>> estate_temp(1, CORE::LINALG::Matrix<nen_, 1>(true));

  // call more general routine
  ExtractNodeValues(estate_temp, discretization, la, statename, nds);

  // copy extracted state variables
  estate = estate_temp[0];
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::ExtractNodeValues(
    std::vector<CORE::LINALG::Matrix<nen_, 1>>& estate, const DRT::Discretization& discretization,
    DRT::Element::LocationArray& la, const std::string& statename, const int& nds) const
{
  // extract global state vector from discretization
  const Teuchos::RCP<const Epetra_Vector> state = discretization.GetState(nds, statename);
  if (state == Teuchos::null)
    dserror("Cannot extract state vector \"" + statename + "\" from discretization!");

  // extract nodal state variables associated with boundary element
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nen_, 1>>(*state, estate, la[nds].lm_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::CalcBoundaryIntegral(
    const DRT::FaceElement* ele, CORE::LINALG::SerialDenseVector& scalar)
{
  // initialize variable for boundary integral
  double boundaryintegral(0.);

  // get integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
  {
    // evaluate values of shape functions and boundary integration factor at current integration
    // point
    const double fac =
        DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvalShapeFuncAndIntFac(
            intpoints, iquad);

    // add contribution from current integration point to boundary integral
    boundaryintegral += fac;
  }  // loop over integration points

  // write result into result vector
  scalar(0) = boundaryintegral;
}  // DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype>::CalcBoundaryIntegral

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::CalcMatMass(
    const DRT::FaceElement* const element, CORE::LINALG::SerialDenseMatrix& massmatrix)
{
  // get integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
  {
    // evaluate values of shape functions and boundary integration factor at current integration
    // point
    const double fac =
        DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvalShapeFuncAndIntFac(
            intpoints, iquad);

    // add contribution from current integration point to element mass matrix
    for (int k = 0; k < numdofpernode_; ++k)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * numdofpernode_ + k;

        for (int ui = 0; ui < nen_; ++ui)
          massmatrix(fvi, ui * numdofpernode_ + k) += funct_(vi) * funct_(ui) * fac;
      }
    }
  }  // loop over integration points
}  // DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype>::CalcBoundaryIntegral

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::CalcRobinBoundary(
    DRT::FaceElement* ele, Teuchos::ParameterList& params, DRT::Discretization& discretization,
    DRT::Element::LocationArray& la, CORE::LINALG::SerialDenseMatrix& elemat1_epetra,
    CORE::LINALG::SerialDenseVector& elevec1_epetra, const double scalar)
{
  //////////////////////////////////////////////////////////////////////
  //              get current condition and parameters
  //////////////////////////////////////////////////////////////////////

  // get current condition
  Teuchos::RCP<DRT::Condition> cond = params.get<Teuchos::RCP<DRT::Condition>>("condition");
  if (cond == Teuchos::null) dserror("Cannot access condition 'TransportRobin'");

  // get on/off flags
  const auto* onoff = cond->Get<std::vector<int>>("onoff");

  // safety check
  if ((int)(onoff->size()) != numscal_)
  {
    dserror(
        "Mismatch in size for Robin boundary conditions, onoff has length %i, but you have %i "
        "scalars",
        onoff->size(), numscal_);
  }

  // extract prefactor and reference value from condition
  const double prefac = cond->GetDouble("prefactor");
  const double refval = cond->GetDouble("refvalue");

  //////////////////////////////////////////////////////////////////////
  //                  read nodal values
  //////////////////////////////////////////////////////////////////////

  std::vector<int>& lm = la[0].lm_;

  // ------------get values of scalar transport------------------
  // extract global state vector from discretization
  Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
  if (phinp == Teuchos::null) dserror("Cannot read state vector \"phinp\" from discretization!");

  // extract local nodal state variables from global state vector
  std::vector<CORE::LINALG::Matrix<nen_, 1>> ephinp(
      numdofpernode_, CORE::LINALG::Matrix<nen_, 1>(true));
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nen_, 1>>(*phinp, ephinp, lm);

  //////////////////////////////////////////////////////////////////////
  //                  build RHS and StiffMat
  //////////////////////////////////////////////////////////////////////

  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over all scalars
  for (int k = 0; k < numscal_; ++k)
  {
    // flag for dofs to be considered by robin conditions
    if ((*onoff)[k] == 1)
    {
      for (int gpid = 0; gpid < intpoints.IP().nquad; gpid++)
      {
        // evaluate values of shape functions and domain integration factor at current integration
        // point
        const double intfac =
            DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvalShapeFuncAndIntFac(
                intpoints, gpid);

        // evaluate reference concentration factor
        const double refconcfac = FacForRefConc(gpid, ele, params, discretization);

        // evaluate overall integration factors
        const double fac_3 = prefac * intfac * refconcfac;

        // evaluate current scalar at current integration point
        const double phinp_gp = funct_.Dot(ephinp[k]);

        // build RHS and matrix
        {
          //////////////////////////////////////////////////////////////////////
          //                  rhs
          //////////////////////////////////////////////////////////////////////
          const double vrhs = scatraparamstimint_->TimeFacRhs() * (phinp_gp - refval) * fac_3;

          for (int vi = 0; vi < nen_; ++vi)
          {
            const int fvi = vi * numscal_ + k;

            elevec1_epetra[fvi] += vrhs * funct_(vi);
          }

          //////////////////////////////////////////////////////////////////////
          //                  matrix
          //////////////////////////////////////////////////////////////////////
          for (int vi = 0; vi < nen_; ++vi)
          {
            const double vlhs = scatraparamstimint_->TimeFac() * fac_3 * funct_(vi);
            const int fvi = vi * numscal_ + k;

            for (int ui = 0; ui < nen_; ++ui)
            {
              const int fui = ui * numdofpernode_ + k;

              elemat1_epetra(fvi, fui) -= vlhs * funct_(ui);
            }
          }
        }
      }  // loop over integration points
    }    // if((*onoff)[k]==1)
    // else //in the case of "OFF", a no flux condition is automatically applied

  }  // loop over scalars
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvaluateSurfacePermeability(
    const DRT::FaceElement* ele, Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la,
    CORE::LINALG::SerialDenseMatrix& elemat1, CORE::LINALG::SerialDenseVector& elevec1)
{
  //////////////////////////////////////////////////////////////////////
  //                  read nodal values
  //////////////////////////////////////////////////////////////////////

  if (scatraparamstimint_->IsGenAlpha() or not scatraparamstimint_->IsIncremental())
    dserror("Not a valid time integration scheme!");

  std::vector<int>& lm = la[0].lm_;

  // ------------get values of scalar transport------------------
  Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
  if (phinp == Teuchos::null) dserror("Cannot get state vector 'phinp'");
  // extract local values from global vector
  std::vector<CORE::LINALG::Matrix<nen_, 1>> ephinp(
      numdofpernode_, CORE::LINALG::Matrix<nen_, 1>(true));
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nen_, 1>>(*phinp, ephinp, lm);

  //------------get membrane concentration at the interface (i.e. within the
  // membrane)------------------
  Teuchos::RCP<const Epetra_Vector> phibar = discretization.GetState("MembraneConcentration");
  if (phibar == Teuchos::null) dserror("Cannot get state vector 'MembraneConcentration'");
  // extract local values from global vector
  std::vector<CORE::LINALG::Matrix<nen_, 1>> ephibar(
      numdofpernode_, CORE::LINALG::Matrix<nen_, 1>(true));
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nen_, 1>>(*phibar, ephibar, lm);

  // ------------get values of wall shear stress-----------------------
  // get number of dofset associated with pressure related dofs
  const int ndswss = scatraparams_->NdsWss();
  if (ndswss == -1) dserror("Cannot get number of dofset of wss vector");
  Teuchos::RCP<const Epetra_Vector> wss = discretization.GetState(ndswss, "WallShearStress");
  if (wss == Teuchos::null) dserror("Cannot get state vector 'WallShearStress'");

  // determine number of velocity (and pressure) related dofs per node
  const int numwssdofpernode = la[ndswss].lm_.size() / nen_;
  // construct location vector for wss related dofs
  std::vector<int> lmwss(nsd_ * nen_, -1);
  for (int inode = 0; inode < nen_; ++inode)
    for (int idim = 0; idim < nsd_; ++idim)
      lmwss[inode * nsd_ + idim] = la[ndswss].lm_[inode * numwssdofpernode + idim];

  CORE::LINALG::Matrix<nsd_, nen_> ewss(true);
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nsd_, nen_>>(*wss, ewss, lmwss);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  // rotsymmpbc_->RotateMyValuesIfNecessary(ewss);

  //////////////////////////////////////////////////////////////////////
  //                  get current condition
  //////////////////////////////////////////////////////////////////////

  Teuchos::RCP<DRT::Condition> cond = params.get<Teuchos::RCP<DRT::Condition>>("condition");
  if (cond == Teuchos::null) dserror("Cannot access condition 'SurfacePermeability'");

  const auto* onoff = cond->Get<std::vector<int>>("onoff");

  const double perm = cond->GetDouble("permeability coefficient");

  // get flag if concentration flux across membrane is affected by local wall shear stresses: 0->no
  // 1->yes
  const bool wss_onoff = (bool)cond->GetInt("wss onoff");

  const auto* coeffs = cond->Get<std::vector<double>>("wss coeffs");

  //////////////////////////////////////////////////////////////////////
  //                  build RHS and StiffMat
  //////////////////////////////////////////////////////////////////////
  {
    // integration points and weights
    const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
        SCATRA::DisTypeToOptGaussRule<distype>::rule);

    // define vector for wss concentration values at nodes
    //  CORE::LINALG::Matrix<nen_,1> fwssnod(true);

    // loop over all scalars
    for (int k = 0; k < numdofpernode_; ++k)
    {
      // flag for dofs to be considered by membrane equations of Kedem and Katchalsky
      if ((*onoff)[k] == 1)
      {
        // loop over all integration points
        for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
        {
          const double fac = EvalShapeFuncAndIntFac(intpoints, iquad, &normal_);
          const double refconcfac = FacForRefConc(iquad, ele, params, discretization);
          // integration factor for right-hand side
          double facfac = 0.0;
          if (scatraparamstimint_->IsIncremental() and not scatraparamstimint_->IsGenAlpha())
            facfac = scatraparamstimint_->TimeFac() * fac * refconcfac;
          else
            dserror("EvaluateSurfacePermeability: Requested scheme not yet implemented");

          // scalar at integration point
          const double phi = funct_.Dot(ephinp[k]);

          // permeabilty scaling factor (depending on the norm of the wss) at integration point
          const double facWSS = WSSinfluence(ewss, wss_onoff, coeffs);

          // matrix
          for (int vi = 0; vi < nen_; ++vi)
          {
            const double vlhs = facfac * facWSS * perm * funct_(vi);
            const int fvi = vi * numdofpernode_ + k;

            for (int ui = 0; ui < nen_; ++ui)
            {
              const int fui = ui * numdofpernode_ + k;

              elemat1(fvi, fui) += vlhs * funct_(ui);
            }
          }

          // rhs
          const double vrhs = facfac * facWSS * perm * phi;

          for (int vi = 0; vi < nen_; ++vi)
          {
            const int fvi = vi * numdofpernode_ + k;

            elevec1[fvi] -= vrhs * funct_(vi);
          }
        }
      }  // if((*onoff)[k]==1)
      // else //in the case of "OFF", a no flux condition is automatically applied
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvaluateKedemKatchalsky(
    const DRT::FaceElement* ele, Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la,
    CORE::LINALG::SerialDenseMatrix& elemat1, CORE::LINALG::SerialDenseVector& elevec1)
{
  // safety checks
  if (scatraparamstimint_->IsGenAlpha() or not scatraparamstimint_->IsIncremental())
    dserror("Not a valid time integration scheme!");

  std::vector<int>& lm = la[0].lm_;

  // ------------get values of scalar transport------------------
  Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
  if (phinp == Teuchos::null) dserror("Cannot get state vector 'phinp'");
  // extract local values from global vector
  std::vector<CORE::LINALG::Matrix<nen_, 1>> ephinp(
      numdofpernode_, CORE::LINALG::Matrix<nen_, 1>(true));
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nen_, 1>>(*phinp, ephinp, lm);


  //------------get membrane concentration at the interface (i.e. within the
  // membrane)------------------
  Teuchos::RCP<const Epetra_Vector> phibar = discretization.GetState("MembraneConcentration");
  if (phibar == Teuchos::null) dserror("Cannot get state vector 'MembraneConcentration'");
  // extract local values from global vector
  std::vector<CORE::LINALG::Matrix<nen_, 1>> ephibar(
      numdofpernode_, CORE::LINALG::Matrix<nen_, 1>(true));
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nen_, 1>>(*phibar, ephibar, lm);


  //--------get values of pressure at the interface ----------------------
  // get number of dofset associated with pressure related dofs
  const int ndspres = scatraparams_->NdsPres();
  if (ndspres == -1) dserror("Cannot get number of dofset of pressure vector");
  Teuchos::RCP<const Epetra_Vector> pressure = discretization.GetState(ndspres, "Pressure");
  if (pressure == Teuchos::null) dserror("Cannot get state vector 'Pressure'");

  // determine number of velocity (and pressure) related dofs per node
  const int numveldofpernode = la[ndspres].lm_.size() / nen_;
  // construct location vector for pressure related dofs
  std::vector<int> lmpres(nen_, -1);
  for (int inode = 0; inode < nen_; ++inode)
    lmpres[inode] = la[ndspres].lm_[inode * numveldofpernode + nsd_];  // only pressure dofs

  CORE::LINALG::Matrix<nen_, 1> epressure(true);
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nen_, 1>>(*pressure, epressure, lmpres);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  // rotsymmpbc_->RotateMyValuesIfNecessary(epressure);


  // ------------get values of wall shear stress-----------------------
  // get number of dofset associated with pressure related dofs
  const int ndswss = scatraparams_->NdsWss();
  if (ndswss == -1) dserror("Cannot get number of dofset of wss vector");
  Teuchos::RCP<const Epetra_Vector> wss = discretization.GetState(ndswss, "WallShearStress");
  if (wss == Teuchos::null) dserror("Cannot get state vector 'WallShearStress'");

  // determine number of velocity (and pressure) related dofs per node
  const int numwssdofpernode = la[ndswss].lm_.size() / nen_;
  // construct location vector for wss related dofs
  std::vector<int> lmwss(nsd_ * nen_, -1);
  for (int inode = 0; inode < nen_; ++inode)
    for (int idim = 0; idim < nsd_; ++idim)
      lmwss[inode * nsd_ + idim] = la[ndswss].lm_[inode * numwssdofpernode + idim];

  CORE::LINALG::Matrix<nsd_, nen_> ewss(true);
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nsd_, nen_>>(*wss, ewss, lmwss);

  // ------------get current condition----------------------------------
  Teuchos::RCP<DRT::Condition> cond = params.get<Teuchos::RCP<DRT::Condition>>("condition");
  if (cond == Teuchos::null)
    dserror("Cannot access condition 'DESIGN SCATRA COUPLING SURF CONDITIONS'");

  const auto* onoff = cond->Get<std::vector<int>>("onoff");

  // get the standard permeability of the interface
  const double perm = cond->GetDouble("permeability coefficient");

  // get flag if concentration flux across membrane is affected by local wall shear stresses: 0->no
  // 1->yes
  const bool wss_onoff = (bool)cond->GetInt("wss onoff");
  const auto* coeffs = cond->Get<std::vector<double>>("wss coeffs");

  // hydraulic conductivity at interface
  const double conductivity = cond->GetDouble("hydraulic conductivity");

  // Staverman filtration coefficient at interface
  const double sigma = cond->GetDouble("filtration coefficient");

  ///////////////////////////////////////////////////////////////////////////
  // ------------do the actual calculations----------------------------------
  ///////////////////////////////////////////////////////////////////////////

  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over all scalars
  for (int k = 0; k < numdofpernode_; ++k)  // numdofpernode_//1
  {
    // flag for dofs to be considered by membrane equations of Kedem and Katchalsky
    if ((*onoff)[k] == 1)
    {
      // loop over all integration points
      for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
      {
        const double fac = EvalShapeFuncAndIntFac(intpoints, iquad, &normal_);

        // integration factor
        double facfac = 0.0;
        if (scatraparamstimint_->IsIncremental() and not scatraparamstimint_->IsGenAlpha())
          facfac = scatraparamstimint_->TimeFac() * fac;
        else
          dserror("Kedem-Katchalsky: Requested time integration scheme not yet implemented");

        // scalar at integration point
        const double phi = funct_.Dot(ephinp[k]);

        // pressure at integration point
        const double p = funct_.Dot(epressure);

        // mean concentration at integration point
        const double phibar_gp = funct_.Dot(ephibar[k]);

        // mean concentration at integration point
        const double facWSS = WSSinfluence(ewss, wss_onoff, coeffs);


        // matrix
        for (int vi = 0; vi < nen_; ++vi)
        {
          const double vlhs = facfac * facWSS * perm * funct_(vi);

          const int fvi = vi * numdofpernode_ + k;

          for (int ui = 0; ui < nen_; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            elemat1(fvi, fui) += vlhs * funct_(ui);
          }
        }

        // rhs
        const double vrhs =
            facfac * facWSS * (perm * phi + (1 - sigma) * phibar_gp * conductivity * p);
        // J_s =f_WSS*[perm*(phi1-phi2)+(1-sigma)*phibar*conductivity*(p1-p2)]
        // J_s := solute flux through scalar scalar interface
        // perm:=membrane permeability
        // sigma:=Staverman filtration coefficient
        // phibar_gp:= mean concentration within the membrane (for now: simply linear interpolated,
        // but other interpolations also possible) conductivity:=local hydraulic conductivity of
        // membrane

        for (int vi = 0; vi < nen_; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          elevec1[fvi] -= vrhs * funct_(vi);
        }
      }
    }  // if((*onoff)[k]==1)
    // else //in the case of "OFF", a no flux condition is automatically applied
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
double DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::WSSinfluence(
    const CORE::LINALG::Matrix<nsd_, nen_>& ewss, const bool wss_onoff,
    const std::vector<double>* coeffs)
{
  // permeabilty scaling factor at integration point
  double facWSS = 1.0;

  if (wss_onoff)
  {
    CORE::LINALG::Matrix<nsd_, 1> wss(true);
    for (int ii = 0; ii < nsd_; ii++)
      for (int jj = 0; jj < nen_; jj++) wss(ii) += ewss(ii, jj) * funct_(jj);

    // euklidian norm of act node wss
    const double wss_norm = sqrt(wss(0) * wss(0) + wss(1) * wss(1) + wss(2) * wss(2));
    facWSS = log10(1 + coeffs->at(0) / (wss_norm + coeffs->at(1))) /
             log10(2);  // empirical function (log law) to account for influence of WSS;
  }
  // else //no WSS influence

  return facWSS;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::IntegrateShapeFunctions(
    const DRT::FaceElement* ele, Teuchos::ParameterList& params,
    CORE::LINALG::SerialDenseVector& elevec1, const bool addarea)
{
  // access boundary area variable with its actual value
  double boundaryint = params.get<double>("area");

  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.IP().nquad; gpid++)
  {
    const double fac = EvalShapeFuncAndIntFac(intpoints, gpid);

    // compute integral of shape functions
    for (int node = 0; node < nen_; ++node)
      for (int k = 0; k < numdofpernode_; ++k)
        elevec1[node * numdofpernode_ + k] += funct_(node) * fac;

    // area calculation
    if (addarea) boundaryint += fac;
  }  // loop over integration points

  // add contribution to the global value
  params.set<double>("area", boundaryint);
}  // DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype>::IntegrateShapeFunctions

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
template <CORE::FE::CellType bdistype, CORE::FE::CellType pdistype>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::WeakDirichlet(DRT::FaceElement* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization,
    Teuchos::RCP<const MAT::Material> material, CORE::LINALG::SerialDenseMatrix& elemat_epetra,
    CORE::LINALG::SerialDenseVector& elevec_epetra)
{
  //------------------------------------------------------------------------
  // Dirichlet boundary condition
  //------------------------------------------------------------------------
  Teuchos::RCP<DRT::Condition> dbc = params.get<Teuchos::RCP<DRT::Condition>>("condition");

  // check of total time
  const double time = scatraparamstimint_->Time();

  // get values and spatial functions from condition
  // (assumed to be constant on element boundary)
  const auto* val = (*dbc).Get<std::vector<double>>("val");
  const auto* func = (*dbc).Get<std::vector<int>>("funct");

  // assign boundary value multiplied by time-curve factor
  double dirichval = (*val)[0];

  // spatial function number
  const int funcnum = (*func)[0];

  //------------------------------------------------------------------------
  // preliminary definitions for (boundary) and parent element and
  // evaluation of nodal values of velocity and scalar based on parent
  // element nodes
  //------------------------------------------------------------------------
  // get the parent element
  DRT::Element* pele = ele->ParentElement();

  // number of spatial dimensions regarding (boundary) element
  static const int bnsd = CORE::DRT::UTILS::DisTypeToDim<bdistype>::dim;

  // number of spatial dimensions regarding parent element
  static const int pnsd = CORE::DRT::UTILS::DisTypeToDim<pdistype>::dim;

  // number of (boundary) element nodes
  static const int bnen = CORE::DRT::UTILS::DisTypeToNumNodePerEle<bdistype>::numNodePerElement;

  // number of parent element nodes
  static const int pnen = CORE::DRT::UTILS::DisTypeToNumNodePerEle<pdistype>::numNodePerElement;

  // parent element location array
  DRT::Element::LocationArray pla(discretization.NumDofSets());
  pele->LocationVector(discretization, pla, false);

  // get number of dofset associated with velocity related dofs
  const int ndsvel = scatraparams_->NdsVel();

  // get convective (velocity - mesh displacement) velocity at nodes
  Teuchos::RCP<const Epetra_Vector> convel =
      discretization.GetState(ndsvel, "convective velocity field");
  if (convel == Teuchos::null) dserror("Cannot get state vector convective velocity");

  // determine number of velocity related dofs per node
  const int numveldofpernode = pla[ndsvel].lm_.size() / pnen;

  // construct location vector for velocity related dofs
  std::vector<int> plmvel(pnsd * pnen, -1);
  for (int inode = 0; inode < pnen; ++inode)
    for (int idim = 0; idim < pnsd; ++idim)
      plmvel[inode * pnsd + idim] = pla[ndsvel].lm_[inode * numveldofpernode + idim];

  // we deal with a nsd_-dimensional flow field
  CORE::LINALG::Matrix<pnsd, pnen> econvel(true);

  // extract local values of convective velocity field from global state vector
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<pnsd, pnen>>(*convel, econvel, plmvel);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  rotsymmpbc_->template RotateMyValuesIfNecessary<pnsd, pnen>(econvel);

  // get scalar values at parent element nodes
  Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
  if (phinp == Teuchos::null) dserror("Cannot get state vector 'phinp'");

  // extract local values from global vectors for parent element
  std::vector<CORE::LINALG::Matrix<pnen, 1>> ephinp(numscal_);
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<pnen, 1>>(*phinp, ephinp, pla[0].lm_);

  //------------------------------------------------------------------------
  // preliminary definitions for integration loop
  //------------------------------------------------------------------------
  // reshape element matrices and vectors and init to zero, construct views
  elemat_epetra.shape(pnen, pnen);
  elevec_epetra.size(pnen);
  CORE::LINALG::Matrix<pnen, pnen> emat(elemat_epetra.values(), true);
  CORE::LINALG::Matrix<pnen, 1> erhs(elevec_epetra.values(), true);

  // (boundary) element local node coordinates
  CORE::LINALG::Matrix<pnsd, bnen> bxyze(true);
  CORE::GEO::fillInitialPositionArray<bdistype, pnsd, CORE::LINALG::Matrix<pnsd, bnen>>(ele, bxyze);

  // parent element local node coordinates
  CORE::LINALG::Matrix<pnsd, pnen> pxyze(true);
  CORE::GEO::fillInitialPositionArray<pdistype, pnsd, CORE::LINALG::Matrix<pnsd, pnen>>(
      pele, pxyze);

  // coordinates of integration points for (boundary) and parent element
  CORE::LINALG::Matrix<bnsd, 1> bxsi(true);
  CORE::LINALG::Matrix<pnsd, 1> pxsi(true);

  // transposed jacobian "dx/ds" and inverse of transposed jacobian "ds/dx"
  // for parent element
  CORE::LINALG::Matrix<pnsd, pnsd> pxjm(true);
  CORE::LINALG::Matrix<pnsd, pnsd> pxji(true);

  // metric tensor for (boundary) element
  CORE::LINALG::Matrix<bnsd, bnsd> bmetrictensor(true);

  // (outward-pointing) unit normal vector to (boundary) element
  CORE::LINALG::Matrix<pnsd, 1> bnormal(true);

  // velocity vector at integration point
  CORE::LINALG::Matrix<pnsd, 1> velint;

  // gradient of scalar value at integration point
  CORE::LINALG::Matrix<pnsd, 1> gradphi;

  // (boundary) element shape functions, local and global derivatives
  CORE::LINALG::Matrix<bnen, 1> bfunct(true);
  CORE::LINALG::Matrix<bnsd, bnen> bderiv(true);
  CORE::LINALG::Matrix<bnsd, bnen> bderxy(true);

  // parent element shape functions, local and global derivatives
  CORE::LINALG::Matrix<pnen, 1> pfunct(true);
  CORE::LINALG::Matrix<pnsd, pnen> pderiv(true);
  CORE::LINALG::Matrix<pnsd, pnen> pderxy(true);

  //------------------------------------------------------------------------
  // additional matrices and vectors for mixed-hybrid formulation
  //------------------------------------------------------------------------
  // for volume integrals
  CORE::LINALG::Matrix<pnsd * pnen, pnsd * pnen> mat_s_q(true);
  CORE::LINALG::Matrix<pnsd * pnen, pnen> mat_s_gradphi(true);

  CORE::LINALG::Matrix<pnsd * pnen, 1> vec_s_gradphi(true);

  // for boundary integrals
  CORE::LINALG::Matrix<pnen, pnsd * pnen> mat_w_q_o_n(true);
  CORE::LINALG::Matrix<pnsd * pnen, pnen> mat_s_o_n_phi(true);

  CORE::LINALG::Matrix<pnsd * pnen, 1> vec_s_o_n_phi_minus_g(true);

  // inverse matrix
  CORE::LINALG::Matrix<pnsd * pnen, pnsd * pnen> inv_s_q(true);

  //------------------------------------------------------------------------
  // check whether Nitsche (default) or mixed-hybrid formulation as well as
  // preliminary definitions and computations for Nitsche stabilization term
  //------------------------------------------------------------------------
  // default is Nitsche formulation
  bool mixhyb = false;

  // stabilization parameter for Nitsche term
  const double nitsche_stab_para = (*dbc).GetDouble("TauBscaling");

  // if stabilization parameter negative: mixed-hybrid formulation
  if (nitsche_stab_para < 0.0) mixhyb = true;

  // pre-factor for adjoint-consistency term:
  // either 1.0 (adjoint-consistent, default) or -1.0 (adjoint-inconsistent)
  double gamma = 1.0;
  const auto* consistency = (*dbc).Get<std::string>("Choice of gamma parameter");
  if (*consistency == "adjoint-consistent")
    gamma = 1.0;
  else if (*consistency == "diffusive-optimal")
    gamma = -1.0;
  else
    dserror("unknown definition for gamma parameter: %s", (*consistency).c_str());

  // use one-point Gauss rule to do calculations at element center
  const CORE::DRT::UTILS::IntPointsAndWeights<bnsd> intpoints_tau(
      SCATRA::DisTypeToStabGaussRule<bdistype>::rule);

  // element surface area (1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double* gpcoord = (intpoints_tau.IP().qxg)[0];
  for (int idim = 0; idim < bnsd; idim++)
  {
    bxsi(idim) = gpcoord[idim];
  }
  CORE::DRT::UTILS::shape_function_deriv1<bdistype>(bxsi, bderiv);
  double drs = 0.0;
  CORE::DRT::UTILS::ComputeMetricTensorForBoundaryEle<bdistype>(
      bxyze, bderiv, bmetrictensor, drs, &bnormal);
  const double area = intpoints_tau.IP().qwgt[0] * drs;

  // get number of dimensions for (boundary) element (convert from int to double)
  const auto dim = (double)bnsd;

  // computation of characteristic length of (boundary) element
  // (2D: square root of element area, 1D: element length)
  const double h = std::pow(area, (1.0 / dim));

  //------------------------------------------------------------------------
  // preliminary computations for integration loop
  //------------------------------------------------------------------------
  // integration points and weights for (boundary) element and parent element
  const CORE::DRT::UTILS::IntPointsAndWeights<bnsd> bintpoints(
      SCATRA::DisTypeToOptGaussRule<bdistype>::rule);

  const CORE::DRT::UTILS::IntPointsAndWeights<pnsd> pintpoints(
      SCATRA::DisTypeToOptGaussRule<pdistype>::rule);

  // transfer integration-point coordinates of (boundary) element to parent element
  CORE::LINALG::SerialDenseMatrix pqxg(pintpoints.IP().nquad, pnsd);
  {
    CORE::LINALG::SerialDenseMatrix gps(bintpoints.IP().nquad, bnsd);

    for (int iquad = 0; iquad < bintpoints.IP().nquad; ++iquad)
    {
      const double* gpcoord = (bintpoints.IP().qxg)[iquad];

      for (int idim = 0; idim < bnsd; idim++)
      {
        gps(iquad, idim) = gpcoord[idim];
      }
    }
    if (pnsd == 2)
    {
      CORE::DRT::UTILS::BoundaryGPToParentGP2(
          pqxg, gps, pdistype, bdistype, ele->FaceParentNumber());
    }
    else if (pnsd == 3)
    {
      CORE::DRT::UTILS::BoundaryGPToParentGP3(
          pqxg, gps, pdistype, bdistype, ele->FaceParentNumber());
    }
  }

  //------------------------------------------------------------------------
  // integration loop 1: volume integrals (only for mixed-hybrid formulation)
  //------------------------------------------------------------------------
  if (mixhyb)
  {
    for (int iquad = 0; iquad < pintpoints.IP().nquad; ++iquad)
    {
      // reference coordinates of integration point from (boundary) element
      const double* gpcoord = (pintpoints.IP().qxg)[iquad];
      for (int idim = 0; idim < pnsd; idim++)
      {
        pxsi(idim) = gpcoord[idim];
      }

      // parent element shape functions and local derivatives
      CORE::DRT::UTILS::shape_function<pdistype>(pxsi, pfunct);
      CORE::DRT::UTILS::shape_function_deriv1<pdistype>(pxsi, pderiv);

      // Jacobian matrix and determinant of parent element (including check)
      pxjm.MultiplyNT(pderiv, pxyze);
      const double det = pxji.Invert(pxjm);
      if (det < 1E-16)
        dserror("GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", pele->Id(), det);

      // compute integration factor
      const double fac = pintpoints.IP().qwgt[iquad] * det;

      // compute global derivatives
      pderxy.Multiply(pxji, pderiv);

      //--------------------------------------------------------------------
      // loop over scalars (not yet implemented for more than one scalar)
      //--------------------------------------------------------------------
      // for(int k=0;k<numdofpernode_;++k)
      int k = 0;
      {
        // get viscosity
        if (material->MaterialType() == INPAR::MAT::m_scatra)
        {
          const auto* actmat = static_cast<const MAT::ScatraMat*>(material.get());

          dsassert(numdofpernode_ == 1, "more than 1 dof per node for SCATRA material");

          // get constant diffusivity
          diffus_[k] = actmat->Diffusivity();
        }
        else
          dserror("Material type is not supported");

        // gradient of current scalar value
        gradphi.Multiply(pderxy, ephinp[k]);

        // integration factor for left-hand side
        const double lhsfac = scatraparamstimint_->TimeFac() * fac;

        // integration factor for right-hand side
        double rhsfac = 0.0;
        if (scatraparamstimint_->IsIncremental() and scatraparamstimint_->IsGenAlpha())
          rhsfac = lhsfac / scatraparamstimint_->AlphaF();
        else if (not scatraparamstimint_->IsIncremental() and scatraparamstimint_->IsGenAlpha())
          rhsfac = lhsfac * (1.0 - scatraparamstimint_->AlphaF()) / scatraparamstimint_->AlphaF();
        else if (scatraparamstimint_->IsIncremental() and not scatraparamstimint_->IsGenAlpha())
          rhsfac = lhsfac;

        //--------------------------------------------------------------------
        //  matrix and vector additions due to mixed-hybrid formulation
        //--------------------------------------------------------------------
        /*
                       /         \
                  1   |   h   h  |
              - ----- |  s , q   |
                kappa |          |
                      \          / Omega
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          // const double vlhs = lhsfac*pfunct(vi);
          const double vlhs = lhsfac * (1.0 / diffus_[k]) * pfunct(vi);

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            for (int i = 0; i < pnsd; ++i)
            {
              mat_s_q(fvi * pnsd + i, fui * pnsd + i) -= vlhs * pfunct(ui);
            }
          }
        }

        /*
                       /                  \
                      |  h         /   h\  |
                    + | s  , grad | phi  | |
                      |            \    /  |
                       \                  / Omega
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          // const double vlhs = lhsfac*diffus_[k]*pfunct(vi);
          const double vlhs = lhsfac * pfunct(vi);

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            for (int i = 0; i < pnsd; ++i)
            {
              mat_s_gradphi(fvi * pnsd + i, fui) += vlhs * pderxy(i, ui);
            }
          }

          // const double vrhs = rhsfac*diffus_[k]*pfunct(vi);
          const double vrhs = rhsfac * pfunct(vi);

          for (int i = 0; i < pnsd; ++i)
          {
            vec_s_gradphi(fvi * pnsd + i) += vrhs * gradphi(i);
          }
        }
      }
    }
  }

  //------------------------------------------------------------------------
  // integration loop 2: boundary integrals
  //------------------------------------------------------------------------
  for (int iquad = 0; iquad < bintpoints.IP().nquad; ++iquad)
  {
    // reference coordinates of integration point from (boundary) element
    const double* gpcoord = (bintpoints.IP().qxg)[iquad];
    for (int idim = 0; idim < bnsd; idim++)
    {
      bxsi(idim) = gpcoord[idim];
    }

    // (boundary) element shape functions
    CORE::DRT::UTILS::shape_function<bdistype>(bxsi, bfunct);
    CORE::DRT::UTILS::shape_function_deriv1<bdistype>(bxsi, bderiv);

    // global coordinates of current integration point from (boundary) element
    CORE::LINALG::Matrix<pnsd, 1> coordgp(true);
    for (int A = 0; A < bnen; ++A)
    {
      for (int j = 0; j < pnsd; ++j)
      {
        coordgp(j) += bxyze(j, A) * bfunct(A);
      }
    }

    // reference coordinates of integration point from parent element
    for (int idim = 0; idim < pnsd; idim++)
    {
      pxsi(idim) = pqxg(iquad, idim);
    }

    // parent element shape functions and local derivatives
    CORE::DRT::UTILS::shape_function<pdistype>(pxsi, pfunct);
    CORE::DRT::UTILS::shape_function_deriv1<pdistype>(pxsi, pderiv);

    // Jacobian matrix and determinant of parent element (including check)
    pxjm.MultiplyNT(pderiv, pxyze);
    const double det = pxji.Invert(pxjm);
    if (det < 1E-16)
      dserror("GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", pele->Id(), det);

    // compute measure tensor for surface element, infinitesimal area element drs
    // and (outward-pointing) unit normal vector
    CORE::DRT::UTILS::ComputeMetricTensorForBoundaryEle<bdistype>(
        bxyze, bderiv, bmetrictensor, drs, &bnormal);

    // for nurbs elements the normal vector must be scaled with a special orientation factor!!
    if (DRT::NURBS::IsNurbs(distype)) bnormal.Scale(normalfac_);

    // compute integration factor
    const double fac = bintpoints.IP().qwgt[iquad] * drs;

    // compute global derivatives
    pderxy.Multiply(pxji, pderiv);

    //--------------------------------------------------------------------
    // check whether integration-point coordinates evaluated from
    // (boundary) and parent element match
    //--------------------------------------------------------------------
    CORE::LINALG::Matrix<pnsd, 1> check(true);
    CORE::LINALG::Matrix<pnsd, 1> diff(true);

    for (int A = 0; A < pnen; ++A)
    {
      for (int j = 0; j < pnsd; ++j)
      {
        check(j) += pxyze(j, A) * pfunct(A);
      }
    }

    diff = check;
    diff -= coordgp;

    const double norm = diff.Norm2();

    if (norm > 1e-9)
    {
      for (int j = 0; j < pnsd; ++j)
      {
        printf("%12.5e %12.5e\n", check(j), coordgp(j));
      }
      dserror("Gausspoint matching error %12.5e\n", norm);
    }

    //--------------------------------------------------------------------
    // factor for Dirichlet boundary condition given by spatial function
    //--------------------------------------------------------------------
    double functfac = 1.0;
    if (funcnum > 0)
    {
      // evaluate function at current integration point (important: a 3D position vector is
      // required)
      std::array<double, 3> coordgp3D;
      coordgp3D[0] = 0.0;
      coordgp3D[1] = 0.0;
      coordgp3D[2] = 0.0;
      for (int i = 0; i < pnsd; i++) coordgp3D[i] = coordgp(i);

      functfac = DRT::Problem::Instance()
                     ->FunctionById<DRT::UTILS::FunctionOfSpaceTime>(funcnum - 1)
                     .Evaluate(coordgp3D.data(), time, 0);
    }
    else
      functfac = 1.0;
    dirichval *= functfac;

    //--------------------------------------------------------------------
    // loop over scalars (not yet implemented for more than one scalar)
    //--------------------------------------------------------------------
    // for(int k=0;k<numdofpernode_;++k)
    int k = 0;
    {
      // get viscosity
      if (material->MaterialType() == INPAR::MAT::m_scatra)
      {
        const auto* actmat = static_cast<const MAT::ScatraMat*>(material.get());

        dsassert(numdofpernode_ == 1, "more than 1 dof per node for SCATRA material");

        // get constant diffusivity
        diffus_[k] = actmat->Diffusivity();
      }
      else
        dserror("Material type is not supported");

      // get scalar value at integration point
      const double phi = pfunct.Dot(ephinp[k]);

      // integration factor for left-hand side
      const double lhsfac = scatraparamstimint_->TimeFac() * fac;

      // integration factor for right-hand side
      double rhsfac = 0.0;
      if (scatraparamstimint_->IsIncremental() and scatraparamstimint_->IsGenAlpha())
        rhsfac = lhsfac / scatraparamstimint_->AlphaF();
      else if (not scatraparamstimint_->IsIncremental() and scatraparamstimint_->IsGenAlpha())
        rhsfac = lhsfac * (1.0 - scatraparamstimint_->AlphaF()) / scatraparamstimint_->AlphaF();
      else if (scatraparamstimint_->IsIncremental() and not scatraparamstimint_->IsGenAlpha())
        rhsfac = lhsfac;

      if (mixhyb)
      {
        //--------------------------------------------------------------------
        //  matrix and vector additions due to mixed-hybrid formulation
        //--------------------------------------------------------------------
        /*  consistency term
                    /           \
                   |  h   h     |
                 - | w , q  o n |
                   |            |
                   \            / Gamma
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          const double vlhs = lhsfac * pfunct(vi);

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            for (int i = 0; i < pnsd; ++i)
            {
              mat_w_q_o_n(fvi, fui * pnsd + i) -= vlhs * pfunct(ui) * bnormal(i);
            }
          }
        }

        /*  adjoint consistency term
                    /                 \
                   |  h          h    |
                 - | s  o n , phi - g |
                   |                  |
                   \                  / Gamma
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          const double vlhs = lhsfac * pfunct(vi);

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            for (int i = 0; i < pnsd; ++i)
            {
              mat_s_o_n_phi(fvi * pnsd + i, fui) -= vlhs * pfunct(ui) * bnormal(i);
            }
          }

          for (int i = 0; i < pnsd; ++i)
          {
            vec_s_o_n_phi_minus_g(fvi * pnsd + i) -=
                pfunct(vi) * bnormal(i) *
                (rhsfac * phi - scatraparamstimint_->TimeFac() * fac * dirichval);
          }
        }
      }
      else
      {
        // parameter alpha for Nitsche stabilization term
        const double alpha = nitsche_stab_para * diffus_[k] / h;

        // get velocity at integration point
        velint.Multiply(econvel, pfunct);

        // normal velocity
        const double normvel = velint.Dot(bnormal);

        // gradient of current scalar value
        gradphi.Multiply(pderxy, ephinp[k]);

        // gradient of current scalar value in normal direction
        const double gradphi_norm = bnormal.Dot(gradphi);

        //--------------------------------------------------------------------
        //  matrix and vector additions due to Nitsche formulation
        //--------------------------------------------------------------------
        /*  consistency term
                    /                           \
                   |  h                  h      |
                 - | w , kappa * grad(phi ) o n |
                   |                            |
                   \                            / Gamma
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          const double vlhs = lhsfac * pfunct(vi) * diffus_[k];

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            for (int i = 0; i < pnsd; ++i)
            {
              emat(fvi, fui) -= vlhs * pderxy(i, ui) * bnormal(i);
            }
          }

          const double vrhs = rhsfac * diffus_[k];

          erhs(fvi) += vrhs * pfunct(vi) * gradphi_norm;
        }

        /*  adjoint consistency term, inflow/outflow part
              / --          --                                        \
             |  |         h  |                      h           h     |
           - |  |(a o n) w  +| gamma * kappa *grad(w ) o n , phi - g  |
             |  |            |                                        |
             \  --          --                                        / Gamma_in/out
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          // compute diffusive part
          double prefac = 0.0;
          for (int i = 0; i < pnsd; ++i)
          {
            prefac += gamma * diffus_[k] * pderxy(i, vi) * bnormal(i);
          }

          // add convective part in case of inflow boundary
          if (normvel < -0.0001) prefac += normvel * pfunct(vi);

          const double vlhs = lhsfac * prefac;

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            emat(fvi, fui) -= vlhs * pfunct(ui);
          }

          erhs(fvi) += prefac * (rhsfac * phi - scatraparamstimint_->TimeFac() * fac * dirichval);
        }

        /*  stabilization term
                            /             \
                           |  h     h     |
                 + alpha * | w , phi - g  |
                           |              |
                           \              / Gamma
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          const double prefac = alpha * pfunct(vi);

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            emat(fvi, fui) += lhsfac * prefac * pfunct(ui);
          }

          erhs(fvi) -= prefac * (rhsfac * phi - scatraparamstimint_->TimeFac() * fac * dirichval);
        }
      }
    }
  }

  //------------------------------------------------------------------------
  // local condensation (only for mixed-hybrid formulation)
  //------------------------------------------------------------------------
  if (mixhyb)
  {
    // matrix inversion of flux-flux block
    inv_s_q = mat_s_q;

    CORE::LINALG::FixedSizeSerialDenseSolver<pnsd * pnen, pnsd * pnen> solver;

    solver.SetMatrix(inv_s_q);
    solver.Invert();

    // computation of matrix-matrix and matrix vector products, local assembly
    for (int vi = 0; vi < pnen; ++vi)
    {
      for (int ui = 0; ui < pnen; ++ui)
      {
        for (int rr = 0; rr < pnsd * pnen; ++rr)
        {
          for (int mm = 0; mm < pnsd * pnen; ++mm)
          {
            emat(vi, ui) -= mat_w_q_o_n(vi, rr) * inv_s_q(rr, mm) *
                            (mat_s_gradphi(mm, ui) + mat_s_o_n_phi(mm, ui));
          }
        }
      }
    }

    for (int vi = 0; vi < pnen; ++vi)
    {
      for (int rr = 0; rr < pnsd * pnen; ++rr)
      {
        for (int mm = 0; mm < pnsd * pnen; ++mm)
        {
          erhs(vi) -= mat_w_q_o_n(vi, rr) * inv_s_q(rr, mm) *
                      (-vec_s_o_n_phi_minus_g(mm) - vec_s_gradphi(mm));
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
template <CORE::FE::CellType bdistype, CORE::FE::CellType pdistype>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::ReinitCharacteristicGalerkinBoundary(
    DRT::FaceElement* ele,                           //!< transport element
    Teuchos::ParameterList& params,                  //!< parameter list
    DRT::Discretization& discretization,             //!< discretization
    Teuchos::RCP<const MAT::Material> material,      //!< material
    CORE::LINALG::SerialDenseMatrix& elemat_epetra,  //!< ele sysmat
    CORE::LINALG::SerialDenseVector& elevec_epetra   //!< ele rhs
)
{
  //------------------------------------------------------------------------
  // preliminary definitions for (boundary) and parent element and
  // evaluation of nodal values of velocity and scalar based on parent
  // element nodes
  //------------------------------------------------------------------------
  // get the parent element
  DRT::Element* pele = ele->ParentElement();

  // number of spatial dimensions regarding (boundary) element
  static const int bnsd = CORE::DRT::UTILS::DisTypeToDim<bdistype>::dim;

  // number of spatial dimensions regarding parent element
  static const int pnsd = CORE::DRT::UTILS::DisTypeToDim<pdistype>::dim;

  // number of (boundary) element nodes
  static const int bnen = CORE::DRT::UTILS::DisTypeToNumNodePerEle<bdistype>::numNodePerElement;

  // number of parent element nodes
  static const int pnen = CORE::DRT::UTILS::DisTypeToNumNodePerEle<pdistype>::numNodePerElement;

  // parent element lm vector
  std::vector<int> plm;
  std::vector<int> plmowner;
  std::vector<int> plmstride;
  pele->LocationVector(discretization, plm, plmowner, plmstride);

  // get scalar values at parent element nodes
  Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
  if (phinp == Teuchos::null) dserror("Cannot get state vector 'phinp'");
  Teuchos::RCP<const Epetra_Vector> phin = discretization.GetState("phin");
  if (phinp == Teuchos::null) dserror("Cannot get state vector 'phin'");

  // extract local values from global vectors for parent element
  std::vector<CORE::LINALG::Matrix<pnen, 1>> ephinp(numscal_);
  std::vector<CORE::LINALG::Matrix<pnen, 1>> ephin(numscal_);
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<pnen, 1>>(*phinp, ephinp, plm);
  DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<pnen, 1>>(*phin, ephin, plm);

  //------------------------------------------------------------------------
  // preliminary definitions for integration loop
  //------------------------------------------------------------------------
  // reshape element matrices and vectors and init to zero, construct views
  elemat_epetra.shape(pnen, pnen);
  elevec_epetra.size(pnen);
  CORE::LINALG::Matrix<pnen, pnen> emat(elemat_epetra.values(), true);
  CORE::LINALG::Matrix<pnen, 1> erhs(elevec_epetra.values(), true);

  // (boundary) element local node coordinates
  CORE::LINALG::Matrix<pnsd, bnen> bxyze(true);
  CORE::GEO::fillInitialPositionArray<bdistype, pnsd, CORE::LINALG::Matrix<pnsd, bnen>>(ele, bxyze);

  // parent element local node coordinates
  CORE::LINALG::Matrix<pnsd, pnen> pxyze(true);
  CORE::GEO::fillInitialPositionArray<pdistype, pnsd, CORE::LINALG::Matrix<pnsd, pnen>>(
      pele, pxyze);

  // coordinates of integration points for (boundary) and parent element
  CORE::LINALG::Matrix<bnsd, 1> bxsi(true);
  CORE::LINALG::Matrix<pnsd, 1> pxsi(true);

  // transposed jacobian "dx/ds" and inverse of transposed jacobian "ds/dx"
  // for parent element
  CORE::LINALG::Matrix<pnsd, pnsd> pxjm(true);
  CORE::LINALG::Matrix<pnsd, pnsd> pxji(true);

  // metric tensor for (boundary) element
  CORE::LINALG::Matrix<bnsd, bnsd> bmetrictensor(true);

  // (outward-pointing) unit normal vector to (boundary) element
  CORE::LINALG::Matrix<pnsd, 1> bnormal(true);

  // velocity vector at integration point
  CORE::LINALG::Matrix<pnsd, 1> velint;

  // gradient of scalar value at integration point
  CORE::LINALG::Matrix<pnsd, 1> gradphi;

  // (boundary) element shape functions, local and global derivatives
  CORE::LINALG::Matrix<bnen, 1> bfunct(true);
  CORE::LINALG::Matrix<bnsd, bnen> bderiv(true);
  CORE::LINALG::Matrix<bnsd, bnen> bderxy(true);

  // parent element shape functions, local and global derivatives
  CORE::LINALG::Matrix<pnen, 1> pfunct(true);
  CORE::LINALG::Matrix<pnsd, pnen> pderiv(true);
  CORE::LINALG::Matrix<pnsd, pnen> pderxy(true);


  // use one-point Gauss rule to do calculations at element center
  const CORE::DRT::UTILS::IntPointsAndWeights<bnsd> intpoints_tau(
      SCATRA::DisTypeToStabGaussRule<bdistype>::rule);

  // element surface area (1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double* gpcoord = (intpoints_tau.IP().qxg)[0];
  for (int idim = 0; idim < bnsd; idim++)
  {
    bxsi(idim) = gpcoord[idim];
  }
  CORE::DRT::UTILS::shape_function_deriv1<bdistype>(bxsi, bderiv);
  double drs = 0.0;
  CORE::DRT::UTILS::ComputeMetricTensorForBoundaryEle<bdistype>(
      bxyze, bderiv, bmetrictensor, drs, &bnormal);

  //------------------------------------------------------------------------
  // preliminary computations for integration loop
  //------------------------------------------------------------------------
  // integration points and weights for (boundary) element and parent element
  const CORE::DRT::UTILS::IntPointsAndWeights<bnsd> bintpoints(
      SCATRA::DisTypeToOptGaussRule<bdistype>::rule);

  const CORE::DRT::UTILS::IntPointsAndWeights<pnsd> pintpoints(
      SCATRA::DisTypeToOptGaussRule<pdistype>::rule);

  // transfer integration-point coordinates of (boundary) element to parent element
  CORE::LINALG::SerialDenseMatrix pqxg(pintpoints.IP().nquad, pnsd);
  {
    CORE::LINALG::SerialDenseMatrix gps(bintpoints.IP().nquad, bnsd);

    for (int iquad = 0; iquad < bintpoints.IP().nquad; ++iquad)
    {
      const double* gpcoord = (bintpoints.IP().qxg)[iquad];

      for (int idim = 0; idim < bnsd; idim++)
      {
        gps(iquad, idim) = gpcoord[idim];
      }
    }
    if (pnsd == 2)
    {
      CORE::DRT::UTILS::BoundaryGPToParentGP2(
          pqxg, gps, pdistype, bdistype, ele->FaceParentNumber());
    }
    else if (pnsd == 3)
    {
      CORE::DRT::UTILS::BoundaryGPToParentGP3(
          pqxg, gps, pdistype, bdistype, ele->FaceParentNumber());
    }
  }


  const double reinit_pseudo_timestepsize_factor = params.get<double>("pseudotimestepsize_factor");

  const double meshsize = GetEleDiameter<pdistype>(pxyze);

  const double pseudo_timestep_size = meshsize * reinit_pseudo_timestepsize_factor;

  //------------------------------------------------------------------------
  // integration loop: boundary integrals
  //------------------------------------------------------------------------
  for (int iquad = 0; iquad < bintpoints.IP().nquad; ++iquad)
  {
    // reference coordinates of integration point from (boundary) element
    const double* gpcoord = (bintpoints.IP().qxg)[iquad];
    for (int idim = 0; idim < bnsd; idim++)
    {
      bxsi(idim) = gpcoord[idim];
    }

    // (boundary) element shape functions
    CORE::DRT::UTILS::shape_function<bdistype>(bxsi, bfunct);
    CORE::DRT::UTILS::shape_function_deriv1<bdistype>(bxsi, bderiv);

    // global coordinates of current integration point from (boundary) element
    CORE::LINALG::Matrix<pnsd, 1> coordgp(true);
    for (int A = 0; A < bnen; ++A)
    {
      for (int j = 0; j < pnsd; ++j)
      {
        coordgp(j) += bxyze(j, A) * bfunct(A);
      }
    }

    // reference coordinates of integration point from parent element
    for (int idim = 0; idim < pnsd; idim++)
    {
      pxsi(idim) = pqxg(iquad, idim);
    }

    // parent element shape functions and local derivatives
    CORE::DRT::UTILS::shape_function<pdistype>(pxsi, pfunct);
    CORE::DRT::UTILS::shape_function_deriv1<pdistype>(pxsi, pderiv);

    // Jacobian matrix and determinant of parent element (including check)
    pxjm.MultiplyNT(pderiv, pxyze);
    const double det = pxji.Invert(pxjm);
    if (det < 1E-16)
      dserror("GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", pele->Id(), det);

    // compute measure tensor for surface element, infinitesimal area element drs
    // and (outward-pointing) unit normal vector
    CORE::DRT::UTILS::ComputeMetricTensorForBoundaryEle<bdistype>(
        bxyze, bderiv, bmetrictensor, drs, &bnormal);

    // for nurbs elements the normal vector must be scaled with a special orientation factor!!
    if (DRT::NURBS::IsNurbs(distype)) bnormal.Scale(normalfac_);

    // compute integration factor
    const double fac_surface = bintpoints.IP().qwgt[iquad] * drs;

    // compute global derivatives
    pderxy.Multiply(pxji, pderiv);

    //--------------------------------------------------------------------
    // loop over scalars (not yet implemented for more than one scalar)
    //--------------------------------------------------------------------
    for (int dofindex = 0; dofindex < numdofpernode_; ++dofindex)
    {
      //----------  --------------      |                    |
      //  mat              -1/4* dtau^2 | w, n*grad(D(psi) ) |
      //--------------------------      |                    |

      CORE::LINALG::Matrix<1, pnen> derxy_normal;
      derxy_normal.Clear();
      derxy_normal.MultiplyTN(bnormal, pderxy);

      for (int vi = 0; vi < pnen; ++vi)
      {
        const int fvi = vi * numdofpernode_ + dofindex;

        for (int ui = 0; ui < pnen; ++ui)
        {
          const int fui = ui * numdofpernode_ + dofindex;

          emat(fvi, fui) -= pfunct(vi) *
                            (fac_surface * pseudo_timestep_size * pseudo_timestep_size / 4.0) *
                            derxy_normal(0, ui);
        }
      }

      //----------  --------------      |              m     |
      //  rhs               0.5* dtau^2 | w, n*grad(psi )    |
      //--------------------------      |                    |

      // update grad_dist_n
      CORE::LINALG::Matrix<pnsd, 1> grad_dist_n(true);
      grad_dist_n.Multiply(pderxy, ephin[dofindex]);

      CORE::LINALG::Matrix<1, 1> grad_dist_n_normal(true);
      grad_dist_n_normal.MultiplyTN(bnormal, grad_dist_n);

      for (int vi = 0; vi < pnen; ++vi)
      {
        const int fvi = vi * numdofpernode_ + dofindex;

        erhs(fvi) += pfunct(vi) * pseudo_timestep_size * pseudo_timestep_size * fac_surface / 2.0 *
                     grad_dist_n_normal(0, 0);
      }


      //                    |              m+1     m  |
      //    1/4*delta_tau^2 | w, n*grad(psi   - psi ) |
      //                    |              i          |
      // update grad_dist_n
      CORE::LINALG::Matrix<pnsd, 1> grad_dist_npi(true);
      grad_dist_npi.Multiply(pderxy, ephinp[dofindex]);

      CORE::LINALG::Matrix<1, 1> grad_dist_npi_normal;
      grad_dist_npi_normal.Clear();
      grad_dist_npi_normal.MultiplyTN(bnormal, grad_dist_npi);

      double Grad_Dpsi_normal = grad_dist_npi_normal(0, 0) - grad_dist_n_normal(0, 0);


      for (int vi = 0; vi < pnen; ++vi)
      {
        const int fvi = vi * numdofpernode_ + dofindex;

        erhs(fvi) += pfunct(vi) * Grad_Dpsi_normal * fac_surface * pseudo_timestep_size *
                     pseudo_timestep_size / 4.0;
      }

    }  // loop over scalars
  }    // loop over integration points
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvaluateNodalSize(
    const DRT::FaceElement* ele, Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la,
    CORE::LINALG::SerialDenseVector& nodalsize)
{
  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(
      SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.IP().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac =
        DRT::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::EvalShapeFuncAndIntFac(
            intpoints, gpid);
    for (int vi = 0; vi < nen_; ++vi)
    {
      nodalsize[numdofpernode_ * vi] += funct_(vi, 0) * fac;
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// explicit instantiation of template methods
template void DRT::ELEMENTS::ScaTraEleBoundaryCalc<
    CORE::FE::CellType::tri3>::EvaluateS2ICouplingAtIntegrationPoint<CORE::FE::CellType::
        tri3>(const std::vector<CORE::LINALG::Matrix<nen_, 1>>&,
    const std::vector<CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::tri3>::numNodePerElement, 1>>&,
    const double, const CORE::LINALG::Matrix<nen_, 1>&,
    const CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::tri3>::numNodePerElement, 1>&,
    const CORE::LINALG::Matrix<nen_, 1>&,
    const CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::tri3>::numNodePerElement, 1>&,
    const int, const DRT::ELEMENTS::ScaTraEleParameterBoundary* const, const double, const double,
    CORE::LINALG::SerialDenseMatrix&, CORE::LINALG::SerialDenseMatrix&,
    CORE::LINALG::SerialDenseMatrix&, CORE::LINALG::SerialDenseMatrix&,
    CORE::LINALG::SerialDenseVector&, CORE::LINALG::SerialDenseVector&);
template void DRT::ELEMENTS::ScaTraEleBoundaryCalc<
    CORE::FE::CellType::tri3>::EvaluateS2ICouplingAtIntegrationPoint<CORE::FE::CellType::
        quad4>(const std::vector<CORE::LINALG::Matrix<nen_, 1>>&,
    const std::vector<CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::quad4>::numNodePerElement,
        1>>&,
    const double, const CORE::LINALG::Matrix<nen_, 1>&,
    const CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::quad4>::numNodePerElement, 1>&,
    const CORE::LINALG::Matrix<nen_, 1>&,
    const CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::quad4>::numNodePerElement, 1>&,
    const int, const DRT::ELEMENTS::ScaTraEleParameterBoundary* const, const double, const double,
    CORE::LINALG::SerialDenseMatrix&, CORE::LINALG::SerialDenseMatrix&,
    CORE::LINALG::SerialDenseMatrix&, CORE::LINALG::SerialDenseMatrix&,
    CORE::LINALG::SerialDenseVector&, CORE::LINALG::SerialDenseVector&);
template void DRT::ELEMENTS::ScaTraEleBoundaryCalc<
    CORE::FE::CellType::quad4>::EvaluateS2ICouplingAtIntegrationPoint<CORE::FE::CellType::
        tri3>(const std::vector<CORE::LINALG::Matrix<nen_, 1>>&,
    const std::vector<CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::tri3>::numNodePerElement, 1>>&,
    const double, const CORE::LINALG::Matrix<nen_, 1>&,
    const CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::tri3>::numNodePerElement, 1>&,
    const CORE::LINALG::Matrix<nen_, 1>&,
    const CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::tri3>::numNodePerElement, 1>&,
    const int, const DRT::ELEMENTS::ScaTraEleParameterBoundary* const, const double, const double,
    CORE::LINALG::SerialDenseMatrix&, CORE::LINALG::SerialDenseMatrix&,
    CORE::LINALG::SerialDenseMatrix&, CORE::LINALG::SerialDenseMatrix&,
    CORE::LINALG::SerialDenseVector&, CORE::LINALG::SerialDenseVector&);
template void DRT::ELEMENTS::ScaTraEleBoundaryCalc<
    CORE::FE::CellType::quad4>::EvaluateS2ICouplingAtIntegrationPoint<CORE::FE::CellType::
        quad4>(const std::vector<CORE::LINALG::Matrix<nen_, 1>>&,
    const std::vector<CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::quad4>::numNodePerElement,
        1>>&,
    const double, const CORE::LINALG::Matrix<nen_, 1>&,
    const CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::quad4>::numNodePerElement, 1>&,
    const CORE::LINALG::Matrix<nen_, 1>&,
    const CORE::LINALG::Matrix<
        CORE::DRT::UTILS::DisTypeToNumNodePerEle<CORE::FE::CellType::quad4>::numNodePerElement, 1>&,
    const int, const DRT::ELEMENTS::ScaTraEleParameterBoundary* const, const double, const double,
    CORE::LINALG::SerialDenseMatrix&, CORE::LINALG::SerialDenseMatrix&,
    CORE::LINALG::SerialDenseMatrix&, CORE::LINALG::SerialDenseMatrix&,
    CORE::LINALG::SerialDenseVector&, CORE::LINALG::SerialDenseVector&);

// template classes
template class DRT::ELEMENTS::ScaTraEleBoundaryCalc<CORE::FE::CellType::quad4, 3>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalc<CORE::FE::CellType::quad8, 3>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalc<CORE::FE::CellType::quad9, 3>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalc<CORE::FE::CellType::tri3, 3>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalc<CORE::FE::CellType::tri6, 3>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalc<CORE::FE::CellType::line2, 2>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalc<CORE::FE::CellType::line2, 3>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalc<CORE::FE::CellType::line3, 2>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalc<CORE::FE::CellType::nurbs3, 2>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalc<CORE::FE::CellType::nurbs9, 3>;
