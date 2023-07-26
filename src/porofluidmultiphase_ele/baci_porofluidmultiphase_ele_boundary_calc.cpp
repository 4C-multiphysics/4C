/*----------------------------------------------------------------------*/
/*! \file
 \brief implementation of evaluation routines of porofluidmultiphase boundary element

   \level 3

 *----------------------------------------------------------------------*/


#include "baci_porofluidmultiphase_ele_boundary_calc.H"

#include "baci_porofluidmultiphase_ele_parameter.H"
#include "baci_porofluidmultiphase_ele_action.H"

#include "baci_inpar_parameterlist_utils.H"

#include "baci_discretization_fem_general_utils_boundary_integration.H"

#include "baci_lib_globalproblem.H"  // for curves and functions

#include "baci_utils_singleton_owner.H"

/*----------------------------------------------------------------------*
 | singleton access method                                   vuong 08/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<distype>*
DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<distype>::Instance(
    const int numdofpernode, const std::string& disname)
{
  static auto singleton_map = CORE::UTILS::MakeSingletonMap<std::string>(
      [](const int numdofpernode, const std::string& disname)
      {
        return std::unique_ptr<PoroFluidMultiPhaseEleBoundaryCalc<distype>>(
            new PoroFluidMultiPhaseEleBoundaryCalc<distype>(numdofpernode, disname));
      });

  return singleton_map[disname].Instance(
      CORE::UTILS::SingletonAction::create, numdofpernode, disname);
}


/*----------------------------------------------------------------------*
 | protected constructor for singletons                      vuong 08/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<distype>::PoroFluidMultiPhaseEleBoundaryCalc(
    const int numdofpernode, const std::string& disname)
    : params_(DRT::ELEMENTS::PoroFluidMultiPhaseEleParameter::Instance(disname)),
      numdofpernode_(numdofpernode),
      xyze_(true),  // initialize to zero
      edispnp_(true),
      xsi_(true),
      funct_(true),
      deriv_(true),
      derxy_(true),
      normal_(true),
      velint_(true),
      metrictensor_(true)
{
  return;
}


/*----------------------------------------------------------------------*
 | setup element evaluation                                  vuong 08/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<distype>::SetupCalc(
    DRT::Element* ele, Teuchos::ParameterList& params, DRT::Discretization& discretization)
{
  // get node coordinates (we have a nsd_+1 dimensional domain!)
  CORE::GEO::fillInitialPositionArray<distype, nsd_ + 1, CORE::LINALG::Matrix<nsd_ + 1, nen_>>(
      ele, xyze_);

  return 0;
}

/*----------------------------------------------------------------------*
 * Evaluate element                                          vuong 08/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<distype>::Evaluate(DRT::Element* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization,
    DRT::Element::LocationArray& la, std::vector<Epetra_SerialDenseMatrix*>& elemat,
    std::vector<Epetra_SerialDenseVector*>& elevec)
{
  //--------------------------------------------------------------------------------
  // preparations for element
  //--------------------------------------------------------------------------------
  if (SetupCalc(ele, params, discretization) == -1) return 0;

  //--------------------------------------------------------------------------------
  // extract element based or nodal values
  //--------------------------------------------------------------------------------
  ExtractElementAndNodeValues(ele, params, discretization, la);

  // check for the action parameter
  const POROFLUIDMULTIPHASE::BoundaryAction action =
      DRT::INPUT::get<POROFLUIDMULTIPHASE::BoundaryAction>(params, "action");
  // evaluate action
  EvaluateAction(ele, params, discretization, action, la, elemat, elevec);

  return 0;
}

/*----------------------------------------------------------------------*
 | extract element based or nodal values                     vuong 08/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<distype>::ExtractElementAndNodeValues(
    DRT::Element* ele, Teuchos::ParameterList& params, DRT::Discretization& discretization,
    DRT::Element::LocationArray& la)
{
  // get additional state vector for ALE case: grid displacement
  if (params_->IsAle())
  {
    // get number of dof-set associated with displacement related dofs
    const int ndsdisp = params_->NdsDisp();

    Teuchos::RCP<const Epetra_Vector> dispnp = discretization.GetState(ndsdisp, "dispnp");
    if (dispnp == Teuchos::null) dserror("Cannot get state vector 'dispnp'");

    // determine number of displacement related dofs per node
    const int numdispdofpernode = la[ndsdisp].lm_.size() / nen_;

    // construct location vector for displacement related dofs
    std::vector<int> lmdisp((nsd_ + 1) * nen_, -1);
    for (int inode = 0; inode < nen_; ++inode)
      for (int idim = 0; idim < nsd_ + 1; ++idim)
        lmdisp[inode * (nsd_ + 1) + idim] = la[ndsdisp].lm_[inode * numdispdofpernode + idim];

    // extract local values of displacement field from global state vector
    DRT::UTILS::ExtractMyValues<CORE::LINALG::Matrix<nsd_ + 1, nen_>>(*dispnp, edispnp_, lmdisp);

    // add nodal displacements to point coordinates
    xyze_ += edispnp_;
  }
  else
    edispnp_.Clear();
}

/*----------------------------------------------------------------------*
 * Action type: Evaluate                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<distype>::EvaluateAction(DRT::Element* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization,
    POROFLUIDMULTIPHASE::BoundaryAction action, DRT::Element::LocationArray& la,
    std::vector<Epetra_SerialDenseMatrix*>& elemat, std::vector<Epetra_SerialDenseVector*>& elevec)
{
  // switch over action type
  switch (action)
  {
    case POROFLUIDMULTIPHASE::bd_calc_Neumann:
    {
      // check if the neumann conditions were set
      DRT::Condition* condition = params.get<DRT::Condition*>("condition");
      if (condition == NULL) dserror("Cannot access Neumann boundary condition!");

      // evaluate neumann loads
      EvaluateNeumann(ele, params, discretization, *condition, la, *elevec[0]);

      break;
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 | evaluate Neumann boundary condition                        vuong 08/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<distype>::EvaluateNeumann(DRT::Element* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization, DRT::Condition& condition,
    DRT::Element::LocationArray& la, Epetra_SerialDenseVector& elevec1)
{
  // integration points and weights
  const CORE::DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(
      POROFLUIDMULTIPHASE::ELEUTILS::DisTypeToOptGaussRule<distype>::rule);

  // find out whether we will use a time curve
  const double time = params_->Time();

  // get values, switches and spatial functions from the condition
  // (assumed to be constant on element boundary)
  const int numdof = condition.GetInt("numdof");
  const std::vector<int>* onoff = condition.Get<std::vector<int>>("onoff");
  const std::vector<double>* val = condition.Get<std::vector<double>>("val");
  const std::vector<int>* func = condition.Get<std::vector<int>>("funct");

  if (numdofpernode_ != numdof)
    dserror(
        "The NUMDOF you have entered in your NEUMANN CONDITION does not equal the number of "
        "scalars.");

  // integration loop
  for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
  {
    double fac = EvalShapeFuncAndIntFac(intpoints, iquad);

    // factor given by spatial function
    double functfac = 1.0;

    // determine global coordinates of current Gauss point
    const int nsd_vol_ele = nsd_ + 1;
    CORE::LINALG::Matrix<nsd_vol_ele, 1> coordgp;  // coordinate has always to be given in 3D!
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
          elevec1[node * numdofpernode_ + dof] += funct_(node) * val_fac_funct_fac;
      }  // if ((*onoff)[dof])
    }    // loop over dofs
  }      // loop over integration points

  return 0;
}

/*----------------------------------------------------------------------*
 | evaluate shape functions and int. factor at int. point     vuong 08/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
double DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<distype>::EvalShapeFuncAndIntFac(
    const CORE::DRT::UTILS::IntPointsAndWeights<nsd_>& intpoints,  ///< integration points
    const int iquad,                                               ///< id of current Gauss point
    CORE::LINALG::Matrix<1 + nsd_, 1>* normalvec  ///< normal vector at Gauss point(optional)
)
{
  // coordinates of the current integration point
  const double* gpcoord = (intpoints.IP().qxg)[iquad];
  for (int idim = 0; idim < nsd_; idim++)
  {
    xsi_(idim) = gpcoord[idim];
  }

  // shape functions and their first derivatives
  CORE::DRT::UTILS::shape_function<distype>(xsi_, funct_);
  CORE::DRT::UTILS::shape_function_deriv1<distype>(xsi_, deriv_);

  // the metric tensor and the area of an infinitesimal surface/line element
  // optional: get normal at integration point as well
  double drs(0.0);
  CORE::DRT::UTILS::ComputeMetricTensorForBoundaryEle<distype>(
      xyze_, deriv_, metrictensor_, drs, normalvec);

  // return the integration factor
  return intpoints.IP().qwgt[iquad] * drs;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes
template class DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<DRT::Element::quad4>;
template class DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<DRT::Element::quad8>;
template class DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<DRT::Element::quad9>;
template class DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<DRT::Element::tri3>;
template class DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<DRT::Element::tri6>;
template class DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<DRT::Element::line2>;
template class DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<DRT::Element::line3>;