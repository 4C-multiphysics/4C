/*----------------------------------------------------------------------*/
/*!
  \file meshfree_scatra_impl.cpp

  \brief Internal implementation of meshfree scalar transport cells

  <pre>
  Maintainer: Keijo Nissen
  nissen@lnm.mw.tum.de
  http://www.lnm.mw.tum.de
  089 - 289-15253
  </pre>
*/
/*----------------------------------------------------------------------*/

#include "meshfree_scatra_cell_calc.H"           // class declarations
#include "meshfree_scatra_cell.H"           // class declarations
#include "drt_meshfree_discret.H"           // for cast to get points
#include "drt_meshfree_node.H"              // for cast to get points
#include "drt_meshfree_cell.H"              // for cast to get points
#include "drt_meshfree_cell_utils.H"        // to get Gauss points in real space
#include "../drt_scatra_ele/scatra_ele_action.H"// for enum of scatra actions
#include "../drt_scatra_ele/scatra_ele_parameter.H"
#include "../drt_scatra_ele/scatra_ele_parameter_std.H"
#include "../drt_fem_general/drt_utils_maxent_basisfunctions.H" // basis function evaluation
#include "../drt_mat/scatra_mat.H"          // in GetMaterialParams(): type ScatraMat
#include "../drt_lib/drt_globalproblem.H"   // in BodyForce(): DRT::Problem::Instance()
#include "../drt_lib/drt_utils.H"           // in Evaluate(): ExtractMyValues()
#include "../drt_lib/drt_condition_utils.H" // in BodyForce(): FindElementConditions()

/*==========================================================================*
 * class MeshfreeScaTraCellCalc                                                 *
 *==========================================================================*/

/*--------------------------------------------------------------------------*
 |  ctor                                                 (public) nis Mar12 |
 *--------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::MeshfreeScaTraCellCalc<distype>::MeshfreeScaTraCellCalc(const int numdofpernode, const int numscal)
: numdofpernode_(numdofpernode),
  numscal_(numscal),
  nen_(),
  discret_(NULL),
  kxyz_(nsd_,nek_),
  gxyz_(nsd_,ngp_),
  gw_(ngp_),
  ephin_(numscal_),
  ephinp_(numscal_),
  ephiam_(numscal_),
  ehist_(numdofpernode_),
  evelnp_(),
  econvelnp_(),
  eaccnp_(),
  eprenp_(),
  edispnp_(true),
  funct_(),
  deriv_(),
  bodyforce_(numdofpernode_), // size of vector
  densn_(numscal_),
  densnp_(numscal_),
  densam_(numscal_),
  diffus_(numscal_),
  is_reactive_(false),
  reacoeff_(numscal_)
{
  // get parameter lists
  scatrapara_ = DRT::ELEMENTS::ScaTraEleParameterStd::Instance();
  scatraparatimint_ = DRT::ELEMENTS::ScaTraEleParameterTimInt::Instance();

  return;
}


/*--------------------------------------------------------------------------*
 |  evaluate meshfree scatra cell                        (public) nis Mar12 |
 *--------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::MeshfreeScaTraCellCalc<distype>::Evaluate(
  DRT::ELEMENTS::MeshfreeTransport* ele,
  Teuchos::ParameterList&    params,
  DRT::Discretization&       discretization,
  const std::vector<int>&    lm,
  Epetra_SerialDenseMatrix&  elemat1_epetra,
  Epetra_SerialDenseMatrix&  elemat2_epetra,
  Epetra_SerialDenseVector&  elevec1_epetra,
  Epetra_SerialDenseVector&  elevec2_epetra,
  Epetra_SerialDenseVector&  elevec3_epetra
  )
{
  // --------mandatory are performed here at first ------------

  // cast to meshfree  discretization
  discret_ = dynamic_cast<DRT::MESHFREE::MeshfreeDiscretization*>(&(discretization));
  if (discret_==NULL)
    dserror("dynamic_cast of discretization to meshfree discretization failed!");

  // cast element pointer to cell pointer to get access to point information
  DRT::MESHFREE::Cell const * cell = dynamic_cast<DRT::MESHFREE::Cell const *>(ele);
  if (cell==NULL)
    dserror("dynamic_cast of element to meshfree cell failed!");

  // get number of nodes
  nen_ = cell->NumNode();

  // get global point coordinates
  double const * ckxyz;
  for (int j=0; j<nek_; j++){
    ckxyz =  cell->Points()[j]->X();
    for (int k=0; k<nsd_; k++){
      kxyz_(k,j) = ckxyz[k];
    }
  }

  // set size of all vectors of SerialDense element arrays
  funct_.LightSize(nen_);
  deriv_.LightShape(nsd_,nen_);
  evelnp_.LightShape(nsd_,nen_);
  econvelnp_.LightShape(nsd_,nen_);
  for (int k=0;k<numscal_;++k){
    // set size of all vectors of SerialDenseVectors
    ephin_[k].LightSize(nen_); // without initialisation
    ephinp_[k].LightSize(nen_); // without initialisation
    ehist_[k].LightSize(nen_); // without initialisation
    bodyforce_[k].LightSize(nen_);
    // set size of all vectors of SerialDenseMatrices
  }
  bodyforce_[numdofpernode_-1].LightSize(nen_);

  // check for the action parameter
  const SCATRA::Action action = DRT::INPUT::get<SCATRA::Action>(params,"action");
  switch (action)
  {
  case SCATRA::calc_mat_and_rhs:
  {
    // get velocity at nodes
    const Teuchos::RCP<Epetra_MultiVector> velocity = params.get< Teuchos::RCP<Epetra_MultiVector> >("velocity field");
    DRT::UTILS::ExtractMyNodeBasedValues(cell,evelnp_,velocity,nsd_);
    const Teuchos::RCP<Epetra_MultiVector> convelocity = params.get< Teuchos::RCP<Epetra_MultiVector> >("convective velocity field");
    DRT::UTILS::ExtractMyNodeBasedValues(cell,econvelnp_,convelocity,nsd_);

    // extract local values from the global vectors
    Teuchos::RCP<const Epetra_Vector> hist = discretization.GetState("hist");
    Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (hist==Teuchos::null || phinp==Teuchos::null)
      dserror("Cannot get state vector 'hist' and/or 'phinp'");
    std::vector<double> myhist(lm.size());
    std::vector<double> myphinp(lm.size());
    DRT::UTILS::ExtractMyValues(*hist,myhist,lm);
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

    // fill all element arrays
    for (int i=0;i<nen_;++i)
    {
      int temp = i*numdofpernode_;
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        ephinp_[k][i] = myphinp[k+temp];
        // the history vectors contains information of time step t_n
        ehist_[k][i] = myhist[k+temp];
      }
    } // for i

    if (scatraparatimint_->IsGenAlpha() and not scatraparatimint_->IsIncremental())
    {
      // extract additional local values from global vector
      Teuchos::RCP<const Epetra_Vector> phin = discretization.GetState("phin");
      if (phin==Teuchos::null) dserror("Cannot get state vector 'phin'");
      std::vector<double> myphin(lm.size());
      DRT::UTILS::ExtractMyValues(*phin,myphin,lm);

      // fill element array
      for (int i=0;i<nen_;++i)
      {
        for (int k = 0; k< numscal_; ++k)
        {
          // split for each transported scalar, insert into element arrays
          ephin_[k][i] = myphin[k+(i*numdofpernode_)];
        }
      } // for i
    }

    // calculate element coefficient matrix and rhs
    Sysmat(
      cell,
      elemat1_epetra,
      elevec1_epetra);
    break;
  }
  case SCATRA::get_material_parameters:
  {
    dserror("oops!");
    // get the material
    Teuchos::RCP<MAT::Material> material = cell->Material();
    params.set("thermodynamic pressure",0.0);

    break;
  }
  default:
  {
    dserror("Not acting on action No. %i. Forgot implementation?",action);
  }
  }
  // work is done
  return 0;
}

/*--------------------------------------------------------------------------*
 |  initiates calculation of system matrix and rhs      (private) nis Mar12 |
 *--------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::MeshfreeScaTraCellCalc<distype>::Sysmat(
  const DRT::MESHFREE::Cell*            cell,
  Epetra_SerialDenseMatrix&             emat,
  Epetra_SerialDenseVector&             erhs
  )
{
  // ---------------------------------------------------------------------
  // call routine for calculation of body force in element nodes
  // (time n+alpha_F for generalized-alpha scheme, at time n+1 otherwise)
  // ---------------------------------------------------------------------
  BodyForce(cell,scatraparatimint_->Time());

  //----------------------------------------------------------------------
  // get material parameters (evaluation at element center)
  //----------------------------------------------------------------------
  if (not scatrapara_->MatGP()) GetMaterialParams(cell);

  //----------------------------------------------------------------------
  // get integrations points and weights in xyz-system
  //----------------------------------------------------------------------
  int ngp = DRT::MESHFREE::CellGaussPointInterface::Impl(distype)->GetCellGaussPointsAtX(kxyz_, gxyz_, gw_);

  LINALG::SerialDenseMatrix distng(nsd_,nen_); // matrix for distance between node and Gauss point
  DRT::Node const * const * const nodes = cell->Nodes(); // node pointer
  double const * cgxyz; // read: current Gauss xyz-coordinate
  double const * cnxyz; // read: current node xyz-coordinate
  double fac;     // current Gauss weight

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  for (int iquad=0; iquad<ngp; ++iquad)
  {
    // get xyz-coordinates and weight of current Gauss point
    cgxyz = gxyz_[iquad]; // read: current gauss xyz-coordinate
    fac = gw_[iquad];     // read: current gauss weight

    // coordinates of the current integration point
    for (int i=0; i<nen_; ++i){
      // get current node xyz-coordinate
      cnxyz = nodes[i]->X();
      for (int j=0; j<nsd_; ++j){
        // get distance between
        distng(j,i) = cnxyz[j] - cgxyz[j];
      }
    }

    // calculate basis functions and derivatives via max-ent optimization
    int error = discret_->GetMeshfreeSolutionApprox()->GetMeshfreeBasisFunction(funct_,deriv_,distng,nsd_);
    if (error) dserror("Something went wrong when calculating the meshfree basis functions.");

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    if (scatrapara_->MatGP()) GetMaterialParams(cell);

    for (int k=0;k<numscal_;++k) // deal with a system of transported scalars
    {
      // get velocity at integration point
      LINALG::SerialDenseVector velint(nsd_,false);
      LINALG::SerialDenseVector convelint(nsd_,false);
      velint.Multiply('N','N',1.0,evelnp_,funct_,0.0);
      convelint.Multiply('N','N',1.0,econvelnp_,funct_,0.0);

      // convective part in convective form: rho*u_x*N,x+ rho*u_y*N,y
      LINALG::SerialDenseVector conv(nen_,false);
      conv.Multiply('T','N',1.0,deriv_,convelint,0.0);

      // scalar at integration point at time step n+1
      const double phinp = funct_.Dot(ephinp_[k]);
      // scalar at integration point at time step n
      const double phin = funct_.Dot(ephin_[k]);

      // gradient of current scalar value
      LINALG::SerialDenseVector gradphi(nsd_,false);
      gradphi.Multiply('N','N',1.0,deriv_,ephinp_[k],0.0);

      // convective term using current scalar value
      double conv_phi(0.0);
      conv_phi = convelint.Dot(gradphi);

      // reactive part of the form: (reaction coefficient)*phi
      double rea_phi(0.0);
      rea_phi = densnp_[k]*phinp*reacoeff_[k];

      // velocity divergence required for conservative form
      double vdiv(0.0);
      if (scatrapara_->IsConservative()) GetDivergence(vdiv,evelnp_,deriv_);

      // get history data (or acceleration)
      double hist(0.0);
      hist = funct_.Dot(ehist_[k]);

      // compute rhs containing bodyforce (divided by specific heat capacity) and,
      // for temperature equation, the time derivative of thermodynamic pressure,
      // if not constant, and for temperature equation of a reactive
      // equation system, the reaction-rate term

      // TODO: this only works for non-negative basis functions
      //       create:         LINALG::SerialDenseVector::Sum()
      //       or even better: LINALG::SerialDenseVector::Sum(double)
      double rhs(0.0);
      rhs = bodyforce_[k].Dot(funct_); // normal bodyforce

      //----------------------------------------------------------------
      // 1) element matrix: stationary terms
      //----------------------------------------------------------------

      // integration factors
      const double timefacfac = scatraparatimint_->TimeFac()*fac;
      const double fac_diffus = timefacfac*diffus_[k];

      //----------------------------------------------------------------
      // standard Galerkin terms
      //----------------------------------------------------------------

      // convective term in convective form
      const double densfac = timefacfac*densnp_[k];
      for (int vi=0; vi<nen_; ++vi)
      {
        const double v = densfac*funct_(vi);
        const int fvi = vi*numdofpernode_+k;

        for (int ui=0; ui<nen_; ++ui)
        {
          const int fui = ui*numdofpernode_+k;

          emat(fvi,fui) += v*conv(ui);
        }
      }

      // addition to convective term for conservative form
      if (scatrapara_->IsConservative())
      {
        const double consfac = timefacfac*densnp_[k]*vdiv;
        for (int vi=0; vi<nen_; ++vi)
        {
          const double v = consfac*funct_(vi);
          const int fvi = vi*numdofpernode_+k;

          for (int ui=0; ui<nen_; ++ui)
          {
            const int fui = ui*numdofpernode_+k;

            emat(fvi,fui) += v*funct_(ui);
          }
        }
      }

      // diffusive term
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        for (int ui=0; ui<nen_; ++ui)
        {
          const int fui = ui*numdofpernode_+k;
          double laplawf(0.0);
          GetLaplacianWeakForm(laplawf, deriv_,ui,vi);
          emat(fvi,fui) += fac_diffus*laplawf;
        }
      }

      //----------------------------------------------------------------
      // 2) element matrix: instationary terms
      //----------------------------------------------------------------
      if (not scatraparatimint_->IsStationary())
      {
        const double densamfac = fac*densam_[k];
        //----------------------------------------------------------------
        // standard Galerkin transient term
        //----------------------------------------------------------------
        // transient term
        for (int vi=0; vi<nen_; ++vi)
        {
          const double v = densamfac*funct_(vi);
          const int fvi = vi*numdofpernode_+k;

          for (int ui=0; ui<nen_; ++ui)
          {
            const int fui = ui*numdofpernode_+k;

            emat(fvi,fui) += v*funct_(ui);
          }
        }

      }

      //----------------------------------------------------------------
      // 3) element matrix: reactive terms
      //----------------------------------------------------------------
      if (is_reactive_)
      {
        const double fac_reac        = timefacfac*densnp_[k]*reacoeff_[k];
        for (int vi=0; vi<nen_; ++vi)
        {
          const double v = fac_reac*funct_(vi);
          const int fvi = vi*numdofpernode_+k;

          for (int ui=0; ui<nen_; ++ui)
          {
            const int fui = ui*numdofpernode_+k;

            emat(fvi,fui) += v*funct_(ui);
          }
        }
      }

      //----------------------------------------------------------------
      // 4) element right hand side
      //----------------------------------------------------------------

      //----------------------------------------------------------------
      // computation of bodyforce (and potentially history) term,
      // residual, integration factors and standard Galerkin transient
      // term (if required) on right hand side depending on respective
      // (non-)incremental stationary or time-integration scheme
      //----------------------------------------------------------------
      double rhsint    = rhs;
      double rhsfac    = scatraparatimint_->TimeFacRhs() * fac;

      //----------------------------------------------------------------
      // compute rhsint
      //----------------------------------------------------------------
      if (scatraparatimint_->IsGenAlpha())
      {
        if (not scatraparatimint_->IsIncremental())
          rhsint   += densam_[k]*hist*(scatraparatimint_->AlphaF()/scatraparatimint_->TimeFac());

        rhsint   *= (scatraparatimint_->TimeFac()/scatraparatimint_->AlphaF());
      }
      else // OST, BDF2
      {
        if (not scatraparatimint_->IsStationary())
        {
          rhsint *= scatraparatimint_->TimeFac();
          rhsint += densnp_[k]*hist;
        }
      }

      //----------------------------------------------------------------
      // adaption of convective term for rhs
      //----------------------------------------------------------------
      if (scatraparatimint_->IsIncremental())
      {
        // addition to convective term for conservative form
        // (not included in residual)
        if (scatrapara_->IsConservative())
        {
          // convective term in conservative form
          conv_phi += phinp*vdiv;
        }

        // multiply convective term by density
        conv_phi *= densnp_[k];
      }
      else if (not scatraparatimint_->IsIncremental() and scatraparatimint_->IsGenAlpha())
      {
        // addition to convective term for conservative form
        // (not included in residual)
        if (scatrapara_->IsConservative())
        {
          // convective term in conservative form
          // caution: velocity divergence is for n+1 and not for n!
          // -> hopefully, this inconsistency is of small amount
          conv_phi += phin*vdiv;
        }

        // multiply convective term by density
        conv_phi *= densn_[k];
      }

      //----------------------------------------------------------------
      // standard Galerkin bodyforce term
      //----------------------------------------------------------------
      double vrhs = fac*rhsint;
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        erhs[fvi] += vrhs*funct_(vi);
      }

      //----------------------------------------------------------------
      // standard Galerkin terms on right hand side
      //----------------------------------------------------------------

      // convective term
      vrhs = rhsfac*conv_phi;
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        erhs[fvi] -= vrhs*funct_(vi);
      }

      //----------------------------------------------------------------
      // diffusive term
      //----------------------------------------------------------------
      vrhs = rhsfac*diffus_[k];
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        double laplawf(0.0);
        GetLaplacianWeakFormRHS(laplawf,deriv_,gradphi,vi);
        erhs[fvi] -= vrhs*laplawf;
      }

      //----------------------------------------------------------------
      // reactive terms (standard Galerkin) on rhs
      //----------------------------------------------------------------

      // standard Galerkin term
      if (is_reactive_)
      {
        vrhs = rhsfac*rea_phi;
        for (int vi=0; vi<nen_; ++vi)
        {
          const int fvi = vi*numdofpernode_+k;

          erhs[fvi] -= vrhs*funct_(vi);
        }
      }
    } // loop over each scalar
  }// integration loop

  return;

} //MeshfreeScaTraCellCalc::Sysmat

/*--------------------------------------------------------------------------*
 |  get the body force                                 (private)  nis Feb12 |
 |                                                                          |
 |  this funtions needs to set bodyforce_ every time even for static or     |
 |  even zero forces, since it is not an element, but an impl bodyforce     |
 *--------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::MeshfreeScaTraCellCalc<distype>::BodyForce(
  DRT::MESHFREE::Cell const *  ele,
  const double                 time
  )
{
  std::vector<DRT::Condition*> myneumcond;

  // check whether all nodes have a unique Neumann condition
  switch(nsd_){
  case 3: DRT::UTILS::FindElementConditions(ele, "VolumeNeumann" , myneumcond); break;
  case 2: DRT::UTILS::FindElementConditions(ele, "SurfaceNeumann", myneumcond); break;
  case 1: DRT::UTILS::FindElementConditions(ele, "LineNeumann"   , myneumcond); break;
  default: dserror("Illegal number of spatial dimensions: %d",nsd_);
  }

  if (myneumcond.size()>1)
    dserror("More than one Neumann condition on one node!");

  if (myneumcond.size()==1)
  {
    // check for potential time curve
    const std::vector<int>* curve  = myneumcond[0]->Get<std::vector<int> >("curve");
    int curvenum = -1;
    if (curve) curvenum = (*curve)[0];

    // initialization of time-curve factor
    double curvefac(0.0);

    // compute potential time curve or set time-curve factor to one
    if (curvenum >= 0)
    {
      // time factor (negative time indicating error)
      if (time >= 0.0)
        curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
      else dserror("Negative time in bodyforce calculation: time = %f",time);
    }
    else curvefac = 1.0;

    // get values and switches from the condition
    const std::vector<int>*    onoff = myneumcond[0]->Get<std::vector<int> >   ("onoff");
    const std::vector<double>* val   = myneumcond[0]->Get<std::vector<double> >("val"  );

    // set this condition to the bodyforce array
    for(int idof=0;idof<numdofpernode_;idof++)
    {
      for (int jnode=0; jnode<nen_; jnode++)
      {
        (bodyforce_[idof])(jnode) = (*onoff)[idof]*(*val)[idof]*curvefac;
      }
    }
  }
  else
  {
    for(int idof=0;idof<numdofpernode_;idof++)
    {
      // no bodyforce - set all entries to zero
      bodyforce_[idof].Zero();
    }
  }

  return;
} //MeshfreeScaTraCellCalc::BodyForce


/*--------------------------------------------------------------------------*
 |  get the material constants                         (private)  nis Feb12 |
 *--------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::MeshfreeScaTraCellCalc<distype>::GetMaterialParams(
  const DRT::MESHFREE::Cell*  ele
  )
{
// get the material
  Teuchos::RCP<MAT::Material> material = ele->Material();

// get diffusivity / diffusivities
  if (material->MaterialType() == INPAR::MAT::m_scatra)
  {
    const MAT::ScatraMat* actmat = static_cast<const MAT::ScatraMat*>(material.get());

    dsassert(numdofpernode_==1,"more than 1 dof per node for SCATRA material");

    // get constant diffusivity
    diffus_[0] = actmat->Diffusivity();

    // in case of reaction with (non-zero) constant coefficient:
    // read coefficient and set reaction flag to true
    reacoeff_[0] = actmat->ReaCoeff();
    if (reacoeff_[0] >  1.e-14) is_reactive_ = true;
    if (reacoeff_[0] < -1.e-14)
      dserror("Reaction coefficient for species %d is not positive: %f",0, reacoeff_[0]);

    // set density at various time steps and density gradient factor to 1.0/0.0
    densn_[0]       = 1.0;
    densnp_[0]      = 1.0;
    densam_[0]      = 1.0;
  }
  else dserror("Material type is not supported");

// check whether there is negative (physical) diffusivity
  if (diffus_[0] < -1.e-15) dserror("negative (physical) diffusivity");

  return;
} //MeshfreeScaTraCellCalc::GetMaterialParams

// template classes
template class DRT::ELEMENTS::MeshfreeScaTraCellCalc<DRT::Element::hex8>;
template class DRT::ELEMENTS::MeshfreeScaTraCellCalc<DRT::Element::tet4>;
template class DRT::ELEMENTS::MeshfreeScaTraCellCalc<DRT::Element::quad4>;
template class DRT::ELEMENTS::MeshfreeScaTraCellCalc<DRT::Element::tri3>;
template class DRT::ELEMENTS::MeshfreeScaTraCellCalc<DRT::Element::line2>;
