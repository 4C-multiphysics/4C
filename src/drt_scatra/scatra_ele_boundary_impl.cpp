/*----------------------------------------------------------------------*/
/*!
\file scatra_ele_boundary_impl.cpp

\brief Internal implementation of scalar transport boundary elements

<pre>
Maintainer: Georg Bauer
            bauer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
 */
/*----------------------------------------------------------------------*/

#if defined(D_FLUID2) || defined(D_FLUID3)
#ifdef CCADISCRET

#include <cstdlib>
#include "scatra_ele_boundary_impl.H"
#include "scatra_ele_impl.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_lib/drt_function.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"
#include "../drt_geometry/position_array.H"
#include "../drt_fem_general/drt_utils_boundary_integration.H"
// material headers
#include "../drt_mat/scatra_mat.H"
#include "../drt_mat/mixfrac.H"
#include "../drt_mat/sutherland.H"
#include "../drt_mat/arrhenius_spec.H"
#include "../drt_mat/arrhenius_temp.H"
#include "../drt_mat/arrhenius_pv.H"
#include "../drt_mat/ferech_pv.H"
#include "../drt_mat/ion.H"
#include "../drt_mat/matlist.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ScaTraBoundaryImplInterface* DRT::ELEMENTS::ScaTraBoundaryImplInterface::Impl(
    const DRT::Element* ele,
    const enum INPAR::SCATRA::ScaTraType scatratype)
{
  // we assume here, that numdofpernode is equal for every node within
  // the discretization and does not change during the computations
  const int numdofpernode = ele->NumDofPerNode(*(ele->Nodes()[0]));
  int numscal = numdofpernode;
  if (scatratype == INPAR::SCATRA::scatratype_elch_enc)
    numscal -= 1;

  switch (ele->Shape())
  {
  case DRT::Element::quad4:
  {
    static ScaTraBoundaryImpl<DRT::Element::quad4>* cp4;
    if (cp4==NULL)
      cp4 = new ScaTraBoundaryImpl<DRT::Element::quad4>(numdofpernode,numscal);
      return cp4;
  }
  case DRT::Element::quad8:
  {
    static ScaTraBoundaryImpl<DRT::Element::quad8>* cp8;
    if (cp8==NULL)
      cp8 = new ScaTraBoundaryImpl<DRT::Element::quad8>(numdofpernode,numscal);
    return cp8;
  }
  case DRT::Element::quad9:
  {
    static ScaTraBoundaryImpl<DRT::Element::quad9>* cp9;
    if (cp9==NULL)
      cp9 = new ScaTraBoundaryImpl<DRT::Element::quad9>(numdofpernode,numscal);
    return cp9;
  }
  case DRT::Element::tri3:
  {
    static ScaTraBoundaryImpl<DRT::Element::tri3>* cp3;
    if (cp3==NULL)
      cp3 = new ScaTraBoundaryImpl<DRT::Element::tri3>(numdofpernode,numscal);
      return cp3;
  }
  /*  case DRT::Element::tri6:
  {
    static ScaTraBoundaryImpl<DRT::Element::tri6>* cp6;
    if (cp6==NULL)
      cp6 = new ScaTraBoundaryImpl<DRT::Element::tri6>(numdofpernode,numscal);
    return cp6;
  }*/
  case DRT::Element::line2:
  {
    static ScaTraBoundaryImpl<DRT::Element::line2>* cl2;
    if (cl2==NULL)
      cl2 = new ScaTraBoundaryImpl<DRT::Element::line2>(numdofpernode,numscal);
      return cl2;
  }/*
  case DRT::Element::line3:
  {
    static ScaTraBoundaryImpl<DRT::Element::line3>* cl3;
    if (cl3==NULL)
      cl3 = new ScaTraBoundaryImpl<DRT::Element::line3>(numdofpernode,numscal);
    return cl3;
  }*/
  default:
    dserror("Element shape %d (%d nodes) not activated. Just do it.", ele->Shape(), ele->NumNode());
  }
  return NULL;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraBoundaryImpl<distype>::ScaTraBoundaryImpl
(int numdofpernode,
    int numscal)
    : numdofpernode_(numdofpernode),
    numscal_(numscal),
    isale_(false),
    xyze_(true),  // initialize to zero
    edispnp_(true),
    diffus_(numscal_,0),
    valence_(numscal_,0),
    shcacp_(0.0),
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
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::ScaTraBoundaryImpl<distype>::Evaluate(
    DRT::ELEMENTS::TransportBoundary* ele,
    ParameterList&             params,
    DRT::Discretization&       discretization,
    vector<int>&               lm,
    Epetra_SerialDenseMatrix&  elemat1_epetra,
    Epetra_SerialDenseMatrix&  elemat2_epetra,
    Epetra_SerialDenseVector&  elevec1_epetra,
    Epetra_SerialDenseVector&  elevec2_epetra,
    Epetra_SerialDenseVector&  elevec3_epetra
)
{
  // First, do the things that are needed for all actions:
  // get the material (of the parent element)
  DRT::ELEMENTS::Transport* parentele = ele->ParentElement();
  RefCountPtr<MAT::Material> mat = parentele->Material();
  // the type of scalar transport problem has to be provided for all actions!
  const INPAR::SCATRA::ScaTraType scatratype = params.get<INPAR::SCATRA::ScaTraType>("scatratype");
  if (scatratype == INPAR::SCATRA::scatratype_undefined)
    dserror("Element parameter SCATRATYPE has not been set!");

  // get additional state vector for ALE case: grid displacement
  isale_ = params.get<bool>("isale");
  if (isale_)
  {
    const RCP<Epetra_MultiVector> dispnp = params.get< RCP<Epetra_MultiVector> >("dispnp",null);
    if (dispnp==null) dserror("Cannot get state vector 'dispnp'");
    DRT::UTILS::ExtractMyNodeBasedValues(ele,edispnp_,dispnp,nsd_);
  }
  else edispnp_.Clear();

  // Now, check for the action parameter
  const string action = params.get<string>("action","none");
  if (action == "calc_normal_vectors")
  {
    // access the global vector
    const RCP<Epetra_MultiVector> normals = params.get< RCP<Epetra_MultiVector> >("normal vectors",null);
    if (normals == Teuchos::null) dserror("Could not access vector 'normal vectors'");

    // get node coordinates (we have a nsd_+1 dimensional domain!)
    GEO::fillInitialPositionArray<distype,nsd_+1,LINALG::Matrix<nsd_+1,iel> >(ele,xyze_);

    // in the ALE case add nodal displacements
    if (isale_) xyze_ += edispnp_;

    // determine constant normal to this element
    GetConstNormal(normal_,xyze_);

    // loop over the element nodes
    for (int j=0;j<iel;j++)
    {
      const int nodegid = (ele->Nodes()[j])->Id();
      if (normals->Map().MyGID(nodegid) )
      { // OK, the node belongs to this processor

        // scaling to a unit vector is performed on the global level after
        // assembly of nodal contributions since we have no reliable information
        // about the number of boundary elements adjacent to a node
        for (int dim=0; dim<(nsd_+1); dim++)
        {
          normals->SumIntoGlobalValue(nodegid,dim,normal_(dim));
        }
      }
      //else: the node belongs to another processor; the ghosted
      //      element will contribute the right value on that proc
    }
  }
  else if (action =="calc_elch_electrode_kinetics")
  {
    // get actual values of transported scalars
    RefCountPtr<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (phinp==null) dserror("Cannot get state vector 'phinp'");

    // extract local values from the global vector
    vector<double> ephinp(lm.size());
    DRT::UTILS::ExtractMyValues(*phinp,ephinp,lm);

    // get current condition
    Teuchos::RCP<DRT::Condition> cond = params.get<Teuchos::RCP<DRT::Condition> >("condition");
    if (cond == Teuchos::null) dserror("Cannot access condition 'ElectrodeKinetics'");

    // access parameters of the condition
    const std::string* kinetics = cond->Get<std::string>("kinetic model");
    const int    speciesid = cond->GetInt("species");
    if (speciesid < 1) dserror("species number is not >= 1");
    double       pot0 = cond->GetDouble("pot");
    const int    curvenum = cond->GetInt("curve");
    const double alphaa = cond->GetDouble("alpha_a");
    const double alphac = cond->GetDouble("alpha_c");
    double       i0 = cond->GetDouble("i0");
    if (i0 >EPS14) dserror("i0 is positive, ergo not pointing INTO the domain: %f",i0);
    const double gamma = cond->GetDouble("gamma");
    const double refcon = cond->GetDouble("refcon");
    if (refcon < EPS12) dserror("reference concentration is too small: %f",refcon);
    const double dlcapacitance = cond->GetDouble("dlcap");
    if (dlcapacitance < -EPS12) dserror("double-layer capacitance is negative: %f",dlcapacitance);
    const double frt = params.get<double>("frt"); // = F/RT

    // get control parameter from parameter list
    bool iselch(true);
    if (scatratype != INPAR::SCATRA::scatratype_elch_enc)
      iselch = false;
    const bool   is_stationary = params.get<bool>("using stationary formulation");
    const double time = params.get<double>("total time");
    double       timefac = 1.0;
    double       alphaF  = 1.0;
    // find out whether we shell use a time curve and get the factor
    // this feature can be also used for stationary "pseudo time loops"
    if (curvenum>=0)
    {
      const double curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
      // adjust potential at metal side accordingly
      pot0 *= curvefac;
    }

    const bool calc_status = params.get<bool>("calc_status",false);
    if (!calc_status)
    {
      if (not is_stationary)
      {
        // One-step-Theta:    timefac = theta*dt
        // BDF2:              timefac = 2/3 * dt
        // generalized-alpha: timefac = (gamma*alpha_F/alpha_M) * dt
        timefac = params.get<double>("time factor");
        alphaF = params.get<double>("alpha_F");
        timefac *= alphaF;
        if (timefac < 0.0) dserror("time factor is negative.");
        // we multiply i0 with timefac. So we do not have to give down this paramater
        // and perform automatically the multiplication of matrix and rhs with timefac
        i0 *= timefac;
      }

# if 0
      // print all parameters read from the current condition
      cout<<"kinetic model  = "<<*kinetics<<endl;
      cout<<"react. species = "<<speciesid<<endl;
      cout<<"pot0(mod.)     = "<<pot0<<endl;
      cout<<"pot0n          = "<<pot0n<<endl;
      cout<<"curvenum       = "<<curvenum<<endl;
      cout<<"alpha_a        = "<<alphaa<<endl;
      cout<<"alpha_c        = "<<alphac<<endl;
      cout<<"i0(mod.)       = "<<i0<<endl;
      cout<<"gamma          = "<<gamma<<endl;
      cout<<"refcon         = "<<refcon<<endl;
      cout<<"F/RT           = "<<frt<<endl<<endl;
      cout<<"time factor    = "<<timefac<<endl;
      cout<<"alpha_F        = "<<alphaF<<endl;
#endif

      EvaluateElectrodeKinetics(
          ele,
          elemat1_epetra,
          elevec1_epetra,
          ephinp,
          mat,
          speciesid,
          *kinetics,
          pot0,
          alphaa,
          alphac,
          i0,
          frt,
          gamma,
          refcon,
          iselch
      );
    }
    else
    {
      // NOTE: add integral value only for elements which are NOT ghosted!
      if(ele->Owner() == discretization.Comm().MyPID())
      {
        // get actual values of transported scalars
        RefCountPtr<const Epetra_Vector> hist = discretization.GetState("hist");
        if (hist==null) dserror("Cannot get state vector 'hist'");
        // extract local values from the global vector
        vector<double> ehist(lm.size());
        DRT::UTILS::ExtractMyValues(*hist,ehist,lm);

        double pot0hist(0.0);
        if (not is_stationary)
        {
          // One-step-Theta:    timefac = theta*dt
          // BDF2:              timefac = 2/3 * dt
          // generalized-alpha: timefac = (gamma*alpha_F/alpha_M) * dt
          timefac = params.get<double>("time factor");
          alphaF = params.get<double>("alpha_F");
          timefac *= alphaF;
          if (timefac < 0.0) dserror("time factor is negative.");
          // access history of electrode potential
          if (dlcapacitance > EPS14)
            pot0hist = cond->GetDouble("pothist");
        }

        ElectrodeStatus(
            ele,
            params,
            ephinp,
            ehist,
            *kinetics,
            speciesid,
            pot0,
            pot0hist,
            alphaa,
            alphac,
            i0,
            frt,
            gamma,
            refcon,
            iselch,
            timefac,
            dlcapacitance);}
    }
  }
  else if (action =="calc_therm_press")
  {
    // we dont know the parent element's lm vector; so we have to build it here
    const int ielparent = parentele->NumNode();
    vector<int> lmparent(ielparent);
    vector<int> lmparentowner;
    parentele->LocationVector(discretization, lmparent, lmparentowner);

    // get velocity values at nodes
    const RCP<Epetra_MultiVector> velocity = params.get< RCP<Epetra_MultiVector> >("velocity field",null);
    // we deal with a (nsd_+1)-dimensional flow field
    Epetra_SerialDenseVector evel((nsd_+1)*ielparent);
    DRT::UTILS::ExtractMyNodeBasedValues(parentele,evel,velocity,nsd_+1);

    // get values of scalar
    RefCountPtr<const Epetra_Vector> phinp  = discretization.GetState("phinp");
    if (phinp==null) dserror("Cannot get state vector 'phinp'");

    // extract local values from the global vectors for the parent(!) element
    vector<double> myphinp(lmparent.size());
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lmparent);

    // define vector for normal fluxes
    vector<double> mydiffflux(lm.size());
    vector<double> mydivu(lm.size());

    // get node coordinates (we have a nsd_+1 dimensional domain!)
    GEO::fillInitialPositionArray<distype,nsd_+1,LINALG::Matrix<nsd_+1,iel> >(ele,xyze_);

    // in the ALE case add nodal displacements
    if (isale_) xyze_ += edispnp_;

    // determine normal to this element
    GetConstNormal(normal_,xyze_);

    // compute fluxes on each node of the parent element
    LINALG::SerialDenseMatrix eflux(3,ielparent);
    DRT::Element* peleptr = (DRT::Element*) parentele;

    // set some parameters (temperature is always last degree of freedom in vector)
    double frt=0.0;
    string fluxtypestring("diffusiveflux");
    int j=numscal_-1;

    // compute elementwise flux
    DRT::ELEMENTS::ScaTraImplInterface::Impl(parentele,scatratype)->CalculateFluxSerialDense(
        eflux,
        peleptr,
        myphinp,
        frt,
        evel,
        fluxtypestring,
        j);

    for (int i=0; i<iel; ++i)
    {
      for(int k = 0; k<ielparent;++k)
      {
        // calculate normal diffusive flux and velocity div. at present node
        mydiffflux[i] = 0.0;
        mydivu[i]     = 0.0;
        for (int l=0; l<nsd_+1; l++)
        {
          mydiffflux[i] += eflux(l,k)*normal_(l);
          mydivu[i]     += evel[i*(nsd_+1)+l]*normal_(l);
        }
      }
    }

    // calculate integral of normal diffusive flux and velocity divergence
    // NOTE: add integral value only for elements which are NOT ghosted!
    if(ele->Owner() == discretization.Comm().MyPID())
    {
      DifffluxAndDivuIntegral(ele,params,mydiffflux,mydivu);
    }
  }
  else if (action =="integrate_shape_functions")
  {
    // NOTE: add area value only for elements which are NOT ghosted!
    const bool addarea = (ele->Owner() == discretization.Comm().MyPID());
    IntegrateShapeFunctions(ele,params,elevec1_epetra,addarea);
  }
  else if (action =="calc_Neumann_inflow")
  {
    // get control parameters
    is_stationary_  = params.get<bool>("using stationary formulation");
    is_genalpha_    = params.get<bool>("using generalized-alpha time integration");
    is_incremental_ = params.get<bool>("incremental solver");

    // get time factor and alpha_F if required
    // one-step-Theta:    timefac = theta*dt
    // BDF2:              timefac = 2/3 * dt
    // generalized-alpha: timefac = alphaF * (gamma*/alpha_M) * dt
    double timefac = 1.0;
    double alphaF  = 1.0;
    if (not is_stationary_)
    {
      timefac = params.get<double>("time factor");
      if (is_genalpha_)
      {
        alphaF = params.get<double>("alpha_F");
        timefac *= alphaF;
      }
      if (timefac < 0.0) dserror("time factor is negative.");
    }

    // set thermodynamic pressure and its time derivative as well as
    // flag for turbulence model if required
    thermpress_ = 0.0;
    if (scatratype==INPAR::SCATRA::scatratype_loma)
      thermpress_ = params.get<double>("thermodynamic pressure");

    // we dont know the parent element's lm vector; so we have to build it here
    const int ielparent = parentele->NumNode();
    vector<int> lmparent(ielparent);
    vector<int> lmparentowner;
    parentele->LocationVector(discretization, lmparent, lmparentowner);

    // get velocity values at nodes
    const RCP<Epetra_MultiVector> velocity = params.get< RCP<Epetra_MultiVector> >("velocity field",null);
    // we deal with a (nsd_+1)-dimensional flow field
    Epetra_SerialDenseVector evel((nsd_+1)*ielparent);
    DRT::UTILS::ExtractMyNodeBasedValues(parentele,evel,velocity,nsd_+1);

    // get values of scalar
    RefCountPtr<const Epetra_Vector> phinp  = discretization.GetState("phinp");
    if (phinp==null) dserror("Cannot get state vector 'phinp'");

    // extract local values from the global vectors for the parent(!) element
    vector<double> myphinp(lmparent.size());
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lmparent);

    // create object for density and solution array
    vector<LINALG::Matrix<iel,1> > ephinp(numscal_);
    LINALG::Matrix<nsd_+1,iel>     evelnp;

    // insert into element arrays
    for (int i=0;i<iel;++i)
    {
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each tranported scalar, insert into element arrays
        ephinp[k](i,0) = myphinp[k+(i*numdofpernode_)];
      }

      // insert velocity field into element array
      for (int idim=0 ; idim < nsd_+1 ; idim++)
      {
        evelnp(idim,i) = evel[idim + (i*nsd_+1)];
      }
    }

    NeumannInflow(ele,
                  mat,
                  ephinp,
                  evelnp,
                  elemat1_epetra,
                  elevec1_epetra,
                  timefac,
                  alphaF);
  }
  else
    dserror("Unknown type of action for Scatra boundary impl.: %s",action.c_str());

  return 0;
}


/*----------------------------------------------------------------------*
 |  Integrate a Surface/Line Neumann boundary condition       gjb 01/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::ScaTraBoundaryImpl<distype>::EvaluateNeumann(
    DRT::Element*             ele,
    ParameterList&            params,
    DRT::Discretization&      discretization,
    DRT::Condition&           condition,
    vector<int>&              lm,
    Epetra_SerialDenseVector& elevec1)
{
  // get node coordinates (we have a nsd_+1 dimensional domain!)
  GEO::fillInitialPositionArray<distype,nsd_+1,LINALG::Matrix<nsd_+1,iel> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // integrations points and weights
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // find out whether we will use a time curve
  bool usetime = true;
  const double time = params.get("total time",-1.0);
  if (time<0.0) usetime = false;

  // find out whether we will use a time curve and get the factor
  const vector<int>* curve  = condition.Get<vector<int> >("curve");
  int curvenum = -1;
  if (curve) curvenum = (*curve)[0];
  double curvefac = 1.0;
  if (curvenum>=0 && usetime)
    curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);

  // get values, switches and spatial functions from the condition
  // (assumed to be constant on element boundary)
  const vector<int>*    onoff = condition.Get<vector<int> >   ("onoff");
  const vector<double>* val   = condition.Get<vector<double> >("val"  );
  const vector<int>*    func  = condition.Get<vector<int> >   ("funct");

  // integration loop
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    double fac = EvalShapeFuncAndIntFac(intpoints,iquad,ele->Id());

    // multiply integration factor with the timecurve factor
    fac *= curvefac;

    // factor given by spatial function
    double functfac = 1.0;
    // determine global coordinates of current Gauss point
    double coordgp[nsd_];
    for (int i = 0; i< nsd_; i++)
    {
      coordgp[i] = 0.0;
      for (int j = 0; j < iel; j++)
      {
        coordgp[i] += xyze_(i,j) * funct_(j);
      }
    }

    int functnum = -1;
    const double* coordgpref = &coordgp[0]; // needed for function evaluation


    for(int dof=0;dof<numdofpernode_;dof++)
    {
      if ((*onoff)[dof]) // is this dof activated?
      {
        // factor given by spatial function
        if (func) functnum = (*func)[dof];
        {
          if (functnum>0)
          {
            // evaluate function at current gauss point
            functfac = DRT::Problem::Instance()->Funct(functnum-1).Evaluate(dof,coordgpref,0.0,NULL);
          }
          else
            functfac = 1.0;
        }

        const double val_fac_functfac = (*val)[dof]*fac*functfac;

        for (int node=0;node<iel;++node)
        {
          elevec1[node*numdofpernode_+dof] += funct_(node)*val_fac_functfac;
        }
      } // if ((*onoff)[dof])
    }
  } //end of loop over integration points

  return 0;
}


/*----------------------------------------------------------------------*
 | calculate Neumann inflow boundary conditions                vg 03/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraBoundaryImpl<distype>::NeumannInflow(
    const DRT::Element*                   ele,
    Teuchos::RCP<const MAT::Material>     material,
    const vector<LINALG::Matrix<iel,1> >& ephinp,
    const LINALG::Matrix<nsd_+1,iel>&     evelnp,
    Epetra_SerialDenseMatrix&             emat,
    Epetra_SerialDenseVector&             erhs,
    const double                          timefac,
    const double                          alphaF)
{
  // get node coordinates (we have a nsd_+1 dimensional domain!)
  GEO::fillInitialPositionArray<distype,nsd_+1,LINALG::Matrix<nsd_+1,iel> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // integrations points and weights
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // determine constant normal to this element
  GetConstNormal(normal_,xyze_);

  // integration loop
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = EvalShapeFuncAndIntFac(intpoints,iquad,ele->Id());

    for(int k=0;k<numdofpernode_;++k)
    {
      // get velocity at integration point
      velint_.Multiply(evelnp,funct_);

      // normal velocity
      const double normvel = velint_.Dot(normal_);

      if (normvel<-0.0001)
      {
        // set density to 1.0
        double dens = 1.0;

        // get density if not constant
        if (material->MaterialType() == INPAR::MAT::m_matlist)
        {
          const MAT::MatList* actmat = static_cast<const MAT::MatList*>(material.get());

          const int matid = actmat->MatID(0);
          Teuchos::RCP<const MAT::Material> singlemat = actmat->MaterialById(matid);

          if (singlemat->MaterialType() == INPAR::MAT::m_arrhenius_temp)
          {
            const MAT::ArrheniusTemp* actsinglemat = static_cast<const MAT::ArrheniusTemp*>(singlemat.get());

            // compute temperature
            const double temp = funct_.Dot(ephinp[k]);

            // compute density based on temperature and thermodynamic pressure
            dens = actsinglemat->ComputeDensity(temp,thermpress_);
          }
          else dserror("type of material found in material list is not supported");
        }
        else if (material->MaterialType() == INPAR::MAT::m_mixfrac)
        {
          const MAT::MixFrac* actmat = static_cast<const MAT::MixFrac*>(material.get());

          // compute mixture fraction
          const double mixfrac = funct_.Dot(ephinp[k]);

          // compute density based on mixture fraction
          dens = actmat->ComputeDensity(mixfrac);
        }
        else if (material->MaterialType() == INPAR::MAT::m_sutherland)
        {
          const MAT::Sutherland* actmat = static_cast<const MAT::Sutherland*>(material.get());

          // compute temperature
          const double temp = funct_.Dot(ephinp[k]);

          // compute density based on temperature and thermodynamic pressure
          dens = actmat->ComputeDensity(temp,thermpress_);
        }
        else if (material->MaterialType() == INPAR::MAT::m_arrhenius_pv)
        {
          const MAT::ArrheniusPV* actmat = static_cast<const MAT::ArrheniusPV*>(material.get());

          // compute progress variable
          const double provar = funct_.Dot(ephinp[k]);

          // compute density
          dens = actmat->ComputeDensity(provar);
        }
        else if (material->MaterialType() == INPAR::MAT::m_ferech_pv)
        {
          const MAT::FerEchPV* actmat = static_cast<const MAT::FerEchPV*>(material.get());

          // compute progress variable
          const double provar = funct_.Dot(ephinp[k]);

          // compute density
          dens = actmat->ComputeDensity(provar);
        }
        else dserror("Material type is not supported");

        // integration factor for left-hand side
        const double lhsfac = dens*normvel*timefac*fac;

        // integration factor for right-hand side
        double rhsfac    = 0.0;
        if (is_incremental_ and is_genalpha_)
          rhsfac = lhsfac/alphaF;
        else if (not is_incremental_ and is_genalpha_)
          rhsfac = lhsfac*(1.0-alphaF)/alphaF;
        else if (is_incremental_ and not is_genalpha_)
        {
          if (not is_stationary_) rhsfac = lhsfac;
          else                    rhsfac = dens*normvel*fac;
        }

        // matrix
        for (int vi=0; vi<iel; ++vi)
        {
          const double vlhs = lhsfac*funct_(vi);

          const int fvi = vi*numdofpernode_+k;

          for (int ui=0; ui<iel; ++ui)
          {
            const int fui = ui*numdofpernode_+k;

            emat(fvi,fui) -= vlhs*funct_(ui);
          }
        }

        // scalar at integration point
        const double phi = funct_.Dot(ephinp[k]);

        // rhs
        const double vrhs = rhsfac*phi;
        for (int vi=0; vi<iel; ++vi)
        {
          const int fvi = vi*numdofpernode_+k;

          erhs[fvi] += vrhs*funct_(vi);
        }
      }
    }
  }

  return;

} //ScaTraBoundaryImpl<distype>::NeumannInflow


/*----------------------------------------------------------------------*
 | evaluate shape functions and int. factor at int. point     gjb 01/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
double DRT::ELEMENTS::ScaTraBoundaryImpl<distype>::EvalShapeFuncAndIntFac(
    const DRT::UTILS::IntPointsAndWeights<nsd_>& intpoints,  ///< integration points
    const int                                    iquad,      ///< id of current Gauss point
    const int                                    eleid       ///< the element id
)
{
  // coordinates of the current integration point
  const double* gpcoord = (intpoints.IP().qxg)[iquad];
  for (int idim=0;idim<nsd_;idim++)
  {xsi_(idim) = gpcoord[idim];}

  // shape functions and their first derivatives
  DRT::UTILS::shape_function<distype>(xsi_,funct_);
  DRT::UTILS::shape_function_deriv1<distype>(xsi_,deriv_);

  // the metric tensor and the area of an infinitesimal surface/line element
  double drs(0.0);
  DRT::UTILS::ComputeMetricTensorForBoundaryEle<distype>(xyze_,deriv_,metrictensor_,drs);

  // return the integration factor
  return intpoints.IP().qwgt[iquad] * drs;
}


/*----------------------------------------------------------------------*
 | evaluate an electrode kinetics boundary condition (private) gjb 01/09|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraBoundaryImpl<distype>::EvaluateElectrodeKinetics(
    const DRT::Element*        ele,
    Epetra_SerialDenseMatrix& emat,
    Epetra_SerialDenseVector& erhs,
    const vector<double>&   ephinp,
    Teuchos::RCP<const MAT::Material> material,
    const int            speciesid,
    const std::string     kinetics,
    const double              pot0,
    const double            alphaa,
    const double            alphac,
    const double                i0,
    const double               frt,
    const double             gamma,
    const double            refcon,
    const bool              iselch
)
{
  //for pre-multiplication of i0 with 1/(F z_k)
  double fz = 1.0/96485.3399; // unit of F: C/mol or mC/mmol or muC / mumol

  // get node coordinates (we have a nsd_+1 dimensional domain!)
  GEO::fillInitialPositionArray<distype,nsd_+1,LINALG::Matrix<nsd_+1,iel> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // index of reactive species (starting from zero)
  const int k = speciesid-1;

  if (iselch) // this is not necessary for secondary current distributions
  {
    // get valence of the single(!) reactant
    if (material->MaterialType() == INPAR::MAT::m_matlist)
    {
      const MAT::MatList* actmat = static_cast<const MAT::MatList*>(material.get());

      const int matid = actmat->MatID(k);
      Teuchos::RCP<const MAT::Material> singlemat = actmat->MaterialById(matid);
      if (singlemat->MaterialType() == INPAR::MAT::m_ion)
      {
        const MAT::Ion* actsinglemat = static_cast<const MAT::Ion*>(singlemat.get());
        fz = fz/actsinglemat->Valence();
      }
      else
        dserror("single material type is not 'ion'");
    }
    else
      dserror("material type is not a 'matlist' material");
  }

  // integrations points and weights
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // concentration values of reactive species at element nodes
  LINALG::Matrix<iel,1> conreact(true);

  // el. potential values at element nodes
  LINALG::Matrix<iel,1> pot(true);
  if(iselch)
  {
    for (int inode=0; inode< iel;++inode)
    {
      conreact(inode) = ephinp[inode*numdofpernode_+k];
      pot(inode) = ephinp[inode*numdofpernode_+numscal_];
    }
  }
  else
  {
    for (int inode=0; inode< iel;++inode)
    {
      conreact(inode) = 1.0;
      pot(inode) = ephinp[inode*numdofpernode_];
    }
  }

  // concentration of active species at integration point
  double conint(0.0);
  // el. potential at integration point
  double potint(0.0);
  // a 'working variable'
  double fac_fz_i0_funct_vi(0.0);


 /*----------------------------------------------------------------------*
  |               start loop over integration points                     |
  *----------------------------------------------------------------------*/
  for (int gpid=0; gpid<intpoints.IP().nquad; gpid++)
  {
    const double fac = EvalShapeFuncAndIntFac(intpoints,gpid,ele->Id());

    // elch-specific values at integration point:
    conint = funct_.Dot(conreact);
    potint = funct_.Dot(pot);

    // surface overpotential eta at integration point
    const double eta = (pot0 - potint);

    if (iselch) // tertiary current distribution
    {
      // concentration-dependent Butler-Volmer law(s)
      if ((kinetics=="Butler-Volmer") or (kinetics=="Butler-Volmer-Yang1997"))
      {
        double pow_conint_gamma_k = 0.0;

#ifdef DEBUG
        // some safety checks/ user warnings
        if ((alphaa*frt*eta) > 100.0)
          cout<<"WARNING: Exp(alpha_a...) in Butler-Volmer law is near overflow!"
          <<exp(alphaa*frt*eta)<<endl;
        if (((-alphac)*frt*eta) > 100.0)
          cout<<"WARNING: Exp(alpha_c...) in Butler-Volmer law is near overflow!"
          <<exp((-alphac)*frt*eta)<<endl;
#endif
        if ((conint/refcon) < EPS13)
        {
          pow_conint_gamma_k = pow(EPS13,gamma);
#ifdef DEBUG
          cout<<"WARNING: Rel. Conc. in Butler-Volmer formula is zero/negative: "<<(conint/refcon)<<endl;
          cout<<"-> Replacement value: pow(EPS,gamma) = "<< pow_conint_gamma_k <<endl;
#endif
        }
        else
          pow_conint_gamma_k = pow(conint/refcon,gamma);
        if (kinetics=="Butler-Volmer")
        {
        // note: gamma==0 deactivates concentration dependency in Butler-Volmer!
        const double expterm = exp(alphaa*frt*eta)-exp((-alphac)*frt*eta);

        double concterm = 0.0;
        if (conint > EPS13)
          concterm = gamma*pow(conint,(gamma-1.0))/pow(refcon,gamma);
        else
          concterm = gamma*pow(EPS13,(gamma-1.0))/pow(refcon,gamma);

        for (int vi=0; vi<iel; ++vi)
        {
          fac_fz_i0_funct_vi = fac*fz*i0*funct_(vi);
          // ---------------------matrix
          for (int ui=0; ui<iel; ++ui)
          {
            emat(vi*numdofpernode_+k,ui*numdofpernode_+k) += fac_fz_i0_funct_vi*concterm*funct_(ui)*expterm;
            emat(vi*numdofpernode_+k,ui*numdofpernode_+numscal_) += fac_fz_i0_funct_vi*pow_conint_gamma_k*(((-alphaa)*frt*exp(alphaa*frt*eta))+((-alphac)*frt*exp((-alphac)*frt*eta)))*funct_(ui);
          }
          // ------------right-hand-side
          erhs[vi*numdofpernode_+k] -= fac_fz_i0_funct_vi*pow_conint_gamma_k*expterm;
        }
        } // if (kinetics=="Butler-Volmer")
        if (kinetics=="Butler-Volmer-Yang1997")
        {
          // note: gamma==0 deactivates concentration dependency in Butler-Volmer!
          double concterm = 0.0;
          if ((conint/refcon) > EPS13)
            concterm = gamma*pow(conint,(gamma-1.0))/pow(refcon,gamma);
          else
            concterm = gamma*pow(EPS13,(gamma-1.0))/pow(refcon,gamma);

          for (int vi=0; vi<iel; ++vi)
          {
            fac_fz_i0_funct_vi = fac*fz*i0*funct_(vi);
            // ---------------------matrix
            for (int ui=0; ui<iel; ++ui)
            {
              emat(vi*numdofpernode_+k,ui*numdofpernode_+k) += fac_fz_i0_funct_vi*funct_(ui)*(-(concterm*exp((-alphac)*frt*eta)));
              emat(vi*numdofpernode_+k,ui*numdofpernode_+numscal_) += fac_fz_i0_funct_vi*(((-alphaa)*frt*exp(alphaa*frt*eta))+(pow_conint_gamma_k*(-alphac)*frt*exp((-alphac)*frt*eta)))*funct_(ui);
            }
            // ------------right-hand-side
            erhs[vi*numdofpernode_+k] -= fac_fz_i0_funct_vi*(exp(alphaa*frt*eta)-(pow_conint_gamma_k*exp((-alphac)*frt*eta)));
          }
        } // if (kinetics=="Butler-Volmer-Yang1997")
      }
      else if(kinetics=="linear") // linear law:  i_n = i_0*(alphaa*frt*(V_M - phi) + 1.0)
      {
        for (int vi=0; vi<iel; ++vi)
        {
          fac_fz_i0_funct_vi = fac*fz*i0*funct_(vi);
          // ---------------------matrix
          for (int ui=0; ui<iel; ++ui)
          {
            emat(vi*numdofpernode_+k,ui*numdofpernode_+numscal_) += fac_fz_i0_funct_vi*(-alphaa)*frt*funct_(ui);
          }
          // ------------right-hand-side
          erhs[vi*numdofpernode_+k] -= fac_fz_i0_funct_vi*(alphaa*frt*eta + 1.0);
        }
      }
      else
        dserror("Kinetic model not implemented: %s",kinetics.c_str());
    }
    else // secondary current distribution
    {
      if (kinetics=="Butler-Volmer")  // concentration-dependent Butler-Volmer law
      {
#ifdef DEBUG
        // some safety checks/ user warnings
        if ((alphaa*eta) > 100.0)
          cout<<"WARNING: Exp(alpha_a...) in Butler-Volmer law is near overflow!"
          <<exp(alphaa*eta)<<endl;
        if (((-alphac)*eta) > 100.0)
          cout<<"WARNING: Exp(alpha_c...) in Butler-Volmer law is near overflow!"
          <<exp((-alphac)*eta)<<endl;
#endif
        // Butler-Volmer kinetics
        const double expterm = exp(alphaa*eta)-exp((-alphac)*eta);
        const double exptermderiv = (((-alphaa)*exp(alphaa*eta))+((-alphac)*exp((-alphac)*eta)));

        for (int vi=0; vi<iel; ++vi)
        {
          const double fac_i0_funct_vi = fac*i0*funct_(vi);
          // ---------------------matrix
          for (int ui=0; ui<iel; ++ui)
          {
            emat(vi*numdofpernode_,ui*numdofpernode_) += fac_i0_funct_vi*exptermderiv*funct_(ui);
          }
          // ------------right-hand-side
          erhs[vi*numdofpernode_] -= fac_i0_funct_vi*expterm;
        }
      }
      else if(kinetics == "Tafel") // Tafel kinetics
      {
#ifdef DEBUG
        // some safety checks/ user warnings
        if ((-alphac)*eta > 100.0)
          cout<<"WARNING: Exp(alpha_c...) in Tafel law is near overflow!"
          <<exp((-alphac)*eta)<<endl;
#endif
        const double expterm = -exp((-alphac)*eta);
        const double exptermderiv = alphac*expterm; // do not forget the (-1) from differentiation of eta!

        for (int vi=0; vi<iel; ++vi)
        {
          const double fac_i0_funct_vi = fac*i0*funct_(vi);
          // ---------------------matrix
          for (int ui=0; ui<iel; ++ui)
          {
            emat(vi*numdofpernode_,ui*numdofpernode_) += fac_i0_funct_vi*exptermderiv*funct_(ui);
          }
          // ------------right-hand-side
          erhs[vi*numdofpernode_] -= fac_i0_funct_vi*expterm;
        }
      }
      else
        dserror("Kinetic model not implemented: %s",kinetics.c_str());
    } // if iselch

  } // end of loop over integration points gpid

  return;

} // ScaTraBoundaryImpl<distype>::EvaluateElectrodeKinetics()


/*----------------------------------------------------------------------*
 | calculate electrode kinetics status information             gjb 01/09|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraBoundaryImpl<distype>::ElectrodeStatus(
    const DRT::Element*        ele,
    ParameterList&          params,
    const vector<double>&   ephinp,
    const vector<double>&    ehist,
    const std::string     kinetics,
    const int            speciesid,
    const double              pot0,
    const double          pot0hist,
    const double            alphaa,
    const double            alphac,
    const double                i0,
    const double               frt,
    const double             gamma,
    const double            refcon,
    const bool              iselch,
    const double           timefac,
    const double     dlcapacitance
)
{
  // get node coordinates (we have a nsd_+1 dimensional domain!)
  GEO::fillInitialPositionArray<distype,nsd_+1,LINALG::Matrix<nsd_+1,iel> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // get variables with their current values
  double currentintegral  = params.get<double>("currentintegral");
  double boundaryint      = params.get<double>("boundaryintegral");
  double overpotentialint = params.get<double>("overpotentialintegral");
  double concentrationint = params.get<double>("concentrationintegral");
  double currderiv        = params.get<double>("currentderiv");
  double currentresidual  = params.get<double>("currentresidual");

  // integrations points and weights
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // concentration values of reactive species at element nodes
  LINALG::Matrix<iel,1> conreact(true);

  // el. potential values at element nodes
  LINALG::Matrix<iel,1> pot(true);
  LINALG::Matrix<iel,1> pothist(true);
  if(iselch)
  {
    for (int inode=0; inode< iel;++inode)
    {
      conreact(inode) = ephinp[inode*numdofpernode_+(speciesid-1)];
      pot(inode) = ephinp[inode*numdofpernode_+numscal_];
      pothist(inode) = ehist[inode*numdofpernode_+numscal_];
    }
  }
  else
  {
    for (int inode=0; inode< iel;++inode)
    {
      conreact(inode) = 1.0;
      pot(inode) = ephinp[inode*numdofpernode_];
      pothist(inode) = ehist[inode*numdofpernode_];
    }
  }

  // concentration of active species at integration point
  double conint;
  // el. potential at integration point
  double potint;
  // history term of el. potential at integration point
  double pothistint;

  // loop over integration points
  for (int gpid=0; gpid<intpoints.IP().nquad; gpid++)
  {
    const double fac = EvalShapeFuncAndIntFac(intpoints,gpid,ele->Id());

    // elch-specific values at integration point:
    conint = funct_.Dot(conreact);
    potint = funct_.Dot(pot);
    pothistint = funct_.Dot(pothist);

    // surface overpotential eta at integration point
    const double eta = (pot0 - potint);

    // linearization of current w.r.t applied electrode potential "pot0"
    double linea(0.0);

    if ((kinetics=="Butler-Volmer") or (kinetics=="Butler-Volmer-Yang1997"))
    {
      double expterm(0.0);
      if (iselch)
      {
        if (kinetics=="Butler-Volmer")
        {
          expterm = pow(conint/refcon,gamma) * (exp(alphaa*frt*eta)-exp((-alphac)*frt*eta));
          linea = pow(conint/refcon,gamma) * frt*((alphaa*exp(alphaa*frt*eta)) + (alphac*exp((-alphac)*frt*eta)));
        }
        if (kinetics=="Butler-Volmer-Yang1997")
        {
          if (((conint/refcon)<EPS13) && (gamma < 1.0))
          {// prevents NaN's in the current density evaluation
            expterm = (exp(alphaa*frt*eta)-(pow(EPS13/refcon,gamma)*exp((-alphac)*frt*eta)));
            linea = ((alphaa)*frt*exp(alphaa*frt*eta))+(pow(EPS13/refcon,gamma)*alphac*frt*exp((-alphac)*frt*eta));
          }
          else
          {
            expterm = (exp(alphaa*frt*eta)-(pow(conint/refcon,gamma)*exp((-alphac)*frt*eta)));
            linea = ((alphaa)*frt*exp(alphaa*frt*eta))+(pow(conint/refcon,gamma)*alphac*frt*exp((-alphac)*frt*eta));
          }
        }
      }
      else // secondary current distribution
      {
        expterm = exp(alphaa*eta)-exp((-alphac)*eta);
        linea = (alphaa*exp(alphaa*eta))+(alphac*exp((-alphac)*eta));
      }

      // compute integrals
      overpotentialint += eta * fac;
      currentintegral += (-i0) * expterm * fac; // the negative(!) normal flux density
      boundaryint += fac;
      concentrationint += conint*fac;

      // tangent and rhs for galvanostatic problems
      currderiv += i0*linea*timefac*fac;
      currentresidual += (-i0) * expterm * timefac *fac;

      if (dlcapacitance > EPS12)
      {
      // add contributions due to double-layer capacitance
        currderiv -= dlcapacitance*fac;
        currentresidual += dlcapacitance*(pot0-pot0hist)*fac;
        currentresidual += fac*dlcapacitance*(-potint+pothistint);
      }

    }
    else if (iselch && (kinetics=="linear")) // linear: i_n = i_0*(alphaa*frt*eta + 1.0)
    {
      // compute integrals
      overpotentialint += eta * fac;
      currentintegral += (-i0) * (alphaa*frt*eta + 1.0) * fac; // the negative(!) normal flux density
      boundaryint += fac;
      concentrationint += conint*fac;
    }
    else if ((!iselch) && (kinetics=="Tafel"))
    {
      // secondary current distribution with Tafel kinetics
      double expterm = -exp((-alphac)*eta);
      //linea = (-alphac)*exp((-alphac)*eta);

      // compute integrals
      overpotentialint += eta * fac;
      currentintegral += (-i0) * expterm * fac; // the negative(!) normal flux density
      boundaryint += fac;
      //concentrationint += conint*fac_;
    }
    else
      dserror("Kinetic model not implemented: %s",kinetics.c_str());

  } // loop over integration points

  // add contributions to the global values
  params.set<double>("currentintegral",currentintegral);
  params.set<double>("boundaryintegral",boundaryint);
  params.set<double>("overpotentialintegral",overpotentialint);
  params.set<double>("concentrationintegral",concentrationint);
  params.set<double>("currentderiv",currderiv);
  params.set<double>("currentresidual",currentresidual);

} //ScaTraBoundaryImpl<distype>::ElectrodeStatus


/*----------------------------------------------------------------------*
 | get constant normal                                        gjb 01/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraBoundaryImpl<distype>::GetConstNormal(
    LINALG::Matrix<nsd_+1,1>&          normal,
    const LINALG::Matrix<nsd_+1,iel>&  xyze
    )
{
  // determine normal to this element
  switch(nsd_)
  {
  case 2:
  {
    LINALG::Matrix<3,1> dist1(true), dist2(true);
    for (int i=0; i<3; i++)
    {
      dist1(i) = xyze(i,1)-xyze(i,0);
      dist2(i) = xyze(i,2)-xyze(i,0);
    }

    normal(0) = dist1(1)*dist2(2) - dist1(2)*dist2(1);
    normal(1) = dist1(2)*dist2(0) - dist1(0)*dist2(2);
    normal(2) = dist1(0)*dist2(1) - dist1(1)*dist2(0);
  }
  break;
  case 1:
  {
    normal(0) = xyze(1,1) - xyze(1,0);
    normal(1) = (-1.0)*(xyze(0,1) - xyze(0,0));
  }
  break;
  default:
    dserror("Illegal number of space dimensions: %d",nsd_);
  } // switch(nsd)

  // length of normal to this element
  const double length = normal.Norm2();
  // outward-pointing normal of length 1.0
  normal.Scale(1/length);

  return;
} // ScaTraBoundaryImpl<distype>::


/*----------------------------------------------------------------------*
 |  Integrate shapefunctions over surface (private)           gjb 02/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraBoundaryImpl<distype>::IntegrateShapeFunctions(
    const DRT::Element*        ele,
    ParameterList&             params,
    Epetra_SerialDenseVector&  elevec1,
    const bool                 addarea
)
{
  // access boundary area variable with its actual value
  double boundaryint = params.get<double>("boundaryint");

  // get node coordinates (we have a nsd_+1 dimensional domain!)
  GEO::fillInitialPositionArray<distype,nsd_+1,LINALG::Matrix<nsd_+1,iel> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // integrations points and weights
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid=0; gpid<intpoints.IP().nquad; gpid++)
  {
    const double fac = EvalShapeFuncAndIntFac(intpoints,gpid,ele->Id());

    // compute integral of shape functions
    for (int node=0;node<iel;++node)
    {
      for (int k=0; k< numscal_; k++)
      {
        elevec1[node*numdofpernode_+k] += funct_(node) * fac;
      }
    }

    if (addarea)
    {
      // area calculation
      boundaryint += fac;
    }

  } //loop ove r integration points

  // add contribution to the global value
  params.set<double>("boundaryint",boundaryint);

  return;

} //ScaTraBoundaryImpl<distype>::IntegrateShapeFunction


/*----------------------------------------------------------------------*
 | calculate integral of normal diffusive flux + velocity div.  vg 09/08|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraBoundaryImpl<distype>::DifffluxAndDivuIntegral(
    const DRT::Element*             ele,
    ParameterList&                  params,
    const vector<double>&           ediffflux,
    const vector<double>&           edivu
)
{
  // get variables for integrals of normal diffusive flux and velocity div.
  double difffluxintegral = params.get<double>("diffusive-flux integral");
  double divuintegral     = params.get<double>("velocity-divergence integral");

  // integrations points and weights
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid=0; gpid<intpoints.IP().nquad; gpid++)
  {
    const double fac = EvalShapeFuncAndIntFac(intpoints,gpid,ele->Id());

    // compute integral of normal flux
    for (int node=0;node<iel;++node)
    {
      difffluxintegral += funct_(node) * ediffflux[node] * fac;
      divuintegral     += funct_(node) * edivu[node] * fac;
    }
  } // loop over integration points

  // add contributions to the global values
  params.set<double>("diffusive-flux integral",difffluxintegral);
  params.set<double>("velocity-divergence integral",divuintegral);

  return;

} //ScaTraBoundaryImpl<distype>::DifffluxAndDivuIntegral


#endif // CCADISCRET
#endif // D_FLUID3 or D_FLUID2
