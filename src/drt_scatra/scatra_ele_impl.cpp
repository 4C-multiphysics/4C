/*----------------------------------------------------------------------*/
/*!
\file scatra_ele_impl.cpp

\brief Internal implementation of scalar transport elements

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

#include "scatra_ele_impl.H"
#include "../drt_mat/scatra_mat.H"
#include "../drt_mat/mixfrac.H"
#include "../drt_mat/sutherland.H"
#include "../drt_mat/arrhenius_spec.H"
#include "../drt_mat/arrhenius_temp.H"
#include "../drt_mat/arrhenius_pv.H"
#include "../drt_mat/ferech_pv.H"
#include "../drt_mat/ion.H"
#include "../drt_mat/matlist.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"
#include "../drt_fem_general/drt_utils_nurbs_shapefunctions.H"
#include "../drt_nurbs_discret/drt_nurbs_discret.H"
#include "../drt_fem_general/drt_utils_gder2.H"
#include "../drt_geometry/position_array.H"
#include "../drt_lib/linalg_serialdensematrix.H"
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include "../drt_lib/drt_condition_utils.H"
#include "../drt_inpar/inpar_scatra.H"

//#define VISUALIZE_ELEMENT_DATA
#include "scatra_element.H" // only for visualization of element data
//#define MIGRATIONSTAB  //stabilization w.r.t migration term (obsolete!)
//#define PRINT_ELCH_DEBUG

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ScaTraImplInterface* DRT::ELEMENTS::ScaTraImplInterface::Impl(
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
  case DRT::Element::hex8:
  {
    static ScaTraImpl<DRT::Element::hex8>* ch8;
    if (ch8==NULL)
      ch8 = new ScaTraImpl<DRT::Element::hex8>(numdofpernode,numscal);
    return ch8;
  }
  case DRT::Element::hex20:
  {
    static ScaTraImpl<DRT::Element::hex20>* ch20;
    if (ch20==NULL)
      ch20 = new ScaTraImpl<DRT::Element::hex20>(numdofpernode,numscal);
    return ch20;
  }
  case DRT::Element::hex27:
  {
    static ScaTraImpl<DRT::Element::hex27>* ch27;
    if (ch27==NULL)
      ch27 = new ScaTraImpl<DRT::Element::hex27>(numdofpernode,numscal);
    return ch27;
  }
  case DRT::Element::nurbs8:
  {
    static ScaTraImpl<DRT::Element::nurbs8>* cn8;
    if (cn8==NULL)
      cn8 = new ScaTraImpl<DRT::Element::nurbs8>(numdofpernode,numscal);
    return cn8;
  }
  case DRT::Element::nurbs27:
  {
    static ScaTraImpl<DRT::Element::nurbs27>* cn27;
    if (cn27==NULL)
      cn27 = new ScaTraImpl<DRT::Element::nurbs27>(numdofpernode,numscal);
    return cn27;
  }
  case DRT::Element::tet4:
  {
    static ScaTraImpl<DRT::Element::tet4>* ct4;
    if (ct4==NULL)
      ct4 = new ScaTraImpl<DRT::Element::tet4>(numdofpernode,numscal);
    return ct4;
  }
 /* case DRT::Element::tet10:
  {
    static ScaTraImpl<DRT::Element::tet10>* ct10;
    if (ct10==NULL)
      ct10 = new ScaTraImpl<DRT::Element::tet10>(numdofpernode,numscal);
    return ct10;
  } */
  case DRT::Element::wedge6:
  {
    static ScaTraImpl<DRT::Element::wedge6>* cw6;
    if (cw6==NULL)
      cw6 = new ScaTraImpl<DRT::Element::wedge6>(numdofpernode,numscal);
    return cw6;
  }
/*  case DRT::Element::wedge15:
  {
    static ScaTraImpl<DRT::Element::wedge15>* cw15;
    if (cw15==NULL)
      cw15 = new ScaTraImpl<DRT::Element::wedge15>(numdofpernode,numscal);
    return cw15;
  } */
  case DRT::Element::pyramid5:
  {
    static ScaTraImpl<DRT::Element::pyramid5>* cp5;
    if (cp5==NULL)
      cp5 = new ScaTraImpl<DRT::Element::pyramid5>(numdofpernode,numscal);
    return cp5;
  }
  case DRT::Element::quad4:
  {
    static ScaTraImpl<DRT::Element::quad4>* cp4;
    if (cp4==NULL)
      cp4 = new ScaTraImpl<DRT::Element::quad4>(numdofpernode,numscal);
    return cp4;
  }
/*  case DRT::Element::quad8:
  {
    static ScaTraImpl<DRT::Element::quad8>* cp8;
    if (cp8==NULL)
      cp8 = new ScaTraImpl<DRT::Element::quad8>(numdofpernode,numscal);
    return cp8;
  }
  case DRT::Element::quad9:
  {
    static ScaTraImpl<DRT::Element::quad9>* cp9;
    if (cp9==NULL)
      cp9 = new ScaTraImpl<DRT::Element::quad9>(numdofpernode,numscal);
    return cp9;
  }*/
  case DRT::Element::nurbs4:
  {
    static ScaTraImpl<DRT::Element::nurbs4>* cn4;
    if (cn4==NULL)
      cn4 = new ScaTraImpl<DRT::Element::nurbs4>(numdofpernode,numscal);
    return cn4;
  }
  case DRT::Element::nurbs9:
  {
    static ScaTraImpl<DRT::Element::nurbs9>* cn9;
    if (cn9==NULL)
      cn9 = new ScaTraImpl<DRT::Element::nurbs9>(numdofpernode,numscal);
    return cn9;
  }
  case DRT::Element::tri3:
  {
    static ScaTraImpl<DRT::Element::tri3>* cp3;
    if (cp3==NULL)
      cp3 = new ScaTraImpl<DRT::Element::tri3>(numdofpernode,numscal);
    return cp3;
  }
/*  case DRT::Element::tri6:
  {
    static ScaTraImpl<DRT::Element::tri6>* cp6;
    if (cp6==NULL)
      cp6 = new ScaTraImpl<DRT::Element::tri6>(numdofpernode,numscal);
    return cp6;
  }*/
  case DRT::Element::line2:
  {
    static ScaTraImpl<DRT::Element::line2>* cl2;
    if (cl2==NULL)
      cl2 = new ScaTraImpl<DRT::Element::line2>(numdofpernode,numscal);
    return cl2;
  }/*
  case DRT::Element::line3:
  {
    static ScaTraImpl<DRT::Element::line3>* cl3;
    if (cl3==NULL)
      cl3 = new ScaTraImpl<DRT::Element::line3>(numdofpernode,numscal);
    return cl3;
  }*/
  default:
    dserror("Element shape %s not activated. Just do it.",DRT::DistypeToString(ele->Shape()).c_str());
  }
  return NULL;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraImpl<distype>::ScaTraImpl(int numdofpernode, int numscal)
  : numdofpernode_(numdofpernode),
    numscal_(numscal),
    iselch_((numdofpernode_ - numscal_) == 1),
    isale_(false),
    diffreastafac_(0.0),
    evelnp_(true),   // initialize to zero
    eaccnp_(true),
    eprenp_(true),
    ephin_(numscal_),
    ephinp_(numscal_),
    ephiam_(numscal_),
    ehist_(numdofpernode_),
    epotnp_(true),
    emagnetnp_(true),
    fsphinp_(numscal_),
    edispnp_(true),
    xyze_(true),
    weights_(true),
    myknots_(nsd_),
    bodyforce_(numdofpernode_),
    densn_(numscal_),
    densnp_(numscal_),
    densam_(numscal_),
    densgradfac_(numscal_),
    diffus_(numscal_),
    reacoeff_(numscal_),
    valence_(numscal_),
    diffusvalence_(numscal_),
    shcacp_(0.0),
    xsi_(true),
    funct_(true),
    deriv_(true),
    deriv2_(true),
    xjm_(true),
    xij_(true),
    derxy_(true),
    derxy2_(true),
    vderxy_(true),
    rhs_(numdofpernode_),
    reatemprhs_(numdofpernode_),
    hist_(numdofpernode_),
    velint_(true),
    sgvelint_(true),
    migvelint_(true),
    vdiv_(0.0),
    tau_(numscal_),
    sgdiff_(numscal_),
    xder2_(true),
    conv_(true),
    sgconv_(true),
    diff_(true),
    migconv_(true),
    migrea_(true),
    gradpot_(true),
    conint_(numscal_),
    gradphi_(true),
    fsgradphi_(true),
    laplace_(true),
    thermpressnp_(0.0),
    thermpressam_(0.0),
    thermpressdt_(0.0)
{
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::ScaTraImpl<distype>::Evaluate(
    DRT::Element*              ele,
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
  // get additional state vector for ALE case: grid displacement
  isale_ = params.get<bool>("isale");
  if (isale_)
  {
    const RCP<Epetra_MultiVector> dispnp = params.get< RCP<Epetra_MultiVector> >("dispnp",null);
    if (dispnp==null) dserror("Cannot get state vector 'dispnp'");
    DRT::UTILS::ExtractMyNodeBasedValues(ele,edispnp_,dispnp,nsd_);
  }
  else edispnp_.Clear();

  // the type of scalar transport problem has to be provided for all actions!
  const INPAR::SCATRA::ScaTraType scatratype = params.get<INPAR::SCATRA::ScaTraType>("scatratype");
  if (scatratype == INPAR::SCATRA::scatratype_undefined)
    dserror("Set parameter SCATRATYPE in your input file!");

  // check for the action parameter
  const string action = params.get<string>("action","none");
  if (action=="calc_condif_systemmat_and_residual")
  {
    // set flag for including reactive terms to false initially
    // flag will be set to true below when reactive material is included
    reaction_ = false;

    // get control parameters
    is_stationary_  = params.get<bool>("using stationary formulation");
    is_genalpha_    = params.get<bool>("using generalized-alpha time integration");
    is_incremental_ = params.get<bool>("incremental solver");

    // get current time and time-step length
    const double time = params.get<double>("total time");
    const double dt   = params.get<double>("time-step length");

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
    bool turbmodel = false;
    if (scatratype == INPAR::SCATRA::scatratype_loma)
    {
      thermpressnp_ = params.get<double>("thermodynamic pressure");
      thermpressdt_ = params.get<double>("time derivative of thermodynamic pressure");
      if (is_genalpha_)
        thermpressam_ = params.get<double>("thermodynamic pressure at n+alpha_M");

      // set flag for turbulence model
      turbmodel = params.get<bool>("turbulence model");
    }

    // set flag for conservative form
    const INPAR::SCATRA::ConvForm convform =
    params.get<INPAR::SCATRA::ConvForm>("form of convective term");
    conservative_ = false;
    if (convform ==INPAR::SCATRA::convform_conservative) conservative_ = true;

    bool reinitswitch = params.get<bool>("reinitswitch",false);

    // set parameters for stabilization
    ParameterList& stablist = params.sublist("STABILIZATION");

    // select tau definition
    INPAR::SCATRA::TauType whichtau = Teuchos::getIntegralValue<INPAR::SCATRA::TauType>(stablist,"DEFINITION_TAU");

    // set (sign) factor for diffusive and reactive stabilization terms
    // (factor is zero for SUPG) and overwrite tau definition when there
    // is no stabilization
    const INPAR::SCATRA::StabType stabinp = Teuchos::getIntegralValue<INPAR::SCATRA::StabType>(stablist,"STABTYPE");

    switch(stabinp)
    {
    case INPAR::SCATRA::stabtype_no_stabilization:
      whichtau = INPAR::SCATRA::tau_zero;
      break;
    case INPAR::SCATRA::stabtype_GLS:
      diffreastafac_ = 1.0;
      break;
    case INPAR::SCATRA::stabtype_USFEM:
      diffreastafac_ = -1.0;
    break;
    default:
      break;
    }

    // set flags for subgrid-scale velocity and all-scale subgrid-diffusivity term
    // (default: "false" for both flags)
    const bool sgvel(Teuchos::getIntegralValue<int>(stablist,"SUGRVEL"));
    sgvel_ = sgvel;
    const bool assgd(Teuchos::getIntegralValue<int>(stablist,"ASSUGRDIFF"));

    // select type of all-scale subgrid diffusivity if included
    const INPAR::SCATRA::AssgdType whichassgd
    = Teuchos::getIntegralValue<INPAR::SCATRA::AssgdType>(stablist,"DEFINITION_ASSGD");

    // set flags for potential evaluation of tau and material law at int. point
    const INPAR::SCATRA::EvalTau tauloc = Teuchos::getIntegralValue<INPAR::SCATRA::EvalTau>(stablist,"EVALUATION_TAU");
    tau_gp_ = (tauloc == INPAR::SCATRA::evaltau_integration_point); // set true/false
    const INPAR::SCATRA::EvalMat matloc = Teuchos::getIntegralValue<INPAR::SCATRA::EvalMat>(stablist,"EVALUATION_MAT");
    mat_gp_ = (matloc == INPAR::SCATRA::evalmat_integration_point); // set true/false

    // set flag for fine-scale subgrid diffusivity and perform some checks
    bool fssgd = false; //default
    const INPAR::SCATRA::FSSUGRDIFF whichfssgd = params.get<INPAR::SCATRA::FSSUGRDIFF>("fs subgrid diffusivity");
    if (whichfssgd == INPAR::SCATRA::fssugrdiff_artificial)
    {
      fssgd = true;

      // check for solver type
      if (is_incremental_) dserror("Artificial fine-scale subgrid-diffusivity approach only in combination with non-incremental solver so far!");
    }
    else if (whichfssgd == INPAR::SCATRA::fssugrdiff_smagorinsky_all)
    {
      fssgd = true;

      // check for solver type
      if (not is_incremental_) dserror("Fine-scale subgrid-diffusivity approach using all-scale Smagorinsky model only in combination with incremental solver so far!");
    }
    else if (whichfssgd == INPAR::SCATRA::fssugrdiff_smagorinsky_small)
      dserror("Fine-scale subgrid-diffusivity approach using fine-scale Smagorinsky model not available so far!");

    // check for combination of all-scale and fine-scale subgrid diffusivity
    if (assgd and fssgd) dserror("No combination of all-scale and fine-scale subgrid-diffusivity approach currently possible!");

    // get velocity at nodes
    const RCP<Epetra_MultiVector> velocity = params.get< RCP<Epetra_MultiVector> >("velocity field",null);
    DRT::UTILS::ExtractMyNodeBasedValues(ele,evelnp_,velocity,nsd_);

    // also get subgrid-scale velocity field if required
    if (sgvel_)
    {
      const RCP<Epetra_MultiVector> accpre = params.get< RCP<Epetra_MultiVector> >("acceleration/pressure field",null);
      LINALG::Matrix<nsd_+1,nen_> eaccprenp;
      DRT::UTILS::ExtractMyNodeBasedValues(ele,eaccprenp,accpre,nsd_+1);

      // split acceleration and pressure values
      for (int i=0;i<nen_;++i)
      {
        for (int j=0;j<nsd_;++j)
        {
          eaccnp_(j,i) = eaccprenp(j,i);
        }
        eprenp_(i) = eaccprenp(nsd_,i);
      }
    }

    // extract local values from the global vectors
    RefCountPtr<const Epetra_Vector> hist = discretization.GetState("hist");
    RefCountPtr<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (hist==null || phinp==null)
      dserror("Cannot get state vector 'hist' and/or 'phinp'");
    vector<double> myhist(lm.size());
    vector<double> myphinp(lm.size());
    DRT::UTILS::ExtractMyValues(*hist,myhist,lm);
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

    // fill all element arrays
    for (int i=0;i<nen_;++i)
    {
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        ephinp_[k](i,0) = myphinp[k+(i*numdofpernode_)];
      }
      for (int k = 0; k< numdofpernode_; ++k)
      {
        // the history vectors contains information of time step t_n
        ehist_[k](i,0) = myhist[k+(i*numdofpernode_)];
      }
    } // for i

    if ((scatratype == INPAR::SCATRA::scatratype_loma) and is_genalpha_)
    {
      // extract additional local values from global vector
      RefCountPtr<const Epetra_Vector> phiam = discretization.GetState("phiam");
      if (phiam==null) dserror("Cannot get state vector 'phiam'");
      vector<double> myphiam(lm.size());
      DRT::UTILS::ExtractMyValues(*phiam,myphiam,lm);

      // fill element array
      for (int i=0;i<nen_;++i)
      {
        for (int k = 0; k< numscal_; ++k)
        {
          // split for each transported scalar, insert into element arrays
          ephiam_[k](i,0) = myphiam[k+(i*numdofpernode_)];
        }
      } // for i
    }

    if (is_genalpha_ and not is_incremental_)
    {
      // extract additional local values from global vector
      RefCountPtr<const Epetra_Vector> phin = discretization.GetState("phin");
      if (phin==null) dserror("Cannot get state vector 'phin'");
      vector<double> myphin(lm.size());
      DRT::UTILS::ExtractMyValues(*phin,myphin,lm);

      // fill element array
      for (int i=0;i<nen_;++i)
      {
        for (int k = 0; k< numscal_; ++k)
        {
          // split for each transported scalar, insert into element arrays
          ephin_[k](i,0) = myphin[k+(i*numdofpernode_)];
        }
      } // for i
    }

    double frt(0.0);
    if (iselch_)
    {
      // get values for el. potential at element nodes
      for (int i=0;i<nen_;++i)
      {
        epotnp_(i) = myphinp[i*numdofpernode_+numscal_];
      }
      // get parameter F/RT needed for ELCH ;-)
      frt = params.get<double>("frt");

      // get magnetic field at nodes (if available)
      // try to get the pointer to the entry (and check if type is RCP<Epetra_MultiVector>)
      RCP<Epetra_MultiVector>* b = params.getPtr< RCP<Epetra_MultiVector> >("magnetic field");
      if (b!= NULL) // magnetic field has been set and is not of type Teuchos::null
          DRT::UTILS::ExtractMyNodeBasedValues(ele,emagnetnp_,*b,nsd_);
       else
        emagnetnp_.Clear();
    }
    else
    {
      epotnp_.Clear();
      emagnetnp_.Clear();
    }

    double Cs(0.0);
    double tpn(1.0);
    // get subgrid-diffusivity vector if turbulence model is used
    if (turbmodel or (is_incremental_ and fssgd))
    {
      // get Smagorinsky constant and turbulent Prandtl number
      Cs  = params.get<double>("Smagorinsky constant");
      tpn = params.get<double>("turbulent Prandtl number");

      if (is_incremental_ and fssgd)
      {
        RCP<const Epetra_Vector> gfsphinp = discretization.GetState("fsphinp");
        if (gfsphinp==null) dserror("Cannot get state vector 'fsphinp'");

        vector<double> myfsphinp(lm.size());
        DRT::UTILS::ExtractMyValues(*gfsphinp,myfsphinp,lm);

        for (int i=0;i<nen_;++i)
        {
          for (int k = 0; k< numscal_; ++k)
          {
            // split for each transported scalar, insert into element arrays
            fsphinp_[k](i,0) = myfsphinp[k+(i*numdofpernode_)];
          }
        }
      }
    }

    // --------------------------------------------------
    // Now do the nurbs specific stuff
    //std::vector<Epetra_SerialDenseVector> myknots_(nsd_);

    // for isogeometric elements
    if(SCATRA::IsNurbs(distype))
    {
      DRT::NURBS::NurbsDiscretization* nurbsdis
        =
        dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(discretization));

      bool zero_size = false;
      zero_size = (*((*nurbsdis).GetKnotVector())).GetEleKnots(myknots_,ele->Id());

      // if we have a zero sized element due to a interpolated
      // point --- exit here
      if(zero_size)
      {
        return(0);
      }
    }

    // calculate element coefficient matrix and rhs
    Sysmat(
        ele,
        elemat1_epetra,
        elevec1_epetra,
        elevec2_epetra,
        time,
        dt,
        timefac,
        alphaF,
        whichtau,
        whichassgd,
        whichfssgd,
        assgd,
        fssgd,
        turbmodel,
        reinitswitch,
        Cs,
        tpn,
        frt,
        scatratype);
  }
  // calculate time derivative for time value t_0
  else if (action =="calc_initial_time_deriv")
  {
    // set flag for including reactive terms to false initially
    // flag will be set to true below when reactive material is included
    reaction_ = false;

    // set flag for conservative form
    const INPAR::SCATRA::ConvForm convform =
      params.get<INPAR::SCATRA::ConvForm>("form of convective term");
    conservative_ = false;
    if (convform ==INPAR::SCATRA::convform_conservative) conservative_ = true;

    // get initial velocity values at the nodes
    const RCP<Epetra_MultiVector> velocity = params.get< RCP<Epetra_MultiVector> >("velocity field",null);
    DRT::UTILS::ExtractMyNodeBasedValues(ele,evelnp_,velocity,nsd_);

    // need initial field -> extract local values from the global vector
    RefCountPtr<const Epetra_Vector> phi0 = discretization.GetState("phi0");
    if (phi0==null) dserror("Cannot get state vector 'phi0'");
    vector<double> myphi0(lm.size());
    DRT::UTILS::ExtractMyValues(*phi0,myphi0,lm);

    // fill element arrays
    for (int i=0;i<nen_;++i)
    {
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        ephinp_[k](i,0) = myphi0[k+(i*numdofpernode_)];
      }
    } // for i

    // set time derivative of thermodynamic pressure if required
    if (scatratype ==INPAR::SCATRA::scatratype_loma)
    {
      thermpressnp_ = params.get<double>("thermodynamic pressure");
      thermpressam_ = thermpressnp_;
      thermpressdt_ = params.get<double>("time derivative of thermodynamic pressure");
    }

    bool reinitswitch = params.get<bool>("reinitswitch",false);

    // get stabilization parameter sublist
    ParameterList& stablist = params.sublist("STABILIZATION");

    // set flags for potential evaluation of material law at int. point
    const INPAR::SCATRA::EvalMat matloc = Teuchos::getIntegralValue<INPAR::SCATRA::EvalMat>(stablist,"EVALUATION_MAT");
    mat_gp_ = (matloc == INPAR::SCATRA::evalmat_integration_point); // set true/false

    double frt(0.0);
    if(scatratype ==INPAR::SCATRA::scatratype_elch_enc)
    {
      for (int i=0;i<nen_;++i)
      {
        // get values for el. potential at element nodes
        epotnp_(i) = myphi0[i*numdofpernode_+numscal_];
      } // for i

      // get parameter F/RT
      frt = params.get<double>("frt");
    }
    else epotnp_.Clear();

    // calculate matrix and rhs
    InitialTimeDerivative(
        ele,
        elemat1_epetra,
        elevec1_epetra,
        reinitswitch,
        frt);
  }
  else if (action =="calc_subgrid_diffusivity_matrix")
  // calculate normalized subgrid-diffusivity matrix
  {
    // get control parameter
    is_genalpha_   = params.get<bool>("using generalized-alpha time integration");
    is_stationary_ = params.get<bool>("using stationary formulation");

    // One-step-Theta:    timefac = theta*dt
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

    // calculate mass matrix and rhs
    CalcSubgridDiffMatrix(
        ele,
        elemat1_epetra,
        timefac);
  }
  else if (action=="calc_condif_flux")
  {
    // get velocity values at the nodes
    const RCP<Epetra_MultiVector> velocity = params.get< RCP<Epetra_MultiVector> >("velocity field",null);
    Epetra_SerialDenseVector evel(nsd_*nen_);
    DRT::UTILS::ExtractMyNodeBasedValues(ele,evel,velocity,nsd_);

    // need current values of transported scalar
    // -> extract local values from global vectors
    RefCountPtr<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (phinp==null) dserror("Cannot get state vector 'phinp'");
    vector<double> myphinp(lm.size());
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

    // assure, that the values are in the same order as the element nodes
    for(int k=0;k<nen_;++k)
    {
      Node* node = (ele->Nodes())[k];
      vector<int> dof = discretization.Dof(node);
      int numdof = dof.size();
      for (int i=0;i<numdof;++i)
      {
        if (dof[i]!=lm[k*numdof+i])
        {
          cout<<"dof[i]= "<<dof[i]<<"  lm[k*numdof+i]="<<lm[k*numdof+i]<<endl;
          dserror("Dofs are not in the same order as the element nodes. Implement some resorting!");
        }
      }
    }

    // access control parameter for flux calculation
    INPAR::SCATRA::FluxType fluxtype = params.get<INPAR::SCATRA::FluxType>("fluxtype");

    // set number of scalars and frt for ELCH
    int numscal = numdofpernode_;
    double frt(0.0);

    if (scatratype ==INPAR::SCATRA::scatratype_elch_enc)
    {
      numscal -= 1; // ELCH case: last dof is for el. potential
      // get parameter F/RT
      frt = params.get<double>("frt");
    }

    // we always get an 3D flux vector for each node
    LINALG::Matrix<3,nen_> eflux(true);

    // do a loop for systems of transported scalars
    for (int i = 0; i<numscal; ++i)
    {
      // calculate flux vectors for actual scalar
      eflux.Clear();
      CalculateFlux(eflux,ele,myphinp,frt,evel,fluxtype,i);

      // assembly
      for (int k=0;k<nen_;k++)
      { // form arithmetic mean of assembled nodal flux vectors
        // => factor is the number of adjacent elements for each node
        double factor = ((ele->Nodes())[k])->NumElement();
        elevec1_epetra[k*numdofpernode_+i]+=eflux(0,k)/factor;
        elevec2_epetra[k*numdofpernode_+i]+=eflux(1,k)/factor;
        elevec3_epetra[k*numdofpernode_+i]+=eflux(2,k)/factor;
      }
    } // loop over numscal
  }
  else if (action=="calc_mean_scalars")
  {
    // NOTE: add integral values only for elements which are NOT ghosted!
    if (ele->Owner() == discretization.Comm().MyPID())
    {
      // get flag for inverting
      bool inverting = params.get<bool>("inverting");

      // need current scalar vector
      // -> extract local values from the global vectors
      RefCountPtr<const Epetra_Vector> phinp = discretization.GetState("phinp");
      if (phinp==null) dserror("Cannot get state vector 'phinp'");
      vector<double> myphinp(lm.size());
      DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

      // calculate scalars and domain integral
      CalculateScalars(ele,myphinp,elevec1_epetra,inverting);
    }
  }
  else if (action=="calc_domain_and_bodyforce")
  {
    // NOTE: add integral values only for elements which are NOT ghosted!
    if (ele->Owner() == discretization.Comm().MyPID())
    {
      const double time = params.get<double>("total time");

      bool reinitswitch = params.get<bool>("reinitswitch",false);

      // calculate domain and bodyforce integral
      CalculateDomainAndBodyforce(elevec1_epetra,ele,time,reinitswitch);
    }
  }
  else if (action=="calc_elch_kwok_error")
  {
    // check if length suffices
    if (elevec1_epetra.Length() < 1) dserror("Result vector too short");

    // need current solution
    RefCountPtr<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (phinp==null) dserror("Cannot get state vector 'phinp'");

    // extract local values from the global vector
    vector<double> myphinp(lm.size());
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

    if (numscal_ != 2)
      dserror("Numscal_ != 2 for error calculation of Kwok & Wu example");

    // fill element arrays
    for (int i=0;i<nen_;++i)
    {
      // split for each transported scalar, insert into element arrays
      for (int k = 0; k< numscal_; ++k)
      {
        ephinp_[k](i) = myphinp[k+(i*numdofpernode_)];
      }
      // get values for el. potential at element nodes
      epotnp_(i) = myphinp[i*numdofpernode_+numscal_];
    } // for i

    CalErrorComparedToAnalytSolution(
        ele,
        params,
        elevec1_epetra);
  }
  else if (action=="integrate_shape_functions")
  {
    // calculate integral of shape functions
    const Epetra_IntSerialDenseVector dofids = params.get<Epetra_IntSerialDenseVector>("dofids");
    IntegrateShapeFunctions(ele,elevec1_epetra,dofids);
  }
  else if (action=="calc_elch_conductivity")
  {
    // calculate conductivity of electrolyte solution
    const double frt = params.get<double>("frt");
    // extract local values from the global vector
    RefCountPtr<const Epetra_Vector> phinp = discretization.GetState("phinp");
    vector<double> myphinp(lm.size());
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

    // fill element arrays
    for (int i=0;i<nen_;++i)
    {
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        ephinp_[k](i,0) = myphinp[k+(i*numdofpernode_)];
      }
    } // for i

    CalculateConductivity(ele,frt,elevec1_epetra);
  }
  else if (action=="get_material_parameters")
  {
    // get the material
    RefCountPtr<MAT::Material> material = ele->Material();

    if (material->MaterialType() == INPAR::MAT::m_sutherland)
    {
      const MAT::Sutherland* actmat = static_cast<const MAT::Sutherland*>(material.get());

      params.set("thermodynamic pressure",actmat->ThermPress());
    }
    else params.set("thermodynamic pressure",0.0);
  }
  else if (action =="calc_time_deriv_reinit")
  {
    // set flag for including reactive terms to false initially
    // flag will be set to true below when reactive material is included
    reaction_ = false;

    // set flag for conservative form
    const INPAR::SCATRA::ConvForm convform = params.get<INPAR::SCATRA::ConvForm>("form of convective term");
    conservative_ = false;
    if (convform ==INPAR::SCATRA::convform_conservative) conservative_ = true;

    // get initial velocity values at the nodes
    const RCP<Epetra_MultiVector> velocity = params.get< RCP<Epetra_MultiVector> >("velocity field",null);
    DRT::UTILS::ExtractMyNodeBasedValues(ele,evelnp_,velocity,nsd_);

    // need initial field -> extract local values from the global vector
    RefCountPtr<const Epetra_Vector> phi0 = discretization.GetState("phi0");
    if (phi0==null) dserror("Cannot get state vector 'phi0'");
    vector<double> myphi0(lm.size());
    DRT::UTILS::ExtractMyValues(*phi0,myphi0,lm);

    // fill element arrays
    for (int i=0;i<nen_;++i)
    {
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        ephinp_[k](i,0) = myphi0[k+(i*numdofpernode_)];
      }
    } // for i

    // get stabilization parameter sublist
    ParameterList& stablist = params.sublist("STABILIZATION");

    // select tau definition
    INPAR::SCATRA::TauType whichtau = Teuchos::getIntegralValue<INPAR::SCATRA::TauType>(stablist,"DEFINITION_TAU");

    // get time-step length
    const double dt   = params.get<double>("time-step length");

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

    // set flags for potential evaluation of material law at int. point
    const INPAR::SCATRA::EvalMat matloc = Teuchos::getIntegralValue<INPAR::SCATRA::EvalMat>(stablist,"EVALUATION_MAT");
    mat_gp_ = (matloc == INPAR::SCATRA::evalmat_integration_point); // set true/false

    epotnp_.Clear();

    // calculate matrix and rhs
    TimeDerivativeReinit(
        ele,
        elemat1_epetra,
        elevec1_epetra,
        whichtau,
        dt,
        timefac
        );
  }
  // calculate initial electric potential field caused by initial ion concentrations
  else if (action =="calc_initial_potential_field")
  {
    // need initial field -> extract local values from the global vector
    RefCountPtr<const Epetra_Vector> phi0 = discretization.GetState("phi0");
    if (phi0==null) dserror("Cannot get state vector 'phi0'");
    vector<double> myphi0(lm.size());
    DRT::UTILS::ExtractMyValues(*phi0,myphi0,lm);

    // fill element arrays
    for (int i=0;i<nen_;++i)
    {
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        ephinp_[k](i,0) = myphi0[k+(i*numdofpernode_)];
      }
    } // for i
    const double frt = params.get<double>("frt");

    CalculateElectricPotentialField(ele,frt,elemat1_epetra,elevec1_epetra);

  }
  else
    dserror("Unknown type of action for Scatra Implementation: %s",action.c_str());
  // work is done
  return 0;
}


/*----------------------------------------------------------------------*
 |  calculate mass flux                              (private) gjb 06/08|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalculateFluxSerialDense(
    LINALG::SerialDenseMatrix&      flux,
    DRT::Element*&                  ele,
    vector<double>&                 ephinp,
    double                          frt,
    Epetra_SerialDenseVector&       evel,
    std::string&                    fluxtypestring,
    int                             dofindex
)
{
  // access control parameter
  INPAR::SCATRA::FluxType fluxtype;
  if (fluxtypestring == "totalflux")          fluxtype = INPAR::SCATRA::flux_total_domain;
  else if (fluxtypestring == "diffusiveflux") fluxtype = INPAR::SCATRA::flux_diffusive_domain;
  else                                        fluxtype = INPAR::SCATRA::flux_no;

  // we always get an 3D flux vector for each node
  LINALG::Matrix<3,nen_> eflux(true); //initialize!
  CalculateFlux(eflux,ele,ephinp,frt,evel,fluxtype,dofindex);
  for (int j = 0; j< nen_; j++)
  {
    flux(0,j) = eflux(0,j);
    flux(1,j) = eflux(1,j);
    flux(2,j) = eflux(2,j);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  calculate system matrix and rhs (public)                 g.bau 08/08|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::Sysmat(
    DRT::Element*                         ele, ///< the element those matrix is calculated
    Epetra_SerialDenseMatrix&             sys_mat,///< element matrix to calculate
    Epetra_SerialDenseVector&             residual, ///< element rhs to calculate
    Epetra_SerialDenseVector&             subgrdiff, ///< subgrid-diff.-scaling vector
    const double                          time, ///< current simulation time
    const double                          dt, ///< current time-step length
    const double                          timefac, ///< time discretization factor
    const double                          alphaF, ///< factor for generalized-alpha time integration
    const enum INPAR::SCATRA::TauType     whichtau, ///< stabilization parameter definition
    const enum INPAR::SCATRA::AssgdType   whichassgd, ///< all-scale subgrid-diffusivity definition
    const enum INPAR::SCATRA::FSSUGRDIFF  whichfssgd, ///< fine-scale subgrid-diffusivity definition
    const bool                            assgd, ///< all-scale subgrid-diff. flag
    const bool                            fssgd, ///< fine-scale subgrid-diff. flag
    const bool                            turbmodel, ///< turbulence model flag
    const bool                            reinitswitch,
    const double                          Cs, ///< Smagorinsky constant
    const double                          tpn, ///< turbulent Prandtl number
    const double                          frt, ///< factor F/RT needed for ELCH calculations
    const enum INPAR::SCATRA::ScaTraType  scatratype ///< type of scalar transport problem
)
{
  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  // get node weights for nurbs elements
  if(SCATRA::IsNurbs(distype))
  {
    for (int inode=0; inode<nen_; inode++)
    {
      DRT::Node** nodes = ele->Nodes();
      DRT::NURBS::ControlPoint* cp
        =
        dynamic_cast<DRT::NURBS::ControlPoint* > (nodes[inode]);

      weights_(inode) = cp->W();
    }
  }

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // ---------------------------------------------------------------------
  // call routine for calculation of body force in element nodes
  // (time n+alpha_F for generalized-alpha scheme, at time n+1 otherwise)
  // ---------------------------------------------------------------------
//REINHARD
  if (reinitswitch == false)
    BodyForce(ele,time);
  else
    BodyForceReinit(ele,time);
//end REINHARD

  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------
  // use one-point Gauss rule to do calculations at the element center
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints_tau(SCATRA::DisTypeToStabGaussRule<distype>::rule);

  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double vol = EvalShapeFuncAndDerivsAtIntPoint(intpoints_tau,0,ele->Id());

  //----------------------------------------------------------------------
  // get material parameters (evaluation at element center)
  //----------------------------------------------------------------------
  if (not mat_gp_ or not tau_gp_) GetMaterialParams(ele);

  //----------------------------------------------------------------------
  // calculation of subgrid diffusivity and stabilization parameter(s)
  // at element center
  //----------------------------------------------------------------------
  if (not tau_gp_)
  {
    // get velocity at element center
    velint_.Multiply(evelnp_,funct_);

    bool twoionsystem(false);
    double resdiffus(diffus_[0]);
    if (iselch_) // electrochemistry problem
    {
#ifdef MIGRATIONSTAB
      // compute global derivatives
      derxy_.Multiply(xij_,deriv_);

      // get "migration velocity" divided by D_k*z_k at element center
      migvelint_.Multiply(-frt,derxy_,epotnp_);
#endif

      // ELCH: special stabilization in case of binary electrolytes
      twoionsystem = SCATRA::IsBinaryElectrolyte(valence_);
      if (twoionsystem)
        resdiffus = SCATRA::CalResDiffCoeff(valence_,diffus_,diffusvalence_);

 //     cout<<"resdiffus = "<<resdiffus<<endl;
    }

    for (int k = 0;k<numscal_;++k) // loop of each transported scalar
    {
      // calculation of all-scale subgrid diffusivity (artificial or due to
      // constant-coefficient Smagorinsky model) at element center
      if (assgd or turbmodel)
        CalcSubgrDiff(dt,timefac,whichassgd,assgd,turbmodel,Cs,tpn,vol,k);

      // calculation of fine-scale artificial subgrid diffusivity at element center
      if (fssgd) CalcFineScaleSubgrDiff(ele,subgrdiff,whichfssgd,Cs,tpn,vol,k);

      // generating copy of diffusivity for use in CalTau routine
      double diffus = diffus_[k];

      // use resulting diffusion coefficient for binary electrolyte solutions
      if (twoionsystem && (k<2)) diffus = resdiffus;

      // calculation of stabilization parameter at element center
      CalTau(ele,diffus,dt,timefac,whichtau,vol,k);
    }
  }

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integrations points and weights
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  if (iselch_) // electrochemistry problem
  {
    for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
    {
      const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad,ele->Id());

      //----------------------------------------------------------------------
      // get material parameters (evaluation at integration point)
      //----------------------------------------------------------------------
      if (mat_gp_) GetMaterialParams(ele);

      // get velocity at integration point
      velint_.Multiply(evelnp_,funct_);

      // convective part in convective form: u_x*N,x+ u_y*N,y
      conv_.MultiplyTN(derxy_,velint_);

      // momentum divergence required for conservative form
      if (conservative_) GetVelocityDivergence(vdiv_,evelnp_,derxy_);

      //--------------------------------------------------------------------
      // calculation of subgrid diffusivity and stabilization parameter(s)
      // at integration point
      //--------------------------------------------------------------------
      if (tau_gp_)
      {
#ifdef MIGRATIONSTAB
        // compute global derivatives
        derxy_.Multiply(xij_,deriv_);

        // get "migration velocity" divided by D_k*z_k at element center
        migvelint_.Multiply(-frt,derxy_,epotnp_);
#endif

        // ELCH: special stabilization in case of binary electrolytes
        bool twoionsystem(false);
        double resdiffus(diffus_[0]);
        twoionsystem = SCATRA::IsBinaryElectrolyte(valence_);
        if (twoionsystem)
          resdiffus = SCATRA::CalResDiffCoeff(valence_,diffus_,diffusvalence_);

        for (int k = 0;k<numscal_;++k) // loop of each transported scalar
        {
          // calculation of all-scale subgrid diffusivity (artificial or due to
          // constant-coefficient Smagorinsky model) at integration point
          if (assgd or turbmodel)
            CalcSubgrDiff(dt,timefac,whichassgd,assgd,turbmodel,Cs,tpn,vol,k);

          // calculation of fine-scale artificial subgrid diffusivity
          // at integration point
          if (fssgd) CalcFineScaleSubgrDiff(ele,subgrdiff,whichfssgd,Cs,tpn,vol,k);

          // generating copy of diffusivity for use in CalTau routine
          double diffus = diffus_[k];

          // use resulting diffusion coefficient for binary electrolyte solutions
          if (twoionsystem && (k<2)) diffus = resdiffus;

          // calculation of stabilization parameter at integration point
          CalTau(ele,diffus,dt,timefac,whichtau,vol,k);
        }
      }

      for (int k = 0;k<numscal_;++k) // loop of each transported scalar
      {
        // get history data at integration point
        hist_[k] = funct_.Dot(ehist_[k]);
        // get bodyforce at integration point
        rhs_[k] = bodyforce_[k].Dot(funct_);
      }

      // check for unsupported time-integration scheme
      if (is_genalpha_) dserror("GenAlpha is not supported by ELCH!");

      // compute matrix and rhs for electrochemistry problem
      CalMatElch(sys_mat,residual,frt,timefac,fac);
    }

  }
  else if ((scatratype == INPAR::SCATRA::scatratype_levelset) and reinitswitch == true)
  {
//REINHARD
    dserror("Due to Volkers commit on 14.9.09 things have to be rearranged!");
// 	  Epetra_SerialDenseMatrix gradphi_gpscur(8,3);
//	  // up to now only implemented for Hex8 elements...
//		  //calculation of expol_temp, is then inverted and overwritten
//	  Epetra_SerialDenseMatrix expol_invert(8,8);
//	  LINALG::Matrix<8,8> expol_temp;
//
//	  double sq3=sqrt(3.0);
//	  expol_temp(0,0)=1.25+0.75*sq3;
//	  expol_temp(0,1)=-0.25-0.25*sq3;
//	  expol_temp(0,2)=-0.25+0.25*sq3;
//	  expol_temp(0,3)=-0.25-0.25*sq3;
//	  expol_temp(0,4)=-0.25-0.25*sq3;
//	  expol_temp(0,5)=-0.25+0.25*sq3;
//	  expol_temp(0,6)=1.25-0.75*sq3;
//	  expol_temp(0,7)=-0.25+0.25*sq3;
//	  expol_temp(1,1)=1.25+0.75*sq3;
//	  expol_temp(1,2)=-0.25-0.25*sq3;
//	  expol_temp(1,3)=-0.25+0.25*sq3;
// 		  expol_temp(1,4)=-0.25+0.25*sq3;
// 		  expol_temp(1,5)=-0.25-0.25*sq3;
// 		  expol_temp(1,6)=-0.25+0.25*sq3;
// 		  expol_temp(1,7)=1.25-0.75*sq3;
// 		  expol_temp(2,2)=1.25+0.75*sq3;
// 		  expol_temp(2,3)=-0.25-0.25*sq3;
// 		  expol_temp(2,4)=1.25-0.75*sq3;
// 		  expol_temp(2,5)=-0.25+0.25*sq3;
// 		  expol_temp(2,6)=-0.25-0.25*sq3;
// 		  expol_temp(2,7)=-0.25+0.25*sq3;
// 		  expol_temp(3,3)=1.25+0.75*sq3;
// 		  expol_temp(3,4)=-0.25+0.25*sq3;
// 		  expol_temp(3,5)=1.25-0.75*sq3;
// 		  expol_temp(3,6)=-0.25+0.25*sq3;
// 		  expol_temp(3,7)=-0.25-0.25*sq3;
// 		  expol_temp(4,4)=1.25+0.75*sq3;
// 		  expol_temp(4,5)=-0.25-0.25*sq3;
// 		  expol_temp(4,6)=-0.25+0.25*sq3;
// 		  expol_temp(4,7)=-0.25-0.25*sq3;
// 		  expol_temp(5,5)=1.25+0.75*sq3;
// 		  expol_temp(5,6)=-0.25-0.25*sq3;
// 		  expol_temp(5,7)=-0.25+0.25*sq3;
// 		  expol_temp(6,6)=1.25+0.75*sq3;
// 		  expol_temp(6,7)=-0.25-0.25*sq3;
// 		  expol_temp(7,7)=1.25+0.75*sq3;
//
// 		  for (int k=0;k<8;++k)
// 			  for (int l=0;l<k;++l)
// 				  expol_temp(k,l)=expol_temp(l,k);
//
// 		  LINALG::FixedSizeSerialDenseSolver<8,8,1> solve_for_inverseexpol_temp;
// 		  solve_for_inverseexpol_temp.SetMatrix(expol_temp);
// 		  int err2 = solve_for_inverseexpol_temp.Factor();
// 		  int err = solve_for_inverseexpol_temp.Invert();
// 		  if ((err != 0) || (err2!=0)) dserror("Inversion of expol_temp failed");
//
// 		  for (int k=0; k<8; ++k)
// 			  for (int l=0; l<8; ++l)
// 				  expol_invert(k,l) = expol_temp(k,l);
//
// 		  //end calculation of expol_invert
//
// 		  RefCountPtr<const Epetra_Vector> phinp = discretization.GetState("phinp");
// 		  //final gradients of phi at all nodes of current element
// 		  Epetra_SerialDenseMatrix gradphi_nodescur(8,3);
//
// 		  const int* nodeids = ele->NodeIds();
// 		  //loop over all nodes of current element
// 		  for (int lnodeid=0; lnodeid < nen_; lnodeid++)
// 		  {
// 			  //get current node
// 			  DRT::Node* actnode = discretization.lRowNode(nodeids[lnodeid]);
// 			  //get all adjacent elements to this node
// 			  DRT::Element** elements=actnode->Elements();
//
// 			  //final gradient of phi at the one relevant node
// 			  Epetra_SerialDenseMatrix gradphi(3,1);
// 			  for (int j=0; j<3; j++)
// 				  gradphi(j,0) = 0.0;
//
// 			  int numadnodes = actnode->NumElement();
// 			  //loop over all adjacent elements of the current node
// 			  for (int elecur=0; elecur<numadnodes; elecur++)
// 			  {
// 				  Epetra_SerialDenseMatrix gradphi_node(3,1);
// 				  //get current element
// 				   const DRT::Element* ele_cur = elements[elecur];
//
// 				  // create vector "ephinp" holding scalar phi values for this element
// 				  Epetra_SerialDenseMatrix ephinp(nen_,1);
//
// 				  //temporal vector necessary just for function ExtractMyValues...
// 				  //that is requiring vectors and not matrices
// 				  vector<double> etemp(nen_);
//
// 				  // remark: vector "lm" is neccessary, because ExtractMyValues() only accepts "vector<int>"
// 				  // arguments, but ele->NodeIds delivers an "int*" argument
// 				  vector<int> lm(nen_);
//
// 				  // get vector of node GIDs for this element
// 				  const int* nodeids_cur = ele_cur->NodeIds();
// 				  for (int inode=0; inode < nen_; inode++)
// 					  lm[inode] = nodeids_cur[inode];
//
// 				  // get entries in "gfuncvalues" corresponding to node GIDs "lm" and store them in "ephinp"
// 				  DRT::UTILS::ExtractMyValues(*phinp, etemp, lm);
//
// 				  for (int k=0; k<nen_; k++)
// 					  ephinp(k,0) = etemp[k];
//
// 				  //current node is number iquadcur of current element
// 				  int iquadcur = 0;
// 				  for (int k=0; k<nen_; k++)
// 				  {
// 					  if (actnode->Id()==nodeids_cur[k])
// 						  iquadcur = k;
// 				  }
// 				  //local coordinates of current node
// 				  LINALG::Matrix<3,1> xsi;
// 				  LINALG::SerialDenseMatrix eleCoordMatrix=DRT::UTILS::getEleNodeNumbering_nodes_paramspace(distype);
// 				  for (int k=0; k<3; k++)
// 					  xsi(k,0)=eleCoordMatrix(k,iquadcur);
// 				  Epetra_SerialDenseMatrix deriv(3,nen_);
// 				  DRT::UTILS::shape_function_3D_deriv1(deriv,xsi(0,0),xsi(1,0),xsi(2,0),distype);
// 				  //xyze are the positions of the nodes in the global coordinate system
// 				  Epetra_SerialDenseMatrix xyze(3,nen_);
// 				  const DRT::Node*const* nodes = ele->Nodes();
// 				  for (int inode=0; inode<nen_; inode++)
// 				  {
// 					  const double* x = nodes[inode]->X();
// 					  xyze(0,inode) = x[0];
// 					  xyze(1,inode) = x[1];
// 					  xyze(2,inode) = x[2];
// 				  }
//
// 				  //get transposed of the jacobian matrix
// 				  Epetra_SerialDenseMatrix xjm(3,3);
// 				  //computing: this = 0*this+1*deriv*(xyze)T
// 				  xjm.Multiply('N','T',1.0,deriv,xyze,0.0);
// 				  //xji=xjm^-1
// 				  Epetra_SerialDenseMatrix xji(xjm);
// 				  LINALG::Matrix<3,3> xjm_temp;
// 				  for (int i1=0; i1<3; i1++)
// 					  for (int i2=0; i2<3; i2++)
// 						  xjm_temp(i1,i2) = xjm(i1,i2);
// 				  LINALG::Matrix<3,3> xji_temp;
// 				  const double det = xji_temp.Invert(xjm_temp);
// 				  if (det < 1e-16)
// 					  dserror("zero or negative jacobian determinant");
//
// 				  for (int i1=0; i1<3; i1++)
// 					  for (int i2=0; i2<3; i2++)
// 						  xji(i1,i2) = xji_temp(i1,i2);
//
// 				  Epetra_SerialDenseMatrix derxy(3,nen_);
// 				  //compute global derivatives
// 				  derxy.Multiply('N','N',1.0,xji,deriv,0.0);
//
// 				  gradphi_node.Multiply('N','N',1.0,derxy,ephinp,0.0);
//
// 				  //an average value is calculated of all adjacent elements to get second order accuracy
// 				  for (int k=0; k<3; k++)
// 					  gradphi(k,0) = gradphi(k,0)+gradphi_node(k,0);
// 				  if (elecur==(numadnodes-1))
// 				  {
// 					  for (int k=0; k<3; k++)
// 						  gradphi(k,0) = gradphi(k,0)/numadnodes;
// 				  }
//
// 			  }//end loop over all adjacent elements
//
// 			  for (int j=0; j<3; j++)
// 				  gradphi_nodescur(lnodeid,j)=gradphi(j,0);
//
// 		  }//end loop over all nodes of current element
//
// 		 gradphi_gpscur.Multiply('N','N',1.0,expol_invert,gradphi_nodescur,0.0);
//end REINHARD

    for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
    {
      const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad,ele->Id());
      //----------------------------------------------------------------------
      // get material parameters (evaluation at integration point)
      //----------------------------------------------------------------------
      if (mat_gp_) GetMaterialParams(ele);

      for (int k=0;k<numscal_;++k) // deal with a system of transported scalars
      {
        // get velocity at integration point
        velint_.Multiply(evelnp_,funct_);

        // convective part in convective form: rho*u_x*N,x+ rho*u_y*N,y
        conv_.MultiplyTN(derxy_,velint_);

        // momentum divergence required for conservative form
        if (conservative_) GetVelocityDivergence(vdiv_,evelnp_,derxy_);

        //--------------------------------------------------------------------
        // calculation of subgrid diffusivity and stabilization parameter(s)
        // at integration point
        //--------------------------------------------------------------------
        if (tau_gp_)
        {
          // calculation of all-scale subgrid diffusivity (artificial or due to
          // constant-coefficient Smagorinsky model) at integration point
          if (assgd or turbmodel)
            CalcSubgrDiff(dt,timefac,whichassgd,assgd,turbmodel,Cs,tpn,vol,k);

          // calculation of fine-scale artificial subgrid diffusivity
          // at integration point
          if (fssgd) CalcFineScaleSubgrDiff(ele,subgrdiff,whichfssgd,Cs,tpn,vol,k);

          // generating copy of diffusivity for use in CalTau routine
          double diffus = diffus_[k];

          // calculation of stabilization parameter at integration point
          CalTau(ele,diffus,dt,timefac,whichtau,vol,k);
        }

        // subgrid-scale velocity
        if (sgvel_)
        {
          // calculate subgrid-scale velocity
          CalcSubgrVelocity(ele,time,dt,timefac,k);

          // calculate subgrid-scale convective part
          sgconv_.MultiplyTN(derxy_,sgvelint_);
        }
        else
        {
          sgvelint_.Clear();
          sgconv_.Clear();
        }

        // get history data (or acceleration)
        hist_[k] = funct_.Dot(ehist_[k]);

        // get bodyforce in gausspoint (divided by shcacp)
        // (For temperature equation, time derivative of thermodynamic pressure
        //  is added, if not constant, and for temperature equation of a reactive
        //  equation system, a reaction-rate term is added.)
        rhs_[k] = bodyforce_[k].Dot(funct_) / shcacp_;
        rhs_[k] += thermpressdt_ / shcacp_;
        rhs_[k] += reatemprhs_[k] / shcacp_;

      // gradient of current scalar value

//REINHARD
//        for (int j=0; j<3; j++)
//          gradphi_(j,0)=gradphi_gpscur(iquad,j);
//end REINHARD

        // compute matrix and rhs
        CalMatAndRHS(sys_mat,
                     residual,
                     fac,
                     fssgd,
                     timefac,
                     alphaF,
                     k);
      } // loop over each scalar
    }
  }
  else // 'standard' scalar transport
  {
    for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
    {
      const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad,ele->Id());

      //----------------------------------------------------------------------
      // get material parameters (evaluation at integration point)
      //----------------------------------------------------------------------
      if (mat_gp_) GetMaterialParams(ele);

      for (int k=0;k<numscal_;++k) // deal with a system of transported scalars
      {
        // get velocity at integration point
        velint_.Multiply(evelnp_,funct_);

        // convective part in convective form: rho*u_x*N,x+ rho*u_y*N,y
        conv_.MultiplyTN(derxy_,velint_);

        // velocity divergence required for conservative form
        if (conservative_) GetVelocityDivergence(vdiv_,evelnp_,derxy_);

        //--------------------------------------------------------------------
        // calculation of subgrid diffusivity and stabilization parameter(s)
        // at integration point
        //--------------------------------------------------------------------
        if (tau_gp_)
        {
          // calculation of all-scale subgrid diffusivity (artificial or due to
          // constant-coefficient Smagorinsky model) at integration point
          if (assgd or turbmodel)
            CalcSubgrDiff(dt,timefac,whichassgd,assgd,turbmodel,Cs,tpn,vol,k);

          // calculation of fine-scale artificial subgrid diffusivity
          // at integration point
          if (fssgd) CalcFineScaleSubgrDiff(ele,subgrdiff,whichfssgd,Cs,tpn,vol,k);

          // generating copy of diffusivity for use in CalTau routine
          double diffus = diffus_[k];

          // calculation of stabilization parameter at integration point
          CalTau(ele,diffus,dt,timefac,whichtau,vol,k);
        }

        // subgrid-scale velocity
        if (sgvel_)
        {
          // calculate subgrid-scale velocity
          CalcSubgrVelocity(ele,time,dt,timefac,k);

          // calculate subgrid-scale convective part
          sgconv_.MultiplyTN(derxy_,sgvelint_);
        }
        else
        {
          sgvelint_.Clear();
          sgconv_.Clear();
        }

        // get history data (or acceleration)
        hist_[k] = funct_.Dot(ehist_[k]);

        // get bodyforce in gausspoint (divided by shcacp)
        // (For temperature equation, time derivative of thermodynamic pressure
        //  is added, if not constant, and for temperature equation of a reactive
        //  equation system, a reaction-rate term is added.)
        rhs_[k] = bodyforce_[k].Dot(funct_)/shcacp_;
        rhs_[k] += thermpressdt_/shcacp_;
        rhs_[k] += densnp_[k]*reatemprhs_[k];

        // compute matrix and rhs
        CalMatAndRHS(sys_mat,
                     residual,
                     fac,
                     fssgd,
                     timefac,
                     alphaF,
                     k);
      } // loop over each scalar
    }
  } // integration loop

  return;
}


/*----------------------------------------------------------------------*
 |  get the body force  (private)                              gjb 06/08|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::BodyForce(
    const DRT::Element*    ele,
    const double           time
)
{
  vector<DRT::Condition*> myneumcond;

  // check whether all nodes have a unique VolumeNeumann condition
  switch(nsd_)
  {
  case 3:
    DRT::UTILS::FindElementConditions(ele, "VolumeNeumann", myneumcond);
  break;
  case 2:
    DRT::UTILS::FindElementConditions(ele, "SurfaceNeumann", myneumcond);
  break;
  case 1:
    DRT::UTILS::FindElementConditions(ele, "LineNeumann", myneumcond);
  break;
  default:
    dserror("Illegal number of space dimensions: %d",nsd_);
  }

  if (myneumcond.size()>1)
    dserror("more than one VolumeNeumann cond on one node");

  if (myneumcond.size()==1)
  {
    // find out whether we will use a time curve
    const vector<int>* curve  = myneumcond[0]->Get<vector<int> >("curve");
    int curvenum = -1;

    if (curve) curvenum = (*curve)[0];

    // initialisation
    double curvefac(0.0);

    if (curvenum >= 0) // yes, we have a timecurve
    {
      // time factor for the intermediate step
      if(time >= 0.0)
      {
        curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
      }
      else
      {
        // A negative time value indicates an error.
        dserror("Negative time value in body force calculation: time = %f",time);
      }
    }
    else // we do not have a timecurve --- timefactors are constant equal 1
    {
      curvefac = 1.0;
    }

    // get values and switches from the condition
    const vector<int>*    onoff = myneumcond[0]->Get<vector<int> >   ("onoff");
    const vector<double>* val   = myneumcond[0]->Get<vector<double> >("val"  );

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
      // we have no dead load
      bodyforce_[idof].Clear();
    }
  }

  return;

} //ScaTraImpl::BodyForce


//REINHARD
/*----------------------------------------------------------------------*
 |  body force for sign function on the right hand side  (private)      |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::BodyForceReinit(
       const DRT::Element* ele,
       const double time)
{
//REINHARD

  double signum = 0.0;
  double epsilon = 0.015; //1.5*h in Sussman1994
  vector<int> onoff(numdofpernode_);
  onoff[0]=1;
  for (int k=1; k<numdofpernode_; k++)
    onoff[k]=0;

  for(int idof=0;idof<numdofpernode_;idof++)
  {
    for (int jnode=0; jnode<nen_; jnode++) //loop over all nodes of this element
    {
      if (ephinp_[idof](jnode,0)<-epsilon)
        signum = -1.0;
      else if (ephinp_[idof](jnode,0)>epsilon)
        signum = 1.0;
      else
        signum = ephinp_[idof](jnode,0)/epsilon + sin(PI*ephinp_[idof](jnode,0)/epsilon)/PI;

      // value_rhs
      (bodyforce_[idof])(jnode) = onoff[idof]*signum;
    }
  }

  return;
} //ScaTraImpl::BodyForceReinit
//end REINHARD


/*----------------------------------------------------------------------*
 |  get the material constants  (private)                      gjb 10/08|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::GetMaterialParams(
    const DRT::Element*  ele
)
{
// get the material
RefCountPtr<MAT::Material> material = ele->Material();

// get diffusivity / diffusivities
if (material->MaterialType() == INPAR::MAT::m_matlist)
{
  const MAT::MatList* actmat = static_cast<const MAT::MatList*>(material.get());

  for (int k = 0;k<numscal_;++k)
  {
    // set reaction coeff. and temperature rhs for reactive equation system to zero
    reacoeff_[k]   = 0.0;
    reatemprhs_[k] = 0.0;

    // set specific heat capacity at constant pressure to 1.0
    shcacp_ = 1.0;

    // set density at various time steps and density gradient factor to 1.0/0.0
    densn_[k]       = 1.0;
    densnp_[k]      = 1.0;
    densam_[k]      = 1.0;
    densgradfac_[k] = 0.0;

    const int matid = actmat->MatID(k);
    Teuchos::RCP<const MAT::Material> singlemat = actmat->MaterialById(matid);

    if (singlemat->MaterialType() == INPAR::MAT::m_ion)
    {
      const MAT::Ion* actsinglemat = static_cast<const MAT::Ion*>(singlemat.get());
      valence_[k] = actsinglemat->Valence();
      diffus_[k] = actsinglemat->Diffusivity();
      diffusvalence_[k] = valence_[k]*diffus_[k];
    }
    else if (singlemat->MaterialType() == INPAR::MAT::m_arrhenius_spec)
    {
      const MAT::ArrheniusSpec* actsinglemat = static_cast<const MAT::ArrheniusSpec*>(singlemat.get());

      // compute temperature
      const double tempnp = funct_.Dot(ephinp_[numscal_-1]);

      // compute diffusivity according to Sutherland law
      diffus_[k] = actsinglemat->ComputeDiffusivity(tempnp);

      // compute reaction coefficient for species equation
      reacoeff_[k] = actsinglemat->ComputeReactionCoeff(tempnp);

      // set reaction flag to true
      reaction_ = true;
    }
    else if (singlemat->MaterialType() == INPAR::MAT::m_arrhenius_temp)
    {
      if (k != numscal_-1) dserror("Temperature equation always needs to be the last variable for reactive equation system!");

      const MAT::ArrheniusTemp* actsinglemat = static_cast<const MAT::ArrheniusTemp*>(singlemat.get());

      // get specific heat capacity at constant pressure
      shcacp_ = actsinglemat->Shc();

      // compute species mass fraction and temperature
      const double spmf   = funct_.Dot(ephinp_[0]);
      const double tempnp = funct_.Dot(ephinp_[k]);

      // compute diffusivity according to Sutherland law
      diffus_[k] = actsinglemat->ComputeDiffusivity(tempnp);

      // compute density based on temperature and thermodynamic pressure
      densnp_[k] = actsinglemat->ComputeDensity(tempnp,thermpressnp_);

      if (is_genalpha_)
      {
        // compute density at n+alpha_M
        const double tempam = funct_.Dot(ephiam_[k]);
        densam_[k] = actsinglemat->ComputeDensity(tempam,thermpressam_);

        if (not is_incremental_)
        {
          // compute density at n (thermodynamic pressure approximated at n+alpha_M)
          const double tempn = funct_.Dot(ephin_[k]);
          densn_[k] = actsinglemat->ComputeDensity(tempn,thermpressam_);
        }
        else densn_[k] = 1.0;
      }
      else densam_[k] = densnp_[k];

      // factor for density gradient
      densgradfac_[k] = -densnp_[k]/tempnp;

      // compute sum of reaction rates for temperature equation divided by specific
      // heat capacity -> will be considered a right-hand side contribution
      reatemprhs_[k] = actsinglemat->ComputeReactionRHS(spmf,tempnp)/shcacp_;

      // set reaction flag to true
      reaction_ = true;
    }
    else if (singlemat->MaterialType() == INPAR::MAT::m_scatra)
    {
      const MAT::ScatraMat* actsinglemat = static_cast<const MAT::ScatraMat*>(singlemat.get());
      diffus_[k] = actsinglemat->Diffusivity();

      // in case of reaction with constant coefficient, read coefficient and
      // set reaction flag to true
      reacoeff_[k] = actsinglemat->ReaCoeff();
      if (reacoeff_[k] > EPS15) reaction_ = true;
    }
    else dserror("material type not allowed");

    // check whether there is zero or negative (physical) diffusivity
    if (diffus_[k] < EPS15) dserror("zero or negative (physical) diffusivity");
  }
}
else if (material->MaterialType() == INPAR::MAT::m_scatra)
{
  const MAT::ScatraMat* actmat = static_cast<const MAT::ScatraMat*>(material.get());

  dsassert(numdofpernode_==1,"more than 1 dof per node for SCATRA material");

  // get constant diffusivity
  diffus_[0] = actmat->Diffusivity();

  // in case of reaction with (non-zero) constant coefficient:
  // read coefficient and set reaction flag to true
  reacoeff_[0] = actmat->ReaCoeff();
  if (reacoeff_[0] > EPS15) reaction_ = true;

  // set specific heat capacity at constant pressure to 1.0
  shcacp_ = 1.0;

  // set temperature rhs for reactive equation system to zero
  reatemprhs_[0] = 0.0;

  // set density at various time steps and density gradient factor to 1.0/0.0
  densn_[0]       = 1.0;
  densnp_[0]      = 1.0;
  densam_[0]      = 1.0;
  densgradfac_[0] = 0.0;
}
else if (material->MaterialType() == INPAR::MAT::m_ion)
{
  const MAT::Ion* actsinglemat = static_cast<const MAT::Ion*>(material.get());

  dsassert(numdofpernode_==1,"more than 1 dof per node for single ion material");

  // set reaction coeff. and temperature rhs for reactive equation system to zero
  reacoeff_[0]   = 0.0;
  reatemprhs_[0] = 0.0;
  // set specific heat capacity at constant pressure to 1.0
  shcacp_ = 1.0;
  // set density at various time steps and density gradient factor to 1.0/0.0
  densn_[0]       = 1.0;
  densnp_[0]      = 1.0;
  densam_[0]      = 1.0;
  densgradfac_[0] = 0.0;

  // get constant diffusivity
  diffus_[0] = actsinglemat->Diffusivity();
  valence_[0] = 0.0; // remains unused -> we only do convection-diffusion in this case!
  diffusvalence_[0] = 0.0; // remains unused
}
else if (material->MaterialType() == INPAR::MAT::m_mixfrac)
{
  const MAT::MixFrac* actmat = static_cast<const MAT::MixFrac*>(material.get());

  dsassert(numdofpernode_==1,"more than 1 dof per node for mixture-fraction material");

  // compute mixture fraction at n+1 or n+alpha_F
  const double mixfracnp = funct_.Dot(ephinp_[0]);

  // compute dynamic diffusivity at n+1 or n+alpha_F based on mixture fraction
  diffus_[0] = actmat->ComputeDiffusivity(mixfracnp);

  // compute density at n+1 or n+alpha_F based on mixture fraction
  densnp_[0] = actmat->ComputeDensity(mixfracnp);

  // set specific heat capacity at constant pressure to 1.0
  shcacp_ = 1.0;

  if (is_genalpha_)
  {
    // compute density at n+alpha_M
    const double mixfracam = funct_.Dot(ephiam_[0]);
    densam_[0] = actmat->ComputeDensity(mixfracam);

    if (not is_incremental_)
    {
      // compute density at n
      const double mixfracn = funct_.Dot(ephin_[0]);
      densn_[0] = actmat->ComputeDensity(mixfracn);
    }
    else densn_[0] = 1.0;
  }
  else densam_[0] = densnp_[0];

  // factor for density gradient
  densgradfac_[0] = -densnp_[0]*densnp_[0]*actmat->EosFacA();

  // set reaction coeff. and temperature rhs for reactive equation system to zero
  reacoeff_[0] = 0.0;
  reatemprhs_[0] = 0.0;

  // get also fluid viscosity if subgrid-scale velocity is to be included
  if (sgvel_) visc_ = actmat->ComputeViscosity(mixfracnp);
}
else if (material->MaterialType() == INPAR::MAT::m_sutherland)
{
  const MAT::Sutherland* actmat = static_cast<const MAT::Sutherland*>(material.get());

  dsassert(numdofpernode_==1,"more than 1 dof per node for Sutherland material");

  // get specific heat capacity at constant pressure
  shcacp_ = actmat->Shc();

  // compute temperature at n+1 or n+alpha_F
  const double tempnp = funct_.Dot(ephinp_[0]);

  // compute diffusivity according to Sutherland law
  diffus_[0] = actmat->ComputeDiffusivity(tempnp);

  // compute density at n+1 or n+alpha_F based on temperature
  // and thermodynamic pressure
  densnp_[0] = actmat->ComputeDensity(tempnp,thermpressnp_);

  if (is_genalpha_)
  {
    // compute density at n+alpha_M
    const double tempam = funct_.Dot(ephiam_[0]);
    densam_[0] = actmat->ComputeDensity(tempam,thermpressam_);

    if (not is_incremental_)
    {
      // compute density at n (thermodynamic pressure approximated at n+alpha_M)
      const double tempn = funct_.Dot(ephin_[0]);
      densn_[0] = actmat->ComputeDensity(tempn,thermpressam_);
    }
    else densn_[0] = 1.0;
  }
  else densam_[0] = densnp_[0];

  // factor for density gradient
  densgradfac_[0] = -densnp_[0]/tempnp;

  // set reaction coeff. and temperature rhs for reactive equation system to zero
  reacoeff_[0] = 0.0;
  reatemprhs_[0] = 0.0;

  // get also fluid viscosity if subgrid-scale velocity is to be included
  if (sgvel_) visc_ = actmat->ComputeViscosity(tempnp);
}
else if (material->MaterialType() == INPAR::MAT::m_arrhenius_pv)
{
  const MAT::ArrheniusPV* actmat = static_cast<const MAT::ArrheniusPV*>(material.get());

  dsassert(numdofpernode_==1,"more than 1 dof per node for progress-variable material");

  // get progress variable at n+1 or n+alpha_F
  const double provarnp = funct_.Dot(ephinp_[0]);

  // get specific heat capacity at constant pressure and
  // compute temperature based on progress variable
  shcacp_ = actmat->ComputeShc(provarnp);
  const double tempnp = actmat->ComputeTemperature(provarnp);

  // compute density at n+1 or n+alpha_F
  densnp_[0] = actmat->ComputeDensity(provarnp);

  if (is_genalpha_)
  {
    // compute density at n+alpha_M
    const double provaram = funct_.Dot(ephiam_[0]);
    densam_[0] = actmat->ComputeDensity(provaram);

    if (not is_incremental_)
    {
      // compute density at n
      const double provarn = funct_.Dot(ephin_[0]);
      densn_[0] = actmat->ComputeDensity(provarn);
    }
    else densn_[0] = 1.0;
  }
  else densam_[0] = densnp_[0];

  // factor for density gradient: unburnt-burnt density difference
  densgradfac_[0] = actmat->BurDens() - actmat->UnbDens();

  // compute diffusivity according to Sutherland law
  diffus_[0] = actmat->ComputeDiffusivity(tempnp);

  // compute reaction coefficient for progress variable
  reacoeff_[0] = actmat->ComputeReactionCoeff(tempnp);

  // compute right-hand side contribution for progress variable
  // -> equal to reaction coefficient
  reatemprhs_[0] = reacoeff_[0];

  // set reaction flag to true
  reaction_ = true;

  // get also fluid viscosity if subgrid-scale velocity is to be included
  if (sgvel_) visc_ = actmat->ComputeViscosity(tempnp);
}
else if (material->MaterialType() == INPAR::MAT::m_ferech_pv)
{
  const MAT::FerEchPV* actmat = static_cast<const MAT::FerEchPV*>(material.get());

  dsassert(numdofpernode_==1,"more than 1 dof per node for progress-variable material");

  // get progress variable at n+1 or n+alpha_F
  const double provarnp = funct_.Dot(ephinp_[0]);

  // get specific heat capacity at constant pressure and
  // compute temperature based on progress variable
  shcacp_ = actmat->ComputeShc(provarnp);
  const double tempnp = actmat->ComputeTemperature(provarnp);

  // compute density at n+1 or n+alpha_F
  densnp_[0] = actmat->ComputeDensity(provarnp);

  if (is_genalpha_)
  {
    // compute density at n+alpha_M
    const double provaram = funct_.Dot(ephiam_[0]);
    densam_[0] = actmat->ComputeDensity(provaram);

    if (not is_incremental_)
    {
      // compute density at n
      const double provarn = funct_.Dot(ephin_[0]);
      densn_[0] = actmat->ComputeDensity(provarn);
    }
    else densn_[0] = 1.0;
  }
  else densam_[0] = densnp_[0];

  // factor for density gradient: unburnt-burnt density difference
  densgradfac_[0] = actmat->BurDens() - actmat->UnbDens();

  // compute diffusivity according to Sutherland law
  diffus_[0] = actmat->ComputeDiffusivity(tempnp);

  // compute reaction coefficient for progress variable
  reacoeff_[0] = actmat->ComputeReactionCoeff(provarnp);

  // compute right-hand side contribution for progress variable
  // -> equal to reaction coefficient
  reatemprhs_[0] = reacoeff_[0];

  // set reaction flag to true
  reaction_ = true;

  // get also fluid viscosity if subgrid-scale velocity is to be included
  if (sgvel_) visc_ = actmat->ComputeViscosity(tempnp);
}
else dserror("Material type is not supported");

// check whether there is zero or negative (physical) diffusivity
if (diffus_[0] < EPS15) dserror("zero or negative (physical) diffusivity");

return;
} //ScaTraImpl::GetMaterialParams


/*----------------------------------------------------------------------*
 |  calculate all-scale art. subgrid diffusivity (private)     vg 10/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalcSubgrDiff(
    const double                          dt,
    const double                          timefac,
    const enum INPAR::SCATRA::AssgdType   whichassgd,
    const bool                            assgd,
    const bool                            turbmodel,
    const double                          Cs,
    const double                          tpn,
    const double                          vol,
    const int                             k
  )
{
  // get number of dimensions
  const double dim = (double) nsd_;

  // get characteristic element length as cubic root of element volume
  // (2D: square root of element area, 1D: element length)
  const double h = pow(vol,(1.0/dim));

  // artficial all-scale subgrid diffusivity
  if (assgd )
  {
    // classical linear artificial all-scale subgrid diffusivity
    if (whichassgd == INPAR::SCATRA::assgd_artificial)
    {
      // get element-type constant
      const double mk = SCATRA::MK<distype>();

      // velocity norm
      const double vel_norm = velint_.Norm2();

      // parameter relating convective and diffusive forces + respective switch
      const double epe = mk * densnp_[k] * vel_norm * h / diffus_[k];
      const double xi = DMAX(epe,1.0);

      // compute subgrid diffusivity
      sgdiff_[k] = (DSQR(h)*mk*DSQR(vel_norm)*DSQR(densnp_[k]))/(2.0*diffus_[k]*xi);
    }
    else
    {
      // gradient of current scalar value
      gradphi_.Multiply(derxy_,ephinp_[k]);

      // gradient norm
      const double grad_norm = gradphi_.Norm2();

      if (grad_norm > EPS10)
      {
        // initialize residual and compute values required for residual
        double residual  = 0.0;

        // get non-density-weighted history data (or acceleration)
        hist_[k] = funct_.Dot(ehist_[k]);

        // convective term using current scalar value
        double conv_phi = velint_.Dot(gradphi_);

        // diffusive term using current scalar value for higher-order elements
        double diff_phi = 0.0;
        if (use2ndderiv_) diff_phi = diff_.Dot(ephinp_[k]);

       // reactive term using current scalar value
        double rea_phi = 0.0;
        if (reaction_)
        {
          // scalar at integration point
          const double phi = funct_.Dot(ephinp_[k]);

          rea_phi = densnp_[k]*reacoeff_[k]*phi;
        }

        // get bodyforce (divided by shcacp)
        // (For temperature equation, time derivative of thermodynamic pressure
        //  is added, if not constant, and for temperature equation of a reactive
        //  equation system, a reaction-rate term is added.)
        rhs_[k] = bodyforce_[k].Dot(funct_)/shcacp_;
        rhs_[k] += thermpressdt_/shcacp_;
        rhs_[k] += densnp_[k]*reatemprhs_[k];

        // computation of residual depending on respective time-integration scheme
        if (is_genalpha_)
          residual = densam_[k]*hist_[k] + densnp_[k]*conv_phi - diff_phi + rea_phi - rhs_[k];
        else if (is_stationary_)
          residual = conv_phi - diff_phi + rea_phi - rhs_[k];
        else
        {
          // compute density-weighted scalar at integration point
          double dens_phi = funct_.Dot(ephinp_[k]);

          residual = (densnp_[k]*(dens_phi - hist_[k]) + timefac * (densnp_[k]*conv_phi - diff_phi + rea_phi - rhs_[k])) / dt;
        }

        // for the present definitions, sigma and a specific term (either
        // residual or convective term) are different
        double sigma = 0.0;
        double specific_term = 0.0;
        switch (whichassgd)
        {
          case INPAR::SCATRA::assgd_hughes:
          {
            // get norm of velocity vector b_h^par
            const double vel_norm_bhpar = abs(conv_phi/grad_norm);

            // compute stabilization parameter based on b_h^par
            // (so far, only exact formula for stationary 1-D implemented)
            // element Peclet number relating convective and diffusive forces
            double epe = 0.5 * vel_norm_bhpar * h / diffus_[k];
            const double pp = exp(epe);
            const double pm = exp(-epe);
            double xi = 0.0;
            double tau_bhpar = 0.0;
            if (epe >= 700.0) tau_bhpar = 0.5*h/vel_norm_bhpar;
            else if (epe < 700.0 and epe > EPS15)
            {
              xi = (((pp+pm)/(pp-pm))-(1.0/epe)); // xi = coth(epe) - 1/epe
              // compute optimal stabilization parameter
              tau_bhpar = 0.5*h*xi/vel_norm_bhpar;
            }

            // compute sigma
            sigma = max(0.0,tau_bhpar-tau_[k]);

            // set specific term to convective term
            specific_term = conv_phi;
          }
          break;
          case INPAR::SCATRA::assgd_tezduyar:
          {
            // velocity norm
            const double vel_norm = velint_.Norm2();

            // get norm of velocity vector b_h^par
            const double vel_norm_bhpar = abs(conv_phi/grad_norm);

            // compute stabilization parameter based on b_h^par
            // (so far, only exact formula for stationary 1-D implemented)

            // compute sigma (version 1 according to John and Knobloch (2007))
            //sigma = (h/vel_norm)*(1.0-(vel_norm_bhpar/vel_norm));

            // compute sigma (version 2 according to John and Knobloch (2007))
            // setting scaling phi_0=1.0 as in John and Knobloch (2007)
            const double phi0 = 1.0;
            sigma = (h*h*grad_norm/(vel_norm*phi0))*(1.0-(vel_norm_bhpar/vel_norm));

            // set specific term to convective term
            specific_term = conv_phi;
          }
          break;
          case INPAR::SCATRA::assgd_docarmo:
          case INPAR::SCATRA::assgd_almeida:
          {
            // velocity norm
            const double vel_norm = velint_.Norm2();

            // get norm of velocity vector z_h
            const double vel_norm_zh = abs(residual/grad_norm);

            // parameter zeta differentiating approaches by doCarmo and Galeao (1991)
            // and Almeida and Silva (1997)
            double zeta = 0.0;
            if (whichassgd == INPAR::SCATRA::assgd_docarmo)
                 zeta = 1.0;
            else zeta = max(1.0,(conv_phi/residual));

            // compute sigma
            sigma = tau_[k]*max(0.0,(vel_norm/vel_norm_zh)-zeta);

            // set specific term to residual
            specific_term = residual;
          }
          break;
          default: dserror("unknown type of all-scale subgrid diffusivity\n");
        } //switch (whichassgd)

        // computation of subgrid diffusivity
        sgdiff_[k] = sigma*residual*specific_term/(grad_norm*grad_norm);
      }
      else sgdiff_[k] = 0.0;
    }
  }
  // all-scale subgrid diffusivity due to Smagorinsky model divided by
  // turbulent Prandtl number
  else if (turbmodel)
  {
    //
    // SMAGORINSKY MODEL
    // -----------------
    //                                   +-                                 -+ 1
    //                               2   |          / h \           / h \    | -
    //    visc          = dens * lmix  * | 2 * eps | u   |   * eps | u   |   | 2
    //        turbulent           |      |          \   / ij        \   / ij |
    //                            |      +-                                 -+
    //                            |
    //                            |      |                                   |
    //                            |      +-----------------------------------+
    //                            |           'resolved' rate of strain
    //                    mixing length
    // -> either provided by dynamic modeling procedure and stored in Cs_delta_sq
    // -> or computed based on fixed Smagorinsky constant Cs:
    //             Cs = 0.17   (Lilly --- Determined from filter
    //                          analysis of Kolmogorov spectrum of
    //                          isotropic turbulence)
    //             0.1 < Cs < 0.24 (depending on the flow)
    //

    // compute (all-scale) rate of strain
    double rateofstrain = -1.0e30;
    rateofstrain = GetStrainRate(evelnp_,derxy_,vderxy_);

    // subgrid diffusivity = subgrid viscosity / turbulent Prandtl number
    sgdiff_[k] = densnp_[k] * Cs * Cs * h * h * rateofstrain / tpn;

    // add subgrid viscosity to physical viscosity for computation
    // of subgrid-scale velocity when turbulence model is applied
    if (sgvel_) visc_ += sgdiff_[k]*tpn;
  }

  // compute sum of physical and all-scale subgrid diffusivity
  // -> set internal variable for use when calculating matrix and rhs
  diffus_[k] += sgdiff_[k];

  return;
} //ScaTraImpl::CalcSubgrDiff


/*----------------------------------------------------------------------*
 |  calculate fine-scale art. subgrid diffusivity (private)    vg 10/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalcFineScaleSubgrDiff(
    DRT::Element*                         ele,
    Epetra_SerialDenseVector&             subgrdiff,
    const enum INPAR::SCATRA::FSSUGRDIFF  whichfssgd,
    const double                          Cs,
    const double                          tpn,
    const double                          vol,
    const int                             k
  )
{
  // get number of dimensions
  const double dim = (double) nsd_;

  // get characteristic element length as cubic root of element volume
  // (2D: square root of element area, 1D: element length)
  const double h = pow(vol,(1.0/dim));

  //----------------------------------------------------------------------
  // computation of fine-scale subgrid diffusivity for non-incremental
  // solver -> only artificial subgrid diffusivity
  // (values are stored in subgrid-diffusivity-scaling vector)
  //----------------------------------------------------------------------
  if (not is_incremental_)
  {
    // get element-type constant
    const double mk = SCATRA::MK<distype>();

    // velocity norm
    const double vel_norm = velint_.Norm2();

    // parameter relating convective and diffusive forces + respective switch
    const double epe = mk * densnp_[k] * vel_norm * h / diffus_[k];
    const double xi = DMAX(epe,1.0);

    // compute artificial subgrid diffusivity
    sgdiff_[k] = (DSQR(h)*mk*DSQR(vel_norm)*DSQR(densnp_[k]))/(2.0*diffus_[k]*xi);

    // compute entries of (fine-scale) subgrid-diffusivity-scaling vector
    for (int vi=0; vi<nen_; ++vi)
    {
      subgrdiff(vi) = sgdiff_[k]/ele->Nodes()[vi]->NumElement();
    }
  }
  //----------------------------------------------------------------------
  // computation of fine-scale subgrid diffusivity for incremental solver
  // -> only all-scale Smagorinsky model
  //----------------------------------------------------------------------
  else
  {
  //if (whichfssgd == INPAR::SCATRA::fssugrdiff_smagorinsky_all)
  //{
    //
    // ALL-SCALE SMAGORINSKY MODEL
    // ---------------------------
    //                                      +-                                 -+ 1
    //                                  2   |          / h \           / h \    | -
    //    visc          = dens * (C_S*h)  * | 2 * eps | u   |   * eps | u   |   | 2
    //        turbulent                     |          \   / ij        \   / ij |
    //                                      +-                                 -+
    //                                      |                                   |
    //                                      +-----------------------------------+
    //                                            'resolved' rate of strain
    //

    // compute (all-scale) rate of strain
    double rateofstrain = -1.0e30;
    rateofstrain = GetStrainRate(evelnp_,derxy_,vderxy_);

    // subgrid diffusivity = subgrid viscosity / turbulent Prandtl number
    sgdiff_[k] = densnp_[k] * Cs * Cs * h * h * rateofstrain / tpn;
  /*}
  else if (whichfssgd == INPAR::SCATRA::fssugrdiff_smagorinsky_small)
  {
    //
    // FINE-SCALE SMAGORINSKY MODEL
    // ----------------------------
    //                                      +-                                 -+ 1
    //                                  2   |          /    \          /   \    | -
    //    visc          = dens * (C_S*h)  * | 2 * eps | fsu |   * eps | fsu |   | 2
    //        turbulent                     |          \   / ij        \   / ij |
    //                                      +-                                 -+
    //                                      |                                   |
    //                                      +-----------------------------------+
    //                                            'resolved' rate of strain
    //

    // fine-scale rate of strain
    double fsrateofstrain = -1.0e30;
    fsrateofstrain = GetStrainRate(fsevelnp_,derxy_,fsvderxy_);

    sgdiff_[k] = densnp_[k] * Cs * Cs * h * h * fsrateofstrain;
  }*/

    // compute gradient of fine-scale part of scalar value
    fsgradphi_.Multiply(derxy_,fsphinp_[k]);
  }

  return;
} //ScaTraImpl::CalcFineScaleSubgrDiff


/*----------------------------------------------------------------------*
 |  calculate stabilization parameter  (private)              gjb 06/08 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalTau(
    DRT::Element*                         ele,
    double                                diffus,
    const double                          dt,
    const double                          timefac,
    const enum INPAR::SCATRA::TauType     whichtau,
    const double                          vol,
    const int                             k
  )
{
  // get element-type constant for tau
  const double mk = SCATRA::MK<distype>();

  //----------------------------------------------------------------------
  // computation of stabilization parameter
  //----------------------------------------------------------------------
  switch (whichtau)
  {
    // stabilization parameter definition according to Bazilevs et al. (2007)
    case INPAR::SCATRA::tau_bazilevs:
    {
      // effective velocity at element center:
      // (weighted) convective velocity + individual migration velocity
      LINALG::Matrix<nsd_,1> veleff(velint_,false);
#ifdef MIGRATIONSTAB
      if (iselch_) // ELCH
      {
        veleff.Update(diffusvalence_[k],migvelint_,1.0);
      }
#endif
      /*
                                                                              1.0
                 +-                                                      -+ - ---
                 |        2                                               |   2.0
                 | 4.0*rho         n+1             n+1          2         |
          tau  = | -------  + rho*u     * G * rho*u     + C * mu  * G : G |
                 |     2                  -                I        -   - |
                 |   dt                   -                         -   - |
                 +-                                                      -+

      */
      /*          +-           -+   +-           -+   +-           -+
                  |             |   |             |   |             |
                  |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
            G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
             ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                  |    i     j  |   |    i     j  |   |    i     j  |
                  +-           -+   +-           -+   +-           -+
      */
      /*          +----
                   \
          G : G =   +   G   * G
          -   -    /     ij    ij
          -   -   +----
                   i,j
      */
      /*                               +----
               n+1             n+1     \         n+1              n+1
          rho*u     * G * rho*u     =   +   rho*u    * G   * rho*u
                      -                /         i     -ij        j
                      -               +----        -
                                        i,j
      */
      double G;
      double normG(0.0);
      double Gnormu(0.0);
      const double dens_sqr = densnp_[k]*densnp_[k];
      for (int nn=0;nn<nsd_;++nn)
      {
        for (int rr=0;rr<nsd_;++rr)
        {
          G = xij_(nn,0)*xij_(rr,0);
          for(int tt=1;tt<nsd_;tt++)
          {
            G += xij_(nn,tt)*xij_(rr,tt);
          }
          normG+=G*G;
          Gnormu+=dens_sqr*veleff(nn,0)*G*veleff(rr,0);
        }
      }

      // definition of constant:
      // 12.0/m_k = 36.0 for linear elements and 144.0 for quadratic elements
      // (differently defined, e.g., in Akkerman et al. (2008))
      const double CI = 12.0/mk;

      // stabilization parameters for stationary and instationary case, respectively
      if (is_stationary_)
        tau_[k] = 1.0/(sqrt(dens_sqr*reacoeff_[k]*reacoeff_[k]+Gnormu+CI*diffus*diffus*normG));
      else
        tau_[k] = 1.0/(sqrt(dens_sqr*(4.0/(dt*dt)+reacoeff_[k]*reacoeff_[k])+Gnormu+CI*diffus*diffus*normG));
    }
    break;
    case INPAR::SCATRA::tau_franca_valentin:
    // stabilization parameter definition according to Franca and Valentin (2000)
    {
      // get Euclidean norm of (weighted) velocity at element center
      double vel_norm;
#ifdef MIGRATIONSTAB
      if (iselch_) // ELCH
      {
        // get Euclidean norm of effective velocity at element center:
        // (weighted) convective velocity + individual migration velocity
        LINALG::Matrix<nsd_,1> veleff(velint_,false);

        veleff.Update(diffusvalence_[k],migvelint_,1.0);
        vel_norm = veleff.Norm2();

#ifdef VISUALIZE_ELEMENT_DATA
        veleff.Update(diffusvalence_[k],migvelint_,0.0);
        double vel_norm_mig = veleff.Norm2();
        double migepe2 = mk * vel_norm_mig * h / diffus;

        DRT::ELEMENTS::Transport* actele = dynamic_cast<DRT::ELEMENTS::Transport*>(ele);
        if (!actele) dserror("cast to Transport* failed");
        vector<double> v(1,migepe2);
        ostringstream temp;
        temp << k;
        string name = "Pe_mig_"+temp.str();
        actele->AddToData(name,v);
        name = "hk_"+temp.str();
        v[0] = h;
        actele->AddToData(name,v);
#endif
      }
      else
#endif
        vel_norm = velint_.Norm2();

      // get characteristic element length
      // There exist different definitions for 'the' characteristic element length h:

      // 1) streamlength (based on velocity vector at element centre) -> default
      // normed velocity at element centre
      LINALG::Matrix<nsd_,1> velino;
      if (vel_norm>=1e-6)
        velino.Update(1.0/vel_norm,velint_);
      else
      {
        velino.Clear();
        velino(0,0) = 1;
      }
      // get streamlength using the normed velocity at element centre
      LINALG::Matrix<nen_,1> tmp;
      tmp.MultiplyTN(derxy_,velino);
      const double val = tmp.Norm1();
      const double h = 2.0/val; // h=streamlength

      // 2) get element length for tau_Mp/tau_C: volume-equival. diameter -> not default
      // const double h = pow((6.*vol/PI),(1.0/3.0)); //warning: 3D formula

      // 3) use cubic root of the element volume as characteristic length -> not default
      //    2D case: characterisitc length is the square root of the element area
      //    1D case: characteristic length is the element length
      // get number of dimensions (convert from int to double)
      // const double dim = (double) nsd_;
      // const double h = pow(vol,(1.0/dim));

      // parameter relating convective and diffusive forces + respective switch
      double epe = mk * densnp_[k] * vel_norm * h / diffus;
      const double xi = DMAX(epe,1.0);

      // stabilization parameter for stationary and instationary case
      if (is_stationary_)
      {
        // parameter relating diffusive and reactive forces + respective switch
        double epe1 = 0.0;
        if (reaction_) epe1 = 2.0*diffus/(mk*densnp_[k]*reacoeff_[k]*DSQR(h));

        const double xi1 = DMAX(epe1,1.0);

        tau_[k] = DSQR(h)/(DSQR(h)*densnp_[k]*reacoeff_[k]*xi1 + 2.0*diffus*xi/mk);
      }
      else
      {
        // parameter relating diffusive and reactive forces + respective switch
        double epe1 = 0.0;
        if (reaction_)
             epe1 = 2.0*(timefac+1.0/reacoeff_[k])*diffus/(mk*densnp_[k]*DSQR(h));
        else epe1 = 2.0*timefac*diffus/(mk*densnp_[k]*DSQR(h));

        const double xi1 = DMAX(epe1,1.0);

        tau_[k] = DSQR(h)/(DSQR(h)*densnp_[k]*(reacoeff_[k]+1.0/timefac)*xi1 + 2.0*diffus*xi/mk);
      }

#ifdef VISUALIZE_ELEMENT_DATA
      // visualize resultant Pe number
      DRT::ELEMENTS::Transport* actele = dynamic_cast<DRT::ELEMENTS::Transport*>(ele);
      if (!actele) dserror("cast to Transport* failed");
      vector<double> v(1,epe);
      ostringstream temp;
      temp << k;
      string name = "Pe_"+temp.str();
      actele->AddToData(name,v);
#endif
    }
    break;
    case INPAR::SCATRA::tau_exact_1d:
    {
      // optimal tau (stationary 1D-problem using linear shape functions)
      if (not is_stationary_)
        dserror("Exact stabilization parameter only available for stationary case");

      // get number of dimensions (convert from int to double)
      const double dim = (double) nsd_;

      // get characteristic element length
      const double h = pow(vol,(1.0/dim));

      // get Euclidean norm of (weighted) velocity at element center
      double vel_norm = velint_.Norm2();

#if 0
      // element Peclet number relating convective and diffusive forces
double resdiffus(0.0);
      if (k==0) resdiffus = 0.00628095; //0.01125;
      if (k==1) resdiffus = 0.0211878;
      if (k==2) resdiffus = 0.03075393; //3075393;
       cout<<"resdiffus  for k "<<k <<" is = "<<resdiffus<<endl;


      double epe = 0.5 * densnp_[k] * vel_norm * h / resdiffus;
#endif
      double epe = 0.5 * densnp_[k] * vel_norm * h / diffus;
      const double pp = exp(epe);
      const double pm = exp(-epe);
      double xi = 0.0;
      if (epe >= 700.0) tau_[k] = 0.5*h/vel_norm;
      else if (epe < 700.0 and epe > EPS15)
      {
        xi = (((pp+pm)/(pp-pm))-(1.0/epe)); // xi = coth(epe) - 1/epe
        // compute optimal stabilization parameter
        tau_[k] = 0.5*h*xi/vel_norm;
#if 0
        cout<<"epe = "<<epe<<endl;
        cout<<"xi_opt  = "<<xi<<endl;
        cout<<"vel_norm  = "<<vel_norm<<endl;
        cout<<"tau_opt = "<<0.5*h*xi/vel_norm<<endl<<endl;
#endif
      }
      else tau_[k] = 0.0;
    }
    break;
    case INPAR::SCATRA::tau_oberai:
    {
      // tau according to Assad Oberai's suggestion
      if (is_stationary_) dserror("This stabilization parameter only available for instationary case");

      // get number of dimensions (convert from int to double)
      const double dim = (double) nsd_;

      // get characteristic element length
      const double h = pow(vol,(1.0/dim));

      // get Euclidean norm of (weighted) velocity at element center
      double vel_norm = velint_.Norm2();

      // compute element Peclet number
      // (caution: element Peclet number defined here without division by 2)
      const double epe = densnp_[k] * vel_norm * h / diffus;

      // compute CFL number
      const double cfl = vel_norm * dt / h;

      // compute k_bar
      const double pp = exp(epe/20.0);
      const double pm = exp(-epe/20.0);
      const double k_bar = 1.0 + (8.0*((pp-pm)/(pp+pm))/(1.0+((pp-pm)/(pp+pm))));

      // compute factors gamma, kappa, alpha and mu
      const double kp = exp(k_bar);
      const double km = exp(-k_bar);
      const double fac_gamma = exp(k_bar*cfl*(1.0-(k_bar/epe)));
      const double fac_kappa = (2.0/(kp+km))-1.0;
      const double fac_alpha = (kp-km)/(kp+km);
      const double fac_mu    = (2.0*(2.0/(kp+km))+1.0)/3.0;

      // compute tau
      tau_[k] = fac_mu*(fac_gamma-1.0)-(fac_gamma+1.0)*cfl*((fac_alpha/2.0)+(fac_kappa/epe))/(fac_alpha*(fac_gamma-1.0)+cfl*fac_kappa*(fac_gamma+1.0));
    }
    break;
    case INPAR::SCATRA::tau_zero:
    {
      // set tau's to zero (-> no stabilization effect)
      tau_[k] = 0.0;
    }
    break;
    default: dserror("Unknown definition of tau\n");
  } //switch (whichtau)

#ifdef VISUALIZE_ELEMENT_DATA
  // visualize stabilization parameter
  DRT::ELEMENTS::Transport* actele = dynamic_cast<DRT::ELEMENTS::Transport*>(ele);
  if (!actele) dserror("cast to Transport* failed");
  vector<double> v(1,tau_[k]);
  ostringstream temp;
  temp << k;
  string name = "tau_"+ temp.str();
  actele->AddToData(name,v);
#endif

  return;
} //ScaTraImpl::CalTau


/*----------------------------------------------------------------------*
 |  calculate subgrid-scale velocity                           vg 10/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalcSubgrVelocity(
    DRT::Element*  ele,
    const double   time,
    const double   dt,
    const double   timefac,
    const int      k
  )
{
  // definitions
  LINALG::Matrix<nsd_,1> acc;
  LINALG::Matrix<nsd_,1> conv;
  LINALG::Matrix<nsd_,1> gradp;
  LINALG::Matrix<nsd_,1> visc;
  LINALG::Matrix<nsd_,1> bodyforce;
  LINALG::Matrix<nsd_,nen_> nodebodyforce;

  // get acceleration or momentum history data
  acc.Multiply(eaccnp_,funct_);

  // get velocity derivatives
  vderxy_.MultiplyNT(evelnp_,derxy_);

  // compute convective fluid term
  conv.Multiply(vderxy_,velint_);

  // get pressure gradient
  gradp.Multiply(derxy_,eprenp_);

  //--------------------------------------------------------------------
  // get nodal values of fluid body force
  //--------------------------------------------------------------------
  vector<DRT::Condition*> myfluidneumcond;

  // check whether all nodes have a unique Fluid Neumann condition
  switch(nsd_)
  {
  case 3:
    DRT::UTILS::FindElementConditions(ele, "FluidVolumeNeumann", myfluidneumcond);
  break;
  case 2:
    DRT::UTILS::FindElementConditions(ele, "FluidSurfaceNeumann", myfluidneumcond);
  break;
  case 1:
    DRT::UTILS::FindElementConditions(ele, "FluidLineNeumann", myfluidneumcond);
  break;
  default:
    dserror("Illegal number of space dimensions: %d",nsd_);
  }

  if (myfluidneumcond.size()>1)
    dserror("more than one Fluid Neumann condition on one node");

  if (myfluidneumcond.size()==1)
  {
    // find out whether we will use a time curve
    const vector<int>* curve  = myfluidneumcond[0]->Get<vector<int> >("curve");
    int curvenum = -1;

    if (curve) curvenum = (*curve)[0];

    // initialisation
    double curvefac(0.0);

    if (curvenum >= 0) // yes, we have a timecurve
    {
      // time factor for the intermediate step
      // (negative time value indicates error)
      if(time >= 0.0)
        curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
      else
        dserror("Negative time value in body force calculation: time = %f",time);
    }
    else // we do not have a timecurve: timefactors are constant equal 1
      curvefac = 1.0;

    // get values and switches from the condition
    const vector<int>*    onoff = myfluidneumcond[0]->Get<vector<int> >   ("onoff");
    const vector<double>* val   = myfluidneumcond[0]->Get<vector<double> >("val"  );

    // set this condition to the body force array
    for(int isd=0;isd<nsd_;isd++)
    {
      for (int jnode=0; jnode<nen_; jnode++)
      {
        nodebodyforce(isd,jnode) = (*onoff)[isd]*(*val)[isd]*curvefac;
      }
    }
  }
  else nodebodyforce.Clear();

  // get fluid body force
  bodyforce.Multiply(nodebodyforce,funct_);

  // get viscous term
  if (use2ndderiv_)
  {
    /*--- viscous term: div(epsilon(u)) --------------------------------*/
    /*   /                                                \
         |  2 N_x,xx + N_x,yy + N_y,xy + N_x,zz + N_z,xz  |
       1 |                                                |
       - |  N_y,xx + N_x,yx + 2 N_y,yy + N_z,yz + N_y,zz  |
       2 |                                                |
         |  N_z,xx + N_x,zx + N_y,zy + N_z,yy + 2 N_z,zz  |
         \                                                /

         with N_x .. x-line of N
         N_y .. y-line of N                                             */

    /*--- subtraction for low-Mach-number flow: div((1/3)*(div u)*I) */
    /*   /                            \
         |  N_x,xx + N_y,yx + N_z,zx  |
       1 |                            |
    -  - |  N_x,xy + N_y,yy + N_z,zy  |
       3 |                            |
         |  N_x,xz + N_y,yz + N_z,zz  |
         \                            /

           with N_x .. x-line of N
           N_y .. y-line of N                                             */

    double prefac = 1.0/3.0;
    derxy2_.Scale(prefac);

    for (int i=0; i<nen_; ++i)
    {
      double sum = (derxy2_(0,i)+derxy2_(1,i)+derxy2_(2,i))/prefac;

      visc(0) = ((sum + derxy2_(0,i))*evelnp_(0,i) + derxy2_(3,i)*evelnp_(1,i) + derxy2_(4,i)*evelnp_(2,i))/2.0;
      visc(1) = (derxy2_(3,i)*evelnp_(0,i) + (sum + derxy2_(1,i))*evelnp_(1,i) + derxy2_(5,i)*evelnp_(2,i))/2.0;
      visc(2) = (derxy2_(4,i)*evelnp_(0,i) + derxy2_(5,i)*evelnp_(1,i) + (sum + derxy2_(2,i))*evelnp_(2,i))/2.0;
    }
  }
  else visc.Clear();

  //--------------------------------------------------------------------
  // calculation of subgrid-scale velocity based on momentum residual
  // and stabilization parameter
  // (different for generalized-alpha and other time-integration schemes)
  //--------------------------------------------------------------------
  if (is_genalpha_)
  {
    for (int rr=0;rr<nsd_;++rr)
    {
      sgvelint_(rr) = -tau_[k]*(densam_[k]*acc(rr)+densnp_[k]*conv(rr)+gradp(rr)-2*visc_*visc(rr)-densnp_[k]*bodyforce(rr));
    }
  }
  else
  {
    for (int rr=0;rr<nsd_;++rr)
    {
       sgvelint_(rr) = -tau_[k]*(densnp_[k]*velint_(rr)+timefac*(densnp_[k]*conv(rr)+gradp(rr)-2*visc_*visc(rr)-densnp_[k]*bodyforce(rr))-densn_[k]*acc(rr))/dt;
    }
  }

  return;
} //ScaTraImpl::CalcSubgrVelocity


/*----------------------------------------------------------------------*
 | evaluate shape functions and derivatives at int. point     gjb 08/08 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
double DRT::ELEMENTS::ScaTraImpl<distype>::EvalShapeFuncAndDerivsAtIntPoint(
    const DRT::UTILS::IntPointsAndWeights<nsd_>& intpoints,  ///< integration points
    const int                                    iquad,      ///< id of current Gauss point
    const int                                    eleid       ///< the element id
)
{
  // coordinates of the current integration point
  const double* gpcoord = (intpoints.IP().qxg)[iquad];
  for (int idim=0;idim<nsd_;idim++)
    {xsi_(idim) = gpcoord[idim];}

  if (not SCATRA::IsNurbs(distype))
  {
    // shape functions and their first derivatives
    DRT::UTILS::shape_function<distype>(xsi_,funct_);
    DRT::UTILS::shape_function_deriv1<distype>(xsi_,deriv_);
    if (use2ndderiv_)
    {
      // get the second derivatives of standard element at current GP
      DRT::UTILS::shape_function_deriv2<distype>(xsi_,deriv2_);
    }
  }
  else // nurbs elements are always somewhat special...
  {
    if (nsd_ == 3)
    {
    if (use2ndderiv_)
    {
      DRT::NURBS::UTILS::nurbs_get_3D_funct_deriv_deriv2
      (funct_  ,
          deriv_  ,
          derxy2_ ,
          xsi_    ,
          myknots_,
          weights_,
          distype );
    }
    else
    {
      DRT::NURBS::UTILS::nurbs_get_3D_funct_deriv
      (funct_  ,
          deriv_  ,
          xsi_    ,
          myknots_,
          weights_,
          distype );
    }
    }
    else if (nsd_ == 2)
    {
      if (use2ndderiv_)
      {
        DRT::NURBS::UTILS::nurbs_get_2D_funct_deriv_deriv2
        (funct_  ,
            deriv_  ,
            derxy2_ ,
            xsi_    ,
            myknots_,
            weights_,
            distype );
      }
      else
      {
        DRT::NURBS::UTILS::nurbs_get_2D_funct_deriv
        (funct_  ,
            deriv_  ,
            xsi_    ,
            myknots_,
            weights_,
            distype );
      }
    }
    else dserror("Only 3D and 2D nurbs supported.");
  } // IsNurbs()

  // compute Jacobian matrix and determinant
  // actually compute its transpose....
  /*
    +-            -+ T      +-            -+
    | dx   dx   dx |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dr   dr   dr |
    |              |        |              |
    | dy   dy   dy |        | dx   dy   dz |
    | --   --   -- |   =    | --   --   -- |
    | dr   ds   dt |        | ds   ds   ds |
    |              |        |              |
    | dz   dz   dz |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dt   dt   dt |
    +-            -+        +-            -+
   */

  xjm_.MultiplyNT(deriv_,xyze_);
  const double det = xij_.Invert(xjm_);

  if (det < 1E-16)
    dserror("GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", eleid, det);

  // set integration factor: fac = Gauss weight * det(J)
  const double fac = intpoints.IP().qwgt[iquad]*det;

  // compute global derivatives
  derxy_.Multiply(xij_,deriv_);

  // compute second global derivatives (if needed)
  if (use2ndderiv_)
  {
    // get global second derivatives
    DRT::UTILS::gder2<distype>(xjm_,derxy_,deriv2_,xyze_,derxy2_);
  }
  else
    derxy2_.Clear();

  // return integration factor for current GP: fac = Gauss weight * det(J)
  return fac;

} //ScaTraImpl::CalcSubgrVelocity


/*----------------------------------------------------------------------*
 |  evaluate element matrix and rhs (private)                   vg 02/09|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalMatAndRHS(
    Epetra_SerialDenseMatrix&             emat,
    Epetra_SerialDenseVector&             erhs,
    const double                          fac,
    const bool                            fssgd,
    const double                          timefac,
    const double                          alphaF,
    const int                             dofindex
    )
{
//----------------------------------------------------------------
// 1) element matrix: stationary terms
//----------------------------------------------------------------
// stabilization parameter and integration factors
const double taufac     = tau_[dofindex]*fac;
const double timefacfac = timefac*fac;
const double timetaufac = timefac*taufac;
const double fac_diffus = timefacfac*diffus_[dofindex];

//----------------------------------------------------------------
// standard Galerkin terms
//----------------------------------------------------------------
// convective term in convective form
const double densfac = timefacfac*densnp_[dofindex];
for (int vi=0; vi<nen_; ++vi)
{
  const double v = densfac*funct_(vi);
  const int fvi = vi*numdofpernode_+dofindex;

  for (int ui=0; ui<nen_; ++ui)
  {
    const int fui = ui*numdofpernode_+dofindex;

    emat(fvi,fui) += v*(conv_(ui)+sgconv_(ui));
  }
}

// addition to convective term for conservative form
if (conservative_)
{
  // gradient of current scalar value
  gradphi_.Multiply(derxy_,ephinp_[dofindex]);

  // convective term using current scalar value
  const double cons_conv_phi = velint_.Dot(gradphi_);

  const double consfac = timefacfac*(densnp_[dofindex]*vdiv_+densgradfac_[dofindex]*cons_conv_phi);
  for (int vi=0; vi<nen_; ++vi)
  {
    const double v = consfac*funct_(vi);
    const int fvi = vi*numdofpernode_+dofindex;

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui = ui*numdofpernode_+dofindex;

      emat(fvi,fui) += v*funct_(ui);
    }
  }
}

// diffusive term
for (int vi=0; vi<nen_; ++vi)
{
  const int fvi = vi*numdofpernode_+dofindex;

  for (int ui=0; ui<nen_; ++ui)
  {
    const int fui = ui*numdofpernode_+dofindex;
    double laplawf(0.0);
    GetLaplacianWeakForm(laplawf, derxy_,ui,vi);
    emat(fvi,fui) += fac_diffus*laplawf;
  }
}

//----------------------------------------------------------------
// convective stabilization term
//----------------------------------------------------------------
// convective stabilization of convective term (in convective form)
const double dens2taufac = timetaufac*densnp_[dofindex]*densnp_[dofindex];
for (int vi=0; vi<nen_; ++vi)
{
  const double v = dens2taufac*(conv_(vi)+sgconv_(vi));
  const int fvi = vi*numdofpernode_+dofindex;

  for (int ui=0; ui<nen_; ++ui)
  {
    const int fui = ui*numdofpernode_+dofindex;

    emat(fvi,fui) += v*conv_(ui);
  }
}

//----------------------------------------------------------------
// stabilization terms for higher-order elements
//----------------------------------------------------------------
if (use2ndderiv_)
{
  // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
  GetLaplacianStrongForm(diff_, derxy2_);
  diff_.Scale(diffus_[dofindex]);

  const double denstaufac = timetaufac*densnp_[dofindex];
  // convective stabilization of diffusive term (in convective form)
  for (int vi=0; vi<nen_; ++vi)
  {
    const double v = denstaufac*(conv_(vi)+sgconv_(vi));
    const int fvi = vi*numdofpernode_+dofindex;

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui = ui*numdofpernode_+dofindex;

      emat(fvi,fui) -= v*diff_(ui);
    }
  }

  const double densdifftaufac = diffreastafac_*denstaufac;
  // diffusive stabilization of convective term (in convective form)
  for (int vi=0; vi<nen_; ++vi)
  {
    const double v = densdifftaufac*diff_(vi);
    const int fvi = vi*numdofpernode_+dofindex;

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui = ui*numdofpernode_+dofindex;

      emat(fvi,fui) -= v*conv_(ui);
    }
  }

  const double difftaufac = diffreastafac_*timetaufac;
  // diffusive stabilization of diffusive term
  for (int vi=0; vi<nen_; ++vi)
  {
    const double v = difftaufac*diff_(vi);
    const int fvi = vi*numdofpernode_+dofindex;

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui = ui*numdofpernode_+dofindex;

      emat(fvi,fui) += v*diff_(ui);
    }
  }
}

//----------------------------------------------------------------
// 2) element matrix: instationary terms
//----------------------------------------------------------------
if (not is_stationary_)
{
  const double densamfac = fac*densam_[dofindex];
  //----------------------------------------------------------------
  // standard Galerkin transient term
  //----------------------------------------------------------------
  // transient term
  for (int vi=0; vi<nen_; ++vi)
  {
    const double v = densamfac*funct_(vi);
    const int fvi = vi*numdofpernode_+dofindex;

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui = ui*numdofpernode_+dofindex;

      emat(fvi,fui) += v*funct_(ui);
    }
  }

  const double densamnptaufac = taufac*densam_[dofindex]*densnp_[dofindex];
  //----------------------------------------------------------------
  // stabilization of transient term
  //----------------------------------------------------------------
  // convective stabilization of transient term (in convective form)
  for (int vi=0; vi<nen_; ++vi)
  {
    const double v = densamnptaufac*(conv_(vi)+sgconv_(vi));
    const int fvi = vi*numdofpernode_+dofindex;

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui = ui*numdofpernode_+dofindex;

      emat(fvi,fui) += v*funct_(ui);
    }
  }

  if (use2ndderiv_)
  {
    const double densamreataufac = diffreastafac_*taufac*densam_[dofindex];
    // diffusive stabilization of transient term
    for (int vi=0; vi<nen_; ++vi)
    {
      const double v = densamreataufac*diff_(vi);
      const int fvi = vi*numdofpernode_+dofindex;

      for (int ui=0; ui<nen_; ++ui)
      {
        const int fui = ui*numdofpernode_+dofindex;

        emat(fvi,fui) -= v*funct_(ui);
      }
    }
  }
}

//----------------------------------------------------------------
// 3) element matrix: reactive terms
//----------------------------------------------------------------
if (reaction_)
{
  const double fac_reac        = timefacfac*densnp_[dofindex]*reacoeff_[dofindex];
  const double timetaufac_reac = timetaufac*densnp_[dofindex]*reacoeff_[dofindex];
  //----------------------------------------------------------------
  // standard Galerkin reactive term
  //----------------------------------------------------------------
  for (int vi=0; vi<nen_; ++vi)
  {
    const double v = fac_reac*funct_(vi);
    const int fvi = vi*numdofpernode_+dofindex;

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui = ui*numdofpernode_+dofindex;

      emat(fvi,fui) += v*funct_(ui);
    }
  }

  //----------------------------------------------------------------
  // stabilization of reactive term
  //----------------------------------------------------------------
  double densreataufac = timetaufac_reac*densnp_[dofindex];
  // convective stabilization of reactive term (in convective form)
  for (int vi=0; vi<nen_; ++vi)
  {
    const double v = densreataufac*(conv_(vi)+sgconv_(vi));
    const int fvi = vi*numdofpernode_+dofindex;

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui = ui*numdofpernode_+dofindex;

      emat(fvi,fui) += v*funct_(ui);
    }
  }

  if (use2ndderiv_)
  {
    // diffusive stabilization of reactive term
    for (int vi=0; vi<nen_; ++vi)
    {
      const double v = diffreastafac_*timetaufac_reac*diff_(vi);
      const int fvi = vi*numdofpernode_+dofindex;

      for (int ui=0; ui<nen_; ++ui)
      {
        const int fui = ui*numdofpernode_+dofindex;

        emat(fvi,fui) -= v*funct_(ui);
      }
    }
  }

  //----------------------------------------------------------------
  // reactive stabilization
  //----------------------------------------------------------------
  densreataufac = diffreastafac_*timetaufac_reac*densnp_[dofindex];
  // reactive stabilization of convective (in convective form) and reactive term
  for (int vi=0; vi<nen_; ++vi)
  {
    const double v = densreataufac*funct_(vi);
    const int fvi = vi*numdofpernode_+dofindex;

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui = ui*numdofpernode_+dofindex;

      emat(fvi,fui) += v*(conv_(ui)+reacoeff_[dofindex]*funct_(ui));
    }
  }

  if (use2ndderiv_)
  {
    // reactive stabilization of diffusive term
    for (int vi=0; vi<nen_; ++vi)
    {
      const double v = diffreastafac_*timetaufac_reac*funct_(vi);
      const int fvi = vi*numdofpernode_+dofindex;

      for (int ui=0; ui<nen_; ++ui)
      {
        const int fui = ui*numdofpernode_+dofindex;

        emat(fvi,fui) -= v*diff_(ui);
      }
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
double rhsint    = rhs_[dofindex];
double residual  = 0.0;
double rhsfac    = 0.0;
double rhstaufac = 0.0;
double conv_phi  = 0.0;
double diff_phi  = 0.0;
double rea_phi   = 0.0;
if (is_incremental_ and is_genalpha_)
{
  // gradient of current scalar value
  gradphi_.Multiply(derxy_,ephinp_[dofindex]);

  // convective term using current scalar value
  conv_phi = velint_.Dot(gradphi_);

  // diffusive term using current scalar value for higher-order elements
  if (use2ndderiv_) diff_phi = diff_.Dot(ephinp_[dofindex]);

  // reactive term using current scalar value
  if (reaction_)
  {
    // scalar at integration point
    const double phi = funct_.Dot(ephinp_[dofindex]);

    rea_phi = densnp_[dofindex]*reacoeff_[dofindex]*phi;
  }

  // time derivative stored on history variable
  residual  = densam_[dofindex]*hist_[dofindex] + densnp_[dofindex]*conv_phi - diff_phi + rea_phi - rhsint;
  rhsfac    = timefacfac/alphaF;
  rhstaufac = timetaufac/alphaF;
  rhsint   *= (timefac/alphaF);

  const double vtrans = rhsfac*densam_[dofindex]*hist_[dofindex];
  for (int vi=0; vi<nen_; ++vi)
  {
    const int fvi = vi*numdofpernode_+dofindex;

    erhs[fvi] -= vtrans*funct_(vi);
  }

  // addition to convective term due to subgrid-scale velocity
  // (not included in residual)
  double sgconv_phi = sgvelint_.Dot(gradphi_);
  conv_phi += sgconv_phi;

  // addition to convective term for conservative form
  // (not included in residual)
  if (conservative_)
  {
    // scalar at integration point at time step n
    const double phi = funct_.Dot(ephinp_[dofindex]);

    // convective term in conservative form
    conv_phi += phi*(vdiv_+(densgradfac_[dofindex]/densnp_[dofindex])*conv_phi);
  }

  // multiply convective term by density
  conv_phi *= densnp_[dofindex];
}
else if (not is_incremental_ and is_genalpha_)
{
  // gradient of current scalar value
  gradphi_.Multiply(derxy_,ephin_[dofindex]);

  // convective term using current scalar value
  conv_phi = velint_.Dot(gradphi_);

  // diffusive term using current scalar value for higher-order elements
  if (use2ndderiv_) diff_phi = diff_.Dot(ephin_[dofindex]);

  // reactive term using current scalar value
  if (reaction_)
  {
    // scalar at integration point
    const double phi = funct_.Dot(ephin_[dofindex]);

    rea_phi = densnp_[dofindex]*reacoeff_[dofindex]*phi;
  }

  rhsint   += densam_[dofindex]*hist_[dofindex]*(alphaF/timefac);
  residual  = (1.0-alphaF) * (densn_[dofindex]*conv_phi - diff_phi + rea_phi) - rhsint;
  rhsfac    = timefacfac*(1.0-alphaF)/alphaF;
  rhstaufac = timetaufac/alphaF;
  rhsint   *= (timefac/alphaF);

  // addition to convective term due to subgrid-scale velocity
  // (not included in residual)
  double sgconv_phi = sgvelint_.Dot(gradphi_);
  conv_phi += sgconv_phi;

  // addition to convective term for conservative form
  // (not included in residual)
  if (conservative_)
  {
    // scalar at integration point at time step n
    const double phi = funct_.Dot(ephin_[dofindex]);

    // convective term in conservative form
    // caution: velocity divergence is for n+1 and not for n!
    // -> hopefully, this inconsistency is of small amount
    conv_phi += phi*(vdiv_+(densgradfac_[dofindex]/densn_[dofindex])*conv_phi);
  }

  // multiply convective term by density
  conv_phi *= densn_[dofindex];
}
else if (is_incremental_ and not is_genalpha_)
{
  // gradient of current scalar value
  gradphi_.Multiply(derxy_,ephinp_[dofindex]);

  // convective term using current scalar value
  conv_phi = velint_.Dot(gradphi_);

  // diffusive term using current scalar value for higher-order elements
  if (use2ndderiv_) diff_phi = diff_.Dot(ephinp_[dofindex]);

  // reactive term using current scalar value
  if (reaction_)
  {
    // scalar at integration point
    const double phi = funct_.Dot(ephinp_[dofindex]);

    rea_phi = densnp_[dofindex]*reacoeff_[dofindex]*phi;
  }

  if (not is_stationary_)
  {
    // compute scalar at integration point
    double dens_phi = funct_.Dot(ephinp_[dofindex]);

    rhsint  *= timefac;
    rhsint  += densnp_[dofindex]*hist_[dofindex];
    residual = densnp_[dofindex]*dens_phi + timefac*(densnp_[dofindex]*conv_phi - diff_phi + rea_phi) - rhsint;
    rhsfac   = timefacfac;

    const double vtrans = fac*densnp_[dofindex]*dens_phi;
    for (int vi=0; vi<nen_; ++vi)
    {
      const int fvi = vi*numdofpernode_+dofindex;

      erhs[fvi] -= vtrans*funct_(vi);
    }
  }
  else
  {
    residual = densnp_[dofindex]*conv_phi - diff_phi + rea_phi - rhsint;
    rhsfac   = fac;
  }
  rhstaufac = taufac;

  // addition to convective term due to subgrid-scale velocity
  // (not included in residual)
  double sgconv_phi = sgvelint_.Dot(gradphi_);
  conv_phi += sgconv_phi;

  // addition to convective term for conservative form
  // (not included in residual)
  if (conservative_)
  {
    // scalar at integration point at time step n
    const double phi = funct_.Dot(ephinp_[dofindex]);

    // convective term in conservative form
    conv_phi += phi*(vdiv_+(densgradfac_[dofindex]/densnp_[dofindex])*conv_phi);
  }

  // multiply convective term by density
  conv_phi *= densnp_[dofindex];
}
else
{
  if (not is_stationary_)
  {
    rhsint *= timefac;
    rhsint += densnp_[dofindex]*hist_[dofindex];
  }
  residual  = -rhsint;
  rhstaufac = taufac;
}

//----------------------------------------------------------------
// standard Galerkin bodyforce term
//----------------------------------------------------------------
double vrhs = fac*rhsint;
for (int vi=0; vi<nen_; ++vi)
{
  const int fvi = vi*numdofpernode_+dofindex;

  erhs[fvi] += vrhs*funct_(vi);
}

//----------------------------------------------------------------
// standard Galerkin terms on right hand side
//----------------------------------------------------------------
// convective term
vrhs = rhsfac*conv_phi;
for (int vi=0; vi<nen_; ++vi)
{
  const int fvi = vi*numdofpernode_+dofindex;

  erhs[fvi] -= vrhs*funct_(vi);
}

// diffusive term
vrhs = rhsfac*diffus_[dofindex];
for (int vi=0; vi<nen_; ++vi)
{
  const int fvi = vi*numdofpernode_+dofindex;

  double laplawf(0.0);
  GetLaplacianWeakFormRHS(laplawf,derxy_,gradphi_,vi);
  erhs[fvi] -= vrhs*laplawf;
}

//----------------------------------------------------------------
// stabilization terms
//----------------------------------------------------------------
// convective rhs stabilization (in convective form)
vrhs = rhstaufac*residual*densnp_[dofindex];
for (int vi=0; vi<nen_; ++vi)
{
  const int fvi = vi*numdofpernode_+dofindex;

  erhs[fvi] -= vrhs*(conv_(vi)+sgconv_(vi));
}

// diffusive rhs stabilization
if (use2ndderiv_)
{
  vrhs = rhstaufac*residual;
  // diffusive stabilization of convective temporal rhs term (in convective form)
  for (int vi=0; vi<nen_; ++vi)
  {
    const int fvi = vi*numdofpernode_+dofindex;

    erhs[fvi] += diffreastafac_*vrhs*diff_(vi);
  }
}

//----------------------------------------------------------------
// reactive terms (standard Galerkin and stabilization) on rhs
//----------------------------------------------------------------
// standard Galerkin term
if (reaction_)
{
  vrhs = rhsfac*rea_phi;
  for (int vi=0; vi<nen_; ++vi)
  {
    const int fvi = vi*numdofpernode_+dofindex;

    erhs[fvi] -= vrhs*funct_(vi);
  }

  // reactive rhs stabilization
  vrhs = diffreastafac_*rhstaufac*densnp_[dofindex]*reacoeff_[dofindex]*residual;
  for (int vi=0; vi<nen_; ++vi)
  {
    const int fvi = vi*numdofpernode_+dofindex;

    erhs[fvi] -= vrhs*funct_(vi);
  }
}

//----------------------------------------------------------------
// fine-scale subgrid-diffusivity term on right hand side
//----------------------------------------------------------------
if (is_incremental_ and fssgd)
{
  vrhs = rhsfac*sgdiff_[dofindex];
  for (int vi=0; vi<nen_; ++vi)
  {
    const int fvi = vi*numdofpernode_+dofindex;

    double laplawf(0.0);
    GetLaplacianWeakFormRHS(laplawf,derxy_,fsgradphi_,vi);
    erhs[fvi] -= vrhs*laplawf;
  }
}

return;
} //ScaTraImpl::CalMatAndRHS


/*----------------------------------------------------------------------*
 | calculate mass matrix + rhs for determ. initial time deriv. gjb 08/08|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::InitialTimeDerivative(
    DRT::Element*                         ele,
    Epetra_SerialDenseMatrix&             emat,
    Epetra_SerialDenseVector&             erhs,
    const bool                            reinitswitch,
    const double                          frt
)
{
  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // dead load in element nodes at initial point in time
  const double time = 0.0;

//REINHARD
  if (reinitswitch == false)
    BodyForce(ele,time);
  else
    BodyForceReinit(ele,time);
//end REINHARD

  //----------------------------------------------------------------------
  // get material parameters (evaluation at element center)
  //----------------------------------------------------------------------
  if (not mat_gp_)
  {
    // use one-point Gauss rule to do calculations at the element center
    DRT::UTILS::IntPointsAndWeights<nsd_> intpoints_tau(SCATRA::DisTypeToStabGaussRule<distype>::rule);

    // evaluate shape functions and derivatives at element center
    EvalShapeFuncAndDerivsAtIntPoint(intpoints_tau,0,ele->Id()); //fac unused

    GetMaterialParams(ele);
  }

  // integrations points and weights
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  /*----------------------------------------------------------------------*/
  // element integration loop
  /*----------------------------------------------------------------------*/
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad,ele->Id());

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    if (mat_gp_) GetMaterialParams(ele);

    //------------ get values of variables at integration point
    for (int k=0;k<numscal_;++k) // deal with a system of transported scalars
    {
      // get bodyforce in gausspoint (divided by shcacp)
      // (For temperature equation, time derivative of thermodynamic pressure
      //  is added, if not constant.)
      rhs_[k] = bodyforce_[k].Dot(funct_) / shcacp_;
      rhs_[k] += thermpressdt_/shcacp_;

      // get gradient of el. potential at integration point
      gradpot_.Multiply(derxy_,epotnp_);

      // migration part
      migconv_.MultiplyTN(-frt,derxy_,gradpot_);

      // get velocity at element center
      velint_.Multiply(evelnp_,funct_);

      // convective part in convective form: u_x*N,x+ u_y*N,y
      conv_.MultiplyTN(derxy_,velint_);

      // velocity divergence required for conservative form
      if (conservative_) GetVelocityDivergence(vdiv_,evelnp_,derxy_);

      // diffusive integration factor
      const double fac_diffus = fac*diffus_[k];

      // get value of current scalar
      conint_[k] = funct_.Dot(ephinp_[k]);

      // gradient of current scalar value
      gradphi_.Multiply(derxy_,ephinp_[k]);

      // convective part in convective form times initial scalar field
      double conv_ephi0_k = conv_.Dot(ephinp_[k]);

      // addition to convective term for conservative form
      // -> spatial variation of density not yet accounted for
      if (conservative_)
        conv_ephi0_k += conint_[k]*(vdiv_+(densgradfac_[k]/densnp_[k])*conv_ephi0_k);

      //----------------------------------------------------------------
      // element matrix: transient term
      //----------------------------------------------------------------
      // transient term
      for (int vi=0; vi<nen_; ++vi)
      {
        const double v = fac*funct_(vi)*densnp_[k];
        const int fvi = vi*numdofpernode_+k;

        for (int ui=0; ui<nen_; ++ui)
        {
          const int fui = ui*numdofpernode_+k;

          emat(fvi,fui) += v*funct_(ui);
        }
      }

      //----------------------------------------------------------------
      // element right hand side: convective term in convective form
      //----------------------------------------------------------------
      double vrhs = fac*densnp_[k]*conv_ephi0_k;
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        erhs[fvi] -= vrhs*funct_(vi);
      }

      //----------------------------------------------------------------
      // element right hand side: diffusive term
      //----------------------------------------------------------------
      vrhs = fac_diffus;
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        double laplawf(0.0);
        GetLaplacianWeakFormRHS(laplawf,derxy_,gradphi_,vi);
        erhs[fvi] -= vrhs*laplawf;
      }

      //----------------------------------------------------------------
      // element right hand side: nonlinear migration term
      //----------------------------------------------------------------
      vrhs = fac_diffus*conint_[k]*valence_[k];
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        erhs[fvi] += vrhs*migconv_(vi);
      }

      //----------------------------------------------------------------
      // element right hand side: reactive term
      //----------------------------------------------------------------
      if (reaction_)
      {
        vrhs = fac*densnp_[k]*reacoeff_[k]*conint_[k];
        for (int vi=0; vi<nen_; ++vi)
        {
          const int fvi = vi*numdofpernode_+k;

          erhs[fvi] -= vrhs*funct_(vi);
        }
      }

      //----------------------------------------------------------------
      // element right hand side: bodyforce term
      //----------------------------------------------------------------
      vrhs = fac*rhs_[k];
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        erhs[fvi] += vrhs*funct_(vi);
      }
    } // loop over each scalar k

    if (iselch_) // ELCH
    {
      // we put a dummy mass matrix here in order to have a regular
      // matrix in the lower right block of the whole system-matrix
      // A identity matrix would cause problems with ML solver in the SIMPLE
      // schemes since ML needs to have off-diagonal entries for the aggregation!
      for (int vi=0; vi<nen_; ++vi)
      {
        const double v = fac*funct_(vi); // density assumed to be 1.0 here
        const int fvi = vi*numdofpernode_+numscal_;

        for (int ui=0; ui<nen_; ++ui)
        {
          const int fui = ui*numdofpernode_+numscal_;

          emat(fvi,fui) += v*funct_(ui);
        }
      }
      // dof for el. potential have no 'velocity' -> rhs is zero!
    }

  } // integration loop

  return;
} // ScaTraImpl::InitialTimeDerivative

/*----------------------------------------------------------------------------*
 | calculate mass matrix + rhs for determ. time deriv. reinit. rasthofer 02/10|
 *----------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::TimeDerivativeReinit(
    DRT::Element*                         ele,
    Epetra_SerialDenseMatrix&             emat,
    Epetra_SerialDenseVector&             erhs,
    const enum INPAR::SCATRA::TauType     whichtau,
    const double                          dt,
    const double                          timefac
)
{
  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------
  // use one-point Gauss rule to do calculations at the element center
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints_tau(SCATRA::DisTypeToStabGaussRule<distype>::rule);

  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double vol = EvalShapeFuncAndDerivsAtIntPoint(intpoints_tau,0,ele->Id());

  //------------------------------------------------------------------------------------
  // get material parameters and stabilization parameters (evaluation at element center)
  //------------------------------------------------------------------------------------
  if (not mat_gp_ or not tau_gp_)
  {

    GetMaterialParams(ele);

    if (not tau_gp_)
      {
        // get velocity at element center
        velint_.Multiply(evelnp_,funct_);

        for (int k = 0;k<numscal_;++k) // loop of each transported scalar
        {
          // generating copy of diffusivity for use in CalTau routine
          double diffus = diffus_[k];

          // calculation of stabilization parameter at element center
          CalTau(ele,diffus,dt,timefac,whichtau,vol,k);
        }
      }
  }

  // integrations points and weights
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  /*----------------------------------------------------------------------*/
  // element integration loop
  /*----------------------------------------------------------------------*/
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad,ele->Id());

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    if (mat_gp_) GetMaterialParams(ele);

    //------------ get values of variables at integration point
    for (int k=0;k<numscal_;++k) // deal with a system of transported scalars
    {
      // get bodyforce in gausspoint (divided by shcacp)
      // (For temperature equation, time derivative of thermodynamic pressure
      //  is added, if not constant.)
      rhs_[k] = bodyforce_[k].Dot(funct_) / shcacp_;

      // get gradient of el. potential at integration point
      gradpot_.Multiply(derxy_,epotnp_);

      // get velocity at element center
      velint_.Multiply(evelnp_,funct_);

      // convective part in convective form: u_x*N,x+ u_y*N,y
      conv_.MultiplyTN(derxy_,velint_);

      // velocity divergence required for conservative form
      if (conservative_) GetVelocityDivergence(vdiv_,evelnp_,derxy_);

      if (tau_gp_)
      {
        // generating copy of diffusivity for use in CalTau routine
        double diffus = diffus_[k];

        // calculation of stabilization parameter at integration point
        CalTau(ele,diffus,dt,timefac,whichtau,vol,k);
      }

      const double fac_tau = fac*tau_[k];

      // diffusive integration factor
      const double fac_diffus = fac*diffus_[k];

      // get value of current scalar
      conint_[k] = funct_.Dot(ephinp_[k]);

      // gradient of current scalar value
      gradphi_.Multiply(derxy_,ephinp_[k]);

      // convective part in convective form times initial scalar field
      double conv_ephi0_k = conv_.Dot(ephinp_[k]);

      // addition to convective term for conservative form
      // -> spatial variation of density not yet accounted for
      if (conservative_)
        conv_ephi0_k += conint_[k]*(vdiv_+(densgradfac_[k]/densnp_[k])*conv_ephi0_k);

      //----------------------------------------------------------------
      // element matrix: transient term
      //----------------------------------------------------------------
      // transient term
      for (int vi=0; vi<nen_; ++vi)
      {
        const double v = fac*funct_(vi)*densnp_[k];
        const int fvi = vi*numdofpernode_+k;

        for (int ui=0; ui<nen_; ++ui)
        {
          const int fui = ui*numdofpernode_+k;

          emat(fvi,fui) += v*funct_(ui);
        }
      }

      //----------------------------------------------------------------
      // element matrix: stabilization of transient term
      //----------------------------------------------------------------
      // convective stabilization of transient term (in convective form)
      for (int vi=0; vi<nen_; ++vi)
      {
        const double v = fac_tau*conv_(vi)*densnp_[k];//v = densamnptaufac*(conv_(vi)+sgconv_(vi));
        const int fvi = vi*numdofpernode_+k;

        for (int ui=0; ui<nen_; ++ui)
        {
          const int fui = ui*numdofpernode_+k;

          emat(fvi,fui) += v*funct_(ui);
        }
      }

      //----------------------------------------------------------------
      // element right hand side: convective term in convective form
      //----------------------------------------------------------------
      double vrhs = fac*densnp_[k]*conv_ephi0_k;
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        erhs[fvi] -= vrhs*funct_(vi);
      }

      //----------------------------------------------------------------
      // element right hand side: convective stabilization term
      //----------------------------------------------------------------
      // convective stabilization of convective term (in convective form)
      vrhs = fac_tau*densnp_[k]*conv_ephi0_k*densnp_[k];
      for (int vi=0; vi<nen_; ++vi)
      {
         const int fvi = vi*numdofpernode_+k;

         erhs[fvi] -= vrhs*conv_(vi);
      }

      if (use2ndderiv_)
      {
         dserror("TimeDerivativePhidt not yet implemented for higher order elements");
      }

      //----------------------------------------------------------------
      // element right hand side: diffusive term
      //----------------------------------------------------------------
      vrhs = fac_diffus;
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        double laplawf(0.0);
        GetLaplacianWeakFormRHS(laplawf,derxy_,gradphi_,vi);
        erhs[fvi] -= vrhs*laplawf;
      }

      //----------------------------------------------------------------
      // element right hand side: reactive term
      //----------------------------------------------------------------
      if (reaction_)
      {
        vrhs = fac*densnp_[k]*reacoeff_[k]*conint_[k];
        for (int vi=0; vi<nen_; ++vi)
        {
          const int fvi = vi*numdofpernode_+k;

          erhs[fvi] -= vrhs*funct_(vi);
        }
      }

      //----------------------------------------------------------------
      // element right hand side: bodyforce term
      //----------------------------------------------------------------
      vrhs = fac*rhs_[k];
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        erhs[fvi] += vrhs*funct_(vi);
      }
    } // loop over each scalar k

  } // integration loop

  return;
} // ScaTraImpl::TimeDerivativeReinit

/*----------------------------------------------------------------------*
 | calculate normalized subgrid-diffusivity matrix              vg 10/08|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalcSubgridDiffMatrix(
    const DRT::Element*           ele,
    Epetra_SerialDenseMatrix&     sys_mat_sd,
    const double                  timefac
    )
{
  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

/*----------------------------------------------------------------------*/
// integration loop for one element
/*----------------------------------------------------------------------*/
// integrations points and weights
DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

// integration loop
for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
{
  const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad,ele->Id());

  for (int k=0;k<numscal_;++k)
  {
    // parameter for artificial diffusivity (scaled to one here)
    double kartfac = fac;
    if (not is_stationary_) kartfac *= timefac;

    for (int vi=0; vi<nen_; ++vi)
    {
      const int fvi = vi*numdofpernode_+k;

      for (int ui=0; ui<nen_; ++ui)
      {
        const int fui = ui*numdofpernode_+k;
        double laplawf(0.0);
        GetLaplacianWeakForm(laplawf, derxy_,ui,vi);
        sys_mat_sd(fvi,fui) += kartfac*laplawf;

        /*subtract SUPG term */
        //sys_mat_sd(fvi,fui) -= taufac*conv(vi)*conv(ui);
      }
    }
  }
} // integration loop

return;
} // ScaTraImpl::CalcSubgridDiffMatrix


/*----------------------------------------------------------------------*
 | calculate matrix and rhs for electrochemistry problem      gjb 10/08 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalMatElch(
    Epetra_SerialDenseMatrix&             emat,
    Epetra_SerialDenseVector&             erhs,
    const double                          frt,
    const double                          timefac,
    const double                          fac
)
{
  // get values of all transported scalars at integration point
  for (int k=0; k<numscal_; ++k)
  {
    conint_[k] = funct_.Dot(ephinp_[k]);

    // when concentration becomes zero, the coupling terms in the system matrix get lost!
    //if (conint_[k] < 1e-18)
    //  printf("WARNING: species concentration %d at GP is zero or negative: %g\n",k,conint_[k]);
  }

  // get gradient of el. potential at integration point
  gradpot_.Multiply(derxy_,epotnp_);

  // migration term (convective part)
  migconv_.MultiplyTN(-frt,derxy_,gradpot_);

  // Laplacian of shape functions at integration point
  if (use2ndderiv_)
  {
    GetLaplacianStrongForm(laplace_, derxy2_);
  }

#if 0
  // DEBUG output
  cout<<endl<<"values at GP:"<<endl;
  cout<<"factor F/RT = "<<frt<<endl;
  for (int k=0;k<numscal_;++k)
  {cout<<"conint_["<<k<<"] = "<<conint_[k]<<endl;}
  for (int k=0;k<3;++k)
  {cout<<"gradpot_["<<k<<"] = "<<gradpot_[k]<<endl;}
#endif
  // some 'working doubles'
  double diffus_valence_k;
  double rhsint;  // rhs at int. point
  // integration factors and coefficients of single terms
  double timefacfac;
  double timetaufac;
  double taufac;

  for (int k = 0; k < numscal_;++k) // loop over all transported scalars
  {
    // stabilization parameters
    taufac = tau_[k]*fac;

    if (is_stationary_)
    {
      timefacfac  = fac;
      timetaufac  = taufac;
      rhsint = rhs_[k];     // set rhs at integration point
    }
    else
    {
      timefacfac  = timefac * fac;
      timetaufac  = timefac * taufac;
      rhsint = hist_[k] + rhs_[k]*timefac;     // set rhs at integration point
    }

    // compute gradient of scalar k at integration point
    gradphi_.Multiply(derxy_,ephinp_[k]);

    // factor D_k * z_k
    diffus_valence_k = diffusvalence_[k];

    if (use2ndderiv_)
    {
      // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
      diff_.Update(diffus_[k],laplace_);

      // get Laplacian of el. potential at integration point
      double lappot = laplace_.Dot(epotnp_);
      // reactive part of migration term
      migrea_.Update(-frt*diffus_valence_k*lappot,funct_);
    }

    const double frt_timefacfac_diffus_valence_k_conint_k = frt*timefacfac*diffus_valence_k*conint_[k];

    // ----------------------------------------matrix entries
    for (int vi=0; vi<nen_; ++vi)
    {
      const int    fvi = vi*numdofpernode_+k;
      double timetaufac_conv_eff_vi = timetaufac*conv_(vi);
#ifdef MIGRATIONSTAB
      timetaufac_conv_eff_vi += timetaufac*diffus_valence_k*migconv_(vi);
#endif
      const double timefacfac_funct_vi = timefacfac*funct_(vi);
      const double timefacfac_diffus_valence_k_mig_vi = timefacfac*diffus_valence_k*migconv_(vi);
      const double valence_k_fac_funct_vi = valence_[k]*fac*funct_(vi);

      for (int ui=0; ui<nen_; ++ui)
      {
        const int fui = ui*numdofpernode_+k;

        /* Standard Galerkin terms: */
        /* convective term */
        emat(fvi, fui) += timefacfac_funct_vi*conv_(ui) ;

        // addition to convective term for conservative form
        if (conservative_)
        {
          // convective term using current scalar value
          emat(fvi,fui) += timefacfac_funct_vi*vdiv_*funct_(ui);
        }

        /* diffusive term */
        double laplawf(0.0);
        GetLaplacianWeakForm(laplawf, derxy_,ui,vi);
        emat(fvi, fui) += timefacfac*diffus_[k]*laplawf;

        /* migration term (directional derivatives) */
        emat(fvi, fui) -= timefacfac_diffus_valence_k_mig_vi*funct_(ui);
        emat(fvi,ui*numdofpernode_+numscal_) += frt_timefacfac_diffus_valence_k_conint_k*laplawf;

        /* electroneutrality condition */
        emat(vi*numdofpernode_+numscal_, fui) += valence_k_fac_funct_vi*funct_(ui);

        /* Stabilization term: */
        /* 0) transient stabilization */
        // not implemented

        /* 1) convective stabilization */

        /* convective term */
        // partial derivative w.r.t concentration
        emat(fvi, fui) += timetaufac_conv_eff_vi*conv_(ui);
        emat(fvi, fui) += timetaufac_conv_eff_vi*diffus_valence_k*migconv_(ui);
        // partial derivative w.r.t potential
#if 1
        double val_ui; GetLaplacianWeakFormRHS(val_ui, derxy_,gradphi_,ui);
        emat(fvi,ui*numdofpernode_+numscal_) -= timetaufac_conv_eff_vi*diffus_valence_k*val_ui;
#else
        // linearization w.r.t potential phi
        double val_ui; GetLaplacianWeakFormRHS(val_ui, derxy_,gradphi_,ui);
        double val_vi; GetLaplacianWeakFormRHS(val_vi, derxy_,gradphi_,vi);
        double norm = gradphi_.Norm2();
        emat(fvi, ui*numdofpernode_+ numscal_) += timetaufac*(laplawf*norm*norm + val_vi*val_ui);
#endif
      } // for ui
    } // for vi

    if (use2ndderiv_)
    {
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;
        const double timetaufac_conv_eff_vi = timetaufac*(conv_(vi)+diffus_valence_k*migconv_(vi));
        for (int ui=0; ui<nen_; ++ui)
        {
          const int fui = ui*numdofpernode_+k;

          /* 1) convective stabilization */

          // diffusive term
          emat(fvi, fui) += -timetaufac_conv_eff_vi*diff_(ui) ;

          // migration term (reactive part)
          emat(fvi, fui) += -timetaufac_conv_eff_vi*migrea_(ui) ;

          /* 2) diffusive stabilization */

          // convective term
          emat(fvi, fui) -= diffreastafac_*timetaufac*diff_(vi)*(conv_(ui)+diffus_valence_k*migconv_(ui));

          // diffusive term
          emat(fvi, fui) += diffreastafac_*timetaufac*diff_(vi)*diff_(ui) ;

          // migration term (reactive part)
          emat(fvi, fui) -= diffreastafac_*timetaufac*diff_(vi)*migrea_(ui) ;

          /* 3) reactive stabilization (reactive part of migration term) */

          // convective terms
          //emat(fvi, ui*numdofpernode_+k) -= diffreastafac_*timetaufac*migrea_(vi)*(conv_(ui)+diffus_valence_k*migconv_(ui));

          // diffusive term
          //emat(fvi, ui*numdofpernode_+k) += diffreastafac_*timetaufac*migrea_(vi)*diff_(ui) ;

          // migration term (reactive part)
          //emat(fvi, ui*numdofpernode_+k) -= diffreastafac_*timetaufac*migrea_(vi)*migrea_(ui) ;

        } // for ui
      } // for vi

    } // use2ndderiv

    // ----------------------------------------------RHS
    const double conv_ephinp_k = conv_.Dot(ephinp_[k]);
    const double Dkzk_mig_ephinp_k = diffus_valence_k*(migconv_.Dot(ephinp_[k]));
    const double conv_eff_k = conv_ephinp_k + Dkzk_mig_ephinp_k;
    const double funct_ephinp_k = funct_.Dot(ephinp_[k]);
    double diff_ephinp_k(0.0);
    double migrea_k(0.0);
    if (use2ndderiv_)
    { // only necessary for higher order elements
      diff_ephinp_k = diff_.Dot(ephinp_[k]);   // diffusion
      migrea_k      = migrea_.Dot(ephinp_[k]); // reactive part of migration term
    }

    // compute residual of strong form for residual-based stabilization
    double taufacresidual = taufac*rhsint - timetaufac*(conv_eff_k - diff_ephinp_k + migrea_k);
    if (!is_stationary_) // add transient term to the residual
      taufacresidual -= taufac*funct_ephinp_k;

#ifdef PRINT_ELCH_DEBUG
    cout<<"tau["<<k<<"]    = "<<tau_[k]<<endl;
    cout<<"taufac["<<k<<"] = "<<taufac<<endl;
    if (tau_[k] != 0.0)
      cout<<"residual["<<k<<"] = "<< taufacresidual/taufac<<endl;
    cout<<"conv_eff_k    = "<<conv_eff_k<<endl;
    cout<<"conv_ephinp_k  = "<<conv_ephinp_k<<endl;
    cout<<"Dkzk_mig_ephinp_k = "<<Dkzk_mig_ephinp_k<<endl;
    cout<<"diff_ephinp_k = "<<diff_ephinp_k<<endl;
    cout<<"migrea_k      = "<<migrea_k <<endl;
    cout<<endl;
#endif

    //------------residual formulation (Newton iteration)
    for (int vi=0; vi<nen_; ++vi)
    {
      const int fvi = vi*numdofpernode_+k;

      // RHS source term
      erhs[fvi] += fac*funct_(vi)*rhsint ;

      // nonlinear migration term
      erhs[fvi] += conint_[k]*timefacfac*diffus_valence_k*migconv_(vi);

      // convective term
      erhs[fvi] -= timefacfac*funct_(vi)*conv_ephinp_k;

      // addition to convective term for conservative form
      // (not included in residual)
      if (conservative_)
      {
        // convective term in conservative form
        erhs[fvi] -= timefacfac*funct_(vi)*funct_ephinp_k*vdiv_;
      }

      // diffusive term
      double laplawf(0.0);
      GetLaplacianWeakFormRHS(laplawf,derxy_,gradphi_,vi);
      erhs[fvi] -= timefacfac*diffus_[k]*laplawf;

      // electroneutrality condition
      // for incremental formulation, there is the residuum on the rhs! : 0-ENC*phi_i
      erhs[vi*numdofpernode_+numscal_] -= valence_[k]*fac*funct_(vi)*funct_ephinp_k;

      // Stabilization terms:

      // 0) transient stabilization
      // not implemented

      // 1) convective stabilization
      erhs[fvi] += conv_(vi)* taufacresidual;
#ifdef MIGRATIONSTAB
      erhs[fvi] +=  diffus_valence_k*migconv_(vi) * taufacresidual;
#endif

    } // for vi

    if (use2ndderiv_)
    {
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;

        // 2) diffusive stabilization
        erhs[fvi] -= diffreastafac_*diff_(vi)*taufacresidual ;

        /* 3) reactive stabilization (reactive part of migration term) */

      } // for vi
    } // use2ndderiv

    // -----------------------------------INSTATIONARY TERMS
    if (!is_stationary_)
    {
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+k;
        const double fac_funct_vi = fac*funct_(vi);
        for (int ui=0; ui<nen_; ++ui)
        {
         const int fui = ui*numdofpernode_+k;

          /* Standard Galerkin terms: */
          /* transient term */
          emat(fvi, fui) += fac_funct_vi*funct_(ui) ;

          /* 1) convective stabilization */
          /* transient term */
          emat(fvi, fui) += taufac*conv_(vi)*funct_(ui);
#ifdef MIGRATIONSTAB
          emat(fvi, fui) += taufac*diffus_valence_k*migconv_(vi)*funct_(ui);
#endif

          if (use2ndderiv_)
          {
            /* 2) diffusive stabilization */
            /* transient term */
            emat(fvi, fui) -= diffreastafac_*taufac*diff_(vi)*funct_(ui);
          }
        } // for ui

        // residuum on RHS:

        /* Standard Galerkin terms: */
        /* transient term */
        erhs[fvi] -= fac_funct_vi*funct_ephinp_k;

      } // for vi
    } // instationary case

  } // loop over scalars

  return;
} // ScaTraImpl::CalMatElch


/*---------------------------------------------------------------------*
 |  calculate error compared to analytical solution           gjb 10/08|
 *---------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalErrorComparedToAnalytSolution(
    const DRT::Element*                   ele,
    ParameterList&                        params,
    Epetra_SerialDenseVector&             errors
)
{
  //at the moment, there is only one analytical test problem available!
  if (params.get<string>("action") != "calc_elch_kwok_error")
    dserror("Unknown analytical solution");

  //------------------------------------------------- Kwok et Wu,1995
  //   Reference:
  //   Kwok, Yue-Kuen and Wu, Charles C. K.
  //   "Fractional step algorithm for solving a multi-dimensional diffusion-migration equation"
  //   Numerical Methods for Partial Differential Equations
  //   1995, Vol 11, 389-397

  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) dserror("No ALE for Kwok & Wu error calculation allowed.");

  // set constants for analytical solution
  const double t = params.get<double>("total time");
  const double frt = params.get<double>("frt");

  // get material constants
  GetMaterialParams(ele);

  // working arrays
  double                  potint;
  LINALG::Matrix<2,1>     conint;
  LINALG::Matrix<nsd_,1>  xint;
  LINALG::Matrix<2,1>     c;
  double                  deltapot;
  LINALG::Matrix<2,1>     deltacon(true);

  // integrations points and weights
  // more GP than usual due to cos/exp fcts in analytical solution
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToGaussRuleForExactSol<distype>::rule);

  // start loop over integration points
  for (int iquad=0;iquad<intpoints.IP().nquad;iquad++)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad,ele->Id());

    // get values of all transported scalars at integration point
    for (int k=0; k<2; ++k)
    {
      conint(k) = funct_.Dot(ephinp_[k]);
    }

    // get el. potential solution at integration point
    potint = funct_.Dot(epotnp_);

    // get global coordinate of integration point
    xint.Multiply(xyze_,funct_);

    // compute various constants
    const double d = frt*((diffus_[0]*valence_[0]) - (diffus_[1]*valence_[1]));
    if (abs(d) == 0.0) dserror("division by zero");
    const double D = frt*((valence_[0]*diffus_[0]*diffus_[1]) - (valence_[1]*diffus_[1]*diffus_[0]))/d;

    // compute analytical solution for cation and anion concentrations
    const double A0 = 2.0;
    const double m = 1.0;
    const double n = 2.0;
    const double k = 3.0;
    const double A_mnk = 1.0;
    double expterm;
    double c_0_0_0_t;

    if (nsd_==3)
    {
      expterm = exp((-D)*(m*m + n*n + k*k)*t*PI*PI);
      c(0) = A0 + (A_mnk*(cos(m*PI*xint(0))*cos(n*PI*xint(1))*cos(k*PI*xint(2)))*expterm);
      c_0_0_0_t = A0 + (A_mnk*exp((-D)*(m*m + n*n + k*k)*t*PI*PI));
    }
    else if (nsd_==2)
    {
      expterm = exp((-D)*(m*m + n*n)*t*PI*PI);
      c(0) = A0 + (A_mnk*(cos(m*PI*xint(0))*cos(n*PI*xint(1)))*expterm);
      c_0_0_0_t = A0 + (A_mnk*exp((-D)*(m*m + n*n)*t*PI*PI));
    }
    else if (nsd_==1)
    {
      expterm = exp((-D)*(m*m)*t*PI*PI);
      c(0) = A0 + (A_mnk*(cos(m*PI*xint(0)))*expterm);
      c_0_0_0_t = A0 + (A_mnk*exp((-D)*(m*m)*t*PI*PI));
    }
    else
      dserror("Illegal number of space dimensions for analyt. solution: %d",nsd_);

    // compute analytical solution for anion concentration
    c(1) = (-valence_[0]/valence_[1])* c(0);
    // compute analytical solution for el. potential
    const double pot = ((diffus_[1]-diffus_[0])/d) * log(c(0)/c_0_0_0_t);

    // compute differences between analytical solution and numerical solution
    deltapot = potint - pot;
    deltacon.Update(1.0,conint,-1.0,c);

    // add square to L2 error
    errors[0] += deltacon(0)*deltacon(0)*fac; // cation concentration
    errors[1] += deltacon(1)*deltacon(1)*fac; // anion concentration
    errors[2] += deltapot*deltapot*fac; // electric potential in electrolyte solution

  } // end of loop over integration points

  return;
} // ScaTraImpl::CalErrorComparedToAnalytSolution


/*----------------------------------------------------------------------*
 |  calculate mass flux (no reactive flux so far)    (private) gjb 06/08|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalculateFlux(
    LINALG::Matrix<3,nen_>&          flux,
    const DRT::Element*             ele,
    const vector<double>&           ephinp,
    const double                    frt,
    const Epetra_SerialDenseVector& evel,
    const INPAR::SCATRA::FluxType   fluxtype,
    const int                       dofindex
)
{
  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // set material variables
  double diffus(0.0);
  double dens(0.0);
  double valence(0.0);
  double diffus_valence_frt(0.0);

  //GetMaterialParams(ele);

  // get the material
  RefCountPtr<MAT::Material> material = ele->Material();

  // use one-point Gauss rule to do calculations at element center
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints_tau(SCATRA::DisTypeToStabGaussRule<distype>::rule);

  // evaluate shape functions and derivatives at element center
  EvalShapeFuncAndDerivsAtIntPoint(intpoints_tau,0,ele->Id());

  // get diffusivity and density
  if (material->MaterialType() == INPAR::MAT::m_matlist)
  {
    const MAT::MatList* actmat = static_cast<const MAT::MatList*>(material.get());

    const int matid = actmat->MatID(dofindex);
    Teuchos::RCP<const MAT::Material> singlemat = actmat->MaterialById(matid);

    // set density to 1.0
    dens = 1.0;

    if (singlemat->MaterialType() == INPAR::MAT::m_scatra)
    {
      const MAT::ScatraMat* actsinglemat = static_cast<const MAT::ScatraMat*>(singlemat.get());
      diffus = actsinglemat->Diffusivity();
    }
    else if (singlemat->MaterialType() == INPAR::MAT::m_ion)
    {
      const MAT::Ion* actsinglemat = static_cast<const MAT::Ion*>(singlemat.get());
      diffus = actsinglemat->Diffusivity();
      valence = actsinglemat->Valence();
      diffus_valence_frt = diffus*valence*frt;
    }
    else if (singlemat->MaterialType() == INPAR::MAT::m_arrhenius_spec)
    {
      const MAT::ArrheniusSpec* actsinglemat = static_cast<const MAT::ArrheniusSpec*>(singlemat.get());

      // compute temperature
      double temp = 0.0;
      for (int i=0; i<nen_; ++i)
      {
        temp += funct_(i)*ephinp[i];
      }

      // compute diffusivity according to Sutherland law
      diffus = actsinglemat->ComputeDiffusivity(temp);
    }
    else if (singlemat->MaterialType() == INPAR::MAT::m_arrhenius_temp)
    {
      const MAT::ArrheniusTemp* actsinglemat = static_cast<const MAT::ArrheniusTemp*>(singlemat.get());

      // get specific heat capacity at constant pressure
      shcacp_ = actsinglemat->Shc();

      // compute temperature
      double temp = 0.0;
      for (int i=0; i<nen_; ++i)
      {
        temp += funct_(i)*ephinp[i];
      }

      // compute thermal conductivity according to Sutherland law
      diffus = shcacp_*actsinglemat->ComputeDiffusivity(temp);

      // compute density based on temperature and thermodynamic pressure
      dens = actsinglemat->ComputeDensity(temp,thermpressnp_);
    }
    else dserror("type of material found in material list not supported");
  }
  else if (material->MaterialType() == INPAR::MAT::m_scatra)
  {
    const MAT::ScatraMat* actmat = static_cast<const MAT::ScatraMat*>(material.get());

    // get constant diffusivity
    diffus = actmat->Diffusivity();

    // set density to 1.0
    dens = 1.0;
  }
  else if (material->MaterialType() == INPAR::MAT::m_ion)
  {
    const MAT::Ion* actmat = static_cast<const MAT::Ion*>(material.get());

    // get constant diffusivity (we only do convection-diffusion in this case!)
    diffus = actmat->Diffusivity();

    // set density to 1.0
    dens = 1.0;
  }
  else if (material->MaterialType() == INPAR::MAT::m_mixfrac)
  {
    const MAT::MixFrac* actmat = static_cast<const MAT::MixFrac*>(material.get());

    // compute mixture fraction
    double mixfrac = 0.0;
    for (int i=0; i<nen_; ++i)
    {
      mixfrac += funct_(i)*ephinp[i];
    }

    // compute dynamic diffusivity based on mixture fraction
    diffus = actmat->ComputeDiffusivity(mixfrac);

    // compute density based on mixture fraction
    dens = actmat->ComputeDensity(mixfrac);
  }
  else if (material->MaterialType() == INPAR::MAT::m_sutherland)
  {
    const MAT::Sutherland* actmat = static_cast<const MAT::Sutherland*>(material.get());

    // get specific heat capacity at constant pressure
    shcacp_ = actmat->Shc();

    // compute temperature
    double temp = 0.0;
    for (int i=0; i<nen_; ++i)
    {
      temp += funct_(i)*ephinp[i];
    }

    // compute thermal conductivity according to Sutherland law
    diffus = shcacp_*actmat->ComputeDiffusivity(temp);

    // compute density based on temperature and thermodynamic pressure
    dens = actmat->ComputeDensity(temp,thermpressnp_);
  }
  else if (material->MaterialType() == INPAR::MAT::m_arrhenius_pv)
  {
    const MAT::ArrheniusPV* actmat = static_cast<const MAT::ArrheniusPV*>(material.get());

    // compute progress variable
    double provar = 0.0;
    for (int i=0; i<nen_; ++i)
    {
      provar += funct_(i)*ephinp[i];
    }

    // get specific heat capacity at constant pressure and
    // compute temperature based on progress variable
    shcacp_ = actmat->ComputeShc(provar);
    const double temp = actmat->ComputeTemperature(provar);

    // compute thermal conductivity according to Sutherland law
    diffus = shcacp_*actmat->ComputeDiffusivity(temp);

    // compute density at n+1 or n+alpha_F
    dens = actmat->ComputeDensity(provar);
  }
  else if (material->MaterialType() == INPAR::MAT::m_ferech_pv)
  {
    const MAT::FerEchPV* actmat = static_cast<const MAT::FerEchPV*>(material.get());

    // compute progress variable
    double provar = 0.0;
    for (int i=0; i<nen_; ++i)
    {
      provar += funct_(i)*ephinp[i];
    }

    // get specific heat capacity at constant pressure and
    // compute temperature based on progress variable
    shcacp_ = actmat->ComputeShc(provar);
    const double temp = actmat->ComputeTemperature(provar);

    // compute thermal conductivity according to Sutherland law
    diffus = shcacp_*actmat->ComputeDiffusivity(temp);

    // compute density at n+1 or n+alpha_F
    dens = actmat->ComputeDensity(provar);
  }
  else dserror("Material type is not supported");

  /*----------------------------------------- declaration of variables ---*/
  LINALG::SerialDenseMatrix nodecoords;
  nodecoords = DRT::UTILS::getEleNodeNumbering_nodes_paramspace(distype);

  if ((int) nodecoords.N() != nen_) dserror("number of nodes does not match");

  // loop over all nodes
  for (int iquad=0; iquad<nen_; ++iquad)
  {
    // reference coordinates of the current node
    for (int idim=0;idim<nsd_;idim++)
      {xsi_(idim) = nodecoords(idim, iquad);}

    // first derivatives
    DRT::UTILS::shape_function_deriv1<distype>(xsi_,deriv_);

    // compute Jacobian matrix and determinant
    // actually compute its transpose....
    xjm_.MultiplyNT(deriv_,xyze_);
    const double det = xij_.Invert(xjm_);

    if (det < 1E-16)
      dserror("GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f",ele->Id(), det);

    // compute global derivatives
    derxy_.Multiply(xij_,deriv_);

    // gradient of electric potential
    gradpot_.Clear();
    if (frt > 0.0) // ELCH
    {
      for (int k=0;k<nen_;k++)
      {
        for (int idim=0; idim<nsd_ ;idim++)
        {
          gradpot_(idim) += derxy_(idim,k)*ephinp[k*numdofpernode_+numscal_];
        }
      }
    }

    const double ephinpatnode = ephinp[iquad*numdofpernode_+dofindex];
    // add different flux contributions as specified by user input
    switch (fluxtype)
    {
      case INPAR::SCATRA::flux_total_domain:
        if (frt > 0.0) // ELCH
        {
          // migration flux terms
          for (int idim=0; idim<nsd_ ;idim++)
          {
            flux(idim,iquad) += diffus_valence_frt*gradpot_(idim)*ephinpatnode;
          }
        }
        // convective flux terms
        for (int idim=0; idim<nsd_ ;idim++)
        {
          flux(idim,iquad) -= dens*evel[idim+iquad*nsd_]*ephinpatnode;
        }
        // no break statement here!
      case INPAR::SCATRA::flux_diffusive_domain:
        //diffusive flux terms
        for (int k=0;k<nen_;k++)
        {
          for (int idim=0; idim<nsd_ ;idim++)
          {
            flux(idim,iquad) += diffus*derxy_(idim,k)*ephinp[k*numdofpernode_+dofindex];
          }
        }
        break;
      default:
        dserror("received illegal flag inside flux evaluation for whole domain");
    };

    //set zeros for unused space dimenions
    for (int idim=nsd_; idim<3; idim++)
    {
      flux(idim,iquad) = 0.0;
    }
  } // loop over nodes

  return;
} // ScaTraImpl::CalculateFlux


/*----------------------------------------------------------------------*
 |  calculate scalar(s) and domain integral                     vg 09/08|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalculateScalars(
    const DRT::Element*             ele,
    const vector<double>&           ephinp,
    Epetra_SerialDenseVector&       scalars,
    const bool                      inverting
)
{
  /*------------------------------------------------- set element data */
  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // integrations points and weights
  const DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad,ele->Id());

    // calculate integrals of (inverted) scalar(s) and domain
    if (inverting)
    {
      for (int i=0; i<nen_; i++)
      {
        const double fac_funct_i = fac*funct_(i);
        for (int k = 0; k < numscal_; k++)
        {
          scalars[k] += fac_funct_i/ephinp[i*numdofpernode_+k];
        }
        scalars[numscal_] += fac_funct_i;
      }
    }
    else
    {
      for (int i=0; i<nen_; i++)
      {
        const double fac_funct_i = fac*funct_(i);
        for (int k = 0; k < numscal_; k++)
        {
          scalars[k] += fac_funct_i*ephinp[i*numdofpernode_+k];
        }
        scalars[numscal_] += fac_funct_i;
      }
    }
  } // loop over integration points

  return;
} // ScaTraImpl::CalculateScalars


/*----------------------------------------------------------------------*
 |  calculate domain integral                                   vg 01/09|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalculateDomainAndBodyforce(
    Epetra_SerialDenseVector&  scalars,
    const DRT::Element*        ele,
    const double               time,
    const bool                 reinitswitch
)
{
  // ---------------------------------------------------------------------
  // call routine for calculation of body force in element nodes
  // (time n+alpha_F for generalized-alpha scheme, at time n+1 otherwise)
  // ---------------------------------------------------------------------

//REINHARD
  if (reinitswitch == false)
    BodyForce(ele,time);
  else
    BodyForceReinit(ele,time);
//end REINHARD

  /*------------------------------------------------- set element data */
  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // integrations points and weights
  const DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad,ele->Id());

    // get bodyforce in gausspoint
    rhs_[0] = bodyforce_[0].Dot(funct_);

    // calculate integrals of domain and bodyforce
    for (int i=0; i<nen_; i++)
    {
      scalars[0] += fac*funct_(i);
    }
    scalars[1] += fac*rhs_[0];

  } // loop over integration points

  return;
} // ScaTraImpl::CalculateDomain


/*----------------------------------------------------------------------*
 |  Integrate shape functions over domain (private)           gjb 07/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::IntegrateShapeFunctions(
    const DRT::Element*             ele,
    Epetra_SerialDenseVector&       elevec1,
    const Epetra_IntSerialDenseVector& dofids
)
{
  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // integrations points and weights
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // safety check
  if (dofids.M() < numdofpernode_)
    dserror("Dofids vector is too short. Received not enough flags");

  // loop over integration points
  for (int gpid=0; gpid<intpoints.IP().nquad; gpid++)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,gpid,ele->Id());

    // compute integral of shape functions (only for dofid)
    for (int k=0;k<numdofpernode_;k++)
    {
      if (dofids[k] >= 0)
      {
        for (int node=0;node<nen_;node++)
        {
          elevec1[node*numdofpernode_+k] += funct_(node) * fac;
        }
      }
    }

  } //loop over integration points

  return;

} //ScaTraImpl<distype>::IntegrateShapeFunction


/*----------------------------------------------------------------------*
 |  Calculate conductivity (ELCH) (private)                   gjb 07/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalculateConductivity(
    const DRT::Element*  ele,
    const double         frt,
    Epetra_SerialDenseVector& sigma
)
{
  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  GetMaterialParams(ele);

  // use one-point Gauss rule to do calculations at the element center
  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints_tau(SCATRA::DisTypeToStabGaussRule<distype>::rule);

  // evaluate shape functions (and not needed derivatives) at element center
  EvalShapeFuncAndDerivsAtIntPoint(intpoints_tau,0,ele->Id());

  // compute the conductivity (1/(\Omega m) = 1 Siemens / m)
  double sigma_all(0.0);
  const double factor = frt*96485.34; // = F^2/RT
  for(int k=0; k < numscal_; k++)
  {
    // concentration of ionic species k at element center
    double conint = funct_.Dot(ephinp_[k]);
    double sigma_k = factor*valence_[k]*diffusvalence_[k]*conint;
    sigma[k] += sigma_k; // insert value for this ionic species
    sigma_all += sigma_k;
  }
  sigma[numscal_] += sigma_all; // conductivity based on ALL ionic species

  return;

} //ScaTraImpl<distype>::CalculateConductivity


/*----------------------------------------------------------------------*
 |  CalculateElectricPotentialField (ELCH) (private)          gjb 04/10 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraImpl<distype>::CalculateElectricPotentialField(
    const DRT::Element*         ele,
    const double                frt,
    Epetra_SerialDenseMatrix&   emat,
    Epetra_SerialDenseVector&   erhs
)
{
  // get node coordinates
  GEO::fillInitialPositionArray<distype,nsd_,LINALG::Matrix<nsd_,nen_> >(ele,xyze_);

  // in the ALE case add nodal displacements
  if (isale_) xyze_ += edispnp_;

  // access material parameters
  GetMaterialParams(ele);

  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad,ele->Id());
    double sigmaint(0.0);
    for (int k=0; k<numscal_; ++k)
    {
      // concentration of ionic species k at element center
      double conintk = funct_.Dot(ephinp_[k]);
      double sigma_k = frt*valence_[k]*diffusvalence_[k]*conintk;
      sigmaint += sigma_k;

      // diffusive terms on rhs
      // gradient of current scalar value
      gradphi_.Multiply(derxy_,ephinp_[k]);
      const double vrhs = fac*diffusvalence_[k];
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = vi*numdofpernode_+numscal_;
        double laplawf(0.0);
        GetLaplacianWeakFormRHS(laplawf,derxy_,gradphi_,vi);
        erhs[fvi] -= vrhs*laplawf;
      }

      // provide something for conc. dofs: a standard mass matrix
      for (int vi=0; vi<nen_; ++vi)
      {
        const int    fvi = vi*numdofpernode_+k;
        for (int ui=0; ui<nen_; ++ui)
        {
          const int fui = ui*numdofpernode_+k;
          emat(fvi,fui) += fac*funct_(vi)*funct_(ui);
        }
      }
    } // for k

    // ----------------------------------------matrix entries
    for (int vi=0; vi<nen_; ++vi)
    {
      const int    fvi = vi*numdofpernode_+numscal_;
      for (int ui=0; ui<nen_; ++ui)
      {
        const int fui = ui*numdofpernode_+numscal_;
        double laplawf(0.0);
        GetLaplacianWeakForm(laplawf, derxy_,ui,vi);
        emat(fvi,fui) += fac*sigmaint*laplawf;
      }
    }
  } // integration loop

  return;

} //ScaTraImpl<distype>::CalculateElectricPotentialField

#endif // CCADISCRET
#endif // D_FLUID3 or D_FLUID2
