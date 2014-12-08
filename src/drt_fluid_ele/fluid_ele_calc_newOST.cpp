/*----------------------------------------------------------------------*/
/*!
\file fluid_ele_calc.cpp

\brief main file containing routines for calculation of fluid element

<pre>
Maintainer: Ursula Rasthofer & Volker Gravemeier
            {rasthofer,vgravem}@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236/-245
</pre>
*/
/*----------------------------------------------------------------------*/

#include "fluid_ele_calc.H"
#include "fluid_ele_parameter.H"
#include "fluid_ele_parameter_std.H"
#include "fluid_ele_parameter_timint.H"
#include "fluid_ele.H"
#include "../drt_lib/drt_element_integration_select.H"
#include "fluid_ele_tds.H"

#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"
#include "../drt_fem_general/drt_utils_gder2.H"
#include "../drt_fem_general/drt_utils_nurbs_shapefunctions.H"

#include "../drt_fluid/fluid_rotsym_periodicbc.H"

#include "../drt_geometry/position_array.H"

#include "../drt_inpar/inpar_turbulence.H"
#include "../drt_inpar/inpar_topopt.H"

#include "../drt_lib/drt_condition_utils.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_elementtype.H"
#include "../drt_lib/standardtypes_cpp.H"

#include "../drt_mat/arrhenius_pv.H"
#include "../drt_mat/carreauyasuda.H"
#include "../drt_mat/ferech_pv.H"
#include "../drt_mat/herschelbulkley.H"
#include "../drt_mat/mixfrac.H"
#include "../drt_mat/modpowerlaw.H"
#include "../drt_mat/newtonianfluid.H"
#include "../drt_mat/permeablefluid.H"
#include "../drt_mat/sutherland.H"
#include "../drt_mat/yoghurt.H"
#include "../drt_mat/matlist.H"

#include "../drt_nurbs_discret/drt_nurbs_utils.H"


/*----------------------------------------------------------------------*
 |  calculate element matrix and right hand side (private)   g.bau 03/07|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::SysmatOSTNew(
  const LINALG::Matrix<nsd_,nen_>&              ebofoaf,
  const LINALG::Matrix<nsd_,nen_>&              eprescpgaf,
  const LINALG::Matrix<nsd_,nen_>&              ebofon,
  const LINALG::Matrix<nsd_,nen_>&              eprescpgn,
  const LINALG::Matrix<nsd_,nen_>&              evelaf,
  const LINALG::Matrix<nsd_,nen_>&              eveln,
  const LINALG::Matrix<nsd_,nen_>&              evelnp,
  const LINALG::Matrix<nsd_,nen_>&              fsevelaf,
  const LINALG::Matrix<nen_,1>&                 fsescaaf,
  const LINALG::Matrix<nsd_,nen_>&              evel_hat,
  const LINALG::Matrix<nsd_*nsd_,nen_>&         ereynoldsstress_hat,
  const LINALG::Matrix<nen_,1>&                 epreaf,
  const LINALG::Matrix<nen_,1>&                 epren,
  const LINALG::Matrix<nen_,1>&                 eprenp,
  const LINALG::Matrix<nsd_,nen_>&              eaccam,
  const LINALG::Matrix<nen_,1>&                 escaaf,
  const LINALG::Matrix<nen_,1>&                 escaam,
  const LINALG::Matrix<nen_,1>&                 escadtam,
  const LINALG::Matrix<nen_,1>&                 escabofoaf,
  const LINALG::Matrix<nen_,1>&                 escabofon,
  const LINALG::Matrix<nsd_,nen_>&              emhist,
  const LINALG::Matrix<nsd_,nen_>&              edispnp,
  const LINALG::Matrix<nsd_,nen_>&              egridv,
  LINALG::Matrix<(nsd_+1)*nen_,(nsd_+1)*nen_>&  estif,
  LINALG::Matrix<(nsd_+1)*nen_,(nsd_+1)*nen_>&  emesh,
  LINALG::Matrix<(nsd_+1)*nen_,1>&              eforce,
  const LINALG::Matrix<nen_,1> &                eporo,
  const LINALG::Matrix<nsd_,nen_> &             egradphi,
  const LINALG::Matrix<nen_,1> &                ecurvature,
  const double                                  thermpressaf,
  const double                                  thermpressam,
  const double                                  thermpressdtaf,
  const double                                  thermpressdtam,
  Teuchos::RCP<const MAT::Material>             material,
  double&                                       Cs_delta_sq,
  double&                                       Ci_delta_sq,
  double&                                       Cv,
  bool                                          isale,
  double * saccn,
  double * sveln,
  double * svelnp,
  const DRT::UTILS::GaussIntegration & intpoints
  )
{
  //------------------------------------------------------------------------
  //  preliminary definitions and evaluations
  //------------------------------------------------------------------------
  // definition of matrices
  LINALG::Matrix<nen_*nsd_,nen_*nsd_>  estif_u(true);
  LINALG::Matrix<nen_*nsd_,nen_>       estif_p_v(true);
  LINALG::Matrix<nen_, nen_*nsd_>      estif_q_u(true);
  LINALG::Matrix<nen_,nen_>            ppmat(true);

  // definition of vectors
  LINALG::Matrix<nen_,1>     preforce(true);
  LINALG::Matrix<nsd_,nen_>  velforce(true);

  // definition of velocity-based momentum residual vectors
  LINALG::Matrix<nsd_*nsd_,nen_>  lin_resM_Du(true);
  LINALG::Matrix<nsd_,1>          resM_Du(true);

  // if polynomial pressure projection: reset variables
  if (fldpara_->PPP())
  {
    D_ = 0;
    E_.Clear();
  }

  // evaluate shape functions and derivatives at element center
  EvalShapeFuncAndDerivsAtEleCenter();

  // set element area or volume
  const double vol = fac_;

  //------------------------------------------------------------------------
  // potential evaluation of material parameters, subgrid viscosity
  // and/or stabilization parameters at element center
  //------------------------------------------------------------------------
  // get material parameters at element center
  if (not fldpara_->MatGp() or not fldpara_->TauGp())
  {

    GetMaterialParams(material,evelaf,escaaf,escaam,escabofoaf,thermpressaf,thermpressam,thermpressdtaf,thermpressdtam,vol);

    // calculate all-scale or fine-scale subgrid viscosity at element center
    visceff_ = visc_;

    if (fldpara_->TurbModAction() == INPAR::FLUID::smagorinsky
        or fldpara_->TurbModAction() == INPAR::FLUID::dynamic_smagorinsky
        or fldpara_->TurbModAction() == INPAR::FLUID::vreman
        or fldpara_->TurbModAction() == INPAR::FLUID::dynamic_vreman)
    {
      if (fldpara_->TurbModAction() == INPAR::FLUID::dynamic_vreman)
        Cs_delta_sq=Cv; //use the declaration of Cs_delta_sq for the dynamic Vreman constant
      CalcSubgrVisc(evelaf,vol,Cs_delta_sq,Ci_delta_sq);
      // effective viscosity = physical viscosity + (all-scale) subgrid viscosity
      visceff_ += sgvisc_;
    }
    else if (fldpara_->Fssgv() != INPAR::FLUID::no_fssgv)
      CalcFineScaleSubgrVisc(evelaf,fsevelaf,vol);
  }

  // potential evaluation of multifractal subgrid-scales at element center
  // coefficient B of fine-scale velocity
  LINALG::Matrix<nsd_,1> B_mfs(true);
  // coefficient D of fine-scale scalar (loma only)
  double D_mfs = 0.0;
  if (fldpara_->TurbModAction() == INPAR::FLUID::multifractal_subgrid_scales)
  {
    if (not fldpara_->BGp())
    {
      // make sure to get material parameters at element center
      if (fldpara_->MatGp())
        //GetMaterialParams(material,evelaf,escaaf,escaam,thermpressaf,thermpressam,thermpressdtam,vol);
        GetMaterialParams(material,evelaf,escaaf,escaam,escabofoaf,thermpressaf,thermpressam,thermpressdtaf,thermpressdtam,vol);

      // provide necessary velocities and gradients at element center
      velint_.Multiply(evelaf,funct_);
      fsvelint_.Multiply(fsevelaf,funct_);
      vderxy_.MultiplyNT(evelaf,derxy_);
      // calculate parameters of multifractal subgrid-scales and, finally,
      // calculate coefficient for multifractal modeling of subgrid velocity
      // if loma, calculate coefficient for multifractal modeling of subgrid scalar
      PrepareMultifractalSubgrScales(B_mfs, D_mfs, evelaf, fsevelaf, vol);
      // clear all velocities and gradients
      velint_.Clear();
      fsvelint_.Clear();
      vderxy_.Clear();
    }
  }


  // calculate stabilization parameter at element center
  if (not fldpara_->TauGp() and fldpara_->StabType()==INPAR::FLUID::stabtype_residualbased)
  {
    // get convective velocity at element center
    // for evaluation of stabilization parameter
    velint_.Multiply(evelaf,funct_);

    // get the grid velocity in case of ALE
    if(isale) gridvelint_.Multiply(egridv,funct_);

    // get convective velocity at integration point
    SetConvectiveVelint(isale);


    if (fldpara_->Tds()==INPAR::FLUID::subscales_time_dependent)
    {
      // get velocity derivatives at integration point
      // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
      vderxy_.MultiplyNT(evelaf,derxy_); //required for time-dependent subscales

      // compute velnp at integration point (required for time-dependent subscales)
      LINALG::Matrix<nsd_,1> velintnp(true);
      velintnp.Multiply(evelnp,funct_);
      vel_normnp_ = velintnp.Norm2();
    }

    // calculate stabilization parameters at element center
    CalcStabParameter(vol);
  }

  // get Gaussian integration points
  //const DRT::UTILS::IntegrationPoints3D intpoints(ele->gaussrule_);
  //const DRT::UTILS::IntPointsAndWeights<nsd_> intpoints(DRT::ELEMENTS::DisTypeToOptGaussRule<distype>::rule);

  //------------------------------------------------------------------------
  //  start loop over integration points
  //------------------------------------------------------------------------
  //for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)

  for ( DRT::UTILS::GaussIntegration::const_iterator iquad=intpoints.begin(); iquad!=intpoints.end(); ++iquad )
  {
    // evaluate shape functions and derivatives at integration point
    EvalShapeFuncAndDerivsAtIntPoint(iquad.Point(),iquad.Weight());

    //----------------------------------------------------------------------
    //  evaluation of various values at integration point:
    //  1) velocity (including derivatives, fine-scale and grid velocity)
    //  2) pressure (including derivatives)
    //  3) body-force vector
    //  4) "history" vector for momentum equation
    //----------------------------------------------------------------------
    // get velocity at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    velint_.Multiply(evelaf,funct_);

    // get velocity derivatives at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    vderxy_.MultiplyNT(evelaf,derxy_);

    // get fine-scale velocity and its derivatives at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    if (fldpara_->Fssgv() != INPAR::FLUID::no_fssgv)
    {
      fsvderxy_.MultiplyNT(fsevelaf,derxy_);
    }
    else
    {
      fsvderxy_.Clear();
    }
    if (fldpara_->TurbModAction() == INPAR::FLUID::multifractal_subgrid_scales)
    {
      fsvelint_.Multiply(fsevelaf,funct_);
      fsvderxy_.MultiplyNT(fsevelaf,derxy_);
    }
    else
    {
      fsvelint_.Clear();
    }

    // get the grid velocity in case of ALE
    if(isale) gridvelint_.Multiply(egridv,funct_);

    // get convective velocity at integration point
    SetConvectiveVelint(isale);


    // get pressure at integration point
    // (value at n+alpha_F for generalized-alpha scheme,
    //  value at n+alpha_F for generalized-alpha-NP schemen, n+1 otherwise)
    double press = 0.0;
    if(fldparatimint_->IsGenalphaNP())
      press= funct_.Dot(eprenp);
    else
      press = funct_.Dot(epreaf);

    // get pressure gradient at integration point
    // (value at n+alpha_F for generalized-alpha scheme,
    //  value at n+alpha_F for generalized-alpha-NP schemen, n+1 otherwise)
    if(fldparatimint_->IsGenalphaNP())
      gradp_.Multiply(derxy_,eprenp);
    else
      gradp_.Multiply(derxy_,epreaf);

    // get bodyforce at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    bodyforce_.Multiply(ebofoaf,funct_);
    bodyforcen_.Multiply(ebofon,funct_);

    // get prescribed pressure gradient acting as body force
    // (required for turbulent channel flow)
    generalbodyforce_.Multiply(eprescpgaf,funct_);
    generalbodyforcen_.Multiply(eprescpgn,funct_);

    //TODO: Need gradphi and curvature at time n. (Can be implemented for interface values at timestep t^(n+1))
    if(fldpara_->GetIncludeSurfaceTension())
      AddSurfaceTensionForce(escaaf,escaam,egradphi,ecurvature);

    // get momentum history data at integration point
    // (only required for one-step-theta and BDF2 time-integration schemes)
    histmom_.Multiply(emhist,funct_);

    // preparation of scale similarity type models
    // get filtered velocities and reynolds-stresses at integration point
    // get fine scale velocity at integration point for advanced models
    if (fldpara_->TurbModAction() == INPAR::FLUID::scale_similarity
     or fldpara_->TurbModAction() == INPAR::FLUID::scale_similarity_basic)
    {
      velinthat_.Clear();
      velhatderxy_.Clear();

      // get filtered velocity at integration point
      velinthat_.Multiply(evel_hat,funct_);
      // get filtered velocity derivatives at integration point
      velhatderxy_.MultiplyNT(evel_hat,derxy_);

      reystressinthat_.Clear();
      // get filtered reynoldsstress at integration point
      for (int dimi=0;dimi<nsd_;dimi++)
      {
        for (int dimj=0;dimj<nsd_;dimj++)
        {
          for (int inode=0;inode<nen_;inode++)
          {
            reystressinthat_(dimi,dimj) += funct_(inode) * ereynoldsstress_hat(3*dimi+dimj,inode);
          }
        }
      }

      // filtered velocity divergence from previous iteration
      velhatdiv_ = 0.0;
      for (int idim = 0; idim <nsd_; ++idim)
      {
        velhatdiv_ += velhatderxy_(idim, idim);
      }

      LINALG::Matrix<nsd_*nsd_,nen_> evelhativelhatj;
      velhativelhatjdiv_.Clear();
      for (int nn=0;nn<nsd_;++nn)
      {
        for (int rr=0;rr<nsd_;++rr)
        {
          for (int mm=0;mm<nen_;++mm)
          {
            velhativelhatjdiv_(nn,0) += derxy_(rr,mm)*evel_hat(nn,mm)*evel_hat(rr,mm);
          }
        }
      }

      // get divergence of filtered reynoldsstress at integration point
      reystresshatdiv_.Clear();
      for (int nn=0;nn<nsd_;++nn)
      {
        for (int rr=0;rr<nsd_;++rr)
        {
            int index = 3*nn+rr;
            for (int mm=0;mm<nen_;++mm)
            {
              reystresshatdiv_(nn,0) += derxy_(rr,mm)*ereynoldsstress_hat(index,mm);
            }
        }
      }

      // get fine scale velocity at integration point
      fsvelint_.Multiply(fsevelaf,funct_);
    }
    else
    {
      velinthat_.Clear();
      velhatderxy_.Clear();
      reystressinthat_.Clear();
      reystresshatdiv_.Clear();
      velhativelhatjdiv_.Clear();
    }


    //----------------------------------------------------------------------
    // potential evaluation of material parameters, subgrid viscosity
    // and/or stabilization parameters at integration point
    //----------------------------------------------------------------------
    // get material parameters at integration point

    if (fldpara_->MatGp())
    {

      GetMaterialParams(material,evelaf,escaaf,escaam,escabofoaf,thermpressaf,thermpressam,thermpressdtaf,thermpressdtam,vol);

      // calculate all-scale or fine-scale subgrid viscosity at integration point
      visceff_ = visc_;

      if (fldpara_->TurbModAction() == INPAR::FLUID::smagorinsky or fldpara_->TurbModAction() == INPAR::FLUID::dynamic_smagorinsky or fldpara_->TurbModAction() == INPAR::FLUID::vreman)
      {
        CalcSubgrVisc(evelaf,vol,Cs_delta_sq,Ci_delta_sq);
        // effective viscosity = physical viscosity + (all-scale) subgrid viscosity
        visceff_ += sgvisc_;
      }
      else if (fldpara_->Fssgv() != INPAR::FLUID::no_fssgv)
        CalcFineScaleSubgrVisc(evelaf,fsevelaf,vol);
    }

    // get reaction coefficient due to porosity for topology optimization
    // !do this only at gauss point since this is nonlinear!
    if (fldpara_->ReactionTopopt()) GetPorosityAtGP(eporo);

    // calculate stabilization parameter at integration point
    if (fldpara_->TauGp() and fldpara_->StabType()==INPAR::FLUID::stabtype_residualbased)
      CalcStabParameter(vol);

    // potential evaluation of coefficient of multifractal subgrid-scales at integration point
    if (fldpara_->TurbModAction() == INPAR::FLUID::multifractal_subgrid_scales)
    {
      if (fldpara_->BGp())
      {
        // make sure to get material parameters at gauss point
        if (not fldpara_->MatGp())
        {
          // GetMaterialParams(material,evelaf,escaaf,escaam,escabofoaf,thermpressaf,thermpressam,thermpressdtaf,thermpressdtam);
          // would overwrite materials at the element center, hence BGp() should always be combined with MatGp()
          dserror("evaluation of B and D at gauss-point should always be combined with evaluation material at gauss-point!");
        }

        // calculate parameters of multifractal subgrid-scales
        PrepareMultifractalSubgrScales(B_mfs, D_mfs, evelaf, fsevelaf, vol);
      }

      // calculate fine-scale velocity, its derivative and divergence for multifractal subgrid-scale modeling
      for (int idim=0; idim<nsd_; idim++)
        mffsvelint_(idim,0) = fsvelint_(idim,0) * B_mfs(idim,0);

      for (int idim=0; idim<nsd_; idim++)
      {
        for (int jdim=0; jdim<nsd_; jdim++)
          mffsvderxy_(idim,jdim) = fsvderxy_(idim,jdim) * B_mfs(idim,0);
      }

      mffsvdiv_ = mffsvderxy_(0,0) + mffsvderxy_(1,1) + mffsvderxy_(2,2);

      // only required for variable-density flow at low Mach number
      if (fldpara_->PhysicalType() == INPAR::FLUID::loma)
      {
        mfssgscaint_ = D_mfs * funct_.Dot(fsescaaf);
        grad_fsscaaf_.Multiply(derxy_,fsescaaf);
        for (int dim=0; dim<nsd_; dim++)
            grad_fsscaaf_(dim,0) *= D_mfs;
      }
      else
      {
        mfssgscaint_ = 0.0;
        grad_fsscaaf_.Clear();
      }

      if (isale) dserror("Multifractal subgrid-scales with ale not supported");
    }
    else
    {
      mffsvelint_.Clear();
      mffsvderxy_.Clear();
      mffsvdiv_ = 0.0;
    }

    //----------------------------------------------------------------------
    //  evaluation of various partial operators at integration point
    //  1) convective term from previous iteration and convective operator
    //  2) viscous term from previous iteration and viscous operator
    //  3) divergence of velocity from previous iteration
    //----------------------------------------------------------------------

    // compute convective term from previous iteration and convective operator
    conv_old_.Multiply(vderxy_,convvelint_);
    conv_c_.MultiplyTN(derxy_,convvelint_);

    // compute viscous term from previous iteration and viscous operator
    if (is_higher_order_ele_) CalcDivEps(evelaf,eveln);
    else
    {
      visc_old_.Clear();
      visc_oldn_.Clear();
      viscs2_.Clear();
    }

    // compute divergence of velocity from previous iteration
    vdiv_ = 0.0;
    vdivn_ = 0.0;
    vderxyn_.MultiplyNT(eveln,derxy_);
    if (not fldparatimint_->IsGenalphaNP())
    {
      for (int idim = 0; idim <nsd_; ++idim)
      {
        vdiv_ += vderxy_(idim, idim);
        vdivn_ += vderxyn_(idim,idim);
      }
    }
    else
    {
      for (int idim = 0; idim <nsd_; ++idim)
      {
        //get vdiv at time n+1 for np_genalpha,
        LINALG::Matrix<nsd_,nsd_> vderxy(true);
        vderxy.MultiplyNT(evelnp,derxy_);
        vdiv_ += vderxy(idim, idim);
      }
    }

    // New One Step Theta variables (for old time step):
    velintn_.Multiply(eveln,funct_);
    conv_oldn_.Multiply(vderxyn_,velintn_);
    const double pressn = funct_.Dot(epren);
    gradpn_.Multiply(derxy_,epren);

    //----------------------------------------------------------------------
    // set time-integration factors for left- and right-hand side
    //----------------------------------------------------------------------
    const double timefacfac    = fldparatimint_->TimeFac()    * fac_;
    const double timefacfacpre = fldparatimint_->TimeFacPre() * fac_;
    const double rhsfac        = fldparatimint_->TimeFacRhs() * fac_;

    //For One Step Theta rhsfac is: \Delta t (1 - \theta)
    const double rhsfacn       = (1-fldparatimint_->Theta()) * fldparatimint_->Dt() * fac_;
    // For One Step Theta,
    // the density multiplied with the instationary term has to be evaluated at time = ( n + \theta ).
    dens_theta_                 = fldparatimint_->Theta() * densaf_ + (1-fldparatimint_->Theta()) * densn_;

    //----------------------------------------------------------------------
    // computation of various subgrid-scale values and residuals
    //----------------------------------------------------------------------
    // compute residual of momentum equation and subgrid-scale velocity
    // -> residual of momentum equation different for generalized-alpha
    //    and other time-integration schemes
    double fac1    = 0.0;
    double fac2    = 0.0;
    double fac3    = 0.0;
    double facMtau = 0.0;
    ComputeSubgridScaleVelocityOSTNew(eaccam,fac1,fac2,fac3,facMtau,*iquad,saccn,sveln,svelnp);

    // compute residual of continuity equation
    // residual contains velocity divergence only for incompressible flow
    conres_old_ = vdiv_;
    conres_oldn_ = vdivn_;

    // following computations only required for variable-density flow at low Mach number
    if (fldpara_->PhysicalType() == INPAR::FLUID::loma)
    {
      // compute additional Galerkin terms on right-hand side of continuity equation
      // -> different for generalized-alpha and other time-integration schemes
      ComputeGalRHSContEq(eveln,escaaf,escaam,escadtam,isale);

      // add to residual of continuity equation
      conres_old_ -= rhscon_;

      if (fldpara_->UpdateMat() or
          fldpara_->ContiSUPG() or
          fldpara_->ContiCross() != INPAR::FLUID::cross_stress_stab_none or
          fldpara_->ContiReynolds() != INPAR::FLUID::reynolds_stress_stab_none or
          fldpara_->MultiFracLomaConti())
      {
        // compute subgrid-scale part of scalar
        // -> different for generalized-alpha and other time-integration schemes
        ComputeSubgridScaleScalar(escaaf,escaam);

        // update material parameters including subgrid-scale part of scalar
        if (fldpara_->UpdateMat())
        {
          // since we update the viscosity in the next step, a potential subgrid-scale velocity would be overwritten
          if (fldpara_->TurbModAction() == INPAR::FLUID::smagorinsky or fldpara_->TurbModAction() == INPAR::FLUID::dynamic_smagorinsky or fldpara_->TurbModAction() == INPAR::FLUID::vreman)
             dserror("No material update in combination with smagorinsky model!");

          if (fldpara_->TurbModAction() == INPAR::FLUID::multifractal_subgrid_scales)
            UpdateMaterialParams(material,evelaf,escaaf,escaam,thermpressaf,thermpressam,mfssgscaint_);
          else
            UpdateMaterialParams(material,evelaf,escaaf,escaam,thermpressaf,thermpressam,sgscaint_);
          visceff_ = visc_;
          if (fldpara_->TurbModAction() == INPAR::FLUID::smagorinsky or fldpara_->TurbModAction() == INPAR::FLUID::dynamic_smagorinsky or fldpara_->TurbModAction() == INPAR::FLUID::vreman)
            visceff_ += sgvisc_;
        }

        // right-hand side of continuity equation based on updated material parameters
        // and including all stabilization terms
        // -> different for generalized-alpha and other time-integration schemes
        RecomputeGalAndComputeCrossRHSContEq();
      }
    }
    else if (fldpara_->PhysicalType() == INPAR::FLUID::artcomp)
    {
      // compute additional Galerkin terms on right-hand side of continuity equation
      // -> different for generalized-alpha and other time-integration schemes
      ComputeGalRHSContEqArtComp(epreaf,epren,escadtam);

      // add to residual of continuity equation
      conres_old_ -= rhscon_;
    }

    // set velocity-based momentum residual vectors to zero
    lin_resM_Du.Clear();
    resM_Du.Clear();

    // compute first version of velocity-based momentum residual containing
    // inertia term, convection term (convective and reactive part),
    // reaction term and cross-stress term
    LinGalMomResUOSTNew(lin_resM_Du,
                  timefacfac);

    // potentially rescale first version of velocity-based momentum residual
    if(fldpara_->Tds()      ==INPAR::FLUID::subscales_time_dependent
       &&
       fldpara_->Transient()==INPAR::FLUID::inertia_stab_keep)
    {
      LinGalMomResU_subscales(estif_p_v,
                              lin_resM_Du,
                              resM_Du,
                              timefacfac,
                              facMtau);
    }

    //----------------------------------------------------------------------
    // computation of standard Galerkin and stabilization contributions to
    // element matrix and right-hand-side vector
    //----------------------------------------------------------------------
    // 1) standard Galerkin inertia, convection and reaction terms
    //    (convective and reactive part for convection term)
    //    as well as first part of cross-stress term on left-hand side
    InertiaConvectionReactionGalPart(estif_u,
                                     velforce,
                                     lin_resM_Du,
                                     resM_Du,
                                     rhsfac,
                                     rhsfacn);

    // 2) standard Galerkin viscous term
    //    (including viscous stress computation,
    //     excluding viscous part for low-Mach-number flow)
    LINALG::Matrix<nsd_,nsd_> viscstress(true);
    ViscousGalPart(estif_u,
                   velforce,
                   viscstress,
                   timefacfac,
                   rhsfac,
                   rhsfacn);

    // 3) stabilization of continuity equation,
    //    standard Galerkin viscous part for low-Mach-number flow and
    //    right-hand-side part of standard Galerkin viscous term
    if (fldpara_->CStab() or
        fldpara_->PhysicalType() == INPAR::FLUID::loma)
      ContStab( estif_u,
                velforce,
                fldparatimint_->TimeFac(),
                timefacfac,
                timefacfacpre,
                rhsfac,
                rhsfacn);

    // 4) standard Galerkin pressure term
    PressureGalPart(estif_p_v,
                    velforce,
                    timefacfac,
                    timefacfacpre,
                    rhsfac,
                    rhsfacn,
                    press,
                    pressn);

    // 5) standard Galerkin continuity term
    ContinuityGalPart(estif_q_u,
                      preforce,
                      timefacfac,
                      timefacfacpre,
                      rhsfac,
                      rhsfacn);

    // 6) standard Galerkin bodyforce term on right-hand side
    BodyForceRhsTerm(velforce,
                     rhsfac,
                     rhsfacn);

    // 7) additional standard Galerkin terms due to conservative formulation
    //    New One Step Theta not implemented for this as of yet!
    if (fldpara_->IsConservative())
    {
      ConservativeFormulation(estif_u,
                              velforce,
                              timefacfac,
                              rhsfac);
    }

    // 8) additional standard Galerkin terms for low-Mach-number flow and
    //    artificial compressibility (only right-hand side in latter case)
    //    New One Step Theta not implemented for LOMA as of yet.
    if (fldpara_->PhysicalType() == INPAR::FLUID::loma or
        fldpara_->PhysicalType() == INPAR::FLUID::artcomp)
    {
      LomaGalPart(estif_q_u,
                  preforce,
                  timefacfac,
                  rhsfac);
    }

    // 9) additional standard Galerkin term for temporal derivative of pressure
    //    in case of artificial compressibility (only left-hand side)
    if (fldpara_->PhysicalType() == INPAR::FLUID::artcomp and
        not fldparatimint_->IsStationary())
      ArtCompPressureInertiaGalPartandContStab(estif_p_v,
                                               ppmat);

    //----------------------------------------------------------------------
    // compute second version of velocity-based momentum residual containing
    // inertia term, convection term (convective and reactive part) and
    // viscous term
    //----------------------------------------------------------------------
    StabLinGalMomResU(lin_resM_Du,
                      timefacfac);

    // 10) PSPG term
    if (fldpara_->PSPG())
    {
      PSPGOSTNew(estif_q_u,
           ppmat,
           preforce,
           lin_resM_Du,
           fac3,
           timefacfac,
           timefacfacpre,
           rhsfac,
           *iquad);
    }

    // 11) SUPG term as well as first part of Reynolds-stress term on
    //     left-hand side and Reynolds-stress term on right-hand side
    if(fldpara_->SUPG())
    {
      SUPGOSTNew(estif_u,
           estif_p_v,
           velforce,
           preforce,
           lin_resM_Du,
           fac3,
           timefacfac,
           timefacfacpre,
           rhsfac);
    }

    // 12) reactive stabilization term
   if (fldpara_->RStab() != INPAR::FLUID::reactive_stab_none)
   {
      ReacStab(estif_u,
               estif_p_v,
               velforce,
               lin_resM_Du,
               timefacfac,
               timefacfacpre,
               rhsfac,
               fac3);
   }

    // 13) viscous stabilization term
    if (is_higher_order_ele_ and
        (fldpara_->VStab() != INPAR::FLUID::viscous_stab_none))
    {
      ViscStab(estif_u,
               estif_p_v,
               velforce,
               lin_resM_Du,
               timefacfac,
               timefacfacpre,
               rhsfac,
               fac3);
    }

    // if ConvDivStab for XFEM
//    {
//      ConvDivStab(estif_u,
//           velforce,
//           timefacfac,
//           rhsfac);
//    }


    // 14) cross-stress term: second part on left-hand side (only for Newton
    //     iteration) as well as cross-stress term on right-hand side
    if(fldpara_->Cross() != INPAR::FLUID::cross_stress_stab_none)
    {
      CrossStressStab(estif_u,
                      estif_p_v,
                      velforce,
                      lin_resM_Du,
                      timefacfac,
                      timefacfacpre,
                      rhsfac,
                      fac3);
    }

    // 15) Reynolds-stress term: second part on left-hand side
    //     (only for Newton iteration)
    if (fldpara_->Reynolds() == INPAR::FLUID::reynolds_stress_stab and
        fldpara_->IsNewton())
    {
      ReynoldsStressStab(estif_u,
                         estif_p_v,
                         lin_resM_Du,
                         timefacfac,
                         timefacfacpre,
                         fac3);
    }

    // 16) fine-scale subgrid-viscosity term
    //     (contribution only to right-hand-side vector)
    if(fldpara_->Fssgv() != INPAR::FLUID::no_fssgv)
    {
      const double fssgviscfac = fssgvisc_*rhsfac;

      FineScaleSubGridViscosityTerm(velforce,
                                    fssgviscfac);
    }

    // 17) subgrid-stress term (scale similarity)
    //     (contribution only to right-hand-side vector)
    if (fldpara_->TurbModAction() == INPAR::FLUID::scale_similarity)
    {
      ScaleSimSubGridStressTermCross(
                        velforce,
                        rhsfac,
                        fldpara_->Cl());

      ScaleSimSubGridStressTermReynolds(
                        velforce,
                        rhsfac,
                        fldpara_->Cl());
    }

    if (fldpara_->TurbModAction() == INPAR::FLUID::scale_similarity_basic)
    {
      ScaleSimSubGridStressTermPrefiltering(
                            velforce,
                            rhsfac,
                            fldpara_->Cl());
    }

    // 18) subgrid-stress term (multifractal subgrid scales)
    if (fldpara_->TurbModAction() == INPAR::FLUID::multifractal_subgrid_scales)
    {
      MultfracSubGridScalesCross(
                        estif_u,
                        velforce,
                        timefacfac,
                        rhsfac);

      MultfracSubGridScalesReynolds(
                        estif_u,
                        velforce,
                        timefacfac,
                        rhsfac);
    }

    // 19) polynomial pressure projection term (Dohrmann, Bochev IJNME 2004)
    //     (parameter-free inf-sub-stabilization, e.g. used instead of PSPG)
    if (fldpara_->PPP())
    {
      PressureProjection(ppmat);
    }

    // linearization wrt mesh motion
    if (emesh.IsInitialized())
    {
      if (nsd_ == 3)
        LinMeshMotion_3D(emesh,
                        evelaf,
                        press,
                        fldparatimint_->TimeFac(),
                        timefacfac);
      else if(nsd_ == 2)
        LinMeshMotion_2D(emesh,
                         evelaf,
                         press,
                         fldparatimint_->TimeFac(),
                         timefacfac);
      else
        dserror("Linearization of the mesh motion is not available in 1D");
    }

  }
  //------------------------------------------------------------------------
  //  end loop over integration points
  //------------------------------------------------------------------------

  // if polynomial pressure projection: finalize matrices and rhs
  if (fldpara_->PPP())
  {
    if (fldparatimint_->IsGenalphaNP())
      PressureProjectionFinalize(ppmat,preforce,eprenp);
    else
      PressureProjectionFinalize(ppmat,preforce,epreaf);
  }

  //------------------------------------------------------------------------
  //  add contributions to element matrix and right-hand-side vector
  //------------------------------------------------------------------------
  // add pressure part to right-hand-side vector
  for (int vi=0; vi<nen_; ++vi)
  {
    eforce(numdofpernode_*vi+nsd_)+=preforce(vi);
  }

  // add velocity part to right-hand-side vector
  for (int vi=0; vi<nen_; ++vi)
  {
    for (int idim=0; idim<nsd_; ++idim)
    {
      eforce(numdofpernode_*vi+idim)+=velforce(idim,vi);
    }
  }

  // add pressure-pressure part to matrix
  for (int ui=0; ui<nen_; ++ui)
  {
    const int fuippp = numdofpernode_*ui+nsd_;

    for (int vi=0; vi<nen_; ++vi)
    {
      const int numdof_vi_p_nsd = numdofpernode_*vi+nsd_;

      estif(numdof_vi_p_nsd,fuippp)+=ppmat(vi,ui);
    }
  }

  // add velocity-velocity part to matrix
  for (int ui=0; ui<nen_; ++ui)
  {
    const int numdof_ui = numdofpernode_*ui;
    const int nsd_ui = nsd_*ui;

    for (int jdim=0; jdim < nsd_;++jdim)
    {
      const int numdof_ui_jdim = numdof_ui+jdim;
      const int nsd_ui_jdim = nsd_ui+jdim;

      for (int vi=0; vi<nen_; ++vi)
      {
        const int numdof_vi = numdofpernode_*vi;
        const int nsd_vi = nsd_*vi;

        for (int idim=0; idim <nsd_; ++idim)
        {
          estif(numdof_vi+idim, numdof_ui_jdim) += estif_u(nsd_vi+idim, nsd_ui_jdim);
        }
      }
    }
  }

  // add velocity-pressure part to matrix
  for (int ui=0; ui<nen_; ++ui)
  {
    const int numdof_ui_nsd = numdofpernode_*ui + nsd_;

    for (int vi=0; vi<nen_; ++vi)
    {
      const int nsd_vi = nsd_*vi;
      const int numdof_vi = numdofpernode_*vi;

      for (int idim=0; idim <nsd_; ++idim)
      {
        estif(numdof_vi+idim, numdof_ui_nsd) += estif_p_v(nsd_vi+idim, ui);
      }
    }
  }

  // add pressure-velocity part to matrix
  for (int ui=0; ui<nen_; ++ui)
  {
    const int numdof_ui = numdofpernode_*ui;
    const int nsd_ui = nsd_*ui;

    for (int jdim=0; jdim < nsd_;++jdim)
    {
      const int numdof_ui_jdim = numdof_ui+jdim;
      const int nsd_ui_jdim = nsd_ui+jdim;

      for (int vi=0; vi<nen_; ++vi)
        estif(numdofpernode_*vi+nsd_, numdof_ui_jdim) += estif_q_u(vi, nsd_ui_jdim);
    }
  }

  return;
}


///*----------------------------------------------------------------------*
// |  compute body force at element nodes (protected)            vg 10/11 |
// *----------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::BodyForce(
//  DRT::ELEMENTS::Fluid*               ele,
//  LINALG::Matrix<nsd_,nen_>&          ebofoaf,
//  LINALG::Matrix<nsd_,nen_> &         eprescpgaf,
//  LINALG::Matrix<nen_,1>&             escabofoaf)
//{
//  BodyForce(ele, fldparatimint_->Time(), fldpara_->PhysicalType(), ebofoaf, eprescpgaf, escabofoaf);
//}
//
//
//
///*----------------------------------------------------------------------*
// |  compute body force at element nodes (public)               vg 10/11 |
// *----------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::BodyForce(DRT::ELEMENTS::Fluid*              ele,
//                                                     const double                       time,
//                                                     const INPAR::FLUID::PhysicalType   physicaltype,
//                                                     LINALG::Matrix<nsd_,nen_> &        ebofoaf,
//                                                     LINALG::Matrix<nsd_,nen_> &        eprescpgaf,
//                                                     LINALG::Matrix<nen_,1> &           escabofoaf)
//{
//  std::vector<DRT::Condition*> myneumcond;
//
//  // check whether all nodes have a unique Neumann condition
//  if (nsd_==3)
//    DRT::UTILS::FindElementConditions(ele, "VolumeNeumann", myneumcond);
//  else if (nsd_==2)
//    DRT::UTILS::FindElementConditions(ele, "SurfaceNeumann", myneumcond);
//  else
//    dserror("Body force for 1D problem not yet implemented!");
//
//  if (myneumcond.size()>1)
//    dserror("More than one Neumann condition on one node!");
//
//  if (myneumcond.size()==1)
//  {
//    const std::string* condtype = myneumcond[0]->Get<std::string>("type");
//
//    // check for potential time curve
//    const std::vector<int>* curve  = myneumcond[0]->Get<std::vector<int> >("curve");
//    int curvenum = -1;
//    if (curve) curvenum = (*curve)[0];
//
//    // initialization of time-curve factor
//    double curvefac = 0.0;
//
//    // compute potential time curve or set time-curve factor to one
//    if (curvenum >= 0)
//    {
//      // time factor (negative time indicating error)
//      if (time >= 0.0)
//           curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
//      else dserror("Negative time in bodyforce calculation: time = %f", time);
//    }
//    else curvefac = 1.0;
//
//    // get values and switches from the condition
//    const std::vector<int>*    onoff = myneumcond[0]->Get<std::vector<int> >   ("onoff");
//    const std::vector<double>* val   = myneumcond[0]->Get<std::vector<double> >("val"  );
//    const std::vector<int>*    functions = myneumcond[0]->Get<std::vector<int> >("funct");
//
//    // factor given by spatial function
//    double functionfac = 1.0;
//    int functnum = -1;
//
//    // set this condition to the ebofoaf array
//    for (int isd=0;isd<nsd_;isd++)
//    {
//      // get factor given by spatial function
//      if (functions) functnum = (*functions)[isd];
//      else functnum = -1;
//
//      double num = (*onoff)[isd]*(*val)[isd]*curvefac;
//
//      if(enrtype==DRT::ELEMENTS::Fluid::xwall)
//      {
//        // for xwall, only the linear shape functions contribute to the bodyforce,
//        // since the sum of all shape functions sum(N_j) != 1.0 if the enriched shape functions are included
//        for ( int jnode=0; jnode<nen_; jnode+=2 )
//        {
//          if (functnum>0)
//          {
//            // evaluate function at the position of the current node
//            // ------------------------------------------------------
//            // comment: this introduces an additional error compared to an
//            // evaluation at the integration point. However, we need a node
//            // based element bodyforce vector for prescribed pressure gradients
//            // in some fancy turbulance stuff.
//            functionfac = DRT::Problem::Instance()->Funct(functnum-1).Evaluate(isd,
//                                                                               (ele->Nodes()[jnode])->X(),
//                                                                               time,
//                                                                               NULL);
//          }
//          else functionfac = 1.0;
//
//          // get usual body force
//          if (*condtype == "neum_dead" or *condtype == "neum_live")
//            ebofoaf(isd,jnode) = num*functionfac;
//          // get prescribed pressure gradient
//          else if (*condtype == "neum_pgrad")
//            eprescpgaf(isd,jnode) = num*functionfac;
//          else
//            dserror("Unknown Neumann condition");
//        }
//      }
//      else
//      for ( int jnode=0; jnode<nen_; ++jnode )
//      {
//        if (functnum>0)
//        {
//          // evaluate function at the position of the current node
//          // ------------------------------------------------------
//          // comment: this introduces an additional error compared to an
//          // evaluation at the integration point. However, we need a node
//          // based element bodyforce vector for prescribed pressure gradients
//          // in some fancy turbulance stuff.
//          functionfac = DRT::Problem::Instance()->Funct(functnum-1).Evaluate(isd,
//                                                                             (ele->Nodes()[jnode])->X(),
//                                                                             time,
//                                                                             NULL);
//        }
//        else functionfac = 1.0;
//
//        // get usual body force
//        if (*condtype == "neum_dead" or *condtype == "neum_live")
//          ebofoaf(isd,jnode) = num*functionfac;
//        // get prescribed pressure gradient
//        else if (*condtype == "neum_pgrad")
//          eprescpgaf(isd,jnode) = num*functionfac;
//        else
//          dserror("Unknown Neumann condition");
//      }
//    }
//  }
//
//  // get nodal values of scatra bodyforce for variable-density flow
//  // at low Mach number
//  if (physicaltype == INPAR::FLUID::loma)
//  {
//    std::vector<DRT::Condition*> myscatraneumcond;
//
//    // check whether all nodes have a unique Neumann condition
//    if (nsd_==3)
//      DRT::UTILS::FindElementConditions(ele,"TransportVolumeNeumann",myscatraneumcond);
//    else if (nsd_==2)
//      DRT::UTILS::FindElementConditions(ele,"TransportSurfaceNeumann",myscatraneumcond);
//    else
//      dserror("Body force for 1D problem not yet implemented!");
//
//    if (myscatraneumcond.size()>1)
//      dserror("More than one Neumann condition on one node!");
//
//    if (myscatraneumcond.size()==1)
//    {
//      // check for potential time curve
//      const std::vector<int>* curve  = myscatraneumcond[0]->Get<std::vector<int> >("curve");
//      int curvenum = -1;
//      if (curve) curvenum = (*curve)[0];
//
//      // initialization of time-curve factor
//      double curvefac = 0.0;
//
//      // compute potential time curve or set time-curve factor to one
//      if (curvenum >= 0)
//      {
//        // time factor (negative time indicating error)
//        if (time >= 0.0)
//          curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
//        else dserror("Negative time in bodyforce calculation: time = %f", time);
//      }
//      else curvefac = 1.0;
//
//      // get values and switches from the condition
//      const std::vector<int>*    onoff = myscatraneumcond[0]->Get<std::vector<int> >   ("onoff");
//      const std::vector<double>* val   = myscatraneumcond[0]->Get<std::vector<double> >("val"  );
//
//      // set this condition to the bodyforce array
//      for (int jnode=0; jnode<nen_; jnode++)
//      {
//        escabofoaf(jnode) = (*onoff)[0]*(*val)[0]*curvefac;
//      }
//    }
//  }
//
//}

///*----------------------------------------------------------------------*
// |  compute surface tension force                              mw 05/14 |
// *----------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::AddSurfaceTensionForce(
//    const LINALG::Matrix<nen_,1>&                 escaaf,
//    const LINALG::Matrix<nen_,1>&                 escaam,
//    const LINALG::Matrix<nsd_,nen_> &             egradphi,
//    const LINALG::Matrix<nen_,1> &                ecurvature
//)
//{
//
//  double                 gaussescaaf;
//  gaussescaaf=funct_.Dot(escaaf);
//  // For surface tension only the values at alpha_f are needed/used.
//  //  double                 gaussescaam;
//  //  gaussescaam=funct_.Dot(escaam);
//
//  double epsilon = fldpara_->GetInterfaceThickness();
//
//  //Add surface force if inside interface thickness, otherwise do not.
//  if(abs(gaussescaaf) <= epsilon)
//  {
//    LINALG::Matrix<nsd_,1> gaussgradphi;
//    double                 gausscurvature;
//
//    gaussgradphi.Multiply(egradphi,funct_);
//    gausscurvature=funct_.Dot(ecurvature);
//
//
//    double Dheavyside_epsilon = 1.0/(2.0 * epsilon)*(1.0+cos(PI*gaussescaaf/epsilon));
//
//    double scalar_fac = Dheavyside_epsilon*gamma_*gausscurvature/gaussgradphi.Norm2();
//
//    generalbodyforce_.Update(scalar_fac,gaussgradphi,1.0);
//  }
//
//  if(fldparatimint_->IsNewOSTImplementation())
//  {
//    double                 gaussescaam;
//    gaussescaam=funct_.Dot(escaam);
//
//    //Add surface force if inside interface thickness, otherwise do not.
//    if(abs(gaussescaam) <= epsilon)
//    {
//      LINALG::Matrix<nsd_,1> gaussgradphi;
//      double                 gausscurvature;
//
//      gaussgradphi.Multiply(egradphi,funct_);
//      gausscurvature=funct_.Dot(ecurvature);
//
//
//      double Dheavyside_epsilon = 1.0/(2.0 * epsilon)*(1.0+cos(PI*gaussescaam/epsilon));
//
//      double scalar_fac = Dheavyside_epsilon*gamma_*gausscurvature/gaussgradphi.Norm2();
//
//      generalbodyforcen_.Update(scalar_fac,gaussgradphi,1.0);
//
//    }
//  }//end IsNewOSTImplementation
//
//  return;
//}

///*----------------------------------------------------------------------*
// | evaluate shape functions and derivatives at element center  vg 09/09 |
// *----------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::EvalShapeFuncAndDerivsAtEleCenter(
//)
//{
//  // use one-point Gauss rule
//  DRT::UTILS::IntPointsAndWeights<nsd_> intpoints_stab(DRT::ELEMENTS::DisTypeToStabGaussRule<distype>::rule);
//
//  EvalShapeFuncAndDerivsAtIntPoint((intpoints_stab.IP().qxg)[0],intpoints_stab.IP().qwgt[0]);
//
//  return;
//}


///*----------------------------------------------------------------------*
// | evaluate shape functions and derivatives at integr. point   vg 09/09 |
// *----------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::EvalShapeFuncAndDerivsAtIntPoint(
//    const double* gpcoord,  // actual integration point (coords)
//    double gpweight// actual integration point (weight)
//)
//{
//  for (int idim=0;idim<nsd_;idim++)
//  {
//     xsi_(idim) = gpcoord[idim];
//  }
//
//  if(not isNurbs_)
//  {
//    // shape functions and their first derivatives
//    DRT::UTILS::shape_function<distype>(xsi_,funct_);
//    DRT::UTILS::shape_function_deriv1<distype>(xsi_,deriv_);
//    derxy2_.Clear();
//    if (is_higher_order_ele_)
//    {
//      // get the second derivatives of standard element at current GP
//      DRT::UTILS::shape_function_deriv2<distype>(xsi_,deriv2_);
//    }
//  }
//  else
//  {
//    if (is_higher_order_ele_)
//      DRT::NURBS::UTILS::nurbs_get_funct_deriv_deriv2
//      (funct_  ,
//          deriv_  ,
//          deriv2_ ,
//          xsi_    ,
//          myknots_,
//          weights_,
//          distype );
//    else
//      DRT::NURBS::UTILS::nurbs_get_funct_deriv
//      (funct_  ,
//          deriv_  ,
//          xsi_    ,
//          myknots_,
//          weights_,
//          distype );
//  }
//
//  //
//
//  // get Jacobian matrix and determinant
//  // actually compute its transpose....
//  /*
//    +-            -+ T      +-            -+
//    | dx   dx   dx |        | dx   dy   dz |
//    | --   --   -- |        | --   --   -- |
//    | dr   ds   dt |        | dr   dr   dr |
//    |              |        |              |
//    | dy   dy   dy |        | dx   dy   dz |
//    | --   --   -- |   =    | --   --   -- |
//    | dr   ds   dt |        | ds   ds   ds |
//    |              |        |              |
//    | dz   dz   dz |        | dx   dy   dz |
//    | --   --   -- |        | --   --   -- |
//    | dr   ds   dt |        | dt   dt   dt |
//    +-            -+        +-            -+
//  */
//  xjm_.MultiplyNT(deriv_,xyze_);
//  det_ = xji_.Invert(xjm_);
//
//  if (det_ < 1E-16)
//    dserror("GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", eid_, det_);
//
//  // compute integration factor
//  fac_ = gpweight*det_;
//
//  // compute global first derivates
//  derxy_.Multiply(xji_,deriv_);
//
//  //--------------------------------------------------------------
//  //             compute global second derivatives
//  //--------------------------------------------------------------
//  if (is_higher_order_ele_)
//  {
//    DRT::UTILS::gder2<distype,nen_>(xjm_,derxy_,deriv2_,xyze_,derxy2_);
//  }
//  else derxy2_.Clear();
//
//  return;
//}

///*---------------------------------------------------------------------------*
// | get ALE grid displacements and grid velocity for element     schott 11/14 |
// *---------------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::GetGridDispVelALE(
//    DRT::Discretization &                   discretization,
//    const std::vector<int> &                lm,
//    LINALG::Matrix<nsd_,nen_>&              edispnp,
//    LINALG::Matrix<nsd_,nen_>&              egridv
//)
//{
//  switch (fldpara_->PhysicalType())
//  {
//  case INPAR::FLUID::oseen:
//  case INPAR::FLUID::stokes:
//  {
//    dserror("ALE with Oseen or Stokes seems to be a tricky combination. Think deep before removing dserror!");
//    break;
//  }
//  default:
//  {
//    ExtractValuesFromGlobalVector(discretization,lm, *rotsymmpbc_, &edispnp, NULL,"dispnp");
//    ExtractValuesFromGlobalVector(discretization,lm, *rotsymmpbc_, &egridv, NULL,"gridv");
//    break;
//  }
//  }
//}
//
///*---------------------------------------------------------------------------*
// |  set the (relative) convective velocity at integration point schott 11/14 |
// *---------------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::SetConvectiveVelint(
//    const bool isale
//)
//{
//  // get convective velocity at integration point
//  switch (fldpara_->PhysicalType())
//  {
//  case INPAR::FLUID::incompressible:
//  case INPAR::FLUID::artcomp:
//  case INPAR::FLUID::varying_density:
//  case INPAR::FLUID::loma:
//  case INPAR::FLUID::boussinesq:
//  case INPAR::FLUID::topopt:
//  {
//    convvelint_.Update(velint_);
//    break;
//  }
//  case INPAR::FLUID::oseen:
//  {
//    convvelint_.Multiply(eadvvel_,funct_);
//    break;
//  }
//  case INPAR::FLUID::stokes:
//  {
//    convvelint_.Clear();
//    break;
//  }
//  default:
//    dserror("Physical type not implemented here. For Poro-problems see derived class FluidEleCalcPoro.");
//    break;
//  }
//
//  // (ALE case handled implicitly here using the (potential
//  //  mesh-movement-dependent) convective velocity, avoiding
//  //  various ALE terms used to be calculated before)
//  if (isale)
//  {
//    convvelint_.Update(-1.0,gridvelint_,1.0);
//  }
//}
//
//
///*---------------------------------------------------------------------------*
// |  set element advective field for Oseen problems              schott 11/14 |
// *---------------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::SetAdvectiveVelOseen(DRT::ELEMENTS::Fluid* ele)
//{
//  // set element advective field for Oseen problems
//  if (fldpara_->PhysicalType()==INPAR::FLUID::oseen)
//  {
//    const int funcnum = fldpara_->OseenFieldFuncNo();
//    const double time = fldparatimint_->Time();
//    for ( int jnode=0; jnode<nen_; ++jnode )
//    {
//      const double * jx = ele->Nodes()[jnode]->X();
//      for(int idim=0;idim<nsd_;++idim)
//        eadvvel_(idim,jnode) = DRT::Problem::Instance()->Funct(funcnum-1).Evaluate(idim,jx,time,NULL);
//    }
//  }
//}


///*----------------------------------------------------------------------*
// |  compute material parameters                                vg 09/09 |
// *----------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::GetMaterialParams(
//  Teuchos::RCP<const MAT::Material>  material,
//  const LINALG::Matrix<nsd_,nen_>&   evelaf,
//  const LINALG::Matrix<nen_,1>&      escaaf,
//  const LINALG::Matrix<nen_,1>&      escaam,
//  const LINALG::Matrix<nen_,1>&      escabofoaf,
//  const double                       thermpressaf,
//  const double                       thermpressam,
//  const double                       thermpressdtaf,
//  const double                       thermpressdtam,
//  const double                       vol
//)
//{
//// initially set density values and values with respect to continuity rhs
//densam_        = 1.0;
//densaf_        = 1.0;
//densn_         = 1.0;
//scadtfac_      = 0.0;
//scaconvfacaf_  = 0.0;
//scaconvfacn_   = 0.0;
//thermpressadd_ = 0.0;
//
//if (material->MaterialType() == INPAR::MAT::m_fluid)
//{
//  const MAT::NewtonianFluid* actmat = static_cast<const MAT::NewtonianFluid*>(material.get());
//
//  // get constant dynamic viscosity
//  visc_ = actmat->Viscosity();
//  viscn_ =visc_;
//
//  // artificial compressibility
//  if (fldpara_->PhysicalType() == INPAR::FLUID::artcomp)
//  {
//    // get norm of convective velocity
//    const double vel_norm = convvelint_.Norm2();
//
//    // calculate characteristic element length
//    double h_u = 0.0;
//    double h_p = 0.0;
//    CalcCharEleLength(vol,vel_norm,h_u,h_p);
//
//    // get constant density
//    densaf_ = actmat->Density();
//    densam_ = densaf_;
//    densn_  = densaf_;
//
//    // compute compressibility parameter c as squared value of
//    // maximum of convective velocity, viscous velocity and constant
//    // value 1/2, as proposed in Nithiarasu (2003)
//    double maxvel = std::max(vel_norm,(visc_/(h_p*densaf_)));
//    maxvel = std::max(maxvel,0.5);
//    scadtfac_ = 1.0/std::pow(maxvel,2);
//  }
//  // varying Density
//  else if (fldpara_->PhysicalType() == INPAR::FLUID::varying_density)
//  {
//    const double density_0 = actmat->Density();
//
//    densaf_ = funct_.Dot(escaaf)*density_0;
//    densam_ = densaf_;
//    densn_  = funct_.Dot(escaam)*density_0;
//  }
//  // Boussinesq approximation: Calculation of delta rho
//  else if (fldpara_->PhysicalType() == INPAR::FLUID::boussinesq)
//  {
//    const double density_0 = actmat->Density();
//
//    if(escaaf(0) < EPS12)
//      dserror("Boussinesq approximation: density in escaaf is zero");
//    densaf_ = density_0;
//    densam_ = densaf_;
//    densn_  = densaf_;
//
//    deltadens_ =  (funct_.Dot(escaaf)- 1.0)*density_0;
//    // divison by density_0 was removed here since we keep the density in all
//    // terms of the momentum equation (no divison by rho -> using dynamic viscosity)
//  }
//  // incompressible flow (standard case)
//  else
//  {
//    densaf_ = actmat->Density();
//    densam_ = densaf_;
//    densn_  = densaf_;
//  }
//}
//else if (material->MaterialType() == INPAR::MAT::m_carreauyasuda)
//{
//  const MAT::CarreauYasuda* actmat = static_cast<const MAT::CarreauYasuda*>(material.get());
//
//  densaf_ = actmat->Density();
//  densam_ = densaf_;
//  densn_  = densaf_;
//
//  double nu_0   = actmat->Nu0();    // parameter for zero-shear viscosity
//  double nu_inf = actmat->NuInf();  // parameter for infinite-shear viscosity
//  double lambda = actmat->Lambda();  // parameter for characteristic time
//  double a      = actmat->AParam(); // constant parameter
//  double b      = actmat->BParam(); // constant parameter
//
//  // compute rate of strain at n+alpha_F or n+1
//  double rateofstrain   = -1.0e30;
//  rateofstrain = GetStrainRate(evelaf);
//
//  // compute viscosity according to the Carreau-Yasuda model for shear-thinning fluids
//  // see Dhruv Arora, Computational Hemodynamics: Hemolysis and Viscoelasticity,PhD, 2005
//  const double tmp = std::pow(lambda*rateofstrain,b);
//  // kinematic viscosity
//  visc_ = nu_inf + ((nu_0 - nu_inf)/pow((1 + tmp),a));
//  // dynamic viscosity
//  visc_ *= densaf_;
//}
//else if (material->MaterialType() == INPAR::MAT::m_modpowerlaw)
//{
//  const MAT::ModPowerLaw* actmat = static_cast<const MAT::ModPowerLaw*>(material.get());
//
//  densaf_ = actmat->Density();
//  densam_ = densaf_;
//  densn_  = densaf_;
//
//  // get material parameters
//  double m     = actmat->MCons();     // consistency constant
//  double delta = actmat->Delta();      // safety factor
//  double a     = actmat->AExp();      // exponent
//
//  // compute rate of strain at n+alpha_F or n+1
//  double rateofstrain   = -1.0e30;
//  rateofstrain = GetStrainRate(evelaf);
//
//  // compute viscosity according to a modified power law model for shear-thinning fluids
//  // see Dhruv Arora, Computational Hemodynamics: Hemolysis and Viscoelasticity,PhD, 2005
//  // kinematic viscosity
//  visc_ = m * pow((delta + rateofstrain), (-1)*a);
//  // dynamic viscosity
//  visc_ *= densaf_;
//}
//else if (material->MaterialType() == INPAR::MAT::m_herschelbulkley)
//{
//  const MAT::HerschelBulkley* actmat = static_cast<const MAT::HerschelBulkley*>(material.get());
//
//  densaf_ = actmat->Density();
//  densam_ = densaf_;
//  densn_  = densaf_;
//
//  double tau0           = actmat->Tau0();            // yield stress
//  double kfac           = actmat->KFac();            // constant factor
//  double nexp           = actmat->NExp();            // exponent
//  double mexp           = actmat->MExp();            // exponent
//  double uplimshearrate = actmat->UpLimShearRate();  // upper limit of shear rate
//  double lolimshearrate = actmat->LoLimShearRate();  // lower limit of shear rate
//
//  // compute rate of strain at n+alpha_F or n+1
//  double rateofstrain   = -1.0e30;
//  rateofstrain = GetStrainRate(evelaf);
//
//  // calculate dynamic viscosity according to Herschel-Bulkley model
//  // (within lower and upper limit of shear rate)
//  if (rateofstrain < lolimshearrate)
//    visc_ = tau0*((1.0-exp(-mexp*lolimshearrate))/lolimshearrate) + kfac*pow(lolimshearrate,(nexp-1.0));
//  else if (rateofstrain > uplimshearrate)
//    visc_ = tau0*((1.0-exp(-mexp*uplimshearrate))/uplimshearrate) + kfac*pow(uplimshearrate,(nexp-1.0));
//  else
//    visc_ = tau0*((1.0-exp(-mexp*rateofstrain))/rateofstrain) + kfac*pow(rateofstrain,(nexp-1.0));
//}
//else if (material->MaterialType() == INPAR::MAT::m_yoghurt)
//{
//  const MAT::Yoghurt* actmat = static_cast<const MAT::Yoghurt*>(material.get());
//
//  // get constant density
//  densaf_ = actmat->Density();
//  densam_ = densaf_;
//  densn_  = densaf_;
//
//  // compute temperature at n+alpha_F or n+1
//  const double tempaf = funct_.Dot(escaaf);
//
//  // compute rate of strain at n+alpha_F or n+1
//  double rateofstrain = -1.0e30;
//  rateofstrain = GetStrainRate(evelaf);
//
//  // compute viscosity for Yoghurt-like flows according to Afonso et al. (2003)
//  visc_ = actmat->ComputeViscosity(rateofstrain,tempaf);
//
//  // compute diffusivity
//  diffus_ = actmat->ComputeDiffusivity();
//}
//else if (material->MaterialType() == INPAR::MAT::m_mixfrac)
//{
//  const MAT::MixFrac* actmat = static_cast<const MAT::MixFrac*>(material.get());
//
//  // compute mixture fraction at n+alpha_F or n+1
//  const double mixfracaf = funct_.Dot(escaaf);
//
//  // compute dynamic viscosity at n+alpha_F or n+1 based on mixture fraction
//  visc_ = actmat->ComputeViscosity(mixfracaf);
//
//  // compute dynamic diffusivity at n+alpha_F or n+1 based on mixture fraction
//  diffus_ = actmat->ComputeDiffusivity(mixfracaf);
//
//  // compute density at n+alpha_F or n+1 based on mixture fraction
//  densaf_ = actmat->ComputeDensity(mixfracaf);
//
//  // factor for convective scalar term at n+alpha_F or n+1
//  scaconvfacaf_ = actmat->EosFacA()*densaf_;
//
//  if (fldparatimint_->IsGenalpha())
//  {
//    // compute density at n+alpha_M based on mixture fraction
//    const double mixfracam = funct_.Dot(escaam);
//    densam_ = actmat->ComputeDensity(mixfracam);
//
//    // factor for scalar time derivative at n+alpha_M
//    scadtfac_ = actmat->EosFacA()*densam_;
//  }
//  else
//  {
//    // set density at n+1 at location n+alpha_M as well
//    densam_ = densaf_;
//
//    if (not fldparatimint_->IsStationary())
//    {
//      // compute density at n based on mixture fraction
//      const double mixfracn = funct_.Dot(escaam);
//      densn_ = actmat->ComputeDensity(mixfracn);
//
//      // factor for convective scalar term at n
//      scaconvfacn_ = actmat->EosFacA()*densn_;
//
//      // factor for scalar time derivative
//      scadtfac_ = scaconvfacaf_;
//    }
//  }
//}
//else if (material->MaterialType() == INPAR::MAT::m_sutherland)
//{
//  const MAT::Sutherland* actmat = static_cast<const MAT::Sutherland*>(material.get());
//
//  // compute temperature at n+alpha_F or n+1
//  const double tempaf = funct_.Dot(escaaf);
//
//  // compute viscosity according to Sutherland law
//  visc_ = actmat->ComputeViscosity(tempaf);
//
//  // compute diffusivity according to Sutherland law
//  diffus_ = actmat->ComputeDiffusivity(tempaf);
//
//  // compute density at n+alpha_F or n+1 based on temperature
//  // and thermodynamic pressure
//  densaf_ = actmat->ComputeDensity(tempaf,thermpressaf);
//
//  // factor for convective scalar term at n+alpha_F or n+1
//  scaconvfacaf_ = 1.0/tempaf;
//
//  if (fldparatimint_->IsGenalpha())
//  {
//    // compute temperature at n+alpha_M
//    const double tempam = funct_.Dot(escaam);
//
//    // factor for scalar time derivative at n+alpha_M
//    scadtfac_ = 1.0/tempam;
//
//    // compute density at n+alpha_M based on temperature
//    densam_ = actmat->ComputeDensity(tempam,thermpressam);
//
//    // addition due to thermodynamic pressure at n+alpha_M
//    thermpressadd_ = -thermpressdtam/thermpressam;
//
//    // first part of right-hand side for scalar equation:
//    // time derivative of thermodynamic pressure at n+alpha_F
//    scarhs_ = thermpressdtaf/actmat->Shc();
//  }
//  else
//  {
//    // set density at n+1 at location n+alpha_M as well
//    densam_ = densaf_;
//
//    if (not fldparatimint_->IsStationary())
//    {
//      // compute temperature at n
//      const double tempn = funct_.Dot(escaam);
//
//      // compute density at n based on temperature at n and
//      // (approximately) thermodynamic pressure at n+1
//      densn_ = actmat->ComputeDensity(tempn,thermpressaf);
//
//      // factor for convective scalar term at n
//      scaconvfacn_ = 1.0/tempn;
//
//      // factor for scalar time derivative
//      scadtfac_ = scaconvfacaf_;
//
//      // addition due to thermodynamic pressure
//      thermpressadd_ = -(thermpressaf-thermpressam)/(fldparatimint_->Dt()*thermpressaf);
//
//      // first part of right-hand side for scalar equation:
//      // time derivative of thermodynamic pressure
//      scarhs_ = (thermpressaf-thermpressam)/fldparatimint_->Dt()/actmat->Shc();
//    }
//  }
//
//  // second part of right-hand side for scalar equation: body force
//  // (value at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
//  const double scatrabodyforce = funct_.Dot(escabofoaf);
//  scarhs_ += scatrabodyforce/actmat->Shc();
//}
//else if (material->MaterialType() == INPAR::MAT::m_arrhenius_pv)
//{
//  const MAT::ArrheniusPV* actmat = static_cast<const MAT::ArrheniusPV*>(material.get());
//
//  // get progress variable at n+alpha_F or n+1
//  const double provaraf = funct_.Dot(escaaf);
//
//  // compute temperature based on progress variable at n+alpha_F or n+1
//  const double tempaf = actmat->ComputeTemperature(provaraf);
//
//  // compute viscosity according to Sutherland law
//  visc_ = actmat->ComputeViscosity(tempaf);
//
//  // compute diffusivity according to Sutherland law
//  diffus_ = actmat->ComputeDiffusivity(tempaf);
//
//  // compute density at n+alpha_F or n+1 based on progress variable
//  densaf_ = actmat->ComputeDensity(provaraf);
//
//  // factor for convective scalar term at n+alpha_F or n+1
//  scaconvfacaf_ = actmat->ComputeFactor(provaraf);
//
//  if (fldparatimint_->IsGenalpha())
//  {
//    // compute density at n+alpha_M based on progress variable
//    const double provaram = funct_.Dot(escaam);
//    densam_ = actmat->ComputeDensity(provaram);
//
//    // factor for scalar time derivative at n+alpha_M
//    scadtfac_ = actmat->ComputeFactor(provaram);
//
//    // right-hand side for scalar equation (including reactive term)
//    scarhs_ = densaf_*actmat->ComputeReactionCoeff(tempaf)*(1.0-provaraf);
//  }
//  else
//  {
//    // set density at n+1 at location n+alpha_M as well
//    densam_ = densaf_;
//
//    if (not fldparatimint_->IsStationary())
//    {
//      // compute density at n based on progress variable
//      const double provarn = funct_.Dot(escaam);
//      densn_ = actmat->ComputeDensity(provarn);
//
//      // factor for convective scalar term at n
//      scaconvfacn_ = actmat->ComputeFactor(provarn);
//
//      // factor for scalar time derivative
//      scadtfac_ = scaconvfacaf_;
//
//      // right-hand side for scalar equation (including reactive term)
//      const double tempn = actmat->ComputeTemperature(provarn);
//      scarhs_ = fldparatimint_->Theta()*
//                (densaf_*actmat->ComputeReactionCoeff(tempaf)*(1.0-provaraf))
//               +fldparatimint_->OmTheta()*
//                (densn_*actmat->ComputeReactionCoeff(tempn)*(1.0-provarn));
//    }
//  }
//}
//else if (material->MaterialType() == INPAR::MAT::m_ferech_pv)
//{
//  const MAT::FerEchPV* actmat = static_cast<const MAT::FerEchPV*>(material.get());
//
//  // get progress variable at n+alpha_F or n+1
//  const double provaraf = funct_.Dot(escaaf);
//
//  // compute temperature based on progress variable at n+alpha_F or n+1
//  const double tempaf = actmat->ComputeTemperature(provaraf);
//
//  // compute viscosity according to Sutherland law
//  visc_ = actmat->ComputeViscosity(tempaf);
//
//  // compute diffusivity according to Sutherland law
//  diffus_ = actmat->ComputeDiffusivity(tempaf);
//
//  // compute density at n+alpha_F or n+1 based on progress variable
//  densaf_ = actmat->ComputeDensity(provaraf);
//
//  // factor for convective scalar term at n+alpha_F or n+1
//  scaconvfacaf_ = actmat->ComputeFactor(provaraf);
//
//  if (fldparatimint_->IsGenalpha())
//  {
//    // compute density at n+alpha_M based on progress variable
//    const double provaram = funct_.Dot(escaam);
//    densam_ = actmat->ComputeDensity(provaram);
//
//    // factor for scalar time derivative at n+alpha_M
//    scadtfac_ = actmat->ComputeFactor(provaram);
//
//    // right-hand side for scalar equation (including reactive term)
//    scarhs_ = densaf_*actmat->ComputeReactionCoeff(tempaf)*(1.0-provaraf);
//  }
//  else
//  {
//    // set density at n+1 at location n+alpha_M as well
//    densam_ = densaf_;
//
//    if (not fldparatimint_->IsStationary())
//    {
//      // compute density at n based on progress variable
//      const double provarn = funct_.Dot(escaam);
//      densn_ = actmat->ComputeDensity(provarn);
//
//      // factor for convective scalar term at n
//      scaconvfacn_ = actmat->ComputeFactor(provarn);
//
//      // factor for scalar time derivative
//      scadtfac_ = scaconvfacaf_;
//
//      // right-hand side for scalar equation (including reactive term)
//      const double tempn = actmat->ComputeTemperature(provarn);
//      scarhs_ = fldparatimint_->Theta()*
//                (densaf_*actmat->ComputeReactionCoeff(tempaf)*(1.0-provaraf))
//               +fldparatimint_->OmTheta()*
//                (densn_*actmat->ComputeReactionCoeff(tempn)*(1.0-provarn));
//    }
//  }
//}
//else if (material->MaterialType() == INPAR::MAT::m_permeable_fluid)
//{
//  const MAT::PermeableFluid* actmat = static_cast<const MAT::PermeableFluid*>(material.get());
//
//  densaf_ = actmat->Density();
//  densam_ = densaf_;
//  densn_  = densaf_;
//
//  // calculate reaction coefficient
//  reacoeff_ = actmat->ComputeReactionCoeff();
//
//  // get constant viscosity (zero for Darcy and greater than zero for Darcy-Stokes)
//  visc_ = actmat->SetViscosity();
//
//  // set darcy flag to true
//  //f3Parameter_->darcy_ = true;
//  // set reaction flag to true
//  //f3Parameter_->reaction_ = true;
//
//  // check stabilization parameter definition for permeable fluid
//  if (not (fldpara_->WhichTau() == INPAR::FLUID::tau_franca_madureira_valentin_badia_codina or
//           fldpara_->WhichTau() == INPAR::FLUID::tau_franca_madureira_valentin_badia_codina_wo_dt))
//    dserror("incorrect definition of stabilization parameter for Darcy or Darcy-Stokes problem");
//}
//else if (material->MaterialType() == INPAR::MAT::m_matlist)
//{
//  // get material list for this element
//  const MAT::MatList* matlist = static_cast<const MAT::MatList*>(material.get());
//
//  int numofmaterials = matlist->NumMat();
//
//  //Error messages
//  if(numofmaterials>2)
//  {
//    dserror("More than two materials is currently not supported.");
//  }
//  if(not fldpara_->MatGp())
//  {
//    dserror("For Two Phase Flow, the material should be evaluated at the Gauss-points. Switch EVALUATION_MAT to integration_point.");
//  }
//
//  std::vector<double> density(numofmaterials); //Assume density[0] is on positive side, and density[1] is on negative side.
//  std::vector<double> viscosity(numofmaterials);
//  std::vector<double> gamma(numofmaterials);
//
//  for(int nmaterial=0; nmaterial<numofmaterials; nmaterial++)
//  {
//    // set default id in list of materials
//      int matid = -1;
//      matid = matlist->MatID(nmaterial);
//
//      Teuchos::RCP<const MAT::Material> matptr = matlist->MaterialById(matid);
//      INPAR::MAT::MaterialType mattype = matptr->MaterialType();
//
//      // choose from different materials
//      switch(mattype)
//      {
//      //--------------------------------------------------------
//      // Newtonian fluid for incompressible flow (standard case)
//      //--------------------------------------------------------
//      case INPAR::MAT::m_fluid:
//      {
//        const MAT::NewtonianFluid* mat = static_cast<const MAT::NewtonianFluid*>(matptr.get());
//        density[nmaterial]=mat->Density();
//        viscosity[nmaterial]=mat->Viscosity();
//        gamma[nmaterial]=mat->Gamma();
//        break;
//      }
//      //------------------------------------------------
//      // different types of materials (to be added here)
//      //------------------------------------------------
//      default:
//        dserror("Only Newtonian fluids supported as input.");
//        break;
//      }
//  }
//
//  double epsilon = fldpara_->GetInterfaceThickness();
//
//  const double gpscaaf = funct_.Dot(escaaf); //Scalar function at gausspoint evaluated
//  const double gpscaam = funct_.Dot(escaam); //Scalar function at gausspoint evaluated
//
//  //Assign material parameter values to positive side by default.
//  double heavyside_epsilon=1.0;
//  densaf_=density[0];
//  visc_=viscosity[0];
//  viscn_=visc_;
//  densam_=densaf_;
//  densn_=densam_;
//
//
//  //Calculate material parameters with phiaf
//  if(abs(gpscaaf) <= epsilon)
//  {
//    heavyside_epsilon = 0.5 * (1.0 + gpscaaf/epsilon + 1.0 / PI * sin(PI*gpscaaf/epsilon));
//
//    densaf_ = heavyside_epsilon*density[0]+(1.0-heavyside_epsilon)*density[1];
//    visc_ = heavyside_epsilon*viscosity[0]+(1.0-heavyside_epsilon)*viscosity[1];
//  }
//  else if (gpscaaf < epsilon)
//  {
//    heavyside_epsilon = 0.0;
//
//    densaf_=density[1];
//    visc_=viscosity[1];
//  }
//
////  //Calculate material parameters with phiam
//  if(abs(gpscaam) <= epsilon)
//  {
//    heavyside_epsilon = 0.5 * (1.0 + gpscaam/epsilon + 1.0 / PI * sin(PI*gpscaam/epsilon));
//
//    densam_ = heavyside_epsilon*density[0]+(1.0-heavyside_epsilon)*density[1];
//    densn_ = densam_;
//
//    viscn_ = heavyside_epsilon*viscosity[0]+(1.0-heavyside_epsilon)*viscosity[1];
//  }
//  else if (gpscaam < epsilon)
//  {
//    heavyside_epsilon = 0.0;
//
//    densam_=density[1];
//    densn_=densam_;
//
//    viscn_=viscosity[1];
//  }
//
//
//  if(gamma[0]!=gamma[1])
//    dserror("Surface tensions have to be equal");
//
//  //Surface tension coefficient assigned.
//  gamma_=gamma[0];
//
//}// end else if m_matlist
//else dserror("Material type is not supported");
//
//// check whether there is zero or negative (physical) viscosity
//// (expect for permeable fluid)
//if (visc_ < EPS15 and not (material->MaterialType() == INPAR::MAT::m_permeable_fluid))
//  dserror("zero or negative (physical) diffusivity");
//
//return;
//} // FluidEleCalc::GetMaterialParams


///*----------------------------------------------------------------------*
// |  Get mk                                                     bk 09/14 |
// *----------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//double DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::GetMK()
//{
//  return DRT::ELEMENTS::MK<distype>();
//}

///*----------------------------------------------------------------------*
// |  calculation of stabilization parameter                     vg 09/09 |
// *----------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::CalcStabParameter(const double vol)
//{
//  //---------------------------------------------------------------------
//  // preliminary definition of values which will already be computed for
//  // tau_M and later be used for tau_C again by some of the subsequent
//  // stabilization parameter definitions
//  //---------------------------------------------------------------------
//  double traceG = 0.0;
//  double Gnormu = 0.0;
//  double Gvisc  = 0.0;
//
//  double h_u      = 0.0;
//  double h_p      = 0.0;
//  double vel_norm = 0.0;
//  double re12     = 0.0;
//  double c3       = 0.0;
//
//  //---------------------------------------------------------------------
//  // first step: computation of tau_M with the following options
//  // (both with or without inclusion of dt-part):
//  // A) definition according to Taylor et al. (1998)
//  //    -> see also Gravemeier and Wall (2010) for version for
//  //       variable-density flow at low Mach number
//  // B) combined definition according to Franca and Valentin (2000) as
//  //    well as Barrenechea and Valentin (2002)
//  //    -> differentiating tau_Mu and tau_Mp for this definition
//  // C) definition according to Shakib (1989) / Shakib and Hughes (1991)
//  //    -> differentiating tau_Mu and tau_Mp for this definition
//  // D) definition according to Codina (1998)
//  //    -> differentiating tau_Mu and tau_Mp for this definition
//  // E) definition according to Franca et al. (2005) as well as Badia
//  //    and Codina (2010)
//  //    -> only for Darcy or Darcy-Stokes/Brinkman flow, hence only
//  //       tau_Mp for this definition
//  //---------------------------------------------------------------------
//  // get element-type constant for tau
//  const double mk = GetMK();
//
//
//  if (fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic) // quasistatic case
//  {
//  // computation depending on which parameter definition is used
//  switch (fldpara_->WhichTau())
//  {
//  case INPAR::FLUID::tau_taylor_hughes_zarins:
//  case INPAR::FLUID::tau_taylor_hughes_zarins_wo_dt:
//  case INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen:
//  case INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen_wo_dt:
//  case INPAR::FLUID::tau_taylor_hughes_zarins_scaled:
//  case INPAR::FLUID::tau_taylor_hughes_zarins_scaled_wo_dt:
//  {
//    /*
//
//    literature:
//    1) C.A. Taylor, T.J.R. Hughes, C.K. Zarins, Finite element modeling
//       of blood flow in arteries, Comput. Methods Appl. Mech. Engrg. 158
//       (1998) 155-196.
//    2) V. Gravemeier, W.A. Wall, An algebraic variational multiscale-
//       multigrid method for large-eddy simulation of turbulent variable-
//       density flow at low Mach number, J. Comput. Phys. 229 (2010)
//       6047-6070.
//       -> version for variable-density low-Mach-number flow as implemented
//          here, which corresponds to version for incompressible flow as
//          given in the previous publications when density is constant
//
//                                                                           1
//                     +-                                               -+ - -
//                     |        2                                        |   2
//                     | c_1*rho                                  2      |
//          tau  = C * | -------   +  c_2*rho*u*G*rho*u  +  c_3*mu *G:G  |
//             M       |     2                                           |
//                     |   dt                                            |
//                     +-                                               -+
//
//          with the constants and covariant metric tensor defined as follows:
//
//          C   = 1.0 (not explicitly defined here),
//          c_1 = 4.0 (for version with dt), 0.0 (for version without dt),
//          c_2 = 1.0 (not explicitly defined here),
//          c_3 = 12.0/m_k (36.0 for linear and 144.0 for quadratic elements)
//
//                  +-           -+   +-           -+   +-           -+
//                  |             |   |             |   |             |
//                  |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
//            G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
//             ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
//                  |    i     j  |   |    i     j  |   |    i     j  |
//                  +-           -+   +-           -+   +-           -+
//
//                  +----
//                   \
//          G : G =   +   G   * G
//                   /     ij    ij
//                  +----
//                   i,j
//                             +----
//                             \
//          rho*u*G*rho*u  =   +   rho*u * G  *rho*u
//                             /        i   ij      j
//                            +----
//                              i,j
//    */
//
//    // total reaction coefficient sigma_tot: sum of "artificial" reaction
//    // due to time factor and reaction coefficient (reaction coefficient
//    // ensured to remain zero in GetMaterialParams for non-reactive material)
//    double sigma_tot = reacoeff_;
//    if (fldpara_->WhichTau() == INPAR::FLUID::tau_taylor_hughes_zarins or
//        fldpara_->WhichTau() == INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen or
//        fldpara_->WhichTau() == INPAR::FLUID::tau_taylor_hughes_zarins_scaled)
//      sigma_tot += 1.0/fldparatimint_->Dt();
//
//    // definition of constants as described above
//    const double c1 = 4.0;
//    c3 = 12.0/mk;
//
//    // computation of various values derived from covariant metric tensor
//    // (trace of covariant metric tensor required for computation of tau_C below)
//    double G;
//    double normG = 0.0;
//    const double dens_sqr = densaf_*densaf_;
//    for (int nn=0;nn<nsd_;++nn)
//    {
//      const double dens_sqr_velint_nn = dens_sqr*convvelint_(nn);
//      for (int mm=0; mm<nsd_; ++mm)
//      {
//        traceG += xji_(nn,mm)*xji_(nn,mm);
//      }
//      for (int rr=0;rr<nsd_;++rr)
//      {
//        G = xji_(nn,0)*xji_(rr,0);
//        for (int mm=1; mm<nsd_; ++mm)
//        {
//          G += xji_(nn,mm)*xji_(rr,mm);
//        }
//        normG  += G*G;
//        Gnormu += dens_sqr_velint_nn*G*convvelint_(rr);
//      }
//    }
//
//    // compute viscous part
//    Gvisc = c3*visceff_*visceff_*normG;
//
//    // compute stabilization parameter tau_Mu
//    tau_(0) = 1.0/(sqrt(c1*dens_sqr*DSQR(sigma_tot) + Gnormu + Gvisc));
//
//    // compute stabilization parameter tau_Mp
//    // ensure that tau_Mp does not become too small for viscosity-dominated flow
//    // lower limit proportional to squared (Braack et al. (2007)) or cubic
//    // Barth et al. (2004) characteristic length
//    // here: lower-limit constant chosen to be 1.0 and cubic char. length
//    const double llc = 1.0;
//    const double powerfac = 3.0;
//    if ((Gnormu < Gvisc) and (std::pow(traceG,(powerfac/2.0)) < llc*sqrt(Gvisc)))
//         tau_(1) = 1.0/(sqrt(c1*dens_sqr*DSQR(sigma_tot) + Gnormu + (std::pow(traceG,powerfac)/DSQR(llc))));
//    else tau_(1) = tau_(0);
//  }
//  break;
//
//  case INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall:
//  {
//    /*
//
//    literature:
//    1) L.P. Franca, F. Valentin, On an improved unusual stabilized
//       finite element method for the advective-reactive-diffusive
//       equation, Comput. Methods Appl. Mech. Engrg. 190 (2000) 1785-1800.
//    2) G.R. Barrenechea, F. Valentin, An unusual stabilized finite
//       element method for a generalized Stokes problem, Numer. Math.
//       92 (2002) 652-677.
//
//
//                  xi1,xi2 ^
//                          |      /
//                          |     /
//                          |    /
//                        1 +---+
//                          |
//                          |
//                          |
//                          +--------------> re1,re2
//                              1
//
//    */
//    // get norm of convective velocity
//    vel_norm = convvelint_.Norm2();
//
//    // total reaction coefficient sigma_tot: sum of "artificial" reaction
//    // due to time factor and reaction coefficient (reaction coefficient
//    // ensured to remain zero in GetMaterialParams for non-reactive material)
//    const double sigma_tot = 1.0/fldparatimint_->TimeFac() + reacoeff_;
//    // NOTE: Gen_Alpha (implementation by Peter Gamnitzer) used a different time factor!
//
//    // calculate characteristic element length
//    CalcCharEleLength(vol,vel_norm,h_u,h_p);
//
//    // various parameter computations for case with dt:
//    // relating viscous to reactive part (re01: tau_Mu, re11: tau_Mp)
//    const double re01 = 4.0 * visceff_ / (mk * densaf_ * sigma_tot * DSQR(h_u));
//    const double re11 = 4.0 * visceff_ / (mk * densaf_ * sigma_tot * DSQR(h_p));
//
//    // relating convective to viscous part (re02: tau_Mu, re12: tau_Mp)
//    const double re02 = mk * densaf_ * vel_norm * h_u / (2.0 * visceff_);
//                 re12 = mk * densaf_ * vel_norm * h_p / (2.0 * visceff_);
//
//    // respective "switching" parameters
//    const double xi01 = std::max(re01,1.0);
//    const double xi11 = std::max(re11,1.0);
//    const double xi02 = std::max(re02,1.0);
//    const double xi12 = std::max(re12,1.0);
//
//    // compute stabilization parameter tau_Mu
//    tau_(0) = DSQR(h_u)/(DSQR(h_u)*densaf_*sigma_tot*xi01+(4.0*visceff_/mk)*xi02);
//
//    // compute stabilization parameter tau_Mp
//    // ensure that tau_Mp does not become too small for viscosity-dominated flow
//    // lower limit proportional to squared (Braack et al. (2007)) or cubic
//    // Barth et al. (2004) characteristic length
//    // here: lower-limit constant chosen to be 1.0 and cubic char. length
//    const double llc = 1.0;
//    const double powerfac = 3.0;
//    if ((re12 < 1.0) and (llc*std::pow(h_p,powerfac) > DSQR(h_p)/(4.0*visceff_/mk)))
//    {
//       if (re11 < 1.0)
//         tau_(1) = 1.0/(densaf_*sigma_tot+(1.0/(llc*std::pow(h_p,powerfac))));
//       else
//         tau_(1) = llc*std::pow(h_p,powerfac);
//    }
//    else tau_(1) = DSQR(h_p)/(DSQR(h_p)*densaf_*sigma_tot*xi11+(4.0*visceff_/mk)*xi12);
//  }
//  break;
//
//  case INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall_wo_dt:
//  {
//    /*
//
//     stabilization parameter as above without inclusion of dt-part
//
//    */
//    // get norm of convective velocity
//    vel_norm = convvelint_.Norm2();
//
//    // calculate characteristic element length
//    CalcCharEleLength(vol,vel_norm,h_u,h_p);
//
//    // various parameter computations for case without dt:
//    // relating viscous to reactive part (re01: tau_Mu, re11: tau_Mp)
//    double re01 = 0.0;
//    double re11 = 0.0;
//    if (fldpara_->Reaction()) // TODO Martin: check influence of reaction to stabilization
//    {
//      re01 = 4.0 * visceff_ / (mk * densaf_ * reacoeff_ * DSQR(h_u));
//      re11 = 4.0 * visceff_ / (mk * densaf_ * reacoeff_ * DSQR(h_p));
//    }
//    // relating convective to viscous part (re02: tau_Mu, re12: tau_Mp)
//    const double re02 = mk * densaf_ * vel_norm * h_u / (2.0 * visceff_);
//                 re12 = mk * densaf_ * vel_norm * h_p / (2.0 * visceff_);
//
//    // respective "switching" parameters
//    const double xi01 = std::max(re01,1.0);
//    const double xi11 = std::max(re11,1.0);
//    const double xi02 = std::max(re02,1.0);
//    const double xi12 = std::max(re12,1.0);
//
//    // compute stabilization parameter tau_Mu
//    tau_(0) = DSQR(h_u)/(DSQR(h_u)*densaf_*reacoeff_*xi01+(4.0*visceff_/mk)*xi02);
//
//    // compute stabilization parameter tau_Mp
//    // ensure that tau_Mp does not become too small for viscosity-dominated flow
//    // lower limit proportional to squared (Braack et al. (2007)) or cubic
//    // Barth et al. (2004) characteristic length
//    // here: lower-limit constant chosen to be 1.0 and cubic char. length
//    const double llc = 1.0;
//    const double powerfac = 3.0;
//    if ((re12 < 1.0) and (llc*std::pow(h_p,powerfac) > DSQR(h_p)/(4.0*visceff_/mk)))
//    {
//       if (re11 < 1.0)
//         tau_(1) = 1.0/(densaf_*reacoeff_+(1.0/(llc*std::pow(h_p,powerfac))));
//       else
//         tau_(1) = llc*std::pow(h_p,powerfac);
//    }
//    else tau_(1) = DSQR(h_p)/(DSQR(h_p)*densaf_*reacoeff_*xi11+(4.0*visceff_/mk)*xi12);
//  }
//  break;
//
//  case INPAR::FLUID::tau_shakib_hughes_codina:
//  case INPAR::FLUID::tau_shakib_hughes_codina_wo_dt:
//  {
//    /*
//
//    literature:
//    1) F. Shakib, Finite element analysis of the compressible Euler and
//       Navier-Stokes equations, PhD thesis, Division of Applied Mechanics,
//       Stanford University, Stanford, CA, USA, 1989.
//    2) F. Shakib, T.J.R. Hughes, A new finite element formulation for
//       computational fluid dynamics: IX. Fourier analysis of space-time
//       Galerkin/least-squares algorithms, Comput. Methods Appl. Mech.
//       Engrg. 87 (1991) 35-58.
//    3) R. Codina, Stabilized finite element approximation of transient
//       incompressible flows using orthogonal subscales, Comput. Methods
//       Appl. Mech. Engrg. 191 (2002) 4295-4321.
//
//       constants defined as in Shakib (1989) / Shakib and Hughes (1991),
//       merely slightly different with respect to c_3:
//
//       c_1 = 4.0 (for version with dt), 0.0 (for version without dt),
//       c_2 = 4.0,
//       c_3 = 4.0/(m_k*m_k) (36.0 for linear, 576.0 for quadratic ele.)
//
//       Codina (2002) proposed present version without dt and explicit
//       definition of constants
//       (condition for constants as defined here: c_2 <= sqrt(c_3)).
//
//    */
//    // get norm of convective velocity
//    vel_norm = convvelint_.Norm2();
//
//    // calculate characteristic element length
//    CalcCharEleLength(vol,vel_norm,h_u,h_p);
//
//    // total reaction coefficient sigma_tot: sum of "artificial" reaction
//    // due to time factor and reaction coefficient (reaction coefficient
//    // ensured to remain zero in GetMaterialParams for non-reactive material)
//    double sigma_tot = reacoeff_;
//    if (fldpara_->WhichTau() == INPAR::FLUID::tau_shakib_hughes_codina)
//      sigma_tot += 1.0/fldparatimint_->Dt();
//
//    // definition of constants as described above
//    const double c1 = 4.0;
//    const double c2 = 4.0;
//    c3 = 4.0/(mk*mk);
//    // alternative value as proposed in Shakib (1989): c3 = 16.0/(mk*mk);
//
//    // compute stabilization parameter tau_Mu
//    tau_(0) = 1.0/(sqrt(c1*DSQR(densaf_)*DSQR(sigma_tot)
//                      + c2*DSQR(densaf_)*DSQR(vel_norm)/DSQR(h_u)
//                      + c3*DSQR(visceff_)/(DSQR(h_u)*DSQR(h_u))));
//
//    // compute stabilization parameter tau_Mp
//    // ensure that tau_Mp does not become too small for viscosity-dominated flow
//    // lower limit proportional to squared (Braack et al. (2007)) or cubic
//    // Barth et al. (2004) characteristic length
//    // here: lower-limit constant chosen to be 1.0 and cubic char. length
//    const double llc = 1.0;
//    const double powerfac = 3.0;
//    const double re12 = mk * densaf_ * vel_norm * h_p / (2.0 * visceff_);
//    if ((re12 < 1.0) and (llc*std::pow(h_p,powerfac) > DSQR(h_p)/(sqrt(c3)*visceff_)))
//         tau_(1) = 1.0/(sqrt(c1*DSQR(densaf_)*DSQR(sigma_tot)
//                      + c2*DSQR(densaf_)*DSQR(vel_norm)/DSQR(h_p)
//                      + 1.0/DSQR(llc*std::pow(h_p,powerfac))));
//    else tau_(1) = 1.0/(sqrt(c1*DSQR(densaf_)*DSQR(sigma_tot)
//                      + c2*DSQR(densaf_)*DSQR(vel_norm)/DSQR(h_p)
//                      + c3*DSQR(visceff_)/(DSQR(h_p)*DSQR(h_p))));
//  }
//  break;
//
//  case INPAR::FLUID::tau_codina:
//  case INPAR::FLUID::tau_codina_wo_dt:
//  case INPAR::FLUID::tau_codina_convscaled:
//  {
//    /*
//
//      literature:
//         R. Codina, Comparison of some finite element methods for solving
//         the diffusion-convection-reaction equation, Comput. Methods
//         Appl. Mech. Engrg. 156 (1998) 185-210.
//
//         constants:
//         c_1 = 1.0 (for version with dt), 0.0 (for version without dt),
//         c_2 = 2.0,
//         c_3 = 4.0/m_k (12.0 for linear, 48.0 for quadratic elements)
//
//         Codina (1998) proposed present version without dt.
//
//    */
//    // get norm of convective velocity
//    vel_norm = convvelint_.Norm2();
//
//    // calculate characteristic element length
//    CalcCharEleLength(vol,vel_norm,h_u,h_p);
//
//    // total reaction coefficient sigma_tot: sum of "artificial" reaction
//    // due to time factor and reaction coefficient (reaction coefficient
//    // ensured to remain zero in GetMaterialParams for non-reactive material)
//    double sigma_tot = reacoeff_;
//    if (fldpara_->WhichTau() == INPAR::FLUID::tau_codina||fldpara_->WhichTau() == INPAR::FLUID::tau_codina_convscaled)
//      sigma_tot += 1.0/fldparatimint_->Dt();
//
//    // definition of constants as described above
//    const double c1 = 1.0;
//    double c2 = 2.0;
//    c3 = 4.0/mk;
//
//    // for high-order elements or non-polynomial enrichments,
//    // a scaling of the convective term like this gives good results
//    if(INPAR::FLUID::tau_codina_convscaled)
//      c2 = sqrt(c3/3.0);
//
//    // compute stabilization parameter tau_Mu
//    tau_(0) = 1.0/(c1*densaf_*sigma_tot
//                 + c2*densaf_*vel_norm/h_u
//                 + c3*visceff_/DSQR(h_u));
//
//    // compute stabilization parameter tau_Mp
//    // ensure that tau_Mp does not become too small for viscosity-dominated flow
//    // lower limit proportional to squared (Braack et al. (2007)) or cubic
//    // Barth et al. (2004) characteristic length
//    // here: lower-limit constant chosen to be 1.0 and cubic char. length
//    const double llc = 1.0;
//    const double powerfac = 3.0;
//    const double re12 = mk * densaf_ * vel_norm * h_p / (2.0 * visceff_);
//    if ((re12 < 1.0) and (llc*std::pow(h_p,powerfac) > DSQR(h_p)/(c3*visceff_)))
//         tau_(1) = 1.0/(c1*densaf_*sigma_tot
//                      + c2*densaf_*vel_norm/h_p
//                      + 1.0/(llc*std::pow(h_p,powerfac)));
//    else tau_(1) = 1.0/(c1*densaf_*sigma_tot
//                      + c2*densaf_*vel_norm/h_p
//                      + c3*visceff_/DSQR(h_p));
//  }
//  break;
//
//  case INPAR::FLUID::tau_franca_madureira_valentin_badia_codina:
//  case INPAR::FLUID::tau_franca_madureira_valentin_badia_codina_wo_dt:
//  {
//    /*
//
//    This stabilization parameter is only intended to be used for
//    (viscous-)reactive problems such as Darcy(-Stokes/Brinkman) problems.
//
//    literature:
//    1) L.P. Franca, A.L. Madureira, F. Valentin, Towards multiscale
//       functions: enriching finite element spaces with local but not
//       bubble-like functions, Comput. Methods Appl. Mech. Engrg. 194
//       (2005) 3006-3021.
//    2) S. Badia, R. Codina, Stabilized continuous and discontinuous
//       Galerkin techniques for Darcy flow, Comput. Methods Appl.
//       Mech. Engrg. 199 (2010) 1654-1667.
//
//    */
//    // total reaction coefficient sigma_tot: sum of "artificial" reaction
//    // due to time factor and reaction coefficient
//    double sigma_tot = reacoeff_;
//    if (fldpara_->WhichTau() == INPAR::FLUID::tau_franca_madureira_valentin_badia_codina)
//      sigma_tot += 1.0/fldparatimint_->TimeFac();
//
//    // calculate characteristic element length
//    CalcCharEleLength(vol,0.0,h_u,h_p);
//
//    // various parameter computations for case with dt:
//    // relating viscous to reactive part
//    const double re11 = 2.0 * visceff_ / (mk * densaf_ * sigma_tot * DSQR(h_p));
//
//    // respective "switching" parameter
//    const double xi11 = std::max(re11,1.0);
//
//    // constant c_u as suggested in Badia and Codina (2010), method A
//    // (set to be 4.0 in Badia and Codina (2010), 1.0 in Franca et al. (2005))
//    const double c_u = 4.0;
//
//    // compute stabilization parameter tau_Mp (tau_Mu not required)
//    tau_(0) = 0.0;
//    tau_(1) = DSQR(h_p)/(c_u*DSQR(h_p)*densaf_*sigma_tot*xi11+(2.0*visceff_/mk));
//  }
//  break;
//
//  case INPAR::FLUID::tau_hughes_franca_balestra_wo_dt:
//  {
//    /*----------------------------------------------------------------------*/
//    /*
//     *  This stabilization parameter is only intended to be used for
//     *  stationary Stokes problems.
//     *
//     *  literature:
//     *  1) T.J.R. Hughes, L.P. Franca, and M. Balestra, A new finite element
//     *     formulation for computational fluid dynamics: V. circumventing the
//     *     Babuska-Brezzi condition: a stable Petrov-Galerkin formulation of
//     *     the Stokes problem accomodating equal-order interpolations,
//     *     Comput. Methods Appl. Mech. Engrg. 59 (1986) 85-99.
//     *
//     *  2) J. Donea and A. Huerta, Finite element methods for flow problems.
//     *     (for alpha_0 = 1/3)
//     */
//    /*----------------------------------------------------------------------*/
//
//    // calculate characteristic element length
//    CalcCharEleLength(vol,0.0,h_u,h_p);
//
//    // compute stabilization parameter tau_Mp (tau_Mu not required)
//    tau_(0) = 0.0;
//    tau_(1) = (1.0/3.0) * std::pow(h_p,2.0) / (4.0 * visceff_);
//
//  }
//  break;
//
//  default:
//  {
//    if (not (fldpara_->StabType() == INPAR::FLUID::stabtype_edgebased and
//        fldpara_->WhichTau() == INPAR::FLUID::tau_not_defined))
//      dserror("unknown definition for tau_M\n %i  ", fldpara_->WhichTau());
//
//    break;
//  }
//  }  // end switch (fldpara_->WhichTau())
//
//
//  //---------------------------------------------------------------------
//  // second step: computation of tau_C with the following options:
//  // A) definition according to Taylor et al. (1998)
//  // B) definition according to Whiting (1999)/Whiting and Jansen (2001)
//  // C) scaled version of definition according to Taylor et al. (1998)
//  // D) definition according to Wall (1999)
//  // E) definition according to Codina (2002)
//  // F) definition according to Badia and Codina (2010)
//  //    (only for Darcy or Darcy-Stokes/Brinkman flow)
//  //---------------------------------------------------------------------
//  // computation depending on which parameter definition is used
//  switch (fldpara_->WhichTau())
//  {
//  case INPAR::FLUID::tau_taylor_hughes_zarins:
//  case INPAR::FLUID::tau_taylor_hughes_zarins_wo_dt:
//  {
//    /*
//
//    literature:
//       C.A. Taylor, T.J.R. Hughes, C.K. Zarins, Finite element modeling
//       of blood flow in arteries, Comput. Methods Appl. Mech. Engrg. 158
//       (1998) 155-196.
//
//                                              1/2
//                           (c_2*rho*u*G*rho*u)
//                    tau  = -------------------
//                       C       trace (G)
//
//
//       -> see respective definitions for computation of tau_M above
//
//    */
//
//    tau_(2) = sqrt(Gnormu)/traceG;
//  }
//  break;
//
//  case INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen:
//  case INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen_wo_dt:
//  {
//    /*
//
//    literature:
//    1) C.H. Whiting, Stabilized finite element methods for fluid dynamics
//       using a hierarchical basis, PhD thesis, Rensselaer Polytechnic
//       Institute, Troy, NY, USA, 1999.
//    2) C.H. Whiting, K.E. Jansen, A stabilized finite element method for
//       the incompressible Navier-Stokes equations using a hierarchical
//       basis, Int. J. Numer. Meth. Fluids 35 (2001) 93-116.
//
//                                  1.0
//                    tau  = ------------------
//                       C    tau  * trace (G)
//                               M
//
//       -> see respective definitions for computation of tau_M above
//
//    */
//
//    tau_(2) = 1.0/(tau_(0)*traceG);
//  }
//  break;
//
//  case INPAR::FLUID::tau_taylor_hughes_zarins_scaled:
//  case INPAR::FLUID::tau_taylor_hughes_zarins_scaled_wo_dt:
//  {
//    /*
//
//      Caution: This is an experimental version of a stabilization
//               parameter definition which scales the definition
//               for tau_C by Taylor et al. (1998) in a similar
//               way as proposed below by Franca and Frey (1992)
//               and Wall (1999) by appropriately defining an
//               element Reynolds number based on the covariant
//               metric tensor.
//
//                  /                        1/2    \
//                  |  /                    \       |                       1/2
//                  | |  c_2*rho*u*G*rho*u  |       |    (c_2*rho*u*G*rho*u)
//      tau  =  MIN | | ------------------- | | 1.0 | *  -------------------
//         C        | |          2          |       |         trace (G)
//                  | \    c_3*mu *G:G      /       |
//                  \                               /
//                    |                     |
//                    -----------------------
//                    element Reynolds number
//                      based on covariant
//                        metric tensor
//
//       -> see respective definitions for computation of tau_M above
//
//    */
//
//    // element Reynolds number based on covariant metric tensor
//    const double reG = sqrt(Gnormu/Gvisc);
//
//    // "switching" parameter
//    const double xi_tau_c = std::min(reG,1.0);
//
//    tau_(2) = xi_tau_c*sqrt(Gnormu)/traceG;
//  }
//  break;
//
//  case INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall:
//  case INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall_wo_dt:
//  {
//    /*
//
//    literature:
//    1) L.P. Franca, S.L. Frey, Stabilized finite element methods:
//       II. The incompressible Navier-Stokes equations, Comput. Methods
//       Appl. Mech. Engrg. 99 (1992) 209-293.
//    2) W.A. Wall, Fluid-Struktur-Interaktion mit stabilisierten Finiten
//       Elementen, Dissertation, Universitaet Stuttgart, 1999.
//
//                 xi_tau_c ^
//                          |
//                        1 |   +-----------
//                          |  /
//                          | /
//                          |/
//                          +--------------> re12
//                              1
//
//       -> see respective definitions for computation of tau_M above
//
//    */
//
//    // "switching" parameter
//    const double xi_tau_c = std::min(re12,1.0);
//
//    tau_(2) = 0.5 * densaf_ * vel_norm * h_p * xi_tau_c;
//  }
//  break;
//
//  case INPAR::FLUID::tau_shakib_hughes_codina:
//  case INPAR::FLUID::tau_shakib_hughes_codina_wo_dt:
//  {
//    /*
//
//    literature:
//       R. Codina, Stabilized finite element approximations of transient
//       incompressible flows using orthogonal subscales, Comput. Methods
//       Appl. Mech. Engrg. 191 (2002) 4295-4321.
//
//       -> see respective definitions for computation of tau_M above
//
//    */
//
//    tau_(2) = DSQR(h_p)/(sqrt(c3)*tau_(1));
//  }
//  break;
//
//  case INPAR::FLUID::tau_codina:
//  case INPAR::FLUID::tau_codina_wo_dt:
//  case INPAR::FLUID::tau_codina_convscaled:
//  {
//    /*
//
//    literature:
//       R. Codina, Stabilized finite element approximations of transient
//       incompressible flows using orthogonal subscales, Comput. Methods
//       Appl. Mech. Engrg. 191 (2002) 4295-4321.
//
//       -> see respective definitions for computation of tau_M above
//
//       fixed bug: before we used sqrt(c3) instead of c3 here   bk 2014
//       see for example Diss Peter Gamnitzer for correct definition
//    */
//
//    tau_(2) = DSQR(h_p)/(c3*tau_(1));
//  }
//  break;
//
//  case INPAR::FLUID::tau_franca_madureira_valentin_badia_codina:
//  case INPAR::FLUID::tau_franca_madureira_valentin_badia_codina_wo_dt:
//  {
//    /*
//
//    This stabilization parameter is only intended to be used for
//    (viscous-)reactive problems such as Darcy(-Stokes/Brinkman) problems.
//
//    literature:
//       S. Badia, R. Codina, Stabilized continuous and discontinuous
//       Galerkin techniques for Darcy flow, Comput. Methods Appl.
//       Mech. Engrg. 199 (2010) 1654-1667.
//
//    */
//
//    // constant c_p as suggested in Badia and Codina (2010), method A
//    // (set to be 4.0 in Badia and Codina (2010))
//    const double c_p = 4.0;
//
//    tau_(2) = c_p*DSQR(h_p)*reacoeff_;
//  }
//  break;
//
//  case INPAR::FLUID::tau_hughes_franca_balestra_wo_dt:
//  {
//    /*----------------------------------------------------------------------*/
//    /*
//     *  This stabilization parameter is only intended to be used for
//     *  stationary Stokes problems.
//     *
//     *  literature:
//     *  1) T.J.R. Hughes, L.P. Franca, and M. Balestra, A new finite element
//     *     formulation for computational fluid dynamics: V. circumventing the
//     *     Babuska-Brezzi condition: a stable Petrov-Galerkin formulation of
//     *     the Stokes problem accomodating equal-order interpolations,
//     *     Comput. Methods Appl. Mech. Engrg. 59 (1986) 85-99.
//     *
//     *  2) J. Donea and A. Huerta, Finite element methods for flow problems.
//     *     (for alpha_0 = 1/3)
//     */
//    /*----------------------------------------------------------------------*/
//
//    // tau_C not required)
//    tau_(2) = 0.0;
//
//  }
//  break;
//
//  default:
//  {
//    if (not (fldpara_->StabType() == INPAR::FLUID::stabtype_edgebased and
//        fldpara_->WhichTau() == INPAR::FLUID::tau_not_defined))
//      dserror("unknown definition for tau_C\n %i  ", fldpara_->WhichTau());
//
//    break;
//  }
//  }  // end switch (fldpara_->WhichTau())
//
//  } // end of quasistatic case
//
//  //-----------------------------------------------------------
//  //-----------------------------------------------------------
//
//  else // INPAR::FLUID::subscales_time_dependent
//  {
//    // norms of velocity in gausspoint, time n+af and time n+1
//    const double vel_normaf = convvelint_.Norm2();
//    const double vel_normnp = vel_normnp_;
//
//    // get velocity (n+alpha_F,i) derivatives at integration point
//    //
//    //       n+af      +-----  dN (x)
//    //   dvel    (x)    \        k         n+af
//    //   ----------- =   +     ------ * vel
//    //       dx         /        dx        k
//    //         j       +-----      j
//    //                 node k
//    //
//    // j : direction of derivative x/y/z
//    //
//    LINALG::Matrix<nsd_,nsd_> vderxyaf_(true);
//    vderxyaf_.Update(vderxy_);
//
//    // Now we are ready. Let's go on!
//
//
//    //-------------------------------------------------------
//    //          TAUS FOR TIME DEPENDENT SUBSCALES
//    //-------------------------------------------------------
//
//    switch(fldpara_->WhichTau())
//    {
//    case INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen:
//    case INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen_wo_dt:
//    {
//      /* INSTATIONARY FLOW PROBLEM, GENERALISED ALPHA
//
//         tau_M: Bazilevs et al. + ideas from Codina
//                                                         1.0
//                 +-                                 -+ - ---
//                 |                                   |   2.0
//             td  |  n+af      n+af         2         |
//          tau  = | u     * G u     + C * nu  * G : G |
//             M   |         -          I        -   - |
//                 |         -                   -   - |
//                 +-                                 -+
//
//         tau_C: Bazilevs et al., derived from the fine scale complement Shur
//                                 operator of the pressure equation
//
//
//                       td         1.0
//                    tau  = -----------------
//                       C       td   /     \
//                            tau  * | g * g |
//                               M    \-   -/
//      */
//
//      /*          +-           -+   +-           -+   +-           -+
//                  |             |   |             |   |             |
//                  |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
//            G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
//             ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
//                  |    i     j  |   |    i     j  |   |    i     j  |
//                  +-           -+   +-           -+   +-           -+
//      */
//      LINALG::Matrix<nsd_,nsd_> G;
//
//      for (int nn=0;nn<nsd_;++nn)
//      {
//        for (int rr=0;rr<nsd_;++rr)
//        {
//          G(nn,rr) = xji_(nn,0)*xji_(rr,0);
//          for (int mm=1;mm<nsd_;++mm)
//          {
//            G(nn,rr) += xji_(nn,mm)*xji_(rr,mm);
//          }
//        }
//      }
//
//      /*          +----
//                   \
//          G : G =   +   G   * G
//          -   -    /     ij    ij
//          -   -   +----
//                   i,j
//      */
//      double normG = 0;
//      for (int nn=0;nn<nsd_;++nn)
//      {
//        for (int rr=0;rr<nsd_;++rr)
//        {
//          normG+=G(nn,rr)*G(nn,rr);
//        }
//      }
//
//      /*                    +----
//           n+af      n+af    \     n+af         n+af
//          u     * G u     =   +   u    * G   * u
//                  -          /     i     -ij    j
//                  -         +----        -
//                             i,j
//      */
//      double Gnormu = 0;
//      for (int nn=0;nn<nsd_;++nn)
//      {
//        for (int rr=0;rr<nsd_;++rr)
//        {
//          Gnormu+=convvelint_(nn)*G(nn,rr)*convvelint_(rr);
//        }
//      }
//
//      // definition of constant
//      // (Akkerman et al. (2008) used 36.0 for quadratics, but Stefan
//      //  brought 144.0 from Austin...)
//      const double CI = 12.0/mk;
//
//      /*                                                 1.0
//                 +-                                 -+ - ---
//                 |                                   |   2.0
//                 |  n+af      n+af         2         |
//          tau  = | u     * G u     + C * nu  * G : G |
//             M   |         -          I        -   - |
//                 |         -                   -   - |
//                 +-                                 -+
//      */
//      tau_(0) = 1.0/sqrt(Gnormu+CI*visceff_*visceff_*normG);
//      tau_(1) = tau_(0);
//
//      /*         +-     -+   +-     -+   +-     -+
//                 |       |   |       |   |       |
//                 |  dr   |   |  ds   |   |  dt   |
//            g  = |  ---  | + |  ---  | + |  ---  |
//             i   |  dx   |   |  dx   |   |  dx   |
//                 |    i  |   |    i  |   |    i  |
//                 +-     -+   +-     -+   +-     -+
//      */
//      LINALG::Matrix<nsd_,1> g;
//
//      for (int rr=0;rr<nsd_;++rr)
//      {
//        g(rr) = xji_(rr,0);
//        for (int mm=1;mm<nsd_;++mm)
//        {
//          g(rr) += xji_(rr,mm);
//        }
//      }
//
//      /*         +----
//                  \
//         g * g =   +   g * g
//         -   -    /     i   i
//                 +----
//                   i
//      */
//      double normgsq = 0.0;
//
//      for (int rr=0;rr<nsd_;++rr)
//      {
//        normgsq+=g(rr)*g(rr);
//      }
//
//      /*
//                                1.0
//                  tau  = -----------------
//                     C            /      \
//                          tau  * | g * g |
//                             M    \-   -/
//      */
//      tau_(2) = 1./(tau_(0)*normgsq);
//
//    }
//    break;
//    case INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall:
//    case INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall_wo_dt:
//    {
//      // INSTATIONARY FLOW PROBLEM, GENERALISED ALPHA
//      //
//      // tau_M: modification of
//      //
//      //    Franca, L.P. and Valentin, F.: On an Improved Unusual Stabilized
//      //    Finite Element Method for the Advective-Reactive-Diffusive
//      //    Equation. Computer Methods in Applied Mechanics and Enginnering,
//      //    Vol. 190, pp. 1785-1800, 2000.
//      //    http://www.lncc.br/~valentin/publication.htm                   */
//      //
//      // tau_Mp: modification of Barrenechea, G.R. and Valentin, F.
//      //
//      //    Barrenechea, G.R. and Valentin, F.: An unusual stabilized finite
//      //    element method for a generalized Stokes problem. Numerische
//      //    Mathematik, Vol. 92, pp. 652-677, 2002.
//      //    http://www.lncc.br/~valentin/publication.htm
//      //
//      //
//      // tau_C: kept Wall definition
//      //
//      // for the modifications see Codina, Principe, Guasch, Badia
//      //    "Time dependent subscales in the stabilized finite  element
//      //     approximation of incompressible flow problems"
//      //
//      //
//      // see also: Codina, R. and Soto, O.: Approximation of the incompressible
//      //    Navier-Stokes equations using orthogonal subscale stabilisation
//      //    and pressure segregation on anisotropic finite element meshes.
//      //    Computer methods in Applied Mechanics and Engineering,
//      //    Vol 193, pp. 1403-1419, 2004.
//
//      // calculate characteristic element length
//      CalcCharEleLength(vol,vel_norm,h_u,h_p);
//
//      //---------------------------------------------- compute tau_Mu = tau_Mp
//      /* convective : viscous forces (element reynolds number)*/
//      const double re_convectaf = (vel_normaf * h_p / visceff_ ) * (mk/2.0);
//      const double xi_convectaf = std::max(re_convectaf,1.0);
//
//      /*
//               xi_convect ^
//                          |      /
//                          |     /
//                          |    /
//                        1 +---+
//                          |
//                          |
//                          |
//                          +--------------> re_convect
//                              1
//      */
//
//      /* the 4.0 instead of the Franca's definition 2.0 results from the viscous
//       * term in the Navier-Stokes-equations, which is scaled by 2.0*nu         */
//
//      tau_(0) = DSQR(h_p) / (4.0 * visceff_ / mk + ( 4.0 * visceff_/mk) * xi_convectaf);
//
//      tau_(1) = tau_(0);
//
//      /*------------------------------------------------------ compute tau_C ---*/
//
//      //-- stability parameter definition according to Wall Diss. 99
//      /*
//               xi_convect ^
//                          |
//                        1 |   +-----------
//                          |  /
//                          | /
//                          |/
//                          +--------------> Re_convect
//                              1
//      */
//      const double re_convectnp = (vel_normnp * h_p / visceff_ ) * (mk/2.0);
//
//      const double xi_tau_c = std::min(re_convectnp,1.0);
//
//      tau_(2) = vel_normnp * h_p * 0.5 * xi_tau_c;
//
//    }
//    break;
//    case INPAR::FLUID::tau_codina:
//    {
//      // Parameter from Codina, Badia (Constants are chosen according to
//      // the values in the standard definition above)
//
//      const double CI  = 4.0/mk;
//      const double CII = 2.0/mk;
//
//      // in contrast to the original definition, we neglect the influence of
//      // the subscale velocity on velnormaf
//      tau_(0)=1.0/(CI*visceff_/(h_p*h_p)+CII*vel_normaf/h_p);
//
//      tau_(1)=tau_(0);
//
//      tau_(2)=(h_p*h_p)/(CI*tau_(0));
//    }
//    break;
//    default:
//    {
//      dserror("Unknown definition of stabilization parameter for time-dependent formulation\n");
//    }
//    break;
//    } // Switch TauType
//
//  } // end Fluid::subscales_time_dependent
//
//  return;
//}


///*----------------------------------------------------------------------*
// |  calculation of characteristic element length               vg 01/11 |
// *----------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::CalcCharEleLength(
//    const double  vol,
//    const double  vel_norm,
//    double&       h_u,
//    double&       h_p
//    )
//{
//  // check for potential 1-D computations
//  if (nsd_==1) dserror("Element length not implemented for 1-D computation!");
//
//  //---------------------------------------------------------------------
//  // select from various definitions for characteristic element length
//  // for tau_Mu
//  //---------------------------------------------------------------------
//  switch (fldpara_->CharEleLengthU())
//  {
//    // a) streamlength due to Tezduyar et al. (1992) -> default
//    // normed velocity vector
//    case INPAR::FLUID::streamlength_u:
//    {
//      LINALG::Matrix<nsd_,1> velino(true);
//      if (vel_norm>=1e-6) velino.Update(1.0/vel_norm,convvelint_);
//      else
//      {
//        velino.Clear();
//        velino(0,0) = 1;
//      }
//
//      // get streamlength using the normed velocity at element centre
//      LINALG::Matrix<nen_,1> tmp;
//
//      //enriched dofs are not interpolatory with respect to geometry
//      if(enrtype == DRT::ELEMENTS::Fluid::xwall)
//      {
//        LINALG::Matrix<nsd_,nen_> derxy_copy(derxy_);
//        for(int inode=1;inode<nen_;inode+=2)
//        {
//          for (int idim=0; idim<nsd_;idim++)
//            derxy_copy(idim,inode)=0.0;
//        }
//        tmp.MultiplyTN(derxy_copy,velino);
//      }
//      else
//        tmp.MultiplyTN(derxy_,velino);
//
//      const double val = tmp.Norm1();
//      h_u = 2.0/val; // h=streamlength
//    }
//    break;
//
//    // b) volume-equivalent diameter (warning: 3-D formula!)
//    case INPAR::FLUID::volume_equivalent_diameter_u:
//    {
//      h_u = std::pow((6.*vol/M_PI),(1.0/3.0))/sqrt(3.0);
//    }
//    break;
//
//    // c) cubic/square root of element volume/area or element length (3- and 2-D)
//    case INPAR::FLUID::root_of_volume_u:
//    {
//      // cast dimension to a double varibale -> pow()
//      const double dim = double (nsd_);
//      h_u = std::pow(vol,1.0/dim);
//    }
//    break;
//
//    default: dserror("unknown characteristic element length for tau_Mu\n");
//    break;
//  } //switch (charelelengthu_)
//
//  //---------------------------------------------------------------------
//  // select from various definitions for characteristic element length
//  // for tau_Mp and tau_C
//  //---------------------------------------------------------------------
//  switch (fldpara_->CharEleLengthPC())
//  {
//    // a) streamlength due to Tezduyar et al. (1992) -> default
//    // normed velocity vector
//    case INPAR::FLUID::streamlength_pc:
//    {
//      LINALG::Matrix<nsd_,1> velino(true);
//      if (vel_norm>=1e-6) velino.Update(1.0/vel_norm,convvelint_);
//      else
//      {
//        velino.Clear();
//        velino(0,0) = 1;
//      }
//
//      // get streamlength using the normed velocity at element centre
//      LINALG::Matrix<nen_,1> tmp;
//      //enriched dofs are not interpolatory with respect to geometry
//      if(enrtype == DRT::ELEMENTS::Fluid::xwall)
//      {
//        LINALG::Matrix<nsd_,nen_> derxy_copy(derxy_);
//        for(int inode=1;inode<nen_;inode+=2)
//        {
//          for (int idim=0; idim<nsd_;idim++)
//            derxy_copy(idim,inode)=0.0;
//        }
//        tmp.MultiplyTN(derxy_copy,velino);
//      }
//      else
//        tmp.MultiplyTN(derxy_,velino);
//      const double val = tmp.Norm1();
//      h_p = 2.0/val; // h=streamlength
//    }
//    break;
//
//    // b) volume-equivalent diameter (warning: 3-D formula!)
//    case INPAR::FLUID::volume_equivalent_diameter_pc:
//    {
//      h_p = std::pow((6.*vol/M_PI),(1.0/3.0))/sqrt(3.0);
//    }
//    break;
//
//    // c) cubic/square root of element volume/area or element length (3- and 2-D)
//    case INPAR::FLUID::root_of_volume_pc:
//    {
//      // cast dimension to a double varibale -> pow()
//      const double dim = double (nsd_);
//      h_p = std::pow(vol,1/dim);
//    }
//    break;
//
//    default: dserror("unknown characteristic element length for tau_Mu and tau_C\n");
//    break;
//  } //switch (charelelengthpc_)
//
//  return;
//}


template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::CalcDivEps(
    const LINALG::Matrix<nsd_,nen_>&      evelaf,
    const LINALG::Matrix<nsd_,nen_>&      eveln)
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

  // set visc_old to zero
  visc_old_.Clear();
  visc_oldn_.Clear();

  double prefac;
  if(fldpara_->PhysicalType() == INPAR::FLUID::loma)
  //if(loma_)
  {
    prefac = 1.0/3.0;
    derxy2_.Scale(prefac);
 }
  else prefac = 1.0;

  if (nsd_==3)
  {
    for (int inode=0; inode<nen_; ++inode)
    {
      double sum = (derxy2_(0,inode)+derxy2_(1,inode)+derxy2_(2,inode))/prefac;
      viscs2_(0,inode) = 0.5 * (sum + derxy2_(0,inode));
      viscs2_(1,inode) = 0.5 *  derxy2_(3,inode);
      viscs2_(2,inode) = 0.5 *  derxy2_(4,inode);
      viscs2_(3,inode) = 0.5 *  derxy2_(3,inode);
      viscs2_(4,inode) = 0.5 * (sum + derxy2_(1,inode));
      viscs2_(5,inode) = 0.5 *  derxy2_(5,inode);
      viscs2_(6,inode) = 0.5 *  derxy2_(4,inode);
      viscs2_(7,inode) = 0.5 *  derxy2_(5,inode);
      viscs2_(8,inode) = 0.5 * (sum + derxy2_(2,inode));

      for (int idim=0; idim<nsd_; ++idim)
      {
        const int nsd_idim = idim*nsd_;
        for (int jdim=0; jdim<nsd_; ++jdim)
        {
          visc_old_(idim) += viscs2_(nsd_idim+jdim,inode)*evelaf(jdim,inode);
          visc_oldn_(idim) += viscs2_(nsd_idim+jdim,inode)*eveln(jdim,inode);
        }
      }
    }
  }
  else if (nsd_==2)
  {
    for (int inode=0; inode<nen_; ++inode)
    {
    double sum = (derxy2_(0,inode)+derxy2_(1,inode))/prefac;
    viscs2_(0,inode) = 0.5 * (sum + derxy2_(0,inode));
    viscs2_(1,inode) = 0.5 * derxy2_(2,inode);
    viscs2_(2,inode) = 0.5 * derxy2_(2,inode);
    viscs2_(3,inode) = 0.5 * (sum + derxy2_(1,inode));

    for (int idim=0; idim<nsd_; ++idim)
    {
      const int nsd_idim = idim*nsd_;
      for (int jdim=0; jdim<nsd_; ++jdim)
      {
        visc_old_(idim) += viscs2_(nsd_idim+jdim,inode)*evelaf(jdim,inode);
        visc_oldn_(idim) += viscs2_(nsd_idim+jdim,inode)*eveln(jdim,inode);
      }
    }
    }
  }
  else dserror("Epsilon(N) is not implemented for the 1D case");

  return;
}


template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::ComputeSubgridScaleVelocityOSTNew(
    const LINALG::Matrix<nsd_,nen_>&  eaccam,
    double &                          fac1,
    double &                          fac2,
    double &                          fac3,
    double &                          facMtau,
    int                               iquad,
    double *                          saccn,
    double *                          sveln,
    double *                          svelnp
    )
{
  //----------------------------------------------------------------------
  // compute residual of momentum equation
  // -> different for generalized-alpha and other time-integration schemes
  //----------------------------------------------------------------------
  if (fldparatimint_->IsGenalpha())
  {
    if (fldpara_->PhysicalType() == INPAR::FLUID::boussinesq)
      dserror("The combination of generalized-alpha time integration and a Boussinesq approximation has not been implemented yet!");

    // rhs of momentum equation: density*bodyforce at n+alpha_F
    rhsmom_.Update(densaf_,bodyforce_,0.0);
    // and pressure gradient prescribed as body force
    // caution: not density weighted
    rhsmom_.Update(1.0,generalbodyforce_,1.0);

    // get acceleration at time n+alpha_M at integration point
    accint_.Multiply(eaccam,funct_);

    // evaluate momentum residual once for all stabilization right hand sides
    for (int rr=0;rr<nsd_;++rr)
    {
      momres_old_(rr) = densam_*accint_(rr)+densaf_*conv_old_(rr)+gradp_(rr)
                       -2*visceff_*visc_old_(rr)+reacoeff_*velint_(rr)-densaf_*bodyforce_(rr)-generalbodyforce_(rr);
    }

    // add consistency terms for MFS if applicable
    MultfracSubGridScalesConsistentResidual();
  }
  else
  {
    if (not fldparatimint_->IsStationary())
    {
      // rhs of instationary momentum equation:
      // density*theta*bodyforce at n+1 + density*(histmom/dt)
      // in the case of a Boussinesq approximation: f = rho_0*[(rho - rho_0)/rho_0]*g = (rho - rho_0)*g
      // else:                                      f = rho * g
      if (fldpara_->PhysicalType() == INPAR::FLUID::boussinesq)
      {
//      Made old OST impl equivalent to gen-alpha (alpha_f=alpha_m=1) (multiplied with \rho_(n+1))
        rhsmom_.Update((densaf_/fldparatimint_->Dt()/fldparatimint_->Theta()),histmom_,deltadens_,bodyforce_);
        // and pressure gradient prescribed as body force
        // caution: not density weighted
        rhsmom_.Update(1.0,generalbodyforce_,1.0);
      }
      else
      {
        //      Made old OST impl equivalent to gen-alpha (alpha_f=alpha_m=1) (multiplied with \rho_(n+1))
        rhsmom_.Update((densaf_/fldparatimint_->Dt()/fldparatimint_->Theta()),histmom_,densaf_,bodyforce_);

        // and pressure gradient prescribed as body force
        // caution: not density weighted
        rhsmom_.Update(1.0,generalbodyforce_,1.0);
      }

      // compute instationary momentum residual:
      // momres_old = u_(n+1)/dt + theta ( ... ) +(1-theta) ( ... ) - theta*bodyforce_
      if(fldparatimint_->IsNewOSTImplementation())
      {
        const double quotfac = (1.0-fldparatimint_->Theta())*fldparatimint_->Dt()/fldparatimint_->TimeFacRhs();
        for (int rr=0;rr<nsd_;++rr)
        {
          momres_old_(rr) =  dens_theta_*(velint_(rr)-velintn_(rr))/(fldparatimint_->Dt()*fldparatimint_->Theta());
          momres_old_(rr) -= (densaf_*bodyforce_(rr)    +  densn_*quotfac*bodyforcen_(rr));
          momres_old_(rr) += reacoeff_*(velint_(rr)     +  quotfac*velintn_(rr));  //TODO: Time dependant reacoef.
          momres_old_(rr) += (densaf_*conv_old_(rr)     +  densn_*quotfac*conv_oldn_(rr));
          momres_old_(rr) += -2*(visceff_*visc_old_(rr) +  viscn_*quotfac*visc_oldn_(rr)); //TODO: Fix viscosity for time step (n)
          momres_old_(rr) += generalbodyforce_(rr)      +  quotfac*generalbodyforcen_(rr);
          if(not fldparatimint_->IsFullImplPressureAndCont()){
            momres_old_(rr) += gradp_(rr)                 +  quotfac*gradpn_(rr);
          }
          else{
            momres_old_(rr) += gradp_(rr)/fldparatimint_->Theta(); //Gradient of p with no pre-factor.
          }

          // Left for implementation in ALE, possibly....
          //        //Furthermore, should rhsmom_ be calculated like this? Not with galerkin terms and integration???!
          //        rhsmom_(rr)=- fldparatimint_->Dt() *fldparatimint_->Theta()* (- densaf_*velintn_(rr)/(fldparatimint_->Dt()*fldparatimint_->Theta())
          //                                          - densn_*quotfac*bodyforcen_(rr)
          //                                          + reacoeff_*quotfac*velintn_(rr) + densaf_*quotfac*conv_oldn_(rr)
          //                                          - 2*viscn_*quotfac*visc_oldn_(rr)  + quotfac*generalbodyforcen_(rr)
          //                                          + quotfac*gradpn_(rr)); //Check what gradpn_ to use! (This is needed for ALE implementations?)

        }
      }
      else
      {
        // momres_old = u_(n+1)/dt + theta ( ... ) - histmom_/dt - theta*bodyforce_
        for (int rr=0;rr<nsd_;++rr)
        {
          momres_old_(rr) = ((densaf_*velint_(rr)/fldparatimint_->Dt()
              +fldparatimint_->Theta()*(densaf_*conv_old_(rr)+gradp_(rr)
                  -2*visceff_*visc_old_(rr)+reacoeff_*velint_(rr)))/fldparatimint_->Theta())-rhsmom_(rr);
        }
      }//end IsNewOSTImplementation
    }
    else
    {
      // rhs of stationary momentum equation: density*bodyforce
      // in the case of a Boussinesq approximation: f = rho_0*[(rho - rho_0)/rho_0]*g = (rho - rho_0)*g
      // else:                                      f = rho * g
      // and pressure gradient prescribed as body force (not density weighted)
      if (fldpara_->PhysicalType() == INPAR::FLUID::boussinesq)
           rhsmom_.Update(deltadens_,bodyforce_, 1.0,generalbodyforce_);
      else rhsmom_.Update(densaf_,bodyforce_,1.0,generalbodyforce_);

      // compute stationary momentum residual:
      for (int rr=0;rr<nsd_;++rr)
      {
        momres_old_(rr) = -rhsmom_(rr);
        momres_old_(rr) += gradp_(rr);
        momres_old_(rr) += reacoeff_*velint_(rr);
        momres_old_(rr) += densaf_*conv_old_(rr);
        momres_old_(rr) += -2*visceff_*visc_old_(rr);
      }

      // add consistency terms for MFS if applicable
      MultfracSubGridScalesConsistentResidual();
    }
  }

  //----------------------------------------------------------------------
  // compute subgrid-scale velocity
  //----------------------------------------------------------------------
  // 1) quasi-static subgrid scales
  // Definition of subgrid-scale velocity is not consistent for the SUPG term and Franca, Valentin, ...
  // Definition of subgrid velocity used by Hughes
  if (fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic)
  {
    sgvelint_.Update(-tau_(1),momres_old_,0.0);
  }
  // 2) time-dependent subgrid scales
  else
  {
    // some checking
    if (fldparatimint_->IsStationary())
      dserror("there is no time dependent subgrid scale closure for stationary problems\n");
    if ( saccn==NULL or sveln==NULL or svelnp==NULL )
      dserror( "no subscale array provided" );

    // parameter definitions
    double alphaF = fldparatimint_->AlphaF();
    double alphaM = fldparatimint_->AlphaM();
    double gamma  = fldparatimint_->Gamma();
    double dt     = fldparatimint_->Dt();

    /*
                                            1.0
       facMtau =  -------------------------------------------------------
                     n+aM                      n+aF
                  rho     * alphaM * tauM + rho     * alphaF * gamma * dt
    */
    facMtau = 1.0/(densam_*alphaM*tau_(1)+densaf_*fldparatimint_->Afgdt());

    /*
       factor for old subgrid velocities:

                 n+aM                      n+aF
       fac1 = rho     * alphaM * tauM + rho     * gamma * dt * (alphaF-1)
    */
    fac1=(densam_*alphaM*tau_(1)+densaf_*gamma*dt*(alphaF-1.0))*facMtau;
    /*
      factor for old subgrid accelerations

                 n+aM
       fac2 = rho     * tauM * dt * (alphaM-gamma)
    */
    fac2=(densam_*dt*tau_(1)*(alphaM-gamma))*facMtau;
    /*
      factor for residual in current subgrid velocities:

       fac3 = gamma * dt * tauM
    */
    fac3=(gamma*dt*tau_(1))*facMtau;

    // warning: time-dependent subgrid closure requires generalized-alpha time
    // integration
    if (!fldparatimint_->IsGenalpha())
    {
      dserror("the time-dependent subgrid closure requires a genalpha time integration\n");
    }

    /*         +-                                       -+
        ~n+1   |        ~n           ~ n            n+1  |
        u    = | fac1 * u  + fac2 * acc  -fac3 * res     |
         (i)   |                                    (i)  |
               +-                                       -+
    */

    /* compute the intermediate value of subscale velocity

            ~n+af            ~n+1                   ~n
            u     = alphaF * u     + (1.0-alphaF) * u
             (i)              (i)

    */

    LINALG::Matrix<1,nsd_> sgvelintaf(true);
    for (int rr=0;rr<nsd_;++rr)
    {
      tds_->UpdateSvelnpInOneDirection(
         fac1           ,
         fac2           ,
         fac3           ,
         momres_old_(rr),
         fldparatimint_->AlphaF(),
         rr             ,
         iquad          ,
         sgvelint_(rr)  , // sgvelint_ is set to sgvelintnp, but is then overwritten below anyway!
         sgvelintaf(rr));

      int pos = rr + nsd_*iquad;

      /*
       *  ~n+1           ~n           ~ n            n+1
       *  u    =  fac1 * u  + fac2 * acc  -fac3 * res
       *   (i)
       *
       */

      svelnp[pos] =
        fac1*sveln[pos]
        +
        fac2*saccn[pos]
        -
        fac3*momres_old_(rr);

      /* compute the intermediate value of subscale velocity
       *
       *          ~n+af            ~n+1                   ~n
       *          u     = alphaF * u     + (1.0-alphaF) * u
       *           (i)              (i)
       *
       */
      sgvelint_(rr) =
        alphaF      *svelnp[pos]
        +
        (1.0-alphaF)*sveln [pos];
    }
  } // end time dependent subgrid scale closure

  //----------------------------------------------------------------------
  // include computed subgrid-scale velocity in convective term
  // -> only required for cross- and Reynolds-stress terms
  //----------------------------------------------------------------------
  if (fldpara_->Cross()         != INPAR::FLUID::cross_stress_stab_none or
      fldpara_->Reynolds()      != INPAR::FLUID::reynolds_stress_stab_none or
      fldpara_->ContiCross()    != INPAR::FLUID::cross_stress_stab_none or
      fldpara_->ContiReynolds() != INPAR::FLUID::reynolds_stress_stab_none)
       sgconv_c_.MultiplyTN(derxy_,sgvelint_);
  else sgconv_c_.Clear();
}


template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::LinGalMomResUOSTNew(
                     LINALG::Matrix<nsd_*nsd_,nen_> &    lin_resM_Du,
                     const double &                      timefacfac)
{
  /*
      instationary                          cross-stress, part 1
       +-----+                             +-------------------+
       |     |                             |                   |

                 /       n+1       \        /      ~n+1       \
       rho*Du + |   rho*u   o nabla | Du + |   rho*u   o nabla | Du +
                 \      (i)        /        \      (i)        /

                 /                \  n+1
              + |   rho*Du o nabla | u      +  sigma*Du
                 \                /   (i)
                |                        |     |       |
                +------------------------+     +-------+
                        Newton                  reaction
  */

  int idim_nsd_p_idim[nsd_];

  for (int idim = 0; idim <nsd_; ++idim)
  {
    idim_nsd_p_idim[idim]=idim*nsd_+idim;
  }

  if (fldparatimint_->IsStationary() == false)
  {

//    double fac_densam= (fldparatimint_->IsOneStepTheta()) ? fac_*dens_theta_ : fac_*densam_;
    double fac_densam;
    if(fldparatimint_->IsNewOSTImplementation())
      fac_densam= fldparatimint_->IsOneStepTheta() ? fac_*dens_theta_ : fac_*densam_;
    else
      fac_densam= fac_*densam_;
    //End of IsNewOSTImplementation()

    for (int ui=0; ui<nen_; ++ui)
    {
      const double v=fac_densam*funct_(ui);

      for (int idim = 0; idim <nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim],ui)+=v;
      }
    }
  }

  const double timefacfac_densaf=timefacfac*densaf_;

  // convection, reactive
  for (int ui=0; ui<nen_; ++ui)
  {
  const double v=timefacfac_densaf*conv_c_(ui);

    for (int idim = 0; idim <nsd_; ++idim)
    {
      lin_resM_Du(idim_nsd_p_idim[idim],ui)+=v;
    }
  }

  // convection, convective (only for Newton)
  if(fldpara_->IsNewton())
  {
    for (int ui=0; ui<nen_; ++ui)
    {
      const double temp=timefacfac_densaf*funct_(ui);

      for (int idim = 0; idim <nsd_; ++idim)
      {
        const int idim_nsd=idim*nsd_;

        for(int jdim=0;jdim<nsd_;++jdim)
        {
          lin_resM_Du(idim_nsd+jdim,ui)+=temp*vderxy_(idim,jdim);
        }
      }
    }
  }

  if (fldpara_->Reaction())
  {
    const double fac_reac=timefacfac*reacoeff_;

    for (int ui=0; ui<nen_; ++ui)
    {
      const double v=fac_reac*funct_(ui);

      for (int idim = 0; idim <nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim],ui)+=v;
      }
    }
  }

  if(fldpara_->Cross()==INPAR::FLUID::cross_stress_stab)
  {
   //const double rhsresfac_densaf=rhsresfac*densaf_;
    for (int ui=0; ui<nen_; ++ui)
    {
      //const double v=rhsresfac_densaf*sgconv_c_(ui);
      const double v=timefacfac_densaf*sgconv_c_(ui);

      for (int idim = 0; idim <nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim],ui)+=v;
      }
    }
  }

  return;
}




//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::LinGalMomResU_subscales(
//            LINALG::Matrix<nen_*nsd_,nen_> &    estif_p_v,
//            LINALG::Matrix<nsd_*nsd_,nen_> &    lin_resM_Du,
//            LINALG::Matrix<nsd_,1> &            resM_Du,
//            const double &                      timefacfac,
//            const double &                      facMtau)
//{
//  // rescale Galerkin residual of all terms which have not been
//  // integrated by parts
//
//  const double C_saccGAL=densaf_*fldparatimint_->Afgdt()*facMtau;
//
//  for (int ui=0; ui<nen_; ++ui)
//  {
//    for (int idim = 0; idim <nsd_; ++idim)
//    {
//      const int idim_nsd=idim*nsd_;
//
//      for(int jdim=0;jdim<nsd_;++jdim)
//      {
//        lin_resM_Du(idim_nsd+jdim,ui)*=C_saccGAL;
//      }
//    }
//  }
//
//  // include all contributions which have been integrated by parts
//  // and thus can not be rescaled
//
//  /* viscous term (intermediate) */
//  /*  factor:
//                                rhoaM*alphaM*tauM                 gamma*dt
//          2*nu*alphaF*---------------------------------------,  * --------
//                      rhoaM*alphaM*tauM+rhoaf*alphaF*gamma*dt      alphaM
//
//
//             /                         \
//            |               /    \      |
//            |  nabla o eps | Dacc | , v |
//            |               \    /      |
//             \                         /
//
//  */
//
//  if (is_higher_order_ele_)
//  {
//    const double v = 2.0*visceff_*timefacfac*(1.0-C_saccGAL);
//    for (int idim = 0; idim <nsd_; ++idim)
//    {
//      const int nsd_idim=nsd_*idim;
//
//      for(int jdim=0;jdim<nsd_;++jdim)
//      {
//        const int nsd_idim_p_jdim=nsd_idim+jdim;
//
//        for (int ui=0; ui<nen_; ++ui)
//        {
//          lin_resM_Du(nsd_idim_p_jdim,ui)+=v*viscs2_(nsd_idim_p_jdim, ui);
//        }
//      }
//    }
//  }
//
//  /*  factor:
//                              rhoaM*alphaM*tauM                gamma*dt
//          alphaF * ---------------------------------------,  * --------
//                   rhoaM*alphaM*tauM+rhoaF*alphaF*gamma*dt      alphaM
//
//                       /               \
//                      |                 |
//                      |  nabla Dp ,  v  |
//                      |                 |
//                       \               /
//  */
//  for (int ui=0; ui<nen_; ++ui)
//  {
//    const double v=(1.0-C_saccGAL)*fac_*fldparatimint_->TimeFacPre();
//    for (int vi=0; vi<nen_; ++vi)
//    {
//      const int fvi = nsd_*vi;
//
//      for (int idim = 0; idim <nsd_; ++idim)
//      {
//        estif_p_v(fvi + idim,ui)-= v*derxy_(idim,ui)*funct_(vi);
//      }
//    }
//  }
//
//  /*  factor: +1
//
//           /                       \
//          |     n+am    ~ n+am      |
//          |  rho     * acc     , v  |
//          |               (i)       |
//           \                       /
//
//
//         using
//                                  n+af             /
//           n+am    ~ n+am      rho        ~n+af   |    n+am      n+am
//        rho     * acc     = - --------- * u     - | rho     * acc     +
//                     (i)           n+af    (i)    |              (i)
//                               tau_M               \
//
//                                  n+af    / n+af        \   n+af            n+1
//                             + rho     * | c     o nabla | u     + nabla o p    -
//                                          \ (i)         /   (i)             (i)
//
//                                                        / n+af \
//                             - 2 * mu * grad o epsilon | u      | -
//                                                        \ (i)  /
//                                               \
//                                  n+af    n+af  |
//                             - rho     * f      |
//                                                |
//                                               /
//  */
//
//  /*
//  For time-dependent subgrid scales closure, we take the implementation
//  by Peter Gamnitzer, reading as documented below. Note that certain terms cancel out,
//  such as those containing acc_(i)^n+am and the convective term!
//  */
//
//  //---------------------------------------------------------------
//  //
//  //      GALERKIN PART AND SUBSCALE ACCELERATION STABILISATION
//  //
//  //---------------------------------------------------------------
//  /*  factor: +1
//
//         /             \     /                     \
//        |   ~ n+am      |   |     n+am    n+af      |
//        |  acc     , v  | + |  acc     - f     , v  |
//        |     (i)       |   |     (i)               |
//         \             /     \                   /
//
//
//       using
//                                                  /
//                  ~ n+am        1.0      ~n+af   |    n+am
//                 acc     = - --------- * u     - | acc     +
//                    (i)           n+af    (i)    |    (i)
//                             tau_M                \
//
//                              / n+af        \   n+af            n+1
//                           + | c     o nabla | u     + nabla o p    -
//                              \ (i)         /   (i)             (i)
//
//                                                      / n+af \
//                           - 2 * nu * grad o epsilon | u      | -
//                                                      \ (i)  /
//                                   \
//                              n+af  |
//                           - f      |
//                                    |
//                                   /
//
//  */
//
//
//  for(int idim = 0; idim <nsd_; ++idim)
//  {
//    if(fldpara_->Tds()==INPAR::FLUID::subscales_time_dependent)
//      // see implementation by Peter Gamnitzer
//      resM_Du(idim)+=(timefacfac/fldparatimint_->AlphaF())*(-densaf_*sgvelint_(idim)/tau_(0) - gradp_(idim)+2*visceff_*visc_old_(idim)); //-momres_old_(idim));
//    else
//      resM_Du(idim)+=fac_*(-densaf_*sgvelint_(idim)/tau_(1)-momres_old_(idim));
//  }
//
//  return;
//}


//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::StabLinGalMomResU(
//    LINALG::Matrix<nsd_*nsd_,nen_> &          lin_resM_Du,
//    const double &                            timefacfac)
//{
//
//  /*
//                 /       n+1       \        /                \  n+1
//       rho*Du + |   rho*u   o nabla | Du + |   rho*Du o nabla | u   +
//                 \      (i)        /        \                /   (i)
//
//                               /  \
//     + sigma*Du + nabla o eps | Du |
//                               \  /
//  */
//  if(fldpara_->Tds()==INPAR::FLUID::subscales_time_dependent
//     ||
//     fldpara_->Cross()==INPAR::FLUID::cross_stress_stab)
//  {
//    //----------------------------------------------------------------------
//    /* GALERKIN residual was rescaled and cannot be reused; so rebuild it */
//
//    lin_resM_Du.Clear();
//
//    int idim_nsd_p_idim[nsd_];
//
//    for (int idim = 0; idim <nsd_; ++idim)
//    {
//      idim_nsd_p_idim[idim]=idim*nsd_+idim;
//    }
//
//    if (fldparatimint_->IsStationary() == false)
//    {
//
//      double fac_densam;
//      if(fldparatimint_->IsNewOSTImplementation())
//        fac_densam= fldparatimint_->IsOneStepTheta() ? fac_*dens_theta_ : fac_*densam_;
//      else
//        fac_densam= fac_*densam_;
//
//      for (int ui=0; ui<nen_; ++ui)
//      {
//        const double v=fac_densam*funct_(ui);
//
//        for (int idim = 0; idim <nsd_; ++idim)
//        {
//          lin_resM_Du(idim_nsd_p_idim[idim],ui)+=v;
//        }
//      }
//    }
//
//    const double timefacfac_densaf=timefacfac*densaf_;
//
//    for (int ui=0; ui<nen_; ++ui)
//    {
//      // deleted +sgconv_c_(ui)
//      const double v=timefacfac_densaf*conv_c_(ui);
//
//      for (int idim = 0; idim <nsd_; ++idim)
//      {
//        lin_resM_Du(idim_nsd_p_idim[idim],ui)+=v;
//      }
//    }
//
//    if (fldpara_->IsNewton())
//    {
////
////
//// dr_j   d    /    du_j \          du_j         dN_B
//// ----= ---- | u_i*----  | = N_B * ---- + u_i * ---- * d_jk
//// du_k  du_k  \    dx_i /          dx_k         dx_i
//
//      for (int ui=0; ui<nen_; ++ui)
//      {
//        const double temp=timefacfac_densaf*funct_(ui);
//
//        for (int idim = 0; idim <nsd_; ++idim)
//        {
//          const int idim_nsd=idim*nsd_;
//
//          for(int jdim=0;jdim<nsd_;++jdim)
//          {
//            lin_resM_Du(idim_nsd+jdim,ui)+=temp*vderxy_(idim,jdim);
//          }
//        }
//      }
//    }
//
//    if (fldpara_->Reaction())
//    {
//      const double fac_reac=timefacfac*reacoeff_;
//
//      for (int ui=0; ui<nen_; ++ui)
//      {
//        const double v=fac_reac*funct_(ui);
//
//        for (int idim = 0; idim <nsd_; ++idim)
//        {
//          lin_resM_Du(idim_nsd_p_idim[idim],ui)+=v;
//        }
//      }
//    }
//  }
//
//  if (is_higher_order_ele_)
//  {
//    const double v = -2.0*visceff_*timefacfac;
//    for (int idim = 0; idim <nsd_; ++idim)
//    {
//      const int nsd_idim=nsd_*idim;
//
//      for(int jdim=0;jdim<nsd_;++jdim)
//      {
//        const int nsd_idim_p_jdim=nsd_idim+jdim;
//
//        for (int ui=0; ui<nen_; ++ui)
//        {
//          lin_resM_Du(nsd_idim_p_jdim,ui)+=v*viscs2_(nsd_idim_p_jdim, ui);
//        }
//      }
//    }
//  }
//
//  return;
//}


template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::InertiaConvectionReactionGalPart(
    LINALG::Matrix<nen_*nsd_,nen_*nsd_> &   estif_u,
    LINALG::Matrix<nsd_,nen_> &             velforce,
    LINALG::Matrix<nsd_*nsd_,nen_> &        lin_resM_Du,
    LINALG::Matrix<nsd_,1> &                resM_Du,
    const double &                          rhsfac,
    const double &                          rhsfacn)
{
  /* inertia (contribution to mass matrix) if not is_stationary */
  /*
            /              \
           |                |
           |    rho*Du , v  |
           |                |
            \              /
  */
  /* convection, convective part (convective form) */
  /*
            /                             \
           |  /       n+1       \          |
           | |   rho*u   o nabla | Du , v  |
           |  \      (i)        /          |
            \                             /
  */
  /*  convection, reactive part (convective form)
            /                               \
           |  /                \   n+1       |
           | |  rho*Du o nabla  | u     , v  |
           |  \                /   (i)       |
            \                               /
  */
  /*  reaction */
  /*
            /                \
           |                  |
           |    sigma*Du , v  |
           |                  |
            \                /
  */
  if ((fldpara_->IsNewton() or
       (is_higher_order_ele_ and fldpara_->Tds()==INPAR::FLUID::subscales_time_dependent)))
  {
    for (int ui=0; ui<nen_; ++ui)
    {
      for (int vi=0; vi<nen_; ++vi)
      {
        for (int jdim= 0; jdim<nsd_;++jdim)
        {
          for (int idim = 0; idim<nsd_; ++idim)
          {
            estif_u(nsd_*vi+idim,nsd_*ui+jdim) += funct_(vi)*lin_resM_Du(idim*nsd_+jdim,ui);
          } // end for (idim)
        } // end for (jdim)
      } // end for (vi)
    } // end for (ui)
  }
  else
  {
    for (int ui=0; ui<nen_; ++ui)
    {
      for (int vi=0; vi<nen_; ++vi)
      {
        for (int idim = 0; idim <nsd_; ++idim)
        {
          estif_u(nsd_*vi+idim,nsd_*ui+idim) += funct_(vi)*lin_resM_Du(idim*nsd_+idim,ui);
        } // end for (idim)
      } //vi
    } // ui
  }

  // inertia terms on the right hand side for instationary fluids
  if (not fldparatimint_->IsStationary())
  {
    for (int idim = 0; idim <nsd_; ++idim)
    {
      if (fldparatimint_->IsGenalpha())
      {
        if(fldpara_->Tds()      ==INPAR::FLUID::subscales_time_dependent &&
           fldpara_->Transient()==INPAR::FLUID::inertia_stab_keep)
        {
          ;// do nothing here! Whole term already set in LinGalMomResU_subscales()
        }
        else
          resM_Du(idim)+=rhsfac*densam_*accint_(idim);
      }
      else
      {
        if(fldparatimint_->IsNewOSTImplementation())
        {
          //this approximates \int_{t_n}^{t_{n+1}} \rho du/dt
          //It could be implemented differently like (through integration by parts):
          // \rho_{n+1}u_{n+1}-\rho_{n}u_{n} -  \int_{t_n}^{t_{n+1}} d \rho/dt u
          //But leaves the second integral to be approximated.
          resM_Du(idim)+=fac_*dens_theta_*(velint_(idim)-velintn_(idim));
        }
        else
        {
          resM_Du(idim)+=fac_*densaf_*velint_(idim);
        }//end IsNewOSTImplementation
      }
    }
  }  // end if (not stationary)

  // convective terms of rhs
  for (int idim = 0; idim <nsd_; ++idim)
  {
    if(fldpara_->Tds()      ==INPAR::FLUID::subscales_time_dependent &&
       fldpara_->Transient()==INPAR::FLUID::inertia_stab_keep)
    {
      ;// do nothing here! Whole term already set in LinGalMomResU_subscales()
    }
    else
    {
      resM_Du(idim)+=rhsfac*densaf_*conv_old_(idim);
      if(fldparatimint_->IsNewOSTImplementation())
      {
        if (fldparatimint_->IsOneStepTheta())
        {
          resM_Du(idim)+=rhsfacn*densn_*conv_oldn_(idim);
        }
      }//end IsNewOSTImplementation
    }
  }  // end for(idim)

  if (fldpara_->Reaction())
  {
    for (int idim = 0; idim <nsd_; ++idim)
    {
      resM_Du(idim) += rhsfac*reacoeff_*velint_(idim);
      if(fldparatimint_->IsNewOSTImplementation())
      {
        if (fldparatimint_->IsOneStepTheta())
        {
          resM_Du(idim) += rhsfacn*reacoeff_*velintn_(idim);
        }
      }//end IsNewOSTImplementation
    }
  }  // end if (reaction_)

  for (int vi=0; vi<nen_; ++vi)
  {
    for(int idim = 0; idim <nsd_; ++idim)
    {
      velforce(idim,vi)-=resM_Du(idim)*funct_(vi);
    }
  }
  return;
}


template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::ViscousGalPart(
    LINALG::Matrix<nen_*nsd_,nen_*nsd_> &   estif_u,
    LINALG::Matrix<nsd_,nen_> &             velforce,
    LINALG::Matrix<nsd_,nsd_> &             viscstress,
    const double &                          timefacfac,
    const double &                          rhsfac,
    const double &                          rhsfacn)
{
  const double visceff_timefacfac = visceff_*timefacfac;

  /* viscosity term */
  /*
                   /                        \
                  |       /  \         / \   |
            2 mu  |  eps | Du | , eps | v |  |
                  |       \  /         \ /   |
                   \                        /
  */

  for (int vi=0; vi<nen_; ++vi)
  {
    for (int ui=0; ui<nen_; ++ui)
    {
      for (int jdim= 0; jdim<nsd_;++jdim)
      {
        const double temp=visceff_timefacfac*derxy_(jdim,vi);

        for (int idim = 0; idim <nsd_; ++idim)
        {
          estif_u(nsd_*vi+idim,nsd_*ui+jdim) += temp*derxy_(idim, ui);
          estif_u(nsd_*vi+idim,nsd_*ui+idim) += temp*derxy_(jdim, ui);
        } // end for (jdim)
      } // end for (idim)
    } // ui
  } //vi

  const double v = visceff_*rhsfac;
  const double vn = viscn_*rhsfacn;

  LINALG::Matrix<nsd_,nsd_> viscstressn(true);

  for (int jdim = 0; jdim < nsd_; ++jdim)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      viscstress(idim,jdim)=v*(vderxy_(jdim,idim)+vderxy_(idim,jdim));
      viscstressn(idim,jdim)=vn*(vderxyn_(jdim,idim)+vderxyn_(idim,jdim));
    }
  }

  // computation of right-hand-side viscosity term
  for (int vi=0; vi<nen_; ++vi)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        /* viscosity term on right-hand side */
        velforce(idim,vi)-= viscstress(idim,jdim)*derxy_(jdim,vi);
        if(fldparatimint_->IsNewOSTImplementation())
        {
          if (fldparatimint_->IsOneStepTheta())
            velforce(idim,vi)-= viscstressn(idim,jdim)*derxy_(jdim,vi);
        }//end IsNewOSTImplementation
      }
    }
  }


  return;
}

template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::ContStab(
    LINALG::Matrix<nen_*nsd_,nen_*nsd_> &     estif_u,
    LINALG::Matrix<nsd_,nen_> &               velforce,
    const double &                            timefac,
    const double &                            timefacfac,
    const double &                            timefacfacpre,
    const double &                            rhsfac,
    const double &                            rhsfacn)
{
  // In the case no continuity stabilization and no LOMA:
  // the factors 'conti_stab_and_vol_visc_fac' and 'conti_stab_and_vol_visc_rhs' are zero
  // therefore there is no contribution to the element stiffness matrix and
  // the viscous stress tensor is NOT altered!!
  //
  // ONLY
  // the rhs contribution of the viscous term is added!!

  double conti_stab_and_vol_visc_fac=0.0;
  double conti_stab_and_vol_visc_rhs=0.0;

  if (fldpara_->CStab())
  {
    if(not fldparatimint_->IsFullImplPressureAndCont())
    {
      conti_stab_and_vol_visc_fac+=timefacfacpre*tau_(2);
      conti_stab_and_vol_visc_rhs-=rhsfac*tau_(2)*conres_old_;
      if(fldparatimint_->IsNewOSTImplementation())
      {
        if (not fldparatimint_->IsImplPressure())
        {
          if (fldparatimint_->IsOneStepTheta())
          {
            conti_stab_and_vol_visc_rhs-=rhsfacn*tau_(2)*conres_oldn_;
          }
        }
      }//end IsNewOSTImplementation
    }
    else
    {
      // Full impl pressure weighted with \Delta t only.
      conti_stab_and_vol_visc_fac+=fac_*fldparatimint_->Dt()*tau_(2);
      conti_stab_and_vol_visc_rhs-=fac_*fldparatimint_->Dt()*tau_(2)*conres_old_;
    }
  }
  if (fldpara_->PhysicalType() == INPAR::FLUID::loma)
  {
    conti_stab_and_vol_visc_fac-=(2.0/3.0)*visceff_*timefacfac;
    conti_stab_and_vol_visc_rhs+=(2.0/3.0)*rhsfac*visceff_*vdiv_;
    // additional term q_sq_ for dynamics Smagorisnky only
    if (fldpara_->TurbModAction()==INPAR::FLUID::dynamic_smagorinsky
        or fldpara_->TurbModAction()==INPAR::FLUID::smagorinsky or fldpara_->TurbModAction()==INPAR::FLUID::vreman)
      conti_stab_and_vol_visc_rhs+=(2.0/3.0)*rhsfac*q_sq_;
  }

  /* continuity stabilisation on left-hand side */
  /*
              /                        \
             |                          |
        tauC | nabla o Du  , nabla o v  |
             |                          |
              \                        /
  */
  /* viscosity term - subtraction for low-Mach-number flow */
  /*
             /                             \             /                        \
            |  1                      / \   |     2 mu  |                          |
     - 2 mu |  - (nabla o u) I , eps | v |  | = - ----- | nabla o Du  , nabla o v  |
            |  3                      \ /   |       3   |                          |
             \                             /             \                        /
  */
  for (int ui=0; ui<nen_; ++ui)
  {
    const int fui = nsd_*ui;

    for (int idim = 0; idim <nsd_; ++idim)
    {
      const int fui_p_idim = fui+idim;
      const double v0 = conti_stab_and_vol_visc_fac*derxy_(idim,ui);
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = nsd_*vi;

        for(int jdim=0;jdim<nsd_;++jdim)
        {
          estif_u(fvi+jdim,fui_p_idim) += v0*derxy_(jdim, vi) ;
        }
      }
    } // end for(idim)
  }

  // computation of right-hand-side viscosity term
  for (int vi=0; vi<nen_; ++vi)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      /* viscosity term on right-hand side */
      velforce(idim,vi)+= conti_stab_and_vol_visc_rhs*derxy_(idim,vi);
    }
  }

  return;
}


template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::PressureGalPart(
    LINALG::Matrix<nen_*nsd_,nen_> &          estif_p_v,
    LINALG::Matrix<nsd_,nen_> &               velforce,
    const double &                            timefacfac,
    const double &                            timefacfacpre,
    const double &                            rhsfac,
    const double &                            rhsfacn,
    const double &                            press,
    const double &                            pressn)
{
  for (int ui=0; ui<nen_; ++ui)
  {
    double v;
    if(not fldparatimint_->IsFullImplPressureAndCont())
    {
      v = -timefacfacpre*funct_(ui);
    }
    else
    {
      v = -fldparatimint_->Dt()*fac_*funct_(ui);
    }

    for (int vi=0; vi<nen_; ++vi)
    {
      const int fvi = nsd_*vi;
      /* pressure term */
      /*
           /                \
          |                  |
          |  Dp , nabla o v  |
          |                  |
           \                /
      */
      for (int idim = 0; idim <nsd_; ++idim)
      {

        estif_p_v(fvi + idim,ui) += v*derxy_(idim, vi);

      }
    }
  }

  // pressure term on right-hand side
  if(not fldparatimint_->IsFullImplPressureAndCont())
  {
    velforce.Update(press*rhsfac,derxy_,1.0);
    if(fldparatimint_->IsNewOSTImplementation())
    {
      if (fldparatimint_->IsOneStepTheta())
      {
        velforce.Update(pressn*rhsfacn , derxy_ , 1.0);
      }
    }//end IsNewOSTImplementation
  }
  else{
//     Full impl pressure weighted with \Delta t.
    velforce.Update(fac_*fldparatimint_->Dt()*press,derxy_,1.0);

  }
  return;
}




template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::ContinuityGalPart(
    LINALG::Matrix<nen_, nen_*nsd_> &         estif_q_u,
    LINALG::Matrix<nen_,1> &                  preforce,
    const double &                            timefacfac,
    const double &                            timefacfacpre,
    const double &                            rhsfac,
    const double &                            rhsfacn)
{
  for (int vi=0; vi<nen_; ++vi)
  {
    double v;
    if(not fldparatimint_->IsFullImplPressureAndCont()){
      v = timefacfacpre*funct_(vi);
    }
    else{
      v = fac_*fldparatimint_->Dt()*funct_(vi);
    }

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui = nsd_*ui;

      for (int idim = 0; idim <nsd_; ++idim)
      {
        /* continuity term */
        /*
             /                \
            |                  |
            | nabla o Du  , q  |
            |                  |
             \                /
        */
        estif_q_u(vi,fui+idim) += v*derxy_(idim,ui);
      }
    }
  }  // end for(idim)
  if(not fldparatimint_->IsFullImplPressureAndCont())
  {
    preforce.Update(-rhsfac * vdiv_,funct_,1.0);
    if(fldparatimint_->IsNewOSTImplementation())
    {
      if (not fldparatimint_->IsImplPressure())
      {
        if (fldparatimint_->IsOneStepTheta())
        {
          preforce.Update(-rhsfacn * vdivn_,funct_,1.0);
        }
      }
    }//end IsNewOSTImplementation
  }
  else{
//     Full impl pressure weighted with \Delta t.
    preforce.Update(-fac_*fldparatimint_->Dt() * vdiv_,funct_,1.0);
  }
  return;
}


///*----------------------------------------------------------------------------*
// |  add integration point contribution to pressure matrices         nis Jan13 |
// *----------------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::PressureProjection(
//  LINALG::Matrix<nen_, nen_> &  ppmat
//  )
//{
//  // mass matrix of pressure basis functions - used as temp
//  ppmat.MultiplyNT(fac_, funct_, funct_, 1.0);
//
//  // "mass matrix" of projection modes - here element volume_equivalent_diameter_pc
//  D_ += fac_;
//
//  // prolongator(?) of projection
//  E_.Update(fac_, funct_, 1.0);
//}

///*----------------------------------------------------------------------------*
// |  finalize pressure projection and compute rhs-contribution       nis Jan13 |
// *----------------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::PressureProjectionFinalize(
//  LINALG::Matrix<nen_, nen_> & ppmat,
//  LINALG::Matrix<nen_, 1   > & preforce,
//  const LINALG::Matrix<nen_, 1   > & epre
//  )
//{
//  // check whether we deal with linear elements
//  if (not (distype==DRT::Element::hex8 or
//           distype==DRT::Element::tet4 or
//           distype==DRT::Element::quad4 or
//           distype==DRT::Element::tri3))
//  {
//    dserror("Polynomial pressure projection only implemented for linear elements so far.");
//  }
//
//  // compute difference of consistent and projection pressure mass matrices
//  ppmat.MultiplyNT(-1.0/D_,E_,E_,1.0);
//  ppmat.Scale(1.0/visc_);
//
//  // compute rhs-contribution
//  LINALG::Matrix<nen_,1> temp(false);
//  temp.Multiply(ppmat,epre);
//  preforce.Update(-fldparatimint_->TimeFacRhs(),temp,1.0);
//
//  // scale pressure-pressure matrix with pressure time factor
//  ppmat.Scale(fldparatimint_->TimeFacPre());
//}

template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::BodyForceRhsTerm(
    LINALG::Matrix<nsd_,nen_> &               velforce,
    const double &                            rhsfac,
    const double                              rhsfacn)
{
  for (int idim = 0; idim <nsd_; ++idim)
  {
    double scaled_rhsmom;
    if (fldparatimint_->IsGenalpha())
      scaled_rhsmom=rhsfac*rhsmom_(idim);
    else
    {
      if(fldparatimint_->IsNewOSTImplementation())
      {
        scaled_rhsmom=rhsfac*(densaf_*bodyforce_(idim)+generalbodyforce_(idim));
        if (fldparatimint_->IsOneStepTheta())
        {
          scaled_rhsmom+=rhsfacn*(densn_*bodyforcen_(idim)+generalbodyforcen_(idim));
        }
      }
      else
      {
        scaled_rhsmom=rhsfac*rhsmom_(idim);
      } // end IsNewOSTImplementation
    }

    for (int vi=0; vi<nen_; ++vi)
    {
      velforce(idim,vi)+=scaled_rhsmom*funct_(vi);
    }
  }  // end for(idim)

  return;
}


//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::ConservativeFormulation(
//    LINALG::Matrix<nen_*nsd_,nen_*nsd_> &     estif_u,
//    LINALG::Matrix<nsd_,nen_> &               velforce,
//    const double &                            timefacfac,
//    const double &                            rhsfac)
//{
//  //----------------------------------------------------------------------
//  // computation of additions to convection term (convective and
//  // reactive part) for conservative form of convection term including
//  // right-hand-side contribution
//  //----------------------------------------------------------------------
//
//  /* convection, convective part (conservative addition) */
//    /*
//      /                                                \
//      |      /              n+1    n+1           \      |
//      |  Du | rho*nabla o u    +  u   *nabla rho | , v  |
//      |      \             (i)     (i)          /       |
//      \                                                 /
//    */
//
//  if(fldparatimint_->IsNewOSTImplementation())
//  {
//    if (fldparatimint_->IsOneStepTheta())
//      dserror("Conservative formulation of Navier-Stokes eq. is not implemented for One Step Theta. Feel free to implement it if needed.");
//  }//end IsNewOSTImplementation
//
//    for (int idim = 0; idim <nsd_; ++idim)
//    {
//      // left hand side
//      {
//      // compute prefactor
//      double v = timefacfac*densaf_*vdiv_;
//      if (fldpara_->PhysicalType() == INPAR::FLUID::loma) v -= timefacfac*densaf_*scaconvfacaf_*conv_scaaf_;
//      else if(fldpara_->PhysicalType() == INPAR::FLUID::varying_density)
//      {
//        v += timefacfac*conv_scaaf_;
//        //         o
//        // (v, Du rho)
//        /*{
//          // interpolation to GP
//          double densdtngp = densdtn.Dot(funct_);
//          v += timefacfac*densdtngp;
//        }*/
//      }
//
//      for (int ui=0; ui<nen_; ++ui)
//      {
//        const int    fui   = nsd_*ui + idim;
//        const double v1 = v*funct_(ui);
//
//        for (int vi=0; vi<nen_; ++vi)
//        {
//          const int fvi   = nsd_*vi + idim;
//          estif_u(fvi  , fui  ) += funct_(vi)*v1;
//        }
//      }
//
//      /*  convection, reactive part (conservative addition) */
//      /*
//        /                              \
//        |  n+1  /               \      |
//        | u    | rho*nabla o Du | , v  |
//        |  (i)  \              /       |
//        \                             /
//      */
//
//      if (fldpara_->IsNewton())
//      {
//        const double v_idim = timefacfac*densaf_*velint_(idim);
//        for (int vi=0; vi<nen_; ++vi)
//        {
//          const int fvi   = nsd_*vi + idim;
//          const double v1_idim = v_idim*funct_(vi);
//
//          for (int ui=0; ui<nen_; ++ui)
//          {
//            const int fui   = nsd_*ui;
//
//            for(int jdim=0; jdim<nsd_;++jdim)
//              estif_u(fvi,  fui+jdim  ) += v1_idim*derxy_(jdim, ui);
//           }
//         }
//
//         /*  convection, reactive part (conservative addition) */
//         /*
//          /                           \
//          |  n+1  /             \      |
//          | u    | Du*nabla rho | , v  |
//          |  (i)  \            /       |
//          \                           /
//         */
//         if (fldpara_->PhysicalType() == INPAR::FLUID::loma
//             or fldpara_->PhysicalType() == INPAR::FLUID::varying_density)
//         {
//           double v_idim = 0.0;
//           if (fldpara_->PhysicalType() == INPAR::FLUID::loma)
//             v_idim = -timefacfac*densaf_*scaconvfacaf_*grad_scaaf_(idim)*velint_(idim);
//           else if (fldpara_->PhysicalType() == INPAR::FLUID::varying_density)
//             v_idim = +timefacfac*grad_scaaf_(idim)*velint_(idim);
//
//           for (int vi=0; vi<nen_; ++vi)
//           {
//             const int    fvi   = nsd_*vi + idim;
//             const double v1_idim = v_idim*funct_(vi);
//
//             for (int ui=0; ui<nen_; ++ui)
//             {
//               const int fui   = nsd_*ui;
//
//               for(int jdim=0;jdim<nsd_;++jdim)
//                 estif_u(fvi,  fui +jdim  ) += v1_idim*funct_(ui) ;
//              }
//            }
//          }
//        }
//      }
//
//      //right hand side
//      {
//        /* convection (conservative addition) on right-hand side */
//        double v = -rhsfac*densaf_*velint_(idim)*vdiv_;
//
//        if (fldpara_->PhysicalType() == INPAR::FLUID::loma)
//          v += rhsfac*velint_(idim)*densaf_*scaconvfacaf_*conv_scaaf_;
//        else if (fldpara_->PhysicalType() == INPAR::FLUID::varying_density)
//          v -= rhsfac*velint_(idim)*conv_scaaf_;
//
//         for (int vi=0; vi<nen_; ++vi)
//           velforce(idim, vi    ) += v*funct_(vi);
//      }
//    }  // end for(idim)
//
//  return;
//}


template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::PSPGOSTNew(
    LINALG::Matrix<nen_, nen_*nsd_> &         estif_q_u,
    LINALG::Matrix<nen_,nen_> &               ppmat,
    LINALG::Matrix<nen_,1> &                  preforce,
    LINALG::Matrix<nsd_*nsd_,nen_> &          lin_resM_Du,
    const double &                            fac3,
    const double &                            timefacfac,
    const double &                            timefacfacpre,
    const double &                            rhsfac,
    const int                                 iquad)
{
  // conservative, stabilization terms are neglected (Hughes)

  /* pressure stabilisation:                                            */
  /*
              /                 \
             |  ~n+af            |
           - |  u     , nabla q  |
             |                   |
              \                 /
  */

    double scal_grad_q = 0.0;

    if(fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic)
    {
      scal_grad_q=tau_(1);
    }
    else // time-dependent subgrid-scales
    {
      scal_grad_q=fac3;
    }

    /* pressure stabilisation: inertia if not stationary*/
    /*
              /                  \
             |                    |
             |  rho*Du , nabla q  |
             |                    |
              \                  /
    */
    /* pressure stabilisation: convection, convective part */
    /*
              /                                   \
             |  /       n+1       \                |
             | |   rho*u   o nabla | Du , nabla q  |
             |  \      (i)        /                |
              \                                   /
    */
    /* pressure stabilisation: convection, reactive part if Newton */
    /*
              /                                   \
             |  /                \   n+1           |
             | |   rho*Du o nabla | u     , grad q |
             |  \                /   (i)           |
              \                                   /
    */
    /* pressure stabilisation: reaction if included */
    /*
              /                     \
             |                      |
             |  sigma*Du , nabla q  |
             |                      |
              \                    /
    */
    /* pressure stabilisation: viscosity (-L_visc_u) */
    /*
              /                              \
             |               /  \             |
         mu  |  nabla o eps | Du | , nabla q  |
             |               \  /             |
              \                              /
    */

    if (is_higher_order_ele_ || fldpara_->IsNewton())
    {
      for (int ui=0; ui<nen_; ++ui)
      {
        for (int vi=0; vi<nen_; ++vi)
        {
          for(int idim=0;idim<nsd_;++idim)
          {
            const double temp_vi_idim=derxy_(idim,vi)*scal_grad_q;
            for(int jdim=0;jdim<nsd_;++jdim)
            {
              estif_q_u(vi,nsd_*ui+jdim) += lin_resM_Du(nsd_*idim+jdim,ui)*temp_vi_idim;
            } // jdim
          } // idim
        } // vi
      } // ui
    } // end if (is_higher_order_ele_) or (newton_)
    else
    {
      for (int vi=0; vi<nen_; ++vi)
      {
        for (int ui=0; ui<nen_; ++ui)
        {
          for(int idim=0;idim<nsd_;++idim)
          {
            estif_q_u(vi,nsd_*ui+idim) += lin_resM_Du(nsd_*idim+idim,ui)*derxy_(idim, vi)*scal_grad_q;
          } // vi
        } // ui
      } //idim
    } // end if not (is_higher_order_ele_) nor (newton_)


    for (int ui=0; ui<nen_; ++ui)
    {
          /* pressure stabilisation: pressure( L_pres_p) */
          /*
               /                    \
              |                      |
              |  nabla Dp , nabla q  |
              |                      |
               \                    /
          */
      for (int vi=0; vi<nen_; ++vi)
      {
        double sum = 0.;
        for (int idim = 0; idim < nsd_; ++idim)
          sum += derxy_(idim,ui) * derxy_(idim,vi);

        if(not fldparatimint_->IsFullImplPressureAndCont())
          ppmat(vi,ui) += timefacfacpre*scal_grad_q*sum;
        else{
          //Weighted with \Delta t
          ppmat(vi,ui) += fac_*fldparatimint_->Dt()*scal_grad_q*sum;
        }
      } // vi
    }  // ui

    for (int idim = 0; idim <nsd_; ++idim)
    {
      double sgvel = 0.0;
      if(fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic)
      {
        sgvel = sgvelint_(idim);
      }
      else // time-dependent subgrid-scales, Np_Genal_Alpha!
      {
        sgvel = (tds_->Svelnp())(idim,iquad);
      }
      const double temp = rhsfac*sgvel;

      for (int vi=0; vi<nen_; ++vi)
      {
        // pressure stabilisation
        preforce(vi) += temp*derxy_(idim, vi);
      }
    } // end for(idim)

  return;
}

template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::SUPGOSTNew(
    LINALG::Matrix<nen_*nsd_,nen_*nsd_> &     estif_u,
    LINALG::Matrix<nen_*nsd_,nen_> &          estif_p_v,
    LINALG::Matrix<nsd_,nen_> &               velforce,
    LINALG::Matrix<nen_,1> &                  preforce,
    LINALG::Matrix<nsd_*nsd_,nen_> &          lin_resM_Du,
    const double &                            fac3,
    const double &                            timefacfac,
    const double &                            timefacfacpre,
    const double &                            rhsfac)
{
  /*
                    /                                \
                   |  ~n+af    /     n+af       \     |
                 - |  u     , | rho*u    o nabla | v  |
                   |           \     (i)        /     |
                    \                                /
   */

  LINALG::Matrix<nsd_,1> temp;

  double supgfac;
  if(fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic)
    supgfac=densaf_*tau_(0);
  else
    supgfac=densaf_*fldparatimint_->AlphaF()*fac3;

  LINALG::Matrix<nen_,1> supg_test;
  for (int vi=0; vi<nen_; ++vi)
  {
    supg_test(vi)=supgfac*conv_c_(vi);
  }

  if(fldpara_->Reynolds() == INPAR::FLUID::reynolds_stress_stab)
  {
    for (int vi=0; vi<nen_; ++vi)
    {
      supg_test(vi)+=supgfac*sgconv_c_(vi);
    }
  }

     /* supg stabilisation: inertia if not stationary */
     /*
            /                                \
           |            /     n+1       \     |
           |  rho*Du , | rho*u   o nabla | v  |
           |            \     (i)       /     |
            \                                /
     */
     /* supg stabilisation: convective part ( L_conv_u) , convective term */
     /*
            /                                                     \
           |    /       n+1        \        /      n+1       \     |
           |   |   rho*u    o nabla | Du , | rho*u    o nabla | v  |
           |    \       (i)        /        \      (i)       /     |
            \                                                     /
     */
     /* supg stabilisation: convective part ( L_conv_u) , reactive term if Newton */
     /*
            /                                                     \
           |    /       n+1        \        /     n+1        \     |
           |   |   rho*u    o nabla | Du , | rho*u    o nabla | v  |
           |    \       (i)        /        \     (i)        /     |
            \                                                     /
     */
     /* supg stabilisation: reaction if included */
     /*
            /                                  \
           |              /     n+1       \     |
           |  sigma*Du , | rho*u   o nabla | v  |
           |              \     (i)       /     |
            \                                  /
     */
     /* supg stabilisation: viscous part  (-L_visc_u) if is_higher_order_ele_ */
     /*
            /                                              \
           |               /  \    /       n+1        \     |
           |  nabla o eps | Du |, |   rho*u    o nabla | v  |
           |               \  /    \       (i)        /     |
            \                                              /
     */

     /* supg stabilisation: inertia, linearisation of testfunction if Newton */
     /*
                 /                                       \
                |         n+1       /              \      |
                |    rho*u      ,  | rho*Du o nabla | v   |
                |         (i)       \              /      |
                 \                                       /
     */
     /* supg stabilisation: reactive part of convection and linearisation of testfunction ( L_conv_u) if Newton */
     /*
                 /                                                       \
                |    /       n+1        \   n+1     /              \      |
                |   |   rho*u    o nabla | u    ,  | rho*Du o nabla | v   |
                |    \       (i)        /   (i)     \              /      |
                 \                                                       /
     */
     /* supg stabilisation: reaction, linearisation of testfunction if Newton */
     /*
                 /                                         \
                |           n+1       /              \      |
                |    sigma*u      ,  | rho*Du o nabla | v   |
                |           (i)       \              /      |
                 \                                         /
     */
     /* supg stabilisation: pressure part, linearisation of test function if Newton ( L_pres_p) */
     /*
                /                                     \
               |         n+1    /                \     |
               |  nabla p    , |   rho*Du o nabla | v  |
               |         (i)    \                /     |
                \                                     /
     */
     /* supg stabilisation: viscous part, linearisation of test function if Newton (-L_visc_u) */
     /*
                /                                               \
               |               / n+1 \    /               \      |
               |  nabla o eps | u     |, |  rho*Du o nabla | v   |
               |               \ (i) /    \               /      |
                \                                               /
     */
     /* supg stabilisation: bodyforce part, linearisation of test function if Newton */
     /*
                /                                      \
               |                  /               \     |
               |  rho*rhsint   , |  rho*Du o nabla | v  |
               |                  \               /     |
                \                                      /
     */
     if (fldpara_->IsNewton())
     {
       if(fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic)
       {
         for(int jdim=0;jdim<nsd_;++jdim)
         {
           temp(jdim)=timefacfac*supgfac*momres_old_(jdim);
         }
       }
       else
       {
         for(int jdim=0;jdim<nsd_;++jdim)
         {
           temp(jdim)=-timefacfac*densaf_*sgvelint_(jdim);
         }
       }
       for (int vi=0; vi<nen_; ++vi)
       {
         for (int ui=0; ui<nen_; ++ui)
         {
           for(int idim=0;idim<nsd_;++idim)
           {
             const double w = temp(idim)*funct_(ui);
             for(int jdim=0;jdim<nsd_;++jdim)
             {
               estif_u(nsd_*vi+idim,nsd_*ui+jdim) += lin_resM_Du(nsd_*idim+jdim,ui)*supg_test(vi)
                   +derxy_(jdim,vi)*w;
             } // jdim
           } // vi
         } // ui
       } //idim
     } // end if (fldpara_->IsNewton())
     else if (is_higher_order_ele_)
     {
       for (int vi=0; vi<nen_; ++vi)
       {
         for (int ui=0; ui<nen_; ++ui)
         {
           for(int idim=0;idim<nsd_;++idim)
           {
             for(int jdim=0;jdim<nsd_;++jdim)
             {
               estif_u(nsd_*vi+idim,nsd_*ui+jdim) += lin_resM_Du(nsd_*idim+jdim,ui)*supg_test(vi);
             }
           }
         }
       }
     } // end if (is_higher_order_ele_)
     else
     {
       for (int vi=0; vi<nen_; ++vi)
       {
         for (int ui=0; ui<nen_; ++ui)
         {
           for(int idim=0;idim<nsd_;++idim)
           {
             estif_u(nsd_*vi+idim,nsd_*ui+idim) += lin_resM_Du(nsd_*idim+idim,ui)*supg_test(vi);
           } // ui
         } //idim
       } // vi
     } // end if not (is_higher_order_ele_) nor (newton_)

     /* supg stabilisation: pressure part  ( L_pres_p) */
     /*
              /                                    \
             |              /       n+1       \     |
             |  nabla Dp , |   rho*u   o nabla | v  |
             |              \       (i)       /     |
              \                                    /
     */
     for (int vi=0; vi<nen_; ++vi)
     {
       //       const double v = timefacfacpre*supg_test(vi);
       double v;
       if(not fldparatimint_->IsFullImplPressureAndCont())
       {
         v = timefacfacpre*supg_test(vi);
       }
       else
       {
         v = fldparatimint_->Dt()*fac_*supg_test(vi);
       }

       for (int ui=0; ui<nen_; ++ui)
       {

         for (int idim = 0; idim <nsd_; ++idim)
         {
           estif_p_v(nsd_*vi+idim,ui) += v*derxy_(idim, ui);
         }
       }
     }  // end for(idim)

     if(fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic)
     {
       for(int jdim=0;jdim<nsd_;++jdim)
       {
         temp(jdim)=rhsfac*momres_old_(jdim);
       }
     }
     else
     {
       for(int jdim=0;jdim<nsd_;++jdim)
       {
         temp(jdim)=-rhsfac*densaf_*sgvelint_(jdim)/(fac3*fldparatimint_->AlphaF());
       }
     }

     for (int idim = 0; idim <nsd_; ++idim)
     {
       for (int vi=0; vi<nen_; ++vi)
       {
         // supg stabilisation
         velforce(idim,vi) -= temp(idim)*supg_test(vi);
       }
     }  // end for(idim)

  // SUPG and Reynolds-stress term on right-hand side of
  // continuity equation for low-Mach-number flow
  if (fldpara_->PhysicalType() == INPAR::FLUID::loma and
      (fldpara_->ContiSUPG() or fldpara_->ContiReynolds() != INPAR::FLUID::reynolds_stress_stab_none))
  {
    const double temp_supg = rhsfac*scaconvfacaf_*sgscaint_;

    if(fldpara_->ContiSUPG())
    {
      for (int vi=0; vi<nen_; ++vi)
      {
        preforce(vi) -= temp_supg*conv_c_(vi);
      }
    }

    if (fldpara_->ContiReynolds() != INPAR::FLUID::reynolds_stress_stab_none)
    {
      for (int vi=0; vi<nen_; ++vi)
      {
        preforce(vi) -= temp_supg*sgconv_c_(vi);
      }
    }
  } // loma

  return;
}

//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::ReacStab(
//    LINALG::Matrix<nen_*nsd_,nen_*nsd_> &     estif_u,
//    LINALG::Matrix<nen_*nsd_,nen_> &          estif_p_v,
//    LINALG::Matrix<nsd_,nen_> &               velforce,
//    LINALG::Matrix<nsd_*nsd_,nen_> &          lin_resM_Du,
//    const double &                            timefacfac,
//    const double &                            timefacfacpre,
//    const double &                            rhsfac,
//    const double &                            fac3)
//{
//   double reac_tau;
//   if (fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic)
//     reac_tau = fldpara_->ViscReaStabFac()*reacoeff_*tau_(1);
//   else
//   {
//     dserror("Is this factor correct? Check for bugs!");
//     reac_tau = fldpara_->ViscReaStabFac()*reacoeff_*fldparatimint_->AlphaF()*fac3;
//   }
//
//
//   /* reactive stabilisation, inertia part if not stationary */
//   /*
//                /                    \
//               |                      |
//           -/+ |    rho*Du , sigma*v  |
//               |                      |
//                \                    /
//   */
//   /* reactive stabilisation, convective part, convective type */
//   /*
//              /                                  \
//             |  /       n+1       \               |
//         -/+ | |   rho*u   o nabla | Du , sigma*v |
//             |  \       (i)       /               |
//              \                                  /
//   */
//   /* reactive stabilisation, reactive part of convection */
//   /*
//              /                                   \
//             |  /                \   n+1           |
//         -/+ | |   rho*Du o nabla | u    , sigma*v |
//             |  \                /   (i)           |
//              \                                   /
//   */
//   /* reactive stabilisation, reaction part if included */
//   /*
//                /                      \
//               |                        |
//           -/+ |    sigma*Du , sigma*v  |
//               |                        |
//                \                      /
//   */
//   /* reactive stabilisation, viscous part (-L_visc_u) */
//   /*
//              /                             \
//             |               /  \            |
//        +/-  |  nabla o eps | Du | , sigma*v |
//             |               \  /            |
//              \                             /
//   */
//   if (is_higher_order_ele_ or fldpara_->IsNewton())
//   {
//     for (int vi=0; vi<nen_; ++vi)
//     {
//       const double v = reac_tau*funct_(vi);
//
//       for(int idim=0;idim<nsd_;++idim)
//       {
//         const int nsd_idim=nsd_*idim;
//
//         const int fvi_p_idim = nsd_*vi+idim;
//
//         for(int jdim=0;jdim<nsd_;++jdim)
//         {
//           const int nsd_idim_p_jdim=nsd_idim+jdim;
//
//           for (int ui=0; ui<nen_; ++ui)
//           {
//             const int fui_p_jdim   = nsd_*ui + jdim;
//
//             estif_u(fvi_p_idim,fui_p_jdim) += v*lin_resM_Du(nsd_idim_p_jdim,ui);
//           } // jdim
//         } // vi
//       } // ui
//     } //idim
//   } // end if (is_higher_order_ele_) or (newton_)
//   else
//   {
//     for (int vi=0; vi<nen_; ++vi)
//     {
//       const double v = reac_tau*funct_(vi);
//
//       for(int idim=0;idim<nsd_;++idim)
//       {
//         const int fvi_p_idim = nsd_*vi+idim;
//
//         const int nsd_idim=nsd_*idim;
//
//         for (int ui=0; ui<nen_; ++ui)
//         {
//           const int fui_p_idim   = nsd_*ui + idim;
//
//           estif_u(fvi_p_idim,fui_p_idim) += v*lin_resM_Du(nsd_idim+idim,ui);
//         } // ui
//       } //idim
//     } // vi
//   } // end if not (is_higher_order_ele_) nor (newton_)
//
//
//   /* reactive stabilisation, pressure part ( L_pres_p) */
//   /*
//              /                    \
//             |                      |
//        -/+  |  nabla Dp , sigma*v  |
//             |                      |
//              \                    /
//   */
//   const double reac_tau_timefacfacpre = reac_tau*timefacfacpre;
//   for (int vi=0; vi<nen_; ++vi)
//   {
//     const double v = reac_tau_timefacfacpre*funct_(vi);
//
//     for (int idim = 0; idim <nsd_; ++idim)
//     {
//       const int fvi = nsd_*vi + idim;
//
//       for (int ui=0; ui<nen_; ++ui)
//       {
//         estif_p_v(fvi,ui) += v*derxy_(idim, ui);
//       }
//     }
//   }  // end for(idim)
//
//   const double reac_fac = fldpara_->ViscReaStabFac()*rhsfac*reacoeff_;
//   for (int idim =0;idim<nsd_;++idim)
//   {
//     const double v = reac_fac*sgvelint_(idim);
//
//     for (int vi=0; vi<nen_; ++vi)
//     {
//         velforce(idim,vi) += v*funct_(vi);
//     }
//   } // end for(idim)
//
//  return;
//}


//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::ViscStab(
//    LINALG::Matrix<nen_*nsd_,nen_*nsd_> &     estif_u,
//    LINALG::Matrix<nen_*nsd_,nen_> &          estif_p_v,
//    LINALG::Matrix<nsd_,nen_> &               velforce,
//    LINALG::Matrix<nsd_*nsd_,nen_> &          lin_resM_Du,
//    const double &                            timefacfac,
//    const double &                            timefacfacpre,
//    const double &                            rhsfac,
//    const double &                            fac3)
//{
//  // preliminary parameter computation
//  double two_visc_tau;
//  if (fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic)
//    two_visc_tau = -fldpara_->ViscReaStabFac()*2.0*visc_*tau_(1);
//  else
//    two_visc_tau = -fldpara_->ViscReaStabFac()*2.0*visc_*fldparatimint_->AlphaF()*fac3;
//
//  /* viscous stabilisation, inertia part if not stationary */
//  /*
//                /                        \
//               |                          |
//           +/- |    rho*Du , div eps (v)  |
//               |                          |
//                \                        /
//  */
//  /* viscous stabilisation, convective part, convective type */
//  /*
//              /                                      \
//             |  /       n+1       \                   |
//         +/- | |   rho*u   o nabla | Du , div eps (v) |
//             |  \       (i)       /                   |
//              \                                      /
//  */
//  /* viscous stabilisation, reactive part of convection */
//  /*
//              /                                       \
//             |  /                \   n+1               |
//         +/- | |   rho*Du o nabla | u    , div eps (v) |
//             |  \                /   (i)               |
//              \                                       /
//  */
//  /* viscous stabilisation, reaction part if included */
//  /*
//                /                          \
//               |                            |
//           +/- |    sigma*Du , div eps (v)  |
//               |                            |
//                \                          /
//  */
//  /* viscous stabilisation, viscous part (-L_visc_u) */
//  /*
//              /                                 \
//             |               /  \                |
//        -/+  |  nabla o eps | Du | , div eps (v) |
//             |               \  /                |
//              \                                 /
//  */
//  for(int jdim=0;jdim<nsd_;++jdim)
//  {
//    for (int ui=0; ui<nen_; ++ui)
//    {
//      const int fui_p_jdim   = nsd_*ui + jdim;
//
//      for(int idim=0;idim<nsd_;++idim)
//      {
//        for(int kdim=0;kdim<nsd_;++kdim)
//        {
//          for (int vi=0; vi<nen_; ++vi)
//          {
//            const int fvi_p_idim = nsd_*vi+idim;
//
//            estif_u(fvi_p_idim,fui_p_jdim) += two_visc_tau*lin_resM_Du(nsd_*kdim+jdim,ui)*viscs2_(nsd_*idim+kdim,vi);
//          } // vi
//        } // kdim
//      } // idim
//    } // ui
//  } //jdim
//
//
//  /* viscous stabilisation, pressure part ( L_pres_p) */
//  /*
//              /                        \
//             |                          |
//        +/-  |  nabla Dp , div eps (v)  |
//             |                          |
//              \                        /
//  */
//  const double two_visc_tau_timefacfacpre = two_visc_tau*timefacfacpre;
//  for (int idim=0;idim<nsd_; ++idim)
//  {
//    for (int ui=0; ui<nen_; ++ui)
//    {
//      for (int vi=0; vi<nen_; ++vi)
//      {
//        for(int jdim=0;jdim<nsd_;++jdim)
//        {
//          estif_p_v(vi*nsd_+idim,ui) += two_visc_tau_timefacfacpre*derxy_(jdim, ui)*viscs2_(jdim+(idim*nsd_),vi);
//        }
//      }
//    }
//  } // end for(idim)
//
//  // viscous stabilization term on right-hand side
//  const double two_visc_fac = -fldpara_->ViscReaStabFac()*rhsfac*2.0*visc_;
//  for (int idim =0;idim<nsd_;++idim)
//  {
//    for (int vi=0; vi<nen_; ++vi)
//    {
//      /* viscous stabilisation */
//      for (int jdim=0;jdim<nsd_;++jdim)
//      {
//        velforce(idim,vi)+= two_visc_fac*sgvelint_(jdim)*viscs2_(jdim+(idim*nsd_),vi);
//      }
//    }
//  } // end for(idim)
//
//  return;
//}



//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::ConvDivStab(
//    LINALG::Matrix<nen_*nsd_,nen_*nsd_> &     estif_u,
//    LINALG::Matrix<nsd_,nen_> &               velforce,
//    const double &                            timefacfac,
//    const double &                            rhsfac)
//{
//
//  /* additional convective stabilization, when continuity is not satisfied*/
//  /*
//              /                           \
//          1  |                             |
//      +  --- |  (nabla o u)  Du  , v       |
//          2  |                             |
//              \                           /
//  */
//
//
//  // compute divergence of u
//  double divergence_timefacfac = 0.5*( vderxy_(0,0)+vderxy_(1,1)+vderxy_(2,2) )*timefacfac;
//  for (int vi=0; vi<nen_; ++vi)
//  {
//    for (int ui=0; ui<nen_; ++ui)
//    {
//      for(int ijdim=0; ijdim<nsd_; ijdim++)
//      {
//        const int fui_ijdim   = nsd_*ui + ijdim;
//        const int fvi_ijdim   = nsd_*vi + ijdim;
//
//        estif_u(fvi_ijdim,fui_ijdim) += divergence_timefacfac*funct_(vi)*funct_(ui);
//      }
//    }
//  }
//
//
//  for (int idim =0;idim<nsd_;++idim)
//  {
//    const double rhs_divergencefac = divergence_timefacfac*velint_(idim);
//
//    for (int vi=0; vi<nen_; ++vi)
//    {
//        velforce(idim,vi) -= rhs_divergencefac*funct_(vi);
//    }
//  } // end for(idim)
//
//
//  return;
//}


//
//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::CrossStressStab(
//    LINALG::Matrix<nen_*nsd_,nen_*nsd_> &     estif_u,
//    LINALG::Matrix<nen_*nsd_,nen_> &          estif_p_v,
//    LINALG::Matrix<nsd_,nen_> &               velforce,
//    LINALG::Matrix<nsd_*nsd_,nen_> &          lin_resM_Du,
//    const double &                            timefacfac,
//    const double &                            timefacfacpre,
//    const double &                            rhsfac,
//    const double &                            fac3)
//{
//  /*
//                               this part is linearised in
//                              combination with the standard
//                                  Galerkin term above
//                                          +----+
//                                          |    |
//                    /                                \
//                   |   /    ~n+af       \   n+af      |
//                 + |  | rho*u    o nabla | u     , v  |
//                   |   \     (i)        /   (i)       |
//                    \                                /
//                        |       |
//                        +-------+
//                     linearisation of
//                  this part is performed
//                     in the following
//
//   */
//
//     double crossfac;
//     if (fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic)
//          crossfac=densaf_*tau_(1);
//     else crossfac=densaf_*fldparatimint_->AlphaF()*fac3;
//
//     // Stabilization of lhs and the rhs
//     if (fldpara_->Cross() == INPAR::FLUID::cross_stress_stab and
//         fldpara_->IsNewton())
//     {
//       /*
//              /                         \
//             |  /          \   n+af      |
//             | | Du o nabla | u     , v  |
//             |  \          /             |
//              \                         /
//       */
//       /*
//              /                                              \
//             |  / / /          \   n+af \         \   n+af    |
//             | | | | Du o nabla | u      | o nabla | u   , v  |
//             |  \ \ \          /        /         /           |
//              \                                              /
//       */
//       /*
//              /                                               \
//             |  / / / n+af        \     \         \   n+af     |
//             | | | | u     o nabla | Du  | o nabla | u    , v  |
//             |  \ \ \             /     /         /            |
//              \                                               /
//       */
//       /*
//              /                               \
//             |  /                \   n+af      |
//             | | sigma*Du o nabla | u     , v  |
//             |  \                /             |
//              \                               /
//       */
//       /*
//              /                                             \
//             |  / /             /  \ \         \   n+af      |
//             | | | nabla o eps | Du | | o nabla | u     , v  |
//             |  \ \             \  / /         /             |
//              \                                             /
//       */
//       for(int jdim=0;jdim<nsd_;++jdim)
//       {
//         for (int ui=0; ui<nen_; ++ui)
//         {
//           const int fui_p_jdim   = nsd_*ui + jdim;
//
//           for(int idim=0;idim<nsd_;++idim)
//           {
//             for (int vi=0; vi<nen_; ++vi)
//             {
//               const int fvi_p_idim = nsd_*vi+idim;
//
//               for(int kdim=0;kdim<nsd_;++kdim)
//               {
//                 estif_u(fvi_p_idim,fui_p_jdim) -= crossfac*lin_resM_Du(nsd_*kdim+jdim,ui)*vderxy_(idim,kdim)*funct_(vi);
//               }
//             } // jdim
//           } // vi
//         } // ui
//       } //idim
//
//       /*
//                       /                               \
//                      |  /                \   n+af      |
//                      | | nabla Dp o nabla | u     , v  |
//                      |  \                /             |
//                       \                               /
//       */
//       for (int vi=0; vi<nen_; ++vi)
//       {
//         for (int idim = 0; idim <nsd_; ++idim)
//         {
//           const int fvi   = nsd_*vi + idim;
//
//           for (int ui=0; ui<nen_; ++ui)
//           {
//             for(int kdim=0;kdim<nsd_;++kdim)
//             {
//               estif_p_v(fvi,ui) -= crossfac*timefacfacpre*vderxy_(idim,kdim)*derxy_(kdim,ui)*funct_(vi);
//             }
//           }
//         }  // end for(idim)
//       } // vi
//     } // end if (cross_ == INPAR::FLUID::cross_stress_stab) and (is_newton)
//
//     // Stabilization only of the rhs
//     LINALG::Matrix<nsd_,1> temp;
//
//     temp.Clear();
//
//     for(int jdim=0;jdim<nsd_;++jdim)
//     {
//       for(int kdim=0;kdim<nsd_;++kdim)
//       {
//         temp(jdim)+=rhsfac*densaf_*sgvelint_(kdim)*vderxy_(jdim,kdim);
//       }
//     }
//
//     for (int idim = 0; idim <nsd_; ++idim)
//     {
//       for (int vi=0; vi<nen_; ++vi)
//       {
//         velforce(idim,vi) -= temp(idim)*funct_(vi);
//       }
//     }  // end for(idim)
//
//
//  return;
//}

//template <DRT::Element::DiscretizationType distype, DRT::ELEMENTS::Fluid::EnrichmentType enrtype>
//void DRT::ELEMENTS::FluidEleCalc<distype,enrtype>::ReynoldsStressStab(
//    LINALG::Matrix<nen_*nsd_,nen_*nsd_> &     estif_u,
//    LINALG::Matrix<nen_*nsd_,nen_> &          estif_p_v,
//    LINALG::Matrix<nsd_*nsd_,nen_> &          lin_resM_Du,
//    const double &                            timefacfac,
//    const double &                            timefacfacpre,
//    const double &                            fac3)
//{
//  /*
//                            linearisation of
//                         this part is performed
//                            in the following
//                                +--------+
//                                |        |
//                   /                                 \
//                  |  ~n+af     /    ~n+af       \     |
//                - |  u     ,  | rho*u    o nabla | v  |
//                  |   (i)      \     (i)        /     |
//                   \                                 /
//                     |   |
//                     +---+
//            this part is linearised
//          in combination with the SUPG
//                  term above
//
//  */
//
//  double reyfac;
//  if (fldpara_->Tds()==INPAR::FLUID::subscales_quasistatic)
//  {
//    //if(fldpara_->IsGenalpha())
//      reyfac=densaf_*tau_(1);
//    //else
//      // reyfac=densaf_*tau_(1)/fldpara_->Theta();
//  }
//  else reyfac=densaf_*fldparatimint_->AlphaF()*fac3;
//
//  /*
//          /                          \
//         |  ~n+af                     |
//         |  u     , ( Du o nabla ) v  |
//         |                            |
//          \                          /
//  */
//  /*
//          /                                                 \
//         |  ~n+af    / / / n+af        \     \         \     |
//         |  u     , | | | u     o nabla | Du  | o nabla | v  |
//         |           \ \ \             /     /         /     |
//          \                                                 /
//  */
//  /*
//          /                                                 \
//         |  ~n+af    / / /          \   n+af \         \     |
//         |  u     , | | | Du o nabla | u      | o nabla | v  |
//         |           \ \ \          /        /         /     |
//          \                                                 /
//  */
//  /*
//          /                                \
//         |  ~n+af                           |
//         |  u     , ( sigma*Du o nabla ) v  |
//         |                                  |
//          \                                /
//  */
//  /*
//          /                                               \
//         |  ~n+af    / /             /  \  \         \     |
//         |  u     , | | nabla o eps | Du |  | o nabla | v  |
//         |           \ \             \  /  /         /     |
//          \                                               /
//  */
//  for(int jdim=0;jdim<nsd_;++jdim)
//  {
//    for (int ui=0; ui<nen_; ++ui)
//    {
//      const int fui_p_jdim   = nsd_*ui + jdim;
//
//      for(int idim=0;idim<nsd_;++idim)
//      {
//        for (int vi=0; vi<nen_; ++vi)
//        {
//          const int fvi_p_idim = nsd_*vi+idim;
//
//          for(int kdim=0;kdim<nsd_;++kdim)
//          {
//            estif_u(fvi_p_idim,fui_p_jdim) += reyfac*lin_resM_Du(nsd_*kdim+jdim,ui)*sgvelint_(idim)*derxy_(kdim,vi);
//          }
//        } // jdim
//      } // vi
//    } // ui
//  } //idim
//
//  /*
//          /                                \
//         |  ~n+af    /                \     |
//         |  u     , | nabla Dp o nabla | v  |
//         |           \                /     |
//          \                                /
//  */
//  for (int vi=0; vi<nen_; ++vi)
//  {
//    for (int idim = 0; idim <nsd_; ++idim)
//    {
//      const int fvi_p_idim   = nsd_*vi + idim;
//
//      for (int ui=0; ui<nen_; ++ui)
//      {
//        for(int kdim=0;kdim<nsd_;++kdim)
//        {
//          estif_p_v(fvi_p_idim,ui) += reyfac*timefacfacpre*sgvelint_(idim)*derxy_(kdim,ui)*derxy_(kdim,vi);
//        }
//      }
//    }  // end for(idim)
//  } // vi
//
//  return;
//}


// template classes
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::hex8,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::hex8,DRT::ELEMENTS::Fluid::xwall>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::hex20,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::hex27,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::tet4,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::tet10,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::wedge6,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::pyramid5,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::quad4,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::quad8,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::quad9,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::tri3,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::tri6,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::nurbs9,DRT::ELEMENTS::Fluid::none>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::nurbs27,DRT::ELEMENTS::Fluid::none>;

