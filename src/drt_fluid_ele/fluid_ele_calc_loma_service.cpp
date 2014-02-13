/*----------------------------------------------------------------------*/
/*!
\file fluid_ele_calc_loma_service.cpp

\brief Low-Mach-number flow service routines for calculation of fluid
       element

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
#include "fluid_ele_parameter_timint.H"

#include "../drt_lib/drt_elementtype.H"

#include "../drt_mat/arrhenius_pv.H"
#include "../drt_mat/ferech_pv.H"
#include "../drt_mat/mixfrac.H"
#include "../drt_mat/sutherland.H"
#include "../drt_mat/yoghurt.H"



template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::FluidEleCalc<distype>::ComputeGalRHSContEq(
    const LINALG::Matrix<nsd_,nen_>&  eveln,
    const LINALG::Matrix<nen_,1>&     escaaf,
    const LINALG::Matrix<nen_,1>&     escaam,
    const LINALG::Matrix<nen_,1>&     escadtam,
    bool                              isale)
{
  //----------------------------------------------------------------------
  // compute additional Galerkin terms on right-hand side of continuity
  // equation (only required for variable-density flow at low Mach number)
  //----------------------------------------------------------------------
  /*

           /                                                dp   \
          |         1     / dT     /         \   \     1      th  |
          |    q , --- * | ---- + | u o nabla | T | - --- * ----  |
          |         T     \ dt     \         /   /    p      dt   |
           \                                           th        /
           +-----------------------------------------------------+
                           Galerkin part of rhscon_
  */

  // convective term (identical for all time-integration schemes,
  // while being the only component for stationary scheme)
  // gradient of scalar value at n+alpha_F/n+1
  grad_scaaf_.Multiply(derxy_,escaaf);

  // convective scalar term at n+alpha_F/n+1
  conv_scaaf_ = convvelint_.Dot(grad_scaaf_);

  // add to rhs of continuity equation
  rhscon_ = scaconvfacaf_*conv_scaaf_;

  // further terms different for general.-alpha and other time-int. schemes
  if (fldparatimint_->IsGenalpha())
  {
    // time derivative of scalar at n+alpha_M
    tder_sca_ = funct_.Dot(escadtam);

    // add to rhs of continuity equation
    rhscon_ += scadtfac_*tder_sca_ + thermpressadd_;
  }
  else
  {
    // instationary case
    if (not fldparatimint_->IsStationary())
    {
      // get velocity at n (including grid velocity in ALE case)
      convvelintn_.Multiply(eveln,funct_);
      if (isale) convvelintn_.Update(-1.0,gridvelint_,1.0);

      // get velocity derivatives at n
      vderxyn_.MultiplyNT(eveln,derxy_);

      // velocity divergence at n
      vdivn_ = 0.0;
      for (int idim = 0; idim<nsd_; ++idim)
      {
        vdivn_ += vderxyn_(idim,idim);
      }

      // scalar value at n+1
      scaaf_ = funct_.Dot(escaaf);

      // scalar value at n
      scan_ = funct_.Dot(escaam);

      // gradient of scalar value at n
      grad_scan_.Multiply(derxy_,escaam);

      // convective scalar term at n
      conv_scan_ = convvelintn_.Dot(grad_scan_);

      // add to rhs of continuity equation
      // (prepared for later multiplication by theta*dt in
      //  evaluation of element matrix and vector contributions)
      rhscon_ += (scadtfac_*(scaaf_-scan_)/fldparatimint_->Dt()
                + fldparatimint_->OmTheta()*(scaconvfacn_*conv_scan_-vdivn_)
                + thermpressadd_)/fldparatimint_->Theta();
    }
  }

  return;
}


template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::FluidEleCalc<distype>::ComputeGalRHSContEqArtComp(
    const LINALG::Matrix<nen_,1>&  epreaf,
    const LINALG::Matrix<nen_,1>&  epren,
    const LINALG::Matrix<nen_,1>&  escadtam)
{
  //----------------------------------------------------------------------
  // compute additional Galerkin terms on right-hand side of continuity
  // equation for artificial compressibility
  //----------------------------------------------------------------------
  /*

            /                      \
           |           1      dp   |
       -   |    q ,   --- *  ----  |
           |           c�    dt    |
            \                     /
            +----------------------+
            Galerkin part of rhscon_
  */

  // terms different for general.-alpha and other time-int. schemes
  if (fldparatimint_->IsGenalpha())
  {
    // time derivative of scalar (i.e., pressure in this case) at n+alpha_M
    tder_sca_ = funct_.Dot(escadtam);

    // add to rhs of continuity equation
    rhscon_ = -scadtfac_*tder_sca_;
  }
  else
  {
    // instationary case
    if (not fldparatimint_->IsStationary())
    {
      // scalar value (i.e., pressure in this case) at n+1
      scaaf_ = funct_.Dot(epreaf);

      // scalar value (i.e., pressure in this case) at n
      scan_ = funct_.Dot(epren);

      // add to rhs of continuity equation
      // (prepared for later multiplication by theta*dt in
      //  evaluation of element matrix and vector contributions)
      rhscon_ = -scadtfac_*(scaaf_-scan_)/(fldparatimint_->Dt()*fldparatimint_->Theta());
    }
  }

  return;
}


template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::FluidEleCalc<distype>::ComputeSubgridScaleScalar(
    const LINALG::Matrix<nen_,1>&             escaaf,
    const LINALG::Matrix<nen_,1>&             escaam)
{
  //----------------------------------------------------------------------
  // compute residual of scalar equation
  // -> different for generalized-alpha and other time-integration schemes
  // (only required for variable-density flow at low Mach number)
  //----------------------------------------------------------------------
  // define residual
  double scares_old = 0.0;

  // compute diffusive term at n+alpha_F/n+1 for higher-order elements
  LINALG::Matrix<nen_,1> diff;
  double diff_scaaf = 0.0;
  if (is_higher_order_ele_)
  {
    diff.Clear();
    // compute N,xx + N,yy + N,zz for each shape function
    for (int i=0; i<nen_; ++i)
    {
      for (int j = 0; j<nsd_; ++j)
      {
        diff(i) += derxy2_(j,i);
      }
    }
    diff.Scale(diffus_);
    diff_scaaf = diff.Dot(escaaf);
  }

  if (fldparatimint_->IsGenalpha())
    scares_old = densam_*tder_sca_+densaf_*conv_scaaf_-diff_scaaf-scarhs_;
  else
  {
    if (not fldparatimint_->IsStationary())
    {
      // compute diffusive term at n for higher-order elements
      double diff_scan = 0.0;
      if (is_higher_order_ele_) diff_scan = diff.Dot(escaam);

      scares_old = densaf_*(scaaf_-scan_)/fldparatimint_->Dt()
                  +fldparatimint_->Theta()*(densaf_*conv_scaaf_-diff_scaaf)
                  +fldparatimint_->OmTheta()*(densn_*conv_scan_-diff_scan)
                  -scarhs_;
    }
    else scares_old = densaf_*conv_scaaf_-diff_scaaf-scarhs_;
  }

  //----------------------------------------------------------------------
  // compute subgrid-scale part of scalar
  // (For simplicity, stabilization parameter tau_Mu is used here instead
  //  of exactly calculating the stabilization parameter tau for the scalar
  //  equation; differences should be minor for Prandtl numbers or ratios
  //  of viscosity and diffusivity (for mixture-fraction equation),
  //  respectively, close to one.)
  //----------------------------------------------------------------------
  sgscaint_ = -tau_(0)*scares_old;

  return;
}


/*----------------------------------------------------------------------*
 |  update material parameters including s.-s. part of scalar  vg 10/11 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::FluidEleCalc<distype>::UpdateMaterialParams(
  Teuchos::RCP<const MAT::Material>  material,
  const LINALG::Matrix<nsd_,nen_>&   evelaf,
  const LINALG::Matrix<nen_,1>&      escaaf,
  const LINALG::Matrix<nen_,1>&      escaam,
  const double                       thermpressaf,
  const double                       thermpressam,
  const double                       sgsca
)
{
if (material->MaterialType() == INPAR::MAT::m_mixfrac)
{
  const MAT::MixFrac* actmat = static_cast<const MAT::MixFrac*>(material.get());

  // compute mixture fraction at n+alpha_F or n+1
  double mixfracaf = funct_.Dot(escaaf);

  // add subgrid-scale part to obtain complete mixture fraction
  mixfracaf += sgsca;

  // compute dynamic viscosity at n+alpha_F or n+1 based on mixture fraction
  visc_ = actmat->ComputeViscosity(mixfracaf);

  // compute density at n+alpha_F or n+1 based on mixture fraction
  densaf_ = actmat->ComputeDensity(mixfracaf);

  // factor for convective scalar term at n+alpha_F or n+1
  scaconvfacaf_ = actmat->EosFacA()*densaf_;

  if (fldparatimint_->IsGenalpha())
  {
    // compute density at n+alpha_M based on mixture fraction
    double mixfracam = funct_.Dot(escaam);
    mixfracam += sgsca;
    densam_ = actmat->ComputeDensity(mixfracam);

    // factor for scalar time derivative at n+alpha_M
    scadtfac_ = actmat->EosFacA()*densam_;
  }
  else
  {
    // set density at n+1 at location n+alpha_M as well
    densam_ = densaf_;

    if (not fldparatimint_->IsStationary())
    {
      // compute density at n based on mixture fraction
      double mixfracn = funct_.Dot(escaam);
      mixfracn += sgsca;
      densn_ = actmat->ComputeDensity(mixfracn);

      // factor for convective scalar term at n
      scaconvfacn_ = actmat->EosFacA()*densn_;

      // factor for scalar time derivative
      scadtfac_ = scaconvfacaf_;
    }
  }
}
else if (material->MaterialType() == INPAR::MAT::m_sutherland)
{
  const MAT::Sutherland* actmat = static_cast<const MAT::Sutherland*>(material.get());

  // compute temperature at n+alpha_F or n+1
  double tempaf = funct_.Dot(escaaf);

  // add subgrid-scale part to obtain complete temperature
  tempaf += sgsca;

  // compute viscosity according to Sutherland law
  visc_ = actmat->ComputeViscosity(tempaf);

  // compute density at n+alpha_F or n+1 based on temperature
  // and thermodynamic pressure
  densaf_ = actmat->ComputeDensity(tempaf,thermpressaf);

  // factor for convective scalar term at n+alpha_F or n+1
  scaconvfacaf_ = 1.0/tempaf;

  if (fldparatimint_->IsGenalpha())
  {
    // compute temperature at n+alpha_M
    double tempam = funct_.Dot(escaam);

    // add subgrid-scale part to obtain complete temperature
    tempam += sgsca;

    // factor for scalar time derivative at n+alpha_M
    scadtfac_ = 1.0/tempam;

    // compute density at n+alpha_M based on temperature
    densam_ = actmat->ComputeDensity(tempam,thermpressam);
  }
  else
  {
    // set density at n+1 at location n+alpha_M as well
    densam_ = densaf_;

    if (not fldparatimint_->IsStationary())
    {
      // compute temperature at n
      double tempn = funct_.Dot(escaam);

      // add subgrid-scale part to obtain complete temperature
      tempn += sgsca;

      // compute density at n based on temperature at n and
      // (approximately) thermodynamic pressure at n+1
      densn_ = actmat->ComputeDensity(tempn,thermpressaf);

      // factor for convective scalar term at n
      scaconvfacn_ = 1.0/tempn;

      // factor for scalar time derivative
      scadtfac_ = scaconvfacaf_;
    }
  }
}
else if (material->MaterialType() == INPAR::MAT::m_arrhenius_pv)
{
  const MAT::ArrheniusPV* actmat = static_cast<const MAT::ArrheniusPV*>(material.get());

  // get progress variable at n+alpha_F or n+1
  double provaraf = funct_.Dot(escaaf);

  // add subgrid-scale part to obtain complete progress variable
  provaraf += sgsca;

  // compute temperature based on progress variable at n+alpha_F or n+1
  const double tempaf = actmat->ComputeTemperature(provaraf);

  // compute viscosity according to Sutherland law
  visc_ = actmat->ComputeViscosity(tempaf);

  // compute density at n+alpha_F or n+1 based on progress variable
  densaf_ = actmat->ComputeDensity(provaraf);

  // factor for convective scalar term at n+alpha_F or n+1
  scaconvfacaf_ = actmat->ComputeFactor(provaraf);

  if (fldparatimint_->IsGenalpha())
  {
    // compute density at n+alpha_M based on progress variable
    double provaram = funct_.Dot(escaam);
    provaram += sgsca;
    densam_ = actmat->ComputeDensity(provaram);

    // factor for scalar time derivative at n+alpha_M
    scadtfac_ = actmat->ComputeFactor(provaram);
  }
  else
  {
    // set density at n+1 at location n+alpha_M as well
    densam_ = densaf_;

    if (not fldparatimint_->IsStationary())
    {
      // compute density at n based on progress variable
      double provarn = funct_.Dot(escaam);
      provarn += sgsca;
      densn_ = actmat->ComputeDensity(provarn);

      // factor for convective scalar term at n
      scaconvfacn_ = actmat->ComputeFactor(provarn);

      // factor for scalar time derivative
      scadtfac_ = scaconvfacaf_;
    }
  }
}
else if (material->MaterialType() == INPAR::MAT::m_ferech_pv)
{
  const MAT::FerEchPV* actmat = static_cast<const MAT::FerEchPV*>(material.get());

  // get progress variable at n+alpha_F or n+1
  double provaraf = funct_.Dot(escaaf);

  // add subgrid-scale part to obtain complete progress variable
  provaraf += sgsca;

  // compute temperature based on progress variable at n+alpha_F or n+1
  const double tempaf = actmat->ComputeTemperature(provaraf);

  // compute viscosity according to Sutherland law
  visc_ = actmat->ComputeViscosity(tempaf);

  // compute density at n+alpha_F or n+1 based on progress variable
  densaf_ = actmat->ComputeDensity(provaraf);

  // factor for convective scalar term at n+alpha_F or n+1
  scaconvfacaf_ = actmat->ComputeFactor(provaraf);

  if (fldparatimint_->IsGenalpha())
  {
    // compute density at n+alpha_M based on progress variable
    double provaram = funct_.Dot(escaam);
    provaram += sgsca;
    densam_ = actmat->ComputeDensity(provaram);

    // factor for scalar time derivative at n+alpha_M
    scadtfac_ = actmat->ComputeFactor(provaram);
  }
  else
  {
    // set density at n+1 at location n+alpha_M as well
    densam_ = densaf_;

    if (not fldparatimint_->IsStationary())
    {
      // compute density at n based on progress variable
      double provarn = funct_.Dot(escaam);
      provarn += sgsca;
      densn_ = actmat->ComputeDensity(provarn);

      // factor for convective scalar term at n
      scaconvfacn_ = actmat->ComputeFactor(provarn);

      // factor for scalar time derivative
      scadtfac_ = scaconvfacaf_;
    }
  }
}
else if (material->MaterialType() == INPAR::MAT::m_yoghurt)
{
  const MAT::Yoghurt* actmat = static_cast<const MAT::Yoghurt*>(material.get());

  // get constant density
  densaf_ = actmat->Density();
  densam_ = densaf_;
  densn_  = densaf_;

  // compute temperature at n+alpha_F or n+1
  const double tempaf = funct_.Dot(escaaf);

  // compute rate of strain at n+alpha_F or n+1
  double rateofstrain = -1.0e30;
  rateofstrain = GetStrainRate(evelaf);

  // compute viscosity for Yoghurt-like flows according to Afonso et al. (2003)
  visc_ = actmat->ComputeViscosity(rateofstrain,tempaf);

  // compute diffusivity
  diffus_ = actmat->ComputeDiffusivity();
}
else dserror("Update of material parameters not required for this material type!");

return;
} // FluidEleCalc::UpdateMaterialParams



template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::FluidEleCalc<distype>::RecomputeGalAndComputeCrossRHSContEq()
{
  //----------------------------------------------------------------------
  // recompute Galerkin terms based on updated material parameters
  // including s.-s. part of scalar and compute cross-stress term on
  // right-hand side of continuity equation
  // (only required for variable-density flow at low Mach number)
  //----------------------------------------------------------------------
  /*

           /                                                       dp   \
          |         1     / dT     /               \   \     1      th  |
          |    q , --- * | ---- + | (u + �) o nabla | T | - --- * ----  |
          |         T     \ dt     \               /   /    p      dt   |
           \                                                 th        /
           +-----------------------------------------------------+
            Galerkin part of rhscon_ including cross-stress term
  */

  // add convective term to rhs of continuity equation
  // (identical for all time-integration schemes)
  rhscon_ = scaconvfacaf_*conv_scaaf_;

  // add (first) subgrid-scale-velocity part to rhs of continuity equation
  // (identical for all time-integration schemes)
  if (fldpara_->ContiCross() != INPAR::FLUID::cross_stress_stab_none)
  {
    rhscon_ += scaconvfacaf_*sgvelint_.Dot(grad_scaaf_);
  }

  if (fldpara_->MultiFracLomaConti())
  {
    rhscon_ += scaconvfacaf_*mffsvelint_.Dot(grad_scaaf_); // first cross-stress term
    rhscon_ += scaconvfacaf_*velint_.Dot(grad_fsscaaf_); // second cross-stress term
    rhscon_ += scaconvfacaf_*mffsvelint_.Dot(grad_fsscaaf_); // Reynolds-stress term
//    rhscon_ -= mffsvdiv_; // multifractal divergence
  }

  // further terms different for general.-alpha and other time-int. schemes
  if (fldparatimint_->IsGenalpha())
  {
    // add to rhs of continuity equation
    rhscon_ += scadtfac_*tder_sca_ + thermpressadd_;
  }
  else
  {
    // instationary case
    if (not fldparatimint_->IsStationary())
    {
      // add to rhs of continuity equation
      // (prepared for later multiplication by theta*dt in
      //  evaluation of element matrix and vector contributions)
      rhscon_ += (scadtfac_*(scaaf_-scan_)/fldparatimint_->Dt()
                + fldparatimint_->OmTheta()*(scaconvfacn_*conv_scan_-vdivn_)
                + thermpressadd_)/fldparatimint_->Theta();

      // add second subgrid-scale-velocity part to rhs of continuity equation
      // (subgrid-scale velocity at n+1 also approximately used at n)
      if (fldpara_->Cross() != INPAR::FLUID::cross_stress_stab_none)
          rhscon_ += (fldparatimint_->OmTheta()/fldparatimint_->Theta())
                     *scaconvfacn_*sgvelint_.Dot(grad_scan_);
    }
  }

  return;
}


template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::FluidEleCalc<distype>::LomaGalPart(
    LINALG::Matrix<nen_, nen_*nsd_> &       estif_q_u,
    LINALG::Matrix<nen_,1> &                preforce,
    const double &                          timefacfac,
    const double &                          rhsfac)
{
  //----------------------------------------------------------------------
  // computation of additional terms for low-Mach-number flow:
  // 2) additional rhs term of continuity equation
  //----------------------------------------------------------------------

  if (fldpara_->IsNewton())
  {
    const double timefacfac_scaconvfacaf=timefacfac*scaconvfacaf_;

    for (int ui=0; ui<nen_; ++ui)
    {
      const int fui=nsd_*ui;

      const double timefacfac_scaconvfacaf_funct_ui=timefacfac_scaconvfacaf*funct_(ui);

      for(int jdim=0;jdim<nsd_;++jdim)
      {
        const double temp=timefacfac_scaconvfacaf_funct_ui*grad_scaaf_(jdim);

        for (int vi=0; vi<nen_; ++vi)
        {
          //const int fvippp= numdofpernode_*vi+nsd_;


          /*
              factor afgtd/am

                      /                    \
                1    |       /         \    |
               --- * |  q , | Du o grad | T |
                T    |       \         /    |
                      \                    /
          */
          estif_q_u(vi,fui+jdim) -= temp*funct_(vi);
        }
      }
    }
  } // end if (is_newton_)

  const double rhsfac_rhscon = rhsfac*rhscon_;
  for (int vi=0; vi<nen_; ++vi)
  {
    /* additional rhs term of continuity equation */
    preforce(vi) += rhsfac_rhscon*funct_(vi) ;
  }

  return;
}



template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::FluidEleCalc<distype>::StabLinGalMomResU(
    LINALG::Matrix<nsd_*nsd_,nen_> &          lin_resM_Du,
    const double &                            timefacfac)
{

  /*
                 /       n+1       \        /                \  n+1
       rho*Du + |   rho*u   o nabla | Du + |   rho*Du o nabla | u   +
                 \      (i)        /        \                /   (i)

                               /  \
     + sigma*Du + nabla o eps | Du |
                               \  /
  */
  if(fldpara_->Tds()==INPAR::FLUID::subscales_time_dependent
     ||
     fldpara_->Cross()==INPAR::FLUID::cross_stress_stab)
  {
    //----------------------------------------------------------------------
    /* GALERKIN residual was rescaled and cannot be reused; so rebuild it */

    lin_resM_Du.Clear();

    int idim_nsd_p_idim[nsd_];

    for (int idim = 0; idim <nsd_; ++idim)
    {
      idim_nsd_p_idim[idim]=idim*nsd_+idim;
    }

    if (fldparatimint_->IsStationary() == false)
    {
      const double fac_densam=fac_*densam_;

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

    for (int ui=0; ui<nen_; ++ui)
    {
      // deleted +sgconv_c_(ui)
      const double v=timefacfac_densaf*conv_c_(ui);

      for (int idim = 0; idim <nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim],ui)+=v;
      }
    }

    if (fldpara_->IsNewton())
    {
//
//
// dr_j   d    /    du_j \          du_j         dN_B
// ----= ---- | u_i*----  | = N_B * ---- + u_i * ---- * d_jk
// du_k  du_k  \    dx_i /          dx_k         dx_i

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
  }

  if (is_higher_order_ele_)
  {
    const double v = -2.0*visceff_*timefacfac;
    for (int idim = 0; idim <nsd_; ++idim)
    {
      const int nsd_idim=nsd_*idim;

      for(int jdim=0;jdim<nsd_;++jdim)
      {
        const int nsd_idim_p_jdim=nsd_idim+jdim;

        for (int ui=0; ui<nen_; ++ui)
        {
          lin_resM_Du(nsd_idim_p_jdim,ui)+=v*viscs2_(nsd_idim_p_jdim, ui);
        }
      }
    }
  }

  return;
}


template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::FluidEleCalc<distype>::ArtCompPressureInertiaGalPartandContStab(
    LINALG::Matrix<nen_*nsd_,nen_> &        estif_p_v,
    LINALG::Matrix<nen_,nen_> &             ppmat)
{
  /* pressure inertia term if not is_stationary */
  /*
            /             \
           |   1           |
           |  ---  Dp , q  |
           | beta�         |
            \             /
  */
  double prefac = scadtfac_*fac_;
  for (int ui=0; ui<nen_; ++ui)
  {
    for (int vi=0; vi<nen_; ++vi)
    {
      ppmat(vi,ui) += prefac*funct_(ui)*funct_(vi);
    } // vi
  }  // ui

  if (fldpara_->CStab())
  {
    /* continuity stabilisation on left-hand side for artificial compressibility */
    /*
                /                      \
               |   1                   |
          tauC |  --- Dp  , nabla o v  |
               |   c�                  |
                \                     /
    */

    prefac *= tau_(2);
    
    for (int ui=0; ui<nen_; ++ui)
    {
      for (int vi=0; vi<nen_; ++vi)
      {
        const int fvi = nsd_*vi;

        for(int jdim=0;jdim<nsd_;++jdim)
        {
          estif_p_v(fvi+jdim,ui) += prefac*funct_(ui)*derxy_(jdim,vi) ;
        }
      }
    }
  }

  return;
}



// Ursula is responsible for this comment!
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::hex8>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::hex20>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::hex27>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::tet4>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::tet10>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::wedge6>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::pyramid5>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::quad4>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::quad8>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::quad9>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::tri3>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::tri6>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::nurbs9>;
template class DRT::ELEMENTS::FluidEleCalc<DRT::Element::nurbs27>;
