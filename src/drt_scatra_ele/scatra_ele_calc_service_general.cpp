/*----------------------------------------------------------------------*/
/*!
\file scatra_ele_calc_service_general.cpp

\brief Internal implementation of ScaTra element

<pre>
Maintainer: Andreas Ehrl
            ehrl@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
*/
/*----------------------------------------------------------------------*/

#include "scatra_ele_calc.H"

#include "scatra_ele.H"
#include "scatra_ele_action.H"

#include "scatra_ele_parameter.H"
#include "scatra_ele_parameter_std.H"
#include "scatra_ele_parameter_timint.H"

#include "../drt_lib/drt_utils.H"
#include "../drt_nurbs_discret/drt_nurbs_utils.H"
#include "../drt_geometry/position_array.H"

#include "../drt_lib/standardtypes_cpp.H"  // for EPS13 and so on
#include "../drt_lib/drt_globalproblem.H"


/*----------------------------------------------------------------------*
 | evaluate action                                           fang 02/15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
int DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::EvaluateAction(
    DRT::Element*               ele,
    Teuchos::ParameterList&     params,
    DRT::Discretization&        discretization,
    const SCATRA::Action&       action,
    const std::vector<int> &    lm,
    Epetra_SerialDenseMatrix&   elemat1_epetra,
    Epetra_SerialDenseMatrix&   elemat2_epetra,
    Epetra_SerialDenseVector&   elevec1_epetra,
    Epetra_SerialDenseVector&   elevec2_epetra,
    Epetra_SerialDenseVector&   elevec3_epetra
    )
{
  // determine and evaluate action
  switch(action)
  {
  // calculate time derivative for time value t_0
  case SCATRA::calc_initial_time_deriv:
  {
    // calculate matrix and rhs
    CalcInitialTimeDerivative(
      ele,
      elemat1_epetra,
      elevec1_epetra,
      params,
      discretization,
      lm
      );
    break;
  }
  case SCATRA::integrate_shape_functions:
  {
    // calculate integral of shape functions
    const Epetra_IntSerialDenseVector dofids = params.get<Epetra_IntSerialDenseVector>("dofids");
    IntegrateShapeFunctions(ele,elevec1_epetra,dofids);

    break;
  }
  case SCATRA::calc_flux_domain:
  {
    // get velocity values at the nodes
    const Teuchos::RCP<Epetra_MultiVector> velocity = params.get< Teuchos::RCP<Epetra_MultiVector> >("velocity field");
    DRT::UTILS::ExtractMyNodeBasedValues(ele,evelnp_,velocity,nsd_);
    const Teuchos::RCP<Epetra_MultiVector> convelocity = params.get< Teuchos::RCP<Epetra_MultiVector> >("convective velocity field");
    DRT::UTILS::ExtractMyNodeBasedValues(ele,econvelnp_,convelocity,nsd_);

    // need current values of transported scalar
    // -> extract local values from global vectors
    Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (phinp==Teuchos::null) dserror("Cannot get state vector 'phinp'");
    std::vector<double> myphinp(lm.size());
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

    // fill all element arrays
    for (int i=0;i<nen_;++i)
    {
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        ephinp_[k](i,0) = myphinp[k+(i*numdofpernode_)];
      }
    } // for i

    // access control parameter for flux calculation
    INPAR::SCATRA::FluxType fluxtype = scatrapara_->WriteFlux();
    Teuchos::RCP<std::vector<int> > writefluxids = scatrapara_->WriteFluxIds();

    // we always get an 3D flux vector for each node
    LINALG::Matrix<3,nen_> eflux(true);

    // do a loop for systems of transported scalars
    for (std::vector<int>::iterator it = writefluxids->begin(); it!=writefluxids->end(); ++it)
    {
      int k = (*it)-1;
      // calculate flux vectors for actual scalar
      eflux.Clear();
      CalculateFlux(eflux,ele,fluxtype,k);
      // assembly
      for (int inode=0;inode<nen_;inode++)
      {
        const int fvi = inode*numdofpernode_+k;
        elevec1_epetra[fvi]+=eflux(0,inode);
        elevec2_epetra[fvi]+=eflux(1,inode);
        elevec3_epetra[fvi]+=eflux(2,inode);
      }
    } // loop over numscal

    break;
  }
  case SCATRA::calc_mean_scalars:
  {
    // get flag for inverting
    bool inverting = params.get<bool>("inverting");

    // need current scalar vector
    // -> extract local values from the global vectors
    Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (phinp==Teuchos::null) dserror("Cannot get state vector 'phinp'");
    std::vector<double> myphinp(lm.size());
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

    // fill all element arrays
    for (int i=0;i<nen_;++i)
    {
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        ephinp_[k](i,0) = myphinp[k+(i*numdofpernode_)];
      }
    } // for i

    // calculate scalars and domain integral
    CalculateScalars(ele,elevec1_epetra,inverting);

    break;
  }
  // calculate filtered fields for calculation of turbulent Prandtl number
  // required for dynamic Smagorinsky model in scatra
  case SCATRA::calc_scatra_box_filter:
  {
    if (nsd_ == 3)
      CalcBoxFilter(ele,params,discretization,lm);
    else
      dserror("action 'calc_scatra_box_filter' is 3D specific action");

    break;
  }
  // calculate turbulent prandtl number of dynamic Smagorinsky model
  case SCATRA::calc_turbulent_prandtl_number:
  {
    if (nsd_ == 3)
    {
      // get required quantities, set in dynamic Smagorinsky class
      Teuchos::RCP<Epetra_MultiVector> col_filtered_vel =
        params.get<Teuchos::RCP<Epetra_MultiVector> >("col_filtered_vel");
      Teuchos::RCP<Epetra_MultiVector> col_filtered_dens_vel =
        params.get<Teuchos::RCP<Epetra_MultiVector> >("col_filtered_dens_vel");
      Teuchos::RCP<Epetra_MultiVector> col_filtered_dens_vel_temp =
        params.get<Teuchos::RCP<Epetra_MultiVector> >("col_filtered_dens_vel_temp");
      Teuchos::RCP<Epetra_MultiVector> col_filtered_dens_rateofstrain_temp =
        params.get<Teuchos::RCP<Epetra_MultiVector> >("col_filtered_dens_rateofstrain_temp");
      Teuchos::RCP<Epetra_Vector> col_filtered_temp =
        params.get<Teuchos::RCP<Epetra_Vector> >("col_filtered_temp");
      Teuchos::RCP<Epetra_Vector> col_filtered_dens =
        params.get<Teuchos::RCP<Epetra_Vector> >("col_filtered_dens");
      Teuchos::RCP<Epetra_Vector> col_filtered_dens_temp =
        params.get<Teuchos::RCP<Epetra_Vector> >("col_filtered_dens_temp");

      // initialize variables to calculate
      double LkMk   = 0.0;
      double MkMk   = 0.0;
      double xcenter  = 0.0;
      double ycenter  = 0.0;
      double zcenter  = 0.0;

      // calculate LkMk and MkMk
      switch (distype)
      {
      case DRT::Element::hex8:
      {
        scatra_calc_smag_const_LkMk_and_MkMk(
            col_filtered_vel                   ,
            col_filtered_dens_vel              ,
            col_filtered_dens_vel_temp         ,
            col_filtered_dens_rateofstrain_temp,
            col_filtered_temp                  ,
            col_filtered_dens                  ,
            col_filtered_dens_temp             ,
            LkMk                               ,
            MkMk                               ,
            xcenter                            ,
            ycenter                            ,
            zcenter                            ,
            ele                                );
        break;
      }
      default:
      {
        dserror("Unknown element type for box filter application\n");
      }
      }

      // set Prt without averaging (only clipping)
      // calculate inverse of turbulent Prandtl number times (C_s*Delta)^2
      double inv_Prt = 0.0;
      if (abs(MkMk) < 1E-16)
      {
        //std::cout << "warning: abs(MkMk) < 1E-16 -> set inverse of turbulent Prandtl number to zero!"  << std::endl;
        inv_Prt = 0.0;
      }
      else  inv_Prt = LkMk / MkMk;
      if (inv_Prt<0.0)
        inv_Prt = 0.0;

      // set all values in parameter list
      params.set<double>("LkMk",LkMk);
      params.set<double>("MkMk",MkMk);
      params.set<double>("xcenter",xcenter);
      params.set<double>("ycenter",ycenter);
      params.set<double>("zcenter",zcenter);
      params.set<double>("ele_Prt",inv_Prt);
    }
    else dserror("action 'calc_turbulent_prandtl_number' is a 3D specific action");

    break;
  }
  case SCATRA::calc_vreman_scatra:
  {
    if (nsd_ == 3)
    {

      Teuchos::RCP<Epetra_MultiVector> col_filtered_phi =
        params.get<Teuchos::RCP<Epetra_MultiVector> >("col_filtered_phi");
      Teuchos::RCP<Epetra_Vector> col_filtered_phi2 =
        params.get<Teuchos::RCP<Epetra_Vector> >("col_filtered_phi2");
      Teuchos::RCP<Epetra_Vector> col_filtered_phiexpression =
        params.get<Teuchos::RCP<Epetra_Vector> >("col_filtered_phiexpression");
      Teuchos::RCP<Epetra_MultiVector> col_filtered_alphaijsc =
        params.get<Teuchos::RCP<Epetra_MultiVector> >("col_filtered_alphaijsc");

      // initialize variables to calculate
      double dt_numerator=0.0;
      double dt_denominator=0.0;

      // calculate LkMk and MkMk
      switch (distype)
      {
      case DRT::Element::hex8:
      {
        scatra_calc_vreman_dt(
            col_filtered_phi                   ,
            col_filtered_phi2              ,
            col_filtered_phiexpression         ,
            col_filtered_alphaijsc,
            dt_numerator,
            dt_denominator,
            ele                                );
        break;
      }
      default:
      {
        dserror("Unknown element type for vreman scatra application\n");
      }
      break;
      }

      elevec1_epetra(0)=dt_numerator;
      elevec1_epetra(1)=dt_denominator;
    }
    else dserror("action 'calc_vreman_scatra' is a 3D specific action");


    break;
  }
  // calculate normalized subgrid-diffusivity matrix
  case SCATRA::calc_subgrid_diffusivity_matrix:
  {
    // calculate mass matrix and rhs
    CalcSubgrDiffMatrix(
      ele,
      elemat1_epetra);

    break;
  }
  // calculate mean Cai of multifractal subgrid-scale modeling approach
  case SCATRA::calc_mean_Cai:
  {
    // get nodel velocites
    const Teuchos::RCP<Epetra_MultiVector> convelocity = params.get< Teuchos::RCP<Epetra_MultiVector> >("convective velocity field");
    DRT::UTILS::ExtractMyNodeBasedValues(ele,econvelnp_,convelocity,nsd_);
    // get phi for material parameters
    Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (phinp==Teuchos::null)
      dserror("Cannot get state vector 'phinp'");
    std::vector<double> myphinp(lm.size());
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);
    // fill all element arrays
    for (int i=0;i<nen_;++i)
    {
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        ephinp_[k](i,0) = myphinp[k+(i*numdofpernode_)];
      }
    }

    if (scatrapara_->TurbModel() != INPAR::FLUID::multifractal_subgrid_scales)
      dserror("Multifractal_Subgrid_Scales expected");

    double Cai = 0.0;
    double vol = 0.0;
    // calculate Cai and volume, do not include elements of potential inflow section
    if (scatrapara_->AdaptCsgsPhi() and scatrapara_->Nwl() and (not SCATRA::InflowElement(ele)))
    {
      // use one-point Gauss rule to do calculations at the element center
      DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToStabGaussRule<distype>::rule);
      vol = EvalShapeFuncAndDerivsAtIntPoint(intpoints,0);

      // adopt integration points and weights for gauss point evaluation of B
      if (scatrapara_->BD_Gp())
      {
        const DRT::UTILS::IntPointsAndWeights<nsd_ele_> gauss_intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);
        intpoints = gauss_intpoints;
      }

      for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
      {
        const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad);

        // density at t_(n)
        double densn(1.0);
        // density at t_(n+1) or t_(n+alpha_F)
        double densnp(1.0);
        // density at t_(n+alpha_M)
        double densam(1.0);

        // diffusivity / diffusivities (in case of systems) or (thermal conductivity/specific heat) in case of loma

        diffmanager_ = Teuchos::rcp(new ScaTraEleDiffManager(numscal_));

        // fluid viscosity
        double visc(0.0);

        // get material
        GetMaterialParams(ele,densn,densnp,densam,visc);

        // get velocity at integration point
        LINALG::Matrix<nsd_,1> convelint(true);
        convelint.Multiply(econvelnp_,funct_);

        // calculate characteristic element length
        double hk = CalcRefLength(vol,convelint);

        // estimate norm of strain rate
        double strainnorm = GetStrainRate(econvelnp_);
        strainnorm /= sqrt(2.0);

        // get Re from strain rate
        double Re_ele_str = strainnorm * hk * hk * densnp / visc;
        if (Re_ele_str < 0.0) dserror("Something went wrong!");
        // ensure positive values
        if (Re_ele_str < 1.0)
          Re_ele_str = 1.0;

        // calculate corrected Cai
        //           -3/16
        //  =(1 - (Re)   )
        //
        Cai += (1.0-pow(Re_ele_str,-3.0/16.0)) * fac;
      }
    }

    // hand down the Cai and volume contribution to the time integration algorithm
    params.set<double>("Cai_int",Cai);
    params.set<double>("ele_vol",vol);

    break;
  }
  // calculate dissipation introduced by stabilization and turbulence models
  case SCATRA::calc_dissipation:
  {
    CalcDissipation(params,
                    ele,
                    discretization,
                    lm);
    break;
  }
  case SCATRA::calc_integr_grad_reac:
  {
    Teuchos::RCP<const Epetra_Vector> dualphi = discretization.GetState("dual phi");
    Teuchos::RCP<const Epetra_Vector> psi     = discretization.GetState("psi");
    if(dualphi==Teuchos::null || psi==Teuchos::null) dserror("Cannot get state vector 'dual phi' or 'psi' in action calc_integr_grad_reac");
    std::vector<double> mydualphi(lm.size());
    std::vector<double> mypsi(lm.size());
    DRT::UTILS::ExtractMyValues(*dualphi,mydualphi,lm);
    DRT::UTILS::ExtractMyValues(*psi,mypsi,lm);

    bool sign = params.get<bool>("signum_mu");
    double sign_fac = 1.0;
    if(sign) sign_fac = -1.0;

    // compute mass matrix
    Epetra_SerialDenseMatrix massmat;
    massmat.Shape(nen_,nen_);
    const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);
    double area = 0.0;
    for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
    {
      const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad);
      area += fac;
      CalcMatMass(massmat,0,fac,1.0);
    }

    Epetra_SerialDenseVector temp(nen_);
    for(int i=0;i<nen_;++i) temp(i) = mydualphi[i];
    elevec1_epetra.Multiply('N','N',sign_fac,massmat,temp,0.0);

    for(int i=0;i<nen_;++i) temp(i) = mypsi[i];

    bool scaleele = params.get<bool>("scaleele");
    if(scaleele) elevec1_epetra += temp;
    else         elevec1_epetra.Multiply('N','N',1.0,massmat,temp,1.0);
    break;
  }
  case SCATRA::recon_gradients_at_nodes:
  {
    // need current scalar vector
    // -> extract local values from the global vectors
    Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (phinp==Teuchos::null) dserror("Cannot get state vector 'phinp'");
    std::vector<double> myphinp(lm.size());
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

    // fill all element arrays
    for (int i=0;i<nen_;++i)
    {
      for (int k = 0; k< numscal_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        ephinp_[k](i,0) = myphinp[k+(i*numdofpernode_)];
      }
    } // for i

    CalcGradientAtNodes(ele, elemat1_epetra, elevec1_epetra, elevec2_epetra, elevec3_epetra);

    break;
  }
  case SCATRA::recon_curvature_at_nodes:
    {
      // need current scalar vector
      // -> extract local values from the global vectors
      Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
      if (phinp==Teuchos::null) dserror("Cannot get state vector 'phinp'");

      Teuchos::RCP<const Epetra_Vector> gradphinp_x = discretization.GetState("gradphinp_x");
      if (phinp==Teuchos::null) dserror("Cannot get state vector 'gradphinp_x'");

      Teuchos::RCP<const Epetra_Vector> gradphinp_y = discretization.GetState("gradphinp_y");
      if (phinp==Teuchos::null) dserror("Cannot get state vector 'gradphinp_y'");

      Teuchos::RCP<const Epetra_Vector> gradphinp_z = discretization.GetState("gradphinp_z");
      if (phinp==Teuchos::null) dserror("Cannot get state vector 'gradphinp_z'");

      std::vector<double> myphinp(lm.size());
      DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

      std::vector<double> mygradphinp_x(lm.size());
      DRT::UTILS::ExtractMyValues(*gradphinp_x,mygradphinp_x,lm);

      std::vector<double> mygradphinp_y(lm.size());
      DRT::UTILS::ExtractMyValues(*gradphinp_y,mygradphinp_y,lm);

      std::vector<double> mygradphinp_z(lm.size());
      DRT::UTILS::ExtractMyValues(*gradphinp_z,mygradphinp_z,lm);

      std::vector<LINALG::Matrix<nen_,nsd_> > egradphinp(numscal_);

      // fill all element arrays
      for (int i=0;i<nen_;++i)
      {
        for (int k = 0; k< numscal_; ++k)
        {
          // split for each transported scalar, insert into element arrays
          ephinp_[k](i,0) = myphinp[k+(i*numdofpernode_)];
          egradphinp[k](i,0) = mygradphinp_x[k+(i*numdofpernode_)];
          egradphinp[k](i,1) = mygradphinp_y[k+(i*numdofpernode_)];
          egradphinp[k](i,2) = mygradphinp_z[k+(i*numdofpernode_)];
        }
      } // for i

      CalcCurvatureAtNodes(ele, elemat1_epetra, elevec1_epetra, egradphinp);

      break;
    }
  case SCATRA::calc_mass_center_smoothingfunct:
    {
      double interface_thickness = params.get<double>("INTERFACE_THICKNESS_TPF");

      if(numscal_>1)
      {
        std::cout << "###########################################################################################################" << std::endl;
        std::cout << "#                                                 WARNING:                                                #" << std::endl;
        std::cout << "# More scalars than the levelset are transported. Mass center calculations have NOT been tested for this. #" << std::endl;
        std::cout << "#                                                                                                         # " << std::endl;
        std::cout << "###########################################################################################################" << std::endl;

      }
      // NOTE: add integral values only for elements which are NOT ghosted!
      if (ele->Owner() == discretization.Comm().MyPID())
      {
        // need current scalar vector
        // -> extract local values from the global vectors
        Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
        if (phinp==Teuchos::null) dserror("Cannot get state vector 'phinp'");
        std::vector<double> myphinp(lm.size());
        DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

        // fill all element arrays
        for (int i=0;i<nen_;++i)
        {
          for (int k = 0; k< numscal_; ++k)
          {
            // split for each transported scalar, insert into element arrays
            ephinp_[k](i,0) = myphinp[k+(i*numdofpernode_)];
          }
        } // for i

        // calculate momentum vector and volume for element.
        CalculateMomentumAndVolume(ele,elevec1_epetra,interface_thickness);
      }
      break;
    }
  case SCATRA::calc_integr_pat_rhsvec:
  {
    // extract local values from the global vectors w and phi
    Teuchos::RCP<const Epetra_Vector> rhsvec = discretization.GetState("rhsnodebasedvals");
    if (rhsvec==Teuchos::null)
      dserror("Cannot get state vector 'rhsnodebasedvals' ");
    std::vector<double> myrhsvec(lm.size());
    DRT::UTILS::ExtractMyValues(*rhsvec,myrhsvec,lm);

    Epetra_SerialDenseVector rhs(nen_);
    for(int i=0; i<nen_; ++i)
      rhs(i) = myrhsvec[i];

    Epetra_SerialDenseMatrix mass(nen_,nen_);
    const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);
    for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
    {
      double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad);
      for (int vi=0; vi<nen_; ++vi)
      {
        const double v = fac*funct_(vi); // no density required here

        for (int ui=0; ui<nen_; ++ui)
        {
          mass(vi,ui) += v*funct_(ui);
        }
      }
    }

    elevec1_epetra.Multiply('N','N',1.0,mass,rhs,0.0);

    break;
  }
  case SCATRA::calc_error:
  {
    // check if length suffices
    if (elevec1_epetra.Length() < 1) dserror("Result vector too short");

    // need current solution
    Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (phinp==Teuchos::null) dserror("Cannot get state vector 'phinp'");

    // extract local values from the global vector
    std::vector<double> myphinp(lm.size());
    DRT::UTILS::ExtractMyValues(*phinp,myphinp,lm);

    // fill element arrays
    for (int i=0;i<nen_;++i)
    {
      // split for each transported scalar, insert into element arrays
      for (int k = 0; k< numscal_; ++k)
      {
        ephinp_[k](i) = myphinp[k+(i*numdofpernode_)];
      }
    } // for i

    CalErrorComparedToAnalytSolution(
      ele,
      params,
      elevec1_epetra);

    break;
  }
  default:
  {
    dserror("Not acting on this action. Forgot implementation?");
    break;
  }
  } // switch(action)

  return 0;
}


/*----------------------------------------------------------------------*
 | evaluate service routine                                  fang 02/15 |
 *----------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distype,int probdim>
int DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::EvaluateService(
    DRT::Element*               ele,
    Teuchos::ParameterList&     params,
    DRT::Discretization&        discretization,
    const std::vector<int>&     lm,
    Epetra_SerialDenseMatrix&   elemat1_epetra,
    Epetra_SerialDenseMatrix&   elemat2_epetra,
    Epetra_SerialDenseVector&   elevec1_epetra,
    Epetra_SerialDenseVector&   elevec2_epetra,
    Epetra_SerialDenseVector&   elevec3_epetra
    )
{
  // setup
  if(SetupCalc(ele,discretization) == -1)
    return 0;

  if(scatrapara_->IsAle())
  {
    const Teuchos::RCP<Epetra_MultiVector> dispnp = params.get< Teuchos::RCP<Epetra_MultiVector> >("dispnp");
    if (dispnp==Teuchos::null) dserror("Cannot get state vector 'dispnp'");
    DRT::UTILS::ExtractMyNodeBasedValues(ele,edispnp_,dispnp,nsd_);
    // add nodal displacements to point coordinates
    xyze_ += edispnp_;
  }
  else edispnp_.Clear();

  // check for the action parameter
  const SCATRA::Action action = DRT::INPUT::get<SCATRA::Action>(params,"action");

  // evaluate action
  EvaluateAction(
      ele,
      params,
      discretization,
      action,
      lm,
      elemat1_epetra,
      elemat2_epetra,
      elevec1_epetra,
      elevec2_epetra,
      elevec3_epetra
      );

  return 0;
}


/*-------------------------------------------------------------------------*
 | Element reconstruct grad phi, one deg of freedom (for now) winter 04/14 |
 *-------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalcGradientAtNodes(
  const DRT::Element*             ele,
  Epetra_SerialDenseMatrix&       elemat1,
  Epetra_SerialDenseVector&       elevec1,
  Epetra_SerialDenseVector&       elevec2,
  Epetra_SerialDenseVector&       elevec3
  )
{
  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // Loop over integration points
  for (int gpid=0; gpid<intpoints.IP().nquad; gpid++)
  {
    // Get integration factor and evaluate shape func and its derivatives at the integration points.
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,gpid);
    // Loop over degrees of freedom
    for (int k=0; k<numdofpernode_; k++)
    {
    LINALG::Matrix<3,1> grad_phi(true);

    // Get gradient of phi at Gauss point
    for (int node = 0; node< nen_; node++)
      for (int idim = 0; idim< nsd_; idim++)
        grad_phi(idim,0) += ephinp_[k](node,0)*derxy_(idim,node);


    //const double eta_smooth=0.0; //Option for smoothing factor, currently not used.
    //diffmanager_->SetIsotropicDiff(eta_smooth,k);

    // Compute element matrix. For L2-projection
    CalcMatDiff(elemat1,k,fac);
    CalcMatMass(elemat1,k,fac,1.0);

    // Compute element vectors. For L2-Projection
    for (int node_i=0;node_i<nen_;node_i++)
    {
      elevec1[node_i*numdofpernode_+k] += funct_(node_i) * fac * grad_phi(0,0);
      elevec2[node_i*numdofpernode_+k] += funct_(node_i) * fac * grad_phi(1,0);
      elevec3[node_i*numdofpernode_+k] += funct_(node_i) * fac * grad_phi(2,0);
    }

    } //loop over degrees of freedom

  } //loop over integration points

  return;

} //ScaTraEleCalc::CalcGradientAtNodes


/*-------------------------------------------------------------------------*
 | Element reconstructed curvature                            winter 04/14 |
 *-------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalcCurvatureAtNodes(
  const DRT::Element*                       ele,
  Epetra_SerialDenseMatrix&                 elemat1,
  Epetra_SerialDenseVector&                 elevec1,
  const std::vector<LINALG::Matrix<nen_,nsd_> >&   egradphinp
  )
{
  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // Loop over integration points
  for (int gpid=0; gpid<intpoints.IP().nquad; gpid++)
  {
    // Get integration factor and evaluate shape func and its derivatives at the integration points.
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,gpid);

    // Loop over degrees of freedom
    for (int k=0; k<numdofpernode_; k++)
    {
      // Get gradient of phi at Gauss point
      LINALG::Matrix<3,1> grad_phi(true);
      grad_phi.Clear();
      for (int rr=0; rr<nsd_; rr ++)
      {
        for(int inode = 0; inode < nen_; inode++)
          grad_phi(rr) += egradphinp[k](inode,rr) *funct_(inode);
      }


      // get second derivatives of phi at Gauss point
      static LINALG::Matrix<9,1> grad2_phi;
      grad2_phi.Clear();

      for(int inode = 0; inode < nen_; inode++)
      {
        grad2_phi(0) += derxy_(0,inode)*egradphinp[k](inode,0); // ,xx
        grad2_phi(1) += derxy_(1,inode)*egradphinp[k](inode,1); // ,yy
        grad2_phi(2) += derxy_(2,inode)*egradphinp[k](inode,2); // ,zz
        grad2_phi(3) += derxy_(1,inode)*egradphinp[k](inode,0); // ,xy
        grad2_phi(4) += derxy_(2,inode)*egradphinp[k](inode,0); // ,xz
        grad2_phi(5) += derxy_(2,inode)*egradphinp[k](inode,1); // ,yz
        grad2_phi(6) += derxy_(0,inode)*egradphinp[k](inode,1); // ,yx
        grad2_phi(7) += derxy_(0,inode)*egradphinp[k](inode,2); // ,zx
        grad2_phi(8) += derxy_(1,inode)*egradphinp[k](inode,2); // ,zy
      }

      // get curvature at Gauss point
      double curvature_gp = 0.0;

      double grad_phi_norm = grad_phi.Norm2();
      {
        // check norm of normal gradient
        if (fabs(grad_phi_norm < 1.0E-9))
        {
          std::cout << "grad phi is small -> set to 1.0E9" << grad_phi_norm << std::endl;
          // phi gradient too small -> there must be a local max or min in the level-set field
          // set curvature to a large value
          // note: we have 1.0E12 in CalcCurvature Function
          curvature_gp = 1.0E2;
        }
        else
        {
          double val = grad_phi_norm*grad_phi_norm*grad_phi_norm;
          double invval = 1.0 / val;
          curvature_gp = -invval * ( grad_phi(0)*grad_phi(0)*grad2_phi(0)
                                    + grad_phi(1)*grad_phi(1)*grad2_phi(1)
                                    + grad_phi(2)*grad_phi(2)*grad2_phi(2) )
                         -invval * ( grad_phi(0)*grad_phi(1)*( grad2_phi(3) + grad2_phi(6) )
                                    + grad_phi(0)*grad_phi(2)*( grad2_phi(4) + grad2_phi(7) )
                                    + grad_phi(1)*grad_phi(2)*( grad2_phi(5) + grad2_phi(8)) )
                         +1.0/grad_phi_norm * ( grad2_phi(0) + grad2_phi(1) + grad2_phi(2) );

          // *-1.0 because of direction of gradient phi vector, which is minus the normal on the interface pointing from + to -
          curvature_gp *= -1.0;
        }
      }

      // check for norm of grad phi almost zero
      if (fabs(grad_phi_norm < 1.0E-9))
      {
        std::cout << "grad phi is small -> set to 1.0E9" << grad_phi_norm << std::endl;
        grad_phi_norm = 1.0;
      }

//      const double eta_smooth=0.0; //Option for smoothing factor, currently not used.
//      diffmanager_->SetIsotropicDiff(eta_smooth,k);

      // Compute element matrix.
      CalcMatDiff(elemat1,k,fac);
      CalcMatMass(elemat1,k,fac,1.0);

      // Compute rhs vector.
      // element matrix and rhs
      for (int vi=0; vi<nen_; ++vi) // loop rows  (test functions)
      {
        elevec1(vi) += fac*funct_(vi)*curvature_gp;
      }

    } //loop over degrees of freedom

  } //loop over integration points


  return;

} //ScaTraEleCalc::CalcCurvatureAtNodes


/*---------------------------------------------------------------------*
 | calculate filtered fields for turbulent Prandtl number   fang 02/15 |
 *---------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalcBoxFilter(
    DRT::Element*               ele,
    Teuchos::ParameterList&     params,
    DRT::Discretization&        discretization,
    const std::vector<int>&     lm
    )
{
  // extract scalar and velocity values from global vectors
  // scalar field
  Teuchos::RCP<const Epetra_Vector> scalar = discretization.GetState("scalar");
  if (scalar == Teuchos::null)
    dserror("Cannot get scalar!");
  std::vector<double> myscalar(lm.size());
  DRT::UTILS::ExtractMyValues(*scalar,myscalar,lm);
  for (int i=0;i<nen_;++i)
      ephinp_[0](i,0) = myscalar[i];
  // velocity field
  const Teuchos::RCP<Epetra_MultiVector> velocity = params.get< Teuchos::RCP<Epetra_MultiVector> >("velocity");
  DRT::UTILS::ExtractMyNodeBasedValues(ele,evelnp_,velocity,nsd_);

  // initialize the contribution of this element to the patch volume to zero
  double volume_contribution = 0.0;
  // initialize the contributions of this element to the filtered scalar quantities
  double dens_hat = 0.0;
  double temp_hat = 0.0;
  double dens_temp_hat = 0.0;
  double phi2_hat=0.0;
  double phiexpression_hat=0.0;
  // get pointers for vector quantities
  Teuchos::RCP<std::vector<double> > vel_hat = params.get<Teuchos::RCP<std::vector<double> > >("vel_hat");
  Teuchos::RCP<std::vector<double> > densvel_hat = params.get<Teuchos::RCP<std::vector<double> > >("densvel_hat");
  Teuchos::RCP<std::vector<double> > densveltemp_hat = params.get<Teuchos::RCP<std::vector<double> > >("densveltemp_hat");
  Teuchos::RCP<std::vector<double> > densstraintemp_hat = params.get<Teuchos::RCP<std::vector<double> > >("densstraintemp_hat");
  Teuchos::RCP<std::vector<double> > phi_hat = params.get<Teuchos::RCP<std::vector<double> > >("phi_hat");
  Teuchos::RCP<std::vector<std::vector<double> > > alphaijsc_hat = params.get<Teuchos::RCP<std::vector<std::vector<double> > > >("alphaijsc_hat");
  // integrate the convolution with the box filter function for this element
  // the results are assembled onto the *_hat arrays
  switch (distype)
  {
  case DRT::Element::hex8:
  {
    scatra_apply_box_filter(
        dens_hat,
        temp_hat,
        dens_temp_hat,
        phi2_hat,
        phiexpression_hat,
        vel_hat,
        densvel_hat,
        densveltemp_hat,
        densstraintemp_hat,
        phi_hat,
        alphaijsc_hat,
        volume_contribution,
        ele,
        params
        );

    break;
  }
  default:
  {
    dserror("Unknown element type for box filter application\n");
    break;
  }
  }

  // hand down the volume contribution to the time integration algorithm
  params.set<double>("volume_contribution",volume_contribution);
  // as well as the filtered scalar quantities
  params.set<double>("dens_hat",dens_hat);
  params.set<double>("temp_hat",temp_hat);
  params.set<double>("dens_temp_hat",dens_temp_hat);
  params.set<double>("phi2_hat",phi2_hat);
  params.set<double>("phiexpression_hat",phiexpression_hat);

  return;
} // DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalcBoxFilter


/*-----------------------------------------------------------------------------*
 | calculate mass matrix + rhs for initial time derivative calc.     gjb 03/12 |
 *-----------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalcInitialTimeDerivative(
    DRT::Element*               ele,              //!< current element
    Epetra_SerialDenseMatrix&   emat,             //!< element matrix
    Epetra_SerialDenseVector&   erhs,             //!< element residual
    Teuchos::ParameterList&     params,           //!< parameter list
    DRT::Discretization&        discretization,   //!< discretization
    const std::vector<int>&     lm                //!< location vector
    )
{
  // extract relevant quantities from discretization and parameter list
  ExtractElementAndNodeValues(ele,params,discretization,lm);

  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------
  // use one-point Gauss rule to do calculations at the element center
  const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints_tau(SCATRA::DisTypeToStabGaussRule<distype>::rule);

  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double vol = EvalShapeFuncAndDerivsAtIntPoint(intpoints_tau,0);

  //------------------------------------------------------------------------------------
  // get material parameters and stabilization parameters (evaluation at element center)
  //------------------------------------------------------------------------------------
    // density at t_(n)
  double densn(1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  double densnp(1.0);
  // density at t_(n+alpha_M)
  double densam(1.0);

  // fluid viscosity
  double visc(0.0);

  // the stabilisation parameters (one per transported scalar)
  std::vector<double> tau(numscal_,0.0);

  if (not scatrapara_->MatGP() or not scatrapara_->TauGP())
  {
    GetMaterialParams(ele,densn,densnp,densam,visc);

    if (not scatrapara_->TauGP())
    {
      // get velocity at element center
      LINALG::Matrix<nsd_,1> velint(true);
      LINALG::Matrix<nsd_,1> convelint(true);
      velint.Multiply(evelnp_,funct_);
      convelint.Multiply(econvelnp_,funct_);

      for (int k = 0;k<numscal_;++k) // loop of each transported scalar
      {
        // calculation of stabilization parameter at element center
        CalcTau(tau[k],diffmanager_->GetIsotropicDiff(k),reamanager_->GetReaCoeff(k),densnp,convelint,vol);
      }
    }
  }

  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  /*----------------------------------------------------------------------*/
  // element integration loop
  /*----------------------------------------------------------------------*/
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad);

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    if (scatrapara_->MatGP()) GetMaterialParams(ele,densn,densnp,densam,visc);

    //------------ get values of variables at integration point
    for (int k=0;k<numscal_;++k) // deal with a system of transported scalars
    {
      SetInternalVariablesForMatAndRHS();

      // get phi at integration point for all scalars
      const double& phiint = scatravarmanager_->Phinp(k);

      // get velocity at integration point
      LINALG::Matrix<nsd_,1> velint(true);
      LINALG::Matrix<nsd_,1> convelint(true);
      velint.Multiply(evelnp_,funct_);
      convelint.Multiply(econvelnp_,funct_);

      // convective part in convective form: rho*u_x*N,x+ rho*u_y*N,y
      LINALG::Matrix<nen_,1> conv(true);
      conv.MultiplyTN(derxy_,convelint);

      // scalar at integration point at time step n+1
      const double phinp = funct_.Dot(ephinp_[k]);

      // velocity divergence required for conservative form
      double vdiv(0.0);
      if (scatrapara_->IsConservative()) GetDivergence(vdiv,evelnp_);

      // diffusive part used in stabilization terms
      LINALG::Matrix<nen_,1> diff(true);
      // diffusive term using current scalar value for higher-order elements
      if (use2ndderiv_)
      {
        // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
        GetLaplacianStrongForm(diff);
        diff.Scale(diffmanager_->GetIsotropicDiff(k));
      }

      // calculation of stabilization parameter at integration point
      if (scatrapara_->TauGP()) CalcTau(tau[k],diffmanager_->GetIsotropicDiff(k),reamanager_->GetReaCoeff(k),densnp,scatravarmanager_->ConVel(),vol);

      const double fac_tau = fac*tau[k];

      //----------------------------------------------------------------
      // element matrix: transient term
      //----------------------------------------------------------------
      // transient term
      CalcMatMass(emat,k,fac,densam);

      //----------------------------------------------------------------
      // element matrix: stabilization of transient term
      //----------------------------------------------------------------
      // the stabilization term is deactivated in PrepareFirstTimeStep()
      if(scatrapara_->StabType()!=INPAR::SCATRA::stabtype_no_stabilization)
      {
        // subgrid-scale velocity (dummy)
        LINALG::Matrix<nen_,1> sgconv(true);
        CalcMatMassStab(emat,k,fac_tau,densam,densnp,sgconv,diff);

        // remove convective stabilization of inertia term
        for (int vi=0; vi<nen_; ++vi)
        {
          const int fvi = vi*numdofpernode_+k;
          erhs(fvi)+=fac_tau*densnp*conv(vi)*densnp*phiint;
        }
      }

      // we solve: (w,dc/dt) = rhs
      // whereas the rhs is based on the standard element evaluation routine
      // including contributions resulting from the time discretization.
      // The contribution from the time discretization has to be removed before solving the system:
      CorrectRHSFromCalcRHSLinMass(erhs,k,fac,densnp,phinp);
    } // loop over each scalar k
  } // integration loop

  // scale element matrix appropriately to be consistent with scaling of global residual vector computed by AssembleMatAndRHS() routine
  // (see CalcInitialTimeDerivative() routine on time integrator level)
  emat.Scale(scatraparatimint_->TimeFacRhs());

  return;
} // ScaTraEleCalc::CalcInitialTimeDerivative()


/*----------------------------------------------------------------------*
 |  CorrectRHSFromCalcRHSLinMass                             ehrl 06/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CorrectRHSFromCalcRHSLinMass(
    Epetra_SerialDenseVector&     erhs,
    const int                     k,
    const double                  fac,
    const double                  densnp,
    const double                  phinp
  )
{
  // fac->-fac to change sign of rhs
  if (scatraparatimint_->IsIncremental())
     CalcRHSLinMass(erhs,k,0.0,-fac,0.0,densnp);
  else
    dserror("Must be incremental!");

  return;
}


/*----------------------------------------------------------------------*
 |  Integrate shape functions over domain (private)           gjb 07/09 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::IntegrateShapeFunctions(
  const DRT::Element*             ele,
  Epetra_SerialDenseVector&       elevec1,
  const Epetra_IntSerialDenseVector& dofids
  )
{
  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // safety check
  if (dofids.M() < numdofpernode_)
    dserror("Dofids vector is too short. Received not enough flags");

  // loop over integration points
  // this order is not efficient since the integration of the shape functions is always the same for all species
  for (int gpid=0; gpid<intpoints.IP().nquad; gpid++)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,gpid);

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

} //ScaTraEleCalc::IntegrateShapeFunction


/*----------------------------------------------------------------------*
  |  calculate weighted mass flux (no reactive flux so far)     gjb 06/08|
  *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalculateFlux(
LINALG::Matrix<3,nen_>&         flux,
const DRT::Element*             ele,
const INPAR::SCATRA::FluxType   fluxtype,
const int                       k
)
{
  /*
  * Actually, we compute here a weighted (and integrated) form of the fluxes!
  * On time integration level, these contributions are then used to calculate
  * an L2-projected representation of fluxes.
  * Thus, this method here DOES NOT YET provide flux values that are ready to use!!
  /                                                         \
  |                /   \                               /   \  |
  | w, -D * nabla | phi | + u*phi - frt*z_k*c_k*nabla | pot | |
  |                \   /                               \   /  |
  \                      [optional]      [ELCH]               /
  */

  // density at t_(n)
  double densn(1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  double densnp(1.0);
  // density at t_(n+alpha_M)
  double densam(1.0);

  // fluid viscosity
  double visc(0.0);

  // get material parameters (evaluation at element center)
  if (not scatrapara_->MatGP()) GetMaterialParams(ele,densn,densnp,densam,visc);

  // integration rule
  const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad=0; iquad< intpoints.IP().nquad; ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad);

    // get material parameters (evaluation at integration point)
    if (scatrapara_->MatGP()) GetMaterialParams(ele,densn,densnp,densam,visc);

    // get velocity at integration point
    LINALG::Matrix<nsd_,1> velint(true);
    LINALG::Matrix<nsd_,1> convelint(true);
    velint.Multiply(evelnp_,funct_);
    convelint.Multiply(econvelnp_,funct_);

    // get scalar at integration point
    const double phi = funct_.Dot(ephinp_[k]);

    // get gradient of scalar at integration point
    LINALG::Matrix<nsd_,1> gradphi(true);
    gradphi.Multiply(derxy_,ephinp_[k]);

    // allocate and initialize!
    LINALG::Matrix<nsd_,1> q(true);

    // add different flux contributions as specified by user input
    switch (fluxtype)
    {
    case INPAR::SCATRA::flux_total_domain:
      // convective flux contribution
      q.Update(densnp*phi,convelint);

      // no break statement here!
    case INPAR::SCATRA::flux_diffusive_domain:
      // diffusive flux contribution
      q.Update(-(diffmanager_->GetIsotropicDiff(k)),gradphi,1.0);

      break;
    default:
      dserror("received illegal flag inside flux evaluation for whole domain"); break;
    };
    // q at integration point

    // integrate and assemble everything into the "flux" vector
    for (int vi=0; vi < nen_; vi++)
    {
      for (int idim=0; idim<nsd_ ;idim++)
      {
        flux(idim,vi) += fac*funct_(vi)*q(idim);
      } // idim
    } // vi

  } // integration loop

  //set zeros for unused space dimensions
  for (int idim=nsd_; idim<3; idim++)
  {
    for (int vi=0; vi < nen_; vi++)
    {
      flux(idim,vi) = 0.0;
    }
  }

  return;
} // ScaTraCalc::CalculateFlux


/*----------------------------------------------------------------------*
|  calculate scalar(s) and domain integral                     vg 09/08|
*----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalculateScalars(
const DRT::Element*             ele,
Epetra_SerialDenseVector&       scalars,
const bool                      inverting
  )
{
  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad);

    // calculate integrals of (inverted) scalar(s) and domain
    if (inverting)
    {
      for (int i=0; i<nen_; i++)
      {
        const double fac_funct_i = fac*funct_(i);
        for (int k = 0; k < numscal_; k++)
        {
          if (std::abs(ephinp_[k](i,0))> EPS14)
            scalars[k] += fac_funct_i/ephinp_[k](i,0);
          else
            dserror("Division by zero");
        }
        // for domain volume
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
          scalars[k] += fac_funct_i*ephinp_[k](i,0);
        }
        // for domain volume
        scalars[numscal_] += fac_funct_i;
      }
    }
  } // loop over integration points

  return;
} // ScaTraEleCalc::CalculateScalars


/*----------------------------------------------------------------------*
|  calculate momentum vector and minus domain integral          mw 06/14|
*----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalculateMomentumAndVolume(
const DRT::Element*             ele,
Epetra_SerialDenseVector&       momandvol,
const double                    interface_thickness
  )
{

  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad);

    // coordinates of the current integration point.
    std::vector<double> gpcoord(nsd_);
    // levelset function at gaussian point.
    double ephi_gp = 0.0;
    //fac*funct at GP
    double fac_funct = 0.0;

    for (int i=0; i<nen_; i++)
    {
      // Levelset function (first scalar stored [0]) at gauss point
      ephi_gp += funct_(i)*ephinp_[0](i,0);

      //Coordinate * shapefunction to get the coordinate value of the gausspoint.
      for(int idim=0; idim<nsd_; idim++)
      {
        gpcoord[idim] += funct_(i)*(ele->Nodes()[i]->X()[idim]);
      }

      //Summation of fac*funct_ for volume calculation.
      fac_funct+=fac*funct_(i);
    }

    double heavyside_epsilon = 1.0; //plus side

    //Smoothing function
    if(abs(ephi_gp) <= interface_thickness)
    {
      heavyside_epsilon = 0.5 * (1.0 + ephi_gp/interface_thickness + 1.0 / PI * sin(PI*ephi_gp/interface_thickness));
    }
    else if(ephi_gp < interface_thickness)
    {
      heavyside_epsilon=0.0; //minus side
    }


    // add momentum vector and volume
    for(int idim=0; idim<nsd_; idim++)
    {
      momandvol(idim)+=gpcoord[idim]*(1.0-heavyside_epsilon)*fac_funct;
    }

    momandvol(nsd_)+=fac_funct*(1.0-heavyside_epsilon);

  } // loop over integration points

return;
} // ScaTraEleCalc::CalculateMomentumAndVolume


/*----------------------------------------------------------------------*
 | calculate normalized subgrid-diffusivity matrix              vg 10/08|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalcSubgrDiffMatrix(
  const DRT::Element*           ele,
  Epetra_SerialDenseMatrix&     emat
  )
{
  /*----------------------------------------------------------------------*/
  // integration loop for one element
  /*----------------------------------------------------------------------*/
  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad);

    for (int k=0;k<numscal_;++k)
    {
      // set diffusion coeff to 1.0
      diffmanager_->SetIsotropicDiff(1.0,k);

      // calculation of diffusive element matrix
      double timefacfac = scatraparatimint_->TimeFac() * fac;
      CalcMatDiff(emat,k,timefacfac);

      /*subtract SUPG term */
      //emat(fvi,fui) -= taufac*conv(vi)*conv(ui);
    }
  } // integration loop

  return;
} // ScaTraImpl::CalcSubgrDiffMatrix


/*----------------------------------------------------------------------------------------*
 | finite difference check on element level (for debugging only) (protected)   fang 10/14 |
 *----------------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::FDCheck(
  DRT::Element*                              ele,
  Epetra_SerialDenseMatrix&                  emat,
  Epetra_SerialDenseVector&                  erhs,
  Epetra_SerialDenseVector&                  subgrdiff
  )
{
  // screen output
  std::cout << "FINITE DIFFERENCE CHECK FOR ELEMENT " << ele->Id();

  // make a copy of state variables to undo perturbations later
  std::vector<LINALG::Matrix<nen_,1> > ephinp_original(numscal_);
  for(int k=0; k<numscal_; ++k)
    for(int i=0; i<nen_; ++i)
      ephinp_original[k](i,0) = ephinp_[k](i,0);

  // generalized-alpha time integration requires a copy of history variables as well
  std::vector<LINALG::Matrix<nen_,1> > ehist_original(numscal_);
  if(scatraparatimint_->IsGenAlpha())
  {
    for(int k=0; k<numscal_; ++k)
      for(int i=0; i<nen_; ++i)
        ehist_original[k](i,0)  = ehist_[k](i,0);
  }

  // initialize element matrix and vectors for perturbed state
  Epetra_SerialDenseMatrix emat_dummy(emat);
  Epetra_SerialDenseVector erhs_perturbed(erhs);
  Epetra_SerialDenseVector subgrdiff_dummy(subgrdiff);

  // initialize counter for failed finite difference checks
  unsigned counter(0);

  // initialize tracking variable for maximum absolute and relative errors
  double maxabserr(0.);
  double maxrelerr(0.);

  // loop over columns of element matrix by first looping over nodes and then over dofs at each node
  for(int inode=0; inode<nen_; ++inode)
  {
    for(int idof=0; idof<numdofpernode_; ++idof)
    {
      // number of current column of element matrix
      unsigned col = inode*numdofpernode_+idof;

      // clear element matrix and vectors for perturbed state
      emat_dummy.Scale(0.0);
      erhs_perturbed.Scale(0.0);
      subgrdiff_dummy.Scale(0.0);

      // fill state vectors with original state variables
      for(int k=0; k<numscal_; ++k)
        for(int i=0; i<nen_; ++i)
          ephinp_[k](i,0) = ephinp_original[k](i,0);
      if(scatraparatimint_->IsGenAlpha())
        for(int k=0; k<numscal_; ++k)
          for(int i=0; i<nen_; ++i)
            ehist_[k](i,0)  = ehist_original[k](i,0);

      // impose perturbation
      if(scatraparatimint_->IsGenAlpha())
      {
        // perturbation of phi(n+alphaF), not of phi(n+1) => scale epsilon by factor alphaF
        ephinp_[idof](inode,0) += scatraparatimint_->AlphaF() * scatrapara_->FDCheckEps();

        // perturbation of phi(n+alphaF) by alphaF*epsilon corresponds to perturbation of phidtam (stored in ehist_)
        // by alphaM*epsilon/(gamma*dt); note: alphaF/timefac = alphaM/(gamma*dt)
        ehist_[idof](inode,0) += scatraparatimint_->AlphaF() / scatraparatimint_->TimeFac() * scatrapara_->FDCheckEps();
      }
      else
        ephinp_[idof](inode,0) += scatrapara_->FDCheckEps();

      // calculate element right-hand side vector for perturbed state
      Sysmat(ele,emat_dummy,erhs_perturbed,subgrdiff_dummy);

      // Now we compare the difference between the current entries in the element matrix
      // and their finite difference approximations according to
      // entries ?= (-erhs_perturbed + erhs_original) / epsilon

      // Note that the element right-hand side equals the negative element residual.
      // To account for errors due to numerical cancellation, we additionally consider
      // entries - erhs_original / epsilon ?= -erhs_perturbed / epsilon

      // Note that we still need to evaluate the first comparison as well. For small entries in the element
      // matrix, the second comparison might yield good agreement in spite of the entries being wrong!
      for(int row=0; row<numdofpernode_*nen_; ++row)
      {
        // get current entry in original element matrix
        const double entry = emat(row,col);

        // finite difference suggestion (first divide by epsilon and then subtract for better conditioning)
        const double fdval = -erhs_perturbed(row) / scatrapara_->FDCheckEps() + erhs(row) / scatrapara_->FDCheckEps();

        // confirm accuracy of first comparison
        if(abs(fdval) > 1.e-17 and abs(fdval) < 1.e-15)
          dserror("Finite difference check involves values too close to numerical zero!");

        // absolute and relative errors in first comparison
        const double abserr1 = entry - fdval;
        if(abs(abserr1) > abs(maxabserr))
          maxabserr = abserr1;
        double relerr1(0.);
        if(abs(entry) > 1.e-17)
          relerr1 = abserr1 / abs(entry);
        else if(abs(fdval) > 1.e-17)
          relerr1 = abserr1 / abs(fdval);
        if(abs(relerr1) > abs(maxrelerr))
          maxrelerr = relerr1;

        // evaluate first comparison
        if(abs(relerr1) > scatrapara_->FDCheckTol())
        {
          if(!counter)
            std::cout << " --> FAILED AS FOLLOWS:" << std::endl;
          std::cout << "emat[" << row << "," << col << "]:  " << entry << "   ";
          std::cout << "finite difference suggestion:  " << fdval << "   ";
          std::cout << "absolute error:  " << abserr1 << "   ";
          std::cout << "relative error:  " << relerr1 << std::endl;

          counter++;
        }

        // first comparison OK
        else
        {
          // left-hand side in second comparison
          const double left  = entry - erhs(row) / scatrapara_->FDCheckEps();

          // right-hand side in second comparison
          const double right = -erhs_perturbed(row) / scatrapara_->FDCheckEps();

          // confirm accuracy of second comparison
          if(abs(right) > 1.e-17 and abs(right) < 1.e-15)
            dserror("Finite difference check involves values too close to numerical zero!");

          // absolute and relative errors in second comparison
          const double abserr2 = left - right;
          if(abs(abserr2) > abs(maxabserr))
            maxabserr = abserr2;
          double relerr2(0.);
          if(abs(left) > 1.e-17)
            relerr2 = abserr2 / abs(left);
          else if(abs(right) > 1.e-17)
            relerr2 = abserr2 / abs(right);
          if(abs(relerr2) > abs(maxrelerr))
            maxrelerr = relerr2;

          // evaluate second comparison
          if(abs(relerr2) > scatrapara_->FDCheckTol())
          {
            if(!counter)
              std::cout << " --> FAILED AS FOLLOWS:" << std::endl;
            std::cout << "emat[" << row << "," << col << "]-erhs[" << row << "]/eps:  " << left << "   ";
            std::cout << "-erhs_perturbed[" << row << "]/eps:  " << right << "   ";
            std::cout << "absolute error:  " << abserr2 << "   ";
            std::cout << "relative error:  " << relerr2 << std::endl;

            counter++;
          }
        }
      }
    }
  }

  // screen output in case finite difference check is passed
  if(!counter)
    std::cout << " --> PASSED WITH MAXIMUM ABSOLUTE ERROR " << maxabserr << " AND MAXIMUM RELATIVE ERROR " << maxrelerr << std::endl;

  // undo perturbations of state variables
  for(int k=0; k<numscal_; ++k)
    for(int i=0; i<nen_; ++i)
      ephinp_[k](i,0) = ephinp_original[k](i,0);
  if(scatraparatimint_->IsGenAlpha())
    for(int k=0; k<numscal_; ++k)
      for(int i=0; i<nen_; ++i)
        ehist_[k](i,0)  = ehist_original[k](i,0);

  return;
}

/*---------------------------------------------------------------------*
  |  calculate error compared to analytical solution           gjb 10/08|
  *---------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype,int probdim>
void DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalErrorComparedToAnalytSolution(
  const DRT::Element*                   ele,
  Teuchos::ParameterList&               params,
  Epetra_SerialDenseVector&             errors
  )
{
  if (DRT::INPUT::get<SCATRA::Action>(params,"action") != SCATRA::calc_error)
    dserror("How did you get here?");

  // -------------- prepare common things first ! -----------------------
  // set constants for analytical solution
  const double t = scatraparatimint_->Time();

  // integration points and weights
  // more GP than usual due to (possible) cos/exp fcts in analytical solutions
  const DRT::UTILS::IntPointsAndWeights<nsd_ele_> intpoints(SCATRA::DisTypeToGaussRuleForExactSol<distype>::rule);

  const INPAR::SCATRA::CalcError errortype = DRT::INPUT::get<INPAR::SCATRA::CalcError>(params, "calcerrorflag");
  switch(errortype)
  {

  case INPAR::SCATRA::calcerror_byfunction:
  {
    const int errorfunctno = params.get<int>("error function number");

    // analytical solution
    double  phi_exact(0.0);
    double  deltaphi(0.0);
    //! spatial gradient of current scalar value
    LINALG::Matrix<nsd_,1>  gradphi(true);
    LINALG::Matrix<nsd_,1>  gradphi_exact(true);
    LINALG::Matrix<nsd_,1>  deltagradphi(true);

    // start loop over integration points
    for (int iquad=0;iquad<intpoints.IP().nquad;iquad++)
    {
      const double fac = EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad);

      // get coordinates at integration point
      // gp reference coordinates
      LINALG::Matrix<nsd_,1> xyzint(true);
      xyzint.Multiply(xyze_,funct_);

      // function evaluation requires a 3D position vector!!
      double position[3]={0.0,0.0,0.0};

      for (int dim=0; dim<nsd_; ++dim)
        position[dim] = xyzint(dim);

      for(int k=0;k<numscal_;k++)
      {
        // scalar at integration point at time step n+1
        const double phinp = funct_.Dot(ephinp_[k]);
        // spatial gradient of current scalar value
        gradphi.Multiply(derxy_,ephinp_[k]);

        phi_exact = DRT::Problem::Instance()->Funct(errorfunctno-1).Evaluate(0,position,t,NULL);

        std::vector<std::vector<double> > gradphi_exact_vec = DRT::Problem::Instance()->Funct(errorfunctno-1).FctDer(0,position,t,NULL);

        if(gradphi_exact_vec.size())
        {
          for (int dim=0; dim<nsd_; ++dim)
            gradphi_exact(dim)=gradphi_exact_vec[0][dim];
        }
        else
          gradphi_exact.Clear();

        // error at gauss point
        deltaphi = phinp - phi_exact;
        deltagradphi.Update(1.0,gradphi,-1.0,gradphi_exact);

        // 0: delta scalar for L2-error norm
        // 1: delta scalar for H1-error norm
        // 2: analytical scalar for L2 norm
        // 3: analytical scalar for H1 norm

        // the error for the L2 and H1 norms are evaluated at the Gauss point

        // integrate delta scalar for L2-error norm
        errors(k*numscal_+0) += deltaphi*deltaphi*fac;
        // integrate delta scalar for H1-error norm
        errors(k*numscal_+1) += deltaphi*deltaphi*fac;
        // integrate analytical scalar for L2 norm
        errors(k*numscal_+2) += phi_exact*phi_exact*fac;
        // integrate analytical scalar for H1 norm
        errors(k*numscal_+3) += phi_exact*phi_exact*fac;

        // integrate delta scalar derivative for H1-error norm
        errors(k*numscal_+1) += deltagradphi.Dot(deltagradphi)*fac;
        // integrate analytical scalar for H1 norm
        errors(k*numscal_+3) += deltaphi*deltaphi*fac;
      }
    } // loop over integration points

  }
  break;
  default: dserror("Unknown analytical solution!"); break;
  } //switch(errortype)

  return;
} // DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::CalErrorComparedToAnalytSolution


// template classes

#include "scatra_ele_calc_fwd.hpp"
