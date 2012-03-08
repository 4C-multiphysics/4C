/*----------------------------------------------------------------------*/
/*!
\file fluid3_impl_parameter.cpp

\brief Evaluation of general fluid parameter

Fluid3ImplParameter::SetParameter(Teuchos::ParameterList& params)
set all general fluid parameter once for all elements.

<pre>
Maintainer: Andreas Ehrl
            ehrl@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
*/
/*----------------------------------------------------------------------*/

#include <string>
#include <iostream>

#include "fluid3_impl_parameter.H"
#include "../drt_lib/drt_dserror.H"

using namespace std;
//----------------------------------------------------------------------*/
//    definition of the instance
//----------------------------------------------------------------------*/
DRT::ELEMENTS::Fluid3ImplParameter* DRT::ELEMENTS::Fluid3ImplParameter::instance_;


//----------------------------------------------------------------------*/
//    definition of the instance
//----------------------------------------------------------------------*/
DRT::ELEMENTS::Fluid3ImplParameter* DRT::ELEMENTS::Fluid3ImplParameter::Instance()
{
  if (instance_==NULL)
    instance_ = new Fluid3ImplParameter();
  return instance_;
}


//----------------------------------------------------------------------*/
// private constructor of Fluid3ImplParameter
//----------------------------------------------------------------------*/
DRT::ELEMENTS::Fluid3ImplParameter::Fluid3ImplParameter()
  :
  set_general_fluid_parameter_(false),
  is_genalpha_(false),
  is_genalpha_np_(false),
  is_conservative_(false),
  is_stationary_(false),
  is_newton_(false),
  is_inconsistent_(false),
  reaction_(false),
  darcy_(false),
  reaction_topopt_(false),
  physicaltype_(INPAR::FLUID::incompressible),
  tds_(INPAR::FLUID::subscales_none),
  transient_(INPAR::FLUID::inertia_stab_drop),
  pspg_(INPAR::FLUID::pstab_use_pspg),
  supg_(INPAR::FLUID::convective_stab_supg),
  vstab_(INPAR::FLUID::viscous_stab_none),
  cstab_(INPAR::FLUID::continuity_stab_yes),
  cross_(INPAR::FLUID::cross_stress_stab_none),
  reynolds_(INPAR::FLUID::reynolds_stress_stab_none),
  whichtau_(INPAR::FLUID::tau_not_defined),
  fssgv_(INPAR::FLUID::no_fssgv),
  turb_mod_action_(INPAR::FLUID::no_model),
  mat_gp_(false),     // standard evaluation of the material at the element center
  tau_gp_(false),     // standard evaluation of tau at the element center
  time_(-1.0),
  dt_(0.0),
  timefac_(0.0),
  theta_(0.0),
  omtheta_(0.0),
  gamma_(0.0),
  alphaF_(0.0),
  alphaM_(0.0),
  afgdt_(1.0),
  timefacrhs_(1.0),
  timefacpre_(1.0),
  viscreastabfac_(0.0),
  Cs_(0.0),
  Cs_averaged_(false),
  l_tau_(0.0),
  Cl_(0.0),
  Csgs_(0.0),
  Csgs_phi_(0.0),
  alpha_(0.0),
  CalcN_(false),
  N_(0.0),
  refvel_(INPAR::FLUID::strainrate),
  reflength_(INPAR::FLUID::cube_edge),
  c_nu_(1.0),
  c_diff_(1.0),
  near_wall_limit_(false),
  B_gp_(false),
  beta_(0.0),
  update_mat_(false),
  conti_supg_(INPAR::FLUID::convective_stab_none),
  conti_cross_(INPAR::FLUID::cross_stress_stab_none),
  conti_reynolds_(INPAR::FLUID::reynolds_stress_stab_none),
  multifrac_loma_conti_(false)

{
}


//----------------------------------------------------------------------*
//  set general parameters                                   ehrl 04/10 |
//---------------------------------------------------------------------*/
void DRT::ELEMENTS::Fluid3ImplParameter::SetElementGeneralFluidParameter( Teuchos::ParameterList& params )
{
  if(set_general_fluid_parameter_ == false)
    set_general_fluid_parameter_ = true;

//----------------------------------------------------------------------
// get flags to switch on/off different fluid formulations
//----------------------------------------------------------------------

  // set flag, time integration scheme
  timealgo_ = DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(params, "TimeIntegrationScheme");

  // set time integration scheme-specific element parameters
  if (timealgo_==INPAR::FLUID::timeint_stationary)
  {
    is_genalpha_ = false;
    is_stationary_ = true;
    is_genalpha_np_ = false;
  }
  else if (timealgo_==INPAR::FLUID::timeint_afgenalpha)
  {
    is_genalpha_ = true;
    is_stationary_ = false;
    is_genalpha_np_ = false;
  }
  else if (timealgo_==INPAR::FLUID::timeint_npgenalpha)
  {
    is_genalpha_ = true;
    is_stationary_ = false;
    is_genalpha_np_ = true;
  }
  else if (timealgo_==INPAR::FLUID::timeint_gen_alpha)
  {
    is_genalpha_ = true;
    is_stationary_ = false;
    is_genalpha_np_ = false;
  }
  else
  {
    is_genalpha_ = false;
    is_stationary_ = false;
    is_genalpha_np_ = false;
  }

  // set flag for type of linearization (fixed-point-like or Newton)
  //std::string newtonstr   = params.get<std::string>("Linearisation");
  if (DRT::INPUT::get<INPAR::FLUID::LinearisationAction>(params, "Linearisation")==INPAR::FLUID::Newton)
    is_newton_       = true;
  if (DRT::INPUT::get<INPAR::FLUID::LinearisationAction>(params, "Linearisation")==INPAR::FLUID::minimal)
    dserror("There is no LinearisationAction minimal in the fluid formulation");

  // set flags for formuation of the convective velocity term (conservative or convective)
  std::string convformstr = params.get<std::string>("form of convective term");
  if (convformstr =="conservative") is_conservative_ = true;

  // set flag for physical type of fluid flow
  physicaltype_ = DRT::INPUT::get<INPAR::FLUID::PhysicalType>(params, "Physical Type");
  if (((physicaltype_ != INPAR::FLUID::boussinesq) and (physicaltype_ != INPAR::FLUID::incompressible))
      and (is_stationary_ == true))
    dserror("physical type is not supported in stationary FLUID implementation.");

  if (is_genalpha_np_ and physicaltype_ == INPAR::FLUID::loma)
    dserror("the combination Np_Gen_Alpha and loma is not supported");

  if (is_genalpha_np_ and is_conservative_)
    dserror("the combination Np_Gen_Alpha and conservative flow is not supported");

  if (not is_stationary_ and is_conservative_)
  {
    cout << endl << "Warning: missing time derivative terms in conservative formulation (for variable density flows)!!" << endl;
  }


// ---------------------------------------------------------------------
// get control parameters for stabilization and higher-order elements
//----------------------------------------------------------------------
  Teuchos::ParameterList& stablist = params.sublist("STABILIZATION");

  // no safety check necessary since all options are used
  tds_      = DRT::INPUT::IntegralValue<INPAR::FLUID::SubscalesTD>(stablist,"TDS");
  transient_= DRT::INPUT::IntegralValue<INPAR::FLUID::Transient>(stablist,"TRANSIENT");
  pspg_     = DRT::INPUT::IntegralValue<INPAR::FLUID::PSPG>(stablist,"PSPG");
  supg_     = DRT::INPUT::IntegralValue<INPAR::FLUID::SUPG>(stablist,"SUPG");
  vstab_    = DRT::INPUT::IntegralValue<INPAR::FLUID::VStab>(stablist,"VSTAB");
  rstab_    = DRT::INPUT::IntegralValue<INPAR::FLUID::RStab>(stablist,"RSTAB");
  cstab_    = DRT::INPUT::IntegralValue<INPAR::FLUID::CStab>(stablist,"CSTAB");
  cross_    = DRT::INPUT::IntegralValue<INPAR::FLUID::CrossStress>(stablist,"CROSS-STRESS");
  reynolds_ = DRT::INPUT::IntegralValue<INPAR::FLUID::ReynoldsStress>(stablist,"REYNOLDS-STRESS");

  if (is_genalpha_np_ and
      (tds_ == INPAR::FLUID::subscales_time_dependent or transient_ != INPAR::FLUID::inertia_stab_drop))
    dserror("time dependent subscales does not work for ost/afgenalpha/npgenalpha. \nOne need to look for bugs");

//-------------------------------
// get tau definition
//-------------------------------

  whichtau_ =  DRT::INPUT::IntegralValue<INPAR::FLUID::TauType>(stablist,"DEFINITION_TAU");
  // check if tau can be handled
  if (not(whichtau_ == INPAR::FLUID::tau_taylor_hughes_zarins or
                       INPAR::FLUID::tau_taylor_hughes_zarins_wo_dt or
                       INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen or
                       INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen_wo_dt or
                       INPAR::FLUID::tau_taylor_hughes_zarins_scaled or
                       INPAR::FLUID::tau_taylor_hughes_zarins_scaled_wo_dt or
                       INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall or
                       INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall_wo_dt or
                       INPAR::FLUID::tau_shakib_hughes_codina or
                       INPAR::FLUID::tau_shakib_hughes_codina_wo_dt or
                       INPAR::FLUID::tau_codina or
                       INPAR::FLUID::tau_codina_wo_dt or
                       INPAR::FLUID::tau_franca_madureira_valentin_badia_codina or
                       INPAR::FLUID::tau_franca_madureira_valentin_badia_codina_wo_dt))
    dserror("Definition of Tau cannot be handled by the element");

  // set correct stationary definition of stabilization parameter automatically
  if (is_stationary_)
  {
    if (whichtau_ == INPAR::FLUID::tau_taylor_hughes_zarins)
      whichtau_ = INPAR::FLUID::tau_taylor_hughes_zarins_wo_dt;
    else if (whichtau_ == INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen)
      whichtau_ = INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen_wo_dt;
    else if (whichtau_ == INPAR::FLUID::tau_taylor_hughes_zarins_scaled)
      whichtau_ = INPAR::FLUID::tau_taylor_hughes_zarins_scaled_wo_dt;
    else if (whichtau_ == INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall)
      whichtau_ = INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall_wo_dt;
    else if (whichtau_ == INPAR::FLUID::tau_shakib_hughes_codina)
      whichtau_ = INPAR::FLUID::tau_shakib_hughes_codina_wo_dt;
    else if (whichtau_ == INPAR::FLUID::tau_codina)
      whichtau_ = INPAR::FLUID::tau_codina_wo_dt;
    else if (whichtau_ == INPAR::FLUID::tau_franca_madureira_valentin_badia_codina)
      whichtau_ = INPAR::FLUID::tau_franca_madureira_valentin_badia_codina_wo_dt;
  }

  // overrule higher_order_ele if input-parameter is set
  // this might be interesting for fast (but slightly
  // less accurate) computations
  if (stablist.get<std::string>("STABTYPE") == "inconsistent") is_inconsistent_ = true;

  // in case of viscous and/or reactive stabilization, decide whether to use
  // GLS or USFEM and ensure compatibility of respective definitions
  if (vstab_ == INPAR::FLUID::viscous_stab_usfem or
      vstab_ == INPAR::FLUID::viscous_stab_usfem_only_rhs)
  {
    viscreastabfac_ = -1.0;
    if (rstab_ == INPAR::FLUID::reactive_stab_gls)
      dserror("inconsistent reactive and viscous stabilization!");
  }
  else if (vstab_ == INPAR::FLUID::viscous_stab_gls or
           vstab_ == INPAR::FLUID::viscous_stab_gls_only_rhs)
  {
    viscreastabfac_ = 1.0;
    if (rstab_ == INPAR::FLUID::reactive_stab_usfem)
      dserror("inconsistent reactive and viscous stabilization!");
  }
  else if (vstab_ == INPAR::FLUID::viscous_stab_none)
  {
    if (rstab_ == INPAR::FLUID::reactive_stab_usfem)    viscreastabfac_ = -1.0;
    else if (rstab_ == INPAR::FLUID::reactive_stab_gls) viscreastabfac_ =  1.0;
  }

  // set flags for potential evaluation of tau and material law at int. point
  // default value: evaluation at element center
  const std::string tauloc = stablist.get<std::string>("EVALUATION_TAU");
  if (tauloc == "integration_point") tau_gp_ = true;
  else                               tau_gp_ = false;
  const std::string matloc = stablist.get<std::string>("EVALUATION_MAT");
  if (matloc == "integration_point") mat_gp_ = true;
  else                               mat_gp_ = false;
}

void DRT::ELEMENTS::Fluid3ImplParameter::SetElementTimeParameter( Teuchos::ParameterList& params )
{
  // second check: timealgo
  // work around to use SetTimeParameter in GenaAlpha (Neumann BC)
  if(set_general_fluid_parameter_!= true)
    dserror("General fluid parameter are not set yet!!");

  //----------------------------------------------------------------------
  // get control parameters for time integration
  //----------------------------------------------------------------------

  // get current time: n+alpha_F for generalized-alpha scheme, n+1 otherwise
  time_ = params.get<double>("total time",-1.0);

  // set global variable timefac to zero
  timefac_ = 0.0;

  if (not is_stationary_)
  {
    // get time-step length and time-integration parameters
    dt_      = params.get<double>("dt",-1.0);
    theta_   = params.get<double>("theta",-1.0);
    omtheta_ = params.get<double>("omtheta",-1.0);

    // compute timefactor for left-hand side:
    // one-step-Theta:    timefac = theta*dt
    // BDF2:              timefac = 2/3 * dt
    // generalized-alpha: timefac = (alpha_F/alpha_M) * gamma * dt
    // (For BDF2 and generalized-alpha, theta was already computed
    //  accordingly in FLD::FluidImplicitTimeInt::PrepareTimeStep().)

    //-----------------------------------------------------------------------
    //       |          timefac         |  timefacpre     |    timefacrhs   |
    // ----------------------------------------------------------------------
    // OST   |                        dt*theta                              |
    //-----------------------------------------------------------------------
    // BDF2  |                        2/3 * dt                              |
    //-----------------------------------------------------------------------
    // Af GA |          alphaF*gamma*dt/alphaM            | gamma*dt/alphaM |
    //----------------------------------------------------------------------
    // NP GA | alphaF*gamma*dt/alphaM   | gamma*dt/alphaM | gamma*dt/alphaM |
    //-----------------------------------------------------------------------
    //   GA  |      alphaF*gamma*dt     |    gamma*dt     |        1        |
    //-----------------------------------------------------------------------

    timefac_ = theta_*dt_;

    // compute generalized-alpha-related values and set them appropriately
    // otherwise
    if (is_genalpha_)
    {
      gamma_  = params.get<double>("gamma",-1.0);
      alphaF_ = params.get<double>("alphaF",-1.0);
      alphaM_ = params.get<double>("alphaM",-1.0);
    }
    else
    {
      gamma_  = theta_;
      alphaF_ = 1.0;
      alphaM_ = 1.0;
    }

    // if not generalized-alpha: afgdt = theta * dt_ = timefac_
    // Peter's generalized alpha: timefacmat_u_ for velocity terms
    afgdt_=alphaF_*gamma_*dt_;

    // timeint_gen_alpha = p(n+1) (Peter's genalpha)
    if (timealgo_ == INPAR::FLUID::timeint_npgenalpha)
    {
      // if not generalized-alpha: timefacrhs_=theta * dt_ = timefac_
      timefacpre_ = gamma_/alphaM_*dt_;
      timefacrhs_ = gamma_/alphaM_*dt_;
    }
    else if (timealgo_ == INPAR::FLUID::timeint_gen_alpha)
    {
      // used for weak boundary conditions and mixid hybrid
      timefac_ = alphaF_*gamma_*dt_;
      timefacpre_ = gamma_*dt_;
      timefacrhs_ = 1.0;
    }
    else if(timealgo_ == INPAR::FLUID::timeint_afgenalpha)
    {
      timefacpre_ = gamma_*alphaF_/alphaM_*dt_;
      timefacrhs_ = gamma_/alphaM_*dt_;
    }
    else
    {
      // if not generalized-alpha: timefacmat_p_=theta * dt_ = timefac_
      timefacpre_ = gamma_*alphaF_/alphaM_*dt_;
      // if not generalized-alpha: timefacrhs_=theta * dt_ = timefac_
      timefacrhs_ = gamma_*alphaF_/alphaM_*dt_;
    }
  }
  else // is_stationary == true
  {
    // set timefactor for stationary case to 1.0
    timefac_ = 1.0;
    timefacrhs_ = 1.0;
  }

  if (dt_ < 0.0 or theta_ < 0.0 or time_ < 0.0 or omtheta_ < 0.0 or gamma_ < 0.0
      or alphaF_ < 0.0 or alphaM_ < 0.0)
  {
    cout<<"dt_: "<<dt_<<endl;
    cout<<"theta_ "<<theta_<<endl;
    cout<<"time_ "<<time_<<endl;
    cout<<"omtheta_ "<<omtheta_<<endl;
    cout<<"gamma_ "<<gamma_<<endl;
    cout<<"alphaF_ "<<alphaF_<<endl;
    cout<<"alphaM_ "<<alphaM_<<endl;
    dserror("Negative (or no) time-integration parameter or time-step length supplied");
  }
}

//----------------------------------------------------------------------*
//  set turbulence parameters                            rasthofer 11/11|
//---------------------------------------------------------------------*/
void DRT::ELEMENTS::Fluid3ImplParameter::SetElementTurbulenceParameter( Teuchos::ParameterList& params )
{
  // get parameter lists
  Teuchos::ParameterList& turbmodelparams = params.sublist("TURBULENCE MODEL");
  Teuchos::ParameterList& turbmodelparamssgvisc = params.sublist("SUBGRID VISCOSITY");
  Teuchos::ParameterList& turbmodelparamsmfs = params.sublist("MULTIFRACTAL SUBGRID SCALES");

  //---------------------------------------------------------------------------------
  // parameter for subgrid-viscosity approach
  //---------------------------------------------------------------------------------

  // get flag for fine-scale subgrid-viscosity approach
  {
    const std::string fssgvdef = turbmodelparams.get<std::string>("FSSUGRVISC","No");

    if (fssgvdef == "Smagorinsky_all")        fssgv_ = INPAR::FLUID::smagorinsky_all;
    else if (fssgvdef == "Smagorinsky_small") fssgv_ = INPAR::FLUID::smagorinsky_small;
  }

  // get Smagorinsky model parameter for fine-scale subgrid viscosity
  // (Since either all-scale Smagorinsky model (i.e., classical LES model
  // as will be inititalized below) or fine-scale Smagorinsky model is
  // used (and never both), the same input parameter can be exploited.)
  if (fssgv_ != INPAR::FLUID::no_fssgv) Cs_ = turbmodelparamssgvisc.get<double>("C_SMAGORINSKY",0.0);

  //---------------------------------------------------------------------------------
  // parameter for turbulence approach
  //---------------------------------------------------------------------------------

  // the default action is no model
  turb_mod_action_ = INPAR::FLUID::no_model;

  // No turbulent flow: TURBULENCE_APPROACH = DNS
  if (turbmodelparams.get<std::string>("TURBULENCE_APPROACH", "none") == "CLASSICAL_LES")
  {
    if (is_stationary_ == true)
      dserror("Stationary turbulent flow does not make any sense");

    std::string& physical_turbulence_model = turbmodelparams.get<std::string>("PHYSICAL_MODEL");

    // --------------------------------------------------
    // standard constant coefficient Smagorinsky model
    if (physical_turbulence_model == "Smagorinsky")
    {
      // the classic Smagorinsky model only requires one constant parameter
      turb_mod_action_ = INPAR::FLUID::smagorinsky;
      Cs_              = turbmodelparamssgvisc.get<double>("C_SMAGORINSKY");
    }
    // --------------------------------------------------
    // Smagorinsky model with van Driest damping
    else if (physical_turbulence_model == "Smagorinsky_with_van_Driest_damping")
    {
      // that's only implemented for turbulent channel flow
      // wall function length is hard implemented
      if (turbmodelparamssgvisc.get<std::string>("CANONICAL_FLOW","no")
          !=
          "channel_flow_of_height_2")
          dserror("van_Driest_damping only for channel_flow_of_height_2\n");

      // for the Smagorinsky model with van Driest damping, we need
      // a viscous length to determine the y+ (heigth in wall units)
      turb_mod_action_ = INPAR::FLUID::smagorinsky_with_van_Driest_damping;

      // get parameters of model
      Cs_              = turbmodelparamssgvisc.get<double>("C_SMAGORINSKY");
      l_tau_           = turbmodelparamssgvisc.get<double>("CHANNEL_L_TAU");
    }

    // --------------------------------------------------
    // Smagorinsky model with dynamic Computation of Cs
    else if (physical_turbulence_model == "Dynamic_Smagorinsky")
    {
      turb_mod_action_ = INPAR::FLUID::dynamic_smagorinsky;

      // In the case of dynamic Smagorinsky:
      // Cs_ is calculated from Cs_sqrt_delta to compare it with the standard
      // it is stored in Cs_ after its calculation in CalcSubgrVisc
      Cs_ = 0.0;
      Cs_averaged_ = DRT::INPUT::IntegralValue<int>(turbmodelparamssgvisc,"C_SMAGORINSKY_AVERAGED");
    }
    else if (physical_turbulence_model == "Scale_Similarity")
    {
      turb_mod_action_ = INPAR::FLUID::scale_similarity;
      Cl_ = turbmodelparamsmfs.get<double>("C_SCALE_SIMILARITY");
    }
    else if (physical_turbulence_model == "Scale_Similarity_basic")
    {
      turb_mod_action_ = INPAR::FLUID::scale_similarity_basic;
      Cl_ = turbmodelparamsmfs.get<double>("C_SCALE_SIMILARITY");
    }
    else if (physical_turbulence_model == "Multifractal_Subgrid_Scales")
    {
      turb_mod_action_ = INPAR::FLUID::multifractal_subgrid_scales;

      // get parameters of model
      Csgs_ = turbmodelparamsmfs.get<double>("CSGS");
      Csgs_phi_ = turbmodelparamsmfs.get<double>("CSGS_PHI");

      if (turbmodelparamsmfs.get<string>("SCALE_SEPARATION") == "algebraic_multigrid_operator")
       alpha_ = 3.0;
      else if (turbmodelparamsmfs.get<string>("SCALE_SEPARATION") == "box_filter"
            or turbmodelparamsmfs.get<string>("SCALE_SEPARATION") == "geometric_multigrid_operator")
       alpha_ = 2.0;
      else
       dserror("Unknown filter type!");

      CalcN_ = DRT::INPUT::IntegralValue<int>(turbmodelparamsmfs,"CALC_N");

      N_ = turbmodelparamsmfs.get<double>("N");

      if (turbmodelparamsmfs.get<string>("REF_VELOCITY") == "strainrate")
       refvel_ = INPAR::FLUID::strainrate;
      else if (turbmodelparamsmfs.get<string>("REF_VELOCITY") == "resolved")
       refvel_ = INPAR::FLUID::resolved;
      else if (turbmodelparamsmfs.get<string>("REF_VELOCITY") == "fine_scale")
       refvel_ = INPAR::FLUID::fine_scale;
      else
       dserror("Unknown velocity!");

      if (turbmodelparamsmfs.get<string>("REF_LENGTH") == "cube_edge")
       reflength_ = INPAR::FLUID::cube_edge;
      else if (turbmodelparamsmfs.get<string>("REF_LENGTH") == "sphere_diameter")
       reflength_ = INPAR::FLUID::sphere_diameter;
      else if (turbmodelparamsmfs.get<string>("REF_LENGTH") == "streamlength")
       reflength_ = INPAR::FLUID::streamlength;
      else if (turbmodelparamsmfs.get<string>("REF_LENGTH") == "gradient_based")
       reflength_ = INPAR::FLUID::gradient_based;
      else if (turbmodelparamsmfs.get<string>("REF_LENGTH") == "metric_tensor")
       reflength_ = INPAR::FLUID::metric_tensor;
      else
       dserror("Unknown length!");

      c_nu_ = turbmodelparamsmfs.get<double>("C_NU");
      c_diff_ = turbmodelparamsmfs.get<double>("C_DIFF"); //loma only

      near_wall_limit_ = DRT::INPUT::IntegralValue<int>(turbmodelparamsmfs,"NEAR_WALL_LIMIT");

      if (turbmodelparamsmfs.get<string>("EVALUATION_B") == "element_center")
      B_gp_ = false;
      else if (turbmodelparamsmfs.get<string>("EVALUATION_B") == "integration_point")
      B_gp_ = true;
      else
        dserror("Unknown evaluation point!");

      beta_ = turbmodelparamsmfs.get<double>("BETA");
    }
    else
    {
      dserror("Up to now, only Smagorinsky, Scale Similarity and Multifractal Subgrid Scales are available");
    }
  } // end if(Classical LES)
}


//----------------------------------------------------------------------*
//  set loma parameters                                  rasthofer 03/12|
//---------------------------------------------------------------------*/
void DRT::ELEMENTS::Fluid3ImplParameter::SetElementLomaParameter( Teuchos::ParameterList& params )
{
  // get parameter lists
  Teuchos::ParameterList& lomaparams = params.sublist("LOMA");
  Teuchos::ParameterList& stabparams = params.sublist("STABILIZATION");
  Teuchos::ParameterList& turbmodelparamsmfs = params.sublist("MULTIFRACTAL SUBGRID SCALES");

  //---------------------------------------------------------------------------------
  // material update with subgrid-scale temperature
  //---------------------------------------------------------------------------------

  update_mat_ = lomaparams.get<bool>("update material",false);

  //---------------------------------------------------------------------------------
  // parameter for additional rbvmm terms in continuity equation
  //---------------------------------------------------------------------------------

  conti_supg_     = DRT::INPUT::IntegralValue<INPAR::FLUID::SUPG>(stabparams,"LOMA_CONTI_SUPG");
  conti_cross_    = DRT::INPUT::IntegralValue<INPAR::FLUID::CrossStress>(stabparams,"LOMA_CONTI_CROSS_STRESS");
  conti_reynolds_ = DRT::INPUT::IntegralValue<INPAR::FLUID::ReynoldsStress>(stabparams,"LOMA_CONTI_REYNOLDS_STRESS");

  //---------------------------------------------------------------------------------
  // parameter for additional multifractal subgrid-scale terms
  //---------------------------------------------------------------------------------

  if (turb_mod_action_ == INPAR::FLUID::multifractal_subgrid_scales)
   multifrac_loma_conti_ = DRT::INPUT::IntegralValue<int>(turbmodelparamsmfs,"LOMA_CONTI");

  return;
}


//----------------------------------------------------------------------*/
// print fluid parameter to screen (AE 01-11)
//----------------------------------------------------------------------*/
void DRT::ELEMENTS::Fluid3ImplParameter::PrintFluidParameter()
{
    cout << endl << "|-----------------------------------------------------------------------------" << endl;
    cout << "|  General Fluid parameter: " << endl;
    cout << "|-----------------------------------------------------------------------------" << endl;
    //! flag SetGeneralParameter was called
    cout << "|    method SetElmentGeneralFluidParameter was called:    " << set_general_fluid_parameter_ << endl;
    //! flag to (de)activate generalized-alpha time-integration scheme
    cout << "|    generalized alpha time integration active:    " << is_genalpha_ << endl;
    //! flag to (de)activate generalized-alpha-np1 time-integration scheme
    cout << "|    generalized alpha time integration np:    " << is_genalpha_np_ << endl;
    //! flag to (de)activate conservative formulation
    cout << "|    conservative formulation:    " << is_conservative_ << endl;
    //! flag to (de)activate stationary formulation
    cout << "|    steady state:    " << is_stationary_ << endl;
    //! flag to (de)activate Newton linearization
    cout << "|    Newton linearization:    " << is_newton_ << endl;
    //! flag to (de)activate second derivatives
    cout << "|    use inconsistent:    " << is_inconsistent_ << endl;
    //! flag to (de)activate potential reactive terms
    cout << "|    reaction term:    " << reaction_ << endl;
    //! flag to (de)aktivate porous darcy flow
    cout << "|    darcy equation:    " << darcy_ << endl;
    //! flag to (de)aktivate reaction due to topology optimization
    cout << "|    reaction term due to topology optimization:    " << reaction_topopt_ << endl;
    //! Flag for physical type of the fluid flow (incompressible, loma, varying_density, Boussinesq)
    cout << "|    physical type:    "<< physicaltype_ << endl;
    //! Flag to (de)activate time-dependent subgrid stabilization
    cout << "|    time-dependent subgrid stabilization:    " << tds_ << endl;
    //! Flag to (de)activate time-dependent term in large-scale momentum equation
    cout << "|    time dependent term:    " << transient_ << endl;
    //! Flag to (de)activate PSPG stabilization
    cout << "|    PSPG:    " << pspg_ << endl;
    //! Flag to (de)activate SUPG stabilization
    cout << "|    SUPG:    " << supg_<< endl ;
    //! Flag to (de)activate viscous term in residual-based stabilization
    cout << "|    VSTAB:    " << vstab_ << endl;
    //! Flag to (de)activate least-squares stabilization of continuity equation
    cout << "|    Grad-Div-Stab:    " << cstab_ << endl ;
    //! Flag to (de)activate reactive term in residual-based stabilization
    cout << "|    reactive stabilization:    " << rstab_ << endl;
    //! Flag to (de)activate cross-stress term -> residual-based VMM
    cout << "|    cross-stress term:    " << cross_ << endl;
    //! Flag to (de)activate Reynolds-stress term -> residual-based VMM
    cout << "|    Reynolds-stress term:    " << reynolds_ << endl;
    //! (sign) factor for viscous and reactive stabilization terms
    cout << "|    viscous and reactive stabilization factor:    " << viscreastabfac_ << endl;
    //! Flag to define tau
    cout << "|    Definition of stabilization parameter:    " << whichtau_ << endl;
    //! flag to (de)activate fine-scale subgrid viscosity
    cout << "|    fine-scale subgrid viscosity::    " << fssgv_ << endl;
    //! flag for material evaluation at Gaussian integration points
    cout << "|    material evaluation at Gaussian integration points:    " << mat_gp_ << endl;
    //! flag for stabilization parameter evaluation at Gaussian integration points
    cout << "|    stabilization parameter evaluation at Gaussian integration points:    " << tau_gp_ << endl;
    cout << "|---------------------------------------------------------------------------" << endl;

    cout << endl << "|---------------------------------------------------------------------------" << endl;
    cout << "|  Time parameter: " << endl;
    cout << "|---------------------------------------------------------------------------" << endl;
    //! time algorithm
    cout << "|    time algorithm:    " << timealgo_ << endl;
    //! actual time to evaluate the body BC
    cout << "|    time:    " << time_ << endl;
    //! time-step length
    cout << "|    time step:    " << dt_ << endl;
    //! timefac = dt_ * ("pseudo"-)theta_
    cout << "|    time factor:    " << timefac_ << endl;
    //! factor for left-hand side due to one-step-theta time-integration scheme
    cout << "|    theta:    " << theta_ << endl;
    //! factor for right-hand side due to one-step-theta time-integration scheme
    cout << "|    (1-theta):    " << omtheta_ << endl;
    //! generalised-alpha parameter (connecting velocity and acceleration)
    cout << "|    gamma:    " << gamma_ << endl;
    //! generalised-alpha parameter (velocity)
    cout << "|    alpha_F:    " << alphaF_ << endl;
    //! generalised-alpha parameter (acceleration)
    cout << "|    alpha_M:    " << alphaM_ << endl;
    //! generalised-alpha parameter, alphaF_*gamma_*dt_
    cout << "|    time factor mat_u:    " << afgdt_ << endl;
    //! time integration factor for the right hand side (boundary elements)
    cout << "|    time factor rhs:    " << timefacrhs_ << endl;
    //! time integration factor for the left hand side (pressure)
    cout << "|    time factor mat_p:    " << timefacpre_ << endl;
    cout << "|---------------------------------------------------------------------------" << endl;

    cout << endl << "|---------------------------------------------------------------------------" << endl;
    cout << "|  Turbulence parameter: " << endl;
    cout << "|---------------------------------------------------------------------------" << endl;
    //! flag to define turbulence model
    cout << "|    turbulence model:    " << turb_mod_action_ << endl;
    //! smagorinsky constant
    cout << "|    smagorinsky constant:    " << Cs_ << endl;
    //! comment missing
    cout << "|    Cs_averaged_ is    " << Cs_averaged_ << endl;
    //! scale similarity constant
    cout << "|    Cl_ is    " << Cl_ << endl;
    /// multifractal subgrid-scales
    cout << "|    Csgs_ is    " << Csgs_ << endl;
    //! comment missing
    cout << "|    alpha_ is    " << alpha_ << endl;
    //! comment missing
    cout << "|    CalcN_ is    " << CalcN_ << endl;
    //! comment missing
    cout << "|    N_ is    " << N_ << endl;
    //! comment missing
    cout << "|    refvel_ is    " << refvel_ << endl;
    //! comment missing
    cout << "|    reflength_ is    " << reflength_ << endl;
    //! comment missing
    cout << "|    c_nu_ is    " << c_nu_ << endl;
    //! comment missing
    cout << "|    near_wall_limit_ is    " << near_wall_limit_ << endl;
    //! comment missing
    cout << "|    B_gp_ is    " << B_gp_ << endl;
    //! comment missing
    cout << "|    beta_ is    " << beta_ << endl;
    //! channel length to normalize the normal wall distance
    cout << "|    channel length to normalize the normal wall distance:    " << l_tau_ << endl;
    cout << "|---------------------------------------------------------------------------" << endl;

}
