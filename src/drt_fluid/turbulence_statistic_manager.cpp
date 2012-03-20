/*!

Manage the computation of averages for several
canonical flows like channel flow, flow around a square
cylinder, flow in a lid driven cavity, flow over a backward-facing step etc.

The manager is intended to remove as much of the averaging
overhead as possible from the time integration method.

Maintainer: Ursula Rasthofer
            rasthofer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236

*/
#ifdef CCADISCRET

#include "turbulence_statistic_manager.H"
#include "../drt_fluid/fluid_genalpha_integration.H"
#include "../drt_fluid/fluidimplicitintegration.H"
#include "../drt_combust/combust_fluidimplicitintegration.H"
#include "../drt_fluid/fluid_utils.H" // for LiftDrag
#include "../drt_lib/drt_dofset_independent_pbc.H"
#include "../drt_fluid/turbulence_statistics_mean_general.H"
#include "../drt_fluid/turbulence_statistics_ccy.H"
#include "../drt_fluid/turbulence_statistics_cha.H"
#include "../drt_fluid/turbulence_statistics_bcf.H"
#include "../drt_fluid/turbulence_statistics_ldc.H"
#include "../drt_fluid/turbulence_statistics_bfs.H"
#include "../drt_fluid/turbulence_statistics_oracles.H"
#include "../drt_fluid/turbulence_statistics_sqc.H"

namespace FLD
{

  /*----------------------------------------------------------------------

    Standard Constructor for Genalpha time integration (public) (Gammis Fluid Algo!!!!!!)

  ----------------------------------------------------------------------*/
  TurbulenceStatisticManager::TurbulenceStatisticManager(FluidGenAlphaIntegration& fluid)
    :
    dt_              (fluid.dt_       ),
    alphaM_          (fluid.alphaM_   ),
    alphaF_          (fluid.alphaF_   ),
    gamma_           (fluid.gamma_    ),
    density_         (fluid.density_  ),
    discret_         (fluid.discret_  ),
//    params_          (fluid.params_   ),
    alefluid_        (fluid.alefluid_ ),
    myaccnp_         (fluid.accnp_    ),
    myaccn_          (fluid.accn_     ),
    myaccam_         (fluid.accam_    ),
    myvelnp_         (fluid.velnp_    ),
    myveln_          (fluid.veln_     ),
    myvelaf_         (fluid.velaf_    ),
    myscanp_         (fluid.scaaf_    ),
    mydispnp_        (fluid.dispnp_   ),
    mydispn_         (fluid.dispn_    ),
    mygridveln_      (fluid.gridveln_ ),
    mygridvelaf_     (fluid.gridvelaf_),
    myforce_         (fluid.force_    ),
    myfilteredvel_   (null            ),
    myfilteredreystr_(null            ),
    myfsvelaf_       (null            )
  {
    params_ = Teuchos::rcp(&fluid.params_);
    // get density

    // activate the computation of subgrid dissipation,
    // residuals etc
    subgrid_dissipation_=true;

    // toogle statistics output for turbulent inflow
    inflow_ = DRT::INPUT::IntegralValue<int>(params_->sublist("TURBULENT INFLOW"),"TURBULENTINFLOW")==true;

    // the flow parameter will control for which geometry the
    // sampling is done
    if(fluid.special_flow_=="channel_flow_of_height_2")
    {
      flow_=channel_flow_of_height_2;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_channel_=rcp(new TurbulenceStatisticsCha(discret_            ,
                                                          alefluid_           ,
                                                          mydispnp_           ,
                                                          *params_             ,
                                                          subgrid_dissipation_));
    }
    else if(fluid.special_flow_=="loma_channel_flow_of_height_2")
    {
      flow_=loma_channel_flow_of_height_2;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_channel_=rcp(new TurbulenceStatisticsCha(discret_            ,
                                                          alefluid_           ,
                                                          mydispnp_           ,
                                                          *params_             ,
                                                          subgrid_dissipation_));
    }
    else if(fluid.special_flow_=="scatra_channel_flow_of_height_2")
    {
      flow_=scatra_channel_flow_of_height_2;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_channel_=rcp(new TurbulenceStatisticsCha(discret_            ,
                                                          alefluid_           ,
                                                          mydispnp_           ,
                                                          *params_             ,
                                                          subgrid_dissipation_));;
    }
    else if(fluid.special_flow_=="lid_driven_cavity")
    {
      flow_=lid_driven_cavity;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ldc_    =rcp(new TurbulenceStatisticsLdc(discret_,*params_));
    }
    else if(fluid.special_flow_=="loma_lid_driven_cavity")
    {
      flow_=loma_lid_driven_cavity;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ldc_    =rcp(new TurbulenceStatisticsLdc(discret_,*params_));
    }
    else if(fluid.special_flow_=="backward_facing_step")
    {
      flow_=backward_facing_step;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = rcp(new TurbulenceStatisticsBfs(discret_,*params_,"geometry_DNS_incomp_flow"));
      if (inflow_)
        dserror("Sorry, no inflow generation for gammi-style fluid!");
    }
    else if(fluid.special_flow_=="loma_backward_facing_step")
    {
      flow_=loma_backward_facing_step;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = rcp(new TurbulenceStatisticsBfs(discret_,*params_,"geometry_LES_flow_with_heating"));
      if (inflow_)
        dserror("Sorry, no inflow generation for gammi-style fluid!");
    }
    else if(fluid.special_flow_=="combust_oracles")
    {
      flow_=combust_oracles;

      if(discret_->Comm().MyPID()==0)
        std::cout << "---  setting up turbulence statistics manager for ORACLES ..." << std::flush;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_oracles_ = rcp(new COMBUST::TurbulenceStatisticsORACLES(discret_,*params_,"geometry_ORACLES",false));

      if(discret_->Comm().MyPID()==0)
        std::cout << " done" << std::endl;
    }
    else if(fluid.special_flow_=="square_cylinder")
    {
      flow_=square_cylinder;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_sqc_    =rcp(new TurbulenceStatisticsSqc(discret_,*params_));
    }
    else if(fluid.special_flow_=="square_cylinder_nurbs")
    {
      flow_=square_cylinder_nurbs;

      // do the time integration independent setup
      Setup();
    }
    else if(fluid.special_flow_=="rotating_circular_cylinder_nurbs")
    {
      flow_=rotating_circular_cylinder_nurbs;
      const bool withscatra = false;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ccy_=rcp(new TurbulenceStatisticsCcy(discret_            ,
                                                      alefluid_           ,
                                                      mydispnp_           ,
                                                      *params_             ,
                                                      withscatra));
    }
    else if(fluid.special_flow_=="rotating_circular_cylinder_nurbs_scatra")
    {
      flow_=rotating_circular_cylinder_nurbs_scatra;
      const bool withscatra = true;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ccy_=rcp(new TurbulenceStatisticsCcy(discret_            ,
                                                      alefluid_           ,
                                                      mydispnp_           ,
                                                      *params_             ,
                                                      withscatra));
    }
    else if(fluid.special_flow_=="time_averaging")
    {
      flow_=time_averaging;

      // do the time integration independent setup
      Setup();
    }
    else
    {
      flow_=no_special_flow;

      // do the time integration independent setup
      Setup();
    }

    // allocate one instance of the flow independent averaging procedure
    // providing colorful output for paraview
    {
      ParameterList *  modelparams =&(params_->sublist("TURBULENCE MODEL"));

      string homdir = modelparams->get<string>("HOMDIR","not_specified");

      if(flow_==rotating_circular_cylinder_nurbs_scatra)
      {
        // additional averaging of scalar field
        statistics_general_mean_
        =rcp(new TurbulenceStatisticsGeneralMean(
            discret_,
            homdir,
            density_,
            fluid.VelPresSplitter(),true));
      }
      else
      {
        statistics_general_mean_
        =rcp(new TurbulenceStatisticsGeneralMean(
            discret_,
            homdir,
            density_,
            fluid.VelPresSplitter(),false));
      }
    }
    return;

  }

  /*----------------------------------------------------------------------

    Standard Constructor for standard time integration (public)

  ----------------------------------------------------------------------*/
  TurbulenceStatisticManager::TurbulenceStatisticManager(FluidImplicitTimeInt& fluid)
    :
    dt_              (fluid.dta_           ),
    alphaM_          (0.0                  ),
    alphaF_          (0.0                  ),
    gamma_           (0.0                  ),
    density_         (1.0                  ),
    discret_         (fluid.discret_       ),
    params_          (fluid.params_        ),
    alefluid_        (fluid.alefluid_      ),
    myaccnp_         (fluid.accnp_         ),
    myaccn_          (fluid.accn_          ),
    myaccam_         (fluid.accam_         ),
    myvelnp_         (fluid.velnp_         ),
    myveln_          (fluid.veln_          ),
    myvelaf_         (fluid.velaf_         ),
    myscanp_         (fluid.scaaf_         ),
    mydispnp_        (fluid.dispnp_        ),
    mydispn_         (fluid.dispn_         ),
    mygridveln_      (fluid.gridv_         ),
    mygridvelaf_     (null                 ),
    myforce_         (fluid.trueresidual_  ),
    myfilteredvel_   (fluid.filteredvel_   ),
    myfilteredreystr_(fluid.filteredreystr_),
    myfsvelaf_       (fluid.fsvelaf_       )
  {

    //subgrid_dissipation_=true;
    //why: read comment in CalcDissipation() (fluid3_impl.cpp)
    subgrid_dissipation_=false;

    // toogle statistics output for turbulent inflow
    inflow_ = DRT::INPUT::IntegralValue<int>(params_->sublist("TURBULENT INFLOW"),"TURBULENTINFLOW")==true;

    // the flow parameter will control for which geometry the
    // sampling is done
    if(fluid.special_flow_=="channel_flow_of_height_2")
    {
      flow_=channel_flow_of_height_2;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_channel_=rcp(new TurbulenceStatisticsCha(discret_            ,
                                                          alefluid_           ,
                                                          mydispnp_           ,
                                                          *params_             ,
                                                          subgrid_dissipation_));
    }
    else if(fluid.special_flow_=="loma_channel_flow_of_height_2")
    {
      flow_=loma_channel_flow_of_height_2;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_channel_=rcp(new TurbulenceStatisticsCha(discret_            ,
                                                          alefluid_           ,
                                                          mydispnp_           ,
                                                          *params_             ,
                                                          subgrid_dissipation_));
    }
    else if(fluid.special_flow_=="scatra_channel_flow_of_height_2")
    {
      flow_=scatra_channel_flow_of_height_2;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_channel_=rcp(new TurbulenceStatisticsCha(discret_            ,
                                                          alefluid_           ,
                                                          mydispnp_           ,
                                                          *params_             ,
                                                          subgrid_dissipation_));
    }
    else if(fluid.special_flow_=="lid_driven_cavity")
    {
      flow_=lid_driven_cavity;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ldc_    =rcp(new TurbulenceStatisticsLdc(discret_,*params_));
    }
    else if(fluid.special_flow_=="loma_lid_driven_cavity")
    {
      flow_=loma_lid_driven_cavity;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ldc_    =rcp(new TurbulenceStatisticsLdc(discret_,*params_));
    }
    else if(fluid.special_flow_=="backward_facing_step")
    {
      flow_=backward_facing_step;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = rcp(new TurbulenceStatisticsBfs(discret_,*params_,"geometry_DNS_incomp_flow"));

      // build statistics manager for inflow channel flow
      if (inflow_)
      {
        if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2")
        {
          // allocate one instance of the averaging procedure for the flow under consideration
          statistics_channel_=rcp(new TurbulenceStatisticsCha(discret_,
                                                              alefluid_,
                                                              mydispnp_,
                                                              *params_,
                                                              subgrid_dissipation_));
        }
      }
    }
    else if(fluid.special_flow_=="loma_backward_facing_step")
    {
      flow_=loma_backward_facing_step;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = rcp(new TurbulenceStatisticsBfs(discret_,*params_,"geometry_LES_flow_with_heating"));

      // build statistics manager for inflow channel flow
      if (inflow_)
      {
        if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2"
         or params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="loma_channel_flow_of_height_2")
        {
          // allocate one instance of the averaging procedure for the flow under consideration
          statistics_channel_=rcp(new TurbulenceStatisticsCha(discret_,
                                                              alefluid_,
                                                              mydispnp_,
                                                              *params_,
                                                              subgrid_dissipation_));
        }
      }
    }
    else if(fluid.special_flow_=="combust_oracles")
    {
      flow_=combust_oracles;

      if(discret_->Comm().MyPID()==0)
        std::cout << "---  setting up turbulence statistics manager for ORACLES ..." << std::flush;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_oracles_ = rcp(new COMBUST::TurbulenceStatisticsORACLES(discret_,*params_,"geometry_ORACLES",false));

      // build statistics manager for inflow channel flow
      if (inflow_)
      {
        if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2")
        {
          // allocate one instance of the averaging procedure for the flow under consideration
          statistics_channel_=rcp(new TurbulenceStatisticsCha(discret_,
                                                              alefluid_,
                                                              mydispnp_,
                                                              *params_,
                                                              subgrid_dissipation_));
        }
      }

      if(discret_->Comm().MyPID()==0)
        std::cout << " done" << std::endl;
    }
    else if(fluid.special_flow_=="square_cylinder")
    {
      flow_=square_cylinder;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_sqc_    =rcp(new TurbulenceStatisticsSqc(discret_,*params_));
    }
    else if(fluid.special_flow_=="square_cylinder_nurbs")
    {
      flow_=square_cylinder_nurbs;

      // do the time integration independent setup
      Setup();
    }
    else if(fluid.special_flow_=="rotating_circular_cylinder_nurbs")
    {
      flow_=rotating_circular_cylinder_nurbs;
      const bool withscatra = false;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ccy_=rcp(new TurbulenceStatisticsCcy(discret_            ,
                                                      alefluid_           ,
                                                      mydispnp_           ,
                                                      *params_             ,
                                                      withscatra));
    }
    else if(fluid.special_flow_=="rotating_circular_cylinder_nurbs_scatra")
    {
      flow_=rotating_circular_cylinder_nurbs_scatra;
      const bool withscatra = true;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ccy_=rcp(new TurbulenceStatisticsCcy(discret_            ,
                                                      alefluid_           ,
                                                      mydispnp_           ,
                                                      *params_             ,
                                                      withscatra));
    }
    else if(fluid.special_flow_=="time_averaging")
    {
      flow_=time_averaging;

      // do the time integration independent setup
      Setup();
    }
    else
    {
      flow_=no_special_flow;

      // do the time integration independent setup
      Setup();
    }

    // allocate one instance of the flow independent averaging procedure
    // providing colorful output for paraview
    {
      ParameterList *  modelparams =&(params_->sublist("TURBULENCE MODEL"));

      string homdir = modelparams->get<string>("HOMDIR","not_specified");

      statistics_general_mean_
        =rcp(new TurbulenceStatisticsGeneralMean(
               discret_,
               homdir,
               density_,
               fluid.VelPresSplitter(),
               false // scatra support not yet activated.
               ));
    }

    return;

  }


  /*----------------------------------------------------------------------

    Standard Constructor for combustion One-Step-Theta time integration (public)

  ----------------------------------------------------------------------*/
  TurbulenceStatisticManager::TurbulenceStatisticManager(CombustFluidImplicitTimeInt& timeint)
    :
    dt_              (timeint.dta_         ),
    alphaM_          (0.0                  ),
    alphaF_          (0.0                  ),
    gamma_           (0.0                  ),
    density_         (1.0                  ),
    discret_         (timeint.discret_     ),
    params_          (timeint.params_      ),
    alefluid_        (false                ),
    myaccnp_         (null                 ), // size is not fixed as we deal with xfem problems
    myaccn_          (null                 ), // size is not fixed
    myaccam_         (null                 ), // size is not fixed
    myveln_          (null                 ), // size is not fixed
    myvelaf_         (null                 ), // size is not fixed
    myscanp_         (null                 ),
    mydispnp_        (Teuchos::null        ),
    mydispn_         (null                 ),
    mygridveln_      (null                 ),
    mygridvelaf_     (null                 ),
    myfilteredvel_   (null                 ),
    myfilteredreystr_(null                 ),
    myfsvelaf_       (null                 )
  {

    // subgrid dissipation
    subgrid_dissipation_ = false;
    // boolean for statistics of transported scalar
    bool withscatra = false;
    // toogle statistics output for turbulent inflow
    inflow_ = DRT::INPUT::IntegralValue<int>(params_->sublist("TURBULENT INFLOW"),"TURBULENTINFLOW")==true;

    // the flow parameter will control for which geometry the
    // sampling is done
    if(timeint.special_flow_=="bubbly_channel_flow")
    {
      flow_=bubbly_channel_flow;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_channel_multiphase_ = rcp(new COMBUST::TurbulenceStatisticsBcf(discret_, *params_ ));
    }
    else if(timeint.special_flow_=="combust_oracles")
    {
      flow_=combust_oracles;
      // statistics for transported scalar (G-function)
      withscatra = true;

      if(discret_->Comm().MyPID()==0)
        std::cout << "---  setting up turbulence statistics manager for ORACLES ..." << std::flush;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_oracles_ = rcp(new COMBUST::TurbulenceStatisticsORACLES(discret_,*params_,"geometry_ORACLES",withscatra));

      // build statistics manager for inflow channel flow
      if (inflow_)
      {
        if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2")
        {
          // allocate one instance of the averaging procedure for the flow under consideration
          statistics_channel_=rcp(new TurbulenceStatisticsCha(discret_,
                                                              alefluid_,
                                                              mydispnp_,
                                                              *params_,
                                                              subgrid_dissipation_));
        }
      }

      if(discret_->Comm().MyPID()==0)
        std::cout << " done" << std::endl;
    }
    else
    {
      flow_=no_special_flow;

      // do the time integration independent setup
      Setup();
    }

    statistics_general_mean_ = Teuchos::null;
    // allocate one instance of the flow independent averaging procedure
    // providing colorful output for paraview
    {
      ParameterList *  modelparams =&(params_->sublist("TURBULENCE MODEL"));

      string homdir = modelparams->get<string>("HOMDIR","not_specified");

      statistics_general_mean_ = Teuchos::rcp(new TurbulenceStatisticsGeneralMean(
          discret_,
          timeint.standarddofset_,
          homdir,
          density_,  // density
          *timeint.velpressplitterForOutput_,
          withscatra // statistics for transported scalar
      ));
    }

    return;

  }


  /*
    Destructor
  */
  TurbulenceStatisticManager::~TurbulenceStatisticManager()
  {

    return;
  }

  /*----------------------------------------------------------------------

    Time integration independent setup called by Constructor (private)

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::Setup()
  {

    ParameterList *  modelparams =&(params_->sublist("TURBULENCE MODEL"));

    smagorinsky_=false;
    scalesimilarity_=false;
    multifractal_=false;
    if (modelparams->get<string>("TURBULENCE_APPROACH","DNS_OR_RESVMM_LES")
        ==
        "CLASSICAL_LES")
    {
      // check if we want to compute averages of Smagorinsky
      // constants, effective viscosities etc
      if(modelparams->get<string>("PHYSICAL_MODEL","no_model")
         ==
         "Dynamic_Smagorinsky"
         ||
         modelparams->get<string>("PHYSICAL_MODEL","no_model")
         ==
         "Smagorinsky_with_van_Driest_damping"
         ||
         modelparams->get<string>("PHYSICAL_MODEL","no_model")
         ==
         "Smagorinsky"
        )
      {
//        if(discret_->Comm().MyPID()==0)
//        {
//          cout << "                             Initialising output for Smagorinsky type models\n\n\n";
//          fflush(stdout);
//        }

        smagorinsky_=true;
      }
      // check if we want to compute averages of scale similarity
      // quantities (tau_SFS)
      else if(modelparams->get<string>("PHYSICAL_MODEL","no_model")
              ==
              "Scale_Similarity")
      {
//        if(discret_->Comm().MyPID()==0)
//        {
//          cout << "                             Initializing output for scale similarity type models\n\n\n";
//          fflush(stdout);
//        }

        scalesimilarity_=true;
      }
      // check if we want to compute averages of multifractal
      // quantities (N, B)
      else if(modelparams->get<string>("PHYSICAL_MODEL","no_model")
           ==
           "Multifractal_Subgrid_Scales")
      {
//        if(discret_->Comm().MyPID()==0)
//        {
//          cout << "                             Initializing output for multifractal subgrid scales type models\n\n\n";
//          fflush(stdout);
//        }

        multifractal_=true;
      }
    }

    // parameters for sampling/dumping period
    if (flow_ != no_special_flow)
    {
      samstart_  = modelparams->get<int>("SAMPLING_START",1         );
      samstop_   = modelparams->get<int>("SAMPLING_STOP", 1000000000);
      dumperiod_ = modelparams->get<int>("DUMPING_PERIOD",1         );
    }
    else
    {
      samstart_  =0;
      samstop_   =0;
      dumperiod_ =0;
    }


    if(discret_->Comm().MyPID()==0)
    {

      if (flow_ == channel_flow_of_height_2 or
          flow_ == loma_channel_flow_of_height_2 or
          flow_ == scatra_channel_flow_of_height_2 or
          flow_ == bubbly_channel_flow)
      {
        string homdir
          =
          modelparams->get<string>("HOMDIR","not_specified");

        if(homdir!="xy" && homdir!="xz" && homdir!="yz")
        {
          dserror("need two homogeneous directions to do averaging in plane channel flows\n");
        }

        cout << "Additional output          : " ;
        cout << "Turbulence statistics are evaluated ";
        cout << "for a turbulent channel flow.\n";
        cout << "                             " ;
        cout << "The solution is averaged over the homogeneous ";
        cout << homdir;
        cout << " plane and over time.\n";
        cout << "\n";
        cout << "                             " ;
        cout << "Sampling period: steps " << samstart_ << " to ";
        cout << modelparams->get<int>("SAMPLING_STOP",1000000000) << ".\n";

        int dumperiod = modelparams->get<int>("DUMPING_PERIOD",1);


        if(dumperiod == 0)
        {
          cout << "                             " ;
          cout << "Using standalone records (i.e. start from 0 for a new record)\n";
        }
        else
        {
          cout << "                             " ;
          cout << "Volker-style incremental dumping is used (";
          cout << dumperiod << ")" << endl;
        }

        cout << endl;
        cout << endl;
      }
    }

    if(discret_->Comm().MyPID()==0)
    {
      if (flow_ == combust_oracles)
      {
        samstart_  = modelparams->get<int>("SAMPLING_START",1);
        samstop_   = modelparams->get<int>("SAMPLING_STOP", 1000000000);
        dumperiod_ = 0; // used as switch for the multi-record statistic output

        string homdir = modelparams->get<string>("HOMDIR","not_specified");
        if(homdir!="not_specified")
          dserror("there is no homogeneous direction for the ORACLES problem\n");

      }
    }

    return;
  }


  /*----------------------------------------------------------------------

    Store values computed during the element call

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::StoreElementValues(int step)
  {

    // sampling takes place only in the sampling period
    if(step>=samstart_ && step<=samstop_ && flow_ != no_special_flow)
    {
      switch(flow_)
      {
      case channel_flow_of_height_2:
      case loma_channel_flow_of_height_2:
      {
        // add computed dynamic Smagorinsky quantities
        // (effective viscosity etc. used during the computation)
        if(smagorinsky_) statistics_channel_->AddDynamicSmagorinskyQuantities();
        break;
      }
      default:
      {
        // there are no values to be stored in these cases
        break;
      }
    }
  }

    return;
  }

  /*----------------------------------------------------------------------

    Store values computed during the element call

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::StoreNodalValues(
       int                        step,
       const RCP<Epetra_Vector>   stress12)
  {
    // sampling takes place only in the sampling period
    if(step>=samstart_ && step<=samstop_ && flow_ != no_special_flow)
    {
      switch(flow_)
      {
      case channel_flow_of_height_2:
      case loma_channel_flow_of_height_2:
      {
        // add computed subfilter stress
        if(scalesimilarity_) statistics_channel_->AddSubfilterStresses(stress12);
        break;
      }
      default:
      {
        // there are no values to be stored in these cases
        break;
      }
    }
  }

    return;
  }

  /*----------------------------------------------------------------------

    Include current quantities in the time averaging procedure

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::DoTimeSample(int          step,
                                                double       time,
                                                const double eosfac
                                                )
  {
    // sampling takes place only in the sampling period
    if(step>=samstart_ && step<=samstop_ && flow_ != no_special_flow)
    {
      double tcpu=Teuchos::Time::wallTime();

      //--------------------------------------------------
      // calculate means, fluctuations etc of velocity,
      // pressure, boundary forces etc.
      switch(flow_)
      {
      case channel_flow_of_height_2:
      {
        if(statistics_channel_==null)
          dserror("need statistics_channel_ to do a time sample for a turbulent channel flow");

        statistics_channel_->DoTimeSample(myvelnp_,*myforce_);
        break;
      }
      case loma_channel_flow_of_height_2:
      {
        if(statistics_channel_==null)
          dserror("need statistics_channel_ to do a time sample for a turbulent channel flow at low Mach number");

        statistics_channel_->DoLomaTimeSample(myvelnp_,myscanp_,*myforce_,eosfac);
        break;
      }
      case scatra_channel_flow_of_height_2:
      {
        if(statistics_channel_==null)
          dserror("need statistics_channel_ to do a time sample for a turbulent passive scalar transport in channel");

        statistics_channel_->DoScatraTimeSample(myvelnp_,myscanp_,*myforce_);
        break;
      }
      case lid_driven_cavity:
      {
        if(statistics_ldc_==null)
          dserror("need statistics_ldc_ to do a time sample for a cavity flow");

        statistics_ldc_->DoTimeSample(myvelnp_);
        break;
      }
      case loma_lid_driven_cavity:
      {
        if(statistics_ldc_==null)
          dserror("need statistics_ldc_ to do a time sample for a cavity flow at low Mach number");

        statistics_ldc_->DoLomaTimeSample(myvelnp_,myscanp_,*myforce_,eosfac);
        break;
      }
      case backward_facing_step:
      {
        if(statistics_bfs_==null)
          dserror("need statistics_bfs_ to do a time sample for a flow over a backward-facing step");

        statistics_bfs_->DoTimeSample(myvelnp_);

        // do time sample for inflow channel flow
        if (inflow_)
        {
          if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2")
          {
            statistics_channel_->DoTimeSample(myvelnp_,*myforce_);
          }
        }

        break;
      }
      case loma_backward_facing_step:
      {
        if(statistics_bfs_==null)
          dserror("need statistics_bfs_ to do a time sample for a flow over a backward-facing step at low Mach number");

        if (DRT::INPUT::get<INPAR::FLUID::PhysicalType>(*params_, "Physical Type") == INPAR::FLUID::incompressible)
        {
          statistics_bfs_->DoTimeSample(myvelnp_);

          // do time sample for inflow channel flow
          if (inflow_)
          {
            if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2")
            {
              statistics_channel_->DoTimeSample(myvelnp_,*myforce_);
            }
          }
        }
        else
        {
          statistics_bfs_->DoLomaTimeSample(myvelnp_,myscanp_,eosfac);

          // do time sample for inflow channel flow
          if (inflow_)
          {
            if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="loma_channel_flow_of_height_2")
            {
              statistics_channel_->DoLomaTimeSample(myvelnp_,myscanp_,*myforce_,eosfac);
            }
          }
        }
        break;
      }
      case combust_oracles:
      {
        subgrid_dissipation_ = false;

        if(statistics_oracles_==null)
          dserror("need statistics_oracles_ to do a time sample for an ORACLES flow step");

        statistics_oracles_->DoTimeSample(myvelnp_,myforce_,Teuchos::null,Teuchos::null);

        // build statistics manager for inflow channel flow
        if (inflow_)
        {
          if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2")
          {
            statistics_channel_->DoTimeSample(myvelnp_,*myforce_);
          }
        }
        break;
      }
      case square_cylinder:
      {
        if(statistics_sqc_==null)
          dserror("need statistics_sqc_ to do a time sample for a flow around a square cylinder");

        statistics_sqc_->DoTimeSample(myvelnp_);

        // computation of Lift&Drag statistics
        {
          RCP<map<int,vector<double> > > liftdragvals;

          FLD::UTILS::LiftDrag(*discret_,*myforce_,*params_,liftdragvals);

          if((*liftdragvals).size()!=1)
          {
            dserror("expecting only one liftdrag label for the sampling of a flow around a square cylinder");
          }
          map<int,vector<double> >::iterator theonlyldval = (*liftdragvals).begin();

          statistics_sqc_->DoLiftDragTimeSample(((*theonlyldval).second)[0],
                                                ((*theonlyldval).second)[1]);
        }
        break;
      }
      case rotating_circular_cylinder_nurbs:
      {

        if(statistics_ccy_==null)
          dserror("need statistics_ccy_ to do a time sample for a flow in a rotating circular cylinder");

        statistics_ccy_->DoTimeSample(myvelnp_,Teuchos::null,Teuchos::null);
        break;
      }
      case rotating_circular_cylinder_nurbs_scatra:
      {

        if(statistics_ccy_==null)
          dserror("need statistics_ccy_ to do a time sample for a flow in a rotating circular cylinder");

        statistics_ccy_->DoTimeSample(myvelnp_,myscanp_,myfullphinp_);
        break;
      }
      default:
      {
        break;
      }
      }

      if(discret_->Comm().MyPID()==0)
      {
        cout << "Computed statistics: mean values, fluctuations, boundary forces etc.             (";
        printf("%10.4E",Teuchos::Time::wallTime()-tcpu);
        cout << ")";
      }

      //--------------------------------------------------
      // do averaging of residuals, dissipation rates etc
      // (all gausspoint-quantities)
      if(subgrid_dissipation_)
      {
        tcpu=Teuchos::Time::wallTime();

        switch(flow_)
        {
        case channel_flow_of_height_2:
        {
          if(statistics_channel_==null)
          {
            dserror("need statistics_channel_ to do a time sample for a turbulent channel flow");
          }

          // set vector values needed by elements
          map<string,RCP<Epetra_Vector> > statevecs;
          map<string,RCP<Epetra_MultiVector> > statetenss;

          if (DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(*params_, "time int algo") == INPAR::FLUID::timeint_gen_alpha)
          {
            statevecs.insert(pair<string,RCP<Epetra_Vector> >("u and p (n+1      ,trial)",myvelnp_));
            statevecs.insert(pair<string,RCP<Epetra_Vector> >("u and p (n+alpha_F,trial)",myvelaf_));
            statevecs.insert(pair<string,RCP<Epetra_Vector> >("acc     (n+alpha_M,trial)",myaccam_));

            if (alefluid_)
            {
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("dispnp"    , mydispnp_   ));
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("gridvelaf" , mygridvelaf_));
            }

            statistics_channel_->EvaluateResiduals(statevecs,time);
          }
          else
          {
            if (DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(*params_, "time int algo") == INPAR::FLUID::timeint_afgenalpha)
            {
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("vel",myvelaf_));
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("acc",myaccam_));
            }
            else if (DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(*params_, "time int algo") == INPAR::FLUID::timeint_one_step_theta)
            {
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("vel",myvelnp_));
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("acc",myaccnp_));
            }
            else
              dserror("Time integartion scheme not supported!");

            if (params_->sublist("SUBGRID VISCOSITY").get<string>("FSSUGRVISC")!= "No")
            {
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("fsvel",myfsvelaf_));
              if (myfsvelaf_==null)
                dserror ("Didn't got fsvel!");
            }
            if (scalesimilarity_)
            {
              statetenss.insert(pair<string,RCP<Epetra_MultiVector> >("filtered vel",myfilteredvel_));
              statetenss.insert(pair<string,RCP<Epetra_MultiVector> >("filtered reystr",myfilteredreystr_));
            }

            statistics_channel_->EvaluateResidualsFluidImplInt(statevecs,statetenss,time);
          }

          break;
        }
        default:
        {
          break;
        }
        }

        if(discret_->Comm().MyPID()==0)
        {
          cout << "                      residuals, dissipation rates etc, ";
          cout << "all gausspoint-quantities (";
          printf("%10.4E",Teuchos::Time::wallTime()-tcpu);
          cout << ")\n";
        }
      }
      if(discret_->Comm().MyPID()==0)
      {
        cout << "\n";
      }

      if(multifractal_)
      {
        switch(flow_)
        {
          case channel_flow_of_height_2:
          {
            // add parameters of multifractal subgrid-scales model
            statistics_channel_->AddModelParamsMultifractal(myvelaf_,myfsvelaf_,false);
            break;
          }
          case scatra_channel_flow_of_height_2:
          {
            // add parameters of multifractal subgrid-scales model
            statistics_channel_->AddModelParamsMultifractal(myvelaf_,myfsvelaf_,true);
            break;
          }
          default:
          {
            break;
          }
        }
      }

      // add vector(s) to general mean value computation
      // scatra vectors may be Teuchos::null
      if (statistics_general_mean_!=Teuchos::null)
        statistics_general_mean_->AddToCurrentTimeAverage(dt_,myvelnp_,myscanp_,myfullphinp_);

    } // end step in sampling period

    return;
  }


  /*----------------------------------------------------------------------

    Include current quantities in the time averaging procedure

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::DoTimeSample(int                             step,
                                                double                          time,
                                                Teuchos::RCP<Epetra_Vector>     velnp,
                                                Teuchos::RCP<Epetra_Vector>     force,
                                                Teuchos::RCP<Epetra_Vector>     phi,
                                                Teuchos::RCP<const DRT::DofSet> stddofset,
                                                Teuchos::RCP<const Epetra_Vector> discretmatchingvelnp /*= Teuchos::null */ // needed for 'bubbly_channel_flow'
)
  {
    // sampling takes place only in the sampling period
    if(step>=samstart_ && step<=samstop_ && flow_ != no_special_flow)
    {
      double tcpu=Teuchos::Time::wallTime();

      //--------------------------------------------------
      // calculate means, fluctuations etc of velocity,
      // pressure, boundary forces etc.
      switch(flow_)
      {
      case bubbly_channel_flow:
      {
        if(statistics_channel_multiphase_ == null)
          dserror("need statistics_channel_multiphase_ to do a time sample for a turbulent channel flow");

        if (velnp == Teuchos::null        or force == Teuchos::null
            or stddofset == Teuchos::null or discretmatchingvelnp == Teuchos::null
            or phi == Teuchos::null)
            dserror("The multi phase channel statistics need a current velnp, force, stddofset, discretmatchingvelnp, phinp.");

        statistics_channel_multiphase_->DoTimeSample(velnp, force, stddofset, discretmatchingvelnp, phi);
        break;
      }
      case combust_oracles:
      {
        subgrid_dissipation_ = false;

        if(statistics_oracles_==null)
          dserror("need statistics_oracles_ to do a time sample for an ORACLES flow step");

        statistics_oracles_->DoTimeSample(velnp,force,phi,stddofset);

        // build statistics manager for inflow channel flow
        if (inflow_)
        {
          if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2")
          {
            statistics_channel_->DoTimeSample(velnp,*force);
          }
        }
        break;
      }
      default:
      {
        dserror("called wrong DoTimeSample() for this kind of special flow");
        break;
      }
      }

      // add vector(s) to general mean value computation
      // scatra vectors may be Teuchos::null
      if (statistics_general_mean_!=Teuchos::null)
        statistics_general_mean_->AddToCurrentTimeAverage(dt_,velnp,myscanp_,myfullphinp_);

      if(discret_->Comm().MyPID()==0)
      {
        cout << "                      taking time sample (";
        printf("%10.4E",Teuchos::Time::wallTime()-tcpu);
        cout << ")\n";
      }

    } // end step in sampling period

    return;
  }

  /*----------------------------------------------------------------------

    Write (dump) the statistics to a file

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::DoOutput(IO::DiscretizationWriter& output,
                                            int                       step,
                                            const double              inflow)
  {
    // sampling takes place only in the sampling period
    if(step>=samstart_ && step<=samstop_ && flow_ != no_special_flow)
    {
      enum format {write_single_record   ,
                   write_multiple_records,
                   do_not_write          } outputformat=do_not_write;
      bool output_inflow = false;

      // sampling a la Volker --- single record is constantly updated
      if(dumperiod_!=0)
      {
        int samstep = step-samstart_+1;

        // dump every dumperiod steps
        if (samstep%dumperiod_==0) outputformat=write_single_record;
      }

      // sampling a la Peter --- for each sampling period a
      // new record is written; they can be combined by a
      // postprocessing script to a single long term sample
      // (allows restarts during sampling)
      if(dumperiod_==0)
      {
        int upres    =params_->get<int>("write solution every");
        int uprestart=params_->get<int>("write restart every" );

        // dump in combination with a restart/output
        if((step%upres == 0 || ( uprestart > 0 && step%uprestart == 0) ) && step>samstart_)
          outputformat=write_multiple_records;
      }
      if(inflow_)
      {
        int upres    =params_->get<int>("write solution every");
        int uprestart=params_->get<int>("write restart every" );

        // dump in combination with a restart/output
        if((step%upres == 0 || ( uprestart > 0 && step%uprestart == 0) ) && step>samstart_)
          output_inflow=true;
      }

      if (discret_->Comm().MyPID()==0 && outputformat != do_not_write )
        std::cout << "---  statistics record: \n" << std::flush;

      // do actual output (time averaging)
      switch(flow_)
      {
      case channel_flow_of_height_2:
      {
        if(statistics_channel_==null)
          dserror("need statistics_channel_ to do a time sample for a turbulent channel flow");


        if(outputformat == write_multiple_records)
        {
          statistics_channel_->TimeAverageMeansAndOutputOfStatistics(step);
          statistics_channel_->ClearStatistics();
        }

        if(outputformat == write_single_record)
          statistics_channel_->DumpStatistics(step);
        break;
      }
      case loma_channel_flow_of_height_2:
      {
        if(statistics_channel_==null)
          dserror("need statistics_channel_ to do a time sample for a turbulent channel flow at low Mach number");

        if(outputformat == write_single_record)
          statistics_channel_->DumpLomaStatistics(step);
        break;
      }
      case scatra_channel_flow_of_height_2:
      {
        if(statistics_channel_==null)
          dserror("need statistics_channel_ to do a time sample for a turbulent channel flow at low Mach number");

        if(outputformat == write_single_record)
          statistics_channel_->DumpScatraStatistics(step);
        break;
      }
      case bubbly_channel_flow:
      {
        if(statistics_channel_multiphase_==null)
          dserror("need statistics_channel_multiphase_ to do a time sample for a turbulent channel flow");

        if(outputformat == write_multiple_records)
        {
          statistics_channel_multiphase_->TimeAverageMeansAndOutputOfStatistics(step);
          statistics_channel_multiphase_->ClearStatistics();
        }

        if(outputformat == write_single_record)
          statistics_channel_multiphase_->DumpStatistics(step);
        break;
      }
      case lid_driven_cavity:
      {
        if(statistics_ldc_==null)
          dserror("need statistics_ldc_ to do a time sample for a lid driven cavity");

        if(outputformat == write_single_record)
          statistics_ldc_->DumpStatistics(step);
        break;
      }
      case loma_lid_driven_cavity:
      {
        if(statistics_ldc_==null)
          dserror("need statistics_ldc_ to do a time sample for a lid driven cavity at low Mach number");

        if(outputformat == write_single_record)
          statistics_ldc_->DumpLomaStatistics(step);
        break;
      }
      case backward_facing_step:
      {
        if(statistics_bfs_==null)
          dserror("need statistics_bfs_ to do a time sample for a flow over a backward-facing step");

        if(outputformat == write_single_record)
          statistics_bfs_->DumpStatistics(step);

        //write statistics of inflow channel flow
        if (inflow_)
        {
          if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2")
          {
            if(output_inflow)
            {
              statistics_channel_->TimeAverageMeansAndOutputOfStatistics(step);
              statistics_channel_->ClearStatistics();
            }
          }
        }
        break;
      }
      case loma_backward_facing_step:
      {
        if(statistics_bfs_==null)
          dserror("need statistics_bfs_ to do a time sample for a flow over a backward-facing step at low Mach number");

        if (DRT::INPUT::get<INPAR::FLUID::PhysicalType>(*params_, "Physical Type") == INPAR::FLUID::incompressible)
        {
          if(outputformat == write_single_record)
            statistics_bfs_->DumpStatistics(step);

          // write statistics of inflow channel flow
          if (inflow_)
          {
            if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2")
            {
              if(output_inflow)
              {
                statistics_channel_->TimeAverageMeansAndOutputOfStatistics(step);
                statistics_channel_->ClearStatistics();
              }
            }
          }
        }
        else
        {
          if(outputformat == write_single_record)
            statistics_bfs_->DumpLomaStatistics(step);

          // write statistics of inflow channel flow
          if (inflow_)
          {
            if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="loma_channel_flow_of_height_2")
            {
              if(outputformat == write_single_record)
                statistics_channel_->DumpLomaStatistics(step);
            }
          }
        }
        break;
      }
      case combust_oracles:
      {
        if(statistics_oracles_==null)
          dserror("need statistics_oracles_ to do a time sample for an ORACLES flow step");
        if(outputformat == write_multiple_records)
        {
          statistics_oracles_->TimeAverageStatistics();
          statistics_oracles_->OutputStatistics(step);
          statistics_oracles_->ClearStatistics();
        }

        // build statistics manager for inflow channel flow
        if (inflow_)
        {
          if(params_->sublist("TURBULENT INFLOW").get<string>("CANONICAL_INFLOW")=="channel_flow_of_height_2")
          {
            if(outputformat == write_multiple_records)
            {
              statistics_channel_->TimeAverageMeansAndOutputOfStatistics(step);
              statistics_channel_->ClearStatistics();
            }
          }
        }
        break;
      }
      case square_cylinder:
      {
        if(statistics_sqc_==null)
          dserror("need statistics_sqc_ to do a time sample for a square cylinder flow");

        if (outputformat == write_single_record)
          statistics_sqc_->DumpStatistics(step);
        break;
      }
      case rotating_circular_cylinder_nurbs:
      case rotating_circular_cylinder_nurbs_scatra:
      {
        if(statistics_ccy_==null)
          dserror("need statistics_ccy_ to do a time sample for a flow in a rotating circular cylinder");

        statistics_ccy_->TimeAverageMeansAndOutputOfStatistics(step);
        statistics_ccy_->ClearStatistics();
        break;
      }
      default:
      {
        break;
      }
      }

//      if (discret_->Comm().MyPID()==0 && outputformat != do_not_write )
//        std::cout << "done" << std::endl;

      if(discret_->Comm().MyPID()==0 && outputformat != do_not_write)
      {
        cout << "XXXXXXXXXXXXXXXXXXXXX              ";
        cout << "wrote statistics record            ";
        cout << "XXXXXXXXXXXXXXXXXXXXX";
        cout << "\n\n";
      }


      // dump general mean value output in combination with a restart/output
      // don't write output if turbulent inflow or twophaseflow is computed
      if (!inflow and flow_ != bubbly_channel_flow)
      {
        int upres    =params_->get<int>("write solution every");
        int uprestart=params_->get<int>("write restart every" );

        if(step%upres == 0 || (uprestart > 0 && step%uprestart == 0) )
        {
          if (discret_->Comm().MyPID()==0)
            std::cout << "---  averaged vector: \n" << std::flush;

          statistics_general_mean_->WriteOldAverageVec(output);

//          if (discret_->Comm().MyPID()==0)
//            std::cout << "done" << std::endl;
        }
      }
    } // end step is in sampling period

    return;
  } // DoOutput


  /*----------------------------------------------------------------------

  Add results from scalar transport fields to statistics

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::AddScaTraResults(
      RCP<DRT::Discretization> scatradis,
      RCP<Epetra_Vector> phinp
  )
  {
    if(discret_->Comm().MyPID()==0)
    {
      cout<<endl<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
      cout<<"TurbulenceStatisticManager: added access to ScaTra results"<<endl;
      cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
    }

    // store the relevant pointers to provide access
    scatradis_   = scatradis;
    myfullphinp_ = phinp;

    if (statistics_general_mean_!=Teuchos::null)
      statistics_general_mean_->AddScaTraResults(scatradis, phinp);

    if (statistics_ccy_!=Teuchos::null)
      statistics_ccy_->AddScaTraResults(scatradis, phinp);

    if(flow_==scatra_channel_flow_of_height_2)
    {
      // store scatra discretization (multifractal subgrid scales only)
      if (multifractal_)
        statistics_channel_->StoreScatraDiscret(scatradis_);
    }

    return;
  }


  /*----------------------------------------------------------------------

  Write (dump) the scatra-specific mean fields to the result file

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::DoOutputForScaTra(
      IO::DiscretizationWriter& output,
      int                       step)
  {
    // sampling takes place only in the sampling period
    if(step>=samstart_ && step<=samstop_ && flow_ != no_special_flow)
    {
      // statistics for scatra fields was already written during DoOutput()
      // Thus, we have to care for the mean field only:

      // dump general mean value output for scatra results
      // in combination with a restart/output
      int upres    =params_->get("write solution every", -1);
      int uprestart=params_->get("write restart every" , -1);

      if(step%upres == 0 || step%uprestart == 0)
      {
        statistics_general_mean_->DoOutputForScaTra(output,step);
      }
    }
    return;
  }


  /*----------------------------------------------------------------------

  Restart statistics collection

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::Restart(
    IO::DiscretizationReader& reader,
    int                       step
    )
  {

    if(statistics_general_mean_!=Teuchos::null)
    {
      if(samstart_<step && step<=samstop_)
      {
        if(discret_->Comm().MyPID()==0)
        {
          cout << "XXXXXXXXXXXXXXXXXXXXX              ";
          cout << "Read general mean values           ";
          cout << "XXXXXXXXXXXXXXXXXXXXX";
          cout << "\n\n";
        }

        statistics_general_mean_->ReadOldStatistics(reader);
      }
    }

    return;
  } // Restart


  /*----------------------------------------------------------------------

  Restart for scatra mean fields (statistics was restarted via Restart() )

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::RestartScaTra(
    IO::DiscretizationReader& scatrareader,
    int                       step
  )
  {
    // we have only to read in the mean field.
    // The rest of the restart was already done during the Restart() call
    if(statistics_general_mean_!=Teuchos::null)
    {
      if(samstart_<step && step<=samstop_)
      {
        if(discret_->Comm().MyPID()==0)
        {
          cout << "XXXXXXXXXXXXXXXXXXXXX        ";
          cout << "Read general mean values for ScaTra      ";
          cout << "XXXXXXXXXXXXXXXXXXXXX";
          cout << "\n\n";
        }

        statistics_general_mean_->ReadOldStatisticsScaTra(scatrareader);
      }
    }

    return;
  } // RestartScaTra


} // end namespace FLD

#endif  // #ifdef CCADISCRET
