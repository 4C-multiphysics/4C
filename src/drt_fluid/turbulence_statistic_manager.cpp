/*!

Manage the computation of averages for several
canonical flows like channel flow, flow around a square
cylinder, flow in a lid driven cavity, flow over a backward-facing step etc.

The manager is intended to remove as much of the averaging
overhead as possible from the time integration method.

Maintainer: Peter Gamnitzer
            gamnitzer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15235

*/
#ifdef CCADISCRET

#include "turbulence_statistic_manager.H"

namespace FLD
{

  /*----------------------------------------------------------------------

    Standard Constructor for Genalpha time integration (public)

  ----------------------------------------------------------------------*/
  TurbulenceStatisticManager::TurbulenceStatisticManager(FluidGenAlphaIntegration& fluid)
    :
    dt_              (fluid.dt_       ),
    alphaM_          (fluid.alphaM_   ),
    alphaF_          (fluid.alphaF_   ),
    gamma_           (fluid.gamma_    ),
    density_         (fluid.density_  ),
    discret_         (fluid.discret_  ),
    params_          (fluid.params_   ),
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
    // get density

    // activate the computation of subgrid dissipation,
    // residuals etc
    subgrid_dissipation_=true;

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
                                                          params_             ,
                                                          smagorinsky_        ,
                                                          scalesimilarity_    ,
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
                                                          params_             ,
                                                          smagorinsky_        ,
                                                          scalesimilarity_    ,
                                                          subgrid_dissipation_));
    }
    else if(fluid.special_flow_=="lid_driven_cavity")
    {
      flow_=lid_driven_cavity;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ldc_    =rcp(new TurbulenceStatisticsLdc(discret_,params_));
    }
    else if(fluid.special_flow_=="loma_lid_driven_cavity")
    {
      flow_=loma_lid_driven_cavity;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ldc_    =rcp(new TurbulenceStatisticsLdc(discret_,params_));
    }
    else if(fluid.special_flow_=="backward_facing_step")
    {
      flow_=backward_facing_step;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = rcp(new TurbulenceStatisticsBfs(discret_,params_,"geometry_DNS_incomp_flow"));
    }
    else if(fluid.special_flow_=="loma_backward_facing_step")
    {
      flow_=loma_backward_facing_step;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = rcp(new TurbulenceStatisticsBfs(discret_,params_,"geometry_LES_flow_with_heating"));
    }
    else if(fluid.special_flow_=="square_cylinder")
    {
      flow_=square_cylinder;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_sqc_    =rcp(new TurbulenceStatisticsSqc(discret_,params_));
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
                                                      params_             ,
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
                                                      params_             ,
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
      ParameterList *  modelparams =&(params_.sublist("TURBULENCE MODEL"));

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

    Standard Constructor for One-Step-Theta time integration (public)

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

    subgrid_dissipation_=true;

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
                                                          params_             ,
                                                          smagorinsky_        ,
                                                          scalesimilarity_    ,
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
                                                          params_             ,
                                                          smagorinsky_        ,
                                                          scalesimilarity_    ,
                                                          subgrid_dissipation_));
    }
    else if(fluid.special_flow_=="lid_driven_cavity")
    {
      flow_=lid_driven_cavity;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ldc_    =rcp(new TurbulenceStatisticsLdc(discret_,params_));
    }
    else if(fluid.special_flow_=="loma_lid_driven_cavity")
    {
      flow_=loma_lid_driven_cavity;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ldc_    =rcp(new TurbulenceStatisticsLdc(discret_,params_));
    }
    else if(fluid.special_flow_=="backward_facing_step")
    {
      flow_=backward_facing_step;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = rcp(new TurbulenceStatisticsBfs(discret_,params_,"geometry_DNS_incomp_flow"));
    }
    else if(fluid.special_flow_=="loma_backward_facing_step")
    {
      flow_=loma_backward_facing_step;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = rcp(new TurbulenceStatisticsBfs(discret_,params_,"geometry_LES_flow_with_heating"));
    }
    else if(fluid.special_flow_=="square_cylinder")
    {
      flow_=square_cylinder;

      // do the time integration independent setup
      Setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_sqc_    =rcp(new TurbulenceStatisticsSqc(discret_,params_));
    }
    else if(fluid.special_flow_=="square_cylinder_nurbs")
    {
      flow_=square_cylinder_nurbs;

      // do the time integration independent setup
      Setup();
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

    // allocate one instance of the flow indepedent averaging procedure
    // providiing colerful output for paraview
    {
      ParameterList *  modelparams =&(params_.sublist("TURBULENCE MODEL"));

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

    ParameterList *  modelparams =&(params_.sublist("TURBULENCE MODEL"));

    // check if we want to compute averages of Smagorinsky
    // constants, effective viscosities etc
    smagorinsky_=false;
    if (modelparams->get<string>("TURBULENCE_APPROACH","DNS_OR_RESVMM_LES")
        ==
        "CLASSICAL_LES")
    {
      if(modelparams->get<string>("PHYSICAL_MODEL","no_model")
         ==
         "Dynamic_Smagorinsky"
         ||
         params_.sublist("TURBULENCE MODEL").get<string>("PHYSICAL_MODEL","no_model")
         ==
         "Smagorinsky_with_van_Driest_damping"
         ||
         params_.sublist("TURBULENCE MODEL").get<string>("PHYSICAL_MODEL","no_model")
         ==
         "Smagorinsky"
        )
      {
        if(discret_->Comm().MyPID()==0)
        {
          cout << "                             Initialising output for Smagorinsky type models\n\n\n";
          fflush(stdout);
        }

        smagorinsky_=true;
      }
    }

    // check if we want to compute averages of scale similarity
    // quantities (tau_SFS)
        scalesimilarity_=false;
        if (modelparams->get<string>("TURBULENCE_APPROACH","DNS_OR_RESVMM_LES")
            ==
            "CLASSICAL_LES")
        {
          if(modelparams->get<string>("PHYSICAL_MODEL","no_model")
             ==
             "Scale_Similarity"
             ||
             params_.sublist("TURBULENCE MODEL").get<string>("PHYSICAL_MODEL","no_model")
             ==
             "Mixed_Scale_Similarity_Eddy_Viscosity_Model"
            )
          {
            if(discret_->Comm().MyPID()==0)
            {
              cout << "                             Initialising output for scale similarity type models\n\n\n";
              fflush(stdout);
            }

            scalesimilarity_=true;
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
          flow_ == loma_channel_flow_of_height_2)
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
        // add computed dynamic Smagorinsky quantities
        // (effective viscosity etc. used during the computation)
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
                                                const double eosfac)
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
        break;
      }
      case loma_backward_facing_step:
      {
        if(statistics_bfs_==null)
          dserror("need statistics_bfs_ to do a time sample for a flow over a backward-facing step at low Mach number");

        if (DRT::INPUT::get<INPAR::FLUID::PhysicalType>(params_, "Physical Type") == INPAR::FLUID::incompressible)
            statistics_bfs_->DoTimeSample(myvelnp_);
        else
            statistics_bfs_->DoLomaTimeSample(myvelnp_,myscanp_,eosfac);
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

          FLD::UTILS::LiftDrag(*discret_,*myforce_,params_,liftdragvals);

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

        statistics_ccy_->DoTimeSample(myvelnp_,Teuchos::null);
        break;
      }
      case rotating_circular_cylinder_nurbs_scatra:
      {

        if(statistics_ccy_==null)
          dserror("need statistics_ccy_ to do a time sample for a flow in a rotating circular cylinder");

        statistics_ccy_->DoTimeSample(myvelnp_,myscanp_);
        break;
      }
      default:
      {
        break;
      }
      }

      if(discret_->Comm().MyPID()==0)
      {
        cout << "Computing statistics: mean values, fluctuations, ";
        cout << "boundary forces etc.             (";
        printf("%10.4E",Teuchos::Time::wallTime()-tcpu);
        cout << ")\n";
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

          if (DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(params_, "time int algo") == INPAR::FLUID::timeint_gen_alpha)
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
            if (DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(params_, "time int algo") == INPAR::FLUID::timeint_afgenalpha)
            {
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("vel",myvelaf_));
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("acc",myaccam_));
            }
            else if (DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(params_, "time int algo") == INPAR::FLUID::timeint_one_step_theta)
            {
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("vel",myvelnp_));
              statevecs.insert(pair<string,RCP<Epetra_Vector> >("acc",myaccnp_));
            }
            else
              dserror("Time integartion scheme not supported!");

            if (params_.get<string>("fs subgrid viscosity")!= "No")
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

      // add vector to general mean value computation
      if(flow_==rotating_circular_cylinder_nurbs_scatra)
        statistics_general_mean_->AddToCurrentTimeAverage(dt_,myvelnp_,myscanp_);
      else
        statistics_general_mean_->AddToCurrentTimeAverage(dt_,myvelnp_);

    } // end step in sampling period

    return;
  }

  /*----------------------------------------------------------------------

    Write (dump) the statistics to a file

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::DoOutput(IO::DiscretizationWriter& output,
                                            int                       step,
                                            const double              eosfac)
  {
    // sampling takes place only in the sampling period
    if(step>=samstart_ && step<=samstop_ && flow_ != no_special_flow)
    {
      enum format {write_single_record   ,
                   write_multiple_records,
                   do_not_write          } outputformat=do_not_write;

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
        int upres    =params_.get("write solution every", -1);
        int uprestart=params_.get("write restart every" , -1);

        // dump in combination with a restart/output
        if((step%upres == 0 || step%uprestart == 0) && step>samstart_)
          outputformat=write_multiple_records;
      }

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
        break;
      }
      case loma_backward_facing_step:
      {
        if(statistics_bfs_==null)
          dserror("need statistics_bfs_ to do a time sample for a flow over a backward-facing step at low Mach number");

        if (DRT::INPUT::get<INPAR::FLUID::PhysicalType>(params_, "Physical Type") == INPAR::FLUID::incompressible)
        {
          if(outputformat == write_single_record)
            statistics_bfs_->DumpStatistics(step);
        }
        else
        {
          if(outputformat == write_single_record)
            statistics_bfs_->DumpLomaStatistics(step,eosfac);
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


      if(discret_->Comm().MyPID()==0 && outputformat != do_not_write)
      {
        cout << "XXXXXXXXXXXXXXXXXXXXX              ";
        cout << "wrote statistics record            ";
        cout << "XXXXXXXXXXXXXXXXXXXXX";
        cout << "\n\n";
      }


      // dump general mean value output in combination with a restart/output
      {
        int upres    =params_.get("write solution every", -1);
        int uprestart=params_.get("write restart every" , -1);

        if(step%upres == 0 || step%uprestart == 0)
        {
          statistics_general_mean_->WriteOldAverageVec(output);
        }
      }
    } // end step is in sampling period

    return;
  } // DoOutput

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

  Clear all statistics collected up to now

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::Reset()
  {
    switch(flow_)
    {
    case channel_flow_of_height_2:
    case loma_channel_flow_of_height_2:
    {
      if (statistics_channel_==null)
        dserror("need statistics_channel_ to do a time sample for a turbulent channel flow");
      statistics_channel_->ClearStatistics();
      break;
    }
    case lid_driven_cavity:
    case loma_lid_driven_cavity:
    case backward_facing_step:
    case loma_backward_facing_step:
    case square_cylinder:
    {
      // nothing to do here
      break;
    }
    default:
    {
      // nothing to do here
      break;
    }
    }

    return;
  }

} // end namespace FLD

#endif  // #ifdef CCADISCRET
