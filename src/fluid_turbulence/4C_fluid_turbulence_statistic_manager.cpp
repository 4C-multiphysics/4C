/*----------------------------------------------------------------------*/
/*! \file

\brief Manage the computation of averages for several
canonical flows like channel flow, flow around a square
cylinder, flow in a lid driven cavity, flow over a backward-facing step etc.

The manager is intended to remove as much of the averaging
overhead as possible from the time integration method.


\level 2

*/
/*----------------------------------------------------------------------*/

#include "4C_fluid_turbulence_statistic_manager.hpp"

#include "4C_fluid_implicit_integration.hpp"
#include "4C_fluid_timint_hdg.hpp"
#include "4C_fluid_turbulence_statistics_bfda.hpp"
#include "4C_fluid_turbulence_statistics_bfs.hpp"
#include "4C_fluid_turbulence_statistics_ccy.hpp"
#include "4C_fluid_turbulence_statistics_cha.hpp"
#include "4C_fluid_turbulence_statistics_hit.hpp"
#include "4C_fluid_turbulence_statistics_ldc.hpp"
#include "4C_fluid_turbulence_statistics_mean_general.hpp"
#include "4C_fluid_turbulence_statistics_ph.hpp"
#include "4C_fluid_turbulence_statistics_sqc.hpp"
#include "4C_fluid_turbulence_statistics_tgv.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_fluid_xwall.hpp"
#include "4C_global_data.hpp"
#include "4C_io_pstream.hpp"
#include "4C_scatra_timint_implicit.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace FLD
{
  /*----------------------------------------------------------------------

    Standard Constructor for standard time integration (public)

  ----------------------------------------------------------------------*/
  TurbulenceStatisticManager::TurbulenceStatisticManager(FluidImplicitTimeInt& fluid)
      : dt_(fluid.dta_),
        discret_(fluid.discret_),
        params_(fluid.params_),
        alefluid_(fluid.alefluid_),
        myaccnp_(fluid.accnp_),
        myaccn_(fluid.accn_),
        myaccam_(fluid.accam_),
        myvelnp_(fluid.velnp_),
        myveln_(fluid.veln_),
        myvelaf_(fluid.velaf_),
        myhist_(fluid.hist_),
        myscaaf_(fluid.scaaf_),
        myscaam_(fluid.scaam_),
        mydispnp_(fluid.dispnp_),
        mydispn_(fluid.dispn_),
        mygridvelaf_(fluid.gridv_),
        myforce_(fluid.trueresidual_),
        myfsvelaf_(fluid.fsvelaf_),
        myfsscaaf_(fluid.fsscaaf_),
        myxwall_(fluid.xwall_),
        mystressmanager_(fluid.stressmanager_),
        flow_(no_special_flow),
        withscatra_(false),
        turbmodel_(Inpar::FLUID::no_model),
        subgrid_dissipation_(false),
        inflow_(false),
        statistics_outfilename_(fluid.statistics_outfilename_),
        statistics_general_mean_(Teuchos::null),
        statistics_channel_(Teuchos::null),
        statistics_ccy_(Teuchos::null),
        statistics_ldc_(Teuchos::null),
        statistics_bfs_(Teuchos::null),
        statistics_ph_(Teuchos::null),
        statistics_bfda_(Teuchos::null),
        statistics_sqc_(Teuchos::null),
        statistics_hit_(Teuchos::null),
        statistics_tgv_(Teuchos::null)
  {
    subgrid_dissipation_ = params_->sublist("TURBULENCE MODEL").get<bool>("SUBGRID_DISSIPATION");
    // initialize
    withscatra_ = false;

    // toogle statistics output for turbulent inflow
    inflow_ = params_->sublist("TURBULENT INFLOW").get<bool>("TURBULENTINFLOW") == true;

    // toogle output of mean velocity for paraview
    out_mean_ = params_->sublist("TURBULENCE MODEL").get<bool>("OUTMEAN");

    // the flow parameter will control for which geometry the
    // sampling is done
    if (fluid.special_flow_ == "channel_flow_of_height_2")
    {
      flow_ = channel_flow_of_height_2;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_channel_ = Teuchos::make_rcp<TurbulenceStatisticsCha>(discret_, alefluid_,
          mydispnp_, *params_, statistics_outfilename_, subgrid_dissipation_, myxwall_);
    }
    else if (fluid.special_flow_ == "loma_channel_flow_of_height_2")
    {
      flow_ = loma_channel_flow_of_height_2;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_channel_ = Teuchos::make_rcp<TurbulenceStatisticsCha>(discret_, alefluid_,
          mydispnp_, *params_, statistics_outfilename_, subgrid_dissipation_, Teuchos::null);
    }
    else if (fluid.special_flow_ == "scatra_channel_flow_of_height_2")
    {
      flow_ = scatra_channel_flow_of_height_2;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_channel_ = Teuchos::make_rcp<TurbulenceStatisticsCha>(discret_, alefluid_,
          mydispnp_, *params_, statistics_outfilename_, subgrid_dissipation_, Teuchos::null);
    }
    else if (fluid.special_flow_ == "decaying_homogeneous_isotropic_turbulence" or
             fluid.special_flow_ == "forced_homogeneous_isotropic_turbulence" or
             fluid.special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence")
    {
      if (fluid.special_flow_ == "decaying_homogeneous_isotropic_turbulence")
        flow_ = decaying_homogeneous_isotropic_turbulence;
      else if (fluid.special_flow_ == "forced_homogeneous_isotropic_turbulence")
        flow_ = forced_homogeneous_isotropic_turbulence;
      else
        flow_ = scatra_forced_homogeneous_isotropic_turbulence;

      // do the time integration independent setup
      setup();
      if (Global::Problem::instance()->spatial_approximation_type() ==
          Core::FE::ShapeFunctionType::hdg)
      {
        TimIntHDG* hdgfluid = dynamic_cast<TimIntHDG*>(&fluid);
        if (hdgfluid == nullptr) FOUR_C_THROW("this should be a hdg time integer");

        // we want to use the interior velocity here
        myvelnp_ = hdgfluid->return_int_velnp();

        // allocate one instance of the averaging procedure for
        // the flow under consideration
        if (flow_ == forced_homogeneous_isotropic_turbulence or
            flow_ == scatra_forced_homogeneous_isotropic_turbulence)
          statistics_hit_ = Teuchos::make_rcp<TurbulenceStatisticsHitHDG>(
              discret_, *params_, statistics_outfilename_, true);
        else
        {
          statistics_hit_ = Teuchos::null;
          FOUR_C_THROW("decaying hit currently not implemented for HDG");
        }
      }
      else
      {
        // allocate one instance of the averaging procedure for
        // the flow under consideration
        if (flow_ == forced_homogeneous_isotropic_turbulence or
            flow_ == scatra_forced_homogeneous_isotropic_turbulence)
          statistics_hit_ = Teuchos::make_rcp<TurbulenceStatisticsHit>(
              discret_, *params_, statistics_outfilename_, true);
        else
          statistics_hit_ = Teuchos::make_rcp<TurbulenceStatisticsHit>(
              discret_, *params_, statistics_outfilename_, false);
      }
    }
    else if (fluid.special_flow_ == "taylor_green_vortex")
    {
      flow_ = taylor_green_vortex;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_tgv_ =
          Teuchos::make_rcp<TurbulenceStatisticsTgv>(discret_, *params_, statistics_outfilename_);
    }
    else if (fluid.special_flow_ == "lid_driven_cavity")
    {
      flow_ = lid_driven_cavity;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ldc_ =
          Teuchos::make_rcp<TurbulenceStatisticsLdc>(discret_, *params_, statistics_outfilename_);
    }
    else if (fluid.special_flow_ == "loma_lid_driven_cavity")
    {
      flow_ = loma_lid_driven_cavity;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ldc_ =
          Teuchos::make_rcp<TurbulenceStatisticsLdc>(discret_, *params_, statistics_outfilename_);
    }
    else if (fluid.special_flow_ == "backward_facing_step")
    {
      flow_ = backward_facing_step;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = Teuchos::make_rcp<TurbulenceStatisticsBfs>(
          discret_, *params_, statistics_outfilename_, "geometry_DNS_incomp_flow");

      // statistics manager for turbulent boundary layer not available
      if (inflow_)
        FOUR_C_THROW(
            "The backward-facing step based on the geometry the DNS requires a turbulent boundary "
            "layer inflow profile which is not supported, yet!");
    }
    else if (fluid.special_flow_ == "periodic_hill")
    {
      flow_ = periodic_hill;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ph_ =
          Teuchos::make_rcp<TurbulenceStatisticsPh>(discret_, *params_, statistics_outfilename_);
    }
    else if (fluid.special_flow_ == "loma_backward_facing_step")
    {
      flow_ = loma_backward_facing_step;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = Teuchos::make_rcp<TurbulenceStatisticsBfs>(
          discret_, *params_, statistics_outfilename_, "geometry_LES_flow_with_heating");

      // build statistics manager for inflow channel flow
      if (inflow_)
      {
        if (params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                "channel_flow_of_height_2" or
            params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                "loma_channel_flow_of_height_2" or
            params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                "scatra_channel_flow_of_height_2")
        {
          // do not write any dissipation rates for inflow channels
          subgrid_dissipation_ = false;
          // allocate one instance of the averaging procedure for the flow under consideration
          statistics_channel_ = Teuchos::make_rcp<TurbulenceStatisticsCha>(discret_, alefluid_,
              mydispnp_, *params_, statistics_outfilename_, subgrid_dissipation_, myxwall_);
        }
      }
    }
    else if (fluid.special_flow_ == "backward_facing_step2")
    {
      flow_ = backward_facing_step2;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfs_ = Teuchos::make_rcp<TurbulenceStatisticsBfs>(
          discret_, *params_, statistics_outfilename_, "geometry_EXP_vogel_eaton");

      // build statistics manager for inflow channel flow
      if (inflow_)
      {
        if (params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                "channel_flow_of_height_2" or
            params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                "loma_channel_flow_of_height_2" or
            params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                "scatra_channel_flow_of_height_2")
        {
          // do not write any dissipation rates for inflow channels
          subgrid_dissipation_ = false;
          // allocate one instance of the averaging procedure for the flow under consideration
          statistics_channel_ = Teuchos::make_rcp<TurbulenceStatisticsCha>(discret_, alefluid_,
              mydispnp_, *params_, statistics_outfilename_, subgrid_dissipation_, myxwall_);
        }
      }
    }
    else if (fluid.special_flow_ == "square_cylinder")
    {
      flow_ = square_cylinder;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_sqc_ =
          Teuchos::make_rcp<TurbulenceStatisticsSqc>(discret_, *params_, statistics_outfilename_);
    }
    else if (fluid.special_flow_ == "square_cylinder_nurbs")
    {
      flow_ = square_cylinder_nurbs;

      // do the time integration independent setup
      setup();
    }
    else if (fluid.special_flow_ == "rotating_circular_cylinder_nurbs")
    {
      flow_ = rotating_circular_cylinder_nurbs;
      const bool withscatra = false;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ccy_ = Teuchos::make_rcp<TurbulenceStatisticsCcy>(
          discret_, alefluid_, mydispnp_, *params_, statistics_outfilename_, withscatra);
    }
    else if (fluid.special_flow_ == "rotating_circular_cylinder_nurbs_scatra")
    {
      flow_ = rotating_circular_cylinder_nurbs_scatra;
      const bool withscatra = true;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_ccy_ = Teuchos::make_rcp<TurbulenceStatisticsCcy>(
          discret_, alefluid_, mydispnp_, *params_, statistics_outfilename_, withscatra);
    }
    else if (fluid.special_flow_ == "time_averaging")
    {
      flow_ = time_averaging;

      // do the time integration independent setup
      setup();
    }
    else if (fluid.special_flow_ == "blood_fda_flow")
    {
      flow_ = blood_fda_flow;

      // do the time integration independent setup
      setup();

      // allocate one instance of the averaging procedure for
      // the flow under consideration
      statistics_bfda_ =
          Teuchos::make_rcp<TurbulenceStatisticsBfda>(discret_, *params_, statistics_outfilename_);
    }
    else
    {
      flow_ = no_special_flow;

      // do the time integration independent setup
      setup();
    }

    // allocate one instance of the flow independent averaging procedure
    // providing colorful output for paraview
    if (out_mean_)
    {
      Teuchos::ParameterList* modelparams = &(params_->sublist("TURBULENCE MODEL"));

      std::string homdir = modelparams->get<std::string>("HOMDIR", "not_specified");

      if (flow_ == rotating_circular_cylinder_nurbs_scatra)
      {
        // additional averaging of scalar field
        statistics_general_mean_ = Teuchos::make_rcp<TurbulenceStatisticsGeneralMean>(
            discret_, homdir, *fluid.vel_pres_splitter(), true);
      }
      else
      {
        statistics_general_mean_ = Teuchos::make_rcp<TurbulenceStatisticsGeneralMean>(
            discret_, homdir, *fluid.vel_pres_splitter(), false);
      }
    }
    else
      statistics_general_mean_ = Teuchos::null;

    return;
  }



  /*----------------------------------------------------------------------

    Time integration independent setup called by Constructor (private)

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::setup()
  {
    Teuchos::ParameterList* modelparams = &(params_->sublist("TURBULENCE MODEL"));

    if (modelparams->get<std::string>("TURBULENCE_APPROACH", "DNS_OR_RESVMM_LES") ==
        "CLASSICAL_LES")
    {
      // check if we want to compute averages of Smagorinsky
      // constants, effective viscosities etc
      if (modelparams->get<std::string>("PHYSICAL_MODEL", "no_model") == "Dynamic_Smagorinsky" ||
          modelparams->get<std::string>("PHYSICAL_MODEL", "no_model") ==
              "Smagorinsky_with_van_Driest_damping" ||
          modelparams->get<std::string>("PHYSICAL_MODEL", "no_model") == "Smagorinsky")
      {
        turbmodel_ = Inpar::FLUID::dynamic_smagorinsky;
      }
      // check if we want to compute averages of multifractal
      // quantities (N, B)
      else if (modelparams->get<std::string>("PHYSICAL_MODEL", "no_model") ==
               "Multifractal_Subgrid_Scales")
      {
        turbmodel_ = Inpar::FLUID::multifractal_subgrid_scales;
      }
      else if (modelparams->get<std::string>("PHYSICAL_MODEL", "no_model") == "Dynamic_Vreman")
      {
        turbmodel_ = Inpar::FLUID::dynamic_vreman;
        // some dummy values into the parameter list
        params_->set<double>("C_vreman", 0.0);
        params_->set<double>("C_vreman_theoretical", 0.0);
        params_->set<double>("Dt_vreman", 0.0);
      }
    }
    else
      turbmodel_ = Inpar::FLUID::no_model;

    // parameters for sampling/dumping period
    if (flow_ != no_special_flow)
    {
      samstart_ = modelparams->get<int>("SAMPLING_START", 1);
      samstop_ = modelparams->get<int>("SAMPLING_STOP", 1000000000);
      dumperiod_ = modelparams->get<int>("DUMPING_PERIOD", 1);
    }
    else
    {
      samstart_ = 0;
      samstop_ = 0;
      dumperiod_ = 0;
    }


    if (discret_->get_comm().MyPID() == 0)
    {
      if (flow_ == channel_flow_of_height_2 or flow_ == loma_channel_flow_of_height_2 or
          flow_ == scatra_channel_flow_of_height_2 or flow_ == bubbly_channel_flow)
      {
        std::string homdir = modelparams->get<std::string>("HOMDIR", "not_specified");

        if (homdir != "xy" && homdir != "xz" && homdir != "yz")
        {
          FOUR_C_THROW("need two homogeneous directions to do averaging in plane channel flows\n");
        }

        std::cout << "Additional output          : ";
        std::cout << "Turbulence statistics are evaluated ";
        std::cout << "for a turbulent channel flow.\n";
        std::cout << "                             ";
        std::cout << "The solution is averaged over the homogeneous ";
        std::cout << homdir;
        std::cout << " plane and over time.\n";
        std::cout << "\n";
        std::cout << "                             ";
        std::cout << "Sampling period: steps " << samstart_ << " to ";
        std::cout << modelparams->get<int>("SAMPLING_STOP", 1000000000) << ".\n";

        int dumperiod = modelparams->get<int>("DUMPING_PERIOD", 1);


        if (dumperiod == 0)
        {
          std::cout << "                             ";
          std::cout << "Using standalone records (i.e. start from 0 for a new record)\n";
        }
        else
        {
          std::cout << "                             ";
          std::cout << "Volker-style incremental dumping is used (";
          std::cout << dumperiod << ")" << std::endl;
        }

        std::cout << std::endl;
        std::cout << std::endl;
      }
    }

    return;
  }


  /*----------------------------------------------------------------------

    Store values computed during the element call

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::store_element_values(int step)
  {
    // sampling takes place only in the sampling period
    if (step >= samstart_ && step <= samstop_ && flow_ != no_special_flow)
    {
      switch (flow_)
      {
        case channel_flow_of_height_2:
        case loma_channel_flow_of_height_2:
        case scatra_channel_flow_of_height_2:
        {
          // add computed dynamic Smagorinsky quantities
          // (effective viscosity etc. used during the computation)
          if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky)
          {
            if (discret_->get_comm().MyPID() == 0)
            {
              std::cout << "\nSmagorinsky constant, effective viscosity, ... etc, ";
              std::cout << "all element-quantities \n";
            }
            statistics_channel_->add_dynamic_smagorinsky_quantities();
          }
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
  void TurbulenceStatisticManager::do_time_sample(int step, const double eosfac,
      const double thermpressaf, const double thermpressam, const double thermpressdtaf,
      const double thermpressdtam)
  {
    // store Smagorinsky statistics if used
    store_element_values(step);

    // sampling takes place only in the sampling period
    if ((step >= samstart_ && step <= samstop_ &&
            flow_ != no_special_flow)  // usual case with statistical-stationary state
        or (step != 0 && flow_ == decaying_homogeneous_isotropic_turbulence))  // time-dependent!
    {
      double tcpu = Teuchos::Time::wallTime();

      //--------------------------------------------------
      // calculate means, fluctuations etc of velocity,
      // pressure, boundary forces etc.
      switch (flow_)
      {
        case channel_flow_of_height_2:
        {
          if (statistics_channel_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_channel_ to do a time sample for a turbulent channel flow");

          statistics_channel_->do_time_sample(*myvelnp_, *myforce_);
          break;
        }
        case loma_channel_flow_of_height_2:
        {
          if (statistics_channel_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_channel_ to do a time sample for a turbulent channel flow at low "
                "Mach number");

          statistics_channel_->do_loma_time_sample(*myvelnp_, *myscaaf_, *myforce_, eosfac);
          break;
        }
        case scatra_channel_flow_of_height_2:
        {
          if (statistics_channel_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_channel_ to do a time sample for a turbulent passive scalar "
                "transport in channel");

          statistics_channel_->do_scatra_time_sample(*myvelnp_, *myscaaf_, *myforce_);
          break;
        }
        case decaying_homogeneous_isotropic_turbulence:
        case forced_homogeneous_isotropic_turbulence:
        {
          if (statistics_hit_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_hit_ to do sampling for homogeneous isotropic turbulence");

          statistics_hit_->do_time_sample(myvelnp_);
          break;
        }
        case scatra_forced_homogeneous_isotropic_turbulence:
        {
          if (statistics_hit_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_hit_ to do sampling for homogeneous isotropic turbulence");

          statistics_hit_->do_scatra_time_sample(myvelnp_, myphinp_);
          break;
        }
        case lid_driven_cavity:
        {
          if (statistics_ldc_ == Teuchos::null)
            FOUR_C_THROW("need statistics_ldc_ to do a time sample for a cavity flow");

          statistics_ldc_->do_time_sample(*myvelnp_);
          break;
        }
        case loma_lid_driven_cavity:
        {
          if (statistics_ldc_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_ldc_ to do a time sample for a cavity flow at low Mach number");

          statistics_ldc_->do_loma_time_sample(*myvelnp_, *myscaaf_, *myforce_, eosfac);
          break;
        }
        case backward_facing_step:
        case backward_facing_step2:
        {
          if (statistics_bfs_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_bfs_ to do a time sample for a flow over a backward-facing step");

          statistics_bfs_->do_time_sample(
              *myvelnp_, *mystressmanager_->get_stresses_wo_agg(myforce_));

          // do time sample for inflow channel flow
          if (inflow_)
          {
            if (params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                "channel_flow_of_height_2")
              statistics_channel_->do_time_sample(*myvelnp_, *myforce_);
            else
              FOUR_C_THROW("channel_flow_of_height_2 expected!");
          }
          break;
        }
        case periodic_hill:
        {
          if (statistics_ph_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_ph_ to do a time sample for a flow over a backward-facing step");

          statistics_ph_->do_time_sample(
              *myvelnp_, *mystressmanager_->get_wall_shear_stresses_wo_agg(myforce_));
          break;
        }
        case loma_backward_facing_step:
        {
          if (statistics_bfs_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_bfs_ to do a time sample for a flow over a backward-facing step "
                "at low Mach number");

          if (Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(*params_, "Physical Type") ==
              Inpar::FLUID::incompressible)
          {
            if (not withscatra_)
            {
              statistics_bfs_->do_time_sample(
                  *myvelnp_, *mystressmanager_->get_stresses_wo_agg(myforce_));

              // do time sample for inflow channel flow
              if (inflow_)
              {
                if (params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                    "channel_flow_of_height_2")
                  statistics_channel_->do_time_sample(*myvelnp_, *myforce_);
                else
                  FOUR_C_THROW("channel_flow_of_height_2 expected!");
              }
            }
            else
            {
              statistics_bfs_->do_scatra_time_sample(*myvelnp_, *myscaaf_);

              // do time sample for inflow channel flow
              if (inflow_)
              {
                if (params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                    "scatra_channel_flow_of_height_2")
                  FOUR_C_THROW(
                      "Use channel_flow_of_height_2 instead of scatra_channel_flow_of_height_2!");
                // to get DoScatraTimeSample() running also for inflow problems pointswise
                // evaluation as available for channel flow is required
                // statistics_channel_->DoScatraTimeSample(myvelnp_,myscaaf_,myforce_);
                // if the inflow channel is not subject to scalar transport using the functions for
                // usual channel flow is ok
                else if (params_->sublist("TURBULENT INFLOW")
                             .get<std::string>("CANONICAL_INFLOW") == "channel_flow_of_height_2")
                  statistics_channel_->do_time_sample(*myvelnp_, *myforce_);
                else
                  FOUR_C_THROW("channel_flow_of_height_2 expected!");
              }
            }
          }
          else
          {
            statistics_bfs_->do_loma_time_sample(*myvelnp_, *myscaaf_, eosfac);

            // do time sample for inflow channel flow
            if (inflow_)
            {
              if (params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                  "loma_channel_flow_of_height_2")
              {
                std::cout << "Warning: no statistics for loma inflow channel!" << std::endl;
                // to get DoLomaTimeSample() running also for inflow problems pointswise evaluation
                // as available for channel flow is required
                // statistics_channel_->DoLomaTimeSample(myvelnp_,myscaaf_,myforce_,eosfac);
                // statistics_channel_->DoTimeSample(myvelnp_,myforce_) is not an option,
                // since it requires visc and dens
                // you may use the statistics of the inlet section instead
              }
            }
          }
          break;
        }
        case blood_fda_flow:
        {
          if (statistics_bfda_ == Teuchos::null)
            FOUR_C_THROW("need statistics_bfda_ to do a time sample for a blood fda flow");

          statistics_bfda_->do_time_sample(*myvelnp_);
          break;
        }
        case square_cylinder:
        {
          if (statistics_sqc_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_sqc_ to do a time sample for a flow around a square cylinder");

          statistics_sqc_->do_time_sample(*myvelnp_);

          // computation of Lift&Drag statistics, if required
          if (params_->get<bool>("LIFTDRAG"))
          {
            Teuchos::RCP<std::map<int, std::vector<double>>> liftdragvals;

            // spatial dimension of problem
            const int ndim = params_->get<int>("number of velocity degrees of freedom");

            FLD::UTILS::lift_drag(discret_, *myforce_, mydispnp_, ndim, liftdragvals, alefluid_);

            if ((*liftdragvals).size() != 1)
            {
              FOUR_C_THROW(
                  "expecting only one liftdrag label for the sampling of a flow around a square "
                  "cylinder");
            }
            std::map<int, std::vector<double>>::iterator theonlyldval = (*liftdragvals).begin();

            statistics_sqc_->do_lift_drag_time_sample(
                ((*theonlyldval).second)[0], ((*theonlyldval).second)[1]);
          }
          break;
        }
        case rotating_circular_cylinder_nurbs:
        {
          if (statistics_ccy_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_ccy_ to do a time sample for a flow in a rotating circular "
                "cylinder");

          statistics_ccy_->do_time_sample(*myvelnp_, Teuchos::null, Teuchos::null);
          break;
        }
        case rotating_circular_cylinder_nurbs_scatra:
        {
          if (statistics_ccy_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_ccy_ to do a time sample for a flow in a rotating circular "
                "cylinder");

          statistics_ccy_->do_time_sample(*myvelnp_, myscaaf_, myphinp_);
          break;
        }
        default:
        {
          break;
        }
      }

      if (discret_->get_comm().MyPID() == 0)
      {
        std::cout
            << "Computed statistics: mean values, fluctuations, boundary forces etc.             (";
        printf("%10.4E", Teuchos::Time::wallTime() - tcpu);
        std::cout << ")";
      }

      //--------------------------------------------------
      // do averaging of residuals, dissipation rates etc
      // (all gausspoint-quantities)
      if (subgrid_dissipation_)
      {
        tcpu = Teuchos::Time::wallTime();

        switch (flow_)
        {
          case channel_flow_of_height_2:
          case loma_channel_flow_of_height_2:
          case scatra_channel_flow_of_height_2:
          case taylor_green_vortex:
          {
            if (statistics_channel_ == Teuchos::null and statistics_tgv_ == Teuchos::null)
              FOUR_C_THROW("No dissipation rates for this flow type!");

            // set vector values needed by elements
            std::map<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>> statevecs;
            std::map<std::string, Teuchos::RCP<Epetra_MultiVector>> statetenss;
            std::map<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>> scatrastatevecs;
            std::map<std::string, Teuchos::RCP<Epetra_MultiVector>> scatrafieldvecs;

            statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                "hist", myhist_));
            statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                "accam", myaccam_));
            statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                "scaaf", myscaaf_));
            statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                "scaam", myscaam_));

            if (alefluid_)
            {
              statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                  "dispnp", mydispnp_));
              statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                  "gridv", mygridvelaf_));
              if (scatradis_ != Teuchos::null) FOUR_C_THROW("Not supported!");
            }

            auto time_int_algo = Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(
                *params_, "time int algo");

            if (time_int_algo == Inpar::FLUID::timeint_afgenalpha or
                time_int_algo == Inpar::FLUID::timeint_npgenalpha)
            {
              statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                  "velaf", myvelaf_));
              if (time_int_algo == Inpar::FLUID::timeint_npgenalpha)
                statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                    "velnp", myvelnp_));

              // additional scatra vectors
              scatrastatevecs.insert(
                  std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                      "phinp", myphiaf_));
              scatrastatevecs.insert(
                  std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                      "phiam", myphiam_));
              scatrastatevecs.insert(
                  std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                      "hist", myphidtam_));
            }
            else
            {
              statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                  "velaf", myvelnp_));
              statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                  "velaf", myvelnp_));

              // additional scatra vectors
              scatrastatevecs.insert(
                  std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                      "phinp", myphinp_));
              scatrastatevecs.insert(
                  std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                      "hist", myscatrahist_));
            }

            if (Teuchos::getIntegralValue<Inpar::FLUID::FineSubgridVisc>(
                    params_->sublist("TURBULENCE MODEL"), "FSSUGRVISC") !=
                    Inpar::FLUID::FineSubgridVisc::no_fssgv or
                turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
            {
              statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                  "fsvelaf", myfsvelaf_));
              if (myfsvelaf_ == Teuchos::null) FOUR_C_THROW("Have not got fsvel!");

              if (Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(
                      *params_, "Physical Type") == Inpar::FLUID::loma and
                  turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
              {
                statevecs.insert(std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                    "fsscaaf", myfsscaaf_));
                if (myfsscaaf_ == Teuchos::null) FOUR_C_THROW("Have not got fssca!");
              }

              // additional scatra vectors
              if (withscatra_)
              {
                scatrastatevecs.insert(
                    std::pair<std::string, Teuchos::RCP<Core::LinAlg::Vector<double>>>(
                        "fsphinp", myfsphi_));
                if (myfsphi_ == Teuchos::null) FOUR_C_THROW("Have not got fsphi!");
              }
            }

            switch (flow_)
            {
              case channel_flow_of_height_2:
              case loma_channel_flow_of_height_2:
              case scatra_channel_flow_of_height_2:
              {
                statistics_channel_->evaluate_residuals(statevecs, statetenss, thermpressaf,
                    thermpressam, thermpressdtaf, thermpressdtam, scatrastatevecs, scatrafieldvecs);
                break;
              }
              case taylor_green_vortex:
              {
                statistics_tgv_->evaluate_residuals(statevecs, statetenss, thermpressaf,
                    thermpressam, thermpressdtaf, thermpressdtam);
                if (step == 0)  // sorry, this function is not called for step=0
                {
                  statistics_tgv_->dump_statistics(0);
                  statistics_tgv_->clear_statistics();
                }
                break;
              }
              default:
                FOUR_C_THROW("Dissipation not supported for this flow type!");
                break;
            }

            break;
          }
          default:
          {
            break;
          }
        }

        if (discret_->get_comm().MyPID() == 0)
        {
          std::cout << "\nresiduals, dissipation rates etc, ";
          std::cout << "all gausspoint-quantities (";
          printf("%10.4E", Teuchos::Time::wallTime() - tcpu);
          std::cout << ")";
        }
      }
      if (discret_->get_comm().MyPID() == 0)
      {
        std::cout << "\n";
      }

      if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales and inflow_ == false and
          myxwall_ == Teuchos::null)
      {
        auto time_int_scheme = Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(
            *params_, "time int algo");
        switch (flow_)
        {
          case channel_flow_of_height_2:
          {
            // add parameters of multifractal subgrid-scales model
            if (time_int_scheme == Inpar::FLUID::timeint_afgenalpha or
                time_int_scheme == Inpar::FLUID::timeint_npgenalpha)
              statistics_channel_->add_model_params_multifractal(myvelaf_, myfsvelaf_, false);
            else
              statistics_channel_->add_model_params_multifractal(myvelnp_, myfsvelaf_, false);
            break;
          }
          case scatra_channel_flow_of_height_2:
          {
            // add parameters of multifractal subgrid-scales model
            if (time_int_scheme == Inpar::FLUID::timeint_afgenalpha or
                time_int_scheme == Inpar::FLUID::timeint_npgenalpha)
              statistics_channel_->add_model_params_multifractal(myvelaf_, myfsvelaf_, true);
            else
              statistics_channel_->add_model_params_multifractal(myvelnp_, myfsvelaf_, false);
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
      if (statistics_general_mean_ != Teuchos::null)
        statistics_general_mean_->add_to_current_time_average(dt_, *myvelnp_, myscaaf_, myphinp_);

    }  // end step in sampling period

    // for homogeneous isotropic turbulence, the initial field is averaged
    // to get the amount of energy at the beginning which depends on the
    // resolution
    if (flow_ == decaying_homogeneous_isotropic_turbulence and step == 0)
    {
      if (discret_->get_comm().MyPID() == 0)
      {
        std::cout << "XXXXXXXXXXXXXXXXXXXXX              ";
        std::cout << "calculate initial energy spectrum  ";
        std::cout << "XXXXXXXXXXXXXXXXXXXXX";
        std::cout << "\n\n";
      }

      statistics_hit_->do_time_sample(myvelnp_);
      statistics_hit_->dump_statistics(0);
      statistics_hit_->clear_statistics();

      if (discret_->get_comm().MyPID() == 0)
      {
        std::cout << "XXXXXXXXXXXXXXXXXXXXX              ";
        std::cout << "wrote statistics record            ";
        std::cout << "XXXXXXXXXXXXXXXXXXXXX";
        std::cout << "\n\n";
      }
    }

    return;
  }


  /*----------------------------------------------------------------------

    Include current quantities in the time averaging procedure

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::do_time_sample(int step,
      Teuchos::RCP<Core::LinAlg::Vector<double>> velnp, Core::LinAlg::Vector<double>& force,
      Core::LinAlg::Vector<double>& phi, const Core::DOFSets::DofSet& stddofset)
  {
    // sampling takes place only in the sampling period
    if (step >= samstart_ && step <= samstop_ && flow_ != no_special_flow)
    {
      double tcpu = Teuchos::Time::wallTime();

      //--------------------------------------------------
      // calculate means, fluctuations etc of velocity,
      // pressure, boundary forces etc.
      switch (flow_)
      {
        default:
        {
          FOUR_C_THROW("called wrong DoTimeSample() for this kind of special flow");
          break;
        }
      }

      // add vector(s) to general mean value computation
      // scatra vectors may be Teuchos::null
      if (statistics_general_mean_ != Teuchos::null)
        statistics_general_mean_->add_to_current_time_average(dt_, *velnp, myscaaf_, myphinp_);

      if (discret_->get_comm().MyPID() == 0)
      {
        std::cout << "                      taking time sample (";
        printf("%10.4E", Teuchos::Time::wallTime() - tcpu);
        std::cout << ")\n";
      }

    }  // end step in sampling period

    return;
  }


  /*----------------------------------------------------------------------

    get current velnp pointer from fluid
    necessary for meshtying                                    bk 02/14
  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::get_current_velnp(
      Teuchos::RCP<Core::LinAlg::Vector<double>> velnp)
  {
    myvelnp_ = velnp;
    return;
  }

  /*----------------------------------------------------------------------

    Write (dump) the statistics to a file

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::do_output(
      Core::IO::DiscretizationWriter& output, int step, const bool inflow)
  {
    // sampling takes place only in the sampling period
    if (step >= samstart_ && step <= samstop_ && flow_ != no_special_flow)
    {
      enum Format
      {
        write_single_record,
        write_multiple_records,
        do_not_write
      } outputformat = do_not_write;
      bool output_inflow = false;

      // sampling a la Volker --- single record is constantly updated
      if (dumperiod_ != 0)
      {
        int samstep = step - samstart_ + 1;

        // dump every dumperiod steps
        if (samstep % dumperiod_ == 0) outputformat = write_single_record;
      }

      // sampling a la Peter --- for each sampling period a
      // new record is written; they can be combined by a
      // postprocessing script to a single long term sample
      // (allows restarts during sampling)
      if (dumperiod_ == 0)
      {
        int upres = params_->get<int>("write solution every");
        int uprestart = params_->get<int>("write restart every");

        // dump in combination with a restart/output
        if ((step % upres == 0 || (uprestart > 0 && step % uprestart == 0)) && step > samstart_)
          outputformat = write_multiple_records;
      }
      if (inflow_)
      {
        int upres = params_->get<int>("write solution every");
        int uprestart = params_->get<int>("write restart every");

        // dump in combination with a restart/output
        if ((step % upres == 0 || (uprestart > 0 && step % uprestart == 0)) && step > samstart_)
          output_inflow = true;
      }

      if (discret_->get_comm().MyPID() == 0 && outputformat != do_not_write)
        std::cout << "---  statistics record: \n" << std::flush;

      // do actual output (time averaging)
      switch (flow_)
      {
        case channel_flow_of_height_2:
        {
          if (statistics_channel_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_channel_ to do a time sample for a turbulent channel flow");


          if (outputformat == write_multiple_records)
          {
            statistics_channel_->time_average_means_and_output_of_statistics(step);
            statistics_channel_->clear_statistics();
          }

          if (outputformat == write_single_record) statistics_channel_->dump_statistics(step);
          break;
        }
        case loma_channel_flow_of_height_2:
        {
          if (statistics_channel_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_channel_ to do a time sample for a turbulent channel flow at low "
                "Mach number");

          if (outputformat == write_single_record) statistics_channel_->dump_loma_statistics(step);
          break;
        }
        case scatra_channel_flow_of_height_2:
        {
          if (statistics_channel_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_channel_ to do a time sample for a turbulent channel flow at low "
                "Mach number");

          if (outputformat == write_single_record)
            statistics_channel_->dump_scatra_statistics(step);
          break;
        }
        case decaying_homogeneous_isotropic_turbulence:
        case forced_homogeneous_isotropic_turbulence:
        {
          if (statistics_hit_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_hit_ to do sampling for homogeneous isotropic turbulence");

          if (flow_ == forced_homogeneous_isotropic_turbulence)
          {
            // write statistics only during sampling period
            if (outputformat == write_single_record)
              statistics_hit_->dump_statistics(step);
            else if (outputformat == write_multiple_records)
            {
              statistics_hit_->dump_statistics(step, true);
              statistics_hit_->clear_statistics();
            }
          }
          else
          {
            // write statistics for every time step,
            // since there is not any statistical-stationary state
            statistics_hit_->dump_statistics(step);
            statistics_hit_->clear_statistics();
          }
          break;
        }
        case scatra_forced_homogeneous_isotropic_turbulence:
        {
          if (statistics_hit_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_hit_ to do sampling for homogeneous isotropic turbulence");

          // write statistics only during sampling period
          if (outputformat == write_single_record)
            statistics_hit_->dump_scatra_statistics(step);
          else if (outputformat == write_multiple_records)
          {
            statistics_hit_->dump_scatra_statistics(step, true);
            statistics_hit_->clear_scatra_statistics();
          }
          break;
        }
        case taylor_green_vortex:
        {
          // write statistics for every time step,
          // since there is not any statistical-stationary state
          statistics_tgv_->dump_statistics(step);
          statistics_tgv_->clear_statistics();
          break;
        }
        case lid_driven_cavity:
        {
          if (statistics_ldc_ == Teuchos::null)
            FOUR_C_THROW("need statistics_ldc_ to do a time sample for a lid driven cavity");

          if (outputformat == write_single_record) statistics_ldc_->dump_statistics(step);

          if (outputformat != do_not_write) statistics_ldc_->write_restart(output);
          break;
        }
        case loma_lid_driven_cavity:
        {
          if (statistics_ldc_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_ldc_ to do a time sample for a lid driven cavity at low Mach "
                "number");

          if (outputformat == write_single_record) statistics_ldc_->dump_loma_statistics(step);
          break;
        }
        case backward_facing_step:
        case backward_facing_step2:
        {
          if (statistics_bfs_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_bfs_ to do a time sample for a flow over a backward-facing step");

          if (outputformat == write_single_record) statistics_bfs_->dump_statistics(step);

          // write statistics of inflow channel flow
          if (inflow_)
          {
            if (params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                "channel_flow_of_height_2")
            {
              if (output_inflow)
              {
                statistics_channel_->time_average_means_and_output_of_statistics(step);
                statistics_channel_->clear_statistics();
              }
            }
          }
          break;
        }
        case periodic_hill:
        {
          if (statistics_ph_ == Teuchos::null)
            FOUR_C_THROW("need statistics_ph_ to do a time sample for a flow over a periodic hill");

          if (outputformat == write_single_record) statistics_ph_->dump_statistics(step);

          break;
        }
        case loma_backward_facing_step:
        {
          if (statistics_bfs_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_bfs_ to do a time sample for a flow over a backward-facing step "
                "at low Mach number");

          if (Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(*params_, "Physical Type") ==
              Inpar::FLUID::incompressible)
          {
            if (not withscatra_)
            {
              if (outputformat == write_single_record) statistics_bfs_->dump_statistics(step);

              // write statistics of inflow channel flow
              if (inflow_)
              {
                if (params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                    "channel_flow_of_height_2")
                {
                  if (output_inflow)
                  {
                    statistics_channel_->time_average_means_and_output_of_statistics(step);
                    statistics_channel_->clear_statistics();
                  }
                }
              }
            }
            else
            {
              if (outputformat == write_single_record)
                statistics_bfs_->dump_scatra_statistics(step);

              // write statistics of inflow channel flow
              if (inflow_)
              {
                if (params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                    "scatra_channel_flow_of_height_2")
                {
                  if (outputformat == write_single_record)
                    statistics_channel_->dump_scatra_statistics(step);
                }
                else if (params_->sublist("TURBULENT INFLOW")
                             .get<std::string>("CANONICAL_INFLOW") == "channel_flow_of_height_2")
                {
                  if (output_inflow)
                  {
                    statistics_channel_->time_average_means_and_output_of_statistics(step);
                    statistics_channel_->clear_statistics();
                  }
                }
              }
            }
          }
          else  // loma
          {
            if (outputformat == write_single_record) statistics_bfs_->dump_loma_statistics(step);

            // write statistics of inflow channel flow
            if (inflow_)
            {
              if (params_->sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW") ==
                  "loma_channel_flow_of_height_2")
              {
                std::cout << "Warning: no statistics for loma inflow channel!" << std::endl;
                //              if(outputformat == write_single_record)
                //                statistics_channel_->DumpLomaStatistics(step);
              }
            }
          }
          break;
        }
        case blood_fda_flow:
        {
          if (statistics_bfda_ == Teuchos::null)
            FOUR_C_THROW("need statistics_bfda_ to do a time sample for a blood fda flow");

          if (outputformat == write_single_record) statistics_bfda_->dump_statistics(step);

          break;
        }
        case square_cylinder:
        {
          if (statistics_sqc_ == Teuchos::null)
            FOUR_C_THROW("need statistics_sqc_ to do a time sample for a square cylinder flow");

          if (outputformat == write_single_record) statistics_sqc_->dump_statistics(step);
          break;
        }
        case rotating_circular_cylinder_nurbs:
        case rotating_circular_cylinder_nurbs_scatra:
        {
          if (statistics_ccy_ == Teuchos::null)
            FOUR_C_THROW(
                "need statistics_ccy_ to do a time sample for a flow in a rotating circular "
                "cylinder");

          statistics_ccy_->time_average_means_and_output_of_statistics(step);
          statistics_ccy_->clear_statistics();
          break;
        }
        default:
        {
          break;
        }
      }

      if (discret_->get_comm().MyPID() == 0 && outputformat != do_not_write)
      {
        std::cout << "XXXXXXXXXXXXXXXXXXXXX              ";
        std::cout << "wrote statistics record            ";
        std::cout << "XXXXXXXXXXXXXXXXXXXXX";
        std::cout << "\n\n";
      }


      // dump general mean value output in combination with a restart/output
      // don't write output if turbulent inflow or twophaseflow is computed
      if (!inflow and flow_ != bubbly_channel_flow)
      {
        int upres = params_->get<int>("write solution every");
        int uprestart = params_->get<int>("write restart every");

        if ((step % upres == 0 || (uprestart > 0 && step % uprestart == 0)) &&
            (statistics_general_mean_ != Teuchos::null))
          statistics_general_mean_->write_old_average_vec(output);
      }
    }  // end step is in sampling period

    if (discret_->get_comm().MyPID() == 0 and turbmodel_ == Inpar::FLUID::dynamic_vreman)
    {
      std::string fnamevreman(statistics_outfilename_);

      fnamevreman.append(".vremanconstant");
      double Cv;
      double Cv_theo;
      double Dt = 0.0;
      std::cout << __LINE__ << std::endl;
      Cv = params_->get<double>("C_vreman", 0.0);
      Cv_theo = params_->get<double>("C_vreman_theoretical");
      if (withscatra_) Dt = params_->get<double>("Dt_vreman");
      std::ofstream fvreman;
      if (step <= 1)
      {
        fvreman.open(fnamevreman.c_str(), std::ofstream::trunc);
        fvreman << "time step C_v         C_v clipped D_T (required for subgrid diffusivity)\n";
      }
      else
        fvreman.open(fnamevreman.c_str(), std::ofstream::app);
      fvreman.width(10);
      fvreman << step;
      fvreman.width(12);
      fvreman << Cv_theo;
      fvreman.width(12);
      fvreman << Cv;
      if (withscatra_)
      {
        fvreman.width(12);
        fvreman << Dt * Cv;  // to make it comparable to the the original paper
      }
      fvreman << "\n";
      fvreman.flush();
      fvreman.close();
    }

    return;
  }  // DoOutput


  /*----------------------------------------------------------------------

  Provide access to scalar transport field

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::add_scatra_field(
      Teuchos::RCP<ScaTra::ScaTraTimIntImpl> scatra_timeint)
  {
    if (discret_->get_comm().MyPID() == 0)
    {
      Core::IO::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                     << Core::IO::endl;
      Core::IO::cout << "turbulence_statistic_manager: provided access to ScaTra time integration"
                     << Core::IO::endl;
      Core::IO::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                     << Core::IO::endl;
    }

    // store the relevant pointers to provide access
    scatradis_ = scatra_timeint->discretization();
    scatraparams_ = scatra_timeint->scatra_parameter_list();
    // and sublists from extraparams
    scatraextraparams_ = scatra_timeint->scatra_extra_parameter_list();
    // remark: this is not a good idea, since the sublists are copied and modifications
    //         during simulation are never seen in the statistics manager; moreover
    //         this would add the following sublists also to the parameter list in the
    //         scatra time integration
    // scatraparams_->sublist("TURBULENCE MODEL") =
    // scatra_timeint->scatra_extra_parameter_list()->sublist("TURBULENCE MODEL");
    // scatraparams_->sublist("SUBGRID VISCOSITY") =
    // scatra_timeint->scatra_extra_parameter_list()->sublist("SUBGRID VISCOSITY");
    // scatraparams_->sublist("MULTIFRACTAL SUBGRID SCALES") =
    // scatra_timeint->scatra_extra_parameter_list()->sublist("MULTIFRACTAL SUBGRID SCALES");
    // scatraparams_->sublist("LOMA") =
    // scatra_timeint->scatra_extra_parameter_list()->sublist("LOMA");
    scatratimeparams_ = scatra_timeint->scatra_time_parameter_list();
    // required vectors
    // remark: Although some of these field are already set for the fluid,
    //         we set set them here once more. They are required for integration
    //         on the element level of the scatra field and have to be set in the
    //         specific form using MultiVectors (cf. scatra time integration). If
    //         these vectors are not taken form the scatra time integration we
    //         would have to transfer them to the scatra dofs here!
    myphinp_ = scatra_timeint->phinp();
    myphiaf_ = scatra_timeint->phiaf();
    myphiam_ = scatra_timeint->phiam();
    myscatrahist_ = scatra_timeint->hist();
    myphidtam_ = scatra_timeint->phidtam();
    myfsphi_ = scatra_timeint->fs_phi();

    if (statistics_general_mean_ != Teuchos::null)
      statistics_general_mean_->add_scatra_results(scatradis_, *myphinp_);

    if (statistics_ccy_ != Teuchos::null)
      statistics_ccy_->add_scatra_results(scatradis_, *myphinp_);

    if (flow_ == scatra_channel_flow_of_height_2 or flow_ == loma_channel_flow_of_height_2)
    {
      statistics_channel_->store_scatra_discret_and_params(
          scatradis_, scatraparams_, scatraextraparams_, scatratimeparams_);
    }

    if (flow_ == scatra_forced_homogeneous_isotropic_turbulence)
      statistics_hit_->store_scatra_discret(scatradis_);

    withscatra_ = true;

    return;
  }


  /*----------------------------------------------------------------------

  Write (dump) the scatra-specific mean fields to the result file

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::do_output_for_scatra(
      Core::IO::DiscretizationWriter& output, int step)
  {
    // sampling takes place only in the sampling period
    if (step >= samstart_ && step <= samstop_ && flow_ != no_special_flow)
    {
      // statistics for scatra fields was already written during DoOutput()
      // Thus, we have to care for the mean field only:

      // dump general mean value output for scatra results
      // in combination with a restart/output
      int upres = params_->get("write solution every", -1);
      int uprestart = params_->get("write restart every", -1);

      if ((step % upres == 0 || step % uprestart == 0) &&
          (statistics_general_mean_ != Teuchos::null))
        statistics_general_mean_->do_output_for_scatra(output, step);
    }
    return;
  }


  /*----------------------------------------------------------------------

  Restart statistics collection

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::read_restart(Core::IO::DiscretizationReader& reader, int step)
  {
    if (samstart_ < step && step <= samstop_)
    {
      if (statistics_general_mean_ != Teuchos::null)
      {
        if (discret_->get_comm().MyPID() == 0)
        {
          std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXX          ";
          std::cout << "Read general mean values           ";
          std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXX";
          std::cout << "\n\n";
        }

        statistics_general_mean_->read_old_statistics(reader);
      }

      if (statistics_ldc_ != Teuchos::null)
      {
        if (discret_->get_comm().MyPID() == 0)
        {
          std::cout << "XXXXXXXXXXXXXXXXXXXXX              ";
          std::cout << "Read ldc statistics                ";
          std::cout << "XXXXXXXXXXXXXXXXXXXXX";
          std::cout << "\n\n";
        }

        statistics_ldc_->read_restart(reader);
      }
    }

    return;
  }  // Restart


  /*----------------------------------------------------------------------

  Restart for scatra mean fields (statistics was restarted via restart() )

  ----------------------------------------------------------------------*/
  void TurbulenceStatisticManager::read_restart_scatra(
      Core::IO::DiscretizationReader& scatrareader, int step)
  {
    // we have only to read in the mean field.
    // The rest of the restart was already done during the restart() call
    if (statistics_general_mean_ != Teuchos::null)
    {
      if (samstart_ < step && step <= samstop_)
      {
        if (discret_->get_comm().MyPID() == 0)
        {
          std::cout << "XXXXXXXXXXXXXXXXXXXXX        ";
          std::cout << "Read general mean values for ScaTra      ";
          std::cout << "XXXXXXXXXXXXXXXXXXXXX";
          std::cout << "\n\n";
        }

        statistics_general_mean_->read_old_statistics_scatra(scatrareader);
      }
    }

    return;
  }  // RestartScaTra


}  // end namespace FLD

FOUR_C_NAMESPACE_CLOSE
