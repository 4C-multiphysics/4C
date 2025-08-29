// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_timint_red.hpp"

#include "4C_adapter_art_net.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_fluid_coupling_red_models.hpp"
#include "4C_fluid_meshtying.hpp"
#include "4C_fluid_volumetric_surfaceFlow_condition.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  Constructor (public)                                       bk 11/13 |
 *----------------------------------------------------------------------*/
FLD::TimIntRedModels::TimIntRedModels(const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis, solver, params, output, alefluid),
      traction_vel_comp_adder_bc_(nullptr),
      coupled3D_redDbc_art_(nullptr),
      ART_timeInt_(nullptr),
      coupled3D_redDbc_airways_(nullptr),
      airway_imp_timeInt_(nullptr),
      vol_surf_flow_bc_(nullptr),
      vol_surf_flow_bc_maps_(nullptr),
      vol_flow_rates_bc_extractor_(nullptr),
      strong_redD_3d_coupling_(false)
{
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                rasthofer 04/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::init()
{
  // Vectors associated to boundary conditions
  // -----------------------------------------

  // create the volumetric-surface-flow condition
  if (alefluid_)
  {
    discret_->set_state("dispnp", *dispn_);
  }

  vol_surf_flow_bc_ = std::make_shared<Utils::FluidVolumetricSurfaceFlowWrapper>(discret_, dta_);

  // evaluate the map of the womersley bcs
  vol_flow_rates_bc_extractor_ = std::make_shared<FLD::Utils::VolumetricFlowMapExtractor>();
  vol_flow_rates_bc_extractor_->setup(*discret_);
  vol_surf_flow_bc_maps_ = std::make_shared<Core::LinAlg::Map>(
      *(vol_flow_rates_bc_extractor_->volumetric_surface_flow_cond_map()));

  // -------------------------------------------------------------------
  // Initialize the reduced models
  // -------------------------------------------------------------------

  strong_redD_3d_coupling_ = params_->get<bool>("Strong 3D_redD coupling", false);

  {
    ART_timeInt_ = dyn_art_net_drt(true);
    // Check if one-dimensional artery network problem exist
    if (ART_timeInt_ != nullptr)
    {
      Core::IO::DiscretizationWriter output_redD(ART_timeInt_->discretization(),
          Global::Problem::instance()->output_control_file(),
          Global::Problem::instance()->spatial_approximation_type());
      discret_->clear_state();
      discret_->set_state("velaf", *zeros_);
      if (alefluid_)
      {
        discret_->set_state("dispnp", *dispnp_);
      }
      coupled3D_redDbc_art_ =
          std::make_shared<Utils::FluidCouplingWrapper<Adapter::ArtNet>>(discret_,
              ART_timeInt_->discretization(), ART_timeInt_, output_redD, dta_, ART_timeInt_->dt());
    }


    airway_imp_timeInt_ = dyn_red_airways_drt(true);
    // Check if one-dimensional artery network problem exist
    if (airway_imp_timeInt_ != nullptr)
    {
      Core::IO::DiscretizationWriter output_redD(airway_imp_timeInt_->discretization(),
          Global::Problem::instance()->output_control_file(),
          Global::Problem::instance()->spatial_approximation_type());
      discret_->clear_state();
      discret_->set_state("velaf", *zeros_);
      if (alefluid_)
      {
        discret_->set_state("dispnp", *dispnp_);
      }
      coupled3D_redDbc_airways_ =
          std::make_shared<Utils::FluidCouplingWrapper<Airway::RedAirwayImplicitTimeInt>>(discret_,
              airway_imp_timeInt_->discretization(), airway_imp_timeInt_, output_redD, dta_,
              airway_imp_timeInt_->dt());
    }


    zeros_->put_scalar(0.0);  // just in case of change
  }

  traction_vel_comp_adder_bc_ = std::make_shared<Utils::TotalTractionCorrector>(discret_, dta_);


  // ------------------------------------------------------------------------------
  // Check, if features are used with the locsys manager that are not supported,
  // or better, not implemented yet.
  // ------------------------------------------------------------------------------
  if (locsysman_ != nullptr)
  {
    // Models
    if ((ART_timeInt_ != nullptr) or (airway_imp_timeInt_ != nullptr))
    {
      FOUR_C_THROW(
          "No problem types involving airways are supported for use with locsys conditions!");
    }
  }
}



/*----------------------------------------------------------------------*
 | evaluate special boundary conditions                        bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::do_problem_specific_boundary_conditions()
{
  if (alefluid_)
  {
    discret_->set_state("dispnp", *dispnp_);
  }

  // Check if one-dimensional artery network problem exist
  if (ART_timeInt_ != nullptr)
  {
    coupled3D_redDbc_art_->evaluate_dirichlet(*velnp_, *(dbcmaps_->cond_map()), time_);
  }
  // update the 3D-to-reduced_D coupling data
  // Check if one-dimensional artery network problem exist
  if (airway_imp_timeInt_ != nullptr)
  {
    coupled3D_redDbc_airways_->evaluate_dirichlet(*velnp_, *(dbcmaps_->cond_map()), time_);
  }

  // Evaluate the womersley velocities
  vol_surf_flow_bc_->evaluate_velocities(*velnp_, time_);
}

/*----------------------------------------------------------------------*
| Update3DToReduced in assemble_mat_and_rhs                       bk 11/13 |
*----------------------------------------------------------------------*/
void FLD::TimIntRedModels::update_3d_to_reduced_mat_and_rhs()
{
  discret_->clear_state();

  discret_->set_state("velaf", *velnp_);
  discret_->set_state("hist", *hist_);

  if (alefluid_)
  {
    discret_->set_state("dispnp", *dispnp_);
  }

  // Check if one-dimensional artery network problem exist
  if (ART_timeInt_ != nullptr)
  {
    if (strong_redD_3d_coupling_)
    {
      coupled3D_redDbc_art_->load_state();
      coupled3D_redDbc_art_->flow_rate_calculation(time_, dta_);
      coupled3D_redDbc_art_->apply_boundary_conditions(time_, dta_, theta_);
    }
    coupled3D_redDbc_art_->update_residual(*residual_);
  }
  // update the 3D-to-reduced_D coupling data
  // Check if one-dimensional artery network problem exist
  if (airway_imp_timeInt_ != nullptr)
  {
    if (strong_redD_3d_coupling_)
    {
      coupled3D_redDbc_airways_->load_state();
      coupled3D_redDbc_airways_->flow_rate_calculation(time_, dta_);
      coupled3D_redDbc_airways_->apply_boundary_conditions(time_, dta_, theta_);
    }
    coupled3D_redDbc_airways_->update_residual(*residual_);
  }

  //----------------------------------------------------------------------
  // add the traction velocity component
  //----------------------------------------------------------------------

  traction_vel_comp_adder_bc_->evaluate_velocities(*velnp_, time_, theta_, dta_);
  traction_vel_comp_adder_bc_->update_residual(*residual_);

  discret_->clear_state();
}

/*----------------------------------------------------------------------*
| call update_3d_to_reduced_mat_and_rhs                              bk 11/13 |
*----------------------------------------------------------------------*/
void FLD::TimIntRedModels::set_custom_ele_params_assemble_mat_and_rhs(
    Teuchos::ParameterList& eleparams)
{
  // these are the only routines that have to be called in assemble_mat_and_rhs
  // before Evaluate in the RedModels case
  update_3d_to_reduced_mat_and_rhs();
}

/*----------------------------------------------------------------------*
 | output of solution vector of ReducedD problem to binio   ismail 01/13|
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::output_reduced_d()
{
  // output of solution
  if (step_ % upres_ == 0)
  {
    // write reduced model problem
    // Check if one-dimensional artery network problem exist
    if (ART_timeInt_ != nullptr)
    {
      std::shared_ptr<Teuchos::ParameterList> redD_export_params;
      redD_export_params = std::make_shared<Teuchos::ParameterList>();

      redD_export_params->set<int>("step", step_);
      redD_export_params->set<int>("upres", upres_);
      redD_export_params->set<int>("uprestart", uprestart_);
      redD_export_params->set<double>("time", time_);

      ART_timeInt_->output(true, redD_export_params);
    }

    // Check if one-dimensional artery network problem exist
    if (airway_imp_timeInt_ != nullptr)
    {
      std::shared_ptr<Teuchos::ParameterList> redD_export_params;
      redD_export_params = std::make_shared<Teuchos::ParameterList>();

      redD_export_params->set<int>("step", step_);
      redD_export_params->set<int>("upres", upres_);
      redD_export_params->set<int>("uprestart", uprestart_);
      redD_export_params->set<double>("time", time_);

      airway_imp_timeInt_->output(true, redD_export_params);
    }
  }
}  // FLD::TimIntRedModels::OutputReducedD

/*----------------------------------------------------------------------*
 | read some additional data in restart                         bk 12/13|
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::read_restart(int step)
{
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step);

  vol_surf_flow_bc_->read_restart(reader);

  traction_vel_comp_adder_bc_->read_restart(reader);

  // Read restart of one-dimensional arterial network
  if (ART_timeInt_ != nullptr)
  {
    coupled3D_redDbc_art_->read_restart(reader);
  }
  // Check if zero-dimensional airway network problem exist
  if (airway_imp_timeInt_ != nullptr)
  {
    coupled3D_redDbc_airways_->read_restart(reader);
  }

  read_restart_reduced_d(step);
}

/*----------------------------------------------------------------------*
 |                                                          ismail 01/13|
 -----------------------------------------------------------------------*/
void FLD::TimIntRedModels::read_restart_reduced_d(int step)
{
  // Check if one-dimensional artery network problem exist
  if (ART_timeInt_ != nullptr)
  {
    ART_timeInt_->read_restart(step, true);
  }

  // Check if one-dimensional artery network problem exist
  if (airway_imp_timeInt_ != nullptr)
  {
    airway_imp_timeInt_->read_restart(step, true);
  }
}  // FLD::TimIntRedModels::ReadRestartReadRestart(int step)

/*----------------------------------------------------------------------*
 | do some additional steps in setup_meshtying                   bk 12/13|
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::setup_meshtying()
{
  FluidImplicitTimeInt::setup_meshtying();
  // Volume surface flow conditions are treated in the same way as Dirichlet condition.
  // Therefore, a volume surface flow condition cannot be defined on the same nodes as the
  // slave side of an internal interface
  // Solution:  Exclude those nodes of your surface
  // but:       The resulting inflow rate (based on the area)
  //            as well as the profile will be different
  //            since it is based on a different surface discretization!!

  if (vol_surf_flow_bc_maps_->num_global_elements() != 0)
  {
    meshtying_->check_overlapping_bc(*vol_surf_flow_bc_maps_);
    meshtying_->dirichlet_on_master(vol_surf_flow_bc_maps_);
  }
}

/*----------------------------------------------------------------------*
 | output of solution vector to binio                        gammi 04/07|
 | overloading function                                         bk 12/13|
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::output()
{
  FluidImplicitTimeInt::output();
  // output of solution
  if (step_ % upres_ == 0)
  {
    vol_surf_flow_bc_->output(*output_);
    traction_vel_comp_adder_bc_->output(*output_);

    if (uprestart_ != 0 && step_ % uprestart_ == 0)  // add restart data
    {
      // Check if one-dimensional artery network problem exist
      if (ART_timeInt_ != nullptr)
      {
        coupled3D_redDbc_art_->write_restart(*output_);
      }
      // Check if zero-dimensional airway network problem exist
      if (airway_imp_timeInt_ != nullptr)
      {
        coupled3D_redDbc_airways_->write_restart(*output_);
      }
    }
  }
  // write restart also when uprestart_ is not a integer multiple of upres_
  else if (uprestart_ > 0 && step_ % uprestart_ == 0)
  {
    // write reduced model problem
    // Check if one-dimensional artery network problem exist
    if (ART_timeInt_ != nullptr)
    {
      coupled3D_redDbc_art_->write_restart(*output_);
    }
    // Check if zero-dimensional airway network problem exist
    if (airway_imp_timeInt_ != nullptr)
    {
      coupled3D_redDbc_airways_->write_restart(*output_);
    }
  }

  output_reduced_d();
}  // TimIntRedModels::Output

/*----------------------------------------------------------------------*
 | read some additional data in restart                         bk 12/13|
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::insert_volumetric_surface_flow_cond_vector(
    std::shared_ptr<Core::LinAlg::Vector<double>> vel,
    std::shared_ptr<Core::LinAlg::Vector<double>> res)
{
  // -------------------------------------------------------------------
  // take surface volumetric flow rate into account
  //    std::shared_ptr<Core::LinAlg::Vector<double>> temp_vec = Teuchos::rcp(new
  //    Core::LinAlg::Vector<double>(*vol_surf_flow_bc_maps_,true));
  //    vol_surf_flow_bc_->insert_cond_vector( *temp_vec , *residual_);
  // -------------------------------------------------------------------
  vol_flow_rates_bc_extractor_->insert_volumetric_surface_flow_cond_vector(
      *vol_flow_rates_bc_extractor_->extract_volumetric_surface_flow_cond_vector(*vel), *res);
}

/*----------------------------------------------------------------------*
 | prepare AVM3-based scale separation                         vg 10/08 |
 | overloaded in TimIntRedModelsModels and TimIntLoma        bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::avm3_preparation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("           + avm3");

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // necessary here, because some application time integrations add something to the residual
  // before the Neumann loads are added
  residual_->put_scalar(0.0);

  // Maybe this needs to be inserted in case of impedanceBC + AVM3
  //  if (nonlinearbc_ && isimpedancebc_)
  //  {
  //    // add impedance Neumann loads
  //    impedancebc_->update_residual(residual_);
  //  }

  avm3_assemble_mat_and_rhs(eleparams);

  // apply Womersley as a Dirichlet BC
  Core::LinAlg::apply_dirichlet_to_system(
      *sysmat_, *incvel_, *residual_, *zeros_, *(vol_surf_flow_bc_maps_));

  // get scale-separation matrix
  avm3_get_scale_separation_matrix();
}  // TimIntRedModels::avm3_preparation

/*----------------------------------------------------------------------*
 | RedModels - specific BC in linear_relaxation_solve            bk 12/13|
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::custom_solve(std::shared_ptr<Core::LinAlg::Vector<double>> relax)
{
  // apply Womersley as a Dirichlet BC
  Core::LinAlg::apply_dirichlet_to_system(*incvel_, *residual_, *relax, *(vol_surf_flow_bc_maps_));

  // apply Womersley as a Dirichlet BC
  sysmat_->apply_dirichlet(*(vol_surf_flow_bc_maps_));
}

/*----------------------------------------------------------------------*
 | RedModels - prepare time step                            ismail 06/14|
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::prepare_time_step()
{
  FluidImplicitTimeInt::prepare_time_step();

  discret_->clear_state();
  discret_->set_state("velaf", *velnp_);
  discret_->set_state("hist", *hist_);

  if (alefluid_) discret_->set_state("dispnp", *dispnp_);

  // Check if one-dimensional artery network problem exist
  if (ART_timeInt_ != nullptr)
  {
    coupled3D_redDbc_art_->save_state();
    coupled3D_redDbc_art_->flow_rate_calculation(time_, dta_);
    coupled3D_redDbc_art_->apply_boundary_conditions(time_, dta_, theta_);
  }


  // Check if one-dimensional artery network problem exist
  if (airway_imp_timeInt_ != nullptr)
  {
    coupled3D_redDbc_airways_->save_state();
    coupled3D_redDbc_airways_->flow_rate_calculation(time_, dta_);
    coupled3D_redDbc_airways_->apply_boundary_conditions(time_, dta_, theta_);
  }

  discret_->clear_state();
}


/*----------------------------------------------------------------------*
 | Apply Womersley bc to shapederivatives                       bk 12/13|
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::assemble_mat_and_rhs()
{
  FluidImplicitTimeInt::assemble_mat_and_rhs();

  if (shapederivatives_ != nullptr)
  {
    // apply the womersley bc as a dirichlet bc
    shapederivatives_->apply_dirichlet(*(vol_surf_flow_bc_maps_), false);
  }
}

/*----------------------------------------------------------------------*
 | Apply Womersley bc to system                                 bk 12/13|
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModels::apply_dirichlet_to_system()
{
  FluidImplicitTimeInt::apply_dirichlet_to_system();

  if (locsys_manager() != nullptr)
  {
    // apply Womersley as a Dirichlet BC
    Core::LinAlg::apply_dirichlet_to_system(
        *Core::LinAlg::cast_to_sparse_matrix_and_check_success(sysmat_), *incvel_, *residual_,
        *locsysman_->trafo(), *zeros_, *(vol_surf_flow_bc_maps_));
  }
  else
  {
    // apply Womersley as a Dirichlet BC
    Core::LinAlg::apply_dirichlet_to_system(
        *sysmat_, *incvel_, *residual_, *zeros_, *(vol_surf_flow_bc_maps_));
  }
}

FOUR_C_NAMESPACE_CLOSE
