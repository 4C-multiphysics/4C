// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_coupling_volmortar_shape.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_scatra_multiscale.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_calc.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_parameter_turbulence.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | evaluate action                                           fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalc<distype, probdim>::evaluate_action(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const ScaTra::Action& action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  //(for now) only first dof set considered
  const std::vector<int>& lm = la[0].lm_;
  // determine and evaluate action
  switch (action)
  {
    // calculate global mass matrix
    case ScaTra::Action::calc_mass_matrix:
    {
      // integration points and weights
      const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
          ScaTra::DisTypeToOptGaussRule<distype>::rule);

      // loop over integration points
      for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
      {
        // evaluate values of shape functions and domain integration factor at current integration
        // point
        const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

        // loop over dofs
        for (int k = 0; k < numdofpernode_; ++k) calc_mat_mass(elemat1_epetra, k, fac, 1.);
      }  // loop over integration points

      break;
    }

    // calculate time derivative for time value t_0
    case ScaTra::Action::calc_initial_time_deriv:
    {
      // calculate matrix and rhs
      calc_initial_time_derivative(ele, elemat1_epetra, elevec1_epetra, params, discretization, la);
      break;
    }

    case ScaTra::Action::integrate_shape_functions:
    {
      // calculate integral of shape functions
      const auto dofids = params.get<std::shared_ptr<Core::LinAlg::IntSerialDenseVector>>("dofids");
      integrate_shape_functions(ele, elevec1_epetra, *dofids);

      break;
    }

    case ScaTra::Action::calc_flux_domain:
    {
      // get number of dofset associated with velocity related dofs
      const int ndsvel = scatrapara_->nds_vel();

      // get velocity values at nodes
      const std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
          discretization.get_state(ndsvel, "convective velocity field");
      const std::shared_ptr<const Core::LinAlg::Vector<double>> vel =
          discretization.get_state(ndsvel, "velocity field");

      // safety check
      if (convel == nullptr or vel == nullptr) FOUR_C_THROW("Cannot get state vector");

      // determine number of velocity related dofs per node
      const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

      // construct location vector for velocity related dofs
      std::vector<int> lmvel(nsd_ * nen_, -1);
      for (unsigned inode = 0; inode < nen_; ++inode)
        for (unsigned idim = 0; idim < nsd_; ++idim)
          lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

      // extract local values of (convective) velocity field from global state vector
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*convel, econvelnp_, lmvel);
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*vel, evelnp_, lmvel);

      // rotate the vector field in the case of rotationally symmetric boundary conditions
      rotsymmpbc_->rotate_my_values_if_necessary(econvelnp_);
      rotsymmpbc_->rotate_my_values_if_necessary(evelnp_);

      // need current values of transported scalar
      // -> extract local values from global vectors
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp_, lm);

      // access control parameter for flux calculation
      Inpar::ScaTra::FluxType fluxtype = scatrapara_->calc_flux_domain();
      std::shared_ptr<std::vector<int>> writefluxids = scatrapara_->write_flux_ids();

      // we always get an 3D flux vector for each node
      Core::LinAlg::Matrix<3, nen_> eflux(Core::LinAlg::Initialization::zero);

      // do a loop for systems of transported scalars
      for (int& writefluxid : *writefluxids)
      {
        int k = writefluxid - 1;
        // calculate flux vectors for actual scalar
        eflux.clear();
        calculate_flux(eflux, ele, fluxtype, k);
        // assembly
        for (unsigned inode = 0; inode < nen_; inode++)
        {
          const int fvi = inode * numdofpernode_ + k;
          elevec1_epetra[fvi] += eflux(0, inode);
          elevec2_epetra[fvi] += eflux(1, inode);
          elevec3_epetra[fvi] += eflux(2, inode);
        }
      }  // loop over numscal

      break;
    }

    case ScaTra::Action::calc_total_and_mean_scalars:
    {
      // get flag for inverting
      const bool inverting = params.get<bool>("inverting");

      const bool calc_grad_phi = params.get<bool>("calc_grad_phi");
      // need current scalar vector
      // -> extract local values from the global vectors
      auto phinp = discretization.get_state("phinp");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp_, lm);

      // calculate scalars and domain integral
      calculate_scalars(ele, elevec1_epetra, inverting, calc_grad_phi);

      break;
    }

    case ScaTra::Action::calc_mean_scalar_time_derivatives:
    {
      calculate_scalar_time_derivatives(discretization, lm, elevec1_epetra);
      break;
    }

    // calculate filtered fields for calculation of turbulent Prandtl number
    // required for dynamic Smagorinsky model in scatra
    case ScaTra::Action::calc_scatra_box_filter:
    {
      if (nsd_ == 3)
        calc_box_filter(ele, params, discretization, la);
      else
        FOUR_C_THROW("action 'calc_scatra_box_filter' is 3D specific action");

      break;
    }

    // calculate turbulent prandtl number of dynamic Smagorinsky model
    case ScaTra::Action::calc_turbulent_prandtl_number:
    {
      if (nsd_ == 3)
      {
        // get required quantities, set in dynamic Smagorinsky class
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_vel =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("col_filtered_vel");
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_dens_vel =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("col_filtered_dens_vel");
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_dens_vel_temp =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
                "col_filtered_dens_vel_temp");
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_dens_rateofstrain_temp =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
                "col_filtered_dens_rateofstrain_temp");
        std::shared_ptr<Core::LinAlg::Vector<double>> col_filtered_temp =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_filtered_temp");
        std::shared_ptr<Core::LinAlg::Vector<double>> col_filtered_dens =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_filtered_dens");
        std::shared_ptr<Core::LinAlg::Vector<double>> col_filtered_dens_temp =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_filtered_dens_temp");

        // initialize variables to calculate
        double LkMk = 0.0;
        double MkMk = 0.0;
        double xcenter = 0.0;
        double ycenter = 0.0;
        double zcenter = 0.0;

        // calculate LkMk and MkMk
        switch (distype)
        {
          case Core::FE::CellType::hex8:
          {
            scatra_calc_smag_const_lk_mk_and_mk_mk(*col_filtered_vel, *col_filtered_dens_vel,
                *col_filtered_dens_vel_temp, *col_filtered_dens_rateofstrain_temp,
                *col_filtered_temp, *col_filtered_dens, *col_filtered_dens_temp, LkMk, MkMk,
                xcenter, ycenter, zcenter, ele);
            break;
          }
          default:
          {
            FOUR_C_THROW("Unknown element type for box filter application\n");
          }
        }

        // set Prt without averaging (only clipping)
        // calculate inverse of turbulent Prandtl number times (C_s*Delta)^2
        double inv_Prt;
        if (abs(MkMk) < 1E-16)
        {
          // std::cout << "warning: abs(MkMk) < 1E-16 -> set inverse of turbulent Prandtl number to
          // zero!"  << std::endl;
          inv_Prt = 0.0;
        }
        else
        {
          inv_Prt = LkMk / MkMk;
        }
        if (inv_Prt < 0.0)
        {
          inv_Prt = 0.0;
        }

        // set all values in parameter list
        params.set<double>("LkMk", LkMk);
        params.set<double>("MkMk", MkMk);
        params.set<double>("xcenter", xcenter);
        params.set<double>("ycenter", ycenter);
        params.set<double>("zcenter", zcenter);
        params.set<double>("ele_Prt", inv_Prt);
      }
      else
      {
        FOUR_C_THROW("action 'calc_turbulent_prandtl_number' is a 3D specific action");
      }

      break;
    }

    case ScaTra::Action::calc_vreman_scatra:
    {
      if (nsd_ == 3)
      {
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_phi =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("col_filtered_phi");
        std::shared_ptr<Core::LinAlg::Vector<double>> col_filtered_phi2 =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_filtered_phi2");
        std::shared_ptr<Core::LinAlg::Vector<double>> col_filtered_phiexpression =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_filtered_phiexpression");
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_alphaijsc =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
                "col_filtered_alphaijsc");

        // initialize variables to calculate
        double dt_numerator = 0.0;
        double dt_denominator = 0.0;

        // calculate LkMk and MkMk
        switch (distype)
        {
          case Core::FE::CellType::hex8:
          {
            scatra_calc_vreman_dt(*col_filtered_phi, *col_filtered_phi2,
                *col_filtered_phiexpression, *col_filtered_alphaijsc, dt_numerator, dt_denominator,
                ele);
            break;
          }
          default:
          {
            FOUR_C_THROW("Unknown element type for vreman scatra application\n");
          }
        }

        elevec1_epetra(0) = dt_numerator;
        elevec1_epetra(1) = dt_denominator;
      }
      else
      {
        FOUR_C_THROW("action 'calc_vreman_scatra' is a 3D specific action");
      }


      break;
    }

    // calculate domain integral, i.e., surface area or volume of domain element
    case ScaTra::Action::calc_domain_integral:
    {
      calc_domain_integral(ele, elevec1_epetra);

      break;
    }

    // calculate normalized subgrid-diffusivity matrix
    case ScaTra::Action::calc_subgrid_diffusivity_matrix:
    {
      // calculate mass matrix and rhs
      calc_subgr_diff_matrix(ele, elemat1_epetra);

      break;
    }

    // calculate mean Cai of multifractal subgrid-scale modeling approach
    case ScaTra::Action::calc_mean_Cai:
    {
      // get number of dofset associated with velocity related dofs
      const int ndsvel = scatrapara_->nds_vel();

      // get convective (velocity - mesh displacement) velocity at nodes
      std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
          discretization.get_state(ndsvel, "convective velocity field");
      if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

      // determine number of velocity related dofs per node
      const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

      // construct location vector for velocity related dofs
      std::vector<int> lmvel(nsd_ * nen_, -1);
      for (unsigned inode = 0; inode < nen_; ++inode)
        for (unsigned idim = 0; idim < nsd_; ++idim)
          lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

      // extract local values of convective velocity field from global state vector
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*convel, econvelnp_, lmvel);

      // rotate the vector field in the case of rotationally symmetric boundary conditions
      rotsymmpbc_->rotate_my_values_if_necessary(econvelnp_);

      // get phi for material parameters
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp_, lm);

      if (turbparams_->turb_model() != Inpar::FLUID::multifractal_subgrid_scales)
        FOUR_C_THROW("Multifractal_Subgrid_Scales expected");

      double Cai = 0.0;
      double vol = 0.0;
      // calculate Cai and volume, do not include elements of potential inflow section
      if (turbparams_->adapt_csgs_phi() and turbparams_->nwl() and
          (not ScaTra::inflow_element(ele)))
      {
        // use one-point Gauss rule to do calculations at the element center
        Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
            ScaTra::DisTypeToStabGaussRule<distype>::rule);
        vol = eval_shape_func_and_derivs_at_int_point(intpoints, 0);

        // adopt integration points and weights for gauss point evaluation of B
        if (turbparams_->bd_gp())
        {
          const Core::FE::IntPointsAndWeights<nsd_ele_> gauss_intpoints(
              ScaTra::DisTypeToOptGaussRule<distype>::rule);
          intpoints = gauss_intpoints;
        }

        for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
        {
          const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

          // density at t_(n)
          std::vector<double> densn(numscal_, 1.0);
          // density at t_(n+1) or t_(n+alpha_F)
          std::vector<double> densnp(numscal_, 1.0);
          // density at t_(n+alpha_M)
          std::vector<double> densam(numscal_, 1.0);

          // diffusivity / diffusivities (in case of systems) or (thermal conductivity/specific
          // heat) in case of loma

          diffmanager_ = std::make_shared<ScaTraEleDiffManager>(numscal_);

          // fluid viscosity
          double visc(0.0);

          // set internal variables
          set_internal_variables_for_mat_and_rhs();

          // get material
          get_material_params(ele, densn, densnp, densam, visc);

          // get velocity at integration point
          Core::LinAlg::Matrix<nsd_, 1> convelint(Core::LinAlg::Initialization::zero);
          convelint.multiply(econvelnp_, funct_);

          // calculate characteristic element length
          double hk = calc_ref_length(vol, convelint);

          // estimate norm of strain rate
          double strainnorm = get_strain_rate(econvelnp_);
          strainnorm /= sqrt(2.0);

          // get Re from strain rate
          double Re_ele_str = strainnorm * hk * hk * densnp[0] / visc;
          if (Re_ele_str < 0.0) FOUR_C_THROW("Something went wrong!");
          // ensure positive values
          if (Re_ele_str < 1.0) Re_ele_str = 1.0;

          // calculate corrected Cai
          //           -3/16
          //  =(1 - (Re)   )
          //
          Cai += (1.0 - pow(Re_ele_str, -3.0 / 16.0)) * fac;
        }
      }

      // hand down the Cai and volume contribution to the time integration algorithm
      params.set<double>("Cai_int", Cai);
      params.set<double>("ele_vol", vol);

      break;
    }

    // calculate dissipation introduced by stabilization and turbulence models
    case ScaTra::Action::calc_dissipation:
    {
      calc_dissipation(params, ele, discretization, la);
      break;
    }

    case ScaTra::Action::calc_mass_center_smoothingfunct:
    {
      double interface_thickness = params.get<double>("INTERFACE_THICKNESS_TPF");

      if (numscal_ > 1)
      {
        std::cout << "#############################################################################"
                     "##############################\n";
        std::cout << "#                                                 WARNING:                   "
                     "                             #\n";
        std::cout << "# More scalars than the levelset are transported. Mass center calculations "
                     "have NOT been tested for this. #\n";
        std::cout << "#                                                                            "
                     "                             # \n";
        std::cout << "#############################################################################"
                     "##############################\n";
        std::cout << std::endl;
      }
      // NOTE: add integral values only for elements which are NOT ghosted!
      if (ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
      {
        // need current scalar vector
        // -> extract local values from the global vectors
        std::shared_ptr<const Core::LinAlg::Vector<double>> phinp =
            discretization.get_state("phinp");
        if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
        Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp_, lm);

        // calculate momentum vector and volume for element.
        calculate_momentum_and_volume(ele, elevec1_epetra, interface_thickness);
      }
      break;
    }

    case ScaTra::Action::calc_error:
    {
      // check if length suffices
      if (elevec1_epetra.length() < 1) FOUR_C_THROW("Result vector too short");

      // need current solution
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp_, lm);

      cal_error_compared_to_analyt_solution(ele, params, elevec1_epetra);

      break;
    }

    case ScaTra::Action::calc_immersed_element_source:
    {
      int scalartoprovidwithsource = 0;
      double segregationconst = params.get<double>("segregation_constant");

      // assembly
      for (unsigned inode = 0; inode < nen_; inode++)
      {
        const int fvi = inode * numdofpernode_ + scalartoprovidwithsource;
        elevec1_epetra[fvi] += segregationconst;
      }

      break;
    }

    case ScaTra::Action::micro_scale_initialize:
    {
      if (ele->material()->material_type() == Core::Materials::m_scatra_multiscale)
      {
        const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
            ScaTra::DisTypeToOptGaussRule<distype>::rule);

        // loop over all Gauss points
        for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
        {
          // initialize micro scale in multi-scale simulations
          std::static_pointer_cast<Mat::ScatraMultiScale>(ele->material())
              ->initialize(ele->id(), iquad, scatrapara_->is_ale());
        }
      }

      break;
    }

    case ScaTra::Action::micro_scale_prepare_time_step:
    case ScaTra::Action::micro_scale_solve:
    {
      if (ele->material()->material_type() == Core::Materials::m_scatra_multiscale)
      {
        // extract state variables at element nodes
        Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(
            *discretization.get_state("phinp"), ephinp_, lm);

        const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
            ScaTra::DisTypeToOptGaussRule<distype>::rule);

        // loop over all Gauss points
        for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
        {
          // evaluate shape functions at Gauss point
          eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

          // evaluate state variables at Gauss point
          set_internal_variables_for_mat_and_rhs();

          if (action == ScaTra::Action::micro_scale_prepare_time_step)
          {
            // prepare time step on micro scale
            std::static_pointer_cast<Mat::ScatraMultiScale>(ele->material())
                ->prepare_time_step(iquad, std::vector<double>(1, scatravarmanager_->phinp(0)));
          }
          else
          {
            const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
                ScaTra::DisTypeToOptGaussRule<distype>::rule);

            const double detF = eval_det_f_at_int_point(ele, intpoints, iquad);

            // solve micro scale
            std::vector<double> dummy(1, 0.);
            std::static_pointer_cast<Mat::ScatraMultiScale>(ele->material())
                ->evaluate(iquad, std::vector<double>(1, scatravarmanager_->phinp(0)), dummy[0],
                    dummy, detF);
          }
        }
      }

      break;
    }

    case ScaTra::Action::micro_scale_update:
    {
      if (ele->material()->material_type() == Core::Materials::m_scatra_multiscale)
      {
        const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
            ScaTra::DisTypeToOptGaussRule<distype>::rule);

        // loop over all Gauss points
        for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
          // update multi-scale scalar transport material
          std::static_pointer_cast<Mat::ScatraMultiScale>(ele->material())->update(iquad);
      }

      break;
    }

    case ScaTra::Action::micro_scale_output:
    {
      if (ele->material()->material_type() == Core::Materials::m_scatra_multiscale)
      {
        const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
            ScaTra::DisTypeToOptGaussRule<distype>::rule);

        // loop over all Gauss points
        for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
          // create output on micro scale
          std::static_pointer_cast<Mat::ScatraMultiScale>(ele->material())->output(iquad);
      }

      break;
    }

    case ScaTra::Action::collect_micro_scale_output:
    {
      if (ele->material()->material_type() == Core::Materials::m_scatra_multiscale)
      {
        const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
            ScaTra::DisTypeToOptGaussRule<distype>::rule);

        // loop over all Gauss points
        for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
        {
          // create output on micro scale
          std::static_pointer_cast<Mat::ScatraMultiScale>(ele->material())
              ->collect_output_data(iquad);
        }
      }

      break;
    }

    case ScaTra::Action::micro_scale_read_restart:
    {
      if (ele->material()->material_type() == Core::Materials::m_scatra_multiscale)
      {
        const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
            ScaTra::DisTypeToOptGaussRule<distype>::rule);

        // loop over all Gauss points
        for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
          // read restart on micro scale
          std::dynamic_pointer_cast<Mat::ScatraMultiScale>(ele->material())->read_restart(iquad);
      }

      break;
    }

    case ScaTra::Action::micro_scale_set_time:
    {
      if (ele->material()->material_type() == Core::Materials::m_scatra_multiscale)
      {
        const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
            ScaTra::DisTypeToOptGaussRule<distype>::rule);

        // loop over all Gauss points
        for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
        {
          std::dynamic_pointer_cast<Mat::ScatraMultiScale>(ele->material())
              ->set_time_stepping(iquad, params.get<double>("dt"), params.get<double>("time"),
                  params.get<int>("step"));
        }
      }

      break;
    }

    case ScaTra::Action::calc_heteroreac_mat_and_rhs:
    {
      //--------------------------------------------------------------------------------
      // extract element based or nodal values
      //--------------------------------------------------------------------------------
      extract_element_and_node_values(ele, params, discretization, la);

      for (int idof = 0; idof < numdofpernode_; idof++)
      {
        // no bodyforce
        bodyforce_[idof].clear();
      }

      calc_hetero_reac_mat_and_rhs(ele, elemat1_epetra, elevec1_epetra);

      break;
    }

    case ScaTra::Action::transform_real_to_reference_point:
    {
      // init quantities
      Core::LinAlg::Matrix<nsd_, 1> x_real;
      for (unsigned int d = 0; d < nsd_; ++d) x_real(d, 0) = params.get<double*>("point")[d];
      xsi_(0, 0) = 0.0;
      for (unsigned int d = 1; d < nsd_; ++d) xsi_(d, 0) = 0.0;
      int count = 0;
      Core::LinAlg::Matrix<nsd_, 1> diff;

      // do the Newton loop
      bool inside = true;
      do
      {
        count++;
        eval_shape_func_and_derivs_in_parameter_space();
        Core::LinAlg::Matrix<nsd_, 1> x_eval;
        for (unsigned int d = 0; d < nsd_; ++d)
        {
          for (unsigned int n = 0; n < nen_; ++n)
          {
            x_eval(d, 0) += funct_(n, 0) * xyze_(d, n);
          }
          x_eval(d, 0) -= x_real(d, 0);
        }
        diff.multiply_tn(xij_, x_eval);

        for (unsigned int d = 0; d < nsd_; ++d)
        {
          xsi_(d, 0) -= diff(d, 0);
          if (xsi_(d, 0) > 10.0 || xsi_(d, 0) < -10.0)
          {
            inside = false;
          }
        }
      } while (count < 20 && diff.norm1() > 1.0e-10 && inside);

      inside = true;
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        if (xsi_(d, 0) > 1.0 || xsi_(d, 0) < -1.0)
        {
          inside = false;
        }
      }

      std::array<double, nsd_> pointarr;
      if (!inside)
      {
        for (unsigned int d = 0; d < nsd_; ++d)
        {
          pointarr[d] = -123.0;
        }
      }
      else
      {
        for (unsigned int d = 0; d < nsd_; ++d)
        {
          pointarr[d] = xsi_(d, 0);
        }
      }

      params.set<std::array<double, nsd_>>("point", pointarr);

      params.set<bool>("inside", inside);

      break;
    }

    case ScaTra::Action::evaluate_field_in_point:
    {
      for (unsigned int d = 0; d < nsd_; ++d) xsi_(d, 0) = params.get<double*>("point")[d];

      eval_shape_func_and_derivs_in_parameter_space();

      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp_, lm);

      if (params.get<int>("NUMSCAL") > numscal_)
      {
        FOUR_C_THROW(
            "you requested the pointvalue of the {}-th scalar but there is only {} scalars",
            params.get<int>("NUMSCAL"), numscal_);
      }

      const double value = funct_.dot(ephinp_[params.get<int>("NUMSCAL")]);

      params.set<double>("value", value);

      break;
    }

    default:
    {
      FOUR_C_THROW("Not acting on this action. Forgot implementation?");
    }
  }  // switch(action)

  return 0;
}


/*----------------------------------------------------------------------*
 | evaluate service routine                                  fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalc<distype, probdim>::evaluate_service(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // setup
  if (setup_calc(ele, discretization) == -1) return 0;

  // check for the action parameter
  const auto action = Teuchos::getIntegralValue<ScaTra::Action>(params, "action");

  if (scatrapara_->is_ale() and action != ScaTra::Action::micro_scale_read_restart)
  {
    // get number of dofset associated with displacement related dofs
    const int ndsdisp = scatrapara_->nds_disp();

    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
        discretization.get_state(ndsdisp, "dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

    // determine number of displacement related dofs per node
    const int numdispdofpernode = la[ndsdisp].lm_.size() / nen_;

    // construct location vector for displacement related dofs
    std::vector<int> lmdisp(nsd_ * nen_, -1);
    for (unsigned inode = 0; inode < nen_; ++inode)
      for (unsigned idim = 0; idim < nsd_; ++idim)
        lmdisp[inode * nsd_ + idim] = la[ndsdisp].lm_[inode * numdispdofpernode + idim];

    // extract local values of displacement field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*dispnp, edispnp_, lmdisp);

    // add nodal displacements to point coordinates
    update_node_coordinates();
  }
  else
  {
    edispnp_.clear();
  }

  // evaluate action
  evaluate_action(ele, params, discretization, action, la, elemat1_epetra, elemat2_epetra,
      elevec1_epetra, elevec2_epetra, elevec3_epetra);

  return 0;
}

/*---------------------------------------------------------------------*
 | calculate filtered fields for turbulent Prandtl number   fang 02/15 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_box_filter(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  // extract scalar values from global vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> scalar = discretization.get_state("scalar");
  if (scalar == nullptr) FOUR_C_THROW("Cannot get scalar!");
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*scalar, ephinp_, la[0].lm_);

  // get number of dofset associated with velocity related dofs
  const int ndsvel = scatrapara_->nds_vel();

  // get convective (velocity - mesh displacement) velocity at nodes
  std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
      discretization.get_state(ndsvel, "convective velocity field");
  if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

  // determine number of velocity related dofs per node
  const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

  // construct location vector for velocity related dofs
  std::vector<int> lmvel(nsd_ * nen_, -1);
  for (unsigned inode = 0; inode < nen_; ++inode)
    for (unsigned idim = 0; idim < nsd_; ++idim)
      lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

  // extract local values of convective velocity field from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*convel, evelnp_, lmvel);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  rotsymmpbc_->rotate_my_values_if_necessary(evelnp_);

  // initialize the contribution of this element to the patch volume to zero
  double volume_contribution = 0.0;
  // initialize the contributions of this element to the filtered scalar quantities
  double dens_hat = 0.0;
  double temp_hat = 0.0;
  double dens_temp_hat = 0.0;
  double phi2_hat = 0.0;
  double phiexpression_hat = 0.0;
  // get pointers for vector quantities
  std::shared_ptr<std::vector<double>> vel_hat =
      params.get<std::shared_ptr<std::vector<double>>>("vel_hat");
  std::shared_ptr<std::vector<double>> densvel_hat =
      params.get<std::shared_ptr<std::vector<double>>>("densvel_hat");
  std::shared_ptr<std::vector<double>> densveltemp_hat =
      params.get<std::shared_ptr<std::vector<double>>>("densveltemp_hat");
  std::shared_ptr<std::vector<double>> densstraintemp_hat =
      params.get<std::shared_ptr<std::vector<double>>>("densstraintemp_hat");
  std::shared_ptr<std::vector<double>> phi_hat =
      params.get<std::shared_ptr<std::vector<double>>>("phi_hat");
  std::shared_ptr<std::vector<std::vector<double>>> alphaijsc_hat =
      params.get<std::shared_ptr<std::vector<std::vector<double>>>>("alphaijsc_hat");
  // integrate the convolution with the box filter function for this element
  // the results are assembled onto the *_hat arrays
  switch (distype)
  {
    case Core::FE::CellType::hex8:
    {
      scatra_apply_box_filter(dens_hat, temp_hat, dens_temp_hat, phi2_hat, phiexpression_hat,
          *vel_hat, *densvel_hat, *densveltemp_hat, *densstraintemp_hat, *phi_hat, *alphaijsc_hat,
          volume_contribution, ele, params);

      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown element type for box filter application\n");
    }
  }

  // hand down the volume contribution to the time integration algorithm
  params.set<double>("volume_contribution", volume_contribution);
  // as well as the filtered scalar quantities
  params.set<double>("dens_hat", dens_hat);
  params.set<double>("temp_hat", temp_hat);
  params.set<double>("dens_temp_hat", dens_temp_hat);
  params.set<double>("phi2_hat", phi2_hat);
  params.set<double>("phiexpression_hat", phiexpression_hat);
}  // Discret::Elements::ScaTraEleCalc<distype,probdim>::calc_box_filter


/*-----------------------------------------------------------------------------*
 | calculate mass matrix + rhs for initial time derivative calc.     gjb 03/12 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_initial_time_derivative(
    Core::Elements::Element* ele,              //!< current element
    Core::LinAlg::SerialDenseMatrix& emat,     //!< element matrix
    Core::LinAlg::SerialDenseVector& erhs,     //!< element residual
    Teuchos::ParameterList& params,            //!< parameter list
    Core::FE::Discretization& discretization,  //!< discretization
    Core::Elements::LocationArray& la          //!< location array
)
{
  // extract relevant quantities from discretization and parameter list
  extract_element_and_node_values(ele, params, discretization, la);

  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------
  // use one-point Gauss rule to do calculations at the element center
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints_tau(
      ScaTra::DisTypeToStabGaussRule<distype>::rule);

  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double vol = eval_shape_func_and_derivs_at_int_point(intpoints_tau, 0);

  //------------------------------------------------------------------------------------
  // get material parameters and stabilization parameters (evaluation at element center)
  //------------------------------------------------------------------------------------
  // density at t_(n)
  std::vector<double> densn(numscal_, 1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  std::vector<double> densnp(numscal_, 1.0);
  // density at t_(n+alpha_M)
  std::vector<double> densam(numscal_, 1.0);

  // fluid viscosity
  double visc(0.0);

  // the stabilisation parameters (one per transported scalar)
  std::vector<double> tau(numscal_, 0.0);

  if (not scatrapara_->mat_gp() or not scatrapara_->tau_gp())
  {
    set_internal_variables_for_mat_and_rhs();

    get_material_params(ele, densn, densnp, densam, visc);

    if (not scatrapara_->tau_gp())
    {
      for (int k = 0; k < numscal_; ++k)  // loop of each transported scalar
      {
        // calculation of stabilization parameter at element center
        calc_tau(tau[k], diffmanager_->get_isotropic_diff(k),
            reamanager_->get_stabilization_coeff(k, scatravarmanager_->phinp(k)), densnp[k],
            scatravarmanager_->con_vel(k), vol);
      }
    }
  }

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  /*----------------------------------------------------------------------*/
  // element integration loop
  /*----------------------------------------------------------------------*/
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    set_internal_variables_for_mat_and_rhs();

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    if (scatrapara_->mat_gp()) get_material_params(ele, densn, densnp, densam, visc, iquad);

    //------------ get values of variables at integration point
    for (int k = 0; k < numscal_; ++k)  // deal with a system of transported scalars
    {
      // get phi at integration point for all scalars
      const double& phiint = scatravarmanager_->phinp(k);

      // convective part in convective form: rho*u_x*N,x+ rho*u_y*N,y
      Core::LinAlg::Matrix<nen_, 1> conv = scatravarmanager_->conv(k);

      // velocity divergence required for conservative form
      double vdiv(0.0);
      if (scatrapara_->is_conservative()) get_divergence(vdiv, evelnp_);

      // diffusive part used in stabilization terms
      Core::LinAlg::Matrix<nen_, 1> diff(Core::LinAlg::Initialization::zero);
      // diffusive term using current scalar value for higher-order elements
      if (use2ndderiv_)
      {
        // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
        get_laplacian_strong_form(diff);
        diff.scale(diffmanager_->get_isotropic_diff(k));
      }

      // calculation of stabilization parameter at integration point
      if (scatrapara_->tau_gp())
      {
        calc_tau(tau[k], diffmanager_->get_isotropic_diff(k),
            reamanager_->get_stabilization_coeff(k, scatravarmanager_->phinp(k)), densnp[k],
            scatravarmanager_->con_vel(k), vol);
      }

      const double fac_tau = fac * tau[k];

      //----------------------------------------------------------------
      // element matrix: transient term
      //----------------------------------------------------------------
      // transient term
      calc_mat_mass(emat, k, fac, densam[k]);

      //----------------------------------------------------------------
      // element matrix: stabilization of transient term
      //----------------------------------------------------------------
      // the stabilization term is deactivated in calc_initial_time_derivative() on time integrator
      // level
      if (scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
      {
        // subgrid-scale velocity (dummy)
        Core::LinAlg::Matrix<nen_, 1> sgconv(Core::LinAlg::Initialization::zero);
        calc_mat_mass_stab(emat, k, fac_tau, densam[k], densnp[k], sgconv, diff);

        // remove convective stabilization of inertia term
        for (unsigned vi = 0; vi < nen_; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;
          erhs(fvi) += fac_tau * densnp[k] * conv(vi) * densnp[k] * phiint;
        }
      }

      // we solve: (w,dc/dt) = rhs
      // whereas the rhs is based on the standard element evaluation routine
      // including contributions resulting from the time discretization.
      // The contribution from the time discretization has to be removed before solving the system:
      correct_rhs_from_calc_rhs_lin_mass(erhs, k, fac, densnp[k], phiint);
    }  // loop over each scalar k
  }  // integration loop

  // scale element matrix appropriately to be consistent with scaling of global residual vector
  // computed by assemble_mat_and_rhs() routine (see calc_initial_time_derivative() routine on time
  // integrator level)
  emat.scale(scatraparatimint_->time_fac_rhs());
}  // ScaTraEleCalc::calc_initial_time_derivative()


/*----------------------------------------------------------------------*
 |  correct_rhs_from_calc_rhs_lin_mass                             ehrl 06/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::correct_rhs_from_calc_rhs_lin_mass(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double fac, const double densnp,
    const double phinp)
{
  // fac->-fac to change sign of rhs
  if (scatraparatimint_->is_incremental())
    calc_rhs_lin_mass(erhs, k, 0.0, -fac, 0.0, densnp);
  else
    FOUR_C_THROW("Must be incremental!");
}


/*----------------------------------------------------------------------*
 |  Integrate shape functions over domain (private)           gjb 07/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::integrate_shape_functions(
    const Core::Elements::Element* ele, Core::LinAlg::SerialDenseVector& elevec1,
    const Core::LinAlg::IntSerialDenseVector& dofids)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // safety check
  if (dofids.numRows() < numdofpernode_)
    FOUR_C_THROW("Dofids vector is too short. Received not enough flags");

  // loop over integration points
  // this order is not efficient since the integration of the shape functions is always the same for
  // all species
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, gpid);

    // compute integral of shape functions (only for dofid)
    for (int k = 0; k < numdofpernode_; k++)
    {
      if (dofids[k] >= 0)
      {
        for (unsigned node = 0; node < nen_; node++)
        {
          elevec1[node * numdofpernode_ + k] += funct_(node) * fac;
        }
      }
    }

  }  // loop over integration points
}  // ScaTraEleCalc::integrate_shape_function

/*----------------------------------------------------------------------*
  |  calculate weighted mass flux (no reactive flux so far)     gjb 06/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calculate_flux(
    Core::LinAlg::Matrix<3, nen_>& flux, const Core::Elements::Element* ele,
    const Inpar::ScaTra::FluxType fluxtype, const int k)
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
  std::vector<double> densn(numscal_, 1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  std::vector<double> densnp(numscal_, 1.0);
  // density at t_(n+alpha_M)
  std::vector<double> densam(numscal_, 1.0);

  // fluid viscosity
  double visc(0.0);

  // get material parameters (evaluation at element center)
  if (not scatrapara_->mat_gp())
  {
    set_internal_variables_for_mat_and_rhs();

    get_material_params(ele, densn, densnp, densam, visc);
  }

  // integration rule
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    set_internal_variables_for_mat_and_rhs();

    // get material parameters (evaluation at integration point)
    if (scatrapara_->mat_gp()) get_material_params(ele, densn, densnp, densam, visc);

    // get velocity at integration point
    Core::LinAlg::Matrix<nsd_, 1> velint(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd_, 1> convelint(Core::LinAlg::Initialization::zero);
    velint.multiply(evelnp_, funct_);
    convelint.multiply(econvelnp_, funct_);

    // get gradient of scalar at integration point
    Core::LinAlg::Matrix<nsd_, 1> gradphi(Core::LinAlg::Initialization::zero);
    gradphi.multiply(derxy_, ephinp_[k]);

    // allocate and initialize!
    Core::LinAlg::Matrix<nsd_, 1> q(Core::LinAlg::Initialization::zero);

    // add different flux contributions as specified by user input
    switch (fluxtype)
    {
      case Inpar::ScaTra::flux_total:
        // convective flux contribution
        q.update(densnp[k] * scatravarmanager_->phinp(k), convelint);

        [[fallthrough]];
      case Inpar::ScaTra::flux_diffusive:
        // diffusive flux contribution
        q.update(-(diffmanager_->get_isotropic_diff(k)), gradphi, 1.0);

        break;
      default:
        FOUR_C_THROW("received illegal flag inside flux evaluation for whole domain");
    }
    // q at integration point

    // integrate and assemble everything into the "flux" vector
    for (unsigned vi = 0; vi < nen_; vi++)
    {
      for (unsigned idim = 0; idim < nsd_; idim++)
      {
        flux(idim, vi) += fac * funct_(vi) * q(idim);
      }  // idim
    }  // vi

  }  // integration loop

  // set zeros for unused space dimensions
  for (unsigned idim = nsd_; idim < 3; idim++)
  {
    for (unsigned vi = 0; vi < nen_; vi++)
    {
      flux(idim, vi) = 0.0;
    }
  }
}  // ScaTraCalc::calculate_flux


/*----------------------------------------------------------------------------------------*
 | calculate domain integral, i.e., surface area or volume of domain element   fang 07/15 |
 *----------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_domain_integral(
    const Core::Elements::Element* ele,      //!< the element we are dealing with
    Core::LinAlg::SerialDenseVector& scalar  //!< result vector for scalar integral to be computed
)
{
  // initialize variable for domain integral
  double domainintegral(0.);

  // get integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // add contribution from current integration point to domain integral
    for (unsigned vi = 0; vi < nen_; ++vi) domainintegral += funct_(vi) * fac;
  }  // loop over integration points

  // write result into result vector
  scalar(0) = domainintegral;
}  // Discret::Elements::ScaTraEleCalc<distype,probdim>::calc_domain_integral


/*----------------------------------------------------------------------*
|  calculate scalar(s) and domain integral                     vg 09/08|
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calculate_scalars(
    const Core::Elements::Element* ele, Core::LinAlg::SerialDenseVector& scalars,
    const bool inverting, const bool calc_grad_phi)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate everything on current GP
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    for (int k = 0; k < numdofpernode_; k++)
    {
      // evaluate 1.0/phi if needed
      Core::LinAlg::Matrix<nen_, 1> inv_ephinp(Core::LinAlg::Initialization::zero);
      if (inverting)
      {
        for (unsigned i = 0; i < nen_; i++)
        {
          const double inv_value = 1.0 / ephinp_[k](i);
          if (std::abs(inv_value) < 1e-14) FOUR_C_THROW("Division by zero");
          inv_ephinp(i) = inv_value;
        }
      }

      // project phi or 1.0/phi to current GP and multiply with domain integration factor
      const double phi_gp = funct_.dot(inverting ? inv_ephinp : ephinp_[k]);
      scalars[k] += phi_gp * fac;
    }
    scalars[numdofpernode_] += fac;

    if (calc_grad_phi)
    {
      Discret::Elements::ScaTraEleCalc<distype, probdim>::set_internal_variables_for_mat_and_rhs();

      for (int k = 0; k < numscal_; k++)
      {
        const double gradphi_l2norm_gp = scatravarmanager_->grad_phi()[k].norm2();
        scalars[numdofpernode_ + 1 + k] += gradphi_l2norm_gp * fac;
      }
    }
  }
}


/*----------------------------------------------------------------------*
 | calculate scalar time derivative(s) and domain integral   fang 03/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calculate_scalar_time_derivatives(
    const Core::FE::Discretization& discretization,  //!< discretization
    const std::vector<int>& lm,                      //!< location vector
    Core::LinAlg::SerialDenseVector& scalars  //!< result vector for scalar integrals to be computed
)
{
  // extract scalar time derivatives from global state vector
  const std::shared_ptr<const Core::LinAlg::Vector<double>> phidtnp =
      discretization.get_state("phidtnp");
  if (phidtnp == nullptr) FOUR_C_THROW("Cannot get state vector \"phidtnp\"!");
  static std::vector<Core::LinAlg::Matrix<nen_, 1>> ephidtnp(numscal_);
  Core::FE::extract_my_values(*phidtnp, ephidtnp, lm);

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // calculate integrals of scalar time derivatives
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const double fac_funct_vi = fac * funct_(vi);

      for (int k = 0; k < numscal_; ++k) scalars(k) += fac_funct_vi * ephidtnp[k](vi);
    }

    // calculate integral of domain
    scalars(numscal_) += fac;
  }  // loop over integration points
}  // Discret::Elements::ScaTraEleCalc<distype,probdim>::calculate_scalar_time_derivatives


/*----------------------------------------------------------------------*
|  calculate momentum vector and minus domain integral          mw 06/14|
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calculate_momentum_and_volume(
    const Core::Elements::Element* ele, Core::LinAlg::SerialDenseVector& momandvol,
    const double interface_thickness)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // coordinates of the current integration point.
    std::vector<double> gpcoord(nsd_);
    // levelset function at gaussian point.
    double ephi_gp = 0.0;
    // fac*funct at GP
    double fac_funct = 0.0;

    for (unsigned i = 0; i < nen_; i++)
    {
      // Levelset function (first scalar stored [0]) at gauss point
      ephi_gp += funct_(i) * ephinp_[0](i, 0);

      // Coordinate * shapefunction to get the coordinate value of the gausspoint.
      for (unsigned idim = 0; idim < nsd_; idim++)
      {
        gpcoord[idim] += funct_(i) * (ele->nodes()[i]->x()[idim]);
      }

      // Summation of fac*funct_ for volume calculation.
      fac_funct += fac * funct_(i);
    }

    double heavyside_epsilon = 1.0;  // plus side

    // Smoothing function
    if (abs(ephi_gp) <= interface_thickness)
    {
      heavyside_epsilon = 0.5 * (1.0 + ephi_gp / interface_thickness +
                                    1.0 / M_PI * sin(M_PI * ephi_gp / interface_thickness));
    }
    else if (ephi_gp < interface_thickness)
    {
      heavyside_epsilon = 0.0;  // minus side
    }


    // add momentum vector and volume
    for (unsigned idim = 0; idim < nsd_; idim++)
    {
      momandvol(idim) += gpcoord[idim] * (1.0 - heavyside_epsilon) * fac_funct;
    }

    momandvol(nsd_) += fac_funct * (1.0 - heavyside_epsilon);

  }  // loop over integration points
}  // ScaTraEleCalc::calculate_momentum_and_volume


/*----------------------------------------------------------------------*
 | calculate normalized subgrid-diffusivity matrix              vg 10/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_subgr_diff_matrix(
    const Core::Elements::Element* ele, Core::LinAlg::SerialDenseMatrix& emat)
{
  /*----------------------------------------------------------------------*/
  // integration loop for one element
  /*----------------------------------------------------------------------*/
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    for (int k = 0; k < numscal_; ++k)
    {
      // set diffusion coeff to 1.0
      diffmanager_->set_isotropic_diff(1.0, k);

      // calculation of diffusive element matrix
      double timefacfac = scatraparatimint_->time_fac() * fac;
      calc_mat_diff(emat, k, timefacfac);

      /*subtract SUPG term */
      // emat(fvi,fui) -= taufac*conv(vi)*conv(ui);
    }
  }  // integration loop
}  // ScaTraImpl::calc_subgr_diff_matrix


/*----------------------------------------------------------------------------------------*
 | finite difference check on element level (for debugging only) (protected)   fang 10/14 |
 *----------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::fd_check(Core::Elements::Element* ele,
    Core::LinAlg::SerialDenseMatrix& emat, Core::LinAlg::SerialDenseVector& erhs,
    Core::LinAlg::SerialDenseVector& subgrdiff)
{
  // screen output
  std::cout << "FINITE DIFFERENCE CHECK FOR ELEMENT " << ele->id();

  // make a copy of state variables to undo perturbations later
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ephinp_original(numscal_);
  for (int k = 0; k < numscal_; ++k)
    for (unsigned i = 0; i < nen_; ++i) ephinp_original[k](i, 0) = ephinp_[k](i, 0);

  // generalized-alpha time integration requires a copy of history variables as well
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ehist_original(numscal_);
  if (scatraparatimint_->is_gen_alpha())
  {
    for (int k = 0; k < numscal_; ++k)
      for (unsigned i = 0; i < nen_; ++i) ehist_original[k](i, 0) = ehist_[k](i, 0);
  }

  // initialize element matrix and vectors for perturbed state
  Core::LinAlg::SerialDenseMatrix emat_dummy(emat);
  Core::LinAlg::SerialDenseVector erhs_perturbed(erhs);
  Core::LinAlg::SerialDenseVector subgrdiff_dummy(subgrdiff);

  // initialize counter for failed finite difference checks
  unsigned counter(0);

  // initialize tracking variable for maximum absolute and relative errors
  double maxabserr(0.);
  double maxrelerr(0.);

  // loop over columns of element matrix by first looping over nodes and then over dofs at each node
  for (unsigned inode = 0; inode < nen_; ++inode)
  {
    for (int idof = 0; idof < numdofpernode_; ++idof)
    {
      // number of current column of element matrix
      unsigned col = inode * numdofpernode_ + idof;

      // clear element matrix and vectors for perturbed state
      emat_dummy.putScalar(0.0);
      erhs_perturbed.putScalar(0.0);
      subgrdiff_dummy.putScalar(0.0);

      // fill state vectors with original state variables
      for (int k = 0; k < numscal_; ++k)
        for (unsigned i = 0; i < nen_; ++i) ephinp_[k](i, 0) = ephinp_original[k](i, 0);
      if (scatraparatimint_->is_gen_alpha())
        for (int k = 0; k < numscal_; ++k)
          for (unsigned i = 0; i < nen_; ++i) ehist_[k](i, 0) = ehist_original[k](i, 0);

      // impose perturbation
      if (scatraparatimint_->is_gen_alpha())
      {
        // perturbation of phi(n+alphaF), not of phi(n+1) => scale epsilon by factor alphaF
        ephinp_[idof](inode, 0) += scatraparatimint_->alpha_f() * scatrapara_->fd_check_eps();

        // perturbation of phi(n+alphaF) by alphaF*epsilon corresponds to perturbation of phidtam
        // (stored in ehist_) by alphaM*epsilon/(gamma*dt); note: alphaF/timefac = alphaM/(gamma*dt)
        ehist_[idof](inode, 0) += scatraparatimint_->alpha_f() / scatraparatimint_->time_fac() *
                                  scatrapara_->fd_check_eps();
      }
      else
      {
        ephinp_[idof](inode, 0) += scatrapara_->fd_check_eps();
      }

      // calculate element right-hand side vector for perturbed state
      sysmat(ele, emat_dummy, erhs_perturbed, subgrdiff_dummy);

      // Now we compare the difference between the current entries in the element matrix
      // and their finite difference approximations according to
      // entries ?= (-erhs_perturbed + erhs_original) / epsilon

      // Note that the element right-hand side equals the negative element residual.
      // To account for errors due to numerical cancellation, we additionally consider
      // entries - erhs_original / epsilon ?= -erhs_perturbed / epsilon

      // Note that we still need to evaluate the first comparison as well. For small entries in the
      // element matrix, the second comparison might yield good agreement in spite of the entries
      // being wrong!
      for (unsigned row = 0; row < static_cast<unsigned>(numdofpernode_ * nen_); ++row)
      {
        // get current entry in original element matrix
        const double entry = emat(row, col);

        // finite difference suggestion (first divide by epsilon and then subtract for better
        // conditioning)
        const double fdval = -erhs_perturbed(row) / scatrapara_->fd_check_eps() +
                             erhs(row) / scatrapara_->fd_check_eps();

        // confirm accuracy of first comparison
        if (abs(fdval) > 1.e-17 and abs(fdval) < 1.e-15)
          FOUR_C_THROW("Finite difference check involves values too close to numerical zero!");

        // absolute and relative errors in first comparison
        const double abserr1 = entry - fdval;
        if (abs(abserr1) > abs(maxabserr)) maxabserr = abserr1;
        double relerr1(0.);
        if (abs(entry) > 1.e-17)
          relerr1 = abserr1 / abs(entry);
        else if (abs(fdval) > 1.e-17)
          relerr1 = abserr1 / abs(fdval);
        if (abs(relerr1) > abs(maxrelerr)) maxrelerr = relerr1;

        // evaluate first comparison
        if (abs(relerr1) > scatrapara_->fd_check_tol())
        {
          if (!counter) std::cout << " --> FAILED AS FOLLOWS:\n";
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
          const double left = entry - erhs(row) / scatrapara_->fd_check_eps();

          // right-hand side in second comparison
          const double right = -erhs_perturbed(row) / scatrapara_->fd_check_eps();

          // confirm accuracy of second comparison
          if (abs(right) > 1.e-17 and abs(right) < 1.e-15)
            FOUR_C_THROW("Finite difference check involves values too close to numerical zero!");

          // absolute and relative errors in second comparison
          const double abserr2 = left - right;
          if (abs(abserr2) > abs(maxabserr)) maxabserr = abserr2;
          double relerr2(0.);
          if (abs(left) > 1.e-17)
            relerr2 = abserr2 / abs(left);
          else if (abs(right) > 1.e-17)
            relerr2 = abserr2 / abs(right);
          if (abs(relerr2) > abs(maxrelerr)) maxrelerr = relerr2;

          // evaluate second comparison
          if (abs(relerr2) > scatrapara_->fd_check_tol())
          {
            if (!counter) std::cout << " --> FAILED AS FOLLOWS:\n";
            std::cout << "emat[" << row << "," << col << "]-erhs[" << row << "]/eps:  " << left
                      << "   ";
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
  if (!counter)
    std::cout << " --> PASSED WITH MAXIMUM ABSOLUTE ERROR " << maxabserr
              << " AND MAXIMUM RELATIVE ERROR " << maxrelerr << std::endl;

  // undo perturbations of state variables
  for (int k = 0; k < numscal_; ++k)
    for (unsigned i = 0; i < nen_; ++i) ephinp_[k](i, 0) = ephinp_original[k](i, 0);
  if (scatraparatimint_->is_gen_alpha())
    for (int k = 0; k < numscal_; ++k)
      for (unsigned i = 0; i < nen_; ++i) ehist_[k](i, 0) = ehist_original[k](i, 0);
}

/*---------------------------------------------------------------------*
  |  calculate error compared to analytical solution           gjb 10/08|
  *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::cal_error_compared_to_analyt_solution(
    const Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& errors)
{
  if (Teuchos::getIntegralValue<ScaTra::Action>(params, "action") != ScaTra::Action::calc_error)
    FOUR_C_THROW("How did you get here?");

  // -------------- prepare common things first ! -----------------------
  // set constants for analytical solution
  const double t = scatraparatimint_->time();

  // integration points and weights
  // more GP than usual due to (possible) cos/exp fcts in analytical solutions
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToGaussRuleForExactSol<distype>::rule);

  const auto errortype =
      Teuchos::getIntegralValue<Inpar::ScaTra::CalcError>(params, "calcerrorflag");
  switch (errortype)
  {
    case Inpar::ScaTra::calcerror_byfunction:
    {
      const int errorfunctno = params.get<int>("error function number");

      // analytical solution
      double phi_exact(0.0);
      double deltaphi(0.0);
      //! spatial gradient of current scalar value
      Core::LinAlg::Matrix<nsd_, 1> gradphi(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, 1> gradphi_exact(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, 1> deltagradphi(Core::LinAlg::Initialization::zero);

      // start loop over integration points
      for (int iquad = 0; iquad < intpoints.ip().nquad; iquad++)
      {
        const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

        // get coordinates at integration point
        // gp reference coordinates
        Core::LinAlg::Matrix<nsd_, 1> xyzint(Core::LinAlg::Initialization::zero);
        xyzint.multiply(xyze_, funct_);

        // function evaluation requires a 3D position vector!!
        double position[3] = {0.0, 0.0, 0.0};

        for (unsigned dim = 0; dim < nsd_; ++dim) position[dim] = xyzint(dim);

        for (int k = 0; k < numdofpernode_; ++k)
        {
          // scalar at integration point at time step n+1
          const double phinp = funct_.dot(ephinp_[k]);
          // spatial gradient of current scalar value
          gradphi.multiply(derxy_, ephinp_[k]);

          phi_exact = Global::Problem::instance()
                          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(errorfunctno)
                          .evaluate(position, t, k);

          std::vector<double> gradphi_exact_vec =
              Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(errorfunctno)
                  .evaluate_spatial_derivative(position, t, k);

          if (gradphi_exact_vec.size())
          {
            if (nsd_ == nsd_ele_)
              for (unsigned dim = 0; dim < nsd_; ++dim) gradphi_exact(dim) = gradphi_exact_vec[dim];
            else
            {
              // std::cout<<"Warning: Gradient of analytical solution cannot be evaluated correctly
              // for transport on curved surfaces!"<<std::endl;
              gradphi_exact.clear();
            }
          }
          else
          {
            std::cout << "Warning: Gradient of analytical solution was not evaluated!" << std::endl;
            gradphi_exact.clear();
          }

          // error at gauss point
          deltaphi = phinp - phi_exact;
          deltagradphi.update(1.0, gradphi, -1.0, gradphi_exact);

          // 0: delta scalar for L2-error norm
          // 1: delta scalar for H1-error norm
          // 2: analytical scalar for L2 norm
          // 3: analytical scalar for H1 norm

          // the error for the L2 and H1 norms are evaluated at the Gauss point

          // integrate delta scalar for L2-error norm
          errors(k * 4 + 0) += deltaphi * deltaphi * fac;
          // integrate delta scalar for H1-error norm
          errors(k * 4 + 1) += deltaphi * deltaphi * fac;
          // integrate analytical scalar for L2 norm
          errors(k * 4 + 2) += phi_exact * phi_exact * fac;
          // integrate analytical scalar for H1 norm
          errors(k * 4 + 3) += phi_exact * phi_exact * fac;

          // integrate delta scalar derivative for H1-error norm
          errors(k * 4 + 1) += deltagradphi.dot(deltagradphi) * fac;
          // integrate analytical scalar derivative for H1 norm
          errors(k * 4 + 3) += gradphi_exact.dot(gradphi_exact) * fac;
        }
      }  // loop over integration points
    }
    break;

    case Inpar::ScaTra::calcerror_spherediffusion:
    {
      FOUR_C_ASSERT_ALWAYS(nsd_ == 3, "Sphere diffusion only implemented for 3D!");
      // analytical solution
      double phi_exact(0.0);
      double deltaphi(0.0);
      //! spatial gradient of current scalar value
      Core::LinAlg::Matrix<nsd_, 1> gradphi(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, 1> gradphi_exact(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, 1> deltagradphi(Core::LinAlg::Initialization::zero);

      // start loop over integration points
      for (int iquad = 0; iquad < intpoints.ip().nquad; iquad++)
      {
        const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

        // get coordinates at integration point
        // gp reference coordinates
        Core::LinAlg::Matrix<nsd_, 1> xyzint(Core::LinAlg::Initialization::zero);
        xyzint.multiply(xyze_, funct_);

        for (int k = 0; k < numscal_; k++)
        {
          const double x = xyzint(0);
          const double y = xyzint(1);
          const double z = xyzint(2);

          // scalar at integration point at time step n+1
          const double phinp = funct_.dot(ephinp_[k]);
          // spatial gradient of current scalar value
          gradphi.multiply(derxy_, ephinp_[k]);

          phi_exact = exp(-6 * t) * x * y + 10;

          gradphi_exact(0) = (1.0 - 2.0 * x * x) * y * exp(-6.0 * t);
          gradphi_exact(1) = (1.0 - 2.0 * y * y) * x * exp(-6.0 * t);
          gradphi_exact(2) = -2.0 * x * y * z * exp(-6.0 * t);

          // error at gauss point
          deltaphi = phinp - phi_exact;
          deltagradphi.update(1.0, gradphi, -1.0, gradphi_exact);

          // 0: delta scalar for L2-error norm
          // 1: delta scalar for H1-error norm
          // 2: analytical scalar for L2 norm
          // 3: analytical scalar for H1 norm

          // the error for the L2 and H1 norms are evaluated at the Gauss point

          // integrate delta scalar for L2-error norm
          errors(k * numscal_ + 0) += deltaphi * deltaphi * fac;
          // integrate delta scalar for H1-error norm
          errors(k * numscal_ + 1) += deltaphi * deltaphi * fac;
          // integrate analytical scalar for L2 norm
          errors(k * numscal_ + 2) += phi_exact * phi_exact * fac;
          // integrate analytical scalar for H1 norm
          errors(k * numscal_ + 3) += phi_exact * phi_exact * fac;

          // integrate delta scalar derivative for H1-error norm
          errors(k * numscal_ + 1) += deltagradphi.dot(deltagradphi) * fac;
          // integrate analytical scalar derivative for H1 norm
          errors(k * numscal_ + 3) += gradphi_exact.dot(gradphi_exact) * fac;
        }
      }  // loop over integration points
    }
    break;
    default:
      FOUR_C_THROW("Unknown analytical solution!");
      break;
  }  // switch(errortype)
}  // Discret::Elements::ScaTraEleCalc<distype,probdim>::cal_error_compared_to_analyt_solution


/*----------------------------------------------------------------------*
|  calculate system matrix and rhs (public)                  vuong 07/16|
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_hetero_reac_mat_and_rhs(
    Core::Elements::Element* ele,           ///< the element whose matrix is calculated
    Core::LinAlg::SerialDenseMatrix& emat,  ///< element matrix to calculate
    Core::LinAlg::SerialDenseVector& erhs   ///< element rhs to calculate
)
{
  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------
  const double vol = eval_shape_func_and_derivs_at_ele_center();

  //----------------------------------------------------------------------
  // get material and stabilization parameters (evaluation at element center)
  //----------------------------------------------------------------------
  // density at t_(n)
  std::vector<double> densn(numscal_, 1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  std::vector<double> densnp(numscal_, 1.0);
  // density at t_(n+alpha_M)
  std::vector<double> densam(numscal_, 1.0);

  // fluid viscosity
  double visc(0.0);

  // the stabilization parameters (one per transported scalar)
  std::vector<double> tau(numscal_, 0.0);

  if (not scatrapara_->tau_gp())
  {
    for (int k = 0; k < numscal_; ++k)  // loop of each transported scalar
    {
      // get velocity at element center
      Core::LinAlg::Matrix<nsd_, 1> convelint = scatravarmanager_->con_vel(k);
      // calculation of stabilization parameter at element center
      calc_tau(tau[k], diffmanager_->get_isotropic_diff(k),
          reamanager_->get_stabilization_coeff(k, scatravarmanager_->phinp(k)), densnp[k],
          convelint, vol);
    }
  }

  // material parameter at the element center are also necessary
  // even if the stabilization parameter is evaluated at the element center
  if (not scatrapara_->mat_gp())
  {
    // set gauss point variables needed for evaluation of mat and rhs
    set_internal_variables_for_mat_and_rhs();

    get_material_params(ele, densn, densnp, densam, visc);
  }

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // set gauss point variables needed for evaluation of mat and rhs
    set_internal_variables_for_mat_and_rhs();

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    if (scatrapara_->mat_gp()) get_material_params(ele, densn, densnp, densam, visc, iquad);

    // loop all scalars
    for (int k = 0; k < numscal_; ++k)  // deal with a system of transported scalars
    {
      // reactive part of the form: (reaction coefficient)*phi
      double rea_phi(0.0);
      rea_phi = densnp[k] * scatravarmanager_->phinp(k) * reamanager_->get_rea_coeff(k);

      // compute rhs containing bodyforce (divided by specific heat capacity) and,
      // for temperature equation, the time derivative of thermodynamic pressure,
      // if not constant, and for temperature equation of a reactive
      // equation system, the reaction-rate term
      double rhsint(0.0);
      get_rhs_int(rhsint, densnp[k], k);

      double scatrares(0.0);
      // calculate strong residual
      calc_strong_residual(k, scatrares, densam[k], densnp[k], rea_phi, rhsint, tau[k]);

      if (scatrapara_->tau_gp())
      {
        // (re)compute stabilization parameter at integration point, since diffusion may have
        // changed
        calc_tau(tau[k], diffmanager_->get_isotropic_diff(k),
            reamanager_->get_stabilization_coeff(k, scatravarmanager_->phinp(k)), densnp[k],
            scatravarmanager_->con_vel(k), vol);  // TODO:(Thon) do we really have to do this??
      }

      //----------------------------------------------------------------
      // standard Galerkin terms
      //----------------------------------------------------------------

      // stabilization parameter and integration factors
      const double taufac = tau[k] * fac;
      const double timefacfac = scatraparatimint_->time_fac() * fac;
      const double timetaufac = scatraparatimint_->time_fac() * taufac;

      //----------------------------------------------------------------
      // 3) element matrix: reactive term
      //----------------------------------------------------------------

      Core::LinAlg::Matrix<nen_, 1> sgconv(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nen_, 1> diff(Core::LinAlg::Initialization::zero);
      // diffusive term using current scalar value for higher-order elements
      if (use2ndderiv_)
      {
        // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
        get_laplacian_strong_form(diff);
        diff.scale(diffmanager_->get_isotropic_diff(k));
      }

      // including stabilization
      if (reamanager_->active())
      {
        calc_mat_react(emat, k, timefacfac, timetaufac, taufac, densnp[k], sgconv, diff);
      }

      //----------------------------------------------------------------
      // 5) element right hand side
      //----------------------------------------------------------------
      //----------------------------------------------------------------
      // computation of bodyforce (and potentially history) term,
      // residual, integration factors and standard Galerkin transient
      // term (if required) on right hand side depending on respective
      // (non-)incremental stationary or time-integration scheme
      //----------------------------------------------------------------
      double rhsfac = scatraparatimint_->time_fac_rhs() * fac;
      double rhstaufac = scatraparatimint_->time_fac_rhs_tau() * taufac;

      compute_rhs_int(rhsint, densam[k], densnp[k], 0.0);

      recompute_scatra_res_for_rhs(scatrares, k, diff, densn[k], densnp[k], rea_phi, rhsint);

      //----------------------------------------------------------------
      // standard Galerkin transient, old part of rhs and bodyforce term
      //----------------------------------------------------------------
      calc_rhs_hist_and_source(erhs, k, fac, rhsint);

      //----------------------------------------------------------------
      // reactive terms (standard Galerkin and stabilization) on rhs
      //----------------------------------------------------------------

      if (reamanager_->active())
        calc_rhs_react(erhs, k, rhsfac, rhstaufac, rea_phi, densnp[k], scatrares);

    }  // end loop all scalars
  }  // end loop Gauss points
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleCalc<distype, probdim>::eval_det_f_at_int_point(
    const Core::Elements::Element* const& ele,
    const Core::FE::IntPointsAndWeights<nsd_ele_>& intpoints, const int iquad)
{
  // get determinant of derivative of spatial coordinate w.r.t. parameter coordinates
  const double det_dxds = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

  // get derivatives of element shape function w.r.t. parameter coordinates
  Core::LinAlg::Matrix<nsd_ele_, nen_> deriv_ele;
  Core::FE::shape_function_deriv1<distype>(xsi_, deriv_ele);

  // reference coordinates of element nodes
  Core::LinAlg::Matrix<nsd_, nen_> XYZ;
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(ele, XYZ);

  // reference coordinates of elemental nodes in space dimension of element
  Core::LinAlg::Matrix<nsd_ele_, nen_> XYZe;
  for (int i = 0; i < static_cast<int>(nsd_ele_); ++i)
    for (int j = 0; j < static_cast<int>(nen_); ++j) XYZe(i, j) = XYZ(i, j);

  // compute derivative of parameter coordinates w.r.t. reference coordinates
  Core::LinAlg::Matrix<nsd_ele_, nsd_ele_> dXds;
  dXds.multiply_nt(deriv_ele, XYZe);

  return det_dxds / dXds.determinant();
}

FOUR_C_NAMESPACE_CLOSE

// template classes

#include "4C_scatra_ele_calc.inst.hpp"
