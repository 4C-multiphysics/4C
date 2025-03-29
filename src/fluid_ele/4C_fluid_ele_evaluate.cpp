// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_utils.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_evaluate_utils.hpp"
#include "4C_fluid_ele_factory.hpp"
#include "4C_fluid_ele_interface.hpp"
#include "4C_fluid_ele_parameter.hpp"
#include "4C_fluid_ele_parameter_intface.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_fluid_ele_parameter_xfem.hpp"
#include "4C_fluid_ele_tds.hpp"
#include "4C_fluid_ele_xwall.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN


/*
  Depending on the type of action and the element type (tet, hex etc.),
  the elements allocate common static arrays.

  */

/*----------------------------------------------------------------------*
|  evaluate the element (public)                            g.bau 03/07|
*----------------------------------------------------------------------*/
int Discret::Elements::Fluid::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // get the action required
  const auto act = Teuchos::getIntegralValue<FLD::Action>(params, "action");

  // get material
  std::shared_ptr<Core::Mat::Material> mat = material();

  // get space dimensions
  const int nsd = Core::FE::get_dimension(shape());

  // Retrieve the physical type from the parameters
  auto physicalType =
      params.get<Inpar::FLUID::PhysicalType>("Physical Type", Inpar::FLUID::incompressible);
  std::string impltype;
  switch (physicalType)
  {
    case Inpar::FLUID::loma:
      impltype = "loma";
      break;
    default:
      impltype = "std";
      break;
  }


  Discret::Elements::FluidXWall* xwallele = dynamic_cast<Discret::Elements::FluidXWall*>(this);
  if (xwallele)  // not a xwall element and the node row maps don't know it's nodes
    impltype = "xw";

  switch (act)
  {
    //-----------------------------------------------------------------------
    // standard implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    //-----------------------------------------------------------------------
    case FLD::calc_fluid_systemmat_and_residual:
    {
      return Discret::Elements::FluidFactory::provide_impl(shape(), impltype)
          ->evaluate(
              this, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2, elevec3);
    }
    break;
    //-----------------------------------------------------------------------
    // standard implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    // for evaluation of off-diagonal matrix block for monolithic
    // low-Mach-number solver
    //-----------------------------------------------------------------------
    case FLD::calc_loma_mono_odblock:
    {
      return Discret::Elements::FluidFactory::provide_impl(shape(), "loma")
          ->evaluate(this, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2,
              elevec3, true);
    }
    break;
    case FLD::calc_turbscatra_statistics:
    {
      if (nsd == 3)
      {
        // do nothing if you do not own this element
        if (this->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
        {
          // --------------------------------------------------
          // extract velocity, pressure, and scalar from global
          // distributed vectors
          // --------------------------------------------------
          // velocity/pressure and scalar values (n+1)
          std::shared_ptr<const Core::LinAlg::Vector<double>> velnp =
              discretization.get_state("u and p (n+1,converged)");
          std::shared_ptr<const Core::LinAlg::Vector<double>> scanp =
              discretization.get_state("scalar (n+1,converged)");
          if (velnp == nullptr || scanp == nullptr)
            FOUR_C_THROW("Cannot get state vectors 'velnp' and/or 'scanp'");

          // extract local values from the global vectors
          std::vector<double> myvelpre = Core::FE::extract_values(*velnp, lm);
          std::vector<double> mysca = Core::FE::extract_values(*scanp, lm);

          // integrate mean values
          const Core::FE::CellType distype = this->shape();

          switch (distype)
          {
            case Core::FE::CellType::hex8:
            {
              FLD::f3_calc_scatra_means<8>(this, discretization, myvelpre, mysca, params);
              break;
            }
            case Core::FE::CellType::hex20:
            {
              FLD::f3_calc_scatra_means<20>(this, discretization, myvelpre, mysca, params);
              break;
            }
            case Core::FE::CellType::hex27:
            {
              FLD::f3_calc_scatra_means<27>(this, discretization, myvelpre, mysca, params);
              break;
            }
            default:
            {
              FOUR_C_THROW(
                  "Unknown element type for turbulent passive scalar mean value evaluation\n");
            }
          }
        }
      }  // end if (nsd == 3)
      else
        FOUR_C_THROW("action 'calc_turbscatra_statistics' is a 3D specific action");
    }
    break;
    case FLD::calc_loma_statistics:
    {
      if (nsd == 3)
      {
        // do nothing if you do not own this element
        if (this->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
        {
          // --------------------------------------------------
          // extract velocity, pressure, and temperature from
          // global distributed vectors
          // --------------------------------------------------
          // velocity/pressure and scalar values (n+1)
          std::shared_ptr<const Core::LinAlg::Vector<double>> velnp =
              discretization.get_state("u and p (n+1,converged)");
          std::shared_ptr<const Core::LinAlg::Vector<double>> scanp =
              discretization.get_state("scalar (n+1,converged)");
          if (velnp == nullptr || scanp == nullptr)
            FOUR_C_THROW("Cannot get state vectors 'velnp' and/or 'scanp'");

          // extract local values from global vectors
          std::vector<double> myvelpre(lm.size());
          std::vector<double> mysca(lm.size());
          myvelpre = Core::FE::extract_values(*velnp, lm);
          mysca = Core::FE::extract_values(*scanp, lm);

          // get factor for equation of state
          const double eosfac = params.get<double>("eos factor", 100000.0 / 287.0);

          // integrate mean values
          const Core::FE::CellType distype = this->shape();

          switch (distype)
          {
            case Core::FE::CellType::hex8:
            {
              FLD::f3_calc_loma_means<8>(this, discretization, myvelpre, mysca, params, eosfac);
              break;
            }
            case Core::FE::CellType::hex20:
            {
              FLD::f3_calc_loma_means<20>(this, discretization, myvelpre, mysca, params, eosfac);
              break;
            }
            case Core::FE::CellType::hex27:
            {
              FLD::f3_calc_loma_means<27>(this, discretization, myvelpre, mysca, params, eosfac);
              break;
            }
            default:
            {
              FOUR_C_THROW("Unknown element type for low-Mach-number mean value evaluation\n");
            }
          }
        }
      }  // end if (nsd == 3)
      else
        FOUR_C_THROW("action 'calc_loma_statistics' is a 3D specific action");
    }
    break;
    case FLD::calc_fluid_box_filter:
    {
      if (nsd == 3)
      {
        const Core::FE::CellType distype = this->shape();
        int nen = 0;
        if (distype == Core::FE::CellType::hex8)
          nen = 8;
        else if (distype == Core::FE::CellType::tet4)
          nen = 4;
        else
          FOUR_C_THROW("not supported");

        // --------------------------------------------------
        // extract velocity and pressure from global
        // distributed vectors
        // --------------------------------------------------
        // velocity, pressure and temperature values (most recent
        // intermediate solution, i.e. n+alphaF for genalpha
        // and n+1 for one-step-theta)
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel =
            discretization.get_state("u and p (trial)");
        if (vel == nullptr) FOUR_C_THROW("Cannot get state vectors 'vel'");
        // extract local values from the global vectors
        std::vector<double> myvel = Core::FE::extract_values(*vel, lm);

        std::vector<double> tmp_temp(lm.size());
        std::vector<double> mytemp(nen);
        double thermpress = 0.0;
        // pointer to class FluidEleParameter (access to the general parameter)
        Discret::Elements::FluidEleParameterStd* fldpara =
            Discret::Elements::FluidEleParameterStd::instance();
        if (fldpara->physical_type() == Inpar::FLUID::loma)
        {
          std::shared_ptr<const Core::LinAlg::Vector<double>> temp =
              discretization.get_state("T (trial)");
          if (temp == nullptr) FOUR_C_THROW("Cannot get state vectors 'temp'");
          tmp_temp = Core::FE::extract_values(*temp, lm);

          for (int i = 0; i < nen; i++) mytemp[i] = tmp_temp[nsd + (i * (nsd + 1))];

          // get thermodynamic pressure
          thermpress = params.get<double>("thermpress");
        }
        // initialize the contribution of this element to the patch volume to zero
        double volume_contribution = 0.0;

        // initialize the contributions of this element to the filtered scalar quantities
        double dens_hat = 0.0;
        double dens_strainrate_hat = 0.0;
        // get pointers for vector quantities
        std::shared_ptr<std::vector<double>> vel_hat =
            params.get<std::shared_ptr<std::vector<double>>>("vel_hat");
        std::shared_ptr<std::vector<double>> densvel_hat =
            params.get<std::shared_ptr<std::vector<double>>>("densvel_hat");
        std::shared_ptr<std::vector<std::vector<double>>> reynoldsstress_hat =
            params.get<std::shared_ptr<std::vector<std::vector<double>>>>("reynoldsstress_hat");
        std::shared_ptr<std::vector<std::vector<double>>> modeled_subgrid_stress =
            params.get<std::shared_ptr<std::vector<std::vector<double>>>>("modeled_subgrid_stress");
        // Vreman
        double expression_hat = 0.0;
        double alpha2_hat = 0.0;
        std::shared_ptr<std::vector<std::vector<double>>> strainrate_hat =
            params.get<std::shared_ptr<std::vector<std::vector<double>>>>("strainrate_hat");
        std::shared_ptr<std::vector<std::vector<double>>> alphaij_hat =
            params.get<std::shared_ptr<std::vector<std::vector<double>>>>("alphaij_hat");
        // integrate the convolution with the box filter function for this element
        // the results are assembled onto the *_hat arrays
        switch (distype)
        {
          case Core::FE::CellType::hex8:
          {
            FLD::f3_apply_box_filter<8>(this, fldpara, myvel, mytemp, thermpress, *vel_hat,
                *densvel_hat, *reynoldsstress_hat, *modeled_subgrid_stress, volume_contribution,
                dens_hat, dens_strainrate_hat, expression_hat, alpha2_hat, *strainrate_hat,
                *alphaij_hat);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            FLD::f3_apply_box_filter<4>(this, fldpara, myvel, mytemp, thermpress, *vel_hat,
                *densvel_hat, *reynoldsstress_hat, *modeled_subgrid_stress, volume_contribution,
                dens_hat, dens_strainrate_hat, expression_hat, alpha2_hat, *strainrate_hat,
                *alphaij_hat);
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
        params.set<double>("dens_strainrate_hat", dens_strainrate_hat);

        params.set<double>("expression_hat", expression_hat);
        params.set<double>("alpha2_hat", alpha2_hat);

      }  // end if (nsd == 3)
      else
        FOUR_C_THROW("action 'calc_fluid_box_filter' is 3D specific action");
    }
    break;
    case FLD::calc_smagorinsky_const:
    {
      if (nsd == 3)
      {
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_vel =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("col_filtered_vel");
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_reynoldsstress =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
                "col_filtered_reynoldsstress");
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_modeled_subgrid_stress =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
                "col_filtered_modeled_subgrid_stress");

        // pointer to class FluidEleParameter (access to the general parameter)
        Discret::Elements::FluidEleParameterStd* fldpara =
            Discret::Elements::FluidEleParameterStd::instance();
        // add potential loma specific vectors
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_dens_vel = nullptr;
        std::shared_ptr<Core::LinAlg::Vector<double>> col_filtered_dens = nullptr;
        std::shared_ptr<Core::LinAlg::Vector<double>> col_filtered_dens_strainrate = nullptr;
        if (fldpara->physical_type() == Inpar::FLUID::loma)
        {
          col_filtered_dens_vel = params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
              "col_filtered_dens_vel");
          col_filtered_dens =
              params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_filtered_dens");
          col_filtered_dens_strainrate = params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>(
              "col_filtered_dens_strainrate");
        }

        double LijMij = 0.0;
        double MijMij = 0.0;
        double CI_numerator = 0.0;
        double CI_denominator = 0.0;
        double xcenter = 0.0;
        double ycenter = 0.0;
        double zcenter = 0.0;

        const Core::FE::CellType distype = this->shape();
        switch (distype)
        {
          case Core::FE::CellType::hex8:
          {
            FLD::f3_calc_smag_const_lij_mij_and_mij_mij<8>(this, fldpara, *col_filtered_vel,
                *col_filtered_reynoldsstress, *col_filtered_modeled_subgrid_stress,
                *col_filtered_dens_vel, *col_filtered_dens, *col_filtered_dens_strainrate, LijMij,
                MijMij, CI_numerator, CI_denominator, xcenter, ycenter, zcenter);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            FLD::f3_calc_smag_const_lij_mij_and_mij_mij<4>(this, fldpara, *col_filtered_vel,
                *col_filtered_reynoldsstress, *col_filtered_modeled_subgrid_stress,
                *col_filtered_dens_vel, *col_filtered_dens, *col_filtered_dens_strainrate, LijMij,
                MijMij, CI_numerator, CI_denominator, xcenter, ycenter, zcenter);
            break;
          }
          default:
          {
            FOUR_C_THROW("Unknown element type for box filter application\n");
          }
        }

        double Cs_delta_sq = 0.0;
        double Ci_delta_sq = 0.0;
        // set Cs_delta_sq without averaging (only clipping)
        if (abs(MijMij) < 1E-16)
          Cs_delta_sq = 0.0;
        else
          Cs_delta_sq = 0.5 * LijMij / MijMij;
        if (Cs_delta_sq < 0.0)
        {
          Cs_delta_sq = 0.0;
        }

        if (fldpara->physical_type() == Inpar::FLUID::loma)
        {
          // and set Ci_delta_sq without averaging (only clipping) for loma
          if (abs(CI_denominator) < 1E-16)
            Ci_delta_sq = 0.0;
          else
            Ci_delta_sq = 0.5 * CI_numerator / CI_denominator;
          if (Ci_delta_sq < 0.0)
          {
            Ci_delta_sq = 0.0;
          }
        }

        // set all values in parameter list
        params.set<double>("ele_Cs_delta_sq", Cs_delta_sq);
        params.set<double>("ele_Ci_delta_sq", Ci_delta_sq);
        params.set<double>("LijMij", LijMij);
        params.set<double>("MijMij", MijMij);
        params.set<double>("CI_numerator", CI_numerator);
        params.set<double>("CI_denominator", CI_denominator);
        params.set<double>("xcenter", xcenter);
        params.set<double>("ycenter", ycenter);
        params.set<double>("zcenter", zcenter);
      }  // end if(nsd == 3)
      else
        FOUR_C_THROW("action 'calc_smagorinsky_const' is a 3D specific action");
    }
    break;
    case FLD::calc_vreman_const:
    {
      if (nsd == 3)
      {
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_strainrate =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
                "col_filtered_strainrate");
        std::shared_ptr<Core::LinAlg::MultiVector<double>> col_filtered_alphaij =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("col_filtered_alphaij");
        // pointer to class FluidEleParameter (access to the general parameter)
        std::shared_ptr<Core::LinAlg::Vector<double>> col_filtered_expression = nullptr;
        std::shared_ptr<Core::LinAlg::Vector<double>> col_filtered_alpha2 = nullptr;
        col_filtered_expression =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_filtered_expression");
        col_filtered_alpha2 =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_filtered_alpha2");

        double cv_numerator = 0.0;
        double cv_denominator = 0.0;
        double volume = 0.0;
        const Core::FE::CellType distype = this->shape();
        switch (distype)
        {
          case Core::FE::CellType::hex8:
          {
            FLD::f3_calc_vreman_const<8>(this, *col_filtered_strainrate, *col_filtered_alphaij,
                *col_filtered_expression, *col_filtered_alpha2, cv_numerator, cv_denominator,
                volume);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            FLD::f3_calc_vreman_const<4>(this, *col_filtered_strainrate, *col_filtered_alphaij,
                *col_filtered_expression, *col_filtered_alpha2, cv_numerator, cv_denominator,
                volume);
            break;
          }
          default:
          {
            FOUR_C_THROW("Unknown element type for dynamic vreman application\n");
          }
        }

        elevec1(0) = cv_numerator;
        elevec1(1) = cv_denominator;
      }  // end if(nsd == 3)
      else
        FOUR_C_THROW("action 'calc_vreman_const' is a 3D specific action");
    }
    break;
    case FLD::calc_fluid_genalpha_update_for_subscales:
    {
      // time update for time-dependent subgrid-scales
      const double dt = params.get<double>("dt");
      const double gamma = params.get<double>("gamma");
      this->tds()->update(dt, gamma);
    }
    break;
    case FLD::calc_model_params_mfsubgr_scales:
    {
      if (nsd == 3)
      {
        // velocity values
        std::shared_ptr<const Core::LinAlg::Vector<double>> velnp =
            discretization.get_state("velnp");
        // fine-scale velocity values
        std::shared_ptr<const Core::LinAlg::Vector<double>> fsvelnp =
            discretization.get_state("fsvelnp");
        if (velnp == nullptr or fsvelnp == nullptr)
        {
          FOUR_C_THROW("Cannot get state vectors");
        }

        // extract local values from the global vectors
        std::vector<double> myvel = Core::FE::extract_values(*velnp, lm);
        std::vector<double> myfsvel = Core::FE::extract_values(*fsvelnp, lm);

        // pointer to class FluidEleParameter (access to the general parameter)
        Discret::Elements::FluidEleParameterStd* fldpara =
            Discret::Elements::FluidEleParameterStd::instance();

        const Core::FE::CellType distype = this->shape();
        switch (distype)
        {
          case Core::FE::CellType::hex8:
          {
            // don't store values of ghosted elements
            if (this->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
            {
              FLD::f3_get_mf_params<8, 3, Core::FE::CellType::hex8>(
                  this, fldpara, params, mat, myvel, myfsvel);
            }
            break;
          }
          default:
          {
            FOUR_C_THROW("Unknown element type\n");
          }
        }
      }
      else
        FOUR_C_THROW("{} D elements does not support calculation of model parameters", nsd);
    }
    break;
    case FLD::calc_mean_Cai:
    {
      if (nsd == 3)
      {
        const Core::FE::CellType distype = this->shape();
        int nen = 0;
        if (distype == Core::FE::CellType::hex8)
          nen = 8;
        else
          FOUR_C_THROW("not supported");

        // velocity values
        // renamed to "velaf" to be consistent i fluidimplicitintegration.cpp (krank 12/13)
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel = discretization.get_state("velaf");
        // scalar values
        std::shared_ptr<const Core::LinAlg::Vector<double>> sca =
            discretization.get_state("scalar");
        if (vel == nullptr or sca == nullptr)
        {
          FOUR_C_THROW("Cannot get state vectors");
        }
        // extract local values from the global vectors
        std::vector<double> myvel = Core::FE::extract_values(*vel, lm);
        std::vector<double> tmp_sca(lm.size());
        std::vector<double> mysca(nen);
        tmp_sca = Core::FE::extract_values(*sca, lm);
        for (int i = 0; i < nen; i++) mysca[i] = tmp_sca[nsd + (i * (nsd + 1))];
        // get thermodynamic pressure
        double thermpress = params.get<double>("thermpress at n+alpha_F/n+1", 0.0);

        // pointer to class FluidEleParameter
        Discret::Elements::FluidEleParameterStd* fldpara =
            Discret::Elements::FluidEleParameterStd::instance();

        double Cai = 0.0;
        double vol = 0.0;

        bool is_inflow_ele = false;

        std::vector<Core::Conditions::Condition*> myinflowcond;

        // check whether all nodes have a unique inflow condition
        Core::Conditions::find_element_conditions(this, "TurbulentInflowSection", myinflowcond);
        if (myinflowcond.size() > 1) FOUR_C_THROW("More than one inflow condition on one node!");

        if (myinflowcond.size() == 1) is_inflow_ele = true;

        // exclude elements of inflow section
        if (not is_inflow_ele)
        {
          switch (distype)
          {
            case Core::FE::CellType::hex8:
            {
              FLD::f3_get_mf_nwc<8, 3, Core::FE::CellType::hex8>(
                  this, fldpara, Cai, vol, myvel, mysca, thermpress);
              break;
            }
            default:
            {
              FOUR_C_THROW("Unknown element type\n");
            }
          }
        }

        // hand down the Cai and volume contribution to the time integration algorithm
        params.set<double>("Cai_int", Cai);
        params.set<double>("ele_vol", vol);
      }
      else
        FOUR_C_THROW("{} D elements does not support calculation of mean Cai", nsd);
    }
    break;
    case FLD::calc_node_normal:
    {
      if (nsd == 3)
      {
        const Core::FE::CellType distype = this->shape();
        switch (distype)
        {
          case Core::FE::CellType::hex27:
          {
            FLD::element_node_normal<Core::FE::CellType::hex27>(
                this, params, discretization, lm, elevec1);
            break;
          }
          case Core::FE::CellType::hex20:
          {
            FLD::element_node_normal<Core::FE::CellType::hex20>(
                this, params, discretization, lm, elevec1);
            break;
          }
          case Core::FE::CellType::hex8:
          {
            FLD::element_node_normal<Core::FE::CellType::hex8>(
                this, params, discretization, lm, elevec1);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            FLD::element_node_normal<Core::FE::CellType::tet4>(
                this, params, discretization, lm, elevec1);
            break;
          }
          case Core::FE::CellType::tet10:
          {
            FLD::element_node_normal<Core::FE::CellType::tet10>(
                this, params, discretization, lm, elevec1);
            break;
          }
          default:
          {
            FOUR_C_THROW("Unknown element type for shape function integration\n");
          }
        }
      }
      else
        FOUR_C_THROW(
            "action 'calculate node normal' should also work in 2D, but 2D elements are not"
            " added to the template yet. Also it is not tested");
      break;
    }
    case FLD::calc_div_u:
    case FLD::calc_mass_matrix:
    case FLD::calc_fluid_error:
    case FLD::calc_dissipation:
    case FLD::integrate_shape:
    case FLD::calc_divop:
    case FLD::calc_turbulence_statistics:
    case FLD::xwall_l2_projection:
    case FLD::xwall_calc_mk:
    case FLD::tauw_via_gradient:
    case FLD::velgradient_projection:
    case FLD::presgradient_projection:
    case FLD::calc_velgrad_ele_center:
    case FLD::calc_dt_via_cfl:
    case FLD::calc_mass_flow_periodic_hill:
    {
      return Discret::Elements::FluidFactory::provide_impl(shape(), impltype)
          ->evaluate_service(
              this, params, mat, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }
    default:
      FOUR_C_THROW("Unknown type of action '{}' for Fluid", act);
      break;
  }  // end of switch(act)

  return 0;
}  // end of Discret::Elements::Fluid::Evaluate


/*----------------------------------------------------------------------*
 |  do nothing (public)                                      gammi 04/07|
 |                                                                      |
 |  The function is just a dummy. For fluid elements, the integration   |
 |  integration of volume Neumann conditions (body forces) takes place  |
 |  in the element. We need it there for the stabilisation terms!       |
 *----------------------------------------------------------------------*/
int Discret::Elements::Fluid::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  return 0;
}

FOUR_C_NAMESPACE_CLOSE
