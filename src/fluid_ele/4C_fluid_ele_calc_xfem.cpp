// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_calc_xfem.hpp"

#include "4C_cut_boundarycell.hpp"
#include "4C_cut_position.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_parameter_xfem.hpp"
#include "4C_fluid_functions.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_par_bundle.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <fstream>
#include <memory>

FOUR_C_NAMESPACE_OPEN

template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcXFEM<distype>*
Discret::Elements::FluidEleCalcXFEM<distype>::instance(Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Discret::Elements::FluidEleCalcXFEM<distype>>(
            new Discret::Elements::FluidEleCalcXFEM<distype>());
      });

  return singleton_owner.instance(action);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcXFEM<distype>::FluidEleCalcXFEM()
    : Discret::Elements::FluidEleCalc<distype>::FluidEleCalc(),
      densaf_master_(0.0),
      densaf_slave_(0.0),
      viscaf_master_(0.0),
      viscaf_slave_(0.0),
      gamma_m_(0.0),
      gamma_s_(0.0),
      evelaf_(Core::LinAlg::Initialization::zero),
      epreaf_(Core::LinAlg::Initialization::zero),
      eveln_(Core::LinAlg::Initialization::zero),
      epren_(Core::LinAlg::Initialization::zero),
      ivelint_jump_(Core::LinAlg::Initialization::zero),
      itraction_jump_(Core::LinAlg::Initialization::zero),
      proj_tangential_(Core::LinAlg::Initialization::zero),
      lb_proj_matrix_(Core::LinAlg::Initialization::zero),
      ivelintn_jump_(Core::LinAlg::Initialization::zero),
      itractionn_jump_(Core::LinAlg::Initialization::zero),
      velint_s_(Core::LinAlg::Initialization::zero),
      velintn_s_(Core::LinAlg::Initialization::zero),
      rst_(Core::LinAlg::Initialization::zero),
      normal_(Core::LinAlg::Initialization::zero),
      x_side_(Core::LinAlg::Initialization::zero),
      x_gp_lin_(Core::LinAlg::Initialization::zero)
{
  // we use the standard parameter list here, since there are not any additional
  // xfem-specific parameters required in this derived class
  my::fldpara_ = Discret::Elements::FluidEleParameterXFEM::instance();
  fldparaxfem_ = static_cast<Discret::Elements::FluidEleParameterXFEM*>(my::fldpara_);
}


namespace Discret
{
  namespace Elements
  {
    /*-------------------------------------------------------------------------------*
              Evaluate routine for cut elements of XFEM  (public)
    *-------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    int FluidEleCalcXFEM<distype>::evaluate_xfem(Discret::Elements::Fluid* ele,
        Core::FE::Discretization& discretization, const std::vector<int>& lm,
        Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
        Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
        Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
        Core::LinAlg::SerialDenseVector& elevec1_epetra,
        Core::LinAlg::SerialDenseVector& elevec2_epetra,
        Core::LinAlg::SerialDenseVector& elevec3_epetra,
        const std::vector<Core::FE::GaussIntegration>& intpoints,
        const Cut::plain_volumecell_set& cells, bool offdiag)
    {
      int err = 0;

      for (std::vector<Core::FE::GaussIntegration>::const_iterator i = intpoints.begin();
          i != intpoints.end(); ++i)
      {
        const Core::FE::GaussIntegration intpoints_cell = *i;
        err = my::evaluate(ele, discretization, lm, params, mat, elemat1_epetra, elemat2_epetra,
            elevec1_epetra, elevec2_epetra, elevec3_epetra, intpoints_cell, offdiag);
        if (err) return err;
      }
      return err;
    }

    /*-------------------------------------------------------------------------------*
              Evaluate routine for cut elements of XFEM  (public)
    *-------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    int FluidEleCalcXFEM<distype>::integrate_shape_function_xfem(Discret::Elements::Fluid* ele,
        Core::FE::Discretization& discretization, const std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& elevec1_epetra,
        const std::vector<Core::FE::GaussIntegration>& intpoints,
        const Cut::plain_volumecell_set& cells)
    {
      int err = 0;

      for (std::vector<Core::FE::GaussIntegration>::const_iterator i = intpoints.begin();
          i != intpoints.end(); ++i)
      {
        const Core::FE::GaussIntegration gint = *i;
        err = my::integrate_shape_function(ele, discretization, lm, elevec1_epetra, gint);
        if (err) return err;
      }

      return err;
    }


    /// error computation
    template <Core::FE::CellType distype>
    int FluidEleCalcXFEM<distype>::compute_error(Discret::Elements::Fluid* ele,
        Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
        Core::FE::Discretization& discretization, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& ele_dom_norms)
    {
      // integrations points and weights
      // more GP than usual due to (possible) cos/exp fcts in analytical solutions
      // degree 5
      const Core::FE::GaussIntegration intpoints(distype, 8);
      return compute_error(ele, params, mat, discretization, lm, ele_dom_norms, intpoints);
    }

    template <Core::FE::CellType distype>
    int FluidEleCalcXFEM<distype>::compute_error(Discret::Elements::Fluid* ele,
        Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
        Core::FE::Discretization& discretization, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& ele_dom_norms,  // squared element domain norms
        const Core::FE::GaussIntegration& intpoints)
    {
      // analytical solution
      Core::LinAlg::Matrix<nsd_, 1> u_analyt(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, nsd_> grad_u_analyt(Core::LinAlg::Initialization::zero);
      double p_analyt = 0.0;

      // error
      Core::LinAlg::Matrix<nsd_, 1> u_err(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, nsd_> grad_u_err(Core::LinAlg::Initialization::zero);
      double p_err = 0.0;

      const auto calcerr =
          Teuchos::getIntegralValue<Inpar::FLUID::CalcError>(params, "calculate error");
      const int calcerrfunctno = params.get<int>("error function number");

      const double t = my::fldparatimint_->time();

      // set element id
      my::eid_ = ele->id();


      //----------------------------------------------------------------------------
      //   Extract velocity/pressure from global vectors
      //----------------------------------------------------------------------------

      // fill the local element vector/matrix with the global values
      Core::LinAlg::Matrix<nsd_, nen_> evelaf(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nen_, 1> epreaf(Core::LinAlg::Initialization::zero);
      this->extract_values_from_global_vector(discretization, lm, *my::rotsymmpbc_, &evelaf,
          &epreaf, "u and p at time n+1 (converged)");

      //----------------------------------------------------------------------------
      //                         ELEMENT GEOMETRY
      //----------------------------------------------------------------------------

      // get node coordinates
      Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
          ele, my::xyze_);

      //----------------------------------------------------------------
      // Now do the nurbs specific stuff (for isogeometric elements)
      //----------------------------------------------------------------
      if (my::isNurbs_)
      {
        FOUR_C_THROW("compute error not implemented for nurbs");
      }  // Nurbs specific stuff

      if (ele->is_ale())
      {
        Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
        this->extract_values_from_global_vector(
            discretization, lm, *my::rotsymmpbc_, &edispnp, nullptr, "dispnp");

        // get new node positions for isale
        my::xyze_ += edispnp;
      }

      //------------------------------------------------------------------
      //                       INTEGRATION LOOP
      //------------------------------------------------------------------

      for (Core::FE::GaussIntegration::iterator iquad = intpoints.begin(); iquad != intpoints.end();
          ++iquad)
      {
        // evaluate shape functions and derivatives at integration point
        my::eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

        // get velocity at integration point
        // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
        my::velint_.multiply(evelaf, my::funct_);

        // get velocity derivatives at integration point
        // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
        my::vderxy_.multiply_nt(evelaf, my::derxy_);

        // get pressure at integration point
        // (value at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
        double preint = my::funct_.dot(epreaf);

        // get coordinates at integration point
        Core::LinAlg::Matrix<nsd_, 1> xyzint(Core::LinAlg::Initialization::zero);
        xyzint.multiply(my::xyze_, my::funct_);

        // get viscosity
        if (mat->material_type() == Core::Materials::m_fluid)
        {
          const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(mat.get());

          // get constant kinematic viscosity
          my::visc_ = actmat->viscosity() / actmat->density();
        }
        else
          FOUR_C_THROW("Material is not Newtonian Fluid");

        analytical_reference(calcerr,  ///< which reference solution
            calcerrfunctno,            ///< error function number
            u_analyt,                  ///< exact velocity
            grad_u_analyt,             ///< exact velocity gradient
            p_analyt,                  ///< exact pressure
            xyzint,                    ///< xyz position of gaussian point
            t, mat);

        // compute difference between analytical solution and numerical solution
        p_err = preint - p_analyt;
        u_err.update(1.0, my::velint_, -1.0, u_analyt, 0.0);
        grad_u_err.update(1.0, my::vderxy_, -1.0, grad_u_analyt, 0.0);

        // error on pre-defined functional
        // here G=sin(x)( u,x - u,x exact )
        double funcerr = (sin(xyzint(0, 0)) * (grad_u_analyt(0, 0) - my::vderxy_(0, 0))) * my::fac_;

        // standard domain errors
        // 1.   || u - u_h ||_L2(Omega)              =   standard L2-norm for velocity
        // 2.   || grad( u - u_h ) ||_L2(Omega)      =   standard H1-seminorm for velocity
        // 3.   || u - u_h ||_H1(Omega)              =   standard H1-norm for velocity
        //                                           =   sqrt( || u - u_h ||^2_L2(Omega) + || grad(
        //                                           u - u_h ) ||^2_L2(Omega) )
        // 4.   || p - p_h ||_L2(Omega)              =   standard L2-norm for pressure
        //
        // viscosity-scaled domain errors
        // 5.   || nu^(+1/2) grad( u - u_h ) ||_L2(Omega)      =   visc-scaled H1-seminorm for
        // velocity
        //                                                     =   nu^(+1/2) * || grad( u - u_h )
        //                                                     ||_L2(Omega) (for homogeneous visc)
        // 6.   || nu^(-1/2) (p - p_h) ||_L2(Omega)            =   visc-scaled L2-norm for pressure
        //                                                     =   nu^(-1/2) * || p - p_h
        //                                                     ||_L2(Omega) (for homogeneous visc)
        // 7.   || sigma^(+1/2) ( u - u_h ) ||_L2(Omega)       =   sigma-scaled L2-norm for velocity
        //                                                     =   sigma^(+1/2) * || u - u_h
        //                                                     ||_L2(Omega) (for homogeneous sigma)
        // 8.   || Phi^(+1/2) (p - p_h) ||_L2(Omega)           =   Phi-scaled L2-norm for pressure
        //                                                     =   Phi^(+1/2) * || p - p_h
        //                                                     ||_L2(Omega) (for homogeneous Phi)
        // with Phi^{-1} = sigma*CP^2 + |beta|*CP + nu + (|beta|*CP/sqrt(sigma*CP^2 + nu))^2, see
        // Massing,Schott,Wall Oseen paper
        //
        // 9. functional G=sin(x)( u,x - u,x exact ) (Sudhakar)


        double u_err_squared = 0.0;
        double grad_u_err_squared = 0.0;
        double p_err_squared = 0.0;

        // evaluate squared errors at gaussian point
        for (int isd = 0; isd < nsd_; isd++)
        {
          u_err_squared += u_err(isd) * u_err(isd) * my::fac_;

          for (int jsd = 0; jsd < nsd_; jsd++)
          {
            grad_u_err_squared += grad_u_err(isd, jsd) * grad_u_err(isd, jsd) * my::fac_;
          }
        }

        p_err_squared = p_err * p_err * my::fac_;

        double Poincare_const = 1.0;  // scales as upper bound for mesh size
        double beta_maximum = 1.0;    // maximal advective velocity in domain
        double sigma =
            1. / my::fldparatimint_
                     ->time_fac();  // sigma scaling in Oseen results from time discretization

        double Phi_tmp = beta_maximum * Poincare_const /
                         sqrt(sigma * Poincare_const * Poincare_const + my::visc_);
        double Phi_inverse_squared = sigma * Poincare_const * Poincare_const +
                                     beta_maximum * Poincare_const + my::visc_ + Phi_tmp * Phi_tmp;

        // standard domain errors
        ele_dom_norms[0] += u_err_squared;
        ele_dom_norms[1] += grad_u_err_squared;
        ele_dom_norms[2] += u_err_squared + grad_u_err_squared;
        ele_dom_norms[3] += p_err_squared;

        // viscosity-scaled domain errors
        ele_dom_norms[4] += my::visc_ * grad_u_err_squared;
        ele_dom_norms[5] += 1.0 / my::visc_ * p_err_squared;
        ele_dom_norms[6] += sigma * u_err_squared;
        ele_dom_norms[7] += 1.0 / Phi_inverse_squared * p_err_squared;

        // error for predefined functional
        ele_dom_norms[8] += funcerr;


      }  // loop gaussian points

      return 0;
    }

    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::analytical_reference(
        const int calcerr,                         ///< which reference solution
        const int calcerrfunctno,                  ///< error function number
        Core::LinAlg::Matrix<nsd_, 1>& u,          ///< exact jump vector (coupled)
        Core::LinAlg::Matrix<nsd_, nsd_>& grad_u,  ///< exact velocity gradient
        double& p,                                 ///< exact pressure
        Core::LinAlg::Matrix<nsd_, 1>& xyzint,     ///< xyz position of gaussian point
        const double& t,                           ///< time
        std::shared_ptr<Core::Mat::Material> mat)
    {
      // Compute analytical solution
      switch (calcerr)
      {
        case Inpar::FLUID::beltrami_stat_stokes:
        case Inpar::FLUID::beltrami_stat_navier_stokes:
        case Inpar::FLUID::beltrami_instat_stokes:
        case Inpar::FLUID::beltrami_instat_navier_stokes:
        {
          // function evaluation requires a 3D position vector!!
          double position[3];

          if (nsd_ == 3)
          {
            position[0] = xyzint(0);
            position[1] = xyzint(1);
            position[2] = xyzint(2);
          }
          else
            FOUR_C_THROW("invalid nsd {}", nsd_);

          // evaluate velocity and pressure
          std::shared_ptr<Core::Utils::FunctionOfSpaceTime> function = nullptr;

          // evaluate the velocity gradient
          std::shared_ptr<Core::Utils::FunctionOfSpaceTime> function_grad = nullptr;

          // get material
          Core::Mat::PAR::Parameter* params = mat->parameter();
          auto* fparams = dynamic_cast<Mat::PAR::NewtonianFluid*>(params);

          if (!fparams) FOUR_C_THROW("Material does not cast to Newtonian fluid");

          // evaluate velocity and pressure
          // evaluate the velocity gradient
          function = std::make_shared<FLD::BeltramiUP>(*fparams);
          function_grad = std::make_shared<FLD::BeltramiGradU>(*fparams);

          if (nsd_ == 3)
          {
            u(0) = function->evaluate(position, t, 0);
            u(1) = function->evaluate(position, t, 1);
            u(2) = function->evaluate(position, t, 2);
            p = function->evaluate(position, t, 3);
          }
          else
            FOUR_C_THROW("case 'kimmoin_stat' is a 3D specific case");


          if (nsd_ == 3)
          {
            grad_u(0, 0) = function_grad->evaluate(position, t, 0);  // u,x
            grad_u(0, 1) = function_grad->evaluate(position, t, 1);  // u,y
            grad_u(0, 2) = function_grad->evaluate(position, t, 2);  // u,z

            grad_u(1, 0) = function_grad->evaluate(position, t, 3);  // v,x
            grad_u(1, 1) = function_grad->evaluate(position, t, 4);  // v,y
            grad_u(1, 2) = function_grad->evaluate(position, t, 5);  // v,z

            grad_u(2, 0) = function_grad->evaluate(position, t, 6);  // w,x
            grad_u(2, 1) = function_grad->evaluate(position, t, 7);  // w,y
            grad_u(2, 2) = function_grad->evaluate(position, t, 8);  // w,z
          }
          else
            FOUR_C_THROW("case 'kimmoin_stat' is a 3D specific case");
        }
        break;

        case Inpar::FLUID::beltrami_flow:
        {
          if (nsd_ == 3)
          {
            const double a = M_PI / 4.0;
            const double d = M_PI / 2.0;

            double x = xyzint(0);
            double y = xyzint(1);
            double z = xyzint(2);

            double visc = my::visc_;

            // calculation of velocities and pressure
            u(0) = -a * (exp(a * x) * sin(a * y + d * z) + exp(a * z) * cos(a * x + d * y)) *
                   exp(-visc * d * d * t);
            u(1) = -a * (exp(a * y) * sin(a * z + d * x) + exp(a * x) * cos(a * y + d * z)) *
                   exp(-visc * d * d * t);
            u(2) = -a * (exp(a * z) * sin(a * x + d * y) + exp(a * y) * cos(a * z + d * x)) *
                   exp(-visc * d * d * t);
            p = -a * a / 2.0 *
                (exp(2.0 * a * x) + exp(2.0 * a * y) + exp(2.0 * a * z) +
                    2.0 * sin(a * x + d * y) * cos(a * z + d * x) * exp(a * (y + z)) +
                    2.0 * sin(a * y + d * z) * cos(a * x + d * y) * exp(a * (z + x)) +
                    2.0 * sin(a * z + d * x) * cos(a * y + d * z) * exp(a * (x + y))) *
                exp(-2.0 * visc * d * d * t);

            // velocity gradients
            grad_u(0, 0) =
                -a * (a * exp(a * x) * sin(a * y + d * z) - a * exp(a * z) * sin(a * x + d * y)) *
                exp(-visc * d * d * t);  // u,x
            grad_u(0, 1) =
                -a * (a * exp(a * x) * cos(a * y + d * z) - d * exp(a * z) * sin(a * x + d * y)) *
                exp(-visc * d * d * t);  // u,y
            grad_u(0, 2) =
                -a * (d * exp(a * x) * cos(a * y + d * z) + a * exp(a * z) * cos(a * x + d * y)) *
                exp(-visc * d * d * t);  // u,z

            grad_u(1, 0) =
                -a * (d * exp(a * y) * cos(a * z + d * x) + a * exp(a * x) * cos(a * y + d * z)) *
                exp(-visc * d * d * t);  // v,x
            grad_u(1, 1) =
                -a * (a * exp(a * y) * sin(a * z + d * x) - a * exp(a * x) * sin(a * y + d * z)) *
                exp(-visc * d * d * t);  // v,y
            grad_u(1, 2) =
                -a * (a * exp(a * y) * cos(a * z + d * x) - d * exp(a * x) * sin(a * y + d * z)) *
                exp(-visc * d * d * t);  // v,z

            grad_u(2, 0) =
                -a * (a * exp(a * z) * cos(a * x + d * y) - d * exp(a * y) * sin(a * z + d * x)) *
                exp(-visc * d * d * t);  // w,x
            grad_u(2, 1) =
                -a * (d * exp(a * z) * cos(a * x + d * y) + a * exp(a * y) * cos(a * z + d * x)) *
                exp(-visc * d * d * t);  // w,y
            grad_u(2, 2) =
                -a * (a * exp(a * z) * sin(a * x + d * y) - a * exp(a * y) * sin(a * z + d * x)) *
                exp(-visc * d * d * t);  // w,z
          }
          else
            FOUR_C_THROW("action 'calc_fluid_beltrami_error' is a 3D specific action");
        }
        break;

        case Inpar::FLUID::kimmoin_stat_stokes:
        case Inpar::FLUID::kimmoin_stat_navier_stokes:
        case Inpar::FLUID::kimmoin_instat_stokes:
        case Inpar::FLUID::kimmoin_instat_navier_stokes:
        {
          // function evaluation requires a 3D position vector!!
          double position[3];

          if (nsd_ == 3)
          {
            position[0] = xyzint(0);
            position[1] = xyzint(1);
            position[2] = xyzint(2);
          }
          else
            FOUR_C_THROW("invalid nsd {}", nsd_);

          // evaluate velocity and pressure
          std::shared_ptr<Core::Utils::FunctionOfSpaceTime> function = nullptr;

          // evaluate the velocity gradient
          std::shared_ptr<Core::Utils::FunctionOfSpaceTime> function_grad = nullptr;

          bool is_stationary = false;

          // evaluate velocity and pressure
          // evaluate the velocity gradient
          if (calcerr == Inpar::FLUID::kimmoin_stat_stokes or
              calcerr == Inpar::FLUID::kimmoin_stat_navier_stokes)
          {
            is_stationary = true;
          }
          else if (calcerr == Inpar::FLUID::kimmoin_instat_stokes or
                   calcerr == Inpar::FLUID::kimmoin_instat_navier_stokes)
          {
            is_stationary = false;
          }

          // get material
          Core::Mat::PAR::Parameter* params = mat->parameter();
          auto* fparams = dynamic_cast<Mat::PAR::NewtonianFluid*>(params);
          if (!fparams) FOUR_C_THROW("Material does not cast to Newtonian fluid");

          function = std::make_shared<FLD::KimMoinUP>(*fparams, is_stationary);
          function_grad = std::make_shared<FLD::KimMoinGradU>(*fparams, is_stationary);

          if (nsd_ == 3)
          {
            u(0) = function->evaluate(position, t, 0);
            u(1) = function->evaluate(position, t, 1);
            u(2) = function->evaluate(position, t, 2);
            p = function->evaluate(position, t, 3);
          }
          else
            FOUR_C_THROW("case 'kimmoin_stat' is a 3D specific case");


          if (nsd_ == 3)
          {
            grad_u(0, 0) = function_grad->evaluate(position, t, 0);  // u,x
            grad_u(0, 1) = function_grad->evaluate(position, t, 1);  // u,y
            grad_u(0, 2) = function_grad->evaluate(position, t, 2);  // u,z

            grad_u(1, 0) = function_grad->evaluate(position, t, 3);  // v,x
            grad_u(1, 1) = function_grad->evaluate(position, t, 4);  // v,y
            grad_u(1, 2) = function_grad->evaluate(position, t, 5);  // v,z

            grad_u(2, 0) = function_grad->evaluate(position, t, 6);  // w,x
            grad_u(2, 1) = function_grad->evaluate(position, t, 7);  // w,y
            grad_u(2, 2) = function_grad->evaluate(position, t, 8);  // w,z
          }
          else
            FOUR_C_THROW("case 'kimmoin_stat' is a 3D specific case");
        }
        break;

        case Inpar::FLUID::shear_flow:
        {
          const double maxvel = 1.0;
          const double height = 1.0;

          // y=0 is located in the middle of the domain
          if (nsd_ == 2)
          {
            p = 1.0;
            u(0) = xyzint(1) * maxvel + height / 2 * maxvel;
            u(1) = 0.0;
          }
          if (nsd_ == 3)
          {
            p = 0.0;
            u(0) = xyzint(1) * maxvel + height / 2 * maxvel;
            u(1) = 0.0;
            u(2) = 0.0;
          }
        }
        break;

        case Inpar::FLUID::gravitation:
        {
          const double gravity = 10.0;
          const double height = 1.0;

          // 2D: rectangle 1.0x1.0
          // 3D: cube 1.0x1.0x1.0
          // y=0 is located in the middle of the domain
          if (nsd_ == 2)
          {
            p = -xyzint(1) * gravity + height / 2 * gravity;
            u(0) = 0.0;
            u(1) = 0.0;
          }
          if (nsd_ == 3)
          {
            p = -xyzint(1) * gravity + height / 2 * gravity;
            u(0) = 0.0;
            u(1) = 0.0;
            u(2) = 0.0;
          }
        }
        break;

        case Inpar::FLUID::channel2D:
        {
          const double maxvel = 1.25;
          const double height = 1.0;
          const double visc = 1.0;
          const double pressure_gradient = 10.0;

          // u_max = 1.25
          // y=0 is located in the middle of the channel
          if (nsd_ == 2)
          {
            p = 1.0;
            // p = -10*xyzint(0)+20;
            u(0) = maxvel - ((height * height) / (2.0 * visc) * pressure_gradient *
                                (xyzint(1) / height) * (xyzint(1) / height));
            u(1) = 0.0;
          }
          else
            FOUR_C_THROW("3D analytical solution is not implemented yet");
        }
        break;

        case Inpar::FLUID::byfunct:
        {
          // function evaluation requires a 3D position vector!!
          double position[3];

          if (nsd_ == 2)
          {
            position[0] = xyzint(0);
            position[1] = xyzint(1);
            position[2] = 0.0;
          }
          else if (nsd_ == 3)
          {
            position[0] = xyzint(0);
            position[1] = xyzint(1);
            position[2] = xyzint(2);
          }
          else
            FOUR_C_THROW("invalid nsd {}", nsd_);

          if (nsd_ == 2)
          {
            const double u_exact_x =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate(position, t, 0);
            const double u_exact_y =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate(position, t, 1);
            const double p_exact =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate(position, t, 2);

            u(0) = u_exact_x;
            u(1) = u_exact_y;
            p = p_exact;


            std::vector<double> uder_exact_x =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate_spatial_derivative(position, t, 0);
            std::vector<double> uder_exact_y =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate_spatial_derivative(position, t, 1);

            if (uder_exact_x.size())
            {
              grad_u(0, 0) = uder_exact_x[0];
              grad_u(0, 1) = uder_exact_x[1];
            }

            if (uder_exact_y.size())
            {
              grad_u(1, 0) = uder_exact_y[0];
              grad_u(1, 1) = uder_exact_y[1];
            }
          }
          else if (nsd_ == 3)
          {
            const double u_exact_x =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate(position, t, 0);
            const double u_exact_y =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate(position, t, 1);
            const double u_exact_z =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate(position, t, 2);
            const double p_exact =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate(position, t, 3);

            u(0) = u_exact_x;
            u(1) = u_exact_y;
            u(2) = u_exact_z;
            p = p_exact;

            std::vector<double> uder_exact_x =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate_spatial_derivative(position, t, 0);
            std::vector<double> uder_exact_y =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate_spatial_derivative(position, t, 1);
            std::vector<double> uder_exact_z =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                    .evaluate_spatial_derivative(position, t, 2);

            if (uder_exact_x.size())
            {
              grad_u(0, 0) = uder_exact_x[0];
              grad_u(0, 1) = uder_exact_x[1];
              grad_u(0, 2) = uder_exact_x[2];
            }

            if (uder_exact_y.size())
            {
              grad_u(1, 0) = uder_exact_y[0];
              grad_u(1, 1) = uder_exact_y[1];
              grad_u(1, 2) = uder_exact_y[2];
            }

            if (uder_exact_z.size())
            {
              grad_u(2, 0) = uder_exact_z[0];
              grad_u(2, 1) = uder_exact_z[1];
              grad_u(2, 2) = uder_exact_z[2];
            }

            //      u(0) = 5.0+30.0*position[1];
            //      u(1) = 0.0;
            //      u(2) = 0.0;
            //
            //      p = 4.0;
            //
            //      grad_u(0,0) = 0.0;       grad_u(0,1) = 30.0;       grad_u(0,2) = 0.0;
            //      grad_u(1,0) = 0.0;       grad_u(1,1) =  0.0;       grad_u(1,2) = 0.0;
            //      grad_u(2,0) = 0.0;       grad_u(2,1) =  0.0;       grad_u(2,2) = 0.0;
          }
          else
            FOUR_C_THROW("invalid dimension");
        }
        break;

        default:
          FOUR_C_THROW("analytical solution is not defined");
          break;
      }
    }


    /*--------------------------------------------------------------------------------
     * compute interface error norms
     *--------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    int FluidEleCalcXFEM<distype>::compute_error_interface(
        Discret::Elements::Fluid* ele,                                ///< fluid element
        Core::FE::Discretization& dis,                                ///< background discretization
        const std::vector<int>& lm,                                   ///< element local map
        const std::shared_ptr<XFEM::ConditionManager>& cond_manager,  ///< XFEM condition manager
        std::shared_ptr<Core::Mat::Material>& mat,                    ///< material
        Core::LinAlg::SerialDenseVector& ele_interf_norms,  /// squared element interface norms
        const std::map<int, std::vector<Cut::BoundaryCell*>>& bcells,  ///< boundary cells
        const std::map<int, std::vector<Core::FE::GaussIntegration>>&
            bintpoints,                          ///< boundary integration points
        const Cut::plain_volumecell_set& vcSet,  ///< set of plain volume cells
        Teuchos::ParameterList& params           ///< parameter list
    )
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (cond_manager == nullptr) FOUR_C_THROW("set the condition manager!");
#endif

      const auto calcerr =
          Teuchos::getIntegralValue<Inpar::FLUID::CalcError>(params, "calculate error");
      const int calcerrfunctno = params.get<int>("error function number");

      const double t = my::fldparatimint_->time();


      //----------------------------------------------------------------------------
      //                         ELEMENT GEOMETRY
      //----------------------------------------------------------------------------

      // ---------------------------------------------------------------------
      // get initial node coordinates for element
      // ---------------------------------------------------------------------
      // get node coordinates
      Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
          ele, my::xyze_);

      // ---------------------------------------------------------------------
      // get additional state vectors for ALE case: grid displacement and vel.
      // ---------------------------------------------------------------------

      Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, nen_> egridv(Core::LinAlg::Initialization::zero);

      if (ele->is_ale()) my::get_grid_disp_vel_ale(dis, lm, edispnp, egridv);
      // add displacement when fluid nodes move in the ALE case
      if (ele->is_ale()) my::xyze_ += edispnp;


      // ---------------------------------------------------------------------

      /// element coordinates in EpetraMatrix
      Core::LinAlg::SerialDenseMatrix ele_xyze(nsd_, nen_);
      for (int i = 0; i < nen_; ++i)
      {
        for (int j = 0; j < nsd_; j++) ele_xyze(j, i) = my::xyze_(j, i);
      }

      // ---------------------------------------------------------------------
      // get velocity state vectors
      // ---------------------------------------------------------------------

      // get element-wise velocity/pressure field
      Core::LinAlg::Matrix<nsd_, nen_> evelaf(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nen_, 1> epreaf(Core::LinAlg::Initialization::zero);
      my::extract_values_from_global_vector(
          dis, lm, *my::rotsymmpbc_, &evelaf, &epreaf, "u and p at time n+1 (converged)");


      // ---------------------------------------------------------------------
      // set element advective field for Oseen problems
      // ---------------------------------------------------------------------
      if (my::fldpara_->physical_type() == Inpar::FLUID::oseen) my::set_advective_vel_oseen(ele);


      // ---------------------------------------------------------------------
      // get the element volume
      // ---------------------------------------------------------------------

      //  // set element area or volume
      const double vol =
          XFEM::Utils::eval_element_volume<distype>(my::xyze_, &(my::weights_), &(my::myknots_));

      //-----------------------------------------------------------------------------------
      //         evaluate element length, stabilization factors and average weights
      //-----------------------------------------------------------------------------------

      // element length
      double h_k = 0.0;
      double inv_hk = 0.0;

      // take a volume based element length
      h_k = XFEM::Utils::compute_vol_eq_diameter(vol);
      inv_hk = 1.0 / h_k;


      // evaluate shape function derivatives
      bool eval_deriv = true;


      //-----------------------------------------------------------------------------------
      //         initialize analytical solution vectors and error variables
      //-----------------------------------------------------------------------------------

      // analytical solution
      Core::LinAlg::Matrix<nsd_, 1> u_analyt(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, nsd_> grad_u_analyt(Core::LinAlg::Initialization::zero);
      double p_analyt = 0.0;

      // error
      Core::LinAlg::Matrix<nsd_, 1> u_err(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, nsd_> grad_u_err(Core::LinAlg::Initialization::zero);
      double p_err = 0.0;

      Core::LinAlg::Matrix<nsd_, 1> flux_u_err(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, 1> flux_p_err(Core::LinAlg::Initialization::zero);


      //--------------------------------------------
      // loop intersecting sides
      //--------------------------------------------
      // map of side-element id and Gauss points
      for (std::map<int, std::vector<Core::FE::GaussIntegration>>::const_iterator i =
               bintpoints.begin();
          i != bintpoints.end(); ++i)
      {
        //-----------------------------------------------------------------------------------

        // interface normal vector, pointing from background domain into the interface
        Core::LinAlg::Matrix<3, 1> normal(Core::LinAlg::Initialization::zero);
        // gauss-point coordinates
        Core::LinAlg::Matrix<3, 1> x_side(Core::LinAlg::Initialization::zero);

        // we need an interface to the boundary element (for projection)
        std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<distype>> si;

        // location array of boundary element
        Core::Elements::LocationArray cutla(1);

        // pointer to boundary element
        Core::Elements::Element* side = nullptr;

        // location array of element to couple with (only used for embedded fluid problems)
        Core::Elements::LocationArray coupl_la(1);

        // coordinates of boundary element
        Core::LinAlg::SerialDenseMatrix side_xyze;

        //-----------------------------------------------------------------------------------
        // only used for couplings:

        // coupling object between background element and each coupling element (side for
        // xfluid-sided coupling, element for other couplings)
        std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<distype>> ci;

        // pointer to coupling element
        Core::Elements::Element* coupl_ele = nullptr;

        // coupling element coordinates
        Core::LinAlg::SerialDenseMatrix coupl_xyze;

        //-----------------------------------------------------------------------------------

        //-----------------------------------------------------------------------------------

        int coup_sid = i->first;  // global coupling side id

        // get the coupling strategy for coupling of two fields
        const XFEM::EleCoupCond& coupcond =
            cond_manager->get_coupling_condition(coup_sid, my::eid_);
        const Inpar::XFEM::EleCouplingCondType& cond_type = coupcond.first;

        const std::vector<Core::FE::GaussIntegration>& cutintpoints = i->second;

        std::map<int, std::vector<Cut::BoundaryCell*>>::const_iterator j = bcells.find(coup_sid);
        if (j == bcells.end()) FOUR_C_THROW("missing boundary cell");

        const std::vector<Cut::BoundaryCell*>& bcs = j->second;
        if (bcs.size() != cutintpoints.size())
          FOUR_C_THROW("boundary cell integration rules mismatch");


        //---------------------------------------------------------------------------------
        // set flags used for coupling with given levelset/mesh coupling side
        bool is_ls_coupling_side = cond_manager->is_level_set_coupling(coup_sid);
        bool is_mesh_coupling_side = cond_manager->is_mesh_coupling(coup_sid);

        std::shared_ptr<Core::FE::Discretization> cutter_dis =
            cond_manager->get_cutter_dis(coup_sid);

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (is_ls_coupling_side and is_mesh_coupling_side)
          FOUR_C_THROW(
              "side cannot be a levelset-coupling side and a mesh coupling side at once: side {}",
              coup_sid);
        if (!is_ls_coupling_side and !is_mesh_coupling_side)
          FOUR_C_THROW("side is neither a levelset-coupling side nor a mesh coupling side: side {}",
              coup_sid);
#endif
        //-----------------------------------------------------------------------------------

        //---------------------------------------------------------------------------------
        // prepare the coupling objects
        if (is_mesh_coupling_side)
        {
          // get the side element and its coordinates for projection of Gaussian points
          side = cond_manager->get_side(coup_sid);
          Core::Geo::initial_position_array(side_xyze, side);

          // create auxiliary coupling object for the boundary element, in order to perform
          // projection
          si = Discret::Elements::XFLUID::SlaveElementInterface<
              distype>::create_slave_element_representation(side, side_xyze);

          // set displacement of side
          side->location_vector(*cutter_dis, cutla, false);
          si->add_slave_ele_disp(*cutter_dis, cutla[0].lm_);

          if (cond_type == Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET or
              cond_type == Inpar::XFEM::CouplingCond_SURF_FSI_PART or
              cond_type == Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP or
              cond_type == Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE)
          {
            si->set_interface_jump_statenp(*cutter_dis, "ivelnp", cutla[0].lm_);
          }

          if (cond_type == Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID)
          {
            // force to get the embedded element, even if background-sided coupling is active
            coupl_ele = cond_manager->get_cond_element(coup_sid);

            Core::Geo::initial_position_array(coupl_xyze, coupl_ele);

            ci = Discret::Elements::XFLUID::SlaveElementInterface<
                distype>::create_slave_element_representation(coupl_ele, coupl_xyze);

            // set velocity (and pressure) of coupling/slave element at current time step
            const int coup_idx = cond_manager->get_coupling_index(coup_sid, my::eid_);
            coupl_ele->location_vector(
                *cond_manager->get_coupling_by_idx(coup_idx)->get_cond_dis(), coupl_la, false);
            ci->set_slave_state(
                *cond_manager->get_coupling_by_idx(coup_idx)->get_cond_dis(), coupl_la[0].lm_);
          }
        }

        if (cond_manager->has_averaging_strategy(Inpar::XFEM::Xfluid_Sided))
        {
          h_k = XFEM::Utils::compute_char_ele_length<distype>(ele, ele_xyze, *cond_manager, vcSet,
              bcells, bintpoints, fldparaxfem_->visc_stab_hk());
          inv_hk = 1.0 / h_k;
        }

        //--------------------------------------------
        // loop boundary cells w.r.t current cut side
        //--------------------------------------------
        for (std::vector<Core::FE::GaussIntegration>::const_iterator i = cutintpoints.begin();
            i != cutintpoints.end(); ++i)
        {
          const Core::FE::GaussIntegration& gi = *i;
          Cut::BoundaryCell* bc =
              bcs[i - cutintpoints.begin()];  // get the corresponding boundary cell

          //--------------------------------------------
          // loop gausspoints w.r.t current boundary cell
          //--------------------------------------------
          for (Core::FE::GaussIntegration::iterator iquad = gi.begin(); iquad != gi.end(); ++iquad)
          {
            double drs =
                0.0;  // transformation factor between reference cell and linearized boundary cell

            const Core::LinAlg::Matrix<2, 1> eta(
                iquad.point());  // xi-coordinates with respect to side

            Core::LinAlg::Matrix<3, 1> rst(
                Core::LinAlg::Initialization::zero);  // local coordinates w.r.t background
                                                      // element

            Core::LinAlg::Matrix<3, 1> x_gp_lin(
                Core::LinAlg::Initialization::zero);  // gp in xyz-system on linearized
                                                      // interface

            // compute transformation factor, normal vector and global Gauss point coordinates
            if (bc->shape() != Core::FE::CellType::dis_none)  // Tessellation approach
            {
              XFEM::Utils::compute_surface_transformation(drs, x_gp_lin, normal, bc, eta);
            }
            else  // MomentFitting approach
            {
              drs = 1.0;
              normal = bc->get_normal_vector();
              const double* gpcord = iquad.point();
              for (int idim = 0; idim < 3; ++idim)
              {
                x_gp_lin(idim, 0) = gpcord[idim];
              }
            }

            // find element local position of gauss point
            std::shared_ptr<Cut::Position> pos =
                Cut::PositionFactory::build_position<nsd_, distype>(my::xyze_, x_gp_lin);
            pos->compute();
            pos->local_coordinates(rst);

            //        if (!levelset_cut)
            //        {
            //          // project gaussian point from linearized interface to warped side (get/set
            //          local side coordinates in SideImpl) Core::LinAlg::Matrix<2,1> xi_side(true);
            //
            //          //          side_impl[sid]->project_on_side(x_gp_lin, x_side, xi_side);
            //          si->project_on_side(x_gp_lin, x_side, xi_side);
            //        }
            //        else x_side = x_gp_lin;


            // TODO unify project_on_side and Evaluate for different spatial dimensions of boundary
            // slave element and volumetric slave element
            if (is_mesh_coupling_side)
            {
              // project gaussian point from linearized interface to warped side (get/set local side
              // coordinates in SideImpl)
              Core::LinAlg::Matrix<3, 1> xi_side(Core::LinAlg::Initialization::zero);
              // project on boundary element
              si->project_on_side(x_gp_lin, x_side, xi_side);

              if (cond_type == Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID)
                ci->evaluate(x_side);  // evaluate embedded element's shape functions at gauss-point
                                       // coordinates
            }
            else if (is_ls_coupling_side)
            {
              // TODO: do we need this here?
              //          if(cond_manager->IsCoupling( coup_sid, my::eid_ ))
              //            ci->evaluate( x_gp_lin ); // evaluate embedded element's shape functions
              //            at gauss-point coordinates
            }


            const double surf_fac = drs * iquad.weight();

            //--------------------------------------------
            // evaluate shape functions (and derivatives)

            if (eval_deriv)
            {
              eval_func_and_deriv(rst);
            }
            else
            {
              Core::FE::shape_function<distype>(rst, my::funct_);
            }


            // get velocity at integration point
            // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
            my::velint_.multiply(evelaf, my::funct_);

            // get velocity derivatives at integration point
            // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
            my::vderxy_.multiply_nt(evelaf, my::derxy_);

            // get pressure at integration point
            // (value at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
            double press = my::funct_.dot(epreaf);

            //----------------------------------------------
            // get convective velocity at integration point
            my::set_convective_velint(ele->is_ale());

            //--------------------------------------------
            // compute errors

            Core::LinAlg::Matrix<nsd_, 1> u_analyt(
                Core::LinAlg::Initialization::zero);  // boundary condition to enforce (xfsi),
                                                      // interfacial jump to enforce
                                                      // (fluidfluid)
            Core::LinAlg::Matrix<nsd_, nsd_> grad_u_analyt(Core::LinAlg::Initialization::zero);
            p_analyt = 0.0;

            analytical_reference(calcerr,  ///< which reference solution
                calcerrfunctno,            ///< error function number
                u_analyt,       ///< exact velocity (onesided), exact jump vector (coupled)
                grad_u_analyt,  ///< exact velocity gradient
                p_analyt,       ///< exact pressure
                x_gp_lin,       ///< xyz position of gaussian point
                t,              ///< time t
                mat);

            Core::LinAlg::Matrix<nsd_, 1> velint_s;
            if (is_mesh_coupling_side)
            {
              si->get_interface_velnp(velint_s);
            }

            if (cond_type == Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID)
            {
              u_err.update(1.0, my::velint_, -1.0, velint_s, 0.0);

              Core::LinAlg::Matrix<nsd_, nsd_> grad_u_side(Core::LinAlg::Initialization::zero);
              ci->get_interface_vel_gradnp(grad_u_side);

              grad_u_err.update(1.0, my::vderxy_, -1.0, grad_u_side, 0.0);

              double press_coupl = 0.0;
              ci->get_interface_presnp(press_coupl);
              // p_err = p_background - p_emb;
              p_err = press - press_coupl;
            }
            else
            {
              u_err.update(1.0, my::velint_, -1.0, u_analyt, 0.0);
              grad_u_err.update(1.0, my::vderxy_, -1.0, grad_u_analyt, 0.0);
              p_err = press - p_analyt;
            }

            flux_u_err.multiply(grad_u_err, normal);
            flux_p_err.update(p_err, normal, 0.0);

            /*
             * Scaling of interface error norms:
             *
             *                       (1)           (2)          (3)
             *                    /  \mu    \rho             h * \rho         \
             *  NIT :=  \gamma * |    --  +  -- * |u|_inf  + ----------------- |
             *                    \   h      6               12 * \theta * dt /
             *
             *                             interface errors
             *  1.   || nu/h * (u_b - u_e - u_jump) ||_L_2(Gamma)        =  broken H1/2 Sobolev norm
             * for boundary/coupling condition
             *  2.   || nu^(+1/2) grad( u_b - u_e )*n ||_H-1/2(Gamma)    =  standard H-1/2 Sobolev
             * norm for normal flux (velocity part)
             *  3.   || (p_b - p_e)*n ||_H-1/2(Gamma)                    =  standard H-1/2 Sobolev
             * norm for normal flux (pressure part)
             *  4.   || (u*n)_inflow (u - u*) ||_L2(Gamma)               =  L^2 Sobolev norm for
             * inflow boundary/coupling condition
             *  5.   || (sigma*h+|u|+nu/h)^(+1/2) (u - u*)*n ||_L2(Gamma) = L^2 Sobolev norm for
             * mass conservation coupling condition
             */
            double u_err_squared = 0.0;
            double u_err_squared_normal = 0.0;
            double flux_u_err_squared = 0.0;
            double flux_p_err_squared = 0.0;

            // evaluate squared errors at gaussian point
            for (int isd = 0; isd < nsd_; isd++)
            {
              u_err_squared += u_err(isd) * u_err(isd) * surf_fac;
              u_err_squared_normal +=
                  u_err(isd) * normal(isd) * normal(isd) * u_err(isd) * surf_fac;
              flux_u_err_squared += flux_u_err(isd) * flux_u_err(isd) * surf_fac;
              flux_p_err_squared += flux_p_err(isd) * flux_p_err(isd) * surf_fac;
            }

            // interface errors
            double nit_stabfac = 0.0;

            // Needs coupling condition to get kappas!
            const double kappa_m = 1.0;
            const double kappa_s = 0.0;
            double visc_stab_fac = 0.0;
            double visc_stab_fac_tang = 0.0;
            cond_manager->get_visc_penalty_stabfac(coup_sid, ele, kappa_m, kappa_s, inv_hk,
                fldparaxfem_, visc_stab_fac, visc_stab_fac_tang);

            XFEM::Utils::nit_compute_full_penalty_stabfac(
                nit_stabfac,  ///< to be filled: full Nitsche's penalty term scaling
                              ///< (viscous+convective part)
                normal, h_k,
                kappa_m,  // weights (only existing for Nitsche currently!!)
                kappa_s,  // weights (only existing for Nitsche currently!!)
                my::convvelint_, velint_s,
                visc_stab_fac,  ///< Nitsche's viscous scaling part of penalty term
                my::fldparatimint_->time_fac(), my::fldparatimint_->is_stationary(), densaf_master_,
                densaf_slave_, fldparaxfem_->mass_conservation_scaling(),
                fldparaxfem_->mass_conservation_combination(), fldparaxfem_->nit_stab_scaling(),
                fldparaxfem_->conv_stab_scaling(), fldparaxfem_->xff_conv_stab_scaling(),
                my::fldpara_->is_conservative(), true);

            const double veln_normal = my::convvelint_.dot(normal);  // TODO: shift this to routine
            double NIT_inflow_stab = std::max(0.0, -veln_normal);

            ele_interf_norms[0] += visc_stab_fac * u_err_squared;
            ele_interf_norms[1] += h_k * my::visc_ * flux_u_err_squared;
            ele_interf_norms[2] += h_k * flux_p_err_squared;
            ele_interf_norms[3] += NIT_inflow_stab * u_err_squared;
            ele_interf_norms[4] += nit_stabfac * u_err_squared_normal;

          }  // end loop gauss points of boundary cell
        }  // end loop boundary cells of side

      }  // end loop cut sides

      return 0;
    }


    /*--------------------------------------------------------------------------------
     * add mixed/hybrid stress-based LM interface condition to element matrix and rhs
     *--------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::element_xfem_interface_hybrid_lm(
        Discret::Elements::Fluid* ele,                                ///< fluid element
        Core::FE::Discretization& dis,                                ///< background discretization
        const std::vector<int>& lm,                                   ///< element local map
        const std::shared_ptr<XFEM::ConditionManager>& cond_manager,  ///< XFEM condition manager
        const std::vector<Core::FE::GaussIntegration>& intpoints,     ///< element gauss points
        const std::map<int, std::vector<Cut::BoundaryCell*>>& bcells,  ///< boundary cells
        const std::map<int, std::vector<Core::FE::GaussIntegration>>&
            bintpoints,  ///< boundary integration points
        const std::map<int, std::vector<int>>&
            patchcouplm,  ///< lm vectors for coupling elements, key= global coupling side-Id
        std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>&
            side_coupling,                          ///< side coupling matrices
        Teuchos::ParameterList& params,             ///< parameter list
        std::shared_ptr<Core::Mat::Material>& mat,  ///< material
        Core::LinAlg::SerialDenseMatrix&
            elemat1_epetra,  ///< local system matrix of intersected element
        Core::LinAlg::SerialDenseVector&
            elevec1_epetra,                      ///< local element vector of intersected element
        Core::LinAlg::SerialDenseMatrix& Cuiui,  ///< coupling matrix of a side with itself
        const Cut::plain_volumecell_set& vcSet   ///< set of plain volume cells
    )
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (cond_manager == nullptr) FOUR_C_THROW("set the condition manager!");
#endif

      //--------------------------------------------------------
      // determine, whether this is a Cauchy stress-based (MHCS)
      // or viscous stress-based LM approach (MHVS)
      //--------------------------------------------------------
      Inpar::XFEM::CouplingMethod coupling_method = fldparaxfem_->get_coupling_method();

      // check for valid boundary integration type
      switch (coupling_method)
      {
        case Inpar::XFEM::Hybrid_LM_viscous_stress:
        case Inpar::XFEM::Hybrid_LM_Cauchy_stress:
          break;
        case Inpar::XFEM::Nitsche:
          FOUR_C_THROW(
              "Wrong evaluation routine for Nitsche coupling. Try element_xfem_interface_nit/NIT2 "
              "instead.");
          break;
        default:
          FOUR_C_THROW(
              "Landed in evaluation routine for stress-based LM, given an unknown or unsupported "
              "coupling method.");
          break;
      }

      const bool is_MHVS = (coupling_method == Inpar::XFEM::Hybrid_LM_viscous_stress);


      // REMARK: to avoid confusion -
      // 'side' = boundary element, part of cutdis (can be warped)
      // 'boundary cell' = belongs to volume-cell

      // do we need convective stabilization?
      bool add_conv_stab(
          fldparaxfem_->xff_conv_stab_scaling() != Inpar::XFEM::XFF_ConvStabScaling_none ||
          fldparaxfem_->conv_stab_scaling() != Inpar::XFEM::ConvStabScaling_none);

      //----------------------------------------------------------------------------
      //                         ELEMENT GEOMETRY
      //----------------------------------------------------------------------------

      // ---------------------------------------------------------------------
      // get initial node coordinates for element
      // ---------------------------------------------------------------------
      // get node coordinates
      Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
          ele, my::xyze_);

      // ---------------------------------------------------------------------
      // get additional state vectors for ALE case: grid displacement and vel.
      // ---------------------------------------------------------------------

      Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, nen_> egridv(Core::LinAlg::Initialization::zero);

      if (ele->is_ale()) my::get_grid_disp_vel_ale(dis, lm, edispnp, egridv);


      // ---------------------------------------------------------------------

      /// element coordinates in EpetraMatrix
      Core::LinAlg::SerialDenseMatrix ele_xyze(nsd_, nen_);
      for (int i = 0; i < nen_; ++i)
      {
        for (int j = 0; j < nsd_; j++) ele_xyze(j, i) = my::xyze_(j, i);
      }

      // ---------------------------------------------------------------------
      // get velocity state vectors
      // ---------------------------------------------------------------------

      // get element-wise velocity/pressure field for current time step
      Core::LinAlg::Matrix<nsd_, nen_> evelaf(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nen_, 1> epreaf(Core::LinAlg::Initialization::zero);
      my::extract_values_from_global_vector(dis, lm, *my::rotsymmpbc_, &evelaf, &epreaf, "velaf");

      // get element-wise velocity/pressure field for previous time step
      Core::LinAlg::Matrix<nsd_, nen_> eveln(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nen_, 1> epren(Core::LinAlg::Initialization::zero);
      if (my::fldparatimint_->is_new_ost_implementation())
        my::extract_values_from_global_vector(dis, lm, *my::rotsymmpbc_, &eveln, &epren, "veln");

      // ---------------------------------------------------------------------
      // set element advective field for Oseen problems
      // ---------------------------------------------------------------------
      if (my::fldpara_->physical_type() == Inpar::FLUID::oseen) my::set_advective_vel_oseen(ele);

      // compute characteristic element length based on the background element
      const double h_k = XFEM::Utils::compute_char_ele_length<distype>(
          ele, ele_xyze, *cond_manager, vcSet, bcells, bintpoints, fldparaxfem_->visc_stab_hk());

      //--------------------------------------------------------
      // declaration of matrices & rhs
      //--------------------------------------------------------

      // sub-blocks of matrix K_{\sigma\sigma} (-->K_ss)
      Core::LinAlg::Matrix<nen_, nen_> bK_ss(Core::LinAlg::Initialization::zero);  // N * N^T
      Core::LinAlg::Matrix<nen_, nen_> invbK_ss(
          Core::LinAlg::Initialization::zero);  // inverse of bK_ss, (N * N^T)^-1
      Core::LinAlg::Matrix<nen_, nen_> halfInvbK_ss(
          Core::LinAlg::Initialization::zero);  // inverse scaled by 1/2

      // The block matrices K_... result from the volume integrals on the cut element.
      // In case of a viscous stress-based approach (MHVS), there is no term like K_sp,
      // that couples the Lagrange multiplier stresses with the pressure.
      // Instead, we have contributions G_up and G_pu from surface coupling terms, that don't play a
      // role in the condensation of the stress-based LM (in contrast to velocity coupling terms s-u
      // and u-s). For compatibility reasons, we include column- & row-blocks for the pressure in
      // K_su and K_us. In the case of a MHVS-approach, these remain empty, as we have G_up and
      // G_pu.

      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_, numdofpernode_>
          K_su;  // s-u, s-p
      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numdofpernode_, numstressdof_>
          K_us;  // u-s, p-s

      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_, numstressdof_>
          invK_ss;  // K_ss^-1
      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1> rhs_s;

      // Only for MHVS:

      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, nsd_, nsd_> K_uu;
      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, nsd_, 1> rhs_uu;

      // surface-based pressure terms, analogous to Nitsche
      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, nsd_, 1> G_up;
      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, 1, nsd_> G_pu;

      // rhs-contributions from interface integration
      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, nsd_, 1> rhs_up;
      Core::LinAlg::Matrix<nen_, 1> rhs_pu(Core::LinAlg::Initialization::zero);

      //--------------------------------------------------------
      // build matrices K (based on volume terms)
      //--------------------------------------------------------

      // in case of MHVS we have a stabilizing parameter!
      double mhvs_param = 1.0;
      if (is_MHVS)
      {
        // MHVS-stabilization parameter n
        // REMARK:
        // NIT_visc_stab_fac =   gamma * mu * C^2
        // (C^2 includes characteristic length scale);
        // the analogous MHVS-factor = 2 * n * mu * meas(\Gamma)/meas(\Omega_K)
        // gamma <--> 2 * n
        double mhvs_param = fldparaxfem_->nit_stab_scaling() / 2.0;
        if (fabs(mhvs_param) < 1.e-8)
          FOUR_C_THROW(
              "MHVS stabilizing parameter n appears in denominator. Please avoid choosing 0.");
      }

      // build volumetric coupling matrices
      hybrid_lm_build_vol_based(intpoints, vcSet, evelaf, epreaf, bK_ss, invbK_ss, K_su, rhs_s,
          K_us, K_uu, rhs_uu, is_MHVS, mhvs_param);

      /*--------------------------------------------------------
        build surface coupling terms
      --------------------------------------------------------*/

      //-----------------------------------------------------------------------------------

      // side coupling implementation between background element and each cutting side OR
      // embedded element
      std::map<int, std::shared_ptr<Discret::Elements::XFLUID::HybridLMInterface<distype>>> ci;

      //-----------------------------------------------------------------------------------
      //                     application-specific flags & parameters
      //-----------------------------------------------------------------------------------

      // map of boundary element gids and coupling matrices, [0]: Gsui, [1]: Guis
      std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>> Cuiui_coupling;

      std::vector<int> patchelementslm;


      // auxiliary coupling implementation for terms
      // find all the intersecting elements of actele
      std::set<int> begids;
      for (std::map<int, std::vector<Cut::BoundaryCell*>>::const_iterator bc = bcells.begin();
          bc != bcells.end(); ++bc)
      {
        const int coup_sid = bc->first;

        // get the coupling strategy for coupling of two fields
        const Inpar::XFEM::AveragingStrategy averaging_strategy =
            cond_manager->get_averaging_strategy(coup_sid, my::eid_);

        begids.insert(coup_sid);

        if (!cond_manager->is_coupling(coup_sid, my::eid_))
          continue;  // no coupling with current side

        if (cond_manager->is_level_set_coupling(coup_sid))
          FOUR_C_THROW(
              "PatchLocationVector for level-set coupling not supported for hybrid-lm methods yet");

        // get coupling matrices for the current side (boundary element)
        std::vector<Core::LinAlg::SerialDenseMatrix>& Cuiui_matrices =
            Cuiui_coupling[coup_sid];  // create new vector of Coupling matrices

        std::map<int, std::vector<int>>::const_iterator j = patchcouplm.find(coup_sid);
        if (j == patchcouplm.end()) FOUR_C_THROW("missing side");

        const std::vector<int>& patchlm = j->second;

        // get number of dofs for coupling side/element
        const size_t ndof_i = j->second.size();

        patchelementslm.reserve(patchelementslm.size() + ndof_i);
        patchelementslm.insert(patchelementslm.end(), patchlm.begin(), patchlm.end());

        if (averaging_strategy != Inpar::XFEM::Xfluid_Sided)
          FOUR_C_THROW(
              "Embedded-sided or Mean or Harmonic coupling for stress-based hybrid LM approach is "
              "not yet available!");

        Cuiui_matrices.resize(2);
        Cuiui_matrices[0].shape(nen_ * numstressdof_,
            ndof_i);  // Gsui (coupling between background elements sigma and current side!)
        Cuiui_matrices[1].shape(ndof_i, nen_ * numstressdof_);  // Guis
      }


      // map of Nitsche-interfaces for building convective stabilization terms
      std::map<int, std::shared_ptr<Discret::Elements::XFLUID::NitscheInterface<distype>>> si_nit;

      // map of boundary element gids and coupling contributions from convective stabilization terms
      std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>> side_coupling_extra;

      // reshape coupling matrices for convective stabilization terms
      if (add_conv_stab || my::fldparatimint_->is_new_ost_implementation())
      {
        hybrid_lm_create_special_contribution_matrices(*cond_manager, begids, side_coupling_extra);
      }


      //--------------------------------------------
      // loop intersecting sides
      //--------------------------------------------
      // map of side-element id and Gauss points
      for (std::map<int, std::vector<Core::FE::GaussIntegration>>::const_iterator i =
               bintpoints.begin();
          i != bintpoints.end(); ++i)
      {
        //-----------------------------------------------------------------------------------

        // interface normal vector, pointing from background domain into the interface
        Core::LinAlg::Matrix<3, 1> normal(Core::LinAlg::Initialization::zero);
        // gauss-point coordinates
        Core::LinAlg::Matrix<3, 1> x_side(Core::LinAlg::Initialization::zero);

        // we need an interface to the boundary element (for projection)
        std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<distype>> si;

        // location array of boundary element
        Core::Elements::LocationArray cutla(1);

        // pointer to boundary element
        Core::Elements::Element* side = nullptr;

        // coordinates of boundary element
        Core::LinAlg::SerialDenseMatrix side_xyze;

        //-----------------------------------------------------------------------------------
        // only used for couplings:

        // pointer to coupling element
        Core::Elements::Element* coupl_ele = nullptr;

        // coupling element coordinates
        Core::LinAlg::SerialDenseMatrix coupl_xyze;

        //-----------------------------------------------------------------------------------


        int coup_sid = i->first;

        // get the coupling strategy for coupling of two fields
        const Inpar::XFEM::AveragingStrategy averaging_strategy =
            cond_manager->get_averaging_strategy(coup_sid, my::eid_);
        const XFEM::EleCoupCond& coupcond =
            cond_manager->get_coupling_condition(coup_sid, my::eid_);
        const Inpar::XFEM::EleCouplingCondType& cond_type = coupcond.first;

        const int coup_idx = cond_manager->get_coupling_index(coup_sid, my::eid_);
        std::shared_ptr<XFEM::CouplingBase> coupling = cond_manager->get_coupling_by_idx(coup_idx);

        const std::vector<Core::FE::GaussIntegration>& cutintpoints = i->second;

        // get side's boundary cells
        std::map<int, std::vector<Cut::BoundaryCell*>>::const_iterator j = bcells.find(coup_sid);
        if (j == bcells.end()) FOUR_C_THROW("missing boundary cell");

        const std::vector<Cut::BoundaryCell*>& bcs = j->second;
        if (bcs.size() != cutintpoints.size())
          FOUR_C_THROW("boundary cell integration rules mismatch");

        //---------------------------------------------------------------------------------
        // set flags used for coupling with given levelset/mesh coupling side
        bool is_ls_coupling_side = cond_manager->is_level_set_coupling(coup_sid);
        bool is_mesh_coupling_side = cond_manager->is_mesh_coupling(coup_sid);

        std::shared_ptr<Core::FE::Discretization> cutter_dis =
            cond_manager->get_cutter_dis(coup_sid);

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (is_ls_coupling_side and is_mesh_coupling_side)
          FOUR_C_THROW(
              "side cannot be a levelset-coupling side and a mesh coupling side at once: side {}",
              coup_sid);
        if (!is_ls_coupling_side and !is_mesh_coupling_side)
          FOUR_C_THROW("side is neither a levelset-coupling side nor a mesh coupling side: side {}",
              coup_sid);
#endif

        //-----------------------------------------------------------------------------------
        std::shared_ptr<XFEM::MeshCouplingFSI> mc_fsi = nullptr;
        bool assemble_iforce = false;

        //---------------------------------------------------------------------------------
        // prepare the coupling objects
        //---------------------------------------------------------------------------------
        // prepare the coupling objects
        if (is_mesh_coupling_side)
        {
          mc_fsi = std::dynamic_pointer_cast<XFEM::MeshCouplingFSI>(coupling);
          if (mc_fsi != nullptr) assemble_iforce = true;

          // get the side element and its coordinates for projection of Gaussian points
          side = cond_manager->get_side(coup_sid);
          Core::Geo::initial_position_array(side_xyze, side);

          // create auxiliary coupling object for the boundary element, in order to perform
          // projection
          si = Discret::Elements::XFLUID::SlaveElementInterface<
              distype>::create_slave_element_representation(side, side_xyze);

          // set displacement of side
          side->location_vector(*cutter_dis, cutla, false);
          si->add_slave_ele_disp(*cutter_dis, cutla[0].lm_);

          if (cond_type == Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET or
              cond_type == Inpar::XFEM::CouplingCond_SURF_FSI_PART or
              cond_type == Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP or
              cond_type == Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE)
          {
            si->set_interface_jump_statenp(*cutter_dis, "ivelnp", cutla[0].lm_);
            if (my::fldparatimint_->is_new_ost_implementation())
              si->set_interface_jump_staten(*cutter_dis, "iveln", cutla[0].lm_);
          }
        }

        std::shared_ptr<Core::FE::Discretization> coupl_dis_ =
            cond_manager->get_coupling_dis(coup_sid);

        if (!(is_ls_coupling_side and
                !cond_manager->is_coupling(coup_sid, my::eid_)))  // not level-set-WDBC case
        {
          if (averaging_strategy == Inpar::XFEM::Embedded_Sided or
              averaging_strategy == Inpar::XFEM::Mean)  // for coupling-sided and two-sided coupling
            FOUR_C_THROW("embedded or two-sided coupling not supported");
          else
          {
            // TODO get the coupling element / the coupling side!!!
            coupl_ele = cond_manager->get_side(coup_sid);
          }

          Core::Geo::initial_position_array(coupl_xyze, coupl_ele);
        }

        if (!cond_manager->is_coupling(coup_sid, my::eid_))
        {
          if (is_ls_coupling_side)  //... for problems with cut interface defined by level-set
                                    // field, currently only one-sided
          {
            ci[coup_sid] = Discret::Elements::XFLUID::HybridLMInterface<
                distype>::create_hybrid_lm_coupling_x_fluid_wdbc(fldparaxfem_
                    ->is_viscous_adjoint_symmetric());
          }
          else if (is_mesh_coupling_side)
          {
            ci[coup_sid] = Discret::Elements::XFLUID::HybridLMInterface<
                distype>::create_hybrid_lm_coupling_x_fluid_wdbc(coupl_ele, coupl_xyze,
                fldparaxfem_->is_viscous_adjoint_symmetric());
          }
        }
        else  // coupling
        {
          std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::iterator c =
              side_coupling.find(coup_sid);

          std::vector<Core::LinAlg::SerialDenseMatrix>& side_matrices = c->second;

          if (side_matrices.size() != 3)
            FOUR_C_THROW(
                "Obtained only {} side coupling matrices. 3 required.", side_matrices.size());

          // coupling matrices between background element and one! side
          Core::LinAlg::SerialDenseMatrix& C_you = side_matrices[0];
          Core::LinAlg::SerialDenseMatrix& C_uui = side_matrices[1];
          Core::LinAlg::SerialDenseMatrix& rhC_ui = side_matrices[2];

          // coupling matrices between one side and itself via the element Kss
          std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::iterator c2 =
              Cuiui_coupling.find(coup_sid);
          std::vector<Core::LinAlg::SerialDenseMatrix>& Cuiui_matrices = c2->second;
          Core::LinAlg::SerialDenseMatrix& eleGsui = Cuiui_matrices[0];
          Core::LinAlg::SerialDenseMatrix& eleGuis = Cuiui_matrices[1];

          if (averaging_strategy == Inpar::XFEM::Embedded_Sided or
              averaging_strategy == Inpar::XFEM::Mean)  // for coupling-sided and two-sided coupling
            FOUR_C_THROW("embedded or two-sided coupling not supported");

          ci[coup_sid] = Discret::Elements::XFLUID::HybridLMInterface<
              distype>::create_hybrid_lm_coupling_x_fluid_sided(coupl_ele, coupl_xyze, C_you, C_uui,
              rhC_ui, eleGsui, eleGuis, fldparaxfem_->is_viscous_adjoint_symmetric());
        }

        if (cond_manager->is_coupling(coup_sid, my::eid_))
        {
          std::map<int, std::vector<int>>::const_iterator k = patchcouplm.find(coup_sid);
          const std::vector<int>& coupl_lm = k->second;

          // set velocity (and pressure) of coupling/slave element at current time step
          ci[coup_sid]->set_slave_state(*coupl_dis_, coupl_lm);

          // note: old state is handled by nitsche coupling si_nit
        }


        if (!(is_ls_coupling_side and
                !cond_manager->is_coupling(coup_sid, my::eid_)))  // not level-set-WDBC case
        {
          std::map<int, std::vector<int>>::const_iterator k = patchcouplm.find(coup_sid);
          const std::vector<int>& coupl_lm = k->second;

          // add displacement of coupling element at current time step
          ci[coup_sid]->add_slave_ele_disp(*coupl_dis_, coupl_lm);
        }


        // define interface force vector w.r.t side
        Core::LinAlg::SerialDenseVector iforce;
        iforce.size(cutla[0].lm_.size());

        // we need an instance of Nitsche-evaluation class for evaluation of
        // inflow terms and for evaluation of terms for the previous time step
        // (new OST)
        if (add_conv_stab || my::fldparatimint_->is_new_ost_implementation())
        {
          if (is_mesh_coupling_side)
          {
            // create si_nit based on side and side_xyze and not w.r.t coup_ele, as no derivatives
            // are used in these coupling terms see also
            // hybrid_lm_create_special_contribution_matrices()
            if (cond_manager->is_coupling(coup_sid, my::eid_))  //... for two-sided problems
            {
              // coupling matrices between background element and one! side
              std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::iterator c =
                  side_coupling_extra.find(coup_sid);
              std::vector<Core::LinAlg::SerialDenseMatrix>& side_matrices_extra = c->second;
              Core::LinAlg::SerialDenseMatrix& C_you = side_matrices_extra[0];
              Core::LinAlg::SerialDenseMatrix& C_uui = side_matrices_extra[1];
              Core::LinAlg::SerialDenseMatrix& rhC_ui = side_matrices_extra[2];
              Core::LinAlg::SerialDenseMatrix& C_uiui = side_matrices_extra[3];

              si_nit[coup_sid] = Discret::Elements::XFLUID::NitscheInterface<
                  distype>::create_nitsche_coupling_x_fluid_sided(side, side_xyze, elemat1_epetra,
                  C_you, C_uui, C_uiui, elevec1_epetra, rhC_ui, *fldparaxfem_);
            }
            else
            {
              si_nit[coup_sid] = Discret::Elements::XFLUID::NitscheInterface<
                  distype>::create_nitsche_coupling_x_fluid_wdbc(side, side_xyze, elemat1_epetra,
                  elevec1_epetra, *fldparaxfem_);
            }

            // set velocity for current time step
            si_nit[coup_sid]->set_slave_state(*cutter_dis, cutla[0].lm_);

            // set displacement of side for current time step
            si_nit[coup_sid]->add_slave_ele_disp(*cutter_dis, cutla[0].lm_);
          }
          else if (is_ls_coupling_side)
          {
            if (cond_manager->is_coupling(coup_sid, my::eid_))  //... for two-sided problems
            {
              FOUR_C_THROW(
                  "convective terms for hybrid lm coupling not implemented yet for level-set cuts");
            }
            else
            {
              si_nit[coup_sid] = Discret::Elements::XFLUID::NitscheInterface<
                  distype>::create_nitsche_coupling_x_fluid_wdbc(elemat1_epetra, elevec1_epetra,
                  *fldparaxfem_);
            }
          }
          else
            FOUR_C_THROW("no mesh-/level-set coupling object for coupling sid {}", coup_sid);
        }

        // Set State for current and previous time
        if (cond_manager->is_coupling(coup_sid, my::eid_))
        {
          if (my::fldparatimint_->is_new_ost_implementation())
          {
            // set velocity for previous time step
            si_nit[coup_sid]->set_slave_staten(*cutter_dis, cutla[0].lm_);
          }
        }


        //--------------------------------------------
        // loop boundary cells w.r.t current cut side
        //--------------------------------------------
        for (std::vector<Core::FE::GaussIntegration>::const_iterator i = cutintpoints.begin();
            i != cutintpoints.end(); ++i)
        {
          const Core::FE::GaussIntegration& gi = *i;
          Cut::BoundaryCell* bc =
              bcs[i - cutintpoints.begin()];  // get the corresponding boundary cell

          //--------------------------------------------
          // loop gausspoints w.r.t current boundary cell
          //--------------------------------------------
          for (Core::FE::GaussIntegration::iterator iquad = gi.begin(); iquad != gi.end(); ++iquad)
          {
            double drs =
                0.0;  // transformation factor between reference cell and linearized boundary cell

            const Core::LinAlg::Matrix<2, 1> eta(
                iquad.point());  // xi-coordinates with respect to side

            Core::LinAlg::Matrix<3, 1> rst(
                Core::LinAlg::Initialization::zero);  // local coordinates w.r.t background
                                                      // element

            Core::LinAlg::Matrix<3, 1> x_gp_lin(
                Core::LinAlg::Initialization::zero);  // gp in xyz-system on linearized
                                                      // interface

            // compute transformation factor, normal vector and global Gauss point coordinates
            if (bc->shape() != Core::FE::CellType::dis_none)  // Tessellation approach
            {
              XFEM::Utils::compute_surface_transformation(drs, x_gp_lin, normal, bc, eta);
            }
            else  // MomentFitting approach
            {
              drs = 1.0;
              normal = bc->get_normal_vector();
              const double* gpcord = iquad.point();
              for (int idim = 0; idim < 3; ++idim)
              {
                x_gp_lin(idim, 0) = gpcord[idim];
              }
            }

            // find element local position of gauss point
            std::shared_ptr<Cut::Position> pos =
                Cut::PositionFactory::build_position<nsd_, distype>(my::xyze_, x_gp_lin);
            pos->compute();
            pos->local_coordinates(rst);

            // TODO unify project_on_side and Evaluate for different spatial dimensions of boundary
            // slave element and volumetric slave element
            if (is_mesh_coupling_side)
            {
              // project gaussian point from linearized interface to warped side (get/set local side
              // coordinates in SideImpl)
              Core::LinAlg::Matrix<3, 1> xi_side(Core::LinAlg::Initialization::zero);
              // project on boundary element
              si->project_on_side(x_gp_lin, x_side, xi_side);

              if (averaging_strategy == Inpar::XFEM::Embedded_Sided or
                  averaging_strategy == Inpar::XFEM::Mean)
                FOUR_C_THROW(
                    "embedded or two-sided weighting not supported");  // evaluate embedded
                                                                       // element's shape functions
                                                                       // at gauss-point coordinates
              else
              {
                ci.at(coup_sid)->evaluate(
                    xi_side);  // evaluate side's shape functions at gauss-point coordinates
                if (add_conv_stab || my::fldparatimint_->is_new_ost_implementation())
                  si_nit.at(coup_sid)->evaluate(
                      xi_side);  // evaluate side's shape functions at gauss-point coordinates
              }
            }
            else if (is_ls_coupling_side)
            {
              if (cond_manager->is_coupling(coup_sid, my::eid_))
                FOUR_C_THROW(
                    "coupling for level-sets not supported here");  // evaluate embedded element's
                                                                    // shape functions at
                                                                    // gauss-point coordinates
            }


            // integration factors
            const double surf_fac = drs * iquad.weight();
            const double timefacfac = surf_fac * my::fldparatimint_->time_fac();

            //--------------------------------------------

            // evaluate shape functions (and derivatives)

            if (assemble_iforce)  // evaluate derivatives to assemble iforce vector
            {
              eval_func_and_deriv(rst);
            }
            else
            {
              Core::FE::shape_function<distype>(rst, my::funct_);
            }


            // get velocity at integration point
            // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
            my::velint_.multiply(evelaf, my::funct_);


            //-----------------------------------------------------------------------------
            // define the prescribed interface jump vectors for velocity and traction

            Core::LinAlg::Matrix<nsd_, 1> ivelint_jump(Core::LinAlg::Initialization::zero);
            Core::LinAlg::Matrix<nsd_, 1> itraction_jump(Core::LinAlg::Initialization::zero);
            Core::LinAlg::Matrix<nsd_, nsd_> proj_tangential(Core::LinAlg::Initialization::zero);
            Core::LinAlg::Matrix<nsd_, nsd_> LB_proj_matrix(Core::LinAlg::Initialization::zero);

            double kappa_m = 0.0;
            double kappa_s = 0.0;
            double visc_m = 0.0;
#ifdef FOUR_C_ENABLE_ASSERTIONS
            // Only Navier Slip used kappa_m,kappa_s and visc_m defined just before!
            // To use Navier Slip specify them correct!
            if (cond_type == Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP ||
                cond_type == Inpar::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP)
              FOUR_C_THROW(
                  "element_xfem_interface_hybrid_lm with Navier Slip, what to do with "
                  "kappa_m/kappa_s "
                  "for the dyn_visc in the traction_jump?");
#endif

            Core::LinAlg::Matrix<3, 1> dummy1;
            std::vector<double> dummy2;

            get_interface_jump_vectors(coupcond, coupling, ivelint_jump, itraction_jump,
                proj_tangential, LB_proj_matrix, x_gp_lin, normal, *si, rst, kappa_m, kappa_s,
                visc_m, dummy1, dummy2);

            if (cond_type == Inpar::XFEM::CouplingCond_LEVELSET_NEUMANN or
                cond_type == Inpar::XFEM::CouplingCond_SURF_NEUMANN)
            {
              //-----------------------------------------------------------------------------
              // evaluate the Neumann boundary condition term
              evaluate_neumann(timefacfac,  ///< theta*dt
                  my::funct_,               ///< coupling master shape functions
                  itraction_jump,  ///< prescribed interface traction, jump height for coupled
                                   ///< problems
                  elevec1_epetra   ///< element rhs vector
              );

              if (my::fldparatimint_->is_new_ost_implementation())
              {
                FOUR_C_THROW(
                    "how to deal with Neumann boundary condition and new OSTImplementation");
              }
            }
            else  // standard Hybrid lm terms
            {
              //--------------------------------------------

              bK_ss.multiply_nt(my::funct_, my::funct_);

              /*                      \
                - |  (virt tau) * n^f , Du  |
                   \                      */

              hybrid_lm_evaluate_surf_based(*ci[coup_sid], bK_ss, K_su, K_us, rhs_s, epreaf, K_uu,
                  rhs_uu, G_up, G_pu, rhs_up, rhs_pu, normal, timefacfac, ivelint_jump,
                  itraction_jump, cond_manager->is_coupling(coup_sid, my::eid_), is_MHVS);

              //--------------------------------------------
              // evaluate additional inflow/convective stabilization terms

              if (add_conv_stab)
              {
                double NIT_full_stab_fac = 0.0;
                const double NIT_visc_stab_fac = 0.0;

                my::set_convective_velint(ele->is_ale());


                //-----------------------------------------------------------------------------

                // define the coupling between two not matching grids
                // for fluidfluidcoupling
                // domain Omega^m := Coupling master (XFluid)
                // domain Omega^s := Alefluid( or monolithic: structure) ( not available for
                // non-coupling (Dirichlet) )

                // [| v |] := vm - vs


                if (cond_manager->is_coupling(coup_sid, my::eid_))
                {
                  Core::LinAlg::Matrix<nsd_, 1> velint_s;
                  ci[coup_sid]->get_interface_velnp(velint_s);


                  // Get Material parameters for the master side!
                  get_material_parameters_volume_cell(
                      mat, densaf_master_, viscaf_master_, gamma_m_);

                  bool non_xfluid_coupling;
                  double kappa_m;
                  double kappa_s;
                  cond_manager->get_average_weights(
                      coup_sid, ele, kappa_m, kappa_s, non_xfluid_coupling);

                  XFEM::Utils::nit_compute_full_penalty_stabfac(
                      NIT_full_stab_fac,  ///< to be filled: full Nitsche's penalty term scaling
                                          ///< (viscous+convective part)
                      normal, h_k,
                      kappa_m,  // weights (only existing for Nitsche currently!!)
                      kappa_s,  // weights (only existing for Nitsche currently!!)
                      my::convvelint_, velint_s,
                      NIT_visc_stab_fac,  ///< Nitsche's viscous scaling part of penalty term
                      my::fldparatimint_->time_fac(), my::fldparatimint_->is_stationary(),
                      densaf_master_, densaf_slave_, fldparaxfem_->mass_conservation_scaling(),
                      fldparaxfem_->mass_conservation_combination(),
                      fldparaxfem_->nit_stab_scaling(), fldparaxfem_->conv_stab_scaling(),
                      fldparaxfem_->xff_conv_stab_scaling(), my::fldpara_->is_conservative());

                  si_nit.at(coup_sid)->apply_conv_stab_terms(ci[coup_sid], my::funct_, my::velint_,
                      normal,
                      my::densaf_,  // Look into this term when changing HybridLM
                      NIT_full_stab_fac, timefacfac, ivelint_jump, cond_type);
                }
                else  // non-coupling
                {
                  Core::LinAlg::Matrix<nsd_, 1> velint_s;
                  ci[coup_sid]->get_interface_velnp(velint_s);

                  // Get Material parameters for the master side!
                  get_material_parameters_volume_cell(
                      mat, densaf_master_, viscaf_master_, gamma_m_);

                  bool non_xfluid_coupling;
                  double kappa_m;
                  double kappa_s;
                  cond_manager->get_average_weights(
                      coup_sid, ele, kappa_m, kappa_s, non_xfluid_coupling);

                  XFEM::Utils::nit_compute_full_penalty_stabfac(
                      NIT_full_stab_fac,  ///< to be filled: full Nitsche's penalty term scaling
                                          ///< (viscous+convective part)
                      normal, h_k,
                      kappa_m,  // weights (only existing for Nitsche currently!!)
                      kappa_s,  // weights (only existing for Nitsche currently!!)
                      my::convvelint_, velint_s,
                      NIT_visc_stab_fac,  ///< Nitsche's viscous scaling part of penalty term
                      my::fldparatimint_->time_fac(), my::fldparatimint_->is_stationary(),
                      densaf_master_, densaf_slave_, fldparaxfem_->mass_conservation_scaling(),
                      fldparaxfem_->mass_conservation_combination(),
                      fldparaxfem_->nit_stab_scaling(), fldparaxfem_->conv_stab_scaling(),
                      fldparaxfem_->xff_conv_stab_scaling(), my::fldpara_->is_conservative());

                  si_nit.at(coup_sid)->apply_conv_stab_terms(ci[coup_sid], my::funct_, my::velint_,
                      normal, my::densaf_, NIT_full_stab_fac, timefacfac, ivelint_jump, cond_type);
                }  // if coupling
              }  // if add conv stab

              if (my::fldparatimint_->is_new_ost_implementation())
              {
                FOUR_C_THROW(
                    "New OST for HybridLM not implemented - check out the code below this "
                    "FOUR_C_THROW!");
                //            // get velocity at integration point
                //            // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
                //            my::velintn_.multiply(eveln,my::funct_);
                //
                //            // get velocity derivatives at integration point
                //            // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
                //            my::vderxyn_.multiply_nt(eveln,my::derxy_);
                //
                //            //-----------------------------------------------------------------------------
                //            // evaluate the coupling terms for coupling with current side
                //            // (or embedded element through current side)
                //            // time step n
                //            const double kappa_m = 1.0;
                //            const double kappa_s = 0.0;
                //            get_material_parameters_volume_cell(mat,densaf_master_,viscaf_master_,gamma_m_);
                //
                //            // REMARK: do not add adjoint and penalty terms at t_n for hybrid LM
                //            approach!
                //            // (these are Nitsche-terms! find best settings for Nitsche's method
                //            first!) Core::LinAlg::Matrix<nsd_,1> ivelintn_jump (true);
                //            Core::LinAlg::Matrix<nsd_,1> itractionn_jump(true);
                //
                //            //Get Configuration Map (finally we should modify the configuration
                //            map here in a way that it fits hybrid LM approach)
                //            std::map<Inpar::XFEM::CoupTerm, std::pair<bool,double> >&
                //            hlm_configmap_n = coupling->GetConfigurationmap();
                //
                //            si_nit.at(coup_sid)->nit_evaluate_coupling_old_state(
                //              normal,
                //              surf_fac * (my::fldparatimint_->Dt()-my::fldparatimint_->TimeFac()),
                //              // scaling of rhs depending on time discretization scheme false,
                //              viscaf_master_,              // dynvisc viscosity in background
                //              fluid viscaf_slave_,               // dynvisc viscosity in embedded
                //              fluid kappa_m,                     // mortaring weighting kappa_s,
                //              // mortaring weighting my::densn_,                  // fluid density
                //              my::funct_,                  // bg shape functions
                //              my::derxy_,                  // bg shape function gradient
                //              my::vderxyn_,                // bg grad u^n
                //              my::funct_.Dot(epren),       // bg p^n
                //              my::velintn_,                // bg u^n
                //              ivelintn_jump,
                //              itractionn_jump,
                //              hlm_configmap_n,
                //              Inpar::XFEM::PreviousState_only_consistency
                //            );
              }
            }  // hybrid lm method

            if (!assemble_iforce) continue;

            //--------------------------------------------
            // calculate interface forces for XFSI

            // get velocity derivatives at integration point
            // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
            my::vderxy_.multiply_nt(evelaf, my::derxy_);

            // get pressure at integration point
            // (value at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
            double press = my::funct_.dot(epreaf);

            //-------------------------------
            // traction vector w.r.t fluid domain, resulting stresses acting on the fluid surface
            // t= (-p*I + 2mu*eps(u))*n^f
            Core::LinAlg::Matrix<nsd_, 1> traction(Core::LinAlg::Initialization::zero);

            build_traction_vector(traction, press, normal);

            ci[coup_sid]->compute_interface_force(iforce, traction, surf_fac);

          }  // end loop gauss points of boundary cell
        }  // end loop boundary cells of side

        if (assemble_iforce)
          assemble_interface_force(*mc_fsi->i_forcecol(), *cutter_dis, cutla[0].lm_, iforce);

      }  // end loop cut sides

      /*--------------------------------------------------------
        build final element and coupling matrices
      --------------------------------------------------------*/

      // compute inverse K_ss^-1
      Core::LinAlg::FixedSizeSerialDenseSolver<nen_, nen_> invsolver;
      invsolver.set_matrix(invbK_ss);
      invsolver.invert();

      // the non-diagonal entries (shear stresses) lead to the factor 2 in the
      // K_ss matrix; inversion leads to 1/2 in the matrix blocks of the shear stress
      halfInvbK_ss.update(0.5, invbK_ss, 0.0);

      invK_ss.add_view(Sigmaxx, Sigmaxx, invbK_ss);
      invK_ss.add_view(Sigmaxy, Sigmaxy, halfInvbK_ss);
      invK_ss.add_view(Sigmaxz, Sigmaxz, halfInvbK_ss);
      invK_ss.add_view(Sigmayy, Sigmayy, invbK_ss);
      invK_ss.add_view(Sigmayz, Sigmayz, halfInvbK_ss);
      invK_ss.add_view(Sigmazz, Sigmazz, invbK_ss);

      // create views
      Core::LinAlg::Matrix<numdofpernode_ * nen_, numdofpernode_ * nen_> elemat(
          elemat1_epetra, true);
      Core::LinAlg::Matrix<numdofpernode_ * nen_, 1> elevec(elevec1_epetra, true);

      // now the matrix products involving the inverse matrix will be computed!

      // REMARK: at this step, the K matrices already include contributions from surface terms G_us,
      // G_su
      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numdofpernode_, numstressdof_>
          KusinvKss;

      // (K_us + G_us) K_ss^-1 (MHVS) or G_us K_ss^-1 (MHCS)
      KusinvKss.multiply(K_us, invK_ss);

      // (K_us + G_us) K_ss^-1 (K_su + G_su) (MHVS) or G_us  K_ss^-1 (K_su + G_su + K_sp) (MHCS)
      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numdofpernode_, numdofpernode_>
          KusinvKssKsu;
      KusinvKssKsu.multiply(KusinvKss, K_su);

      // (K_us + G_us) K_ss^-1 rhs_s (MHVS) or G_us K_ss^-1 rhs_s (MHCS)
      Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numdofpernode_, 1> KusinvKssrhs_s;

      KusinvKssrhs_s.multiply(KusinvKss, rhs_s);

      // REMARK: The following term goes into the interface element's rhs-vector rhC_ui!
      /*
       *  G_uis  *(K_ss^-1 rhs_s)
       *  |___________________|
       *        |
       *        |
       *    -->side-impl
       */


      /*--------------------------------------------------------
        setup element matrix
      --------------------------------------------------------*/
      /*
       * Matrix of intersected fluid element for VISCOUS stress-based LM
       *  _                                                                          _
       * |                                                                            |
       *             _
       *      K_uu + K_uu - (K_us + G_us) K_ss^-1 (K_su + G_su) | K_up + G_up
       *
       *  ----------------------------------------------------- |------------
       *
       *      K_pu + G_pu                                       | K_pp
       * |_                                                                          _|
       *
       * Matrix of intersected fluid element for CAUCHY stress-based LM
       *
       *  _                                                                          _
       * |                                                                            |
       *
       *      K_uu  - G_us  K_ss^-1 (K_su + G_su) | K_up - G_us K_ss^-1 K_sp
       *
       *  ----------------------------------------|------------
       *
       *      K_pu                                | K_pp
       * |_                                                                          _|
       *
       */

      // -KusinvKssKsu --> -(K_us + G_us) K_ss^-1 (K_su + G_su) (MHVS) or -G_us  K_ss^-1 (K_su +
      // G_su) -G_us K_ss^-1 K_sp (MHCS)

      // complete the background element matrix with the calculated entries:

      // number of row blocks
      const unsigned numbrow = nsd_;  // no (p,u)-block for MHVS & MHCS
      // number of column blocks
      const unsigned numdofpernode =
          numdofpernode_;  // avoid possible linker error on some compilers
      const unsigned numbcol =
          (is_MHVS ? numbrow : numdofpernode);  // we have (u,p)-block for MHCS (-G_us K_ss^-1 K_sp)

      // loop over row blocks (only velocities)
      for (unsigned ibr = 0; ibr < numbrow; ++ibr)
      {
        // loop over column blocks (only velocities for MHVS!)
        for (unsigned ibc = 0; ibc < numbcol; ++ibc)
        {
          // add -KusinvKssKsu
          if (KusinvKssKsu.is_used(ibr, ibc))
          {
            Core::LinAlg::Matrix<nen_, nen_>& bKusinvKssKsu = *KusinvKssKsu(ibr, ibc);

            for (int ir = 0; ir < nen_; ++ir)
            {
              // row position in the final element matrix
              unsigned row = ibr + ir * numdofpernode_;

              for (int ic = 0; ic < nen_; ++ic)
              {
                // column position in the final element matrix
                unsigned col = ibc + ic * numdofpernode_;

                // - (K_us + G_us) K_ss^-1 (K_su + G_su ) (MHVS) or
                // - G_us  K_ss^-1 (K_su + G_su + K_sp) (MHCS)
                elemat(row, col) -= bKusinvKssKsu(ir, ic);
              }
            }
          }  // -KusinvKssKsu

          if (!is_MHVS) continue;

          /*
           * ONLY MHVS:
           * add contribution from viscous term scaled by (-1/n) (n: MHVS-parameter):
           * _
           * K_uu = (-1/n) * K_uu^{viscous}
           *
           * */

          if (K_uu.is_used(ibr, ibc))
          {
            Core::LinAlg::Matrix<nen_, nen_>& bK_uu = *K_uu(ibr, ibc);

            for (int ir = 0; ir < nen_; ++ir)
            {
              // row position in the final element matrix
              unsigned velrow = ibr + ir * numdofpernode_;

              for (int ic = 0; ic < nen_; ++ic)
              {
                // column position in the final element matrix
                unsigned velcol = ibc + ic * numdofpernode_;

                // + K_uu
                elemat(velrow, velcol) += bK_uu(ir, ic);
              }
            }
          }  // K_uu
        }  // end column block loop

        if (!is_MHVS) continue;

        // ONLY MHVS: velocity-pressure coupling entries G_up
        // loop over row blocks
        if (G_up.is_used(ibr, 0))
        {
          Core::LinAlg::Matrix<nen_, nen_>& bGup = *G_up(ibr, 0);

          for (int ic = 0; ic < nen_; ++ic)
          {
            // column position in final element matrix
            unsigned prescol = nsd_ + ic * numdofpernode_;

            // loop over rows of velocity-pressure submatrix
            for (int ir = 0; ir < nen_; ++ir)
            {
              // row position in the final element matrix
              unsigned velrow = ibr + ir * numdofpernode_;

              // + G_up
              elemat(velrow, prescol) += bGup(ir, ic);
            }
          }
        }  // G_up
      }  // end row block loop

      if (is_MHVS)
      {
        // ONLY MHVS: pressure-velocity coupling entries G_pu
        // loop over column blocks
        for (unsigned ibc = 0; ibc < numbcol; ++ibc)
        {
          if (G_pu.is_used(0, ibc))
          {
            Core::LinAlg::Matrix<nen_, nen_>& bGpu = *G_pu(0, ibc);

            // pressure-velocity entries
            for (int ir = 0; ir < nen_; ++ir)
            {
              // row position in final element matrix
              unsigned presrow = nsd_ + ir * numdofpernode_;

              for (int ic = 0; ic < nen_; ++ic)
              {
                // column position in final element matrix
                unsigned velcol = ibc + ic * numdofpernode_;

                // + Gpu
                elemat(presrow, velcol) += bGpu(ir, ic);
              }
            }
          }  // G_pu
        }  // end column block loop
      }
      // element matrix complete!

      /*--------------------------------------------------------
        setup rhs-vector
      --------------------------------------------------------*/
      /*
       *  for MHVS:
       *  - (K_us + G_us) K_ss^-1 * rhs_s + rhs_uu + rhs_up
       *          +
       *  rhs_pu + rhs_pui
       *  |_______________|
       *        |
       *       united in rhs_pu!
       *
       *  for MHCS:
       *  - G_us K_ss^-1 * rhs_s
       *
       */
      // loop over row blocks
      for (unsigned ibr = 0; ibr < numbrow; ++ibr)
      {
        if (KusinvKssrhs_s.is_used(ibr, 0))
        {
          Core::LinAlg::Matrix<nen_, 1>& bKusinvKssrhs_s = *KusinvKssrhs_s(ibr, 0);

          for (int ir = 0; ir < nen_; ++ir)
          {
            // row position in final element matrix
            unsigned int velrow = ibr + ir * numdofpernode_;

            // - (K_us + G_us) K_ss^-1 * rhs_s (MHVS) or
            // - G_us K_ss^-1 * rhs_s (MHCS)
            elevec(velrow, 0) -= bKusinvKssrhs_s(ir, 0);
          }
        }  // -KusinvKssrhs_s

        // ONLY MHVS!
        if (!is_MHVS) continue;

        if (rhs_uu.is_used(ibr, 0))
        {
          Core::LinAlg::Matrix<nen_, 1>& brhs_uu = *rhs_uu(ibr, 0);

          for (int ir = 0; ir < nen_; ++ir)
          {
            // row position in final element matrix
            unsigned velrow = ibr + ir * numdofpernode_;

            // + rhs_uu
            elevec(velrow, 0) += brhs_uu(ir, 0);
          }
        }  // rhs_uu

        if (rhs_up.is_used(ibr, 0))
        {
          Core::LinAlg::Matrix<nen_, 1>& brhs_up = *rhs_up(ibr, 0);

          for (int ir = 0; ir < nen_; ++ir)
          {
            // row position in final element matrix
            unsigned int velrow = ibr + ir * numdofpernode_;
            // + rhs_up
            elevec(velrow, 0) += brhs_up(ir, 0);
          }
        }  // rhs_up
      }  // end row block loop

      if (is_MHVS)
      {
        // add rhs_pu
        for (int ir = 0; ir < nen_; ++ir)
        {
          // row position in final element matrix
          unsigned int presrow = nsd_ + ir * numdofpernode_;
          // + rhs_pu + rhs_pui
          elevec(presrow, 0) += rhs_pu(ir, 0);
        }  // rhs_pu
      }

      // coupling contributions are added matrix & rhs-vector for background element!

      // in case that no coupling objects are available, we are done here
      if (side_coupling.empty()) return;

      //-------------------------------------------------
      // finalize creation of side coupling terms
      // Cuiu, Cuui, rhCui & Gsui, Guis (-->Cuiui)
      //-------------------------------------------------

      // build interface coupling matrices - therefore iterate through the interface elements
      for (typename std::map<int,
               std::shared_ptr<Discret::Elements::XFLUID::HybridLMInterface<distype>>>::iterator
               sit = ci.begin();
          sit != ci.end(); ++sit)
      {
        std::shared_ptr<Discret::Elements::XFLUID::HybridLMInterface<distype>> si = sit->second;
        const int coup_sid = sit->first;

        // creation of Cuiu,Cuui,rhCui,Guis and Gsui:
        si->hybrid_lm_build_final_coupling_matrices(invK_ss, KusinvKss, K_su, rhs_s);

        // add contributions from convective stabilization, if active
        if (add_conv_stab)
        {
          std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::iterator c =
              side_coupling.find(coup_sid);
          std::vector<Core::LinAlg::SerialDenseMatrix>& side_matrices = c->second;
          std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::iterator cc =
              side_coupling_extra.find(coup_sid);
          std::vector<Core::LinAlg::SerialDenseMatrix>& side_matrices_extra = cc->second;
#ifdef FOUR_C_ENABLE_ASSERTIONS
          if (side_matrices.size() != 3)
            FOUR_C_THROW(
                "Obtained only {} side coupling matrices. 3 required.", side_matrices.size());
          if (side_matrices_extra.size() != 4)
            FOUR_C_THROW("Obtained only {} conv. side coupling matrices. 4 required.",
                side_matrices_extra.size());
#endif

          for (int i = 0; i < 3; ++i)
          {
#ifdef FOUR_C_ENABLE_ASSERTIONS
            if (side_matrices[i].numRows() != side_matrices_extra[i].numRows() ||
                side_matrices[i].numCols() != side_matrices_extra[i].numCols())
              FOUR_C_THROW(
                  "Mismatch in matrix dimensions of convective stabilization matrix and MHCS/MHVS "
                  "coupling matrix");
#endif
            side_matrices[i] += side_matrices_extra[i];
          }

          // in case of new OST and with active convective stab. terms, we have already
          //
          continue;
        }

        // add contributions from old time step to RHS
        if (my::fldparatimint_->is_new_ost_implementation())
        {
          std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::iterator c =
              side_coupling.find(coup_sid);
          std::vector<Core::LinAlg::SerialDenseMatrix>& side_matrices = c->second;
          std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::iterator cc =
              side_coupling_extra.find(coup_sid);
          std::vector<Core::LinAlg::SerialDenseMatrix>& side_matrices_extra = cc->second;
#ifdef FOUR_C_ENABLE_ASSERTIONS
          if (side_matrices.size() != 3)
            FOUR_C_THROW(
                "Obtained only {} side coupling matrices. 3 required.", side_matrices.size());
          if (side_matrices_extra.size() != 4)
            FOUR_C_THROW("Obtained only {} conv. side coupling matrices. 4 required.",
                side_matrices_extra.size());
          if (side_matrices[2].numRows() != side_matrices_extra[2].numRows() ||
              side_matrices[2].numCols() != side_matrices_extra[2].numCols())
            FOUR_C_THROW(
                "Mismatch in matrix dimensions of convective stabilization matrix and MHCS/MHVS "
                "coupling matrix");
#endif

          // we only add the RHS contribution
          side_matrices[2] += side_matrices_extra[2];
        }
      }  // end loop over map of coupling matrices

      //-------------------------------------------------
      // build C_uiui
      //-------------------------------------------------

      /*
       * "patchelementslm" includes u,v,w,p- DOFs
       * Therefore, the final matrices G_sui and G_uis have pressure columns/rows.
       * The pressure entries are set to zero.
       * The task is now, to build the final G_uis and G_sui matrices
       * out of the element submatrices collected in vector 'Cuiui_matrices'.
       * In there, the G_sui & G_uis contributions from the sides are collected!
       */
      Core::LinAlg::SerialDenseMatrix G_sui(numstressdof_ * nen_, patchelementslm.size());
      Core::LinAlg::SerialDenseMatrix G_uis(patchelementslm.size(), numstressdof_ * nen_);
      Core::LinAlg::SerialDenseMatrix Cuiui_conv(patchelementslm.size(), patchelementslm.size());

      // transform the block matrix invK_ss to an EpetraSerialDenseMatrix,
      // to be later multiplied with G_sui & G_uis!
      Core::LinAlg::SerialDenseMatrix InvKss(nen_ * numstressdof_, nen_ * numstressdof_);

      //--------------------------------------------
      // Build InvKss ( K_ss^(-1) )
      //--------------------------------------------
      for (int ibc = 0; ibc < numstressdof_; ++ibc)
      {
        for (int ibr = 0; ibr < numstressdof_; ++ibr)
        {
          if (invK_ss.is_used(ibr, ibc))
          {
            Core::LinAlg::Matrix<nen_, nen_>& binvK_ss = *invK_ss(ibr, ibc);
            for (int ic = 0; ic < nen_; ++ic)
            {
              unsigned col = ibc + numstressdof_ * ic;
              for (int ir = 0; ir < nen_; ++ir)
              {
                unsigned row = ibr + numstressdof_ * ir;
                InvKss(row, col) -= binvK_ss(ir, ic);
              }
            }
          }
        }
      }

      //--------------------------------------------
      // assemble G_sui & G_uis
      //--------------------------------------------
      int ipatchsizesbefore = 0;

      for (std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::const_iterator m =
               Cuiui_coupling.begin();
          m != Cuiui_coupling.end(); ++m)
      {
        const std::vector<Core::LinAlg::SerialDenseMatrix>& Cuiui_matrices = m->second;

        // assemble Gsui
        for (int ibc = 0; ibc < Cuiui_matrices[0].numCols(); ++ibc)
        {
          for (int ibr = 0; ibr < numstressdof_ * nen_; ++ibr)
          {
            G_sui(ibr, ibc + ipatchsizesbefore) = Cuiui_matrices[0](ibr, ibc);
          }
        }

        // assemble Guis
        for (int ibc = 0; ibc < numstressdof_ * nen_; ++ibc)
        {
          for (int ibr = 0; ibr < Cuiui_matrices[1].numRows(); ++ibr)
          {
            G_uis(ibr + ipatchsizesbefore, ibc) = Cuiui_matrices[1](ibr, ibc);
          }
        }

        if (add_conv_stab)
        {
          const int coup_sid = m->first;
          std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::const_iterator c =
              side_coupling_extra.find(coup_sid);
          const std::vector<Core::LinAlg::SerialDenseMatrix>& Cuiui_conv_matrices = c->second;

          for (int ibc = 0; ibc < Cuiui_conv_matrices[3].numCols(); ++ibc)
          {
            for (int ibr = 0; ibr < Cuiui_conv_matrices[3].numRows(); ++ibr)
            {
              Cuiui_conv(ibr + ipatchsizesbefore, ibc + ipatchsizesbefore) =
                  Cuiui_conv_matrices[3](ibr, ibc);
            }
          }
        }

        ipatchsizesbefore += Cuiui_matrices[0].numCols();
      }

      Core::LinAlg::SerialDenseMatrix GuisInvKss(patchelementslm.size(), numstressdof_ * nen_);

      // G_uis * K_ss^-1
      Core::LinAlg::multiply(1.0, GuisInvKss, 1.0, G_uis, InvKss);

      // Cuiui <--> (-)G_uis * K_ss^-1 * G_sui
      Core::LinAlg::multiply(1.0, Cuiui, 1.0, GuisInvKss, G_sui);

      if (add_conv_stab) Cuiui += Cuiui_conv;
    }

    /*--------------------------------------------------------------------------------
     * setup volume-based terms
     * (mixed/hybrid viscous stress-based LM approach)
     *-------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::hybrid_lm_build_vol_based(
        const std::vector<Core::FE::GaussIntegration>& intpoints,
        const Cut::plain_volumecell_set& cells,
        const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,  ///< element velocity
        const Core::LinAlg::Matrix<nen_, 1>& epreaf,     ///< element pressure
        Core::LinAlg::Matrix<nen_, nen_>& bK_ss,         ///< block K_ss matrix
        Core::LinAlg::Matrix<nen_, nen_>& invbK_ss,      ///< inverse of block K_ss matrix
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_, numdofpernode_>&
            K_su,  ///< K_su matrix
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>&
            rhs_s,  ///< rhs_s vector
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numdofpernode_, numstressdof_>&
            K_us,  ///< K_us matrix
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, nsd_, nsd_>&
            K_uu,  ///< K_uu matrix
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, nsd_, 1>&
            rhs_uu,              ///< rhs_u(u) vector
        const bool is_MHVS,      ///< viscous (true) or Cauchy (false) stress-based LM
        const double mhvs_param  ///< stabilizing parameter for viscous stress-based LM
    )
    {
      // full L2-projection means integration over the full background element,
      // not only the physical part
      if (fldparaxfem_->hybrid_lm_l2_proj() == Inpar::XFEM::Hybrid_LM_L2_Proj_full)
      {
        // get the standard set of gauss-points from the intersected element
        for (Core::FE::GaussIntegration::const_iterator iquad = my::intpoints_.begin();
            iquad != my::intpoints_.end(); ++iquad)
        {
          my::eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

          if (is_MHVS)
          {
            mhvs_evaluate_vol_based(
                evelaf, bK_ss, invbK_ss, K_su, rhs_s, K_us, K_uu, rhs_uu, mhvs_param);
          }
          else
          {
            mhcs_evaluate_vol_based(evelaf, epreaf, bK_ss, invbK_ss, K_su, rhs_s);
          }
        }
      }
      else
      {
        for (std::vector<Core::FE::GaussIntegration>::const_iterator i = intpoints.begin();
            i != intpoints.end(); ++i)
        {
          const Core::FE::GaussIntegration intcell = *i;
          for (Core::FE::GaussIntegration::iterator iquad = intcell.begin(); iquad != intcell.end();
              ++iquad)
          {
            // evaluate shape functions and derivatives at integration point
            my::eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());
            if (is_MHVS)
            {
              mhvs_evaluate_vol_based(
                  evelaf, bK_ss, invbK_ss, K_su, rhs_s, K_us, K_uu, rhs_uu, mhvs_param);
            }
            else
            {
              mhcs_evaluate_vol_based(evelaf, epreaf, bK_ss, invbK_ss, K_su, rhs_s);
            }
          }
        }
      }

      return;
    }

    /*--------------------------------------------------------------------------------
     * evaluate volume-based terms for current gauss point
     * (mixed/hybrid Cauchy stress-based LM approach)
     *-------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::mhcs_evaluate_vol_based(
        const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,  ///< element velocity
        const Core::LinAlg::Matrix<nen_, 1>& epreaf,     ///< element pressure
        Core::LinAlg::Matrix<nen_, nen_>& bK_ss,         ///< block K_ss matrix
        Core::LinAlg::Matrix<nen_, nen_>& invbK_ss,      ///< inverse of block K_ss matrix
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_, numdofpernode_>&
            K_su,  ///< K_su matrix
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>&
            rhs_s  ///< rhs_s vector
    )
    {
      Core::LinAlg::Matrix<nen_, 1> dx;
      Core::LinAlg::Matrix<nen_, 1> dy;
      Core::LinAlg::Matrix<nen_, 1> dz;

      Core::LinAlg::Matrix<nen_, nen_> conv_x;
      Core::LinAlg::Matrix<nen_, nen_> conv_y;
      Core::LinAlg::Matrix<nen_, nen_> conv_z;

      //----------------------------------------------------------------------
      // set time-integration factors for left- and right-hand side
      // (two right-hand-side factors: general and for residuals)
      //----------------------------------------------------------------------

      const double viscfac = 1.0 / (2.0 * my::visceff_);

      // get velocity at integration point
      // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
      my::velint_.multiply(evelaf, my::funct_);

      // get velocity derivatives at integration point
      // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
      my::vderxy_.multiply_nt(evelaf, my::derxy_);

      // get pressure at integration point
      // (value at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
      double press = my::funct_.dot(epreaf);

      // time integration factor & spatial integration factor
      const double timefacfac = my::fldparatimint_->time_fac() * my::fac_;


      //--------------------------------------------

      for (int i = 0; i < nen_; ++i)
      {
        dx(i) = my::derxy_(0, i);
        dy(i) = my::derxy_(1, i);
        dz(i) = my::derxy_(2, i);
      }

      // block - K_ss
      bK_ss.multiply_nt(my::funct_, my::funct_);


      conv_x.multiply_nt(my::funct_, dx);
      conv_y.multiply_nt(my::funct_, dy);
      conv_z.multiply_nt(my::funct_, dz);

      /*                     \
    - |  virt tau , eps(Dtau)  |
      \                     */

      invbK_ss.update(-viscfac * timefacfac, bK_ss, 1.0);

      /*                 \
     | virt tau , eps(Du) |
      \                 */


      // K_su

      K_su(Sigmaxx, Velx)->update(timefacfac, conv_x, 1.0);
      K_su(Sigmaxy, Velx)->update(timefacfac, conv_y, 1.0);
      K_su(Sigmayx, Vely)->update(timefacfac, conv_x, 1.0);
      K_su(Sigmaxz, Velx)->update(timefacfac, conv_z, 1.0);
      K_su(Sigmazx, Velz)->update(timefacfac, conv_x, 1.0);
      K_su(Sigmayy, Vely)->update(timefacfac, conv_y, 1.0);
      K_su(Sigmayz, Vely)->update(timefacfac, conv_z, 1.0);
      K_su(Sigmazy, Velz)->update(timefacfac, conv_y, 1.0);
      K_su(Sigmazz, Velz)->update(timefacfac, conv_z, 1.0);

      // r_su

      rhs_s(Sigmaxx, 0)->update(-timefacfac * my::vderxy_(0, 0), my::funct_, 1.0);
      rhs_s(Sigmaxy, 0)
          ->update(-timefacfac * (my::vderxy_(0, 1) + my::vderxy_(1, 0)), my::funct_, 1.0);
      rhs_s(Sigmaxz, 0)
          ->update(-timefacfac * (my::vderxy_(0, 2) + my::vderxy_(2, 0)), my::funct_, 1.0);
      rhs_s(Sigmayy, 0)->update(-timefacfac * my::vderxy_(1, 1), my::funct_, 1.0);
      rhs_s(Sigmayz, 0)
          ->update(-timefacfac * (my::vderxy_(1, 2) + my::vderxy_(2, 1)), my::funct_, 1.0);
      rhs_s(Sigmazz, 0)->update(-timefacfac * my::vderxy_(2, 2), my::funct_, 1.0);

      // stressbar-pressure coupling
      /*
                      /                    \
                     |                      |
                   - | tr(virt tau^e) , p I |
                     |                      |
                      \                    /
      */

      // K_sp
      K_su(Sigmaxx, Pres)->update(-viscfac * timefacfac, bK_ss, 1.0);
      K_su(Sigmayy, Pres)->update(-viscfac * timefacfac, bK_ss, 1.0);
      K_su(Sigmazz, Pres)->update(-viscfac * timefacfac, bK_ss, 1.0);

      // r_sp
      rhs_s(Sigmaxx, 0)->update(viscfac * timefacfac * press, my::funct_, 1.0);
      rhs_s(Sigmayy, 0)->update(viscfac * timefacfac * press, my::funct_, 1.0);
      rhs_s(Sigmazz, 0)->update(viscfac * timefacfac * press, my::funct_, 1.0);

      return;

    }  // EvaluateMatricesMSH

    /*--------------------------------------------------------------------------------
     * evaluate volume-based matrices K for current gauss point
     * (mixed/hybrid viscous stress-based LM approach)
     *-------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::mhvs_evaluate_vol_based(
        const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, Core::LinAlg::Matrix<nen_, nen_>& bK_ss,
        Core::LinAlg::Matrix<nen_, nen_>& invbK_ss,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_, numdofpernode_>&
            K_su,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>& rhs_s,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numdofpernode_, numstressdof_>&
            K_us,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, nsd_, nsd_>& K_uu,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, nsd_, 1>& rhs_uu,
        const double& mhvs_param)
    {
      // velocities at current gauss point
      my::velint_.multiply(evelaf, my::funct_);

      // velocity gradient at current gauss point
      my::vderxy_.multiply_nt(evelaf, my::derxy_);

      // compute shape function derivatives:
      // get derivatives of nodal shape function vector w. r. t. x,y,z
      Core::LinAlg::Matrix<nen_, 1> dx;
      Core::LinAlg::Matrix<nen_, 1> dy;
      Core::LinAlg::Matrix<nen_, 1> dz;

      // derxy(coordinate,node)
      for (int i = 0; i < nen_; ++i)
      {
        dx(i) = my::derxy_(0, i);
        dy(i) = my::derxy_(1, i);
        dz(i) = my::derxy_(2, i);
      }

      // fill the (nen_ x nen_) matrix block of K_ss
      bK_ss.multiply_nt(my::funct_, my::funct_);  // N * N^T

      // scaling with inverse dynamic effective viscosity
      const double viscfac = -1.0 / (2.0 * my::visceff_);

      // time integration factor & spatial integration factor, scaled with mhvs-parameter 1/n
      const double timefacfac = my::fldparatimint_->time_fac() * my::fac_ / mhvs_param;


      /* K_ss
       *
       *       1        /                \
       * (-)  ---   *  |  \tau, \sigma    |
       *   (2n \mu)     \                /
       */
      invbK_ss.update(viscfac * timefacfac, bK_ss, 1.0);

      /*
       *  K_su
       *  from:
       *
       *  1     /                    \
       *  - *  |  \tau, \epsilon(u)   |
       *  n     \                    /
       *
       */

      // create blocks
      Core::LinAlg::Matrix<nen_, nen_> NdNdxT;
      Core::LinAlg::Matrix<nen_, nen_> NdNdyT;
      Core::LinAlg::Matrix<nen_, nen_> NdNdzT;

      NdNdxT.multiply_nt(my::funct_, dx);
      NdNdyT.multiply_nt(my::funct_, dy);
      NdNdzT.multiply_nt(my::funct_, dz);

      // add main diagonal blocks
      K_su(Sigmaxx, Velx)->update(timefacfac, NdNdxT, 1.0);
      K_su(Sigmayy, Vely)->update(timefacfac, NdNdyT, 1.0);
      K_su(Sigmazz, Velz)->update(timefacfac, NdNdzT, 1.0);

      // add off-diagonal blocks
      K_su(Sigmaxy, Velx)->update(timefacfac, NdNdyT, 1.0);
      K_su(Sigmaxz, Velx)->update(timefacfac, NdNdzT, 1.0);
      K_su(Sigmayx, Vely)->update(timefacfac, NdNdxT, 1.0);
      K_su(Sigmayz, Vely)->update(timefacfac, NdNdzT, 1.0);
      K_su(Sigmazx, Velz)->update(timefacfac, NdNdxT, 1.0);
      K_su(Sigmazy, Velz)->update(timefacfac, NdNdyT, 1.0);

      /*
       * rhs_s contribution
       * from:
       *
       *  1     /                   \
       *  -  * |  \tau, \epsilon(u)  |
       *  n     \                   /
       *
       */

      rhs_s(Sigmaxx, 0)->update(-timefacfac * my::vderxy_(0, 0), my::funct_, 1.0);
      rhs_s(Sigmaxy, 0)
          ->update(-timefacfac * (my::vderxy_(1, 0) + my::vderxy_(0, 1)), my::funct_, 1.0);
      rhs_s(Sigmaxz, 0)
          ->update(-timefacfac * (my::vderxy_(0, 2) + my::vderxy_(2, 0)), my::funct_, 1.0);
      rhs_s(Sigmayy, 0)->update(-timefacfac * my::vderxy_(1, 1), my::funct_, 1.0);
      rhs_s(Sigmayz, 0)
          ->update(-timefacfac * (my::vderxy_(1, 2) + my::vderxy_(2, 1)), my::funct_, 1.0);
      rhs_s(Sigmazz, 0)->update(-timefacfac * my::vderxy_(2, 2), my::funct_, 1.0);

      // nothing like K_sp here, as this is a purely viscous stress-based approach

      /*
       * coupling matrix K_us which results from testing the LM-stress field with
       * test strains computed from the test velocities
       *
       *     1     /                         \
       *     -  * |  \epsilon(v), \sigma      | * \alpha
       *     n     \                         /
       *
       */

      // as the viscous stresses are condensed, there is no contribution of this term to
      // the RHS vector
      Core::LinAlg::Matrix<nen_, nen_> dNdxNT;
      Core::LinAlg::Matrix<nen_, nen_> dNdyNT;
      Core::LinAlg::Matrix<nen_, nen_> dNdzNT;

      dNdxNT.update_t(NdNdxT);
      dNdyNT.update_t(NdNdyT);
      dNdzNT.update_t(NdNdzT);

      // leads to terms, that are analogous to a symmetric/non-symmetric Nitsche-formulation
      // REMARK: behaves unstable for betau=-1.0 in fluid-fluid problems, so keep that in mind!
      // betau (-)1 <--> symmetric Nitsche
      // betau (+)1 <--> non-symmetric Nitsche
      // the stabilizing parameter n has been applied to timefacfac

      const double alpha = fldparaxfem_->is_viscous_adjoint_symmetric() ? 1.0 : -1.0;

      // add main diagonal submatrices
      K_us(Velx, Sigmaxx)->update(alpha * timefacfac, dNdxNT, 1.0);
      K_us(Vely, Sigmayy)->update(alpha * timefacfac, dNdyNT, 1.0);
      K_us(Velz, Sigmazz)->update(alpha * timefacfac, dNdzNT, 1.0);

      // add off-diagonal blocks
      K_us(Velx, Sigmaxy)->update(alpha * timefacfac, dNdyNT, 1.0);
      K_us(Velx, Sigmaxz)->update(alpha * timefacfac, dNdzNT, 1.0);
      K_us(Vely, Sigmayx)->update(alpha * timefacfac, dNdxNT, 1.0);
      K_us(Vely, Sigmayz)->update(alpha * timefacfac, dNdzNT, 1.0);
      K_us(Velz, Sigmazx)->update(alpha * timefacfac, dNdxNT, 1.0);
      K_us(Velz, Sigmazy)->update(alpha * timefacfac, dNdyNT, 1.0);

      // computation of additional stress term, scaled with inverse MHVS-parameter

      // build K_uu coupling matrix
      /*
       *   /                           \     1
       * -|  \epsilon(v), \epsilon(u)   | *  - * 2 \mu * \alpha
       *   \                           /     n
       *
       */

      // factor 2 from above is cancelled out
      const double visc_timefac_mhvs = -alpha * my::visceff_ * timefacfac;

      std::vector<const Core::LinAlg::Matrix<nen_, 1>*> dN;
      dN.push_back(&dx);
      dN.push_back(&dy);
      dN.push_back(&dz);

      Core::LinAlg::Matrix<nen_, nen_> dNidxj(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nen_, nen_> dNjdxj(Core::LinAlg::Initialization::zero);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          dNidxj.multiply_nt(*dN[jdim], *dN[idim]);
          K_uu(idim, jdim)->update(visc_timefac_mhvs, dNidxj, 1.0);
          dNjdxj.multiply_nt(*dN[jdim], *dN[jdim]);
          K_uu(idim, idim)->update(visc_timefac_mhvs, dNjdxj, 1.0);
          rhs_uu(idim, 0)->update(
              -visc_timefac_mhvs * (my::vderxy_(idim, jdim) + my::vderxy_(jdim, idim)), *dN[jdim],
              1.0);
        }
      }
    }

    /*--------------------------------------------------------------------------------
     * build surface-based terms for current gauss point
     * (mixed/hybrid Cauchy or viscous stress-based LM approach)
     *-------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::hybrid_lm_evaluate_surf_based(
        Discret::Elements::XFLUID::HybridLMInterface<distype>& si,
        const Core::LinAlg::Matrix<nen_, nen_>& bK_ss,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_, numdofpernode_>&
            K_su,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numdofpernode_, numstressdof_>&
            K_us,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>& rhs_s,
        const Core::LinAlg::Matrix<nen_, 1>& epreaf,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, nsd_, nsd_>& K_uu,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, nsd_, 1>& rhs_uu,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, nsd_, 1>& G_up,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, 1, nsd_>& G_pu,
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, nsd_, 1>& rhs_up,
        Core::LinAlg::Matrix<nen_, 1>& rhs_pu, const Core::LinAlg::Matrix<nsd_, 1>& normal,
        const double& timesurffac, const Core::LinAlg::Matrix<nsd_, 1>& ivelint_jump,
        const Core::LinAlg::Matrix<nsd_, 1>& itraction_jump, const bool eval_side_coupling,
        const bool is_MHVS)
    {
      K_us(Velx, Sigmaxx)->update(-timesurffac * normal(Velx), bK_ss, 1.0);
      K_us(Velx, Sigmaxy)->update(-timesurffac * normal(Vely), bK_ss, 1.0);
      K_us(Velx, Sigmaxz)->update(-timesurffac * normal(Velz), bK_ss, 1.0);
      K_us(Vely, Sigmayx)->update(-timesurffac * normal(Velx), bK_ss, 1.0);
      K_us(Vely, Sigmayy)->update(-timesurffac * normal(Vely), bK_ss, 1.0);
      K_us(Vely, Sigmayz)->update(-timesurffac * normal(Velz), bK_ss, 1.0);
      K_us(Velz, Sigmazx)->update(-timesurffac * normal(Velx), bK_ss, 1.0);
      K_us(Velz, Sigmazy)->update(-timesurffac * normal(Vely), bK_ss, 1.0);
      K_us(Velz, Sigmazz)->update(-timesurffac * normal(Velz), bK_ss, 1.0);


      // K_su - add the blocks from the surface contribution
      // viscous stress-tested interface continuity term
      /*
       *
       *     /                             \
       *   - |       \tau_{ij} * n_j, u^i   |
       *     \                             /
       */
      K_su(Sigmaxx, Velx)->update(-timesurffac * normal(Velx), bK_ss, 1.0);
      K_su(Sigmaxy, Velx)->update(-timesurffac * normal(Vely), bK_ss, 1.0);
      K_su(Sigmaxz, Velx)->update(-timesurffac * normal(Velz), bK_ss, 1.0);
      K_su(Sigmayx, Vely)->update(-timesurffac * normal(Velx), bK_ss, 1.0);
      K_su(Sigmayy, Vely)->update(-timesurffac * normal(Vely), bK_ss, 1.0);
      K_su(Sigmayz, Vely)->update(-timesurffac * normal(Velz), bK_ss, 1.0);
      K_su(Sigmazx, Velz)->update(-timesurffac * normal(Velx), bK_ss, 1.0);
      K_su(Sigmazy, Velz)->update(-timesurffac * normal(Vely), bK_ss, 1.0);
      K_su(Sigmazz, Velz)->update(-timesurffac * normal(Velz), bK_ss, 1.0);

      // Add surface integral contribution to rhs_s

      // from diagonal terms
      rhs_s(Sigmaxx, 0)->update(timesurffac * normal(Velx) * my::velint_(Velx), my::funct_, 1.0);
      rhs_s(Sigmayy, 0)->update(timesurffac * normal(Vely) * my::velint_(Vely), my::funct_, 1.0);
      rhs_s(Sigmazz, 0)->update(timesurffac * normal(Velz) * my::velint_(Velz), my::funct_, 1.0);

      // from off-diagonal terms
      rhs_s(Sigmaxy, 0)
          ->update(
              timesurffac * (normal(Vely) * my::velint_(Velx) + normal(Velx) * my::velint_(Vely)),
              my::funct_, 1.0);
      rhs_s(Sigmaxz, 0)
          ->update(
              timesurffac * (normal(Velz) * my::velint_(Velx) + normal(Velx) * my::velint_(Velz)),
              my::funct_, 1.0);
      rhs_s(Sigmayz, 0)
          ->update(
              timesurffac * (normal(Velz) * my::velint_(Vely) + normal(Vely) * my::velint_(Velz)),
              my::funct_, 1.0);

      // get pressure at current integration point
      double press = my::funct_.dot(epreaf);

      // MHVS terms
      if (is_MHVS)
      {
        // interface pressure term
        /*
         *
         *     /          \
         *    |   v, p n   |
         *     \          /
         */
        G_up(Velx, 0)->update(timesurffac * normal(0), bK_ss, 1.0);
        G_up(Vely, 0)->update(timesurffac * normal(1), bK_ss, 1.0);
        G_up(Velz, 0)->update(timesurffac * normal(2), bK_ss, 1.0);

        // velocity residual rhs_up
        rhs_up(Velx, 0)->update(-timesurffac * normal(Velx) * press, my::funct_, 1.0);
        rhs_up(Vely, 0)->update(-timesurffac * normal(Vely) * press, my::funct_, 1.0);
        rhs_up(Velz, 0)->update(-timesurffac * normal(Velz) * press, my::funct_, 1.0);

        // pressure-tested interface continuity term
        /*
         *
         *      /         \
         *    -|  q n, u   |
         *      \         /
         */
        G_pu(0, Velx)->update(-timesurffac * normal(0), bK_ss, 1.0);
        G_pu(0, Vely)->update(-timesurffac * normal(1), bK_ss, 1.0);
        G_pu(0, Velz)->update(-timesurffac * normal(2), bK_ss, 1.0);

        // pressure residual rhs_pu
        // this results from -(q, u_i * n_i)_{\Gamma} (pressure-tested kinematic continuity)
        const double normalvel = my::velint_.dot(normal);
        rhs_pu.update(timesurffac * normalvel, my::funct_, 1.0);
      }

      // the terms involving side-DOF are treated by the side implementation class,
      // as the sides have their own shape functions!
      if (eval_side_coupling)
      {
        if (is_MHVS)
          si.mhvs_build_coupling_matrices(
              normal, timesurffac, my::funct_, rhs_s, press, rhs_pu, ivelint_jump, itraction_jump);
        else
          si.mhcs_build_coupling_matrices(
              normal, timesurffac, my::funct_, rhs_s, ivelint_jump, itraction_jump);
      }
      else
      {
        // from velocity jump tested with viscous test stresses
        /*
         *
         *     /                    _     \
         *   +|  \tau_{ij} * n_j,   u^i    |
         *     \                          /
         */

        // add surface integral contribution to rhs_s

        // from diagonal terms
        rhs_s(Sigmaxx, 0)
            ->update(-timesurffac * normal(Velx) * ivelint_jump(Velx), my::funct_, 1.0);
        rhs_s(Sigmayy, 0)
            ->update(-timesurffac * normal(Vely) * ivelint_jump(Vely), my::funct_, 1.0);
        rhs_s(Sigmazz, 0)
            ->update(-timesurffac * normal(Velz) * ivelint_jump(Velz), my::funct_, 1.0);

        // from off-diagonal terms
        rhs_s(Sigmaxy, 0)
            ->update(-timesurffac *
                         (normal(Vely) * ivelint_jump(Velx) + normal(Velx) * ivelint_jump(Vely)),
                my::funct_, 1.0);
        rhs_s(Sigmaxz, 0)
            ->update(-timesurffac *
                         (normal(Velz) * ivelint_jump(Velx) + normal(Velx) * ivelint_jump(Velz)),
                my::funct_, 1.0);
        rhs_s(Sigmayz, 0)
            ->update(-timesurffac *
                         (normal(Velz) * ivelint_jump(Vely) + normal(Vely) * ivelint_jump(Velz)),
                my::funct_, 1.0);

        if (!is_MHVS) return;

        // ONLY MHVS:
        // pressure-tested kinematic continuity term
        /*
         *
         *      /      _  \
         *    +|  q n, u   |
         *      \         /
         */
        const double normalvel = ivelint_jump.dot(normal);

        rhs_pu.update(-timesurffac * normalvel, my::funct_, 1.0);
      }
    }

    /*--------------------------------------------------------------------------------
     *--------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::element_xfem_interface_nit(Discret::Elements::Fluid* ele,
        Core::FE::Discretization& dis, const std::vector<int>& lm,
        const std::shared_ptr<XFEM::ConditionManager>& cond_manager,  ///< XFEM condition manager
        const std::map<int, std::vector<Cut::BoundaryCell*>>& bcells,
        const std::map<int, std::vector<Core::FE::GaussIntegration>>& bintpoints,
        const std::map<int, std::vector<int>>&
            patchcouplm,  ///< lm vectors for coupling elements, key= global coupling side-Id
        Teuchos::ParameterList& params,
        std::shared_ptr<Core::Mat::Material>& mat_master,  ///< material for the background
        std::shared_ptr<Core::Mat::Material>& mat_slave,   ///< material for the coupled side
        Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
        Core::LinAlg::SerialDenseVector& elevec1_epetra, const Cut::plain_volumecell_set& vcSet,
        std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>& side_coupling,
        Core::LinAlg::SerialDenseMatrix& Cuiui, bool evaluated_cut)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (cond_manager == nullptr) FOUR_C_THROW("set the condition manager!");
#endif


      //----------------------------------------------------------------------------
      //                         ELEMENT GEOMETRY
      //----------------------------------------------------------------------------


      // ---------------------------------------------------------------------
      // get initial node coordinates for element
      // ---------------------------------------------------------------------
      // get node coordinates
      Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
          ele, my::xyze_);

      // ---------------------------------------------------------------------
      // get additional state vectors for ALE case: grid displacement and vel.
      // ---------------------------------------------------------------------

      my::edispnp_.clear();
      my::egridv_.clear();

      if (ele->is_ale()) my::get_grid_disp_vel_ale(dis, lm, my::edispnp_, my::egridv_);


      // ---------------------------------------------------------------------

      /// element coordinates in EpetraMatrix
      Core::LinAlg::SerialDenseMatrix ele_xyze(nsd_, nen_);
      for (int i = 0; i < nen_; ++i)
      {
        for (int j = 0; j < nsd_; ++j) ele_xyze(j, i) = my::xyze_(j, i);
      }

      // ---------------------------------------------------------------------
      // get velocity state vectors
      // ---------------------------------------------------------------------

      // get element-wise velocity/pressure field for current time step
      evelaf_.clear();
      epreaf_.clear();
      my::extract_values_from_global_vector(dis, lm, *my::rotsymmpbc_, &evelaf_, &epreaf_, "velaf");

      // get element-wise velocity/pressure field for previous time step
      eveln_.clear();
      epren_.clear();
      if (my::fldparatimint_->is_new_ost_implementation())
        my::extract_values_from_global_vector(dis, lm, *my::rotsymmpbc_, &eveln_, &epren_, "veln");

      // ---------------------------------------------------------------------
      // set element advective field for Oseen problems
      // ---------------------------------------------------------------------
      if (my::fldpara_->physical_type() == Inpar::FLUID::oseen) my::set_advective_vel_oseen(ele);


      //-----------------------------------------------------------------------------------
      //                     application-specific flags & parameters
      //-----------------------------------------------------------------------------------

      // map of boundary element gids, to coupling matrices Cuiui
      std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>> Cuiui_coupling;

      //-----------------------------------------------------------------------------------
      //            preparation of Cuiui-coupling matrices for each side
      //-----------------------------------------------------------------------------------

      // create Cuiui coupling matrices for each coupling side (we don't have Cuiui for standard
      // Dirichlet problems...)

      // loop all the intersecting sides of actele
      for (std::map<int, std::vector<Cut::BoundaryCell*>>::const_iterator bc = bcells.begin();
          bc != bcells.end(); ++bc)
      {
        const int coup_sid = bc->first;

        if (!cond_manager->is_coupling(coup_sid, my::eid_))
          continue;  // no couplings to be evaluated for current side

        // get coupling matrices for the current side (boundary element)
        std::vector<Core::LinAlg::SerialDenseMatrix>& Cuiui_matrices =
            Cuiui_coupling[coup_sid];  // create new vector of Coupling matrices

        std::map<int, std::vector<int>>::const_iterator j = patchcouplm.find(coup_sid);
        if (j == patchcouplm.end()) FOUR_C_THROW("missing side");

        // get number of dofs for coupling side/element
        const size_t ndof_i = j->second.size();

        Cuiui_matrices.resize(1);
        Cuiui_matrices[0].shape(ndof_i, ndof_i);  // Cuiui
      }

      //-----------------------------------------------------------------------------------
      //         evaluate element length, stabilization factors and average weights
      //-----------------------------------------------------------------------------------

      // initialize the characteristic element length and it's inverse
      double h_k = 0.0;
      double inv_hk = 1.0;

      //-----------------------------------------------------------------------------------
      // compute characteristic element length for background element in case of background-sided
      // coupling

      if (cond_manager->has_averaging_strategy(Inpar::XFEM::Xfluid_Sided))
      {
        h_k = XFEM::Utils::compute_char_ele_length<distype>(
            ele, ele_xyze, *cond_manager, vcSet, bcells, bintpoints, fldparaxfem_->visc_stab_hk());
        inv_hk = 1.0 / h_k;
      }

      // Get materials for both master and slave side
      // If non-constant density or viscosity wants to be calculated on the different sides. A
      // review over how the scalar variables are set at the surface should be made.
      get_material_parameters_volume_cell(mat_master, densaf_master_, viscaf_master_, gamma_m_);
      if (mat_slave != nullptr)
      {
        get_material_parameters_volume_cell(mat_slave, densaf_slave_, viscaf_slave_, gamma_s_);
        // Security check:
        if (gamma_s_ != gamma_m_)
        {
          std::cout << "Surface tension for master side: " << gamma_m_
                    << ", is not equal to surface tension on slave side:" << gamma_s_ << std::endl;
          FOUR_C_THROW("Non-matching surface tension provided for Master and Slave side.");
        }
      }
      else
      {
        densaf_slave_ = 0.0;
        viscaf_slave_ = 0.0;
        gamma_s_ = 0.0;
      }

      //-----------------------------------------------------------------------------------
      //      surface integral --- loop sides
      //-----------------------------------------------------------------------------------
      // map of side-element id and Gauss points
      for (std::map<int, std::vector<Core::FE::GaussIntegration>>::const_iterator i =
               bintpoints.begin();
          i != bintpoints.end(); ++i)
      {
        TEUCHOS_FUNC_TIME_MONITOR("FluidEleCalcXFEM::GaussIntegrationloop");

        //-----------------------------------------------------------------------------------

        // interface normal vector, pointing from background domain into the interface
        normal_.clear();
        // gauss-point coordinates
        x_side_.clear();

        // we need an interface to the boundary element (for projection)
        std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<distype>> si;

        // location array of boundary element
        Core::Elements::LocationArray cutla(1);

        // pointer to boundary element
        Core::Elements::Element* side = nullptr;

        // coordinates of boundary element
        Core::LinAlg::SerialDenseMatrix side_xyze;

        //-----------------------------------------------------------------------------------
        // only used for couplings:

        // coupling object between background element and each coupling element (side for
        // xfluid-sided coupling, element for other couplings)
        std::shared_ptr<Discret::Elements::XFLUID::NitscheInterface<distype>> ci;

        // pointer to coupling element
        Core::Elements::Element* coupl_ele = nullptr;

        // coupling element coordinates
        Core::LinAlg::SerialDenseMatrix coupl_xyze;

        //-----------------------------------------------------------------------------------

        int coup_sid = i->first;  // global coupling side id

        // get the coupling strategy for coupling of two fields
        const XFEM::EleCoupCond& coupcond =
            cond_manager->get_coupling_condition(coup_sid, my::eid_);
        const Inpar::XFEM::EleCouplingCondType& cond_type = coupcond.first;

        const int coup_idx = cond_manager->get_coupling_index(coup_sid, my::eid_);
        std::shared_ptr<XFEM::CouplingBase> coupling = cond_manager->get_coupling_by_idx(coup_idx);


        const std::vector<Core::FE::GaussIntegration>& cutintpoints = i->second;

        std::map<int, std::vector<Cut::BoundaryCell*>>::const_iterator j = bcells.find(coup_sid);
        if (j == bcells.end()) FOUR_C_THROW("missing boundary cell");

        const std::vector<Cut::BoundaryCell*>& bcs = j->second;
        if (bcs.size() != cutintpoints.size())
          FOUR_C_THROW("boundary cell integration rules mismatch");

        //-----------------------------------------------------------------------------------
        // define average weights

        bool non_xfluid_coupling;
        double kappa_m;
        double kappa_s;

        cond_manager->get_average_weights(coup_sid, ele, kappa_m, kappa_s, non_xfluid_coupling);

        //---------------------------------------------------------------------------------
        // set flags used for coupling with given levelset/mesh coupling side
        bool is_ls_coupling_side = cond_manager->is_level_set_coupling(coup_sid);
        bool is_mesh_coupling_side = cond_manager->is_mesh_coupling(coup_sid);

        std::shared_ptr<Core::FE::Discretization> cutter_dis =
            cond_manager->get_cutter_dis(coup_sid);

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (is_ls_coupling_side and is_mesh_coupling_side)
          FOUR_C_THROW(
              "side cannot be a levelset-coupling side and a mesh coupling side at once: side {}",
              coup_sid);
        if (!is_ls_coupling_side and !is_mesh_coupling_side)
          FOUR_C_THROW("side is neither a levelset-coupling side nor a mesh coupling side: side {}",
              coup_sid);
#endif

        //-----------------------------------------------------------------------------------
        std::shared_ptr<XFEM::MeshCouplingFSI> mc_fsi = nullptr;
        std::shared_ptr<XFEM::MeshCouplingFPI> mc_fpi = nullptr;
        bool assemble_iforce = false;

        //---------------------------------------------------------------------------------
        // prepare the coupling objects
        if (is_mesh_coupling_side)
        {
          mc_fsi = std::dynamic_pointer_cast<XFEM::MeshCouplingFSI>(coupling);
          mc_fpi = std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(coupling);

          if (mc_fsi != nullptr || mc_fpi != nullptr)
          {
            if (coupling->get_averaging_strategy() == Inpar::XFEM::Xfluid_Sided &&
                mc_fsi != nullptr)
              assemble_iforce = true;
            else
            {
              static bool flipflop = false;
              if (!flipflop)
              {
                std::cout
                    << "==| WARNING We do not assemble ifore (problem in case  theta != 1!!!) |=="
                    << std::endl;
                flipflop = !flipflop;
              }
            }
          }

          // get the side element and its coordinates for projection of Gaussian points
          side = cond_manager->get_side(coup_sid);
          Core::Geo::initial_position_array(side_xyze, side);

          // create auxiliary coupling object for the boundary element, in order to perform
          // projection
          si = Discret::Elements::XFLUID::SlaveElementInterface<
              distype>::create_slave_element_representation(side, side_xyze);

          // set displacement of side
          side->location_vector(*cutter_dis, cutla, false);
          si->add_slave_ele_disp(*cutter_dis, cutla[0].lm_);

          if (cond_type == Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET or
              cond_type == Inpar::XFEM::CouplingCond_SURF_FSI_PART or
              cond_type == Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP or
              cond_type == Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE)
          {
            si->set_interface_jump_statenp(*cutter_dis, "ivelnp", cutla[0].lm_);
            if (my::fldparatimint_->is_new_ost_implementation())
              si->set_interface_jump_staten(*cutter_dis, "iveln", cutla[0].lm_);
          }
        }

        std::shared_ptr<Core::FE::Discretization> coupl_dis_ =
            cond_manager->get_coupling_dis(coup_sid);

        if (!(is_ls_coupling_side and
                !cond_manager->is_coupling(coup_sid, my::eid_)))  // not level-set-WDBC case
        {
          coupl_ele = cond_manager->get_coupling_element(coup_sid, ele);
          if (coupl_ele == nullptr)
            FOUR_C_THROW("Failed to obtain coupling element for global coup_sid {}", coup_sid);
          Core::Geo::initial_position_array(coupl_xyze, coupl_ele);
        }

        if (!cond_manager->is_coupling(coup_sid, my::eid_))
        {
          if (is_ls_coupling_side)  //... for problems with cut interface defined by level-set
                                    // field, currently only one-sided
          {
            ci = Discret::Elements::XFLUID::NitscheInterface<
                distype>::create_nitsche_coupling_x_fluid_wdbc(elemat1_epetra, elevec1_epetra,
                *fldparaxfem_);
          }
          else if (is_mesh_coupling_side)
          {
            ci = Discret::Elements::XFLUID::NitscheInterface<
                distype>::create_nitsche_coupling_x_fluid_wdbc(coupl_ele, coupl_xyze,
                elemat1_epetra, elevec1_epetra, *fldparaxfem_);
          }
        }
        else  // coupling
        {
          std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::iterator c =
              side_coupling.find(coup_sid);

          std::vector<Core::LinAlg::SerialDenseMatrix>& side_matrices = c->second;

          // coupling matrices between background element and one! side
          Core::LinAlg::SerialDenseMatrix& C_you = side_matrices[0];
          Core::LinAlg::SerialDenseMatrix& C_uui = side_matrices[1];
          Core::LinAlg::SerialDenseMatrix& rhC_ui = side_matrices[2];

          // coupling matrices between one side and itself via the element Kss
          std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::iterator c2 =
              Cuiui_coupling.find(coup_sid);
          std::vector<Core::LinAlg::SerialDenseMatrix>& Cuiui_matrices = c2->second;
          Core::LinAlg::SerialDenseMatrix& eleCuiui = Cuiui_matrices[0];

          if (non_xfluid_coupling)
          {
            // create interface for the embedded element and the associated side
            ci = Discret::Elements::XFLUID::NitscheInterface<
                distype>::create_nitsche_coupling_two_sided(coupl_ele, coupl_xyze, elemat1_epetra,
                C_you, C_uui, eleCuiui, elevec1_epetra, rhC_ui, *fldparaxfem_);
          }
          else  // ... for xfluid-sided coupling
          {
            ci = Discret::Elements::XFLUID::NitscheInterface<
                distype>::create_nitsche_coupling_x_fluid_sided(coupl_ele, coupl_xyze,
                elemat1_epetra, C_you, C_uui, eleCuiui, elevec1_epetra, rhC_ui, *fldparaxfem_);
          }
        }

        if (cond_manager->is_coupling(coup_sid, my::eid_))
        {
          std::map<int, std::vector<int>>::const_iterator k = patchcouplm.find(coup_sid);
          const std::vector<int>& coupl_lm = k->second;

          // set velocity (and pressure) of coupling/slave element at current time step
          ci->set_slave_state(*coupl_dis_, coupl_lm);

          // set velocity (and pressure) of coupling element at old time step
          if (my::fldparatimint_->is_new_ost_implementation())
            ci->set_slave_staten(*coupl_dis_, coupl_lm);
        }

        std::vector<double> eledisp;
        if (!(is_ls_coupling_side and
                !cond_manager->is_coupling(coup_sid, my::eid_)))  // not level-set-WDBC case
        {
          std::map<int, std::vector<int>>::const_iterator k = patchcouplm.find(coup_sid);
          const std::vector<int>& coupl_lm = k->second;

          // add displacement of coupling element at current time step
          eledisp = std::vector<double>(coupl_lm.size());
          ci->add_slave_ele_disp(*coupl_dis_, coupl_lm, eledisp);
        }



        if (cond_manager->is_coupling(coup_sid, my::eid_) and non_xfluid_coupling)
        {
          //---------------------------------------------------------------------------------
          // compute characteristic element length for the case of embedded-sided coupling

          // char. length defined by local eigenvalue problem
          if (fldparaxfem_->visc_stab_trac_estimate() ==
              Inpar::XFEM::ViscStab_TraceEstimate_eigenvalue)
          {
            inv_hk = cond_manager->get_trace_estimate_max_eigenvalue(coup_sid);
            h_k = 1.0 / inv_hk;
          }
          else  // ... char. length defined otherwise
          {
            // compute characteristic element length based on the embedded element
            h_k = XFEM::Utils::compute_char_ele_length<distype>(coupl_ele, coupl_xyze,
                *cond_manager, vcSet, bcells, bintpoints, fldparaxfem_->visc_stab_hk(), ci, side);
            inv_hk = 1.0 / h_k;
          }
        }


        //---------------------------------------------------------------------------------
        // compute viscous part of Nitsche's penalty term scaling for Nitsche's method
        // based on the inverse characteristic element length
        //---------------------------------------------------------------------------------

        double NIT_visc_stab_fac = 0.0;
        double NIT_visc_stab_fac_tang = 0.0;
        cond_manager->get_visc_penalty_stabfac(coup_sid, ele, kappa_m, kappa_s, inv_hk,
            fldparaxfem_, NIT_visc_stab_fac, NIT_visc_stab_fac_tang);

        // define interface force vector w.r.t side (for XFSI)
        Core::LinAlg::SerialDenseVector iforce;
        iforce.size(cutla[0].lm_.size());

        //---------------------------------------------------------------------------------
        // loop boundary cells w.r.t current cut side
        //---------------------------------------------------------------------------------
        for (std::vector<Core::FE::GaussIntegration>::const_iterator i = cutintpoints.begin();
            i != cutintpoints.end(); ++i)
        {
          const Core::FE::GaussIntegration& gi = *i;
          Cut::BoundaryCell* bc =
              bcs[i - cutintpoints.begin()];  // get the corresponding boundary cell

          //-------------------------------------------------------------------------------
          // loop gausspoints w.r.t current boundary cell
          //-------------------------------------------------------------------------------
          //-------------------------------------------------------------------------------
          for (Core::FE::GaussIntegration::iterator iquad = gi.begin(); iquad != gi.end(); ++iquad)
          {
            double drs =
                0.0;  // transformation factor between reference cell and linearized boundary cell

            const Core::LinAlg::Matrix<2, 1> eta(
                iquad.point());  // xi-coordinates with respect to side



            // compute transformation factor, normal vector and global Gauss point coordinates
            if (bc->shape() != Core::FE::CellType::dis_none)  // Tessellation approach
            {
              XFEM::Utils::compute_surface_transformation(drs, x_gp_lin_, normal_, bc, eta);
            }
            else  // MomentFitting approach
            {
              drs = 1.0;
              normal_ = bc->get_normal_vector();
              const double* gpcord = iquad.point();
              for (int idim = 0; idim < 3; ++idim)
              {
                x_gp_lin_(idim, 0) = gpcord[idim];
              }
            }

            {
              TEUCHOS_FUNC_TIME_MONITOR("FluidEleCalcXFEM::Cut::Position");

              if (!evaluated_cut)  // compute the local coordinate based on the reference position
                                   // (first time the cut was frozen)
              {
                Core::LinAlg::Matrix<3, 1> x_ref = x_gp_lin_;
                double tmp_drs;
                Core::LinAlg::Matrix<3, 1> tmp_normal;
                if (bc->shape() != Core::FE::CellType::dis_none)  // Tessellation approach
                {
                  XFEM::Utils::compute_surface_transformation(
                      tmp_drs, x_ref, tmp_normal, bc, eta, true);
                }

                // find element local position of gauss point
                std::shared_ptr<Cut::Position> pos =
                    Cut::PositionFactory::build_position<nsd_, distype>(my::xyze_, x_ref);
                pos->compute();
                pos->local_coordinates(rst_);
              }
              else  // compute the local coordinate based on the current position
              {
                // find element local position of gauss point
                std::shared_ptr<Cut::Position> pos =
                    Cut::PositionFactory::build_position<nsd_, distype>(my::xyze_, x_gp_lin_);
                pos->compute();
                pos->local_coordinates(rst_);
              }
            }

            Core::LinAlg::Matrix<3, 1> rst_slave;  // local coordinates of slave element
            if (is_mesh_coupling_side)
            {
              // project gaussian point from linearized interface to warped side (get/set local side
              // coordinates in SideImpl)
              Core::LinAlg::Matrix<3, 1> xi_side;

              // project on boundary element
              si->project_on_side(x_gp_lin_, x_side_, xi_side);

              if (non_xfluid_coupling)
                ci->evaluate(x_side_, rst_slave);  // evaluate embedded element's shape functions at
                                                   // gauss-point coordinates
              else
                ci->evaluate(xi_side,
                    rst_slave);  // evaluate side's shape functions at gauss-point coordinates
            }
            else if (is_ls_coupling_side)
            {
              if (cond_manager->is_coupling(coup_sid, my::eid_))
                ci->evaluate(x_gp_lin_, rst_slave);  // evaluate embedded element's shape functions
                                                     // at gauss-point coordinates
            }

            // integration factors
            const double surf_fac = drs * iquad.weight();
            const double timefacfac = surf_fac * my::fldparatimint_->time_fac();

            // evaluate background element shape functions
            eval_func_and_deriv(rst_);

            // get velocity at integration point
            // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
            my::velint_.multiply(evelaf_, my::funct_);

            // get velocity derivatives at integration point
            // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
            my::vderxy_.multiply_nt(evelaf_, my::derxy_);

            // get pressure at integration point
            // (value at n+1)
            double press = my::funct_.dot(epreaf_);

            //----------------------------------------------
            // get convective velocity at integration point
            my::set_convective_velint(ele->is_ale());

            //-----------------------------------------------------------------------------
            // compute stabilization factors

            double NIT_full_stab_fac = 0.0;

            // Extract slave velocity at Gausspoint
            ci->get_interface_velnp(velint_s_);

            XFEM::Utils::nit_compute_full_penalty_stabfac(
                NIT_full_stab_fac,  ///< to be filled: full Nitsche's penalty term scaling
                                    ///< (viscous+convective part)
                normal_, h_k, kappa_m, kappa_s, my::convvelint_, velint_s_,
                NIT_visc_stab_fac,  ///< Nitsche's viscous scaling part of penalty term
                my::fldparatimint_->time_fac(), my::fldparatimint_->is_stationary(), densaf_master_,
                densaf_slave_, fldparaxfem_->mass_conservation_scaling(),
                fldparaxfem_->mass_conservation_combination(), fldparaxfem_->nit_stab_scaling(),
                fldparaxfem_->conv_stab_scaling(), fldparaxfem_->xff_conv_stab_scaling(),
                my::fldpara_->is_conservative());


            //-----------------------------------------------------------------------------
            // define the prescribed interface jump vectors for velocity and traction

            ivelint_jump_.clear();
            itraction_jump_.clear();
            proj_tangential_.clear();
            lb_proj_matrix_.clear();

            get_interface_jump_vectors(coupcond, coupling, ivelint_jump_, itraction_jump_,
                proj_tangential_, lb_proj_matrix_, x_gp_lin_, normal_, *si, rst_, kappa_m,
                viscaf_master_, viscaf_slave_, rst_slave, eledisp, coupl_ele);

            double fulltraction = 0.0;
            if (cond_type == Inpar::XFEM::CouplingCond_LEVELSET_NEUMANN or
                cond_type == Inpar::XFEM::CouplingCond_SURF_NEUMANN)
            {
              //-----------------------------------------------------------------------------
              // evaluate the Neumann boundary condition term
              evaluate_neumann(timefacfac,  ///< theta*dt
                  my::funct_,               ///< coupling master shape functions
                  itraction_jump_,  ///< prescribed interface traction, jump height for coupled
                                    ///< problems
                  elevec1_epetra    ///< element rhs vector
              );

              if (my::fldparatimint_->is_new_ost_implementation())
              {
                FOUR_C_THROW(
                    "how to deal with Neumann boundary condition and new OSTImplementation");
              }
            }

            {
              TEUCHOS_FUNC_TIME_MONITOR("FluidEleCalcXFEM::nit_evaluate_coupling");
              if (mc_fsi != nullptr)
              {
                if (mc_fsi->get_interface_law() == Inpar::XFEM::navierslip_contact)
                  fulltraction =
                      XFEM::Utils::evaluate_full_traction(press, my::vderxy_, viscaf_master_,
                          NIT_full_stab_fac, my::velint_, velint_s_, normal_, normal_, velint_s_);
              }
              else if (mc_fpi != nullptr)
              {
                double J = 0;
                double porosity = mc_fpi->calc_porosity(side, rst_slave, J);
                static Core::LinAlg::Matrix<3, 1> vel_s(Core::LinAlg::Initialization::zero);
                static Core::LinAlg::Matrix<3, 1> velpf_s(Core::LinAlg::Initialization::zero);
                XFEM::Utils::evaluate_stateat_gp(side, rst_slave,
                    *cond_manager->get_mesh_coupling("XFEMSurfFPIMono_ps_ps")->get_cutter_dis(),
                    "ivelnp", vel_s);
                XFEM::Utils::evaluate_stateat_gp(side, rst_slave,
                    *cond_manager->get_mesh_coupling("XFEMSurfFPIMono_pf_pf")->get_cutter_dis(),
                    "ivelnp", velpf_s);
                fulltraction =
                    XFEM::Utils::evaluate_full_traction(press, my::vderxy_, viscaf_master_,
                        NIT_full_stab_fac, my::velint_, vel_s, normal_, normal_, velpf_s, porosity);
              }

              // Get Configuration Map
              std::map<Inpar::XFEM::CoupTerm, std::pair<bool, double>>& configmap =
                  coupling->get_configurationmap(kappa_m, viscaf_master_, viscaf_slave_,
                      my::densaf_, NIT_visc_stab_fac_tang, NIT_full_stab_fac, x_gp_lin_,
                      coupcond.second, ele, side, my::funct_.data(), my::derxy_.data(), rst_slave,
                      normal_, my::velint_, &fulltraction);

              //-----------------------------------------------------------------------------
              // evaluate the coupling terms for coupling with current side
              // (or embedded element through current side)
              // time step n+1

              const bool isImplPressureNewOst(
                  my::fldparatimint_->is_full_impl_pressure_and_cont() &&
                  my::fldparatimint_->is_new_ost_implementation());

              const double pres_timefacfac(
                  isImplPressureNewOst ? my::fldparatimint_->dt() * surf_fac : timefacfac);

              ci->nit_evaluate_coupling(normal_,  // normal vector
                  timefacfac,                     // theta*dt*fac
                  pres_timefacfac,  // impl. pressure with new OST: dt * fac, else theta*dt*fac
                  viscaf_master_,   // dynvisc viscosity in background fluid
                  viscaf_slave_,    // dynvisc viscosity in embedded fluid
                  my::densaf_,      // fluid density
                  my::funct_,       // bg shape functions
                  my::derxy_,       // bg shape function gradient
                  my::vderxy_,      // bg grad u^n+1
                  press,            // bg p^n+1
                  my::velint_,      // bg u^n+1
                  ivelint_jump_,  // prescribed interface velocity, Dirichlet values or jump height
                                  // for coupled problems
                  itraction_jump_,   // traction jump at interface (i.e. [| -pI + \mu*[\nabla u +
                                     // (\nabla u)^T]  |] \cdot n)
                  proj_tangential_,  // tangential projection matrix
                  lb_proj_matrix_,   // prescribed projection matrix for laplace-beltrami problems
                  solid_stress_,     // hold information about solid stress ([0]...traction,
                                     // [1]...dtraction_dv, [2-4]...d2traction_dv2)
                  configmap          // Configuration Map
              );

              if (my::fldparatimint_->is_new_ost_implementation())
              {
                // get velocity at integration point
                // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
                my::velintn_.multiply(eveln_, my::funct_);

                // get velocity derivatives at integration point
                // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
                my::vderxyn_.multiply_nt(eveln_, my::derxy_);

                ivelintn_jump_.clear();
                itractionn_jump_.clear();

                // Safety check
                if (cond_type == Inpar::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP or
                    cond_type == Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP or
                    cond_type == Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE)
                {
                  if (my::fldparatimint_->is_new_ost_implementation())
                  {
                    FOUR_C_THROW(
                        "How to deal with NavierSlip boundary condition and new "
                        "OSTImplementation?");
                  }
                }

                if (cond_type != Inpar::XFEM::CouplingCond_LEVELSET_TWOPHASE and
                    cond_type != Inpar::XFEM::CouplingCond_LEVELSET_COMBUSTION)
                {
                  get_interface_jump_vectors_old_state(coupcond, *coupling, ivelintn_jump_,
                      itractionn_jump_, x_gp_lin_, normal_, *si,
                      my::funct_.dot(epren_),  // bg p^n
                      rst_);
                }
                else
                {
                  get_interface_jump_vectors_old_state(coupcond, *coupling, ivelintn_jump_,
                      itractionn_jump_, x_gp_lin_, normal_, *ci,
                      my::funct_.dot(epren_),  // bg p^n
                      rst_);
                }

                //-----------------------------------------------------------------------------
                // evaluate the coupling terms for coupling with current side
                // (or embedded element through current side)
                // time step n

                // REMARK: evaluation of all Nitsche terms for the previous solution
                // only makes sense for stationary interfaces!

                double NIT_full_stab_fac_n = 0.0;
                if (fldparaxfem_->interface_terms_previous_state() ==
                    Inpar::XFEM::PreviousState_full)
                {
                  velintn_s_.clear();
                  ci->get_interface_veln(velintn_s_);

                  XFEM::Utils::nit_compute_full_penalty_stabfac(
                      NIT_full_stab_fac_n,  ///< to be filled: full Nitsche's penalty term scaling
                                            ///< (viscous+convective part)
                      normal_, h_k,
                      kappa_m,  // weights (only existing for Nitsche currently!!)
                      kappa_s,  // weights (only existing for Nitsche currently!!)
                      my::convvelintn_, velintn_s_,
                      NIT_visc_stab_fac,  ///< Nitsche's viscous scaling part of penalty term
                      my::fldparatimint_->time_fac(), my::fldparatimint_->is_stationary(),
                      densaf_master_, densaf_slave_, fldparaxfem_->mass_conservation_scaling(),
                      fldparaxfem_->mass_conservation_combination(),
                      fldparaxfem_->nit_stab_scaling(), fldparaxfem_->conv_stab_scaling(),
                      fldparaxfem_->xff_conv_stab_scaling(), my::fldpara_->is_conservative());
                }

                // Get Configuration Map
                std::map<Inpar::XFEM::CoupTerm, std::pair<bool, double>> configmap_n =
                    coupling->get_configurationmap(kappa_m, viscaf_master_, viscaf_slave_,
                        my::densaf_, NIT_visc_stab_fac_tang, NIT_full_stab_fac, x_gp_lin_,
                        coupcond.second, ele, side, my::funct_.data(), my::derxy_.data(), rst_slave,
                        normal_, my::velint_, &fulltraction);

                const double timefacfacn =
                    surf_fac * (my::fldparatimint_->dt() - my::fldparatimint_->time_fac());
                ci->nit_evaluate_coupling_old_state(normal_, timefacfacn, isImplPressureNewOst,
                    viscaf_master_,          // dynvisc viscosity in background fluid
                    viscaf_slave_,           // dynvisc viscosity in embedded fluid
                    my::densn_,              // fluid density
                    my::funct_,              // bg shape functions
                    my::derxy_,              // bg shape function gradient
                    my::vderxyn_,            // bg grad u^n
                    my::funct_.dot(epren_),  // bg p^n
                    my::velintn_,            // bg u^n
                    ivelintn_jump_,          // velocity jump at interface (i.e. [| u |])
                    proj_tangential_,        // tangential projection matrix
                    itractionn_jump_,  // traction jump at interface (i.e. [| -pI + \mu*[\nabla u +
                                       // (\nabla u)^T]  |] \cdot n)
                    configmap_n);
              }
            }

            if (!assemble_iforce) continue;

            //-----------------------------------------------------------------------------
            // calculate interface forces for XFSI

            //-------------------------------
            // traction vector w.r.t fluid domain, resulting stresses acting on the fluid surface
            // t= (-p*I + 2mu*eps(u))*n^f
            Core::LinAlg::Matrix<nsd_, 1> traction;

            build_traction_vector(traction, press, normal_);
            ci->compute_interface_force(iforce, traction, surf_fac);

          }  // end loop gauss points of boundary cell
        }  // end loop boundary cells of side

        if (assemble_iforce)
          assemble_interface_force(*mc_fsi->i_forcecol(), *cutter_dis, cutla[0].lm_, iforce);
      }  // end loop cut sides


      //-----------------------------------------------------------------------------------
      // build Cuiui coupling matrix (includes patch of Cuiui matrices for all sides)
      //-----------------------------------------------------------------------------------

      nit_build_patch_cuiui(Cuiui, Cuiui_coupling);

      return;
    }



    /*--------------------------------------------------------------------------------
     * get the interface jump vectors for velocity and traction at the Gaussian point
     *--------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::get_interface_jump_vectors(
        const XFEM::EleCoupCond& coupcond,  ///< coupling condition for given interface side
        std::shared_ptr<XFEM::CouplingBase> coupling,  ///< coupling object
        Core::LinAlg::Matrix<nsd_, 1>&
            ivelint_jump,  ///< prescribed interface jump vector for velocity
        Core::LinAlg::Matrix<nsd_, 1>&
            itraction_jump,  ///< prescribed interface jump vector for traction
        Core::LinAlg::Matrix<nsd_, nsd_>& proj_tangential,  ///< tangential projection matrix
        Core::LinAlg::Matrix<nsd_, nsd_>&
            LB_proj_matrix,  ///< prescribed projection matrix for laplace-beltrami problems
        const Core::LinAlg::Matrix<nsd_, 1>& x,       ///< global coordinates of Gaussian point
        const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector at Gaussian point
        Discret::Elements::XFLUID::SlaveElementInterface<distype>&
            si,                                 ///< side implementation for cutter element
        Core::LinAlg::Matrix<3, 1>& rst,        ///< local coordinates of GP for bg element
        double& kappa_m,                        ///< fluid sided weighting
        double& visc_m,                         ///< fluid sided weighting
        double& visc_s,                         ///< slave sided dynamic viscosity
        Core::LinAlg::Matrix<3, 1>& rst_slave,  ///< local coord of gp in slave element
        std::vector<double>& eledisp,           ///< slave element displacement vector
        Core::Elements::Element* coupl_ele      ///< slave coupling element
    )
    {
      TEUCHOS_FUNC_TIME_MONITOR("FluidEleCalcXFEM::get_interface_jump_vectors");

      // [| v |] := vm - vs

      const Inpar::XFEM::EleCouplingCondType& cond_type =
          coupcond.first;  ///< condition type for given interface side
      const Core::Conditions::Condition* cond = coupcond.second;  ///< condition to be evaluated

      switch (cond_type)
      {
        case Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET:
        {
          const std::string& evaltype = cond->parameters().get<std::string>("EVALTYPE");

          if (evaltype == "funct_gausspoint")
          {
            // evaluate function at Gaussian point at current time
            coupling->evaluate_coupling_conditions(
                ivelint_jump, itraction_jump, x, cond);  // itraction_jump.clear() called here...
          }
          else
          {
            // evaluate function at nodes at current time
            si.get_interface_jump_velnp(ivelint_jump);
          }

          break;
        }
        case Inpar::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET:
        {
          // evaluate condition function at Gaussian point
          coupling->evaluate_coupling_conditions(ivelint_jump, itraction_jump, x, cond);
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_NEUMANN:
        case Inpar::XFEM::CouplingCond_LEVELSET_NEUMANN:
        {
          // evaluate condition function at Gaussian point
          if (cond->parameters().get<int>("NUMDOF") == 6)
          {
            Core::LinAlg::Matrix<6, 1> fulltraction(
                Core::LinAlg::Initialization::zero);  // sigma_xx, sigma_yy, sigma_zz, sigma_xy,
                                                      // sigma_yz, sigma_zx
            coupling->evaluate_coupling_conditions(ivelint_jump, fulltraction, x, cond);
            itraction_jump(0, 0) = fulltraction(0, 0) * normal(0, 0) +
                                   fulltraction(3, 0) * normal(1, 0) +
                                   fulltraction(5, 0) * normal(2, 0);
            itraction_jump(1, 0) = fulltraction(3, 0) * normal(0, 0) +
                                   fulltraction(1, 0) * normal(1, 0) +
                                   fulltraction(4, 0) * normal(2, 0);
            itraction_jump(2, 0) = fulltraction(5, 0) * normal(0, 0) +
                                   fulltraction(4, 0) * normal(1, 0) +
                                   fulltraction(2, 0) * normal(2, 0);
          }
          else
          {
            coupling->evaluate_coupling_conditions(ivelint_jump, itraction_jump, x, cond);
          }
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_FSI_PART:
        {
          // evaluate function at nodes at current time
          si.get_interface_jump_velnp(ivelint_jump);
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID:
        {
          // nothing to evaluate as continuity coupling conditions have to be evaluated
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_FSI_MONO:
        {
          std::dynamic_pointer_cast<XFEM::MeshCouplingFSI>(coupling)
              ->evaluate_structural_cauchy_stress(
                  coupl_ele, rst_slave, eledisp, normal, solid_stress_);
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_FPI_MONO:
        {
          std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(coupling)
              ->evaluate_coupling_conditions<distype>(proj_tangential, normal);
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP:
        {
          bool eval_dirich_at_gp =
              ((cond->parameters().get<std::string>("EVALTYPE")) == "funct_gausspoint");

          // The velocity is evaluated twice in this framework...
          std::dynamic_pointer_cast<XFEM::MeshCouplingNavierSlip>(coupling)
              ->evaluate_coupling_conditions(ivelint_jump, itraction_jump, proj_tangential, x,
                  normal, cond, eval_dirich_at_gp, kappa_m, visc_m, visc_s);

          if (!eval_dirich_at_gp)
          {
            si.get_interface_jump_velnp(ivelint_jump);
          }

          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE:
        {
          bool eval_dirich_at_gp =
              ((cond->parameters().get<std::string>("EVALTYPE")) == "funct_gausspoint");

          std::dynamic_pointer_cast<XFEM::MeshCouplingNavierSlipTwoPhase>(coupling)
              ->evaluate_coupling_conditions<distype>(ivelint_jump, itraction_jump, x, cond,
                  proj_tangential, my::eid_, my::funct_, my::derxy_, normal, eval_dirich_at_gp,
                  kappa_m, visc_m, visc_s);

          //    if(!eval_dirich_at_gp)
          //    {
          //
          //      si->get_interface_jump_velnp(ivelint_jump);
          //
          //    }

          break;
        }
        case Inpar::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP:
        {
          std::dynamic_pointer_cast<XFEM::LevelSetCouplingNavierSlip>(coupling)
              ->evaluate_coupling_conditions<distype>(ivelint_jump, itraction_jump, x, cond,
                  proj_tangential, my::eid_, my::funct_, my::derxy_, normal, kappa_m, visc_m,
                  visc_s);

          break;
        }
        // Neumann boundary conditions for Mesh and Levelset
        default:
          FOUR_C_THROW(
              "invalid type of condition {}, which prescribed interface vectors have to be set?",
              cond_type);
          break;
      }

      // Create a projection matrix.
      //  If it is a Navier-Slip coupling, the matrix is provided from the Evaluation.
      //   Furthermore, if it is a Laplace-Beltrami way of calculating the surface tension,
      //   do not fill the matrix as it contains the "projection matrix" for LB implementation.
      if (cond_type != Inpar::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP and
          cond_type != Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP and
          cond_type != Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE)
      {
        // Create normal projection matrix.
        Core::LinAlg::Matrix<nsd_, nsd_> eye(Core::LinAlg::Initialization::zero);
        for (int i = 0; i < nsd_; ++i) eye(i, i) = 1;
        for (int i = 0; i < nsd_; ++i)
        {
          for (int j = 0; j < nsd_; ++j)
          {
            proj_tangential(i, j) = eye(i, j) - normal(i, 0) * normal(j, 0);
          }
        }
      }

      return;
    }

    /*--------------------------------------------------------------------------------
     * get the interface jump vectors for velocity and traction at the Gaussian point
     * for previous time step
     *--------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::get_interface_jump_vectors_old_state(
        const XFEM::EleCoupCond& coupcond,  ///< coupling condition for given interface side
        XFEM::CouplingBase& coupling,       ///< coupling object
        Core::LinAlg::Matrix<nsd_, 1>&
            ivelintn_jump,  ///< prescribed interface jump vector for velocity
        Core::LinAlg::Matrix<nsd_, 1>&
            itractionn_jump,                     ///< prescribed interface jump vector for traction
        const Core::LinAlg::Matrix<nsd_, 1>& x,  ///< global coordinates of Gaussian point
        const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector at Gaussian point
        Discret::Elements::XFLUID::SlaveElementInterface<distype>&
            si,                          ///< side implementation for cutter element
        const double& presn_m,           ///< coupling master pressure
        Core::LinAlg::Matrix<3, 1>& rst  ///< local coordinates of GP for bg element
    )
    {
      // [| v |] := vm - vs

      const Inpar::XFEM::EleCouplingCondType& cond_type =
          coupcond.first;  ///< condition type for given interface side
      const Core::Conditions::Condition* cond = coupcond.second;  ///< condition to be evaluated

      switch (cond_type)
      {
        case Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET:
        {
          const std::string& evaltype = cond->parameters().get<std::string>("EVALTYPE");

          if (evaltype == "funct_gausspoint")
          {
            // evaluate function at Gaussian point at current time
            coupling.evaluate_coupling_conditions_old_state(
                ivelintn_jump, itractionn_jump, x, cond);
          }
          else
          {
            // evaluate function at nodes for previous time
            si.get_interface_jump_veln(ivelintn_jump);
          }

          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_NEUMANN:
        case Inpar::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET:
        case Inpar::XFEM::CouplingCond_LEVELSET_NEUMANN:
        {
          // evaluate condition function at Gaussian point
          coupling.evaluate_coupling_conditions_old_state(ivelintn_jump, itractionn_jump, x, cond);
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP:
        case Inpar::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP:
        case Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE:
        {
          FOUR_C_THROW("Navier Slip Condition not implemented for NEWOst yet!");
          // here you would need the dyn_visc for summing up vel_jump and traction_jump...
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_FSI_PART:
        {
          // evaluate function at nodes at current time
          si.get_interface_jump_veln(ivelintn_jump);
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID:
        case Inpar::XFEM::CouplingCond_SURF_FSI_MONO:
        {
          // nothing to evaluate as continuity coupling conditions have to be evaluated
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_FPI_MONO:
        {
          FOUR_C_THROW("Fluid Poro Structure Interaction not implemented for NEWOst yet!");
          break;
        }
        case Inpar::XFEM::CouplingCond_LEVELSET_TWOPHASE:
        {
          // Spatial velocity gradient for slave side
          Core::LinAlg::Matrix<nsd_, nsd_> vderxyn_s(Core::LinAlg::Initialization::zero);
          si.get_interface_vel_gradn(vderxyn_s);


          // Calculate the old jump using the reconstructed values of p and u for the new interface
          // position.
          // i.e.

          // itractionn_jump = [|  -pI + \mu*[\nabla u + (\nabla u)^T]  |] \cdot n

          // Pressure part
          double presn_s = 0.0;
          si.get_interface_presn(presn_s);

          itractionn_jump.update(-(presn_m - presn_s), normal, 0.0);

          // Shear tensor part
          //===================
          Core::LinAlg::Matrix<nsd_, nsd_> tmp_matrix(Core::LinAlg::Initialization::zero);
          tmp_matrix.update(viscaf_master_, my::vderxyn_, -viscaf_slave_, vderxyn_s);

          // Initialize dummy variable
          Core::LinAlg::Matrix<nsd_, 1> tmp_vector(Core::LinAlg::Initialization::zero);

          // Normal
          tmp_vector.multiply(tmp_matrix, normal);
          itractionn_jump.update(1.0, tmp_vector, 1.0);

          // Transposed
          tmp_vector.multiply_tn(tmp_matrix, normal);
          itractionn_jump.update(1.0, tmp_vector, 1.0);
          //===================

          break;
        }
        case Inpar::XFEM::CouplingCond_LEVELSET_COMBUSTION:
        {
          // TODO: evaluate the ivelint_jump and the itraction_jump
          break;
        }
        // Neumann boundary conditions for Mesh and Levelset
        default:
          FOUR_C_THROW(
              "invalid type of condition {}, which prescribed interface vectors have to be set?",
              cond_type);
          break;
      }

      return;
    }



    /*--------------------------------------------------------------------------------
     * build the patch coupling matrix Cuiui containing Cuiui for all cutting sides
     *--------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::nit_build_patch_cuiui(
        Core::LinAlg::SerialDenseMatrix&
            Cuiui,  ///< ui-ui patch coupling matrix containing Cuiui for all cutting sides
        std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>&
            Cuiui_coupling  ///< Cuiui matrices for all cutting sides
    )
    {
      // TEUCHOS_FUNC_TIME_MONITOR("FluidEleCalcXFEM::nit_build_patch_cuiui");

      // build patch-Cuiui matrix
      int ipatchsizesbefore = 0;
      for (std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>::const_iterator m =
               Cuiui_coupling.begin();
          m != Cuiui_coupling.end(); ++m)
      {
        int coup_sid = m->first;
        std::vector<Core::LinAlg::SerialDenseMatrix>& Cuiui_mats = Cuiui_coupling[coup_sid];

        // Cuiui matrices in Cuiui_mats[0]

        // assemble Cuiui
        for (int ic = 0; ic < Cuiui_mats[0].numCols();
            ++ic)  // Cuiui includes only ui,ui coupling, not (ui,p) ...
        {
          for (int ir = 0; ir < Cuiui_mats[0].numRows(); ++ir)
          {
            Cuiui(ir + ipatchsizesbefore, ic + ipatchsizesbefore) = Cuiui_mats[0](ir, ic);
          }
        }

        ipatchsizesbefore += Cuiui_mats[0].numCols();
      }

      return;
    }

    /*--------------------------------------------------------------------------------
     * prepare coupling matrices, that include contributions from convective stabilization
     * and contributions from previous time steps (rhs)
     *--------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::hybrid_lm_create_special_contribution_matrices(
        XFEM::ConditionManager& cond_manager,  ///< XFEM condition manager
        std::set<int>& begids,                 ///< ids of intersecting boundary elements
        std::map<int, std::vector<Core::LinAlg::SerialDenseMatrix>>&
            side_coupling_extra  ///< contributions to coupling matrices from convective
                                 ///< stabilizations
    )
    {
      if (fldparaxfem_->get_coupling_method() != Inpar::XFEM::Hybrid_LM_Cauchy_stress &&
          fldparaxfem_->get_coupling_method() != Inpar::XFEM::Hybrid_LM_viscous_stress)
        FOUR_C_THROW("Do not call this method with a non-Lagrange multiplier based approach!");

      for (std::set<int>::const_iterator bgid = begids.begin(); bgid != begids.end(); ++bgid)
      {
        const int coup_sid = *bgid;

        if (!cond_manager.is_coupling(coup_sid, my::eid_))
          continue;  // no coupling with current side

        if (cond_manager.is_level_set_coupling(coup_sid))
          FOUR_C_THROW(
              "hybrid_lm_create_special_contribution_matrices for level-set coupling not supported "
              "yet");

        std::shared_ptr<Core::FE::Discretization> cutter_dis = nullptr;
        if (cond_manager.is_mesh_coupling(coup_sid))
          cutter_dis = cond_manager.get_cutter_dis(coup_sid);

        Core::Elements::Element* side = cond_manager.get_side(
            coup_sid);  // for each boundary element there is one corresponding side

        std::vector<int> patchlm;
        std::vector<int> patchlmowner;
        std::vector<int> patchlmstride;
        side->location_vector(*cutter_dis, patchlm, patchlmowner, patchlmstride);

        // get coupling matrices for the current side (boundary element)
        std::vector<Core::LinAlg::SerialDenseMatrix>& side_matrices_extra =
            side_coupling_extra[coup_sid];

        side_matrices_extra.resize(4);
        side_matrices_extra[0].shape(patchlm.size(), nen_ * numdofpernode_);  // Cuiu
        side_matrices_extra[1].shape(nen_ * numdofpernode_, patchlm.size());  // Cuui
        side_matrices_extra[2].shape(patchlm.size(), 1);                      // rhs_Cui
        side_matrices_extra[3].shape(patchlm.size(), patchlm.size());         // Cuiui
      }
    }


    /*----------------------------------------------------------------------*
     | evaluate shape functions and derivatives at given local coordinates  |
     *----------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::eval_func_and_deriv(Core::LinAlg::Matrix<3, 1>& rst)
    {
      // evaluate shape functions
      Core::FE::shape_function<distype>(rst, my::funct_);

      // evaluate the derivatives of shape functions
      Core::FE::shape_function_deriv1<distype>(rst, my::deriv_);
      my::xjm_.multiply_nt(my::deriv_, my::xyze_);
      my::det_ = my::xji_.invert(my::xjm_);

      // compute global first derivates
      my::derxy_.multiply(my::xji_, my::deriv_);

      return;
    }


    /*----------------------------------------------------------------------*
     | build traction vector w.r.t fluid domain                             |
     *----------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::build_traction_vector(
        Core::LinAlg::Matrix<nsd_, 1>& traction,  ///< traction vector
        double& press,                            ///< pressure at gaussian point
        Core::LinAlg::Matrix<nsd_, 1>& normal     ///< normal vector
    )
    {
      // compute the stresses at the current Gaussian point for computing the interface force
      Core::LinAlg::Matrix<nsd_, nsd_> two_eps;
      for (int i = 0; i < nsd_; ++i)
      {
        for (int j = 0; j < nsd_; ++j)
        {
          two_eps(j, i) = my::vderxy_(i, j) + my::vderxy_(j, i);
        }
      }
      //-------------------------------

      // t = ( -pI + 2mu eps(u) )*n^f
      traction.multiply(two_eps, normal);

      // add the pressure part and scale the viscous part with the viscosity
      traction.update(-press, normal, viscaf_master_);

      return;
    }

    /*----------------------------------------------------------------------*
     | assemble side's interface force                                      |
     *----------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::assemble_interface_force(
        Core::LinAlg::Vector<double>& iforcecol,  ///< interface force column vector
        Core::FE::Discretization& cutdis,         ///< cut discretization
        std::vector<int>& lm,                     ///< local dof map
        Core::LinAlg::SerialDenseVector& iforce   ///< interface force vector
    )
    {
      // TEUCHOS_FUNC_TIME_MONITOR("FluidEleCalcXFEM::assemble_interface_force");

      const Core::LinAlg::Map* dofcolmap = cutdis.dof_col_map();

      for (int idof = 0; idof < (int)(lm.size()); ++idof)
      {
        int gdof = lm[idof];

        // f^i = ( N^i, t ) = ( N^i, (-pI+2mu*eps(u))*n )
        (iforcecol)[dofcolmap->LID(gdof)] += iforce[idof];
      }

      return;
    }


    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::evaluate_neumann(const double& timefacfac,  ///< theta*dt
        const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< coupling master shape functions
        const Core::LinAlg::Matrix<nsd_, 1>&
            itraction_jump,  ///< prescribed interface traction, jump height for coupled problems
        Core::LinAlg::SerialDenseMatrix::Base& elevec1_epetra  ///< element vector
    )
    {
      const int master_numdof = nsd_ + 1;
      Core::LinAlg::Matrix<master_numdof * nen_, 1> rhC_um(elevec1_epetra.values(), true);

      // funct_m * timefac * fac
      Core::LinAlg::Matrix<nen_, 1> funct_m_timefacfac(funct_m);
      funct_m_timefacfac.scale(timefacfac);

      //-----------------------------------------------------------------
      // standard consistency Neumann term

      /*            \
   - |    v  ,   t   |   with t = [sigma * n]
      \             /     */

      // loop over velocity components
      for (int ivel = 0; ivel < nsd_; ++ivel)
      {
        //-----------------------------------------------
        //    - (vm, t)
        //-----------------------------------------------
        for (int ir = 0; ir < nen_; ++ir)
        {
          const unsigned row = ir * (nsd_ + 1) + ivel;
          rhC_um(row, 0) += funct_m_timefacfac(ir) * itraction_jump(ivel);
        }
      }  // end loop over velocity components
    }


    /*--------------------------------------------------------------------------------
     *--------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::calculate_continuity_xfem(
        Discret::Elements::Fluid* ele,                    ///< fluid element
        Core::FE::Discretization& dis,                    ///< discretization
        const std::vector<int>& lm,                       ///< local map
        Core::LinAlg::SerialDenseVector& elevec1_epetra,  ///< element vector
        const Core::FE::GaussIntegration& intpoints       ///< integration points
    )
    {
      Core::LinAlg::Matrix<numdofpernode_ * nen_, 1> elevec1(elevec1_epetra, true);
      Core::LinAlg::Matrix<numdofpernode_, nen_> tmpvel;
      my::eid_ = ele->id();

      // get node coordinates and number of elements per node
      Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
          ele, my::xyze_);

      //------------------------------------------------------------------------
      //  start loop over integration points
      //------------------------------------------------------------------------
      for (Core::FE::GaussIntegration::const_iterator iquad = intpoints.begin();
          iquad != intpoints.end(); ++iquad)
      {
        // evaluate shape functions and derivatives at integration point
        my::eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

        for (int ui = 0; ui < nen_; ++ui)
        {
          for (int idim = 0; idim < nsd_; ++idim)
          {
            const int fui = numdofpernode_ * ui;
            /* continuity term */
            /*
                 /           \
                |             |
                | nabla o Du  |
                |             |
                 \           /
            */

            elevec1(fui + idim) += my::fac_ * my::derxy_(idim, ui);
          }
        }
      }

      return;
    }

    /*--------------------------------------------------------------------------------
     *--------------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::calculate_continuity_xfem(
        Discret::Elements::Fluid* ele,                   ///< fluid element
        Core::FE::Discretization& dis,                   ///< discretization
        const std::vector<int>& lm,                      ///< local map
        Core::LinAlg::SerialDenseVector& elevec1_epetra  ///< element vector
    )
    {
      calculate_continuity_xfem(ele, dis, lm, elevec1_epetra, my::intpoints_);
    }



    template <Core::FE::CellType distype>
    void FluidEleCalcXFEM<distype>::get_material_parameters_volume_cell(
        std::shared_ptr<const Core::Mat::Material> material,
        double& densaf,  // done
        double& viscaf,  // done
        double& gamma    // done
    )
    {
      // TEUCHOS_FUNC_TIME_MONITOR("FluidEleCalcXFEM::get_material_parameters_volume_cell");

      // Initiate dummy variables:
      //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      // get element-wise velocity/pressure field for current time step
      my::evelaf_.clear();
      // Scatra field
      my::escabofoaf_.clear();
      my::escaaf_.clear();
      my::escaam_.clear();
      // set thermodynamic pressure at n+1/n+alpha_F and n+alpha_M/n and
      // its time derivative at n+alpha_M/n+1 (LOMA specific!!)
      const double thermpressaf = 1.0;
      const double thermpressam = 1.0;
      const double thermpressdtaf = 0.0;
      const double thermpressdtam = 0.0;
      const double vol = 0.0;
      // Values of material parameters at not needed time steps.
      double densn = 0.0;
      double densam = 0.0;
      double viscn = 0.0;
      //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      // Get material from FluidEleCalc routine.
      // If non-constant density or viscosity wants to be calculated on the different sides. A
      // review over how the scalar variables are set at the surface should be made.
      my::get_material_params(material, my::evelaf_, my::epreaf_, my::epream_, my::escaaf_,
          my::escaam_, my::escabofoaf_, thermpressaf, thermpressam, thermpressdtaf, thermpressdtam,
          vol, densam, densaf, densn, viscaf, viscn, gamma);

      return;
    }

  }  // end namespace Elements
}  // end namespace Discret

// Ursula is responsible for this comment!
template class Discret::Elements::FluidEleCalcXFEM<Core::FE::CellType::hex8>;
template class Discret::Elements::FluidEleCalcXFEM<Core::FE::CellType::hex20>;
template class Discret::Elements::FluidEleCalcXFEM<Core::FE::CellType::hex27>;
template class Discret::Elements::FluidEleCalcXFEM<Core::FE::CellType::tet4>;
template class Discret::Elements::FluidEleCalcXFEM<Core::FE::CellType::tet10>;
template class Discret::Elements::FluidEleCalcXFEM<Core::FE::CellType::wedge6>;
template class Discret::Elements::FluidEleCalcXFEM<Core::FE::CellType::wedge15>;
// template class Discret::Elements::FluidEleCalcXFEM<Core::FE::CellType::pyramid5>;

FOUR_C_NAMESPACE_CLOSE
