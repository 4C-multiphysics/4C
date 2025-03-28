// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_MULTIPORO_REAC_HPP
#define FOUR_C_SCATRA_ELE_CALC_MULTIPORO_REAC_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_fluidporo_multiphase_reactions.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_mat_scatra_multiporo.hpp"
#include "4C_porofluid_pressure_based_ele_action.hpp"
#include "4C_porofluid_pressure_based_ele_calc.hpp"
#include "4C_porofluid_pressure_based_ele_calc_utils.hpp"
#include "4C_porofluid_pressure_based_ele_evaluator.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"
#include "4C_porofluid_pressure_based_ele_phasemanager.hpp"
#include "4C_porofluid_pressure_based_ele_variablemanager.hpp"
#include "4C_scatra_ele_calc_poro_reac.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Mat
{
  class ScatraMat;
}

namespace Discret
{
  namespace Elements
  {
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerMultiPoro;

    template <Core::FE::CellType distype>
    class ScaTraEleCalcMultiPoroReac : public ScaTraEleCalcPoroReac<distype>
    {
     private:
      /// private constructor, since we are a Singleton.
      ScaTraEleCalcMultiPoroReac(
          const int numdofpernode, const int numscal, const std::string& disname);

      typedef ScaTraEleCalc<distype> my;
      typedef ScaTraEleCalcPoroReac<distype> pororeac;
      typedef ScaTraEleCalcPoro<distype> poro;
      typedef ScaTraEleCalcAdvReac<distype> advreac;
      using my::nen_;
      using my::nsd_;

     public:
      /// Singleton access method
      static ScaTraEleCalcMultiPoroReac<distype>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);

      /// Setup element evaluation
      int setup_calc(
          Core::Elements::Element* ele, Core::FE::Discretization& discretization) override;

     protected:
      //! extract element based or nodal values
      //  return extracted values of phinp
      void extract_element_and_node_values(Core::Elements::Element* ele,
          Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la) override;

      //! extract element based or nodal values --> L2-projection case: called within
      //! extract_element_and_node_values
      //  return extracted values of phinp
      virtual void extract_nodal_flux(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          const int numfluidphases);

      //! set internal variables
      void set_internal_variables_for_mat_and_rhs() override;

      //! evaluate material
      void materials(const std::shared_ptr<const Core::Mat::Material>
                         material,  //!< pointer to current material
          const int k,              //!< id of current scalar
          double& densn,            //!< density at t_(n)
          double& densnp,           //!< density at t_(n+1) or t_(n+alpha_F)
          double& densam,           //!< density at t_(n+alpha_M)
          double& visc,             //!< fluid viscosity
          const int iquad = -1      //!< id of current gauss point (default = -1)
          ) override;

      //! material mat_multi_poro_fluid
      virtual void mat_multi_poro_fluid(const std::shared_ptr<const Core::Mat::Material>
                                            material,  //!< pointer to current material
          const int k,                                 //!< id of current scalar
          double& densn,                               //!< density at t_(n)
          double& densnp,                              //!< density at t_(n+1) or t_(n+alpha_F)
          double& densam,                              //!< density at t_(n+alpha_M)
          double& visc,                                //!< fluid viscosity
          const int iquad = -1                         //!< id of current gauss point (default = -1)
      );

      //! material mat_multi_poro_vol_frac
      virtual void mat_multi_poro_vol_frac(const std::shared_ptr<const Core::Mat::Material>
                                               material,  //!< pointer to current material
          const int k,                                    //!< id of current scalar
          double& densn,                                  //!< density at t_(n)
          double& densnp,                                 //!< density at t_(n+1) or t_(n+alpha_F)
          double& densam,                                 //!< density at t_(n+alpha_M)
          double& visc,                                   //!< fluid viscosity
          const int iquad = -1  //!< id of current gauss point (default = -1)
      );

      //! material mat_multi_poro_solid
      virtual void mat_multi_poro_solid(const std::shared_ptr<const Core::Mat::Material>
                                            material,  //!< pointer to current material
          const int k,                                 //!< id of current scalar
          double& densn,                               //!< density at t_(n)
          double& densnp,                              //!< density at t_(n+1) or t_(n+alpha_F)
          double& densam,                              //!< density at t_(n+alpha_M)
          double& visc,                                //!< fluid viscosity
          const int iquad = -1                         //!< id of current gauss point (default = -1)
      );

      //! material mat_multi_poro_temperature
      virtual void mat_multi_poro_temperature(const std::shared_ptr<const Core::Mat::Material>
                                                  material,  //!< pointer to current material
          const int k,                                       //!< id of current scalar
          double& densn,                                     //!< density at t_(n)
          double& densnp,       //!< density at t_(n+1) or t_(n+alpha_F)
          double& densam,       //!< density at t_(n+alpha_M)
          double& visc,         //!< fluid viscosity
          const int iquad = -1  //!< id of current gauss point (default = -1)
      );

      //! Set advanced reaction terms and derivatives
      void set_advanced_reaction_terms(const int k,                  //!< index of current scalar
          const std::shared_ptr<Mat::MatListReactions> matreaclist,  //!< index of current scalar
          const double* gpcoord  //!< current Gauss-point coordinates
          ) override;

      //! compute pore pressure
      double compute_pore_pressure() override;

      //! get internal variable manager for multiporo formulation
      std::shared_ptr<ScaTraEleInternalVariableManagerMultiPoro<nsd_, nen_>> var_manager()
      {
        return std::static_pointer_cast<ScaTraEleInternalVariableManagerMultiPoro<nsd_, nen_>>(
            my::scatravarmanager_);
      };

      //! calculation of convective element matrix in convective form
      //! the only difference to the base class version is, that there is no scaling with the
      //! density
      void calc_mat_conv(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                                           //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double densnp,      //!< density at time_(n+1)
          const Core::LinAlg::Matrix<nen_, 1>& sgconv  //!< subgrid-scale convective operator
          ) override;

      //! adaption of convective term for rhs
      //! the only difference to the base class version is, that there is no scaling with the
      //! density
      void recompute_conv_phi_for_rhs(const int k,        //!< index of current scalar
          const Core::LinAlg::Matrix<nsd_, 1>& sgvelint,  //!< subgrid-scale velocity at Gauss point
          const double densnp,                            //!< density at time_(n+1)
          const double densn,                             //!< density at time_(n)
          const double vdiv                               //!< velocity divergence
          ) override;

      //! calculation of convective element matrix: add conservative contributions
      //! the only difference to the base class version is, that there is no scaling with the
      //! density
      void calc_mat_conv_add_cons(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double vdiv,        //!< velocity divergence
          const double densnp       //!< density at time_(n+1)
          ) override;

      //! calculation of mass element matrix (standard shape functions)
      void calc_mat_mass(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int& k,                                          //!< index of current scalar
          const double& fac,                                     //!< domain-integration factor
          const double& densam                                   //!< density at time_(n+am)
          ) override;

      //! calculation of convective element matrix (OD term structure coupling)
      //! difference to base class: linearization of mesh motion + shapederivatives pressure
      //! gradient have to be included
      void calc_conv_od_mesh(Core::LinAlg::SerialDenseMatrix& emat, const int k,
          const int ndofpernodemesh, const double fac, const double rhsfac, const double densnp,
          const double J, const Core::LinAlg::Matrix<nsd_, 1>& gradphi,
          const Core::LinAlg::Matrix<nsd_, 1>& convelint) override;

      //! calculation of linearized mass (off diagonal/shapederivative term mesh)
      //! difference to base class: linearization of porosity is included
      void calc_lin_mass_od_mesh(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodemesh,              //!< number of dofs per node of ale element
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double fac,     //!< domain-integration factor
          const double densam,  //!< density at time_(n+am)
          const double densnp,  //!< density at time_(n+1)
          const double phinp,   //!< scalar at time_(n+1)
          const double hist,    //!< history of time integartion
          const double J,       //!< determinant of Jacobian det(dx/ds)
          const Core::LinAlg::Matrix<1, nsd_ * nen_>&
              dJ_dmesh  //!< derivative of det(dx/ds) w.r.t. mesh displacement
          ) override;

      //! standard Galerkin transient, old part of rhs and source term (off diagonal/shapederivative
      //! term mesh) difference to base class: linearization of porosity and advanced reaction terms
      //! are included
      void calc_hist_and_source_od_mesh(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element current to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodemesh,              //!< number of dofs per node of ale element
          const double fac,                       //!< domain-integration factor
          const double rhsint,                    //!< rhs at Gauss point
          const double J,                         //!< determinant of Jacobian det(dx/ds)
          const Core::LinAlg::Matrix<1, nsd_ * nen_>&
              dJ_dmesh,        //!< derivative of det(dx/ds) w.r.t. mesh displacement
          const double densnp  //!< density
          ) override;

      //! standard Galerkin diffusive term (off diagonal/shapederivative term mesh)
      //! difference to base class: linearization of porosity and effective diffusivity
      void calc_diff_od_mesh(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element current to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodemesh,              //!< number of dofs per node of ale element
          const double diffcoeff,                 //!< diffusion coefficient
          const double fac,                       //!< domain-integration factor
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double J,       //!< determinant of Jacobian det(dx/ds)
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi,    //!< scalar gradient at Gauss point
          const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity
          const Core::LinAlg::Matrix<1, nsd_ * nen_>&
              dJ_dmesh  //!< derivative of det(dx/ds) w.r.t. mesh displacement
          ) override;

      //! reactive terms (standard Galerkin) (off diagonal/shapederivative term mesh)
      //! difference to base class: linearization of porosity
      void calc_react_od_mesh(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element current to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodemesh,              //!< number of dofs per node of ale element
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double rea_phi,  //!< reactive term
          const double J,        //!< determinant of Jacobian det(dx/ds)
          const Core::LinAlg::Matrix<1, nsd_ * nen_>&
              dJ_dmesh  //!< derivative of det(dx/ds) w.r.t. mesh displacement
          ) override;

      //! calculation of convective element matrix in convective form (off diagonal term fluid)
      void calc_mat_conv_od_fluid(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodefluid,  //!< number of dofs per node of fluid element // only a dummy
                                       //!< variable
          const double rhsfac,         //!< domain-integration factor times time-integration factor
          const double densnp,         //!< density at time_(n+1)
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi  //!< scalar gradient
          ) override;

      //! calculation of convective element matrix in convective form -- additional conservative
      //! contributions (off diagonal term fluid) not yet implemented --> FOUR_C_THROW
      void calc_mat_conv_add_cons_od_fluid(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodefluid,             //!< number of dofs per node of fluid element
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double densnp,      //!< density at time_(n+1)
          const double phinp        //!< scalar at time_(n+1)
          ) override;

      //! calculation of linearized mass (off diagonal terms fluid)
      //! linearization of porosity*saturation*vtrans
      void calc_lin_mass_od_fluid(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodemesh,  //!< number of dofs per node of fluid element // only a dummy
                                      //!< variable
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double fac,     //!< domain-integration factor
          const double densam,  //!< density at time_(n+am)
          const double densnp,  //!< density at time_(n+1)
          const double phinp,   //!< scalar at time_(n+1)
          const double hist     //!< history of time integartion
          ) override;

      // calculate linearization of a mass matrix type matrix (OD-fluid terms)
      virtual void calc_lin_mass_matrix_type_od_fluid(Core::LinAlg::SerialDenseMatrix& emat,
          const int k, const std::vector<double>* prefaclinmassodfluid,
          const int totalnummultiphasedofpernode, double prefac);

      //! standard Galerkin transient, old part of rhs and source term (off diagonal terms fluid)
      //! linearization of porosity*saturation*vtrans + advanced reaction terms
      void calc_hist_and_source_od_fluid(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element current to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodemesh,              //!< number of dofs per node of ale element
          const double fac,                       //!< domain-integration factor
          const double rhsint,                    //!< rhs at Gauss point
          const double densnp                     //!< density
          ) override;

      //! standard Galerkin reactive term (off diagonal terms fluid)
      //! linearization of porosity*saturation*vreact
      void calc_react_od_fluid(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element current to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodemesh,              //!< number of dofs per node of ale element
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double rea_phi  //!< rhs at Gauss point
          ) override;

      //! standard Galerkin diffusive term (off diagonal terms fluid)
      //! linearization of porosity*saturation*d_eff
      void calc_diff_od_fluid(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element current to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodemesh,              //!< number of dofs per node of ale element
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi  //!< scalar gradient at Gauss point
          ) override;

      //! fill coupling vector and add variables to reaction in order to compute reaction values and
      //! derivatives
      void fill_coupling_vector_and_add_variables(
          const int k, Mat::MatListReactions& matreaclist, ScaTraEleReaManagerAdvReac& remanager);

      //! Get right hand side including reaction bodyforce term
      void get_rhs_int(double& rhsint,  //!< rhs containing bodyforce at Gauss point
          const double densnp,          //!< density at t_(n+1)
          const int k                   //!< index of current scalar
          ) override;

      //! calculation of reactive element matrix
      void calc_mat_react(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double
              timetaufac,  //!< domain-integration factor times time-integration factor times tau
          const double taufac,                          //!< domain-integration factor times tau
          const double densnp,                          //!< density at time_(n+1)
          const Core::LinAlg::Matrix<nen_, 1>& sgconv,  //!< subgrid-scale convective operator
          const Core::LinAlg::Matrix<nen_, 1>& diff     //!< laplace term
          ) override;

      //! "shapederivatives" pressure gradient
      virtual void apply_shape_derivs_pressure_grad(Core::LinAlg::SerialDenseMatrix& emat,
          const int k, const double vrhs, const Core::LinAlg::Matrix<nsd_, 1>& gradphi,
          const Core::LinAlg::Matrix<nsd_, 1> refgradpres);

     protected:
      //! nodal flux values at t_(n+1)
      std::vector<Core::LinAlg::Matrix<nsd_, nen_>> efluxnp_;

      //! a vector containing all quantities, the equation is coupled with
      //! (i.e. pressures, saturations and porosity)
      std::vector<std::pair<std::string, double>> couplingvalues_;

      //! penalty factor to avoid very small "densities"
      const double penalty_ = 1.0;

      //! do we use L2-projection or evaluation at GP
      bool L2_projection_ = false;
    };


    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerMultiPoro
        : public ScaTraEleInternalVariableManager<nsd, nen>
    {
      typedef ScaTraEleInternalVariableManager<nsd, nen> my;

     public:
      ScaTraEleInternalVariableManagerMultiPoro(int numscal)
          : ScaTraEleInternalVariableManager<nsd, nen>(numscal),
            pressure_(0),
            saturation_(0),
            density_(0),
            solidpressure_(0.0),
            delta_(numscal, 0.0),
            relative_mobility_funct_id_(numscal),
            heatcapacity_(0),
            thermaldiffusivity_(0),
            heatcapacityeff_(0.0),
            min_val_of_phase_(numscal, 0.0),
            evaluate_scalar_(numscal, true),
            scalartophasemap_(
                numscal, {-1, Mat::ScaTraMatMultiPoro::SpeciesType::species_undefined}),
            materialset_(false),
            myaction_(ScaTra::Action::calc_mat_and_rhs)
      {
        return;
      }

      // compute and set internal variables -- no L2-projection but evaluation at GP
      void set_internal_variables_multi_poro(
          const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
          const Core::LinAlg::Matrix<nsd, nen>&
              derxy,  //! global derivatives of shape functions w.r.t x,y,z
          const Core::LinAlg::Matrix<nsd, nen>&
              deriv,  //! global derivatives of shape functions w.r.t r,s,t
          const Core::LinAlg::Matrix<nsd, nsd>& xjm, const Core::LinAlg::Matrix<nsd, nen>& xyze0,
          const std::vector<Core::LinAlg::Matrix<nen, 1>>&
              ephinp,  //! scalar at t_(n+1) or t_(n+alpha_F)
          const std::vector<Core::LinAlg::Matrix<nen, 1>>& ephin,  //! scalar at t_(n)
          const std::vector<Core::LinAlg::Matrix<nen, 1>>&
              ehist,  //! history vector of transported scalars
          const Core::LinAlg::Matrix<nsd, nen>&
              eforcevelocity  //! nodal velocity due to external force
      )
      {
        // call base class (scatra) with dummy variable
        const Core::LinAlg::Matrix<nsd, nen> dummy_econv(Core::LinAlg::Initialization::zero);
        my::set_internal_variables(funct, derxy, ephinp, ephin, dummy_econv, ehist, dummy_econv);

        // velocity due to the external force
        Core::LinAlg::Matrix<nsd, 1> force_velocity;
        force_velocity.multiply(eforcevelocity, funct);

        //------------------------get determinant of Jacobian dX / ds
        // transposed jacobian "dX/ds"
        Core::LinAlg::Matrix<nsd, nsd> xjm0;
        xjm0.multiply_nt(deriv, xyze0);

        // inverse of transposed jacobian "ds/dX"
        const double det0 = xjm0.determinant();

        const double det = xjm.determinant();

        // determinant of deformationgradient det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds)
        // )^-1
        const double JacobianDefGradient = det / det0;

        // clear current gauss point data for safety
        phasemanager_->clear_gp_state();
        variablemanager_->evaluate_gp_variables(funct, derxy);
        // access from outside to the phasemanager: scatra-discretization has fluid-dis on dofset 2
        phasemanager_->evaluate_gp_state(
            JacobianDefGradient, *variablemanager_, ndsscatra_porofluid_);

        const int numfluidphases = phasemanager_->num_fluid_phases();
        const int numvolfrac = phasemanager_->num_vol_frac();

        // resize all phase related vectors
        pressure_.resize(numfluidphases);
        saturation_.resize(numfluidphases);
        density_.resize(numfluidphases + numvolfrac);
        heatcapacity_.resize(numfluidphases + numvolfrac + 1);
        thermaldiffusivity_.resize(numfluidphases + numvolfrac + 1);
        pressuregrad_.resize(numfluidphases + numvolfrac);
        difftensorsfluid_.resize(numfluidphases + numvolfrac);
        abspressuregrad_.resize(numfluidphases);
        volfrac_.resize(numvolfrac);
        volfracpressure_.resize(numvolfrac);
        relative_mobility_funct_id_.resize(my::numscal_);

        const std::vector<Core::LinAlg::Matrix<nsd, 1>>& fluidgradphi =
            *(variablemanager_->grad_phinp());

        volfrac_ = phasemanager_->vol_frac();
        volfracpressure_ = phasemanager_->vol_frac_pressure();

        //! convective velocity
        std::vector<Core::LinAlg::Matrix<nsd, 1>> phase_fluid_velocity(0.0);
        phase_fluid_velocity.resize(numfluidphases + numvolfrac);
        //! convective part in convective form: u_x*N,x + u_y*N,y
        std::vector<Core::LinAlg::Matrix<nen, 1>> phase_fluid_velocity_conv(0.0);
        phase_fluid_velocity_conv.resize(numfluidphases + numvolfrac);

        //! temperature convective velocity
        Core::LinAlg::Matrix<nsd, 1> temperatureconvelint(Core::LinAlg::Initialization::zero);
        //! temperature convective part in convective form
        Core::LinAlg::Matrix<nen, 1> temperatureconv(Core::LinAlg::Initialization::zero);

        for (int i_phase = 0; i_phase < numfluidphases; ++i_phase)
        {
          // current pressure gradient
          pressuregrad_[i_phase].clear();

          // phase density
          density_[i_phase] = phasemanager_->density(i_phase);

          // compute the pressure gradient from the phi gradients
          for (int idof = 0; idof < numfluidphases; ++idof)
            pressuregrad_[i_phase].update(
                phasemanager_->pressure_deriv(i_phase, idof), fluidgradphi[idof], 1.0);

          // compute the absolute value of the pressure gradient from the phi gradients
          abspressuregrad_[i_phase] = 0.0;
          for (int i = 0; i < nsd; i++)
            abspressuregrad_[i_phase] += pressuregrad_[i_phase](i) * pressuregrad_[i_phase](i);
          abspressuregrad_[i_phase] = sqrt(abspressuregrad_[i_phase]);

          // diffusion tensor
          difftensorsfluid_[i_phase].clear();
          phasemanager_->permeability_tensor(i_phase, difftensorsfluid_[i_phase]);
          difftensorsfluid_[i_phase].scale(phasemanager_->rel_permeability(i_phase) /
                                           phasemanager_->dyn_viscosity(i_phase,
                                               abspressuregrad_[i_phase], ndsscatra_porofluid_));

          // Insert Darcy's law: porosity*S^\pi*(v^\pi - v_s) = - phase/\mu * grad p
          phase_fluid_velocity[i_phase].multiply(
              -1.0, difftensorsfluid_[i_phase], pressuregrad_[i_phase]);
          temperatureconvelint.update(
              heatcapacity_[i_phase] * density_[i_phase], phase_fluid_velocity[i_phase], 1.0);
          // in convective form: u_x*N,x + u_y*N,y
          phase_fluid_velocity_conv[i_phase].multiply_tn(derxy, phase_fluid_velocity[i_phase]);
          temperatureconv.update(
              heatcapacity_[i_phase] * density_[i_phase], phase_fluid_velocity_conv[i_phase], 1.0);

          // phase pressure
          pressure_[i_phase] = phasemanager_->pressure(i_phase);
          // phase saturation
          saturation_[i_phase] = phasemanager_->saturation(i_phase);
        }

        for (int i_volfrac = numfluidphases; i_volfrac < numfluidphases + numvolfrac; ++i_volfrac)
        {
          // current pressure gradient
          pressuregrad_[i_volfrac].update(1.0, fluidgradphi[i_volfrac + numvolfrac], 0.0);

          // vol frac density
          density_[i_volfrac] = phasemanager_->vol_frac_density(i_volfrac - numfluidphases);

          // diffusion tensor
          difftensorsfluid_[i_volfrac].clear();
          phasemanager_->permeability_tensor_vol_frac_pressure(
              i_volfrac - numfluidphases, difftensorsfluid_[i_volfrac]);
          difftensorsfluid_[i_volfrac].scale(
              1.0 / phasemanager_->dyn_viscosity_vol_frac_pressure(i_volfrac - numfluidphases, -1.0,
                        ndsscatra_porofluid_));  // -1.0 --> don't need abspressgrad

          // Insert Darcy's law: porosity*(v^\pi - v_s) = - k/\mu * grad p
          phase_fluid_velocity[i_volfrac].multiply(
              -1.0, difftensorsfluid_[i_volfrac], pressuregrad_[i_volfrac]);
          temperatureconvelint.update(
              heatcapacity_[i_volfrac] * density_[i_volfrac], phase_fluid_velocity[i_volfrac], 1.0);
          // in convective form: u_x*N,x + u_y*N,y
          phase_fluid_velocity_conv[i_volfrac].multiply_tn(derxy, phase_fluid_velocity[i_volfrac]);
          temperatureconv.update(heatcapacity_[i_volfrac] * density_[i_volfrac],
              phase_fluid_velocity_conv[i_volfrac], 1.0);
        }

        // solid pressure
        solidpressure_ = phasemanager_->solid_pressure();

        for (int k = 0; k < my::numscal_; ++k)
        {
          // overwrite convective term
          // - rho * k/\mu*grad p * grad phi
          switch (scalartophasemap_[k].species_type)
          {
            case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
            case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
              my::convelint_[k] = phase_fluid_velocity[scalartophasemap_[k].phaseID];
              // if the scalar reacts to the external force, add the velocity due to the external
              // force scaled with the relative mobility and the porosity * saturation
              if (my::reacts_to_force_[k])
              {
                const auto prefactor = evaluate_relative_mobility(k) * phasemanager_->porosity() *
                                       phasemanager_->saturation(scalartophasemap_[k].phaseID);
                my::convelint_[k].update(prefactor, force_velocity, 1.0);
              }
              my::conv_[k].multiply_tn(derxy, my::convelint_[k]);
              my::conv_phi_[k] = my::convelint_[k].dot(my::gradphi_[k]);
              break;

            case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
              my::convelint_[k] = Core::LinAlg::Matrix<nsd, 1>(Core::LinAlg::Initialization::zero);
              my::conv_[k] = Core::LinAlg::Matrix<nen, 1>(Core::LinAlg::Initialization::zero);
              my::conv_phi_[k] = 0;
              break;

            case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
              my::convelint_[k] = temperatureconvelint;
              my::conv_[k] = temperatureconv;
              my::conv_phi_[k] = temperatureconvelint.dot(my::gradphi_[k]);
              break;

            default:
              FOUR_C_THROW(
                  "unknown species type {} for species {}!", scalartophasemap_[k].species_type, k);
              break;
          }
          // set flag if we actually have to evaluate the species
          set_evaluate_scalar_flag(k);
        }
      };

      // adapt convective term in case of L2-projection
      void adapt_convective_term_for_l2(
          const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
          const Core::LinAlg::Matrix<nsd, nen>&
              derxy,  //! global derivatives of shape functions w.r.t x,y,z
          const std::vector<Core::LinAlg::Matrix<nsd, nen>>&
              efluxnp  //! nodal flux values at t_(n+1) or t_(n+alpha_F)
      )
      {
        const int numfluidphases = efluxnp.size();

        std::vector<Core::LinAlg::Matrix<nsd, 1>> flux(0.0);
        flux.resize(numfluidphases);

        // in convective form: q_x*N,x + q_y*N,y
        std::vector<Core::LinAlg::Matrix<nen, 1>> flux_conv(0.0);
        flux_conv.resize(numfluidphases);

        for (int i_phase = 0; i_phase < numfluidphases; ++i_phase)
        {
          flux[i_phase].multiply(1.0, efluxnp[i_phase], funct);
          flux_conv[i_phase].multiply_tn(derxy, flux[i_phase]);
        }

        for (auto& k : scalartophasemap_)
        {
          if (k.phaseID < 0 or k.phaseID >= numfluidphases)
            FOUR_C_THROW("Invalid phase ID {}", k.phaseID);
        }

        // set convective term
        for (int k = 0; k < my::numscal_; ++k)
        {
          switch (scalartophasemap_[k].species_type)
          {
            case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
            case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
              my::conv_phi_[k] = flux[scalartophasemap_[k].phaseID].dot(my::gradphi_[k]);
              break;

            case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
            case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
              my::conv_phi_[k] = 0;
              break;

            default:
              FOUR_C_THROW(
                  "unknown species type {} for species {}!", scalartophasemap_[k].species_type, k);
              break;
          }
        }
        return;
      };

      // Set the fluid-material in the scatra-Varmanager
      void set_fluid_poromultiphase_material(Core::Elements::Element* ele)
      {
        // check if we actually have three materials
        if (ele->num_material() < 3) FOUR_C_THROW("no third material available");

        // here we rely that the PoroMultiPhase material has been added as third material
        multiphasemat_ = std::dynamic_pointer_cast<Mat::FluidPoroMultiPhase>(
            ele->material(ndsscatra_porofluid_));
        if (multiphasemat_ == nullptr) FOUR_C_THROW("cast to Mat::FluidPoroMultiPhase failed!");

        materialset_ = true;
      }

      /*========================================================================*/
      //! @name return methods for internal variables
      /*========================================================================*/

      //! return pressure associated with scalar k
      double pressure(const int k) const
      {
        switch (scalartophasemap_[k].species_type)
        {
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
            return pressure_[scalartophasemap_[k].phaseID];

          default:
            FOUR_C_THROW("ScalartophaseID = {} for species {}", scalartophasemap_[k].phaseID, k);
            return 0;
        }
      };

      //! return pressures
      const std::vector<double>& pressure() const { return pressure_; };

      //! return saturation associated with scalar k
      double saturation(const int k) const
      {
        switch (scalartophasemap_[k].species_type)
        {
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
            return saturation_[scalartophasemap_[k].phaseID];

          default:
            FOUR_C_THROW("ScalartophaseID = {} for species {}", scalartophasemap_[k].phaseID, k);
            return 0;
        }
      };

      //! return saturation associated with scalar k
      const std::vector<double>& saturation() const { return saturation_; };

      //! return density associated with scalar k
      double density(const int k) const
      {
        switch (scalartophasemap_[k].species_type)
        {
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
            return density_[scalartophasemap_[k].phaseID];

          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
            return phasemanager_->solid_density();

          case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
            // set to 1.0 because densities are included in the effective heat capacity
            return 1.0;

          default:
            FOUR_C_THROW(
                "unknown species type {} for species {}!", scalartophasemap_[k].species_type, k);
            return 0;
        }
      };

      //! return density vector
      const std::vector<double>& density() const { return density_; };

      //! return volfrac vector
      const std::vector<double>& vol_frac() const { return volfrac_; };

      //! return volume fraction associated with scalar k
      double vol_frac(const int k) const
      {
        switch (scalartophasemap_[k].species_type)
        {
          // case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
            return volfrac_[get_phase_id(k) - phasemanager_->num_fluid_phases()];

          default:
            FOUR_C_THROW("ScalartophaseID = {} for species {}", scalartophasemap_[k].phaseID, k);
            return 0;
        }
      };

      //! return volfrac pressure vector
      const std::vector<double>& vol_frac_pressure() const { return volfracpressure_; };

      //! return solid pressure
      double solid_pressure() const { return solidpressure_; };

      //! set scalar ID to phase ID mapping and species type
      void set_phase_id_and_species_type(const int scalarID, const int phaseID,
          const Mat::ScaTraMatMultiPoro::SpeciesType& spectype)
      {
        scalartophasemap_[scalarID].phaseID = phaseID;
        scalartophasemap_[scalarID].species_type = spectype;
      };

      //! get phase ID from scalar ID
      int get_phase_id(const int scalarID) const
      {
        if (scalartophasemap_[scalarID].phaseID < 0)
          FOUR_C_THROW(
              "ScalartophaseID = {} for species {}", scalartophasemap_[scalarID].phaseID, scalarID);

        return scalartophasemap_[scalarID].phaseID;
      };

      //! get species type of scalar 'k'
      Mat::ScaTraMatMultiPoro::SpeciesType get_species_type(const int k)
      {
        return scalartophasemap_[k].species_type;
      };

      //! set delta for evaluation of effective diffusivity
      void set_delta(const double delta, const int k) { delta_[k] = delta; };

      //! set relative mobility function ID
      void set_relative_mobility_function_id(
          const int relative_mobility_funct_id, const int current_scalar)
      {
        relative_mobility_funct_id_[current_scalar] = relative_mobility_funct_id;
      };

      //! set heat capacity
      void set_heat_capacity(std::vector<double> cp) { heatcapacity_ = cp; };

      //! set effective heat capacity
      void set_effective_heat_capacity(double cp_eff) { heatcapacityeff_ = cp_eff; };

      //! set thermal diffusivity
      void set_thermal_diffusivity(std::vector<double> kappa) { thermaldiffusivity_ = kappa; };

      //! set minimum value of corresponding phase under which we assume that mass fraction is
      //! also zero
      void set_min_val_of_phase(const double minvalofphase, const int k)
      {
        min_val_of_phase_[k] = minvalofphase;
      };

      //! set action
      void set_action(const ScaTra::Action action) { myaction_ = action; };

      //! get delta
      double get_delta(const int k) { return delta_[k]; };

      //! get heat capacity
      //! heat capacity order [ <fluid>  <volfrac>  <solid> ]
      double get_heat_capacity(const int j) { return heatcapacity_[j]; }

      //! get effective heat capacity
      double get_effective_heat_capacity() { return heatcapacityeff_; }

      //! get thermal diffusivity
      //! thermal diffusivity order [ <fluid>  <volfrac>  <solid> ]
      double get_thermal_diffusivity(const int j) { return thermaldiffusivity_[j]; }

      //! get evaluate scalar flag
      bool evaluate_scalar(const int k) { return evaluate_scalar_[k]; };

      //! get minimum value of corresponding phase under which we assume that mass fraction is
      //! also zero
      double get_min_val_of_phase(const int k) { return min_val_of_phase_[k]; };

      //! get pre-factor needed for OD-mesh-linearization of mass matrix
      double get_pre_factor_for_mass_matrix_od_mesh(const int k, const double fac)
      {
        double prefac = fac;

        switch (scalartophasemap_[k].species_type)
        {
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          {
            // linearization of porosity
            //------------------------------------------------dporosity/dd = dporosity/dJ * dJ/dd =
            // dporosity/dJ * J * N_x
            // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X )
            // = det (dx/ds) * ( det(dX/ds) )^-1 in our case: prefac is scaled with density, i.e.
            // porosity --> scale it with 1.0/porosity here

            // we will pass fac/poro*J*dporosity/dJ into CalcMatMassODMesh, where it will internally
            // be scaled correctly

            if (phasemanager_->porosity_depends_on_struct())
              prefac += fac / (phasemanager_->porosity()) * phasemanager_->jacobian_def_grad() *
                        phasemanager_->porosity_deriv_wrt_jacobian_def_grad();
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
          {
            // do nothing: correct prefac = fac
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
          {
            if (phasemanager_->porosity_depends_on_struct())
              prefac -= fac / (1 - phasemanager_->porosity() - phasemanager_->sum_add_vol_frac()) *
                        phasemanager_->jacobian_def_grad() *
                        phasemanager_->porosity_deriv_wrt_jacobian_def_grad();

            // else
            // do nothing: correct prefac = fac

            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
          {
            if (phasemanager_->porosity_depends_on_struct())
            {
              // evaluate the effective heat capacity factor
              const int numdofs = phasemanager_->num_fluid_phases() + phasemanager_->num_vol_frac();
              const int numfluidphases = phasemanager_->num_fluid_phases();

              double cp_eff = get_effective_heat_capacity();

              prefac += fac / cp_eff * phasemanager_->solid_density() * get_heat_capacity(numdofs) *
                        (-1.0) * phasemanager_->jacobian_def_grad() *
                        phasemanager_->porosity_deriv_wrt_jacobian_def_grad();

              for (int phase = 0; phase < numfluidphases; ++phase)
              {
                prefac += fac / cp_eff * phasemanager_->density(phase) * get_heat_capacity(phase) *
                          phasemanager_->jacobian_def_grad() * phasemanager_->saturation(phase) *
                          phasemanager_->porosity_deriv_wrt_jacobian_def_grad();
              }
            }
            break;
          }
          default:
            FOUR_C_THROW(
                "unknown species type {} for species {}!", scalartophasemap_[k].species_type, k);
            break;
        }  // switch(species type)

        return prefac;
      };

      //! get pre-factor needed for OD-fluid-linearization of mass matrix
      void get_pre_fac_lin_mass_od_fluid(const int k, std::vector<double>* prefaclinmassodfluid)
      {
        // reset to zero
        std::fill(prefaclinmassodfluid->begin(), prefaclinmassodfluid->end(), 0.0);

        const int numfluidphases = phasemanager_->num_fluid_phases();

        switch (scalartophasemap_[k].species_type)
        {
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          {
            const int curphase = get_phase_id(k);
            const int totalnummultiphasedofpernode = multiphase_mat()->num_mat();

            // d poro / d S_j = SaturationDeriv; scaling with 1.0/S since term is re-scaled with
            // densnp = poro*S*rho
            for (int idof = 0; idof < numfluidphases; ++idof)
              (*prefaclinmassodfluid)[idof] += phasemanager_->saturation_deriv(curphase, idof) /
                                               phasemanager_->saturation(curphase);

            // linearization of porosity only if porosity is pressure-dependent
            // d poro / d psi_j = porosityDeriv; scaling with 1.0/poro since term is re-scaled with
            // densnp = poro*S*rho
            if (phasemanager_->porosity_depends_on_fluid())
            {
              for (int idof = 0; idof < totalnummultiphasedofpernode; ++idof)
                (*prefaclinmassodfluid)[idof] +=
                    phasemanager_->porosity_deriv(idof) / phasemanager_->porosity();
            }
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
          {
            const int curphase = get_phase_id(k);
            // d volfrac_j / d volfrac_j = 1.0; scaling with 1.0/volfrac since term is re-scaled
            // with densnp = rho*volfrac
            (*prefaclinmassodfluid)[curphase] += 1.0 / vol_frac(k);

            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
          {
            // porosity of solid epsilon_s does not depend on fluid primary variables

            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
          {
            // evaluate the effective heat capacity factor
            const int numdofs = phasemanager_->num_fluid_phases() + phasemanager_->num_vol_frac();
            const int totalnummultiphasedofpernode = multiphase_mat()->num_mat();

            double cp_eff = get_effective_heat_capacity();

            for (int phase = 0; phase < numfluidphases; ++phase)
            {
              for (int idof = 0; idof < numfluidphases; ++idof)
                (*prefaclinmassodfluid)[idof] += phasemanager_->saturation_deriv(phase, idof) /
                                                 cp_eff * phasemanager_->density(phase) *
                                                 get_heat_capacity(phase) *
                                                 phasemanager_->porosity();

              if (phasemanager_->porosity_depends_on_fluid())
              {
                for (int idof = 0; idof < totalnummultiphasedofpernode; ++idof)
                  (*prefaclinmassodfluid)[idof] +=
                      phasemanager_->porosity_deriv(idof) / cp_eff * phasemanager_->density(phase) *
                      get_heat_capacity(phase) * phasemanager_->saturation(phase);
              }
            }

            for (int phase = numfluidphases; phase < numdofs; ++phase)
            {
              (*prefaclinmassodfluid)[phase] +=
                  get_heat_capacity(phase) *
                  phasemanager_->vol_frac_density(phase - numfluidphases) / cp_eff;
            }
            break;
          }
          default:
          {
            FOUR_C_THROW(
                "unknown species type {} for species {}!", scalartophasemap_[k].species_type, k);
            break;
          }
        }  // switch(species type)

        return;
      };

      //! evaluate relative mobility
      auto evaluate_relative_mobility(const int current_scalar)
      {
        std::vector<std::pair<std::string, double>> varfunction_variables;
        std::vector<std::pair<std::string, double>> varfunction_constants;
        const auto number_fluidphases = phasemanager_->num_fluid_phases();

        for (int i = 0; i < number_fluidphases; i++)
        {
          std::ostringstream temp;
          temp << i + 1;

          varfunction_variables.emplace_back("S" + temp.str(), phasemanager_->saturation(i));
          varfunction_variables.emplace_back("p" + temp.str(), phasemanager_->pressure(i));
        }
        varfunction_variables.emplace_back("porosity", phasemanager_->porosity());

        const auto relative_mobility =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfAnything>(
                    relative_mobility_funct_id_[current_scalar])
                .evaluate(varfunction_variables, varfunction_constants, 0);

        return relative_mobility;
      }

      //! get pre-factor needed for OD-fluid-linearization of convective term
      void get_pre_fac_lin_conv_od_fluid(const int k, const unsigned ui,
          std::vector<double>* prefaclinconvodfluid, const Core::LinAlg::Matrix<nsd, 1>& gradphi,
          const Core::LinAlg::Matrix<1, nsd>& gradphiTdifftensor,
          const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
          const int phase)
      {
        // reset to zero
        std::fill(prefaclinconvodfluid->begin(), prefaclinconvodfluid->end(), 0.0);

        // get correct factor
        double laplawf(0.0);
        for (unsigned j = 0; j < nsd; j++) laplawf += derxy(j, ui) * gradphiTdifftensor(0, j);

        const int numfluidphases = phasemanager_->num_fluid_phases();
        const int numvolfrac = phasemanager_->num_vol_frac();

        // fluid phase
        if (phase >= 0 && phase < numfluidphases)
        {
          // derivative after fluid pressures
          for (int idof = 0; idof < numfluidphases; ++idof)
            (*prefaclinconvodfluid)[idof] += laplawf * phasemanager_->pressure_deriv(phase, idof);

          // derivative after relative permeabilty
          if (not phasemanager_->has_constant_rel_permeability(phase))
          {
            const Core::LinAlg::Matrix<nsd, 1> gradpres = pressure_gradient(phase);
            const double abspressgrad = abs_pressure_gradient(phase);

            static Core::LinAlg::Matrix<nsd, nsd> difftensor(Core::LinAlg::Initialization::zero);
            phasemanager_->permeability_tensor(phase, difftensor);
            difftensor.scale(phasemanager_->rel_permeability_deriv(phase) /
                             phasemanager_->dyn_viscosity(phase, abspressgrad, 2));

            static Core::LinAlg::Matrix<1, nsd> gradphiTdifftensor(
                Core::LinAlg::Initialization::zero);
            gradphiTdifftensor.multiply_tn(gradphi, difftensor);

            double laplawf(0.0);
            for (unsigned j = 0; j < nsd; j++) laplawf += gradpres(j) * gradphiTdifftensor(0, j);

            for (int idof = 0; idof < numfluidphases; ++idof)
              (*prefaclinconvodfluid)[idof] +=
                  laplawf * funct(ui) * phasemanager_->saturation_deriv(phase, idof);
          }
        }
        // volfrac
        else if (phase < numfluidphases + numvolfrac)
        {
          // derivative after volfrac-pressure
          (*prefaclinconvodfluid)[phase + numvolfrac] += laplawf;
        }
        else
          FOUR_C_THROW("get_pre_fac_lin_conv_od_fluid has been called with phase {}", phase);

        return;
      };

      //! get pre-factor needed for OD-mesh-linearization of diffusive term
      double get_pre_factor_for_diff_matrix_od_mesh(
          const int k, const double rhsfac, const double diffcoeff)
      {
        double prefac = 0.0;

        switch (scalartophasemap_[k].species_type)
        {
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          {
            if (phasemanager_->porosity_depends_on_struct())
            {
              const double delta = get_delta(k);

              // linearization of porosity
              //------------------------------------------------dporosity/dd = dporosity/dJ * dJ/dd
              //= dporosity/dJ * J * N_x
              // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X
              // ) = det (dx/ds) * ( det(dX/ds) )^-1 in our case: diffusivity is scaled with
              // porosity^(delta + 1) --> scale it with 1.0/porosity^(delta + 1) here
              //              and build derivative d diff/d porosity = (delta + 1) * porosity^delta
              prefac = rhsfac * diffcoeff / std::pow(phasemanager_->porosity(), delta + 1.0) *
                       (delta + 1.0) * std::pow(phasemanager_->porosity(), delta) *
                       phasemanager_->jacobian_def_grad() *
                       phasemanager_->porosity_deriv_wrt_jacobian_def_grad();
            }
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
          {
            // do nothing: linearization performed in base class
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
          {
            if (phasemanager_->porosity_depends_on_struct())
            {
              const double delta = get_delta(k);

              // linearization of porosity
              //------------------------------------------------dporosity/dd =
              // dporosity/dJ * dJ/dd = dporosity/dJ * J * N_x
              // J denotes the determinant of the deformation gradient, i.e. det F =
              // det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1 in our case:
              // diffusivity is scaled with (1-porosity-sumaddvolfrac)^(delta + 1) -->
              // scale it with 1.0/(1-porosity-sumaddvolfrac)^(delta + 1) here
              //              and build derivative d diff/d porosity = (delta + 1) *
              //              (1-porosity-sumaddvolfrac)^delta
              prefac = -1.0 * rhsfac * diffcoeff /
                       std::pow(1.0 - phasemanager_->porosity() - phasemanager_->sum_add_vol_frac(),
                           delta + 1.0) *
                       (delta + 1.0) *
                       std::pow(1.0 - phasemanager_->porosity() - phasemanager_->sum_add_vol_frac(),
                           delta) *
                       phasemanager_->jacobian_def_grad() *
                       phasemanager_->porosity_deriv_wrt_jacobian_def_grad();
            }
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
          {
            if (phasemanager_->porosity_depends_on_struct())
            {
              const int numdofs = phasemanager_->num_fluid_phases() + phasemanager_->num_vol_frac();
              const int numfluidphases = phasemanager_->num_fluid_phases();

              prefac = rhsfac * get_thermal_diffusivity(numdofs) * (-1.0) *
                       phasemanager_->jacobian_def_grad() *
                       phasemanager_->porosity_deriv_wrt_jacobian_def_grad();

              for (int phase = 0; phase < numfluidphases; ++phase)
                prefac += rhsfac * get_thermal_diffusivity(phase) *
                          phasemanager_->jacobian_def_grad() * phasemanager_->saturation(phase) *
                          phasemanager_->porosity_deriv_wrt_jacobian_def_grad();
            }
            break;
          }
          default:
            FOUR_C_THROW(
                "unknown species type {} for species {}!", scalartophasemap_[k].species_type, k);
            break;
        }  // switch(species type)

        return prefac;
      };

      //! get pre-factor needed for OD-fluid-linearization of diffusive term
      void get_pre_fac_diff_od_fluid(const int k, const double rhsfac, const double diffcoeff,
          std::vector<double>* prefacdiffodfluid)
      {
        // reset to zero
        std::fill(prefacdiffodfluid->begin(), prefacdiffodfluid->end(), 0.0);

        switch (scalartophasemap_[k].species_type)
        {
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          {
            const int curphase = get_phase_id(k);
            const int numfluidphases = phasemanager_->num_fluid_phases();
            const int totalnummultiphasedofpernode = multiphase_mat()->num_mat();

            const double delta = get_delta(k);

            // linearization of saturation*porosity*d_eff = (saturation * porosity)^(delta+1) w.r.t
            // saturation
            //-------------------------------------------------------------------------------------------------------------
            // in our case: diffusivity is scaled with saturation^(delta + 1) --> scale it
            // with 1.0/saturation^(delta + 1) here
            //              and build derivative d diff/d saturation = (delta + 1) *
            //              saturation^delta
            const double vrhs_sat =
                rhsfac * diffcoeff / std::pow(phasemanager_->saturation(curphase), delta + 1.0) *
                (delta + 1.0) * std::pow(phasemanager_->saturation(curphase), delta);

            // linearization of saturation*porosity*d_eff = (saturation * porosity)^(delta+1) w.r.t
            // porosity
            //-------------------------------------------------------------------------------------------------------------
            // in our case: diffusivity is scaled with porosity^(delta + 1) --> scale it
            // with 1.0/porosity^(delta + 1) here
            //              and build derivative d diff/d porosity = (delta + 1) * porosity^delta
            const double vrhs_poro = rhsfac * diffcoeff /
                                     std::pow(phasemanager_->porosity(), delta + 1.0) *
                                     (delta + 1.0) * std::pow(phasemanager_->porosity(), delta);

            for (int idof = 0; idof < numfluidphases; ++idof)
              (*prefacdiffodfluid)[idof] +=
                  vrhs_sat * phasemanager_->saturation_deriv(curphase, idof);

            // linearization of porosity only if porosity is pressure-dependent
            if (phasemanager_->porosity_depends_on_fluid())
            {
              for (int idof = 0; idof < totalnummultiphasedofpernode; ++idof)
                (*prefacdiffodfluid)[idof] += vrhs_poro * phasemanager_->porosity_deriv(idof);
            }
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
          {
            const int curphase = get_phase_id(k);
            const double delta = get_delta(k);
            // diffusivity is scaled with volfrac^(delta + 1) --> scale it
            // with 1.0/volfrac^(delta + 1) here
            // and build derivative d diff/d volfrac = (delta + 1) * volfrac^delta
            (*prefacdiffodfluid)[curphase] += rhsfac * diffcoeff /
                                              std::pow(vol_frac(k), delta + 1.0) * (delta + 1.0) *
                                              std::pow(vol_frac(k), delta);
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
          {
            // porosity of solid epsilon_s does not depend on fluid primary variables
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
          {
            const int numfluidphases = phasemanager_->num_fluid_phases();
            const int numdofs = numfluidphases + phasemanager_->num_vol_frac();
            const int totalnummultiphasedofpernode = multiphase_mat()->num_mat();

            for (int phase = 0; phase < numfluidphases; ++phase)
            {
              for (int idof = 0; idof < numfluidphases; ++idof)
                (*prefacdiffodfluid)[idof] +=
                    rhsfac * phasemanager_->saturation_deriv(phase, idof) *
                    get_thermal_diffusivity(phase) * phasemanager_->porosity();

              if (phasemanager_->porosity_depends_on_fluid())
                for (int idof = 0; idof < totalnummultiphasedofpernode; ++idof)
                  (*prefacdiffodfluid)[idof] += rhsfac * phasemanager_->porosity_deriv(idof) *
                                                get_thermal_diffusivity(phase) *
                                                phasemanager_->saturation(phase);
            }

            for (int phase = numfluidphases; phase < numdofs; ++phase)
              (*prefacdiffodfluid)[phase] += rhsfac * get_thermal_diffusivity(phase);

            break;
          }
          default:
          {
            FOUR_C_THROW(
                "unknown species type {} for species {}!", scalartophasemap_[k].species_type, k);
            break;
          }
        }  // switch(species type)

        return;
      };

      //! get difftensor of fluid phases (permeability*relpermeability/mu)
      void get_diff_tensor_fluid(
          const int k, Core::LinAlg::Matrix<nsd, nsd>& difftensor, const int phase)
      {
        switch (scalartophasemap_[k].species_type)
        {
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
          {
            difftensor = difftensorsfluid_[phase];
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
          default:
            FOUR_C_THROW(
                "unknown species type {} for species {}!", scalartophasemap_[k].species_type, k);
            break;
        }  // switch(species type)

        return;
      }

      //! compute gradient of pressure in reference configuration
      void get_ref_grad_pres(const int k, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
          Core::LinAlg::Matrix<nsd, 1>& refgradpres, const int phase)
      {
        refgradpres.clear();

        const int numfluidphases = phasemanager_->num_fluid_phases();
        const int numvolfrac = phasemanager_->num_vol_frac();

        // fluid phase
        if (phase >= 0 && phase < numfluidphases)
        {
          const std::vector<Core::LinAlg::Matrix<nsd, 1>>& fluidgradphi =
              *(variablemanager_->grad_phinp());

          // gradient of phi w.r.t. reference coordinates
          std::vector<Core::LinAlg::Matrix<nsd, 1>> reffluidgradphi(
              numfluidphases, Core::LinAlg::Matrix<nsd, 1>(Core::LinAlg::Initialization::zero));
          for (int idof = 0; idof < numfluidphases; ++idof)
            reffluidgradphi[idof].multiply(xjm, fluidgradphi[idof]);

          // compute the pressure gradient from the phi gradients
          for (int idof = 0; idof < numfluidphases; ++idof)
            refgradpres.update(
                phasemanager_->pressure_deriv(phase, idof), reffluidgradphi[idof], 1.0);
        }
        // volfrac
        else if (phase < numfluidphases + numvolfrac)
        {
          refgradpres.multiply(xjm, pressuregrad_[phase]);
        }
        else
          FOUR_C_THROW("GetRefGradPres has been called with phase {}", phase);

        return;
      }

      // set flag if we actually evaluate the scalar or set it to zero
      void set_evaluate_scalar_flag(const int k)
      {
        const int numfluidphases = phasemanager_->num_fluid_phases();

        switch (scalartophasemap_[k].species_type)
        {
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          {
            // if smaller than threshold (at GP) we do not evaluate
            evaluate_scalar_[k] =
                (fabs(saturation_[scalartophasemap_[k].phaseID]) > min_val_of_phase_[k]);
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
          {
            // if smaller than minimum nodal volume fraction in element we do not evaluate
            evaluate_scalar_[k] = variablemanager_->element_has_valid_vol_frac_species(
                scalartophasemap_[k].phaseID - numfluidphases);
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
          {
            evaluate_scalar_[k] = true;
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
          {
            evaluate_scalar_[k] = true;
            break;
          }
          default:
          {
            FOUR_C_THROW(
                "unknown species type {} for species {}!", scalartophasemap_[k].species_type, k);
            break;
          }
        }  // switch(species type)

        return;
      }

      //! get pre-factor needed for OD-mesh-linearization of hist and source term
      double get_pre_factor_for_hist_and_source_od_mesh(const int k, const double fac,
          const double densnp, const double hist, const double rhsint)
      {
        // 1) linearization of mesh motion:
        //    call base class with fac*rhsint
        double prefac = fac * rhsint;

        switch (scalartophasemap_[k].species_type)
        {
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          {
            // 2) linearization of history:
            //    prefactor is densnp = porosity * rho * S
            //    --> porosity has to be linearized
            // with
            // linearization of porosity
            //------------------------------------------------dporosity/dd = dporosity/dJ * dJ/dd =
            // dporosity/dJ * J * N_x
            // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X )
            // = det (dx/ds) * ( det(dX/ds) )^-1 in our case: prefac is scaled with density, i.e.
            // porosity --> scale it with 1.0/porosity here

            if (phasemanager_->porosity_depends_on_struct())
              prefac += 1.0 * fac * hist * densnp / (phasemanager_->porosity()) *
                        phasemanager_->jacobian_def_grad() *
                        phasemanager_->porosity_deriv_wrt_jacobian_def_grad();
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
          {
            // do nothing: correct prefac = fac*rhsint
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
          {
            if (phasemanager_->porosity_depends_on_struct())
              prefac -= fac * hist * densnp /
                        (1 - phasemanager_->porosity() - phasemanager_->sum_add_vol_frac()) *
                        phasemanager_->jacobian_def_grad() *
                        phasemanager_->porosity_deriv_wrt_jacobian_def_grad();
            // else
            // do nothing: prefac = fac
            break;
          }
          case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
          {
            // evaluate the effective heat capacity factor
            const int numdofs = phasemanager_->num_fluid_phases() + phasemanager_->num_vol_frac();
            const int numfluidphases = phasemanager_->num_fluid_phases();

            double cp_eff = get_effective_heat_capacity();

            prefac += fac * hist * densnp / cp_eff * phasemanager_->solid_density() *
                      get_heat_capacity(numdofs) * (-1.0) * phasemanager_->jacobian_def_grad() *
                      phasemanager_->porosity_deriv_wrt_jacobian_def_grad();

            for (int phase = 0; phase < numfluidphases; ++phase)
              prefac += fac * hist * densnp / cp_eff * phasemanager_->density(phase) *
                        get_heat_capacity(phase) * phasemanager_->saturation(phase) *
                        phasemanager_->porosity_deriv_wrt_jacobian_def_grad() *
                        phasemanager_->jacobian_def_grad();

            break;
          }

          default:
            FOUR_C_THROW(
                "unknown species type {} for species {}!", scalartophasemap_[k].species_type, k);
            break;
        }  // switch(species type)

        return prefac;
      };

      //! get action
      ScaTra::Action get_action()
      {
        // return
        return myaction_;
      };

      //! return pressure gradient of current phase
      const Core::LinAlg::Matrix<nsd, 1>& pressure_gradient(const int curphase) const
      {
        return pressuregrad_[curphase];
      };

      //! return absolute value of pressure gradient of current phase
      double abs_pressure_gradient(const int curphase) const { return abspressuregrad_[curphase]; };

      //! return fluid material
      std::shared_ptr<Mat::FluidPoroMultiPhase> multiphase_mat()
      {
        if (!materialset_)
          FOUR_C_THROW("Fluid-Multiphase Material has not yet been set in Variablemanager");

        return multiphasemat_;
      }

      //! Setup phasemanager and variablemanager of fluid
      void setup_poro_fluid_managers(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          const int numfluidphases, const int totalnummultiphasedofpernode)
      {
        // dummy parameter list
        Discret::Elements::PoroFluidMultiPhaseEleParameter* para =
            Discret::Elements::PoroFluidMultiPhaseEleParameter::instance(discretization.name());

        // create phase-manager
        phasemanager_ =
            Discret::Elements::PoroFluidManager::PhaseManagerInterface::create_phase_manager(*para,
                nsd, multiphase_mat()->material_type(),
                POROFLUIDMULTIPHASE::Action::get_access_from_scatra, totalnummultiphasedofpernode,
                numfluidphases);

        // access from outside to the phasemanager: scatra-discretization has fluid-dis on dofset 2
        phasemanager_->setup(ele, ndsscatra_porofluid_);

        // create variablemanager
        variablemanager_ = Discret::Elements::PoroFluidManager::VariableManagerInterface<nsd,
            nen>::create_variable_manager(*para,
            POROFLUIDMULTIPHASE::Action::get_access_from_scatra, multiphase_mat(),
            totalnummultiphasedofpernode, numfluidphases);

        return;
      }

      // extract the element and node values of the poro-fluid --> extract them from its
      // variablemanager
      void extract_element_and_node_values_of_poro_fluid(Core::Elements::Element* ele,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::Matrix<nsd, nen>& xyze)
      {
        // access from outside to the variablemananger: scatra-discretization has fluid-dis on
        // dofset 2
        variablemanager_->extract_element_and_node_values(
            *ele, discretization, la, xyze, ndsscatra_porofluid_);

        return;
      }

      // get the phasemanager of the fluid
      std::shared_ptr<Discret::Elements::PoroFluidManager::PhaseManagerInterface>
      fluid_phase_manager()
      {
        return phasemanager_;
      }

      // get the variablemanager of the fluid
      std::shared_ptr<Discret::Elements::PoroFluidManager::VariableManagerInterface<nsd, nen>>
      fluid_var_manager()
      {
        return variablemanager_;
      }

     private:
      //! phase pressure
      std::vector<double> pressure_;
      //! phase saturation
      std::vector<double> saturation_;
      //! phase density
      std::vector<double> density_;
      //! solid pressure
      double solidpressure_;
      //! pressure gradient
      std::vector<Core::LinAlg::Matrix<nsd, 1>> pressuregrad_;
      //! pressure gradient
      std::vector<Core::LinAlg::Matrix<nsd, nsd>> difftensorsfluid_;
      //! norm of pressure-gradient
      std::vector<double> abspressuregrad_;
      //! volume fraction
      std::vector<double> volfrac_;
      //! volume fraction pressure
      std::vector<double> volfracpressure_;

      //! fluid-poro multiphase material
      std::shared_ptr<Mat::FluidPoroMultiPhase> multiphasemat_;

      //! delta for effective diffusivity
      std::vector<double> delta_;

      //! function IDs of relative mobility functions
      std::vector<int> relative_mobility_funct_id_;

      //! heat capacity
      //! order [ <fluid>  <volfrac>  <solid> ]
      std::vector<double> heatcapacity_;
      //! thermal diffusivity
      //! order [ <fluid>  <volfrac>  <solid> ]
      std::vector<double> thermaldiffusivity_;
      //! effective heat capacity
      double heatcapacityeff_;

      //! minimum saturation under which the corresponding mass fraction is assumed to be zero
      std::vector<double> min_val_of_phase_;
      //! flag to check if we have to evaluate the species equation in this element
      std::vector<bool> evaluate_scalar_;
      //! scalar to phase map
      std::vector<Mat::ScaTraMatMultiPoro::ScalarToPhaseMap> scalartophasemap_;
      //! check if multiphase material has been set
      bool materialset_;

      ScaTra::Action myaction_;

      //! phase manager of the fluid
      std::shared_ptr<Discret::Elements::PoroFluidManager::PhaseManagerInterface> phasemanager_;

      //! variable manager of the fluid
      std::shared_ptr<Discret::Elements::PoroFluidManager::VariableManagerInterface<nsd, nen>>
          variablemanager_;

      //! dofset of fluid field on scatra dis
      // TODO: find a better way to do this
      const int ndsscatra_porofluid_ = 2;
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
