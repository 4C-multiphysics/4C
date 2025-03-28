// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_ARTERY_HPP
#define FOUR_C_SCATRA_ELE_CALC_ARTERY_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_scatra_ele_calc.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Discret
{
  namespace Elements
  {
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerArtery;

    template <Core::FE::CellType distype, int probdim>
    class ScaTraEleCalcArtery : public ScaTraEleCalc<distype, probdim>
    {
     private:
     protected:
      //! (private) protected constructor, since we are a Singleton.
      ScaTraEleCalcArtery(const int numdofpernode, const int numscal, const std::string& disname);

     private:
     public:
      typedef ScaTraEleCalc<distype, probdim> my;
      using my::nen_;
      using my::nsd_;
      using my::nsd_ele_;


      //! Singleton access method
      static ScaTraEleCalcArtery<distype, probdim>* instance(
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

      //! evaluate shape functions and their derivatives at current integration point
      double eval_shape_func_and_derivs_at_int_point(
          const Core::FE::IntPointsAndWeights<nsd_ele_>& intpoints,  //!< integration points
          const int iquad                                            //!< id of current Gauss point
          ) override
      {
        const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

        // scale fac with the area of the artery pi*D^2/4
        return fac * M_PI * var_manager()->diam() * var_manager()->diam() / 4.0;
      }

      //! evaluate shape functions and their derivatives at element center
      double eval_shape_func_and_derivs_at_ele_center() override
      {
        // use one-point Gauss rule to do calculations at the element center
        const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints_tau(
            ScaTra::DisTypeToStabGaussRule<distype>::rule);

        // volume of the element (2D: element surface area; 1D: element length)
        // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
        // base class has to be called since we do not want the scaling with artery area here
        const double vol = my::eval_shape_func_and_derivs_at_int_point(intpoints_tau, 0);

        return vol;
      }

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

      //! calculation of convective element matrix in convective form (off diagonal term fluid)
      void calc_mat_conv_od_fluid(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodefluid,             //!< number of dofs per node of fluid element
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double densnp,      //!< density at time_(n+1)
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi  //!< scalar gradient
          ) override;

      //! calculation of convective element matrix: add conservative contributions (off diagonal
      //! term fluid)
      void calc_mat_conv_add_cons_od_fluid(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const int ndofpernodefluid,             //!< number of dofs per node of fluid element
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double densnp,      //!< density at time_(n+1)
          const double phinp        //!< scalar at time_(n+1)
          ) override
      {
        FOUR_C_THROW("not possible");
        return;
      }

      //! get internal variable manager for multiporo formulation
      std::shared_ptr<ScaTraEleInternalVariableManagerArtery<nsd_, nen_>> var_manager()
      {
        return std::static_pointer_cast<ScaTraEleInternalVariableManagerArtery<nsd_, nen_>>(
            my::scatravarmanager_);
      };

      //! set internal variables
      void set_internal_variables_for_mat_and_rhs() override;

      //! artery pressure values at t_(n+1)
      Core::LinAlg::Matrix<nen_, 1> earterypressurenp_;
    };

    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerArtery : public ScaTraEleInternalVariableManager<nsd, nen>
    {
      typedef ScaTraEleInternalVariableManager<nsd, nen> my;

     public:
      ScaTraEleInternalVariableManagerArtery(int numscal)
          : ScaTraEleInternalVariableManager<nsd, nen>(numscal), materialset_(false)
      {
        return;
      }

      // compute and set internal variables -- no L2-projection but evaluation at GP
      void set_internal_variables_artery(
          const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
          const Core::LinAlg::Matrix<nsd, nen>&
              derxy,  //! global derivatives of shape functions w.r.t x,y,z
          const Core::LinAlg::Matrix<nsd, nen>&
              deriv,  //! global derivatives of shape functions w.r.t r,s,t
          const Core::LinAlg::Matrix<nsd, nsd>& xjm,
          const std::vector<Core::LinAlg::Matrix<nen, 1>>&
              ephinp,  //! scalar at t_(n+1) or t_(n+alpha_F)
          const std::vector<Core::LinAlg::Matrix<nen, 1>>& ephin,  //! scalar at t_(n)
          const std::vector<Core::LinAlg::Matrix<nen, 1>>&
              ehist,  //! history vector of transported scalars
          const Core::LinAlg::Matrix<nen, 1>& earterypressure)
      {
        // call base class (scatra) with dummy variable econvelnp
        const Core::LinAlg::Matrix<nsd, nen> econvelnp(Core::LinAlg::Initialization::zero);
        const Core::LinAlg::Matrix<nsd, nen> eforcevelocity(Core::LinAlg::Initialization::zero);
        my::set_internal_variables(funct, derxy, ephinp, ephin, econvelnp, ehist, eforcevelocity);

        static Core::LinAlg::Matrix<nsd, 1> pressuregrad(Core::LinAlg::Initialization::zero);
        pressuregrad.multiply(derxy, earterypressure);

        for (int k = 0; k < my::numscal_; ++k)
        {
          // convective velocity
          my::convelint_[k].update(-diam() * diam() / 32.0 / visc(), pressuregrad, 0.0);
          // convective part in convective form: rho*u_x*N,x+ rho*u_y*N,y
          my::conv_[k].multiply_tn(derxy, my::convelint_[k]);
          // overwrite convective term
          // - k/\mu*grad p * grad phi
          my::conv_phi_[k] = my::convelint_[k].dot(my::gradphi_[k]);
        }
      };

      // Set the artery material in the scatra-Varmanager
      void set_artery_material(Core::Elements::Element* ele)
      {
        // check if we actually have two materials
        if (ele->num_material() < 2) FOUR_C_THROW("no second material available");
        // check for artery material
        if (ele->material(1)->material_type() != Core::Materials::MaterialType::m_cnst_art)
          FOUR_C_THROW("Secondary material is not of type m_cnst_art, but {}",
              ele->material(1)->material_type());

        // here we rely that the Artery material has been added as second material
        arterymat_ = std::dynamic_pointer_cast<Mat::Cnst1dArt>(ele->material(1));

        materialset_ = true;
      }

      // return density
      double dens() { return artery_mat()->Density(); }

      // return diameter
      double diam() { return artery_mat()->diam(); }

      // return viscosity
      double visc() { return artery_mat()->viscosity(); }

      //! return artery material
      std::shared_ptr<Mat::Cnst1dArt> artery_mat()
      {
        if (!materialset_) FOUR_C_THROW("Artery Material has not yet been set in Variablemanager");

        return arterymat_;
      }

     private:
      //! artery material
      std::shared_ptr<Mat::Cnst1dArt> arterymat_;

      //! check if artery material has been set
      bool materialset_;
    };
  }  // namespace Elements
}  // namespace Discret



FOUR_C_NAMESPACE_CLOSE

#endif
