// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_ADVANCED_REACTION_HPP
#define FOUR_C_SCATRA_ELE_CALC_ADVANCED_REACTION_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_calc.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Mat
{
  class MatListReactions;
}

namespace Discret
{
  namespace Elements
  {
    // forward declaration
    class ScaTraEleReaManagerAdvReac;

    /*
     This class calls all 'advanced reaction terms' calculations and applies them correctly.
     Thereby, no assumption on the shape of the (potentially nonlinear) reaction term f(c_1,...,c_n)
     have to be made. The actual calculations of the reaction term f(c_1,...,c_n) as well as all its
     linearizations \partial_c f(c) are done within the material MAT_matlist_reactions and
     MAT_scatra_reaction
     */
    template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype>>
    class ScaTraEleCalcAdvReac : public virtual ScaTraEleCalc<distype, probdim>
    {
     protected:
      /// (private) protected constructor, since we are a Singleton.
      ScaTraEleCalcAdvReac(const int numdofpernode, const int numscal, const std::string& disname);

     private:
      using my = ScaTraEleCalc<distype, probdim>;

     protected:
      using my::nen_;
      using my::nsd_;

     public:
      /// Singleton access method
      static ScaTraEleCalcAdvReac<distype, probdim>* instance(const int numdofpernode,
          const int numscal, const std::string& disname  //!< creation/destruction indication
      );

     protected:
      //! set internal variables
      void set_internal_variables_for_mat_and_rhs() override;

      //! get the material parameters
      void get_material_params(
          const Core::Elements::Element* ele,  //!< the element we are dealing with
          std::vector<double>& densn,          //!< density at t_(n)
          std::vector<double>& densnp,         //!< density at t_(n+1) or t_(n+alpha_F)
          std::vector<double>& densam,         //!< density at t_(n+alpha_M)
          double& visc,                        //!< fluid viscosity
          const int iquad = -1                 //!< id of current gauss point (default = -1)
          ) override;


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


      //! Set advanced reaction terms and derivatives
      virtual void set_advanced_reaction_terms(const int k,          //!< index of current scalar
          const std::shared_ptr<Mat::MatListReactions> matreaclist,  //!< index of current scalar
          const double* gpcoord,  //!< current Gauss-point coordinates
          const double scale,     //!< scale factor for reaction body force (default is 1.0)
          int element_id          //!< current element id
      );

      //! evaluate shape functions and their derivatives at element center
      double eval_shape_func_and_derivs_at_ele_center() override;

      //! array for shape function at element center
      Core::LinAlg::Matrix<nen_, 1> funct_elementcenter_;

      //! get current Gauss-point coordinates
      virtual const double* get_gp_coord() const { return gpcoord_; }

      //! get reaction manager for advanced reaction
      std::shared_ptr<ScaTraEleReaManagerAdvReac> rea_manager()
      {
        return std::static_pointer_cast<ScaTraEleReaManagerAdvReac>(my::reamanager_);
      };

     private:
      //! number of spatial dimensions for Gauss point coordinates (always three)
      static constexpr unsigned int numdim_gp_ = 3;
      //! current Gauss-point coordinates
      double gpcoord_[numdim_gp_];

    };  // end ScaTraEleCalcAdvReac


    /// Scatra reaction manager for Advanced_Reaction
    /*!
      This class keeps all advanced reaction terms specific sutffs needed for the evaluation of an
      element. The ScaTraEleReaManagerAdvReac is derived from the standard ScaTraEleReaManager.
    */
    class ScaTraEleReaManagerAdvReac : public ScaTraEleReaManager
    {
     public:
      ScaTraEleReaManagerAdvReac(int numscal)
          : ScaTraEleReaManager(numscal),
            reabodyforce_(numscal, 0.0),  // size of vector + initialized to zero
            reabodyforcederiv_(numscal, std::vector<double>(numscal, 0.0)),
            numaddvariables_(0),
            reabodyforcederivaddvariables_(numscal, std::vector<double>(numaddvariables_, 0.0))
      {
      }

      //! @name set routines

      //! Clear everything and resize to length numscal
      void clear(int numscal) override
      {
        // clear base class
        ScaTraEleReaManager::clear(numscal);
        // clear
        reabodyforce_.resize(0);
        reabodyforcederiv_.resize(0);
        reabodyforcederivaddvariables_.resize(0);
        // resize
        reabodyforce_.resize(numscal, 0.0);
        reabodyforcederiv_.resize(numscal, std::vector<double>(numscal, 0.0));
        reabodyforcederivaddvariables_.resize(numscal, std::vector<double>(numaddvariables_, 0.0));
      }

      //! Add to the body force due to reaction
      void add_to_rea_body_force(const double reabodyforce, const int k)
      {
        reabodyforce_[k] += reabodyforce;
        if (reabodyforce != 0.0) include_me_ = true;
      }

      //! Return one line of the jacobian of the reaction vector
      std::vector<double>& get_rea_body_force_deriv_vector(const int k)
      {
        return reabodyforcederiv_[k];
      }

      //! Add to the derivative of the body force due to reaction
      void add_to_rea_body_force_deriv_matrix(
          const double reabodyforcederiv, const int k, const int j)
      {
        (reabodyforcederiv_[k])[j] += reabodyforcederiv;
        return;
      }

      //@}

      //! @name access routines

      //! Return the reaction coefficient
      double get_rea_body_force(const int k) const { return reabodyforce_[k]; }

      //! Return the reaction coefficient
      double get_rea_body_force_deriv_matrix(const int k, const int j) const
      {
        return (reabodyforcederiv_[k])[j];
      }

      //! Return the stabilization coefficient
      double get_stabilization_coeff(const int k, const double phinp_k) const override
      {
        double stabboeff = ScaTraEleReaManager::get_stabilization_coeff(k, phinp_k);

        if (phinp_k > 1.0e-10) stabboeff += fabs(reabodyforce_[k] / phinp_k);

        return stabboeff;
      }

      // initialize
      void initialize_rea_body_force_deriv_vector_add_variables(
          const int numscal, const int newsize)
      {
        reabodyforcederivaddvariables_.resize(numscal, std::vector<double>(newsize, 0.0));
        numaddvariables_ = newsize;
      }

      //! Return one line of the jacobian of the reaction vector -- derivatives after additional
      //! variables
      std::vector<double>& get_rea_body_force_deriv_vector_add_variables(const int k)
      {
        return reabodyforcederivaddvariables_[k];
      }

      //@}

     private:
      //! @name protected variables

      //! scalar reaction coefficients
      std::vector<double> reabodyforce_;

      //! jacobian of the scalar reaction coefficients w.r.t. scalar variables
      std::vector<std::vector<double>> reabodyforcederiv_;

      //! number of additional variables
      int numaddvariables_;

      //! derivatives after additional variables (for OD-terms)
      std::vector<std::vector<double>> reabodyforcederivaddvariables_;

      //@}
    };


  }  // namespace Elements

}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
