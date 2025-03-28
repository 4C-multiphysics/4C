// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_STI_THERMO_HPP
#define FOUR_C_SCATRA_ELE_STI_THERMO_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_calc.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declaration
    class ScaTraEleDiffManagerSTIThermo;

    // class implementation
    template <Core::FE::CellType distype>
    class ScaTraEleSTIThermo
    {
     public:
      virtual ~ScaTraEleSTIThermo() = default;

     protected:
      using my = ScaTraEleCalc<distype>;

      //! number of element nodes
      static constexpr int nen_ = my::nen_;

      //! number of space dimensions
      static constexpr int nsd_ = my::nsd_;

      //! protected constructor for singletons
      ScaTraEleSTIThermo(const int& numscal  //!< number of transported scalars
      );

      //! extract quantities for element evaluation
      virtual void extract_element_and_node_values(
          Core::Elements::Element* ele,              //!< current element
          Teuchos::ParameterList& params,            //!< parameter list
          Core::FE::Discretization& discretization,  //!< discretization
          Core::Elements::LocationArray& la          //!< location array
      );

      //! provide element matrix with linearizations of Soret effect term in discrete scatra
      //! residuals w.r.t. scatra dofs
      void calc_mat_soret(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
          const double& timefacfac,  //!< domain integration factor times time integration factor
          const double& conc,        //!< concentration
          const double& diffcoeff,   //!< diffusion coefficient
          const double&
              diffcoeffderiv,  //!< derivative of diffusion coefficient w.r.t. concentration
          const double& temp,  //!< temperature
          const Core::LinAlg::Matrix<nsd_, 1>& gradtemp,  //!< gradient of temperature
          const Core::LinAlg::Matrix<nen_, 1>& funct,     //!< shape functions
          const Core::LinAlg::Matrix<nsd_, nen_>& derxy  //!< spatial derivatives of shape functions
      );

      //! provide element matrix with linearizations of Soret effect term in discrete scatra
      //! residuals w.r.t. thermo dofs
      void calc_mat_soret_od(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
          const double& timefacfac,     //!< time integration factor times domain integration factor
          const double& concentration,  //!< concentration
          const double& diffcoeff,      //!< diffusion coefficient
          const double& temperature,    //!< temperature
          const Core::LinAlg::Matrix<nsd_, 1>& gradtemp,  //!< gradient of temperature
          const Core::LinAlg::Matrix<nen_, 1>& funct,     //!< shape functions
          const Core::LinAlg::Matrix<nsd_, nen_>& derxy  //!< spatial derivatives of shape functions
      );

      //! provide element right-hand side vector with contributions of Soret effect term to discrete
      //! scatra residuals
      void calc_rhs_soret(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
          const double& rhsfac,     //!< domain integration factor times time integration factor for
                                    //!< right-hand side vector
          const double& conc,       //!< concentration
          const double& diffcoeff,  //!< diffusion coefficient
          const double& temp,       //!< temperature
          const Core::LinAlg::Matrix<nsd_, 1>& gradtemp,  //!< gradient of temperature
          const Core::LinAlg::Matrix<nsd_, nen_>& derxy  //!< spatial derivatives of shape functions
      );

      //! provide element matrix with linearization of diffusion coefficient with respect to
      //! discrete temperature
      //!
      //! \param emat           element matrix
      //! \param numdofpernode  number of dof per node
      //! \param timefacfac     time integration factor times domain integration factor
      //! \param invF  1/F      with F being the faraday constant
      //! \param gradconc       spatial derivative of concentration
      //! \param gradpot        spatial derivative of electric potential
      //! \param tempderivisodiffcoef  derivative of isotropic diffusion coefficient wrt temperature
      //! \param tempderivcond         derivative of electric conductivity wrt temperature
      //! \param funct     shape functions
      //! \param derxy     spatial derivatives of shape functions
      //! \param scalefac  scaling factor for pot. contributions
      void calc_mat_diff_thermo_od(Core::LinAlg::SerialDenseMatrix& emat, const int& numdofpernode,
          const double& timefacfac, const double& invF,
          const Core::LinAlg::Matrix<nsd_, 1>& gradconc,
          const Core::LinAlg::Matrix<nsd_, 1>& gradpot, const double& tempderivisodiffcoef,
          const double& tempderivcond, const Core::LinAlg::Matrix<nen_, 1>& funct,
          const Core::LinAlg::Matrix<nsd_, nen_>& derxy, const double& scalefac);

      //! evaluate Soret material
      void mat_soret(const std::shared_ptr<const Core::Mat::Material> material  //!< Soret material
      );

      //! compute gradient of test function times gradient of shape function
      void get_laplacian_weak_form(double& result,       //!< result variable
          const int& vi,                                 //!< index of test function
          const int& ui,                                 //!< index of shape function
          const Core::LinAlg::Matrix<nsd_, nen_>& derxy  //!< spatial derivatives of shape functions
      )
      {
        // initialize result variable
        result = 0.;

        // compute gradient of test function times gradient of shape function
        for (int idim = 0; idim < nsd_; ++idim) result += derxy(idim, vi) * derxy(idim, ui);
      };

      //! compute gradient of test function times given gradient
      void get_laplacian_weak_form_rhs(double& result,    //!< result variable
          const int& vi,                                  //!< index of test function
          const Core::LinAlg::Matrix<nsd_, 1>& gradient,  //!< given gradient
          const Core::LinAlg::Matrix<nsd_, nen_>& derxy  //!< spatial derivatives of shape functions
      )
      {
        // initialize result variable
        result = 0.;

        // compute gradient of test function times given gradient
        for (int idim = 0; idim < nsd_; ++idim) result += derxy(idim, vi) * gradient(idim);
      };

      //! local nodal values of temperature
      Core::LinAlg::Matrix<nen_, 1> etempnp_;

      //! thermo diffusion manager
      const std::shared_ptr<ScaTraEleDiffManagerSTIThermo> diffmanagerstithermo_;
    };  // class ScaTraEleSTIThermo


    //! implementation of ScaTraEleDiffManagerSTIThermo
    class ScaTraEleDiffManagerSTIThermo : public ScaTraEleDiffManager
    {
     public:
      //! constructor
      ScaTraEleDiffManagerSTIThermo(const int& numscal  //!< number of transported scalars
          )
          :  // constructor of base class
            ScaTraEleDiffManager(numscal),

            // initialize internal member variable
            soret_(0.) {};

      //! set Soret coefficient
      void set_soret(const double& soret  //!< Soret coefficient
      )
      {
        soret_ = soret;
      };

      //! get Soret coefficient
      const double& get_soret() const { return soret_; };

     protected:
      double soret_;
    };  // class ScaTraEleDiffManagerSTIThermo


    // implementation of ScaTraEleInternalVariableManagerSTIThermo
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerSTIThermo
    {
     public:
      //! constructor
      ScaTraEleInternalVariableManagerSTIThermo()
          : temp_(0.), gradtemp_(Core::LinAlg::Initialization::zero) {};

      //! set internal variables for element evaluation
      void set_internal_variables_sti_thermo(
          const Core::LinAlg::Matrix<nen, 1>& funct,    //!< shape functions
          const Core::LinAlg::Matrix<nsd, nen>& derxy,  //!< spatial derivatives of shape functions
          const Core::LinAlg::Matrix<nen, 1>&
              etempnp  //!< nodal temperature values at time t_(n+1) or t_(n+alpha_F)
      )
      {
        // set local values of thermo variables at time t_(n+1) or t_(n+alpha_F)
        temp_ = funct.dot(etempnp);  // temperature
        gradtemp_.multiply(derxy, etempnp);
      };

      //! return temperature
      const double& temp() const { return temp_; };

      //! return gradient of temperature
      const Core::LinAlg::Matrix<nsd, 1>& grad_temp() const { return gradtemp_; };

     protected:
      //! temperature
      double temp_;

      //! gradient of temperature
      Core::LinAlg::Matrix<nsd, 1> gradtemp_;
    };
  }  // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
