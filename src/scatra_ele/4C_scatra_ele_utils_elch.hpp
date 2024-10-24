// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_UTILS_ELCH_HPP
#define FOUR_C_SCATRA_ELE_UTILS_ELCH_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_inpar_elch.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declaration
    class ScaTraEleDiffManagerElch;

    // class implementation
    template <Core::FE::CellType distype>
    class ScaTraEleUtilsElch
    {
     protected:
      //! number of element nodes
      static constexpr int nen_ = Core::FE::num_nodes<distype>;

      //! number of space dimensions
      static constexpr int nsd_ = Core::FE::dim<distype>;

     public:
      //! singleton access method
      static ScaTraEleUtilsElch<distype>* instance(
          const int numdofpernode,    ///< number of degrees of freedom per node
          const int numscal,          ///< number of transported scalars per node
          const std::string& disname  ///< name of discretization
      );

      //! destructor
      virtual ~ScaTraEleUtilsElch() = default;

      //! evaluation of electrochemistry kinetics at integration point on domain or boundary element
      void evaluate_elch_kinetics_at_integration_point(
          const Core::Elements::Element* ele,     ///< current element
          Core::LinAlg::SerialDenseMatrix& emat,  ///< element matrix
          Core::LinAlg::SerialDenseVector& erhs,  ///< element right-hand side vector
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
              ephinp,  ///< state variables at element nodes
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
              ehist,                                   ///< history variables at element nodes
          const double timefac,                        ///< time factor
          const double fac,                            ///< Gauss integration factor
          const Core::LinAlg::Matrix<nen_, 1>& funct,  ///< shape functions at int. point
          Core::Conditions::Condition& cond,           ///< condition
          const int nume,                              ///< number of transferred electrons
          const std::vector<int>& stoich,              ///< stoichiometry of the reaction
          const double valence_k,                      ///< valence of the single reactant
          const int kinetics,                          ///< desired electrode kinetics model
          const double pot0,                           ///< actual electrode potential on metal side
          const double frt,                            ///< factor F/RT
          const double fns,     ///< factor fns = s_k / (nume * faraday * (-1))
          const double scalar,  ///< scaling factor for element matrix and right-hand side vector
                                ///< contributions
          const int k           ///< index of evaluated scalar
      ) const;

      //! evaluate electrode kinetics status information at integration point on domain or boundary
      //! element
      void evaluate_electrode_status_at_integration_point(
          const Core::Elements::Element* ele,        ///< current element
          Core::LinAlg::SerialDenseVector& scalars,  ///< scalars to be computed
          const Teuchos::ParameterList& params,      ///< parameter list
          Core::Conditions::Condition& cond,         ///< condition
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
              ephinp,  ///< nodal values of concentration and electric potential
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
              ephidtnp,                                ///< nodal time derivative vector
          const Core::LinAlg::Matrix<nen_, 1>& funct,  ///< shape functions at integration point
          const int zerocur,                           ///< flag for zero current
          const int kinetics,                          ///< desired electrode kinetics model
          const std::vector<int>& stoich,              ///< stoichiometry of the reaction
          const int nume,                              ///< number of transferred electrons
          const double pot0,     ///< actual electrode potential on metal side at t_{n+1}
          const double frt,      ///< factor F/RT
          const double timefac,  ///< factor due to time discretization
          const double fac,      ///< integration factor
          const double scalar,   ///< scaling factor for current related quantities
          const int k            ///< index of evaluated scalar
      ) const;

      //! evaluate ion material
      void mat_ion(const Teuchos::RCP<const Core::Mat::Material> material,  //!< ion material
          const int k,                                                      //!< ID of ion material
          const Inpar::ElCh::EquPot equpot,  //!< type of closing equation for electric potential
          const Teuchos::RCP<ScaTraEleDiffManagerElch>& diffmanager  //!< diffusion manager
      );

     protected:
      //! protected constructor for singletons
      ScaTraEleUtilsElch(const int numdofpernode,  ///< number of degrees of freedom per node
          const int numscal,                       ///< number of transported scalars per node
          const std::string& disname               ///< name of discretization
      );

     private:
      //! number of degrees of freedom per node
      const int numdofpernode_;

      //! number of transported scalars
      const int numscal_;
    };  // class ScaTraEleUtilsElch
  }     // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
