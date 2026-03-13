// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELE_PARAMETER_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELE_PARAMETER_HPP


#include "4C_config.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  class FunctionManager;
}  // namespace Core::Utils

namespace Discret
{
  namespace Elements
  {
    /// Evaluation of general parameters (constant over time)
    class PoroFluidMultiPhaseEleParameter
    {
     public:
      //! singleton access method
      static PoroFluidMultiPhaseEleParameter* instance(
          const std::string& disname  //!< name of discretization
      );

      //! set parameters
      void set_time_step_parameters(Teuchos::ParameterList& parameters  //!< parameter list
      );

      //! set parameters
      void set_general_parameters(Teuchos::ParameterList& parameters  //!< parameter list
      );

      //! @name access methods
      double time() const { return time_; };
      bool is_gen_alpha() const { return is_genalpha_; };
      bool is_stationary() const { return is_stationary_; };
      double dt() const { return dt_; };
      double time_fac() const { return timefac_; };
      double time_fac_rhs() const { return timefacrhs_; };
      double time_fac_rhs_tau() const { return timefacrhstau_; };
      double alpha_f() const { return alpha_f_; };
      bool is_ale() const { return is_ale_; };
      bool biot_stab() const { return stab_biot_; };
      int nds_disp() const { return nds_disp_; };
      int nds_vel() const { return nds_vel_; };
      int nds_solid_pressure() const { return nds_solidpressure_; };
      int nds_scalar() const { return nds_scalar_; };
      bool has_scalar() const { return nds_scalar_ > -1; };
      bool has_bodyforce_contribution() const { return has_bodyforce_contribution_; };
      const Core::Utils::FunctionManager* function_manager() const { return function_manager_; };
      const std::vector<double>& bodyforce_contribution_values() const
      {
        return bodyforce_contribution_values_;
      };
      int num_domain_int_functions() const { return domainint_funct_.size(); };
      std::vector<int> domain_int_functions() const { return domainint_funct_; };
      //@}

     private:
      //! private constructor for singletons
      PoroFluidMultiPhaseEleParameter(const std::string& disname  //!< name of discretization
      );

      //! @name parameters potentially changing every time step

      //! current total time
      double time_;
      //! current time step
      double dt_;
      //! time integration factor for left hand side
      double timefac_;
      //! time integration factor for right hand side
      double timefacrhs_;
      //! (time integration factor for right hand side (* (stabilization parameter)
      double timefacrhstau_;
      //! alpha_f parameter from generalized alpha time integration
      double alpha_f_;

      //@}

      //! @name (almost) constant parameters over simulation time

      //! generalized-alpha flag
      bool is_genalpha_;
      //! instationary flag
      bool is_stationary_;
      //! ALE flag
      bool is_ale_;
      //! flag for biot stabilization
      bool stab_biot_;
      //! number of dof set related to mesh displacements
      int nds_disp_;
      //! number of dof set related to mesh velocities
      int nds_vel_;
      //! number of dof set related to solid pressure
      int nds_solidpressure_;
      //! number of dof set related to scalar field
      int nds_scalar_;
      //! setup flag
      bool isset_generalparams_;
      //! domain integral functions
      std::vector<int> domainint_funct_;

      //! function manager for configured runtime functions
      const Core::Utils::FunctionManager* function_manager_;

      //! flag for bodyforce contribution
      bool has_bodyforce_contribution_;

      //! bodyforce contribution values
      std::vector<double> bodyforce_contribution_values_;
      //@}

    };  // class PoroFluidMultiPhaseEleParameter
  }  // namespace Elements
}  // namespace Discret



FOUR_C_NAMESPACE_CLOSE

#endif
