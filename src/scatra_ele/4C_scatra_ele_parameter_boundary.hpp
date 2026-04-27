// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_PARAMETER_BOUNDARY_HPP
#define FOUR_C_SCATRA_ELE_PARAMETER_BOUNDARY_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_scatra_ele_parameter_base.hpp"
#include "4C_scatra_s2i_input.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // class implementation
    class ScaTraEleParameterBoundary : public ScaTraEleParameterBase
    {
     public:
      //! singleton access method
      static ScaTraEleParameterBoundary* instance(const std::string& disname);

      //! set parameters for specific kintetic model
      void set_parameters(Teuchos::ParameterList& parameters) override;

      /// Set actual parameters
      ///@{
      void set_alpha(Teuchos::ParameterList& parameters);
      void set_capacitance(Teuchos::ParameterList& parameters);
      void set_charge_transfer_constant(Teuchos::ParameterList& parameters);
      void set_conv_tol_iter_num(Teuchos::ParameterList& parameters);
      void set_density_molar_mass(Teuchos::ParameterList& parameters);
      void set_energy_substance_ratio(Teuchos::ParameterList& parameters);
      void set_is_pseudo_contact(Teuchos::ParameterList& parameters);
      void set_num_electrons(Teuchos::ParameterList& parameters);
      void set_num_scal(Teuchos::ParameterList& parameters);
      void set_on_off(Teuchos::ParameterList& parameters);
      void set_peltier(Teuchos::ParameterList& parameters);
      void set_permeabilities(Teuchos::ParameterList& parameters);
      void set_regularization(Teuchos::ParameterList& parameters);
      void set_resistance(Teuchos::ParameterList& parameters);
      void set_resistivity(Teuchos::ParameterList& parameters);
      void set_stoichiometries(Teuchos::ParameterList& parameters);
      void set_thermo_perm(Teuchos::ParameterList& parameters);
      ///@}

      /// Return parameters
      ///@{
      double alphadata() const { return alphaa_; }
      double alpha_c() const { return alphac_; }
      double capacitance() const { return capacitance_; }
      double charge_transfer_constant() const { return kr_; }
      Core::Conditions::ConditionType condition_type() const { return conditiontype_; }
      double convtolimplicit_bv() const { return convtolimplicit_bv_; }
      double density() const { return density_; }
      bool is_pseudo_contact() const { return is_pseudo_contact_; }
      int itemaximplicit_bv() const { return itemaxmimplicit_bv_; }
      int kinetic_model() const { return kineticmodel_; }
      double molar_heat_capacity() const { return molar_heat_capacity_; }
      double molar_mass() const { return molarmass_; }
      int num_electrons() const { return numelectrons_; }
      int num_scal() const { return numscal_; }
      const std::vector<int>* on_off() const { return onoff_; }
      double peltier() const { return peltier_; }
      const std::vector<double>* permeabilities() const { return permeabilities_; }
      double regularization_parameter() const { return regularizationparameter_; }
      S2I::RegularizationType regularization_type() const { return regularizationtype_; }
      double resistance() const { return resistance_; }
      double resistivity() const { return resistivity_; }
      const std::vector<int>* stoichiometries() const { return stoichiometries_; }
      double thermo_perm() const { return thermoperm_; }

      ///@}

     private:
      //! private constructor for singletons
      ScaTraEleParameterBoundary(const std::string& disname);

      /// symmetry coefficient of anodic intercalation reaction
      double alphaa_;

      /// symmetry coefficient of cathodic intercalation reaction
      double alphac_;

      /// condition type of stored condition
      Core::Conditions::ConditionType conditiontype_;

      /// convergence tolerance of local Newton-Raphson iteration for modified Butler-Volmer
      /// equation
      double convtolimplicit_bv_;

      /// density of the interface layer material associated to S2IKineticsGrowth condition
      double density_;

      /// ratio between flux of species (amount) and flux of energy
      double molar_heat_capacity_;

      /// indicating if pseudo contact is considered at the s2i interface, i.e. no flux if interface
      /// is under tensile stresses
      bool is_pseudo_contact_;

      /// maximum number of iterations of local Newton-Raphson iteration for modified Butler-Volmer
      /// equation
      int itemaxmimplicit_bv_;

      /// kinetic model of s2i condition
      int kineticmodel_;

      /// charge transfer constant
      double kr_;

      /// molar mass of the interface layer material associated to S2IKineticsGrowth condition
      double molarmass_;

      /// number of electrons involved in charge transfer
      int numelectrons_;

      /// total number of transported scalars
      int numscal_;

      /// which dofs are constrained by this condition
      const std::vector<int>* onoff_;

      /// peltier coefficient
      double peltier_;

      /// vector of constant permeabilities
      const std::vector<double>* permeabilities_;

      /// regularization factor for S2IKineticsGrowth condition
      double regularizationparameter_;

      /// type of regularization for S2IKineticsGrowth condition
      S2I::RegularizationType regularizationtype_;

      /// interface resistance associated with S2ICoupling condition
      double resistance_;

      /// resistivity of the interface layer material associated to S2IKineticsGrowth condition
      double resistivity_;

      /// surface capacitance at interface associated with S2ICoupling condition
      double capacitance_;

      /// vector of stoichiometric coefficients for scatra-scatra interface
      const std::vector<int>* stoichiometries_;

      /// constant permeability for heat at interface
      double thermoperm_;
    };
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
