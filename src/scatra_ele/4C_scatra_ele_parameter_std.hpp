// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_PARAMETER_STD_HPP
#define FOUR_C_SCATRA_ELE_PARAMETER_STD_HPP

#include "4C_config.hpp"

#include "4C_inpar_cardiac_monodomain.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_scatra_ele_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declaration
    class ScaTraEleParameterTimInt;

    // class implementation
    class ScaTraEleParameterStd : public ScaTraEleParameterBase
    {
     public:
      //! singleton access method
      static ScaTraEleParameterStd* instance(const std::string& disname);

      //! set parameters
      void set_parameters(Teuchos::ParameterList& parameters) override;

      //! set the nodeset parameters
      void set_nodeset_parameters(Teuchos::ParameterList& parameters);

      //! @name return general parameters
      //! @{
      bool is_ale() const { return is_ale_; };
      bool is_conservative() const { return is_conservative_; };
      bool spherical_coords() const { return sphericalcoords_; };
      Inpar::ScaTra::FluxType calc_flux_domain() const { return calcflux_domain_; };
      std::shared_ptr<std::vector<int>> write_flux_ids() const { return writefluxids_; };
      Inpar::ScaTra::FdCheck fd_check() const { return fdcheck_; };
      double fd_check_eps() const { return fdcheckeps_; };
      double fd_check_tol() const { return fdchecktol_; };
      int prob_num() const { return probnum_; };
      bool semi_implicit() const { return semiimplicit_; };
      double int_layer_growth_conv_tol() const { return intlayergrowth_convtol_; };
      unsigned int_layer_growth_ite_max() const { return intlayergrowth_itemax_; };
      bool partitioned_multi_scale() const { return partitioned_multiscale_; };
      //! return true if external force is applied
      [[nodiscard]] bool has_external_force() const { return has_external_force_; };
      //! number of dofset associated with displacement dofs
      int nds_disp() const;
      //! number of dofset associated with interface growth dofs
      int nds_growth() const;
      //! number of dofset to write micro scale values on
      int nds_micro() const;
      //! number of dofset associated with pressure dofs
      int nds_pres() const;
      //! number of dofset associated with scalar transport dofs
      int nds_scatra() const;
      //! number of dofset associated with temperature dofs
      int nds_thermo() const;
      //! number of dofset associated with two-tensor quantity dofs, e.g. stresses, strains
      int nds_two_tensor_quantity() const;
      //! number of dofset associated with velocity related dofs
      int nds_vel() const;
      //! number of dofset associated with wall shear stress dofs
      int nds_wss() const;

      //! @}

      //! @name return stabilization parameters
      //! @{
      Inpar::ScaTra::StabType stab_type() const { return stabtype_; };
      Inpar::ScaTra::TauType tau_def() const { return whichtau_; };
      Inpar::ScaTra::CharEleLength char_ele_length() const { return charelelength_; };
      double usfemgls_fac() const { return diffreastafac_; };
      bool rb_sub_gr_vel() const { return sgvel_; };
      bool assgd() const { return assgd_; };
      Inpar::ScaTra::AssgdType assgd_type() const { return whichassgd_; };
      bool tau_gp() const { return tau_gp_; };
      bool mat_gp() const { return mat_gp_; };
      double tau_value() const { return tau_value_; };
      //! @}

     private:
      //! private constructor for singletons
      ScaTraEleParameterStd(const std::string& disname  //!< name of discretization
      );

      //! @name general parameters
      //! @{
      //! flag for ALE
      bool is_ale_;

      //! flag for conservative form
      bool is_conservative_;

      //! flag for use of spherical coordinates
      bool sphericalcoords_;

      //! flag for writing the flux vector fields
      Inpar::ScaTra::FluxType calcflux_domain_;

      //! ids of scalars for which flux vectors are written (starting with 1)
      std::shared_ptr<std::vector<int>> writefluxids_;

      //! flag for finite difference check
      Inpar::ScaTra::FdCheck fdcheck_;

      //! perturbation magnitude for finite difference check
      double fdcheckeps_;

      //! relative tolerance for finite difference check
      double fdchecktol_;

      //! number of dofset associated with displacement dofs
      int nds_disp_;

      //! number of dofset associated with interface growth dofs
      int nds_growth_;

      //! number of dofset to write micro scale values on
      int nds_micro_;

      //! number of dofset associated with pressure dofs
      int nds_pres_;

      //! number of dofset associated with scalar transport dofs
      int nds_scatra_;

      //! number of dofset associated with temperature dofs
      int nds_thermo_;

      //! number of dofset associated with two-tensor quantity dofs, e.g. stresses, strains
      int nds_two_tensor_quantity_;

      //! number of dofset associated with velocity related dofs
      int nds_vel_;

      //! number of dofset associated with wall shear stress dofs
      int nds_wss_;

      //! problem number
      int probnum_;

      //! evaluation type
      bool semiimplicit_;

      //! local Newton-Raphson convergence tolerance for scatra-scatra interface coupling involving
      //! interface layer growth
      double intlayergrowth_convtol_;

      //! maximum number of local Newton-Raphson iterations for scatra-scatra interface coupling
      //! involving interface layer growth
      unsigned intlayergrowth_itemax_;

      // flag for truly partitioned multi-scale simulation
      bool partitioned_multiscale_;

      /// flag for external force
      bool has_external_force_;

      //! @}

      //! @name stabilization parameters
      //! @{
      //! type of stabilization
      Inpar::ScaTra::StabType stabtype_;

      //! definition of stabilization parameter
      Inpar::ScaTra::TauType whichtau_;

      //! definition of characteristic element length
      Inpar::ScaTra::CharEleLength charelelength_;

      //! parameter to switch between SUPG, GLS and USFEM
      double diffreastafac_;

      //! flag to include residual-based subgrid-scale velocity
      bool sgvel_;

      //! flag to active artificial diffusion
      bool assgd_;

      //! definition of artificial diffusion
      Inpar::ScaTra::AssgdType whichassgd_;

      //! flag for evaluation of tau at Gauss point
      bool tau_gp_;

      //! flag for evaluation of material at Gauss point
      bool mat_gp_;

      //! numerical value for stabilization parameter
      double tau_value_;
      //! @}

      //! parameter class for time integration
      Discret::Elements::ScaTraEleParameterTimInt* scatraparatimint_;
    };
  }  // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
