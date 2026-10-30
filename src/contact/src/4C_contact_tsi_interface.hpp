// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_TSI_INTERFACE_HPP
#define FOUR_C_CONTACT_TSI_INTERFACE_HPP

#include "4C_config.hpp"

#include "4C_contact_input.hpp"
#include "4C_contact_interface.hpp"

FOUR_C_NAMESPACE_OPEN


namespace CONTACT
{
  class TSIInterface : public Interface
  {
   public:
    /*!
    \brief Constructor

    */
    TSIInterface(const std::shared_ptr<Mortar::InterfaceDataContainer>& interfaceData_ptr,
        const int id, MPI_Comm comm, const int dim, const Teuchos::ParameterList& icontact,
        bool selfcontact);

    enum LinDmXMode
    {
      LinDM_Diss,
      LinDM_ThermoLM,
      linDM_ContactLMnormal
    };

    /*!
    \brief Assemble matrix LinStick containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the tangential stick condition.

    */
    virtual void assemble_lin_stick(Core::LinAlg::SparseMatrix& linstickLMglobal,
        Core::LinAlg::SparseMatrix& linstickDISglobal,
        Core::LinAlg::SparseMatrix& linstickTEMPglobal,
        Core::LinAlg::Vector<double>& linstickRHSglobal);

    /*!
    \brief Assemble matrix LinSlip containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the tangential slip condition. Concretely, this
    includes assembling the linearizations of the source side
    nodal tangents and of the Mortar matrices D  and M.

    */
    virtual void assemble_lin_slip(Core::LinAlg::SparseMatrix& linslipLMglobal,
        Core::LinAlg::SparseMatrix& linslipDISglobal, Core::LinAlg::SparseMatrix& linslipTEMPglobal,
        Core::LinAlg::Vector<double>& linslipRHSglobal);


    /*!
    \brief Assemble contact heat conduction

    This method assembles the Thermo-Lagrange-Multiplier line
    in the global system of equations, containing linearizations
    wrt displacements, contact forces, temperatures
    */
    virtual void assemble_lin_conduct(Core::LinAlg::SparseMatrix& linConductDISglobal,
        Core::LinAlg::SparseMatrix& linConductTEMPglobal,
        Core::LinAlg::SparseMatrix& linConductThermoLMglobal,
        Core::LinAlg::SparseMatrix& linConductContactLMglobal);

    /*!
    \brief Assemble lumped mass matrix of the dual basis
           This is actually the D-Matrix. However, to keep
           the info where it comes from, we stick to that name

    */
    virtual void assemble_dual_mass_lumped(
        Core::LinAlg::SparseMatrix& dualMassGlobal,    /// dual mass matrix
        Core::LinAlg::SparseMatrix& linDualMassGlobal  // derivative of dual mass matrix wrt
                                                       // displacements multiplied with thermo-LM
    );

    /*!
      \brief Assemble the linearization of \f$\mathbf{D}\f$ and \f${\mathbf{M}}\f$ matrices times
      some nodal value.

      Specifically, this adds
      \f[
        \mathtt{linD\_X}_{kc} \mathrel{+}= \mathtt{fac}
          \frac{\mathrm{d}D_{jk}}{\mathrm{d}u_c} X_j, \qquad
        \mathtt{linM\_X}_{lc} \mathrel{-}= \mathtt{fac}
          \frac{\mathrm{d}M_{jl}}{\mathrm{d}u_c} X_j
      \f]
      for all \f$j\f$ in @p node_rowmap. \f$X_j\f$ is a (scalar) nodal value
      determined by @p mode. For example, if mode is @ref LinDmXMode::LinDM_Diss, then
      \f$X_j = \mathcal{P}_j\f$ is the nodal frictional dissipation power, see
      @ref assemble_dm_lin_diss() for the definition.
      */
    virtual void assemble_lin_dm_x(Core::LinAlg::SparseMatrix* linD_X,
        Core::LinAlg::SparseMatrix* linM_X, const double fac, const LinDmXMode mode,
        const std::shared_ptr<const Core::LinAlg::Map> node_rowmap);

    /*!
      \brief Assemble \f$\mathbf{D}\f$ and \f$\mathbf{M}\f$ matrices times the linearization of the
      nodal frictional dissipation \f$\mathcal{P}\f$.

      Specifically, this adds
      \f[
        \mathtt{d\_LinDissX}_{kc} \mathrel{+}= \mathtt{fac}\,
          D_{jk} \frac{\mathrm{d}\,\mathcal{P}_j}{\mathrm{d}X_c}, \qquad
        \mathtt{m\_LinDissX}_{kc} \mathrel{+}= \mathtt{fac}\,
          M_{jk} \frac{\mathrm{d}\,\mathcal{P}_j}{\mathrm{d}X_c}
      \f]
      for all active nodes \f$j\f$ and \f$X_c\f$ a displacement (DISP) or contact Lagrange
      multiplier (ContactLM) dof. The nodal frictional dissipation power (cf.
      https://mediatum.ub.tum.de/?id=1454145, equation 4.57) is
      \f[
        \mathcal{P}_j = \mathbf{t}_\tau \cdot \mathbf{v}_\tau
        = \mathbf{t} \cdot \left(\mathbf{I} - \mathbf{n}\otimes\mathbf{n}\right) \cdot
          \frac{\Delta\tilde{\mathbf{u}}}{\Delta t D_{jj}}
      \f]
      with the traction \f$\mathbf{t} = -\boldsymbol{\lambda}_j\f$, the normal \f$\mathbf{n}\f$
      and the weighted slip increment (jump) \f$\Delta\tilde{\mathbf{u}}\f$ (no summation over
      \f$j\f$). Its derivative w.r.t. a displacement dof is
      \f[
        \frac{\mathrm{d}\mathcal{P}_j}{\mathrm{d}u_c}
        = \left[\mathbf{t}_\tau \cdot \frac{\mathrm{d}\Delta\tilde{\mathbf{u}}}{\mathrm{d}u_c}
          - \left(t_n \Delta\tilde{\mathbf{u}} + \Delta\tilde{u}_n \mathbf{t}\right) \cdot
            \frac{\mathrm{d}\mathbf{n}}{\mathrm{d}u_c}\right] \frac{1}{\Delta t D_{jj}}
        - \frac{\mathcal{P}_j}{D_{jj}} \frac{\mathrm{d}D_{jj}}{\mathrm{d}u_c}
      \f]
      and w.r.t. the contact Lagrange multiplier
      \f[
        \frac{\mathrm{d}\mathcal{P}_j}{\mathrm{d}\boldsymbol{\lambda}_j}
        = -\left(\mathbf{I} - \mathbf{n}\otimes\mathbf{n}\right) \cdot
          \frac{\Delta\tilde{\mathbf{u}}}{\Delta t D_{jj}}
      \f]
      */
    virtual void assemble_dm_lin_diss(Core::LinAlg::SparseMatrix* d_LinDissDISP,
        Core::LinAlg::SparseMatrix* m_LinDissDISP, Core::LinAlg::SparseMatrix* d_LinDissContactLM,
        Core::LinAlg::SparseMatrix* m_LinDissContactLM, const double fac);

    /*!
      \brief Assemble the linearization of D and M times the temperature

      This assembles the derivative wrt displacements
      linDM_Temp_{jl}= LMj_n* ( D_{jk,c) T_k
                               -M_{jm,c} T_m )
                       + LMj_d * n_{d,l} * ( D_{jk} T_k
                                            -M_{jm} T_m )
      and the derivative wrt the contact Lagrange multiplier
      lin_lm_{jk} = nj_{k} * (D_jk T_k - M_jl T_l)
                                 (mind the minus sign!)
      */
    virtual void assemble_lin_l_mn_dm_temp(
        const double fac, Core::LinAlg::SparseMatrix* lin_disp, Core::LinAlg::SparseMatrix* lin_lm);

    /*!
      \brief Assemble the D and M times the normal Lagrange multiplier

      This assembles lambda_n* ( D_{jk) LMn_j
                                -M_{jl} LMn_j ) (mind the minus sign! No sum over j!)
      */
    virtual void assemble_dm_l_mn(const double fac, Core::LinAlg::SparseMatrix* DM_LMn);

    /*!
      \brief Assemble inactive part of the thermal heat conduction equation

      This assembles lambda(thermo) = 0 for all inactive contact nodes
      */
    virtual void assemble_inactive(Core::LinAlg::SparseMatrix* linConductThermoLM);


    /*!
    \brief Initialize / reset interface for contact

    Derived version with some additional TSI related stuff!

    */
    void initialize() final;

   protected:
  };  // class


}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
