// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_WEAR_INTERFACE_HPP
#define FOUR_C_CONTACT_WEAR_INTERFACE_HPP

#include "4C_config.hpp"

#include "4C_contact_input.hpp"
#include "4C_contact_interface.hpp"
#include "4C_inpar_wear.hpp"
#include "4C_linalg_fevector.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Wear
{
  class WearInterface : public CONTACT::Interface
  {
   public:
    /*!
    \brief Constructor

    */
    WearInterface(const std::shared_ptr<Mortar::InterfaceDataContainer>& interfaceData_ptr,
        const int id, MPI_Comm comm, const int dim, const Teuchos::ParameterList& icontact,
        bool selfcontact);

    /*!
    \brief Assemble second mortar D matrix for both-sided wear

    */
    virtual void assemble_d2(Core::LinAlg::SparseMatrix& dglobal);

    /*!
    \brief Assemble Mortar wear matrices T and E

    */
    virtual void assemble_te(
        Core::LinAlg::SparseMatrix& tglobal, Core::LinAlg::SparseMatrix& eglobal);

    /*!
    \brief Assemble Mortar wear matrices T and E (maser side)

    */
    virtual void assemble_te_master(
        Core::LinAlg::SparseMatrix& tglobal, Core::LinAlg::SparseMatrix& eglobal);

    /*!
    \brief Assemble matrices LinT containing linearizations
           w.r.t. displacements
    */
    virtual void assemble_lin_t_d(Core::LinAlg::SparseMatrix& lintglobal);

    /*!
    \brief Assemble matrices LinT containing linearizations
           w.r.t. displacements (for master side)
    */
    virtual void assemble_lin_t_d_master(Core::LinAlg::SparseMatrix& lintglobal);

    /*!
    \brief Assemble matrices LinT containing linearizations
           w.r.t. LM
    */
    virtual void assemble_lin_t_lm(Core::LinAlg::SparseMatrix& lintglobal);

    /*!
    \brief Assemble matrices LinT containing linearizations
           w.r.t. LM
    */
    virtual void assemble_lin_t_lm_master(Core::LinAlg::SparseMatrix& lintglobal);

    /*!
    \brief Assemble matrices LinE containing linearizations
           w.r.t. displacements
    */
    virtual void assemble_lin_e_d(Core::LinAlg::SparseMatrix& lineglobal);

    /*!
    \brief Assemble matrices LinE containing linearizations
           w.r.t. displacements (for master side)
    */
    virtual void assemble_lin_e_d_master(Core::LinAlg::SparseMatrix& lineglobal);

    /*!
    \brief Assemble matrix S containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the normal contact condition g~ = 0. Concretely, this
    includes assembling the linearizations of the slave side
    nodal normals and of the Mortar matrices D  and M.

    */
    void assemble_s(Core::LinAlg::SparseMatrix& sglobal) override;

    /*!
    \brief Assemble matrix S containing linearizations w

    */
    virtual void assemble_lin_g_w(Core::LinAlg::SparseMatrix& sglobal);

    /*!
    \brief Assemble matrix LinStick containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the tangential stick condition delta tg = 0. Concretely, this
    includes assembling the linearizations of the slave side
    nodal tangents and of the Mortar matrices D  and M.

    */
    void assemble_lin_stick(Core::LinAlg::SparseMatrix& linstickLMglobal,
        Core::LinAlg::SparseMatrix& linstickDISglobal,
        Core::LinAlg::Vector<double>& linstickRHSglobal) override;
    /*!
    \brief Assemble matrix LinSlip containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the tangential slip condition. Concretely, this
    includes assembling the linearizations of the slave side
    nodal tangents and of the Mortar matrices D  and M.

    */
    void assemble_lin_slip(Core::LinAlg::SparseMatrix& linslipLMglobal,
        Core::LinAlg::SparseMatrix& linslipDISglobal,
        Core::LinAlg::Vector<double>& linslipRHSglobal) override;

    /*!
    \brief Assemble matrix LinSlip containing w linearizations

    */
    virtual void assemble_lin_slip_w(Core::LinAlg::SparseMatrix& linslipWglobal);

    /*!
    \brief Assemble matrices W containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the normal contact and slip contact condition for ~w.
    --> w.r.t. lagr. mult.

    */
    virtual void assemble_lin_w_lm(Core::LinAlg::SparseMatrix& sglobal);
    virtual void assemble_lin_w_lm_sl(Core::LinAlg::SparseMatrix& sglobal);
    virtual void assemble_lin_w_lm_st(Core::LinAlg::SparseMatrix& sglobal);

    /*!
    \brief Assemble wear w

     This method assembles the weighted wear vector.
     */
    virtual void assemble_wear(Core::LinAlg::Vector<double>& wglobal);

    /*!
    \brief Build active set (nodes / dofs) of this interface

    If the flag init==true, the active set is initialized (for t=0)
    according to the contact initialization defined in the input file.

    */
    bool build_active_set(bool init = false) override;

    /*!
    \brief Build corresponding active set for master side

    */
    virtual bool build_active_set_master();

    /*!
    \brief Check mortar wear T derivatives with finite differences

    */
    void fd_check_mortar_t_deriv();

    /*!
    \brief Check mortar wear T derivatives with finite differences (Master)

    */
    void fd_check_mortar_t_master_deriv();

    /*!
    \brief Check mortar wear E derivatives with finite differences

    */
    void fd_check_mortar_e_deriv();

    /*!
    \brief Check mortar wear E derivatives with finite differences (for master)

    */
    void fd_check_mortar_e_master_deriv();

    /*!
    \brief Check mortar wear T derivatives with finite differences
      --> for wear condition

    */
    void fd_check_deriv_t_d(Core::LinAlg::SparseMatrix& lintdis);

    /*!
    \brief Check mortar wear T derivatives with finite differences
      --> for wear condition (Master)

    */
    void fd_check_deriv_t_d_master(Core::LinAlg::SparseMatrix& lintdis);

    /*!
    \brief Check mortar wear E derivatives with finite differences
      --> for wear condition

    */
    void fd_check_deriv_e_d(Core::LinAlg::SparseMatrix& linedis);

    /*!
    \brief Check mortar wear E derivatives with finite differences
      --> for wear condition (Master)

    */
    void fd_check_deriv_e_d_master(Core::LinAlg::SparseMatrix& linedis);
    /*!
    \brief Check weighted gap g derivatives with finite differences

    */
    void fd_check_gap_deriv();

    /*!
    \brief Check weighted gap g derivatives with finite differences

    */
    void fd_check_gap_deriv_w();

    /*!
    \brief Check weighted wear ~w derivatives with finite differences
           derivation w.r.t. displ.

    */
    void fd_check_wear_deriv();

    /*!
    \brief Check weighted wear ~w derivatives with finite differences
           derivation w.r.t. lagr.-mult.

    */
    void fd_check_wear_deriv_lm();

    /*!
    \brief Check slip condition derivatives with finite differences

    */
    virtual void fd_check_slip_deriv(Core::LinAlg::SparseMatrix& linslipLMglobal,
        Core::LinAlg::SparseMatrix& linslipDISglobal, Core::LinAlg::SparseMatrix& linslipWglobal);

    /*!
    \brief Assemble inactive rhs (incremental delta_w_)
    */
    virtual void assemble_inactive_wear_rhs(Core::LinAlg::Vector<double>& inactiverhs);

    /*!
    \brief Assemble inactive rhs (incremental delta_w_)
    */
    virtual void assemble_inactive_wear_rhs_master(Core::LinAlg::FEVector<double>& inactiverhs);

    /*!
    \brief Assemble wear-cond. rhs
    */
    virtual void assemble_wear_cond_rhs(Core::LinAlg::Vector<double>& rhs);

    /*!
    \brief Assemble wear-cond. rhs
    */
    virtual void assemble_wear_cond_rhs_master(Core::LinAlg::FEVector<double>& rhs);

    /*!
    \brief Initialize / reset interface for contact

    Derived version!

    */
    void initialize() final;


    /*!
    \brief Returning dofs for both-sided wear mapping

    */
    virtual std::shared_ptr<const Core::LinAlg::Map> involved_dofs() const { return involveddofs_; }

    virtual std::shared_ptr<const Core::LinAlg::Map> involved_nodes() const
    {
      return involvednodes_;
    }

    /*!
    \brief Set element areas

    Derived version!

    */
    void split_slave_dofs();
    void split_master_dofs();
    /*!
    \brief Set element areas

    Derived version!

    */
    void set_element_areas() final;

    /*!
    \brief Evaluate nodal normals

    */
    void evaluate_nodal_normals() const final;


    /*!
    \brief Evaluate nodal normals

    */
    void export_nodal_normals() const final;

    /*!
    \brief Update interface Wear variable sets

    This update is usually only done ONCE in the initialization phase
    and sets up the wear unknowns (only dofs) for the whole
    simulation.

    */
    virtual void update_w_sets(int offset_if, int maxdofwear, bool bothdiscr);

    /*!
    \brief Get map of slave wear dofs (Filled()==true is prerequisite)

    */
    virtual std::shared_ptr<const Core::LinAlg::Map> w_dofs() const
    {
      if (filled())
        return wdofmap_;
      else
        FOUR_C_THROW("CONTACT::WearInterface::fill_complete was not called");
    }

    /*!
    \brief Get map of master wear dofs (Filled()==true is prerequisite)

    */
    virtual std::shared_ptr<const Core::LinAlg::Map> wm_dofs() const
    {
      if (filled())
        return wmdofmap_;
      else
        FOUR_C_THROW("CONTACT::WearInterface::fill_complete was not called");
    }

    /*!
    \brief Get map of Lagrange multiplier dofs (Filled()==true is prerequisite)

    */
    virtual std::shared_ptr<const Core::LinAlg::Map> sn_dofs() const
    {
      if (filled())
        return sndofmap_;
      else
        FOUR_C_THROW("CONTACT::WearInterface::fill_complete was not called");
    }

    /*!
    \brief Get map of Lagrange multiplier dofs (Filled()==true is prerequisite)

    */
    virtual std::shared_ptr<const Core::LinAlg::Map> mn_dofs() const
    {
      if (filled())
        return mndofmap_;
      else
        FOUR_C_THROW("CONTACT::WearInterface::fill_complete was not called");
    }

    /*!
    \brief Get row map of active nodes (Filled()==true is prerequisite)

    */
    virtual std::shared_ptr<const Core::LinAlg::Map> active_master_nodes() const
    {
      if (filled())
        return activmasternodes_;
      else
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
    }

    /*!
    \brief Get row map of active nodes (Filled()==true is prerequisite)

    */
    virtual std::shared_ptr<const Core::LinAlg::Map> slip_master_nodes() const
    {
      if (filled())
        return slipmasternodes_;
      else
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
    }

    /*!
    \brief Get row map of active nodes (Filled()==true is prerequisite)

    */
    virtual std::shared_ptr<const Core::LinAlg::Map> slip_master_n_dofs() const
    {
      if (filled())
        return slipmn_;
      else
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
    }


    /*!
    \brief Get type of wear shapefnct

    */
    Inpar::Wear::WearShape wear_shape_fcn()
    {
      return Teuchos::getIntegralValue<Inpar::Wear::WearShape>(imortar_, "WEAR_SHAPEFCN");
    }

   private:
    /*!
    \brief initialize node and element data container

    Derived version!

    */
    void initialize_data_container() final;


    // both-sided wear specific stuff
    std::shared_ptr<Core::LinAlg::Map> involvednodes_;  // row map of all involved master nodes
    std::shared_ptr<Core::LinAlg::Map> involveddofs_;   // row map of all involved master dofs

    std::shared_ptr<Core::LinAlg::Map> wdofmap_;   // row map of all slave wear dofs
    std::shared_ptr<Core::LinAlg::Map> wmdofmap_;  // row map of all master wear dofs

    std::shared_ptr<Core::LinAlg::Map> sndofmap_;  // row map of all slave dofs (first entries)
    std::shared_ptr<Core::LinAlg::Map> mndofmap_;  // row map of all master dofs (first entries)

    std::shared_ptr<Core::LinAlg::Map>
        activmasternodes_;  // row map of all active master nodes (first entries)
    std::shared_ptr<Core::LinAlg::Map>
        slipmasternodes_;  // row map of all active master nodes (first entries)
    std::shared_ptr<Core::LinAlg::Map>
        slipmn_;  // row map of all active master nodes (first entries)

    bool wear_;      // bool for wear
    bool wearimpl_;  // bool for implicit wear
    bool wearpv_;    // bool for wear with own discretization
    bool wearboth_;  // bool for wear on both sides
    bool sswear_;    // bool steady state wear

  };  // class


}  // namespace Wear

FOUR_C_NAMESPACE_CLOSE

#endif
