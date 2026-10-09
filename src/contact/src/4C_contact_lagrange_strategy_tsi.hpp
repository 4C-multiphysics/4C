// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_LAGRANGE_STRATEGY_TSI_HPP
#define FOUR_C_CONTACT_LAGRANGE_STRATEGY_TSI_HPP

#include "4C_config.hpp"

#include "4C_contact_defines.hpp"
#include "4C_contact_lagrange_strategy.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseMatrix;
  class BlockSparseMatrixBase;
}  // namespace Core::LinAlg


namespace FSI
{
  namespace Utils
  {
    class MatrixRowTransform;
    class MatrixColTransform;
    class MatrixRowColTransform;
  }  // namespace Utils
}  // namespace FSI

namespace CONTACT
{
  // forward declaration
  // class WearInterface;
  /*!
   \brief Contact solving strategy with (standard/dual) Lagrangian multipliers.

   This is a specialization of the abstract contact algorithm as defined in AbstractStrategy.
   For a more general documentation of the involved functions refer to CONTACT::AbstractStrategy.

   */
  class LagrangeStrategyTsi : public LagrangeStrategy
  {
   public:
    /*!
      \brief Standard Constructor

     */
    LagrangeStrategyTsi(const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
        const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interface,
        int dim, MPI_Comm comm, double alphaf, int maxdof);


    //! @name Access methods

    //@}

    //! @name Evaluation methods

    /*!
      \brief Set current state
      ...Standard Implementation in Abstract Strategy:
      All interfaces are called to set the current deformation state
      (u, xspatial) in their nodes. Additionally, the new contact
      element areas are computed.

      ... + Overloaded Implementation in Poro Lagrange Strategy
      Set structure & fluid velocity and lagrangean multiplier to Contact nodes data container!!!

      \param statetype (in): enumerator defining which quantity to set (see mortar_interface.H for
      an overview) \param vec (in): current global state of the quantity defined by statetype
     */
    void set_state(
        const Mortar::StateType& statetype, const Core::LinAlg::Vector<double>& vec) override;

    // Overload CONTACT::AbstractStrategy::apply_force_stiff_cmt as this is called in the structure
    // --> to early for monolithically coupled algorithms!
    void apply_force_stiff_cmt(std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
        std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
        bool predictor) override
    {
      // structure single-field predictors (e.g.TangDis) may evaluate the structural contact part
      if (predictor) AbstractStrategy::apply_force_stiff_cmt(dis, kt, f, step, iter, predictor);
    }

    /*!
      \brief Apply thermo-contact to matrix blocks

      In the TSI case, the contact terms are applied to the global system here.
      The "usual" place, i.e. the
      evaluate(
        std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
      std::shared_ptr<Core::LinAlg::Vector<double>> dis) in the Contact_lagrange_strategy is
      overloaded to do nothing, since in a coupled problem, we need to be very careful, when
      condensating the Lagrange multipliers.

     */
    virtual void evaluate(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>>& combined_RHS,
        std::shared_ptr<Coupling::Adapter::Coupling> coupST,
        std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> temp);

    /*!
    \brief Overload CONTACT::LagrangeStrategy::recover as this is called in the structure

    --> not enough information available for monolithically coupled algorithms!
    */
    void recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi) override { return; };

    virtual void recover_coupled(
        std::shared_ptr<Core::LinAlg::Vector<double>> sinc,  /// displacement  increment
        std::shared_ptr<Core::LinAlg::Vector<double>> tinc,  /// thermal  increment
        std::shared_ptr<Coupling::Adapter::Coupling> coupST);

    void store_nodal_quantities(
        Mortar::StrategyBase::QuantityType type, Coupling::Adapter::Coupling& coupST);

    /*!
     \brief Update contact at end of time step

     \param dis (in):  current displacements (-> old displacements)

     */
    void update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

    /*!
     \brief Set time integration parameter from Thermo time integration

     */
    void set_alphaf_thermo(const Teuchos::ParameterList& tdyn);


    /*!
    \brief Perform a write restart

    A write restart is initiated by the contact manager. However, the manager has no
    direct access to the nodal quantities. Hence, a portion of the restart has to be
    performed on the level of the contact algorithm, for short: here's the right place.

    */
    void do_write_restart(
        std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>>& restart_vectors,
        bool forcedrestart = false) const override;

    /*!
    \brief Perform a write restart

    A write restart is initiated by the contact manager. However, the manager has no
    direct access to the nodal quantities. Hence, all the restart action has to be
    performed on the level of the contact algorithm, for short: here's the right place.

    */
    void do_read_restart(Core::IO::DiscretizationReader& reader,
        const Core::LinAlg::Vector<double>& disp_n) override;

    void set_coupling(std::shared_ptr<Coupling::Adapter::Coupling> coupST)
    {
      structure_thermo_coupling_ = coupST;
    };

    //@}

    // residual and increment norms
    double mech_contact_res_;
    double mech_contact_incr_;
    double thermo_contact_incr_;

   protected:
    // don't want = operator and cctor
    LagrangeStrategyTsi operator=(const LagrangeStrategyTsi& old) = delete;
    LagrangeStrategyTsi(const LagrangeStrategyTsi& old) = delete;

    // time integration
    double tsi_alpha_;

    std::shared_ptr<Core::LinAlg::Vector<double>>
        fscn_;  // structural contact forces of last time step (needed for time integration)
    std::shared_ptr<Core::LinAlg::Vector<double>>
        ftcn_;  // thermal    contact forces of last time step (needed for time integration)
    std::shared_ptr<Core::LinAlg::Vector<double>>
        ftcnp_;  // thermal   contact forces of this time step (needed for time integration)

    std::shared_ptr<Core::LinAlg::Vector<double>>
        z_thermo_;  // current vector of Thermo-Lagrange multipliers at t_n+1
    std::shared_ptr<Core::LinAlg::Map> thermo_act_dofs_;  // active thermo dofs
    std::shared_ptr<Core::LinAlg::Map> thermo_s_dofs_;    // source thermo dofs

    std::shared_ptr<Core::LinAlg::SparseMatrix>
        dinvA_;  // dinv on active displacement dofs (for recovery)
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        dinvAthr_;  // dinv on active thermal dofs (for recovery)
    // recovery of contact LM
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        kss_a_;  // Part of structure-stiffness (kss) that corresponds to active source rows
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        kst_a_;  // Part of coupling-stiffness  (kst) that corresponds to active source rows
    std::shared_ptr<Core::LinAlg::Vector<double>>
        rs_a_;  // Part of structural residual that corresponds to active source rows

    // recovery of thermal LM
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        ktt_a_;  // Part of structure-stiffness (ktt) that corresponds to active source rows
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        kts_a_;  // Part of coupling-stiffness  (kts) that corresponds to active source rows
    std::shared_ptr<Core::LinAlg::Vector<double>>
        rt_a_;  // Part of structural residual that corresponds to active source rows

    // pointer to TSI coupling object
    std::shared_ptr<Coupling::Adapter::Coupling> structure_thermo_coupling_;
  };  // class LagrangeStrategyTsi

  namespace Utils
  {
    //! @name little helpers
    void add_vector(Core::LinAlg::Vector<double>& src, Core::LinAlg::Vector<double>& dst);
  }  // namespace Utils
}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif
