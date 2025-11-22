// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_MESHTYING_LAGRANGE_STRATEGY_HPP
#define FOUR_C_CONTACT_MESHTYING_LAGRANGE_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_contact_meshtying_abstract_strategy.hpp"

FOUR_C_NAMESPACE_OPEN


namespace CONTACT
{
  /*!
   \brief Meshtying solving strategy with (standard/dual) Lagrangian multipliers.

   This is a specialization of the abstract meshtying algorithm as defined in MtAbstractStrategy.
   For a more general documentation of the involved functions refer to MtAbstractStrategy.

   \sa MtAbstractStrategy

   */
  class MtLagrangeStrategy : public MtAbstractStrategy
  {
   public:
    /*!
    \brief Standard Constructor

    \param[in] dof_row_map Dof row map of underlying problem
    \param[in] NodeRowMap Node row map of underlying problem
    \param[in] params List of contact/parameters
    \param[in] interface All contact interface objects
    \param[in] spatialDim Spatial dimension of the problem
    \param[in] comm Communicator
    \param[in] alphaf Mid-point for Generalized-alpha time integration
    \param[in] maxdof Highest DOF number in global problem
    */
    MtLagrangeStrategy(const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<std::shared_ptr<Mortar::Interface>> interface,
        const int spatialDim, const MPI_Comm& comm, const double alphaf, const int maxdof);



    /*!
    \brief Do mortar coupling in reference configuration

    Only do this ONCE for meshtying upon initialization!

    */
    void mortar_coupling(const std::shared_ptr<const Core::LinAlg::Vector<double>>& dis) override;

    /*!
    \brief Mesh initialization for rotational invariance

    Compute necessary modifications to the reference configuration of interface nodes, such that the
    weighted gap in the modified reference configuration is zero.

    \note Only do this \em once for meshtying upon initialization!

    \warning This is only implemented for mortar coupling. No implementation for node-to-segment
    approach.

    \return Vector with modified nodal positions
    */
    std::shared_ptr<const Core::LinAlg::Vector<double>> mesh_initialization() override;

    /*!
    \brief Evaluate meshtying

    This is the main routine of our meshtying algorithms on a global level.
    It contains the setup of the global linear system including meshtying.

    For a Lagrangian strategy this involves heavy modification to the initial kteff and feff.
    Hence, they are in fact build from scratch here. The application of modifications to
    groups of dofs (slave, master, etc.) results in some matrix and vector splitting and a
    lot of matrix-vector calculation in here!

    \param kteff (in/out): effective stiffness matrix (without -> with contact)
    \param feff (in/out): effective residual / force vector (without -> with contact)
    \param dis (in): current displacement state

    */
    void evaluate_meshtying(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
        std::shared_ptr<Core::LinAlg::Vector<double>> dis) override;

    /*!
    \brief Build 2x2 saddle point system

    \param kdd (in): the displacement dof stiffness (upper left block)
    \param fd (in): the displacement dof r.h.s. (upper block)
    \param sold (in): the displacement dof solution increment
    \param dirichtoggle (in): toggle vector for dirichlet conditions
    \param blockMat (out): Core::LinAlg::SparseOperator containing the 2x2 block sparse matrix
    object
    \param mergedsol (out): Core::LinAlg::Vector<double> for merged solution vector
    \param mergedrhs (out): Core::LinAlg::Vector<double> for merged right hand side vector
    */
    void build_saddle_point_system(std::shared_ptr<Core::LinAlg::SparseOperator> kdd,
        std::shared_ptr<Core::LinAlg::Vector<double>> fd,
        std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps,
        std::shared_ptr<Core::LinAlg::SparseOperator>& blockMat,
        std::shared_ptr<Core::LinAlg::Vector<double>>& blocksol,
        std::shared_ptr<Core::LinAlg::Vector<double>>& blockrhs) override;

    /*!
    \brief Update internal member variables after solving the 2x2 saddle point contact system

    \param sold (out): the displacement dof solution increment (associated with displacement dofs)
    \param mergedsol (in): Core::LinAlg::Vector<double> for merged solution vector (containing the
    new solution vector of the full merged linear system)
    */
    void update_displacements_and_l_mincrements(std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<const Core::LinAlg::Vector<double>> blocksol) override;


    void eval_constr_rhs()
    {
      std::cout
          << "Warning: The evaluate_constr_rhs() function is not yet implemented for meshtying."
          << std::endl;
    }


    /*!
    \brief Recovery method

    We only recover the Lagrange multipliers here, which had been statically condensed during
    the setup of the global problem!

    */
    void recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi) override;

    //@}

    /*! @name Empty functions (Penalty meshtying)
     *
     * All these functions only have functionality in Penalty meshtying simulations,
     * thus they are defined as FOUR_C_THROW here in the case of Lagrange meshtying.
     */

    double constraint_norm() const override { return 0.0; }
    void initialize_uzawa(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) override
    {
    }
    double initial_penalty() const override { return 0.0; }
    void reset_penalty() override {}
    void modify_penalty() override {}
    void save_reference_state(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override {}
    void update_uzawa_augmented_lagrange() override {}
    void update_constraint_norm(int uzawaiter = 0) override {}
    bool is_penalty() const override { return false; };

    //@}

    //! @name New time integration
    //!@{

    /*! \brief Evaluate residual
     *
     * @param[in] dis Current displacement field
     * @return Boolean flag indicating successful evaluation
     */
    bool evaluate_force(const std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

    /*! \brief Evaluate stiffness term
     *
     * @param[in] dis Current displacement field
     * @return Boolean flag indicating successful evaluation
     */
    bool evaluate_stiff(const std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

    /*! \brief Evaluate residual and stiffness matrix
     *
     * @param[in] dis Current displacement field
     * @return Boolean flag indicating successful evaluation
     */
    bool evaluate_force_stiff(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

    //! Return the desired right-hand-side block pointer (read-only) [derived]
    std::shared_ptr<const Core::LinAlg::Vector<double>> get_rhs_block_ptr(
        const CONTACT::VecBlockType& bt) const override;

    //! Return the desired matrix block pointer (read-only) [derived]
    std::shared_ptr<Core::LinAlg::SparseMatrix> get_matrix_block_ptr(
        const CONTACT::MatBlockType& bt) const override;

    /*! \brief Modify system before linear solve
     *
     * Perform the static condensation of mortar terms if a condensed formulation is used.
     *
     * This exploit the biorthogonality condition of the dual shape functions.
     *
     * @param[in/out] kteff Stiffness matrix
     * @param[in/out] rhs right-hand side vector
     *
     * \warning This only works for dual shape functions. Standard shape functions are prohibited
     * as they are too expensive.
     *
     * \todo Is this really the right-hand side vector or the residual?
     */
    void run_pre_apply_jacobian_inverse(std::shared_ptr<Core::LinAlg::SparseMatrix> kteff,
        Core::LinAlg::Vector<double>& rhs) override;

    void run_post_apply_jacobian_inverse(Core::LinAlg::Vector<double>& result) override;

    void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
        const Core::LinAlg::Vector<double>& dir, const Core::LinAlg::Vector<double>& xnew) override;

    void remove_condensed_contributions_from_rhs(Core::LinAlg::Vector<double>& rhs) const override;

    //!@}

   protected:
    //! @name Accessors
    //!@{

    //! Access to #mhatmatrix_
    virtual std::shared_ptr<const Core::LinAlg::SparseMatrix> get_m_hat() { return mhatmatrix_; };

    //! Access to non-redistributed mhatmatrix
    std::shared_ptr<const Core::LinAlg::SparseMatrix> get_non_redist_m_hat();

    //! Access to #invd_
    virtual std::shared_ptr<const Core::LinAlg::SparseMatrix> get_d_inverse() { return invd_; };

    //!@}

   private:
    //! don't want = operator
    MtLagrangeStrategy operator=(const MtLagrangeStrategy& old);

    //! don't want copy constructor
    MtLagrangeStrategy(const MtLagrangeStrategy& old);

    //! Constraint matrix for saddle point system
    std::shared_ptr<Core::LinAlg::SparseMatrix> conmatrix_;

    //! Mortar projection matrix \f$P = D^{-1} M\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> mhatmatrix_;

    //! Slave side effective forces (needed for Lagrange multipliers)
    std::shared_ptr<Core::LinAlg::Vector<double>> fs_;

    //! Inverse \f$D^{-1}\f$ of Mortar matrix \f$D\f$ (needed for Lagrange multipliers)
    std::shared_ptr<Core::LinAlg::SparseMatrix> invd_;

    /*! @name Blocks for Jacobian matrix
     *
     * Subscripts are defined as follows
     * - s: slave
     * - m: master
     * - n: non-mortar nodes (i.e. all non-interface nodes in all subdomains)
     */
    //!@{

    //! Stiffness block \f$K_{sn}\f$ (needed for Lagrange multipliers)
    std::shared_ptr<Core::LinAlg::SparseMatrix> ksn_;

    //! Stiffness block \f$K_{sm}\f$ (needed for Lagrange multipliers)
    std::shared_ptr<Core::LinAlg::SparseMatrix> ksm_;

    //! Stiffness block \f$K_{ss}\f$ (needed for Lagrange multipliers)
    std::shared_ptr<Core::LinAlg::SparseMatrix> kss_;

    //!@}

  };  // class MtLagrangeStrategy
}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif
