// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_MESHTYING_PENALTY_STRATEGY_HPP
#define FOUR_C_CONTACT_MESHTYING_PENALTY_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_contact_meshtying_abstract_strategy.hpp"

FOUR_C_NAMESPACE_OPEN


namespace CONTACT
{
  /*!
   \brief Meshtying solving strategy with regularization of Lagrangian multipliers,
   also known as Penalty Method or regularization. An Augmented Lagrangian version
   based on the Uzawa algorithm is included, too.

   This is a specialization of the abstract meshtying algorithm as defined in MtAbstractStrategy.
   For a more general documentation of the involved functions refer to MtAbstractStrategy.

   */
  class MtPenaltyStrategy : public MtAbstractStrategy
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
    MtPenaltyStrategy(const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<std::shared_ptr<Mortar::Interface>> interface,
        const int spatialDim, const MPI_Comm& comm, const double alphaf, const int maxdof);



    //! @name Access methods

    /*!
    \brief Return L2-norm of active constraints

    */
    double constraint_norm() const override { return constrnorm_; }

    /*!
    \brief Return initial penalty parameter

    */
    double initial_penalty() const override { return initialpenalty_; }

    //@}

    //! @name Evaluation methods

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

    For a penalty strategy this includes the evaluation of regularized forces
    and results in a simple addition of extra stiffness contributions to kteff
    and extra meshtying forces to feff.

    \param kteff (in/out): effective stiffness matrix (without -> with contact)
    \param feff (in/out): effective residual / force vector (without -> with contact)
    \param dis (in): current displacement state

    */
    void evaluate_meshtying(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
        std::shared_ptr<Core::LinAlg::Vector<double>> dis) override;

    /*!
    \brief Initialize Uzawa step

    LM is updated to z = zuzawa - pp * gap. This method is called at the
    beginning of the second, third, ... Uzawa iterarion in order to
    create an out-of-balance force again.

    */
    void initialize_uzawa(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) override;

    /*!
    \brief Reset penalty parameter to initial value

    When applying an Augmented Lagrangian version of the penalty approach,
    the penalty parameter is sometimes updated during the Uzawa steps in
    order to accelerate convergence of the constraint norm. This increase
    in penalty stiffness can be dealt with, because at the time it is applied
    the constraint norm is already quite low. Yet, for a new time step, we have
    to come back to the initial penalty parameter. Thus, this method is called
    at the beginning of each time step and resets the penalty parameter to its initial value.

    */
    void reset_penalty() override;

    void modify_penalty() override;

    /*!
    \brief Compute L2-norm of active constraints

    In a classical penalty approach, the constraint norm is only monitored.
    When applying an Augmented Lagrangian version, the constraint norm is the
    relevant stopping criterion of the Uzawa iteration. In order to accelerate
    convergence, a heuristic update formula for the penalty parameter is applied
    in this method, too.

    */
    void update_constraint_norm(int uzawaiter = 0) override;

    /*!
    \brief Store Lagrange multipliers for next Uzawa step

    A method ONLY called for the Uzawa Augmented Lagrangian version of the penalty method.
    At the end of an Uzawa step, the converged Lagrange multiplier value is stored
    in the variable zuzawa_, which is then used in the next Uzawa step.

    */
    void update_uzawa_augmented_lagrange() override;

    /*!
    \brief Tell that this is a penalty strategy
    */
    bool is_penalty() const override { return true; };
    //@}

    //! @name Empty functions (Lagrange meshtying)

    // All these functions only have functionality in Lagrange meshtying simulations,
    // thus they are defined empty here in the case of Penalty meshtying.

    void recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi) override { return; };
    void build_saddle_point_system(std::shared_ptr<Core::LinAlg::SparseOperator> kdd,
        std::shared_ptr<Core::LinAlg::Vector<double>> fd,
        std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps,
        std::shared_ptr<Core::LinAlg::SparseOperator>& blockMat,
        std::shared_ptr<Core::LinAlg::Vector<double>>& blocksol,
        std::shared_ptr<Core::LinAlg::Vector<double>>& blockrhs) override
    {
      FOUR_C_THROW(
          "A penalty approach does not have Lagrange multiplier DOFs. So, saddle point system "
          "makes no sense here.");
    };
    void update_displacements_and_l_mincrements(std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<const Core::LinAlg::Vector<double>> blocksol) override
    {
      FOUR_C_THROW(
          "A penalty approach does not have Lagrange multiplier DOFs. So, saddle point system "
          "makes no sense here.");
    };
    void eval_constr_rhs()
    {
      std::cout << "Warning: No constraint RHS in contact penalty strategy" << std::endl;
    }
    //@}

    //! @name New time integration

    /*! \brief Evaluate residual
     *
     * @param[in] dis Current displacement field
     * @return Boolean flag indicating successful evaluation
     */
    void evaluate_force(const std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

    /*! \brief Evaluate stiffness term
     *
     * \note We assume that the stiffness matrix has already been evaluated,
     * so we do nothing in here.
     *
     * @param[in] dis Current displacement field
     * @return Boolean flag indicating successful evaluation
     */
    void evaluate_stiff(const std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

    /*! \brief Evaluate residual and stiffness matrix
     *
     * @param[in] dis Current displacement field
     * @return Boolean flag indicating successful evaluation
     */
    void evaluate_force_stiff(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

    //! Return the desired right-hand-side block pointer (read-only) [derived]
    std::shared_ptr<const Core::LinAlg::Vector<double>> get_rhs_block_ptr(
        const CONTACT::VecBlockType& bt) const override;

    //! Return the desired matrix block pointer (read-only) [derived]
    std::shared_ptr<Core::LinAlg::SparseMatrix> get_matrix_block_ptr(
        const CONTACT::MatBlockType& bt) const override;

    //@}

   protected:
    // don't want = operator and cctor
    MtPenaltyStrategy operator=(const MtPenaltyStrategy& old) = delete;
    MtPenaltyStrategy(const MtPenaltyStrategy& old) = delete;

    //! L2-norm of normal contact constraints
    double constrnorm_;

    //! Initial penalty parameter
    double initialpenalty_;

    //! Mortar matrix product \f$M^T M\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> mtm_;

    //! Mortar matrix product \f$M^T D\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> mtd_;

    //! Mortar matrix product \f$D^T M\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> dtm_;

    //! Mortar matrix product \f$D^T D\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> dtd_;


    //! Global stiffness matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_;

    /*! \brief Global residual vector
     *
     * \todo Is this the residual or the right-hand side vector?
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> force_;


  };  // class MtPenaltyStrategy
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
