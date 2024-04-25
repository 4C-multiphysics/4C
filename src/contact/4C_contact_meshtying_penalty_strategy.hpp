/*---------------------------------------------------------------------*/
/*! \file
\brief Penalty mesh-tying solving strategy.

\level 2


*/
/*---------------------------------------------------------------------*/
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

    \param[in] DofRowMap Dof row map of underlying problem
    \param[in] NodeRowMap Node row map of underlying problem
    \param[in] params List of contact/parameters
    \param[in] interface All contact interface objects
    \param[in] spatialDim Spatial dimension of the problem
    \param[in] comm Communicator
    \param[in] alphaf Mid-point for Generalized-alpha time integration
    \param[in] maxdof Highest DOF number in global problem
    */
    MtPenaltyStrategy(const Epetra_Map* DofRowMap, const Epetra_Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<Teuchos::RCP<MORTAR::Interface>> interface,
        const int spatialDim, const Teuchos::RCP<const Epetra_Comm>& comm, const double alphaf,
        const int maxdof);



    //! @name Access methods

    /*!
    \brief Return L2-norm of active constraints

    */
    double ConstraintNorm() const override { return constrnorm_; }

    /*!
    \brief Return initial penalty parameter

    */
    double InitialPenalty() override { return initialpenalty_; }

    //@}

    //! @name Evaluation methods

    /*!
    \brief Do mortar coupling in reference configuration

    Only do this ONCE for meshtying upon initialization!

    */
    void MortarCoupling(const Teuchos::RCP<const Epetra_Vector>& dis) override;

    /*!
    \brief Mesh initialization for rotational invariance

    Compute necessary modifications to the reference configuration of interface nodes, such that the
    weighted gap in the modified reference configuration is zero.

    \note Only do this \em once for meshtying upon initialization!

    \warning This is only implemented for mortar coupling. No implementation for node-to-segment
    approach.

    \return Vector with modified nodal positions
    */
    Teuchos::RCP<const Epetra_Vector> MeshInitialization() override;

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
    void EvaluateMeshtying(Teuchos::RCP<CORE::LINALG::SparseOperator>& kteff,
        Teuchos::RCP<Epetra_Vector>& feff, Teuchos::RCP<Epetra_Vector> dis) override;

    /*!
    \brief Initialize Uzawa step

    LM is updated to z = zuzawa - pp * gap. This mehtod is called at the
    beginning of the second, third, ... Uzawa iterarion in order to
    create an out-of-balance force again.

    */
    void InitializeUzawa(Teuchos::RCP<CORE::LINALG::SparseOperator>& kteff,
        Teuchos::RCP<Epetra_Vector>& feff) override;

    /*!
    \brief Reset penalty parameter to intial value

    When applying an Augmented Lagrangian version of the penalty approach,
    the penalty parameter is sometimes updated during the Uzawa steps in
    order to accelerate convergence of the constraint norm. This increase
    in penalty stiffness can be dealt with, because at the time it is applied
    the constraint norm is already quite low. Yet, for a new time step, we have
    to come back to the initial penalty parameter. Thus, this method is called
    at the beginning of each time step and resets the penalty parameter to its initial value.

    */
    void ResetPenalty() override;

    void ModifyPenalty() override;

    /*!
    \brief Compute L2-norm of active constraints

    In a classical penalty approach, the constraint norm is only monitored.
    When applying an Augmented Lagrangian version, the constraint norm is the
    relevant stopping criterion of the Uzawa iteration. In order to accelerate
    convergence, a heuristic update formula for the penalty parameter is applied
    in this method, too.

    */
    void UpdateConstraintNorm(int uzawaiter = 0) override;

    /*!
    \brief Store Lagrange multipliers for next Uzawa step

    A method ONLY called for the Uzawa Augmented Lagrangian version of the penalty method.
    At the end of an Uzawa step, the converged Lagrange multiplier value is stored
    in the variable zuzawa_, which is then used in the next Uzawa step.

    */
    void UpdateUzawaAugmentedLagrange() override;

    /*!
    \brief Tell that this is a penalty strategy
    */
    bool IsPenalty() const override { return true; };
    //@}

    //! @name Empty functions (Lagrange meshtying)

    // All these functions only have functionality in Lagrange meshtying simulations,
    // thus they are defined empty here in the case of Penalty meshtying.

    void Recover(Teuchos::RCP<Epetra_Vector> disi) override { return; };
    void BuildSaddlePointSystem(Teuchos::RCP<CORE::LINALG::SparseOperator> kdd,
        Teuchos::RCP<Epetra_Vector> fd, Teuchos::RCP<Epetra_Vector> sold,
        Teuchos::RCP<CORE::LINALG::MapExtractor> dbcmaps, Teuchos::RCP<Epetra_Operator>& blockMat,
        Teuchos::RCP<Epetra_Vector>& blocksol, Teuchos::RCP<Epetra_Vector>& blockrhs) override
    {
      FOUR_C_THROW(
          "A penalty approach does not have Lagrange multiplier DOFs. So, saddle point system "
          "makes no sense here.");
    };
    void UpdateDisplacementsAndLMincrements(
        Teuchos::RCP<Epetra_Vector> sold, Teuchos::RCP<const Epetra_Vector> blocksol) override
    {
      FOUR_C_THROW(
          "A penalty approach does not have Lagrange multiplier DOFs. So, saddle point system "
          "makes no sense here.");
    };
    void EvalConstrRHS()
    {
      std::cout << "Warning: No constraint RHS in contact penalty strategy" << std::endl;
    }
    //@}

    //! @name New time integration

    /*! \brief Evaluate residual
     *
     * @param[in] dis Current displacement field
     * @return Boolean flag indicating successfull evaluation
     */
    bool EvaluateForce(const Teuchos::RCP<const Epetra_Vector> dis) override;

    /*! \brief Evaluate stiffness term
     *
     * \note We assume that the stiffness matrix has already been evaluated,
     * so we do nothing in here.
     *
     * @param[in] dis Current displacement field
     * @return Boolean flag indicating successfull evaluation
     */
    bool EvaluateStiff(const Teuchos::RCP<const Epetra_Vector> dis) override;

    /*! \brief Evaluate residual and stiffness matrix
     *
     * @param[in] dis Current displacement field
     * @return Boolean flag indicating successfull evaluation
     */
    bool EvaluateForceStiff(const Teuchos::RCP<const Epetra_Vector> dis) override;

    //! Return the desired right-hand-side block pointer (read-only) [derived]
    Teuchos::RCP<const Epetra_Vector> GetRhsBlockPtr(
        const enum CONTACT::VecBlockType& bt) const override;

    //! Return the desired matrix block pointer (read-only) [derived]
    Teuchos::RCP<CORE::LINALG::SparseMatrix> GetMatrixBlockPtr(
        const enum CONTACT::MatBlockType& bt) const override;

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
    Teuchos::RCP<CORE::LINALG::SparseMatrix> mtm_;

    //! Mortar matrix product \f$M^T D\f$
    Teuchos::RCP<CORE::LINALG::SparseMatrix> mtd_;

    //! Mortar matrix product \f$D^T M\f$
    Teuchos::RCP<CORE::LINALG::SparseMatrix> dtm_;

    //! Mortar matrix product \f$D^T D\f$
    Teuchos::RCP<CORE::LINALG::SparseMatrix> dtd_;


    //! Global stiffness matrix
    Teuchos::RCP<CORE::LINALG::SparseMatrix> stiff_;

    /*! \brief Global residual vector
     *
     * \todo Is this the residual or the right-hand side vector?
     */
    Teuchos::RCP<Epetra_Vector> force_;


  };  // class MtPenaltyStrategy
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
