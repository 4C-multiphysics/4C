// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_LAGRANGE_STRATEGY_HPP
#define FOUR_C_CONTACT_LAGRANGE_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_contact_abstract_strategy.hpp"

FOUR_C_NAMESPACE_OPEN


namespace CONTACT
{
  /*! \brief Contact solving strategy with (standard/dual) Lagrangian multipliers.

   This is a specialization of the abstract contact algorithm as defined in AbstractStrategy.
   For a more general documentation of the involved functions refer to AbstractStrategy.
  */

  class LagrangeStrategy : public AbstractStrategy
  {
   public:
    /*!
    \brief Shared data constructor

    Creates the strategy base object and initializes all global variables.

    \param[in] data_ptr Data container object
    \param[in] dof_row_map Dof row map of underlying problem
    \param[in] NodeRowMap Node row map of underlying problem
    \param[in] params List of contact/parameters
    \param[in] interface All contact interface objects
    \param[in] spatialDim Spatial dimension of the problem
    \param[in] comm Communicator
    \param[in] alphaf Mid-point for Generalized-alpha time integration
    \param[in] maxdof Highest DOF number in global problem
    */
    LagrangeStrategy(const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
        const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interface,
        const int spatialDim, MPI_Comm comm, const double alphaf, const int maxdof);

    /*!
    \brief Return convergence status of fixed-point active set search

    If this Lagrange contact strategy is based on a semi-smooth
    Newton approach and not on a fixed-point approach with two nested
    loops, then this method simply returns true, of course. Convergence
    of the active set is monitored with the flag activesetssconv_ in
    this case and activesetconv_ is meaningless.

    */
    bool active_set_converged() const override
    {
      bool semismooth = params().get<bool>("SEMI_SMOOTH_NEWTON");
      if (semismooth)
        return activesetssconv_;
      else
        return activesetconv_;
    }

    /*!
    \brief Return no. of fixed-point active sets in this time step

    */
    int active_set_steps() const override { return activesetsteps_; }

    /*! \brief Return the desired right-hand-side block pointer (read-only)
     *
     *  \remark Please note, that a nullptr pointer is returned, if no active contact
     *  contributions are present.
     *
     *  \param bt (in): Desired vector block type, e.g. displ, constraint,*/
    std::shared_ptr<const Core::LinAlg::Vector<double>> get_rhs_block_ptr(
        const CONTACT::VecBlockType& bt) const override;


    /*! \brief recover the current state
     *
     * The main task of this method is to recover the Lagrange multiplier solution.
     * The Lagrange multiplier solution will be stored inside the corresponding strategy
     * and is necessary for different internal evaluation methods. If the Lagrange multiplier
     * is condensed, this method is the right place to recover it from the displacement solution.
     * If it is not condensed (saddle-point system) use the ResetLagrangeMultiplier routine
     * instead.
     *
     * \param cparams (in): parameter interface between the contact objects and the structural time
     *                       integration
     * \param xold (in): old solution vector of the NOX solver
     * \param dir (in): current search direction
     * \param xnew (in): new solution vector of the NOX solver
     *
     * \note The search direction \c dir in general differs from the actual step since the step
     * length can be differ from 1.0.
     * */
    void run_post_compute_x(const CONTACT::ParamsInterface& cparams,
        const Core::LinAlg::Vector<double>& xold, const Core::LinAlg::Vector<double>& dir,
        const Core::LinAlg::Vector<double>& xnew) override;

    /*! \brief Return the desired matrix block pointer (read-only)
     *
     *  \remark Please note, that a nullptr pointer is returned, if no active contact
     *  contributions are present.
     *
     *  \param bt (in): Desired matrix block type, e.g. displ_displ, displ_lm, ...
     *  \param cparams (in): contact parameter interface (read-only) */
    std::shared_ptr<Core::LinAlg::SparseMatrix> get_matrix_block_ptr(
        const CONTACT::MatBlockType& bt,
        const CONTACT::ParamsInterface* cparams = nullptr) const override;

    /*! \brief Apply modifications (e.g. condensation) directly before linear solve
     *
     * \todo Complete documentation of parameters.
     *
     * @param kteff Jacobian matrix
     * @param rhs ?? Right-hand side of linear system or residual?
     */
    void run_pre_apply_jacobian_inverse(std::shared_ptr<Core::LinAlg::SparseMatrix> kteff,
        Core::LinAlg::Vector<double>& rhs) override;

    /*! \brief Perform condensation of frictionless contact
     *
     * \todo Complete documentation of parameters.
     *
     * @param kteff Jacobian matrix
     * @param rhs ?? Right-hand side of linear system or residual?
     */
    virtual void condense_frictionless(
        std::shared_ptr<Core::LinAlg::SparseMatrix> kteff, Core::LinAlg::Vector<double>& rhs);

    /*! \brief Perform condensation of frictional contact
     *
     * \todo Complete documentation of parameters.
     *
     * @param kteff Jacobian matrix
     * @param rhs ?? Right-hand side of linear system or residual?
     */
    virtual void condense_friction(
        std::shared_ptr<Core::LinAlg::SparseMatrix> kteff, Core::LinAlg::Vector<double>& rhs);

    /*! recover condensed Lagrange multiplier after linear solve
     *
     * \todo Complete documentation of parameters.
     *
     * @param cparams Parameter interface between contact and structural time integration
     * @param rhs ?? Right-hand side of linear system or residual?
     * @param result ??
     * @param xold ??
     * @param grp ??
     */
    void run_post_apply_jacobian_inverse(const CONTACT::ParamsInterface& cparams,
        const Core::LinAlg::Vector<double>& rhs, Core::LinAlg::Vector<double>& result,
        const Core::LinAlg::Vector<double>& xold, const NOX::Nln::Group& grp) override;

    //! The contributions to the structural right-hand-side block are calculated.
    virtual void eval_str_contact_rhs();

    //! Assemble contact contributions to the rhs
    virtual void assemble_contact_rhs();

    //! Assemble all contact contributions
    virtual void assemble_all_contact_terms();

    //! Assemble all contact contributions (frictionless contact)
    virtual void assemble_all_contact_terms_frictionless();

    //! Assemble all contact contributions (frictional contact)
    virtual void assemble_all_contact_terms_friction();

    const Core::LinAlg::Map& slave_n_dof_row_map(const bool& redist) const override
    {
      if ((not redist) and parallel_redistribution_status())
        FOUR_C_THROW("The original / not redistributed slave normal row map is not available!");

      return *gsdofrowmap_;
    }

    //! Get the active node row map of the previous Newton step
    std::shared_ptr<const Core::LinAlg::Map> get_old_active_row_nodes() const override
    {
      return gOldActiveSlaveNodes_;
    };

    //! Get the slip node row map of the previous Newton step
    std::shared_ptr<const Core::LinAlg::Map> get_old_slip_row_nodes() const override
    {
      return gOldslipnodes_;
    };

    //@}

    //! @name Evaluation methods

    /*!
    \brief Build 2x2 saddle point system

    The saddle-point system reads
    \f[
    \left[\begin{array}{cc}
    K_{dd} & K_{dz}\\
    K_{zd} & K_{zz}
    \end{array}\right]
    \left[\begin{array}{c}\Delta d\\ \Delta z\end{array}\right]
    = - \left[\begin{array}{c}f_d\\ f_z\end{array}\right]
    \f]
    with \f$d\f$ and \f$z\f$ denoting the displacement and Lagrange multiplier degrees of freedom
    (DOFs), respectively.

    In terms of data structures and parallel layout, the Lagrange multiplier DOFs have the same
    parallel layout as their slave side displacement DOFs counterparts, but distinct global IDs
    (GIDs). For evaluation of mortar matrices, the salve side dof_row_map #gsdofrowmap_ is used.
    However, for assembling the saddle-point system, the constraint equations need their own unique
    map, #glmdofrowmap_. To also account for the possibility of a parallel redistribution of the
    Mortar interface discretizations, the assembly is performed in two steps:

    1. Transform maps related to constraint DOFs of matrix blocks \f$K_{dz}\f$, \f$K_{zd}\f$, and
       \f$K_{zz}\f$ to the Lagrange multiplier dof_row_map #glmdofrowmap_.
    2. If parallel redistribution is active, do another transformation to go from the redistributed
       constraint dof_row_map #glmdofrowmap_ to the original distribution #non_redist_glmdofrowmap_.

    \param[in] kdd Displacement dof stiffness (upper left block)
    \param[in] fd  Displacement dof r.h.s. (upper block)
    \param[in] sold Displacement dof solution increment
    \param[in] dbcmaps Map extractor to apply Dirichlet boundary conditions
    \param[out] blockMat Core::LinAlg::SparseOperator containing the 2x2 block sparse matrix object
    \param[out] blocksol Core::LinAlg::Vector<double> for merged solution vector
    \param[out] blockrhs Core::LinAlg::Vector<double> for merged right hand side vector

    \todo Update documentation once class members have been removed and replaced by accessing the
    data container directly.
    \todo Argument \c numiter not used. Can it be removed?

    \sa update_displacements_and_l_mincrements
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

    We have to extract the displacement and Lagrange multiplier solution increment from the blocked
    solution vector. This requires to also transfer constraint GIDs back to #gsdofrowmap_ since
    constraints internally use the same GIDs as their displacement DOF counterparts.

    \note In case of parallel redistribution, we also have to transfer constraint data from the
    undredistributed layout (used to build the saddle-point system) to the redistributed layout
    (used to perform contact evaluations).

    \note More details on handling of maps w/ and w/o parallel redistribution are given in
    build_saddle_point_system().

    \param[out] sold Displacement dof solution increment (associated with displacement dofs)
    \param[in] blocksol Merged solution vector (containing the new solution vector of the full
                    merged linear system, i.e. displacement and Lagrange multiplier DOFs)

    \sa build_saddle_point_system
    */
    void update_displacements_and_l_mincrements(std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<const Core::LinAlg::Vector<double>> blocksol) override;

    /*!
    \brief The entries of the constraint right-hand side are calculated.

    This function is outsourced cause the vector is needed for the line search algorithm.

    */
    void evaluate_constr_rhs() override;

    /*! \brief Compute force and stiffness terms
     *
     * \param cparams (in): parameter interface between the contact objects and the structural time
     * integration*/
    void evaluate_force_stiff(CONTACT::ParamsInterface& cparams) override;

    /*! \brief Run at the beginning of the evaluate() routine
     *         set force evaluation flag
     *
     */
    void pre_evaluate(CONTACT::ParamsInterface& cparams) override;

    /*! \brief Run in the end of the evaluate() routine to reset
     *         force evaluation flag
     */
    void post_evaluate(CONTACT::ParamsInterface& cparams) override;

    /*! \brief This is a postprocessing functionality for nonsmooth contact
     */
    void compute_contact_stresses() final;

    /*!
    \brief Recovery method

    We only recover the Lagrange multipliers here, which had been
    statically condensed during the setup of the global problem!
    Optionally satisfaction or violation of the contact boundary
    conditions can be checked, too.

    */
    void recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi) override;

    /*! \brief Reset the internal stored Lagrange multipliers
     *
     * \param cparams (in): parameter interface between the contact objects and the structural time
     *                      integration
     * \param xnew    (in): new solution vector of the NOX solver
     */
    void reset_lagrange_multipliers(
        const CONTACT::ParamsInterface& cparams, const Core::LinAlg::Vector<double>& xnew) override;

    /*! \brief Compute force terms
     *
     *  \param cparams (in): parameter interface between the contact objects and the structural time
     * integration*/
    void evaluate_force(CONTACT::ParamsInterface& cparams) override;

    /*!
    \brief Update active set and check for convergence

    In this function we loop over all interfaces and then over all
    slave nodes to check, whether the assumption of them being active
    or inactive respectively has been correct. If a single node changes
    state, the active set is adapted accordingly and the convergence
    flag is kept on false. Here we have the "standard" case of two
    nested iteration loops, and as a consequence this method is
    called AFTER convergence of the inner Newton iteration. If there
    is a change in the active set, another full Newton iteration has
    to be performed for the current time / load step.

    */
    void update_active_set() override;

    /*!
    \brief Update active set and check for convergence

    In this function we loop over all interfaces and then over all
    slave nodes to check, whether the assumption of them being active
    or inactive respectively has been correct. If a single node changes
    state, the active set is adapted accordingly and the convergence
    flag is kept on false.

    Here we have the semi-smooth Newton case
    with one combined iteration loop for active set search and large
    deformations. As a consequence this method is called AFTER each
    (not yet converged) Newton step. If there is a change in the active
    set or the residual and displacement increment norm are still above their limits,
    another Newton step has to be performed.

    \note We use the flag \c firstStepPredictor to overwrite the active set status
    for each node in the predictor of the first time step (or the first time step
    after a restart).

    \param[in] firstStepPredictor Boolean flag to indicate the predictor step in the first time step
    */
    void update_active_set_semi_smooth(const bool firstStepPredictor = false) override;

    /*!
    \brief Reset active set status for next time step

    */
    void reset_active_set() override
    {
      activesetssconv_ = false;
      activesetconv_ = false;
      activesetsteps_ = 1;
      return;
    }

    /*!
    \brief Return matrix T

    */
    std::shared_ptr<const Core::LinAlg::SparseMatrix> t_matrix() const override { return tmatrix_; }

    //@}

    //! @name Debugging and visualization methods

    /*!
    \brief Check linear and angular momentum conservation

    */
    void check_conservation_laws(
        const Core::LinAlg::Vector<double>& fs, const Core::LinAlg::Vector<double>& fm);

    /*!
    \brief Saving reference state is required for penalty support (LTL)

    */
    void save_reference_state(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;
    //@}

    //! @name Empty methods

    /*!
    \brief Empty methods only relevant for other strategies

    For a Lagrange strategy these are functions without functionality.
    Call them whenever you like.

    */
    double constraint_norm() const override { return 0.0; }
    void predict_relative_movement() override {}
    double initial_penalty() const override { return 0.0; }
    void initialize_uzawa(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) override
    {
    }
    void reset_penalty() override {}
    void modify_penalty() override {}
    void update_uzawa_augmented_lagrange() override {}
    void update_constraint_norm(int uzawaiter = 0) override {}
    bool is_penalty() const override { return false; };

   protected:
    //! derived
    std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces() override { return interface_; }

    //! derived
    const std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces() const override
    {
      return interface_;
    }

    /*!
    \brief Initialize general contact variables for next Newton step

    For a lagrangian strategy this includes the global normal / tangent matrices N and T,
    the global derivative matrices S and P and Tresca friction matrix L + vector r.

    */
    void initialize() override;

    /*!
    \brief Evaluate contact

    For a lagrangian strategy this involves heavy modification to the initial kteff and feff.
    Hence, they are in fact build from scratch here.
    The application of modifications to groups of dofs (slave, master, active etc.)
    results in some matrix and vector splitting and a lot of matrix-vector calculation in here!

    */
    void evaluate_contact(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) override;

    /*!
    \brief Evaluate frictional contact

    */
    void evaluate_friction(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) override;

    void update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;


   protected:
    /*!
    \brief Add penalty terms for LTL edge contact

    */
    void add_line_to_lin_contributions(Core::LinAlg::SparseOperator& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff, bool add_time_integration = true);

    /*!
    \brief Add penalty terms for master contact

    */
    void add_master_contributions(Core::LinAlg::SparseOperator& kteff,
        Core::LinAlg::Vector<double>& feff, bool add_time_integration = true);

    /*!
    \brief Add penalty terms for LTL edge contact

    */
    void add_line_to_lin_contributions_friction(Core::LinAlg::SparseOperator& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff, bool add_time_integration = true);

    // don't want = operator and cctor
    LagrangeStrategy operator=(const LagrangeStrategy& old) = delete;
    LagrangeStrategy(const LagrangeStrategy& old) = delete;

    // Store Coupling Matrices in case of Poro Lagrange Strategy ... here just ignore!
    virtual void save_coupling_matrices(std::shared_ptr<Core::LinAlg::SparseMatrix> dhat,
        std::shared_ptr<Core::LinAlg::SparseMatrix> mhataam,
        std::shared_ptr<Core::LinAlg::SparseMatrix> invda)
    {
      return;
    }

    std::vector<std::shared_ptr<CONTACT::Interface>> interface_;

    bool evalForceCalled_;  //< flag for evaluate force call
    bool activesetssconv_;  //< convergence flag for semi-smooth active set search
    bool activesetconv_;    //< convergence flag for fixed-point active set search
    int activesetsteps_;    //< number of fixed-point active set steps in this time step

    std::shared_ptr<Core::LinAlg::SparseMatrix>
        mhatmatrix_;  //< product of global Mortar matrices inv(D)*M

    std::shared_ptr<Core::LinAlg::SparseMatrix>
        tmatrix_;  //< global Matrix T containing active node tangents
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        nmatrix_;  //< global Matrix N containing active node normals

    std::shared_ptr<Core::LinAlg::SparseMatrix>
        smatrix_;  //< global Matrix S containing normal+D+M derivatives
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        smatrixW_;  //< global Matrix S containing W derivatives
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        tderivmatrix_;  //< global Matrix containing tangent derivatives
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        nderivmatrix_;  //< global Matrix containing normal derivatives


    std::shared_ptr<Core::LinAlg::Vector<double>>
        fs_;  //< slave side effective forces (needed for LM)
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        invd_;  //< inverse of Mortar matrix D (needed for LM)
    std::shared_ptr<Core::LinAlg::SparseMatrix> ksn_;  //< stiffness block K_sn (needed for LM)
    std::shared_ptr<Core::LinAlg::SparseMatrix> ksm_;  //< stiffness block K_sm (needed for LM)
    std::shared_ptr<Core::LinAlg::SparseMatrix> kss_;  //< stiffness block K_ss (needed for LM)

    std::shared_ptr<Core::LinAlg::SparseMatrix>
        linslipLM_;  //< global matrix containing derivatives (LM) of slip condition
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        linslipDIS_;  //< global matrix containing derivatives (DIS) of slip condition
    std::shared_ptr<Core::LinAlg::Vector<double>> linslipRHS_;  //< r.h.s vector friction slip nodes
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        linstickLM_;  //< global matrix containing derivatives (LM) of slip condition
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        linstickDIS_;  //< global matrix containing derivatives (DIS) of stick condition
    std::shared_ptr<Core::LinAlg::Vector<double>>
        linstickRHS_;  //< r.h.s vector for friction stick condition

    std::shared_ptr<Core::LinAlg::Map> zigzagone_;  //< active node set of last active set try
    std::shared_ptr<Core::LinAlg::Map>
        zigzagtwo_;  //< active node set of second-last active set try
    std::shared_ptr<Core::LinAlg::Map>
        zigzagthree_;  //< active node set of third-last active set try

    std::shared_ptr<Core::LinAlg::Map>
        gOldActiveSlaveNodes_;  // active slave nodes from previous newton step

    std::shared_ptr<Core::LinAlg::Map>
        gOldslipnodes_;  // active slave nodes from previous newton step

    std::shared_ptr<Core::LinAlg::Vector<double>> fLTLOld_;  //< old line to line forces
    std::shared_ptr<Core::LinAlg::Vector<double>> fLTL_;   //< current line to line forces combined
    std::shared_ptr<Core::LinAlg::Vector<double>> fLTLn_;  //< current line to line forces normal
    std::shared_ptr<Core::LinAlg::Vector<double>> fLTLt_;  //< current line to line forces tangent
    std::shared_ptr<Core::LinAlg::Vector<double>>
        fconservation_;  //< current line to line forces normal

    std::shared_ptr<Core::LinAlg::Vector<double>>
        nonsmooth_Penalty_force_;  //< penalty forces of non-smooth contact
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        nonsmooth_Penalty_stiff_;  //< tangent to penalty forces of non-smooth contact

  };  // class LagrangeStrategy
}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif
