// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_LAGRANGE_STRATEGY_WEAR_HPP
#define FOUR_C_CONTACT_LAGRANGE_STRATEGY_WEAR_HPP

#include "4C_config.hpp"

#include "4C_contact_lagrange_strategy.hpp"
#include "4C_linalg_fevector.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Wear
{
  // forward declarations
  class WearInterface;


  class LagrangeStrategyWear : public CONTACT::LagrangeStrategy
  {
   public:
    /*!
    \brief Standard Constructor

    */
    LagrangeStrategyWear(const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
        const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interfaces,
        int dim, MPI_Comm comm, double alphaf, int maxdof);


    /*!
    \brief Condense discr. wear and lm. for frictional contact

    */
    void condense_wear_discr(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff, Core::LinAlg::Vector<double>& gact);

    /*!
    \brief Condense lm. for frictional contact with explicit/implicit wear algorithm

    */
    void condense_wear_impl_expl(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff, Core::LinAlg::Vector<double>& gact);

    /*!
    \brief Prepare SaddlePointSystem

    */
    void prepare_saddle_point_system(
        Core::LinAlg::SparseOperator& kteff, Core::LinAlg::Vector<double>& feff);

    /*!
    \brief Recovery method

    We only recover the Lagrange multipliers here, which had been
    statically condensed during the setup of the global problem!
    Optionally satisfaction or violation of the contact boundary
    conditions can be checked, too.

    */
    void recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi) override;

    /*!
    \brief Redistribute all contact interfaces in parallel

    We hand in the current global displacement state so that a contact search can be performed and
    set state called.

    The current velocity state is required in case of extedning the ghosting via binning to account
    for relative motion between interfaces.

    \param[in] dis Current displacement state
    \param[in] vel Current velocity state

    \return TRUE if the interface has been redistributed. Return FALSE otherwise.
    */
    bool redistribute_contact(std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel) final;

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

    /*!
      \brief Reset the wear vector
        */
    void reset_wear() override;

    /*!
    \brief Evaluate wear vector

    Evaluates the unweighted wear vector.
    Refer also to the Semesterarbeit of Karl Wichmann 2010

    \warning In case of weighted wear, this is only implemented for dual/Petrov-Galerkin shape
    functions.

    \note This requires to solve for the real wear where we need to invert the slave side mortar
    matrix. Since we restrict ourselves to dual/Petrov-Galerkin shape functions, we can exploit
    the diagonality of the #dmatrix_. Hence, instead of solving we just divide by the diagonal
    elements of #dmatrix_.

    */
    void output_wear() override;


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
        std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

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
    set or the residual and disp norm are still above their limits,
    another Newton step has to be performed.

    \note We use the flag \c firstStepPredictor to overwrite the active set status
    for each node in the predictor of the first time step

    \param[in] firstStepPredictor Boolean flag to indicate the predictor step in the first time step
    */
    void update_active_set_semi_smooth(const bool firstStepPredictor = false) override;

    /*!
    \brief Store/Reset nodal wear quantities for pv approach

    */
    void update_wear_discret_iterate(bool store);

    /*!
    \brief Store wear for accumulation due to different pseudo time scales!

    */
    void update_wear_discret_accumulation();

    /*!
    \brief Update wear contact at end of time step

    */
    void update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

    /*!
    \brief Store wear data into wear data container

    */
    void store_nodal_quantities(Mortar::StrategyBase::QuantityType type) override;

    /*!
    \brief Return vector of wear (t_n+1) - D^-1 \times weighted wear!

    */
    std::shared_ptr<const Core::LinAlg::Vector<double>> contact_wear() const override
    {
      return wearoutput_;
    }  // for slave side

    std::shared_ptr<const Core::LinAlg::Vector<double>> contact_wear2() const
    {
      return wearoutput2_;
    }  // for master side

    /*!
    \brief Return wear interfaces

    */
    std::vector<std::shared_ptr<Wear::WearInterface>> wear_interfaces() { return interface_; }

    /*!
    \brief Return master map for both sided wear (slip), mapped from slave side

    */
    std::shared_ptr<const Core::LinAlg::Map> master_slip_nodes() const override
    {
      return gmslipnodes_;
    };

    /*!
    \brief Return master map for both sided wear (active), mapped from slave side

    */
    std::shared_ptr<const Core::LinAlg::Map> master_active_nodes() const override
    {
      return gmactivenodes_;
    };

    /*!
     \brief Return discrete wear vector (t_n+1)

     */
    std::shared_ptr<const Core::LinAlg::Vector<double>> wear_var() const { return w_; }

    /*!
     \brief Return discrete wear vector (t_n+1) Master

     */
    std::shared_ptr<const Core::LinAlg::Vector<double>> wear_var_m() const { return wm_; }

    /*!
     \brief Return wear rhs vector (only in saddle-point formulation

     */
    std::shared_ptr<const Core::LinAlg::Vector<double>> wear_rhs() const override
    {
      return wearrhs_;
    }

    /*!
     \brief Return wear-master rhs vector (only in saddle-point formulation

     */
    std::shared_ptr<const Core::LinAlg::Vector<double>> wear_m_rhs() const override
    {
      return wearmrhs_;
    }

    /*!
     \brief Returns increment of W solution vector in SaddlePointSolve routine

     */
    std::shared_ptr<const Core::LinAlg::Vector<double>> w_solve_incr() const override
    {
      return wincr_;
    }

    /*!
     \brief Returns increment of W-master solution vector in SaddlePointSolve routine

     */
    std::shared_ptr<const Core::LinAlg::Vector<double>> wm_solve_incr() const override
    {
      return wmincr_;
    }

    /*!
     \brief Return global both sided wear status

     */
    bool wear_both_discrete() const override { return wbothpv_; }

    /*!
     \brief Return global wear status

     */
    bool weighted_wear() const override { return weightedwear_; }

   private:
    /*!
    \brief Evaluate frictional contact

    */
    void evaluate_friction(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) override;

    /*!
    \brief Initialize and evaluate Mortar stuff for the next Newton step

    This method first checks if we are dealing with self contact and updates
    the interface slave and master sets if so. Then it resets the global
    Mortar matrices D and M and the global gap vector g accordingly.

    The nodal quantities computed in initialize_and_evaluate_interface() are then assembled
    to global matrices and vectors respectively. No setup of the global system
    is to be done here yet, so there is no need to pass in the effective
    stiffness K or the effective load vector f.

    */
    void initialize_mortar() override;
    void assemble_mortar() override;

    /*!
    \brief Initialize general contact variables for next Newton step

    For a lagrangian strategy this includes the global normal / tangent matrices N and T,
    the global derivative matrices S and P and Tresca friction matrix L + vector r.

    */
    void initialize() override;

    /*!
    \brief Setup this strategy object (maps, vectors, etc.)

    All global maps and vectors are initialized by collecting
    the necessary information from all interfaces. In the case
    of a parallel redistribution, this method is called again
    to re-setup the above mentioned quantities. In this case
    we set the input parameter redistributed=TRUE. Moreover,
    when called for the first time (in the constructor) this
    method is given the input parameter init=TRUE to account
    for initialization of the active set.
      */
    void setup(bool redistributed, bool init) override;

    /*!
    \brief Setup this strategy object (maps, vectors, etc.)

    All wear specific maps here
    */
    void setup_wear(bool redistributed, bool init);

   private:
    // don't want = operator and cctor
    LagrangeStrategyWear operator=(const LagrangeStrategyWear& old) = delete;
    LagrangeStrategyWear(const LagrangeStrategyWear& old) = delete;

    std::vector<std::shared_ptr<Wear::WearInterface>> interface_;

    // basic data
    bool weightedwear_;  // flag for contact with wear (is) --> weighted wear
    bool wbothpv_;       // flag for both sided wear discrete
    std::shared_ptr<Core::LinAlg::Vector<double>> w_;  // current vector of pv wear at t_n+1 (slave)
    std::shared_ptr<Core::LinAlg::Vector<double>>
        wincr_;  // Wear variables vector increment within SaddlePointSolve (this is NOT the
                 // increment of w_ between t_{n+1} and t_{n}!)
    std::shared_ptr<Core::LinAlg::Vector<double>> wearrhs_;

    std::shared_ptr<Core::LinAlg::Vector<double>>
        wm_;  // current vector of pv wear at t_n+1 (master)
    std::shared_ptr<Core::LinAlg::Vector<double>>
        wmincr_;  // Wear variables vector increment within SaddlePointSolve (this is NOT the
                  // increment of w_ between t_{n+1} and t_{n}!)
    std::shared_ptr<Core::LinAlg::Vector<double>> wearmrhs_;

    // implicit wear algorithm
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        wlinmatrix_;  // global Matrix Wg containing wear-lm derivatives
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        wlinmatrixsl_;  // global Matrix Wsl containing wear-lm slip derivatives
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        wlinmatrixst_;  // global Matrix Wst containing wear-lm stick derivatives

    // both-sided wear weak dirich cond
    std::shared_ptr<Core::LinAlg::SparseMatrix> d2matrix_;  // global Mortar matrix D2

    std::shared_ptr<Core::LinAlg::Map>
        gminvolvednodes_;  // global involved master node row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map>
        gminvolveddofs_;  // global involved master dof row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map>
        gslipn_;  // global row map of matrix N for slip dofs (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map>
        gwinact_;  // global row map of matrix N for slip dofs (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map>
        gmslipn_;  // global row map of matrix N for slip dofs (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map>
        gwminact_;  // global row map of matrix N for slip dofs (of all interfaces)

    std::shared_ptr<Core::LinAlg::Map>
        gwmdofrowmap_;  // global master wear dof row map (of all interfaces) -active
    std::shared_ptr<Core::LinAlg::Map>
        gwdofrowmap_;  // global slave wear dof row map (of all interfaces) -active
    std::shared_ptr<Core::LinAlg::Map>
        gsdofnrowmap_;  // global slave wear dof row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map>
        gmdofnrowmap_;  // global master wear dof row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map>
        galldofnrowmap_;  // global master wear dof row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map> gwalldofrowmap_;  // all
    std::shared_ptr<Core::LinAlg::Map> gmslipnodes_;     // global master slip nodes
    std::shared_ptr<Core::LinAlg::Map> gmactivenodes_;   // global master active nodes

    std::shared_ptr<Core::LinAlg::Vector<double>>
        wearoutput_;  // vector of unweighted wear at t_n+1  -- slave
    std::shared_ptr<Core::LinAlg::Vector<double>>
        wearoutput2_;  // vector of unweighted wear at t_n+1  -- master
    std::shared_ptr<Core::LinAlg::Vector<double>> wearvector_;  // global weighted wear vector w

    int maxdofwear_;  // highest dof number in problem discretization

    bool wearimpl_;        // weartype: implicit
    bool wearprimvar_;     // bool for wear with own discretization
    bool wearbothpv_;      // bool for both-sided discrete wear
    bool weartimescales_;  // bool for different time scales
    bool sswear_;          // bool steady state wear

    // discrete wear algorithm (SLAVE)
    std::shared_ptr<Core::LinAlg::SparseMatrix> twmatrix_;  // global Mortar wear matrix T
    std::shared_ptr<Core::LinAlg::SparseMatrix> ematrix_;   // global Mortar wear matrix E
    std::shared_ptr<Core::LinAlg::SparseMatrix> eref_;      // global Mortar wear matrix E
    std::shared_ptr<Core::LinAlg::SparseMatrix> lintdis_;   // Lin T w.r.t. displ: Lin(T*n*lm)
    std::shared_ptr<Core::LinAlg::SparseMatrix> lintlm_;    // Lin T w.r.t. lm: (T*n)
    std::shared_ptr<Core::LinAlg::SparseMatrix> linedis_;   // Lin E w.r.t. displ: Lin(E*w)
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        linslip_w_;  // global matrix containing derivatives (LM) of slip condition
    std::shared_ptr<Core::LinAlg::Vector<double>> inactive_wear_rhs_;  // inactive wear rhs: -w_i
    std::shared_ptr<Core::LinAlg::Vector<double>>
        wear_cond_rhs_;  // rhs wear condition: -E*w_i + k*T*n*lm_i

    // discrete wear algorithm (MASTER)
    std::shared_ptr<Core::LinAlg::SparseMatrix> twmatrix_m_;  // global Mortar wear matrix T
    std::shared_ptr<Core::LinAlg::SparseMatrix> ematrix_m_;   // global Mortar wear matrix E
    std::shared_ptr<Core::LinAlg::SparseMatrix> lintdis_m_;   // Lin T w.r.t. displ: Lin(T*n*lm)
    std::shared_ptr<Core::LinAlg::SparseMatrix> lintlm_m_;    // Lin T w.r.t. lm: (T*n)
    std::shared_ptr<Core::LinAlg::SparseMatrix> linedis_m_;   // Lin E w.r.t. displ: Lin(E*w)
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        linslip_wm_;  // global matrix containing derivatives (LM) of slip condition
    std::shared_ptr<Core::LinAlg::FEVector<double>>
        inactive_wear_rhs_m_;  // inactive wear rhs: -w_i
    std::shared_ptr<Core::LinAlg::FEVector<double>>
        wear_cond_rhs_m_;  // rhs wear condition: -E*w_i + k*T*n*lm_i

    // matrix blocks for recovering
    std::shared_ptr<Core::LinAlg::SparseMatrix> dnblock_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> dmblock_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> diblock_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> dablock_;
    std::shared_ptr<Core::LinAlg::Vector<double>> fw_;

    std::shared_ptr<Core::LinAlg::Map> gidofs_;

  };  // class

}  // namespace Wear

FOUR_C_NAMESPACE_CLOSE

#endif
