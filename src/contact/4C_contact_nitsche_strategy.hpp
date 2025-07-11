// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_STRATEGY_HPP
#define FOUR_C_CONTACT_NITSCHE_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_contact_abstract_strategy.hpp"
#include "4C_contact_utils.hpp"
#include "4C_linalg_fevector.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  /*!
   \brief Contact solving strategy with Nitsche's method.

   This is a specialization of the abstract contact algorithm as defined in AbstractStrategy.
   For a more general documentation of the involved functions refer to CONTACT::AbstractStrategy.

   */
  class NitscheStrategy : public AbstractStrategy
  {
   public:
    //! Standard constructor
    NitscheStrategy(const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        const Teuchos::ParameterList& params,
        std::vector<std::shared_ptr<CONTACT::Interface>> interface, int dim, const MPI_Comm& comm,
        double alphaf, int maxdof)
        : AbstractStrategy(std::make_shared<CONTACT::AbstractStrategyDataContainer>(), dof_row_map,
              NodeRowMap, params, dim, comm, alphaf, maxdof),
          interface_(std::move(interface)),
          curr_state_eval_(false)
    { /* empty */
    }

    //! Shared data constructor
    NitscheStrategy(const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
        const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        const Teuchos::ParameterList& params,
        std::vector<std::shared_ptr<CONTACT::Interface>> interface, int dim, const MPI_Comm& comm,
        double alphaf, int maxdof)
        : AbstractStrategy(data_ptr, dof_row_map, NodeRowMap, params, dim, comm, alphaf, maxdof),
          interface_(std::move(interface)),
          curr_state_eval_(false)
    { /* empty */
    }

    // don't want = operator and cctor
    NitscheStrategy operator=(const NitscheStrategy& old) = delete;
    NitscheStrategy(const NitscheStrategy& old) = delete;

    void apply_force_stiff_cmt(std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
        std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
        bool predictor) override;

    void do_read_restart(Core::IO::DiscretizationReader& reader,
        std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<CONTACT::ParamsInterface> cparams_ptr) override;

    bool is_saddle_point_system() const override { return false; }

    bool is_condensed_system() const override { return false; }

    /*!
     * @brief Integrate all contact interfaces
     *
     * @note this method is called from the new structural time integration
     *
     * @param[in] cparams  contact data container
     */
    virtual void integrate(const CONTACT::ParamsInterface& cparams);

    std::shared_ptr<const Core::LinAlg::Vector<double>> get_rhs_block_ptr(
        const enum CONTACT::VecBlockType& bt) const override;

    std::shared_ptr<Core::LinAlg::SparseMatrix> get_matrix_block_ptr(
        const enum CONTACT::MatBlockType& bt, const ParamsInterface* cparams) const override;

    /*! \brief Setup this strategy object (maps, vectors, etc.)

     derived from contact abstract strategy.
     The Nitsche strategy does not have
      */
    void setup(bool redistributed, bool init) override;

    virtual void update_trace_ineq_etimates();

    /*! \brief Get dirichlet B.C. status and store into Nodes

     This is called once at the beginning of the simulation
     to set the D.B.C. status in each CNode.

     \param dbcmaps (in): MapExtractor carrying global dbc map */
    void store_dirichlet_status(
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmaps) override {
      /* we don't care about dirichlet for now */
    };
    void update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;
    void evaluate_reference_state() override;
    void do_write_restart(
        std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>>& restart_vectors,
        bool forcedrestart) const override {
      /* nothing stored in nitsche strategy that would need to be written */
    };
    void compute_contact_stresses() final { /* nothing stress output in nitsche strategy yet */ };
    virtual void reconnect_parent_elements();
    void set_state(
        const enum Mortar::StateType& statename, const Core::LinAlg::Vector<double>& vec) override;

    /*!
     * @brief  Set the parent state
     *
     * @param[in] statename  name of state to be set
     * @param[in] vec        corresponding state vector
     * @param[in] dis        corresponding discretization
     */
    void set_parent_state(const enum Mortar::StateType& statename,
        const Core::LinAlg::Vector<double>& vec, const Core::FE::Discretization& dis) override;

    std::shared_ptr<const Core::LinAlg::Vector<double>> lagrange_multiplier_n(
        const bool& redist) const override
    {
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> lagrange_multiplier_np(
        const bool& redist) const override
    {
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> lagrange_multiplier_old() const override
    {
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Map> lm_dof_row_map_ptr(const bool& redist) const override
    {
      return nullptr;
    }
    // All these functions only have functionality in Lagrange contact simulations,
    // thus they are defined empty here in the case of Penalty contact.

    //! Get the active node row map of the previous Newton step
    std::shared_ptr<const Core::LinAlg::Map> get_old_active_row_nodes() const override
    {
      return nullptr;
    };
    std::shared_ptr<const Core::LinAlg::Map> get_old_slip_row_nodes() const override
    {
      return nullptr;
    };
    bool is_nitsche() const override { return true; }
    void print_active_set() const override {};
    bool active_set_converged() const override { return true; }
    int active_set_steps() const override { return 0; }
    void reset_active_set() override {}
    void recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi) override {}
    void build_saddle_point_system(std::shared_ptr<Core::LinAlg::SparseOperator> kdd,
        std::shared_ptr<Core::LinAlg::Vector<double>> fd,
        std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps,
        std::shared_ptr<Epetra_Operator>& blockMat,
        std::shared_ptr<Core::LinAlg::Vector<double>>& blocksol,
        std::shared_ptr<Core::LinAlg::Vector<double>>& blockrhs) override
    {
      FOUR_C_THROW(
          "Nitsche does not have Lagrange multiplier DOFs. So, saddle point system makes no sense "
          "here.");
    }
    void update_displacements_and_l_mincrements(std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<const Core::LinAlg::Vector<double>> blocksol) override
    {
      FOUR_C_THROW(
          "Nitsche does not have Lagrange multiplier DOFs. So, saddle point system makes no sense "
          "here.");
    }
    void evaluate_constr_rhs() override {}
    void update_active_set() override {}
    void update_active_set_semi_smooth(const bool firstStepPredictor) override {}
    void predict_relative_movement() override {}
    void modify_penalty() override {}
    void update_uzawa_augmented_lagrange() override {}
    void update_constraint_norm(int uzawaiter) override {}
    void initialize() override {};
    void evaluate_contact(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) override
    {
      FOUR_C_THROW("not supported in this strategy");
    }
    void evaluate_friction(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) override
    {
      FOUR_C_THROW("not supported in this strategy");
    }
    void initialize_uzawa(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) override
    {
    }
    void reset_penalty() override {}
    void save_reference_state(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override {}
    double initial_penalty() const override { return 0.0; }
    double constraint_norm() const override { return 0.0; }
    bool is_penalty() const override { return false; }

   protected:
    std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces() override { return interface_; }

    const std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces() const override
    {
      return interface_;
    }

    void evaluate_force(CONTACT::ParamsInterface& cparams) override;

    void evaluate_force_stiff(CONTACT::ParamsInterface& cparams) override;

    void reset(const CONTACT::ParamsInterface& cparams, const Core::LinAlg::Vector<double>& dispnp,
        const Core::LinAlg::Vector<double>& xnew) override;

    void run_post_compute_x(const CONTACT::ParamsInterface& cparams,
        const Core::LinAlg::Vector<double>& xold, const Core::LinAlg::Vector<double>& dir,
        const Core::LinAlg::Vector<double>& xnew) override;

    /*!
     * @brief Fill RHS vector of vector block type
     *
     * @param[in] bt  vector block type
     * @return the filled RHS vector of given vector block type
     */
    virtual std::shared_ptr<Core::LinAlg::FEVector<double>> create_rhs_block_ptr(
        const enum CONTACT::VecBlockType& bt) const;

    /*!
     * @brief  Create an appropriate vector for the RHS
     *
     * @param[in] bt  block type
     * @return  vector for given vector block type
     */
    virtual std::shared_ptr<Core::LinAlg::FEVector<double>> setup_rhs_block_vec(
        const enum CONTACT::VecBlockType& bt) const;

    /*!
     * @brief Create appropriate matrix block
     *
     * @param[in] bt  matrix block type
     * @return matrix block for given matrix block type
     */
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> setup_matrix_block_ptr(
        const enum MatBlockType& bt);

    /*!
     * @brief Complete the matrix block with correct maps
     *
     * @param[in] bt      matrix block type
     * @param[in,out] kc  matrix block of given matrix block type that has to be completed
     */
    virtual void complete_matrix_block_ptr(
        const enum MatBlockType& bt, std::shared_ptr<Core::LinAlg::SparseMatrix> kc);

    /*!
     * @brief Fill block matrix of given matrix block type
     *
     * @param[in] bt  matrix block type
     * @return the filled block matrix of given matrix block type
     */
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> create_matrix_block_ptr(
        const enum MatBlockType& bt);

    std::vector<std::shared_ptr<CONTACT::Interface>> interface_;

    std::shared_ptr<Core::LinAlg::Vector<double>> curr_state_;
    bool curr_state_eval_;

    std::shared_ptr<Core::LinAlg::FEVector<double>> fc_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> kc_;
  };
}  // namespace CONTACT
FOUR_C_NAMESPACE_CLOSE

#endif
