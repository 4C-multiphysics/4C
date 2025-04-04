// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_STR_WRAPPER_HPP
#define FOUR_C_ADAPTER_STR_WRAPPER_HPP

#include "4C_config.hpp"

#include "4C_adapter_str_structure.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  /// This class is a wrapper of Adapter::Structure. It follows the "decorator" design pattern. This
  /// approach allows to dynamically add functionalities to an instance of Adapter::Structure. For
  /// example, Adapter::StructureTimeLoop implements the Integrate function for sequential time
  /// marching.
  class StructureWrapper : public Structure
  {
   public:
    /// constructor
    explicit StructureWrapper(std::shared_ptr<Structure> structure)
        : structure_(std::move(structure))
    {
    }

    //! @name Construction
    //@{

    /*! \brief Setup all class internal objects and members

     setup() is not supposed to have any input arguments !

     Must only be called after init().

     Construct all objects depending on the parallel distribution and
     relying on valid maps like, e.g. the state vectors, system matrices, etc.

     Call all setup() routines on previously initialized internal objects and members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, e.g. vectors may have wrong maps.

    \warning none
    \return void

    */
    void setup() override { structure_->setup(); };

    //@}

    /*!
     * @brief Perform all necessary tasks after setting up structure
     * wrapper. Currently, the method only calls the post setup routine of the
     * underlying structural time integration.
     */
    void post_setup() override { structure_->post_setup(); }

    //! @name Vector access
    //@{

    /// initial guess of Newton's method
    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() override
    {
      return structure_->initial_guess();
    }

    /// right-hand-side of Newton's method
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override { return structure_->rhs(); }

    /// unknown displacements at \f$t_{n+1}\f$
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() const override
    {
      return structure_->dispnp();
    }

    /// known displacements at \f$t_{n}\f$
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() const override
    {
      return structure_->dispn();
    }

    /// unknown velocity at \f$t_{n+1}\f$
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> velnp() const override
    {
      return structure_->velnp();
    }

    /// known velocity at \f$t_{n}\f$
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> veln() const override
    {
      return structure_->veln();
    }

    /// known velocity at \f$t_{n-1}\f$
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> velnm() const override
    {
      return structure_->velnm();
    }

    /// unknown acceleration at \f$t_{n+1}\f$
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> accnp() const override
    {
      return structure_->accnp();
    }

    /// known acceleration at \f$t_{n}\f$
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> accn() const override
    {
      return structure_->accn();
    }

    //@}


    //! @name Misc
    //@{

    /// dof map of vector of unknowns
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() override
    {
      return structure_->dof_row_map();
    }

    /// dof map of vector of unknowns for multiple dof sets
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map(unsigned nds) override
    {
      return structure_->dof_row_map(nds);
    }

    /// view of dof map of vector of vector of unknowns
    const Core::LinAlg::Map* dof_row_map_view() override { return structure_->dof_row_map_view(); }

    /// domain map of system matrix
    [[nodiscard]] const Core::LinAlg::Map& domain_map() const override
    {
      return structure_->domain_map();
    }

    /// direct access to system matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override
    {
      return structure_->system_matrix();
    }

    /// direct access to system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override
    {
      return structure_->block_system_matrix();
    }

    /// switch structure field to block matrix
    void use_block_matrix(std::shared_ptr<const Core::LinAlg::MultiMapExtractor> domainmaps,
        std::shared_ptr<const Core::LinAlg::MultiMapExtractor> rangemaps) override
    {
      structure_->use_block_matrix(domainmaps, rangemaps);
    }

    // access to contact/meshtying bridge
    std::shared_ptr<CONTACT::MeshtyingContactBridge> meshtying_contact_bridge() override
    {
      return structure_->meshtying_contact_bridge();
    }

    // access to locsys manager
    std::shared_ptr<Core::Conditions::LocsysManager> locsys_manager() override
    {
      return structure_->locsys_manager();
    }

    /// direct access to discretization
    std::shared_ptr<Core::FE::Discretization> discretization() override
    {
      return structure_->discretization();
    }

    /// read only access to discretization
    [[nodiscard]] virtual std::shared_ptr<const Core::FE::Discretization> get_discretization() const
    {
      return structure_->discretization();
    }

    /// are there any algebraic constraints?
    bool have_constraint() override { return structure_->have_constraint(); }

    /// are there any spring dashpot BCs?
    bool have_spring_dashpot() override { return structure_->have_spring_dashpot(); }

    /// get constraint manager defined in the structure
    std::shared_ptr<CONSTRAINTS::ConstrManager> get_constraint_manager() override
    {
      return structure_->get_constraint_manager();
    }

    /// get constraint manager defined in the structure
    std::shared_ptr<CONSTRAINTS::SpringDashpotManager> get_spring_dashpot_manager() override
    {
      return structure_->get_spring_dashpot_manager();
    }

    /// get type of thickness scaling for thin shell structures
    Inpar::Solid::StcScale get_stc_algo() override { return structure_->get_stc_algo(); }

    /// access to scaling matrix for STC
    std::shared_ptr<Core::LinAlg::SparseMatrix> get_stc_mat() override
    {
      return structure_->get_stc_mat();
    }

    /// Return MapExtractor for Dirichlet boundary conditions
    std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() override
    {
      return structure_->get_dbc_map_extractor();
    }

    /// expand dirichlet bc map
    void add_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoadd) override
    {
      structure_->add_dirich_dofs(maptoadd);
    };

    /// contract dirichlet bc map
    void remove_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoremove) override
    {
      structure_->remove_dirich_dofs(maptoremove);
    };

    /// reset step and state vectors
    void reset() override { structure_->reset(); }

    /// reset last time step, needed for time step size adaptivity in FSI
    void reset_step() override { structure_->reset_step(); }

    //@}


    /// @name Time step helpers
    //@{

    /// return time integration factor
    [[nodiscard]] double tim_int_param() const override { return structure_->tim_int_param(); }

    //! Sets the current time \f$t_{n}\f$
    void set_time(const double time) override { structure_->set_time(time); }

    //! Sets the target time \f$t_{n+1}\f$ of this time step
    void set_timen(const double time) override { structure_->set_timen(time); }

    //! Sets the target step \f$n\f$
    void set_step(int step) override { structure_->set_step(step); }

    //! Sets the target step \f$n+1\f$
    void set_stepn(int step) override { structure_->set_stepn(step); }

    //! Return current time \f$t_{n}\f$
    [[nodiscard]] double time_old() const override { return structure_->time_old(); }

    //! Return target time \f$t_{n+1}\f$
    [[nodiscard]] double time() const override { return structure_->time(); }

    /// get upper limit of time range of interest
    [[nodiscard]] double get_time_end() const override { return structure_->get_time_end(); }

    //! Set upper limit of time range of interest //HACK for parameter continuation
    void set_time_end(double timemax) override { structure_->set_time_end(timemax); }

    /// get time step size \f$\Delta t_n\f$
    [[nodiscard]] double dt() const override { return structure_->dt(); }

    /// Return current step number $n$
    [[nodiscard]] int step_old() const override { return structure_->step_old(); }

    /// Return current step number $n+1$
    [[nodiscard]] int step() const override { return structure_->step(); }

    /// get number of time steps
    [[nodiscard]] int num_step() const override { return structure_->num_step(); }

    /// integrate from t1 to t2
    int integrate() override { return structure_->integrate(); }

    //! do something in case nonlinear solution does not converge for some reason
    Inpar::Solid::ConvergenceStatus perform_error_action(
        Inpar::Solid::ConvergenceStatus nonlinsoldiv) override
    {
      return structure_->perform_error_action(nonlinsoldiv);
    }

    /// tests if there are more time steps to do
    [[nodiscard]] bool not_finished() const override { return structure_->not_finished(); }

    /// set time step size
    void set_dt(const double dtnew) override { structure_->set_dt(dtnew); }

    /// start new time step
    void prepare_time_step() override { structure_->prepare_time_step(); }

    /// update displacement
    void update_state_incrementally(
        std::shared_ptr<const Core::LinAlg::Vector<double>> disi  ///< iterative solution increment
        ) override
    {
      structure_->update_state_incrementally(disi);
    }

    void determine_stress_strain() override { structure_->determine_stress_strain(); }

    /// update displacement and evaluate elements (implicit only)
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc) override
    {
      structure_->evaluate(disiterinc);
    }

    /// don't update displacement but evaluate elements (implicit only)
    void evaluate() override { structure_->evaluate(); }

    /// update at time step end
    void update() override { structure_->update(); }

    /// update at time step end
    void update(const double endtime) override { structure_->update(endtime); }

    /// resize MStep objects for AB2
    void resize_m_step_tim_ada() override { structure_->resize_m_step_tim_ada(); }

    /// update iteration; add residual increment to Lagrange multipliers stored in Constraint
    /// manager
    void update_iter_incr_constr(std::shared_ptr<Core::LinAlg::Vector<double>> lagrincr) override
    {
      structure_->update_iter_incr_constr(lagrincr);
    }

    /// update iteration; add residual increment to pressures stored in 0D cardiovascular manager
    void update_iter_incr_cardiovascular0_d(
        std::shared_ptr<Core::LinAlg::Vector<double>> presincr) override
    {
      structure_->update_iter_incr_cardiovascular0_d(presincr);
    }

    /// access to output object
    std::shared_ptr<Core::IO::DiscretizationWriter> disc_writer() override
    {
      return structure_->disc_writer();
    }

    /// prepare output (i.e. calculate stresses, strains, energies)
    void prepare_output(bool force_prepare) override { structure_->prepare_output(force_prepare); }

    /// Get restart data
    void get_restart_data(std::shared_ptr<int> step, std::shared_ptr<double> time,
        std::shared_ptr<Core::LinAlg::Vector<double>> disn,
        std::shared_ptr<Core::LinAlg::Vector<double>> veln,
        std::shared_ptr<Core::LinAlg::Vector<double>> accn,
        std::shared_ptr<std::vector<char>> elementdata,
        std::shared_ptr<std::vector<char>> nodedata) override
    {
      structure_->get_restart_data(step, time, disn, veln, accn, elementdata, nodedata);
    }

    /// output results
    void output(bool forced_writerestart = false) override
    {
      structure_->output(forced_writerestart);
    }

    /// Write Gmsh output for structural field
    void write_gmsh_struct_output_step() override { structure_->write_gmsh_struct_output_step(); }

    /// output results to screen
    void print_step() override { structure_->print_step(); }

    /// read restart information for given time step
    void read_restart(const int step) override { structure_->read_restart(step); }

    /// set restart information for parameter continuation
    void set_restart(int step, double time, std::shared_ptr<Core::LinAlg::Vector<double>> disn,
        std::shared_ptr<Core::LinAlg::Vector<double>> veln,
        std::shared_ptr<Core::LinAlg::Vector<double>> accn,
        std::shared_ptr<std::vector<char>> elementdata,
        std::shared_ptr<std::vector<char>> nodedata) override
    {
      structure_->set_restart(step, time, disn, veln, accn, elementdata, nodedata);
    }

    /// set the state of the nox group and the global state data container (implicit only)
    void set_state(const std::shared_ptr<Core::LinAlg::Vector<double>>& x) override
    {
      structure_->set_state(x);
    }

    /// set evaluation action
    void set_action_type(const Core::Elements::ActionType& action) override
    {
      structure_->set_action_type(action);
    }

    /// wrapper for things that should be done before prepare_time_step is called
    void pre_predict() override { structure_->pre_predict(); }

    /// wrapper for things that should be done before solving the nonlinear iterations
    void pre_solve() override { structure_->pre_solve(); }

    /// wrapper for things that should be done before updating
    void pre_update() override { structure_->pre_update(); }

    /// wrapper for things that should be done after solving the update
    void post_update() override { structure_->post_update(); }

    /// wrapper for things that should be done after the output
    void post_output() override { structure_->post_output(); }

    /// wrapper for things that should be done after the actual time loop is finished
    void post_time_loop() override { structure_->post_time_loop(); }

    //@}


    //! @name Solver calls
    //@{

    /// nonlinear solve
    Inpar::Solid::ConvergenceStatus solve() override { return structure_->solve(); }

    //! linear structure solve with just an interface load
    std::shared_ptr<Core::LinAlg::Vector<double>> solve_relaxation_linear() override
    {
      return structure_->solve_relaxation_linear();
    }

    /// get the linear solver object used for this field
    std::shared_ptr<Core::LinAlg::Solver> linear_solver() override
    {
      return structure_->linear_solver();
    }

    //@}


    //! @name volume coupled specific methods
    //@{

    /// set forces due to interface with fluid, the force is expected external-force-like
    void set_force_interface(std::shared_ptr<Core::LinAlg::MultiVector<double>> iforce) override
    {
      structure_->set_force_interface(iforce);
    }

    //! specific method for iterative staggered partitioned TSI
    //! will be obsolete after switch to new structural timint.
    void prepare_partition_step() override { structure_->prepare_partition_step(); }

    //@}


    //! @name Write access to field solution variables at \f$t^{n+1}\f$
    //@{

    /// write access to extract displacements at \f$t^{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispnp() override
    {
      return structure_->write_access_dispnp();
    }

    /// write access to extract velocities at \f$t^{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_velnp() override
    {
      return structure_->write_access_velnp();
    }

    /// write access to extract displacements at \f$t^{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispn() override
    {
      return structure_->write_access_dispn();
    }

    /// write access to extract velocities at \f$t^{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_veln() override
    {
      return structure_->write_access_velnp();
    }

    //@}

    /// extract rhs (used to calculate reaction force for post-processing)
    std::shared_ptr<Core::LinAlg::Vector<double>> freact() override { return structure_->freact(); }

    /// create result test for encapsulated structure algorithm
    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override
    {
      return structure_->create_field_test();
    }

    //! @name Biofilm specific methods
    //@{

    void set_str_gr_disp(std::shared_ptr<Core::LinAlg::Vector<double>> struct_growth_disp) override
    {
      structure_->set_str_gr_disp(struct_growth_disp);
    }

    //@}

    /// do we have this model
    bool have_model(Inpar::Solid::ModelType model) override
    {
      return structure_->have_model(model);
    }

    /// return model evaluator
    Solid::ModelEvaluator::Generic& model_evaluator(Inpar::Solid::ModelType mtype) override
    {
      return structure_->model_evaluator(mtype);
    }

    [[nodiscard]] bool has_final_state_been_written() const override
    {
      return structure_->has_final_state_been_written();
    }

   protected:
    std::shared_ptr<Structure> structure_;  ///< underlying structural time integration
  };


  /// Calculate increments from absolute values
  class StructureNOXCorrectionWrapper : public StructureWrapper
  {
   public:
    explicit StructureNOXCorrectionWrapper(std::shared_ptr<Structure> structure)
        : StructureWrapper(structure)
    {
    }

    void prepare_time_step() override;

    //! evaluate() routine that can handle NOX step increments by computing the
    //! last iteration increment needed for structural evaluate() call
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> disstepinc) override;

   private:
    /// sum of displacement increments already applied,
    ///
    /// there are two increments around
    ///
    /// x^n+1_i+1 = x^n+1_i + disiterinc  (also referred to as residual increment)
    ///
    /// x^n+1_i+1 = x^n     + disstepinc
    std::shared_ptr<Core::LinAlg::Vector<double>> disstepinc_;
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
