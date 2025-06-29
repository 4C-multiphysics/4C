// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_MONOLITHIC_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_MONOLITHIC_HPP

#include "4C_config.hpp"

#include "4C_porofluid_pressure_based_elast_base.hpp"
#include "4C_porofluid_pressure_based_input.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;
  class SparseOperator;
  class MultiMapExtractor;
  class BlockSparseMatrixBase;
  class Solver;
  class Equilibration;
  enum class EquilibrationMethod;
}  // namespace Core::LinAlg

namespace Core::LinearSolver
{
  enum class SolverType;
}

namespace Core::Conditions
{
  class LocsysManager;
}

namespace PoroPressureBased
{
  //! Monolithic solution scheme for porofluid-elasticity problems
  class PorofluidElastMonolithicAlgorithm : public PorofluidElastAlgorithm
  {
   public:
    PorofluidElastMonolithicAlgorithm(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    /// initialization
    void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& algoparams, const Teuchos::ParameterList& structparams,
        const Teuchos::ParameterList& fluidparams, const std::string& struct_disname,
        const std::string& fluid_disname, bool isale, int nds_disp, int nds_vel,
        int nds_solidpressure, int ndsporofluid_scatra,
        const std::map<int, std::set<int>>* nearby_ele_pairs) override;

    /// setup
    void setup_system() override;

    /// time step of coupled problem
    void time_step() override;

    //! extractor to communicate between full monolithic map and block maps
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> extractor() const override
    {
      return blockrowdofmap_;
    }

    //! evaluate all fields at x^n+1 with x^n+1 = x_n + stepinc
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fx, const bool firstcall) override;

    //! update all fields after convergence (add increment on displacements and fluid primary
    //! variables) public for access from monolithic scatra problem
    void update_fields_after_convergence(std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx) override;

    //! access to monolithic rhs vector
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() const override { return rhs_; }

    //! access to monolithic block system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() const override
    {
      return systemmatrix_;
    }

    //! unique map of all dofs that should be constrained with DBC
    std::shared_ptr<const Core::LinAlg::Map> combined_dbc_map() const override
    {
      return combinedDBCMap_;
    };

   protected:
    //! Newton output to screen
    virtual void newton_output();

    //! Newton error check after loop
    virtual void newton_error_check();

    //! build the combined dirichletbcmap
    virtual void build_combined_dbc_map();

    //! full monolithic dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map();

    virtual void setup_rhs();

    virtual std::shared_ptr<Core::LinAlg::Vector<double>> setup_structure_partof_rhs();

    //! build block vector from field vectors, e.g. rhs, increment vector
    void setup_vector(Core::LinAlg::Vector<double>& f,  //!< vector of length of all dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            sv,  //!< vector containing only structural dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            fv  //!< vector containing only fluid dofs
    );

    //! extract the field vectors from a given composed vector x.
    /*!
     \param x  (i) composed vector that contains all field vectors
     \param sx (o) structural vector (e.g. displacements)
     \param fx (o) fluid vector (primary variables of fluid field, i.e. pressures or saturations,
     and 1D artery pressure)
     */
    virtual void extract_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> x,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx);

    //! extract only the structure and fluid field vectors from a given composed vector x.
    /*!
     \param x  (i) composed vector that contains all field vectors
     \param sx (o) structural vector (e.g. displacements)
     \param fx (o) fluid vector (primary variables of fluid field, i.e. pressures or saturations)
     */
    void extract_structure_and_fluid_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> x,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx);

    /// setup composed system matrix from field solvers
    virtual void setup_system_matrix() { setup_system_matrix(*systemmatrix_); }

    /// setup composed system matrix from field solvers
    virtual void setup_system_matrix(Core::LinAlg::BlockSparseMatrixBase& mat);

    /// setup composed system matrix from field solvers
    virtual void setup_maps();

    //! Setup solver for monolithic system
    bool setup_solver() override;

    //! build the block null spaces
    void build_block_null_spaces(std::shared_ptr<Core::LinAlg::Solver>& solver) override;

    //! Evaluate mechanical-fluid system matrix
    virtual void apply_str_coupl_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> k_sf  //!< mechanical-fluid stiffness matrix
    );

    //! Evaluate fluid-mechanical system matrix
    virtual void apply_fluid_coupl_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> k_fs  //!< fluid-mechanical tangent matrix
    );

    //! evaluate all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    virtual void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> iterinc);

    //! return structure fluid coupling sparse matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> struct_fluid_coupling_matrix();

    //! return fluid structure coupling sparse matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> fluid_struct_coupling_matrix();

    //! Solve the linear system of equations
    void linear_solve();

    //! Create the linear solver
    virtual void create_linear_solver(const Teuchos::ParameterList& solverparams,
        const Core::LinearSolver::SolverType solvertype);

    //! Setup Newton-Raphson
    void setup_newton();

    //! Print Header to screen
    void print_header() override;

    //! update all fields after convergence (add increment on displacements and fluid primary
    //! variables)
    void update_fields_after_convergence();

    //! build norms for convergence check
    virtual void build_convergence_norms();

    void poro_fd_check();

    // check for convergence
    bool converged();

    /// Print user output that structure field is disabled
    void print_structure_disabled_info();

    //! convergence tolerance for increments
    double ittolinc_;
    //! convergence tolerance for residuals
    double ittolres_;
    //! maximally permitted iterations
    int itmax_;
    //! minimally necessary iterations
    int itmin_;
    //! current iteration step
    int itnum_;
    //! @name Global vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;  //!< a zero vector of full length

    std::shared_ptr<Core::LinAlg::Vector<double>>
        iterinc_;  //!< increment between Newton steps k and k+1
    //!< \f$\Delta{x}^{<k>}_{n+1}\f$

    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_;  //!< rhs of Poroelasticity system

    std::shared_ptr<Core::LinAlg::Solver> solver_;  //!< linear algebraic solver
    double solveradaptolbetter_;                    //!< tolerance to which is adapted ?
    bool solveradapttol_;                           //!< adapt solver tolerance


    //@}

    //! @name Global matrixes

    //! block systemmatrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> systemmatrix_;

    //! structure-fluid coupling matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> k_sf_;
    //! fluid-structure coupling matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> k_fs_;

    //@}

    //! dof row map (not split)
    std::shared_ptr<Core::LinAlg::Map> fullmap_;

    //! dof row map split in (field) blocks
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockrowdofmap_;

    //! all equilibration of global system matrix and RHS is done in here
    std::shared_ptr<Core::LinAlg::Equilibration> equilibration_;

    //! equilibration method applied to system matrix
    Core::LinAlg::EquilibrationMethod equilibration_method_;

    //! dirichlet map of monolithic system
    std::shared_ptr<Core::LinAlg::Map> combinedDBCMap_;

    double tolinc_;   //!< tolerance residual increment
    double tolfres_;  //!< tolerance force residual

    double tolinc_struct_;   //!< tolerance residual increment for structure displacements
    double tolfres_struct_;  //!< tolerance force residual for structure displacements

    double tolinc_fluid_;   //!< tolerance residual increment for fluid
    double tolfres_fluid_;  //!< tolerance force residual for fluid

    double normrhs_;  //!< norm of residual forces

    double normrhsfluid_;  //!< norm of residual forces (fluid )
    double normincfluid_;  //!< norm of residual unknowns (fluid )

    double normrhsstruct_;  //!< norm of residual forces (structure)
    double normincstruct_;  //!< norm of residual unknowns (structure)

    double normrhsart_;       //!< norm of residual (artery)
    double normincart_;       //!< norm of residual unknowns (artery)
    double arterypressnorm_;  //!< norm of artery pressure

    double maxinc_;  //!< maximum increment
    double maxres_;  //!< maximum residual

    VectorNorm vectornormfres_;  //!< type of norm for residual
    VectorNorm vectornorminc_;   //!< type of norm for increments

    Teuchos::Time timernewton_;  //!< timer for measurement of solution time of newton iterations
    double dtsolve_;             //!< linear solver time
    double dtele_;               //!< time for element evaluation + build-up of system matrix

    //! Dirichlet BCs with local co-ordinate system
    std::shared_ptr<Core::Conditions::LocsysManager> locsysman_;

    //! flag for finite difference check
    bool fdcheck_;
  };
}  // namespace PoroPressureBased

FOUR_C_NAMESPACE_CLOSE

#endif
