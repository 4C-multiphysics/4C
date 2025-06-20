// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_MONOLITHIC_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_MONOLITHIC_HPP

#include "4C_config.hpp"

#include "4C_inpar_solver.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_base.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_utils.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
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

namespace PoroPressureBased
{
  //! Monolithic solution scheme of porofluid-elasticity problems with scalar transport
  class PorofluidElastScatraMonolithicAlgorithm : public PorofluidElastScatraBaseAlgorithm
  {
   public:
    PorofluidElastScatraMonolithicAlgorithm(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    /// initialization
    void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& algoparams, const Teuchos::ParameterList& poroparams,
        const Teuchos::ParameterList& structparams, const Teuchos::ParameterList& fluidparams,
        const Teuchos::ParameterList& scatraparams, const std::string& struct_disname,
        const std::string& fluid_disname, const std::string& scatra_disname, bool isale,
        int nds_disp, int nds_vel, int nds_solidpressure, int ndsporofluid_scatra,
        const std::map<int, std::set<int>>* nearby_ele_pairs) override;

    /// setup
    void setup_system() override;

    /// setup solver (only needed for poromultiphase monolithic coupling)
    void setup_solver() override;

    /// time step of coupled problem
    void time_step() override;

    //! extractor to communicate between full monolithic map and block maps
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> extractor() const
    {
      return blockrowdofmap_;
    }

    //! unique map of all dofs that should be constrained with DBC
    std::shared_ptr<const Core::LinAlg::Map> combined_dbc_map() const { return combinedDBCMap_; };


   protected:
    //! Setup Newton-Raphson iteration
    void setup_newton();

    //! Setup full map
    virtual void setup_maps();

    //! Setup monolithic rhs-vector
    virtual void setup_rhs();

    //! build the combined dirichletbcmap
    virtual void build_combined_dbc_map();

    //! build the block null spaces
    virtual void build_block_null_spaces();

    //! create the linear solver
    void create_linear_solver(const Teuchos::ParameterList& solverparams,
        const Core::LinearSolver::SolverType solvertype);

    //! full monolithic dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map();

    //! evaluate all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    virtual void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> iterinc);

    //! extract the field vectors from a given composed vector x.
    /*!
     \param x   (i) composed vector that contains all field vectors
     \param stx (o) structural vector (e.g. displacements)
     \param flx (o) fluid vector (primary variables of fluid field, i.e. pressures or saturations)
     and pressures of artery network
     \param scx (o) scatra vector (primary variables of scatra field, i.e. mass fraction)
      and mass fractions in 1D artery network
     */
    virtual void extract_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> x,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& stx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& flx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& scx);

    //! extract only the 3D field vectors from a given composed vector x.
    /*!
     \param x   (i) composed vector that contains all field vectors
     \param stx (o) structural vector (e.g. displacements)
     \param flx (o) fluid vector (primary variables of fluid field, i.e. pressures or saturations)
     of 3D field
     \param scx (o) scatra vector (primary variables of scatra field, i.e. mass fraction)
     of 3D field
     */
    void extract_3d_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> x,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& stx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& flx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& scx);

    //! build block vector from field vectors, e.g. rhs, increment vector
    void setup_vector(Core::LinAlg::Vector<double>& f,  //!< vector of length of all dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            pv,  //!< vector containing structural + fluid dofs, i.e. poro dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            sv  //!< vector containing only scatra dofs
    );

    //! setup monolithic system matrix
    virtual void setup_system_matrix();

    //! print header
    void print_header() override;

    //! solve linear system of equations
    void linear_solve();

    //! convergence check
    bool converged();

    //! build norms
    virtual void build_convergence_norms();

    //! output
    void newton_output();

    //! check for convergence
    void newton_error_check();

    //! update the single fields after convergence
    void update_fields_after_convergence();

    //! update the scatra field
    virtual void update_scatra(std::shared_ptr<const Core::LinAlg::Vector<double>> scatrainc);

    //! return structure fluid coupling sparse matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> poro_fluid_scatra_coupling_matrix();

    //! return scatra structure coupling sparse matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> scatra_struct_coupling_matrix();

    //! return scatra fluid coupling sparse matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> scatra_poro_fluid_coupling_matrix();

    //! evaluate scatra field
    virtual void evaluate_scatra();

    //! evaluate porofluid-scatra coupling sparse matrix
    void apply_poro_fluid_scatra_coupl_matrix(std::shared_ptr<Core::LinAlg::SparseOperator> k_pfs);

    //! evaluate scatra-structure coupling sparse matrix
    void apply_scatra_struct_coupl_matrix(std::shared_ptr<Core::LinAlg::SparseOperator> k_sps);

    //! evaluate scatra-porofluid coupling sparse matrix
    void apply_scatra_poro_fluid_coupl_matrix(std::shared_ptr<Core::LinAlg::SparseOperator> k_spf);

    // update the single fields after convergence
    void print_structure_disabled_info();

    //! FD-Check
    void poro_multi_phase_scatra_fd_check();

    //! convergence tolerance (increment)
    double iter_tol_inc_;
    //! convergence tolerance (residual)
    double iter_tol_res_;
    //! maximally permitted iterations
    int iter_max_;
    //! minimally necessary iterations
    int iter_min_;
    //! current iteration step
    int iter_number_;

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

    //! @name Global vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;  //!< a zero vector of full length

    //! increment between Newton steps k and k+1 \f$\Delta{x}^{<k>}_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> iter_inc_;

    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_;  //!< rhs of struct-fluid-scatra system

    std::shared_ptr<Core::LinAlg::Solver> solver_;  //!< linear algebraic solver
    double solveradaptolbetter_;                    //!< tolerance to which is adapted ?
    bool solveradapttol_;                           //!< adapt solver tolerance

    // do we solve the structure?
    bool solve_structure_;

    // for building blocks
    int struct_offset_;

    //! block systemmatrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> systemmatrix_;

    //! structure-scatra coupling matrix --> we do not have it (yet)
    // std::shared_ptr<Core::LinAlg::SparseMatrix> k_pss_;

    //! fluid-scatra coupling matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> k_pfs_;

    //! scatra-structure coupling matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> k_sps_;
    //! scatra-fluid coupling matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> k_spf_;

    double tol_inc_;  //!< tolerance residual increment
    double tol_res_;  //!< tolerance force residual

    double tol_inc_structure_;  //!< tolerance residual increment for structure displacements
    double tol_res_structure_;  //!< tolerance force residual for structure displacements

    double tol_inc_porofluid_;  //!< tolerance residual increment for fluid
    double tol_res_porofluid_;  //!< tolerance force residual for fluid

    double tol_inc_scatra_;  //!< tolerance residual increment for scatra
    double tol_res_scatra_;  //!< tolerance force residual for scatra

    double norm_rhs_;  //!< norm of residual forces

    double norm_rhs_porofluid_;  //!< norm of residual forces (fluid )
    double norm_inc_porofluid_;  //!< norm of residual unknowns (fluid )

    double norm_rhs_structure_;  //!< norm of residual forces (structure)
    double norm_inc_structure_;  //!< norm of residual unknowns (structure)

    double norm_rhs_scatra_;  //!< norm of residual forces (scatra)
    double norm_inc_scatra_;  //!< norm of residual unknowns (scatra)

    double norm_rhs_artery_;       //!< norm of residual (artery)
    double norm_inc_artery_;       //!< norm of residual unknowns (artery)
    double norm_artery_pressure_;  //!< norm of artery pressure

    double norm_rhs_artery_scatra_;  //!< norm of residual (artery-scatra)
    double norm_inc_artery_scatra_;  //!< norm of residual unknowns (artery-scatra)
    double norm_artery_scatra_;      //!< norm of artery scatra mass-fractions

    double max_inc_;  //!< maximum increment
    double max_res_;  //!< maximum residual

    enum PoroPressureBased::VectorNorm vector_norm_res_;  //!< type of norm for residual
    enum PoroPressureBased::VectorNorm vector_norm_inc_;  //!< type of norm for increments

    Teuchos::Time timernewton_;  //!< timer for measurement of solution time of newton iterations
    double dtsolve_;             //!< linear solver time
    double dtele_;               //!< time for element evaluation + build-up of system matrix

    //! flag for finite difference check
    bool fdcheck_;
  };

  //! Monolithic solution scheme for porofluid-elasticity problems with scalar transport and
  //! coupled with a 1D artery network
  class PorofluidElastScatraMonolithicArteryCouplingAlgorithm
      : public PorofluidElastScatraMonolithicAlgorithm
  {
   public:
    PorofluidElastScatraMonolithicArteryCouplingAlgorithm(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

   private:
    // Setup full map
    void setup_maps() override;

    // update the scatra field
    void update_scatra(std::shared_ptr<const Core::LinAlg::Vector<double>> scatrainc) override;

    //! extract the field vectors from a given composed vector x.
    /*!
     \param x   (i) composed vector that contains all field vectors
     \param stx (o) structural vector (e.g. displacements)
     \param flx (o) fluid vector (primary variables of fluid field, i.e. pressures or saturations)
     and pressures of artery network
     \param scx (o) scatra vector (primary variables of scatra field, i.e. mass fraction)
      and mass fractions in 1D artery network
     */
    void extract_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> x,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& stx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& flx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& scx) override;

    //! setup monolithic system matrix
    void setup_system_matrix() override;

    // Setup monolithic rhs-vector
    void setup_rhs() override;

    //! evaluate scatra field
    void evaluate_scatra() override;

    //! build the combined dirichletbcmap
    void build_combined_dbc_map() override;

    /// setup
    void setup_system() override;

    //! return arteryscatra-artery coupling sparse matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> artery_scatra_artery_coupling_matrix();

    //! evaluate arteryscatra-artery coupling sparse matrix
    void apply_artery_scatra_artery_coupl_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> k_asa);

    //! build the block null spaces
    void build_block_null_spaces() override;

    //! build norms for convergence check
    void build_convergence_norms() override;

    //! dof row map (not split), only artery and porofluid
    std::shared_ptr<Core::LinAlg::Map> fullmap_artporo_;

    //! dof row map split in (field) blocks, only artery and porofluid
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockrowdofmap_artporo_;

    //! dof row map (not split), only artery and artery-scatra
    std::shared_ptr<Core::LinAlg::Map> fullmap_artscatra_;

    //! dof row map split in (field) blocks, only artery and artery-scatra
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockrowdofmap_artscatra_;

    //! artscatra-artery coupling matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> k_asa_;

    //! flag if nodal coupling active or not
    bool nodal_coupl_inactive_;
  };


}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
