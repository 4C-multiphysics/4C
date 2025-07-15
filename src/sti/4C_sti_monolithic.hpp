// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STI_MONOLITHIC_HPP
#define FOUR_C_STI_MONOLITHIC_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_sti_algorithm.hpp"

FOUR_C_NAMESPACE_OPEN


// forward declarations
namespace Core::LinAlg
{
  class BlockSparseMatrixBase;
  class MapExtractor;
  class MultiMapExtractor;
  class Solver;
  class SparseMatrix;
  class SparseOperator;
  class Equilibration;
  enum class MatrixType;
}  // namespace Core::LinAlg

namespace STI
{
  class ScatraThermoOffDiagCoupling;

  //! monolithic coupling algorithm for scatra-thermo interaction
  class Monolithic : public Algorithm
  {
   public:
    //! constructor
    explicit Monolithic(MPI_Comm comm,         //! communicator
        const Teuchos::ParameterList& stidyn,  //! parameter list for scatra-thermo interaction
        const Teuchos::ParameterList&
            scatradyn,  //! scalar transport parameter list for scatra and thermo fields
        const Teuchos::ParameterList&
            solverparams,  //! solver parameter list for scatra-thermo interaction
        const Teuchos::ParameterList&
            solverparams_scatra,  //! solver parameter list for scatra field
        const Teuchos::ParameterList&
            solverparams_thermo  //! solver parameter list for thermo field
    );

    //! output matrix to *.csv file for debugging purposes, with global row and column IDs of matrix
    //! components in ascending order across all processors
    static void output_matrix_to_file(
        const std::shared_ptr<const Core::LinAlg::SparseOperator>
            sparseoperator,           //!< sparse or block sparse matrix to be output
        const int precision = 16,     //!< output precision
        const double tolerance = -1.  //!< output omission tolerance
    );

    //! output vector to *.csv file for debugging purposes, with global IDs of vector components in
    //! ascending order across all processors
    static void output_vector_to_file(
        const Core::LinAlg::MultiVector<double>& vector,  //!< vector to be output
        const int precision = 16,                         //!< output precision
        const double tolerance = -1.                      //!< output omission tolerance
    );

    //! return algebraic solver for global system of equations
    [[nodiscard]] const Core::LinAlg::Solver& solver() const { return *solver_; };

   private:
    //! Apply Dirichlet conditions to assembled OD blocks
    void apply_dirichlet_off_diag(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermo_domain_interface,
        std::shared_ptr<Core::LinAlg::SparseOperator> thermoscatra_domain_interface) const;

    //! Assemble interface and domain contributions of OD blocks
    void assemble_domain_interface_off_diag(
        std::shared_ptr<Core::LinAlg::SparseOperator>& scatrathermo_domain_interface,
        std::shared_ptr<Core::LinAlg::SparseOperator>& thermoscatra_domain_interface) const;

    //! assemble global system of equations
    void assemble_mat_and_rhs();

    //! build null spaces associated with blocks of global system matrix
    void build_null_spaces() const;

    //! compute null space information associated with global system matrix if applicable
    void compute_null_space_if_necessary(Teuchos::ParameterList&
            solverparams  //! solver parameter list for scatra-thermo interaction
    ) const;

    //! global map of degrees of freedom
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const;

    //! check termination criterion for Newton-Raphson iteration
    bool exit_newton_raphson();

    //! finite difference check for global system matrix
    void fd_check();

    //! prepare time step
    void prepare_time_step() override;

    //! evaluate time step using Newton-Raphson iteration
    void solve() override;

    //! absolute tolerance for residual vectors
    const double restol_;

    //! global map extractor (0: scatra, 1: thermo)
    std::shared_ptr<Core::LinAlg::MapExtractor> maps_;

    // flag for double condensation of linear equations associated with temperature field
    const bool condensationthermo_;

    //! global system matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix_;

    //! type of global system matrix in global system of equations
    const Core::LinAlg::MatrixType matrixtype_;

    //! scatra-thermo block of global system matrix (derivatives of scatra residuals w.r.t. thermo
    //! degrees of freedom), domain contributions
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermoblockdomain_;

    //! scatra-thermo block of global system matrix (derivatives of scatra residuals w.r.t. thermo
    //! degrees of freedom), interface contributions
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermoblockinterface_;

    //! thermo-scatra block of global system matrix (derivatives of thermo residuals w.r.t. scatra
    //! degrees of freedom), domain contributions
    std::shared_ptr<Core::LinAlg::SparseOperator> thermoscatrablockdomain_;

    //! thermo-scatra block of global system matrix (derivatives of thermo residuals w.r.t. scatra
    //! degrees of freedom), interface contributions
    std::shared_ptr<Core::LinAlg::SparseOperator> thermoscatrablockinterface_;

    //! map extractor associated with blocks of global system matrix
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockmaps_;

    //! map extractor associated with all degrees of freedom inside temperature field
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockmapthermo_;

    //! global increment vector for Newton-Raphson iteration
    std::shared_ptr<Core::LinAlg::Vector<double>> increment_;

    //! global residual vector on right-hand side of global system of equations
    std::shared_ptr<Core::LinAlg::Vector<double>> residual_;

    //! time for element evaluation and assembly of global system of equations
    double dtele_;

    //! time for solution of global system of equations
    double dtsolve_;

    //! algebraic solver for global system of equations
    std::shared_ptr<Core::LinAlg::Solver> solver_;

    //! inverse sums of absolute values of row entries in global system matrix
    std::shared_ptr<Core::LinAlg::Vector<double>> invrowsums_;

    //! interface coupling adapter for scatra discretization
    std::shared_ptr<const Coupling::Adapter::Coupling> icoupscatra_;

    //! interface coupling adapter for thermo discretization
    std::shared_ptr<const Coupling::Adapter::Coupling> icoupthermo_;

    //! slave-to-master row transformation operator for scatra-thermo block of global system matrix
    std::shared_ptr<Coupling::Adapter::MatrixRowTransform> islavetomasterrowtransformscatraod_;

    //! slave-to-master column transformation operator for thermo-scatra block of global system
    //! matrix
    std::shared_ptr<Coupling::Adapter::MatrixColTransform> islavetomastercoltransformthermood_;

    //! master-to-slave row transformation operator for thermo-scatra block of global system matrix
    std::shared_ptr<Coupling::Adapter::MatrixRowTransform> islavetomasterrowtransformthermood_;

    //! evaluation of OD blocks for scatra-thermo coupling
    std::shared_ptr<STI::ScatraThermoOffDiagCoupling> scatrathermooffdiagcoupling_;

    //! all equilibration of global system matrix and RHS is done in here
    std::shared_ptr<Core::LinAlg::Equilibration> equilibration_;
  };  // class Monolithic : public Algorithm
}  // namespace STI
FOUR_C_NAMESPACE_CLOSE

#endif
