/*----------------------------------------------------------------------*/
/*! \file

\brief monolithic coupling algorithm for scatra-thermo interaction

\level 2

*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_STI_MONOLITHIC_HPP
#define FOUR_C_STI_MONOLITHIC_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_inpar_sti.hpp"
#include "4C_sti_algorithm.hpp"

// forward declarations
class Epetra_Map;

FOUR_C_NAMESPACE_OPEN


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
    explicit Monolithic(const Epetra_Comm& comm,  //! communicator
        const Teuchos::ParameterList& stidyn,     //! parameter list for scatra-thermo interaction
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
        const Teuchos::RCP<const Core::LinAlg::SparseOperator>
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
    const Core::LinAlg::Solver& solver() const { return *solver_; };

   private:
    //! Apply Dirichlet conditions to assembled OD blocks
    void apply_dirichlet_off_diag(
        Teuchos::RCP<Core::LinAlg::SparseOperator>& scatrathermo_domain_interface,
        Teuchos::RCP<Core::LinAlg::SparseOperator>& thermoscatra_domain_interface);

    //! Assemble interface and domain contributions of OD blocks
    void assemble_domain_interface_off_diag(
        Teuchos::RCP<Core::LinAlg::SparseOperator>& scatrathermo_domain_interface,
        Teuchos::RCP<Core::LinAlg::SparseOperator>& thermoscatra_domain_interface);

    //! assemble global system of equations
    void assemble_mat_and_rhs();

    //! assemble off-diagonal scatra-thermo block of global system matrix
    void assemble_od_block_scatra_thermo();

    //! assemble off-diagonal thermo-scatra block of global system matrix
    void assemble_od_block_thermo_scatra();

    //! build null spaces associated with blocks of global system matrix
    void build_null_spaces() const;

    //! compute null space information associated with global system matrix if applicable
    void compute_null_space_if_necessary(Teuchos::ParameterList&
            solverparams  //! solver parameter list for scatra-thermo interaction
    ) const;

    //! global map of degrees of freedom
    const Teuchos::RCP<const Epetra_Map>& dof_row_map() const;

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
    Teuchos::RCP<Core::LinAlg::MapExtractor> maps_;

    // flag for double condensation of linear equations associated with temperature field
    const bool condensationthermo_;

    //! global system matrix
    Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix_;

    //! type of global system matrix in global system of equations
    const Core::LinAlg::MatrixType matrixtype_;

    //! scatra-thermo block of global system matrix (derivatives of scatra residuals w.r.t. thermo
    //! degrees of freedom), domain contributions
    Teuchos::RCP<Core::LinAlg::SparseOperator> scatrathermoblockdomain_;

    //! scatra-thermo block of global system matrix (derivatives of scatra residuals w.r.t. thermo
    //! degrees of freedom), interface contributions
    Teuchos::RCP<Core::LinAlg::SparseOperator> scatrathermoblockinterface_;

    //! thermo-scatra block of global system matrix (derivatives of thermo residuals w.r.t. scatra
    //! degrees of freedom), domain contributions
    Teuchos::RCP<Core::LinAlg::SparseOperator> thermoscatrablockdomain_;

    //! thermo-scatra block of global system matrix (derivatives of thermo residuals w.r.t. scatra
    //! degrees of freedom), interface contributions
    Teuchos::RCP<Core::LinAlg::SparseOperator> thermoscatrablockinterface_;

    //! map extractor associated with blocks of global system matrix
    Teuchos::RCP<Core::LinAlg::MultiMapExtractor> blockmaps_;

    //! map extractor associated with all degrees of freedom inside temperature field
    Teuchos::RCP<Core::LinAlg::MultiMapExtractor> blockmapthermo_;

    //! global increment vector for Newton-Raphson iteration
    Teuchos::RCP<Core::LinAlg::Vector<double>> increment_;

    //! global residual vector on right-hand side of global system of equations
    Teuchos::RCP<Core::LinAlg::Vector<double>> residual_;

    //! time for element evaluation and assembly of global system of equations
    double dtele_;

    //! time for solution of global system of equations
    double dtsolve_;

    //! algebraic solver for global system of equations
    Teuchos::RCP<Core::LinAlg::Solver> solver_;

    //! inverse sums of absolute values of row entries in global system matrix
    Teuchos::RCP<Core::LinAlg::Vector<double>> invrowsums_;

    //! interface coupling adapter for scatra discretization
    Teuchos::RCP<const Coupling::Adapter::Coupling> icoupscatra_;

    //! interface coupling adapter for thermo discretization
    Teuchos::RCP<const Coupling::Adapter::Coupling> icoupthermo_;

    //! slave-to-master row transformation operator for scatra-thermo block of global system matrix
    Teuchos::RCP<Coupling::Adapter::MatrixRowTransform> islavetomasterrowtransformscatraod_;

    //! slave-to-master column transformation operator for thermo-scatra block of global system
    //! matrix
    Teuchos::RCP<Coupling::Adapter::MatrixColTransform> islavetomastercoltransformthermood_;

    //! master-to-slave row transformation operator for thermo-scatra block of global system matrix
    Teuchos::RCP<Coupling::Adapter::MatrixRowTransform> islavetomasterrowtransformthermood_;

    //! evaluation of OD blocks for scatra-thermo coupling
    Teuchos::RCP<STI::ScatraThermoOffDiagCoupling> scatrathermooffdiagcoupling_;

    //! all equilibration of global system matrix and RHS is done in here
    Teuchos::RCP<Core::LinAlg::Equilibration> equilibration_;
  };  // class Monolithic : public Algorithm
}  // namespace STI
FOUR_C_NAMESPACE_CLOSE

#endif
