// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_NOX_JACOBIAN_HPP
#define FOUR_C_FSI_NOX_JACOBIAN_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparseoperator.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian_base.hpp"
#include "4C_solver_nonlin_nox_interface_required_base.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

#include <Epetra_Operator.h>
#include <NOX_Abstract_Group.H>
#include <NOX_Utils.H>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    class Vector;
  }  // namespace Nln

  namespace FSI
  {
    /// Matrix Free Newton Krylov based on an approximation of the residuum derivatives
    class FSIMatrixFree : public Core::LinAlg::SparseOperator,
                          public virtual NOX::Nln::Interface::JacobianBase
    {
     public:
      /*! \brief Constructor

        The vector \c x is used to clone the solution vector.
      */
      FSIMatrixFree(Teuchos::ParameterList& printParams,
          const std::shared_ptr<NOX::Nln::Interface::RequiredBase> i, const NOX::Nln::Vector& x);

      // Methods of Core::LinAlg::SparseOperator interface
      Epetra_Operator& epetra_operator() override;

      void zero() override;

      void reset() override;

      MPI_Comm get_comm() const override;

      void assemble(int eid, const std::vector<int>& lmstride,
          const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
          const std::vector<int>& lmrowowner, const std::vector<int>& lmcol) override;

      void assemble(double val, int rgid, int cgid) override;

      bool filled() const override;

      void complete(Core::LinAlg::OptionsMatrixComplete options_matrix_complete = {}) override;

      void complete(const Core::LinAlg::Map& domainmap, const Core::LinAlg::Map& rangemap,
          Core::LinAlg::OptionsMatrixComplete options_matrix_complete = {}) override;

      void un_complete() override;

      void apply_dirichlet(
          const Core::LinAlg::Vector<double>& dbctoggle, bool diagonalblock = true) override;

      void apply_dirichlet(const Core::LinAlg::Map& dbcmap, bool diagonalblock = true) override;

      const Core::LinAlg::Map& domain_map() const override;

      void add(const Core::LinAlg::SparseOperator& A, const bool transposeA, const double scalarA,
          const double scalarB) override;

      void scale(double ScalarConstant) override;

      void multiply(bool TransA, const Core::LinAlg::MultiVector<double>& X,
          Core::LinAlg::MultiVector<double>& Y) const override;

      //! Compute Jacobian given the specified input vector, x.  Returns true if computation was
      //! successful.
      bool compute_jacobian(
          const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac) override;

      //! Clone a ::NOX::Abstract::Group derived object and use the computeF() method of that group
      //! for the perturbation instead of the NOX::Nln::Interface::RequiredBase::computeF() method.
      //! This is required for LOCAL to get the operators correct during homotopy.
      void set_group_for_compute_f(const ::NOX::Abstract::Group& group);

     protected:
      //! Label for matrix
      std::string label;

      //! User provided interface function
      std::shared_ptr<NOX::Nln::Interface::RequiredBase> interface;

      //! The current solution vector
      NOX::Nln::Vector currentX;

      //! Perturbed solution vector
      mutable NOX::Nln::Vector perturbX;

      //! Perturbed solution vector
      mutable NOX::Nln::Vector perturbY;

      //! Core::LinAlg::Map object used in the returns of the Epetra_Operator derived methods.
      /*! If the user is using Core::LinAlg::Maps, then ::NOX::Epetra::MatrixFree must create an
       * equivalent Core::LinAlg::Map from the Core::LinAlg::Map that can be used as the return
       * object of the OperatorDomainMap() and OperatorRangeMap() methods.
       */
      std::shared_ptr<const Epetra_Map> epetraMap;

      //! Flag to enables the use of a group instead of the interface for the computeF() calls in
      //! the directional difference calculation.
      bool useGroupForComputeF;

      //! Pointer to the group for possible use in computeF() calls.
      std::shared_ptr<::NOX::Abstract::Group> groupPtr;

      //! Printing utilities.
      ::NOX::Utils utils;
    };

  }  // namespace FSI
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
