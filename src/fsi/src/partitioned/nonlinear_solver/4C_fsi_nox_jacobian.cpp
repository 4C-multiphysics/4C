// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_nox_jacobian.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <NOX_Abstract_Group.H>
#include <NOX_Utils.H>

#include <iostream>

FOUR_C_NAMESPACE_OPEN


NOX::FSI::FSIMatrixFree::FSIMatrixFree(Teuchos::ParameterList& printParams,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> i, const NOX::Nln::Vector& x)
    : label("FSI-Matrix-Free"),
      interface(i),
      currentX(x),
      perturbX(x),
      perturbY(x),
      useGroupForComputeF(false),
      utils(printParams)
{
  perturbX.init(0.0);
  perturbY.init(0.0);

  // Epetra_Operators require Epetra_Maps, so anyone using block maps
  // (Core::LinAlg::Map) won't be able to directly use the iterative solver.
  // We get around this by creating an Epetra_Map from the Core::LinAlg::Map.
  try
  {
    epetraMap =
        std::make_shared<Epetra_Map>(currentX.get_linalg_vector().get_map().get_epetra_map());
  }
  catch (const Core::Exception&)
  {
    int size = currentX.get_linalg_vector().get_map().num_global_points();
    int mySize = currentX.get_linalg_vector().get_map().num_my_points();
    int indexBase = currentX.get_linalg_vector().get_map().index_base();
    const auto& comm = currentX.get_linalg_vector().get_map().get_comm();
    epetraMap = std::make_shared<Epetra_Map>(
        size, mySize, indexBase, Core::Communication::as_epetra_comm(comm));
  }
}

Epetra_Operator& NOX::FSI::FSIMatrixFree::epetra_operator() { FOUR_C_THROW("Not implemented"); }

void NOX::FSI::FSIMatrixFree::zero() { FOUR_C_THROW("Not implemented"); }

void NOX::FSI::FSIMatrixFree::reset() { FOUR_C_THROW("Not implemented"); }

MPI_Comm NOX::FSI::FSIMatrixFree::get_comm() const { FOUR_C_THROW("Not implemented"); }

void NOX::FSI::FSIMatrixFree::assemble(int eid, const std::vector<int>& lmstride,
    const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
    const std::vector<int>& lmrowowner, const std::vector<int>& lmcol)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::FSI::FSIMatrixFree::assemble(double val, int rgid, int cgid)
{
  FOUR_C_THROW("Not implemented");
}

bool NOX::FSI::FSIMatrixFree::filled() const { FOUR_C_THROW("Not implemented"); }

void NOX::FSI::FSIMatrixFree::complete(Core::LinAlg::OptionsMatrixComplete options_matrix_complete)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::FSI::FSIMatrixFree::complete(const Core::LinAlg::Map& domainmap,
    const Core::LinAlg::Map& rangemap, Core::LinAlg::OptionsMatrixComplete options_matrix_complete)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::FSI::FSIMatrixFree::un_complete() { FOUR_C_THROW("Not implemented"); }

void NOX::FSI::FSIMatrixFree::apply_dirichlet(
    const Core::LinAlg::Vector<double>& dbctoggle, bool diagonalblock)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::FSI::FSIMatrixFree::apply_dirichlet(const Core::LinAlg::Map& dbcmap, bool diagonalblock)
{
  FOUR_C_THROW("Not implemented");
}


const Core::LinAlg::Map& NOX::FSI::FSIMatrixFree::domain_map() const
{
  FOUR_C_THROW("Not implemented");
}

void NOX::FSI::FSIMatrixFree::add(const Core::LinAlg::SparseOperator& A, const bool transposeA,
    const double scalarA, const double scalarB)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::FSI::FSIMatrixFree::scale(double ScalarConstant) { FOUR_C_THROW("Not implemented"); }

void NOX::FSI::FSIMatrixFree::multiply(bool TransA, const Core::LinAlg::MultiVector<double>& X,
    Core::LinAlg::MultiVector<double>& Y) const
{
  if (TransA == true)
  {
    utils.out()
        << "ERROR: FSIMatrixFree::multiply() - Transpose is unavailable in Matrix-Free mode!"
        << std::endl;
    throw "NOX Error";
  }

  // Calculate the matrix-vector product:
  //
  // y = R' x = S'(F(d)) F'(d) x - x
  //
  // that comes down to a FSI residuum call with linear field solvers.
  //
  // We make use of the special structure of the FSI Residuum (this
  // approach is not general purpose) and neglect the dependence of
  // the fluid field on the interface displacements.

  // Convert X and Y from an Epetra_MultiVector to a Core::LinAlg::Vectors
  // and NOX::Nln::Vectors.  This is done so we use a consistent
  // vector space for norms and inner products.

  // There is a const_cast introduced - should be removed
  NOX::Nln::Vector nevX(
      Core::Utils::shared_ptr_from_ref(const_cast<Core::LinAlg::Vector<double>&>(X.get_vector(0))),
      NOX::Nln::Vector::MemoryType::View);
  NOX::Nln::Vector nevY(
      Core::Utils::shared_ptr_from_ref(Y.get_vector(0)), NOX::Nln::Vector::MemoryType::View);

  // The trial vector x is not guaranteed to be a suitable interface
  // displacement. It might be much too large to fit the ALE
  // algorithm. But we know our residual to be linear, so we can
  // easily scale x.

  double xscale = 1e4 * nevX.norm();
  // double xscale = nevX.norm();
  if (xscale == 0)
  {
    // In the first call is x=0. No need to calculate the
    // residuum. y=0 in that case.
    nevY.init(0.);
    return;
  }

  // For some strange reason currentX.Map()!=X.Map() and we are bound
  // to call computeF with the right map.
  perturbX = currentX;
  // perturbX.update(1./xscale,nevX,0.0);
  perturbX.update(1., nevX, 0.0);

  if (!useGroupForComputeF)
  {
    interface->compute_f(
        perturbX.get_linalg_vector(), perturbY.get_linalg_vector(), NOX::Nln::FillType::User);
  }
  else
  {
    groupPtr->setX(perturbX);
    groupPtr->computeF();
    perturbY = groupPtr->getF();
  }

  // scale back
  // nevY.update(xscale, perturbY, 0.0);
  nevY.update(1., perturbY, 0.0);
}


bool NOX::FSI::FSIMatrixFree::compute_jacobian(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac)
{
  // Remember the current interface displacements.
  currentX = x;

  // Nothing to do here. The work is done when we apply a vector to
  // the Jacobian.
  bool ok = true;
  return ok;
}


void NOX::FSI::FSIMatrixFree::set_group_for_compute_f(const ::NOX::Abstract::Group& group)
{
  useGroupForComputeF = true;
  groupPtr = std::shared_ptr<::NOX::Abstract::Group>(group.clone().release().get());
}

FOUR_C_NAMESPACE_CLOSE