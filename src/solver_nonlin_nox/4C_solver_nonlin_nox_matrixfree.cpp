// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_matrixfree.hpp"

#include "4C_linear_solver_thyra_utils.hpp"

FOUR_C_NAMESPACE_OPEN

// Implementation of NOX::Nln::MatrixFree::ThyraModelWrapper
NOX::Nln::MatrixFree::ThyraModelWrapper::ThyraModelWrapper(
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> model,
    const std::shared_ptr<const Core::LinAlg::Map> map)
    : model_(model), map_(map), vector_space_(Core::LinearSolver::Utils::create_thyra_map(*map))
{
  ::Thyra::ModelEvaluatorBase::InArgsSetup<double> inArgs;
  inArgs.setModelEvalDescription(description());
  inArgs.setSupports(::Thyra::ModelEvaluatorBase::IN_ARG_x);
  prototype_in_args_ = inArgs;

  ::Thyra::ModelEvaluatorBase::OutArgsSetup<double> outArgs;
  outArgs.setModelEvalDescription(description());
  outArgs.setSupports(::Thyra::ModelEvaluatorBase::OUT_ARG_f);
  prototype_out_args_ = outArgs;
}

::Thyra::ModelEvaluatorBase::InArgs<double>
NOX::Nln::MatrixFree::ThyraModelWrapper::getNominalValues() const
{
  return prototype_in_args_;
}

::Thyra::ModelEvaluatorBase::InArgs<double> NOX::Nln::MatrixFree::ThyraModelWrapper::createInArgs()
    const
{
  return prototype_in_args_;
}

Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>>
NOX::Nln::MatrixFree::ThyraModelWrapper::get_x_space() const
{
  return vector_space_;
}

Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>>
NOX::Nln::MatrixFree::ThyraModelWrapper::get_f_space() const
{
  return vector_space_;
}

Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>>
NOX::Nln::MatrixFree::ThyraModelWrapper::get_p_space(int l) const
{
  return Teuchos::null;
}

Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>>
NOX::Nln::MatrixFree::ThyraModelWrapper::get_g_space(int j) const
{
  return Teuchos::null;
}

::Thyra::ModelEvaluatorBase::OutArgs<double>
NOX::Nln::MatrixFree::ThyraModelWrapper::createOutArgsImpl() const
{
  return prototype_out_args_;
}

void NOX::Nln::MatrixFree::ThyraModelWrapper::evalModelImpl(
    const ::Thyra::ModelEvaluatorBase::InArgs<double>& inArgs,
    const ::Thyra::ModelEvaluatorBase::OutArgs<double>& outArgs) const
{
  const Teuchos::RCP<const ::Thyra::VectorBase<double>> x_in = inArgs.get_x();
  const Teuchos::RCP<::Thyra::VectorBase<double>> f_out = outArgs.get_f();

  const Teuchos::RCP<const Epetra_Vector> x_in_epetra =
      Core::LinearSolver::Utils::get_epetra_vector_from_thyra(*map_, x_in);
  const Teuchos::RCP<Epetra_Vector> f_out_epetra =
      Core::LinearSolver::Utils::get_epetra_vector_from_thyra(*map_, f_out);

  model_->compute_f(Core::LinAlg::Vector<double>(*x_in_epetra), Core::LinAlg::View(*f_out_epetra),
      NOX::Nln::FillType::Residual);
}

// Implementation of NOX::Nln::MatrixFree::SparseOperatorWrapper
NOX::Nln::MatrixFree::SparseOperatorWrapper::SparseOperatorWrapper(
    const ::Thyra::LinearOpBase<double>& op, const std::shared_ptr<const Core::LinAlg::Map> map)
    : operator_(op), map_(map)
{
}

Epetra_Operator& NOX::Nln::MatrixFree::SparseOperatorWrapper::epetra_operator()
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::zero() { FOUR_C_THROW("Not implemented"); }

void NOX::Nln::MatrixFree::SparseOperatorWrapper::reset() { FOUR_C_THROW("Not implemented"); }

MPI_Comm NOX::Nln::MatrixFree::SparseOperatorWrapper::get_comm() const
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::assemble(int eid,
    const std::vector<int>& lmstride, const Core::LinAlg::SerialDenseMatrix& Aele,
    const std::vector<int>& lmrow, const std::vector<int>& lmrowowner,
    const std::vector<int>& lmcol)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::assemble(double val, int rgid, int cgid)
{
  FOUR_C_THROW("Not implemented");
}

bool NOX::Nln::MatrixFree::SparseOperatorWrapper::filled() const
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::complete(
    Core::LinAlg::OptionsMatrixComplete options_matrix_complete)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::complete(const Core::LinAlg::Map& domainmap,
    const Core::LinAlg::Map& rangemap, Core::LinAlg::OptionsMatrixComplete options_matrix_complete)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::un_complete() { FOUR_C_THROW("Not implemented"); }

void NOX::Nln::MatrixFree::SparseOperatorWrapper::apply_dirichlet(
    const Core::LinAlg::Vector<double>& dbctoggle, bool diagonalblock)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::apply_dirichlet(
    const Core::LinAlg::Map& dbcmap, bool diagonalblock)
{
  FOUR_C_THROW("Not implemented");
}

const Core::LinAlg::Map& NOX::Nln::MatrixFree::SparseOperatorWrapper::domain_map() const
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::add(const Core::LinAlg::SparseOperator& A,
    const bool transposeA, const double scalarA, const double scalarB)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::scale(double ScalarConstant)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::multiply(bool TransA,
    const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y) const
{
  FOUR_C_ASSERT(!TransA, "Transposed multiplication is not supported");

  auto rcp_x = Core::LinearSolver::Utils::create_thyra_multi_vector(X, *map_);
  auto rcp_y = Core::LinearSolver::Utils::create_thyra_multi_vector(Y, *map_);

  operator_.apply(::Thyra::NOTRANS, *rcp_x, rcp_y.ptr(), 1.0, 0.0);
}

// Implementation of NOX::Nln::MatrixFree
NOX::Nln::MatrixFree::MatrixFree(Teuchos::ParameterList& printParams,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase>& required,
    const NOX::Nln::Vector& cloneVector, double lambda)
    : matrix_free_(printParams),
      map_(build_map(cloneVector)),
      thyra_model_wrapper_(required, map_),
      sparse_operator_wrapper_(matrix_free_, map_)
{
  Teuchos::RCP<Teuchos::ParameterList> jfnkParams = Teuchos::parameterList();
  jfnkParams->set("Difference Type", "Forward");
  jfnkParams->set("Perturbation Algorithm", "KSP NOX 2001");
  jfnkParams->set("lambda", lambda);

  matrix_free_.setParameterList(jfnkParams);
}

std::shared_ptr<const Core::LinAlg::Map> NOX::Nln::MatrixFree::build_map(
    const NOX::Nln::Vector& cloneVector)
{
  std::shared_ptr<const Core::LinAlg::Map> map = nullptr;

  // If the vector is based on Epetra_Map, use it directly
  try
  {
    const auto& epetraMap = cloneVector.get_linalg_vector().get_map().get_epetra_map();
    map = std::make_shared<const Core::LinAlg::Map>(epetraMap);
  }
  // Otherwise, create Epetra_Map manually from the underlying Epetra_BlockMap
  catch (const Core::Exception&)
  {
    map = std::make_shared<const Core::LinAlg::Map>(
        cloneVector.get_linalg_vector().get_map().num_global_points(),
        cloneVector.get_linalg_vector().get_map().num_my_points(),
        cloneVector.get_linalg_vector().get_map().my_global_elements(),
        cloneVector.get_linalg_vector().get_map().index_base(),
        cloneVector.get_linalg_vector().get_map().get_comm());
  }

  return map;
}

Core::LinAlg::SparseOperator& NOX::Nln::MatrixFree::get_operator()
{
  return sparse_operator_wrapper_;
}

bool NOX::Nln::MatrixFree::compute_jacobian(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac)
{
  (void)jac;

  // This is another way of initializing the Thyra matrix-free operator, but
  // setBaseEvaluationToRawThyra() is broken in Trilinos and needs fixes. This can not be done at
  // the moment, since we are boned to a particular Trilinos version.
  /*
    // These need to by private members:
    Core::LinAlg::MultiVector<double> x_base_;
    Core::LinAlg::MultiVector<double> f_base_;
    std::shared_ptr<NOX::Nln::Interface::RequiredBase> required_;

    Core::LinAlg::View x_epetra_view(x);
    x_base_ = x_epetra_view.underlying().as_multi_vector();

    required_->computeF(x, f_base_.get_vector(0), ::NOX::Nln::FillType::Residual);

    auto rcp_thyra_x_base =
        Core::LinearSolver::Utils::create_thyra_multi_vector(x_base_, x_base_.get_map());
    auto rcp_thyra_f_base =
        Core::LinearSolver::Utils::create_thyra_multi_vector(f_base_, f_base_.get_map());

    matrix_free_.setBaseEvaluationToRawThyra(
        rcp_thyra_x_base, rcp_thyra_f_base, Teuchos::rcpFromRef(thyra_model_wrapper_));
  */

  // Create NOX::Thyra::Vector from x_base
  auto rcp_thyra_x_base = Core::LinearSolver::Utils::create_thyra_multi_vector(x, *map_);

  // Wrap into NOX::Thyra::Vector
  ::NOX::Thyra::Vector initial_guess(rcp_thyra_x_base->col(0));

  // Create NOX Thyra Group
  auto nox_group = Teuchos::rcp(new ::NOX::Thyra::Group(initial_guess,
      Teuchos::rcpFromRef(thyra_model_wrapper_), Teuchos::null, Teuchos::null, Teuchos::null,
      Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null));

  // This needs to be done to initialize the force vector of the NOX::Thyra::Group
  nox_group->computeF();

  // This initializes the matrix-free operator with the NOX::Thyra::Group
  matrix_free_.setBaseEvaluationToNOXGroup(nox_group);

  return true;
}

FOUR_C_NAMESPACE_CLOSE
