// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_vector.hpp"

#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

FOUR_C_NAMESPACE_OPEN

NOX::Nln::Vector::Vector(const std::shared_ptr<Core::LinAlg::Vector<double>>& source,
    MemoryType memory, ::NOX::CopyType type)
{
  if (memory == MemoryType::View)
    linalg_vec = source;
  else
  {
    switch (type)
    {
      case ::NOX::DeepCopy:
        linalg_vec = std::make_shared<Core::LinAlg::Vector<double>>(*source);
        break;

      case ::NOX::ShapeCopy:
        linalg_vec = std::make_shared<Core::LinAlg::Vector<double>>(source->get_map());
        break;
    }
  }
}

NOX::Nln::Vector::Vector(Core::LinAlg::Vector<double>&& source)
    : linalg_vec(std::make_shared<Core::LinAlg::Vector<double>>(std::move(source)))
{
}

NOX::Nln::Vector::Vector(const Core::LinAlg::Vector<double>& source, ::NOX::CopyType type)
{
  switch (type)
  {
    case ::NOX::DeepCopy:
      linalg_vec = std::make_shared<Core::LinAlg::Vector<double>>(source);
      break;

    case ::NOX::ShapeCopy:
      linalg_vec = std::make_shared<Core::LinAlg::Vector<double>>(source.get_map());
      break;
  }
}

NOX::Nln::Vector::Vector(const NOX::Nln::Vector& source, ::NOX::CopyType type)
{
  switch (type)
  {
    case ::NOX::DeepCopy:
      linalg_vec = std::make_shared<Core::LinAlg::Vector<double>>(source.get_linalg_vector());
      break;

    case ::NOX::ShapeCopy:
      linalg_vec =
          std::make_shared<Core::LinAlg::Vector<double>>(source.get_linalg_vector().get_map());
      break;
  }
}

::NOX::Abstract::Vector& NOX::Nln::Vector::operator=(const Core::LinAlg::Vector<double>& source)
{
  linalg_vec->scale(1.0, source);
  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::operator=(const NOX::Nln::Vector& source)
{
  linalg_vec->scale(1.0, source.get_linalg_vector());
  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::operator=(const ::NOX::Abstract::Vector& source)
{
  return operator=(dynamic_cast<const NOX::Nln::Vector&>(source));
}

Teuchos::RCP<::NOX::Abstract::Vector> NOX::Nln::Vector::clone(::NOX::CopyType type) const
{
  Teuchos::RCP<::NOX::Abstract::Vector> newVec =
      Teuchos::rcp(new NOX::Nln::Vector(*linalg_vec, type));
  return newVec;
}

Core::LinAlg::Vector<double>& NOX::Nln::Vector::get_linalg_vector() { return *linalg_vec; }

const Core::LinAlg::Vector<double>& NOX::Nln::Vector::get_linalg_vector() const
{
  return *linalg_vec;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::init(double value)
{
  linalg_vec->put_scalar(value);
  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::random(bool, int)
{
  FOUR_C_THROW("NOX::Nln::Vector::random() is not implemented");

  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::abs(const ::NOX::Abstract::Vector& y)
{
  const auto& nln_vector = dynamic_cast<const NOX::Nln::Vector&>(y);
  linalg_vec->abs(nln_vector.get_linalg_vector());

  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::reciprocal(const ::NOX::Abstract::Vector& y)
{
  const auto& nln_vector = dynamic_cast<const NOX::Nln::Vector&>(y);
  linalg_vec->reciprocal(nln_vector.get_linalg_vector());

  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::scale(double gamma)
{
  linalg_vec->scale(gamma);

  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::scale(const ::NOX::Abstract::Vector& a)
{
  const auto& nln_vector = dynamic_cast<const NOX::Nln::Vector&>(a);
  linalg_vec->scale(1.0, nln_vector.get_linalg_vector());

  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::update(
    double alpha, const ::NOX::Abstract::Vector& a, double gamma)
{
  const auto& nln_vector = dynamic_cast<const NOX::Nln::Vector&>(a);
  linalg_vec->update(alpha, nln_vector.get_linalg_vector(), gamma);

  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::update(double alpha, const ::NOX::Abstract::Vector& a,
    double beta, const ::NOX::Abstract::Vector& b, double gamma)
{
  const auto& nln_vector_a = dynamic_cast<const NOX::Nln::Vector&>(a);
  const auto& nln_vector_b = dynamic_cast<const NOX::Nln::Vector&>(b);
  linalg_vec->update(
      alpha, nln_vector_a.get_linalg_vector(), beta, nln_vector_b.get_linalg_vector(), gamma);

  return *this;
}

double NOX::Nln::Vector::norm(::NOX::Abstract::Vector::NormType type) const
{
  return NOX::Nln::Aux::calc_vector_norm(*linalg_vec, type, false);
}

double NOX::Nln::Vector::norm(const ::NOX::Abstract::Vector&) const
{
  FOUR_C_THROW("NOX::Nln::Vector::norm(weights) is not implemented");

  return 0.0;
}

double NOX::Nln::Vector::innerProduct(const ::NOX::Abstract::Vector& y) const
{
  double res = 0.;
  const auto& nln_vector = dynamic_cast<const NOX::Nln::Vector&>(y);
  linalg_vec->dot(nln_vector.get_linalg_vector(), &res);

  return res;
}

::NOX::size_type NOX::Nln::Vector::length() const { return linalg_vec->global_length(); }

Epetra_Vector& NOX::Nln::Vector::getEpetraVector()
{
  return linalg_vec->get_ref_of_epetra_vector();
}

const Epetra_Vector& NOX::Nln::Vector::getEpetraVector() const
{
  return linalg_vec->get_ref_of_epetra_vector();
}

NOX::Nln::Vector::operator ::NOX::Epetra::Vector()
{
  return ::NOX::Epetra::Vector(Teuchos::rcp(&linalg_vec->get_ref_of_epetra_vector(), false));
}

NOX::Nln::Vector::operator ::NOX::Epetra::Vector() const
{
  return ::NOX::Epetra::Vector(Teuchos::rcp(&linalg_vec->get_ref_of_epetra_vector(), false));
}

FOUR_C_NAMESPACE_CLOSE
