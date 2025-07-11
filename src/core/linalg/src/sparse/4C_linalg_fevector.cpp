// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_fevector.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_utils_exceptions.hpp"


FOUR_C_NAMESPACE_OPEN



template <typename T>
Core::LinAlg::FEVector<T>::FEVector(const Map& Map, bool zeroOut)
    : vector_(Utils::make_owner<Epetra_FEVector>(Map.get_epetra_block_map(), zeroOut))
{
}

template <typename T>
Core::LinAlg::FEVector<T>::FEVector(const Epetra_BlockMap& Map, bool zeroOut)
    : vector_(Utils::make_owner<Epetra_FEVector>(Map, zeroOut))
{
}

template <typename T>
Core::LinAlg::FEVector<T>::FEVector(
    const Epetra_BlockMap& Map, int numVectors, bool ignoreNonLocalEntries)
    : vector_(Utils::make_owner<Epetra_FEVector>(Map, numVectors, ignoreNonLocalEntries))
{
}

template <typename T>
Core::LinAlg::FEVector<T>::FEVector(const Epetra_FEVector& Source)
    : vector_(Utils::make_owner<Epetra_FEVector>(Source))
{
}



template <typename T>
Core::LinAlg::FEVector<T>::FEVector(const FEVector& other)
    : vector_(Utils::make_owner<Epetra_FEVector>(other.get_ref_of_epetra_fevector()))
{
}


template <typename T>
Core::LinAlg::FEVector<T>& Core::LinAlg::FEVector<T>::operator=(const FEVector& other)
{
  *vector_ = other.get_ref_of_epetra_fevector();
  return *this;
}



template <typename T>
Core::LinAlg::FEVector<T>::operator const Core::LinAlg::MultiVector<T>&() const
{
  // We may safely const-cast here, since constness is restored by the returned const reference.
  return multi_vector_view_.sync(const_cast<Epetra_FEVector&>(*vector_));
}


template <typename T>
Core::LinAlg::FEVector<T>::operator Core::LinAlg::MultiVector<T>&()
{
  return multi_vector_view_.sync(*vector_);
}


template <typename T>
const Core::LinAlg::MultiVector<T>& Core::LinAlg::FEVector<T>::as_multi_vector() const
{
  return static_cast<const Core::LinAlg::MultiVector<T>&>(*this);
}


template <typename T>
Core::LinAlg::MultiVector<T>& Core::LinAlg::FEVector<T>::as_multi_vector()
{
  return static_cast<Core::LinAlg::MultiVector<T>&>(*this);
}


template <typename T>
int Core::LinAlg::FEVector<T>::norm_1(double* Result) const
{
  return vector_->Norm1(Result);
}

template <typename T>
int Core::LinAlg::FEVector<T>::norm_2(double* Result) const
{
  return vector_->Norm2(Result);
}

template <typename T>
int Core::LinAlg::FEVector<T>::norm_inf(double* Result) const
{
  return vector_->NormInf(Result);
}

template <typename T>
int Core::LinAlg::FEVector<T>::min_value(double* Result) const
{
  return vector_->MinValue(Result);
}

template <typename T>
int Core::LinAlg::FEVector<T>::max_value(double* Result) const
{
  return vector_->MaxValue(Result);
}

template <typename T>
int Core::LinAlg::FEVector<T>::mean_value(double* Result) const
{
  return vector_->MeanValue(Result);
}

template <typename T>
int Core::LinAlg::FEVector<T>::dot(const Epetra_MultiVector& A, double* Result) const
{
  return vector_->Dot(A, Result);
}

template <typename T>
int Core::LinAlg::FEVector<T>::abs(const Epetra_MultiVector& A)
{
  return vector_->Abs(A);
}

template <typename T>
int Core::LinAlg::FEVector<T>::scale(double ScalarA, const Epetra_MultiVector& A)
{
  return vector_->Scale(ScalarA, A);
}

template <typename T>
int Core::LinAlg::FEVector<T>::update(
    double ScalarA, const Epetra_MultiVector& A, double ScalarThis)
{
  return vector_->Update(ScalarA, A, ScalarThis);
}

template <typename T>
int Core::LinAlg::FEVector<T>::update(double ScalarA, const Epetra_MultiVector& A, double ScalarB,
    const Epetra_MultiVector& B, double ScalarThis)
{
  return vector_->Update(ScalarA, A, ScalarB, B, ScalarThis);
}


template <typename T>
int Core::LinAlg::FEVector<T>::dot(const FEVector& A, double* Result) const
{
  return vector_->Dot(A, Result);
}

template <typename T>
int Core::LinAlg::FEVector<T>::abs(const FEVector& A)
{
  return vector_->Abs(A);
}

template <typename T>
int Core::LinAlg::FEVector<T>::scale(double ScalarA, const FEVector& A)
{
  return vector_->Scale(ScalarA, A);
}

template <typename T>
int Core::LinAlg::FEVector<T>::update(double ScalarA, const FEVector& A, double ScalarThis)
{
  return vector_->Update(ScalarA, A, ScalarThis);
}

template <typename T>
int Core::LinAlg::FEVector<T>::update(
    double ScalarA, const FEVector& A, double ScalarB, const FEVector& B, double ScalarThis)
{
  return vector_->Update(ScalarA, A, ScalarB, B.get_ref_of_epetra_fevector(), ScalarThis);
}

template <typename T>
int Core::LinAlg::FEVector<T>::put_scalar(double ScalarConstant)
{
  return vector_->PutScalar(ScalarConstant);
}



template <typename T>
int Core::LinAlg::FEVector<T>::replace_map(const Map& map)
{
  multi_vector_view_.invalidate();
  map_.invalidate();
  auto rv = vector_->ReplaceMap(map.get_epetra_block_map());
  return rv;
}

template <typename T>
std::unique_ptr<Core::LinAlg::FEVector<T>> Core::LinAlg::FEVector<T>::create_view(
    Epetra_FEVector& view)
{
  std::unique_ptr<FEVector<T>> ret(new FEVector<T>);
  ret->vector_ = Utils::make_view(&view);
  return ret;
}

template <typename T>
std::unique_ptr<const Core::LinAlg::FEVector<T>> Core::LinAlg::FEVector<T>::create_view(
    const Epetra_FEVector& view)
{
  std::unique_ptr<FEVector<T>> ret(new FEVector<T>);
  // We may const-cast here, since constness is restored inside the returned unique_ptr.
  ret->vector_ = Utils::make_view(const_cast<Epetra_FEVector*>(&view));
  return ret;
}

template <typename T>
MPI_Comm Core::LinAlg::FEVector<T>::get_comm() const
{
  return Core::Communication::unpack_epetra_comm(vector_->Comm());
}

template <typename T>
const Core::LinAlg::Map& Core::LinAlg::FEVector<T>::get_map() const
{
  return map_.sync(vector_->Map());
}


// explicit instantiation
template class Core::LinAlg::FEVector<double>;



FOUR_C_NAMESPACE_CLOSE
