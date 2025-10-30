// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_fevector.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_multi_vector.hpp"

FOUR_C_NAMESPACE_OPEN

template <typename T>
Core::LinAlg::FEVector<T>::FEVector(const Map& Map, bool zeroOut)
    : vector_(Utils::make_owner<Epetra_FEVector>(Map.get_epetra_block_map(), zeroOut))
{
}

template <typename T>
Core::LinAlg::FEVector<T>::FEVector(const Map& Map, int numVectors, bool ignoreNonLocalEntries)
    : vector_(Utils::make_owner<Epetra_FEVector>(
          Map.get_epetra_block_map(), numVectors, ignoreNonLocalEntries))
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
void Core::LinAlg::FEVector<T>::norm_1(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Norm1(Result));
#else
  vector_->Norm1(Result);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::norm_2(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Norm2(Result));
#else
  vector_->Norm2(Result);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::norm_inf(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->NormInf(Result));
#else
  vector_->NormInf(Result);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::min_value(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->MinValue(Result));
#else
  vector_->MinValue(Result);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::max_value(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->MaxValue(Result));
#else
  vector_->MaxValue(Result);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::mean_value(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->MeanValue(Result));
#else
  vector_->MeanValue(Result);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::scale(double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Scale(ScalarValue));
#else
  vector_->Scale(ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::dot(const Epetra_MultiVector& A, double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Dot(A, Result));
#else
  vector_->Dot(A, Result);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::abs(const Epetra_MultiVector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Abs(A));
#else
  vector_->Abs(A);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::scale(double ScalarA, const Epetra_MultiVector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Scale(ScalarA, A));
#else
  vector_->Scale(ScalarA, A);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::update(
    double ScalarA, const Epetra_MultiVector& A, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Update(ScalarA, A, ScalarThis));
#else
  vector_->Update(ScalarA, A, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::update(double ScalarA, const Epetra_MultiVector& A, double ScalarB,
    const Epetra_MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Update(ScalarA, A, ScalarB, B, ScalarThis));
#else
  vector_->Update(ScalarA, A, ScalarB, B, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::dot(const FEVector& A, double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Dot(A, Result));
#else
  vector_->Dot(A, Result);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::abs(const FEVector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Abs(A));
#else
  vector_->Abs(A);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::scale(double ScalarA, const FEVector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Scale(ScalarA, A));
#else
  vector_->Scale(ScalarA, A);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::update(double ScalarA, const FEVector& A, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Update(ScalarA, A, ScalarThis));
#else
  vector_->Update(ScalarA, A, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::update(
    double ScalarA, const FEVector& A, double ScalarB, const FEVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(
      vector_->Update(ScalarA, A, ScalarB, B.get_ref_of_epetra_fevector(), ScalarThis));
#else
  vector_->Update(ScalarA, A, ScalarB, B.get_ref_of_epetra_fevector(), ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::multiply(
    double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Multiply(ScalarAB, A, B, ScalarThis));
#else
  vector_->Multiply(ScalarAB, A, B, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::reciprocal_multiply(
    double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ReciprocalMultiply(ScalarAB, A, B, ScalarThis));
#else
  vector_->ReciprocalMultiply(ScalarAB, A, B, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::put_scalar(double ScalarConstant)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->PutScalar(ScalarConstant));
#else
  vector_->PutScalar(ScalarConstant);
#endif
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

template <typename T>
void Core::LinAlg::FEVector<T>::import(const Epetra_SrcDistObject& A,
    const Core::LinAlg::Import& Importer, Epetra_CombineMode CombineMode,
    const Epetra_OffsetIndex* Indexor)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Import(A, Importer.get_epetra_import(), CombineMode, Indexor));
#else
  vector_->Import(A, Importer.get_epetra_import(), CombineMode, Indexor);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::import(const Epetra_SrcDistObject& A, const Epetra_Export& Exporter,
    Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Import(A, Exporter, CombineMode, Indexor));
#else
  vector_->Import(A, Exporter, CombineMode, Indexor);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::export_to(const Epetra_SrcDistObject& A,
    const Core::LinAlg::Import& Importer, Epetra_CombineMode CombineMode,
    const Epetra_OffsetIndex* Indexor)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Export(A, Importer.get_epetra_import(), CombineMode, Indexor));
#else
  vector_->Export(A, Importer.get_epetra_import(), CombineMode, Indexor);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::export_to(const Epetra_SrcDistObject& A,
    const Core::LinAlg::Export& Exporter, Epetra_CombineMode CombineMode,
    const Epetra_OffsetIndex* Indexor)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Export(A, Exporter.get_epetra_export(), CombineMode, Indexor));
#else
  vector_->Export(A, Exporter.get_epetra_export(), CombineMode, Indexor);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::export_to(const Epetra_SrcDistObject& A,
    const Epetra_Export& Exporter, Epetra_CombineMode CombineMode,
    const Epetra_OffsetIndex* Indexor)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Export(A, Exporter, CombineMode, Indexor));
#else
  vector_->Export(A, Exporter, CombineMode, Indexor);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::complete(Epetra_CombineMode mode, bool reuse_map_and_exporter)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->GlobalAssemble(mode, reuse_map_and_exporter));
#else
  vector_->GlobalAssemble(mode, reuse_map_and_exporter);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::sum_into_local_value(
    int MyRow, int FEVectorIndex, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoMyValue(MyRow, FEVectorIndex, ScalarValue));
#else
  vector_->SumIntoMyValue(MyRow, FEVectorIndex, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::sum_into_global_value(
    int GlobalRow, int FEVectorIndex, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoGlobalValue(GlobalRow, FEVectorIndex, ScalarValue));
#else
  vector_->SumIntoGlobalValue(GlobalRow, FEVectorIndex, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::sum_into_global_value(
    long long GlobalRow, int FEVectorIndex, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoGlobalValue(GlobalRow, FEVectorIndex, ScalarValue));
#else
  vector_->SumIntoGlobalValue(GlobalRow, FEVectorIndex, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::sum_into_global_values(
    int numIDs, const int* GIDs, const int* numValuesPerID, const double* values, int vectorIndex)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(
      vector_->SumIntoGlobalValues(numIDs, GIDs, numValuesPerID, values, vectorIndex));
#else
  vector_->SumIntoGlobalValues(numIDs, GIDs, numValuesPerID, values, vectorIndex);
#endif
}

template <typename T>
void Core::LinAlg::FEVector<T>::sum_into_global_values(
    int numIDs, const int* GIDs, const double* values, int vectorIndex)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoGlobalValues(numIDs, GIDs, values, vectorIndex));
#else
  vector_->SumIntoGlobalValues(numIDs, GIDs, values, vectorIndex);
#endif
}

// explicit instantiation
template class Core::LinAlg::FEVector<double>;

FOUR_C_NAMESPACE_CLOSE
