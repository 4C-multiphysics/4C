// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_multi_vector.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

template <typename T>
Core::LinAlg::MultiVector<T>::MultiVector(const Epetra_BlockMap& Map, int num_columns, bool zeroOut)
    : vector_(Utils::make_owner<Epetra_MultiVector>(Map, num_columns, zeroOut))
{
}

template <typename T>
Core::LinAlg::MultiVector<T>::MultiVector(
    const Core::LinAlg::Map& Map, int num_columns, bool zeroOut)
    : vector_(
          Utils::make_owner<Epetra_MultiVector>(Map.get_epetra_block_map(), num_columns, zeroOut))
{
}

template <typename T>
Core::LinAlg::MultiVector<T>::MultiVector(const Epetra_MultiVector& source)
    : vector_(Utils::make_owner<Epetra_MultiVector>(source))
{
}

template <typename T>
Core::LinAlg::MultiVector<T>::MultiVector(const Epetra_FEVector& source)
    : vector_(Utils::make_owner<Epetra_MultiVector>(source))
{
}

template <typename T>
Core::LinAlg::MultiVector<T>::MultiVector(const MultiVector& other)
    : vector_(Utils::make_owner<Epetra_MultiVector>(*other.vector_))
{
}

template <typename T>
Core::LinAlg::MultiVector<T>& Core::LinAlg::MultiVector<T>::operator=(const MultiVector& other)
{
  *vector_ = *other.vector_;
  return *this;
}

template <typename T>
void Core::LinAlg::MultiVector<T>::norm_1(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Norm1(Result));
#else
  vector_->Norm1(Result);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::norm_2(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Norm2(Result));
#else
  vector_->Norm2(Result);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::norm_inf(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->NormInf(Result));
#else
  vector_->NormInf(Result);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::min_value(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->MinValue(Result));
#else
  vector_->MinValue(Result);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::max_value(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->MaxValue(Result));
#else
  vector_->MaxValue(Result);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::mean_value(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->MeanValue(Result));
#else
  vector_->MeanValue(Result);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::scale(double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Scale(ScalarValue));
#else
  vector_->Scale(ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::dot(const MultiVector& A, double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Dot(A, Result));
#else
  vector_->Dot(A, Result);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::abs(const MultiVector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Abs(A));
#else
  vector_->Abs(A);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::scale(double ScalarA, const MultiVector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Scale(ScalarA, A));
#else
  vector_->Scale(ScalarA, A);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::update(double ScalarA, const MultiVector& A, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Update(ScalarA, A, ScalarThis));
#else
  vector_->Update(ScalarA, A, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::update(
    double ScalarA, const MultiVector& A, double ScalarB, const MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Update(ScalarA, A, ScalarB, *B.vector_, ScalarThis));
#else
  vector_->Update(ScalarA, A, ScalarB, *B.vector_, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::put_scalar(double ScalarConstant)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->PutScalar(ScalarConstant));
#else
  vector_->PutScalar(ScalarConstant);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::reciprocal_multiply(
    double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ReciprocalMultiply(ScalarAB, A, B, ScalarThis));
#else
  vector_->ReciprocalMultiply(ScalarAB, A, B, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::multiply(
    double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Multiply(ScalarAB, A, B, ScalarThis));
#else
  vector_->Multiply(ScalarAB, A, B, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::multiply(char TransA, char TransB, double ScalarAB,
    const Epetra_MultiVector& A, const Epetra_MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Multiply(TransA, TransB, ScalarAB, A, B, ScalarThis));
#else
  vector_->Multiply(TransA, TransB, ScalarAB, A, B, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::reciprocal(const Epetra_MultiVector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Reciprocal(A));
#else
  vector_->Reciprocal(A);
#endif
}

template <typename T>
int Core::LinAlg::MultiVector<T>::replace_map(const Core::LinAlg::Map& map)
{
  column_vector_view_.clear();
  return vector_->ReplaceMap(map.get_epetra_block_map());
}

template <typename T>
MPI_Comm Core::LinAlg::MultiVector<T>::get_comm() const
{
  return Core::Communication::unpack_epetra_comm(vector_->Comm());
}

template <typename T>
Core::LinAlg::Vector<double>& Core::LinAlg::MultiVector<T>::operator()(int i)
{
  FOUR_C_ASSERT_ALWAYS(
      i < vector_->NumVectors(), "Index {} out of bounds [0,{}).", i, vector_->NumVectors());
  column_vector_view_.resize(vector_->NumVectors());
  return column_vector_view_[i].sync(*(*vector_)(i));
}

template <typename T>
const Core::LinAlg::Vector<double>& Core::LinAlg::MultiVector<T>::operator()(int i) const
{
  FOUR_C_ASSERT_ALWAYS(
      i < vector_->NumVectors(), "Index {} out of bounds [0,{}).", i, vector_->NumVectors());
  column_vector_view_.resize(vector_->NumVectors());
  // We may safely const_cast here, since constness is restored by the returned const reference.
  return column_vector_view_[i].sync(const_cast<Epetra_Vector&>(*(*vector_)(i)));
}

template <typename T>
std::unique_ptr<Core::LinAlg::MultiVector<T>> Core::LinAlg::MultiVector<T>::create_view(
    Epetra_MultiVector& view)
{
  std::unique_ptr<MultiVector<T>> ret(new MultiVector<T>);
  ret->vector_ = Utils::make_view<Epetra_MultiVector>(&view);
  return ret;
}

template <typename T>
std::unique_ptr<const Core::LinAlg::MultiVector<T>> Core::LinAlg::MultiVector<T>::create_view(
    const Epetra_MultiVector& view)
{
  std::unique_ptr<MultiVector<T>> ret(new MultiVector<T>);
  // We may safely const_cast here, since constness is restored inside the returned unique_ptr.
  ret->vector_ = Utils::make_view<Epetra_MultiVector>(const_cast<Epetra_MultiVector*>(&view));
  return ret;
}

template <typename T>
void Core::LinAlg::MultiVector<T>::import(const Epetra_SrcDistObject& A,
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
void Core::LinAlg::MultiVector<T>::import(const Epetra_SrcDistObject& A,
    const Core::LinAlg::Export& Exporter, Epetra_CombineMode CombineMode,
    const Epetra_OffsetIndex* Indexor)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Import(A, Exporter.get_epetra_export(), CombineMode, Indexor));
#else
  vector_->Import(A, Exporter.get_epetra_export(), CombineMode, Indexor);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::export_to(const Epetra_SrcDistObject& A,
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
void Core::LinAlg::MultiVector<T>::export_to(const Epetra_SrcDistObject& A,
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
void Core::LinAlg::MultiVector<T>::replace_local_value(
    int MyRow, int VectorIndex, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ReplaceMyValue(MyRow, VectorIndex, ScalarValue));
#else
  vector_->ReplaceMyValue(MyRow, VectorIndex, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::replace_global_value(
    int GlobalRow, int VectorIndex, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ReplaceGlobalValue(GlobalRow, VectorIndex, ScalarValue));
#else
  vector_->ReplaceGlobalValue(GlobalRow, VectorIndex, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::replace_global_value(
    long long GlobalRow, int VectorIndex, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ReplaceGlobalValue(GlobalRow, VectorIndex, ScalarValue));
#else
  vector_->ReplaceGlobalValue(GlobalRow, VectorIndex, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::sum_into_global_value(
    int GlobalRow, int VectorIndex, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoGlobalValue(GlobalRow, VectorIndex, ScalarValue));
#else
  vector_->SumIntoGlobalValue(GlobalRow, VectorIndex, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::sum_into_global_value(
    long long GlobalRow, int VectorIndex, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoGlobalValue(GlobalRow, VectorIndex, ScalarValue));
#else
  vector_->SumIntoGlobalValue(GlobalRow, VectorIndex, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::sum_into_local_value(
    int MyRow, int VectorIndex, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoMyValue(MyRow, VectorIndex, ScalarValue));
#else
  vector_->SumIntoMyValue(MyRow, VectorIndex, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::sum_into_local_value(
    int MyBlockRow, int BlockRowOffset, int VectorIndex, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoMyValue(MyBlockRow, BlockRowOffset, VectorIndex, ScalarValue));
#else
  vector_->SumIntoMyValue(MyBlockRow, BlockRowOffset, VectorIndex, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::extract_view(double*** ArrayOfPointers) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ExtractView(ArrayOfPointers));
#else
  vector_->ExtractView(ArrayOfPointers);
#endif
}

template <typename T>
void Core::LinAlg::MultiVector<T>::extract_copy(double* A, int MyLDA) const

{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ExtractCopy(A, MyLDA));
#else
  vector_->ExtractCopy(A, MyLDA);
#endif
}

// explicit instantiation
template class Core::LinAlg::MultiVector<double>;

FOUR_C_NAMESPACE_CLOSE
