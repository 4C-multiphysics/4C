// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_vector.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_utils_epetra_exceptions.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_Vector.h>

FOUR_C_NAMESPACE_OPEN



template <typename T>
Core::LinAlg::Vector<T>::Vector(const Map& Map, bool zeroOut)
    : vector_(Utils::make_owner<Epetra_Vector>(Map.get_epetra_block_map(), zeroOut))
{
}

template <typename T>
Core::LinAlg::Vector<T>::Vector(const Epetra_BlockMap& Map, bool zeroOut)
    : vector_(Utils::make_owner<Epetra_Vector>(Map, zeroOut))
{
}

template <typename T>
Core::LinAlg::Vector<T>::Vector(const Epetra_Vector& Source)
    : vector_(Utils::make_owner<Epetra_Vector>(Source))
{
}

template <typename T>
Core::LinAlg::Vector<T>::Vector(const Epetra_FEVector& Source)
    : vector_(Utils::make_owner<Epetra_Vector>(Epetra_DataAccess::Copy, Source, 0))
{
  FOUR_C_ASSERT(Source.NumVectors() == 1, "Can only convert a FE vector with a single column.");
}

template <typename T>
Core::LinAlg::Vector<T>::Vector(const Vector& other)
    : vector_(Utils::make_owner<Epetra_Vector>(other.get_ref_of_epetra_vector()))
{
}


template <typename T>
Core::LinAlg::Vector<T>& Core::LinAlg::Vector<T>::operator=(const Vector& other)
{
  *vector_ = other.get_ref_of_epetra_vector();
  return *this;
}


template <typename T>
Core::LinAlg::Vector<T>::operator const Core::LinAlg::MultiVector<T>&() const
{
  // We may safely const-cast here, since constness is restored by the returned const reference.
  return multi_vector_view_.sync(const_cast<Epetra_Vector&>(*vector_));
}


template <typename T>
Core::LinAlg::Vector<T>::operator Core::LinAlg::MultiVector<T>&()
{
  return multi_vector_view_.sync(*vector_);
}


template <typename T>
const Core::LinAlg::MultiVector<T>& Core::LinAlg::Vector<T>::as_multi_vector() const
{
  return static_cast<const Core::LinAlg::MultiVector<T>&>(*this);
}


template <typename T>
Core::LinAlg::MultiVector<T>& Core::LinAlg::Vector<T>::as_multi_vector()
{
  return static_cast<Core::LinAlg::MultiVector<T>&>(*this);
}


template <typename T>
void Core::LinAlg::Vector<T>::norm_1(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Norm1(Result));
#else
  vector_->Norm1(Result);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::norm_2(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Norm2(Result));
#else
  vector_->Norm2(Result);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::norm_inf(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->NormInf(Result));
#else
  vector_->NormInf(Result);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::min_value(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->MinValue(Result));
#else
  vector_->MinValue(Result);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::max_value(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->MaxValue(Result));
#else
  vector_->MaxValue(Result);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::mean_value(double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->MeanValue(Result));
#else
  vector_->MeanValue(Result);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::scale(double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Scale(ScalarValue));
#else
  vector_->Scale(ScalarValue);
#endif
}


template <typename T>
void Core::LinAlg::Vector<T>::dot(const Epetra_MultiVector& A, double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Dot(A, Result));
#else
  vector_->Dot(A, Result);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::abs(const Epetra_MultiVector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Abs(A));
#else
  vector_->Abs(A);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::scale(double ScalarA, const Epetra_MultiVector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Scale(ScalarA, A));
#else
  vector_->Scale(ScalarA, A);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::update(double ScalarA, const Epetra_MultiVector& A, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Update(ScalarA, A, ScalarThis));
#else
  vector_->Update(ScalarA, A, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::update(double ScalarA, const Epetra_MultiVector& A, double ScalarB,
    const Epetra_MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Update(ScalarA, A, ScalarB, B, ScalarThis));
#else
  vector_->Update(ScalarA, A, ScalarB, B, ScalarThis);
#endif
}


template <typename T>
void Core::LinAlg::Vector<T>::dot(const Vector& A, double* Result) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Dot(A, Result));
#else
  vector_->Dot(A, Result);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::abs(const Vector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Abs(A));
#else
  vector_->Abs(A);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::scale(double ScalarA, const Vector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Scale(ScalarA, A));
#else
  vector_->Scale(ScalarA, A);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::update(double ScalarA, const Vector& A, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Update(ScalarA, A, ScalarThis));
#else
  vector_->Update(ScalarA, A, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::update(
    double ScalarA, const Vector& A, double ScalarB, const Vector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Update(ScalarA, A, ScalarB, B.get_ref_of_epetra_vector(), ScalarThis));
#else
  vector_->Update(ScalarA, A, ScalarB, B.get_ref_of_epetra_vector(), ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::put_scalar(double ScalarConstant)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->PutScalar(ScalarConstant));
#else
  vector_->PutScalar(ScalarConstant);
#endif
}



template <typename T>
int Core::LinAlg::Vector<T>::replace_map(const Map& map)
{
  multi_vector_view_.invalidate();
  map_.invalidate();
  auto rv = vector_->ReplaceMap(map.get_epetra_block_map());
  return rv;
}

template <typename T>
void Core::LinAlg::Vector<T>::replace_local_value(int MyRow, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ReplaceMyValue(MyRow, 0, ScalarValue));
#else
  vector_->ReplaceMyValue(MyRow, 0, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::replace_local_values(
    int NumEntries, const double* Values, const int* Indices)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ReplaceMyValues(NumEntries, Values, Indices));
#else
  vector_->ReplaceMyValues(NumEntries, Values, Indices);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::replace_global_value(int GlobalRow, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ReplaceGlobalValue(GlobalRow, 0, ScalarValue));
#else
  vector_->ReplaceGlobalValue(GlobalRow, 0, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::replace_global_values(
    int NumEntries, const double* Values, const int* Indices)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ReplaceGlobalValues(NumEntries, Values, Indices));
#else
  vector_->ReplaceGlobalValues(NumEntries, Values, Indices);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::sum_into_local_value(int MyRow, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoMyValue(MyRow, 0, ScalarValue));
#else
  vector_->SumIntoMyValue(MyRow, 0, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::sum_into_global_value(int GlobalRow, double ScalarValue)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoGlobalValue(GlobalRow, 0, ScalarValue));
#else
  vector_->SumIntoGlobalValue(GlobalRow, 0, ScalarValue);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::sum_into_global_values(
    int NumEntries, const double* Values, const int* Indices)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->SumIntoGlobalValues(NumEntries, Values, Indices));
#else
  vector_->SumIntoGlobalValues(NumEntries, Values, Indices);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::multiply(char TransA, char TransB, double ScalarAB,
    const Epetra_MultiVector& A, const Epetra_MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Multiply(TransA, TransB, ScalarAB, A, B, ScalarThis));
#else
  vector_->Multiply(TransA, TransB, ScalarAB, A, B, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::multiply(
    double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Multiply(ScalarAB, A, B, ScalarThis));
#else
  vector_->Multiply(ScalarAB, A, B, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::reciprocal(const Epetra_MultiVector& A)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->Reciprocal(A));
#else
  vector_->Reciprocal(A);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::reciprocal_multiply(
    double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B, double ScalarThis)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(vector_->ReciprocalMultiply(ScalarAB, A, B, ScalarThis));
#else
  vector_->ReciprocalMultiply(ScalarAB, A, B, ScalarThis);
#endif
}

template <typename T>
void Core::LinAlg::Vector<T>::import(const Epetra_SrcDistObject& A,
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
void Core::LinAlg::Vector<T>::import(const Epetra_SrcDistObject& A,
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
void Core::LinAlg::Vector<T>::export_to(const Epetra_SrcDistObject& A,
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
void Core::LinAlg::Vector<T>::export_to(const Epetra_SrcDistObject& A,
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
std::unique_ptr<Core::LinAlg::Vector<T>> Core::LinAlg::Vector<T>::create_view(Epetra_Vector& view)
{
  std::unique_ptr<Vector<T>> ret(new Vector<T>);
  ret->vector_ = Utils::make_view(&view);
  return ret;
}

template <typename T>
std::unique_ptr<const Core::LinAlg::Vector<T>> Core::LinAlg::Vector<T>::create_view(
    const Epetra_Vector& view)
{
  std::unique_ptr<Vector<T>> ret(new Vector<T>);
  // We may const-cast here, since constness is restored inside the returned unique_ptr.
  ret->vector_ = Utils::make_view(const_cast<Epetra_Vector*>(&view));
  return ret;
}

template <typename T>
MPI_Comm Core::LinAlg::Vector<T>::get_comm() const
{
  return Core::Communication::unpack_epetra_comm(vector_->Comm());
}

template <typename T>
const Core::LinAlg::Map& Core::LinAlg::Vector<T>::get_map() const
{
  return map_.sync(vector_->Map());
}


// explicit instantiation
template class Core::LinAlg::Vector<double>;



Core::LinAlg::Vector<int>::Vector(const Map& map, bool zeroOut)
    : vector_(std::make_shared<Epetra_IntVector>(map.get_epetra_block_map(), zeroOut))
{
}


Core::LinAlg::Vector<int>::Vector(const Map& map, int* values)
    : vector_(std::make_shared<Epetra_IntVector>(
          Epetra_DataAccess::Copy, map.get_epetra_block_map(), values))
{
}

Core::LinAlg::Vector<int>::Vector(const Vector& other)
    : vector_(std::make_shared<Epetra_IntVector>(*other.vector_))
{
}

Core::LinAlg::Vector<int>::Vector(Vector&& other) noexcept : vector_(std::move(other.vector_)) {}

Core::LinAlg::Vector<int>& Core::LinAlg::Vector<int>::operator=(const Vector& other)
{
  vector_ = std::make_shared<Epetra_IntVector>(*other.vector_);
  return *this;
}

Core::LinAlg::Vector<int>& Core::LinAlg::Vector<int>::operator=(Vector&& other) noexcept
{
  vector_ = std::move(other.vector_);
  return *this;
}


void Core::LinAlg::Vector<int>::put_value(int Value)
{
  CHECK_EPETRA_CALL(vector_->PutValue(Value));
}

int Core::LinAlg::Vector<int>::max_value() { return vector_->MaxValue(); }

int Core::LinAlg::Vector<int>::min_value() { return vector_->MinValue(); }

void Core::LinAlg::Vector<int>::print(std::ostream& os) const { vector_->Print(os); }

void Core::LinAlg::Vector<int>::import(const Vector& A, const Core::LinAlg::Import& Importer,
    Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(
      vector_->Import(*A.vector_, Importer.get_epetra_import(), CombineMode, Indexor));
#else
  vector_->Import(*A.vector_, Importer.get_epetra_import(), CombineMode, Indexor);
#endif
}

void Core::LinAlg::Vector<int>::import(const Vector& A, const Core::LinAlg::Export& Exporter,
    Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(
      vector_->Import(*A.vector_, Exporter.get_epetra_export(), CombineMode, Indexor));
#else
  vector_->Import(*A.vector_, Exporter.get_epetra_export(), CombineMode, Indexor);
#endif
}

void Core::LinAlg::Vector<int>::export_to(const Vector& A, const Core::LinAlg::Import& Importer,
    Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(
      vector_->Export(*A.vector_, Importer.get_epetra_import(), CombineMode, Indexor));
#else
  vector_->Export(*A.vector_, Importer.get_epetra_import(), CombineMode, Indexor);
#endif
}

void Core::LinAlg::Vector<int>::export_to(const Vector& A, const Core::LinAlg::Export& Exporter,
    Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(
      vector_->Export(*A.vector_, Exporter.get_epetra_export(), CombineMode, Indexor));
#else
  vector_->Export(*A.vector_, Exporter.get_epetra_export(), CombineMode, Indexor);
#endif
}

MPI_Comm Core::LinAlg::Vector<int>::get_comm() const
{
  return Core::Communication::unpack_epetra_comm(vector_->Comm());
}


FOUR_C_NAMESPACE_CLOSE
