// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_MULTI_VECTOR_HPP
#define FOUR_C_LINALG_MULTI_VECTOR_HPP

#include "4C_config.hpp"

#include "4C_linalg_map.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_linalg_view.hpp"
#include "4C_utils_epetra_exceptions.hpp"
#include "4C_utils_owner_or_view.hpp"

#include <Epetra_FEVector.h>
#include <Epetra_MultiVector.h>
#include <mpi.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename T>
  class Vector;

  template <typename T>
  class MultiVector
  {
    static_assert(std::is_same_v<T, double>, "Only double is supported for now");

   public:
    /// Basic multi-vector constructor to create vector based on a map and initialize memory with
    /// zeros.
    explicit MultiVector(const Epetra_BlockMap& Map, int num_columns, bool zeroOut = true);

    explicit MultiVector(const Map& Map, int num_columns, bool zeroOut = true);

    explicit MultiVector(const Epetra_MultiVector& source);

    explicit MultiVector(const Epetra_FEVector& source);


    MultiVector(const MultiVector& other);

    MultiVector& operator=(const MultiVector& other);

    ~MultiVector() = default;

    // (Implicit) conversions: they all return references, never copies
    operator Epetra_MultiVector&() { return *vector_; }

    operator const Epetra_MultiVector&() const { return *vector_; }

    const Epetra_MultiVector& get_epetra_multi_vector() const { return *vector_; }

    Epetra_MultiVector& get_epetra_multi_vector() { return *vector_; }

    //! Compute 1-norm of each vector
    void norm_1(double* Result) const;

    //! Compute 2-norm of each vector
    void norm_2(double* Result) const;

    //! Compute Inf-norm of each vector
    void norm_inf(double* Result) const;

    //! Compute minimum value of each vector
    void min_value(double* Result) const;

    //! Compute maximum value of each vector
    void max_value(double* Result) const;

    //! Compute mean (average) value of each vector
    void mean_value(double* Result) const;

    //! Scale the current values of a multi-vector, \e this = ScalarValue*\e this.
    void scale(double ScalarValue);

    //! Computes dot product of each corresponding pair of vectors.
    void dot(const MultiVector& A, double* Result) const;

    //! Puts element-wise absolute values of input Multi-vector in target.
    void abs(const MultiVector& A);

    //! Replace multi-vector values with scaled values of A, \e this = ScalarA*A.
    void scale(double ScalarA, const MultiVector& A);

    //! Update multi-vector values with scaled values of A, \e this = ScalarThis*\e this +
    //! ScalarA*A.
    void update(double ScalarA, const MultiVector& A, double ScalarThis);

    //! Update multi-vector with scaled values of A and B, \e this = ScalarThis*\e this + ScalarA*A
    //! + ScalarB*B.
    void update(double ScalarA, const MultiVector& A, double ScalarB, const MultiVector& B,
        double ScalarThis);

    //! Initialize all values in a multi-vector with const value.
    void put_scalar(double ScalarConstant);

    //! Returns a view of the EpetraMap as type Map for this multi-vector.
    const Map& get_map() const { return map_.sync(vector_->Map()); };

    //! Returns the MPI_Comm for this multi-vector.
    [[nodiscard]] MPI_Comm get_comm() const;

    //! Returns true if this multi-vector is distributed global, i.e., not local replicated.
    bool is_distributed_global() const { return (vector_->Map().DistributedGlobal()); };

    //! Print method
    void print(std::ostream& os) const { vector_->Print(os); }

    //! Returns the number of vectors in the multi-vector.
    int num_vectors() const { return vector_->NumVectors(); }

    //! Returns the local vector length on the calling processor of vectors in the multi-vector.
    int local_length() const { return vector_->MyLength(); }

    //! Returns the global vector length of vectors in the multi-vector.
    int global_length() const { return vector_->GlobalLength(); }

    const double* get_values() const { return vector_->Values(); }

    double* get_values() { return vector_->Values(); }

    /**
     * Replace map, only if new map has same point-structure as current map.
     *
     * @warning This call may invalidate any views of this vector.
     *
     * @returns 0 if map is replaced, -1 if not.
     */
    int replace_map(const Map& map);

    void replace_local_value(int MyRow, int VectorIndex, double ScalarValue)
    {
      CHECK_EPETRA_CALL(vector_->ReplaceMyValue(MyRow, VectorIndex, ScalarValue));
    }

    void replace_global_value(int GlobalRow, int VectorIndex, double ScalarValue)
    {
      CHECK_EPETRA_CALL(vector_->ReplaceGlobalValue(GlobalRow, VectorIndex, ScalarValue));
    }

    void replace_global_value(long long GlobalRow, int VectorIndex, double ScalarValue)
    {
      CHECK_EPETRA_CALL(vector_->ReplaceGlobalValue(GlobalRow, VectorIndex, ScalarValue));
    }

    //! Matrix-Matrix multiplication, \e this = ScalarThis*\e this + ScalarAB*A*B.
    void multiply(char TransA, char TransB, double ScalarAB, const Epetra_MultiVector& A,
        const Epetra_MultiVector& B, double ScalarThis)
    {
      CHECK_EPETRA_CALL(vector_->Multiply(TransA, TransB, ScalarAB, A, B, ScalarThis));
    }

    //! Puts element-wise reciprocal values of input Multi-vector in target.
    void reciprocal(const Epetra_MultiVector& A) { CHECK_EPETRA_CALL(vector_->Reciprocal(A)); }

    //! Multiply a Core::LinAlg::MultiVector<double> with another, element-by-element.
    void multiply(double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B,
        double ScalarThis)
    {
      CHECK_EPETRA_CALL(vector_->Multiply(ScalarAB, A, B, ScalarThis));
    }

    //! Imports an Epetra_DistObject using the Core::LinAlg::Import object.
    void import(const Epetra_SrcDistObject& A, const Core::LinAlg::Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      CHECK_EPETRA_CALL(vector_->Import(A, Importer.get_epetra_import(), CombineMode, Indexor));
    }

    //! Imports an Epetra_DistObject using the Core::LinAlg::Export object.
    void import(const Epetra_SrcDistObject& A, const Core::LinAlg::Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      CHECK_EPETRA_CALL(vector_->Import(A, Exporter.get_epetra_export(), CombineMode, Indexor));
    }

    void export_to(const Epetra_SrcDistObject& A, const Core::LinAlg::Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      CHECK_EPETRA_CALL(vector_->Export(A, Importer.get_epetra_import(), CombineMode, Indexor));
    }

    void export_to(const Epetra_SrcDistObject& A, const Core::LinAlg::Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      CHECK_EPETRA_CALL(vector_->Export(A, Exporter.get_epetra_export(), CombineMode, Indexor));
    }

    void sum_into_global_value(int GlobalRow, int VectorIndex, double ScalarValue)
    {
      CHECK_EPETRA_CALL(vector_->SumIntoGlobalValue(GlobalRow, VectorIndex, ScalarValue));
    }

    void sum_into_global_value(long long GlobalRow, int VectorIndex, double ScalarValue)
    {
      CHECK_EPETRA_CALL(vector_->SumIntoGlobalValue(GlobalRow, VectorIndex, ScalarValue));
    }

    void reciprocal_multiply(double ScalarAB, const Epetra_MultiVector& A,
        const Epetra_MultiVector& B, double ScalarThis)
    {
      CHECK_EPETRA_CALL(vector_->ReciprocalMultiply(ScalarAB, A, B, ScalarThis));
    }

    void sum_into_local_value(int MyRow, int VectorIndex, double ScalarValue)
    {
      CHECK_EPETRA_CALL(vector_->SumIntoMyValue(MyRow, VectorIndex, ScalarValue));
    }

    void sum_into_local_value(
        int MyBlockRow, int BlockRowOffset, int VectorIndex, double ScalarValue)
    {
      CHECK_EPETRA_CALL(
          vector_->SumIntoMyValue(MyBlockRow, BlockRowOffset, VectorIndex, ScalarValue));
    }

    void extract_view(double*** ArrayOfPointers) const
    {
      CHECK_EPETRA_CALL(vector_->ExtractView(ArrayOfPointers));
    }

    void extract_copy(double* A, int MyLDA) const
    {
      CHECK_EPETRA_CALL(vector_->ExtractCopy(A, MyLDA));
    }

    Core::LinAlg::Vector<double>& operator()(int i);

    const Core::LinAlg::Vector<double>& operator()(int i) const;

    /**
     * View a given Epetra_MultiVector under our own MultiVector wrapper.
     */
    [[nodiscard]] static std::unique_ptr<MultiVector<T>> create_view(Epetra_MultiVector& view);

    [[nodiscard]] static std::unique_ptr<const MultiVector<T>> create_view(
        const Epetra_MultiVector& view);

   private:
    MultiVector() = default;

    //! The actual vector.
    Utils::OwnerOrView<Epetra_MultiVector> vector_;

    //! Vector view of the single columns stored inside the vector. This is used to allow
    //! access to the single columns of the MultiVector.
    mutable std::vector<View<Vector<T>>> column_vector_view_;

    mutable View<const Map> map_;

    friend class Vector<T>;
  };

  template <>
  struct EnableViewFor<Epetra_MultiVector>
  {
    using type = MultiVector<double>;
  };



}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE


#endif