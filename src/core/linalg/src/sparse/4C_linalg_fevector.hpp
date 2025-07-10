// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_FEVECTOR_HPP
#define FOUR_C_LINALG_FEVECTOR_HPP


#include "4C_config.hpp"

#include "4C_linalg_map.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_linalg_view.hpp"

#include <Epetra_FEVector.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{

  // Sparse FEVector which will replace the Epetra_FEVector
  template <typename T>
  class FEVector
  {
    static_assert(std::is_same_v<T, double>, "Only double is supported for now");

   public:
    /// Basic vector constructor to create vector based on a map and initialize memory with zeros
    explicit FEVector(const Epetra_BlockMap& Map, bool zeroOut = true);

    explicit FEVector(const Epetra_BlockMap& Map, int numVectors, bool ignoreNonLocalEntries);

    explicit FEVector(const Map& Map, bool zeroOut = true);

    /// Copy constructor from epetra to vector
    explicit FEVector(const Epetra_FEVector& Source);

    explicit FEVector(const Epetra_Vector& Source);


    // Rule of five: We currently need to take care to make a deep copy of the Epetra_FEVector.
    FEVector(const FEVector& other);

    FEVector& operator=(const FEVector& other);

    ~FEVector() = default;

    //! Element access function
    double& operator[](int index) { return *(*vector_)[index]; }

    double operator[](int const index) const { return *(*vector_)[index]; }
    // Implicit conversion to MultiVector: the MultiVector will view the same content and only have
    // a single column.
    operator const MultiVector<T>&() const;
    operator MultiVector<T>&();

    // Explicit conversion to MultiVector: the MultiVector will view the same content and only have
    // a single column.
    const MultiVector<T>& as_multi_vector() const;
    MultiVector<T>& as_multi_vector();

    // (Implicit) conversions: they all return references or RCPs, never copies
    const Epetra_FEVector& get_ref_of_epetra_fevector() const { return *vector_; }

    Epetra_FEVector& get_ref_of_epetra_fevector() { return *vector_; }

    operator Epetra_MultiVector&() { return *vector_; }

    operator const Epetra_MultiVector&() const { return *vector_; }

    operator Epetra_FEVector&() { return *vector_; }

    operator const Epetra_FEVector&() const { return *vector_; }

    Epetra_Vector* operator()(int i)
    {
      return (*vector_)(i);  // Forward to underlying Epetra_FEVector
    }

    const Epetra_Vector* operator()(int i) const
    {
      return (*vector_)(i);  // Forward to const version
    }


    //! Computes dot product of each corresponding pair of vectors.
    int dot(const Epetra_MultiVector& A, double* Result) const;

    //! Puts element-wise absolute values of input Multi-vector in target.
    int abs(const Epetra_MultiVector& A);

    //! Replace multi-vector values with scaled values of A, \e this = ScalarA*A.
    int scale(double ScalarA, const Epetra_MultiVector& A);

    //! Update multi-vector values with scaled values of A, \e this = ScalarThis*\e this +
    //! ScalarA*A.
    int update(double ScalarA, const Epetra_MultiVector& A, double ScalarThis);

    //! Update multi-vector with scaled values of A and B, \e this = ScalarThis*\e this + ScalarA*A
    //! + ScalarB*B.
    int update(double ScalarA, const Epetra_MultiVector& A, double ScalarB,
        const Epetra_MultiVector& B, double ScalarThis);


    ///

    //! Compute 1-norm of each vector
    int norm_1(double* Result) const;

    //! Compute 2-norm of each vector
    int norm_2(double* Result) const;

    //! Compute Inf-norm of each vector
    int norm_inf(double* Result) const;

    //! Compute minimum value of each vector
    int min_value(double* Result) const;

    //! Compute maximum value of each vector
    int max_value(double* Result) const;

    //! Compute mean (average) value of each vector
    int mean_value(double* Result) const;

    //! Scale the current values of a multi-vector, \e this = ScalarValue*\e this.
    int scale(double ScalarValue) { return vector_->Scale(ScalarValue); }

    //! Computes dot product of each corresponding pair of vectors.
    int dot(const FEVector& A, double* Result) const;

    //! Puts element-wise absolute values of input Multi-vector in target.
    int abs(const FEVector& A);

    //! Replace multi-vector values with scaled values of A, \e this = ScalarA*A.
    int scale(double ScalarA, const FEVector& A);

    //! Update multi-vector values with scaled values of A, \e this = ScalarThis*\e this +
    //! ScalarA*A.
    int update(double ScalarA, const FEVector& A, double ScalarThis);

    //! Update multi-vector with scaled values of A and B, \e this = ScalarThis*\e this + ScalarA*A
    //! + ScalarB*B.
    int update(
        double ScalarA, const FEVector& A, double ScalarB, const FEVector& B, double ScalarThis);

    //! Initialize all values in a multi-vector with const value.
    int put_scalar(double ScalarConstant);

    //! Returns the address of the Core::LinAlg::Map for this multi-vector.
    const Map& get_map() const;

    //! Returns the MPI_Comm for this multi-vector.
    MPI_Comm get_comm() const;

    //! Returns true if this multi-vector is distributed global, i.e., not local replicated.
    bool distributed_global() const { return (vector_->Map().DistributedGlobal()); };

    //! Print method
    void print(std::ostream& os) const { vector_->Print(os); }

    //! Returns the number of vectors in the multi-vector.
    int num_vectors() const { return vector_->NumVectors(); }

    //! Returns the local vector length on the calling processor of vectors in the multi-vector.
    int local_length() const { return vector_->MyLength(); }

    //! Returns the global vector length of vectors in the multi-vector.
    int global_length() const { return vector_->GlobalLength(); }

    int replace_local_value(int MyRow, int FEVectorIndex, double ScalarValue)
    {
      return vector_->ReplaceMyValue(MyRow, FEVectorIndex, ScalarValue);
    }

    double* get_values() const { return vector_->Values(); }

    /**
     * Replace map, only if new map has same point-structure as current map.
     *
     * @warning This call may invalidate any views of this vector.
     *
     * @returns 0 if map is replaced, -1 if not.
     */
    int replace_map(const Map& map);

    int replace_global_value(int GlobalRow, int FEVectorIndex, double ScalarValue)
    {
      return vector_->ReplaceGlobalValue(GlobalRow, FEVectorIndex, ScalarValue);
    }


    int replace_global_values(
        int numIDs, const int* GIDs, const double* values, int vectorIndex = 0)
    {
      return vector_->ReplaceGlobalValues(numIDs, GIDs, values, vectorIndex);
    }
    //! Matrix-Matrix multiplication, \e this = ScalarThis*\e this + ScalarAB*A*B.
    int multiply(char TransA, char TransB, double ScalarAB, const Epetra_MultiVector& A,
        const Epetra_MultiVector& B, double ScalarThis)
    {
      return vector_->Multiply(TransA, TransB, ScalarAB, A, B, ScalarThis);
    }

    //! Puts element-wise reciprocal values of input Multi-vector in target.
    int reciprocal(const Epetra_MultiVector& A) { return vector_->Reciprocal(A); }

    //! Multiply a Core::LinAlg::MultiVector<double> with another, element-by-element.
    int multiply(double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B,
        double ScalarThis)
    {
      return vector_->Multiply(ScalarAB, A, B, ScalarThis);
    }

    //! Imports an Epetra_DistObject using the Core::LinAlg::Import object.
    int import(const Epetra_SrcDistObject& A, const Core::LinAlg::Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Import(A, Importer.get_epetra_import(), CombineMode, Indexor);
    }

    //! Imports an Epetra_DistObject using the Epetra_Export object.
    int import(const Epetra_SrcDistObject& A, const Epetra_Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Import(A, Exporter, CombineMode, Indexor);
    }

    int export_to(const Epetra_SrcDistObject& A, const Core::LinAlg::Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Export(A, Importer.get_epetra_import(), CombineMode, Indexor);
    }

    int export_to(const Epetra_SrcDistObject& A, const Epetra_Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Export(A, Exporter, CombineMode, Indexor);
    }

    int sum_into_global_value(int GlobalRow, int FEVectorIndex, double ScalarValue)
    {
      return vector_->SumIntoGlobalValue(GlobalRow, FEVectorIndex, ScalarValue);
    }

    int global_assemble(Epetra_CombineMode mode = Add, bool reuse_map_and_exporter = false)
    {
      return vector_->GlobalAssemble(mode, reuse_map_and_exporter);
    };

    int sum_into_global_value(long long GlobalRow, int FEVectorIndex, double ScalarValue)
    {
      return vector_->SumIntoGlobalValue(GlobalRow, FEVectorIndex, ScalarValue);
    }

    int sum_into_global_values(int numIDs, const int* GIDs, const int* numValuesPerID,
        const double* values, int vectorIndex = 0)
    {
      return vector_->SumIntoGlobalValues(numIDs, GIDs, numValuesPerID, values, vectorIndex);
    }

    int sum_into_global_values(
        int numIDs, const int* GIDs, const double* values, int vectorIndex = 0)
    {
      return vector_->SumIntoGlobalValues(numIDs, GIDs, values, vectorIndex);
    }


    int reciprocal_multiply(double ScalarAB, const Epetra_MultiVector& A,
        const Epetra_MultiVector& B, double ScalarThis)
    {
      return vector_->ReciprocalMultiply(ScalarAB, A, B, ScalarThis);
    }

    int sum_into_local_value(int MyRow, int FEVectorIndex, double ScalarValue)
    {
      return vector_->SumIntoMyValue(MyRow, FEVectorIndex, ScalarValue);
    }


    int sum_into_local_value(
        int MyBlockRow, int BlockRowOffset, int FEVectorIndex, double ScalarValue)
    {
      return vector_->SumIntoMyValue(MyBlockRow, BlockRowOffset, FEVectorIndex, ScalarValue);
    }
    /**
     * View a given Epetra_FEVector object under our own FEVector wrapper.
     */
    [[nodiscard]] static std::unique_ptr<FEVector<T>> create_view(Epetra_FEVector& view);

    [[nodiscard]] static std::unique_ptr<const FEVector<T>> create_view(
        const Epetra_FEVector& view);


   private:
    FEVector() = default;

    //! The actual Epetra_FEVector object.
    Utils::OwnerOrView<Epetra_FEVector> vector_;

    //! Map from Epetra_FEVector
    mutable View<const Map> map_;

    //! MultiVector view of the FEVector. This is used to allow implicit conversion to MultiVector.
    mutable View<MultiVector<T>> multi_vector_view_;

    friend class MultiVector<T>;
  };

}  // namespace Core::LinAlg



FOUR_C_NAMESPACE_CLOSE


#endif