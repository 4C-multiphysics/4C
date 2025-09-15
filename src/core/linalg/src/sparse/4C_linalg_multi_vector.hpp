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
#include "4C_utils_owner_or_view.hpp"

#include <Epetra_FEVector.h>
#include <Epetra_MultiVector.h>
#include <mpi.h>

#include <memory>



// Do not lint the file for identifier names, since the naming of the Wrapper functions follow the
// naming of the Core::LinAlg::MultiVector<double>

// NOLINTBEGIN(readability-identifier-naming)

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
    int Norm1(double* Result) const;

    //! Compute 2-norm of each vector
    int Norm2(double* Result) const;

    //! Compute Inf-norm of each vector
    int NormInf(double* Result) const;

    //! Compute minimum value of each vector
    int MinValue(double* Result) const;

    //! Compute maximum value of each vector
    int MaxValue(double* Result) const;

    //! Compute mean (average) value of each vector
    int MeanValue(double* Result) const;

    //! Scale the current values of a multi-vector, \e this = ScalarValue*\e this.
    int Scale(double ScalarValue) { return vector_->Scale(ScalarValue); }

    //! Computes dot product of each corresponding pair of vectors.
    int Dot(const MultiVector& A, double* Result) const;

    //! Puts element-wise absolute values of input Multi-vector in target.
    int Abs(const MultiVector& A);

    //! Replace multi-vector values with scaled values of A, \e this = ScalarA*A.
    int Scale(double ScalarA, const MultiVector& A);

    //! Update multi-vector values with scaled values of A, \e this = ScalarThis*\e this +
    //! ScalarA*A.
    int Update(double ScalarA, const MultiVector& A, double ScalarThis);

    //! Update multi-vector with scaled values of A and B, \e this = ScalarThis*\e this + ScalarA*A
    //! + ScalarB*B.
    int Update(double ScalarA, const MultiVector& A, double ScalarB, const MultiVector& B,
        double ScalarThis);

    //! Initialize all values in a multi-vector with const value.
    int PutScalar(double ScalarConstant);

    //! Returns a view of the EpetraMap as type Map for this multi-vector.
    const Map& get_map() const { return map_.sync(vector_->Map()); };

    //! Returns the MPI_Comm for this multi-vector.
    [[nodiscard]] MPI_Comm Comm() const;

    //! Returns true if this multi-vector is distributed global, i.e., not local replicated.
    bool DistributedGlobal() const { return (vector_->Map().DistributedGlobal()); };

    //! Print method
    void Print(std::ostream& os) const { vector_->Print(os); }

    //! Returns the number of vectors in the multi-vector.
    int NumVectors() const { return vector_->NumVectors(); }

    //! Returns the local vector length on the calling processor of vectors in the multi-vector.
    int MyLength() const { return vector_->MyLength(); }

    //! Returns the global vector length of vectors in the multi-vector.
    int GlobalLength() const { return vector_->GlobalLength(); }

    int ReplaceMyValue(int MyRow, int VectorIndex, double ScalarValue)
    {
      return vector_->ReplaceMyValue(MyRow, VectorIndex, ScalarValue);
    }

    const double* Values() const { return vector_->Values(); }
    double* Values() { return vector_->Values(); }

    /**
     * Replace map, only if new map has same point-structure as current map.
     *
     * @warning This call may invalidate any views of this vector.
     *
     * @returns 0 if map is replaced, -1 if not.
     */
    int ReplaceMap(const Map& map);

    int ReplaceGlobalValue(int GlobalRow, int VectorIndex, double ScalarValue)
    {
      return vector_->ReplaceGlobalValue(GlobalRow, VectorIndex, ScalarValue);
    }

    int ReplaceGlobalValue(long long GlobalRow, int VectorIndex, double ScalarValue)
    {
      return vector_->ReplaceGlobalValue(GlobalRow, VectorIndex, ScalarValue);
    }

    //! Matrix-Matrix multiplication, \e this = ScalarThis*\e this + ScalarAB*A*B.
    int Multiply(char TransA, char TransB, double ScalarAB, const Epetra_MultiVector& A,
        const Epetra_MultiVector& B, double ScalarThis)
    {
      return vector_->Multiply(TransA, TransB, ScalarAB, A, B, ScalarThis);
    }

    //! Puts element-wise reciprocal values of input Multi-vector in target.
    int Reciprocal(const Epetra_MultiVector& A) { return vector_->Reciprocal(A); }

    //! Multiply a Core::LinAlg::MultiVector<double> with another, element-by-element.
    int Multiply(double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B,
        double ScalarThis)
    {
      return vector_->Multiply(ScalarAB, A, B, ScalarThis);
    }

    //! Imports an Epetra_DistObject using the Core::LinAlg::Import object.
    int Import(const Epetra_SrcDistObject& A, const Core::LinAlg::Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Import(A, Importer.get_epetra_import(), CombineMode, Indexor);
    }

    //! Imports an Epetra_DistObject using the Core::LinAlg::Export object.
    int Import(const Epetra_SrcDistObject& A, const Core::LinAlg::Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Import(A, Exporter.get_epetra_export(), CombineMode, Indexor);
    }

    int Export(const Epetra_SrcDistObject& A, const Core::LinAlg::Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Export(A, Importer.get_epetra_import(), CombineMode, Indexor);
    }

    int Export(const Epetra_SrcDistObject& A, const Core::LinAlg::Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Export(A, Exporter.get_epetra_export(), CombineMode, Indexor);
    }

    int SumIntoGlobalValue(int GlobalRow, int VectorIndex, double ScalarValue)
    {
      return vector_->SumIntoGlobalValue(GlobalRow, VectorIndex, ScalarValue);
    }

    int SumIntoGlobalValue(long long GlobalRow, int VectorIndex, double ScalarValue)
    {
      return vector_->SumIntoGlobalValue(GlobalRow, VectorIndex, ScalarValue);
    }

    int ReciprocalMultiply(double ScalarAB, const Epetra_MultiVector& A,
        const Epetra_MultiVector& B, double ScalarThis)
    {
      return vector_->ReciprocalMultiply(ScalarAB, A, B, ScalarThis);
    }

    int SumIntoMyValue(int MyRow, int VectorIndex, double ScalarValue)
    {
      return vector_->SumIntoMyValue(MyRow, VectorIndex, ScalarValue);
    }


    int SumIntoMyValue(int MyBlockRow, int BlockRowOffset, int VectorIndex, double ScalarValue)
    {
      return vector_->SumIntoMyValue(MyBlockRow, BlockRowOffset, VectorIndex, ScalarValue);
    }

    int ExtractView(double*** ArrayOfPointers) const
    {
      return vector_->ExtractView(ArrayOfPointers);
    }

    int ExtractCopy(double* A, int MyLDA) const { return vector_->ExtractCopy(A, MyLDA); }

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

// NOLINTEND(readability-identifier-naming)

#endif