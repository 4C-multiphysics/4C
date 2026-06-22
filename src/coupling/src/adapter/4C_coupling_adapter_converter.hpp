// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_COUPLING_ADAPTER_CONVERTER_HPP
#define FOUR_C_COUPLING_ADAPTER_CONVERTER_HPP

/*----------------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"

#include <map>
#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/* forward declarations */
namespace Coupling::Adapter
{
  class Coupling;
}

/*----------------------------------------------------------------------------*/
/* definition of classes */
namespace Coupling::Adapter
{
  /*! \class CouplingConverter
   *  \brief Abstract converter base for target/source conversion of data
   *
   *  The point is that many generic coupling algorithms that transfer data
   *  between target and source might be used in both directions. These
   *  algorithms can utilize a Converter to enable use in both directions.
   */
  class CouplingConverter
  {
   public:
    virtual ~CouplingConverter() = default;
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> src_to_dst(
        std::shared_ptr<const Core::LinAlg::Vector<double>> s) const = 0;

    virtual std::shared_ptr<Core::LinAlg::Vector<double>> dst_to_src(
        std::shared_ptr<const Core::LinAlg::Vector<double>> d) const = 0;

    virtual std::shared_ptr<const Core::LinAlg::Map> src_map() const = 0;

    virtual std::shared_ptr<const Core::LinAlg::Map> dst_map() const = 0;

    virtual std::shared_ptr<const Core::LinAlg::Map> perm_src_map() const = 0;

    virtual std::shared_ptr<const Core::LinAlg::Map> perm_dst_map() const = 0;

    virtual void fill_src_to_dst_map(std::map<int, int>& rowmap) const = 0;
  };

  /// target to source converter
  class CouplingTargetConverter : public CouplingConverter
  {
   public:
    explicit CouplingTargetConverter(const Coupling& coup) : coup_(coup) {}

    std::shared_ptr<Core::LinAlg::Vector<double>> src_to_dst(
        std::shared_ptr<const Core::LinAlg::Vector<double>> s) const override;

    std::shared_ptr<Core::LinAlg::Vector<double>> dst_to_src(
        std::shared_ptr<const Core::LinAlg::Vector<double>> d) const override;

    std::shared_ptr<const Core::LinAlg::Map> src_map() const override;

    std::shared_ptr<const Core::LinAlg::Map> dst_map() const override;

    std::shared_ptr<const Core::LinAlg::Map> perm_src_map() const override;

    std::shared_ptr<const Core::LinAlg::Map> perm_dst_map() const override;

    void fill_src_to_dst_map(std::map<int, int>& rowmap) const override;

   private:
    const Coupling& coup_;
  };

  /// source to target converter
  class CouplingSourceConverter : public CouplingConverter
  {
   public:
    explicit CouplingSourceConverter(const Coupling& coup) : coup_(coup) {}

    std::shared_ptr<Core::LinAlg::Vector<double>> src_to_dst(
        std::shared_ptr<const Core::LinAlg::Vector<double>> s) const override;

    std::shared_ptr<Core::LinAlg::Vector<double>> dst_to_src(
        std::shared_ptr<const Core::LinAlg::Vector<double>> d) const override;

    std::shared_ptr<const Core::LinAlg::Map> src_map() const override;

    std::shared_ptr<const Core::LinAlg::Map> dst_map() const override;

    std::shared_ptr<const Core::LinAlg::Map> perm_src_map() const override;

    std::shared_ptr<const Core::LinAlg::Map> perm_dst_map() const override;

    void fill_src_to_dst_map(std::map<int, int>& rowmap) const override;

   private:
    const Coupling& coup_;
  };
  /// extract submatrix of the src map and transform it to a new col map
  /*!

  Monolithic multiphysics add matrices from different fields at the interface. These matrices
      belong to different row maps. Thus adding them requires moving one of them to a new row map.
  The relations between these maps are managed by Adapter::Coupling objects. In a parallel setting
              there is a target and a source side (in case of matrix transformations we use source
  and destination abstraction via Adapter::CouplingConverter). The parallel distribution of both is
  arbitrary. And in addition there are permuted target and source maps, that match the respective
      other side. So the row map transformation requires a parallel redistribution followed by a row
  map exchange.

      This operator does not utilize Core::LinAlg::SparseMatrix::replace_row_map() but
      copies the temporary matrix. This is required both to preserve the
      internal Core::LinAlg::SparseMatrix from the destination Core::LinAlg::SparseMatrix and
      because the destination matrix row map may be much larger than the source
      matrix row map.

      The operator is meant to be usable on its own and operate on both row and
      column transformations (if the respective converter is given).

      An additional feature of this class is that it can assign matrix blocks
      from one field to block matrix slots on another field. As opposed to
      MatrixColTransform, this method extracts a logical block from the input
              matrix without any split call.

  \note All matrix transformation operators work with filled and unfilled
              destination matrices. The source matrix is never changed. The destination
              matrix is not reallocated and its filled state is not explicitly
              changed. There is a Core::LinAlg::SparseMatrix::Zero() call if addmatrix==false
          and this can reset the filled state if the matrix graph is not preserved
              by the Core::LinAlg::SparseMatrix object.


  \sa MatrixColTransform, MatrixRowTransform, MatrixRowColTransform

                              */
  class MatrixLogicalSplitAndTransform
  {
   public:
    /// construct
    MatrixLogicalSplitAndTransform() : havegidmap_(false) {}

    /// transformation operation
    /*!
      The call operator to be used for a matrix data copy between \p src and
      \p dst matrices.

      \param src            (i) source matrix
      \param logical_range_map  (i) sub-map that defines a logical block matrix row within src
      \param logical_domain_map (i) sub-map that defines a logical block matrix column within src
      \param scale          (i) scaling factor to be applied
      \param row_converter  (i) src to dst abstraction on Adapter::Coupling, if nullptr: no row
      transform is done \param col_converter  (i) src to dst abstraction on Adapter::Coupling,
      if nullptr: no column transform is done \param dst          (i/o) destination matrix \param
      exactmatch     (i) do not drop any source values if true \param addmatrix      (i) remove
      current dst values if false
     */
    bool operator()(const Core::LinAlg::SparseMatrix& src,
        const Core::LinAlg::Map& logical_range_map, const Core::LinAlg::Map& logical_domain_map,
        const double scale, const CouplingConverter* row_converter,
        const CouplingConverter* col_converter, Core::LinAlg::SparseMatrix& dst,
        bool exactmatch = true, bool addmatrix = false);

   private:
    /// setup column map matching between source and destination gids
    /*!
      Internal method.
     */
    void setup_gid_map(const Core::LinAlg::Map& rowmap, const Core::LinAlg::Map& colmap,
        const CouplingConverter* converter, MPI_Comm comm);

    /// copy values from source to destination matrix
    /*!
      Internal method.
     */
    void internal_add(Core::LinAlg::SparseMatrix& esrc, const Core::LinAlg::Map& logical_range_map,
        const Core::LinAlg::Map& logical_domain_map, const Core::LinAlg::Map& matching_dst_rows,
        Core::LinAlg::SparseMatrix& edst, bool exactmatch, double scale);

    /// fast method that adds into filled matrices
    /*!
      Internal method called by internal_add.
     */
    void add_into_filled(Core::LinAlg::SparseMatrix& esrc,
        const Core::LinAlg::Map& logical_range_map, const Core::LinAlg::Map& logical_domain_map,
        const Core::LinAlg::Vector<double>& selector, const Core::LinAlg::Map& matching_dst_rows,
        Core::LinAlg::SparseMatrix& edst, bool exactmatch, double scale);

    /// slow method that adds into unfilled matrices
    /*!
      Internal method called by internal_add.
     */
    void add_into_unfilled(Core::LinAlg::SparseMatrix& esrc,
        const Core::LinAlg::Map& logical_range_map, const Core::LinAlg::Map& logical_domain_map,
        const Core::LinAlg::Vector<double>& selector, const Core::LinAlg::Map& matching_dst_rows,
        Core::LinAlg::SparseMatrix& edst, bool exactmatch, double scale);

    /// source and destination gid matching
    std::map<int, int> gidmap_;

    /// setup done flag
    bool havegidmap_;

    /// localized version of gidmap_
    std::vector<int> lidvector_;

    /// exporter to communicate matrix to new row map
    std::shared_ptr<Core::LinAlg::Export> exporter_;
  };



  /// communicate matrix to new row map
  /*!

    Monolithic multiphysics add matrices from different fields at the interface. These matrices
  belong to different row maps. Thus adding them requires moving one of them to a new row map. The
  relations between these maps are managed by Adapter::Coupling objects. In a parallel setting
  there is a target and a source side (in case of matrix transformations we use source and
  destination abstraction via Adapter::CouplingConverter). The parallel distribution of both
  is arbitrary. And in addition there are permuted target and source maps, that match the respective
  other side. So the row map transformation requires a parallel redistribution followed by a row map
  exchange.

  This operator does not utilize Core::LinAlg::SparseMatrix::replace_row_map() but
  copies the temporary matrix. This is required both to preserve the
  internal Core::LinAlg::SparseMatrix from the destination Core::LinAlg::SparseMatrix and
  because the destination matrix row map may be much larger than the source
  matrix row map.

    The operator is meant to be usable on its own or as part of the composed
    MatrixRowColTransform operator.

    \note The implementation is done by MatrixLogicalSplitAndTransform

    \note All matrix transformation operators work with filled and unfilled
    destination matrices. The source matrix is never changed. The destination
    matrix is not reallocated and its filled state is not explicitly
    changed. There is a Core::LinAlg::SparseMatrix::Zero() call if addmatrix==false
    and this can reset the filled state if the matrix graph is not preserved
    by the Core::LinAlg::SparseMatrix object.

    \sa MatrixLogicalSplitAndTransform, MatrixColTransform, MatrixRowColTransform

   */
  class MatrixRowTransform
  {
   public:
    /// transformation operation
    /*!
      The call operator to be used for a matrix data copy between \p src and
      \p dst matrices.

      \param src       (i) source matrix
      \param scale     (i) scaling factor to be applied
      \param converter (i) src to dst abstraction on Adapter::Coupling
      \param dst     (i/o) destination matrix
      \param addmatrix (i) remove current dst values if false
     */
    bool operator()(const Core::LinAlg::SparseMatrix& src, double scale,
        const CouplingConverter& converter, Core::LinAlg::SparseMatrix& dst,
        bool addmatrix = false);

   private:
    /// object that does the actual work
    MatrixLogicalSplitAndTransform transformer_;
  };


  /// communicate matrix to new col map
  /*!

    Monolithic multifields need to assign matrix blocks from one field to
    block matrix slots belonging to another field. For some matrix blocks the
    row map stays the same but the column map changes.

    A special point here is that the source matrix column might include more
    values than the respective destination matrix column, e.g. for
    fluid matrices that include velocity and pressure values whereas the
    corresponding ale matrices just contain displacement values. In such a
    case it is possible to advice the transformation to drop the pressure
    values (\p exactmatch=false ). By default additional values raise a
    \p FOUR_C_THROW .

    \note The implementation is done by MatrixLogicalSplitAndTransform

    \note All matrix transformation operators work with filled and unfilled
    destination matrices. The source matrix is never changed. The destination
    matrix is not reallocated and its filled state is not explicitly
    changed. There is a Core::LinAlg::SparseMatrix::Zero() call if addmatrix==false
    and this can reset the filled state if the matrix graph is not preserved
    by the Core::LinAlg::SparseMatrix object.


    \sa MatrixLogicalSplitAndTransform, MatrixRowTransform, MatrixRowColTransform

   */
  class MatrixColTransform
  {
   public:
    /// transformation operation
    /*!
      The call operator to be used for a matrix data copy between \p src and
      \p dst matrices.

      \param rowmap    (i) row map of full source block matrix
      \param colmap    (i) col map of full source block matrix
      \param src       (i) source matrix
      \param scale     (i) scaling factor to be applied
      \param converter (i) src to dst abstraction on Adapter::Coupling
      \param dst     (i/o) destination matrix
      \param exactmatch (i) do not drop any source values if true
      \param addmatrix (i) remove current dst values if false
     */
    bool operator()(const Core::LinAlg::Map& rowmap, const Core::LinAlg::Map& colmap,
        const Core::LinAlg::SparseMatrix& src, double scale, const CouplingConverter& converter,
        Core::LinAlg::SparseMatrix& dst, bool exactmatch = true, bool addmatrix = false);

   private:
    /// object that does the actual work
    MatrixLogicalSplitAndTransform transformer_;
  };


  /// communicate matrix to new row map and col map
  /*!

    A combined row and column map exchange between source and destination
    matrix.

    \note The implementation is done by MatrixLogicalSplitAndTransform

    \note All matrix transformation operators work with filled and unfilled
    destination matrices. The source matrix is never changed. The destination
    matrix is not reallocated and its filled state is not explicitly
    changed. There is a Core::LinAlg::SparseMatrix::Zero() call if addmatrix==false
    and this can reset the filled state if the matrix graph is not preserved
    by the Core::LinAlg::SparseMatrix object.

    \sa MatrixLogicalSplitAndTransform, MatrixRowTransform, MatrixColTransform

   */
  class MatrixRowColTransform
  {
   public:
    /// transformation operation
    /*!
      The call operator to be used for a matrix data copy between \p src and
      \p dst matrices.

      \param src          (i) source matrix
      \param scale        (i) scaling factor to be applied
      \param rowconverter (i) src to dst abstraction on Adapter::Coupling
      \param colconverter (i) src to dst abstraction on Adapter::Coupling
      \param dst        (i/o) destination matrix
      \param exactmatch   (i) do not drop any source values if true
      \param addmatrix    (i) remove current dst values if false
     */
    bool operator()(const Core::LinAlg::SparseMatrix& src, double scale,
        const CouplingConverter& rowconverter, const CouplingConverter& colconverter,
        Core::LinAlg::SparseMatrix& dst, bool exactmatch = true, bool addmatrix = false);

   private:
    /// object that does the actual work
    MatrixLogicalSplitAndTransform transformer_;
  };

}  // namespace Coupling::Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
