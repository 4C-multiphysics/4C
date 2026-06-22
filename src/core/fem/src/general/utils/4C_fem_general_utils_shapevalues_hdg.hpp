// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_UTILS_SHAPEVALUES_HDG_HPP
#define FOUR_C_FEM_GENERAL_UTILS_SHAPEVALUES_HDG_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  /*!
  \brief helper data type to evaluate only nonzero element shape functions on face

   Author: schoeder 10/14
  */
  struct ShapeValuesInteriorOnFace
  {
    // shape function
    void shape(int numrows, int numcols)
    {
      matrix_.shape(numrows, numcols);
      isNonzero_.resize(numrows);
      return;
    }

    // for analog usage as a pure matrix
    double& operator()(const int row, const int col) { return matrix_(row, col); }

    // for analog usage as a pure matrix
    double operator()(const int row, const int col) const { return matrix_(row, col); }

    bool nonzero_on_face(const int index) const { return isNonzero_[index]; }

    Core::LinAlg::SerialDenseMatrix matrix_;
    std::vector<bool> isNonzero_;
  };

  /*!
   \brief helper holding the parameters, which we need to construct a the ShapeValuesFace
  */
  struct ShapeValuesFaceParams
  {
    explicit ShapeValuesFaceParams(
        unsigned int degree, bool completepoly, unsigned int quadraturedegree)
        : degree_(degree),
          completepoly_(completepoly),
          quadraturedegree_(quadraturedegree),
          face_(0)
    {
    }

    unsigned int degree_;
    bool completepoly_;
    unsigned int quadraturedegree_;
    unsigned int face_;

    /// convert the data structure to integers to simplify comparisons
    /// (rather than implementing operator < with many if statements)
    std::size_t to_int() const
    {
      FOUR_C_ASSERT(degree_ < 32, "Not implemented");
      FOUR_C_ASSERT(quadraturedegree_ < 64, "Not implemented");
      FOUR_C_ASSERT(face_ < 8, "Not implemented");
      // Simply encode the various integer values into a single number, all of
      // them take small numbers so this is easily possible. Beware about this when
      // adding further information, tough (might overflow when larger than 4 billion).
      return std::size_t(degree_) + 32 * completepoly_ + 64 * quadraturedegree_ + 4096 * face_;
    }
  };


  /// Helper class for evaluating HDG polynomials, geometry, etc.
  /*!

  */
  template <Core::FE::CellType distype>
  class ShapeValues
  {
   public:
    //! nen_: number of element nodes (T. Hughes: The Finite Element Method)
    static constexpr unsigned int nen_ = Core::FE::num_nodes(distype);

    //! number of space dimensions
    static constexpr unsigned int nsd_ = Core::FE::dim<distype>;

    ///! number of faces on element
    static constexpr unsigned int nfaces_ = Core::FE::num_faces<distype>;

    ///! number of nodes on faces
    static constexpr unsigned int nfn_ = Core::FE::DisTypeToNumNodePerFace<distype>::numNodePerFace;

    /*!
    \brief Initialize the data structure for shape functions, set element-independent data

    @param degree  Degree of the polynomial
    @param completepoly  Sets whether to use a polynomial of complete degree or tensor degree (on
    hex); on tet, only complete degree makes sense
    @param quadratureDegree  Sets the degree of polynomials the Gauss formula should integrate
    exactly; typically 2*degree
     */
    ShapeValues(
        const unsigned int degree, const bool completepoly, const unsigned int quadratureDegree);

    /*!
    \brief Compute element-dependent data, like gradients in real cell, integration weights, for
    element dofs
    */
    void evaluate(const Core::Elements::Element& ele, const std::vector<double>& aleDis = {});

    /// polynomial degree
    unsigned int degree_;

    /// underlying polynomial space for element interior, created in constructor
    std::shared_ptr<Core::FE::PolynomialSpace<nsd_>> polySpace_;

    /// scalar dofs per element
    unsigned int ndofs_;

    /// quadrature rule
    std::shared_ptr<Core::FE::GaussPoints> quadrature_;

    /// complete poly
    bool usescompletepoly_;

    /// number of integration points
    unsigned int nqpoints_;

    Core::LinAlg::SerialDenseMatrix
        funct;  /// values of mapping shape functions on all quadrature points
    Core::LinAlg::Matrix<nsd_, nen_>
        deriv;  /// gradients of mapping shape functions in unit coordinates
    Core::LinAlg::SerialDenseMatrix derxy;    /// gradients of mapping shape functions in real
                                              /// coordinates on all quadrature points
    Core::LinAlg::SerialDenseMatrix xyzreal;  /// coordinates of all quadrature points in real space
    Core::LinAlg::SerialDenseMatrix
        nodexyzunit;  /// coordinates of all node (support) points of Lagrange
                      /// basis functions in unit coordinates (all points at
                      /// cell center for Legendre-type polynomials)
    Core::LinAlg::SerialDenseMatrix
        nodexyzreal;  /// coordinates of all node (support) points of Lagrange basis functions in
                      /// real space (all points at cell center for Legendre-type polynomials)

    Core::LinAlg::SerialDenseMatrix
        shfunct;  /// evaluated HDG shape functions on all quadrature points
    Core::LinAlg::SerialDenseVector shfunctAvg;  /// average of shfunctF on cell
    Core::LinAlg::SerialDenseMatrix
        shderiv;  /// evaluated HDG shape function gradients in unit coordinates
    Core::LinAlg::SerialDenseMatrix
        shderxy;  /// evaluated HDG shape function gradients in real coordinates

    Core::LinAlg::Matrix<nsd_, 1> xsi;      /// quadrature points
    Core::LinAlg::Matrix<nsd_, nsd_> xjm;   /// Jacobi matrix of transformation
    Core::LinAlg::Matrix<nsd_, nsd_> xji;   /// inverse of Jacobi matrix of transformation
    Core::LinAlg::Matrix<nsd_, nen_> xyze;  /// element nodes
    Core::LinAlg::SerialDenseVector jfac;   /// Jacobian determinant times quadrature weight
  };

  /// Helper class for evaluating HDG polynomials, geometry, etc.
  /*!

  */
  template <Core::FE::CellType distype>
  class ShapeValuesFace
  {
   public:
    //! nen_: number of element nodes (T. Hughes: The Finite Element Method)
    static constexpr unsigned int nen_ = Core::FE::num_nodes(distype);

    ///! number of nodes on faces
    static constexpr unsigned int nfn_ = Core::FE::DisTypeToNumNodePerFace<distype>::numNodePerFace;

    //! number of space dimensions
    static constexpr unsigned int nsd_ = Core::FE::dim<distype>;

    ///! number of faces on element
    static constexpr unsigned int nfaces_ = Core::FE::num_faces<distype>;

    /*!
    \brief Constructor which does the things which do not need to be redone for the same
    parameters
    */
    ShapeValuesFace(ShapeValuesFaceParams params);

    /*!
    \brief Compute element-dependent data on faces, like integration weights, normal vectors,
    correctly oriented trace variables
    */
    void evaluate_face(const Core::Elements::Element& ele, const unsigned int face,
        const std::vector<double>& aleDis = {});

    /*!
    \brief Consider the orientation of faces for face degrees of freedom
     */
    void adjust_face_orientation(const Core::Elements::Element& ele, const unsigned int face);

    /// Parameters underlying this structure, necessary for evaluating element basis functions on
    /// faces
    ShapeValuesFaceParams params_;

    /// polynomial degree
    unsigned int degree_;

    /// underlying polynomial space for faces
    std::shared_ptr<Core::FE::PolynomialSpace<nsd_ - 1>> polySpace_;

    /// scalar dofs per face
    unsigned int nfdofs_;

    /// quadrature rule
    std::shared_ptr<Core::FE::GaussPoints> quadrature_;

    /// number of integration points on face
    unsigned int nqpoints_;

    Core::LinAlg::SerialDenseMatrix
        funct;  /// values of mapping shape functions on all face quadrature points
    Core::LinAlg::Matrix<nsd_ - 1, nfn_> deriv;  /// gradients of mapping shape functions on face
    Core::LinAlg::Matrix<nsd_ - 1, nsd_ - 1> metricTensor;  /// metric tensor on face
    Core::LinAlg::Matrix<nsd_, 1> normal;                   /// normal vector
    Core::LinAlg::Matrix<nsd_, nsd_ - 1>
        tangent;  /// Face reference frame wrt real one (face -> real)

    Core::LinAlg::SerialDenseMatrix
        xyzreal;  /// coordinates of face quadrature points in real space
    Core::LinAlg::SerialDenseMatrix
        nodexyzunit;  /// coordinates of all node (support) points of face Lagrange basis
                      /// functions in unit coordinates (invalid for Legendre-type polynomials)
    Core::LinAlg::SerialDenseMatrix
        nodexyzreal;  /// coordinates of all node (support) points of face Lagrange basis
                      /// functions in real space (invalid for Legendre-type polynomials)

    Core::LinAlg::SerialDenseMatrix
        shfunct;  /// evaluated shape functions for HDG face polynomials,
                  /// permuted to account for face orientation
    Core::LinAlg::SerialDenseMatrix shfunctNoPermute;  /// evaluated shape functions for HDG face
                                                       /// polynomials in natural ordering
    ShapeValuesInteriorOnFace
        shfunctI;  /// evaluated shape functions on face for interior HDG polynomials
    Core::LinAlg::SerialDenseMatrix
        normals;  /// normal vectors on a single face for all quadrature points

    Core::LinAlg::Matrix<nsd_ - 1, 1> xsi;  /// face quadrature points
    Core::LinAlg::Matrix<nsd_, nfn_> xyze;  /// face nodes
    Core::LinAlg::SerialDenseVector jfac;   /// face Jacobian determinant times quadrature weight

    std::vector<std::vector<int>> faceNodeOrder;  /// numbering of nodes belonging to faces

   private:
    Core::LinAlg::SerialDenseVector face_values_;  /// Evaluated basis functions on face

    /*!
    \brief Computes the face reference system considering the ordering of the target element

    The face reference system created is orthonormal and therefore the inverse of the
    transformationm matrix is just the transpose of the matrix (T)^{-1} = (T)'.

    \note
    The routine is only used in 3D so far and therefore tested only in this case.
     */
    void compute_face_reference_system(const Core::Elements::Element& ele, const unsigned int face);
  };

  /*!
  \brief Cache for the face-based shape values, so we do not need to calculate again

   Author: schoeder 08/14
  */
  template <Core::FE::CellType distype>
  class ShapeValuesFaceCache
  {
   public:
    /// return instance
    static ShapeValuesFaceCache<distype>& instance();

    /// give pointer to corresponding shape values face
    std::shared_ptr<ShapeValuesFace<distype>> create(ShapeValuesFaceParams params);

   private:
    ShapeValuesFaceCache() = default;

    /// the actual cache; contains all pairs
    std::map<std::size_t, std::shared_ptr<ShapeValuesFace<distype>>> svf_cache_;
  };

  /*!
  \brief Cache for the interior values of shape velocities

   Author: kronbichler 09/14
  */
  template <Core::FE::CellType distype>
  class ShapeValuesInteriorOnFaceCache
  {
   public:
    /// return instance
    static ShapeValuesInteriorOnFaceCache& instance();

    /// give pointer to corresponding shape values face
    std::shared_ptr<ShapeValuesInteriorOnFace> create(ShapeValuesFaceParams params);

   private:
    ShapeValuesInteriorOnFaceCache() = default;

    /// the actual cache; contains all pairs
    std::map<std::size_t, std::shared_ptr<ShapeValuesInteriorOnFace>> cache_;
  };

}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
