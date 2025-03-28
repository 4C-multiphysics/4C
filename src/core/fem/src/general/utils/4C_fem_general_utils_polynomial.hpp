// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_UTILS_POLYNOMIAL_HPP
#define FOUR_C_FEM_GENERAL_UTILS_POLYNOMIAL_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_serialdensevector.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

#include <numeric>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  /*!
   \brief helper holding the parameters, which we need to know, to construct a polynomial space
   */
  struct PolynomialSpaceParams
  {
    explicit PolynomialSpaceParams(
        Core::FE::CellType distype, unsigned int degree, bool completeSpace)
        : distype_(distype), degree_(degree), completeSpace_(completeSpace)
    {
    }

    Core::FE::CellType distype_;
    unsigned int degree_;
    bool completeSpace_;

    bool operator<(const PolynomialSpaceParams& otherparams) const
    {
      return std::tie(distype_, degree_, completeSpace_) <
             std::tie(otherparams.distype_, otherparams.degree_, otherparams.completeSpace_);
    }
  };

  /*!
   \brief A class of 1D Lagrange polynomials on a given number of points
   */
  class LagrangePolynomial
  {
   public:
    /*!
     \brief Constructor.
     */
    LagrangePolynomial(const std::vector<double>& supportPoints, const double nodePoint)
        : support_points_(supportPoints), node_point_(nodePoint)
    {
      long double weight = 1.;
      for (double supportPoint : supportPoints) weight *= nodePoint - supportPoint;
      FOUR_C_ASSERT((std::abs(weight) > std::numeric_limits<double>::min() &&
                        std::abs(weight) < std::numeric_limits<double>::max()),
          "Error in evaluation of polynomial");
      weight_ = 1. / weight;
    }

    /*!
     \brief Copy constructor
     */
    LagrangePolynomial(const LagrangePolynomial& other)
        : support_points_(other.support_points_),
          node_point_(other.node_point_),
          weight_(other.weight_)
    {
    }

    /*!
     \brief Assignment operator
     */
    LagrangePolynomial& operator=(const LagrangePolynomial& other)
    {
      support_points_ = other.support_points_;
      node_point_ = other.node_point_;
      weight_ = other.weight_;
      return *this;
    }

    /*!
     \brief Evaluates the polynomial on a given point on the unit interval [-1, 1].
     */
    [[nodiscard]] double evaluate(const double point) const
    {
      double value = weight_;
      for (double supportPoint : support_points_) value *= point - supportPoint;
      return value;
    }

    /*!
     \brief Evaluates the polynomial and its derivatives (up to the order given by
            the length of values) on the given point
     */
    template <typename M>
    void evaluate(const double point, M& derivatives) const
    {
      FOUR_C_ASSERT(derivatives.num_cols() == 1, "Only column vectors supported");
      if (derivatives.num_rows() == 1)
      {
        derivatives(0) = evaluate(point);
        return;
      }

      // compute the value and derivatives by expanding the derivatives
      derivatives(0) = weight_;
      for (unsigned int k = 1; k < derivatives.num_rows(); ++k) derivatives(k) = 0;
      for (double supportPoint : support_points_)
      {
        const double v = point - supportPoint;
        for (int k = derivatives.num_rows() - 1; k > 0; --k)
          derivatives(k) = v * derivatives(k) + derivatives(k - 1);
        derivatives(0) *= v;
      }
      double faculty = 1;
      for (unsigned int k = 2; k < derivatives.num_rows(); ++k)
      {
        faculty *= static_cast<double>(k);
        derivatives(k) *= faculty;
      }
    }

    [[nodiscard]] double node_point() const { return node_point_; }

   private:
    std::vector<double> support_points_;
    double node_point_;
    double weight_;
  };



  /*!
   \brief A class of polynomials based on monomial coefficients

   This class takes a vector of coefficients of the monomials and evaluates a
   polynomial from these coefficients. The evaluation is done by the Horner's method.
   */
  class Polynomial
  {
   public:
    /*!
     \brief Constructor.
     */
    Polynomial(std::vector<double> coefficients) : coefficients_(std::move(coefficients)) {}

    /*!
     \brief Copy constructor
     */
    Polynomial(const Polynomial& other) : coefficients_(other.coefficients_) {}

    /*!
    \brief Assignment operator
    */
    Polynomial& operator=(const Polynomial& other)
    {
      coefficients_ = other.coefficients_;
      return *this;
    }

    /*!
     \brief Evaluates the polynomial on a given point
     */
    [[nodiscard]] double evaluate(const double point) const
    {
      // classical Horner's method
      return std::accumulate(coefficients_.rbegin(), coefficients_.rend(), 0.0,
          [point](const double& acc, const double& coeff) { return acc * point + coeff; });
    }

    /*!
     \brief Evaluates the polynomial and its derivatives (up to the order given by
         the length of values) on the given point
     */
    template <typename M>
    void evaluate(const double point, M& derivatives) const
    {
      FOUR_C_ASSERT(derivatives.num_cols() == 1, "Only column vectors supported");
      if (derivatives.num_rows() == 1)
      {
        derivatives(0) = evaluate(point);
        return;
      }

      // compute the value and derivatives by Horner scheme
      std::vector<double> temp(coefficients_);
      const int length = coefficients_.size();
      const int nderivs = derivatives.num_rows();
      const int maxder = std::min(length, nderivs);
      double kfaculty = 1.;
      // skip derivatives that are necessarily zero
      for (int k = 0; k < maxder; ++k)
      {
        for (int i = length - 2; i >= k; --i) temp[i] += point * temp[i + 1];
        derivatives(k) = kfaculty * temp[k];
        kfaculty *= static_cast<double>(k + 1);
      }
      for (int k = maxder; k < nderivs; ++k) derivatives(k) = 0;
    }

    /*!
     \brief Evaluates the polynomial or its derivative (of the given order) on the given point
     */
    [[nodiscard]] double evaluate_derivative(const double point, const int deriv_order) const
    {
      switch (deriv_order)
      {
        case 1:
        {
          Core::LinAlg::Matrix<2, 1> derivatives(Core::LinAlg::Initialization::uninitialized);
          evaluate(point, derivatives);
          return derivatives(1, 0);
        }
        case 2:
        {
          Core::LinAlg::Matrix<3, 1> derivatives(Core::LinAlg::Initialization::uninitialized);
          evaluate(point, derivatives);
          return derivatives(2, 0);
        }
        case 3:
        {
          Core::LinAlg::Matrix<4, 1> derivatives(Core::LinAlg::Initialization::uninitialized);
          evaluate(point, derivatives);
          return derivatives(3, 0);
        }
        default:
        {
          FOUR_C_THROW("Only derivatives up to order 2 supported");
          return 0;
        }
      }
    }

    // Here, we do not know about nodes, so put the center of the 1d element
    [[nodiscard]] double node_point() const { return 0.; }


   private:
    std::vector<double> coefficients_;
  };



  /*!
   \brief generates complete Lagrange basis in 1D for [-1,1]^d elements
   */
  std::vector<LagrangePolynomial> generate_lagrange_basis1_d(const unsigned int degree);



  /*!
   \brief generates complete Legendre basis in 1D

   Legendre polynomials are orthogonal on the unit interval [-1, 1]. As opposed
   to the usual mathematical definition, we also scale the polynomials such that
   they are actually orthonormal on the unit interval.
   */
  std::vector<Polynomial> generate_legendre_basis1_d(const unsigned int degree);



  /*!
  \brief Base class for polynomial spaces in nsd_ dimensions used by HDG
  */
  template <int nsd>
  class PolynomialSpaceBase
  {
   public:
    virtual ~PolynomialSpaceBase() = default;
    /*
     \brief Return the number of polynomials (over all dimensions)
     */
    [[nodiscard]] virtual std::size_t size() const = 0;

    /*
     \brief Evaluates the values of all polynomials on the given point
     */
    virtual void evaluate(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseVector& values) const = 0;

    /*
     \brief Evaluates the values of all polynomials on the given point
     */
    virtual void evaluate_deriv1(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseMatrix& derivatives) const = 0;

    /*
     \brief Evaluates the first derivative of all polynomials on the given point
     */
    virtual void evaluate_deriv2(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseMatrix& derivatives) const = 0;

    /*
     \brief Creates an array with coordinates of the nodes supporting the polynomials.
     */
    virtual void fill_unit_node_points(Core::LinAlg::SerialDenseMatrix& matrix) const = 0;
  };



  /*!
   \brief A class of tensor product polynomials expanded from 1D polynomials

   The 1D polynomial class must feature two Evaluate functions, one taking just
   the point as an argument (evaluating the polynomial's value) and one taking
   the point and an array with as many elements as derivatives are requested.

   Base class for LagrangeBasis
   */
  template <int nsd, class POLY>
  class PolynomialSpaceTensor : public PolynomialSpaceBase<nsd>
  {
   public:
    /*
     \brief Constructor from a vector of one-dimensional polynomials
     */
    PolynomialSpaceTensor(const std::vector<POLY> polySpace1d) : poly_space1d_(polySpace1d)
    {
      // renumbering is included in case we want to use these polynomials also
      // for standard finite elements
      renumbering_.resize(size());

      for (unsigned int i = 0; i < renumbering_.size(); ++i) renumbering_[i] = i;
    }


    /*
     \brief Return the number of polynomials (over all dimensions)
     */
    static std::size_t size(const std::size_t degree)
    {
      std::size_t size = degree + 1;
      for (unsigned int d = 1; d < nsd; ++d) size *= degree + 1;
      return size;
    }

    /*
     \brief Return the number of polynomials (over all dimensions)
     */
    std::size_t size() const override { return size(poly_space1d_.size() - 1); }

    /*
     \brief Evaluates the values of the whole polynomial space in the given point
     */
    void evaluate(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseVector& values) const override;

    /*
     \brief Evaluates the first derivative of the whole polynomial space in the given point
     */
    void evaluate_deriv1(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseMatrix& derivatives) const override;

    /*
     \brief Evaluates the second derivative of the whole polynomial space in the given point
     */
    void evaluate_deriv2(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseMatrix& derivatives) const override;

    /*
     \brief Evaluates the second derivative of the whole polynomial space in the given point
     */
    template <typename M>
    void evaluate_deriv2(const Core::LinAlg::Matrix<nsd, 1>& point, M& derivatives) const
    {
    }

    /*
     \brief Convert from a index within the polynomial space to the tensor indices in the
     individual dimensions
     */
    Core::LinAlg::Matrix<nsd, 1, unsigned int> get_indices(const unsigned int index) const
    {
      FOUR_C_ASSERT(index < size(), "Access out of range");
      Core::LinAlg::Matrix<nsd, 1, unsigned int> indices;
      const unsigned int npoly = poly_space1d_.size();
      switch (nsd)
      {
        case 1:
          indices(0) = index;
          break;
        case 2:
          indices(0) = index % npoly;
          indices(1) = index / npoly;
          break;
        case 3:
          indices(0) = index % npoly;
          indices(1) = (index / npoly) % npoly;
          indices(2) = index / (npoly * npoly);
          break;
        default:
          FOUR_C_THROW("Invalid dimension");
          break;
      }
      return indices;
    }

    /*
     \brief Creates an array with coordinates of the nodes supporting the polynomials.
     */
    void fill_unit_node_points(Core::LinAlg::SerialDenseMatrix& matrix) const override;

   private:
    std::vector<POLY> poly_space1d_;
    std::vector<int> renumbering_;
  };



  /*!
   \brief A class of polynomials of complete degree expanded from 1D polynomials by truncated
   tensor products

   The 1D polynomial class must feature two Evaluate functions, one taking just
   the point as an argument (evaluating the polynomial's value) and one taking
   the point and an array with as many elements as derivatives are requested.

   Base class for LagrangeBasis
   */
  template <int nsd, class POLY>
  class PolynomialSpaceComplete : public PolynomialSpaceBase<nsd>
  {
   public:
    /*
     \brief Constructor from a vector of one-dimensional polynomials
     */
    PolynomialSpaceComplete(const std::vector<POLY> polySpace1d) : poly_space1d_(polySpace1d)
    {
      renumbering_.resize(size());
      for (unsigned int i = 0; i < renumbering_.size(); ++i) renumbering_[i] = i;
    }


    /*
     \brief Return the number of polynomials (over all dimensions)
     */
    static std::size_t size(const std::size_t degree)
    {
      std::size_t size = degree + 1;
      for (unsigned int d = 1; d < nsd; ++d)
      {
        size *= degree + 1 + d;
        size /= (d + 1);  // This integer division is always without remainder
      }
      return size;
    }

    /*
     \brief Return the number of polynomials (over all dimensions)
     */
    std::size_t size() const override { return size(poly_space1d_.size() - 1); }

    /*
     \brief Evaluates the values of the whole polynomial space in the given point
     */
    void evaluate(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseVector& values) const override;

    /*
     \brief Evaluates the first derivative of the whole polynomial space in the given point
     */
    void evaluate_deriv1(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseMatrix& derivatives) const override;

    /*
     \brief Evaluates the second derivative of the whole polynomial space in the given point
     */
    void evaluate_deriv2(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseMatrix& derivatives) const override;

    /*
     \brief Creates an array with coordinates of the nodes supporting the polynomials.
     */
    void fill_unit_node_points(Core::LinAlg::SerialDenseMatrix& matrix) const override;

   private:
    std::vector<POLY> poly_space1d_;
    std::vector<int> renumbering_;
  };



  /*!
   \brief Class of tensor product Lagrange polynomials based on 1D Gauss-Lobatto
   support points (gives standard Q1 and Q2 basis functions and good polynomials
   for higher order). This class is usually not directly called in user code, as
   its functionality can be accessed through PolynomialSpace<nsd_>.
  */
  template <int nsd>
  class LagrangeBasis : public PolynomialSpaceTensor<nsd, LagrangePolynomial>
  {
   public:
    /*
    \brief Constructor from a vector of one-dimensional polynomials
     */
    LagrangeBasis(const unsigned int degree)
        : PolynomialSpaceTensor<nsd, LagrangePolynomial>(generate_lagrange_basis1_d(degree))
    {
    }
  };



  /*!
  \brief Class of Legendre polynomials that are pairwise orthogonal. This class
   is usually not directly called in user code, as its functionality can be
   accessed through PolynomialSpace<nsd_>.
  */
  template <int nsd>
  class LegendreBasis : public PolynomialSpaceComplete<nsd, Polynomial>
  {
   public:
    /*
   \brief Constructor from a vector of one-dimensional polynomials
     */
    LegendreBasis(const unsigned int degree)
        : PolynomialSpaceComplete<nsd, Polynomial>(generate_legendre_basis1_d(degree))
    {
    }
  };



  /*!
  \brief Lagrange basis for triangular/tetrahedral elements.

   This class constructs a Lagrangian basis for tetrahedral elements of arbitrary degree
   by a transformation of Legendre polynomials onto so-called Fekete nodes, which are
   a generalization of Gauss-Lobatto points to higher dimensions on triangles. For the
   transformation, a Vandermonde matrix is used to map the nodeless orthonormal Legendre
   polynomials onto the desired node distribution.

   The transformation with the Vandermonde matrix is described in
   R. Sevilla, S. Fernandez-Mendez, A. Huerta, NURBS-Enhanced Finite Element Method,
   Arch. Comput. Methods Eng. (2011), 441-484

   Author: kronbichler 08/14
   */
  template <int nsd>
  class LagrangeBasisTet : public PolynomialSpaceBase<nsd>
  {
   public:
    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;

    LagrangeBasisTet(const unsigned int degree) : legendre_(degree)
    {
      fill_fekete_points(degree);
      compute_vandermonde_matrices(degree);
    }


    /*
     \brief Return the number of polynomials (over all dimensions)
     */
    static std::size_t size(const std::size_t degree)
    {
      std::size_t size = degree + 1;
      for (unsigned int d = 1; d < nsd; ++d)
      {
        size *= degree + 1 + d;
        size /= (d + 1);  // This integer division is always without remainder
      }
      return size;
    }

    /*
     \brief Return the number of polynomials (over all dimensions)
     */
    std::size_t size() const override { return legendre_.size(); }

    /*
     \brief Evaluates the values of the whole polynomial space in the given point
     */
    void evaluate(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseVector& values) const override;

    /*
     \brief Evaluates the first derivative of the whole polynomial space in the given point
     */
    void evaluate_deriv1(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseMatrix& derivatives) const override;

    /*
     \brief Evaluates the second derivative of the whole polynomial space in the given point
     */
    void evaluate_deriv2(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseMatrix& derivatives) const override;

    /*
     \brief Creates an array with coordinates of the nodes supporting the polynomials.
     */
    void fill_unit_node_points(Core::LinAlg::SerialDenseMatrix& matrix) const override;

   private:
    void fill_fekete_points(const unsigned int degree);
    void compute_vandermonde_matrices(const unsigned int degree);

    Core::LinAlg::SerialDenseMatrix vandermonde_;
    mutable Teuchos::SerialDenseSolver<ordinalType, scalarType> vandermonde_factor_;
    mutable Core::LinAlg::SerialDenseMatrix evaluate_vec_;
    Core::LinAlg::SerialDenseMatrix fekete_points_;
    LegendreBasis<nsd> legendre_;
  };



  /*!
   \brief General polynomial class that encapsulates the various polynomial shapes

   For QUAD/HEX elements, this class allows to select between the usual tensor
   polynomial space LagrangeBasisHEX and the complete polynomial space
   LegendreBasis. For TRI/TET elements, Legendre polynomials are always used.
   We might want to implement a Lagrange basis for triangles at some point but
   for that we need a special transform to be able to use the above truncated
   tensor product.
   */
  template <int nsd>
  class PolynomialSpace
  {
   public:
    PolynomialSpace(
        const Core::FE::CellType distype, const unsigned int degree, const bool completeSpace)
        : polyspace_((Core::FE::get_number_of_element_faces(distype) == 1 + nsd && nsd > 1)
                         ? static_cast<Core::FE::PolynomialSpaceBase<nsd>*>(
                               new Core::FE::LagrangeBasisTet<nsd>(degree))
                     : completeSpace ? static_cast<Core::FE::PolynomialSpaceBase<nsd>*>(
                                           new Core::FE::LegendreBasis<nsd>(degree))
                                     : static_cast<Core::FE::PolynomialSpaceBase<nsd>*>(
                                           new Core::FE::LagrangeBasis<nsd>(degree)))
    {
      if (nsd != Core::FE::get_dimension(distype))
        FOUR_C_THROW("Dimension of shape does not match template argument nsd_ in PolynomialSpace");
    }

    PolynomialSpace(PolynomialSpaceParams params)
        : polyspace_((Core::FE::get_number_of_element_faces(params.distype_) == 1 + nsd && nsd > 1)
                         ? static_cast<Core::FE::PolynomialSpaceBase<nsd>*>(
                               new Core::FE::LagrangeBasisTet<nsd>(params.degree_))
                     : params.completeSpace_
                         ? static_cast<Core::FE::PolynomialSpaceBase<nsd>*>(
                               new Core::FE::LegendreBasis<nsd>(params.degree_))
                         : static_cast<Core::FE::PolynomialSpaceBase<nsd>*>(
                               new Core::FE::LagrangeBasis<nsd>(params.degree_)))
    {
    }

    /*
   \brief Return the number of polynomials (over all dimensions)
     */
    std::size_t size() const { return polyspace_->size(); }

    /*
   \brief Evaluates the values of all polynomials on the given point
     */
    void evaluate(
        const Core::LinAlg::Matrix<nsd, 1>& point, Core::LinAlg::SerialDenseVector& values) const
    {
      polyspace_->evaluate(point, values);
    }

    /*
   \brief Evaluates the values of all polynomials on the given point
     */
    void evaluate_deriv1(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseMatrix& derivatives) const
    {
      polyspace_->evaluate_deriv1(point, derivatives);
    }

    /*
   \brief Evaluates the first derivative of all polynomials on the given point
     */
    void evaluate_deriv2(const Core::LinAlg::Matrix<nsd, 1>& point,
        Core::LinAlg::SerialDenseMatrix& derivatives) const
    {
      polyspace_->evaluate_deriv2(point, derivatives);
    }

    /*
   \brief Creates an array with coordinates of the nodes supporting the polynomials.

   For Lagrange polynomials, these are the node points of the respective Gauss-Lobatto
   quadrature formula. For Legendre polynomials which are non-nodal, all points will be
   the element center coordinates.

   The output matrix is resized to the correct dimension.
     */
    void fill_unit_node_points(Core::LinAlg::SerialDenseMatrix& matrix) const
    {
      polyspace_->fill_unit_node_points(matrix);
    }

   private:
    std::shared_ptr<PolynomialSpaceBase<nsd>> polyspace_;
  };

  /*!
  \brief Cache for polynomial space so we do not need to calculate again

   In analogy to GaussPointCache
   Author: schoeder 06/14
   */
  template <int nsd>
  class PolynomialSpaceCache
  {
   public:
    static PolynomialSpaceCache<nsd>& instance();

    std::shared_ptr<PolynomialSpace<nsd>> create(PolynomialSpaceParams params);

   private:
    PolynomialSpaceCache() = default;

    /// cache of already created polynomial spaces
    std::map<PolynomialSpaceParams, std::shared_ptr<PolynomialSpace<nsd>>> ps_cache_;
  };

  /*!
   \brief Returns the size of the polynomial space.

   Note: This class must always be synchronized with PolynomialSpace above!
   */
  inline int get_basis_size(
      const Core::FE::CellType distype, const int degree, const bool completeSpace)
  {
    const int dim = Core::FE::get_dimension(distype);
    const int nfaces = Core::FE::get_number_of_element_faces(distype);
    switch (dim)
    {
      case 3:
        if (completeSpace || (nfaces == dim + 1))
          return LegendreBasis<3>::size(degree);
        else
          return LagrangeBasis<3>::size(degree);
        break;
      case 2:
        if (completeSpace || (nfaces == dim + 1))
          return LegendreBasis<2>::size(degree);
        else
          return LagrangeBasis<2>::size(degree);
        break;
      case 1:
        return degree + 1;
        break;
      default:
        FOUR_C_THROW("Invalid dimension");
        return -1;
    }
  }

}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif