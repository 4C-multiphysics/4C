// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_UTILS_NURBS_SHAPEFUNCTIONS_HPP
#define FOUR_C_FEM_GENERAL_UTILS_NURBS_SHAPEFUNCTIONS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_bspline.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_tensor.hpp"

#include <optional>
#include <type_traits>

FOUR_C_NAMESPACE_OPEN


namespace Core::FE::Nurbs
{
  /*

  \brief Evaluate basis functions of nurbs basis functions (line element version).

  \verbatim

          reference element          element in knotspan
                              Psi
          --+XXX+XXX+-->    ------>   ****+XXX+**** --->
           -1      +1   u                             xi
                  \            knot(degree)   knot(degree+1)D

                    \                       |
                     \                      |
                      \                     |
                       \                    |
                        \                   |
                         \                  |
                          \                 |
                           \                |      +----
                            \               |
                 Phi o Psi   \              | Phi =  +  N_{i}(xi)*f_{i}
                  ^           \             |       /
                  N_{i}(u,v)   \            |      +----
                                \           |
                                 \          |
                                  \         |
                                   \        |
                                    \       |
                                            v
                                         *
                                          *
                                           +
                                           X
                                           X
                                           X
                                           +
                                            *
                                             *
                                 'real' geometry element

   \endverbatim

   PART   I: EVALUATION OF  BEZIER SHAPE FUNCTIONS

   \verbatim

   B_{i}(xi) = N_i(xi)

   \endverbatim

   PART  II: PROJECTING TO NURBS SHAPE FUNCTIONS

   \verbatim

                        w_{i} * B_{i}(xi)
            N_{i} = --------------------------
                     +----+

                       +    w_{k} * B_{k}(xi)
                      /
                     +----+
                       k
   \endverbatim


    \param nurbs_shape_funct  (o)
    \param u                  (i)
    \param knots              (i)
    \param weights            (i)

    \return TRUE if successful

  */

  template <int degree, class V, class UV, class WG>
  bool nurbs_get_1d_funct(V& nurbs_shape_funct, const UV& u,
      const Core::LinAlg::SerialDenseVector& knots, const WG& weights)
  {
    // size is the number of control points/basis
    // functions of this element
    const int size = (degree + 1);

    // Gausspoint is in [-1:1]
    //
    // has to be mapped into knotvector evaluation interval

    /*
         reference element          element in knotspan

                              Psi
          --+XXX+XXX+-->    ------>   ****+XXX+**** --->

           -1      +1   u                             xi

                  \       (knot[0])(degree)   (knot[0])(degree+1)

                    \                       |
                     \                      |
                      \                     |
                       \                    |
                        \                   |
                         \                  |
                          \                 |
                           \                |      +----
                            \               |
                 Phi o Psi   \              | Phi =  +  N_{i}(xi)*f_{i}
                  ^           \             |       /
                  N_{i}(u,v)   \            |      +----
                                \           |
                                 \          |
                                  \         |
                                   \        |
                                    \       |
                                            v
                                         *
                                          *
                                           +
                                           X
                                           X
                                           X
                                           +
                                            *
                                             *
                                'real' geometry element
    */

    // the jacobian matrix of Psi is of the very
    // simple form
    //
    //              Ju
    //              |
    //              |
    //          +-  du -+
    //   Psi' = |  ---  |
    //          +- 2.0 -+
    //
    //
    //
    //

    double du = knots(degree + 1) - knots(degree);

    double Ju = du / 2.0;

    // get mapped coordinates

    double xi = knots(degree) + (u + 1.0) * Ju;

    // -----------------------------------------------
    //  PART I: EVALUATION OF  BEZIER SHAPE FUNCTIONS
    // -----------------------------------------------

    Core::LinAlg::SerialDenseVector bezier_shape_funct(size);

    // allocate bspline polynomials for both direction
    Core::FE::Nurbs::BsplinePolynomial bspline_xi(degree, knots);

    // get temporary double
    double bspline_xi_value;

    //
    //  Bezier basis function
    //          |
    //          |
    //       B_{i}(xi) = N_i(xi)
    //                     |
    //                     |
    //            bspline_u_value
    //

    // loop all basis functions (corresponding to
    // control points)
    for (int rr = 0; rr < degree + 1; ++rr)
    {
      // in first direction:

      // get bsplinevalue
      bspline_xi.evaluate_bspline(bspline_xi_value, xi, rr);

      // add value to bezier_shape_funct
      bezier_shape_funct(rr) = bspline_xi_value;
    }

    // -----------------------------------------------
    //  PART II: PROJECTING TO NURBS SHAPE FUNCTIONS
    // -----------------------------------------------

    // alloc temporary doubles, initialise to zero
    double sum_funct_weight;

    sum_funct_weight = 0.0;

    /*loop all control points, compute sums

                           +----+

         sum_funct_weight =  +    w_{i} * B_{i}(xi)
                            /
                           +----+
    */

    for (int rr = 0; rr < size; ++rr)
    {
      sum_funct_weight += weights(rr) * bezier_shape_funct(rr);
    }

    if (sum_funct_weight == 0.0) return false;

    /* Compute Nurbs basis functions


                      w_{i} * B_{i}(xi)

         N_{i} = ------------------------
                 +----+

                   +    w_{k} * B_{k}(xi)
                  /
                 +----+
                   k
    */

    // loop all basis functions
    for (int rr = 0; rr < size; ++rr)
    {
      // get shape function
      nurbs_shape_funct(rr) = weights(rr) * bezier_shape_funct(rr) / sum_funct_weight;
    }

    return true;
  }

  /**
   * @brief Evaluate the 1D NURBS shape functions based on the discretization type of the element.
   */
  template <class V, class UV, class WG>
  bool nurbs_get_1d_funct(V& nurbs_shape_funct, const UV& u,
      const Core::LinAlg::SerialDenseVector& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (distype)
    {
      case Core::FE::CellType::nurbs2:
      {
        return nurbs_get_1d_funct<1>(nurbs_shape_funct, u, knots, weights);
      }
      case Core::FE::CellType::nurbs3:
      {
        return nurbs_get_1d_funct<2>(nurbs_shape_funct, u, knots, weights);
      }
      default:
      {
        FOUR_C_THROW("Unknown distype for nurbs line element evaluation\n");
      }
    }
  }


  /*!
    \brief Evaluate basis functions and derivatives (with
    respect to the coordinates of the reference element)
    of nurbs basis functions --- version for 1d line
    elements.

   \verbatim

           reference element          element in knotspan
                                 Psi
          --+XXX+XXX+-->    ------>   ****+XXX+**** --->
           -1      +1   u                             xi
                  \            knot(degree)   knot(degree+1)D

                    \                       |
                     \                      |
                      \                     |
                       \                    |
                        \                   |
                         \                  |
                          \                 |
                           \                |      +----
                            \               |
                 Phi o Psi   \              | Phi =  +  N_{i}(xi)*f_{i}
                  ^           \             |       /
                  N_{i}(u,v)   \            |      +----
                                \           |
                                 \          |
                                  \         |
                                   \        |
                                    \       |
                                            v
                                         *
                                          *
                                           +
                                           X
                                           X
                                           X
                                           +
                                            *
                                             *
                                 'real' geometry element

   \endverbatim

   PART   I: EVALUATION OF  BEZIER SHAPE FUNCTIONS

  \verbatim

              B_{i}(xi) = N_i(xi)


              dB_{i}
              ------(xi) = N_i'(xi)
               dxi

  \endverbatim

  PART  II: PROJECTING TO NURBS SHAPE FUNCTIONS

  \verbatim

                      w_{i} * B_{i}(xi)
          N_{i} = --------------------------
                  +----+

                    +    w_{k} * B_{k}(xi)
                   /
                  +----+
                    k

  \endverbatim

  Derivatives are evaluated using the chain rule

  PART III: TRANSFORMING DERIVATIVES FROM xi TO u

  \verbatim

            +-   -+   +-   -+
            | dxi |   |  du |
     Psi' = | --- | = | --- |
            |  du |   | 2.0 |
            +-   -+   +-   -+

  \endverbatim

  \param nurbs_shape_funct  (o)
  \param nurbs_shape_deriv  (o)
  \param uv                 (i)
  \param knots              (i)
  \param weights            (i)

  \return TRUE, if successful

  */
  template <int degree, class VF, class VD, class UV, class WG>
  bool nurbs_get_1d_funct_deriv(VF& nurbs_shape_funct, VD& nurbs_shape_deriv, const UV& u,
      const Core::LinAlg::SerialDenseVector& knots, const WG& weights)
  {
    // size is the number of control points/basis
    // functions of this element
    const int size = (degree + 1);

    // Gausspoint is in [-1:1]
    //
    // has to be mapped into knotvector evaluation interval

    /*
         reference element          element in knotspan

                              Psi
          --+XXX+XXX+-->    ------>   ****+XXX+**** --->

           -1      +1   u                             xi

                  \       (knot[0])(degree)   (knot[0])(degree+1)

                    \                       |
                     \                      |
                      \                     |
                       \                    |
                        \                   |
                         \                  |
                          \                 |
                           \                |      +----
                            \               |
                 Phi o Psi   \              | Phi =  +  N_{i}(xi)*f_{i}
                  ^           \             |       /
                  N_{i}(u,v)   \            |      +----
                                \           |
                                 \          |
                                  \         |
                                   \        |
                                    \       |
                                            v
                                         *
                                          *
                                           +
                                           X
                                           X
                                           X
                                           +
                                            *
                                             *
                                'real' geometry element
    */

    // the jacobian matrix of Psi is of the very
    // simple form
    //
    //              Ju
    //              |
    //              |
    //          +-  du -+
    //   Psi' = |  ---  |
    //          +- 2.0 -+
    //
    //
    //
    //

    double du = knots(degree + 1) - knots(degree);

    double Ju = du / 2.0;

    // get mapped coordinates

    double xi = knots(degree) + (u + 1.0) * Ju;

    // -----------------------------------------------
    //  PART I: EVALUATION OF  BEZIER SHAPE FUNCTIONS
    // -----------------------------------------------

    Core::LinAlg::SerialDenseVector bezier_shape_funct(size);
    Core::LinAlg::SerialDenseVector bezier_shape_deriv(size);

    // allocate bspline polynomials for both direction
    Core::FE::Nurbs::BsplinePolynomial bspline_xi(degree, knots);

    // get temporary doubles
    double bspline_xi_value;
    double bspline_xi_derivative;

    //
    //  Bezier basis function
    //          |
    //          |
    //       B_{i}(xi,eta) = N_i(xi)
    //                         |
    //                         |
    //                bspline_u_value
    //

    //      dB_{i}
    //      ------(xi,eta) = N_i'(xi)
    //       dxi
    //

    // loop all basis functions (corresponding to
    // control points)
    for (int rr = 0; rr < degree + 1; ++rr)
    {
      // in first direction:

      // get bsplinevalue and derivative
      bspline_xi.evaluate_bspline_and_deriv(bspline_xi_value, bspline_xi_derivative, xi, rr);

      // add value to bezier_shape_funct
      bezier_shape_funct(rr) = bspline_xi_value;

      // add values to bezier_shape_deriv
      bezier_shape_deriv(rr) = bspline_xi_derivative;
    }

    // -----------------------------------------------
    //  PART II: PROJECTING TO NURBS SHAPE FUNCTIONS
    // -----------------------------------------------

    // alloc temporary doubles, initialise to zero
    double sum_funct_weight;
    double sum_deriv_weight;

    sum_funct_weight = 0.0;
    sum_deriv_weight = 0.0;

    /*loop all control points, compute sums

                           +----+

         sum_funct_weight =  +    w_{i} * B_{i}(xi)
                            /
                           +----+


                           +----+
                            \             dB'_{i}
         sum_deriv_weight =  +    w_{i} * -------(xi)
                            /               dxi
                           +----+
    */

    for (int rr = 0; rr < size; ++rr)
    {
      sum_funct_weight += weights(rr) * bezier_shape_funct(rr);

      sum_deriv_weight += weights(rr) * bezier_shape_deriv(rr);
    }

    if (sum_funct_weight == 0.0) return false;

    /* Compute Nurbs basis functions


                    w_{i} * B_{i}(xi)

         N_{i} = ------------------------
                 +----+

                   +    w_{k} * B_{k}(xi)
                  /
                 +----+
                    k
    */

    // Nurbs derivatives are defined by the chain rule
    //
    //                         +- +----+          -+               +- +----+             -+
    //                         |   \               |               |   \           dB_{k} |
    //                  dB_{i} |    +  w_{k}*B_{k} |               |    +    w_{k}*------ |
    //            w_{i}*------*|   /               | - w_{i}*B_{i}*|   /             dxi  |
    //                   dxi   |  +----+           |               |  +----+              |
    // dN_{i}                  +-   k             -+               +-   k                -+
    // ------ = ----------------------------------------------------------------------------
    //  dxi                         +- +----+                  -+ 2
    //                              |   \                       |
    //                              |    +    w_{k} * B_{k}(xi) |
    //                              |   /                       |
    //                              |  +----+                   |
    //                              +-   k                     -+

    // loop all basis functions
    for (int rr = 0; rr < size; ++rr)
    {
      // get shape function
      nurbs_shape_funct(rr) = weights(rr) * bezier_shape_funct(rr) / sum_funct_weight;

      // loop directions to compute derivatives
      nurbs_shape_deriv(0, rr) =
          bezier_shape_deriv(rr) * sum_funct_weight - bezier_shape_funct(rr) * sum_deriv_weight;

      nurbs_shape_deriv(0, rr) *= weights(rr) / (sum_funct_weight * sum_funct_weight);
    }

    // -----------------------------------------------
    //  PART III: TRANSFORMING DERIVATIVES FROM xi
    //            TO u
    // -----------------------------------------------

    // we already know the jacobian matrix of psi
    //
    //          +- dxi -+   +-  du -+
    //   Psi' = |  ---  | = |  ---  |
    //          +-  du -+   +- 2.0 -+
    //
    // we will obtain the derivatives with respect to
    // u,v just by multiplication

    // loop all basis function derivatives
    for (int rr = 0; rr < size; ++rr)
    {
      nurbs_shape_deriv(0, rr) *= Ju;
    }

    return true;
  }


  /**
   * @brief Evaluate the 1D NURBS shape functions and their 1st derivatives based on the
   * discretization type of the element.
   */
  template <class VF, class VD, class UV, class WG>
  bool nurbs_get_1d_funct_deriv(VF& nurbs_shape_funct, VD& nurbs_shape_deriv, const UV& u,
      const Core::LinAlg::SerialDenseVector& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (distype)
    {
      case Core::FE::CellType::nurbs2:
      {
        return nurbs_get_1d_funct_deriv<1>(nurbs_shape_funct, nurbs_shape_deriv, u, knots, weights);
      }
      case Core::FE::CellType::nurbs3:
      {
        return nurbs_get_1d_funct_deriv<2>(nurbs_shape_funct, nurbs_shape_deriv, u, knots, weights);
      }
      default:
      {
        FOUR_C_THROW("Unknown distype for nurbs line element evaluation\n");
      }
    }
  }

  template <int degree, class VF, class MD, class MSD, class UV, class WG>
  bool nurbs_get_1d_funct_deriv_deriv2(VF& nurbs_shape_funct, MD& nurbs_shape_deriv,
      MSD& nurbs_shape_deriv2, const UV& u, const Core::LinAlg::SerialDenseVector& knots,
      const WG& weights)
  {
    // size is the number of control points/basis
    // functions of this element
    const int size = (degree + 1);

    double du = knots(degree + 1) - knots(degree);

    double Ju = du / 2.0;

    // get mapped coordinates
    double xi = knots(degree) + (u + 1.0) * Ju;

    // -----------------------------------------------
    //  PART I: EVALUATION OF  BEZIER SHAPE FUNCTIONS
    // -----------------------------------------------

    Core::LinAlg::SerialDenseVector bezier_shape_funct(size);
    Core::LinAlg::SerialDenseVector bezier_shape_deriv(size);
    Core::LinAlg::SerialDenseVector bezier_shape_deriv2(size);

    // allocate bspline polynomials for both direction
    Core::FE::Nurbs::BsplinePolynomial bspline_xi(degree, knots);

    // get temporary doubles
    double bspline_xi_value;
    double bspline_xi_derivative;
    double bspline_xi_deriv2;

    // loop all basis functions (corresponding to
    // control points)
    for (int rr = 0; rr < degree + 1; ++rr)
    {
      // in first direction:

      // get bsplinevalue and derivative
      bspline_xi.evaluate_bspline_first_and_second_deriv(
          bspline_xi_value, bspline_xi_derivative, bspline_xi_deriv2, xi, rr);

      // add value to bezier_shape_funct
      bezier_shape_funct(rr) = bspline_xi_value;

      // add values to bezier_shape_deriv
      bezier_shape_deriv(rr) = bspline_xi_derivative;

      // add values to bspline_xi_deriv2
      bezier_shape_deriv2(rr) = bspline_xi_deriv2;
    }
    // -----------------------------------------------
    //  PART II: PROJECTING TO NURBS SHAPE FUNCTIONS
    // -----------------------------------------------

    // allocate temporary doubles, initialize to zero
    double sum_funct_weight;
    double sum_deriv_weight;
    double sum_deriv2_weight;

    sum_funct_weight = 0.0;
    sum_deriv_weight = 0.0;
    sum_deriv2_weight = 0.0;


    for (int rr = 0; rr < size; ++rr)
    {
      sum_funct_weight += weights(rr) * bezier_shape_funct(rr);

      sum_deriv_weight += weights(rr) * bezier_shape_deriv(rr);

      sum_deriv2_weight += weights(rr) * bezier_shape_deriv2(rr);
    }

    if (sum_funct_weight == 0.0) return false;

    // loop all basis functions
    for (int rr = 0; rr < size; ++rr)
    {
      // get shape function
      nurbs_shape_funct(rr) = weights(rr) * bezier_shape_funct(rr) / sum_funct_weight;

      // loop directions to compute derivatives
      nurbs_shape_deriv(0, rr) = bezier_shape_deriv(rr) * sum_funct_weight;
      nurbs_shape_deriv(0, rr) -= bezier_shape_funct(rr) * sum_deriv_weight;
      nurbs_shape_deriv(0, rr) *= weights(rr) / (sum_funct_weight * sum_funct_weight);

      // second derivative
      nurbs_shape_deriv2(0, rr) = 2 * bezier_shape_funct(rr) * sum_deriv_weight * sum_deriv_weight;
      nurbs_shape_deriv2(0, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(0, rr) -= bezier_shape_funct(rr) * sum_deriv2_weight;
      nurbs_shape_deriv2(0, rr) -= bezier_shape_deriv(rr) * sum_deriv_weight;
      nurbs_shape_deriv2(0, rr) -= bezier_shape_deriv(rr) * sum_deriv_weight;
      nurbs_shape_deriv2(0, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(0, rr) += bezier_shape_deriv2(rr);
      nurbs_shape_deriv2(0, rr) *= weights(rr) / sum_funct_weight;
    }

    // -----------------------------------------------
    //  PART III: TRANSFORMING DERIVATIVES FROM xi
    //            TO u
    // -----------------------------------------------
    for (int rr = 0; rr < size; ++rr)
    {
      nurbs_shape_deriv(0, rr) *= Ju;
      nurbs_shape_deriv2(0, rr) *= Ju * Ju;
    }

    return true;
  }

  /**
   * @brief Evaluate the 1D NURBS shape functions and their 2nd derivatives based on the
   * discretization type of the element.
   */
  template <class VF, class MD, class MSD, class UV, class WG>
  bool nurbs_get_1d_funct_deriv_deriv2(VF& nurbs_shape_funct, MD& nurbs_shape_deriv,
      MSD& nurbs_shape_deriv2, const UV& u, const Core::LinAlg::SerialDenseVector& knots,
      const WG& weights, const Core::FE::CellType& distype)
  {
    switch (distype)
    {
      case Core::FE::CellType::nurbs2:
      {
        return nurbs_get_1d_funct_deriv_deriv2<1>(
            nurbs_shape_funct, nurbs_shape_deriv, nurbs_shape_deriv2, u, knots, weights);
      }
      case Core::FE::CellType::nurbs3:
      {
        return nurbs_get_1d_funct_deriv_deriv2<2>(
            nurbs_shape_funct, nurbs_shape_deriv, nurbs_shape_deriv2, u, knots, weights);
      }
      default:
      {
        FOUR_C_THROW("Unknown distype for nurbs line element evaluation\n");
      }
    }
  }


  /*!
  \brief Evaluate 2d basis functions of nurbs basis functions.

  \verbatim

         reference element          element in knotspan

               v^                        eta^
                |                           |
           -1   |  +1                 *   *   *   *
         +1 +---+---+ +1             *+***+---+***+*
            |XXX|XXX|         Psi     *   |XXX|   *
          --+---+---+-->    ------>   *   |XXX|   * --->
            |XXX|XXX|   u             *   |XXX|   *   xi
         -1 +---+---+ -1              *+***+---+***+*
           -1   |  +1                 *   *   *   *
                                          *   *
                  \       (knot[0])(degree)   (knot[0])(degree+1)

                    \                       |
                     \                      |
                      \                     |
                       \                    |
                        \                   |
                         \                  |
                          \                 |
                           \                |      +----
                            \               |
                 Phi o Psi   \              | Phi =  +  N_{i,j}(xi,eta)*f_{i,j}
                ^             \             |       /
                N_{i,j}(u,v)   \            |      +----
                                \           |
                                 \          |
                                  \         |
                                   \        |
                                    \       |
                                            v
                                    *
                                    *+-------+*
                                     \XXXXXX|
                                      \XXXXX|
                                       \XXXX|
                                      **+---+**
                                        *   *

                              'real' geometry element

  \endverbatim

  PART   I: EVALUATION OF  BEZIER SHAPE FUNCTIONS

  \verbatim

             B_{i,j}(xi,eta) = N_i(xi)*M_j(eta)

  \endverbatim

  PART  II: PROJECTING TO NURBS SHAPE FUNCTIONS

  \verbatim

                        w_{i,j} * B_{i,j}(xi,eta)

         N_{i,j} = ----------------------------------
                    +----+

                      +    w_{k,l} * B_{k,l}(xi,eta)
                     /
                    +----+
                     k,l

  \endverbatim


  \param nurbs_shape_funct  (o)
  \param uv                 (i)
  \param knots              (i)
  \param weights            (i)

  \return TRUE, if successful

  */
  template <int degree, typename ScalarType = double, class VF, class UV, class WG>
  bool nurbs_get_2d_funct(VF& nurbs_shape_funct, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights)
  {
    // size is the number of control points/basis
    // functions of this element
    const int size = (degree + 1) * (degree + 1);

    // Gausspoint is in [-1:1]x[-1:1]
    //
    // has to be mapped into knotvector evaluation interval

    /*
         reference element          element in knotspan

               v^                        eta^
                |                           |
           -1   |  +1                 *   *   *   *
         +1 +---+---+ +1             *+***+---+***+*
            |XXX|XXX|         Psi     *   |XXX|   *
          --+---+---+-->    ------>   *   |XXX|   * --->
            |XXX|XXX|   u             *   |XXX|   *   xi
         -1 +---+---+ -1              *+***+---+***+*
           -1   |  +1                 *   *   *   *
                                          *   *
                  \       (knot[0])(degree)   (knot[0])(degree+1)

                    \                       |
                     \                      |
                      \                     |
                       \                    |
                        \                   |
                         \                  |
                          \                 |
                           \                |      +----
                            \               |
                 Phi o Psi   \              | Phi =  +  N_{i,j}(xi,eta)*f_{i,j}
                ^             \             |       /
                N_{i,j}(u,v)   \            |      +----
                                \           |
                                 \          |
                                  \         |
                                   \        |
                                    \       |
                                            v
                                    *
                                   *+-------+*
                                     \XXXXXX|
                                      \XXXXX|
                                       \XXXX|
                                      **+---+**
                                        *   *

                              'real' geometry element
    */

    // the jacobian matrix of Psi is of the very
    // simple form
    //
    //               Ju
    //               |
    //               |
    //          +-   du       -+
    //          |   ---   0    |
    //          |   2.0        |
    //   Psi' = |              |
    //          |         dv   |
    //          |    0   ---   |
    //          +-       2.0  -+
    //                    |
    //                    |
    //                    Jv
    //

    double du = (knots[0])(degree + 1) - (knots[0])(degree);
    double dv = (knots[1])(degree + 1) - (knots[1])(degree);

    double Ju = du / 2.0;
    double Jv = dv / 2.0;

    // get mapped coordinates

    ScalarType xi = (knots[0])(degree) + ((uv(0) + 1)) * Ju;
    ScalarType eta = (knots[1])(degree) + ((uv(1) + 1)) * Jv;

    // -----------------------------------------------
    //  PART I: EVALUATION OF  BEZIER SHAPE FUNCTIONS
    // -----------------------------------------------

    std::vector<ScalarType> bezier_shape_funct(size);

    // allocate bspline polynomials for both direction
    Core::FE::Nurbs::BsplinePolynomial bspline_xi(degree, knots[0]);
    Core::FE::Nurbs::BsplinePolynomial bspline_eta(degree, knots[1]);

    // get temporary variables
    ScalarType bspline_xi_value;
    ScalarType bspline_eta_value;

    // define temporary int variable to compute the
    // number of the basis function from i,j
    int id;

    //
    //  Bezier basis function
    //          |
    //          |
    //       B_{i,j}(xi,eta) = N_i(xi)*M_j(eta)
    //                           |       |
    //                           |       |
    //                  bspline_u_value  |
    //                                   |
    //                             bspline_v_value
    //

    // loop all basis functions (corresponding to
    // control points)
    for (int rr = 0; rr < degree + 1; ++rr)
    {
      // in first direction:

      // get bsplinevalue and derivative
      bspline_xi.evaluate_bspline(bspline_xi_value, xi, rr);

      for (int mm = 0; mm < degree + 1; ++mm)
      {
        // in second direction:

        // get bsplinevalue and derivative
        bspline_eta.evaluate_bspline(bspline_eta_value, eta, mm);

        // get the number of the basis function
        id = rr + mm * (degree + 1);

        // add value to bezier_shape_funct
        bezier_shape_funct[id] = bspline_xi_value * bspline_eta_value;
      }
    }

    // -----------------------------------------------
    //  PART II: PROJECTING TO NURBS SHAPE FUNCTIONS
    // -----------------------------------------------

    // alloc temporary variable, initialise to zero
    ScalarType sum_funct_weight;

    sum_funct_weight = 0.0;

    /*loop all control points, compute sums

                           +----+

         sum_funct_weight =  +    w_{i,j} * B_{i,j}(xi,eta)
                            /
                           +----+
    */

    for (int rr = 0; rr < size; ++rr)
    {
      sum_funct_weight += weights(rr) * bezier_shape_funct[rr];
    }

    if (sum_funct_weight == 0.0) return false;

    /* Compute Nurbs basis functions


                        w_{i,j} * B_{i,j}(xi,eta)

         N_{i,j} = ----------------------------------
                    +----+

                      +    w_{k,l} * B_{k,l}(xi,eta)
                     /
                    +----+
                     k,l
    */

    // loop all basis functions
    for (int rr = 0; rr < size; ++rr)
    {
      // get shape function
      nurbs_shape_funct(rr) = weights(rr) * bezier_shape_funct[rr] / sum_funct_weight;
    }

    return true;
  }

  /**
   * @brief Evaluate the 2D NURBS shape functions based on the discretization type of the element.
   */
  template <typename ScalarType = double, class VF, class UV, class WG>
  bool nurbs_get_2d_funct(VF& nurbs_shape_funct, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (distype)
    {
      case Core::FE::CellType::nurbs4:
      {
        return nurbs_get_2d_funct<1, ScalarType>(nurbs_shape_funct, uv, knots, weights);
      }
      case Core::FE::CellType::nurbs9:
      {
        return nurbs_get_2d_funct<2, ScalarType>(nurbs_shape_funct, uv, knots, weights);
      }
      default:
      {
        FOUR_C_THROW("Unknown distype for nurbs element evaluation\n");
      }
    }
    return false;
  }


  /*!
  \brief Evaluate 2d basis functions and derivatives (with
  respect to the coordinates of the reference element)
  of nurbs basis functions.

  \verbatim

         reference element          element in knotspan

               v^                        eta^
                |                           |
           -1   |  +1                 *   *   *   *
         +1 +---+---+ +1             *+***+---+***+*
            |XXX|XXX|         Psi     *   |XXX|   *
          --+---+---+-->    ------>   *   |XXX|   * --->
            |XXX|XXX|   u             *   |XXX|   *   xi
         -1 +---+---+ -1              *+***+---+***+*
           -1   |  +1                  *   *   *   *
                                          *   *
                  \       (knot[0])(degree)   (knot[0])(degree+1)

                    \                       |
                     \                      |
                      \                     |
                       \                    |
                        \                   |
                         \                  |
                          \                 |
                           \                |      +----
                            \               |
                 Phi o Psi   \              | Phi =  +  N_{i,j}(xi,eta)*f_{i,j}
                ^             \             |       /
                N_{i,j}(u,v)   \            |      +----
                                \           |
                                 \          |
                                  \         |
                                   \        |
                                    \       |
                                            v
                                    *
                                    *+-------+*
                                     \XXXXXX|
                                      \XXXXX|
                                       \XXXX|
                                      **+---+**
                                        *   *

                              'real' geometry element

  \endverbatim

  PART   I: EVALUATION OF  BEZIER SHAPE FUNCTIONS

  \verbatim


             B_{i,j}(xi,eta) = N_i(xi) *M_j (eta)


            dB_{i,j}
            --------(xi,eta) = N_i'(xi)*M_j (eta)
               dxi


            dB_{i,j}
            --------(xi,eta) = N_i (xi)*M_j'(eta)
              deta

  \endverbatim

  PART  II: PROJECTING TO NURBS SHAPE FUNCTIONS

  \verbatim

                        w_{i,j} * B_{i,j}(xi,eta)

         N_{i,j} = ----------------------------------
                    +----+

                      +    w_{k,l} * B_{k,l}(xi,eta)
                     /
                    +----+
                     k,l

  \endverbatim

  Derivatives are evaluated using the chain rule


  PART III: TRANSFORMING DERIVATIVES FROM xi/eta TO u/v

  \verbatim

                +-  dxi   dxi  -+   +-   du       -+
                |   ---   ---   |   |   ---   0    |
                |    du    dv   |   |   2.0        |
         Psi' = |               | = |              |
                |  deta  deta   |   |         dv   |
                |  ----  ----   |   |    0   ---   |
                +-  du    dv   -+   +-       2.0  -+

  \endverbatim

  \param nurbs_shape_funct  (o)
  \param nurbs_shape_deriv  (o)
  \param uv                 (i)
  \param knots              (i)
  \param weights            (i)

  \return TRUE, if successful

  */
  template <int degree, typename ScalarType = double, class VF, class MD, class UV, class WG>
  bool nurbs_get_2d_funct_deriv(VF& nurbs_shape_funct, MD& nurbs_shape_deriv, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights)
  {
    // size is the number of control points/basis
    // functions of this element
    const int size = (degree + 1) * (degree + 1);

    // Gausspoint is in [-1:1]x[-1:1]
    //
    // has to be mapped into knotvector evaluation interval

    /*
         reference element          element in knotspan

               v^                        eta^
                |                           |
           -1   |  +1                 *   *   *   *
         +1 +---+---+ +1             *+***+---+***+*
            |XXX|XXX|         Psi     *   |XXX|   *
          --+---+---+-->    ------>   *   |XXX|   * --->
            |XXX|XXX|   u             *   |XXX|   *   xi
         -1 +---+---+ -1             *+***+---+***+*
           -1   |  +1                 *   *   *   *
                                          *   *
                  \       (knot[0])(degree)   (knot[0])(degree+1)

                    \                       |
                     \                      |
                      \                     |
                       \                    |
                        \                   |
                         \                  |
                          \                 |
                           \                |      +----
                            \               |
                 Phi o Psi   \              | Phi =  +  N_{i,j}(xi,eta)*f_{i,j}
                ^             \             |       /
                N_{i,j}(u,v)   \            |      +----
                                \           |
                                 \          |
                                  \         |
                                   \        |
                                    \       |
                                            v
                                    *
                                   *+-------+*
                                     \XXXXXX|
                                      \XXXXX|
                                       \XXXX|
                                      **+---+**
                                        *   *

                              'real' geometry element
    */

    // the jacobian matrix of Psi is of the very
    // simple form
    //
    //               Ju
    //               |
    //               |
    //          +-   du       -+
    //          |   ---   0    |
    //          |   2.0        |
    //   Psi' = |              |
    //          |         dv   |
    //          |    0   ---   |
    //          +-       2.0  -+
    //                    |
    //                    |
    //                    Jv
    //

    double du = (knots[0])(degree + 1) - (knots[0])(degree);
    double dv = (knots[1])(degree + 1) - (knots[1])(degree);

    double Ju = du / 2.0;
    double Jv = dv / 2.0;

    // get mapped coordinates

    ScalarType xi = (knots[0])(degree) + ((uv(0) + 1)) * Ju;
    ScalarType eta = (knots[1])(degree) + ((uv(1) + 1)) * Jv;

    // -----------------------------------------------
    //  PART I: EVALUATION OF  BEZIER SHAPE FUNCTIONS
    // -----------------------------------------------

    std::vector<ScalarType> bezier_shape_funct(size);
    std::vector<std::vector<ScalarType>> bezier_shape_deriv(2, std::vector<ScalarType>(size));

    // allocate bspline polynomials for both direction
    Core::FE::Nurbs::BsplinePolynomial bspline_xi(degree, knots[0]);
    Core::FE::Nurbs::BsplinePolynomial bspline_eta(degree, knots[1]);

    // get temporary variables
    ScalarType bspline_xi_value;
    ScalarType bspline_xi_derivative;
    ScalarType bspline_eta_value;
    ScalarType bspline_eta_derivative;

    // define temporary int variable to compute the
    // number of the basis function from i,j
    int id;

    //
    //  Bezier basis function
    //          |
    //          |
    //       B_{i,j}(xi,eta) = N_i(xi)*M_j(eta)
    //                           |       |
    //                           |       |
    //                  bspline_u_value  |
    //                                   |
    //                             bspline_v_value
    //

    //      dB_{i,j}
    //      --------(xi,eta) = N_i'(xi)*M_j (eta)
    //         dxi
    //
    //      dB_{i,j}
    //      --------(xi,eta) = N_i (xi)*M_j'(eta)
    //        deta

    // loop all basis functions (corresponding to
    // control points)
    for (int rr = 0; rr < degree + 1; ++rr)
    {
      // in first direction:

      // get bsplinevalue and derivative
      bspline_xi.evaluate_bspline_and_deriv(bspline_xi_value, bspline_xi_derivative, xi, rr);

      for (int mm = 0; mm < degree + 1; ++mm)
      {
        // in second direction:

        // get bsplinevalue and derivative
        bspline_eta.evaluate_bspline_and_deriv(bspline_eta_value, bspline_eta_derivative, eta, mm);

        // get the number of the basis function
        id = rr + mm * (degree + 1);

        // add value to bezier_shape_funct
        bezier_shape_funct[id] = bspline_xi_value * bspline_eta_value;

        // add values to bezier_shape_deriv
        bezier_shape_deriv[0][id] = bspline_xi_derivative * bspline_eta_value;
        bezier_shape_deriv[1][id] = bspline_xi_value * bspline_eta_derivative;
      }
    }

    // -----------------------------------------------
    //  PART II: PROJECTING TO NURBS SHAPE FUNCTIONS
    // -----------------------------------------------

    // alloc temporary variable, initialise to zero
    ScalarType sum_funct_weight;
    ScalarType sum_deriv_weight[2];

    sum_funct_weight = 0.0;
    sum_deriv_weight[0] = 0.0;
    sum_deriv_weight[1] = 0.0;

    /*loop all control points, compute sums

                           +----+

         sum_funct_weight =  +    w_{i,j} * B_{i,j}(xi,eta)
                            /
                           +----+


                              +----+
                               \               dB'_{i,j}
         sum_deriv_weight[.] =  +    w_{i,j} * ---------(xi,eta)
                               /                  d.
                              +----+
    */

    for (int rr = 0; rr < size; ++rr)
    {
      sum_funct_weight += weights(rr) * bezier_shape_funct[rr];

      for (int mm = 0; mm < 2; ++mm)
      {
        sum_deriv_weight[mm] += weights(rr) * bezier_shape_deriv[mm][rr];
      }
    }

    if (sum_funct_weight == 0.0) return false;

    /* Compute Nurbs basis functions


                        w_{i,j} * B_{i,j}(xi,eta)

         N_{i,j} = ----------------------------------
                    +----+

                      +    w_{k,l} * B_{k,l}(xi,eta)
                     /
                    +----+
                     k,l
    */

    // Nurbs derivatives are defined by the chain rule
    //
    //                             +- +----+              -+                   +- +----+ -+ |
    //                             |                   |   \             dB_{k,l} |
    //                    dB_{i,j} |    +  w_{k,l}*B_{k,l} |                   |    +
    //                    w_{k,l}*-------- |
    //            w_{i,j}*--------*|   /                   | - w_{i,j}*B_{i,j}*|   / dxi   |
    //                        dxi  |  +----+               |                   |  +----+ |
    // dN_{i,j}                    +-   k,l               -+                   +-   k,l -+
    // -------- =
    // -----------------------------------------------------------------------------------------
    //   dxi                             +- +----+                          -+ 2
    //                                   |   \                               |
    //                                   |    +    w_{k,l} * B_{k,l}(xi,eta) |
    //                                   |   /                               |
    //                                   |  +----+                           |
    //                                   +-   k,l                           -+

    // loop all basis functions
    for (int rr = 0; rr < size; ++rr)
    {
      // get shape function
      nurbs_shape_funct(rr) = weights(rr) * bezier_shape_funct[rr] / sum_funct_weight;

      // loop directions to compute derivatives
      for (int mm = 0; mm < 2; ++mm)
      {
        nurbs_shape_deriv(mm, rr) = bezier_shape_deriv[mm][rr] * sum_funct_weight -
                                    bezier_shape_funct[rr] * sum_deriv_weight[mm];

        nurbs_shape_deriv(mm, rr) *= weights(rr) / (sum_funct_weight * sum_funct_weight);
      }
    }

    // -----------------------------------------------
    //  PART III: TRANSFORMING DERIVATIVES FROM xi/eta
    //            TO u/v
    // -----------------------------------------------

    // we already know the jacobian matrix of psi
    //
    //          +-   dxi   dxi  -+   +-   du       -+
    //          |    ---   ---   |   |   ---   0    |
    //          |     du    dv   |   |   2.0        |
    //   Psi' = |                | = |              |
    //          |   deta  deta   |   |         dv   |
    //          |   ----  ----   |   |    0   ---   |
    //          +-   du    dv   -+   +-       2.0  -+
    //
    // we will obtain the derivatives with respect to
    // u,v just by multiplication

    // loop all basis function derivatives
    for (int rr = 0; rr < size; ++rr)
    {
      nurbs_shape_deriv(0, rr) *= Ju;
      nurbs_shape_deriv(1, rr) *= Jv;
    }

    return true;
  }

  /**
   * @brief Evaluate the 2D NURBS shape functions and their 1st derivatives based on the
   * discretization type of the element.
   */
  template <typename ScalarType = double, class VF, class MD, class UV, class WG>
  bool nurbs_get_2d_funct_deriv(VF& nurbs_shape_funct, MD& nurbs_shape_deriv, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (distype)
    {
      case Core::FE::CellType::nurbs4:
      {
        return nurbs_get_2d_funct_deriv<1, ScalarType>(
            nurbs_shape_funct, nurbs_shape_deriv, uv, knots, weights);
      }
      case Core::FE::CellType::nurbs9:
      {
        return nurbs_get_2d_funct_deriv<2, ScalarType>(
            nurbs_shape_funct, nurbs_shape_deriv, uv, knots, weights);
      }
      default:
      {
        FOUR_C_THROW("Unknown distype for nurbs element evaluation\n");
      }
    }
  }

  /*!

  \brief Evaluate 2d basis functions, first and second derivatives
         (with respect to the coordinates of the reference
         element) of nurbs basis functions.

  \verbatim

         reference element          element in knotspan

               v^                        eta^
                |                           |
           -1   |  +1                 *   *   *   *
         +1 +---+---+ +1             *+***+---+***+*
            |XXX|XXX|         Psi     *   |XXX|   *
          --+---+---+-->    ------>   *   |XXX|   * --->
            |XXX|XXX|   u             *   |XXX|   *   xi
         -1 +---+---+ -1             *+***+---+***+*
           -1   |  +1                 *   *   *   *
                                          *   *
                  \       (knot[0])(degree)   (knot[0])(degree+1)

                    \                       |
                     \                      |
                      \                     |
                       \                    |
                        \                   |
                         \                  |
                          \                 |
                           \                |      +----
                            \               |
                 Phi o Psi   \              | Phi =  +  N_{i,j}(xi,eta)*f_{i,j}
                ^             \             |       /
                N_{i,j}(u,v)   \            |      +----
                                \           |
                                 \          |
                                  \         |
                                   \        |
                                    \       |
                                            v
                                    *
                                   *+-------+*
                                     \XXXXXX|
                                      \XXXXX|
                                       \XXXX|
                                      **+---+**
                                        *   *

                              'real' geometry element

  \endverbatim

  PART   I: EVALUATION OF  BEZIER SHAPE FUNCTIONS

  \verbatim


             B_{i,j}(xi,eta) = N_i(xi) *M_j (eta)


            dB_{i,j}
            --------(xi,eta) = N_i'(xi)*M_j (eta)
               dxi


            dB_{i,j}
            --------(xi,eta) = N_i (xi)*M_j'(eta)
              deta

  \endverbatim

  PART  II: PROJECTING TO NURBS SHAPE FUNCTIONS

  \verbatim

                        w_{i,j} * B_{i,j}(xi,eta)

         N_{i,j} = ----------------------------------
                    +----+

                      +    w_{k,l} * B_{k,l}(xi,eta)
                     /
                    +----+
                      k,l

  \endverbatim

  First and second derivatives are evaluated using
  the chain rule. The second derivatives are already
  quite painful to compute...


  PART III: TRANSFORMING DERIVATIVES FROM xi/eta TO u/v

  \verbatim

                +-  dxi   dxi  -+   +-   du       -+
                |   ---   ---   |   |   ---   0    |
                |    du    dv   |   |   2.0        |
         Psi' = |               | = |              |
                |  deta  deta   |   |         dv   |
                |  ----  ----   |   |    0   ---   |
                +-  du    dv   -+   +-       2.0  -+

  \endverbatim


  \param nurbs_shape_funct  (o)
  \param nurbs_shape_deriv  (o)
  \param nurbs_shape_deriv2 (o)
  \param uv                 (i)
  \param knots              (i)
  \param weights            (i)

  \return TRUE, if successful

  */

  template <int degree, class VF, class MD, class MSD, class UV, class WG>
  bool nurbs_get_2d_funct_deriv_deriv2(VF& nurbs_shape_funct, MD& nurbs_shape_deriv,
      MSD& nurbs_shape_deriv2, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights)
  {
    // size is the number of control points/basis
    // functions of this element
    const int size = (degree + 1) * (degree + 1);

    // Gausspoint is in [-1:1]x[-1:1]
    //
    // has to be mapped into knotvector evaluation interval

    /*
         reference element          element in knotspan

               v^                        eta^
                |                           |
           -1   |  +1                 *   *   *   *
         +1 +---+---+ +1             *+***+---+***+*
            |XXX|XXX|         Psi     *   |XXX|   *
          --+---+---+-->    ------>   *   |XXX|   * --->
            |XXX|XXX|   u             *   |XXX|   *   xi
         -1 +---+---+ -1             *+***+---+***+*
           -1   |  +1                 *   *   *   *
                                          *   *
                  \       (knot[0])(degree)   (knot[0])(degree+1)

                    \                       |
                     \                      |
                      \                     |
                       \                    |
                        \                   |
                         \                  |
                          \                 |
                           \                |      +----
                            \               |
                 Phi o Psi   \              | Phi =  +  N_{i,j}(xi,eta)*f_{i,j}
                ^             \             |       /
                N_{i,j}(u,v)   \            |      +----
                                \           |
                                 \          |
                                  \         |
                                   \        |
                                    \       |
                                            v
                                    *
                                    *+-------+*
                                     \XXXXXX|
                                      \XXXXX|
                                       \XXXX|
                                      **+---+**
                                        *   *

                              'real' geometry element
    */

    // the jacobian matrix of Psi is of the very
    // simple form
    //
    //               Ju
    //               |
    //               |
    //          +-   du       -+
    //          |   ---   0    |
    //          |   2.0        |
    //   Psi' = |              |
    //          |         dv   |
    //          |    0   ---   |
    //          +-       2.0  -+
    //                    |
    //                    |
    //                    Jv
    //

    double du = (knots[0])(degree + 1) - (knots[0])(degree);
    double dv = (knots[1])(degree + 1) - (knots[1])(degree);

    double Ju = du / 2.0;
    double Jv = dv / 2.0;

    // get mapped coordinates

    double xi = (knots[0])(degree) + ((uv(0) + 1)) * Ju;
    double eta = (knots[1])(degree) + ((uv(1) + 1)) * Jv;

    // -----------------------------------------------
    //  PART I: EVALUATION OF  BEZIER SHAPE FUNCTIONS
    // -----------------------------------------------

    Core::LinAlg::SerialDenseVector bezier_shape_funct(size);
    Core::LinAlg::SerialDenseMatrix bezier_shape_deriv(2, size);
    Core::LinAlg::SerialDenseMatrix bezier_shape_deriv2(3, size);

    // allocate bspline polynomials for both direction
    Core::FE::Nurbs::BsplinePolynomial bspline_xi(degree, knots[0]);
    Core::FE::Nurbs::BsplinePolynomial bspline_eta(degree, knots[1]);

    // get temporary doubles for derivatives and
    // values of the above devlared polynomials
    double bspline_xi_value;
    double bspline_eta_value;

    double bspline_xi_derivative;
    double bspline_eta_derivative;

    double bspline_xi_deriv2;
    double bspline_eta_deriv2;

    // define temporary int variable to compute the
    // number of the basis function from i,j
    int id;

    //
    //  Bezier basis function
    //          |
    //          |
    //       B_{i,j}(xi,eta) = N_i(xi)*M_j(eta)
    //                           |       |
    //                           |       |
    //                  bspline_u_value  |
    //                                   |
    //                             bspline_v_value
    //

    //      dB_{i,j}
    //      --------(xi,eta) = N_i'(xi)*M_j (eta)
    //         dxi
    //
    //      dB_{i,j}
    //      --------(xi,eta) = N_i (xi)*M_j'(eta)
    //        deta

    //       2
    //      d B_{i,j}
    //      ---------(xi,eta) = N_i"(xi)*M_j (eta)
    //           2
    //        dxi
    //
    //       2
    //      d B_{i,j}
    //      ---------(xi,eta) = N_i (xi)*M_j"(eta)
    //           2
    //       deta
    //
    //       2
    //      d B_{i,j}
    //      ---------(xi,eta) = N_i'(xi)*M_j'(eta)
    //      dxi deta
    //


    // loop all basis functions (corresponding to
    // control points)
    for (int rr = 0; rr < degree + 1; ++rr)
    {
      // in first direction:

      // get bsplinevalue and derivative
      bspline_xi.evaluate_bspline_first_and_second_deriv(
          bspline_xi_value, bspline_xi_derivative, bspline_xi_deriv2, xi, rr);

      for (int mm = 0; mm < degree + 1; ++mm)
      {
        // in second direction:

        // get bsplinevalue and derivative
        bspline_eta.evaluate_bspline_first_and_second_deriv(
            bspline_eta_value, bspline_eta_derivative, bspline_eta_deriv2, eta, mm);

        // get the number of the basis function
        id = rr + mm * (degree + 1);

        // set value to bezier_shape_funct
        bezier_shape_funct(id) = bspline_xi_value * bspline_eta_value;

        // set values to bezier_shape_deriv
        bezier_shape_deriv(0, id) = bspline_xi_derivative * bspline_eta_value;
        bezier_shape_deriv(1, id) = bspline_xi_value * bspline_eta_derivative;

        // set values to bezier_shape_deriv2
        bezier_shape_deriv2(0, id) = bspline_xi_deriv2 * bspline_eta_value;
        bezier_shape_deriv2(1, id) = bspline_xi_value * bspline_eta_deriv2;
        bezier_shape_deriv2(2, id) = bspline_xi_derivative * bspline_eta_derivative;
      }
    }

    // -----------------------------------------------
    //  PART II: PROJECTING TO NURBS SHAPE FUNCTIONS
    // -----------------------------------------------

    // alloc temporary doubles, initialise to zero
    double sum_funct_weight;
    double sum_deriv_weight[2];
    double sum_deriv2_weight[3];

    sum_funct_weight = 0.0;
    sum_deriv_weight[0] = 0.0;
    sum_deriv_weight[1] = 0.0;
    sum_deriv2_weight[0] = 0.0;
    sum_deriv2_weight[1] = 0.0;
    sum_deriv2_weight[2] = 0.0;

    /*loop all control points, compute sums

                           +----+

         sum_funct_weight =  +    w_{i,j} * B_{i,j}(xi,eta)
                            /
                           +----+


                              +----+
                               \               dB_{i,j}
         sum_deriv_weight[0] =  +    w_{i,j} * --------(xi,eta)
                               /                  dxi
                              +----+

                              +----+
                               \               dB_{i,j}
         sum_deriv_weight[1] =  +    w_{i,j} * --------(xi,eta)
                               /                 deta
                              +----+



                               +----+            2
                                \               d B_{i,j}
         sum_deriv2_weight[0] =  +    w_{i,j} * --------(xi,eta)
                                /                     2
                               +----+              dxi

                               +----+            2
                                \               d B_{i,j}
         sum_deriv2_weight[1] =  +    w_{i,j} * ---------(xi,eta)
                                /                     2
                               +----+             deta

                               +----+            2
                                \               d B_{i,j}
         sum_deriv2_weight[2] =  +    w_{i,j} * ---------(xi,eta)
                                /               dxi deta
                               +----+

    */

    for (int rr = 0; rr < size; ++rr)
    {
      sum_funct_weight += weights(rr) * bezier_shape_funct(rr);

      for (int mm = 0; mm < 2; ++mm)
      {
        sum_deriv_weight[mm] += weights(rr) * bezier_shape_deriv(mm, rr);
        sum_deriv2_weight[mm] += weights(rr) * bezier_shape_deriv2(mm, rr);
      }
      sum_deriv2_weight[2] += weights(rr) * bezier_shape_deriv2(2, rr);
    }

    if (sum_funct_weight == 0.0) return false;

    /* Compute Nurbs basis functions


                        w_{i,j} * B_{i,j}(xi,eta)

         N_{i,j} = ----------------------------------
                    +----+

                      +    w_{k,l} * B_{k,l}(xi,eta)
                     /
                    +----+
                     k,l
    */

    /* Nurbs derivatives are defined by the chain rule

                                   +- +----+              -+                   +- +----+ -+ |
       |                   |   \             dB_{k,l} | dB_{i,j} |    +  w_{k,l}*B_{k,l} | | +
       w_{k,l}*-------- | w_{i,j}*--------*|   /                   | - w_{i,j}*B_{i,j}*|   / dxi
       | dxi  |  +----+               |                   |  +----+                  | dN_{i,j}
       +-   k,l               -+                   +-   k,l                  -+
       -------- =
       -----------------------------------------------------------------------------------------
         dxi                             +- +----+                          -+ 2
                                         |   \                               |
                                         |    +    w_{k,l} * B_{k,l}(xi,eta) |
                                         |   /                               |
                                         |  +----+                           |
                                         +-   k,l                           -+
    */

    /* Nurbs second derivatives are calculated by a
       second application of the chain rule



                                                           +-
                                                           |
                                                           |
                                                           |
                                w_{i,j}                    |   2
       d N_{i,j}                                           |  d B_{i,j}
       --------- = ------------------------------------- * |  --------- -
       dxi  dxi    +- +----+                          -+   |  dxi  dxi
          k    l   |   \                               |   |     k    l
                   |    +    w_{n,m} * B_{n,m}(xi,eta) |   |
                   |   /                               |   |
                   |  +----+                           |   |
                   +-   n,m                           -+   +-

                     +- +----+                           -+
            dB_{i,j} |   \               dB_{n,m}         |
            -------- |    +    w_{n,m} * --------(xi,eta) |
             dxi     |   /                  dxi           |
                k    |  +----+                 l          |
                     +-   n,m                            -+
          - ----------------------------------------------- -
                 +- +----+                          -+
                 |   \                               |
                 |    +    w_{n,m} * B_{n,m}(xi,eta) |
                 |   /                               |
                 |  +----+                           |
                 +-   n,m                           -+



                     +- +----+                           -+
            dB_{i,j} |   \               dB_{n,m}         |
            -------- |    +    w_{n,m} * --------(xi,eta) |
             dxi     |   /                  dxi           |
                l    |  +----+                 k          |
                     +-   n,m                            -+
          - ----------------------------------------------- -
                 +- +----+                          -+
                 |   \                               |
                 |    +    w_{n,m} * B_{n,m}(xi,eta) |
                 |   /                               |
                 |  +----+                           |
                 +-   n,m                           -+


                       +- +----+            2               -+
                       |   \               d B_{n,m}         |
                       |    +    w_{n,m} * ---------(xi,eta) |
             B_{i,j} * |   /               dxi  dxi          |
                       |  +----+              k    l         |
                       +-   n,m                             -+
          - -------------------------------------------------- +
                   +- +----+                          -+
                   |   \                               |
                   |    +    w_{n,m} * B_{n,m}(xi,eta) |
                   |   /                               |
                   |  +----+                           |
                   +-   n,m                           -+


                           +- +----+                            -+   +- +----+ -+  -+ |
       dB_{n,m}          |   |   \               dB_{n,m}          |   | |    +    w_{n,m} *
       ---------(xi,eta) |   |    +    w_{n,m} * ---------(xi,eta) |   | 2 * B_{i,j} * |   / dxi
       | * |   /                 dxi             |   | |  +----+                k            |
       |  +----+                l            |   |
                           +-   n,m                             -+   +-   n,m -+   |
          +
       ------------------------------------------------------------------------------------------------
       |
                                    +- +----+                          -+ 2 | |   \ | | |    +
       w_{n,m} * B_{n,m}(xi,eta) |                                      | |   / | | |  +----+ |
       |
                                    +-   n,m                           -+ -+
    */


    // loop all basis functions
    for (int rr = 0; rr < size; ++rr)
    {
      // get shape function
      nurbs_shape_funct(rr) = weights(rr) * bezier_shape_funct(rr) / sum_funct_weight;

      // loop directions to compute derivatives
      for (int mm = 0; mm < 2; ++mm)
      {
        nurbs_shape_deriv(mm, rr) = bezier_shape_deriv(mm, rr) * sum_funct_weight -
                                    bezier_shape_funct(rr) * sum_deriv_weight[mm];

        nurbs_shape_deriv(mm, rr) *= weights(rr) / (sum_funct_weight * sum_funct_weight);
      }

      // we apply a Horner-type scheme for the
      // multiplication with sum_funct_weight
      nurbs_shape_deriv2(0, rr) =
          2 * bezier_shape_funct(rr) * sum_deriv_weight[0] * sum_deriv_weight[0];
      nurbs_shape_deriv2(0, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(0, rr) -= bezier_shape_funct(rr) * sum_deriv2_weight[0];
      nurbs_shape_deriv2(0, rr) -= bezier_shape_deriv(0, rr) * sum_deriv_weight[0];
      nurbs_shape_deriv2(0, rr) -= bezier_shape_deriv(0, rr) * sum_deriv_weight[0];
      nurbs_shape_deriv2(0, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(0, rr) += bezier_shape_deriv2(0, rr);
      nurbs_shape_deriv2(0, rr) *= weights(rr) / sum_funct_weight;

      nurbs_shape_deriv2(1, rr) =
          2 * bezier_shape_funct(rr) * sum_deriv_weight[1] * sum_deriv_weight[1];
      nurbs_shape_deriv2(1, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(1, rr) -= bezier_shape_funct(rr) * sum_deriv2_weight[1];
      nurbs_shape_deriv2(1, rr) -= bezier_shape_deriv(1, rr) * sum_deriv_weight[1];
      nurbs_shape_deriv2(1, rr) -= bezier_shape_deriv(1, rr) * sum_deriv_weight[1];
      nurbs_shape_deriv2(1, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(1, rr) += bezier_shape_deriv2(1, rr);
      nurbs_shape_deriv2(1, rr) *= weights(rr) / sum_funct_weight;

      nurbs_shape_deriv2(2, rr) =
          2 * bezier_shape_funct(rr) * sum_deriv_weight[0] * sum_deriv_weight[1];
      nurbs_shape_deriv2(2, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(2, rr) -= bezier_shape_funct(rr) * sum_deriv2_weight[2];
      nurbs_shape_deriv2(2, rr) -= bezier_shape_deriv(0, rr) * sum_deriv_weight[1];
      nurbs_shape_deriv2(2, rr) -= bezier_shape_deriv(1, rr) * sum_deriv_weight[0];
      nurbs_shape_deriv2(2, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(2, rr) += bezier_shape_deriv2(2, rr);
      nurbs_shape_deriv2(2, rr) *= weights(rr) / sum_funct_weight;
    }

    // -----------------------------------------------
    //  PART III: TRANSFORMING DERIVATIVES FROM xi/eta
    //            TO u/v
    // -----------------------------------------------

    // we already know the jacobian matrix of psi
    //
    //          +-   dxi   dxi  -+   +-   du       -+
    //          |    ---   ---   |   |   ---   0    |
    //          |     du    dv   |   |   2.0        |
    //   Psi' = |                | = |              |
    //          |   deta  deta   |   |         dv   |
    //          |   ----  ----   |   |    0   ---   |
    //          +-   du    dv   -+   +-       2.0  -+
    //
    // we will obtain the derivatives with respect to
    // u,v just by multiplication

    // loop all basis function derivatives
    for (int rr = 0; rr < size; ++rr)
    {
      nurbs_shape_deriv(0, rr) *= Ju;
      nurbs_shape_deriv(1, rr) *= Jv;

      nurbs_shape_deriv2(0, rr) *= Ju * Ju;
      nurbs_shape_deriv2(1, rr) *= Jv * Jv;
      nurbs_shape_deriv2(2, rr) *= Ju * Jv;
    }

    return true;
  }

  /**
   * @brief Evaluate the 2D NURBS shape functions and their 2nd derivatives based on the
   * discretization type of the element.
   */
  template <class VF, class MD, class MSD, class UV, class WG>
  bool nurbs_get_2d_funct_deriv_deriv2(VF& nurbs_shape_funct, MD& nurbs_shape_deriv,
      MSD& nurbs_shape_deriv2, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (distype)
    {
      case Core::FE::CellType::nurbs4:
      {
        return nurbs_get_2d_funct_deriv_deriv2<1>(
            nurbs_shape_funct, nurbs_shape_deriv, nurbs_shape_deriv2, uv, knots, weights);
      }
      case Core::FE::CellType::nurbs9:
      {
        return nurbs_get_2d_funct_deriv_deriv2<2>(
            nurbs_shape_funct, nurbs_shape_deriv, nurbs_shape_deriv2, uv, knots, weights);
      }
      default:
      {
        FOUR_C_THROW("Unknown distype for nurbs element evaluation\n");
      }
    }
  }


  /*!
  \brief Evaluate 3d basis functions of nurbs basis functions.

  \verbatim

         reference element          element in knotspan

                                                nu ^
                                                   |
                  w                                |
                  |                                |
                  |                                |
                +-|--+----+                        +---------+
               /  | /    /|                       /         /|
              /   |/    / |                      /         / |
             +----+----+  +        Psi          /         /  |
            /    / 1  /| /|      ------->      /         /   |
           /    /    / |/ |                   /         /    |
          +----+----+ 1+-------- v           +---------+     +-------->
          |    |    | /| /                   |         |    /          xi
          |    |1   |/ |/                    |         |   /
          +----+----+  +                     |         |  /
          |   /|    | /                      |         | /
          |  / |    |/                       |         |/
          +-/--+----+                        +---------+
           /                                /|         |
        u /     \                          / |         |
                 \                        /  |         |
                  \                      /   |         |
                   \                eta v    |         |
                    \                        |         |
                     \                       |         |
                      \                      |         |
                       \                     |         |
                        \       (knot[0])(degree)   (knot[0])(degree+1)

                          \                       |
                           \                      |
                            \                     |
                             \                    |
                              \                   |
                               \                  |
                                \                 |
                                 \                |      +----
                                  \               |
                       Phi o Psi   \              | Phi =  +  N_{i,j,k}(xi,eta,nu)*f_{i,j,k}
                  ^                 \             |       /
                  N_{i,j,k}(u,v,w)   \            |      +----
                                      \           |
                                       \          |
                                        \         |
                                         \        |
                                          \       |

                                               _____
                                            +--     --+
                                           /         /
                                          /         /   |
                                         /         /    |
                                        /         /     |
                                       / _____   /     /
                                      +--     --+     +
                                       \       /     /
                                        |     |     /
                                        |     |    /
                                        |     |   /
                                       /       \ /
                                      +---------+

                               'real' geometry element

  \endverbatim

  PART   I: EVALUATION OF  BEZIER SHAPE FUNCTIONS

  \verbatim

             B_{i,j,k}(xi,eta,nu) = N_i(xi) *M_j (eta)*L_k (nu)

  \endverbatim

  PART  II: PROJECTING TO NURBS SHAPE FUNCTIONS

  \verbatim

                          w_{i,j,k} * B_{i,j,k}(xi,eta,nu)

         N_{i,j,k} = ---------------------------------------
                     +----+

                       +    w_{s,t,r} * B_{s,t,r}(xi,eta,nu)
                      /
                     +----+
                     s,t,r

  \endverbatim


  \param nurbs_shape_funct  (o)
  \param uv                 (i)
  \param knots              (i)
  \param weights            (i)

  \return TRUE, if successful

  */

  template <int degree, class V, class UV, class WG>
  bool nurbs_get_3d_funct(V& nurbs_shape_funct, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights)
  {
    // size is the number of control points/basis
    // functions of this element
    const int size = (degree + 1) * (degree + 1) * (degree + 1);

    // Gausspoint is in [-1:1]x[-1:1]x[-1:1]
    //
    // has to be mapped into knotvector evaluation interval

    /*
         reference element                element in knotspan

                                                nu ^
                                                   |
                  w                                |
                  |                                |
                  |                                |
                +-|--+----+                        +---------+
               /  | /    /|                       /         /|
              /   |/    / |                      /         / |
             +----+----+  +        Psi          /         /  |
            /    / 1  /| /|      ------->      /         /   |
           /    /    / |/ |                   /         /    |
          +----+----+ 1+-------- v           +---------+     +-------->
          |    |    | /| /                   |         |    /          xi
          |    |1   |/ |/                    |         |   /
          +----+----+  +                     |         |  /
          |   /|    | /                      |         | /
          |  / |    |/                       |         |/
          +-/--+----+                        +---------+
           /                                /|         |
        u /     \                          / |         |
                 \                        /  |         |
                  \                      /   |         |
                   \                eta v    |         |
                    \                        |         |
                     \                       |         |
                      \                      |         |
                       \                     |         |
                        \       (knot[0])(degree)   (knot[0])(degree+1)

                          \                       |
                           \                      |
                            \                     |
                             \                    |
                              \                   |
                               \                  |
                                \                 |
                                 \                |      +----
                                  \               |
                       Phi o Psi   \              | Phi =  +  N_{i,j,k}(xi,eta,nu)*f_{i,j,k}
                  ^                 \             |       /
                  N_{i,j,k}(u,v,w)   \            |      +----
                                      \           |
                                       \          |
                                        \         |
                                         \        |
                                          \       |

                                               _____
                                            +--     --+
                                           /         /
                                          /         /   |
                                         /         /    |
                                        /         /     |
                                       / _____   /     /
                                      +--     --+     +
                                       \       /     /
                                        |     |     /
                                        |     |    /
                                        |     |   /
                                       /       \ /
                                      +---------+

                               'real' geometry element

    */

    // the jacobian matrix of Psi is of the very
    // simple form
    //
    //
    //                                        Ju     Jv
    //                                        |       |
    //                                        |       |
    //           +   dxi   dxi   dxi -+   +-  du      |    -+
    //           |   ---   ---   ---  |   |  ---   0  | 0   |
    //           |    du    dv    dw  |   |  2.0      |     |
    //           |                    |   |          /      |
    //           |  deta  deta  deta  |   |        dv       |
    //    Psi' = |  ----  ----  ----  | = |   0   ---   0   |
    //           |   du    dv    dw   |   |       2.0       |
    //           |                    |   |                 |
    //           |   dnu   dnu   dnu  |   |             dw  |
    //           |  ----  ----  ----  |   |   0    0   ---  |
    //           +-  du    dv    dw  -+   +-           2.0 -+
    //                                                  |
    //                                                  |
    //                                                  Jw

    double du = (knots[0])(degree + 1) - (knots[0])(degree);
    double dv = (knots[1])(degree + 1) - (knots[1])(degree);
    double dw = (knots[2])(degree + 1) - (knots[2])(degree);

    double Ju = du / 2.0;
    double Jv = dv / 2.0;
    double Jw = dw / 2.0;

    // get mapped coordinates

    double xi = (knots[0])(degree) + ((uv(0) + 1)) * Ju;
    double eta = (knots[1])(degree) + ((uv(1) + 1)) * Jv;
    double nu = (knots[2])(degree) + ((uv(2) + 1)) * Jw;

    // -----------------------------------------------
    //  PART I: EVALUATION OF  BEZIER SHAPE FUNCTIONS
    // -----------------------------------------------

    Core::LinAlg::SerialDenseVector bezier_shape_funct(size);

    // allocate bspline polynomials for both direction
    Core::FE::Nurbs::BsplinePolynomial bspline_xi(degree, knots[0]);
    Core::FE::Nurbs::BsplinePolynomial bspline_eta(degree, knots[1]);
    Core::FE::Nurbs::BsplinePolynomial bspline_nu(degree, knots[2]);

    // get temporary doubles for derivatives and
    // values of the above devlared polynomials
    double bspline_xi_value;
    double bspline_eta_value;
    double bspline_nu_value;

    // define temporary int variable to compute the
    // number of the basis function from i,j,k
    int id;


    /*
         Bezier basis function
                  |
                  |
             B_{i,j,k}(xi,eta,nu) = N_i(xi) *M_j (eta)*L_k (nu)
                                    |        |         |
                                    |        |         |
                             bspline_u_value |         |
                                             |         |
                                      bspline_v_value  |
                                                       |
                                                 bspline_w_value
    */

    // loop all basis functions (corresponding to
    // control points)
    for (int rr = 0; rr < degree + 1; ++rr)
    {
      // in first direction:

      // get bsplinevalue and derivative
      bspline_xi.evaluate_bspline(bspline_xi_value, xi, rr);

      for (int mm = 0; mm < degree + 1; ++mm)
      {
        // in second direction:

        // get bsplinevalue and derivative
        bspline_eta.evaluate_bspline(bspline_eta_value, eta, mm);

        for (int nn = 0; nn < degree + 1; ++nn)
        {
          // in third direction:

          // get bsplinevalue and derivative
          bspline_nu.evaluate_bspline(bspline_nu_value, nu, nn);

          // get the number of the basis function
          id = rr + (degree + 1) * (mm + nn * (degree + 1));

          // set value to bezier_shape_funct
          bezier_shape_funct(id) = bspline_xi_value * bspline_eta_value * bspline_nu_value;
        }
      }
    }

    // -----------------------------------------------
    //  PART II: PROJECTING TO NURBS SHAPE FUNCTIONS
    // -----------------------------------------------

    // alloc temporary doubles, initialise to zero
    double sum_funct_weight;

    sum_funct_weight = 0.0;

    /*loop all control points, compute sums

                           +----+

         sum_funct_weight =  +    w_{i,j,k} * B_{i,j,k}(xi,eta,nu)
                            /
                           +----+

    */

    for (int rr = 0; rr < size; ++rr)
    {
      sum_funct_weight += weights(rr) * bezier_shape_funct(rr);
    }

    if (sum_funct_weight == 0.0) return false;

    // loop all basis functions
    for (int rr = 0; rr < size; ++rr)
    {
      /* Compute Nurbs basis functions


                        w_{i,j,k} * B_{i,j,k}(xi,eta,nu)

         N_{i,j,k} = ---------------------------------------
                     +----+

                       +    w_{r,s,t} * B_{r,s,t}(xi,eta,nu)
                      /
                     +----+
                     r,s,t
      */

      nurbs_shape_funct(rr) = weights(rr) * bezier_shape_funct(rr) / sum_funct_weight;
    }

    return true;
  }

  /**
   * @brief Evaluate the 3D NURBS shape functions based on the discretization type of the element.
   */
  template <class V, class UV, class WG>
  bool nurbs_get_3d_funct(V& nurbs_shape_funct, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (distype)
    {
      case Core::FE::CellType::nurbs8:
      {
        return nurbs_get_3d_funct<1>(nurbs_shape_funct, uv, knots, weights);
      }
      case Core::FE::CellType::nurbs27:
      {
        return nurbs_get_3d_funct<2>(nurbs_shape_funct, uv, knots, weights);
      }
      default:
      {
        FOUR_C_THROW("Unknown distype for nurbs element evaluation\n");
      }
    }
  }


  /*!
  \brief Evaluate 3d basis functions and derivatives (with
  respect to the coordinates of the reference element)
  of nurbs basis functions.

  \verbatim

         reference element          element in knotspan

                                                nu ^
                                                   |
                  w                                |
                  |                                |
                  |                                |
                +-|--+----+                        +---------+
               /  | /    /|                       /         /|
              /   |/    / |                      /         / |
             +----+----+  +        Psi          /         /  |
            /    / 1  /| /|      ------->      /         /   |
           /    /    / |/ |                   /         /    |
          +----+----+ 1+-------- v           +---------+     +-------->
          |    |    | /| /                   |         |    /          xi
          |    |1   |/ |/                    |         |   /
          +----+----+  +                     |         |  /
          |   /|    | /                      |         | /
          |  / |    |/                       |         |/
          +-/--+----+                        +---------+
           /                                /|         |
        u /     \                          / |         |
                 \                        /  |         |
                  \                      /   |         |
                   \                eta v    |         |
                    \                        |         |
                     \                       |         |
                      \                      |         |
                       \                     |         |
                        \       (knot[0])(degree)   (knot[0])(degree+1)

                          \                       |
                           \                      |
                            \                     |
                             \                    |
                              \                   |
                               \                  |
                                \                 |
                                 \                |      +----
                                  \               |
                       Phi o Psi   \              | Phi =  +  N_{i,j,k}(xi,eta,nu)*f_{i,j,k}
                  ^                 \             |       /
                  N_{i,j,k}(u,v,w)   \            |      +----
                                      \           |
                                       \          |
                                        \         |
                                         \        |
                                          \       |

                                               _____
                                            +--     --+
                                           /         /
                                          /         /   |
                                         /         /    |
                                        /         /     |
                                       / _____   /     /
                                      +--     --+     +
                                       \       /     /
                                        |     |     /
                                        |     |    /
                                        |     |   /
                                       /       \ /
                                      +---------+

                               'real' geometry element

  \endverbatim

  PART   I: EVALUATION OF  BEZIER SHAPE FUNCTIONS

  \verbatim


             B_{i,j,k}(xi,eta,nu) = N_i(xi) *M_j (eta)*L_k (nu)


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i'(xi)*M_j (eta)*L_k (nu)
                dxi


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i (xi)*M_j'(eta)*L_k (nu)
               deta


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i (xi)*M_j (eta)*L_k'(nu)
               dnu

  \endverbatim

  PART  II: PROJECTING TO NURBS SHAPE FUNCTIONS

  \verbatim

                          w_{i,j,k} * B_{i,j,k}(xi,eta,nu)

         N_{i,j,k} = ---------------------------------------
                     +----+

                       +    w_{s,t,r} * B_{s,t,r}(xi,eta,nu)
                     /
                    +----+
                    s,t,r

  \endverbatim

  Derivatives are evaluated using the chain rule


  PART III: TRANSFORMING DERIVATIVES FROM xi/eta/nu TO u/v/w

  \verbatim

                +   dxi   dxi   dxi -+   +-  du           -+
                |   ---   ---   ---  |   |  ---   0    0   |
                |    du    dv    dw  |   |  2.0            |
                |                    |   |                 |
                |  deta  deta  deta  |   |        dv       |
         Psi' = |  ----  ----  ----  | = |   0   ---   0   |
                |   du    dv    dw   |   |       2.0       |
                |                    |   |                 |
                |   dnu   dnu   dnu  |   |             dw  |
                |  ----  ----  ----  |   |   0    0   ---  |
                +-  du    dv    dw  -+   +-           2.0 -+

  \endverbatim

  \param nurbs_shape_funct  (o)
  \param nurbs_shape_deriv  (o)
  \param uv                 (i)
  \param knots              (i)
  \param weights            (i)

  \return TRUE, if successful

  */

  template <int degree, class VF, class UV, class MD, class WG>
  bool nurbs_get_3d_funct_deriv(VF& nurbs_shape_funct, MD& nurbs_shape_deriv, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights)
  {
    // size is the number of control points/basis
    // functions of this element
    const int size = (degree + 1) * (degree + 1) * (degree + 1);

    // Gausspoint is in [-1:1]x[-1:1]x[-1:1]
    //
    // has to be mapped into knotvector evaluation interval

    /*
         reference element                element in knotspan

                                                nu ^
                                                   |
                  w                                |
                  |                                |
                  |                                |
                +-|--+----+                        +---------+
               /  | /    /|                       /         /|
              /   |/    / |                      /         / |
             +----+----+  +        Psi          /         /  |
            /    / 1  /| /|      ------->      /         /   |
           /    /    / |/ |                   /         /    |
          +----+----+ 1+-------- v           +---------+     +-------->
          |    |    | /| /                   |         |    /          xi
          |    |1   |/ |/                    |         |   /
          +----+----+  +                     |         |  /
          |   /|    | /                      |         | /
          |  / |    |/                       |         |/
          +-/--+----+                        +---------+
           /                                /|         |
        u /     \                          / |         |
                 \                        /  |         |
                  \                      /   |         |
                   \                eta v    |         |
                    \                        |         |
                     \                       |         |
                      \                      |         |
                       \                     |         |
                        \       (knot[0])(degree)   (knot[0])(degree+1)

                          \                       |
                           \                      |
                            \                     |
                             \                    |
                              \                   |
                               \                  |
                                \                 |
                                 \                |      +----
                                  \               |
                       Phi o Psi   \              | Phi =  +  N_{i,j,k}(xi,eta,nu)*f_{i,j,k}
                  ^                 \             |       /
                  N_{i,j,k}(u,v,w)   \            |      +----
                                      \           |
                                       \          |
                                        \         |
                                         \        |
                                          \       |

                                               _____
                                            +--     --+
                                           /         /
                                          /         /   |
                                         /         /    |
                                        /         /     |
                                       / _____   /     /
                                      +--     --+     +
                                       \       /     /
                                        |     |     /
                                        |     |    /
                                        |     |   /
                                       /       \ /
                                      +---------+

                               'real' geometry element

    */

    // the jacobian matrix of Psi is of the very
    // simple form
    //
    //
    //                                        Ju     Jv
    //                                        |       |
    //                                        |       |
    //           +   dxi   dxi   dxi -+   +-  du      |    -+
    //           |   ---   ---   ---  |   |  ---   0  | 0   |
    //           |    du    dv    dw  |   |  2.0      |     |
    //           |                    |   |          /      |
    //           |  deta  deta  deta  |   |        dv       |
    //    Psi' = |  ----  ----  ----  | = |   0   ---   0   |
    //           |   du    dv    dw   |   |       2.0       |
    //           |                    |   |                 |
    //           |   dnu   dnu   dnu  |   |             dw  |
    //           |  ----  ----  ----  |   |   0    0   ---  |
    //           +-  du    dv    dw  -+   +-           2.0 -+
    //                                                  |
    //                                                  |
    //                                                  Jw

    double du = (knots[0])(degree + 1) - (knots[0])(degree);
    double dv = (knots[1])(degree + 1) - (knots[1])(degree);
    double dw = (knots[2])(degree + 1) - (knots[2])(degree);

    double Ju = du / 2.0;
    double Jv = dv / 2.0;
    double Jw = dw / 2.0;

    // get mapped coordinates

    double xi = (knots[0])(degree) + ((uv(0) + 1)) * Ju;
    double eta = (knots[1])(degree) + ((uv(1) + 1)) * Jv;
    double nu = (knots[2])(degree) + ((uv(2) + 1)) * Jw;

    // -----------------------------------------------
    //  PART I: EVALUATION OF  BEZIER SHAPE FUNCTIONS
    // -----------------------------------------------

    Core::LinAlg::SerialDenseVector bezier_shape_funct(size);
    Core::LinAlg::SerialDenseMatrix bezier_shape_deriv(3, size);

    // allocate bspline polynomials for both direction
    Core::FE::Nurbs::BsplinePolynomial bspline_xi(degree, knots[0]);
    Core::FE::Nurbs::BsplinePolynomial bspline_eta(degree, knots[1]);
    Core::FE::Nurbs::BsplinePolynomial bspline_nu(degree, knots[2]);

    // get temporary doubles for derivatives and
    // values of the above devlared polynomials
    double bspline_xi_value;
    double bspline_eta_value;
    double bspline_nu_value;

    double bspline_xi_derivative;
    double bspline_eta_derivative;
    double bspline_nu_derivative;

    // define temporary int variable to compute the
    // number of the basis function from i,j,k
    int id;


    /*
         Bezier basis function
                  |
                  |
             B_{i,j,k}(xi,eta,nu) = N_i(xi) *M_j (eta)*L_k (nu)
                                    |        |         |
                                    |        |         |
                             bspline_u_value |         |
                                             |         |
                                      bspline_v_value  |
                                                       |
                                                 bspline_w_value


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i'(xi)*M_j (eta)*L_k (nu)
                dxi


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i (xi)*M_j'(eta)*L_k (nu)
               deta


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i (xi)*M_j (eta)*L_k'(nu)
               dnu

    */

    // loop all basis functions (corresponding to
    // control points)
    for (int rr = 0; rr < degree + 1; ++rr)
    {
      // in first direction:

      // get bsplinevalue and derivative
      bspline_xi.evaluate_bspline_and_deriv(bspline_xi_value, bspline_xi_derivative, xi, rr);

      for (int mm = 0; mm < degree + 1; ++mm)
      {
        // in second direction:

        // get bsplinevalue and derivative
        bspline_eta.evaluate_bspline_and_deriv(bspline_eta_value, bspline_eta_derivative, eta, mm);

        for (int nn = 0; nn < degree + 1; ++nn)
        {
          // in third direction:

          // get bsplinevalue and derivative
          bspline_nu.evaluate_bspline_and_deriv(bspline_nu_value, bspline_nu_derivative, nu, nn);

          // get the number of the basis function
          id = rr + (degree + 1) * (mm + nn * (degree + 1));

          // set value to bezier_shape_funct
          bezier_shape_funct(id) = bspline_xi_value * bspline_eta_value * bspline_nu_value;

          // set values to bezier_shape_deriv
          bezier_shape_deriv(0, id) = bspline_xi_derivative * bspline_eta_value * bspline_nu_value;
          bezier_shape_deriv(1, id) = bspline_xi_value * bspline_eta_derivative * bspline_nu_value;
          bezier_shape_deriv(2, id) = bspline_xi_value * bspline_eta_value * bspline_nu_derivative;
        }
      }
    }

    // -----------------------------------------------
    //  PART II: PROJECTING TO NURBS SHAPE FUNCTIONS
    // -----------------------------------------------

    // alloc temporary doubles, initialise to zero
    double sum_funct_weight;
    double sum_deriv_weight[3];

    sum_funct_weight = 0.0;
    sum_deriv_weight[0] = 0.0;
    sum_deriv_weight[1] = 0.0;
    sum_deriv_weight[2] = 0.0;

    /*loop all control points, compute sums

                           +----+

         sum_funct_weight =  +    w_{i,j,k} * B_{i,j,k}(xi,eta,nu)
                            /
                           +----+


                              +----+
                               \                 dB_{i,j,k}
         sum_deriv_weight[0] =  +    w_{i,j,k} * ----------(xi,eta,nu)
                               /                     dxi
                              +----+

                              +----+
                               \                 dB_{i,j,k}
         sum_deriv_weight[1] =  +    w_{i,j,k} * ----------(xi,eta,nu)
                               /                     deta
                              +----+

                              +----+
                               \                 dB_{i,j,k}
         sum_deriv_weight[2] =  +    w_{i,j,k} * ----------(xi,eta,nu)
                               /                     dnu
                              +----+

    */

    for (int rr = 0; rr < size; ++rr)
    {
      sum_funct_weight += weights(rr) * bezier_shape_funct(rr);

      for (int mm = 0; mm < 3; ++mm)
      {
        sum_deriv_weight[mm] += weights(rr) * bezier_shape_deriv(mm, rr);
      }
    }

    if (sum_funct_weight == 0.0) return false;

    // loop all basis functions
    for (int rr = 0; rr < size; ++rr)
    {
      /* Compute Nurbs basis functions


                        w_{i,j,k} * B_{i,j,k}(xi,eta,nu)

         N_{i,j,k} = ---------------------------------------
                     +----+

                       +    w_{r,s,t} * B_{r,s,t}(xi,eta,nu)
                      /
                     +----+
                     r,s,t
      */
      nurbs_shape_funct(rr) = weights(rr) * bezier_shape_funct(rr) / sum_funct_weight;

      /* Nurbs derivatives are defined by the chain rule

                                         +- +----+                  -+                       +-
       +----+                     -+ |   \                       |                       |
       dB_{r,s,t} | dB_{i,j,k} |    +  w_{r,s,t}*B_{r,s,t} |                       |    +
       w_{r,s,t}*---------- | w_{i,j,k}*----------*|   /                       | -
       w_{i,j,k}*B_{i,j,k}*|   /                   dxi    | dxi    |  +----+                   |
       |  +----+                      | dN_{i,j,k}                        +-  r,s,t -+ +-  r,s,t
       -+
       ---------- =
       ---------------------------------------------------------------------------------------------------------
           dxi                           +- +----+                                 -+ 2
                                         |   \                                      |
                                         |    +    w_{r,s,t} * B_{r,s,t}(xi,eta,nu) |
                                         |   /                                      |
                                         |  +----+                                  |
                                         +-  r,s,t                                 -+
       */
      // loop directions to compute derivatives
      for (int mm = 0; mm < 3; ++mm)
      {
        nurbs_shape_deriv(mm, rr) = bezier_shape_deriv(mm, rr) * sum_funct_weight -
                                    bezier_shape_funct(rr) * sum_deriv_weight[mm];

        nurbs_shape_deriv(mm, rr) *= weights(rr) / (sum_funct_weight * sum_funct_weight);
      }
    }


    // -----------------------------------------------
    //  PART III: TRANSFORMING DERIVATIVES FROM xi/eta
    //            TO u/v
    // -----------------------------------------------

    // we already know the jacobian matrix of psi
    //
    //            +   dxi   dxi   dxi -+   +-  du           -+
    //            |   ---   ---   ---  |   |  ---   0    0   |
    //            |    du    dv    dw  |   |  2.0            |
    //            |                    |   |                 |
    //            |  deta  deta  deta  |   |        dv       |
    //     Psi' = |  ----  ----  ----  | = |   0   ---   0   |
    //            |   du    dv    dw   |   |       2.0       |
    //            |                    |   |                 |
    //            |   dnu   dnu   dnu  |   |             dw  |
    //            |  ----  ----  ----  |   |   0    0   ---  |
    //            +-  du    dv    dw  -+   +-           2.0 -+
    //
    //
    // we will obtain the derivatives with respect to
    // u,v,w just by multiplication

    // loop all basis function derivatives
    for (int rr = 0; rr < size; ++rr)
    {
      nurbs_shape_deriv(0, rr) *= Ju;
      nurbs_shape_deriv(1, rr) *= Jv;
      nurbs_shape_deriv(2, rr) *= Jw;
    }

    return true;
  }

  /**
   * @brief Evaluate the 3D NURBS shape functions and their 1st derivatives based on the
   * discretization type of the element.
   */
  template <class VF, class UV, class MD, class WG>
  bool nurbs_get_3d_funct_deriv(VF& nurbs_shape_funct, MD& nurbs_shape_deriv, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (distype)
    {
      case Core::FE::CellType::nurbs8:
      {
        return nurbs_get_3d_funct_deriv<1>(
            nurbs_shape_funct, nurbs_shape_deriv, uv, knots, weights);
      }
      case Core::FE::CellType::nurbs27:
      {
        return nurbs_get_3d_funct_deriv<2>(
            nurbs_shape_funct, nurbs_shape_deriv, uv, knots, weights);
      }
      default:
      {
        FOUR_C_THROW("Unknown distype for nurbs element evaluation\n");
      }
    }
  }


  /*!

  \brief Evaluate 3d basis functions, first and second derivatives
         (with respect to the coordinates of the reference
         element) of nurbs basis functions.

  \verbatim

         reference element          element in knotspan

                                                nu ^
                                                   |
                  w                                |
                  |                                |
                  |                                |
                +-|--+----+                        +---------+
               /  | /    /|                       /         /|
              /   |/    / |                      /         / |
             +----+----+  +        Psi          /         /  |
            /    / 1  /| /|      ------->      /         /   |
           /    /    / |/ |                   /         /    |
          +----+----+ 1+-------- v           +---------+     +-------->
          |    |    | /| /                   |         |    /          xi
          |    |1   |/ |/                    |         |   /
          +----+----+  +                     |         |  /
          |   /|    | /                      |         | /
          |  / |    |/                       |         |/
          +-/--+----+                        +---------+
           /                                /|         |
        u /     \                          / |         |
                 \                        /  |         |
                  \                      /   |         |
                   \                eta v    |         |
                    \                        |         |
                     \                       |         |
                      \                      |         |
                       \                     |         |
                        \       (knot[0])(degree)   (knot[0])(degree+1)

                          \                       |
                           \                      |
                            \                     |
                             \                    |
                              \                   |
                               \                  |
                                \                 |
                                 \                |      +----
                                  \               |
                       Phi o Psi   \              | Phi =  +  N_{i,j,k}(xi,eta,nu)*f_{i,j,k}
                  ^                 \             |       /
                  N_{i,j,k}(u,v,w)   \            |      +----
                                      \           |
                                       \          |
                                        \         |
                                         \        |
                                          \       |

                                               _____
                                            +--     --+
                                           /         /
                                          /         /   |
                                         /         /    |
                                        /         /     |
                                       / _____   /     /
                                      +--     --+     +
                                       \       /     /
                                        |     |     /
                                        |     |    /
                                        |     |   /
                                       /       \ /
                                      +---------+
                               'real' geometry element

  \endverbatim

  PART   I: EVALUATION OF  BEZIER SHAPE FUNCTIONS

  \verbatim



             B_{i,j,k}(xi,eta,nu) = N_i(xi) *M_j (eta)*L_k (nu)


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i'(xi)*M_j (eta)*L_k (nu)
                dxi


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i (xi)*M_j'(eta)*L_k (nu)
               deta


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i (xi)*M_j (eta)*L_k'(nu)
               dnu


            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i"(xi)*M_j (eta)*L_k (nu)
             dxi dxi

            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i (xi)*M_j"(eta)*L_k (nu)
            deta deta

            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i (xi)*M_j (eta)*L_k"(nu)
             dnu dnu

            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i'(xi)*M_j'(eta)*L_k (nu)
             dxi deta

            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i'(xi)*M_j (eta)*L_k'(nu)
             dxi dnu

            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i (xi)*M_j'(eta)*L_k'(nu)
             deta dnu


  \endverbatim

  PART  II: PROJECTING TO NURBS SHAPE FUNCTIONS

  \verbatim


                          w_{i,j,k} * B_{i,j,k}(xi,eta,nu)

         N_{i,j,k} = ---------------------------------------
                     +----+

                       +    w_{s,t,r} * B_{s,t,r}(xi,eta,nu)
                      /
                     +----+
                     s,t,r

  \endverbatim

  First and second derivatives are evaluated using
  the chain rule. The second derivatives are already
  quite painful to compute...


  PART III: TRANSFORMING DERIVATIVES FROM xi/eta TO u/v

  \verbatim

                +   dxi   dxi   dxi -+   +-  du           -+
                |   ---   ---   ---  |   |  ---   0    0   |
                |    du    dv    dw  |   |  2.0            |
                |                    |   |                 |
                |  deta  deta  deta  |   |        dv       |
         Psi' = |  ----  ----  ----  | = |   0   ---   0   |
                |   du    dv    dw   |   |       2.0       |
                |                    |   |                 |
                |   dnu   dnu   dnu  |   |             dw  |
                |  ----  ----  ----  |   |   0    0   ---  |
                +-  du    dv    dw  -+   +-           2.0 -+

  \endverbatim
  \verbatim

     bezier_shape_deriv2(.,...):

     +-------------+---    ......   ---+-------------------+
     |  2          |                   |  2                |
     | d B_{0,0,0} |                   | d B_{N-1,M-1,L-1} |
     | ----------- |                   | ----------------- |
     |    dxi dxi  |                   |       dxi dxi     |
     |             |                   |                   |
     +-------------+---    ......   ---+-------------------+
     |             |                   |                   |
     |  2          |                   |  2                |
     | d B_{0,0,0} |                   | d B_{N-1,M-1,L-1} |
     | ----------- |                   | ----------------- |
     |  deta deta  |                   |      deta deta    |
     |             |                   |                   |
     +-------------+---    ......   ---+-------------------+
     |             |                   |                   |
     |  2          |                   |  2                |
     | d B_{0,0,0} |                   | d B_{N-1,M-1,L-1} |
     | ----------- |                   | ----------------- |
     |   dnu dnu   |                   |       dnu dnu     |
     |             |                   |                   |
     +-------------+---   ......    ---+-------------------+
     |             |                   |                   |
     |  2          |                   |  2                |
     | d B_{0,0,0} |                   | d B_{N-1,M-1,L-1} |
     | ----------- |                   | ----------------- |
     |   dxi deta  |                   |      dxi deta     |
     |             |                   |                   |
     +-------------+---   ......    ---+-------------------+
     |             |                   |                   |
     |  2          |                   |  2                |
     | d B_{0,0,0} |                   | d B_{N-1,M-1,L-1} |
     | ----------- |                   | ----------------- |
     |   dxi dnu   |                   |       dxi dnu     |
     |             |                   |                   |
     +-------------+---   ......    ---+-------------------+
     |             |                   |                   |
     |  2          |                   |  2                |
     | d B_{0,0,0} |                   | d B_{N-1,M-1,L-1} |
     | ----------- |                   | ----------------- |
     |   deta dnu  |                   |      deta dnu     |
     +-------------+---   .......   ---+-------------------+


  \endverbatim


  \param nurbs_shape_funct  (o)
  \param nurbs_shape_deriv  (o)
  \param nurbs_shape_deriv2 (o)
  \param uv                 (i)
  \param knots              (i)
  \param weights            (i)

  \return TRUE, if successful

  */

  template <int degree, class VF, class MD, class MSD, class UV, class WG>
  bool nurbs_get_3d_funct_deriv_deriv2(VF& nurbs_shape_funct, MD& nurbs_shape_deriv,
      MSD& nurbs_shape_deriv2, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights)
  {
    const int degreep = degree + 1;

    // size is the number of control points/basis
    // functions of this element
    const int size = degreep * degreep * degreep;

    // Gausspoint is in [-1:1]x[-1:1]x[-1:1]
    //
    // has to be mapped into knotvector evaluation interval

    /*
         reference element                element in knotspan

                                                nu ^
                                                   |
                  w                                |
                  |                                |
                  |                                |
                +-|--+----+                        +---------+
               /  | /    /|                       /         /|
              /   |/    / |                      /         / |
             +----+----+  +        Psi          /         /  |
            /    / 1  /| /|      ------->      /         /   |
           /    /    / |/ |                   /         /    |
          +----+----+ 1+-------- v           +---------+     +-------->
          |    |    | /| /                   |         |    /          xi
          |    |1   |/ |/                    |         |   /
          +----+----+  +                     |         |  /
          |   /|    | /                      |         | /
          |  / |    |/                       |         |/
          +-/--+----+                        +---------+
           /                                /|         |
        u /     \                          / |         |
                 \                        /  |         |
                  \                      /   |         |
                   \                eta v    |         |
                    \                        |         |
                     \                       |         |
                      \                      |         |
                       \                     |         |
                        \       (knot[0])(degree)   (knot[0])(degree+1)

                          \                       |
                           \                      |
                            \                     |
                             \                    |
                              \                   |
                               \                  |
                                \                 |
                                 \                |      +----
                                  \               |
                       Phi o Psi   \              | Phi =  +  N_{i,j,k}(xi,eta,nu)*f_{i,j,k}
                  ^                 \             |       /
                  N_{i,j,k}(u,v,w)   \            |      +----
                                      \           |
                                       \          |
                                        \         |
                                         \        |
                                          \       |

                                               _____
                                            +--     --+
                                           /         /
                                          /         /   |
                                         /         /    |
                                        /         /     |
                                       / _____   /     /
                                      +--     --+     +
                                       \       /     /
                                        |     |     /
                                        |     |    /
                                        |     |   /
                                       /       \ /
                                      +---------+

                               'real' geometry element
    */

    // the jacobian matrix of Psi is of the very
    // simple form
    //
    //
    //                                        Ju     Jv
    //                                        |       |
    //                                        |       |
    //           +   dxi   dxi   dxi -+   +-  du      |    -+
    //           |   ---   ---   ---  |   |  ---   0  | 0   |
    //           |    du    dv    dw  |   |  2.0      |     |
    //           |                    |   |          /      |
    //           |  deta  deta  deta  |   |        dv       |
    //    Psi' = |  ----  ----  ----  | = |   0   ---   0   |
    //           |   du    dv    dw   |   |       2.0       |
    //           |                    |   |                 |
    //           |   dnu   dnu   dnu  |   |             dw  |
    //           |  ----  ----  ----  |   |   0    0   ---  |
    //           +-  du    dv    dw  -+   +-           2.0 -+
    //                                                  |
    //                                                  |
    //                                                  Jw

    double du = (knots[0])(degreep) - (knots[0])(degree);
    double dv = (knots[1])(degreep) - (knots[1])(degree);
    double dw = (knots[2])(degreep) - (knots[2])(degree);

    double Ju = du / 2.0;
    double Jv = dv / 2.0;
    double Jw = dw / 2.0;

    // get mapped coordinates

    double xi = (knots[0])(degree) + ((uv(0) + 1)) * Ju;
    double eta = (knots[1])(degree) + ((uv(1) + 1)) * Jv;
    double nu = (knots[2])(degree) + ((uv(2) + 1)) * Jw;

    // -----------------------------------------------
    //  PART I: EVALUATION OF  BEZIER SHAPE FUNCTIONS
    // -----------------------------------------------

    Core::LinAlg::SerialDenseVector bezier_shape_funct(size);
    Core::LinAlg::SerialDenseMatrix bezier_shape_deriv(3, size);
    Core::LinAlg::SerialDenseMatrix bezier_shape_deriv2(6, size);

    // allocate bspline polynomials for both direction
    Core::FE::Nurbs::BsplinePolynomial bspline_xi(degree, knots[0]);
    Core::FE::Nurbs::BsplinePolynomial bspline_eta(degree, knots[1]);
    Core::FE::Nurbs::BsplinePolynomial bspline_nu(degree, knots[2]);

    // get temporary doubles for derivatives and
    // values of the above declared polynomials
    double bspline_xi_value;
    double bspline_eta_value;
    double bspline_nu_value;

    double bspline_xi_derivative;
    double bspline_eta_derivative;
    double bspline_nu_derivative;

    double bspline_xi_deriv2;
    double bspline_eta_deriv2;
    double bspline_nu_deriv2;

    // define temporary int variable to compute the
    // number of the basis function from i,j,k
    int id;

    /*
         Bezier basis function
                  |
                  |
             B_{i,j,k}(xi,eta,nu) = N_i(xi) *M_j (eta)*L_k (nu)
                                    |        |         |
                                    |        |         |
                             bspline_u_value |         |
                                             |         |
                                      bspline_v_value  |
                                                       |
                                                 bspline_w_value


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i'(xi)*M_j (eta)*L_k (nu)
                dxi


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i (xi)*M_j'(eta)*L_k (nu)
               deta


            dB_{i,j,k}
            ----------(xi,eta,nu) = N_i (xi)*M_j (eta)*L_k'(nu)
               dnu


            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i"(xi)*M_j (eta)*L_k (nu)
              dxi dxi

            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i (xi)*M_j"(eta)*L_k (nu)
            deta deta

            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i (xi)*M_j (eta)*L_k"(nu)
             dnu dnu

            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i'(xi)*M_j'(eta)*L_k (nu)
             dxi deta

            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i'(xi)*M_j (eta)*L_k'(nu)
             dxi dnu

            2
           d B_{i,j,k}
           -----------(xi,eta,nu) = N_i (xi)*M_j'(eta)*L_k'(nu)
             deta dnu

    */

    // loop all basis functions (corresponding to
    // control points)


    for (int nn = 0; nn < degreep; ++nn)
    {
      // in third direction:

      // get bsplinevalue and derivative
      bspline_nu.evaluate_bspline_first_and_second_deriv(
          bspline_nu_value, bspline_nu_derivative, bspline_nu_deriv2, nu, nn);

      for (int mm = 0; mm < degreep; ++mm)
      {
        // in second direction:

        // get bsplinevalue and derivative
        bspline_eta.evaluate_bspline_first_and_second_deriv(
            bspline_eta_value, bspline_eta_derivative, bspline_eta_deriv2, eta, mm);

        const double M_eta_L_nu = bspline_eta_value * bspline_nu_value;
        const double M_eta_dL_nu = bspline_eta_value * bspline_nu_derivative;
        const double dM_eta_L_nu = bspline_eta_derivative * bspline_nu_value;
        const double dM_eta_dL_nu = bspline_eta_derivative * bspline_nu_derivative;
        const double M_eta_ddL_nu = bspline_eta_value * bspline_nu_deriv2;
        const double ddM_eta_L_nu = bspline_eta_deriv2 * bspline_nu_value;

        const int layerid = (degreep) * (mm + nn * (degreep));

        for (int rr = 0; rr < degreep; ++rr)
        {
          // in first direction:

          // get bsplinevalue and derivative
          bspline_xi.evaluate_bspline_first_and_second_deriv(
              bspline_xi_value, bspline_xi_derivative, bspline_xi_deriv2, xi, rr);


          // get the number of the basis function
          id = rr + layerid;

          // set value to bezier_shape_funct
          bezier_shape_funct(id) = bspline_xi_value * M_eta_L_nu;

          // set values to bezier_shape_deriv
          bezier_shape_deriv(0, id) = bspline_xi_derivative * M_eta_L_nu;
          bezier_shape_deriv(1, id) = bspline_xi_value * dM_eta_L_nu;
          bezier_shape_deriv(2, id) = bspline_xi_value * M_eta_dL_nu;

          // set values to bezier_shape_deriv2
          bezier_shape_deriv2(0, id) = bspline_xi_deriv2 * M_eta_L_nu;
          bezier_shape_deriv2(1, id) = bspline_xi_value * ddM_eta_L_nu;
          bezier_shape_deriv2(2, id) = bspline_xi_value * M_eta_ddL_nu;
          bezier_shape_deriv2(3, id) = bspline_xi_derivative * dM_eta_L_nu;
          bezier_shape_deriv2(4, id) = bspline_xi_derivative * M_eta_dL_nu;
          bezier_shape_deriv2(5, id) = bspline_xi_value * dM_eta_dL_nu;
        }
      }
    }

    // -----------------------------------------------
    //  PART II: PROJECTING TO NURBS SHAPE FUNCTIONS
    // -----------------------------------------------

    // alloc temporary doubles, initialise to zero
    double sum_funct_weight;
    double sum_deriv_weight[3];
    double sum_deriv2_weight[6];

    sum_funct_weight = 0.0;
    sum_deriv_weight[0] = 0.0;
    sum_deriv_weight[1] = 0.0;
    sum_deriv_weight[2] = 0.0;
    sum_deriv2_weight[0] = 0.0;
    sum_deriv2_weight[1] = 0.0;
    sum_deriv2_weight[2] = 0.0;
    sum_deriv2_weight[3] = 0.0;
    sum_deriv2_weight[4] = 0.0;
    sum_deriv2_weight[5] = 0.0;

    /*loop all control points, compute sums

                           +----+

         sum_funct_weight =  +    w_{i,j,k} * B_{i,j,k}(xi,eta,nu)
                            /
                           +----+


                              +----+
                               \                 dB_{i,j,k}
         sum_deriv_weight[0] =  +    w_{i,j,k} * ----------(xi,eta,nu)
                               /                     dxi
                              +----+

                              +----+
                               \                 dB_{i,j,k}
         sum_deriv_weight[1] =  +    w_{i,j,k} * ----------(xi,eta,nu)
                               /                     deta
                              +----+

                              +----+
                               \                 dB_{i,j,k}
         sum_deriv_weight[2] =  +    w_{i,j,k} * ----------(xi,eta,nu)
                               /                     dnu
                              +----+



                               +----+              2
                                \                 d B_{i,j,k}
         sum_deriv2_weight[0] =  +    w_{i,j,k} * -----------(xi,eta,nu)
                                /                        2
                               +----+                 dxi

                               +----+              2
                                \                 d B_{i,j,k}
         sum_deriv2_weight[1] =  +    w_{i,j,k} * -----------(xi,eta,nu)
                                /                        2
                               +----+                deta

                               +----+              2
                                \                 d B_{i,j,k}
         sum_deriv2_weight[2] =  +    w_{i,j,k} * -----------(xi,eta,nu)
                                /                       2
                               +----+                dnu


                               +----+              2
                                \                 d B_{i,j,k}
         sum_deriv2_weight[3] =  +    w_{i,j,k} * -----------(xi,eta,nu)
                                /                  dxi deta
                               +----+

                               +----+              2
                                \                 d B_{i,j,k}
         sum_deriv2_weight[4] =  +    w_{i,j,k} * -----------(xi,eta,nu)
                                /                   dxi dnu
                               +----+

                               +----+              2
                                \                 d B_{i,j,k}
         sum_deriv2_weight[5] =  +    w_{i,j,k} * -----------(xi,eta,nu)
                                /                   deta dnu
                               +----+
    */

    for (int rr = 0; rr < size; ++rr)
    {
      sum_funct_weight += weights(rr) * bezier_shape_funct(rr);

      for (int mm = 0; mm < 3; ++mm)
      {
        sum_deriv_weight[mm] += weights(rr) * bezier_shape_deriv(mm, rr);
      }


      for (int mm = 0; mm < 6; ++mm)
      {
        sum_deriv2_weight[mm] += weights(rr) * bezier_shape_deriv2(mm, rr);
      }
    }

    if (sum_funct_weight == 0.0) return false;

    const double fact = 2.0 / sum_funct_weight;

    // loop all basis functions
    for (int rr = 0; rr < size; ++rr)
    {
      const double normed_weight = weights(rr) / sum_funct_weight;

      /* Compute Nurbs basis functions


                        w_{i,j,k} * B_{i,j,k}(xi,eta,nu)

         N_{i,j,k} = ---------------------------------------
                     +----+

                       +    w_{r,s,t} * B_{r,s,t}(xi,eta,nu)
                      /
                     +----+
                      r,s,t
      */
      nurbs_shape_funct(rr) = bezier_shape_funct(rr) * normed_weight;

      /* Nurbs derivatives are defined by the chain rule

                                         +- +----+                  -+                       +-
       +----+                     -+ |   \                       |                       |
       dB_{r,s,t} | dB_{i,j,k} |    +  w_{r,s,t}*B_{r,s,t} |                       |    +
       w_{r,s,t}*---------- | w_{i,j,k}*----------*|   /                       | -
       w_{i,j,k}*B_{i,j,k}*|   /                   dxi    | dxi    |  +----+                   |
       |  +----+                      | dN_{i,j,k}                        +-  r,s,t -+ +-  r,s,t
       -+
       ---------- =
       ---------------------------------------------------------------------------------------------------------
           dxi                           +- +----+                                 -+ 2
                                         |   \                                      |
                                         |    +    w_{r,s,t} * B_{r,s,t}(xi,eta,nu) |
                                         |   /                                      |
                                         |  +----+                                  |
                                         +-  r,s,t                                 -+
       */
      // loop directions to compute derivatives
      for (int mm = 0; mm < 3; ++mm)
      {
        nurbs_shape_deriv(mm, rr) = bezier_shape_deriv(mm, rr) - bezier_shape_funct(rr) *
                                                                     sum_deriv_weight[mm] /
                                                                     sum_funct_weight;

        nurbs_shape_deriv(mm, rr) *= normed_weight;
      }

      /* Nurbs second derivatives are calculated by a
         second application of the chain rule

                                                                    +-
                                                                    |
                                                                    |
                                                                    |
                                       w_{i,j,k}                    |   2
       d N_{i,j,k}                                                  |  d B_{i,j,k}
       ----------- = -------------------------------------------- * |  ----------- -
        dxi  dxi     +- +----+                                 -+   |   dxi  dxi
           n    m    |   \                                      |   |      n    m
                     |    +    w_{r,s,t} * B_{r,s,t}(xi,eta,nu) |   |
                     |   /                                      |   |
                     |  +----+                                  |   |
                     +-  r,s,t                                 -+   +-

                       +- +----+                                  -+
            dB_{i,j,k} |   \                 dB_{r,s,t}            |
            ---------- |    +    w_{r,s,t} * ----------(xi,eta,nu) |
               dxi     |   /                    dxi                |
                  n    |  +----+                   m               |
                       +-  r,s,t                                  -+
          - -------------------------------------------------------- -
                 +- +----+                                 -+
                 |   \                                      |
                 |    +    w_{r,s,t} * B_{r,s,t}(xi,eta,nu) |
                 |   /                                      |
                 |  +----+                                  |
                 +-  r,s,t                                 -+



                       +- +----+                                  -+
            dB_{i,j,k} |   \                 dB_{r,s,t}            |
            ---------- |    +    w_{r,s,t} * ----------(xi,eta,nu) |
               dxi     |   /                    dxi                |
                  m    |  +----+                   n               |
                       +-  r,s,t                                  -+
          - -------------------------------------------------------- -
                 +- +----+                                 -+
                 |   \                                      |
                 |    +    w_{r,s,t} * B_{r,s,t}(xi,eta,nu) |
                 |   /                                      |
                 |  +----+                                  |
                 +-  r,s,t                                 -+


                         +- +----+              2                    -+
                         |   \                 d B_{r,s,t}            |
                         |    +    w_{r,s,t} * -----------(xi,eta,nu) |
             B_{i,j,k} * |   /                  dxi  dxi              |
                         |  +----+                 n    m             |
                         +-  r,s,t                                   -+
          - ----------------------------------------------------------- +
                   +- +----+                                 -+
                   |   \                                      |
                   |    +    w_{r,s,t} * B_{r,s,t}(xi,eta,nu) |
                   |   /                                      |
                   |  +----+                                  |
                   +-  r,s,t                                 -+


                             +- +----+                                  -+   +- +----+ -+  -+ |
       \                 dB_{r,s,t}            |   |   \                 dB_{r,s,t}            |
       | |    +    w_{r,s,t} * ----------(xi,eta,nu) |   |    +    w_{r,s,t} *
       ----------(xi,eta,nu) |   | 2 * B_{i,j,k} * |   /                    dxi                |
       * |   /                    dxi                |   | |  +----+                   n |   |
       +----+                   m               |   |
                             +-  r,s,t                                  -+   +-  r,s,t -+   |
          +
       --------------------------------------------------------------------------------------------------------------
       |
                                         +- +----+                                 -+ 2 | |
       |                                        | |    +    w_{r,s,t} * B_{r,s,t}(xi,eta,nu) | |
                                         |   /                                      | | | +----+
       |                                        |
                                         +-  r,s,t                                 -+ -+
      */

      const double fact_bezier_shape_funct_rr = fact * bezier_shape_funct(rr);

      // we apply a Horner-type scheme for the
      // multiplication with sum_funct_weight
      nurbs_shape_deriv2(0, rr) =
          fact_bezier_shape_funct_rr * sum_deriv_weight[0] * sum_deriv_weight[0];
      nurbs_shape_deriv2(0, rr) -= bezier_shape_funct(rr) * sum_deriv2_weight[0];
      nurbs_shape_deriv2(0, rr) -= 2 * bezier_shape_deriv(0, rr) * sum_deriv_weight[0];
      nurbs_shape_deriv2(0, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(0, rr) += bezier_shape_deriv2(0, rr);
      nurbs_shape_deriv2(0, rr) *= normed_weight;

      nurbs_shape_deriv2(1, rr) =
          fact_bezier_shape_funct_rr * sum_deriv_weight[1] * sum_deriv_weight[1];
      nurbs_shape_deriv2(1, rr) -= bezier_shape_funct(rr) * sum_deriv2_weight[1];
      nurbs_shape_deriv2(1, rr) -= 2 * bezier_shape_deriv(1, rr) * sum_deriv_weight[1];
      nurbs_shape_deriv2(1, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(1, rr) += bezier_shape_deriv2(1, rr);
      nurbs_shape_deriv2(1, rr) *= normed_weight;

      nurbs_shape_deriv2(2, rr) =
          fact_bezier_shape_funct_rr * sum_deriv_weight[2] * sum_deriv_weight[2];
      nurbs_shape_deriv2(2, rr) -= bezier_shape_funct(rr) * sum_deriv2_weight[2];
      nurbs_shape_deriv2(2, rr) -= 2 * bezier_shape_deriv(2, rr) * sum_deriv_weight[2];
      nurbs_shape_deriv2(2, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(2, rr) += bezier_shape_deriv2(2, rr);
      nurbs_shape_deriv2(2, rr) *= normed_weight;

      nurbs_shape_deriv2(3, rr) =
          fact_bezier_shape_funct_rr * sum_deriv_weight[0] * sum_deriv_weight[1];
      nurbs_shape_deriv2(3, rr) -= bezier_shape_funct(rr) * sum_deriv2_weight[3];
      nurbs_shape_deriv2(3, rr) -= bezier_shape_deriv(0, rr) * sum_deriv_weight[1];
      nurbs_shape_deriv2(3, rr) -= bezier_shape_deriv(1, rr) * sum_deriv_weight[0];
      nurbs_shape_deriv2(3, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(3, rr) += bezier_shape_deriv2(3, rr);
      nurbs_shape_deriv2(3, rr) *= normed_weight;

      nurbs_shape_deriv2(4, rr) =
          fact_bezier_shape_funct_rr * sum_deriv_weight[0] * sum_deriv_weight[2];
      nurbs_shape_deriv2(4, rr) -= bezier_shape_funct(rr) * sum_deriv2_weight[4];
      nurbs_shape_deriv2(4, rr) -= bezier_shape_deriv(0, rr) * sum_deriv_weight[2];
      nurbs_shape_deriv2(4, rr) -= bezier_shape_deriv(2, rr) * sum_deriv_weight[0];
      nurbs_shape_deriv2(4, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(4, rr) += bezier_shape_deriv2(4, rr);
      nurbs_shape_deriv2(4, rr) *= normed_weight;

      nurbs_shape_deriv2(5, rr) =
          fact_bezier_shape_funct_rr * sum_deriv_weight[1] * sum_deriv_weight[2];
      nurbs_shape_deriv2(5, rr) -= bezier_shape_funct(rr) * sum_deriv2_weight[5];
      nurbs_shape_deriv2(5, rr) -= bezier_shape_deriv(1, rr) * sum_deriv_weight[2];
      nurbs_shape_deriv2(5, rr) -= bezier_shape_deriv(2, rr) * sum_deriv_weight[1];
      nurbs_shape_deriv2(5, rr) /= sum_funct_weight;
      nurbs_shape_deriv2(5, rr) += bezier_shape_deriv2(5, rr);
      nurbs_shape_deriv2(5, rr) *= normed_weight;
    }


    // -----------------------------------------------
    //  PART III: TRANSFORMING DERIVATIVES FROM xi/eta
    //            TO u/v
    // -----------------------------------------------

    // we already know the jacobian matrix of psi
    //
    //            +   dxi   dxi   dxi -+   +-  du           -+
    //            |   ---   ---   ---  |   |  ---   0    0   |
    //            |    du    dv    dw  |   |  2.0            |
    //            |                    |   |                 |
    //            |  deta  deta  deta  |   |        dv       |
    //     Psi' = |  ----  ----  ----  | = |   0   ---   0   |
    //            |   du    dv    dw   |   |       2.0       |
    //            |                    |   |                 |
    //            |   dnu   dnu   dnu  |   |             dw  |
    //            |  ----  ----  ----  |   |   0    0   ---  |
    //            +-  du    dv    dw  -+   +-           2.0 -+
    //
    //
    // we will obtain the derivatives with respect to
    // u,v,w just by multiplication

    const double JuJu = Ju * Ju;
    const double JvJv = Jv * Jv;
    const double JwJw = Jw * Jw;
    const double JuJv = Ju * Jv;
    const double JuJw = Ju * Jw;
    const double JvJw = Jv * Jw;


    // loop all basis function derivatives
    for (int rr = 0; rr < size; ++rr)
    {
      nurbs_shape_deriv(0, rr) *= Ju;
      nurbs_shape_deriv(1, rr) *= Jv;
      nurbs_shape_deriv(2, rr) *= Jw;

      nurbs_shape_deriv2(0, rr) *= JuJu;
      nurbs_shape_deriv2(1, rr) *= JvJv;
      nurbs_shape_deriv2(2, rr) *= JwJw;
      nurbs_shape_deriv2(3, rr) *= JuJv;
      nurbs_shape_deriv2(4, rr) *= JuJw;
      nurbs_shape_deriv2(5, rr) *= JvJw;
    }

    return true;
  }

  /**
   * @brief Evaluate the 3D NURBS shape functions and their 2nd derivatives based on the
   * discretization type of the element.
   */
  template <class VF, class MD, class MSD, class UV, class WG>
  bool nurbs_get_3d_funct_deriv_deriv2(VF& nurbs_shape_funct, MD& nurbs_shape_deriv,
      MSD& nurbs_shape_deriv2, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (distype)
    {
      case Core::FE::CellType::nurbs8:
      {
        return nurbs_get_3d_funct_deriv_deriv2<1>(
            nurbs_shape_funct, nurbs_shape_deriv, nurbs_shape_deriv2, uv, knots, weights);
      }
      case Core::FE::CellType::nurbs27:
      {
        return nurbs_get_3d_funct_deriv_deriv2<2>(
            nurbs_shape_funct, nurbs_shape_deriv, nurbs_shape_deriv2, uv, knots, weights);
      }
      default:
      {
        FOUR_C_THROW("Unknown distype for nurbs element evaluation\n");
      }
    }
  }


  namespace Internal
  {
    /*!
     * \brief Number of parameter coordinates offered by @p T that is known at compile time.
     *
     * Returns std::nullopt for containers whose number of entries is not part of the type, e.g.
     * Core::LinAlg::Matrix or Core::LinAlg::SerialDenseVector. For those, the number of parameter
     * coordinates can only be checked at runtime.
     */
    template <typename T>
    consteval std::optional<std::size_t> static_num_coords()
    {
      if constexpr (Core::LinAlg::is_tensor<T>)
        return std::remove_cvref_t<T>::template extent<0>();
      else
        return std::nullopt;
    }

    /*!
     * \brief Whether the parameter coordinate container @p T can hold @p dim coordinates.
     *
     * Only instantiates the dimension-specific shape function kernels for dimensions that the
     * container can actually hold. Since Core::FE::get_dimension() is evaluated at runtime, all
     * branches of a switch on it are instantiated. Without this guard, kernels for the wrong
     * dimension are instantiated for containers of insufficient size, which reads out of bounds.
     */
    template <std::size_t dim, typename T>
    consteval bool has_enough_coords()
    {
      constexpr std::optional<std::size_t> num_coords = static_num_coords<T>();
      return !num_coords.has_value() || *num_coords >= dim;
    }
  }  // namespace Internal

  //! Evaluate basis functions, first and second derivatives of nurbs basis functions.
  template <class VF, class MD, class MSD, class UV, class WG>
  bool nurbs_get_funct_deriv_deriv2(VF& nurbs_shape_funct, MD& nurbs_shape_deriv,
      MSD& nurbs_shape_deriv2, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (Core::FE::get_dimension(distype))
    {
      case 3:
      {
        if constexpr (Internal::has_enough_coords<3, UV>())
        {
          return Core::FE::Nurbs::nurbs_get_3d_funct_deriv_deriv2(nurbs_shape_funct,
              nurbs_shape_deriv, nurbs_shape_deriv2, uv, knots, weights, distype);
        }
        else
        {
          FOUR_C_THROW("The parameter coordinate container is too small for a 3D nurbs element.");
        }
      }
      case 2:
      {
        if constexpr (Internal::has_enough_coords<2, UV>())
        {
          return Core::FE::Nurbs::nurbs_get_2d_funct_deriv_deriv2(nurbs_shape_funct,
              nurbs_shape_deriv, nurbs_shape_deriv2, uv, knots, weights, distype);
        }
        else
        {
          FOUR_C_THROW("The parameter coordinate container is too small for a 2D nurbs element.");
        }
      }
      case 1:
      {
        if constexpr (Internal::has_enough_coords<1, UV>())
        {
          return Core::FE::Nurbs::nurbs_get_1d_funct_deriv_deriv2(nurbs_shape_funct,
              nurbs_shape_deriv, nurbs_shape_deriv2, uv(0), knots[0], weights, distype);
        }
        else
        {
          FOUR_C_THROW(
              "The parameter coordinate container holds no coordinate for a 1D nurbs element.");
        }
      }
      default:
        FOUR_C_THROW("dimension of the element is not correct");
    }
  }

  //! Evaluate basis functions and first derivatives of nurbs basis functions.
  template <class VF, class MD, class UV, class WG>
  bool nurbs_get_funct_deriv(VF& nurbs_shape_funct, MD& nurbs_shape_deriv, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (Core::FE::get_dimension(distype))
    {
      case 3:
      {
        if constexpr (Internal::has_enough_coords<3, UV>())
        {
          return Core::FE::Nurbs::nurbs_get_3d_funct_deriv(
              nurbs_shape_funct, nurbs_shape_deriv, uv, knots, weights, distype);
        }
        else
        {
          FOUR_C_THROW("The parameter coordinate container is too small for a 3D nurbs element.");
        }
      }
      case 2:
      {
        if constexpr (Internal::has_enough_coords<2, UV>())
        {
          return Core::FE::Nurbs::nurbs_get_2d_funct_deriv(
              nurbs_shape_funct, nurbs_shape_deriv, uv, knots, weights, distype);
        }
        else
        {
          FOUR_C_THROW("The parameter coordinate container is too small for a 2D nurbs element.");
        }
      }
      case 1:
      {
        if constexpr (Internal::has_enough_coords<1, UV>())
        {
          return Core::FE::Nurbs::nurbs_get_1d_funct_deriv(
              nurbs_shape_funct, nurbs_shape_deriv, uv(0), knots[0], weights, distype);
        }
        else
        {
          FOUR_C_THROW(
              "The parameter coordinate container holds no coordinate for a 1D nurbs element.");
        }
      }
      default:
        FOUR_C_THROW("dimension of the element is not correct");
    }
  }

  //! Evaluate basis functions of nurbs basis functions.
  template <class VF, class UV, class WG>
  bool nurbs_shape_function_dim(VF& nurbs_shape_funct, const UV& uv,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots, const WG& weights,
      const Core::FE::CellType& distype)
  {
    switch (Core::FE::get_dimension(distype))
    {
      case 3:
      {
        if constexpr (Internal::has_enough_coords<3, UV>())
        {
          return Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, distype);
        }
        else
        {
          FOUR_C_THROW("The parameter coordinate container is too small for a 3D nurbs element.");
        }
      }
      case 2:
      {
        if constexpr (Internal::has_enough_coords<2, UV>())
        {
          return Core::FE::Nurbs::nurbs_get_2d_funct(
              nurbs_shape_funct, uv, knots, weights, distype);
        }
        else
        {
          FOUR_C_THROW("The parameter coordinate container is too small for a 2D nurbs element.");
        }
      }
      case 1:
      {
        if constexpr (Internal::has_enough_coords<1, UV>())
        {
          return Core::FE::Nurbs::nurbs_get_1d_funct(
              nurbs_shape_funct, uv(0), knots[0], weights, distype);
        }
        else
        {
          FOUR_C_THROW(
              "The parameter coordinate container holds no coordinate for a 1D nurbs element.");
        }
      }
      default:
        FOUR_C_THROW("dimension of the element is not correct");
    }
  }

}  // namespace Core::FE::Nurbs

FOUR_C_NAMESPACE_CLOSE

#endif
