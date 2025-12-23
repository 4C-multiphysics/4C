// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_KRYLOV_PROJECTOR_HPP
#define FOUR_C_LINALG_KRYLOV_PROJECTOR_HPP

#include "4C_config.hpp"

#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linear_solver_method_projector.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SerialDenseMatrix;
  class SerialDenseVector;
  class SparseMatrix;
  class Map;

  /*!
  A class providing a Krylov projectors. Used for projected preconditioner,
  projected operator, and directly in direct solver.

  */

  class KrylovProjector : public LinearSystemProjector
  {
   public:
    /*!
    \brief Standard Constructor, sets mode-ids and weighttype. kernel and weight
           vector as well as their inner product matrix are allocated, but still
           have to be set. Use GetNonConstKernel() and GetNonConstWeights() to
           get pointer and set the vectors and call fill_complete() tobe able to
           use projector.
    */
    KrylovProjector(const std::vector<int>
                        modeids,        //! ids of to-be-projected modes according element nullspace
        const std::string* weighttype,  //! type of weights: integration or pointvalues
        const Core::LinAlg::Map* map    //! map for kernel and weight vectors
    );

    //! give out std::shared_ptr to c_ for change
    std::shared_ptr<Core::LinAlg::MultiVector<double>> get_non_const_kernel();

    //! give out std::shared_ptr to w_ for change
    std::shared_ptr<Core::LinAlg::MultiVector<double>> get_non_const_weights();
    // set c_ and w_ from outside
    void set_cw(Core::LinAlg::MultiVector<double>& c0, Core::LinAlg::MultiVector<double>& w0,
        const Core::LinAlg::Map* newmap);
    void set_cw(Core::LinAlg::MultiVector<double>& c0, Core::LinAlg::MultiVector<double>& w0);
    //! compute (w^T c)^(-1) and completes projector for use
    void fill_complete();

    //! give out projector matrix - build it if not yet built (thus not const)
    Core::LinAlg::SparseMatrix get_p();

    //! give out transposed projector matrix - build it if not yet built (thus not const)
    Core::LinAlg::SparseMatrix get_pt();

    //! wrapper for applying projector to vector for iterative solver
    [[nodiscard]] Core::LinAlg::Vector<double> to_full(
        const Core::LinAlg::Vector<double>& Y) const override;

    //! wrapper for applying transpose of projector to vector for iterative solver
    [[nodiscard]] Core::LinAlg::Vector<double> to_reduced(
        const Core::LinAlg::Vector<double>& Y) const override;

    //! give out projection P^T A P
    [[nodiscard]] std::unique_ptr<Core::LinAlg::SparseOperator> to_reduced(
        const Core::LinAlg::SparseOperator& A) const override;

    //! return dimension of nullspace
    int nsdim() const { return nsdim_; }

    //! return mode-ids corresponding to element nullspace
    std::vector<int> modes() const { return modeids_; }

    //! return type of projection weights: integration or pointvalues
    const std::string* weight_type() const { return weighttype_; }

   private:
    //! creates actual projector matrix P (or its transpose) for use in direct solver
    Core::LinAlg::SparseMatrix create_projector(const Core::LinAlg::MultiVector<double>& v1,
        const Core::LinAlg::MultiVector<double>& v2,
        const Core::LinAlg::SerialDenseMatrix& inv_v1Tv2);

    //! applies projector (or its transpose) to vector for iterative solver
    Core::LinAlg::Vector<double> apply_projector(const Core::LinAlg::Vector<double>& Y,
        const Core::LinAlg::MultiVector<double>& v1, const Core::LinAlg::MultiVector<double>& v2,
        const Core::LinAlg::SerialDenseMatrix& inv_v1Tv2) const;

    /*
      (Modified) ApplyInverse call

      This method calls ApplyInverse on the actual preconditioner and, the
      solution is krylovized against a set of weight vectors provided in a
      multivector.

      This is done using a projector P defined by

                                      T
                                     x * w
                          P  x = x - ------ c
                                      T
                                     w * c

      w is the vector of weights, c a vector of ones (in the dofs under
      consideration) corresponding to the matrix kernel.

      The system we are solving with this procedure is not Au=b for u (since A
      might be singular), but we are solving

                          / T \         T
                         | P A | P u = P b ,
                          \   /

      for the projection of the solution Pu, i.e. in the preconditioned case


                                                            -+
             / T   \     /      -1 \          T              |
            | P * A | * |  P * M    | * xi = P  * b          |
             \     /     \         /                         |
                                                  -1         |
                                         x = P * M  * xi     |
                                                            -+


      Hence, P is always associated with the apply inverse call of the
      preconditioner (the right bracket) and always called after the call
      to ApplyInverse.


      Properties of P are:

      1) c defines the kernel of P, i.e. P projects out the matrix kernel

                            T
                           c * w
                P c = c - ------- c = c - c = 0
                            T
                           w * c

      2) The space spanned by P x is krylov to the weight vector

                         /      T      \              T
       T   /   \     T  |      x * w    |    T       x * w     T       T       T
      w * | P x | = w * | x - ------- c | = w * x - ------- * w * c = w * x - w * x = 0
           \   /        |       T       |             T
                         \     w * c   /             w * c


      This modified Apply call is for singular matrices A when c is
      a vector defining A's nullspace. The preceding projection
      operator ensures
                              |           |
                             -+-         -+-T
                    A u = A u     where u    * c =0,

      even if A*c != 0 (for numerical inaccuracies during the computation
      of A)

      See the following article for further reading:

      @article{1055401,
       author = {Bochev,, Pavel and Lehoucq,, R. B.},
       title = {On the Finite Element Solution of the Pure Neumann Problem},
       journal = {SIAM Rev.},
       volume = {47},
       number = {1},
       year = {2005},
       issn = {0036-1445},
       pages = {50--66},
       doi = {http://dx.doi.org/10.1137/S0036144503426074},
       publisher = {Society for Industrial and Applied Mathematics},
       address = {Philadelphia, PA, USA},
       }

    */

    //! flag whether inverse of (w_^T c_) was computed after w_ or c_ have been
    // given out for change with GetNonConstW() or GetNonConstC(). This is not
    // fool proof since w and c can also be changed after having called
    // fill_complete().
    bool complete_;

    //! dimension of nullspace
    int nsdim_;

    const std::vector<int> modeids_;

    const std::string* weighttype_;

    //! projector matrix - only built if necessary (e.g. for direct solvers)
    std::shared_ptr<Core::LinAlg::SparseMatrix> p_;

    //! transposed projector matrix - only built if necessary (e.g. for direct solvers)
    std::shared_ptr<Core::LinAlg::SparseMatrix> pt_;

    //! a set of vectors defining weighted (basis integral) vector for the projector
    std::shared_ptr<Core::LinAlg::MultiVector<double>> w_;

    //! a set of vectors defining the vectors of ones (in the respective components)
    //! for the matrix kernel
    std::shared_ptr<Core::LinAlg::MultiVector<double>> c_;

    //! inverse of product (c_^T * w_), computed once after setting c_ and w_
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> invw_tc_;
  };

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
