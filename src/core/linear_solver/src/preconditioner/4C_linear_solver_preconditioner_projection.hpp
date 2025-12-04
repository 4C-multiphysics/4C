// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_PRECONDITIONER_PROJECTION_HPP
#define FOUR_C_LINEAR_SOLVER_PRECONDITIONER_PROJECTION_HPP

#include "4C_config.hpp"

#include "4C_linalg_krylov_projector.hpp"
#include "4C_linear_solver_method_projector.hpp"
#include "4C_linear_solver_preconditioner_type.hpp"

#include <Epetra_Operator.h>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinearSolver
{
  /*!
   * A preconditioner that applies a linear projection first and then the usual preconditioner.
   */
  class ProjectionPreconditioner : public PreconditionerTypeBase
  {
   public:
    ProjectionPreconditioner(std::shared_ptr<PreconditionerTypeBase> preconditioner,
        std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector);

    void setup(Core::LinAlg::SparseOperator& matrix, Core::LinAlg::MultiVector<double>& b) override;

    /// linear operator used for preconditioning
    std::shared_ptr<Epetra_Operator> prec_operator() const override { return p_; }

    Teuchos::RCP<const Thyra::LinearOpBase<double>> thyra_operator() const override
    {
      return p_thyra_;
    }

   private:
    std::shared_ptr<PreconditionerTypeBase> preconditioner_;

    /// projector object that does the actual work
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector_;

    std::shared_ptr<Epetra_Operator> p_;

    Teuchos::RCP<const Thyra::LinearOpBase<double>> p_thyra_;
  };

  /*!
  A common interface for ifpack, ml and simpler preconditioners.
  This interface allows a modification of the vector returned
  by the ApplyInverse call, which is necessary to do a solution on
  a Krylov space krylovized to certain (for example rigid body)
  modes.

  The linalg preconditioner interface class holds a pointer (rcp)
  to the actual preconditioner. All methods implemented to support
  the Epetra_Operator interface just call the corresponding functions
  of the actual preconditioner.

  Only the ApplyInverse method is modified and performs the
  projection if desired.

  See linalg_projected_operator.H for related docu and code.

  */
  class LinalgPrecondOperator : public Thyra::LinearOpBase<double>
  {
   public:
    LinalgPrecondOperator(Teuchos::RCP<const Thyra::LinearOpBase<double>> precond, bool project,
        std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector);

    /** @brief Range space of this operator */
    virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double>> range() const;

    /** @brief Domain space of this operator */
    virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double>> domain() const;

    virtual bool opSupportedImpl(const Thyra::EOpTransp M_trans) const;

    //! @}

    //! @name Mathematical functions required to support the Epetra_Operator interface (modified)
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
    virtual void applyImpl(const Thyra::EOpTransp M_trans, const Thyra::MultiVectorBase<double>& x,
        const Teuchos::Ptr<Thyra::MultiVectorBase<double>>& y, const double alpha,
        const double beta) const;

   private:
    //! flag whether to do a projection or just pass through
    bool project_;

    //! the actual preconditioner
    Teuchos::RCP<const Thyra::LinearOpBase<double>> precond_;

    //! projector
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector_;
  };
}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif
