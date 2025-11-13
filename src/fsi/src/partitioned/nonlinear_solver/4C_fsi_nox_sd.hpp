// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_NOX_SD_HPP
#define FOUR_C_FSI_NOX_SD_HPP

#include "4C_config.hpp"

#include <NOX_GlobalData.H>
#include <NOX_LineSearch_Generic.H>
#include <NOX_LineSearch_UserDefinedFactory.H>
#include <NOX_Utils.H>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  // Forward declaration
  namespace Nln::Interface
  {
    class RequiredBase;
  }

  namespace FSI
  {
    //! Use NonlinearCG linesearch
    /*!

      This is a simple linesearch intended to be used with
      ::NOX::Direction::NonlinearCG, which provides search direction \f$
      d \f$, in computing an update to the current solution vector \f$
      x_{new} = x_{old} + \lambda d \f$.  It is designed to compute a step
      length \f$ \lambda \f$ consistent with the exact linesearch of %Linear
      CG for linear problems, and it avoids use of matrices by employing a
      directional derivative (details below).  The step length, \f$ \lambda
      \f$ is computed from a single evaluation of,

      \f[
        \lambda = - \frac{R(x_{old})^T d}{d^T J(x_{old})d}
      \f]

      where \f$ J \f$ is the n x n Jacobian matrix.  Explicit construction of
      \f$ J \f$ is avoided by performing the product \f$ Jd \f$ using
      approximated fluid field derivatives (cf FSIMatrixFree):

      \f[
      J(x_{old})d \approx S'(F(x_{old})) F'(x_{old}) d - d
      \f]

      <b> Derivation / Theory: </b>

      This linesearch is derived by attempting to achieve in a single step,
      the following minimization:

      \f[
        \min_\lambda \phi(\lambda)\equiv\phi (x_{old}+ \lambda d)
      \f]

      where \f$ \phi \f$ is a merit function chosen (but never explicitly
      given) so that an equivalence to %Linear CG holds, ie \f$ \nabla\phi(x)
      \leftrightarrow R(x) \f$.  The minimization above can now be cast as
      an equation:

      \f[
        \phi ' (\lambda) = \nabla\phi (x_{old}+ \lambda d)^T d =
        R(x_{old}+ \lambda d)^T d = 0~~.
      \f]

      An approximate solution to this equation can be obtained from a
      second-order expansion of \f[ \phi(\lambda) \f],

      \f[
        \phi(\lambda)\approx\phi (0) + \phi ' (0)\lambda + \phi '' (0)
        \frac{\lambda^2}{2}
      \f]

      from which it immediately follows

      \f[
        \lambda_{min} \approx - \frac{\phi ' (0)}{\phi '' (0)} =
        - \frac{R(x_{old})^T d}{d^T J(x_{old})d}
      \f]


      <b>References</b>

      <ul>

      This linesearch is adapted from ideas presented in Section 14.2 of:

      <li>Jonathan Richard Shewchuk,
      <A HREF="http://www-2.cs.cmu.edu/~jrs/jrspapers.html"/> "An
      Introduction to the Conjugate Gradient Method Without the Agonizing
      Pain</A>," 1994.</li> Though presented within the context of nonlinear
      optimization, the connection to solving nonlinear equation systems is
      made via the equivalence \f$ f'(x) \leftrightarrow R(x) \f$.

    */

    class SDRelaxation : public ::NOX::LineSearch::Generic
    {
     public:
      //! Constructor
      SDRelaxation(const Teuchos::RCP<::NOX::Utils>& utils, Teuchos::ParameterList& params);


      // derived
      bool reset(const Teuchos::RCP<::NOX::GlobalData>& gd, Teuchos::ParameterList& params);

      // derived
      bool compute(::NOX::Abstract::Group& newgrp, double& step, const ::NOX::Abstract::Vector& dir,
          const ::NOX::Solver::Generic& s) override;

     private:
      //! Method for computing directional derivatives numerically
      ::NOX::Abstract::Vector& compute_directional_derivative(
          const ::NOX::Abstract::Vector& dir, NOX::Nln::Interface::RequiredBase& interface);

      //! Printing utilities
      Teuchos::RCP<::NOX::Utils> utils_;

      //! Temporary Vector pointer used to compute directional derivatives
      Teuchos::RCP<::NOX::Abstract::Vector> vec_ptr_;
    };


    /// simple factory that creates SD relaxation class
    class SDFactory : public ::NOX::LineSearch::UserDefinedFactory
    {
     public:
      Teuchos::RCP<::NOX::LineSearch::Generic> buildLineSearch(
          const Teuchos::RCP<::NOX::GlobalData>& gd, Teuchos::ParameterList& params) const override
      {
        return Teuchos::make_rcp<SDRelaxation>(gd->getUtils(), params);
      }
    };

  }  // namespace FSI
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
