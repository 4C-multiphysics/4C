/*-----------------------------------------------------------*/
/*! \file

\brief Implementation of the Lagrangian merit function for
       constrained problems.



\level 3

*/
/*-----------------------------------------------------------*/

#ifndef FOUR_C_SOLVER_NONLIN_NOX_MERITFUNCTION_LAGRANGIAN_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_MERITFUNCTION_LAGRANGIAN_HPP

#include "baci_config.hpp"

#include "baci_solver_nonlin_nox_enum_lists.hpp"
#include "baci_solver_nonlin_nox_forward_decl.hpp"
#include "baci_utils_exceptions.hpp"

#include <NOX_LineSearch_Utils_Slope.H>
#include <NOX_MeritFunction_Generic.H>  // base class
#include <Teuchos_RCP.hpp>

BACI_NAMESPACE_OPEN

namespace NOX
{
  namespace NLN
  {
    namespace MeritFunction
    {
      class Lagrangian : public virtual ::NOX::MeritFunction::Generic
      {
       public:
        //! Constructor
        Lagrangian(const std::string& identifier, const Teuchos::RCP<::NOX::Utils>& u);

        /** \brief Computes the Lagrangian merit function
         *
         *  \f$ f = \mathcal{L}(x,\lambda) \f$. */
        double computef(const ::NOX::Abstract::Group& grp) const override;

        /** \brief Computes the gradient
         *
         *  \f$ g = \nabla_{x,\lambda_N} f = [ r_s - \nabla_x \tilde{g}_N^{\mathcal{A}}
         *          \lambda_{N}^{\mathcal{A}}, -\tilde{g}_N^{\mathcal{A}}, -\frac{2}{c_N}
         * A^{\mathcal{I}} \lambda_{N}^{\mathcal{I}} ] \f$. */
        void computeGradient(
            const ::NOX::Abstract::Group& group, ::NOX::Abstract::Vector& result) const override
        {
          dserror("computeGradient is not implemented.");
        }

        //! Compute the slope of the Lagrangian merit function.
        double computeSlope(
            const ::NOX::Abstract::Vector& dir, const ::NOX::Abstract::Group& grp) const override;

        double computeMixed2ndOrderTerms(
            const ::NOX::Abstract::Vector& dir, const ::NOX::Abstract::Group& grp) const;

        /** \brief Computes the quadratic model.
         *
         *  An alternative is the SaddlePointModel. */
        double computeQuadraticModel(
            const ::NOX::Abstract::Vector& dir, const ::NOX::Abstract::Group& grp) const override
        {
          dserror("computeQuadraticModel is not implemented.");
          return 0.0;
        }

        //! Computes the quadratic model minimizer.
        void computeQuadraticMinimizer(
            const ::NOX::Abstract::Group& grp, ::NOX::Abstract::Vector& result) const override
        {
          dserror("computeQuadraticMinimizer is not implemented.");
        }

        /** \brief Computes the saddle-point model of the Lagrangian functional
         *
         *  The function returns the linearized model without the actual function value!
         *
         *  \author hiermeier \date 04/17 */
        virtual double computeSaddlePointModel(const double& stepPV, const double& stepLM,
            const ::NOX::Abstract::Vector& dir, const ::NOX::Abstract::Group& grp) const;

        /** \brief Alternative function call.
         *
         *  Here we choose the same step size for
         *  the primary and Lagrange multiplier degrees of freedom. */
        double computeSaddlePointModel(const double& step, const ::NOX::Abstract::Vector& dir,
            const ::NOX::Abstract::Group& grp) const;

        //! return the name of the merit function
        const std::string& name() const override;

        //! return the name of the merit function
        inline enum MeritFctName Type() const { return lagrangian_type_; }

       private:
        /// \brief Get a list of currently supported Lagrangian merit function types
        /** This list is a sub-list of the merit function enumerator list.
         *
         *  \author hiermeier \date 12/17 */
        std::map<std::string, MeritFctName> GetSupportedTypeList() const;

        /// Set the Lagrangian merit function type
        void SetType(const std::string& identifier);

        //! Throws NOX error
        void throwError(const std::string& functionName, const std::string& errorMsg) const;

       private:
        //! Printing utilities.
        Teuchos::RCP<::NOX::Utils> utils_;

        enum MeritFctName lagrangian_type_;

        //! name of this function
        std::string meritFunctionName_;
      };

    }  // namespace MeritFunction
  }    // namespace NLN
}  // namespace NOX

BACI_NAMESPACE_CLOSE

#endif  // SOLVER_NONLIN_NOX_MERITFUNCTION_LAGRANGIAN_H
