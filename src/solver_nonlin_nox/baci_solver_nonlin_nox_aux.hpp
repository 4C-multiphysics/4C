/*-----------------------------------------------------------*/
/*! \file

\brief Auxiliary methods.



\level 3

*/
/*-----------------------------------------------------------*/

#ifndef FOUR_C_SOLVER_NONLIN_NOX_AUX_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_AUX_HPP

#include "baci_config.hpp"

#include "baci_solver_nonlin_nox_enum_lists.hpp"
#include "baci_solver_nonlin_nox_forward_decl.hpp"
#include "baci_solver_nonlin_nox_statustest_factory.hpp"

#include <NOX_Abstract_Vector.H>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace CORE::LINALG
{
  class Solver;
  class SparseOperator;
}  // namespace CORE::LINALG

namespace NOX
{
  namespace NLN
  {
    namespace AUX
    {
      /*! Set printing parameters
       *
       *  Note: The Yes/No tuples are translated to booleans! */
      void SetPrintingParameters(Teuchos::ParameterList& p_nox, const Epetra_Comm& comm);

      /*! \brief Returns the type of operator that is passed in.
       *
       *   Uses dynamic casting to identify the underlying object type. */
      NOX::NLN::LinSystem::OperatorType GetOperatorType(const CORE::LINALG::SparseOperator& op);

      /// return linear system type
      NOX::NLN::LinSystem::LinearSystemType GetLinearSystemType(
          const std::map<enum NOX::NLN::SolutionType, Teuchos::RCP<CORE::LINALG::Solver>>&
              linsolvers);

      /*! \brief Calculate the root mean square for the NOX status test
       *  \f[
       *    \delta_{rms} = \sqrt{\frac{1}{N} \sum\limits_{i=1}^{N} \left( \frac{x_{i}^{k} -
       * x_{i}^{k-1}}{\mathcal{RTOL} | x_{i}^{k-1} | + \mathcal{ATOL}} \right)} \f]
       *
       *  \param atol  : absolute tolerance
       *  \param rtol  : relative tolerance
       *  \param xnew  : new / current iterate $x_{i}^{k}$
       *  \param xincr : current step increment $x_{i}^{k} - x_{i}^{k-1}$
       */
      double RootMeanSquareNorm(const double& atol, const double& rtol,
          Teuchos::RCP<const Epetra_Vector> xnew, Teuchos::RCP<const Epetra_Vector> xincr,
          const bool& disable_implicit_weighting = false);

      /*! \brief Do a recursive search for a NOX::NLN::StatusTest::NormWRMS object in the StatusTest
       * object list and return the class variable value of the desired quantity.
       *
       * \param test              : StatusTest object which will be scanned.
       * \param qType             : Quantity type of the NormWRMS test which we are looking for.
       * \param classVariableName : Name of the class variable which will be returned. (Type:
       * double) */
      double GetNormWRMSClassVariable(const ::NOX::StatusTest::Generic& test,
          const NOX::NLN::StatusTest::QuantityType& qType, const std::string& classVariableName);

      /*! \brief Do a recursive search for a NOX::NLN::StatusTest::NormF object in the StatusTest
       * object list and return the class variable value of the desired quantity.
       *
       * \param test              : StatusTest object which will be scanned.
       * \param qType             : Quantity type of the NormF test which we are looking for.
       * \param classVariableName : Name of the class variable which will be returned. (Type:
       * double) */
      double GetNormFClassVariable(const ::NOX::StatusTest::Generic& test,
          const NOX::NLN::StatusTest::QuantityType& qType, const std::string& classVariableName);

      /*! Do a recursive search for a <T> status test and the given quantity
       *
       *  True is returned as soon as a status test of type <T> is found, which
       *  holds the given quantity. */
      template <class T>
      bool IsQuantity(
          const ::NOX::StatusTest::Generic& test, const NOX::NLN::StatusTest::QuantityType& qtype);

      /*! \brief Do a recursive search for a <T> status test class and return the NormType of the
       * given quantity.
       *
       *  If there are more than one status tests of the type <T> which hold the given quantity, the
       * normtype of the first we can find, will be returned! */
      template <class T>
      int GetNormType(
          const ::NOX::StatusTest::Generic& test, const NOX::NLN::StatusTest::QuantityType& qtype);

      /// \brief Do a recursive search for a <T> status test class.
      template <class T>
      ::NOX::StatusTest::Generic* GetOuterStatusTest(::NOX::StatusTest::Generic& full_otest);

      /** \brief Do a recursive search for a <T> status test class containing
       *  the given quantity. */
      template <class T>
      ::NOX::StatusTest::Generic* GetOuterStatusTestWithQuantity(
          ::NOX::StatusTest::Generic& test, const NOX::NLN::StatusTest::QuantityType qtype);

      /*! \brief Do a recursive search for a <T> status test class and return its status.
       *
       * If more than one of the given status test objects is combined in a combination list,
       * the AND combination of the different status is returned. I.e. if one of the status
       * is unconverged, the return status is unconverged.
       * If we cannot find the given status test class, a default value of -100 is returned.
       *
       * \param test (in) : StatusTest object which will be scanned.
       */
      template <class T>
      int GetOuterStatus(const ::NOX::StatusTest::Generic& test);

      /*! \brief Convert the quantity type to a solution type
       *
       * \param qtype : Quantity type which has to be converted.
       */
      enum NOX::NLN::SolutionType ConvertQuantityType2SolutionType(
          const enum NOX::NLN::StatusTest::QuantityType& qtype);

      /*! \brief Map norm type stl_string to norm type enum
       *
       * \param name : Name of the vector norm type.
       */
      enum ::NOX::Abstract::Vector::NormType String2NormType(const std::string& name);

      /// add pre/post operator to pre/post operator vector
      void AddToPrePostOpVector(
          Teuchos::ParameterList& p_nox_opt, const Teuchos::RCP<::NOX::Observer>& ppo_ptr);

      /// return the name of the parameter list corresponding to the set direction method
      std::string GetDirectionMethodListName(const Teuchos::ParameterList& p);

    }  // namespace AUX
  }    // namespace NLN
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
