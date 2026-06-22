// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_NLN_SOLVER_UTILS_HPP
#define FOUR_C_STRUCTURE_NEW_NLN_SOLVER_UTILS_HPP

#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_statustest_factory.hpp"

#include <set>

// forward declaration
FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Solver;
}
namespace Solid
{
  namespace TimeInt
  {
    class Base;
    class BaseDataSDyn;
  }  // namespace TimeInt
  namespace Nln
  {
    namespace SOLVER
    {
      //! Check if a xml status test file is specified in the input file
      bool is_xml_status_test_file(const Teuchos::ParameterList& pstatus);

      /*! \brief Create quantity types
       *
       *  This function translates the model type enums and element technology enums
       *  into quantity types.
       *
       */
      void create_quantity_types(std::set<NOX::Nln::StatusTest::QuantityType>& qtypes,
          const Solid::TimeInt::BaseDataSDyn& datasdyn);

      void convert_model_type_to_quantity_type(
          const Inpar::Solid::ModelType& mt, std::vector<NOX::Nln::StatusTest::QuantityType>& qt);

      void convert_ele_tech_to_quantity_type(
          const Inpar::Solid::EleTech& et, std::vector<NOX::Nln::StatusTest::QuantityType>& qt);

      /*! \brief Create a status test parameter list
       *
       *  A status test parameter list for the outer status test is created. The
       *  information comes from the input file. Actually we consider only convergence
       *  tests which were already in 4C before the NOX framework was introduced.
       *
       *  Feel free to extend the given framework, to generalize it or to use a xml
       *  file, where the path is specified in the "STRUCT NOX/Status Test" sublist.
       *  The last option comes for free and there is no need to modify any code
       *  fragments.
       *
       */
      void set_status_test_params(Teuchos::ParameterList& pstatus,
          const Solid::TimeInt::BaseDataSDyn& datasdyn,
          const std::set<NOX::Nln::StatusTest::QuantityType>& qt);

      //! Split the given tests into and/or combinations
      void split_and_or_combo(std::vector<NOX::Nln::StatusTest::QuantityType>& combo_or,
          std::vector<NOX::Nln::StatusTest::QuantityType>& combo_and,
          const Solid::TimeInt::BaseDataSDyn& datasdyn, const std::string& testname,
          const std::set<NOX::Nln::StatusTest::QuantityType>& qtypes);

      /*! \brief Set the combination of different NormF or NormUpdate tests in the status test
       * parameter list
       *
       *  Maybe the current implementation needs a short explanation:
       *  You can specify different combinations in your input file. Let's concentrate on
       *  the NormF case and imagine the following status test settings:
       *
       *  NORMCOMBI_RESFINCO           AND        (RESIDUAL and PRESSURE)
       *  NORMCOMBI_RESFCONSTR         AND        (RESIDUAL and CONTACT)
       *
       *  Following will happen:
       *
       *  (STRUCTURE and PRESSURE and CONTACT) or EAS or PLASTICITY
       *
       *  This means, that the OR-combination is a strong OR and ignores/omits all remaining tests!
       *  In the old implementation the exact behavior was more or less random, because it depended
       * on the order of the implemented tests. In this way the OR-combination is a debugging tool
       * and you can check your results by forcing only one of the corresponding residuals to zero
       *  (i.e. the named partial residual or all remaining parts of the residual and-combinations).
       *
       *  If you want a different behavior, please use a xml file instead.
       *
       *  One last note: In the case you want to commit your input file as a test case, your
       * algorithm should not use any OR-combination, since the goal of your algorithm should be to
       * reduce the whole residual. If you think a part of your residual can not be reduced in a
       * sufficient way, think again and if it stays your opinion, do not check it (by using a
       * xml-file for example). This makes things easier to read and understand. If you use only
       * AND-combinations, you can use the QUANTITY parameter list name option for NormF, NormWRMS
       * and NormUpdate tests. See the NOX::Nln::StatusTest::Factory for more information.
       *

       */
      void set_combo_quantity_test_params(Teuchos::ParameterList& p,
          const Solid::TimeInt::BaseDataSDyn& datasdyn, const std::size_t& count,
          const std::string& testname, const std::set<NOX::Nln::StatusTest::QuantityType>& qtypes);

      /*! \brief Set the status test corresponding to the given quantity
       *
       *  Create a new sublist corresponding to the count variable and create the specific status
       * test entries in the parameter list for the given quantity
       *
       */
      void set_quantity_test_params(Teuchos::ParameterList& p,
          const Solid::TimeInt::BaseDataSDyn& datasdyn,
          const NOX::Nln::StatusTest::QuantityType& qtype, const std::size_t& count,
          const std::string& testname);

      //! Set the status test parameters corresponding to the given quantity
      void set_quantity_test_params(Teuchos::ParameterList& p,
          const Solid::TimeInt::BaseDataSDyn& datasdyn,
          const NOX::Nln::StatusTest::QuantityType& qtype, const std::string& testname);

      //! Set the NormUpdate status test parameters
      void set_norm_update_params(Teuchos::ParameterList& qlist,
          const NOX::Nln::StatusTest::QuantityType& qtype, const Inpar::Solid::ConvNorm& toltype,
          const double& tol, const Inpar::Solid::VectorNorm& normtype);
      void set_norm_update_params(Teuchos::ParameterList& qlist,
          const NOX::Nln::StatusTest::QuantityType& qtype, const double& alpha, const double& beta,
          const Inpar::Solid::ConvNorm& toltype, const double& tol,
          const Inpar::Solid::VectorNorm& normtype, const bool& isscaled);

      //! Set the NormF status test parameters
      void set_norm_f_params(Teuchos::ParameterList& qlist,
          const NOX::Nln::StatusTest::QuantityType& qtype, const Inpar::Solid::ConvNorm& toltype,
          const double& tol, const Inpar::Solid::VectorNorm& normtype);
      void set_norm_f_params(Teuchos::ParameterList& qlist,
          const NOX::Nln::StatusTest::QuantityType& qtype, const Inpar::Solid::ConvNorm& toltype,
          const double& tol, const Inpar::Solid::VectorNorm& normtype, const bool& isscaled);

      //! Set the ActiveSet status test parameters
      void set_active_set_params(
          Teuchos::ParameterList& qlist, const NOX::Nln::StatusTest::QuantityType& qtype);

    }  // namespace SOLVER
  }  // namespace Nln
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
