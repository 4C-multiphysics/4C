// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_aux.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"
#include "4C_solver_nonlin_nox_statustest_activeset.hpp"
#include "4C_solver_nonlin_nox_statustest_combo.hpp"
#include "4C_solver_nonlin_nox_statustest_normf.hpp"
#include "4C_solver_nonlin_nox_statustest_normupdate.hpp"
#include "4C_solver_nonlin_nox_statustest_normwrms.hpp"

#include <NOX_Observer_Vector.hpp>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Aux::set_printing_parameters(Teuchos::ParameterList& p_nox, MPI_Comm comm)
{
  // adjust printing parameter list
  Teuchos::ParameterList& printParams = p_nox.sublist("Printing");
  printParams.set<int>("MyPID", Core::Communication::my_mpi_rank(comm));
  printParams.set<int>("Output Precision", 5);
  printParams.set<int>("Output Processor", 0);
  int outputinformationlevel = ::NOX::Utils::Error;  // ::NOX::Utils::Error==0
  if (printParams.get<bool>("Error", true)) outputinformationlevel += ::NOX::Utils::Error;
  if (printParams.get<bool>("Warning", true)) outputinformationlevel += ::NOX::Utils::Warning;
  if (printParams.get<bool>("Outer Iteration", true))
    outputinformationlevel += ::NOX::Utils::OuterIteration;
  if (printParams.get<bool>("Inner Iteration", true))
    outputinformationlevel += ::NOX::Utils::InnerIteration;
  if (printParams.get<bool>("Parameters", false))
    outputinformationlevel += ::NOX::Utils::Parameters;
  if (printParams.get<bool>("Details", false)) outputinformationlevel += ::NOX::Utils::Details;
  if (printParams.get<bool>("Outer Iteration StatusTest", true))
    outputinformationlevel += ::NOX::Utils::OuterIterationStatusTest;
  if (printParams.get<bool>("Linear Solver Details", false))
    outputinformationlevel += ::NOX::Utils::LinearSolverDetails;
  if (printParams.get<bool>("Test Details", false))
    outputinformationlevel += ::NOX::Utils::TestDetails;
  if (printParams.get<bool>("Debug", false)) outputinformationlevel += ::NOX::Utils::Debug;
  printParams.set("Output Information", outputinformationlevel);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::OperatorType NOX::Nln::Aux::get_operator_type(
    const Core::LinAlg::SparseOperator& op)
{
  const Core::LinAlg::SparseOperator* testOperator = nullptr;

  // Is it a LINALG_BlockSparseMatrix
  testOperator = dynamic_cast<
      const Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>*>(&op);
  if (testOperator != nullptr) return NOX::Nln::LinSystem::LinalgBlockSparseMatrix;

  // Is it a LINALG_SparseMatrix?
  testOperator = dynamic_cast<const Core::LinAlg::SparseMatrix*>(&op);
  if (testOperator != nullptr) return NOX::Nln::LinSystem::LinalgSparseMatrix;

  // Otherwise it must be a LINALG_SparseOperator
  return NOX::Nln::LinSystem::LinalgSparseOperator;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::LinearSystemType NOX::Nln::Aux::get_linear_system_type(
    const NOX::Nln::LinearSystem::SolverMap& linsolvers)
{
  const unsigned int num_ls = linsolvers.size();
  const auto ci_end = linsolvers.end();

  switch (num_ls)
  {
    case 1:
    {
      // --- Pure structural case (+ spring dashpot)
      if (linsolvers.find(NOX::Nln::sol_structure) != ci_end)
      {
        return NOX::Nln::LinSystem::linear_system_structure;
      }
      else if (linsolvers.find(NOX::Nln::sol_scatra) != ci_end)
      {
        return NOX::Nln::LinSystem::linear_system_scatra;
      }
      // --- ToDo has to be extended

      FOUR_C_THROW(
          "There is no capable linear system type for the given linear "
          "solver combination! ( 1 linear solver )");
    }
    case 2:
    {
      // --- Structure/Contact case (+ spring dashpot)
      if (linsolvers.find(NOX::Nln::sol_structure) != ci_end and
          linsolvers.find(NOX::Nln::sol_contact) != ci_end)
      {
        return NOX::Nln::LinSystem::linear_system_structure_contact;
      }
      // --- Structure/CardioVascular0D case (+ spring dashpot)
      else if (linsolvers.find(NOX::Nln::sol_structure) != ci_end and
               linsolvers.find(NOX::Nln::sol_cardiovascular0d) != ci_end)
      {
        return NOX::Nln::LinSystem::linear_system_structure_cardiovascular0d;
      }
      // --- Structure/Lagrange|Penalty Constraint case (+ spring dashpot)
      else if (linsolvers.find(NOX::Nln::sol_structure) != ci_end and
               linsolvers.find(NOX::Nln::sol_lag_pen_constraint) != ci_end)
      {
        return NOX::Nln::LinSystem::linear_system_structure_lag_pen_constraint;
      }
      else if (linsolvers.find(NOX::Nln::sol_structure) != ci_end and
               linsolvers.find(NOX::Nln::sol_meshtying) != ci_end)
      {
        return NOX::Nln::LinSystem::linear_system_structure_meshtying;
      }
      // --- ToDo has to be extended

      FOUR_C_THROW(
          "There is no capable linear system type for the given linear "
          "solver combination ( 2 linear solvers )!");
    }
    case 3:
    {
      // --- Structure/Contact case (+ spring dashpot)
      if (linsolvers.find(NOX::Nln::sol_structure) != ci_end and
          linsolvers.find(NOX::Nln::sol_contact) != ci_end and
          linsolvers.find(NOX::Nln::sol_meshtying) != ci_end)
      {
        return NOX::Nln::LinSystem::linear_system_structure_contact;
      }
      else if (linsolvers.find(NOX::Nln::sol_structure) != ci_end and
               linsolvers.find(NOX::Nln::sol_contact) != ci_end and
               linsolvers.find(NOX::Nln::sol_cardiovascular0d) != ci_end)
      {
        return NOX::Nln::LinSystem::linear_system_structure_contact;
      }
      FOUR_C_THROW(
          "There is no capable linear system type for the given linear "
          "solver combination ( 3 linear solvers )!");
    }
    default:
    {
      FOUR_C_THROW(
          "There is no capable linear system type for the given linear "
          "solver combination!");
    }
  }

  return NOX::Nln::LinSystem::linear_system_undefined;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double NOX::Nln::Aux::root_mean_square_norm(const double& atol, const double& rtol,
    const Core::LinAlg::Vector<double>& xnew, const Core::LinAlg::Vector<double>& xincr,
    const bool& disable_implicit_weighting)
{
  double rval = 0.0;

  // calculate the old iterate (k-1)
  Core::LinAlg::Vector<double> v(xnew);
  v.update(-1.0, xincr, 1.0);

  // new auxiliary vector
  Core::LinAlg::Vector<double> u(xnew.get_map(), false);

  // create the weighting factor u = RTOL |x^(k-1)| + ATOL
  u.put_scalar(1.0);
  u.update(rtol, v, atol);

  // v = xincr/u (elementwise)
  v.reciprocal_multiply(1.0, u, xincr, 0);

  // rval = sqrt (v * v / N)
  v.norm_2(&rval);
  rval /= std::sqrt(static_cast<double>(v.global_length()));

  return rval;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double NOX::Nln::Aux::get_norm_wrms_class_variable(const ::NOX::StatusTest::Generic& test,
    const NOX::Nln::StatusTest::QuantityType& qType, const std::string& classVariableName)
{
  // try to cast the given test to a NOX_StatusTest_Combo
  const NOX::Nln::StatusTest::Combo* comboTest =
      dynamic_cast<const NOX::Nln::StatusTest::Combo*>(&test);

  // if it is no combo test, we just have to check for the desired type
  if (comboTest == nullptr)
  {
    const NOX::Nln::StatusTest::NormWRMS* normWRMSTest =
        dynamic_cast<const NOX::Nln::StatusTest::NormWRMS*>(&test);

    // no normF StatusTest...
    if (normWRMSTest == nullptr) return -1.0;
    // yeah we found one...
    else
    {
      // look for the right quantity and get the desired class variable value
      if (classVariableName == "ATOL")
        return normWRMSTest->get_absolute_tolerance(qType);
      else if (classVariableName == "RTOL")
        return normWRMSTest->get_relative_tolerance(qType);
    }
  }
  // if the nox_nln_statustest_combo Test cast was successful
  else
  {
    const std::vector<Teuchos::RCP<::NOX::StatusTest::Generic>>& tests =
        comboTest->get_test_vector();
    double ret = -1.0;
    for (std::size_t i = 0; i < tests.size(); ++i)
    {
      // recursive function call
      ret = get_norm_wrms_class_variable(*(tests[i]), qType, classVariableName);
      if (ret != -1.0) return ret;
    }
  }

  // default return
  return -1.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double NOX::Nln::Aux::get_norm_f_class_variable(const ::NOX::StatusTest::Generic& test,
    const NOX::Nln::StatusTest::QuantityType& qType, const std::string& classVariableName)
{
  // try to cast the given test to a NOX_StatusTest_Combo
  const NOX::Nln::StatusTest::Combo* comboTest =
      dynamic_cast<const NOX::Nln::StatusTest::Combo*>(&test);

  // if it is no combo test, we just have to check for the desired type
  if (comboTest == nullptr)
  {
    const NOX::Nln::StatusTest::NormF* normFTest =
        dynamic_cast<const NOX::Nln::StatusTest::NormF*>(&test);

    // no normF StatusTest...
    if (normFTest == nullptr) return -1.0;
    // yeah we found one...
    else
    {
      // look for the right quantity and get the desired class variable value
      if (classVariableName == "NormF")
        return normFTest->get_norm_f(qType);
      else if (classVariableName == "TrueTolerance")
        return normFTest->get_true_tolerance(qType);
      else if (classVariableName == "SpecifiedTolerance")
        return normFTest->get_specified_tolerance(qType);
      else if (classVariableName == "InitialTolerance")
        return normFTest->get_initial_tolerance(qType);
    }
  }
  // if the nox_nln_statustest_combo Test cast was successful
  else
  {
    const std::vector<Teuchos::RCP<::NOX::StatusTest::Generic>>& tests =
        comboTest->get_test_vector();
    double ret = -1.0;
    for (std::size_t i = 0; i < tests.size(); ++i)
    {
      // recursive function call
      ret = get_norm_f_class_variable(*(tests[i]), qType, classVariableName);
      if (ret != -1.0) return ret;
    }
  }

  // default return
  return -1.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <class T>
bool NOX::Nln::Aux::is_quantity(
    const ::NOX::StatusTest::Generic& test, const NOX::Nln::StatusTest::QuantityType& qtype)
{
  // try to cast the given test to a NOX_StatusTest_Combo
  const NOX::Nln::StatusTest::Combo* comboTest =
      dynamic_cast<const NOX::Nln::StatusTest::Combo*>(&test);

  // if it is no combo test, we just have to check for the desired type
  if (comboTest == nullptr)
  {
    const T* desiredTest = dynamic_cast<const T*>(&test);

    // not the desired status test...
    if (desiredTest == nullptr) return false;
    // yeah we found one...
    else
    {
      // check for the quantity
      return desiredTest->is_quantity(qtype);
    }
  }
  // if the nox_nln_statustest_combo Test cast was successful
  else
  {
    const std::vector<Teuchos::RCP<::NOX::StatusTest::Generic>>& tests =
        comboTest->get_test_vector();
    for (std::size_t i = 0; i < tests.size(); ++i)
    {
      // recursive function call
      bool isfound = is_quantity<T>(*(tests[i]), qtype);
      if (isfound) return isfound;
    }
  }

  // default return
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <class T>
int NOX::Nln::Aux::get_norm_type(
    const ::NOX::StatusTest::Generic& test, const NOX::Nln::StatusTest::QuantityType& qtype)
{
  // try to cast the given test to a NOX_StatusTest_Combo
  const NOX::Nln::StatusTest::Combo* comboTest =
      dynamic_cast<const NOX::Nln::StatusTest::Combo*>(&test);

  // if it is no combo test, we just have to check for the desired type
  if (comboTest == nullptr)
  {
    const T* desiredTest = dynamic_cast<const T*>(&test);

    // not the desired status test...
    if (desiredTest == nullptr) return -100;
    // yeah we found one...
    else
    {
      // get the norm type of the given quantity
      return desiredTest->get_norm_type(qtype);
    }
  }
  // if the nox_nln_statustest_combo Test cast was successful
  else
  {
    const std::vector<Teuchos::RCP<::NOX::StatusTest::Generic>>& tests =
        comboTest->get_test_vector();
    for (std::size_t i = 0; i < tests.size(); ++i)
    {
      // recursive function call
      int ret = get_norm_type<T>(*(tests[i]), qtype);
      if (ret != -100) return ret;
    }
  }

  // default return
  return -100;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <class T>
::NOX::StatusTest::Generic* NOX::Nln::Aux::get_outer_status_test(
    ::NOX::StatusTest::Generic& full_otest)
{
  // try to cast the given test to a NOX_StatusTest_Combo
  const NOX::Nln::StatusTest::Combo* comboTest =
      dynamic_cast<const NOX::Nln::StatusTest::Combo*>(&full_otest);

  // if it is no combo test, we just have to check for the desired type
  if (not comboTest)
  {
    return dynamic_cast<T*>(&full_otest);
  }
  // if the nox_nln_statustest_combo Test cast was successful
  else
  {
    const std::vector<Teuchos::RCP<::NOX::StatusTest::Generic>>& tests =
        comboTest->get_test_vector();

    ::NOX::StatusTest::Generic* gdesired_test = nullptr;
    for (const auto& test : tests)
    {
      // recursive function call
      ::NOX::StatusTest::Generic* desired_test = get_outer_status_test<T>(*test);

      // the test is not of the specified type, go to the next one
      if (not desired_test) continue;

      // first found test
      if (not gdesired_test) gdesired_test = desired_test;
      // we've found already one test of the same type
      else
      {
        const enum ::NOX::StatusTest::StatusType gstatus = gdesired_test->getStatus();

        // If there are more tests of the same type, we return the
        // test which is possible unconverged (conservative choice, AND-combination).
        gdesired_test = (gstatus == ::NOX::StatusTest::Converged ? desired_test : gdesired_test);
      }
    }
    return gdesired_test;
  }

  // default return
  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <class T>
int NOX::Nln::Aux::get_outer_status(const ::NOX::StatusTest::Generic& test)
{
  // try to cast the given test to a NOX_StatusTest_Combo
  const NOX::Nln::StatusTest::Combo* comboTest =
      dynamic_cast<const NOX::Nln::StatusTest::Combo*>(&test);

  // if it is no combo test, we just have to check for the desired type
  if (comboTest == nullptr)
  {
    const T* desiredTest = dynamic_cast<const T*>(&test);

    // not the desired status test...
    if (desiredTest == nullptr) return -100;
    // yeah we found one...
    else
    {
      // get the global status
      return desiredTest->getStatus();
    }
  }
  // if the nox_nln_statustest_combo Test cast was successful
  else
  {
    const std::vector<Teuchos::RCP<::NOX::StatusTest::Generic>>& tests =
        comboTest->get_test_vector();
    int gRet = -100;
    for (std::size_t i = 0; i < tests.size(); ++i)
    {
      // recursive function call
      int lRet = get_outer_status<T>(*(tests[i]));

      // the test is not of the specified type, go to the next one
      if (lRet == -100) continue;

      // first found test
      if (gRet == -100) gRet = lRet;
      // we've found already one test of the same type
      else
      {
        ::NOX::StatusTest::StatusType gstatus =
            static_cast<enum ::NOX::StatusTest::StatusType>(gRet);

        // If there are more tests of the same type, we return the
        // status of the possible unconverged test (conservative choice).
        gRet = (gstatus == ::NOX::StatusTest::Converged ? lRet : gRet);
      }
    }
    return gRet;
  }

  // default return
  return -100;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::SolutionType NOX::Nln::Aux::convert_quantity_type_to_solution_type(
    const NOX::Nln::StatusTest::QuantityType& qtype)
{
  NOX::Nln::SolutionType soltype = NOX::Nln::sol_unknown;
  switch (qtype)
  {
    case NOX::Nln::StatusTest::quantity_structure:
    case NOX::Nln::StatusTest::quantity_eas:
    case NOX::Nln::StatusTest::quantity_pressure:
      soltype = NOX::Nln::sol_structure;
      break;
    case NOX::Nln::StatusTest::quantity_lag_pen_constraint:
      soltype = NOX::Nln::sol_lag_pen_constraint;
      break;
    case NOX::Nln::StatusTest::quantity_contact_normal:
    case NOX::Nln::StatusTest::quantity_contact_friction:
      soltype = NOX::Nln::sol_contact;
      break;
    case NOX::Nln::StatusTest::quantity_meshtying:
      soltype = NOX::Nln::sol_meshtying;
      break;
    case NOX::Nln::StatusTest::quantity_cardiovascular0d:
      soltype = NOX::Nln::sol_cardiovascular0d;
      break;
    case NOX::Nln::StatusTest::quantity_beaminteraction_lm:
      soltype = NOX::Nln::sol_beaminteraction_lm;
      break;
    case NOX::Nln::StatusTest::quantity_unknown:
    default:
      FOUR_C_THROW("Unknown conversion for the quantity type \"{}\".",
          NOX::Nln::StatusTest::quantity_type_to_string(qtype));
  }
  // return the corresponding solution type
  return soltype;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum ::NOX::Abstract::Vector::NormType NOX::Nln::Aux::string_to_norm_type(const std::string& name)
{
  enum ::NOX::Abstract::Vector::NormType norm_type = ::NOX::Abstract::Vector::TwoNorm;
  if (name == "Two Norm")
    norm_type = ::NOX::Abstract::Vector::TwoNorm;
  else if (name == "One Norm")
    norm_type = ::NOX::Abstract::Vector::OneNorm;
  else if (name == "Max Norm")
    norm_type = ::NOX::Abstract::Vector::MaxNorm;
  else
    FOUR_C_THROW("Unknown conversion from STL_STRING to NormType enum for {}.", name);

  return norm_type;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Aux::add_to_pre_post_op_vector(
    Teuchos::ParameterList& p_nox_opt, const Teuchos::RCP<::NOX::Observer>& ppo_ptr)
{
  // if there is already a pre/post operator, we will convert the pre/post op
  // to a pre/post op vector and add the previous and new pre/post op
  if (p_nox_opt.isType<Teuchos::RCP<::NOX::Observer>>("User Defined Pre/Post Operator"))
  {
    Teuchos::RCP<::NOX::Observer> user_ppo =
        p_nox_opt.get<Teuchos::RCP<::NOX::Observer>>("User Defined Pre/Post Operator");

    Teuchos::RCP<::NOX::ObserverVector> user_ppo_vec =
        Teuchos::rcp_dynamic_cast<::NOX::ObserverVector>(user_ppo, false);

    if (user_ppo_vec.is_null())
    {
      user_ppo_vec = Teuchos::make_rcp<::NOX::ObserverVector>();
      user_ppo_vec->pushBack(user_ppo);
    }

    user_ppo_vec->pushBack(ppo_ptr);

    p_nox_opt.set<Teuchos::RCP<::NOX::Observer>>("User Defined Pre/Post Operator", user_ppo_vec);
  }
  // if there is no pre/post operator, we will just add the new one
  else
    p_nox_opt.set<Teuchos::RCP<::NOX::Observer>>("User Defined Pre/Post Operator", ppo_ptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::string NOX::Nln::Aux::get_direction_method_list_name(const Teuchos::ParameterList& p)
{
  if (not p.isSublist("Direction"))
    FOUR_C_THROW("There is no \"Direction\" sub-list in the parameter list!");
  const Teuchos::ParameterList& pdir = p.sublist("Direction");

  if (not pdir.isParameter("Method"))
    FOUR_C_THROW("There is no \"Method\" parameter in the Direction sub-list!");

  const std::string* dir_str = &pdir.get<std::string>("Method");
  if (*dir_str == "User Defined")
  {
    dir_str = &pdir.get<std::string>("User Defined Method");
  }
  if (*dir_str == "Newton" or *dir_str == "Modified Newton")
    return "Newton";
  else
  {
    FOUR_C_THROW("Currently unsupported direction method string: {}", *dir_str);
  }
}

template <typename T>
T NOX::Nln::Aux::calc_vector_norm(const Core::LinAlg::Vector<T>& vector,
    const ::NOX::Abstract::Vector::NormType type, const bool is_scaled)
{
  T value;

  switch (type)
  {
    case ::NOX::Abstract::Vector::OneNorm:
      vector.norm_1(&value);
      break;
    case ::NOX::Abstract::Vector::TwoNorm:
      vector.norm_2(&value);
      break;
    case ::NOX::Abstract::Vector::MaxNorm:
      vector.norm_inf(&value);
      break;
    default:
      FOUR_C_THROW("The norm type is not supported");
  }

  // do the scaling if desired
  if (is_scaled) value /= static_cast<T>(vector.global_length());

  return value;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template bool NOX::Nln::Aux::is_quantity<NOX::Nln::StatusTest::NormF>(
    const ::NOX::StatusTest::Generic& test, const NOX::Nln::StatusTest::QuantityType& qtype);
template bool NOX::Nln::Aux::is_quantity<NOX::Nln::StatusTest::NormUpdate>(
    const ::NOX::StatusTest::Generic& test, const NOX::Nln::StatusTest::QuantityType& qtype);
template bool NOX::Nln::Aux::is_quantity<NOX::Nln::StatusTest::NormWRMS>(
    const ::NOX::StatusTest::Generic& test, const NOX::Nln::StatusTest::QuantityType& qtype);
template int NOX::Nln::Aux::get_norm_type<NOX::Nln::StatusTest::NormF>(
    const ::NOX::StatusTest::Generic& test, const NOX::Nln::StatusTest::QuantityType& qtype);
template int NOX::Nln::Aux::get_norm_type<NOX::Nln::StatusTest::NormUpdate>(
    const ::NOX::StatusTest::Generic& test, const NOX::Nln::StatusTest::QuantityType& qtype);
template ::NOX::StatusTest::Generic*
NOX::Nln::Aux::get_outer_status_test<NOX::Nln::StatusTest::ActiveSet>(
    ::NOX::StatusTest::Generic& full_otest);
template int NOX::Nln::Aux::get_outer_status<NOX::Nln::StatusTest::NormF>(
    const ::NOX::StatusTest::Generic& test);
template int NOX::Nln::Aux::get_outer_status<NOX::Nln::StatusTest::NormUpdate>(
    const ::NOX::StatusTest::Generic& test);
template int NOX::Nln::Aux::get_outer_status<NOX::Nln::StatusTest::NormWRMS>(
    const ::NOX::StatusTest::Generic& test);
template int NOX::Nln::Aux::get_outer_status<NOX::Nln::StatusTest::ActiveSet>(
    const ::NOX::StatusTest::Generic& test);
template double NOX::Nln::Aux::calc_vector_norm<double>(const Core::LinAlg::Vector<double>& vector,
    const ::NOX::Abstract::Vector::NormType type, const bool is_scaled);

FOUR_C_NAMESPACE_CLOSE
