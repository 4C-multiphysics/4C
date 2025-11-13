// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_globaldata.hpp"  // class definition

#include "4C_comm_mpi_utils.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_direction_factory.hpp"
#include "4C_solver_nonlin_nox_interface_required_base.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"
#include "4C_solver_nonlin_nox_meritfunction_factory.hpp"
#include "4C_solver_nonlin_nox_scaling.hpp"
#include "4C_solver_nonlin_nox_solver_prepostop_generic.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_Epetra_Interface_Jacobian.H>
#include <NOX_Utils.H>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include <filesystem>
#include <stdexcept>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::GlobalData::GlobalData(MPI_Comm comm, Teuchos::ParameterList& noxParams,
    const NOX::Nln::LinearSystem::SolverMap& linSolvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::Nln::OptimizationProblemType& type,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const std::shared_ptr<NOX::Nln::Scaling>& iScale)
    : comm_(comm),
      nlnparams_(Teuchos::rcpFromRef(noxParams)),
      opt_type_(type),
      lin_solvers_(linSolvers),
      i_req_ptr_(iReq),
      i_jac_ptr_(iJac),
      i_constr_(iConstr),
      i_constr_prec_(iConstrPrec),
      i_scale_(iScale),
      mrt_fct_ptr_(Teuchos::null),
      pre_post_op_ptr_(Teuchos::null),
      is_constrained_(type != opt_unconstrained)
{
  check_input();
  // do some setup things
  setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::GlobalData::GlobalData(MPI_Comm comm, Teuchos::ParameterList& noxParams,
    const NOX::Nln::LinearSystem::SolverMap& linSolvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const OptimizationProblemType& type, const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr)
    : comm_(comm),
      nlnparams_(Teuchos::rcpFromRef(noxParams)),
      opt_type_(type),
      lin_solvers_(linSolvers),
      i_req_ptr_(iReq),
      i_jac_ptr_(iJac),
      i_constr_(iConstr),
      mrt_fct_ptr_(Teuchos::null),
      pre_post_op_ptr_(Teuchos::null),
      is_constrained_(type != opt_unconstrained)
{
  check_input();
  // do some setup things
  setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::GlobalData::GlobalData(MPI_Comm comm, Teuchos::ParameterList& noxParams,
    const NOX::Nln::LinearSystem::SolverMap& linSolvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac)
    : comm_(comm),
      nlnparams_(Teuchos::rcpFromRef(noxParams)),
      opt_type_(opt_unconstrained),
      lin_solvers_(linSolvers),
      i_req_ptr_(iReq),
      i_jac_ptr_(iJac),
      mrt_fct_ptr_(Teuchos::null),
      pre_post_op_ptr_(Teuchos::null),
      is_constrained_(false)
{
  check_input();
  // do some setup things
  setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::GlobalData::check_input() const
{
  // short input check
  if (lin_solvers_.size() == 0)
    FOUR_C_THROW("The linear solver map has the size 0! Required size > 0.");

  using CI = std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>::const_iterator;
  for (CI iter = lin_solvers_.begin(); iter != lin_solvers_.end(); ++iter)
    if (iter->second.is_null())
    {
      std::ostringstream msg;
      const std::string act_msg("        WARNING: The entry \"" +
                                solution_type_to_string(iter->first) +
                                "\" of the linear solver vector is not (yet) initialized!");
      msg << std::setw(109) << std::setfill('!') << "\n" << std::setfill(' ');
      msg << "!!! " << std::left << std::setw(100) << act_msg << " !!!";
      msg << std::setw(109) << std::setfill('!') << "\n" << std::setfill(' ') << std::right;
      if (Core::Communication::my_mpi_rank(comm_) == 0) std::cout << msg.str() << std::endl;
    }
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::GlobalData::setup()
{
  // set the nonlinear optimization problem type
  nlnparams_->set("Optimization Problem Type", opt_type_);

  // set printing parameters
  set_printing_parameters();

  // construct the nox utils class
  nox_utils_ = Teuchos::make_rcp<::NOX::Utils>(nlnparams_->sublist("Printing"));

  // set (non-linear) solver option parameters
  set_solver_option_parameters();

  // set status test parameters
  set_status_test_parameters();

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::GlobalData::set_printing_parameters()
{
  NOX::Nln::Aux::set_printing_parameters(*nlnparams_, comm_);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::GlobalData::set_solver_option_parameters()
{
  /* NOX gives you the option to use a user defined merit function by inserting
   * a Teuchos::RCP<::NOX::MeritFunction::Generic> pointer into the parameter
   * sublist "Solver Options". Since we are forced to use the
   * NOX_GlobalData class, we have to use this to define our own merit functions
   * by creating the particular class in the NOX::Nln::MeritFunction::Factory
   * and insert it into the parameter list. */
  Teuchos::ParameterList& solverOptionsList = nlnparams_->sublist("Solver Options");

  /* If we do a full newton method, we don't need a merit function and can
   * skip the following */
  const std::string& nlnsolver_name = nlnparams_->get<std::string>("Nonlinear Solver");
  const std::string& ls_method_name = nlnparams_->sublist("Line Search").get<std::string>("Method");
  if (((nlnsolver_name == "Line Search Based") or (nlnsolver_name == "Pseudo Transient")) and
      (ls_method_name != "Full Step"))
  {
    // Pure reading access to the unfinished nox_nln_globaldata class
    mrt_fct_ptr_ = NOX::Nln::MeritFunction::build_merit_function(*this);
  }

  // If the mrtFctPtr is Teuchos::null the default "Sum of Squares" NOX internal
  // merit function is used.
  if (not mrt_fct_ptr_.is_null())
    solverOptionsList.set<Teuchos::RCP<::NOX::MeritFunction::Generic>>(
        "User Defined Merit Function", mrt_fct_ptr_);

  /* NOX has become quite strict about parameter lists. They may only contain parameters,
   * that are then actually used. Hence, we have to remove some.
   *
   * ToDo: Clean this up by not setting unused parameters in the first place.
   */
  {
    // remove "Merit Function" parameter entry after building the user-defined function
    // (NOX does not accept this parameter)
    solverOptionsList.remove("Merit Function", false);

    // also remove *PrintEqualSign*
    solverOptionsList.remove("*PrintEqualSign*", false);
  }

  // Similar to the merit function we provide our own direction factory, if
  // the direction method is set to "User Defined"
  {
    Teuchos::ParameterList& pdir = nlnparams_->sublist("Direction", true);

    // Set the direction method to user-defined if the method is set to Newton,
    // since we want to use our own NOX::Nln::Newton direction method and not the
    // one provided by NOX.
    if (pdir.get<std::string>("Method") == "Newton")
    {
      pdir.set("Method", "User Defined");
      pdir.set("User Defined Method", "Newton");
    }

    // Set the associated factory
    if (pdir.get<std::string>("Method") == "User Defined")
    {
      direction_factory_ = Teuchos::make_rcp<NOX::Nln::Direction::Factory>();
      pdir.set<Teuchos::RCP<::NOX::Direction::UserDefinedFactory>>(
          "User Defined Direction Factory", direction_factory_);
    }
  }

  /* We use the parameter list to define a PrePostOperator class for the
   * non-linear iteration process. */
  pre_post_op_ptr_ = Teuchos::make_rcp<NOX::Nln::Solver::PrePostOp::Generic>();
  NOX::Nln::Aux::add_to_pre_post_op_vector(solverOptionsList, pre_post_op_ptr_);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::GlobalData::set_status_test_parameters()
{
  Teuchos::ParameterList& statusTestParams = nlnparams_->sublist("Status Test", true);

  // check if the status test was already set via the input file
  if (statusTestParams.isSublist("Outer Status Test") and
      statusTestParams.sublist("Outer Status Test").numParams() != 0)
    return;

  // initialize the sublist "Outer Status Test". This one has to be specified.
  Teuchos::ParameterList& outerStatusTestParams =
      statusTestParams.sublist("Outer Status Test", false);

  Teuchos::ParameterList xmlParams;
  auto xmlfilename = statusTestParams.get<std::optional<std::filesystem::path>>("XML File");

  // check the input: path to the "Status Test" xml-file
  if (!xmlfilename)
    FOUR_C_THROW("Please specify a \"Status Test\"->\"XML File\" for the nox nonlinear solver!");

  if (xmlfilename->extension() == ".xml")
  {
    try
    {
      xmlParams = *(Teuchos::getParametersFromXmlFile(xmlfilename->string()));
    }
    catch (const std::runtime_error&)
    {
      FOUR_C_THROW(
          "The \"Status Test\"->\"XML File\" was not found! "
          "Please check the path in your input file! \n"
          "CURRENT PATH = {}",
          xmlfilename->c_str());
    }
  }
  else
    FOUR_C_THROW("The file name '{}' is not a valid XML file name.", xmlfilename->string());

  // copy the "Outer Status Test" into the nox parameter list
  if (not xmlParams.isSublist("Outer Status Test"))
    FOUR_C_THROW("You have to specify a sublist \"Outer Status Test\" in your xml-file.");
  else
    outerStatusTestParams = xmlParams.sublist("Outer Status Test");

  /* If we use a line search based backtracking non-linear solver and an inner-status test
   * list is necessary. We check it during the nox_nln_linesearch_factory call. */
  if (xmlParams.isSublist("Inner Status Test"))
  {
    Teuchos::ParameterList& innerStatusTestParams =
        statusTestParams.sublist("Inner Status Test", false);
    // copy the "Inner Status Test" into the nox parameter list
    if (xmlParams.sublist("Inner Status Test").numParams())
      innerStatusTestParams = xmlParams.sublist("Inner Status Test");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const ::NOX::Utils& NOX::Nln::GlobalData::get_nox_utils() const
{
  if (nox_utils_.is_null()) FOUR_C_THROW("noxUtils_ was not initialized!");

  return *nox_utils_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Teuchos::RCP<::NOX::Utils>& NOX::Nln::GlobalData::get_nox_utils_ptr() const
{
  if (nox_utils_.is_null()) FOUR_C_THROW("noxUtils_ was not initialized!");

  return nox_utils_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Teuchos::ParameterList& NOX::Nln::GlobalData::get_nln_parameter_list() const
{
  if (nlnparams_.is_null()) FOUR_C_THROW("nlnparams_ was not initialized!");

  return *nlnparams_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::ParameterList& NOX::Nln::GlobalData::get_nln_parameter_list()
{
  if (nlnparams_.is_null()) FOUR_C_THROW("nlnparams_ was not initialized!");

  return *nlnparams_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Teuchos::RCP<Teuchos::ParameterList>& NOX::Nln::GlobalData::get_nln_parameter_list_ptr()
{
  if (nlnparams_.is_null()) FOUR_C_THROW("nlnparams_ was not initialized!");

  return nlnparams_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
MPI_Comm NOX::Nln::GlobalData::get_comm() const { return comm_; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const NOX::Nln::LinearSystem::SolverMap& NOX::Nln::GlobalData::get_linear_solvers()
{
  return lin_solvers_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<NOX::Nln::Interface::RequiredBase> NOX::Nln::GlobalData::get_required_interface()
{
  FOUR_C_ASSERT(i_req_ptr_, "Required interface pointer iReqPtr_ was not initialized!");

  return i_req_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Epetra::Interface::Jacobian> NOX::Nln::GlobalData::get_jacobian_interface()
{
  if (i_jac_ptr_.is_null())
    FOUR_C_THROW("Jacobian interface pointer iJacPtr_ was not initialized!");

  return i_jac_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const NOX::Nln::CONSTRAINT::ReqInterfaceMap& NOX::Nln::GlobalData::get_constraint_interfaces()
{
  if (i_constr_.size() == 0) FOUR_C_THROW("The constraint interface map is empty!");

  return i_constr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const NOX::Nln::CONSTRAINT::PrecInterfaceMap& NOX::Nln::GlobalData::get_constraint_prec_interfaces()
{
  if (i_constr_prec_.size() == 0)
    FOUR_C_THROW("The constraint preconditioner interface map is empty!");

  return i_constr_prec_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool NOX::Nln::GlobalData::is_constrained() const { return is_constrained_; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::shared_ptr<NOX::Nln::Scaling>& NOX::Nln::GlobalData::get_scaling_object()
{
  return i_scale_;
}

FOUR_C_NAMESPACE_CLOSE
