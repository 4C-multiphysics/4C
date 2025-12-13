// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_global_data.hpp"

#include "4C_comm_utils.hpp"
#include "4C_contact_constitutivelaw_bundle.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_global_legacy_module_problem_type_string.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_rebalance_graph_based.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Teuchos_ParameterListExceptions.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Global::Problem* Global::Problem::instance(int num)
{
  static auto singleton_map =
      Core::Utils::make_singleton_map<int>([]() { return std::unique_ptr<Problem>(new Problem); });

  return singleton_map[num].instance(Core::Utils::SingletonAction::create);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Global::Problem::Problem()
    : probtype_(Core::ProblemType::none), restartstep_(0), communicators_(nullptr)
{
  materials_ = std::make_shared<Mat::PAR::Bundle>();
  contactconstitutivelaws_ = std::make_shared<CONTACT::CONSTITUTIVELAW::Bundle>();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::ProblemType Global::Problem::get_problem_type() const { return probtype_; }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::string Global::Problem::problem_name() const
{
  std::map<std::string, Core::ProblemType> map = Core::string_to_problem_type_map();
  std::map<std::string, Core::ProblemType>::const_iterator i;

  for (i = map.begin(); i != map.end(); ++i)
  {
    if (i->second == probtype_) return i->first;
  }
  FOUR_C_THROW("Could not determine valid problem name");
  return "Undefined";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Global::Problem::restart() const { return restartstep_; }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Global::Problem::n_dim() const
{
  const Teuchos::ParameterList& sizeparams = problem_size_params();
  return sizeparams.get<int>("DIM");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Teuchos::ParameterList& Global::Problem::solver_params(int solverNr) const
{
  std::stringstream ss;
  ss << "SOLVER " << solverNr;
  return parameters_->sublist(ss.str());
}

std::function<const Teuchos::ParameterList&(int)> Global::Problem::solver_params_callback() const
{
  return [this](int id) -> const Teuchos::ParameterList& { return this->solver_params(id); };
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::Problem::set_communicators(Core::Communication::Communicators communicators)
{
  if (communicators_ != nullptr) FOUR_C_THROW("Communicators were already set.");
  communicators_ = std::make_unique<Core::Communication::Communicators>(std::move(communicators));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::Communication::Communicators& Global::Problem::get_communicators() const
{
  if (communicators_ == nullptr) FOUR_C_THROW("No communicators allocated yet.");
  return *communicators_;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::Problem::open_control_file(MPI_Comm comm, const std::string& inputfile,
    std::string prefix, const std::string& restartkenner)
{
  if (restartstep_ != 0)
  {
    inputcontrol_ = std::make_shared<Core::IO::InputControl>(restartkenner, comm);

    if (restartstep_ < 0)
    {
      int r = Core::IO::get_last_possible_restart_step(*inputcontrol_);
      set_restart_step(r);
    }
  }

  outputcontrol_ = std::make_shared<Core::IO::OutputControl>(comm, problem_name(),
      spatial_approximation_type(), inputfile, restartkenner, std::move(prefix), n_dim(), restart(),
      io_params().get<int>("FILESTEPS"), io_params().get<bool>("OUTPUT_BIN"), true);

  if (!io_params().get<bool>("OUTPUT_BIN") && Core::Communication::my_mpi_rank(comm) == 0)
  {
    Core::IO::cout << "==================================================\n"
                   << "=== WARNING: No binary output will be written. ===\n"
                   << "==================================================\n"
                   << Core::IO::endl;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::Problem::set_parameter_list(
    std::shared_ptr<Teuchos::ParameterList> const& parameter_list)
{
  parameters_ = parameter_list;
}


std::shared_ptr<const Teuchos::ParameterList> Global::Problem::get_parameter_list() const
{
  return parameters_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::Problem::add_dis(
    const std::string& name, std::shared_ptr<Core::FE::Discretization> dis)
{
  // safety checks
  if (dis == nullptr) FOUR_C_THROW("Received nullptr.");
  if (dis->name().empty()) FOUR_C_THROW("discretization has empty name string.");

  if (!discretizationmap_.insert(std::make_pair(name, dis)).second)
  {
    // if the same key already exists we have to inform the user since
    // the insert statement did not work in this case
    FOUR_C_THROW(
        "Could not insert discretization '{}' under (duplicate) key '{}'.", dis->name(), name);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Discretization> Global::Problem::get_dis(const std::string& name) const
{
  auto iter = discretizationmap_.find(name);

  if (iter != discretizationmap_.end())
  {
    return iter->second;
  }
  else
  {
    FOUR_C_THROW("Could not find discretization '{}'.", name);
    return nullptr;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<std::string> Global::Problem::get_dis_names() const
{
  unsigned mysize = num_fields();
  std::vector<std::string> vec;
  vec.reserve(mysize);

  std::transform(discretizationmap_.begin(), discretizationmap_.end(), std::back_inserter(vec),
      [](const auto& key_value) { return key_value.first; });

  return vec;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Global::Problem::does_exist_dis(const std::string& name) const
{
  auto iter = discretizationmap_.find(name);
  return iter != discretizationmap_.end();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::Problem::set_restart_step(int r) { restartstep_ = r; }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::Problem::set_problem_type(Core::ProblemType targettype) { probtype_ = targettype; }


void Global::Problem::set_function_manager(Core::Utils::FunctionManager&& function_manager_in)
{
  functionmanager_ = std::move(function_manager_in);
}

void Global::Problem::set_spatial_approximation_type(
    Core::FE::ShapeFunctionType shape_function_type)
{
  shapefuntype_ = shape_function_type;
}

FOUR_C_NAMESPACE_CLOSE
