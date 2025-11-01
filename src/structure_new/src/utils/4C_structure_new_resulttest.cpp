// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_resulttest.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <optional>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace
{
  struct QuantityNameAndComponent
  {
    std::string name;
    int component;
  };

  [[nodiscard]] std::optional<QuantityNameAndComponent> get_gauss_point_data_name_and_component(
      const std::string& name_with_component,
      const std::unordered_map<std::string, int>& quantities)
  {
    for (const auto& [quantity_name, num_quantities] : quantities)
    {
      if (num_quantities == 1)
      {
        if (name_with_component == quantity_name)
          return std::make_optional<QuantityNameAndComponent>({quantity_name, 0});
        else
          continue;
      }

      for (auto i = 0; i < num_quantities; ++i)
      {
        if (name_with_component == quantity_name + "_" + std::to_string(i + 1))
        {
          return std::make_optional<QuantityNameAndComponent>({quantity_name, i});
        }
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] double get_gauss_point_data_value(
      const QuantityNameAndComponent& name_and_component, int node_id,
      const std::unordered_map<std::string, std::shared_ptr<Core::LinAlg::MultiVector<double>>>&
          all_data)
  {
    const Core::LinAlg::MultiVector<double>& data = *all_data.at(name_and_component.name);

    int local_id = data.get_map().lid(node_id);

    if (local_id < 0)
    {
      FOUR_C_THROW("You tried to test {} on a proc that does not own node {}.",
          name_and_component.name, node_id);
    }

    return data(name_and_component.component).local_values_as_span()[local_id];
  }

  /*!
   * @brief Returns the stress or strain component requested in label at the given node
   *
   * @param prefix (in) : prefix (either stress or strain)
   * @param label (in) : label of the quantity, e.g. stress_xx
   * @param node_id (in) : Id of the node
   * @param nodal_data (in) : Nodal data
   * @return double
   */
  double get_nodal_stress_strain_component(const std::string& prefix, const std::string& label,
      int node_id, const Core::LinAlg::MultiVector<double>& nodal_data)
  {
    int voigt_index = -1;
    if (label == prefix + "_xx")
    {
      voigt_index = Core::LinAlg::Voigt::IndexMappings::symmetric_tensor_to_voigt6_index(0, 0);
    }
    else if (label == prefix + "_yy")
    {
      voigt_index = Core::LinAlg::Voigt::IndexMappings::symmetric_tensor_to_voigt6_index(1, 1);
    }
    else if (label == prefix + "_zz")
    {
      voigt_index = Core::LinAlg::Voigt::IndexMappings::symmetric_tensor_to_voigt6_index(2, 2);
    }
    else if (label == prefix + "_xy")
    {
      voigt_index = Core::LinAlg::Voigt::IndexMappings::symmetric_tensor_to_voigt6_index(0, 1);
    }
    else if (label == prefix + "_xz")
    {
      voigt_index = Core::LinAlg::Voigt::IndexMappings::symmetric_tensor_to_voigt6_index(0, 2);
    }
    else if (label == prefix + "_yz")
    {
      voigt_index = Core::LinAlg::Voigt::IndexMappings::symmetric_tensor_to_voigt6_index(1, 2);
    }

    if (voigt_index < 0)
    {
      FOUR_C_THROW(
          "You try to test an unknown {} component {}. Use one of {}<_xx, _yy, _zz, _xy, _xz, "
          "_yz>.",
          label.c_str(), prefix.c_str(), prefix.c_str());
    }

    int local_id = nodal_data.get_map().lid(node_id);

    if (local_id < 0)
    {
      FOUR_C_THROW("You tried to test {} on a proc that does not own node {}.", label, node_id);
    }

    return nodal_data(voigt_index).local_values_as_span()[local_id];
  }
}  // namespace

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ResultTest::ResultTest()
    : Core::Utils::ResultTest("STRUCTURE"),
      isinit_(false),
      issetup_(false),
      strudisc_(nullptr),
      disn_(nullptr),
      veln_(nullptr),
      accn_(nullptr),
      reactn_(nullptr),
      gstate_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ResultTest::init(
    const Solid::TimeInt::BaseDataGlobalState& gstate, const Solid::ModelEvaluator::Data& data)
{
  issetup_ = false;

  disn_ = gstate.get_dis_n();
  veln_ = gstate.get_vel_n();
  accn_ = gstate.get_acc_n();
  reactn_ = gstate.get_freact_n();
  gstate_ = Core::Utils::shared_ptr_from_ref(gstate);
  data_ = Core::Utils::shared_ptr_from_ref(data);
  strudisc_ = gstate.get_discret();

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ResultTest::setup()
{
  check_init();
  // currently unused
  issetup_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Solid::ResultTest::test_node(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  check_init_setup();

  // care for the case of multiple discretizations of the same field type
  std::string dis = container.get<std::string>("DIS");
  if (dis != strudisc_->name()) return;

  int node = container.get<int>("NODE");
  node -= 1;

  int havenode(strudisc_->have_global_node(node));
  int isnodeofanybody(0);
  isnodeofanybody = Core::Communication::sum_all(havenode, strudisc_->get_comm());

  if (isnodeofanybody == 0)
  {
    FOUR_C_THROW("Node {} does not belong to discretization {}", node + 1, strudisc_->name());
  }

  std::string position = container.get<std::string>("QUANTITY");

  if (strudisc_->have_global_node(node))
  {
    double result;
    int error_code = get_nodal_result(result, node, position);

    if (error_code == 0)
    {
      // compare values
      const int err = compare_values(result, "NODE", container);
      nerr += err;
      test_count++;
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Solid::ResultTest::get_nodal_result(
    double& result, const int node, const std::string& position) const
{
  result = 0.0;

  const Core::Nodes::Node* actnode = strudisc_->g_node(node);

  // Here we are just interested in the nodes that we own (i.e. a row node)!
  if (actnode->owner() != Core::Communication::my_mpi_rank(strudisc_->get_comm())) return -1;

  bool unknownpos = true;  // make sure the result value std::string can be handled

  // test displacements or pressure
  if (disn_ != nullptr)
  {
    const Core::LinAlg::Map& disnpmap = disn_->get_map();
    int idx = -1;
    if (position == "dispx")
      idx = 0;
    else if (position == "dispy")
      idx = 1;
    else if (position == "dispz")
      idx = 2;
    else if (position == "press")
      idx = 3;

    if (idx >= 0)
    {
      unknownpos = false;
      int lid = disnpmap.lid(strudisc_->dof(0, actnode, idx));
      if (lid < 0)
        FOUR_C_THROW(
            "You tried to test {} on nonexistent dof {} on node {}", position, idx, actnode->id());
      result = disn_->local_values_as_span()[lid];
    }
  }

  // test velocities
  if (veln_ != nullptr)
  {
    const Core::LinAlg::Map& velnpmap = veln_->get_map();
    int idx = -1;
    if (position == "velx")
      idx = 0;
    else if (position == "vely")
      idx = 1;
    else if (position == "velz")
      idx = 2;

    if (idx >= 0)
    {
      unknownpos = false;
      int lid = velnpmap.lid(strudisc_->dof(0, actnode, idx));
      if (lid < 0)
        FOUR_C_THROW(
            "You tried to test {} on nonexistent dof {} on node {}", position, idx, actnode->id());
      result = veln_->local_values_as_span()[lid];
    }
  }

  // test accelerations
  if (accn_ != nullptr)
  {
    const Core::LinAlg::Map& accnpmap = accn_->get_map();
    int idx = -1;
    if (position == "accx")
      idx = 0;
    else if (position == "accy")
      idx = 1;
    else if (position == "accz")
      idx = 2;

    if (idx >= 0)
    {
      unknownpos = false;
      int lid = accnpmap.lid(strudisc_->dof(0, actnode, idx));
      if (lid < 0)
        FOUR_C_THROW(
            "You tried to test {} on nonexistent dof {} on node {}", position, idx, actnode->id());
      result = accn_->local_values_as_span()[lid];
    }
  }

  // test nodal stresses
  if (position.rfind("stress", 0) == 0)
  {
    if (data_->get_stress_data_node_postprocessed() == nullptr)
    {
      FOUR_C_THROW(
          "It looks like you don't write stresses. You have to specify the stress type in "
          "IO->STRUCT_STRESS");
    }
    result = get_nodal_stress_strain_component(
        "stress", position, node, *data_->get_stress_data_node_postprocessed());
    unknownpos = false;
  }

  // test nodal strain
  if (position.rfind("strain", 0) == 0)
  {
    if (data_->get_stress_data_node_postprocessed() == nullptr)
    {
      FOUR_C_THROW(
          "It looks like you don't write strains. You have to specify the strain type in "
          "IO->STRUCT_STRAIN");
    }
    result = get_nodal_stress_strain_component(
        "strain", position, node, *data_->get_strain_data_node_postprocessed());
    unknownpos = false;
  }

  // test for any postprocessed gauss point data
  if (data_->get_gauss_point_data_output_manager_ptr() != nullptr)
  {
    std::optional<QuantityNameAndComponent> name_and_component =
        get_gauss_point_data_name_and_component(
            position, data_->get_gauss_point_data_output_manager_ptr()->get_quantities());
    if (name_and_component.has_value())
    {
      result = get_gauss_point_data_value(*name_and_component, node,
          data_->get_gauss_point_data_output_manager_ptr()->get_nodal_data());
      unknownpos = false;
    }
  }

  // test reaction
  if (reactn_ != nullptr)
  {
    const Core::LinAlg::Map& reactmap = reactn_->get_map();
    int idx = -1;
    if (position == "reactx")
      idx = 0;
    else if (position == "reacty")
      idx = 1;
    else if (position == "reactz")
      idx = 2;

    if (idx >= 0)
    {
      unknownpos = false;
      int lid = reactmap.lid(strudisc_->dof(0, actnode, idx));
      if (lid < 0)
        FOUR_C_THROW(
            "You tried to test {} on nonexistent dof {} on node {}", position, idx, actnode->id());
      result = reactn_->local_values_as_span()[lid];
    }
  }

  // catch position std::strings, which are not handled by structure result test
  if (unknownpos) FOUR_C_THROW("Quantity '{}' not supported in structure testing", position);

  return 0;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Solid::ResultTest::test_node_on_geometry(const Core::IO::InputParameterContainer& container,
    int& nerr, int& test_count, const std::vector<std::vector<std::vector<int>>>& nodeset)
{
  check_init_setup();

  // care for the case of multiple discretizations of the same field type
  std::string dis = container.get<std::string>("DIS");
  if (dis != strudisc_->name()) return;

  auto get_geometry_info = [container]() -> std::tuple<int, int>
  {
    int geometry_type = 0, geometry_id = 0;
    if (container.get_if<int>("NODE") != nullptr)
    {
      geometry_id = container.get<int>("NODE");
      geometry_id -= 1;
      geometry_type = 0;
    }
    else if (container.get_if<int>("LINE") != nullptr)
    {
      geometry_id = container.get<int>("LINE");
      geometry_id -= 1;
      geometry_type = 1;
    }
    else if (container.get_if<int>("SURFACE") != nullptr)
    {
      geometry_id = container.get<int>("SURFACE");
      geometry_id -= 1;
      geometry_type = 2;
    }
    else if (container.get_if<int>("VOLUME") != nullptr)
    {
      geometry_id = container.get<int>("VOLUME");
      geometry_id -= 1;
      geometry_type = 3;
    }
    else
      FOUR_C_THROW("Invalid input parameter container found");

    return {geometry_type, geometry_id};
  };
  const auto [geometry_type, geometry_id] = get_geometry_info();

  FOUR_C_ASSERT(geometry_type >= 0 && geometry_type <= 3, "The geometry type is invalid");
  if (geometry_id < 0 || geometry_id >= static_cast<int>(nodeset[geometry_type].size()))
    FOUR_C_THROW("Invalid geometry id {}", geometry_id);

  const std::vector<int>& nodes = nodeset[geometry_type][geometry_id];

  TestOp op = container.get<TestOp>("OP");
  std::string position = container.get<std::string>("QUANTITY");

  // collect the local result
  double tmp_result;
  switch (op)
  {
    case TestOp::max:
      tmp_result = -1e99;
      break;
    case TestOp::min:
      tmp_result = 1e99;
      break;
    default:
      tmp_result = 0.0;
      break;
  }
  for (int node : nodes)
  {
    double tmp = 0.0;
    if (strudisc_->have_global_node(node)) get_nodal_result(tmp, node, position);

    switch (op)
    {
      case TestOp::sum:
        tmp_result += tmp;
        break;
      case TestOp::max:
        if (tmp > tmp_result) tmp_result = tmp;
        break;
      case TestOp::min:
        if (tmp < tmp_result) tmp_result = tmp;
        break;
      default:
        break;
    }
  }

  // gather the result across processes
  auto gather_result = [op](const Core::FE::Discretization& disc,
                           const double local_result) -> double
  {
    double tmp_result = local_result, result = 0.0;
    switch (op)
    {
      case TestOp::sum:
        result = Core::Communication::sum_all(tmp_result, disc.get_comm());
        break;
      case TestOp::max:
        result = Core::Communication::max_all(tmp_result, disc.get_comm());
        break;
      case TestOp::min:
        result = Core::Communication::min_all(tmp_result, disc.get_comm());
        break;
      default:
        break;
    }
    return result;
  };
  const double result = gather_result(*strudisc_, tmp_result);

  // test the value; we only do at rank 0
  if (Core::Communication::my_mpi_rank(strudisc_->get_comm()) == 0)
  {
    int err = 0;
    switch (geometry_type)
    {
      case 0:
        err = compare_values(result, "NODE", container);
        break;
      case 1:
        err = compare_values(result, "LINE", container);
        break;
      case 2:
        err = compare_values(result, "SURFACE", container);
        break;
      case 3:
        err = compare_values(result, "VOLUME", container);
        break;
      default:
        break;
    }
    nerr += err;
    test_count++;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ResultTest::test_special(const Core::IO::InputParameterContainer& container, int& nerr,
    int& test_count, int& uneval_test_count)
{
  check_init_setup();

  std::string quantity = container.get<std::string>("QUANTITY");

  Status special_status = Status::unevaluated;
  const std::optional<double> result = get_special_result(quantity, special_status);

  switch (special_status)
  {
    case Status::evaluated:
    {
      if (result.has_value())
        nerr += compare_values(*result, "SPECIAL", container);
      else
        FOUR_C_THROW(
            "Solid::ResultTest::test_special: Special result has no defined value assigned to it!");
      ++test_count;
      break;
    }
    case Status::unevaluated:
    {
      ++uneval_test_count;
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "Solid::ResultTest::test_special: Undefined status type (enum={})!", special_status);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::optional<double> Solid::ResultTest::get_special_result(
    const std::string& quantity, Status& special_status) const
{
  if (quantity.find("num_iter_step_") != quantity.npos)
  {
    return get_nln_iteration_number(quantity, special_status);
  }
  else if (quantity.find("lin_iter_step_") != quantity.npos)
  {
    return get_last_lin_iteration_number(quantity, special_status);
  }
  else if (quantity.find("nodes_proc") != quantity.npos)
  {
    return get_nodes_per_proc_number(quantity, special_status);
  }
  else if (quantity == "internal_energy" or quantity == "kinetic_energy" or
           quantity == "total_energy" or quantity == "beam_contact_penalty_potential" or
           quantity == "beam_interaction_potential" or
           quantity == "beam_to_beam_link_internal_energy" or
           quantity == "beam_to_beam_link_kinetic_energy" or
           quantity == "beam_to_sphere_link_internal_energy" or
           quantity == "beam_to_sphere_link_kinetic_energy")
  {
    return get_energy(quantity, special_status);
  }
  else
    FOUR_C_THROW(
        "Quantity '{}' not supported by special result testing functionality "
        "for structure field!",
        quantity.c_str());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::optional<int> Solid::ResultTest::get_last_lin_iteration_number(
    const std::string& quantity, Status& special_status) const
{
  std::optional<int> result = std::nullopt;

  if (Core::Communication::my_mpi_rank(strudisc_->get_comm()) == 0)
  {
    const int stepn = get_integer_number_at_last_position_of_name(quantity);

    const int restart = Global::Problem::instance()->restart();
    if (stepn <= restart) return -1;

    special_status = Status::evaluated;
    result = gstate_->get_last_lin_iteration_number(stepn);
  }

  return result;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::optional<int> Solid::ResultTest::get_nln_iteration_number(
    const std::string& quantity, Status& special_status) const
{
  std::optional<int> result = std::nullopt;

  if (Core::Communication::my_mpi_rank(strudisc_->get_comm()) == 0)
  {
    const int stepn = get_integer_number_at_last_position_of_name(quantity);

    const int restart = Global::Problem::instance()->restart();
    if (stepn <= restart) return -1;

    special_status = Status::evaluated;
    result = gstate_->get_nln_iteration_number(stepn);
  }

  return result;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::optional<int> Solid::ResultTest::get_nodes_per_proc_number(
    const std::string& quantity, Status& special_status) const
{
  std::optional<int> result = std::nullopt;

  std::string proc_string = quantity.substr(quantity.find("nodes_proc") + 10);
  const int proc_num = std::stoi(proc_string);

  // extract processor ID
  if (proc_num >= Core::Communication::num_mpi_ranks(strudisc_->get_comm()))
    FOUR_C_THROW("Solid::ResultTest::get_nodes_per_proc_number: Invalid processor ID!");

  if (Core::Communication::my_mpi_rank(strudisc_->get_comm()) == proc_num)
  {
    // extract number of nodes owned by specified processor
    special_status = Status::evaluated;
    result = strudisc_->node_row_map()->num_my_elements();
  }

  return result;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::optional<double> Solid::ResultTest::get_energy(
    const std::string& quantity, Status& special_status) const
{
  std::optional<double> result = std::nullopt;

  if (Core::Communication::my_mpi_rank(strudisc_->get_comm()) == 0)
  {
    special_status = Status::evaluated;
    result = data_->get_energy_data(quantity);
  }

  return result;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::get_integer_number_at_last_position_of_name(const std::string& quantity)
{
  std::stringstream ss(quantity);
  std::string s;

  std::vector<std::string> split_strings;
  while (std::getline(ss, s, '_')) split_strings.push_back(s);

  try
  {
    return std::stoi(split_strings.back());
  }
  catch (const std::invalid_argument& e)
  {
    FOUR_C_THROW(
        "You provided the wrong format. The integer number must be "
        "at the very last position of the name, separated by an underscore. "
        "The correct format is:\n"
        "\"<prefix_name>_<number>\"");
  }
}

FOUR_C_NAMESPACE_CLOSE
