// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_discretization_helpers.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_builder.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_rebalance.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <array>
#include <memory>

FOUR_C_NAMESPACE_OPEN

Discret::Elements::ReducedLungLineType Discret::Elements::ReducedLungLineType::instance_;

Discret::Elements::ReducedLungLineType& Discret::Elements::ReducedLungLineType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::ReducedLungLineType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::ReducedLungLine(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::ReducedLungLineType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  (void)eledistype;
  if (eletype == "REDUCED_LUNG_LINE")
  {
    return std::make_shared<Discret::Elements::ReducedLungLine>(id, owner);
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::ReducedLungLineType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::ReducedLungLine>(id, owner);
}

void Discret::Elements::ReducedLungLineType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = dwele->num_dof_per_node(*(dwele->nodes()[0]));
  dimns = numdf;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::ReducedLungLineType::compute_null_space(
    Core::Nodes::Node& node, std::span<const double> x0, const int numdof)
{
  (void)node;
  (void)x0;
  if (numdof != 1)
  {
    FOUR_C_THROW("ReducedLungLine expects a single dof per node, got {}.", numdof);
  }

  Core::LinAlg::SerialDenseMatrix nullspace(1, 1, true);
  nullspace(0, 0) = 1.0;
  return nullspace;
}

void Discret::Elements::ReducedLungLineType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["REDUCED_LUNG_LINE"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::line2] = all_of({});
}

Discret::Elements::ReducedLungLine::ReducedLungLine(int id, int owner)
    : Core::Elements::Element(id, owner)
{
}

Discret::Elements::ReducedLungLine::ReducedLungLine(const Discret::Elements::ReducedLungLine& old)
    : Core::Elements::Element(old)
{
}

Core::Elements::Element* Discret::Elements::ReducedLungLine::clone() const
{
  return new Discret::Elements::ReducedLungLine(*this);
}

Core::FE::CellType Discret::Elements::ReducedLungLine::shape() const
{
  if (num_node() == 2)
  {
    return Core::FE::CellType::line2;
  }
  FOUR_C_THROW("Unexpected number of nodes {} for ReducedLungLine.", num_node());
}

int Discret::Elements::ReducedLungLine::num_line() const
{
  if (num_node() == 2)
  {
    return 1;
  }
  FOUR_C_THROW("Could not determine number of lines for ReducedLungLine.");
}

std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ReducedLungLine::lines()
{
  FOUR_C_ASSERT(num_line() == 1, "ReducedLungLine must have only one line");
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

int Discret::Elements::ReducedLungLine::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  (void)params;
  (void)discretization;
  (void)condition;
  (void)lm;
  (void)elevec1;
  (void)elemat1;
  FOUR_C_THROW("ReducedLungLine does not support Neumann evaluations.");
  return 0;
}

void Discret::Elements::ReducedLungLine::print(std::ostream& os) const
{
  Core::Elements::Element::print(os);
}

bool Discret::Elements::ReducedLungLine::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  (void)eletype;
  (void)distype;
  (void)container;
  return true;
}

namespace ReducedLung
{
  void build_discretization_from_topology(Core::FE::Discretization& discretization,
      const ReducedLungParameters::LungTree::Topology& topology,
      const Core::Rebalance::RebalanceParameters& rebalance_parameters)
  {
    Core::FE::DiscretizationBuilder<3> builder;

    const int my_rank = Core::Communication::my_mpi_rank(discretization.get_comm());
    if (my_rank == 0)
    {
      if (topology.num_nodes <= 0)
      {
        FOUR_C_THROW("Topology num_nodes must be positive, got {}.", topology.num_nodes);
      }
      if (topology.num_elements <= 0)
      {
        FOUR_C_THROW("Topology num_elements must be positive, got {}.", topology.num_elements);
      }

      for (int node_id = 0; node_id < topology.num_nodes; ++node_id)
      {
        const auto coords = topology.node_coordinates.at(node_id, "node_coordinates");
        if (coords.size() != 3u)
        {
          FOUR_C_THROW("Topology node_coordinates entry {} must have 3 components, got {}.",
              node_id + 1, coords.size());
        }
        const std::array<double, 3> coord_array{coords[0], coords[1], coords[2]};
        builder.add_node(coord_array, node_id, nullptr);
      }

      for (int element_id = 0; element_id < topology.num_elements; ++element_id)
      {
        const auto nodes = topology.element_nodes.at(element_id, "element_nodes");
        if (nodes.size() != 2u)
        {
          FOUR_C_THROW("Topology element_nodes entry {} must have 2 entries, got {}.",
              element_id + 1, nodes.size());
        }
        if (nodes[0] < 1 || nodes[1] < 1)
        {
          FOUR_C_THROW(
              "Topology element_nodes entry {} must use 1-based node ids.", element_id + 1);
        }

        const int node_in = nodes[0] - 1;
        const int node_out = nodes[1] - 1;
        if (node_in >= topology.num_nodes || node_out >= topology.num_nodes)
        {
          FOUR_C_THROW("Topology element_nodes entry {} references node ids outside [1, {}].",
              element_id + 1, topology.num_nodes);
        }
        if (node_in == node_out)
        {
          FOUR_C_THROW(
              "Topology element_nodes entry {} uses identical in/out node ids.", element_id + 1);
        }

        const std::array<int, 2> node_ids{node_in, node_out};
        auto element = std::make_shared<Discret::Elements::ReducedLungLine>(element_id, my_rank);
        element->set_node_ids(static_cast<int>(node_ids.size()), node_ids.data());
        builder.add_element(Core::FE::CellType::line2, node_ids, element_id, element);
      }
    }

    builder.build(discretization, rebalance_parameters);
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
