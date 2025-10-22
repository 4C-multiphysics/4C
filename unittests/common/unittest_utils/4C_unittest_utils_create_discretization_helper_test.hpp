// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UNITTEST_UTILS_CREATE_DISCRETIZATION_HELPER_TEST_HPP
#define FOUR_C_UNITTEST_UTILS_CREATE_DISCRETIZATION_HELPER_TEST_HPP

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_builder.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_rebalance.hpp"
#include "4C_rebalance_graph_based.hpp"

#include <mpi.h>
#include <Teuchos_ParameterList.hpp>

#include <array>

namespace TESTING
{
  using namespace FourC;

  class PureGeometryElementType : public Core::Elements::ElementType
  {
   public:
    std::string name() const override { return "PureGeometryElementType"; }

    static PureGeometryElementType& instance()
    {
      static PureGeometryElementType instance;
      return instance;
    }

    std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;



    void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override
    {
      FOUR_C_THROW("Not implemented.");
    }

    Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, std::span<const double> x0, const int numdof) override
    {
      FOUR_C_THROW("Not implemented.");
    }
  };

  /**
   * A minimal element implementation that only stores the cell type and has access to the
   * nodes stored in the base class.
   */
  class PureGeometryElement : public Core::Elements::Element
  {
   public:
    struct Data
    {
      Core::FE::CellType cell_type;
      int num_dof_per_node;
      int num_dof_per_element = 0;
    };

    PureGeometryElement(int id, int owner, Data data)
        : Core::Elements::Element(id, owner), data_(data)
    {
    }

    void pack(Core::Communication::PackBuffer& data) const override
    {
      int type = unique_par_object_id();
      add_to_pack(data, type);

      Element::pack(data);

      data.add_to_pack(data_.cell_type);
      data.add_to_pack(data_.num_dof_per_node);
      data.add_to_pack(data_.num_dof_per_element);
    }

    void unpack(Core::Communication::UnpackBuffer& buffer) override
    {
      extract_and_assert_id(buffer, unique_par_object_id());
      Element::unpack(buffer);
      buffer.extract_from_pack(data_.cell_type);
      buffer.extract_from_pack(data_.num_dof_per_node);
      buffer.extract_from_pack(data_.num_dof_per_element);
    }

    Element* clone() const override { FOUR_C_THROW("Not implemented."); }

    int unique_par_object_id() const override { return element_type().unique_par_object_id(); }

    Core::Elements::ElementType& element_type() const override
    {
      return PureGeometryElementType::instance();
    }

    Core::FE::CellType shape() const override { return data_.cell_type; }
    int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        const Core::Conditions::Condition& condition, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseMatrix* elemat1) override
    {
      FOUR_C_THROW("Not implemented.");
    }

    int num_dof_per_node(const Core::Nodes::Node& node) const override
    {
      return data_.num_dof_per_node;
    }

    int num_dof_per_element() const override { return data_.num_dof_per_element; }

   private:
    Data data_;
  };

  inline std::shared_ptr<Core::Elements::Element> PureGeometryElementType::create(
      const int id, const int owner)
  {
    return std::make_shared<PureGeometryElement>(id, owner, PureGeometryElement::Data{});
  }

  inline Core::Communication::ParObject* PureGeometryElementType::create(
      Core::Communication::UnpackBuffer& buffer)
  {
    auto* object = new PureGeometryElement(-1, -1, PureGeometryElement::Data{});
    object->unpack(buffer);
    return object;
  }

  /**
   * Fill the given @p discretization with a hypercube mesh. A total of `subdivisions^3` elements
   * are created and partitioned among all processes in @p comm.
   * */
  inline void fill_discretization_hyper_cube(
      Core::FE::Discretization& discretization, int subdivisions, MPI_Comm comm)
  {
    discretization.clear_discret();

    const int my_rank = Core::Communication::my_mpi_rank(comm);
    const int total_ranks = Core::Communication::num_mpi_ranks(comm);

    [[maybe_unused]] const int total_elements = subdivisions * subdivisions * subdivisions;
    // function to convert indices into a node lid
    const auto lid = [subdivisions](int i, int j, int k)
    { return i * (subdivisions + 1) * (subdivisions + 1) + j * (subdivisions + 1) + k; };

    const double imbalance_tol(1.1);

    Core::Rebalance::RebalanceParameters rebalance_parameters;
    rebalance_parameters.mesh_partitioning_parameters.min_ele_per_proc = total_ranks;
    rebalance_parameters.mesh_partitioning_parameters.imbalance_tol = imbalance_tol;
    rebalance_parameters.mesh_partitioning_parameters.rebalance_type =
        Core::Rebalance::RebalanceType::hypergraph;

    Core::FE::DiscretizationBuilder<3> builder;

    // Dummy call to ensure the element type is registered
    PureGeometryElementType::instance();

    if (my_rank == 0)
    {
      // Connect the nodes into elements on the owning ranks
      for (int i = 0; i < subdivisions; ++i)
      {
        for (int j = 0; j < subdivisions; ++j)
        {
          for (int k = 0; k < subdivisions; ++k)
          {
            const int ele_id = i * subdivisions * subdivisions + j * subdivisions + k;

            const std::array nodeids = {lid(i, j, k), lid(i + 1, j, k), lid(i + 1, j + 1, k),
                lid(i, j + 1, k), lid(i, j, k + 1), lid(i + 1, j, k + 1), lid(i + 1, j + 1, k + 1),
                lid(i, j + 1, k + 1)};


            auto ele = std::make_unique<PureGeometryElement>(ele_id, my_rank,
                PureGeometryElement::Data{
                    .cell_type = Core::FE::CellType::hex8, .num_dof_per_node = 3});
            ele->set_node_ids(8, nodeids.begin());

            builder.add_element(Core::FE::CellType::hex8, nodeids, ele_id, std::move(ele));
          }
        }
      }

      const double increment = 1.0 / subdivisions;
      // Add all nodes of the partitioned hypercube
      for (int i = 0; i < subdivisions + 1; ++i)
      {
        for (int j = 0; j < subdivisions + 1; ++j)
        {
          for (int k = 0; k < subdivisions + 1; ++k)
          {
            const std::array<double, 3> coords = {i * increment, j * increment, k * increment};
            builder.add_node(coords, lid(i, j, k), nullptr);
          }
        }
      }
    }
    builder.build(discretization, rebalance_parameters);
    discretization.fill_complete();

    FOUR_C_ASSERT(discretization.num_global_elements() == total_elements, "Internal error.");
  }


  /**
   * Fill the given @p discretization with the cells and points from the given @p mesh with @p
   * PureGeometryElements.
   */
  inline void fill_discretization_from_mesh(
      Core::FE::Discretization& dis, const FourC::Core::IO::MeshInput::Mesh<3>& mesh)
  {
    // Create discretization from mesh and redistribute
    std::map<int, std::shared_ptr<Core::Elements::Element>> user_elements;
    int cell_id = 0;
    for (const auto& [_, block] : mesh.cell_blocks())
    {
      for (const auto& cell : block.cells())
      {
        auto e = std::make_shared<TESTING::PureGeometryElement>(cell_id, 0,
            TESTING::PureGeometryElement::Data{
                .cell_type = block.cell_type, .num_dof_per_node = 3});
        e->set_node_ids(cell.size(), cell.data());
        user_elements[cell_id] = e;

        cell_id++;
      }
    }
    Core::Rebalance::RebalanceParameters rebalance_parameters;

    dis.fill_from_mesh(mesh, user_elements, {}, rebalance_parameters);
  }


  /**
   * Create a tree of 1D line elements with a total of `2^levels - 1` elements.
   */
  inline void fill_tree_1d_lines(
      Core::FE::Discretization& discretization, unsigned levels, MPI_Comm comm)
  {
    FOUR_C_ASSERT(levels > 0, "Invalid number of levels.");
    discretization.clear_discret();

    const int my_rank = Core::Communication::my_mpi_rank(comm);
    const int total_ranks = Core::Communication::num_mpi_ranks(comm);

    const int n_total_elements = std::pow(2, levels) - 1;

    // Create a map for all elements with some initial distribution
    auto row_elements =
        std::make_shared<Core::LinAlg::Map>(n_total_elements, 0, discretization.get_comm());

    const auto leaf_on_level = [&](unsigned level, unsigned ele_on_level) -> int
    { return std::pow(2, level) + ele_on_level; };

    // Add the root element
    if (row_elements->lid(0) != -1)
    {
      PureGeometryElementType::instance();
      auto ele = std::make_unique<PureGeometryElement>(0, my_rank,
          PureGeometryElement::Data{.cell_type = Core::FE::CellType::line2, .num_dof_per_node = 1});
      const std::array nodeids{0, 1};
      ele->set_node_ids(2, nodeids.data());
      discretization.add_element(std::move(ele));
    }

    int ele_id = 1;
    for (unsigned level = 1; level < levels; ++level)
    {
      const unsigned n_elements = std::pow(2, level);
      for (unsigned ele_on_level = 0; ele_on_level < n_elements; ++ele_on_level)
      {
        if (row_elements->lid(ele_id) != -1)
        {
          const unsigned parent_ele = ele_on_level / 2;
          const std::array nodeids{
              leaf_on_level(level - 1, parent_ele), leaf_on_level(level, ele_on_level)};

          auto ele = std::make_unique<PureGeometryElement>(ele_id, my_rank,
              PureGeometryElement::Data{
                  .cell_type = Core::FE::CellType::line2, .num_dof_per_node = 1});
          ele->set_node_ids(2, nodeids.data());

          discretization.add_element(std::move(ele));
        }
        ++ele_id;
      }
    }

    auto graph = Core::Rebalance::build_graph(discretization, *row_elements);

    const double imbalance_tol(1.1);
    Teuchos::ParameterList rebalanceParams;
    rebalanceParams.set<std::string>("num_global_parts", std::to_string(total_ranks));
    rebalanceParams.set<std::string>("imbalance_tolerance", std::to_string(imbalance_tol));

    std::shared_ptr<Core::LinAlg::Map> col_elements;
    const auto [row_nodes, col_nodes] =
        Core::Rebalance::rebalance_node_maps(*graph, rebalanceParams);

    std::tie(row_elements, col_elements) =
        discretization.build_element_row_column(*row_nodes, *col_nodes);

    discretization.export_row_elements(*row_elements);
    discretization.export_column_elements(*col_elements);

    // Now create the nodes on the owning ranks
    {
      if (row_nodes->lid(0) != -1)
      {
        const std::array coords = {0.0, 0.0, 0.0};
        discretization.add_node(coords, 0, nullptr);
      }

      const double y_inc = 1.0;
      const double x_inc = 1.0;
      for (unsigned level = 0; level < levels; ++level)
      {
        const unsigned n_elements = std::pow(2, level);
        for (unsigned ele_on_level = 0; ele_on_level < n_elements; ++ele_on_level)
        {
          const int node_id = leaf_on_level(level, ele_on_level);
          if (row_nodes->lid(node_id) != -1)
          {
            const std::array coords = {ele_on_level * x_inc, level * y_inc, 0.0};
            discretization.add_node(coords, node_id, nullptr);
          }
        }
      }
    }

    discretization.export_column_nodes(*col_nodes);
    discretization.fill_complete();

    FOUR_C_ASSERT(discretization.num_global_elements() == n_total_elements, "Internal error.");
  }

  inline void fill_single_tet(Core::FE::Discretization& discretization)
  {
    if (Core::Communication::my_mpi_rank(discretization.get_comm()) == 0)
    {
      const std::array nodeids{0, 1, 2, 3};

      PureGeometryElementType::instance();
      auto ele = std::make_unique<PureGeometryElement>(0, 0,
          PureGeometryElement::Data{.cell_type = Core::FE::CellType::tet4, .num_dof_per_node = 3});
      ele->set_node_ids(4, nodeids.begin());

      discretization.add_element(std::move(ele));

      const auto add_node = [&](int id, std::array<double, 3> coords)
      { discretization.add_node(coords, id, nullptr); };

      add_node(0, {0.0, 0.0, 0.0});
      add_node(1, {1.0, 0.0, 0.0});
      add_node(2, {0.0, 1.0, 0.0});
      add_node(3, {0.0, 0.0, 1.0});
    }

    discretization.fill_complete();
  }

  inline void fill_cylindrical_hex27(Core::FE::Discretization& discretization, MPI_Comm comm)
  {
    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      std::array<int, 27> nodeids{};
      std::iota(nodeids.begin(), nodeids.end(), 0);

      const double inner = 1.0;
      const double outer = 1.1;
      const double depth = -0.1;
      const double half = (outer + inner) / 2;
      const double angle = 0.1;


      PureGeometryElementType::instance();
      auto ele = std::make_unique<PureGeometryElement>(0, 0,
          PureGeometryElement::Data{.cell_type = Core::FE::CellType::hex27, .num_dof_per_node = 3});
      ele->set_node_ids(27, nodeids.begin());

      discretization.add_element(std::move(ele));

      const auto add_node_polar = [&](int id, std::array<double, 3> coords_polar)
      {
        const std::array coords_cartesian{coords_polar[0] * std::cos(coords_polar[1]),
            coords_polar[0] * std::sin(coords_polar[1]), coords_polar[2]};
        discretization.add_node(coords_cartesian, id, nullptr);
      };

      add_node_polar(0, {inner, 0.0, 0.0});
      add_node_polar(1, {outer, 0.0, 0.0});
      add_node_polar(2, {outer, 0.0, depth});
      add_node_polar(3, {inner, 0.0, depth});

      add_node_polar(4, {inner, angle, 0.0});
      add_node_polar(5, {outer, angle, 0.0});
      add_node_polar(6, {outer, angle, depth});
      add_node_polar(7, {inner, angle, depth});

      add_node_polar(8, {half, 0.0, 0.0});
      add_node_polar(10, {half, 0.0, depth});

      add_node_polar(9, {outer, 0.0, depth / 2});
      add_node_polar(11, {inner, 0.0, depth / 2});

      add_node_polar(16, {half, angle, 0.0});
      add_node_polar(18, {half, angle, depth});

      add_node_polar(12, {inner, (angle / 2), 0.0});
      add_node_polar(15, {inner, (angle / 2), depth});

      add_node_polar(13, {outer, (angle / 2), 0.0});
      add_node_polar(14, {outer, (angle / 2), depth});

      add_node_polar(19, {inner, angle, depth / 2});
      add_node_polar(17, {outer, angle, depth / 2});

      add_node_polar(20, {half, 0.0, depth / 2});
      add_node_polar(21, {half, (angle / 2), 0});
      add_node_polar(22, {outer, (angle / 2), depth / 2});
      add_node_polar(23, {half, (angle / 2), depth});
      add_node_polar(24, {inner, (angle / 2), depth / 2});
      add_node_polar(25, {half, (angle), depth / 2});
      add_node_polar(26, {half, (angle / 2), depth / 2});
    }

    discretization.fill_complete();
  }

  inline void fill_undeformed_hex27(Core::FE::Discretization& discretization, MPI_Comm comm)
  {
    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      std::array<int, 27> nodeids{};
      std::iota(nodeids.begin(), nodeids.end(), 0);

      PureGeometryElementType::instance();
      auto ele = std::make_unique<PureGeometryElement>(0, 0,
          PureGeometryElement::Data{.cell_type = Core::FE::CellType::hex27, .num_dof_per_node = 3});
      ele->set_node_ids(27, nodeids.begin());

      discretization.add_element(std::move(ele));

      const auto add_node = [&](int id, std::array<double, 3> coords)
      { discretization.add_node(coords, id, nullptr); };

      const std::vector<std::array<double, 3>> coords{{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0},
          {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0}, {1.0, 1.0, 1.0},
          {-1.0, 1.0, 1.0}, {0.0, -1.0, -1.0}, {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0},
          {-1.0, 0.0, -1.0}, {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0},
          {0.0, -1.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {-1.0, 0.0, 1.0}, {0.0, 0.0, -1.0},
          {0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
          {0.0, 0.0, 0.0}};

      int counter = 0;
      for (const auto& coord : coords) add_node(counter++, coord);
    }

    discretization.fill_complete();
  }

  inline void fill_deformed_hex27(Core::FE::Discretization& discretization, MPI_Comm comm)
  {
    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      const std::string elementType = "SOlidH27";
      const std::string disType = "HEX27";
      std::array<int, 27> nodeids{};
      std::iota(nodeids.begin(), nodeids.end(), 0);

      PureGeometryElementType::instance();
      auto ele = std::make_unique<PureGeometryElement>(0, 0,
          PureGeometryElement::Data{.cell_type = Core::FE::CellType::hex27, .num_dof_per_node = 3});
      ele->set_node_ids(27, nodeids.begin());

      discretization.add_element(std::move(ele));

      const auto add_node = [&](int id, std::array<double, 3> coords)
      { discretization.add_node(coords, id, nullptr); };

      const std::vector<std::array<double, 3>> coords{{-0.9, -1.0, -1.0}, {1.0, -1.0, -1.0},
          {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.2}, {1.0, 1.0, 1.0},
          {-1.0, 1.0, 1.0}, {0.0, -1.0, -1.0}, {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0},
          {-1.0, 0.0, -1.0}, {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {.9, 1.0, 0.0}, {-1.0, 1.0, 0.0},
          {0.0, -1.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {-1.0, 0.0, 1.0}, {0.0, 0.0, -1.0},
          {0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
          {0.0, 0.0, 0.0}};

      int counter = 0;
      for (const auto& coord : coords) add_node(counter++, coord);
    }

    discretization.fill_complete();
  }

}  // namespace TESTING

#endif
