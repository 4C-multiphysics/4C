// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_comm_utils.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_parameter_container.templates.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_solid_3D_ele_properties.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <benchmark/benchmark.h>
#include <Teuchos_ParameterList.hpp>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  void setup_material_in_global_problem()
  {
    Core::IO::InputParameterContainer container{};
    container.add("YOUNG", 1.0);
    container.add("NUE", 0.1);
    container.add("DENS", 0.1);
    container.add("THEXPANS", 0.1);

    // add material to problem instance
    const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
    Global::Problem::instance(probinst)->materials()->insert(
        1, Mat::make_parameter(1, Core::Materials::MaterialType::m_stvenant, container));
  }

  template <Core::FE::CellType celltype>
  void add_nodes_to_discretization(Core::FE::Discretization& dis)
  {
    const auto add_node_and_coords = [&](int id, std::array<double, 3> coords)
    { dis.add_node(coords, id, nullptr); };

    Core::LinAlg::SerialDenseMatrix reference_nodes =
        Core::FE::get_ele_node_numbering_nodes_paramspace(celltype);
    for (int i = 0; i < reference_nodes.num_cols(); ++i)
    {
      std::array<double, Core::FE::dim<celltype>> coord;
      for (int j = 0; j < Core::FE::dim<celltype>; ++j)
      {
        // create some deterministic coordinates
        coord[j] = reference_nodes(j, i) + 0.01 * (std::sin(541.0 * i) + 0.05 * std::cos(41.0 * j));
      }
      add_node_and_coords(i + 1, coord);
    }
  }

  template <Core::FE::CellType celltype>
  std::shared_ptr<Core::Elements::Element> make_solid_element(
      const Discret::Elements::ElementTechnology& element_technology,
      Inpar::Solid::KinemType kinem_type)
  {
    constexpr auto ele_type = "SOLID";
    std::array<int, Core::FE::num_nodes(celltype)> node_ids{};
    for (std::size_t i = 0; i < Core::FE::num_nodes(celltype); ++i) node_ids[i] = i + 1;

    std::shared_ptr<Core::Elements::Element> ele =
        Core::Communication::factory(ele_type, Core::FE::celltype_string<celltype>, 1, 0);
    ele->set_node_ids(Core::FE::num_nodes(celltype), node_ids.data());


    Core::IO::InputParameterContainer container{};
    container.add("MAT", 1);
    container.add("KINEM", kinem_type);
    container.add("TECH", element_technology);

    ele->read_element(ele_type, Core::FE::cell_type_to_string(celltype), container);

    return ele;
  }

  template <Core::FE::CellType celltype>
  std::unique_ptr<Core::FE::Discretization> make_discretization(
      Discret::Elements::ElementTechnology ele_tech, Inpar::Solid::KinemType kinem_type)
  {
    auto dis = std::make_unique<Core::FE::Discretization>("solid",
        Global::Problem::instance()->get_communicators().global_comm(), Core::FE::dim<celltype>);

    add_nodes_to_discretization<celltype>(*dis);
    dis->add_element(make_solid_element<celltype>(ele_tech, kinem_type));
    dis->fill_complete();

    return dis;
  }

  Teuchos::ParameterList make_parameter_list()
  {
    Teuchos::ParameterList params{};
    params.set("action", "calc_struct_nlnstiff");
    params.set("total time", 1.0);
    params.set("delta time", 1.0);
    return params;
  }

  Core::Elements::LocationArray make_location_array(
      const Core::FE::Discretization& dis, const Core::Elements::Element& ele)
  {
    Core::Elements::LocationArray la{1};
    ele.location_vector(dis, la);
    return la;
  }

  void set_dummy_state(Core::FE::Discretization& dis)
  {
    auto displacement = std::make_shared<Core::LinAlg::Vector<double>>(*dis.dof_row_map(), true);
    auto residual_displacements =
        std::make_shared<Core::LinAlg::Vector<double>>(displacement->get_map(), true);

    // set an arbitrary state
    for (int i = 0; i < displacement->local_length(); ++i)
    {
      (*displacement).get_values()[i] = 0.1 * std::sin(4321.0 * i);
      (*residual_displacements).get_values()[i] = 0.0001 * std::cos(1234.0 * i);
    }

    dis.set_state("displacement", *displacement);
    dis.set_state("residual displacement", *residual_displacements);
  }

  struct ElementMatrices
  {
    Core::LinAlg::SerialDenseMatrix stiffness_matrix{};
    Core::LinAlg::SerialDenseMatrix unused_matrix{};
    Core::LinAlg::SerialDenseVector force_vector{};
    Core::LinAlg::SerialDenseVector unused_vector1{};
    Core::LinAlg::SerialDenseVector unused_vector2{};
  };

  ElementMatrices make_element_matrices(const Core::Elements::LocationArray& la)
  {
    ElementMatrices matrices{};
    matrices.stiffness_matrix.shape(la[0].size(), la[0].size());
    matrices.force_vector.resize(la[0].size());
    return matrices;
  }
}  // namespace

using CellType = Core::FE::CellType;
using ElementTechnology = Discret::Elements::ElementTechnology;
using KinemType = Inpar::Solid::KinemType;

template <Core::FE::CellType celltype,
    Discret::Elements::ElementTechnology ele_tech = Discret::Elements::ElementTechnology::none,
    Inpar::Solid::KinemType kinem_type = Inpar::Solid::KinemType::nonlinearTotLag>
static void evaluate_force_stiff(benchmark::State& state)
{
  Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  Core::Communication::CommConfig config;
  Global::Problem::instance()->set_communicators(Core::Communication::create_comm(config));
  setup_material_in_global_problem();

  auto dis = make_discretization<celltype>(ele_tech, kinem_type);
  set_dummy_state(*dis);

  Core::Elements::Element& ele = *dis->g_element(1);
  auto la = make_location_array(*dis, ele);
  ElementMatrices matrices = make_element_matrices(la);
  Teuchos::ParameterList params = make_parameter_list();
  for (auto _ : state)
  {
    ele.evaluate(params, *dis, la, matrices.stiffness_matrix, matrices.unused_matrix,
        matrices.force_vector, matrices.unused_vector1, matrices.unused_vector2);
  }
}
BENCHMARK(evaluate_force_stiff<CellType::hex8>);
BENCHMARK(evaluate_force_stiff<CellType::hex20>);
BENCHMARK(evaluate_force_stiff<CellType::hex27>);
BENCHMARK(evaluate_force_stiff<CellType::tet4>);
BENCHMARK(evaluate_force_stiff<CellType::tet10>);
BENCHMARK(evaluate_force_stiff<CellType::wedge6>);
BENCHMARK(evaluate_force_stiff<CellType::pyramid5>);
BENCHMARK(evaluate_force_stiff<CellType::hex8, ElementTechnology::none, KinemType::linear>);
BENCHMARK(evaluate_force_stiff<CellType::hex8, ElementTechnology::eas_mild>);
BENCHMARK(evaluate_force_stiff<CellType::hex8, ElementTechnology::eas_full>);
BENCHMARK(evaluate_force_stiff<CellType::hex8, ElementTechnology::fbar>);

FOUR_C_NAMESPACE_CLOSE