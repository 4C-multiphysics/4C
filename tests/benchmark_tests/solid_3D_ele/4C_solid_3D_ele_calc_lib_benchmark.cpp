// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"

#include <benchmark/benchmark.h>
#include <Teuchos_ParameterList.hpp>

#include <cstdlib>


FOUR_C_NAMESPACE_OPEN

namespace
{

  template <Core::FE::CellType celltype>
  Discret::Elements::ElementNodes<celltype> get_element_nodes()
  {
    Discret::Elements::ElementNodes<celltype> nodes;
    Core::LinAlg::SerialDenseMatrix reference_nodes =
        Core::FE::get_ele_node_numbering_nodes_paramspace(celltype);

    for (int i = 0; i < Core::FE::num_nodes(celltype); ++i)
    {
      for (int d = 0; d < Core::FE::dim<celltype>; ++d)
      {
        nodes.reference_coordinates(d, i) =
            reference_nodes(d, i) + 0.01 * (std::sin(541.0 * i) + 0.05 * std::cos(41.0 * d));
        nodes.displacements(d, i) = static_cast<double>(i + 1) * (d + 1) * 0.01;


        nodes.current_coordinates(d, i) =
            nodes.current_coordinates(d, i) + nodes.displacements(d, i);
      }
    }

    return nodes;
  }

  template <Core::FE::CellType celltype>
  Discret::Elements::JacobianMapping<celltype> get_jacobian_mapping(
      const Discret::Elements::ElementNodes<celltype>& nodes)
  {
    return Discret::Elements::evaluate_jacobian_mapping(
        Discret::Elements::evaluate_shape_functions_and_derivs({{0.1, 0.2, 0.3}}, nodes), nodes);
  }

  template <Core::FE::CellType celltype>
  Discret::Elements::Stress<celltype> get_stress()
  {
    Discret::Elements::Stress<celltype> stress;
    stress.pk2_ = Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{{
        {1.1, 1.2, 1.3},
        {1.2, 2.2, 2.3},
        {1.3, 2.3, 3.3},
    }});
    stress.cmat_ = Core::LinAlg::dyadic(stress.pk2_, stress.pk2_);
    return stress;
  }
}  // namespace

template <Core::FE::CellType celltype>
static void add_internal_force_vector_bop(benchmark::State& state)
{
  constexpr int num_nodes = Core::FE::num_nodes(celltype);
  constexpr int num_dim = Core::FE::dim<celltype>;
  constexpr int num_str = num_dim * (num_dim + 1) / 2;
  constexpr int num_dofs = num_nodes * num_dim;

  Discret::Elements::ElementNodes<celltype> nodes = get_element_nodes<celltype>();
  Discret::Elements::JacobianMapping<celltype> jacobian_mapping =
      get_jacobian_mapping<celltype>(nodes);
  Discret::Elements::SpatialMaterialMapping<celltype> spatial_material_mapping =
      Discret::Elements::evaluate_spatial_material_mapping(jacobian_mapping, nodes);
  Discret::Elements::Stress<celltype> stress = get_stress<celltype>();

  Core::LinAlg::Matrix<num_str, num_dofs> Bop =
      Discret::Elements::evaluate_strain_gradient(jacobian_mapping, spatial_material_mapping);

  double integration_factor = 1.1;
  Core::LinAlg::Matrix<num_dofs, 1> force_vector{};
  for (auto _ : state)
  {
    add_internal_force_vector(Bop, stress, integration_factor, force_vector);
    benchmark::DoNotOptimize(force_vector);
  }
}

using Core::FE::CellType;
BENCHMARK(add_internal_force_vector_bop<CellType::hex8>);
BENCHMARK(add_internal_force_vector_bop<CellType::hex20>);
BENCHMARK(add_internal_force_vector_bop<CellType::hex27>);
BENCHMARK(add_internal_force_vector_bop<CellType::tet4>);
BENCHMARK(add_internal_force_vector_bop<CellType::tet10>);
BENCHMARK(add_internal_force_vector_bop<CellType::wedge6>);
BENCHMARK(add_internal_force_vector_bop<CellType::pyramid5>);



template <Core::FE::CellType celltype>
static void add_internal_force_vector_bfree(benchmark::State& state)
{
  constexpr int num_nodes = Core::FE::num_nodes(celltype);
  constexpr int num_dim = Core::FE::dim<celltype>;
  constexpr int num_dofs = num_nodes * num_dim;

  Discret::Elements::ElementNodes<celltype> nodes = get_element_nodes<celltype>();
  Discret::Elements::JacobianMapping<celltype> jacobian_mapping =
      get_jacobian_mapping<celltype>(nodes);
  Discret::Elements::SpatialMaterialMapping<celltype> spatial_material_mapping =
      Discret::Elements::evaluate_spatial_material_mapping(jacobian_mapping, nodes);
  Discret::Elements::Stress<celltype> stress = get_stress<celltype>();

  Core::LinAlg::Matrix<num_dofs, 1> force_vector{};
  double integration_factor = 1.1;
  for (auto _ : state)
  {
    add_internal_force_vector(jacobian_mapping, spatial_material_mapping.deformation_gradient_,
        stress.pk2_, integration_factor, force_vector);
    benchmark::DoNotOptimize(force_vector);
  }
}

using Core::FE::CellType;
BENCHMARK(add_internal_force_vector_bfree<CellType::hex8>);
BENCHMARK(add_internal_force_vector_bfree<CellType::hex20>);
BENCHMARK(add_internal_force_vector_bfree<CellType::hex27>);
BENCHMARK(add_internal_force_vector_bfree<CellType::tet4>);
BENCHMARK(add_internal_force_vector_bfree<CellType::tet10>);
BENCHMARK(add_internal_force_vector_bfree<CellType::wedge6>);
BENCHMARK(add_internal_force_vector_bfree<CellType::pyramid5>);


template <Core::FE::CellType celltype>
static void add_stiffness_matrix_bop(benchmark::State& state)
{
  constexpr int num_nodes = Core::FE::num_nodes(celltype);
  constexpr int num_dim = Core::FE::dim<celltype>;
  constexpr int num_dofs = num_nodes * num_dim;
  constexpr int num_str = num_dim * (num_dim + 1) / 2;


  Discret::Elements::ElementNodes<celltype> nodes = get_element_nodes<celltype>();
  Discret::Elements::JacobianMapping<celltype> jacobian_mapping =
      get_jacobian_mapping<celltype>(nodes);
  Discret::Elements::SpatialMaterialMapping<celltype> spatial_material_mapping =
      Discret::Elements::evaluate_spatial_material_mapping(jacobian_mapping, nodes);
  Discret::Elements::Stress<celltype> stress = get_stress<celltype>();

  double integration_factor = 1.1;
  Core::LinAlg::Matrix<num_dofs, num_dofs> stiffness_matrix{};
  for (auto _ : state)
  {
    Core::LinAlg::Matrix<num_str, num_dofs> Bop =
        Discret::Elements::evaluate_strain_gradient(jacobian_mapping, spatial_material_mapping);
    add_elastic_stiffness_matrix(Bop, stress, integration_factor, stiffness_matrix);
    add_geometric_stiffness_matrix(jacobian_mapping, stress, integration_factor, stiffness_matrix);
    benchmark::DoNotOptimize(stiffness_matrix);
  }
}

using Core::FE::CellType;
BENCHMARK(add_stiffness_matrix_bop<CellType::hex8>);
BENCHMARK(add_stiffness_matrix_bop<CellType::hex20>);
BENCHMARK(add_stiffness_matrix_bop<CellType::hex27>);
BENCHMARK(add_stiffness_matrix_bop<CellType::tet4>);
BENCHMARK(add_stiffness_matrix_bop<CellType::tet10>);
BENCHMARK(add_stiffness_matrix_bop<CellType::wedge6>);
BENCHMARK(add_stiffness_matrix_bop<CellType::pyramid5>);


template <Core::FE::CellType celltype>
static void add_stiffness_matrix_bfree(benchmark::State& state)
{
  constexpr int num_nodes = Core::FE::num_nodes(celltype);
  constexpr int num_dim = Core::FE::dim<celltype>;
  constexpr int num_dofs = num_nodes * num_dim;


  Discret::Elements::ElementNodes<celltype> nodes = get_element_nodes<celltype>();
  Discret::Elements::JacobianMapping<celltype> jacobian_mapping =
      get_jacobian_mapping<celltype>(nodes);
  Discret::Elements::SpatialMaterialMapping<celltype> spatial_material_mapping =
      Discret::Elements::evaluate_spatial_material_mapping(jacobian_mapping, nodes);
  Discret::Elements::Stress<celltype> stress = get_stress<celltype>();
  double integration_factor = 1.1;

  Core::LinAlg::Matrix<num_dofs, num_dofs> stiffness_matrix{};
  for (auto _ : state)
  {
    add_stiffness_matrix(jacobian_mapping, spatial_material_mapping.deformation_gradient_, stress,
        integration_factor, stiffness_matrix);
    benchmark::DoNotOptimize(stiffness_matrix);
  }
}

using Core::FE::CellType;
BENCHMARK(add_stiffness_matrix_bfree<CellType::hex8>);
BENCHMARK(add_stiffness_matrix_bfree<CellType::hex20>);
BENCHMARK(add_stiffness_matrix_bfree<CellType::hex27>);
BENCHMARK(add_stiffness_matrix_bfree<CellType::tet4>);
BENCHMARK(add_stiffness_matrix_bfree<CellType::tet10>);
BENCHMARK(add_stiffness_matrix_bfree<CellType::wedge6>);
BENCHMARK(add_stiffness_matrix_bfree<CellType::pyramid5>);

FOUR_C_NAMESPACE_CLOSE