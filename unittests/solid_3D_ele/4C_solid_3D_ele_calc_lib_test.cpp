// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_solid_3D_ele_calc_lib.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_tensor_symmetric_einstein.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <type_traits>

namespace
{
  using namespace FourC;

  TEST(EvaluateParameterCoordinateCentroid, DisTypeHex)
  {
    // only tested for hex8, but equivalent for hex18, hex27, ...
    const auto distype = Core::FE::CellType::hex8;

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> xi_centroid_ref{};

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> xi_centroid =
        Discret::Elements::evaluate_parameter_coordinate_centroid<distype>();

    FOUR_C_EXPECT_NEAR(xi_centroid, xi_centroid_ref, 1e-15);
  }

  TEST(EvaluateParameterCoordinateCentroid, DisTypeTet)
  {
    // only tested for tet4, but equivalent for tet10
    const auto distype = Core::FE::CellType::tet4;

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> xi_centroid_ref = {
        {0.25, 0.25, 0.25}};

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> xi_centroid =
        Discret::Elements::evaluate_parameter_coordinate_centroid<distype>();

    FOUR_C_EXPECT_NEAR(xi_centroid, xi_centroid_ref, 1e-15);
  }

  TEST(EvaluateParameterCoordinateCentroid, DisTypePyramid)
  {
    const auto distype = Core::FE::CellType::pyramid5;

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> xi_centroid_ref = {
        {0.0, 0.0, 0.25}};

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> xi_centroid =
        Discret::Elements::evaluate_parameter_coordinate_centroid<distype>();

    FOUR_C_EXPECT_NEAR(xi_centroid, xi_centroid_ref, 1e-15);
  }

  TEST(EvaluateParameterCoordinateCentroid, DisTypeWedge)
  {
    const auto distype = Core::FE::CellType::wedge6;

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> xi_centroid_ref = {
        {1.0 / 3.0, 1.0 / 3.0, 0.0}};


    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> xi_centroid =
        Discret::Elements::evaluate_parameter_coordinate_centroid<distype>();

    FOUR_C_EXPECT_NEAR(xi_centroid, xi_centroid_ref, 1e-15);
  }

  TEST(EvaluateReferenceCoordinateCentroid, DisTypeHex)
  {
    const auto distype = Core::FE::CellType::hex8;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1>
        reference_coords_centroid_ref(Core::LinAlg::Initialization::zero);

    Discret::Elements::ElementNodes<distype> nodal_coordinates;

    nodal_coordinates.reference_coordinates(0, 0) = 0;
    nodal_coordinates.reference_coordinates(1, 0) = 0;
    nodal_coordinates.reference_coordinates(2, 0) = 0;

    nodal_coordinates.reference_coordinates(0, 1) = 4;
    nodal_coordinates.reference_coordinates(1, 1) = 0;
    nodal_coordinates.reference_coordinates(2, 1) = 0;

    nodal_coordinates.reference_coordinates(0, 2) = 4;
    nodal_coordinates.reference_coordinates(1, 2) = 1;
    nodal_coordinates.reference_coordinates(2, 2) = 0;

    nodal_coordinates.reference_coordinates(0, 3) = 0;
    nodal_coordinates.reference_coordinates(1, 3) = 1;
    nodal_coordinates.reference_coordinates(2, 3) = 0;

    nodal_coordinates.reference_coordinates(0, 4) = 0;
    nodal_coordinates.reference_coordinates(1, 4) = 0;
    nodal_coordinates.reference_coordinates(2, 4) = 2;

    nodal_coordinates.reference_coordinates(0, 5) = 4;
    nodal_coordinates.reference_coordinates(1, 5) = 0;
    nodal_coordinates.reference_coordinates(2, 5) = 2;

    nodal_coordinates.reference_coordinates(0, 6) = 4;
    nodal_coordinates.reference_coordinates(1, 6) = 1;
    nodal_coordinates.reference_coordinates(2, 6) = 2;

    nodal_coordinates.reference_coordinates(0, 7) = 0;
    nodal_coordinates.reference_coordinates(1, 7) = 1;
    nodal_coordinates.reference_coordinates(2, 7) = 2;


    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> x_centroid_ref = {
        {2, 0.5, 1.0}};

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> x_centroid =
        Discret::Elements::evaluate_reference_coordinate_centroid<distype>(nodal_coordinates);

    FOUR_C_EXPECT_NEAR(x_centroid, x_centroid_ref, 1e-15);
  }

  TEST(EvaluateReferenceCoordinateCentroid, DisTypeTet)
  {
    const auto distype = Core::FE::CellType::tet4;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1>
        reference_coords_centroid_ref(Core::LinAlg::Initialization::zero);

    Discret::Elements::ElementNodes<distype> nodal_coordinates;

    nodal_coordinates.reference_coordinates(0, 0) = 0;
    nodal_coordinates.reference_coordinates(1, 0) = 0;
    nodal_coordinates.reference_coordinates(2, 0) = 0;

    nodal_coordinates.reference_coordinates(0, 1) = 1;
    nodal_coordinates.reference_coordinates(1, 1) = 0;
    nodal_coordinates.reference_coordinates(2, 1) = 0;

    nodal_coordinates.reference_coordinates(0, 2) = 0;
    nodal_coordinates.reference_coordinates(1, 2) = 2;
    nodal_coordinates.reference_coordinates(2, 2) = 0;

    nodal_coordinates.reference_coordinates(0, 3) = 0;
    nodal_coordinates.reference_coordinates(1, 3) = 0;
    nodal_coordinates.reference_coordinates(2, 3) = 4;

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> x_centroid_ref = {
        {0.25, 0.5, 1.0}};

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> x_centroid =
        Discret::Elements::evaluate_reference_coordinate_centroid<distype>(nodal_coordinates);

    FOUR_C_EXPECT_NEAR(x_centroid, x_centroid_ref, 1e-15);
  }

  TEST(EvaluateReferenceCoordinateCentroid, DisTypePyramid)
  {
    const auto distype = Core::FE::CellType::pyramid5;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1>
        reference_coords_centroid_ref(Core::LinAlg::Initialization::zero);

    Discret::Elements::ElementNodes<distype> nodal_coordinates;

    nodal_coordinates.reference_coordinates(0, 0) = -2;
    nodal_coordinates.reference_coordinates(1, 0) = -1;
    nodal_coordinates.reference_coordinates(2, 0) = 0;

    nodal_coordinates.reference_coordinates(0, 1) = 4;
    nodal_coordinates.reference_coordinates(1, 1) = -1;
    nodal_coordinates.reference_coordinates(2, 1) = 0;

    nodal_coordinates.reference_coordinates(0, 2) = 4;
    nodal_coordinates.reference_coordinates(1, 2) = 1;
    nodal_coordinates.reference_coordinates(2, 2) = 0;

    nodal_coordinates.reference_coordinates(0, 3) = -2;
    nodal_coordinates.reference_coordinates(1, 3) = 1;
    nodal_coordinates.reference_coordinates(2, 3) = 0;

    nodal_coordinates.reference_coordinates(0, 4) = 1;
    nodal_coordinates.reference_coordinates(1, 4) = 0;
    nodal_coordinates.reference_coordinates(2, 4) = 4;


    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> x_centroid_ref = {
        {1.0, 0.0, 1.0}};

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> x_centroid =
        Discret::Elements::evaluate_reference_coordinate_centroid<distype>(nodal_coordinates);

    FOUR_C_EXPECT_NEAR(x_centroid, x_centroid_ref, 1e-15);
  }

  TEST(EvaluateReferenceCoordinateCentroid, DisTypeWedge)
  {
    const auto distype = Core::FE::CellType::wedge6;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1>
        reference_coords_centroid_ref(Core::LinAlg::Initialization::zero);

    Discret::Elements::ElementNodes<distype> nodal_coordinates;

    nodal_coordinates.reference_coordinates(0, 0) = 0;
    nodal_coordinates.reference_coordinates(1, 0) = 0;
    nodal_coordinates.reference_coordinates(2, 0) = 0;

    nodal_coordinates.reference_coordinates(0, 1) = 3;
    nodal_coordinates.reference_coordinates(1, 1) = 0;
    nodal_coordinates.reference_coordinates(2, 1) = 0;

    nodal_coordinates.reference_coordinates(0, 2) = 0;
    nodal_coordinates.reference_coordinates(1, 2) = 6;
    nodal_coordinates.reference_coordinates(2, 2) = 0;

    nodal_coordinates.reference_coordinates(0, 3) = 0;
    nodal_coordinates.reference_coordinates(1, 3) = 0;
    nodal_coordinates.reference_coordinates(2, 3) = 1;

    nodal_coordinates.reference_coordinates(0, 4) = 3;
    nodal_coordinates.reference_coordinates(1, 4) = 0;
    nodal_coordinates.reference_coordinates(2, 4) = 1;

    nodal_coordinates.reference_coordinates(0, 5) = 0;
    nodal_coordinates.reference_coordinates(1, 5) = 6;
    nodal_coordinates.reference_coordinates(2, 5) = 1;

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> x_centroid_ref = {
        {1.0, 2.0, 0.5}};

    Core::LinAlg::Tensor<double, Discret::Elements::Internal::num_dim<distype>> x_centroid =
        Discret::Elements::evaluate_reference_coordinate_centroid<distype>(nodal_coordinates);

    FOUR_C_EXPECT_NEAR(x_centroid, x_centroid_ref, 1e-15);
  }

  namespace
  {
    template <Core::FE::CellType celltype>
    Discret::Elements::ElementNodes<celltype> get_element_nodes()
    {
      std::srand(0);
      Discret::Elements::ElementNodes<celltype> nodes;
      Core::LinAlg::SerialDenseMatrix reference_nodes =
          Core::FE::get_ele_node_numbering_nodes_paramspace(celltype);

      for (int i = 0; i < Core::FE::num_nodes(celltype); ++i)
      {
        for (int d = 0; d < Core::FE::dim<celltype>; ++d)
        {
          nodes.reference_coordinates(d, i) =
              reference_nodes(d, i) + 0.03 * ((double)std::rand()) / RAND_MAX;
          nodes.displacements(d, i) = 0.02 * ((double)std::rand()) / RAND_MAX;


          nodes.current_coordinates(d, i) =
              nodes.reference_coordinates(d, i) + nodes.displacements(d, i);
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
      stress.cmat_ = Core::LinAlg::einsum_sym<"AC", "BD">(stress.pk2_, stress.pk2_) +
                     Core::LinAlg::einsum_sym<"AD", "BC">(stress.pk2_, stress.pk2_);
      return stress;
    }

    template <Core::FE::CellType... celltypes>
    using make_celltyped_test =
        testing::Types<std::integral_constant<Core::FE::CellType, celltypes>...>;
  }  // namespace

  using celltype_list = make_celltyped_test<Core::FE::CellType::hex8, Core::FE::CellType::hex18,
      Core::FE::CellType::hex20, Core::FE::CellType::hex27, Core::FE::CellType::tet4,
      Core::FE::CellType::tet10, Core::FE::CellType::pyramid5, Core::FE::CellType::wedge6,
      Core::FE::CellType::wedge15>;

  template <typename>
  struct SolidEleCalcLibTest : public testing::Test
  {
  };
  TYPED_TEST_SUITE(SolidEleCalcLibTest, celltype_list);

  TYPED_TEST(SolidEleCalcLibTest, BFreeElementStiffnessMatrix)
  {
    constexpr Core::FE::CellType celltype = TypeParam();
    constexpr int num_nodes = Core::FE::num_nodes(celltype);
    constexpr int num_dim = Core::FE::dim<celltype>;
    constexpr int num_dofs = num_nodes * num_dim;
    constexpr int num_str = num_dim * (num_dim + 1) / 2;

    const Discret::Elements::ElementNodes<celltype> nodes = get_element_nodes<celltype>();
    const FourC::Discret::Elements::JacobianMapping<celltype> jacobian_mapping =
        get_jacobian_mapping(nodes);
    const Discret::Elements::SpatialMaterialMapping<celltype> spatial_material_mapping =
        Discret::Elements::evaluate_spatial_material_mapping(jacobian_mapping, nodes);
    Discret::Elements::Stress<celltype> stress = get_stress<celltype>();
    const double integration_factor = 1.0;


    // compute stiffness matrix using the B-operator
    Core::LinAlg::Matrix<num_dofs, num_dofs> stiffness_matrix_bop{};
    Core::LinAlg::Matrix<num_str, num_dofs> Bop =
        Discret::Elements::evaluate_strain_gradient(jacobian_mapping, spatial_material_mapping);

    add_elastic_stiffness_matrix(Bop, stress, integration_factor, stiffness_matrix_bop);
    add_geometric_stiffness_matrix(
        jacobian_mapping, stress, integration_factor, stiffness_matrix_bop);


    // compute stiffness matrix without the B-operator
    Core::LinAlg::Matrix<num_dofs, num_dofs> stiffness_matrix_bfree{};
    Discret::Elements::add_stiffness_matrix(jacobian_mapping,
        spatial_material_mapping.deformation_gradient_, stress, integration_factor,
        stiffness_matrix_bfree);

    // Compare both for equality
    FOUR_C_EXPECT_NEAR(stiffness_matrix_bop, stiffness_matrix_bfree, 1e-13);
  }
}  // namespace