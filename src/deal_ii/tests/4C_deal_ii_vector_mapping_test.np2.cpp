// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_deal_ii_create_discretization_helper_test.hpp"
#include "4C_deal_ii_dofs.hpp"
#include "4C_deal_ii_triangulation.hpp"
#include "4C_deal_ii_vector_conversion.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools_interpolate.templates.h>
#include <Teuchos_ParameterList.hpp>

namespace
{
  using namespace FourC;

  constexpr int dim = 3;

  // Small function that computes different entries for different location and components
  class Function : public dealii::Function<dim>
  {
   public:
    Function() : dealii::Function<dim>(dim) {}

    double value(const dealii::Point<dim>& p, const unsigned int component) const override
    {
      return (component + 1) * (p[0] + 1) * (p[1] + 2) * (p[2] + 3);
    }
  } function{};

  class VectorMappingTest
      : public ::testing::TestWithParam<std::function<void(Core::FE::Discretization&, MPI_Comm)>>
  {
   protected:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

    VectorMappingTest()
    {
      const auto& generator = GetParam();
      generator(discret, MPI_COMM_WORLD);
      context = std::make_unique<DealiiWrappers::Context<dim>>(
          DealiiWrappers::create_triangulation(tria, discret));
      DealiiWrappers::assign_fes_and_dofs(*context, dof_handler);
    }


    dealii::parallel::fullydistributed::Triangulation<dim> tria{MPI_COMM_WORLD};
    Core::FE::Discretization discret{"empty", MPI_COMM_WORLD, dim};
    dealii::DoFHandler<dim> dof_handler{tria};
    std::unique_ptr<DealiiWrappers::Context<dim>> context;
  };

  TEST_P(VectorMappingTest, VectorOfOnesDealiiToFourCToDealii)
  {
    VectorType dealii_vector;
    dealii::IndexSet locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    dealii_vector.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);
    dealii_vector = 1.0;

    const auto four_c_vector = Core::LinAlg::create_vector(*discret.dof_row_map());


    DealiiWrappers::VectorConverter<VectorType, dim> vector_mapping{dof_handler, *context};
    vector_mapping.to_four_c(*four_c_vector, dealii_vector);

    {
      double r;
      four_c_vector->max_value(&r);
      EXPECT_DOUBLE_EQ(r, 1.0);
      four_c_vector->min_value(&r);
      EXPECT_DOUBLE_EQ(r, 1.0);
    }

    // Now convert the 4C vector to a deal.II vector
    dealii_vector = 0.0;

    vector_mapping.to_dealii(dealii_vector, *four_c_vector);

    {
      EXPECT_DOUBLE_EQ(dealii_vector.mean_value(), 1.0);
    }
  }


  TEST_P(VectorMappingTest, ComplicatedVectorDealiiToFourCToDealii)
  {
    VectorType dealii_vector;
    dealii::IndexSet locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    dealii_vector.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);
    dealii_vector = 1.0;

    dealii::VectorTools::interpolate(
        dealii::MappingQ<dim>(1), dof_handler, function, dealii_vector);

    const auto four_c_vector = Core::LinAlg::create_vector(*discret.dof_row_map());
    ASSERT_EQ(four_c_vector->global_length(), dealii_vector.size());

    DealiiWrappers::VectorConverter<VectorType, dim> vector_mapping{dof_handler, *context};
    vector_mapping.to_four_c(*four_c_vector, dealii_vector);

    // check that every dof as seen by Discretization has the value prescribed in the Function
    {
      // to column map
      auto tmp = Core::LinAlg::create_vector(*discret.dof_col_map(), false);
      Core::LinAlg::export_to(*four_c_vector, *tmp);

      Teuchos::ParameterList params;
      Core::FE::AssembleStrategy strategy(0, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
      discret.evaluate(params, strategy,
          [&](Core::Elements::Element& ele, Core::Elements::LocationArray& la,
              Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
              Core::LinAlg::SerialDenseVector&, Core::LinAlg::SerialDenseVector&,
              Core::LinAlg::SerialDenseVector&)
          {
            std::vector<double> local_values = Core::FE::extract_values(*tmp, la[0].lm_);
            EXPECT_EQ(local_values.size(), dim * ele.num_node());

            for (int node_index = 0; node_index < ele.num_node(); ++node_index)
            {
              const auto& x = ele.nodes()[node_index]->x();
              const dealii::Point<dim> coords(x[0], x[1], x[2]);

              for (unsigned component = 0; component < dim; ++component)
              {
                EXPECT_DOUBLE_EQ(
                    function.value(coords, component), local_values[node_index * dim + component])
                    << "ele: " << ele.id() << ", node: " << node_index
                    << ", component: " << component;
              }
            }

            return 0;
          });
    }

    // Now convert the 4C vector to a deal.II vector
    VectorType converted_dealii_vector;
    // only sets up partitioning, does not copy
    converted_dealii_vector.reinit(dealii_vector);

    vector_mapping.to_dealii(converted_dealii_vector, *four_c_vector);

    ASSERT_EQ(dealii_vector.locally_owned_size(), converted_dealii_vector.locally_owned_size());

    // Now check that the vector that is converted back hast the same entries as the initial vector
    for (unsigned i = 0; i < converted_dealii_vector.locally_owned_size(); ++i)
    {
      EXPECT_DOUBLE_EQ(dealii_vector.local_element(i), converted_dealii_vector.local_element(i))
          << "i=" << i;
    }
  }


  // Specify for which parameters the tests defined above should run
  INSTANTIATE_TEST_SUITE_P(VectorMappingTest, VectorMappingTest,
      // create different discretization generators
      ::testing::Values([](Core::FE::Discretization& discret, MPI_Comm comm)
          { TESTING::fill_discretization_hyper_cube(discret, 4, comm); },
          [](Core::FE::Discretization& discret, MPI_Comm comm)
          { TESTING::fill_undeformed_hex27(discret, comm); }),
      // descriptive name for the parameters, purely for information in test output
      [](const ::testing::TestParamInfo<VectorMappingTest::ParamType>& info)
      {
        if (info.index == 0)
          return "MultipleHex8";
        else if (info.index == 1)
          return "SingleHex27";
        else
          return "";
      });
}  // namespace
