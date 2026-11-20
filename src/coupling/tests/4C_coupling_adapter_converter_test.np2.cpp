// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_coupling_adapter_converter.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_sparsematrix.hpp"

#include <array>
#include <map>
#include <memory>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace
{
  // The row transform uses only the map methods; vector conversions are not needed here.
  class IdentityRowConverter : public Coupling::Adapter::CouplingConverter
  {
   public:
    IdentityRowConverter(std::shared_ptr<const Core::LinAlg::Map> source,
        std::shared_ptr<const Core::LinAlg::Map> destination)
        : source_(std::move(source)), destination_(std::move(destination))
    {
    }

    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> src_to_dst(
        [[maybe_unused]] std::shared_ptr<const Core::LinAlg::Vector<double>> source) const override
    {
      return nullptr;
    }

    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> dst_to_src(
        [[maybe_unused]] std::shared_ptr<const Core::LinAlg::Vector<double>> destination)
        const override
    {
      return nullptr;
    }

    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> src_map() const override
    {
      return source_;
    }
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> dst_map() const override
    {
      return destination_;
    }
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> perm_src_map() const override
    {
      return destination_;
    }
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> perm_dst_map() const override
    {
      return source_;
    }

    void fill_src_to_dst_map(std::map<int, int>& rows) const override
    {
      for (int i = 0; i < source_->num_my_elements(); ++i) rows[source_->gid(i)] = source_->gid(i);
    }

   private:
    std::shared_ptr<const Core::LinAlg::Map> source_;
    std::shared_ptr<const Core::LinAlg::Map> destination_;
  };

  TEST(CouplingAdapterConverterTest, MatrixRowTransformImportsRowsOntoRankWithNoSourceRows)
  {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = Core::Communication::my_mpi_rank(comm);
    const std::array source_rows = {0, 1};
    auto source_map = std::make_shared<Core::LinAlg::Map>(
        -1, rank == 0 ? std::span<const int>(source_rows) : std::span<const int>{}, 0, comm);
    const int destination_row = rank;
    auto destination_map =
        std::make_shared<Core::LinAlg::Map>(-1, std::span<const int>(&destination_row, 1), 0, comm);
    const std::array columns = {10, 11, 12};
    Core::LinAlg::Map domain_map(
        -1, rank == 0 ? std::span<const int>(columns) : std::span<const int>{}, 0, comm);

    Core::LinAlg::SparseMatrix source(
        *source_map, 3, Core::LinAlg::SparseMatrix::StorageProfile::static_profile);
    if (rank == 0)
    {
      source.assemble(7.0, 0, 10);
      for (int i = 0; i < 3; ++i) source.assemble(1.0 + i, 1, columns[i]);
    }
    source.complete(domain_map, *source_map);
    EXPECT_EQ(source.max_num_entries(), rank == 0 ? 3 : 0);
    EXPECT_EQ(source.global_max_num_entries(), 3);

    Core::LinAlg::SparseMatrix transformed(*destination_map, 3);
    IdentityRowConverter converter(source_map, destination_map);
    Coupling::Adapter::MatrixRowTransform transform;
    ASSERT_TRUE(transform(source, 1.0, converter, transformed));
    transformed.complete(domain_map, *destination_map);

    EXPECT_TRUE(transformed.row_map().same_as(*destination_map));
    EXPECT_TRUE(transformed.domain_map().same_as(domain_map));
    int count = 0;
    std::array<double, 3> values{};
    std::array<int, 3> indices{};
    transformed.extract_global_row_copy(destination_row, 3, count, values.data(), indices.data());
    ASSERT_EQ(count, rank == 0 ? 1 : 3);
    for (int i = 0; i < count; ++i)
    {
      EXPECT_GE(indices[i], 10);
      EXPECT_LE(indices[i], 12);
      EXPECT_DOUBLE_EQ(values[i], rank == 0 ? 7.0 : indices[i] - 9.0);
    }
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE
