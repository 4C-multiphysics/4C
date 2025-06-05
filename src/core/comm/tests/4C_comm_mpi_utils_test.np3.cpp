// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_mpi_utils.hpp"



namespace
{
  using namespace FourC;

  TEST(AllGather, Maps)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    std::map<int, double> map_in;
    for (int i = 0; i < myPID; ++i)
    {
      map_in.emplace(i, static_cast<double>(i));
    }

    const auto maps_out = Core::Communication::all_gather(map_in, comm);

    std::vector<std::map<int, double>> maps_expected;
    for (int pid = 0; pid < Core::Communication::num_mpi_ranks(comm); ++pid)
    {
      std::map<int, double> map_expected;
      for (int i = 0; i < pid; ++i)
      {
        map_expected.insert(std::make_pair(i, static_cast<double>(i)));
      }
      maps_expected.emplace_back(map_expected);
    }

    EXPECT_EQ(maps_out, maps_expected);
  }

  TEST(AllReduce, Vector)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    const std::vector<double> vec_in(myPID + 1, static_cast<double>(myPID));
    const auto vec_out = Core::Communication::all_reduce(vec_in, comm);

    std::vector<double> vec_expected;
    for (int pid = 0; pid < Core::Communication::num_mpi_ranks(comm); ++pid)
    {
      for (int i = 0; i < pid + 1; ++i) vec_expected.emplace_back(pid);
    }

    EXPECT_EQ(vec_out, vec_expected);
  }

  TEST(AllReduce, PairVector)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    const std::vector<std::pair<int, double>> pair_vec_in(
        myPID + 1, std::make_pair(myPID, static_cast<double>(myPID)));
    const auto pair_vec_out = Core::Communication::all_reduce(pair_vec_in, comm);

    std::vector<std::pair<int, double>> pair_vec_expected;
    for (int pid = 0; pid < Core::Communication::num_mpi_ranks(comm); ++pid)
    {
      for (int i = 0; i < pid + 1; ++i)
      {
        pair_vec_expected.emplace_back(std::make_pair(pid, static_cast<double>(pid)));
      }
    }

    EXPECT_EQ(pair_vec_out, pair_vec_expected);
  }

  TEST(AllReduce, Map)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    const std::map<int, double> map_in = {std::make_pair(myPID, static_cast<double>(myPID))};
    const auto map_out = Core::Communication::all_reduce(map_in, comm);

    std::map<int, double> map_expected;
    for (int pid = 0; pid < Core::Communication::num_mpi_ranks(comm); ++pid)
    {
      map_expected.insert(std::make_pair(pid, static_cast<double>(pid)));
    }

    EXPECT_EQ(map_out, map_expected);
  }

  TEST(AllReduce, UnorderedMap)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    const std::unordered_map<int, double> map_in = {
        std::make_pair(myPID, static_cast<double>(myPID))};
    const auto map_out = Core::Communication::all_reduce(map_in, comm);

    std::unordered_map<int, double> map_expected;
    for (int pid = 0; pid < Core::Communication::num_mpi_ranks(comm); ++pid)
    {
      map_expected.insert(std::make_pair(pid, static_cast<double>(pid)));
    }

    EXPECT_EQ(map_out, map_expected);
  }

  TEST(AllReduce, UnorderedMultiMap)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    const std::unordered_multimap<int, int> map_in = {std::make_pair(1, myPID)};
    const auto map_out = Core::Communication::all_reduce(map_in, comm);

    std::unordered_multimap<int, int> map_expected;
    for (int pid = 0; pid < Core::Communication::num_mpi_ranks(comm); ++pid)
    {
      map_expected.insert(std::make_pair(1, pid));
    }

    EXPECT_EQ(map_out, map_expected);
  }

  TEST(AllReduce, Set)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    const std::set<int> set_in = {myPID};
    const std::set<int> set_out = Core::Communication::all_reduce(set_in, comm);

    std::set<int> set_expected;
    for (int pid = 0; pid < Core::Communication::num_mpi_ranks(comm); ++pid)
    {
      set_expected.insert(pid);
    }

    EXPECT_EQ(set_out, set_expected);
  }


  TEST(AllReduce, VectorNonTrivial)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    const std::vector<std::string> vec_in(myPID + 1, std::to_string(myPID));
    const auto vec_out = Core::Communication::all_reduce(vec_in, comm);

    std::vector<std::string> vec_expected;
    for (int pid = 0; pid < Core::Communication::num_mpi_ranks(comm); ++pid)
    {
      for (int i = 0; i < pid + 1; ++i) vec_expected.emplace_back(std::to_string(pid));
    }

    EXPECT_EQ(vec_out, vec_expected);
  }

  TEST(AllReduce, ComplicatedMap)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    const std::map<std::string, std::pair<int, double>> in = {
        {std::to_string(myPID), {myPID, 2.0}}};
    const auto out = Core::Communication::all_reduce(in, comm);

    std::map<std::string, std::pair<int, double>> expected;
    for (int pid = 0; pid < Core::Communication::num_mpi_ranks(comm); ++pid)
    {
      expected.insert({std::to_string(pid), {pid, 2.0}});
    }

    EXPECT_EQ(out, expected);
  }

  TEST(AllReduce, MapValuesOr)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    using MapType = std::map<std::string, bool>;
    const MapType in{{std::to_string(myPID), (myPID % 2) == 0}, {"key", false}};
    const auto reduced_map = Core::Communication::all_reduce<MapType>(
        in,
        [](const MapType& r, const MapType& in) -> MapType
        {
          MapType result = r;
          for (const auto& [key, value] : in)
          {
            result[key] |= value;
          }
          return result;
        },
        comm);

    MapType expected;
    for (int pid = 0; pid < Core::Communication::num_mpi_ranks(comm); ++pid)
    {
      expected[std::to_string(pid)] = (pid % 2) == 0;
    }
    expected["key"] = false;
    EXPECT_EQ(reduced_map, expected);
  }

  TEST(Broadcast, Vector)
  {
    MPI_Comm comm(MPI_COMM_WORLD);

    // at least two procs required
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    const int myPID = Core::Communication::my_mpi_rank(comm);
    std::vector<double> vec;
    if (myPID == 0)
    {
      vec = {1.0, 2.0, 3.0};
    }
    else
    {
      vec.resize(3);
    }

    Core::Communication::broadcast(vec, 0, comm);

    EXPECT_EQ(vec, std::vector<double>({1.0, 2.0, 3.0}));
  }

  TEST(Broadcast, Enum)
  {
    MPI_Comm comm(MPI_COMM_WORLD);
    ASSERT_GT(Core::Communication::num_mpi_ranks(comm), 1);

    enum class MyEnum : int
    {
      Value1,
      Value2,
      Value3
    };

    MyEnum val;

    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      val = MyEnum::Value3;
    }
    Core::Communication::broadcast(val, 0, comm);
    EXPECT_EQ(val, MyEnum::Value3);
  }


}  // namespace
