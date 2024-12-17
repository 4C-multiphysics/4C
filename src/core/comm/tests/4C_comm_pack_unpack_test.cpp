// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_pack_helpers.hpp"

using namespace FourC;

namespace
{
  void fill_data(int& data) { data = 42; }

  void fill_data(double& data) { data = 3.14; }

  void fill_data(bool& data) { data = true; }

  void fill_data(char& data) { data = 'a'; }

  void fill_data(std::string& data) { data = "Hello, World!"; }

  template <typename T>
  void fill_data(std::vector<T>& data)
  {
    data.resize(2);
    fill_data(data[0]);
    fill_data(data[1]);
  }

  template <typename K, typename V>
  void fill_data(std::map<K, V>& data)
  {
    K k;
    fill_data(k);
    fill_data(data[k]);
  }

  template <typename First, typename Second>
  void fill_data(std::pair<First, Second>& data)
  {
    fill_data(data.first);
    fill_data(data.second);
  }


  /**
   * A test fixture to test that basic types can round-trip through the pack/unpack mechanism.
   */
  template <typename T>
  class PackUnpackStandardTypes : public ::testing::Test
  {
  };

  using MyTypes = ::testing::Types<int, double, char, std::string, std::vector<int>,
      std::vector<std::vector<std::vector<int>>>, std::map<std::string, bool>>;

  TYPED_TEST_SUITE(PackUnpackStandardTypes, MyTypes);

  TYPED_TEST(PackUnpackStandardTypes, Empty)
  {
    TypeParam data{};
    Core::Communication::PackBuffer pack_buffer;
    Core::Communication::add_to_pack(pack_buffer, data);

    TypeParam unpacked_data;
    Core::Communication::UnpackBuffer unpack_buffer(pack_buffer());
    Core::Communication::extract_from_pack(unpack_buffer, unpacked_data);

    EXPECT_EQ(data, unpacked_data);
  }

  TYPED_TEST(PackUnpackStandardTypes, NonEmpty)
  {
    TypeParam data;
    fill_data(data);
    Core::Communication::PackBuffer pack_buffer;
    Core::Communication::add_to_pack(pack_buffer, data);

    TypeParam unpacked_data;
    Core::Communication::UnpackBuffer unpack_buffer(pack_buffer());
    Core::Communication::extract_from_pack(unpack_buffer, unpacked_data);

    EXPECT_EQ(data, unpacked_data);
  }

}  // namespace