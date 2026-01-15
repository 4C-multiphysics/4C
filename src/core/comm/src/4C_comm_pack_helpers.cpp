// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_pack_helpers.hpp"

FOUR_C_NAMESPACE_OPEN


void Core::Communication::add_to_pack(
    PackBuffer& data, const Core::LinAlg::SerialDenseMatrix& stuff)
{
  int m = stuff.numRows();
  int n = stuff.numCols();
  add_to_pack(data, m);
  add_to_pack(data, n);
  double* A = stuff.values();
  add_to_pack(data, A, n * m * sizeof(double));
}

void Core::Communication::add_to_pack(
    PackBuffer& data, const Core::LinAlg::SerialDenseVector& stuff)
{
  int m = stuff.length();
  add_to_pack(data, m);
  double* A = stuff.values();
  add_to_pack(data, A, m * sizeof(double));
}

void Core::Communication::add_to_pack(PackBuffer& data, const std::string& stuff)
{
  int numele = stuff.size();
  add_to_pack(data, numele);
  add_to_pack(data, stuff.data(), numele * sizeof(char));
}

void Core::Communication::extract_from_pack(
    UnpackBuffer& buffer, Core::LinAlg::SerialDenseMatrix& stuff)
{
  int m = 0;
  extract_from_pack(buffer, m);
  int n = 0;
  extract_from_pack(buffer, n);
  stuff.reshape(m, n);
  double* a = stuff.values();
  extract_from_pack(buffer, a, n * m * sizeof(double));
}

void Core::Communication::extract_from_pack(
    UnpackBuffer& buffer, Core::LinAlg::SerialDenseVector& stuff)
{
  int m = 0;
  extract_from_pack(buffer, m);
  stuff.resize(m);
  double* a = stuff.values();
  extract_from_pack(buffer, a, m * sizeof(double));
}


void Core::Communication::extract_from_pack(
    Core::Communication::UnpackBuffer& buffer, std::string& stuff)
{
  int dim = 0;
  extract_from_pack(buffer, dim);
  stuff.resize(dim);
  int size = dim * sizeof(char);
  extract_from_pack(buffer, stuff.data(), size);
}

int Core::Communication::extract_and_assert_id(UnpackBuffer& buffer, const int desired_type_id)
{
  int type_id = 0;
  Core::Communication::extract_from_pack(buffer, type_id);

  FOUR_C_ASSERT(type_id == desired_type_id,
      "Wrong instance type data. The extracted type id is {}, while the desired type id is {}",
      type_id, desired_type_id);

  return type_id;
}

FOUR_C_NAMESPACE_CLOSE
