// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "cut_test_loader.hpp"

#include "4C_linalg_serialdensematrix.hpp"

void MeshLoader::get_cut_node(int nid, double x, double y, double z, double lsv)
{
  if (nid > -1)
  {
    std::vector<double>& values = cut_nodes_[nid];
    values.reserve(3);
    values.push_back(x);
    values.push_back(y);
    values.push_back(z);
    // values.push_back( lsv );
  }
}

void MeshLoader::get_node(int nid, double x, double y, double z, double lsv)
{
  if (nid > -1)
  {
    std::vector<double>& values = nodes_[nid];
    values.reserve(3);
    values.push_back(x);
    values.push_back(y);
    values.push_back(z);
    // values.push_back( lsv );
  }
}

void MeshLoader::create_side(
    int sid, int nid1, int nid2, int nid3, int nid4, Core::FE::CellType shape)
{
  switch (shape)
  {
    case Core::FE::CellType::quad4:
    {
      Core::LinAlg::SerialDenseMatrix xyz(3, 4);
      fill(cut_nodes_, nid1, &xyz(0, 0));
      fill(cut_nodes_, nid2, &xyz(0, 1));
      fill(cut_nodes_, nid3, &xyz(0, 2));
      fill(cut_nodes_, nid4, &xyz(0, 3));

      std::vector<int> nids;
      nids.reserve(4);
      nids.push_back(nid1);
      nids.push_back(nid2);
      nids.push_back(nid3);
      nids.push_back(nid4);
      mesh_.add_cut_side(sid, nids, xyz, Core::FE::CellType::quad4);

      break;
    }
    default:
      FOUR_C_THROW("unknown shape creating a side in mesh loader");
  }
}

void MeshLoader::create_element(int eid, int nid1, int nid2, int nid3, int nid4, int nid5, int nid6,
    int nid7, int nid8, Core::FE::CellType shape)
{
  switch (shape)
  {
    case Core::FE::CellType::hex8:
    {
      Core::LinAlg::SerialDenseMatrix xyz(3, 8);
      fill(nodes_, nid1, &xyz(0, 0));
      fill(nodes_, nid2, &xyz(0, 1));
      fill(nodes_, nid3, &xyz(0, 2));
      fill(nodes_, nid4, &xyz(0, 3));
      fill(nodes_, nid5, &xyz(0, 4));
      fill(nodes_, nid6, &xyz(0, 5));
      fill(nodes_, nid7, &xyz(0, 6));
      fill(nodes_, nid8, &xyz(0, 7));

      std::vector<int> nids;
      nids.reserve(8);
      nids.push_back(nid1);
      nids.push_back(nid2);
      nids.push_back(nid3);
      nids.push_back(nid4);
      nids.push_back(nid5);
      nids.push_back(nid6);
      nids.push_back(nid7);
      nids.push_back(nid8);
      mesh_.add_element(eid, nids, xyz, Core::FE::CellType::hex8);

      break;
    }
    default:
      FOUR_C_THROW("unknown shape creating an element in mesh loader");
  }
}

void MeshLoader::fill(std::map<int, std::vector<double>>& nodes, int nid, double* values)
{
  if (nodes.find(nid) == nodes.end())
  {
    FOUR_C_THROW("node not defined in mesh loader");
  }
  std::vector<double>& v = nodes[nid];
  std::copy(v.begin(), v.end(), values);
}
