// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "cut_test_utils.hpp"

#include "4C_cut_element.hpp"
#include "4C_cut_mesh.hpp"
#include "4C_cut_meshintersection.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"

#include <numbers>

int numnode;
int numele;

Cut::Element* create_hex8(Cut::Mesh& mesh, Core::LinAlg::SerialDenseMatrix& xyze)
{
  std::vector<int> nids;
  nids.reserve(8);
  for (int i = 0; i < 8; ++i)
  {
    mesh.get_node(numnode, &xyze(0, i));
    nids.push_back(numnode++);
  }

  return mesh.create_hex8(numele++, nids);
}

Cut::Element* create_tet4(Cut::Mesh& mesh, Core::LinAlg::SerialDenseMatrix& xyze)
{
  std::vector<int> nids;
  nids.reserve(4);
  for (int i = 0; i < 4; ++i)
  {
    mesh.get_node(numnode, &xyze(0, i));
    nids.push_back(numnode++);
  }

  return mesh.create_tet4(numele++, nids);
}

Cut::Element* create_wedge6(Cut::Mesh& mesh, Core::LinAlg::SerialDenseMatrix& xyze)
{
  std::vector<int> nids;
  nids.reserve(6);
  for (int i = 0; i < 6; ++i)
  {
    mesh.get_node(numnode, &xyze(0, i));
    nids.push_back(numnode++);
  }

  return mesh.create_wedge6(numele++, nids);
}

Cut::Element* create_pyramid5(Cut::Mesh& mesh, Core::LinAlg::SerialDenseMatrix& xyze)
{
  std::vector<int> nids;
  nids.reserve(5);
  for (int i = 0; i < 5; ++i)
  {
    mesh.get_node(numnode, &xyze(0, i));
    nids.push_back(numnode++);
  }

  return mesh.create_pyramid5(numele++, nids);
}

Cut::Side* create_quad4(Cut::Mesh& mesh, Core::LinAlg::SerialDenseMatrix& xyze)
{
  std::vector<int> nids;
  nids.reserve(4);
  for (int i = 0; i < 4; ++i)
  {
    mesh.get_node(numnode, &xyze(0, i));
    nids.push_back(numnode++);
  }

  return mesh.create_quad4_side(numele++, nids);
}

void create_hex8(Core::LinAlg::SerialDenseMatrix& xyze, double dx, double dy, double dz)
{
  xyze(0, 0) = 0;
  xyze(1, 0) = 0;
  xyze(2, 0) = 0;

  xyze(0, 1) = 1;
  xyze(1, 1) = 0;
  xyze(2, 1) = 0;

  xyze(0, 2) = 1;
  xyze(1, 2) = 1;
  xyze(2, 2) = 0;

  xyze(0, 3) = 0;
  xyze(1, 3) = 1;
  xyze(2, 3) = 0;

  xyze(0, 4) = 0;
  xyze(1, 4) = 0;
  xyze(2, 4) = 1;

  xyze(0, 5) = 1;
  xyze(1, 5) = 0;
  xyze(2, 5) = 1;

  xyze(0, 6) = 1;
  xyze(1, 6) = 1;
  xyze(2, 6) = 1;

  xyze(0, 7) = 0;
  xyze(1, 7) = 1;
  xyze(2, 7) = 1;

  for (int i = 0; i < 8; ++i)
  {
    xyze(0, i) += dx;
    xyze(1, i) += dy;
    xyze(2, i) += dz;
  }
}

Cut::Element* create_hex8(Cut::Mesh& mesh, double dx, double dy, double dz)
{
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);
  create_hex8(xyze, dx, dy, dz);
  return create_hex8(mesh, xyze);
}

void create_hex8_mesh(Cut::Mesh& mesh, int rows, int cols, int depth)
{
  for (int i = 0; i < rows + 1; ++i)
  {
    for (int j = 0; j < cols + 1; ++j)
    {
      for (int k = 0; k < depth + 1; ++k)
      {
        int id = i + j * (rows + 1) + k * (rows + 1) * (cols + 1);
        double coord[3];

        coord[0] = 1. / rows * i;
        coord[1] = 1. / cols * j;
        coord[2] = 1. / depth * k;

        mesh.get_node(numnode + id, coord);
      }
    }
  }

  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      for (int k = 0; k < depth; ++k)
      {
        std::vector<int> nids;
        nids.reserve(8);
        nids.push_back(numnode + i + j * (rows + 1) + k * (rows + 1) * (cols + 1));
        nids.push_back(numnode + i + j * (rows + 1) + 1 + k * (rows + 1) * (cols + 1));
        nids.push_back(numnode + i + (j + 1) * (rows + 1) + 1 + k * (rows + 1) * (cols + 1));
        nids.push_back(numnode + i + (j + 1) * (rows + 1) + k * (rows + 1) * (cols + 1));

        nids.push_back(numnode + i + j * (rows + 1) + (k + 1) * (rows + 1) * (cols + 1));
        nids.push_back(numnode + i + j * (rows + 1) + 1 + (k + 1) * (rows + 1) * (cols + 1));
        nids.push_back(numnode + i + (j + 1) * (rows + 1) + 1 + (k + 1) * (rows + 1) * (cols + 1));
        nids.push_back(numnode + i + (j + 1) * (rows + 1) + (k + 1) * (rows + 1) * (cols + 1));

        mesh.create_hex8(numele++, nids);
      }
    }
  }

  numnode += (rows + 1) * (cols + 1) * (depth + 1);
}

void create_quad4_mesh(Cut::Mesh& mesh, int rows, int cols, std::vector<Cut::Side*>& sides)
{
  double sqrt2 = 1. / sqrt(2.);

  for (int i = 0; i < rows + 1; ++i)
  {
    for (int j = 0; j < cols + 1; ++j)
    {
      int id = i + j * (rows + 1);
      double coord[3];

      double x = (2. / rows * i - 1);
      double y = (2. / cols * j - 1);

      coord[0] = x * sqrt2 - y * sqrt2 + 0.5;
      coord[1] = x * sqrt2 + y * sqrt2 + 0.5;
      coord[2] = 0.5;

      mesh.get_node(numnode + id, coord);
    }
  }

  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      std::vector<int> nids;
      nids.reserve(4);
      nids.push_back(numnode + i + j * (rows + 1));
      nids.push_back(numnode + i + j * (rows + 1) + 1);
      nids.push_back(numnode + i + (j + 1) * (rows + 1) + 1);
      nids.push_back(numnode + i + (j + 1) * (rows + 1));

      sides.push_back(mesh.create_quad4_side(numele++, nids));
    }
  }

  numnode += (rows + 1) * (cols + 1);
}

void create_quad4_cylinder_mesh(
    Cut::MeshIntersection& intersection, double x, double y, int rows, int cols)
{
  double r = 1.;

  int rownodes = rows + 1;
  int colnodes = cols + 1;

  for (int i = 0; i < rownodes; ++i)
  {
    for (int j = 0; j < colnodes; ++j)
    {
      int id = i + j * rownodes;
      double coord[3];

      double alpha = static_cast<double>(i) / rows;

      coord[0] = x + r * cos(2 * std::numbers::pi * alpha);
      coord[1] = y + r * sin(2 * std::numbers::pi * alpha);
      coord[2] = static_cast<double>(j) / cols;

      intersection.cut_mesh().get_node(numnode + id, coord);
    }
  }

  for (int i = 0; i < rows + 1; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      std::vector<int> nids;
      nids.reserve(4);
      nids.push_back(numnode + ((i) % (rows)) + j * rownodes);
      nids.push_back(numnode + ((i + 1) % (rows)) + j * rownodes);
      nids.push_back(numnode + ((i + 1) % (rows)) + (j + 1) * rownodes);
      nids.push_back(numnode + ((i) % (rows)) + (j + 1) * rownodes);

      intersection.add_cut_side(numele++, nids, Core::FE::CellType::quad4);
    }
  }

  numnode += rownodes * colnodes;
}

void cutmesh(Cut::Mesh& mesh)
{
  mesh.make_cut_lines();
  mesh.make_facets();
  mesh.make_volume_cells();

  if (mesh.create_options().find_positions())
  {
    mesh.find_node_positions();
    mesh.find_nodal_dof_sets(true);
  }

  mesh.create_integration_cells(0);

  // Run safety checks
  mesh.test_element_volume(false);
}


SimpleWrapper::SimpleWrapper() : side_count_(0)
{
  mesh_ = new Cut::MeshIntersection;
  mesh_->get_options().init_for_cuttests();  // use full cln
}

SimpleWrapper::~SimpleWrapper() { delete mesh_; }

void SimpleWrapper::create_hex8(const Core::LinAlg::SerialDenseMatrix& xyze)
{
  create_element(Core::FE::CellType::hex8, xyze);
}

void SimpleWrapper::create_tet4(const Core::LinAlg::SerialDenseMatrix& xyze)
{
  create_element(Core::FE::CellType::tet4, xyze);
}

void SimpleWrapper::create_pyramid5(const Core::LinAlg::SerialDenseMatrix& xyze)
{
  create_element(Core::FE::CellType::pyramid5, xyze);
}

void SimpleWrapper::create_wedge6(const Core::LinAlg::SerialDenseMatrix& xyze)
{
  create_element(Core::FE::CellType::wedge6, xyze);
}

void SimpleWrapper::create_hex8_sides(const Core::LinAlg::SerialDenseMatrix& xyze)
{
  create_element_sides(Core::FE::CellType::hex8, xyze);
}

void SimpleWrapper::create_tet4_sides(const Core::LinAlg::SerialDenseMatrix& xyze)
{
  create_element_sides(Core::FE::CellType::tet4, xyze);
}

void SimpleWrapper::create_pyramid5_sides(const Core::LinAlg::SerialDenseMatrix& xyze)
{
  create_element_sides(Core::FE::CellType::pyramid5, xyze);
}

void SimpleWrapper::create_wedge6_sides(const Core::LinAlg::SerialDenseMatrix& xyze)
{
  create_element_sides(Core::FE::CellType::wedge6, xyze);
}

void SimpleWrapper::create_tri3(const Core::LinAlg::SerialDenseMatrix& xyze)
{
  create_side(Core::FE::CellType::tri3, xyze);
}

void SimpleWrapper::create_quad4(const Core::LinAlg::SerialDenseMatrix& xyze)
{
  create_side(Core::FE::CellType::quad4, xyze);
}

void SimpleWrapper::create_hex8(double dx, double dy, double dz)
{
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);
  ::create_hex8(xyze, dx, dy, dz);
  create_hex8(xyze);
}

void SimpleWrapper::create_hex8_sides(double dx, double dy, double dz)
{
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);
  ::create_hex8(xyze, dx, dy, dz);
  create_hex8_sides(xyze);
}

void SimpleWrapper::create_tet4_sides()
{
  Core::LinAlg::SerialDenseMatrix xyze(3, 4);

  xyze(0, 0) = 2;
  xyze(1, 0) = 0;
  xyze(2, 0) = 0;

  xyze(0, 1) = 2;
  xyze(1, 1) = 0;
  xyze(2, 1) = 1;

  xyze(0, 2) = 2;
  xyze(1, 2) = 1;
  xyze(2, 2) = 0;

  xyze(0, 3) = 0.5;
  xyze(1, 3) = 0.5;
  xyze(2, 3) = 0.5;

  create_tet4_sides(xyze);
}

void SimpleWrapper::create_quad4_mesh(int rows, int cols)
{
  double sqrt2 = 1. / sqrt(2.);

  for (int i = 0; i < rows + 1; ++i)
  {
    for (int j = 0; j < cols + 1; ++j)
    {
      double coord[3];

      double x = (2. / rows * i - 1);
      double y = (2. / cols * j - 1);

      coord[0] = x * sqrt2 - y * sqrt2 + 0.5;
      coord[1] = x * sqrt2 + y * sqrt2 + 0.5;
      coord[2] = 0.5;

      get_id(Core::LinAlg::Matrix<3, 1>(coord, true), side_points_);
    }
  }

  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      std::vector<int> nids;
      nids.reserve(4);
      nids.push_back(i + j * (rows + 1));
      nids.push_back(i + j * (rows + 1) + 1);
      nids.push_back(i + (j + 1) * (rows + 1) + 1);
      nids.push_back(i + (j + 1) * (rows + 1));

      Core::LinAlg::SerialDenseMatrix xyze(3, 4);
      for (int l = 0; l < 4; ++l)
      {
        Core::LinAlg::Matrix<3, 1>& x = side_points_[nids[l]];
        std::copy(x.data(), x.data() + 3, &xyze(0, l));
      }
      create_quad4(xyze);
    }
  }
}

void SimpleWrapper::assume_volume_cells(unsigned num)
{
  unsigned numvc = mesh_->normal_mesh().volume_cells().size();
  if (numvc != num)
  {
    std::stringstream str;
    str << "expected " << num << " volume cells, but got " << numvc;
    FOUR_C_THROW("{}", str.str());
  }
}

void SimpleWrapper::cut_test_cut(bool include_inner, bool do_Cut_Positions_Dofsets)
{
  mesh_->cut_test_cut(include_inner, Cut::VCellGaussPts_DirectDivergence,
      Cut::BCellGaussPts_Tessellation, true, true, do_Cut_Positions_Dofsets);
}

void SimpleWrapper::create_element(
    Core::FE::CellType distype, const Core::LinAlg::SerialDenseMatrix& xyze)
{
  int& id = element_count_[distype];
  id += 1;

  std::vector<int> nids;
  nids.reserve(xyze.numCols());
  for (int i = 0; i < xyze.numCols(); ++i)
  {
    Core::LinAlg::Matrix<3, 1> x(&xyze(0, i));
    nids.push_back(get_id(x, element_points_));
  }

  mesh_->add_element(id, nids, xyze, distype);
}

void SimpleWrapper::create_element_sides(
    Core::FE::CellType distype, const Core::LinAlg::SerialDenseMatrix& xyze)
{
  //   int & id = side_count_[distype];
  //   id += 1;

  switch (distype)
  {
    case Core::FE::CellType::hex8:
    {
      Core::LinAlg::SerialDenseMatrix side_xyze(3, 4);
      for (int i = 0; i < 6; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          int node = Core::FE::eleNodeNumbering_hex27_surfaces[i][j];
          std::copy(&xyze(0, node), &xyze(0, node) + 3, &side_xyze(0, j));
        }
        create_side(Core::FE::CellType::quad4, side_xyze);
      }
      break;
    }
    case Core::FE::CellType::tet4:
    {
      Core::LinAlg::SerialDenseMatrix side_xyze(3, 3);
      for (int i = 0; i < 4; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          int node = Core::FE::eleNodeNumbering_tet10_surfaces[i][j];
          std::copy(&xyze(0, node), &xyze(0, node) + 3, &side_xyze(0, j));
        }
        create_side(Core::FE::CellType::tri3, side_xyze);
      }
      break;
    }
    case Core::FE::CellType::pyramid5:
    {
      Core::LinAlg::SerialDenseMatrix quad4_side_xyze(3, 4);
      Core::LinAlg::SerialDenseMatrix tri3_side_xyze(3, 3);
      for (int i = 0; i < 4; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          int node = Core::FE::eleNodeNumbering_pyramid5_trisurfaces[i][j];
          std::copy(&xyze(0, node), &xyze(0, node) + 3, &tri3_side_xyze(0, j));
        }
        create_side(Core::FE::CellType::tri3, tri3_side_xyze);
      }
      for (int i = 0; i < 1; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          int node = Core::FE::eleNodeNumbering_pyramid5_quadsurfaces[i][j];
          std::copy(&xyze(0, node), &xyze(0, node) + 3, &quad4_side_xyze(0, j));
        }
        create_side(Core::FE::CellType::quad4, quad4_side_xyze);
      }
      break;
    }
    case Core::FE::CellType::wedge6:
    {
      Core::LinAlg::SerialDenseMatrix quad4_side_xyze(3, 4);
      Core::LinAlg::SerialDenseMatrix tri3_side_xyze(3, 3);
      for (int i = 0; i < 2; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          int node = Core::FE::eleNodeNumbering_wedge18_trisurfaces[i][j];
          std::copy(&xyze(0, node), &xyze(0, node) + 3, &tri3_side_xyze(0, j));
        }
        create_side(Core::FE::CellType::tri3, tri3_side_xyze);
      }
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          int node = Core::FE::eleNodeNumbering_wedge18_quadsurfaces[i][j];
          std::copy(&xyze(0, node), &xyze(0, node) + 3, &quad4_side_xyze(0, j));
        }
        create_side(Core::FE::CellType::quad4, quad4_side_xyze);
      }
      break;
    }
    default:
      FOUR_C_THROW("distype not supported");
  }
}

void SimpleWrapper::create_side(
    Core::FE::CellType distype, const Core::LinAlg::SerialDenseMatrix& xyze)
{
  // int & id = side_count_[distype];
  int& id = side_count_;
  id += 1;

  std::vector<int> nids;
  nids.reserve(xyze.numCols());
  for (int i = 0; i < xyze.numCols(); ++i)
  {
    Core::LinAlg::Matrix<3, 1> x(&xyze(0, i));
    nids.push_back(get_id(x, side_points_));
  }

  mesh_->add_cut_side(id, nids, xyze, distype);
}

int SimpleWrapper::get_id(
    const Core::LinAlg::Matrix<3, 1>& x, std::vector<Core::LinAlg::Matrix<3, 1>>& points)
{
  unsigned size = points.size();
  for (unsigned i = 0; i < size; ++i)
  {
    Core::LinAlg::Matrix<3, 1> p = points[i];
    p.update(-1, x, 1);
    if (p.norm2() < 1e-13)
    {
      return i;
    }
  }
  points.push_back(x);
  return size;
}
