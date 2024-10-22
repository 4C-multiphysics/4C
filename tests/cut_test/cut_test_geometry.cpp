// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_intersection.hpp"  // for IntersectionStatus
#include "4C_cut_kernel.hpp"
#include "4C_cut_output.hpp"
#include "4C_cut_position.hpp"

#include <iostream>

using namespace FourC;

void test_geometry_schleifend1()
{
  Core::LinAlg::Matrix<3, 3> tri3;

  // 38
  tri3(0, 0) = 0.90538448100000001872;
  tri3(1, 0) = 0.66671353600000005102;
  tri3(2, 0) = 0.43846240600000002674;

  // 2
  tri3(0, 1) = 0.92070621299999999554;
  tri3(1, 1) = 0.66671353600000005102;
  tri3(2, 1) = 0.4999144669999999735;

  // 1
  tri3(0, 2) = 0.93551695349999997031;
  tri3(1, 2) = 0.68831014649999999744;
  tri3(2, 2) = 0.46358564499999999065;

  Core::LinAlg::Matrix<3, 2> line;

  // 28
  line(0, 0) = 0.91666668699999998005;
  line(1, 0) = 0.66666668699999998005;
  line(2, 0) = 0.483920493638093141;

  // 31
  line(0, 1) = 0.92080880009095389394;
  line(1, 1) = 0.66678706244096386246;
  line(2, 1) = 0.49999999999999994449;

  Core::LinAlg::Matrix<3, 1> xsi;

  // Cut::Kernel::DebugComputeIntersection<Core::FE::CellType::line2,
  // Core::FE::CellType::tri3,true> ci;
  Cut::Kernel::ComputeIntersection<3, Core::FE::CellType::line2, Core::FE::CellType::tri3, true> ci(
      xsi);  // use cln

  if (ci(tri3, line))
  {
  }
  else
  {
    FOUR_C_THROW("not intersected");
  }
}

void test_geometry_parallel1()
{
  int s[] = {0, 1072693248, -1717986918, 1070176665, -858993459, 1071959244, 0, 1072693248,
      -858993459, 1071959244, -1717986918, 1070176665, -2, 1072693247, -1717986918, 1072273817,
      1717986919, 1071015526};
  int l[] = {
      0,
      -1075838976,
      -1717986918,
      1072273817,
      -1717986918,
      1070176665,
      0,
      -1075838976,
      -1717986918,
      1072273817,
      -1717986918,
      1072273817,
  };

  Core::LinAlg::Matrix<3, 3> tri3(reinterpret_cast<double*>(s));
  Core::LinAlg::Matrix<3, 2> line(reinterpret_cast<double*>(l));

  std::cout << tri3 << line;

  Core::LinAlg::Matrix<3, 1> xsi;

  // Cut::Kernel::DebugComputeIntersection<Core::FE::CellType::line2,
  // Core::FE::CellType::tri3,true> ci;
  Cut::Kernel::ComputeIntersection<3, Core::FE::CellType::line2, Core::FE::CellType::tri3, true> ci(
      xsi);  // use cln



  bool conv = ci(tri3, line);

  if (!conv)
  {
    if ((ci.get_edge_location().within_side()) and (ci.get_side_location().within_side()))
      FOUR_C_THROW("intersected");
  }
  else
  {
  }
}

void test_geometry_distance()
{
  double xyze_data[] = {0.90999999999999992, 0.069230769230768999, 0.31212930977131004,
      0.90999999999999992, 0.061656666666666672, 0.2943944262758405, 0.90999999999999992,
      0.061538461538461306, 0.29411764705882476};
  double xyz_data[] = {0.91044776119402959, 0.061538461538461306, 0.29411764705882476};

  Core::LinAlg::Matrix<3, 3> xyze(xyze_data);
  Core::LinAlg::Matrix<3, 1> xyz(xyz_data);

  Cut::PositionFactory::specify_general_dist_floattype(Cut::floattype_cln);    // use cln
  Cut::PositionFactory::specify_general_pos_floattype(Cut::floattype_double);  // use
                                                                               // double
  Teuchos::RCP<Cut::Position> pos = Cut::Position::create(xyze, xyz, Core::FE::CellType::tri3);
  if (pos->compute())
  {
  }
}

void test_geometry_distance2()
{
  double xyze_row_data[] = {0, 0, 0, 0, 0.737999, -0.737999, -0.737999, 0.737999, -0.207634,
      -0.207634, -0.207472, 0.62274};
  double xyz_data[] = {-1.476, -0.737999, -0.207634};

  Core::LinAlg::Matrix<3, 4> xyze;
  Core::LinAlg::Matrix<3, 1> xyz(xyz_data);

  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      xyze(i, j) = xyze_row_data[i * 4 + j];
    }
  }

  Cut::PositionFactory::specify_general_dist_floattype(Cut::floattype_cln);    // use cln
  Cut::PositionFactory::specify_general_pos_floattype(Cut::floattype_double);  // use
                                                                               // double
  Teuchos::RCP<Cut::Position> pos = Cut::Position::create(xyze, xyz, Core::FE::CellType::quad4);
  if (pos->compute())
  {
  }
}

void test_geometry_distance3()
{
  double xyze_row_data[] = {0, 0, 0, 0, -0.1327641128640012, -0.1327641128640012,
      0.3981781258443317, -0.132649900116329, 0.8469286675746165, -0.8469286675746165,
      -0.8469286675746165, 0.8469286675746165};
  double xyz_data[] = {1.693857335149233, -0.1327438687864578, 0.8469286675746165};

  Core::LinAlg::Matrix<3, 4> xyze;
  Core::LinAlg::Matrix<3, 1> xyz(xyz_data);

  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      xyze(i, j) = xyze_row_data[i * 4 + j];
    }
  }

  Cut::PositionFactory::specify_general_dist_floattype(Cut::floattype_cln);    // use cln
  Cut::PositionFactory::specify_general_pos_floattype(Cut::floattype_double);  // use
                                                                               // double
  Teuchos::RCP<Cut::Position> pos = Cut::Position::create(xyze, xyz, Core::FE::CellType::quad4);
  if (pos->compute())
  {
  }
  else
  {
  }
}

void test_geometry()
{
  test_geometry_schleifend1();
  test_geometry_parallel1();
  test_geometry_distance();
  test_geometry_distance2();
  test_geometry_distance3();
}
