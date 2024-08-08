/*----------------------------------------------------------------------*/
/*! \file
\brief Test for the CUT Library

\level 1

*----------------------------------------------------------------------*/

#include "4C_cut_mesh.hpp"
#include "4C_cut_options.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cut_test_utils.hpp"

void test_cut_volumes()
{
  Cut::Options options;
  options.init_for_cuttests();
  // this is meant to be used with matching boundaries. Thus, no
  // inside/outside positions.
  options.set_find_positions(false);

  Cut::Mesh mesh1(options);
  Cut::Mesh mesh2(options, 1, mesh1.points());

  create_hex8_mesh(mesh1, 4, 4, 4);
  create_hex8_mesh(mesh2, 3, 5, 2);

  mesh2.create_side_ids_cut_test();

  Cut::plain_element_set elements_done;

  mesh2.cut(mesh1, elements_done);

  cutmesh(mesh1);

  mesh2.assign_other_volume_cells_cut_test(mesh1);
}

void test_cut_volumes2()
{
  for (int i = 2; i < 5; ++i)
  {
    for (int j = 2; j < 5; ++j)
    {
      for (int k = 2; k < 5; ++k)
      {
        Cut::Options options;
        options.init_for_cuttests();
        // this is meant to be used with matching boundaries. Thus, no
        // inside/outside positions.
        options.set_find_positions(false);

        Cut::Mesh mesh1(options);
        Cut::Mesh mesh2(options, 1, mesh1.points());

        create_hex8_mesh(mesh1, 1, 1, 1);
        create_hex8_mesh(mesh2, i, j, k);

        mesh2.create_side_ids_cut_test();

        Cut::plain_element_set elements_done;

        mesh2.cut(mesh1, elements_done);

        cutmesh(mesh1);

        mesh2.assign_other_volume_cells_cut_test(mesh1);
      }
    }
  }
}

void test_cut_volumes3()
{
  SimpleWrapper w;

  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  xyze(0, 0) = -1;
  xyze(1, 0) = -1;
  xyze(2, 0) = -1;

  xyze(0, 1) = 1;
  xyze(1, 1) = -1;
  xyze(2, 1) = -1;

  xyze(0, 2) = 1;
  xyze(1, 2) = 1;
  xyze(2, 2) = -1;

  xyze(0, 3) = -1;
  xyze(1, 3) = 1;
  xyze(2, 3) = -1;

  xyze(0, 4) = -1;
  xyze(1, 4) = -1;
  xyze(2, 4) = 1;

  xyze(0, 5) = 1;
  xyze(1, 5) = -1;
  xyze(2, 5) = 1;

  xyze(0, 6) = 1;
  xyze(1, 6) = 1;
  xyze(2, 6) = 1;

  xyze(0, 7) = -1;
  xyze(1, 7) = 1;
  xyze(2, 7) = 1;

  w.create_hex8(xyze);

  xyze(0, 0) = 0;
  xyze(1, 0) = -1;
  xyze(2, 0) = -1;

  xyze(0, 1) = 0;
  xyze(1, 1) = -1;
  xyze(2, 1) = 1;

  xyze(0, 2) = 0;
  xyze(1, 2) = 1;
  xyze(2, 2) = 1;

  xyze(0, 3) = 0;
  xyze(1, 3) = 1;
  xyze(2, 3) = -1;

  xyze(0, 4) = -0.5;
  xyze(1, 4) = 0;
  xyze(2, 4) = 0;

  w.create_pyramid5_sides(xyze);

  w.cut_test_cut();
}
