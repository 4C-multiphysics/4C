// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_element.hpp"
#include "4C_cut_levelsetintersection.hpp"
#include "4C_cut_mesh.hpp"
#include "4C_cut_options.hpp"

#include <iterator>

#include "cut_test_utils.hpp"

void test_ls_hex8_florian1()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = 0.331317;
  lsvs[1] = 0.331355;
  lsvs[2] = -0.0718546;
  lsvs[3] = -0.0719582;
  lsvs[4] = -1.06152;
  lsvs[5] = -1.06155;
  lsvs[6] = 0.178732;
  lsvs[7] = 0.17888;

  xyze(0, 0) = 92.3077;
  xyze(1, 0) = 122.481;
  xyze(2, 0) = 0.75;

  xyze(0, 1) = 92.3077;
  xyze(1, 1) = 122.481;
  xyze(2, 1) = -0.75;

  xyze(0, 2) = 92.3077;
  xyze(1, 2) = 124.031;
  xyze(2, 2) = -0.75;

  xyze(0, 3) = 92.3077;
  xyze(1, 3) = 124.031;
  xyze(2, 3) = 0.75;

  xyze(0, 4) = 93.8462;
  xyze(1, 4) = 122.481;
  xyze(2, 4) = 0.75;

  xyze(0, 5) = 93.8462;
  xyze(1, 5) = 122.481;
  xyze(2, 5) = -0.75;

  xyze(0, 6) = 93.8462;
  xyze(1, 6) = 124.031;
  xyze(2, 6) = -0.75;

  xyze(0, 7) = 93.8462;
  xyze(1, 7) = 124.031;
  xyze(2, 7) = 0.75;

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian2()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -1.06142;
  lsvs[1] = -1.06145;
  lsvs[2] = 0.178845;
  lsvs[3] = 0.178992;
  lsvs[4] = 0.331525;
  lsvs[5] = 0.331563;
  lsvs[6] = -0.071624;
  lsvs[7] = -0.0717273;

  xyze(0, 0) = 6.15385;
  xyze(1, 0) = 122.481;
  xyze(2, 0) = 0.75;

  xyze(0, 1) = 6.15385;
  xyze(1, 1) = 122.481;
  xyze(2, 1) = -0.75;

  xyze(0, 2) = 6.15385;
  xyze(1, 2) = 124.031;
  xyze(2, 2) = -0.75;

  xyze(0, 3) = 6.15385;
  xyze(1, 3) = 124.031;
  xyze(2, 3) = 0.75;

  xyze(0, 4) = 7.69231;
  xyze(1, 4) = 122.481;
  xyze(2, 4) = 0.75;

  xyze(0, 5) = 7.69231;
  xyze(1, 5) = 122.481;
  xyze(2, 5) = -0.75;

  xyze(0, 6) = 7.69231;
  xyze(1, 6) = 124.031;
  xyze(2, 6) = -0.75;

  xyze(0, 7) = 7.69231;
  xyze(1, 7) = 124.031;
  xyze(2, 7) = 0.75;

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian3()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -0.0524131;
  lsvs[1] = -0.0566379;
  lsvs[2] = 0.0747609;
  lsvs[3] = 0.0696167;
  lsvs[4] = 0.0151905;
  lsvs[5] = 0.00655406;
  lsvs[6] = -0.0626401;
  lsvs[7] = -0.11719;

  xyze(0, 0) = 49.2308;
  xyze(1, 0) = 100.775;
  xyze(2, 0) = 0.75;

  xyze(0, 1) = 49.2308;
  xyze(1, 1) = 100.775;
  xyze(2, 1) = -0.75;

  xyze(0, 2) = 49.2308;
  xyze(1, 2) = 102.326;
  xyze(2, 2) = -0.75;

  xyze(0, 3) = 49.2308;
  xyze(1, 3) = 102.326;
  xyze(2, 3) = 0.75;

  xyze(0, 4) = 50.7692;
  xyze(1, 4) = 100.775;
  xyze(2, 4) = 0.75;

  xyze(0, 5) = 50.7692;
  xyze(1, 5) = 100.775;
  xyze(2, 5) = -0.75;

  xyze(0, 6) = 50.7692;
  xyze(1, 6) = 102.326;
  xyze(2, 6) = -0.75;

  xyze(0, 7) = 50.7692;
  xyze(1, 7) = 102.326;
  xyze(2, 7) = 0.75;

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian4()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -0.0102494;
  lsvs[1] = 0.0126085;
  lsvs[2] = -0.0312269;
  lsvs[3] = 0.0440133;
  lsvs[4] = 0.0628933;
  lsvs[5] = 0.0952024;
  lsvs[6] = -0.366343;
  lsvs[7] = -0.322604;

  xyze(0, 0) = 47.0588;
  xyze(1, 0) = 109.091;
  xyze(2, 0) = 2.94118;

  xyze(0, 1) = 47.0588;
  xyze(1, 1) = 109.091;
  xyze(2, 1) = -2.94118;

  xyze(0, 2) = 47.0588;
  xyze(1, 2) = 115.152;
  xyze(2, 2) = -2.94118;

  xyze(0, 3) = 47.0588;
  xyze(1, 3) = 115.152;
  xyze(2, 3) = 2.94118;

  xyze(0, 4) = 52.9412;
  xyze(1, 4) = 109.091;
  xyze(2, 4) = 2.94118;

  xyze(0, 5) = 52.9412;
  xyze(1, 5) = 109.091;
  xyze(2, 5) = -2.94118;

  xyze(0, 6) = 52.9412;
  xyze(1, 6) = 115.152;
  xyze(2, 6) = -2.94118;

  xyze(0, 7) = 52.9412;
  xyze(1, 7) = 115.152;
  xyze(2, 7) = 2.94118;

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian5()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = 0.0986988;
  lsvs[1] = 0.00224424;
  lsvs[2] = -0.00388199;
  lsvs[3] = 0.17228;
  lsvs[4] = -0.116327;
  lsvs[5] = 0.0639577;
  lsvs[6] = 0.042944;
  lsvs[7] = -0.307789;

  xyze(0, 0) = 49.2308;
  xyze(1, 0) = 83.7209;
  xyze(2, 0) = 0.769231;

  xyze(0, 1) = 49.2308;
  xyze(1, 1) = 83.7209;
  xyze(2, 1) = -0.769231;

  xyze(0, 2) = 49.2308;
  xyze(1, 2) = 85.2713;
  xyze(2, 2) = -0.769231;

  xyze(0, 3) = 49.2308;
  xyze(1, 3) = 85.2713;
  xyze(2, 3) = 0.769231;

  xyze(0, 4) = 50.7692;
  xyze(1, 4) = 83.7209;
  xyze(2, 4) = 0.769231;

  xyze(0, 5) = 50.7692;
  xyze(1, 5) = 83.7209;
  xyze(2, 5) = -0.769231;

  xyze(0, 6) = 50.7692;
  xyze(1, 6) = 85.2713;
  xyze(2, 6) = -0.769231;

  xyze(0, 7) = 50.7692;
  xyze(1, 7) = 85.2713;
  xyze(2, 7) = 0.769231;

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian6()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -0.0780914;
  lsvs[1] = 0;
  lsvs[2] = -0.0798239;
  lsvs[3] = 0.214841;
  lsvs[4] = -0.390728;
  lsvs[5] = -0.381449;
  lsvs[6] = 0.29728;
  lsvs[7] = 0.333705;

  xyze(0, 0) = 49.2308;
  xyze(1, 0) = 83.7209;
  xyze(2, 0) = 0.769231;

  xyze(0, 1) = 49.2308;
  xyze(1, 1) = 83.7209;
  xyze(2, 1) = -0.769231;

  xyze(0, 2) = 49.2308;
  xyze(1, 2) = 85.2713;
  xyze(2, 2) = -0.769231;

  xyze(0, 3) = 49.2308;
  xyze(1, 3) = 85.2713;
  xyze(2, 3) = 0.769231;

  xyze(0, 4) = 50.7692;
  xyze(1, 4) = 83.7209;
  xyze(2, 4) = 0.769231;

  xyze(0, 5) = 50.7692;
  xyze(1, 5) = 83.7209;
  xyze(2, 5) = -0.769231;

  xyze(0, 6) = 50.7692;
  xyze(1, 6) = 85.2713;
  xyze(2, 6) = -0.769231;

  xyze(0, 7) = 50.7692;
  xyze(1, 7) = 85.2713;
  xyze(2, 7) = 0.769231;

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian7()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = 2.32342;
  lsvs[1] = 2.32636;
  lsvs[2] = -0.10882;
  lsvs[3] = -0.110644;
  lsvs[4] = 2.96376;
  lsvs[5] = 2.98289;
  lsvs[6] = 0;
  lsvs[7] = -0.00394479;

  xyze(0, 0) = 48.4848;
  xyze(1, 0) = 129.231;
  xyze(2, 0) = 1.51515;

  xyze(0, 1) = 48.4848;
  xyze(1, 1) = 129.231;
  xyze(2, 1) = -1.51515;

  xyze(0, 2) = 48.4848;
  xyze(1, 2) = 132.308;
  xyze(2, 2) = -1.51515;

  xyze(0, 3) = 48.4848;
  xyze(1, 3) = 132.308;
  xyze(2, 3) = 1.51515;

  xyze(0, 4) = 51.5152;
  xyze(1, 4) = 129.231;
  xyze(2, 4) = 1.51515;

  xyze(0, 5) = 51.5152;
  xyze(1, 5) = 129.231;
  xyze(2, 5) = -1.51515;

  xyze(0, 6) = 51.5152;
  xyze(1, 6) = 132.308;
  xyze(2, 6) = -1.51515;

  xyze(0, 7) = 51.5152;
  xyze(1, 7) = 132.308;
  xyze(2, 7) = 1.51515;

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian8()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -0.1999;
  lsvs[1] = -0.1999;
  lsvs[2] = -0.0999;
  lsvs[3] = -0.0999;
  lsvs[4] = -0.0999;
  lsvs[5] = -0.0999;
  lsvs[6] = 0.0001;
  lsvs[7] = 0.0001;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian9()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -2.93768;
  lsvs[1] = -2.93768;
  lsvs[2] = 0.0351257;
  lsvs[3] = 0.0351257;
  lsvs[4] = -4.03311;
  lsvs[5] = -4.03311;
  lsvs[6] = -1.09719;
  lsvs[7] = -1.09719;

  xyze(0, 0) = 33.3333;
  xyze(1, 0) = 95.3846;
  xyze(2, 0) = 1.51515;

  xyze(0, 1) = 33.3333;
  xyze(1, 1) = 95.3846;
  xyze(2, 1) = -1.51515;

  xyze(0, 2) = 33.3333;
  xyze(1, 2) = 98.4615;
  xyze(2, 2) = -1.51515;

  xyze(0, 3) = 33.3333;
  xyze(1, 3) = 98.4615;
  xyze(2, 3) = 1.51515;

  xyze(0, 4) = 36.3636;
  xyze(1, 4) = 95.3846;
  xyze(2, 4) = 1.51515;

  xyze(0, 5) = 36.3636;
  xyze(1, 5) = 95.3846;
  xyze(2, 5) = -1.51515;

  xyze(0, 6) = 36.3636;
  xyze(1, 6) = 98.4615;
  xyze(2, 6) = -1.51515;

  xyze(0, 7) = 36.3636;
  xyze(1, 7) = 98.4615;
  xyze(2, 7) = 1.51515;

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian10()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -0.482161;
  lsvs[1] = -0.482161;
  lsvs[2] = 0.608283;
  lsvs[3] = 0.608283;
  lsvs[4] = -1.57503;
  lsvs[5] = -1.57503;
  lsvs[6] = -0.531381;
  lsvs[7] = -0.53138;

  xyze(0, 0) = 46.1538;
  xyze(1, 0) = 77.5194;
  xyze(2, 0) = 0.769231;

  xyze(0, 1) = 46.1538;
  xyze(1, 1) = 77.5194;
  xyze(2, 1) = -0.769231;

  xyze(0, 2) = 46.1538;
  xyze(1, 2) = 79.0698;
  xyze(2, 2) = -0.769231;

  xyze(0, 3) = 46.1538;
  xyze(1, 3) = 79.0698;
  xyze(2, 3) = 0.769231;

  xyze(0, 4) = 47.6923;
  xyze(1, 4) = 77.5194;
  xyze(2, 4) = 0.769231;

  xyze(0, 5) = 47.6923;
  xyze(1, 5) = 77.5194;
  xyze(2, 5) = -0.769231;

  xyze(0, 6) = 47.6923;
  xyze(1, 6) = 79.0698;
  xyze(2, 6) = -0.769231;

  xyze(0, 7) = 47.6923;
  xyze(1, 7) = 79.0698;
  xyze(2, 7) = 0.769231;

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian11()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = 0.0294072;
  lsvs[1] = -0.000866295;
  lsvs[2] = 0.000429135;
  lsvs[3] = -0.0272108;
  lsvs[4] = 1.89476;
  lsvs[5] = 1.87922;
  lsvs[6] = 2.05645;
  lsvs[7] = 2.05328;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian12()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = 0.00157316;
  lsvs[1] = 0.00253598;
  lsvs[2] = -0.143268;
  lsvs[3] = -0.14371;
  lsvs[4] = -0.0122296;
  lsvs[5] = -0.0115942;
  lsvs[6] = 0.173587;
  lsvs[7] = 0.174208;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_florian13()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -0.117499;
  lsvs[1] = -0.117555;
  lsvs[2] = 0.398494;
  lsvs[3] = 0.398705;
  lsvs[4] = 0.150091;
  lsvs[5] = 0.150048;
  lsvs[6] = -0.132417;
  lsvs[7] = -0.132556;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_ursula1()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -5.60331e-18;
  lsvs[1] = 3.24413e-18;
  lsvs[2] = 0.0298305;
  lsvs[3] = 0.0298305;
  lsvs[4] = -0.00931143;
  lsvs[5] = -0.00931143;
  lsvs[6] = 0.0205881;
  lsvs[7] = 0.0205881;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_ursula2()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = 0.0280187;
  lsvs[1] = 0.0280187;
  lsvs[2] = 0.016173;
  lsvs[3] = 0.016173;
  lsvs[4] = 0.0123106;
  lsvs[5] = 0.0123106;
  lsvs[6] = -8.6e-09;
  lsvs[7] = -8.6e-09;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_ursula3()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = 0.0201562;
  lsvs[1] = 4.33333e-09;
  lsvs[2] = -0.0307418;
  lsvs[3] = -0.0084524;
  lsvs[4] = -0.0084524;
  lsvs[5] = -0.0307418;
  lsvs[6] = -0.0654792;
  lsvs[7] = -0.0401924;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_ursula4()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -0.0307418;
  lsvs[1] = -0.0654792;
  lsvs[2] = -0.0307418;
  lsvs[3] = 4.33333e-09;
  lsvs[4] = -0.0084524;
  lsvs[5] = -0.0401924;
  lsvs[6] = -0.0084524;
  lsvs[7] = 0.0201562;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_ursula5()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -0.0558438;
  lsvs[1] = -1.15154e-05;
  lsvs[2] = -0.062386;
  lsvs[3] = -0.114866;
  lsvs[4] = 0.0507185;
  lsvs[5] = 0.105143;
  lsvs[6] = 0.0485058;
  lsvs[7] = 0.0124904;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_ursula6()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = -0.009363004430403965;
  lsvs[1] = -0.009364717959862903;
  lsvs[2] = 0.02050769832923868;
  lsvs[3] = 0.02050765301070952;
  lsvs[4] = -2.010252267408201e-06;
  lsvs[5] = -3.562907829424899e-07;
  lsvs[6] = 0.02981111597008253;
  lsvs[7] = 0.0298114792053332;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_simple()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  std::fill(&lsvs[0], &lsvs[4], -1.);
  std::fill(&lsvs[4], &lsvs[8], 1.);

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_simple2()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8, -1);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[3] = 1;
  lsvs[7] = 1;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_simple3()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8, -1);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[3] = 1;
  lsvs[4] = 1;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_simple4()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8, 0);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = 1;
  lsvs[1] = 1;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_simple5()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8, 1);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = 0;
  lsvs[1] = 0;
  lsvs[2] = 0;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_simple6()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8, -1);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[2] = -.99981;

  lsvs[0] = 0.5;
  lsvs[1] = 0.43;
  lsvs[6] = 0.4123;
  // lsvs[7] = 0.300091;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void test_ls_hex8_simple7()
{
  for (int a = 0; a < 11; ++a)
  {
    double dy = 0.1 * a;

    Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
    lsi.get_options().init_for_cuttests();

    // simple hex8 element
    std::vector<int> nids(8);
    std::vector<double> lsvs(8);
    Core::LinAlg::SerialDenseMatrix xyze(3, 8);

    for (int i = 0; i < 8; ++i)
    {
      nids[i] = i;
    }

    lsvs[0] = 0 - dy;
    lsvs[1] = -1 - dy;
    lsvs[2] = 0 - dy;
    lsvs[3] = 1 - dy;
    lsvs[4] = 0 + dy;
    lsvs[5] = -1 + dy;
    lsvs[6] = 0 + dy;
    lsvs[7] = 1 + dy;

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

    lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
    lsi.cut();
  }
}

void test_ls_hex8_touch()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8, -1);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  // this is the impossible (undefined) case

  lsvs[4] = 0;
  lsvs[5] = 0;
  lsvs[6] = 0;
  lsvs[7] = 0;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}

void ls_hex8_node_value(
    int element, int node, int& nid, double& x, double& y, double& z, double& lsv)
{
  static int xpos[] = {-1, 1, 1, -1};
  static int ypos[] = {-1, -1, 1, 1};
  int base = node % 4;

  nid = node + 4 * element;
  x = xpos[base];
  y = ypos[base];
  z = node / 4 + element;
  lsv = z - 1 + 0.5 * REFERENCETOL;
  // lsv = z - 1 + 0.5;
}

void test_ls_hex8_between()
{
  for (int e = 0; e < 2; ++e)
  {
    Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
    lsi.get_options().init_for_cuttests();

    // simple hex8 element
    std::vector<int> nids(8);
    std::vector<double> lsvs(8, -1);
    Core::LinAlg::SerialDenseMatrix xyze(3, 8);

    for (int i = 0; i < 8; ++i)
    {
      ls_hex8_node_value(e, i, nids[i], xyze(0, i), xyze(1, i), xyze(2, i), lsvs[i]);
    }

    // std::copy( lsvs.begin(), lsvs.end(), std::ostream_iterator<double>( std::cout, " " ) );
    // std::cout << "\n";

    lsi.add_element(e, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
    lsi.cut(true, false, Cut::VCellGaussPts_DirectDivergence);
  }
}

void test_ls_hex8_experiment()
{
  Cut::LevelSetIntersection lsi(MPI_COMM_WORLD);
  lsi.get_options().init_for_cuttests();

  Cut::Options options;
  lsi.get_options(options);
  options.set_simple_shapes(false);

  // simple hex8 element
  std::vector<int> nids(8);
  std::vector<double> lsvs(8);
  Core::LinAlg::SerialDenseMatrix xyze(3, 8);

  for (int i = 0; i < 8; ++i)
  {
    nids[i] = i;
  }

  lsvs[0] = 1;
  lsvs[1] = 1;
  lsvs[2] = -1;
  lsvs[3] = -1;
  lsvs[4] = 1;
  lsvs[5] = -1;
  lsvs[6] = 1;
  lsvs[7] = -1;

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

  lsi.add_element(1, nids, xyze, Core::FE::CellType::hex8, &lsvs[0]);
  lsi.cut();
}
