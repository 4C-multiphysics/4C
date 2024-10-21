#include "4C_cut_combintersection.hpp"
#include "4C_cut_levelsetintersection.hpp"
#include "4C_cut_meshintersection.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_tetmeshintersection.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cut_test_utils.hpp"

void test_generated_75358()
{
  Cut::MeshIntersection intersection;
  intersection.GetOptions().Init_for_Cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.1109345100177161203;
    tri3_xyze(1, 0) = 0.13707227871460303525;
    tri3_xyze(2, 0) = 0.069999999999995080047;
    nids.push_back(66994);
    tri3_xyze(0, 1) = 0.10222895637890296039;
    tri3_xyze(1, 1) = 0.12677166975466452881;
    tri3_xyze(2, 1) = 0.069999999999987308485;
    nids.push_back(66254);
    tri3_xyze(0, 2) = 0.11340783113674074434;
    tri3_xyze(1, 2) = 0.12640161558345713866;
    tri3_xyze(2, 2) = 0.0699999999999913608;
    nids.push_back(-10622);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.10222895637890296039;
    tri3_xyze(1, 0) = 0.12677166975466452881;
    tri3_xyze(2, 0) = 0.069999999999987308485;
    nids.push_back(66254);
    tri3_xyze(0, 1) = 0.1109345100177161203;
    tri3_xyze(1, 1) = 0.13707227871460303525;
    tri3_xyze(2, 1) = 0.069999999999995080047;
    nids.push_back(66994);
    tri3_xyze(0, 2) = 0.099473874168712145272;
    tri3_xyze(1, 2) = 0.13715544790944794729;
    tri3_xyze(2, 2) = 0.069999999999991208144;
    nids.push_back(-10623);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.10606601717800392959;
    tri3_xyze(1, 0) = 0.10606601717800288875;
    tri3_xyze(2, 0) = 0.069999999999997647437;
    nids.push_back(43350);
    tri3_xyze(0, 1) = 0.093523470278829234914;
    tri3_xyze(1, 1) = 0.11727472237022774915;
    tri3_xyze(2, 1) = 0.06999999999999763356;
    nids.push_back(43354);
    tri3_xyze(0, 2) = 0.099794743728413085049;
    tri3_xyze(1, 2) = 0.11167036977411193277;
    tri3_xyze(2, 2) = 0.082499999999975648368;
    nids.push_back(-12199);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.093523470278829234914;
    tri3_xyze(1, 0) = 0.11727472237022774915;
    tri3_xyze(2, 0) = 0.06999999999999763356;
    nids.push_back(43354);
    tri3_xyze(0, 1) = 0.093523470278822712354;
    tri3_xyze(1, 1) = 0.11727472237022047719;
    tri3_xyze(2, 1) = 0.094999999999953635421;
    nids.push_back(68056);
    tri3_xyze(0, 2) = 0.099794743728413085049;
    tri3_xyze(1, 2) = 0.11167036977411193277;
    tri3_xyze(2, 2) = 0.082499999999975648368;
    nids.push_back(-12199);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.093523470278822712354;
    tri3_xyze(1, 0) = 0.11727472237022047719;
    tri3_xyze(2, 0) = 0.094999999999953635421;
    nids.push_back(68056);
    tri3_xyze(0, 1) = 0.10606601717799646334;
    tri3_xyze(1, 1) = 0.10606601717799662987;
    tri3_xyze(2, 1) = 0.09499999999995370481;
    nids.push_back(68054);
    tri3_xyze(0, 2) = 0.099794743728413085049;
    tri3_xyze(1, 2) = 0.11167036977411193277;
    tri3_xyze(2, 2) = 0.082499999999975648368;
    nids.push_back(-12199);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.1156316801243056358;
    tri3_xyze(1, 0) = 0.11549157211239709231;
    tri3_xyze(2, 0) = 0.06999999999998740563;
    nids.push_back(66253);
    tri3_xyze(0, 1) = 0.10222895637890296039;
    tri3_xyze(1, 1) = 0.12677166975466452881;
    tri3_xyze(2, 1) = 0.069999999999987308485;
    nids.push_back(66254);
    tri3_xyze(0, 2) = 0.10436253099001044364;
    tri3_xyze(1, 2) = 0.11640099535382306128;
    tri3_xyze(2, 2) = 0.069999999999992498778;
    nids.push_back(-5013);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.10222895637890296039;
    tri3_xyze(1, 0) = 0.12677166975466452881;
    tri3_xyze(2, 0) = 0.069999999999987308485;
    nids.push_back(66254);
    tri3_xyze(0, 1) = 0.093523470278829234914;
    tri3_xyze(1, 1) = 0.11727472237022774915;
    tri3_xyze(2, 1) = 0.06999999999999763356;
    nids.push_back(43354);
    tri3_xyze(0, 2) = 0.10436253099001044364;
    tri3_xyze(1, 2) = 0.11640099535382306128;
    tri3_xyze(2, 2) = 0.069999999999992498778;
    nids.push_back(-5013);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.093523470278829234914;
    tri3_xyze(1, 0) = 0.11727472237022774915;
    tri3_xyze(2, 0) = 0.06999999999999763356;
    nids.push_back(43354);
    tri3_xyze(0, 1) = 0.10606601717800392959;
    tri3_xyze(1, 1) = 0.10606601717800288875;
    tri3_xyze(2, 1) = 0.069999999999997647437;
    nids.push_back(43350);
    tri3_xyze(0, 2) = 0.10436253099001044364;
    tri3_xyze(1, 2) = 0.11640099535382306128;
    tri3_xyze(2, 2) = 0.069999999999992498778;
    nids.push_back(-5013);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.093523470278829234914;
    tri3_xyze(1, 0) = 0.11727472237022774915;
    tri3_xyze(2, 0) = 0.06999999999999763356;
    nids.push_back(43354);
    tri3_xyze(0, 1) = 0.10222895637890296039;
    tri3_xyze(1, 1) = 0.12677166975466452881;
    tri3_xyze(2, 1) = 0.069999999999987308485;
    nids.push_back(66254);
    tri3_xyze(0, 2) = 0.090918717785945069032;
    tri3_xyze(1, 2) = 0.12703330088892686445;
    tri3_xyze(2, 2) = 0.069999999999992623678;
    nids.push_back(-5014);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.10222895637890296039;
    tri3_xyze(1, 0) = 0.12677166975466452881;
    tri3_xyze(2, 0) = 0.069999999999987308485;
    nids.push_back(66254);
    tri3_xyze(0, 1) = 0.088117633008731377497;
    tri3_xyze(1, 1) = 0.13707818154654718978;
    tri3_xyze(2, 1) = 0.069999999999987697064;
    nids.push_back(66255);
    tri3_xyze(0, 2) = 0.090918717785945069032;
    tri3_xyze(1, 2) = 0.12703330088892686445;
    tri3_xyze(2, 2) = 0.069999999999992623678;
    nids.push_back(-5014);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.079804811477316717205;
    tri3_xyze(1, 0) = 0.12700862988426800393;
    tri3_xyze(2, 0) = 0.069999999999997841726;
    nids.push_back(43358);
    tri3_xyze(0, 1) = 0.093523470278829234914;
    tri3_xyze(1, 1) = 0.11727472237022774915;
    tri3_xyze(2, 1) = 0.06999999999999763356;
    nids.push_back(43354);
    tri3_xyze(0, 2) = 0.090918717785945069032;
    tri3_xyze(1, 2) = 0.12703330088892686445;
    tri3_xyze(2, 2) = 0.069999999999992623678;
    nids.push_back(-5014);
    intersection.AddCutSide(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.10067694839195882872;
    hex8_xyze(1, 0) = 0.12559552859795883317;
    hex8_xyze(2, 0) = 0.070000000000000284217;
    nids.push_back(84506);
    hex8_xyze(0, 1) = 0.095536134220700688613;
    hex8_xyze(1, 1) = 0.1288531085323437464;
    hex8_xyze(2, 1) = 0.070000000000000284217;
    nids.push_back(84508);
    hex8_xyze(0, 2) = 0.091233824131263088764;
    hex8_xyze(1, 2) = 0.12277029364375977027;
    hex8_xyze(2, 2) = 0.070000000000000284217;
    nids.push_back(84540);
    hex8_xyze(0, 3) = 0.09622673700752132353;
    hex8_xyze(1, 3) = 0.11970644102578015255;
    hex8_xyze(2, 3) = 0.070000000000000284217;
    nids.push_back(84538);
    hex8_xyze(0, 4) = 0.10067694839195878709;
    hex8_xyze(1, 4) = 0.12559552859795888868;
    hex8_xyze(2, 4) = 0.089999999999999968914;
    nids.push_back(84507);
    hex8_xyze(0, 5) = 0.09553613422070064698;
    hex8_xyze(1, 5) = 0.12885310853234380191;
    hex8_xyze(2, 5) = 0.089999999999999968914;
    nids.push_back(84509);
    hex8_xyze(0, 6) = 0.09123382413126304713;
    hex8_xyze(1, 6) = 0.12277029364375982579;
    hex8_xyze(2, 6) = 0.089999999999999968914;
    nids.push_back(84541);
    hex8_xyze(0, 7) = 0.096226737007521281897;
    hex8_xyze(1, 7) = 0.11970644102578020807;
    hex8_xyze(2, 7) = 0.089999999999999968914;
    nids.push_back(84539);

    intersection.add_element(75359, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.10114857734509359599;
    hex8_xyze(1, 0) = 0.11655251697512837572;
    hex8_xyze(2, 0) = 0.070000000000000284217;
    nids.push_back(84536);
    hex8_xyze(0, 1) = 0.09622673700752132353;
    hex8_xyze(1, 1) = 0.11970644102578015255;
    hex8_xyze(2, 1) = 0.070000000000000284217;
    nids.push_back(84538);
    hex8_xyze(0, 2) = 0.09177652562308374895;
    hex8_xyze(1, 2) = 0.11381735345360149969;
    hex8_xyze(2, 2) = 0.070000000000000284217;
    nids.push_back(84570);
    hex8_xyze(0, 3) = 0.09656231008877048938;
    hex8_xyze(1, 3) = 0.11087216862480045454;
    hex8_xyze(2, 3) = 0.070000000000000284217;
    nids.push_back(84568);
    hex8_xyze(0, 4) = 0.10114857734509355436;
    hex8_xyze(1, 4) = 0.11655251697512843123;
    hex8_xyze(2, 4) = 0.089999999999999968914;
    nids.push_back(84537);
    hex8_xyze(0, 5) = 0.096226737007521281897;
    hex8_xyze(1, 5) = 0.11970644102578020807;
    hex8_xyze(2, 5) = 0.089999999999999968914;
    nids.push_back(84539);
    hex8_xyze(0, 6) = 0.091776525623083707317;
    hex8_xyze(1, 6) = 0.1138173534536015552;
    hex8_xyze(2, 6) = 0.089999999999999968914;
    nids.push_back(84571);
    hex8_xyze(0, 7) = 0.096562310088770447747;
    hex8_xyze(1, 7) = 0.11087216862480051005;
    hex8_xyze(2, 7) = 0.089999999999999968914;
    nids.push_back(84569);

    intersection.add_element(75373, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.10573484460141675811;
    hex8_xyze(1, 0) = 0.12223286532545636629;
    hex8_xyze(2, 0) = 0.050000000000000113798;
    nids.push_back(106578);
    hex8_xyze(0, 1) = 0.10067694839195878709;
    hex8_xyze(1, 1) = 0.12559552859795883317;
    hex8_xyze(2, 1) = 0.050000000000000113798;
    nids.push_back(106579);
    hex8_xyze(0, 2) = 0.096226737007521281897;
    hex8_xyze(1, 2) = 0.11970644102578015255;
    hex8_xyze(2, 2) = 0.050000000000000113798;
    nids.push_back(106595);
    hex8_xyze(0, 3) = 0.10114857734509355436;
    hex8_xyze(1, 3) = 0.11655251697512837572;
    hex8_xyze(2, 3) = 0.050000000000000113798;
    nids.push_back(106594);
    hex8_xyze(0, 4) = 0.10573484460141679975;
    hex8_xyze(1, 4) = 0.12223286532545636629;
    hex8_xyze(2, 4) = 0.070000000000000284217;
    nids.push_back(84504);
    hex8_xyze(0, 5) = 0.10067694839195882872;
    hex8_xyze(1, 5) = 0.12559552859795883317;
    hex8_xyze(2, 5) = 0.070000000000000284217;
    nids.push_back(84506);
    hex8_xyze(0, 6) = 0.09622673700752132353;
    hex8_xyze(1, 6) = 0.11970644102578015255;
    hex8_xyze(2, 6) = 0.070000000000000284217;
    nids.push_back(84538);
    hex8_xyze(0, 7) = 0.10114857734509359599;
    hex8_xyze(1, 7) = 0.11655251697512837572;
    hex8_xyze(2, 7) = 0.070000000000000284217;
    nids.push_back(84536);

    intersection.add_element(87058, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.11032111185773997575;
    hex8_xyze(1, 0) = 0.12791321367578428747;
    hex8_xyze(2, 0) = 0.070000000000000284217;
    nids.push_back(84472);
    hex8_xyze(0, 1) = 0.10512715977639641718;
    hex8_xyze(1, 1) = 0.13148461617013745828;
    hex8_xyze(2, 1) = 0.070000000000000284217;
    nids.push_back(84474);
    hex8_xyze(0, 2) = 0.10067694839195882872;
    hex8_xyze(1, 2) = 0.12559552859795883317;
    hex8_xyze(2, 2) = 0.070000000000000284217;
    nids.push_back(84506);
    hex8_xyze(0, 3) = 0.10573484460141679975;
    hex8_xyze(1, 3) = 0.12223286532545636629;
    hex8_xyze(2, 3) = 0.070000000000000284217;
    nids.push_back(84504);
    hex8_xyze(0, 4) = 0.11032111185773993411;
    hex8_xyze(1, 4) = 0.12791321367578434298;
    hex8_xyze(2, 4) = 0.089999999999999968914;
    nids.push_back(84473);
    hex8_xyze(0, 5) = 0.10512715977639637555;
    hex8_xyze(1, 5) = 0.13148461617013751379;
    hex8_xyze(2, 5) = 0.089999999999999968914;
    nids.push_back(84475);
    hex8_xyze(0, 6) = 0.10067694839195878709;
    hex8_xyze(1, 6) = 0.12559552859795888868;
    hex8_xyze(2, 6) = 0.089999999999999968914;
    nids.push_back(84507);
    hex8_xyze(0, 7) = 0.10573484460141675811;
    hex8_xyze(1, 7) = 0.1222328653254564218;
    hex8_xyze(2, 7) = 0.089999999999999968914;
    nids.push_back(84505);

    intersection.add_element(75343, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.10573484460141679975;
    hex8_xyze(1, 0) = 0.12223286532545636629;
    hex8_xyze(2, 0) = 0.070000000000000284217;
    nids.push_back(84504);
    hex8_xyze(0, 1) = 0.10067694839195882872;
    hex8_xyze(1, 1) = 0.12559552859795883317;
    hex8_xyze(2, 1) = 0.070000000000000284217;
    nids.push_back(84506);
    hex8_xyze(0, 2) = 0.09622673700752132353;
    hex8_xyze(1, 2) = 0.11970644102578015255;
    hex8_xyze(2, 2) = 0.070000000000000284217;
    nids.push_back(84538);
    hex8_xyze(0, 3) = 0.10114857734509359599;
    hex8_xyze(1, 3) = 0.11655251697512837572;
    hex8_xyze(2, 3) = 0.070000000000000284217;
    nids.push_back(84536);
    hex8_xyze(0, 4) = 0.10573484460141675811;
    hex8_xyze(1, 4) = 0.1222328653254564218;
    hex8_xyze(2, 4) = 0.089999999999999968914;
    nids.push_back(84505);
    hex8_xyze(0, 5) = 0.10067694839195878709;
    hex8_xyze(1, 5) = 0.12559552859795888868;
    hex8_xyze(2, 5) = 0.089999999999999968914;
    nids.push_back(84507);
    hex8_xyze(0, 6) = 0.096226737007521281897;
    hex8_xyze(1, 6) = 0.11970644102578020807;
    hex8_xyze(2, 6) = 0.089999999999999968914;
    nids.push_back(84539);
    hex8_xyze(0, 7) = 0.10114857734509355436;
    hex8_xyze(1, 7) = 0.11655251697512843123;
    hex8_xyze(2, 7) = 0.089999999999999968914;
    nids.push_back(84537);

    intersection.add_element(75358, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  intersection.CutTest_Cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.Cut_Finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
