// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_cut_element.hpp"
#include "4C_cut_mesh.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_global_data.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <fenv.h>
#include <Kokkos_Core.hpp>
#include <Teuchos_CommandLineProcessor.hpp>

#include <map>
#include <sstream>
#include <string>

#include "cut_test_utils.hpp"

void test_generated_26182();
void test_generated_238343();
void test_generated_369096();
void test_generated_79216();
void test_generated_197489();
void test_generated_238425();

void test_generated_622320();
void test_generated_622829();
void test_generated_627558();

void test_generated_7022();
void test_generated_227469();
void test_generated_463638();
void test_generated_43244();
void test_generated_41534();
void test_generated_6923();
void test_generated_7019();
void test_generated_6920();
void test_generated_6890();
void test_generated_1890();
void test_generated_1858();
void test_generated_1860();
void test_generated_2010();
void test_generated_1901();
void test_generated_1970();
void test_generated_1910();

void test_hex8_simple();
void test_tet4_simple();
void test_pyramid5_simple();
void test_wedge6_simple();
void test_hex8_fullside();
void test_hex8_diagonal();
void test_hex8_tet4();
void test_hex8_hex8();
void test_hex8_touch();
void test_hex8_touch2();
void test_hex8_schraeg();
void test_hex8_quad4_woelbung();
void test_hex8_tet4_touch();
void test_hex8_tet4_touch2();
void test_hex8_mesh();
void test_hex8_double();
void test_hex8_multiple();
void test_hex8_bad1();
void test_hex8_bad2();
void test_hex8_bad3();
void test_hex8_bad4();
void test_hex8_wedge6();
void test_hex8_quad4_touch();
void test_hex8_quad4_touch2();
void test_hex8_quad4_touch3();
void test_hex8_quad4_cut();
void test_hex8_quad4_gedreht();
void test_hex8_hex8_durchstoss();
void test_hex8_hex8_onside();
void test_hex8_hex8_internal();
void test_hex8_hex8_sideintersection();
void test_hex8_hex8_inside();
void test_hex8_quad4_schnitt();
void test_hex8_quad4_touch4();
void test_hex8_quad4_touch5();
void test_hex8_quad4_touch6();
void test_hex8_quad4_touch7();
void test_hex8_quad4_mesh();
void test_position2d();
void test_facet_split();


void test_quad4_line2();
void test_hex8_quad4_qhull1();
void test_hex8_quad4_alex1();
void test_hex8_quad4_alex2();
void test_hex8_quad4_alex3();
void test_hex8_quad4_alex4();
void test_hex8_quad4_alex5();
void test_hex8_quad4_alex6();
void test_hex8_quad4_alex7();
void test_hex8_quad4_alex8();
void test_tet4_quad4_alex9();
void test_tet4_quad4_alex10();
void test_tet4_quad4_alex11();
void test_hex8_quad4_alex12();
void test_hex8_quad4_alex13();
void test_hex8_quad4_alex14();
void test_hex8_quad4_alex15();
void test_tet4_quad4_alex16();
void test_hex8_quad4_alex17();
void test_hex8_quad4_alex18();
void test_hex8_quad4_alex19();
void test_hex8_quad4_alex20();
void test_hex8_quad4_alex21();
void test_hex8_quad4_alex22();
void test_hex8_quad4_alex23();
void test_hex8_quad4_alex24();
void test_hex8_quad4_alex25();
void test_hex8_quad4_alex26();
void test_hex8_quad4_alex27();
void test_hex8_quad4_alex28();
void test_hex8_quad4_alex29();
void test_hex8_quad4_alex30();
void test_hex8_quad4_alex31();
void test_hex8_quad4_alex32();
void test_hex8_quad4_alex33();
void test_hex8_quad4_alex34();
void test_hex8_quad4_alex35();
void test_hex8_quad4_alex36();
void test_hex8_quad4_alex37();
void test_hex8_quad4_alex38();
void test_hex8_twintri();
void test_hex8_twin_quad();
void test_hex8_chair_cut();
void test_hex8_v_cut();
void test_alex39();
void test_alex40();
void test_alex41();
void test_alex42();
void test_alex43();
void test_alex44();
void test_alex45();
void test_alex46();
void test_alex47();
void test_alex48();
void test_alex49();
void test_alex50();
void test_alex51();
void test_alex52();
void test_alex53();
void test_alex54();
void test_alex55();
void test_alex56();
void test_alex57();
void test_alex58();
void test_alex59();
void test_alex60();
void test_alex61();
void test_alex62();
void test_hex8_quad4_axel1();
void test_hex8_quad4_axel2();
void test_hex8_quad4_axel3();
void test_hex8_quad4_axel4();
void test_hex8_quad4_axel5();
void test_hex8_quad4_axel6();
void test_hex8_quad4_axel7();
void test_axel8();
void test_axel9();
void test_axel10();
void test_hex8_quad4_shadan1();
void test_hex8_quad4_shadan2();
void test_hex8_quad4_shadan3();
void test_hex8_quad4_shadan4();
void test_hex8_quad4_shadan5();
void test_shadan6();
void test_hex8_tri3_ursula1();
void test_hex8_quad4_mesh_many();
void test_hex8_quad4_mesh_edgecut();
void test_hex8_quad4_mesh_edgecut2();
void test_hex8_quad4_mesh_inner();
void test_hex27_quad9_simple();
void test_hex20_quad9_simple();
void test_hex20_quad9_moved();
void test_tet10_quad9_simple();
void test_tet10_quad9_moved();
void test_tet4_quad4_double();
void test_tet4_tri3_double();
void test_benedikt1();
void test_christoph_1();

void test_ls_hex8_florian1();
void test_ls_hex8_florian2();
void test_ls_hex8_florian3();
void test_ls_hex8_florian4();
void test_ls_hex8_florian5();
void test_ls_hex8_florian6();
void test_ls_hex8_florian7();
void test_ls_hex8_florian8();
void test_ls_hex8_florian9();
void test_ls_hex8_florian10();
void test_ls_hex8_florian11();
void test_ls_hex8_florian12();
void test_ls_hex8_florian13();
void test_ls_hex8_ursula1();
void test_ls_hex8_ursula2();
void test_ls_hex8_ursula3();
void test_ls_hex8_ursula4();
void test_ls_hex8_ursula5();
void test_ls_hex8_ursula6();
void test_ls_hex8_simple();
void test_ls_hex8_simple2();
void test_ls_hex8_simple3();
void test_ls_hex8_simple4();
void test_ls_hex8_simple5();
void test_ls_hex8_simple6();
void test_ls_hex8_simple7();
void test_ls_hex8_touch();
void test_ls_hex8_between();
void test_ls_hex8_experiment();
void test_ls_hex8_experiment_magnus();
void test_ls_mesh_hex8_simple();  // Same cut with LS and mesh
void test_ls_hex8_magnus1();      // Loss of volume-cell     (prec 24)
void test_ls_hex8_magnus12();     // Not loss of volume cell (prec 16)
void test_ls_hex8_magnus2();
void test_ls_hex8_magnus3();
void test_ls_hex8_magnus4();
void test_ls_hex8_magnus5();  // Qhull QbB input fail in sphere
void test_ls_hex8_magnus6();  // Variable surftens issue with Combust -> Issue with cut in local
                              // coord
void test_ls_hex8_magnus7();  // Problem with Global Cut with DD
void test_ls_hex8_tes_dd_simple();


void test_hex8_quad4_double_cut();

void test_unit_intersection_touch();

void test_geometry();

void test_cut_volumes();
void test_cut_volumes2();
void test_cut_volumes3();

void test_fluidfluid();
void test_fluidfluid2();

//--- test cases for self-cut library---
void test_hex8quad4selfcut20();
void test_hex8quad4selfcut21();
void test_hex8quad4selfcut22();
void test_hex8quad4selfcut23();
void test_hex8quad4selfcut24();

void test_hex8quad4selfcut30();
void test_hex8quad4selfcut31();
void test_hex8quad4selfcut32();
void test_hex8quad4selfcut33();
void test_hex8quad4selfcut34();
void test_hex8quad4selfcut35();
void test_hex8quad4selfcut36();
void test_hex8quad4selfcut37();
void test_hex8quad4selfcut38();
void test_hex8quad4selfcut39();

void test_hex8quad4selfcut41();
void test_hex8quad4selfcut42();
void test_hex8quad4selfcut43();

void test_hex8quad4selfcut51();
void test_hex8quad4selfcut52();
void test_hex8quad4selfcut53();

void test_hex8quad4selfcut61();
void test_hex8quad4selfcut62();
void test_hex8quad4selfcut63();
void test_hex8quad4selfcut64();
void test_hex8quad4selfcut65();
void test_hex8quad4selfcut66();

void test_hex8quad4selfcut71();
void test_hex8quad4selfcut72();

void test_hex8quad4selfcut81();
void test_hex8quad4selfcut82();
void test_hex8quad4selfcut83();
void test_hex8quad4selfcut84();
void test_hex8quad4selfcut85();
void test_hex8quad4selfcut86();

void test_hex8quad4selfcut91();
void test_hex8quad4selfcut92();

void test_hex8quad4aligned_edges();

typedef void (*testfunct)();

/**
 * \brief Run a given test and store errors if necessary.
 *
 * @param iterator Iterator of a map with the cuttest functions.
 * @param failures Vector that stores the errors if they occur.
 * @param msgs Vector that stores an error message.
 */
void testfunction(std::map<std::string, testfunct>::const_iterator iterator,
    std::vector<std::string>& failures, std::vector<std::string>& msgs)
{
  try
  {
    std::cout << "Testing " << iterator->first << " ...\n";
    (*iterator->second)();
  }
  catch (Core::Exception& err)
  {
    std::cout << "FAILED: " << err.what_with_stacktrace() << "\n";
    failures.push_back(iterator->first);
    msgs.push_back(err.what_with_stacktrace());
  }
}

/**
 * \brief Run the cuttests.
 *
 * @param argv
 * @param functable Map that stores all cuttest with a string key.
 * @param testname If this string is different than "(all)", the function runs the cuttests that
 * contain this string.
 * @param ignore_testname If this string is defined, the function doesn't run the cuttests that
 * contain this string.
 */
int runtests(char** argv, std::map<std::string, testfunct>& functable, std::string testname,
    std::string ignore_testname)
{
  // Check if a test name was specified. The default case of testname is (all)
  bool select_testcases;
  if (testname == "(all)")
    select_testcases = false;
  else
    select_testcases = true;

  // Remove tests that should be ignored from functable
  if (!ignore_testname.empty())
  {
    for (std::map<std::string, testfunct>::const_iterator iterator = functable.begin();
        iterator != functable.end();)
    {
      if (iterator->first.find(ignore_testname) != std::string::npos)
        iterator = functable.erase(iterator);
      else
        ++iterator;
    }
  }

  // Define helper variables and the final functable size
  std::vector<std::string> failures;
  std::vector<std::string> msgs;
  int counter_tests = 0;
  int number_tests = functable.size();

  if (select_testcases)
  {
    bool testname_exist = false;

    for (std::map<std::string, testfunct>::const_iterator iterator = functable.begin();
        iterator != functable.end(); iterator++)
    {
      if (iterator->first.find(testname) != std::string::npos)
      {
        counter_tests++;

        testname_exist = true;
        testfunction(iterator, failures, msgs);
      }
      else if ((testname_exist == false) && (counter_tests == number_tests))
      {
        std::cerr << argv[0] << ": test '" << testname << "' not found\n";
        return 1;
      }
    }
  }
  else
  {
    for (std::map<std::string, testfunct>::const_iterator iterator = functable.begin();
        iterator != functable.end(); ++iterator)
    {
      counter_tests++;
      testfunction(iterator, failures, msgs);
    }
  }

  // Show failed tests if there are any
  if (failures.size() > 0)
  {
    std::cout << "\n" << failures.size() << " out of " << counter_tests << " tests failed.\n";
    for (std::vector<std::string>::iterator i = failures.begin(); i != failures.end(); ++i)
    {
      std::string& txt = *i;
      std::cout << "    " << txt;
      for (unsigned j = 0; j < 40 - txt.length(); ++j) std::cout << " ";
      std::cout << "(" << msgs[i - failures.begin()] << ")"
                << "\n";
    }
  }
  else
  {
    std::cout << "\nall " << counter_tests << " tests succeeded.\n";
  }

  return failures.size();
}

/**
 * \brief Set the dimension and the parameters for this problem.
 *
 */
void set_problem_dimension(const std::map<std::string, testfunct>& functable)
{
  Global::Problem& problem = (*Global::Problem::instance());
  std::shared_ptr<Teuchos::ParameterList> pptr = std::make_shared<Teuchos::ParameterList>();
  Teuchos::ParameterList& size_params = pptr->sublist("PROBLEM SIZE", false);
  int probdim = 3;
  //  for ( std::map<std::string, testfunct>::const_iterator cit=functable.begin();
  //      cit!=functable.end(); ++cit )
  //  {
  //
  //  }
  size_params.set<int>("DIM", probdim);

  // set the parameter list in the global problem
  problem.set_parameter_list(pptr);
}

/**
 * \brief Main function that defines and runs the cuttests.
 *
 */
int main(int argc, char** argv)
{
  Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  MPI_Init(&argc, &argv);
  Kokkos::ScopeGuard kokkos_guard{};

  std::map<std::string, testfunct> functable;

  Core::IO::cout.setup(
      false, false, false, Core::IO::standard, MPI_COMM_WORLD, 0, 0, "dummyFilePrefix");

  // these tests were generated from the real computation
  // all of then expreience the same problem:
  // "Could not find reference plane with all gausspoints inside!" in
  // src/cut/direct_divergence_refplane.cpp
  functable["cluster_comp_fail1"] = test_generated_622829;
  // functable["cluster_comp_fail2"] = test_generated_627558;
  functable["cluster_comp_fail3"] = test_generated_622320;

  functable["touching_hole"] = test_generated_79216;

  // tests to test small edge cases in triangulation
  // they fail, but due to other problems, not related to triangulation
  functable["narrow_triangulation"] = test_generated_369096;
  functable["split_facet"] = test_generated_238343;

  // selfcut 3d-Island detection
  functable["selfcut_island"] = test_generated_26182;

  // tests that failed, due to the problem with splitting surface with holes
  // in the colored
  functable["split_colored_graph_1"] = test_generated_197489;
  functable["split_colored_graph_2"] = test_generated_238425;

  // all the tests below were generated from testing of a rotating and slowly moving (with very
  // small displacement) tori in the rectangular mesh in which they failed on the previous version
  // of the cut. Now they must run fine
  functable["touching_failed"] = test_generated_7022;
  functable["failing_cut_kernel"] = test_generated_463638;
  functable["tolerance_mismatch"] = test_generated_6923;
  functable["failing_topology"] = test_generated_1890;
  functable["find_node_position"] = test_generated_1858;
  functable["failing_tolerance"] = test_generated_2010;
  functable["failing_cln_facetgraph"] = test_generated_1901;
  functable["failing_cln_doublearc1910"] = test_generated_1910;
  functable["failing_cln_doublearc1970"] = test_generated_1970;
  functable["failing_tori1"] = test_generated_41534;
  functable["failing_tori2"] = test_generated_43244;
  functable["failing_tori3"] = test_generated_7019;
  functable["failing_tori4"] = test_generated_6920;
  functable["failing_tori5"] = test_generated_6890;
  functable["failing_tori6"] = test_generated_1860;
  functable["failing_tori7"] = test_generated_227469;

  functable["hex8_simple"] = test_hex8_simple;
  functable["tet4_simple"] = test_tet4_simple;
  functable["pyramid5_simple"] = test_pyramid5_simple;
  functable["wedge6_simple"] = test_wedge6_simple;
  functable["hex8_diagonal"] = test_hex8_diagonal;
  functable["hex8_fullside"] = test_hex8_fullside;
  functable["hex8_hex8"] = test_hex8_hex8;
  functable["hex8_tet4"] = test_hex8_tet4;
  functable["hex8_touch"] = test_hex8_touch;
  functable["hex8_touch2"] = test_hex8_touch2;
  functable["hex8_schraeg"] = test_hex8_schraeg;
  functable["hex8_quad4_woelbung"] = test_hex8_quad4_woelbung;
  functable["hex8_tet4_touch"] = test_hex8_tet4_touch;
  functable["hex8_tet4_touch2"] = test_hex8_tet4_touch2;
  functable["hex8_mesh"] = test_hex8_mesh;
  functable["hex8_double"] = test_hex8_double;
  functable["hex8_bad1"] = test_hex8_bad1;
  functable["hex8_bad2"] = test_hex8_bad2;
  functable["hex8_bad3"] = test_hex8_bad3;
  functable["hex8_bad4"] = test_hex8_bad4;
  functable["hex8_wedge6"] = test_hex8_wedge6;
  functable["hex8_quad4_touch"] = test_hex8_quad4_touch;
  functable["hex8_quad4_touch2"] = test_hex8_quad4_touch2;
  functable["hex8_quad4_touch3"] = test_hex8_quad4_touch3;
  functable["hex8_quad4_cut"] = test_hex8_quad4_cut;
  functable["hex8_quad4_gedreht"] = test_hex8_quad4_gedreht;
  functable["hex8_hex8_durchstoss"] = test_hex8_hex8_durchstoss;
  functable["hex8_hex8_onside"] = test_hex8_hex8_onside;
  functable["hex8_hex8_internal"] = test_hex8_hex8_internal;
  functable["hex8_hex8_sideintersection"] = test_hex8_hex8_sideintersection;
  functable["facet_split"] = test_facet_split;

  // Cells within cells without contact to any surface are not supported.
  //
  // functable["hex8_hex8_inside"] = test_hex8_hex8_inside;

  // functable["hex8_quad4_schnitt"] = test_hex8_quad4_schnitt;
  functable["hex8_quad4_touch4"] = test_hex8_quad4_touch4;
  functable["hex8_quad4_touch5"] = test_hex8_quad4_touch5;
  functable["hex8_quad4_touch6"] = test_hex8_quad4_touch6;
  // functable["hex8_quad4_touch7"] = test_hex8_quad4_touch7;
  functable["hex8_quad4_mesh"] = test_hex8_quad4_mesh;
  functable["position2d"] = test_position2d;

  functable["sc20"] = test_hex8quad4selfcut20;
  functable["sc21"] = test_hex8quad4selfcut21;
  functable["sc22"] = test_hex8quad4selfcut22;
  functable["sc23"] = test_hex8quad4selfcut23;
  functable["sc24"] = test_hex8quad4selfcut24;

  functable["sc30"] = test_hex8quad4selfcut30;
  functable["sc31"] = test_hex8quad4selfcut31;
  functable["sc32"] = test_hex8quad4selfcut32;
  functable["sc33"] = test_hex8quad4selfcut33;
  functable["sc34"] = test_hex8quad4selfcut34;
  functable["sc35"] = test_hex8quad4selfcut35;
  functable["sc36"] = test_hex8quad4selfcut36;
  functable["sc37"] = test_hex8quad4selfcut37;
  functable["sc38"] = test_hex8quad4selfcut38;
  functable["sc39"] = test_hex8quad4selfcut39;

  functable["sc41"] = test_hex8quad4selfcut41;
  functable["sc42"] = test_hex8quad4selfcut42;
  functable["sc43"] = test_hex8quad4selfcut43;

  functable["sc51"] = test_hex8quad4selfcut51;
  functable["sc52"] = test_hex8quad4selfcut52;
  functable["sc53"] = test_hex8quad4selfcut53;

  functable["sc61"] = test_hex8quad4selfcut61;
  functable["sc62"] = test_hex8quad4selfcut62;
  functable["sc63"] = test_hex8quad4selfcut63;
  functable["sc64"] = test_hex8quad4selfcut64;
  functable["sc65"] = test_hex8quad4selfcut65;
  functable["sc66"] = test_hex8quad4selfcut66;

  functable["sc71"] = test_hex8quad4selfcut71;
  functable["sc72"] = test_hex8quad4selfcut72;

  functable["sc81"] = test_hex8quad4selfcut81;
  functable["sc82"] = test_hex8quad4selfcut82;
  functable["sc83"] = test_hex8quad4selfcut83;
  functable["sc84"] = test_hex8quad4selfcut84;
  functable["sc85"] = test_hex8quad4selfcut85;
  functable["sc86"] = test_hex8quad4selfcut86;

  functable["sc91"] = test_hex8quad4selfcut91;
  functable["sc92"] = test_hex8quad4selfcut92;

  functable["sc101"] = test_hex8quad4aligned_edges;

  functable["quad4_line2"] = test_quad4_line2;
  functable["hex8_quad4_qhull1"] = test_hex8_quad4_qhull1;
  functable["hex8_quad4_alex1"] = test_hex8_quad4_alex1;
  functable["hex8_quad4_alex2"] = test_hex8_quad4_alex2;
  functable["hex8_quad4_alex3"] = test_hex8_quad4_alex3;
  functable["hex8_quad4_alex4"] = test_hex8_quad4_alex4;
  functable["hex8_quad4_alex5"] = test_hex8_quad4_alex5;
  functable["hex8_quad4_alex6"] = test_hex8_quad4_alex6;
  functable["hex8_quad4_alex7"] = test_hex8_quad4_alex7;
  functable["hex8_quad4_alex8"] = test_hex8_quad4_alex8;
  functable["tet4_quad4_alex9"] = test_tet4_quad4_alex9;
  functable["tet4_quad4_alex10"] = test_tet4_quad4_alex10;
  // functable["tet4_quad4_alex11"] = test_tet4_quad4_alex11;
  functable["hex8_quad4_alex12"] = test_hex8_quad4_alex12;
  functable["hex8_quad4_alex13"] = test_hex8_quad4_alex13;
  functable["hex8_quad4_alex14"] = test_hex8_quad4_alex14;
  functable["hex8_quad4_alex15"] = test_hex8_quad4_alex15;
  functable["tet4_quad4_alex16"] = test_tet4_quad4_alex16;
  functable["hex8_quad4_alex17"] = test_hex8_quad4_alex17;
  functable["hex8_quad4_alex18"] = test_hex8_quad4_alex18;
  functable["hex8_quad4_alex19"] = test_hex8_quad4_alex19;
  functable["hex8_quad4_alex20"] = test_hex8_quad4_alex20;
  functable["hex8_quad4_alex21"] = test_hex8_quad4_alex21;
  functable["hex8_quad4_alex22"] = test_hex8_quad4_alex22;
  functable["hex8_quad4_alex23"] = test_hex8_quad4_alex23;
  functable["hex8_quad4_alex24"] = test_hex8_quad4_alex24;
  functable["hex8_quad4_alex25"] = test_hex8_quad4_alex25;
  functable["hex8_quad4_alex26"] = test_hex8_quad4_alex26;
  functable["hex8_quad4_alex27"] = test_hex8_quad4_alex27;
  functable["hex8_quad4_alex28"] = test_hex8_quad4_alex28;
  functable["hex8_quad4_alex29"] = test_hex8_quad4_alex29;
  functable["hex8_quad4_alex30"] = test_hex8_quad4_alex30;
  functable["hex8_quad4_alex31"] = test_hex8_quad4_alex31;
  functable["hex8_quad4_alex32"] = test_hex8_quad4_alex32;
  functable["hex8_quad4_alex33"] = test_hex8_quad4_alex33;
  functable["hex8_quad4_alex34"] = test_hex8_quad4_alex34;
  functable["hex8_quad4_alex35"] = test_hex8_quad4_alex35;
  functable["hex8_quad4_alex36"] = test_hex8_quad4_alex36;
  functable["hex8_quad4_alex37"] = test_hex8_quad4_alex37;
  functable["hex8_quad4_alex38"] = test_hex8_quad4_alex38;
  functable["hex8_twintri"] = test_hex8_twintri;
  functable["hex8_twinQuad"] = test_hex8_twin_quad;
  functable["hex8_chairCut"] = test_hex8_chair_cut;
  functable["hex8_VCut"] = test_hex8_v_cut;
  functable["alex39"] = test_alex39;
  functable["alex40"] = test_alex40;
  functable["alex41"] = test_alex41;
  functable["alex42"] = test_alex42;
  functable["alex43"] = test_alex43;
  functable["alex44"] = test_alex44;
  functable["alex45"] = test_alex45;
  functable["alex46"] = test_alex46;
  functable["alex47"] = test_alex47;
  functable["alex48"] = test_alex48;
  functable["alex49"] = test_alex49;
  functable["alex50"] = test_alex50;
  functable["alex51"] = test_alex51;
  functable["alex52"] = test_alex52;
  functable["alex53"] = test_alex53;
  functable["alex54"] = test_alex54;
  functable["alex55"] = test_alex55;
  functable["alex56"] = test_alex56;
  functable["alex57"] = test_alex57;
  functable["alex58"] = test_alex58;
  functable["alex59"] = test_alex59;
  functable["alex60"] = test_alex60;
  functable["alex61"] = test_alex61;
  functable["alex62"] = test_alex62;
  functable["hex8_quad4_axel1"] = test_hex8_quad4_axel1;
  functable["hex8_quad4_axel2"] = test_hex8_quad4_axel2;
  functable["hex8_quad4_axel3"] = test_hex8_quad4_axel3;
  functable["hex8_quad4_axel4"] = test_hex8_quad4_axel4;
  functable["hex8_quad4_axel5"] = test_hex8_quad4_axel5;
  functable["hex8_quad4_axel6"] = test_hex8_quad4_axel6;
  functable["hex8_quad4_axel7"] = test_hex8_quad4_axel7;
  functable["axel8"] = test_axel8;
  functable["axel9"] = test_axel9;
  functable["axel10"] = test_axel10;
  functable["hex8_quad4_shadan1"] = test_hex8_quad4_shadan1;
  functable["hex8_quad4_shadan2"] = test_hex8_quad4_shadan2;
  functable["hex8_quad4_shadan3"] = test_hex8_quad4_shadan3;
  // Switched this cut-test to direct divergence. It is failing some times...
  //   Specifically when all cut-tests are run....
  // functable["hex8_quad4_shadan4"] = test_hex8_quad4_shadan4; // switch off this testcase for the
  // moment. failing as a result of commit 22818. We need a closer look into that!
  functable["hex8_quad4_shadan5"] = test_hex8_quad4_shadan5;
  functable["shadan6"] = test_shadan6;
  // functable["hex8_tri3_ursula1"] = test_hex8_tri3_ursula1;
  functable["hex8_quad4_mesh_edgecut"] = test_hex8_quad4_mesh_edgecut;
  functable["hex8_quad4_mesh_edgecut2"] = test_hex8_quad4_mesh_edgecut2;
  // functable["hex8_quad4_mesh_inner"] = test_hex8_quad4_mesh_inner;
  functable["hex8_quad4_mesh_many"] = test_hex8_quad4_mesh_many;
  functable["hex27_quad9_simple"] = test_hex27_quad9_simple;
  functable["hex20_quad9_simple"] = test_hex20_quad9_simple;
  functable["hex20_quad9_moved"] = test_hex20_quad9_moved;
  functable["tet10_quad9_simple"] = test_tet10_quad9_simple;
  functable["tet10_quad9_moved"] = test_tet10_quad9_moved;
  functable["tet4_quad4_double"] = test_tet4_quad4_double;
#ifdef LOCAL
  functable["tet4_tri3_double"] = test_tet4_tri3_double;
#else
  std::cout << "functable[tet4_tri3_double] = test_tet4_tri3_double; RUNS INTO DSERROR IN GLOBAL "
               "CONFIGURATION!"
            << std::endl;
#endif
  functable["benedikt1"] = test_benedikt1;
  // functable["test_christoph_1"] = test_christoph_1;

  functable["ls_hex8_florian1"] = test_ls_hex8_florian1;
  functable["ls_hex8_florian2"] = test_ls_hex8_florian2;
  functable["ls_hex8_florian3"] = test_ls_hex8_florian3;
  functable["ls_hex8_florian4"] = test_ls_hex8_florian4;
  functable["ls_hex8_florian5"] = test_ls_hex8_florian5;
  functable["ls_hex8_florian6"] = test_ls_hex8_florian6;
  functable["ls_hex8_florian7"] = test_ls_hex8_florian7;
  functable["ls_hex8_florian8"] = test_ls_hex8_florian8;
  functable["ls_hex8_florian9"] = test_ls_hex8_florian9;
  functable["ls_hex8_florian10"] = test_ls_hex8_florian10;
  functable["ls_hex8_florian11"] = test_ls_hex8_florian11;
  functable["ls_hex8_florian12"] = test_ls_hex8_florian12;
  functable["ls_hex8_florian13"] = test_ls_hex8_florian13;
  functable["ls_hex8_ursula1"] = test_ls_hex8_ursula1;
  functable["ls_hex8_ursula2"] = test_ls_hex8_ursula2;
  functable["ls_hex8_ursula3"] = test_ls_hex8_ursula3;
  functable["ls_hex8_ursula4"] = test_ls_hex8_ursula4;
  functable["ls_hex8_ursula5"] = test_ls_hex8_ursula5;
  functable["ls_hex8_ursula6"] = test_ls_hex8_ursula6;
  functable["ls_hex8_simple"] = test_ls_hex8_simple;
  functable["ls_hex8_simple2"] = test_ls_hex8_simple2;
  functable["ls_hex8_simple3"] = test_ls_hex8_simple3;
  // functable["ls_hex8_simple4"] = test_ls_hex8_simple4;
  functable["ls_hex8_simple5"] = test_ls_hex8_simple5;
  functable["ls_hex8_simple6"] = test_ls_hex8_simple6;
  functable["ls_hex8_simple7"] = test_ls_hex8_simple7;
  functable["ls_hex8_touch"] = test_ls_hex8_touch;
  functable["ls_hex8_between"] = test_ls_hex8_between;
  functable["ls_hex8_experiment"] = test_ls_hex8_experiment;
  functable["ls_hex8_experiment_magnus"] = test_ls_hex8_experiment_magnus;
  functable["ls_mesh_hex8_simple"] = test_ls_mesh_hex8_simple;
  functable["ls_hex8_magnus1"] = test_ls_hex8_magnus1;
  functable["ls_hex8_magnus12"] = test_ls_hex8_magnus12;
  functable["ls_hex8_magnus2"] = test_ls_hex8_magnus2;
  functable["ls_hex8_magnus3"] = test_ls_hex8_magnus3;
  functable["ls_hex8_magnus4"] = test_ls_hex8_magnus4;
  functable["ls_hex8_magnus5"] = test_ls_hex8_magnus5;
  functable["ls_hex8_magnus6"] = test_ls_hex8_magnus6;
  functable["ls_hex8_magnus7"] = test_ls_hex8_magnus7;  // Issues in Global Cut for DD
  functable["ls_hex8_tes_dd_simple"] = test_ls_hex8_tes_dd_simple;

  functable["hex8_quad4_double_cut"] = test_hex8_quad4_double_cut;

  functable["unit_intersection_touch"] = test_unit_intersection_touch;

  functable["geometry"] = test_geometry;

  functable["cut_volumes"] = test_cut_volumes;

  functable["fluidfluid"] = test_fluidfluid;
  functable["fluidfluid2"] = test_fluidfluid2;

  Teuchos::CommandLineProcessor clp(false);

  std::string indent = "\t\t\t\t\t";
  std::stringstream doc;
  doc << "Available tests:\n" << indent << "(all)\n";
  for (std::map<std::string, testfunct>::iterator i = functable.begin(); i != functable.end(); ++i)
  {
    const std::string& name = i->first;
    doc << indent << name << "\n";
  }

  std::string testname = "(all)";
  std::string ignore_testname = "";

  clp.setOption("test", &testname, doc.str().c_str());
  clp.setOption("ignore_test", &ignore_testname, doc.str().c_str());

  switch (clp.parse(argc, argv))
  {
    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
      break;
    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
      return 0;
    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
    default:
      std::cerr << argv[0] << ": unrecognized option\n";
      MPI_Finalize();
      return 1;
  }

  set_problem_dimension(functable);
  int result = runtests(argv, functable, testname, ignore_testname);
  MPI_Finalize();
  return result;
}
