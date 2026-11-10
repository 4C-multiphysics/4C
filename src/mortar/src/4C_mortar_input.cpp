// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
FOUR_C_NAMESPACE_OPEN



std::vector<Core::IO::InputSpec> Mortar::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  /* parameters for mortar coupling */

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("MORTAR COUPLING",
      {

          deprecated_selection<Mortar::ShapeFcn>("LM_SHAPEFCN",
              {
                  {"Dual", shape_dual},
                  {"dual", shape_dual},
                  {"Standard", shape_standard},
                  {"standard", shape_standard},
                  {"std", shape_standard},
                  {"PetrovGalerkin", shape_petrovgalerkin},
                  {"petrovgalerkin", shape_petrovgalerkin},
                  {"pg", shape_petrovgalerkin},
              },
              {.description = "Type of employed set of shape functions",
                  .default_value = shape_dual}),

          deprecated_selection<Mortar::SearchAlgorithm>("SEARCH_ALGORITHM",
              {
                  {"BinaryTree", search_binarytree},
                  {"Binarytree", search_binarytree},
                  {"binarytree", search_binarytree},
                  {"bounding_volume_hierarchy", bounding_volume_hierarchy},
              },
              {.description = "Type of contact search",
                  .default_value = bounding_volume_hierarchy}),


          deprecated_selection<Mortar::BinaryTreeUpdateType>("BINARYTREE_UPDATETYPE",
              {
                  {"BottomUp", binarytree_bottom_up},
                  {"TopDown", binarytree_top_down},
              },
              {.description = "Type of binary tree update, which is either a bottom up or a top "
                              "down approach.",
                  .default_value = binarytree_bottom_up}),

          parameter<double>("SEARCH_PARAM",
              {.description = "Inflation factor for the bounding volumes (k-DOPs) used in the "
                              "contact search. The bounding volumes are enlarged by this factor "
                              "times the minimum edge length of all contact pair elements.",
                  .default_value = 0.3}),

          parameter<bool>("SEARCH_USE_AUX_POS",
              {.description = "If chosen auxiliary position is used for computing dops",
                  .default_value = true}),

          deprecated_selection<Mortar::LagMultQuad>("LM_QUAD",
              {
                  {"undefined", lagmult_undefined},
                  {"quad", lagmult_quad},
                  {"quadratic", lagmult_quad},
                  {"pwlin", lagmult_pwlin},
                  {"piecewiselinear", lagmult_pwlin},
                  {"lin", lagmult_lin},
                  {"linear", lagmult_lin},
                  {"const", lagmult_const},
              },
              {.description = "Type of LM interpolation for quadratic FE",
                  .default_value = lagmult_undefined}),

          parameter<bool>("CROSSPOINTS",
              {.description = "If chosen, multipliers are removed from crosspoints / edge nodes",
                  .default_value = false}),


          deprecated_selection<Mortar::ConsistentDualType>("LM_DUAL_CONSISTENT",
              {
                  {"none", consistent_none},
                  {"boundary", consistent_boundary},
                  {"all", consistent_all},
              },
              {.description =
                      "For which elements should the dual basis be calculated on EXACTLY the "
                      "same GPs as the contact terms",
                  .default_value = consistent_boundary}),

          deprecated_selection<Mortar::MeshRelocation>("MESH_RELOCATION",
              {
                  {"Initial", relocation_initial},
                  {"None", relocation_none},
              },
              {.description = "Type of mesh relocation", .default_value = relocation_initial}),

          deprecated_selection<Mortar::AlgorithmType>("ALGORITHM",
              {
                  {"mortar", algorithm_mortar},
                  {"Mortar", algorithm_mortar},
                  {"nts", algorithm_nts},
                  {"NTS", algorithm_nts},
                  {"gpts", algorithm_gpts},
                  {"GPTS", algorithm_gpts},
                  {"lts", algorithm_lts},
                  {"LTS", algorithm_lts},
                  {"ltl", algorithm_ltl},
                  {"LTL", algorithm_ltl},
                  {"stl", algorithm_stl},
                  {"STL", algorithm_stl},
              },
              {.description = "Type of meshtying/contact algorithm",
                  .default_value = algorithm_mortar}),

          deprecated_selection<Mortar::IntType>("INTTYPE",
              {
                  {"Segments", inttype_segments},
                  {"segments", inttype_segments},
                  {"Elements", inttype_elements},
                  {"elements", inttype_elements},
                  {"Elements_BS", inttype_elements_BS},
                  {"elements_BS", inttype_elements_BS},
              },
              {.description = "Type of numerical integration scheme",
                  .default_value = inttype_segments}),

          parameter<int>("NUMGP_PER_DIM",
              {.description = "Number of employed integration points per dimension",
                  .default_value = 0}),

          deprecated_selection<Mortar::Triangulation>("TRIANGULATION",
              {
                  {"Delaunay", triangulation_delaunay},
                  {"delaunay", triangulation_delaunay},
                  {"Center", triangulation_center},
                  {"center", triangulation_center},
              },
              {.description = "Type of triangulation for segment-based integration",
                  .default_value = triangulation_delaunay}),

          parameter<bool>("RESTART_WITH_MESHTYING",
              {.description = "Must be chosen if a non-meshtying simulation is to be restarted "
                              "with meshtying",
                  .default_value = false}),

          parameter<bool>("OUTPUT_INTERFACES",
              {.description =
                      "Write output for each mortar interface separately.\nThis is an additional "
                      "feature, purely to enhance visualization. Currently, this is limited to "
                      "solid meshtying and contact w/o friction.",
                  .default_value = false})},
      {.required = false}));
  /*--------------------------------------------------------------------*/
  // parameters for parallel redistribution of mortar interfaces
  specs.push_back(group("MORTAR COUPLING/PARALLEL REDISTRIBUTION",
      {

          parameter<bool>("EXPLOIT_PROXIMITY",
              {.description = "Exploit information on geometric proximity to split slave interface "
                              "into close and "
                              "non-close parts and redistribute them independently. [Contact only]",
                  .default_value = true}),

          deprecated_selection<ExtendGhosting>("GHOSTING_STRATEGY",
              {
                  {"redundant_all", ExtendGhosting::redundant_all},
                  {"redundant_master", ExtendGhosting::redundant_target},
                  {"round_robin", ExtendGhosting::roundrobin},
                  {"binning", ExtendGhosting::binning},
              },
              {.description = "Type of interface ghosting and ghosting extension algorithm",
                  .default_value = ExtendGhosting::redundant_target}),

          parameter<double>("IMBALANCE_TOL",
              {.description = "Max. relative imbalance of subdomain size after redistribution",
                  .default_value = 1.1}),

          parameter<double>("MAX_BALANCE_EVAL_TIME",
              {.description = "Max-to-min ratio of contact evaluation time per "
                              "processor to trigger parallel redistribution",
                  .default_value = 2.0}),

          parameter<double>("MAX_BALANCE_SLAVE_ELES",
              {.description = "Max-to-min ratio of mortar slave elements per "
                              "processor to trigger parallel redistribution",
                  .default_value = 0.5}),

          parameter<int>("MIN_ELEPROC",
              {.description = "Minimum no. of elements per processor for parallel redistribution",
                  .default_value = 0}),

          deprecated_selection<ParallelRedist>("PARALLEL_REDIST",
              {
                  {"None", ParallelRedist::redist_none},
                  {"Static", ParallelRedist::redist_static},
                  {"Dynamic", ParallelRedist::redist_dynamic},
              },
              {.description = "Type of redistribution algorithm",
                  .default_value = ParallelRedist::redist_static}),

          parameter<bool>("PRINT_DISTRIBUTION",
              {.description = "Print details of the parallel distribution, i.e. "
                              "number of nodes/elements for each rank.",
                  .default_value = true})},
      {.required = false}));
  return specs;
}

void Mortar::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // mortar contact

  Core::Conditions::ConditionDefinition linecontact("DESIGN LINE MORTAR CONTACT CONDITIONS 2D",
      "Contact", "Line Contact Coupling", Core::Conditions::Contact, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfcontact("DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
      "Contact", "Surface Contact Coupling", Core::Conditions::Contact, true,
      Core::Conditions::geometry_type_surface);

  const auto make_contact = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("InterfaceID"));
    cond.add_component(deprecated_selection<std::string>(
        "Side", {"Master", "Slave", "Selfcontact"}, {.description = "interface side"}));
    cond.add_component(deprecated_selection<std::string>("Initialization", {"Inactive", "Active"},
        {.description = "initialization", .default_value = "Inactive"}));

    cond.add_component(parameter<double>(
        "FrCoeffOrBound", {.description = "friction coefficient bound", .default_value = 0.0}));
    cond.add_component(parameter<double>(
        "AdhesionBound", {.description = "adhesion bound", .default_value = 0.0}));

    cond.add_component(deprecated_selection<std::string>("Application",
        {"Solidcontact", "Beamtosolidcontact", "Beamtosolidmeshtying"},
        {.description = "application", .default_value = "Solidcontact"}));

    cond.add_component(parameter<double>(
        "TwoHalfPass", {.description = "optional two half pass approach", .default_value = 0.0}));
    cond.add_component(parameter<double>("RefConfCheckNonSmoothSelfContactSurface",
        {.description =
                "optional reference configuration check for non-smooth self contact surfaces",
            .default_value = 0.0}));
    cond.add_component(parameter<std::optional<int>>(
        "ConstitutiveLawID", {.description = "material id of the constitutive law"}));
    condlist.push_back(cond);
  };

  make_contact(linecontact);
  make_contact(surfcontact);

  /*--------------------------------------------------------------------*/
  // mortar coupling (for ALL kinds of interface problems except contact)

  {
    Core::Conditions::ConditionDefinition linemortar("DESIGN LINE MORTAR COUPLING CONDITIONS 2D",
        "Mortar", "Line Mortar Coupling", Core::Conditions::Mortar, true,
        Core::Conditions::geometry_type_line);
    Core::Conditions::ConditionDefinition surfmortar("DESIGN SURF MORTAR COUPLING CONDITIONS 3D",
        "Mortar", "Surface Mortar Coupling", Core::Conditions::Mortar, true,
        Core::Conditions::geometry_type_surface);

    const auto make_mortar = [&condlist](Core::Conditions::ConditionDefinition& cond)
    {
      cond.add_component(parameter<int>("InterfaceID"));
      cond.add_component(deprecated_selection<std::string>(
          "Side", {"Master", "Slave"}, {.description = "interface side"}));
      cond.add_component(deprecated_selection<std::string>("Initialization", {"Inactive", "Active"},
          {.description = "initialization", .default_value = "Inactive"}));

      condlist.push_back(cond);
    };

    make_mortar(linemortar);
    make_mortar(surfmortar);
  }


  /*--------------------------------------------------------------------*/
  // mortar coupling symmetry condition

  Core::Conditions::ConditionDefinition linemrtrsym("DESIGN LINE MORTAR SYMMETRY CONDITIONS 3D",
      "mrtrsym", "Symmetry plane normal for 3D contact", Core::Conditions::LineMrtrSym, true,
      Core::Conditions::geometry_type_line);

  Core::Conditions::ConditionDefinition pointmrtrsym(
      "DESIGN POINT MORTAR SYMMETRY CONDITIONS 2D/3D", "mrtrsym",
      "Symmetry plane normal for 2D/3D contact", Core::Conditions::PointMrtrSym, true,
      Core::Conditions::geometry_type_point);

  const auto make_mrtrsym = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<std::vector<int>>("ONOFF", {.description = "", .size = 3}));

    condlist.push_back(cond);
  };

  make_mrtrsym(linemrtrsym);
  make_mrtrsym(pointmrtrsym);

  /*--------------------------------------------------------------------*/
  // mortar edge/corner condition

  Core::Conditions::ConditionDefinition edgemrtr("DESIGN LINE MORTAR EDGE CONDITIONS 3D",
      "mrtredge", "Geometrical edge for 3D contact", Core::Conditions::EdgeMrtr, true,
      Core::Conditions::geometry_type_line);

  Core::Conditions::ConditionDefinition cornermrtr("DESIGN POINT MORTAR CORNER CONDITIONS 2D/3D",
      "mrtrcorner", "Geometrical corner for 2D/3D contact", Core::Conditions::CornerMrtr, true,
      Core::Conditions::geometry_type_point);

  condlist.push_back(edgemrtr);
  condlist.push_back(cornermrtr);


  {
    /*--------------------------------------------------------------------*/
    // mortar coupling (for ALL kinds of interface problems except contact)

    Core::Conditions::ConditionDefinition linemortar(
        "DESIGN LINE MORTAR MULTI-COUPLING CONDITIONS 2D", "MortarMulti",
        "Line Mortar Multi-Coupling", Core::Conditions::MortarMulti, true,
        Core::Conditions::geometry_type_line);
    Core::Conditions::ConditionDefinition surfmortar(
        "DESIGN SURF MORTAR MULTI-COUPLING CONDITIONS 3D", "MortarMulti",
        "Surface Mortar Multi-Coupling", Core::Conditions::MortarMulti, true,
        Core::Conditions::geometry_type_surface);

    const auto make_mortar_multi = [&condlist](Core::Conditions::ConditionDefinition& cond)
    {
      cond.add_component(parameter<int>("InterfaceID"));
      cond.add_component(deprecated_selection<std::string>(
          "Side", {"Master", "Slave"}, {.description = "interface side"}));
      cond.add_component(deprecated_selection<std::string>("Initialization", {"Inactive", "Active"},
          {.description = "initialization", .default_value = "Inactive"}));

      condlist.push_back(cond);
    };

    make_mortar_multi(linemortar);
    make_mortar_multi(surfmortar);
  }
}

FOUR_C_NAMESPACE_CLOSE