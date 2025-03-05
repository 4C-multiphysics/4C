// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_mortar.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::Mortar::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  /* parameters for mortar coupling */
  Core::Utils::SectionSpecs mortar{"MORTAR COUPLING"};

  Core::Utils::string_to_integral_parameter<Inpar::Mortar::ShapeFcn>("LM_SHAPEFCN", "Dual",
      "Type of employed set of shape functions",
      tuple<std::string>(
          "Dual", "dual", "Standard", "standard", "std", "PetrovGalerkin", "petrovgalerkin", "pg"),
      tuple<Inpar::Mortar::ShapeFcn>(shape_dual, shape_dual, shape_standard, shape_standard,
          shape_standard, shape_petrovgalerkin, shape_petrovgalerkin, shape_petrovgalerkin),
      mortar);

  Core::Utils::string_to_integral_parameter<Inpar::Mortar::SearchAlgorithm>("SEARCH_ALGORITHM",
      "Binarytree", "Type of contact search",
      tuple<std::string>("BruteForce", "bruteforce", "BruteForceEleBased", "bruteforceelebased",
          "BinaryTree", "Binarytree", "binarytree"),
      tuple<Inpar::Mortar::SearchAlgorithm>(search_bfele, search_bfele, search_bfele, search_bfele,
          search_binarytree, search_binarytree, search_binarytree),
      mortar);

  Core::Utils::string_to_integral_parameter<Inpar::Mortar::BinaryTreeUpdateType>(
      "BINARYTREE_UPDATETYPE", "BottomUp",
      "Type of binary tree update, which is either a bottom up or a top down approach.",
      tuple<std::string>("BottomUp", "TopDown"),
      tuple<Inpar::Mortar::BinaryTreeUpdateType>(binarytree_bottom_up, binarytree_top_down),
      mortar);

  mortar.specs.emplace_back(parameter<double>(
      "SEARCH_PARAM", {.description = "Radius / Bounding volume inflation for contact search",
                          .default_value = 0.3}));

  mortar.specs.emplace_back(parameter<bool>("SEARCH_USE_AUX_POS",
      {.description = "If chosen auxiliary position is used for computing dops",
          .default_value = true}));

  Core::Utils::string_to_integral_parameter<Inpar::Mortar::LagMultQuad>("LM_QUAD", "undefined",
      "Type of LM interpolation for quadratic FE",
      tuple<std::string>(
          "undefined", "quad", "quadratic", "pwlin", "piecewiselinear", "lin", "linear", "const"),
      tuple<Inpar::Mortar::LagMultQuad>(lagmult_undefined, lagmult_quad, lagmult_quad,
          lagmult_pwlin, lagmult_pwlin, lagmult_lin, lagmult_lin, lagmult_const),
      mortar);

  mortar.specs.emplace_back(parameter<bool>("CROSSPOINTS",
      {.description = "If chosen, multipliers are removed from crosspoints / edge nodes",
          .default_value = false}));

  Core::Utils::string_to_integral_parameter<Inpar::Mortar::ConsistentDualType>("LM_DUAL_CONSISTENT",
      "boundary",
      "For which elements should the dual basis be calculated on EXACTLY the same GPs as the "
      "contact terms",
      tuple<std::string>("none", "boundary", "all"),
      tuple<Inpar::Mortar::ConsistentDualType>(
          consistent_none, consistent_boundary, consistent_all),
      mortar);

  Core::Utils::string_to_integral_parameter<Inpar::Mortar::MeshRelocation>("MESH_RELOCATION",
      "Initial", "Type of mesh relocation", tuple<std::string>("Initial", "Every_Timestep", "None"),
      tuple<Inpar::Mortar::MeshRelocation>(
          relocation_initial, relocation_timestep, relocation_none),
      mortar);

  Core::Utils::string_to_integral_parameter<Inpar::Mortar::AlgorithmType>("ALGORITHM", "Mortar",
      "Type of meshtying/contact algorithm",
      tuple<std::string>("mortar", "Mortar", "nts", "NTS", "gpts", "GPTS", "lts", "LTS", "ltl",
          "LTL", "stl", "STL"),
      tuple<Inpar::Mortar::AlgorithmType>(algorithm_mortar, algorithm_mortar, algorithm_nts,
          algorithm_nts, algorithm_gpts, algorithm_gpts, algorithm_lts, algorithm_lts,
          algorithm_ltl, algorithm_ltl, algorithm_stl, algorithm_stl),
      mortar);

  Core::Utils::string_to_integral_parameter<Inpar::Mortar::IntType>("INTTYPE", "Segments",
      "Type of numerical integration scheme",
      tuple<std::string>(
          "Segments", "segments", "Elements", "elements", "Elements_BS", "elements_BS"),
      tuple<Inpar::Mortar::IntType>(inttype_segments, inttype_segments, inttype_elements,
          inttype_elements, inttype_elements_BS, inttype_elements_BS),
      mortar);

  Core::Utils::int_parameter(
      "NUMGP_PER_DIM", 0, "Number of employed integration points per dimension", mortar);

  Core::Utils::string_to_integral_parameter<Inpar::Mortar::Triangulation>("TRIANGULATION",
      "Delaunay", "Type of triangulation for segment-based integration",
      tuple<std::string>("Delaunay", "delaunay", "Center", "center"),
      tuple<Inpar::Mortar::Triangulation>(triangulation_delaunay, triangulation_delaunay,
          triangulation_center, triangulation_center),
      mortar);

  mortar.specs.emplace_back(parameter<bool>("RESTART_WITH_MESHTYING",
      {.description =
              "Must be chosen if a non-meshtying simulation is to be restarted with meshtying",
          .default_value = false}));

  mortar.specs.emplace_back(parameter<bool>("OUTPUT_INTERFACES",
      {.description = "Write output for each mortar interface separately.\nThis is an additional "
                      "feature, purely to enhance visualization. Currently, this is limited to "
                      "solid meshtying and contact w/o friction.",
          .default_value = false}));

  mortar.move_into_collection(list);

  /*--------------------------------------------------------------------*/
  // parameters for parallel redistribution of mortar interfaces
  Core::Utils::SectionSpecs parallelRedist{mortar, "PARALLEL REDISTRIBUTION"};

  parallelRedist.specs.emplace_back(parameter<bool>("EXPLOIT_PROXIMITY",
      {.description =
              "Exploit information on geometric proximity to split slave interface into close and "
              "non-close parts and redistribute them independently. [Contact only]",
          .default_value = true}));

  Core::Utils::string_to_integral_parameter<ExtendGhosting>("GHOSTING_STRATEGY", "redundant_master",
      "Type of interface ghosting and ghosting extension algorithm",
      tuple<std::string>("redundant_all", "redundant_master", "round_robin", "binning"),
      tuple<ExtendGhosting>(ExtendGhosting::redundant_all, ExtendGhosting::redundant_master,
          ExtendGhosting::roundrobin, ExtendGhosting::binning),
      parallelRedist);

  parallelRedist.specs.emplace_back(parameter<double>("IMBALANCE_TOL",
      {.description = "Max. relative imbalance of subdomain size after redistribution",
          .default_value = 1.1}));

  parallelRedist.specs.emplace_back(parameter<double>(
      "MAX_BALANCE_EVAL_TIME", {.description = "Max-to-min ratio of contact evaluation time per "
                                               "processor to trigger parallel redistribution",
                                   .default_value = 2.0}));

  parallelRedist.specs.emplace_back(parameter<double>(
      "MAX_BALANCE_SLAVE_ELES", {.description = "Max-to-min ratio of mortar slave elements per "
                                                "processor to trigger parallel redistribution",
                                    .default_value = 0.5}));

  Core::Utils::int_parameter("MIN_ELEPROC", 0,
      "Minimum no. of elements per processor for parallel redistribution", parallelRedist);

  Core::Utils::string_to_integral_parameter<ParallelRedist>("PARALLEL_REDIST", "Static",
      "Type of redistribution algorithm", tuple<std::string>("None", "Static", "Dynamic"),
      tuple<ParallelRedist>(ParallelRedist::redist_none, ParallelRedist::redist_static,
          ParallelRedist::redist_dynamic),
      parallelRedist);

  parallelRedist.specs.emplace_back(parameter<bool>(
      "PRINT_DISTRIBUTION", {.description = "Print details of the parallel distribution, i.e. "
                                            "number of nodes/elements for each rank.",
                                .default_value = true}));

  parallelRedist.move_into_collection(list);
}

void Inpar::Mortar::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
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
    cond.add_component(selection<std::string>(
        "Side", {"Master", "Slave", "Selfcontact"}, {.description = "interface side"}));
    cond.add_component(selection<std::string>("Initialization", {"Inactive", "Active"},
        {.description = "initialization", .default_value = "Inactive"}));

    cond.add_component(parameter<double>(
        "FrCoeffOrBound", {.description = "friction coefficient bound", .default_value = 0.0}));
    cond.add_component(parameter<double>(
        "AdhesionBound", {.description = "adhesion bound", .default_value = 0.0}));

    cond.add_component(selection<std::string>("Application",
        {"Solidcontact", "Beamtosolidcontact", "Beamtosolidmeshtying"},
        {.description = "application", .default_value = "Solidcontact"}));

    // optional DBC handling
    cond.add_component(selection<int>("DbcHandling",
        {{"DoNothing", static_cast<int>(DBCHandling::do_nothing)},
            {"RemoveDBCSlaveNodes",
                static_cast<int>(DBCHandling::remove_dbc_nodes_from_slave_side)}},
        {.description = "DbcHandling",
            .default_value = static_cast<int>(DBCHandling::do_nothing)}));
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
      cond.add_component(
          selection<std::string>("Side", {"Master", "Slave"}, {.description = "interface side"}));
      cond.add_component(selection<std::string>("Initialization", {"Inactive", "Active"},
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
      cond.add_component(
          selection<std::string>("Side", {"Master", "Slave"}, {.description = "interface side"}));
      cond.add_component(selection<std::string>("Initialization", {"Inactive", "Active"},
          {.description = "initialization", .default_value = "Inactive"}));

      condlist.push_back(cond);
    };

    make_mortar_multi(linemortar);
    make_mortar_multi(surfmortar);
  }
}

FOUR_C_NAMESPACE_CLOSE
