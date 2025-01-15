// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_mortar.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::Mortar::set_valid_parameters(Teuchos::ParameterList& list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  /* parameters for mortar coupling */
  Teuchos::ParameterList& mortar = list.sublist("MORTAR COUPLING", false, "");

  setStringToIntegralParameter<Inpar::Mortar::ShapeFcn>("LM_SHAPEFCN", "Dual",
      "Type of employed set of shape functions",
      tuple<std::string>(
          "Dual", "dual", "Standard", "standard", "std", "PetrovGalerkin", "petrovgalerkin", "pg"),
      tuple<Inpar::Mortar::ShapeFcn>(shape_dual, shape_dual, shape_standard, shape_standard,
          shape_standard, shape_petrovgalerkin, shape_petrovgalerkin, shape_petrovgalerkin),
      &mortar);

  setStringToIntegralParameter<Inpar::Mortar::SearchAlgorithm>("SEARCH_ALGORITHM", "Binarytree",
      "Type of contact search",
      tuple<std::string>("BruteForce", "bruteforce", "BruteForceEleBased", "bruteforceelebased",
          "BinaryTree", "Binarytree", "binarytree"),
      tuple<Inpar::Mortar::SearchAlgorithm>(search_bfele, search_bfele, search_bfele, search_bfele,
          search_binarytree, search_binarytree, search_binarytree),
      &mortar);

  setStringToIntegralParameter<Inpar::Mortar::BinaryTreeUpdateType>("BINARYTREE_UPDATETYPE",
      "BottomUp", "Type of binary tree update, which is either a bottom up or a top down approach.",
      tuple<std::string>("BottomUp", "TopDown"),
      tuple<Inpar::Mortar::BinaryTreeUpdateType>(binarytree_bottom_up, binarytree_top_down),
      &mortar);

  Core::Utils::double_parameter(
      "SEARCH_PARAM", 0.3, "Radius / Bounding volume inflation for contact search", &mortar);

  Core::Utils::bool_parameter("SEARCH_USE_AUX_POS", "Yes",
      "If chosen auxiliary position is used for computing dops", &mortar);

  setStringToIntegralParameter<Inpar::Mortar::LagMultQuad>("LM_QUAD", "undefined",
      "Type of LM interpolation for quadratic FE",
      tuple<std::string>(
          "undefined", "quad", "quadratic", "pwlin", "piecewiselinear", "lin", "linear", "const"),
      tuple<Inpar::Mortar::LagMultQuad>(lagmult_undefined, lagmult_quad, lagmult_quad,
          lagmult_pwlin, lagmult_pwlin, lagmult_lin, lagmult_lin, lagmult_const),
      &mortar);

  Core::Utils::bool_parameter("CROSSPOINTS", "No",
      "If chosen, multipliers are removed from crosspoints / edge nodes", &mortar);

  setStringToIntegralParameter<Inpar::Mortar::ConsistentDualType>("LM_DUAL_CONSISTENT", "boundary",
      "For which elements should the dual basis be calculated on EXACTLY the same GPs as the "
      "contact terms",
      tuple<std::string>("none", "boundary", "all"),
      tuple<Inpar::Mortar::ConsistentDualType>(
          consistent_none, consistent_boundary, consistent_all),
      &mortar);

  setStringToIntegralParameter<Inpar::Mortar::MeshRelocation>("MESH_RELOCATION", "Initial",
      "Type of mesh relocation",
      tuple<std::string>("Initial", "initial", "Every_Timestep", "every_timestep", "No", "no"),
      tuple<Inpar::Mortar::MeshRelocation>(relocation_initial, relocation_initial,
          relocation_timestep, relocation_timestep, relocation_none, relocation_none),
      &mortar);

  setStringToIntegralParameter<Inpar::Mortar::AlgorithmType>("ALGORITHM", "Mortar",
      "Type of meshtying/contact algorithm",
      tuple<std::string>("mortar", "Mortar", "nts", "NTS", "gpts", "GPTS", "lts", "LTS", "ltl",
          "LTL", "stl", "STL"),
      tuple<Inpar::Mortar::AlgorithmType>(algorithm_mortar, algorithm_mortar, algorithm_nts,
          algorithm_nts, algorithm_gpts, algorithm_gpts, algorithm_lts, algorithm_lts,
          algorithm_ltl, algorithm_ltl, algorithm_stl, algorithm_stl),
      &mortar);

  setStringToIntegralParameter<Inpar::Mortar::IntType>("INTTYPE", "Segments",
      "Type of numerical integration scheme",
      tuple<std::string>(
          "Segments", "segments", "Elements", "elements", "Elements_BS", "elements_BS"),
      tuple<Inpar::Mortar::IntType>(inttype_segments, inttype_segments, inttype_elements,
          inttype_elements, inttype_elements_BS, inttype_elements_BS),
      &mortar);

  Core::Utils::int_parameter(
      "NUMGP_PER_DIM", 0, "Number of employed integration points per dimension", &mortar);

  setStringToIntegralParameter<Inpar::Mortar::Triangulation>("TRIANGULATION", "Delaunay",
      "Type of triangulation for segment-based integration",
      tuple<std::string>("Delaunay", "delaunay", "Center", "center"),
      tuple<Inpar::Mortar::Triangulation>(triangulation_delaunay, triangulation_delaunay,
          triangulation_center, triangulation_center),
      &mortar);

  Core::Utils::bool_parameter("RESTART_WITH_MESHTYING", "No",
      "Must be chosen if a non-meshtying simulation is to be restarted with meshtying", &mortar);

  Core::Utils::bool_parameter("OUTPUT_INTERFACES", "No",
      "Write output for each mortar interface separately.\nThis is an additional feature, purely "
      "to enhance visualization. Currently, this is limited to solid meshtying and contact w/o "
      "friction.",
      &mortar);

  /*--------------------------------------------------------------------*/
  // parameters for parallel redistribution of mortar interfaces
  Teuchos::ParameterList& parallelRedist = mortar.sublist("PARALLEL REDISTRIBUTION", false,
      "Parameters to control parallel redistribution of mortar interfaces");

  Core::Utils::bool_parameter("EXPLOIT_PROXIMITY", "Yes",
      "Exploit information on geometric proximity to split slave interface into close and "
      "non-close parts and redistribute them independently. [Contact only]",
      &parallelRedist);

  setStringToIntegralParameter<ExtendGhosting>("GHOSTING_STRATEGY", "redundant_master",
      "Type of interface ghosting and ghosting extension algorithm",
      tuple<std::string>("redundant_all", "redundant_master", "round_robin", "binning"),
      tuple<ExtendGhosting>(ExtendGhosting::redundant_all, ExtendGhosting::redundant_master,
          ExtendGhosting::roundrobin, ExtendGhosting::binning),
      &parallelRedist);

  Core::Utils::double_parameter("IMBALANCE_TOL", 1.1,
      "Max. relative imbalance of subdomain size after redistribution", &parallelRedist);

  Core::Utils::double_parameter("MAX_BALANCE_EVAL_TIME", 2.0,
      "Max-to-min ratio of contact evaluation time per processor to trigger parallel "
      "redistribution",
      &parallelRedist);

  Core::Utils::double_parameter("MAX_BALANCE_SLAVE_ELES", 0.5,
      "Max-to-min ratio of mortar slave elements per processor to trigger parallel "
      "redistribution",
      &parallelRedist);

  Core::Utils::int_parameter("MIN_ELEPROC", 0,
      "Minimum no. of elements per processor for parallel redistribution", &parallelRedist);

  setStringToIntegralParameter<ParallelRedist>("PARALLEL_REDIST", "Static",
      "Type of redistribution algorithm",
      tuple<std::string>("None", "none", "No", "no", "Static", "static", "Dynamic", "dynamic"),
      tuple<ParallelRedist>(ParallelRedist::redist_none, ParallelRedist::redist_none,
          ParallelRedist::redist_none, ParallelRedist::redist_none, ParallelRedist::redist_static,
          ParallelRedist::redist_static, ParallelRedist::redist_dynamic,
          ParallelRedist::redist_dynamic),
      &parallelRedist);

  Core::Utils::bool_parameter("PRINT_DISTRIBUTION", "Yes",
      "Print details of the parallel distribution, i.e. number of nodes/elements for each rank.",
      &parallelRedist);
}

void Inpar::Mortar::set_valid_conditions(
    std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  /*--------------------------------------------------------------------*/
  // mortar contact

  std::shared_ptr<Core::Conditions::ConditionDefinition> linecontact =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE MORTAR CONTACT CONDITIONS 2D", "Contact", "Line Contact Coupling",
          Core::Conditions::Contact, true, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfcontact =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF MORTAR CONTACT CONDITIONS 3D", "Contact", "Surface Contact Coupling",
          Core::Conditions::Contact, true, Core::Conditions::geometry_type_surface);

  for (const auto& cond : {linecontact, surfcontact})
  {
    add_named_int(cond, "InterfaceID");
    add_named_selection_component(cond, "Side", "interface side", "Master",
        Teuchos::tuple<std::string>("Master", "Slave", "Selfcontact"),
        Teuchos::tuple<std::string>("Master", "Slave", "Selfcontact"));
    add_named_selection_component(cond, "Initialization", "initialization", "Inactive",
        Teuchos::tuple<std::string>("Inactive", "Active"),
        Teuchos::tuple<std::string>("Inactive", "Active"), true);

    add_named_real(cond, "FrCoeffOrBound", "friction coefficient bound", 0.0, true);
    add_named_real(cond, "AdhesionBound", "adhesion bound", 0.0, true);

    add_named_selection_component(cond, "Application", "application", "Solidcontact",
        Teuchos::tuple<std::string>("Solidcontact", "Beamtosolidcontact", "Beamtosolidmeshtying"),
        Teuchos::tuple<std::string>("Solidcontact", "Beamtosolidcontact", "Beamtosolidmeshtying"),
        true);

    // optional DBC handling
    add_named_selection_component(cond, "DbcHandling", "DbcHandling", "DoNothing",
        Teuchos::tuple<std::string>("DoNothing", "RemoveDBCSlaveNodes"),
        Teuchos::tuple<int>(static_cast<int>(DBCHandling::do_nothing),
            static_cast<int>(DBCHandling::remove_dbc_nodes_from_slave_side)),
        true);
    add_named_real(cond, "TwoHalfPass", "optional two half pass approach", 0.0, true);
    add_named_real(cond, "RefConfCheckNonSmoothSelfContactSurface",
        "optional reference configuration check for non-smooth self contact surfaces", 0.0, true);
    add_named_int(cond, "ConstitutiveLawID", "material id of the constitutive law", 0, true);

    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // mortar coupling (for ALL kinds of interface problems except contact)

  std::shared_ptr<Core::Conditions::ConditionDefinition> linemortar =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE MORTAR COUPLING CONDITIONS 2D", "Mortar", "Line Mortar Coupling",
          Core::Conditions::Mortar, true, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfmortar =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF MORTAR COUPLING CONDITIONS 3D", "Mortar", "Surface Mortar Coupling",
          Core::Conditions::Mortar, true, Core::Conditions::geometry_type_surface);

  for (const auto& cond : {linemortar, surfmortar})
  {
    add_named_int(cond, "InterfaceID");
    add_named_selection_component(cond, "Side", "interface side", "Master",
        Teuchos::tuple<std::string>("Master", "Slave"),
        Teuchos::tuple<std::string>("Master", "Slave"));
    add_named_selection_component(cond, "Initialization", "initialization", "Inactive",
        Teuchos::tuple<std::string>("Inactive", "Active"),
        Teuchos::tuple<std::string>("Inactive", "Active"), true);

    condlist.push_back(cond);
  }


  /*--------------------------------------------------------------------*/
  // mortar coupling symmetry condition

  std::shared_ptr<Core::Conditions::ConditionDefinition> linemrtrsym =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE MORTAR SYMMETRY CONDITIONS 3D", "mrtrsym",
          "Symmetry plane normal for 3D contact", Core::Conditions::LineMrtrSym, true,
          Core::Conditions::geometry_type_line);

  std::shared_ptr<Core::Conditions::ConditionDefinition> pointmrtrsym =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT MORTAR SYMMETRY CONDITIONS 2D/3D", "mrtrsym",
          "Symmetry plane normal for 2D/3D contact", Core::Conditions::PointMrtrSym, true,
          Core::Conditions::geometry_type_point);

  for (const auto& cond : {linemrtrsym, pointmrtrsym})
  {
    add_named_int_vector(cond, "ONOFF", "", 3);

    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // mortar edge/corner condition

  std::shared_ptr<Core::Conditions::ConditionDefinition> edgemrtr =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE MORTAR EDGE CONDITIONS 3D", "mrtredge", "Geometrical edge for 3D contact",
          Core::Conditions::EdgeMrtr, true, Core::Conditions::geometry_type_line);

  std::shared_ptr<Core::Conditions::ConditionDefinition> cornermrtr =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT MORTAR CORNER CONDITIONS 2D/3D", "mrtrcorner",
          "Geometrical corner for 2D/3D contact", Core::Conditions::CornerMrtr, true,
          Core::Conditions::geometry_type_point);

  condlist.push_back(edgemrtr);
  condlist.push_back(cornermrtr);


  {
    /*--------------------------------------------------------------------*/
    // mortar coupling (for ALL kinds of interface problems except contact)

    std::shared_ptr<Core::Conditions::ConditionDefinition> linemortar =
        std::make_shared<Core::Conditions::ConditionDefinition>(
            "DESIGN LINE MORTAR MULTI-COUPLING CONDITIONS 2D", "MortarMulti",
            "Line Mortar Multi-Coupling", Core::Conditions::MortarMulti, true,
            Core::Conditions::geometry_type_line);
    std::shared_ptr<Core::Conditions::ConditionDefinition> surfmortar =
        std::make_shared<Core::Conditions::ConditionDefinition>(
            "DESIGN SURF MORTAR MULTI-COUPLING CONDITIONS 3D", "MortarMulti",
            "Surface Mortar Multi-Coupling", Core::Conditions::MortarMulti, true,
            Core::Conditions::geometry_type_surface);

    for (const auto& cond : {linemortar, surfmortar})
    {
      add_named_int(cond, "InterfaceID");
      add_named_selection_component(cond, "Side", "interface side", "Master",
          Teuchos::tuple<std::string>("Master", "Slave"),
          Teuchos::tuple<std::string>("Master", "Slave"));
      add_named_selection_component(cond, "Initialization", "initialization", "Inactive",
          Teuchos::tuple<std::string>("Inactive", "Active"),
          Teuchos::tuple<std::string>("Inactive", "Active"), true);

      condlist.push_back(cond);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
