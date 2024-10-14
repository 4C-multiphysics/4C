/*----------------------------------------------------------------------*/
/*! \file
\brief Input parameters for immersed

\level 1


*/

/*----------------------------------------------------------------------*/


#include "4C_inpar_immersed.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void Inpar::Immersed::set_valid_parameters(Teuchos::ParameterList& list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;
  Teuchos::ParameterList& immersedmethod =
      list.sublist("IMMERSED METHOD", false, "General parameters for any immersed problem");

  Teuchos::Tuple<std::string, 3> coupname;
  Teuchos::Tuple<Inpar::Immersed::PartitionedScheme, 3> couplabel;

  coupname[0] = "basic_sequ_stagg";
  couplabel[0] = cell_basic_sequ_stagg;
  coupname[1] = "iter_stagg_fixed_rel_param";
  couplabel[1] = cell_iter_stagg_fixed_rel_param;
  coupname[2] = "iter_stagg_AITKEN_rel_param";
  couplabel[2] = cell_iter_stagg_AITKEN_rel_param;

  setStringToIntegralParameter<Inpar::Immersed::ImmersedCoupling>("COUPALGO", "partitioned",
      "Coupling strategies for immersed method.", tuple<std::string>("partitioned", "monolithic"),
      tuple<Inpar::Immersed::ImmersedCoupling>(partitioned, monolithic), &immersedmethod);

  setStringToIntegralParameter<Inpar::Immersed::ImmersedCouplingScheme>("SCHEME",
      "dirichletneumann", "Coupling schemes for partitioned immersed method.",
      tuple<std::string>("neumannneumann", "dirichletneumann"),
      tuple<Inpar::Immersed::ImmersedCouplingScheme>(neumannneumann, dirichletneumann),
      &immersedmethod);

  setStringToIntegralParameter<Inpar::Immersed::ImmersedNlnsolver>("DIVERCONT", "stop",
      "What to do after maxiter is reached.", tuple<std::string>("stop", "continue"),
      tuple<Inpar::Immersed::ImmersedNlnsolver>(nlnsolver_stop, nlnsolver_continue),
      &immersedmethod);

  Core::Utils::bool_parameter("OUTPUT_EVRY_NLNITER", "no",
      "write output after every solution step of the nonlin. part. iter. scheme", &immersedmethod);

  Core::Utils::bool_parameter("CORRECT_BOUNDARY_VELOCITIES", "no",
      "correct velocities in fluid elements cut by surface of immersed structure", &immersedmethod);

  Core::Utils::bool_parameter("DEFORM_BACKGROUND_MESH", "no",
      "switch between immersed with fixed or deformable background mesh", &immersedmethod);

  std::vector<std::string> timestats_valid_input = {"everyiter", "endofsim"};
  Core::Utils::string_parameter("TIMESTATS", "everyiter",
      "summarize time monitor every nln iteration", &immersedmethod, timestats_valid_input);

  Core::Utils::double_parameter(
      "FLD_SRCHRADIUS_FAC", 1.0, "fac times fluid ele. diag. length", &immersedmethod);

  Core::Utils::double_parameter(
      "STRCT_SRCHRADIUS_FAC", 0.5, "fac times structure bounding box diagonal", &immersedmethod);

  Core::Utils::int_parameter("NUM_GP_FLUID_BOUND", 8,
      "number of gp in fluid elements cut by surface of immersed structure (higher number yields "
      "better mass conservation)",
      &immersedmethod);

  /*----------------------------------------------------------------------*/
  /* parameters for paritioned immersed solvers */
  Teuchos::ParameterList& immersedpart = immersedmethod.sublist("PARTITIONED SOLVER", false, "");

  setStringToIntegralParameter<Inpar::Immersed::PartitionedScheme>("COUPALGO",
      "iter_stagg_fixed_rel_param", "Iteration Scheme over the fields", coupname, couplabel,
      &immersedpart);

  std::vector<std::string> predictor_valid_input = {
      "d(n)", "d(n)+dt*(1.5*v(n)-0.5*v(n-1))", "d(n)+dt*v(n)", "d(n)+dt*v(n)+0.5*dt^2*a(n)"};
  Core::Utils::string_parameter("PREDICTOR", "d(n)", "Predictor for interface displacements",
      &immersedpart, predictor_valid_input);

  std::vector<std::string> coupvariable_valid_input = {"Displacement", "Force"};
  Core::Utils::string_parameter("COUPVARIABLE", "Displacement",
      "Coupling variable at the fsi interface", &immersedpart, coupvariable_valid_input);


  Core::Utils::double_parameter("CONVTOL", 1e-6,
      "Tolerance for iteration over fields in case of partitioned scheme", &immersedpart);
  Core::Utils::double_parameter(
      "RELAX", 1.0, "fixed relaxation parameter for partitioned FSI solvers", &immersedpart);
  Core::Utils::double_parameter("MAXOMEGA", 0.0,
      "largest omega allowed for Aitken relaxation (0.0 means no constraint)", &immersedpart);
  Core::Utils::int_parameter(
      "ITEMAX", 100, "Maximum number of iterations over fields", &immersedpart);
}



void Inpar::Immersed::set_valid_conditions(
    std::vector<Teuchos::RCP<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  /*--------------------------------------------------------------------*/
  // IMMERSED FSI

  Teuchos::RCP<Core::Conditions::ConditionDefinition> immersedsearchbox =
      Teuchos::make_rcp<Core::Conditions::ConditionDefinition>("DESIGN VOLUME IMMERSED SEARCHBOX",
          "ImmersedSearchbox", "Immersed Searchbox", Core::Conditions::ImmersedSearchbox, true,
          Core::Conditions::geometry_type_volume);

  condlist.push_back(immersedsearchbox);

  /*--------------------------------------------------------------------*/
  // IMMERSED COUPLING

  Teuchos::RCP<Core::Conditions::ConditionDefinition> lineimmersed =
      Teuchos::make_rcp<Core::Conditions::ConditionDefinition>(
          "DESIGN IMMERSED COUPLING LINE CONDITIONS", "IMMERSEDCoupling", "IMMERSED Coupling",
          Core::Conditions::IMMERSEDCoupling, true, Core::Conditions::geometry_type_line);
  Teuchos::RCP<Core::Conditions::ConditionDefinition> surfimmersed =
      Teuchos::make_rcp<Core::Conditions::ConditionDefinition>(
          "DESIGN IMMERSED COUPLING SURF CONDITIONS", "IMMERSEDCoupling", "IMMERSED Coupling",
          Core::Conditions::IMMERSEDCoupling, true, Core::Conditions::geometry_type_surface);

  lineimmersed->add_component(Teuchos::make_rcp<Input::IntComponent>("coupling id"));
  surfimmersed->add_component(Teuchos::make_rcp<Input::IntComponent>("coupling id"));

  condlist.push_back(lineimmersed);
  condlist.push_back(surfimmersed);
}

FOUR_C_NAMESPACE_CLOSE
