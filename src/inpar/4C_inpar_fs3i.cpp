/*----------------------------------------------------------------------*/
/*! \file
\brief Input parameters for fs3i

\level 2


*/

/*----------------------------------------------------------------------*/



#include "4C_inpar_fs3i.hpp"

#include "4C_inpar_scatra.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::FS3I::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& fs3idyn =
      list->sublist("FS3I DYNAMIC", false, "control parameters for FS3I problems\n");

  Core::UTILS::DoubleParameter("TIMESTEP", 0.1, "Time increment dt", &fs3idyn);
  Core::UTILS::IntParameter("NUMSTEP", 20, "Total number of time steps", &fs3idyn);
  Core::UTILS::DoubleParameter("MAXTIME", 1000.0, "Total simulation time", &fs3idyn);
  Core::UTILS::IntParameter("RESULTSEVRY", 1, "Increment for writing solution", &fs3idyn);
  Core::UTILS::IntParameter("RESTARTEVRY", 1, "Increment for writing restart", &fs3idyn);
  setStringToIntegralParameter<int>("SCATRA_SOLVERTYPE", "nonlinear",
      "type of scalar transport solver", tuple<std::string>("linear", "nonlinear"),
      tuple<int>(Inpar::ScaTra::solvertype_linear_incremental, Inpar::ScaTra::solvertype_nonlinear),
      &fs3idyn);
  Core::UTILS::BoolParameter("INF_PERM", "yes", "Flag for infinite permeability", &fs3idyn);
  setStringToIntegralParameter<int>("CONSTHERMPRESS", "Yes",
      "treatment of thermodynamic pressure in time",
      tuple<std::string>("No_energy", "No_mass", "Yes"), tuple<int>(0, 1, 2), &fs3idyn);

  // number of linear solver used for fs3i problems
  Core::UTILS::IntParameter(
      "COUPLED_LINEAR_SOLVER", -1, "number of linear solver used for fs3i problem", &fs3idyn);
  Core::UTILS::IntParameter(
      "LINEAR_SOLVER1", -1, "number of linear solver used for fluid problem", &fs3idyn);
  Core::UTILS::IntParameter(
      "LINEAR_SOLVER2", -1, "number of linear solver used for structural problem", &fs3idyn);

  setStringToIntegralParameter<int>("STRUCTSCAL_CONVFORM", "conservative",
      "form of convective term of structure scalar",
      tuple<std::string>("convective", "conservative"),
      tuple<int>(Inpar::ScaTra::convform_convective, Inpar::ScaTra::convform_conservative),
      &fs3idyn);

  setStringToIntegralParameter<int>("STRUCTSCAL_INITIALFIELD", "zero_field",
      "Initial Field for structure scalar transport problem",
      tuple<std::string>("zero_field", "field_by_function"),
      tuple<int>(Inpar::ScaTra::initfield_zero_field, Inpar::ScaTra::initfield_field_by_function),
      &fs3idyn);

  Core::UTILS::IntParameter("STRUCTSCAL_INITFUNCNO", -1,
      "function number for structure scalar transport initial field", &fs3idyn);

  // Type of coupling strategy between structure and structure-scalar field
  setStringToIntegralParameter<int>("STRUCTSCAL_FIELDCOUPLING", "volume_matching",
      "Type of coupling strategy between structure and structure-scalar field",
      tuple<std::string>("volume_matching", "volume_nonmatching"),
      tuple<int>(coupling_match, coupling_nonmatch), &fs3idyn);

  // Type of coupling strategy between fluid and fluid-scalar field
  setStringToIntegralParameter<int>("FLUIDSCAL_FIELDCOUPLING", "volume_matching",
      "Type of coupling strategy between fluid and fluid-scalar field",
      tuple<std::string>("volume_matching", "volume_nonmatching"),
      tuple<int>(coupling_match, coupling_nonmatch), &fs3idyn);

  // type of scalar transport
  setStringToIntegralParameter<int>("FLUIDSCAL_SCATRATYPE", "ConvectionDiffusion",
      "Type of scalar transport problem",
      tuple<std::string>("Undefined", "ConvectionDiffusion", "Loma", "Advanced_Reaction",
          "Chemotaxis", "Chemo_Reac"),
      tuple<int>(Inpar::ScaTra::impltype_undefined, Inpar::ScaTra::impltype_std,
          Inpar::ScaTra::impltype_loma, Inpar::ScaTra::impltype_advreac,
          Inpar::ScaTra::impltype_chemo, Inpar::ScaTra::impltype_chemoreac),
      &fs3idyn);

  // Restart from FSI instead of FS3I
  Core::UTILS::BoolParameter("RESTART_FROM_PART_FSI", "No",
      "restart from partitioned fsi problem (e.g. from prestress calculations) instead of fs3i",
      &fs3idyn);

  /*----------------------------------------------------------------------*/
  /* parameters for partitioned FS3I */
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fs3idynpart = fs3idyn.sublist(
      "PARTITIONED", false, "partioned fluid-structure-scalar-scalar interaction control section");

  // Coupling strategy for partitioned FS3I
  setStringToIntegralParameter<int>("COUPALGO", "fs3i_IterStagg",
      "Coupling strategies for FS3I solvers",
      tuple<std::string>("fs3i_SequStagg", "fs3i_IterStagg"),
      tuple<int>(fs3i_SequStagg, fs3i_IterStagg), &fs3idynpart);

  // convergence tolerance of outer iteration loop
  Core::UTILS::DoubleParameter("CONVTOL", 1e-6,
      "tolerance for convergence check of outer iteration within partitioned FS3I", &fs3idynpart);

  Core::UTILS::IntParameter("ITEMAX", 10, "Maximum number of outer iterations", &fs3idynpart);

  /*----------------------------------------------------------------------  */
  /* parameters for stabilization of the structure-scalar field             */
  /*----------------------------------------------------------------------  */
  Teuchos::ParameterList& fs3idynstructscalstab = fs3idyn.sublist("STRUCTURE SCALAR STABILIZATION",
      false, "parameters for stabilization of the structure-scalar field");

  Teuchos::ParameterList& scatradyn = list->sublist(
      "SCALAR TRANSPORT DYNAMIC", true, "control parameters for scalar transport problems\n");
  fs3idynstructscalstab = scatradyn.sublist("STABILIZATION", true,
      "control parameters for the stabilization of scalar transport problems");
}

FOUR_C_NAMESPACE_CLOSE
