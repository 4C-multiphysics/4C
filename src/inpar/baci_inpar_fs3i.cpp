/*----------------------------------------------------------------------*/
/*! \file
\brief Input parameters for fs3i

\level 2


*/

/*----------------------------------------------------------------------*/



#include "baci_inpar_fs3i.hpp"

#include "baci_inpar_scatra.hpp"
#include "baci_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void INPAR::FS3I::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace INPUT;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& fs3idyn =
      list->sublist("FS3I DYNAMIC", false, "control parameters for FS3I problems\n");

  CORE::UTILS::DoubleParameter("TIMESTEP", 0.1, "Time increment dt", &fs3idyn);
  CORE::UTILS::IntParameter("NUMSTEP", 20, "Total number of time steps", &fs3idyn);
  CORE::UTILS::DoubleParameter("MAXTIME", 1000.0, "Total simulation time", &fs3idyn);
  CORE::UTILS::IntParameter("RESULTSEVRY", 1, "Increment for writing solution", &fs3idyn);
  CORE::UTILS::IntParameter("RESTARTEVRY", 1, "Increment for writing restart", &fs3idyn);
  setStringToIntegralParameter<int>("SCATRA_SOLVERTYPE", "nonlinear",
      "type of scalar transport solver", tuple<std::string>("linear", "nonlinear"),
      tuple<int>(INPAR::SCATRA::solvertype_linear_incremental, INPAR::SCATRA::solvertype_nonlinear),
      &fs3idyn);
  CORE::UTILS::BoolParameter("INF_PERM", "yes", "Flag for infinite permeability", &fs3idyn);
  setStringToIntegralParameter<int>("CONSTHERMPRESS", "Yes",
      "treatment of thermodynamic pressure in time",
      tuple<std::string>("No_energy", "No_mass", "Yes"), tuple<int>(0, 1, 2), &fs3idyn);

  // number of linear solver used for fs3i problems
  CORE::UTILS::IntParameter(
      "COUPLED_LINEAR_SOLVER", -1, "number of linear solver used for fs3i problem", &fs3idyn);
  CORE::UTILS::IntParameter(
      "LINEAR_SOLVER1", -1, "number of linear solver used for fluid problem", &fs3idyn);
  CORE::UTILS::IntParameter(
      "LINEAR_SOLVER2", -1, "number of linear solver used for structural problem", &fs3idyn);

  setStringToIntegralParameter<int>("STRUCTSCAL_CONVFORM", "conservative",
      "form of convective term of structure scalar",
      tuple<std::string>("convective", "conservative"),
      tuple<int>(INPAR::SCATRA::convform_convective, INPAR::SCATRA::convform_conservative),
      &fs3idyn);

  setStringToIntegralParameter<int>("STRUCTSCAL_INITIALFIELD", "zero_field",
      "Initial Field for structure scalar transport problem",
      tuple<std::string>("zero_field", "field_by_function"),
      tuple<int>(INPAR::SCATRA::initfield_zero_field, INPAR::SCATRA::initfield_field_by_function),
      &fs3idyn);

  CORE::UTILS::IntParameter("STRUCTSCAL_INITFUNCNO", -1,
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
      tuple<int>(INPAR::SCATRA::impltype_undefined, INPAR::SCATRA::impltype_std,
          INPAR::SCATRA::impltype_loma, INPAR::SCATRA::impltype_advreac,
          INPAR::SCATRA::impltype_chemo, INPAR::SCATRA::impltype_chemoreac),
      &fs3idyn);

  // Restart from FSI instead of FS3I
  CORE::UTILS::BoolParameter("RESTART_FROM_PART_FSI", "No",
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
  CORE::UTILS::DoubleParameter("CONVTOL", 1e-6,
      "tolerance for convergence check of outer iteration within partitioned FS3I", &fs3idynpart);

  CORE::UTILS::IntParameter("ITEMAX", 10, "Maximum number of outer iterations", &fs3idynpart);

  /*----------------------------------------------------------------------  */
  /* parameters for stabilization of the structure-scalar field             */
  /*----------------------------------------------------------------------  */
  Teuchos::ParameterList& fs3idynstructscalstab = fs3idyn.sublist("STRUCTURE SCALAR STABILIZATION",
      false, "parameters for stabilization of the structure-scalar field");

  Teuchos::ParameterList& scatradyn = list->sublist(
      "SCALAR TRANSPORT DYNAMIC", true, "control parameters for scalar transport problems\n");
  fs3idynstructscalstab = scatradyn.sublist("STABILIZATION", true,
      "control parameters for the stabilization of scalar transport problems");

  /*----------------------------------------------------------------------*/
  /* parameters for Atherosclerosis FSI */
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fs3idynac = fs3idyn.sublist(
      "AC", false, "Atherosclerosis fluid-structure-scalar-scalar interaction control section");

  // FSI time steps per SSI time step
  CORE::UTILS::IntParameter(
      "FSI_STEPS_PER_SCATRA_STEP", 1, "FSI time steps per SSI time step", &fs3idynac);

  // Periodicity of the FSI problem
  CORE::UTILS::DoubleParameter("PERIODICITY", -1.0, "Periodicity of the FSI problem", &fs3idynac);

  // relative tolerance for the WK of the fluid sub-problem. Determines if the fsi problem is
  // already periodic
  CORE::UTILS::DoubleParameter("WINDKESSEL_REL_TOL", -1.0,
      "Tolerance for the fluid windkessel to decide if the FSI problem is periodic", &fs3idynac);

  // relative tolerance for the fluid scatra. Determines if the fluid scatra is already periodic
  CORE::UTILS::DoubleParameter("FLUID_SCATRA_REL_TOL", -1.0,
      "Tolerance for the fluid scatra field to decide if it is periodic", &fs3idynac);

  // relative tolerance for the fluid scatra. Determines if the fluid scatra is already periodic
  CORE::UTILS::DoubleParameter("WSS_REL_TOL", -1.0,
      "Tolerance for the wall shear stresses to decide if the FSI problem is periodic", &fs3idynac);

  // amount of growth updates in the large time scale loop
  CORE::UTILS::IntParameter(
      "GROWTH_UPDATES", 1.0, "Amount of growth updates in the large time scale loop", &fs3idynac);

  // realtive tolerance for the structure scatra field to decide if a FSI update is necessary
  CORE::UTILS::DoubleParameter("FSI_UPDATE_TOL", -1.0,
      "Tolerance for the structure scatra field to decide if a FSI update is necessary",
      &fs3idynac);

  // time step of the large time scale
  CORE::UTILS::DoubleParameter(
      "LARGE_TIMESCALE_TIMESTEP", -1.0, "time step of the large time scale", &fs3idynac);
}

FOUR_C_NAMESPACE_CLOSE
