/*-----------------------------------------------------------*/
/*! \file

\brief pseudo transient solution method


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_nln_solver_ptc.hpp"  // class definition

#include "4C_structure_new_nln_solver_utils.hpp"
#include "4C_structure_new_timint_base.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::NLN::SOLVER::PseudoTransient::PseudoTransient()
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::NLN::SOLVER::PseudoTransient::Setup()
{
  check_init();

  // setup the nox parameter list for a pseudo transient solution method
  set_pseudo_transient_params();

  // Call the Setup() function of the base class
  // Note, that the issetup_ flag is also updated during this call.
  Nox::Setup();

  FOUR_C_ASSERT(is_setup(), "issetup_ should be \"true\" at this point!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::NLN::SOLVER::PseudoTransient::set_pseudo_transient_params()
{
  check_init();

  /* get the nox parameter list and set the necessary parameters for a
   * pseudo transient solution procedure */
  Teuchos::ParameterList& pnox = data_s_dyn().GetNoxParams();

  // ---------------------------------------------------------------------------
  // Set-up the pseudo transient method
  // ---------------------------------------------------------------------------
  // Non-linear solver
  pnox.set("Nonlinear Solver", "Pseudo Transient");

  // Direction
  Teuchos::ParameterList& pdir = pnox.sublist("Direction", true);
  pdir.set("Method", "Newton");

  // Line Search
  Teuchos::ParameterList& plinesearch = pnox.sublist("Line Search", true);

  // Line Search/Full Step (line search is deactivated)
  Teuchos::ParameterList& pfullstep = plinesearch.sublist("Full Step", true);
  // check if the default value is set
  double fullstep = pfullstep.get<double>("Full Step");
  if (fullstep != 1.0)
  {
    std::string markerline;
    markerline.assign(40, '!');
    std::cout << markerline << std::endl
              << "WARNING: The Full Step length is " << fullstep << " (default=1.0)" << std::endl
              << markerline << std::endl;
  }
  /* The following parameters create a NOX::NLN::Solver::PseudoTransient
   * solver which is equivalent to the old 4C implementation.
   *
   * If you are keen on using the new features, please use the corresponding
   * input section "STRUCT NOX/Pseudo Transient" in your input file. */
  Teuchos::ParameterList& pptc = pnox.sublist("Pseudo Transient");

  pptc.set<double>("deltaInit", data_s_dyn().get_initial_ptc_pseudo_time_step());
  pptc.set<double>("deltaMax", std::numeric_limits<double>::max());
  pptc.set<double>("deltaMin", 0.0);
  pptc.set<int>("Maximum Number of Pseudo-Transient Iterations", (data_s_dyn().GetIterMax() + 1));
  pptc.set<std::string>("Time Step Control", "SER");
  pptc.set<double>("SER_alpha", 1.0);
  pptc.set<double>("ScalingFactor", 1.0);
  pptc.set<std::string>("Norm Type for TSC", "Max Norm");
  pptc.set<std::string>("Build scaling operator", "every timestep");
  pptc.set<std::string>("Scaling Type", "Identity");

  // ---------------------------------------------------------------------------
  // STATUS TEST
  // ---------------------------------------------------------------------------
  /* This is only necessary for the special case, that you use no xml-file for
   * the definition of your convergence tests, but you use the dat-file instead.
   */
  if (not IsXMLStatusTestFile(data_s_dyn().GetNoxParams().sublist("Status Test")))
  {
    std::set<enum NOX::NLN::StatusTest::QuantityType> qtypes;
    CreateQuantityTypes(qtypes, data_s_dyn());
    SetStatusTestParams(data_s_dyn().GetNoxParams().sublist("Status Test"), data_s_dyn(), qtypes);
  }
}

FOUR_C_NAMESPACE_CLOSE