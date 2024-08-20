/*----------------------------------------------------------------------*/
/*! \file

\brief singleton class holding all static levelset reinitialization parameters required for element
evaluation

This singleton class holds all static levelset reinitialization parameters required for element
evaluation. All parameters are usually set only once at the beginning of a simulation, namely during
initialization of the global time integrator, and then never touched again throughout the
simulation. This parameter class needs to coexist with the general parameter class holding all
general static parameters required for scalar transport element evaluation.


\level 2
*/
/*----------------------------------------------------------------------*/
#include "4C_scatra_ele_parameter_lsreinit.hpp"

#include "4C_utils_exceptions.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

//----------------------------------------------------------------------*/
//    definition of the instance
//----------------------------------------------------------------------*/
Discret::ELEMENTS::ScaTraEleParameterLsReinit*
Discret::ELEMENTS::ScaTraEleParameterLsReinit::instance(
    const std::string& disname  //!< name of discretization
)
{
  static auto singleton_map = Core::UTILS::make_singleton_map<std::string>(
      [](const std::string& disname) {
        return std::unique_ptr<ScaTraEleParameterLsReinit>(new ScaTraEleParameterLsReinit(disname));
      });

  return singleton_map[disname].instance(Core::UTILS::SingletonAction::create, disname);
}

//----------------------------------------------------------------------*/
//    constructor
//----------------------------------------------------------------------*/
Discret::ELEMENTS::ScaTraEleParameterLsReinit::ScaTraEleParameterLsReinit(
    const std::string& disname  //!< name of discretization
    )
    : reinittype_(Inpar::ScaTra::reinitaction_none),
      signtype_(Inpar::ScaTra::signtype_nonsmoothed),
      charelelengthreinit_(Inpar::ScaTra::root_of_volume_reinit),
      interfacethicknessfac_(1.0),
      useprojectedreinitvel_(false),
      linform_(Inpar::ScaTra::fixed_point),
      artdiff_(Inpar::ScaTra::artdiff_none),
      alphapen_(0.0),
      project_(true),
      projectdiff_(0.0),
      lumping_(false),
      difffct_(Inpar::ScaTra::hyperbolic)
{
}


//----------------------------------------------------------------------*
//  set parameters                                      rasthofer 12/13 |
//----------------------------------------------------------------------*/
void Discret::ELEMENTS::ScaTraEleParameterLsReinit::set_parameters(
    Teuchos::ParameterList& parameters  //!< parameter list
)
{
  // get reinitialization parameters list
  Teuchos::ParameterList& reinitlist = parameters.sublist("REINITIALIZATION");

  // reinitialization strategy
  reinittype_ =
      Core::UTILS::integral_value<Inpar::ScaTra::ReInitialAction>(reinitlist, "REINITIALIZATION");

  // get signum function
  signtype_ = Core::UTILS::integral_value<Inpar::ScaTra::SmoothedSignType>(
      reinitlist, "SMOOTHED_SIGN_TYPE");

  // characteristic element length for signum function
  charelelengthreinit_ = Core::UTILS::integral_value<Inpar::ScaTra::CharEleLengthReinit>(
      reinitlist, "CHARELELENGTHREINIT");

  // interface thickness for signum function
  interfacethicknessfac_ = reinitlist.get<double>("INTERFACE_THICKNESS");

  // form of linearization for nonlinear terms
  linform_ =
      Core::UTILS::integral_value<Inpar::ScaTra::LinReinit>(reinitlist, "LINEARIZATIONREINIT");

  // set form of velocity evaluation
  Inpar::ScaTra::VelReinit velreinit =
      Core::UTILS::integral_value<Inpar::ScaTra::VelReinit>(reinitlist, "VELREINIT");
  if (velreinit == Inpar::ScaTra::vel_reinit_node_based) useprojectedreinitvel_ = true;

  // set flag for artificial diffusion term
  artdiff_ = Core::UTILS::integral_value<Inpar::ScaTra::ArtDiff>(reinitlist, "ARTDIFFREINIT");

  // set penalty parameter for elliptic reinitialization
  alphapen_ = reinitlist.get<double>("PENALTY_PARA");

  // get diffusivity function
  difffct_ = Core::UTILS::integral_value<Inpar::ScaTra::DiffFunc>(reinitlist, "DIFF_FUNC");

  // L2-projection
  project_ = Core::UTILS::integral_value<bool>(reinitlist, "PROJECTION");

  // diffusion for L2-projection
  projectdiff_ = reinitlist.get<double>("PROJECTION_DIFF");
  if (projectdiff_ < 0.0) FOUR_C_THROW("Diffusivity has to be positive!");

  // lumping for L2-projection
  lumping_ = Core::UTILS::integral_value<bool>(reinitlist, "LUMPING");

  // check for illegal combination
  if (projectdiff_ > 0.0 and lumping_ == true) FOUR_C_THROW("Illegal combination!");
  if (projectdiff_ > 0.0 and reinittype_ == Inpar::ScaTra::reinitaction_sussman)
    FOUR_C_THROW("Illegal combination!");
  // The second FOUR_C_THROW is added here for safety reasons. I think that using a diffusive term
  // for the reconstruction of the velocity for reinitialization is possible, but I have not yet
  // further investigated this option. Therefore, you should test it first.

  return;
}

FOUR_C_NAMESPACE_CLOSE
