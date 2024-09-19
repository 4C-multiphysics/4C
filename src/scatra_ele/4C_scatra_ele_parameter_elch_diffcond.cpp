/*----------------------------------------------------------------------*/
/*! \file

\brief singleton class holding all static diffusion-conduction parameters required for element
evaluation

This singleton class holds all static diffusion-conduction parameters required for element
evaluation. All parameters are usually set only once at the beginning of a simulation, namely during
initialization of the global time integrator, and then never touched again throughout the
simulation. This parameter class needs to coexist with more general parameter classes holding
additional static parameters required for scalar transport element evaluation.


\level 2
*/
/*----------------------------------------------------------------------*/
#include "4C_scatra_ele_parameter_elch_diffcond.hpp"

#include "4C_inpar_elch.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Teuchos_ParameterList.hpp>

#include <map>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | singleton access method                                   fang 02/15 |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::ScaTraEleParameterElchDiffCond*
Discret::ELEMENTS::ScaTraEleParameterElchDiffCond::instance(
    const std::string& disname  //!< name of discretization
)
{
  static auto singleton_map = Core::UTILS::make_singleton_map<std::string>(
      [](const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleParameterElchDiffCond>(
            new ScaTraEleParameterElchDiffCond(disname));
      });

  return singleton_map[disname].instance(Core::UTILS::SingletonAction::create, disname);
}

/*----------------------------------------------------------------------*
 | private constructor for singletons                        fang 02/15 |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::ScaTraEleParameterElchDiffCond::ScaTraEleParameterElchDiffCond(
    const std::string& disname  //!< name of discretization
    )
    : cursolvar_(false),
      diffusioncoefbased_(true),
      newmanconsta_(2.),
      newmanconstb_(-2.),
      newmanconstc_(-1.)
{
  return;
}


/*---------------------------------------------------------------------*
 | set parameters                                           fang 02/15 |
 *---------------------------------------------------------------------*/
void Discret::ELEMENTS::ScaTraEleParameterElchDiffCond::set_parameters(
    Teuchos::ParameterList& parameters  //!< parameter list
)
{
  // access parameter sublist for diffusion-conduction formulation
  Teuchos::ParameterList& diffcondparams = parameters.sublist("DIFFCOND");

  // flag if current is used as a solution variable
  cursolvar_ = diffcondparams.get<bool>("CURRENT_SOLUTION_VAR");

  // mat_diffcond: flag if diffusion potential is based on diffusion coefficients or transference
  // number
  diffusioncoefbased_ = diffcondparams.get<bool>("MAT_DIFFCOND_DIFFBASED");

  // switch for dilute and concentrated solution theory (diffusion potential in current equation):
  //    A          B
  //   |--|  |----------|
  //   z_1 + (z_2 - z_1) t_1
  // ------------------------ (RT/F kappa 1/c_k grad c_k)
  //      z_1 z_2
  //     |________|
  //         C
  newmanconsta_ = diffcondparams.get<double>("MAT_NEWMAN_CONST_A");
  newmanconstb_ = diffcondparams.get<double>("MAT_NEWMAN_CONST_B");
  newmanconstc_ = diffcondparams.get<double>("MAT_NEWMAN_CONST_C");

  return;
}

FOUR_C_NAMESPACE_CLOSE
