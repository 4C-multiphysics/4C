// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_global_data.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Solid::ModelEvaluator::BrownianDynData::BrownianDynData()
    : isinit_(false),
      issetup_(false),
      str_data_ptr_(nullptr),
      viscosity_(0.0),
      kt_(0.0),
      maxrandforce_(0.0),
      timeintconstrandnumb_(0.0),
      beam_damping_coeff_specified_via_(Inpar::BrownianDynamics::vague),
      beams_damping_coefficient_prefactors_perunitlength_{0.0, 0.0, 0.0},
      randomforces_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDynData::init(
    const std::shared_ptr<const Solid::ModelEvaluator::Data>& str_data_ptr)
{
  issetup_ = false;

  str_data_ptr_ = str_data_ptr;

  const Teuchos::ParameterList& browndyn_params_list =
      Global::Problem::instance()->brownian_dynamics_params();

  // viscosity
  viscosity_ = browndyn_params_list.get<double>("VISCOSITY");
  // thermal energy
  kt_ = browndyn_params_list.get<double>("KT");
  // maximum random force (specified as multiple of standard deviation around mean value)
  maxrandforce_ = browndyn_params_list.get<double>("MAXRANDFORCE");
  // time interval with constant random forces
  timeintconstrandnumb_ = browndyn_params_list.get<double>("TIMESTEP");

  // the way how damping coefficient values for beams are specified
  beam_damping_coeff_specified_via_ =
      Teuchos::getIntegralValue<Inpar::BrownianDynamics::BeamDampingCoefficientSpecificationType>(
          browndyn_params_list, "BEAMS_DAMPING_COEFF_SPECIFIED_VIA");

  // if input file is chosen, get the required values and check them for sanity
  if (beam_damping_coeff_specified_via_ == Inpar::BrownianDynamics::input_file)
  {
    std::string input_file_linecontent(Teuchos::getNumericStringParameter(
        browndyn_params_list, "BEAMS_DAMPING_COEFF_PER_UNITLENGTH"));

    Core::IO::ValueParser beam_dampening_coefficients_parser(input_file_linecontent,
        {.user_scope_message = "While reading beam damping coefficient values: "});

    beams_damping_coefficient_prefactors_perunitlength_ =
        beam_dampening_coefficients_parser.read<std::array<double, 3>>();

    if (!beam_dampening_coefficients_parser.at_end())
    {
      FOUR_C_THROW(
          "Too many values for beam damping coefficients are provided within the input file!");
    }

    FOUR_C_ASSERT_ALWAYS(std::all_of(beams_damping_coefficient_prefactors_perunitlength_.begin(),
                             beams_damping_coefficient_prefactors_perunitlength_.end(),
                             [](double damping_coefficient) { return damping_coefficient >= 0.0; }),
        "The damping coefficients for beams must not be negative!");
  }
  // safety check for valid input parameter
  else if (beam_damping_coeff_specified_via_ == Inpar::BrownianDynamics::vague)
  {
    FOUR_C_THROW("The way how beam damping coefficients are specified is not properly set!");
  }

  // set flag
  isinit_ = true;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDynData::setup()
{
  check_init();

  // set flag
  issetup_ = true;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDynData::resize_random_force_m_vector(
    std::shared_ptr<Core::FE::Discretization> discret_ptr, int maxrandnumelement)
{
  check_init_setup();

  // resize in case of new crosslinkers that were set and are now part of the discretization
  randomforces_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *(discret_ptr->element_col_map()), maxrandnumelement, true);

  return;
}

FOUR_C_NAMESPACE_CLOSE
