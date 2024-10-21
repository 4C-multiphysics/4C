#include "4C_beaminteraction_beam_to_sphere_contact_params.hpp"

#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
BEAMINTERACTION::BeamToSphereContactParams::BeamToSphereContactParams()
    : isinit_(false), issetup_(false), penalty_parameter_(-1.0)
{
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/

void BEAMINTERACTION::BeamToSphereContactParams::init()
{
  issetup_ = false;

  // Teuchos parameter list for beam contact
  const Teuchos::ParameterList& beam_to_sphere_contact_params_list =
      Global::Problem::instance()->beam_interaction_params().sublist("BEAM TO SPHERE CONTACT");

  penalty_parameter_ = beam_to_sphere_contact_params_list.get<double>("PENALTY_PARAMETER");

  if (penalty_parameter_ < 0.0)
    FOUR_C_THROW("beam-to-sphere penalty parameter must not be negative!");


  isinit_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BEAMINTERACTION::BeamToSphereContactParams::setup()
{
  check_init();

  // empty for now

  issetup_ = true;
}

FOUR_C_NAMESPACE_CLOSE
