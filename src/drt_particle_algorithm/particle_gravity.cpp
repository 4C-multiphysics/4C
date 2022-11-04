/*---------------------------------------------------------------------------*/
/*! \file
\brief gravity handler for particle simulations
\level 2
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "particle_gravity.H"

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/function_of_time.H"

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEALGORITHM::GravityHandler::GravityHandler(const Teuchos::ParameterList& params)
    : params_(params), gravityrampfctnumber_(params.get<int>("GRAVITY_RAMP_FUNCT"))
{
  // empty constructor
}

void PARTICLEALGORITHM::GravityHandler::Init(const std::vector<double>& gravity)
{
  // set gravity acceleration vector
  gravity_ = gravity;

  // safety check
  if (static_cast<int>(gravity_.size()) != 3)
    dserror("dimension (dim = %d) of gravity acceleration vector is wrong!",
        static_cast<int>(gravity_.size()));
}

void PARTICLEALGORITHM::GravityHandler::Setup()
{
  // nothing to do
}

void PARTICLEALGORITHM::GravityHandler::GetGravityAcceleration(
    const double time, std::vector<double>& scaled_gravity)
{
  scaled_gravity = gravity_;

  // evaluate gravity ramp function
  if (gravityrampfctnumber_ > 0)
  {
    const double fac = DRT::Problem::Instance()
                           ->FunctionById<DRT::UTILS::FunctionOfTime>(gravityrampfctnumber_ - 1)
                           .Evaluate(time);

    for (int dim = 0; dim < 3; ++dim) scaled_gravity[dim] *= fac;
  }
}
