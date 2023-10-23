/*---------------------------------------------------------------------------*/
/*! \file
\brief particle result test for particle simulations
\level 1
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_particle_algorithm_result_test.H"

#include "baci_io_linedefinition.H"
#include "baci_particle_engine_container.H"
#include "baci_particle_engine_container_bundle.H"
#include "baci_particle_engine_interface.H"

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEALGORITHM::ParticleResultTest::ParticleResultTest() : DRT::ResultTest("PARTICLE")
{
  // empty constructor
}

void PARTICLEALGORITHM::ParticleResultTest::Init()
{
  // nothing to do
}

void PARTICLEALGORITHM::ParticleResultTest::Setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;
}

void PARTICLEALGORITHM::ParticleResultTest::TestSpecial(
    DRT::INPUT::LineDefinition& res, int& nerr, int& test_count)
{
  // extract global particle id
  int globalid;
  res.ExtractInt("ID", globalid);

  // get local index in specific particle container
  PARTICLEENGINE::LocalIndexTupleShrdPtr localindextuple =
      particleengineinterface_->GetLocalIndexInSpecificContainer(globalid);

  // particle with global id found on this processor
  if (localindextuple)
  {
    // access values of local index tuple
    PARTICLEENGINE::TypeEnum particleType;
    PARTICLEENGINE::StatusEnum particleStatus;
    int index;
    std::tie(particleType, particleStatus, index) = *localindextuple;

    // consider only owned particle
    if (particleStatus == PARTICLEENGINE::Owned)
    {
      // get particle container bundle
      PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
          particleengineinterface_->GetParticleContainerBundle();

      // get container of owned particles of current particle type
      PARTICLEENGINE::ParticleContainer* container =
          particlecontainerbundle->GetSpecificContainer(particleType, PARTICLEENGINE::Owned);

      // get result
      std::string quantity;
      res.ExtractString("QUANTITY", quantity);

      // init actual result
      double actresult = 0.0;

      // component of result
      int dim = 0;

      // declare enum of particle state
      PARTICLEENGINE::StateEnum particleState;

      // position
      if (quantity == "posx" or quantity == "posy" or quantity == "posz")
      {
        // get enum of particle state
        particleState = PARTICLEENGINE::Position;

        // get component of result
        if (quantity == "posx")
          dim = 0;
        else if (quantity == "posy")
          dim = 1;
        else if (quantity == "posz")
          dim = 2;
      }
      // velocity
      else if (quantity == "velx" or quantity == "vely" or quantity == "velz")
      {
        // get enum of particle state
        particleState = PARTICLEENGINE::Velocity;

        // get component of result
        if (quantity == "velx")
          dim = 0;
        else if (quantity == "vely")
          dim = 1;
        else if (quantity == "velz")
          dim = 2;
      }
      // acceleration
      else if (quantity == "accx" or quantity == "accy" or quantity == "accz")
      {
        // get enum of particle state
        particleState = PARTICLEENGINE::Acceleration;

        // get component of result
        if (quantity == "accx")
          dim = 0;
        else if (quantity == "accy")
          dim = 1;
        else if (quantity == "accz")
          dim = 2;
      }
      // angular velocity
      else if (quantity == "angvelx" or quantity == "angvely" or quantity == "angvelz")
      {
        // get enum of particle state
        particleState = PARTICLEENGINE::AngularVelocity;

        // get component of result
        if (quantity == "angvelx")
          dim = 0;
        else if (quantity == "angvely")
          dim = 1;
        else if (quantity == "angvelz")
          dim = 2;
      }
      // radius
      else if (quantity == "radius")
      {
        // get enum of particle state
        particleState = PARTICLEENGINE::Radius;

        // get component of result
        dim = 0;
      }
      // density
      else if (quantity == "density")
      {
        // get enum of particle state
        particleState = PARTICLEENGINE::Density;

        // get component of result
        dim = 0;
      }
      // pressure
      else if (quantity == "pressure")
      {
        // get enum of particle state
        particleState = PARTICLEENGINE::Pressure;

        // get component of result
        dim = 0;
      }
      // temperature
      else if (quantity == "temperature")
      {
        // get enum of particle state
        particleState = PARTICLEENGINE::Temperature;

        // get component of result
        dim = 0;
      }
      // temperature gradient
      else if (quantity == "tempgradx" or quantity == "tempgrady" or quantity == "tempgradz")
      {
        // get enum of particle state
        particleState = PARTICLEENGINE::TemperatureGradient;

        // get component of result
        if (quantity == "tempgradx")
          dim = 0;
        else if (quantity == "tempgrady")
          dim = 1;
        else if (quantity == "tempgradz")
          dim = 2;
      }
      else
        dserror("result check failed with unknown quantity '%s'!", quantity.c_str());

      // container contains current particle state
      if (not container->HaveStoredState(particleState))
        dserror("state '%s' not found in container!",
            PARTICLEENGINE::EnumToStateName(particleState).c_str());

      // get pointer to particle state
      const double* state = container->GetPtrToState(particleState, 0);

      // get particle state dimension
      int statedim = container->GetStateDim(particleState);

      // get actual result
      actresult = state[statedim * index + dim];

      // compare values
      const int err = CompareValues(actresult, "SPECIAL", res);
      nerr += err;
      test_count++;
    }
  }
}
