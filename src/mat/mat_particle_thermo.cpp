/*---------------------------------------------------------------------------*/
/*! \file
\brief particle material thermo

\level 3


*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                    sfuchs 06/2018 |
 *---------------------------------------------------------------------------*/
#include "mat_particle_thermo.H"

#include "mat_par_bundle.H"

/*---------------------------------------------------------------------------*
 | constructor                                                sfuchs 06/2018 |
 *---------------------------------------------------------------------------*/
MAT::PAR::ParticleMaterialThermo::ParticleMaterialThermo(Teuchos::RCP<MAT::PAR::Material> matdata)
    : Parameter(matdata),
      initTemperature_(matdata->GetDouble("INITTEMPERATURE")),
      thermalCapacity_(matdata->GetDouble("THERMALCAPACITY")),
      invThermalCapacity_((thermalCapacity_ > 0.0) ? (1.0 / thermalCapacity_) : 0.0),
      thermalConductivity_(matdata->GetDouble("THERMALCONDUCTIVITY")),
      thermalAbsorptivity_(matdata->GetDouble("THERMALABSORPTIVITY"))
{
  // empty constructor
}
