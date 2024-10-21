#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_EQUATIONOFSTATE_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_EQUATIONOFSTATE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class SPHEquationOfStateBase
  {
   public:
    //! constructor
    explicit SPHEquationOfStateBase();

    //! virtual destructor
    virtual ~SPHEquationOfStateBase() = default;

    //! init equation of state handler
    virtual void init();

    //! setup equation of state handler
    virtual void setup();

    //! determine the pressure
    virtual double density_to_pressure(const double& density, const double& density0) const = 0;

    //! determine the density
    virtual double pressure_to_density(const double& pressure, const double& density0) const = 0;

    //! determine the energy
    virtual double density_to_energy(
        const double& density, const double& mass, const double& density0) const = 0;
  };

  class SPHEquationOfStateGenTait final : public SPHEquationOfStateBase
  {
   public:
    //! constructor
    explicit SPHEquationOfStateGenTait(
        const double& speedofsound, const double& refdensfac, const double& exponent);

    //! determine the pressure
    double density_to_pressure(const double& density, const double& density0) const override;

    //! determine the density
    double pressure_to_density(const double& pressure, const double& density0) const override;

    //! determine the energy
    double density_to_energy(
        const double& density, const double& mass, const double& density0) const override;

   private:
    //! speed of sound
    const double speedofsound_;

    //! reference density factor
    const double refdensfac_;

    //! exponent
    const double exponent_;
  };

  class SPHEquationOfStateIdealGas final : public SPHEquationOfStateBase
  {
   public:
    //! constructor
    explicit SPHEquationOfStateIdealGas(const double& speedofsound);

    //! determine the pressure
    double density_to_pressure(const double& density, const double& density0) const override;

    //! determine the density
    double pressure_to_density(const double& pressure, const double& density0) const override;

    //! determine the energy
    double density_to_energy(
        const double& density, const double& mass, const double& density0) const override;

   private:
    //! speed of sound
    const double speedofsound_;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
