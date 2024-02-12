/*---------------------------------------------------------------------------*/
/*! \file
\brief temperature boundary condition handler for particle simulations
\level 2
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
#ifndef BACI_PARTICLE_ALGORITHM_TEMPERATURE_BC_HPP
#define BACI_PARTICLE_ALGORITHM_TEMPERATURE_BC_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_config.hpp"

#include "baci_particle_engine_typedefs.hpp"

#include <Teuchos_ParameterList.hpp>

BACI_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace PARTICLEENGINE
{
  class ParticleEngineInterface;
}

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace PARTICLEALGORITHM
{
  /*!
   * \brief temperature boundary condition handler for particle simulations
   *
   * \author Sebastian Fuchs \date 09/2018
   */
  class TemperatureBoundaryConditionHandler
  {
   public:
    /*!
     * \brief constructor
     *
     * \author Sebastian Fuchs \date 09/2018
     *
     * \param[in] params particle simulation parameter list
     */
    explicit TemperatureBoundaryConditionHandler(const Teuchos::ParameterList& params);

    /*!
     * \brief init temperature boundary condition handler
     *
     * \author Sebastian Fuchs \date 09/2018
     */
    void Init();

    /*!
     * \brief setup temperature boundary condition handler
     *
     * \author Sebastian Fuchs \date 09/2018
     *
     * \param[in] particleengineinterface interface to particle engine
     */
    void Setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface);

    /*!
     * \brief get reference to set of particle types subjected to temperature boundary conditions
     *
     * \author Sebastian Fuchs \date 09/2018
     *
     * \return set of particle types subjected to temperature boundary conditions
     */
    const std::set<PARTICLEENGINE::TypeEnum>& GetParticleTypesSubjectedToTemperatureBCSet() const
    {
      return typessubjectedtotemperaturebc_;
    };

    /*!
     * \brief insert temperature boundary condition dependent states of all particle types
     *
     * \author Sebastian Fuchs \date 09/2018
     *
     * \param[out] particlestatestotypes map of particle types and corresponding states
     */
    void InsertParticleStatesOfParticleTypes(
        std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>&
            particlestatestotypes) const;

    /*!
     * \brief set particle reference position
     *
     * \author Sebastian Fuchs \date 09/2018
     */
    void SetParticleReferencePosition() const;

    /*!
     * \brief evaluate temperature boundary condition
     *
     * \author Sebastian Fuchs \date 09/2018
     *
     * \param[in] evaltime evaluation time
     */
    void EvaluateTemperatureBoundaryCondition(const double& evaltime) const;

   protected:
    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! interface to particle engine
    std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface_;

    //! relating particle types to function ids of temperature boundary conditions
    std::map<PARTICLEENGINE::TypeEnum, int> temperaturebctypetofunctid_;

    //! set of particle types subjected to temperature boundary conditions
    std::set<PARTICLEENGINE::TypeEnum> typessubjectedtotemperaturebc_;
  };

}  // namespace PARTICLEALGORITHM

/*---------------------------------------------------------------------------*/
BACI_NAMESPACE_CLOSE

#endif
