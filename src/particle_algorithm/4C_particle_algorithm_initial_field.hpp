/*---------------------------------------------------------------------------*/
/*! \file
\brief initial field handler for particle simulations
\level 2
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
#ifndef FOUR_C_PARTICLE_ALGORITHM_INITIAL_FIELD_HPP
#define FOUR_C_PARTICLE_ALGORITHM_INITIAL_FIELD_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

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
   * \brief initial field handler for particle simulations
   *
   * \author Sebastian Fuchs \date 07/2018
   */
  class InitialFieldHandler
  {
   public:
    /*!
     * \brief constructor
     *
     * \author Sebastian Fuchs \date 07/2018
     *
     * \param[in] params particle simulation parameter list
     */
    explicit InitialFieldHandler(const Teuchos::ParameterList& params);

    /*!
     * \brief init initial field handler
     *
     * \author Sebastian Fuchs \date 07/2018
     */
    void init();

    /*!
     * \brief setup initial field handler
     *
     * \author Sebastian Fuchs \date 07/2018
     *
     * \param[in] particleengineinterface interface to particle engine
     */
    void setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface);

    /*!
     * \brief set initial fields
     *
     * \author Sebastian Fuchs \date 07/2018
     */
    void set_initial_fields();

   protected:
    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! interface to particle engine
    std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface_;

    //! relating particle types to function ids
    std::map<PARTICLEENGINE::StateEnum, std::map<PARTICLEENGINE::TypeEnum, int>>
        statetotypetofunctidmap_;
  };

}  // namespace PARTICLEALGORITHM

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
