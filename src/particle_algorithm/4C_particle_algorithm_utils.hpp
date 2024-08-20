/*---------------------------------------------------------------------------*/
/*! \file
\brief utils for particle algorithm
\level 2
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
#ifndef FOUR_C_PARTICLE_ALGORITHM_UTILS_HPP
#define FOUR_C_PARTICLE_ALGORITHM_UTILS_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_utils_parameter_list.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN

namespace PARTICLEALGORITHM
{
  namespace UTILS
  {
    /*!
     * \brief read parameters relating particle types to values
     *
     * Read parameters relating particle types to specific values from the parameter list.
     *
     * \author Sebastian Fuchs \date 07/2018
     *
     * \tparam valtype type of value
     *
     * \param[in]  params       particle simulation parameter list
     * \param[in]  name         parameter name
     * \param[out] typetovalmap map relating particle types to specific values
     */
    template <typename Valtype>
    void read_params_types_related_to_values(const Teuchos::ParameterList& params,
        const std::string& name, std::map<PARTICLEENGINE::TypeEnum, Valtype>& typetovalmap);

  }  // namespace UTILS

}  // namespace PARTICLEALGORITHM

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
