/*----------------------------------------------------------------------*/
/*! \file

\brief base class for geometry pair evaluation data subcontainers.

\level 1
*/
// End doxygen header.


#ifndef BACI_GEOMETRY_PAIR_EVALUATION_DATA_BASE_HPP
#define BACI_GEOMETRY_PAIR_EVALUATION_DATA_BASE_HPP


#include "baci_config.hpp"

#include <Teuchos_ParameterList.hpp>

BACI_NAMESPACE_OPEN

namespace GEOMETRYPAIR
{
  /**
   * \brief A base class that all geometry pair evaluation data container have to inherit from.
   */
  class GeometryEvaluationDataBase
  {
   public:
    /**
     * \brief Constructor.
     *
     * \param input_parameter_list (in) Parameter list with the geometry evaluation parameters.
     */
    GeometryEvaluationDataBase(const Teuchos::ParameterList& input_parameter_list){};

    /**
     * \brief Destructor.
     */
    virtual ~GeometryEvaluationDataBase() = default;

    /**
     * \brief Clear data in this evaluation container.
     */
    virtual void Clear(){};
  };
}  // namespace GEOMETRYPAIR

BACI_NAMESPACE_CLOSE

#endif
