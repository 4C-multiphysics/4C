/*----------------------------------------------------------------------*/
/*! \file

\brief base class for geometry pair evaluation data subcontainers.

\level 1
*/
// End doxygen header.


#ifndef FOUR_C_GEOMETRY_PAIR_EVALUATION_DATA_BASE_HPP
#define FOUR_C_GEOMETRY_PAIR_EVALUATION_DATA_BASE_HPP


#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

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
    virtual void clear(){};
  };
}  // namespace GEOMETRYPAIR

FOUR_C_NAMESPACE_CLOSE

#endif
