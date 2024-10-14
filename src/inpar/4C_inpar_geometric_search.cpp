/*-----------------------------------------------------------*/
/*! \file

\brief input parameter for geometric search strategy

\level 2

*/
/*-----------------------------------------------------------*/

#include "4C_inpar_geometric_search.hpp"

#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::GeometricSearch::set_valid_parameters(Teuchos::ParameterList& list)
{
  Teuchos::ParameterList& boundingvolumestrategy =
      list.sublist("BOUNDINGVOLUME STRATEGY", false, "");

  Core::Utils::double_parameter("BEAM_RADIUS_EXTENSION_FACTOR", 2.0,
      "Beams radius is multiplied with the factor and then the bounding volume only depending on "
      "the beam centerline is extended in all directions (+ and -) by that value.",
      &boundingvolumestrategy);

  Core::Utils::double_parameter("SPHERE_RADIUS_EXTENSION_FACTOR", 2.0,
      "Bounding volume of the sphere is the sphere center extended by this factor times the sphere "
      "radius in all directions (+ and -).",
      &boundingvolumestrategy);

  Core::Utils::bool_parameter("WRITE_GEOMETRIC_SEARCH_VISUALIZATION", "no",
      "If visualization output for the geometric search should be written",
      &boundingvolumestrategy);
}

FOUR_C_NAMESPACE_CLOSE
