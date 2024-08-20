/*----------------------------------------------------------------------*/
/*! \file
\brief convert problem type string to enum
\level 1
*/

/*----------------------------------------------------------------------*/
#ifndef FOUR_C_INPAR_PROBLEMTYPE_HPP
#define FOUR_C_INPAR_PROBLEMTYPE_HPP

#include "4C_config.hpp"

#include "4C_legacy_enum_definitions_problem_type.hpp"
#include "4C_utils_parameter_list.hpp"

#include <map>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Inpar
{
  namespace PROBLEMTYPE
  {
    /*! \brief Define valid parameters
     *
     * @param[in/out] list Parameter list to be filled with valid parameters and their defaults
     */
    void set_valid_parameters(Teuchos::RCP<Teuchos::ParameterList> list);

    /// create map of problem name and problem type enum
    std::map<std::string, Core::ProblemType> string_to_problem_type_map();

    /// return problem type enum for a given problem name
    Core::ProblemType string_to_problem_type(std::string name);


  }  // namespace PROBLEMTYPE
}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
