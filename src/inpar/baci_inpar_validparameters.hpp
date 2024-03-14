/*----------------------------------------------------------------------*/
/*! \file
\brief Setup of the list of valid input parameters

\level 1

*/
/*----------------------------------------------------------------------*/



#ifndef BACI_INPAR_VALIDPARAMETERS_HPP
#define BACI_INPAR_VALIDPARAMETERS_HPP

#include "baci_config.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <iostream>
#include <string>

BACI_NAMESPACE_OPEN

namespace IO
{
  class Pstream;
}


namespace INPUT
{
  /// construct list with all parameters and documentation
  Teuchos::RCP<const Teuchos::ParameterList> ValidParameters();

  /// print all parameters that have a default value
  void PrintDefaultParameters(IO::Pstream& stream, const Teuchos::ParameterList& list);

  /// print flag sections of dat file with given list
  void PrintDatHeader(std::ostream& stream, const Teuchos::ParameterList& list,
      std::string parentname = "", bool comment = true);

  /**
   * Return true if the @p list contains any parameter that has whitespace in the key name.
   *
   * @note This is needed for the NOX parameters whose keywords and value have white spaces and
   * thus '=' are inserted to distinguish them.
   */
  bool NeedToPrintEqualSign(const Teuchos::ParameterList& list);

}  // namespace INPUT


/*! print list of valid parameters with documentation */
void PrintValidParameters();

/*! print help message */
void PrintHelpMessage();

/*! print flag sections of dat file with default flags */
void PrintDefaultDatHeader();


BACI_NAMESPACE_CLOSE

#endif