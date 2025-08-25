// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_FILE_UTILS_HPP
#define FOUR_C_IO_INPUT_FILE_UTILS_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <functional>
#include <ostream>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;

  namespace Nurbs
  {
    class Knotvector;
  }
}  // namespace Core::FE

namespace Core::IO
{
  class InputFile;
  class InputSpec;
}  // namespace Core::IO

namespace Core::IO
{

  /**
   * Print a section header padded with dashes to 67 characters.
   */
  void print_section_header(std::ostream& out, const std::string& header);

  /**
   * Print @p spec into a dat file section with given @p header.
   */
  void print_section(std::ostream& out, const std::string& header, const InputSpec& spec);

  /**
   * Split the given @p line into a key-value pair. Key and value are normally separated by
   * whitespace. In case there are multiple distinct whitespace groups in one line, the first of
   * these is assumed to be the separator and all the other whitespace is assumed to be part of
   * the value. Key and value may also be separated by an equals sign "=" and at least one
   * whitespace character on both sides. In this case, key and value may contain spaces
   * internally. Leading and trailing whitespace is trimmed from both key and value.
   *
   * @throws Core::Exception If the @p line cannot be read.
   *
   * @return A pair of key and value.
   */
  std::pair<std::string, std::string> read_key_value(const std::string& line);

  void read_parameters_in_section(
      InputFile& input, const std::string& section_name, Teuchos::ParameterList& list);

  /**
   * Read a node-design topology section. This is a collective call that propagates data that
   * may only be available on rank 0 to all ranks.
   *
   * @param input The input file.
   * @param name Name of the topology to read
   * @param dobj_fenode Resulting collection of all nodes that belong to a design.
   * @param get_discretization Callback to return a discretization by name.
   */
  void read_design(InputFile& input, const std::string& name,
      std::vector<std::vector<int>>& dobj_fenode,
      const std::function<const Core::FE::Discretization&(const std::string& name)>&
          get_discretization);

  /**
   * @brief Read the knot vector section (for isogeometric analysis)
   *
   * @param  input         (in ): InputFile object
   * @param  name           (in ): Name/type of discretisation
   *
   * @return The Knotvector object read from the input file.
   */
  std::unique_ptr<Core::FE::Nurbs::Knotvector> read_knots(
      InputFile& input, const std::string& name);

}  // namespace Core::IO


FOUR_C_NAMESPACE_CLOSE

#endif
