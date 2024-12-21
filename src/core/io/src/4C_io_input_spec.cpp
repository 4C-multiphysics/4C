// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_spec.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

void Core::IO::fully_parse(ValueParser& parser, const Core::IO::InputSpec& input_spec,
    Core::IO::InputParameterContainer& container)
{
  input_spec.parse(parser, container);
  FOUR_C_ASSERT_ALWAYS(parser.at_end(), "After parsing, the line still contains '%s'.",
      std::string(parser.get_unparsed_remainder()).c_str());
}

FOUR_C_NAMESPACE_CLOSE
