// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_SPEC_HPP
#define FOUR_C_IO_INPUT_SPEC_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputSpec;
  class InputParameterContainer;
  class ValueParser;

  /**
   * Use the @p parser to parse whatever @p input_spec expects. The results are stored in the
   * @p container. If parsing fails, an exception is thrown.
   */
  void fully_parse(
      ValueParser& parser, const InputSpec& input_spec, InputParameterContainer& container);

}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
