// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_spec_validators.hpp"

#include <regex>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace
{
  struct PatternValidator
  {
    explicit PatternValidator(std::string pattern)
        : pattern(std::move(pattern)), regex(this->pattern, std::regex::ECMAScript)
    {
    }

    bool operator()(const std::string& v) const { return std::regex_search(v, regex); }

    void describe(std::ostream& os) const { os << "pattern{" << pattern << "}"; }

    void emit_metadata(Core::IO::YamlNodeRef yaml) const
    {
      auto& node = yaml.node;
      node |= ryml::MAP;
      node["pattern"] |= ryml::MAP;
      node["pattern"]["pattern"] << pattern;
    }


    std::string pattern;
    std::regex regex;
  };
}  // namespace

Core::IO::InputSpecBuilders::Validators::Validator<std::string>
Core::IO::InputSpecBuilders::Validators::pattern(std::string pattern)
{
  return PatternValidator(std::move(pattern));
}

FOUR_C_NAMESPACE_CLOSE
