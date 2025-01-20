// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_spec.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_io_yaml_emitter.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Core::IO::InputSpec::InputSpec(std::unique_ptr<Internal::InputSpecTypeErasedBase> pimpl)
    : pimpl_(std::move(pimpl))
{
}

Core::IO::InputSpec::~InputSpec() = default;

Core::IO::InputSpec::InputSpec(const InputSpec& other)
    : pimpl_(other.pimpl_ ? other.pimpl_->clone() : nullptr)
{
}

Core::IO::InputSpec& Core::IO::InputSpec::operator=(const InputSpec& other)
{
  if (this == &other) return *this;

  pimpl_ = other.pimpl_ ? other.pimpl_->clone() : nullptr;

  return *this;
}

void Core::IO::InputSpec::fully_parse(
    ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");

  pimpl_->parse(parser, container);
  FOUR_C_ASSERT_ALWAYS(parser.at_end(), "After parsing, the line still contains '%s'.",
      std::string(parser.get_unparsed_remainder()).c_str());
}

void Core::IO::InputSpec::print_as_dat(
    std::ostream& stream, const Core::IO::InputParameterContainer& container) const
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");

  pimpl_->print(stream, container);
}

void Core::IO::InputSpec::emit_metadata(YamlEmitter& yaml) const
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");

  auto root = yaml.node;
  root |= ryml::MAP;
  pimpl_->emit_metadata(root.append_child());
}

Core::IO::Internal::InputSpecTypeErasedBase& Core::IO::InputSpec::impl()
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");
  return *pimpl_;
}

const Core::IO::Internal::InputSpecTypeErasedBase& Core::IO::InputSpec::impl() const
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");
  return *pimpl_;
}

FOUR_C_NAMESPACE_CLOSE
