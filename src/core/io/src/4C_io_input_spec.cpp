// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_spec.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_io_yaml.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Core::IO::InputSpec::InputSpec(std::unique_ptr<Internal::InputSpecImpl> pimpl)
    : pimpl_(std::move(pimpl))
{
}

void Core::IO::InputSpec::fully_parse(
    ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");

  pimpl_->parse(parser, container);
  if (!parser.at_end())
  {
    std::stringstream ss;
    container.print(ss);
    std::string remainder(parser.get_unparsed_remainder());
    FOUR_C_THROW(
        "After parsing, the line still contains '{}'.\nParsed parameters: {}", remainder, ss.str());
  }
}

void Core::IO::InputSpec::match(ConstYamlNodeRef yaml, InputParameterContainer& container) const
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");

  const auto& spec_name = impl().name();
  FOUR_C_ASSERT_ALWAYS(spec_name != "",
      "You are trying to match an unnamed InputSpec. "
      "You can only match a named spec, not an all_of or one_of spec.");

  Internal::MatchTree match_tree{*this, yaml};

  if (auto* stores_to = impl().data.stores_to;
      stores_to && *stores_to != typeid(InputParameterContainer))
  {
    FOUR_C_THROW(
        "Not implemented: the top-level InputSpec that is used for matching must store to the "
        "InputParameterContainer type, but this one stores to '{}'.",
        Core::Utils::try_demangle(stores_to->name()).c_str());
  }

  InputSpecBuilders::Storage storage;
  Internal::init_storage_with_container(storage);
  impl().match(yaml, storage, match_tree.root());
  container.merge(std::any_cast<InputParameterContainer&&>(std::move(storage)));

  match_tree.assert_match();
}

void Core::IO::InputSpec::emit(YamlNodeRef yaml,
    FourC::Core::IO::InputParameterContainer& container, InputSpecEmitOptions options) const
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");

  bool success = pimpl_->emit(yaml, container, options);
  if (!success)
  {
    std::stringstream ss;
    ss << "Failed to emit this data\n";
    container.print(ss);
    ss << "under the following specification\n";
    print_as_dat(ss);
    FOUR_C_THROW("{}", ss.str());
  }
}

void Core::IO::InputSpec::print_as_dat(std::ostream& stream) const
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");

  pimpl_->print(stream, 0u);
}

void Core::IO::InputSpec::emit_metadata(YamlNodeRef yaml) const
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");

  pimpl_->emit_metadata(yaml);
}

Core::IO::Internal::InputSpecImpl& Core::IO::InputSpec::impl()
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");
  return *pimpl_;
}

const Core::IO::Internal::InputSpecImpl& Core::IO::InputSpec::impl() const
{
  FOUR_C_ASSERT(pimpl_, "InputSpec is empty.");
  return *pimpl_;
}

FOUR_C_NAMESPACE_CLOSE
