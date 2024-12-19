// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_linedefinition.hpp"

#include "4C_io_input_file.hpp"
#include "4C_io_input_line.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_utils_exceptions.hpp"

#include <functional>
#include <iterator>
#include <memory>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace Input
{
  using namespace Core::IO::InputLineBuilders;

  namespace Internal
  {
    /**
     * Internal data used in the implementation. This type has value semantics.
     */
    class LineDefinitionImplementation
    {
     public:
      /// Components that make up an InputLine.
      std::vector<Core::IO::Component> components_;

      /// Store the read data.
      Core::IO::InputParameterContainer container_;
    };
  }  // namespace Internal


  LineDefinition::LineDefinition()
      : pimpl_(std::make_unique<Internal::LineDefinitionImplementation>())
  {
  }


  // The PIMPL idiom forces us to default a few special members in the implementation file.
  LineDefinition::~LineDefinition() = default;

  LineDefinition::LineDefinition(LineDefinition&&) noexcept = default;

  LineDefinition& LineDefinition::operator=(LineDefinition&&) noexcept = default;

  LineDefinition::LineDefinition(const LineDefinition& other)
      : pimpl_(std::make_unique<Internal::LineDefinitionImplementation>(*other.pimpl_))
  {
  }

  LineDefinition& LineDefinition::operator=(const LineDefinition& other)
  {
    pimpl_ = std::make_unique<Internal::LineDefinitionImplementation>(*other.pimpl_);
    return *this;
  }

  LineDefinition::LineDefinition(std::unique_ptr<Internal::LineDefinitionImplementation>&& pimpl)
      : pimpl_(std::move(pimpl))
  {
  }



  LineDefinition::Builder::Builder()
      : pimpl_(std::make_unique<Internal::LineDefinitionImplementation>())
  {
  }

  // The PIMPL idiom forces us to default this in the implementation file.
  LineDefinition::Builder::~Builder() = default;

  LineDefinition::Builder::Builder(LineDefinition::Builder&&) noexcept = default;

  LineDefinition::Builder& LineDefinition::Builder::operator=(
      LineDefinition::Builder&&) noexcept = default;

  LineDefinition::Builder::Builder(const LineDefinition::Builder& other)
      : pimpl_(std::make_unique<Internal::LineDefinitionImplementation>(*other.pimpl_))
  {
  }

  LineDefinition::Builder& LineDefinition::Builder::operator=(const LineDefinition::Builder& other)
  {
    pimpl_ = std::make_unique<Internal::LineDefinitionImplementation>(*other.pimpl_);
    return *this;
  }

  LineDefinition::Builder::Builder(const LineDefinition& line_definition)
      : pimpl_(std::make_unique<Internal::LineDefinitionImplementation>(*line_definition.pimpl_))
  {
  }

  LineDefinition LineDefinition::Builder::build() &&
  {
    // Steal the internal data since this operation is performed on an rvalue.
    return LineDefinition(std::move(pimpl_));
  }

  LineDefinition LineDefinition::Builder::build() const&
  {
    // Make a copy of the implementation details
    return LineDefinition(std::make_unique<Internal::LineDefinitionImplementation>(*pimpl_));
  }


  LineDefinition::Builder& LineDefinition::Builder::add_tag(std::string name)
  {
    pimpl_->components_.emplace_back(tag({.name = std::move(name)}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_string(std::string name)
  {
    pimpl_->components_.emplace_back(entry<std::string>({.name = std::move(name)}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_int(std::string name)
  {
    pimpl_->components_.emplace_back(entry<int>({.name = std::move(name)}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_int_vector(
      std::string name, int length)
  {
    pimpl_->components_.emplace_back(
        entry<std::vector<int>>({.name = std::move(name), .size = length}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_double(std::string name)
  {
    pimpl_->components_.emplace_back(entry<double>({.name = std::move(name)}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_double_vector(
      std::string name, int length)
  {
    pimpl_->components_.emplace_back(
        entry<std::vector<double>>({.name = std::move(name), .size = length}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_string_vector(
      std::string name, int length)
  {
    pimpl_->components_.emplace_back(
        entry<std::vector<std::string>>({.name = std::move(name), .size = length}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_double_vector(
      std::string name, LengthDefinition length_definition)
  {
    pimpl_->components_.emplace_back(
        entry<std::vector<double>>({.name = std::move(name), .size = length_definition}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_path(std::string name)
  {
    pimpl_->components_.emplace_back(entry<std::filesystem::path>({.name = std::move(name)}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_tag(const std::string& name)
  {
    pimpl_->components_.emplace_back(tag({.name = name, .default_value = false}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_string(
      const std::string& name)
  {
    pimpl_->components_.emplace_back(entry<std::string>({.name = name, .default_value = "''"}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_int(const std::string& name)
  {
    pimpl_->components_.emplace_back(entry<int>({.name = name, .default_value = 0}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_int_vector(
      const std::string& name, int length)
  {
    pimpl_->components_.emplace_back(entry<std::vector<int>>(
        {.name = name, .default_value = std::vector<int>(length), .size = length}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_double(
      const std::string& name)
  {
    pimpl_->components_.emplace_back(entry<double>({.name = name, .default_value = 0.0}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_double_vector(
      const std::string& name, int length)
  {
    pimpl_->components_.emplace_back(entry<std::vector<double>>(
        {.name = name, .default_value = std::vector<double>(length), .size = length}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_double_vector(
      const std::string& name, LengthDefinition lengthdef)
  {
    pimpl_->components_.emplace_back(entry<std::vector<double>>(
        {.name = name, .default_value = std::vector<double>(), .size = lengthdef}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_string_vector(
      const std::string& name, int length)
  {
    pimpl_->components_.emplace_back(entry<std::vector<std::string>>(
        {.name = name, .default_value = std::vector<std::string>(length), .size = length}));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_string_vector(
      const std::string& name, LengthDefinition lengthdef)
  {
    pimpl_->components_.emplace_back(entry<std::vector<std::string>>(
        {.name = name, .default_value = std::vector<std::string>(), .size = lengthdef}));
    return *this;
  }



  LineDefinition::Builder&
  LineDefinition::Builder::add_optional_named_pair_of_string_and_double_vector(
      const std::string& name, LengthDefinition lengthdef)
  {
    pimpl_->components_.emplace_back(
        entry<std::vector<std::pair<std::string, double>>>({.name = name,
            .default_value = std::vector<std::pair<std::string, double>>{},
            .size = lengthdef}));
    return *this;
  }



  void LineDefinition::print(std::ostream& stream) const
  {
    Core::IO::InputLine input_line(pimpl_->components_);
    input_line.print_default(stream);
  }



  std::optional<Core::IO::InputParameterContainer> LineDefinition::read(
      std::istream& stream, const ReadContext& context)
  {
    pimpl_->container_ = Core::IO::InputParameterContainer();

    // extract everything from the stream into a string
    std::stringstream ss;
    ss << stream.rdbuf();
    std::string line = ss.str();

    Core::IO::InputLine input_line(pimpl_->components_);
    if (not input_line.parse(line, pimpl_->container_,
            {
                .throw_on_error = false,
                .store_default_values = false,
            },
            {
                .base_path = context.input_file.parent_path(),
            }))
    {
      return std::nullopt;
    }

    return pimpl_->container_;
  }

  const Core::IO::InputParameterContainer& LineDefinition::container() const
  {
    return pimpl_->container_;
  }


  LengthFromIntNamed::LengthFromIntNamed(std::string definition_name)
      : definition_name_(std::move(definition_name))
  {
  }


  std::size_t LengthFromIntNamed::operator()(
      const Core::IO::InputParameterContainer& already_read_line)
  {
    int length = already_read_line.get<int>(definition_name_);
    return length;
  }
}  // namespace Input

FOUR_C_NAMESPACE_CLOSE
