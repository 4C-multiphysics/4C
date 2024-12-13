// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_LINEDEFINITION_HPP
#define FOUR_C_IO_LINEDEFINITION_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputFile;
}

namespace Input
{
  namespace Internal
  {
    class LineDefinitionImplementation;
  }

  /**
   * LineDefinition defines how one specific line in a dat file looks like. The
   * idea is, that each line consists of a list of components.
   *
   * Reading a LineDefinition means filling the components with those values
   * found at the input line.
   *
   * @note This class has value semantics.
   */
  class LineDefinition
  {
   public:
    /**
     * An empty LineDefinition without any components.
     */
    LineDefinition();

    /**
     * Destructor.
     */
    ~LineDefinition();

    /**
     * Copy constructor.
     */
    LineDefinition(const LineDefinition& other);

    /**
     * Copy assignment.
     */
    LineDefinition& operator=(const LineDefinition& other);

    /**
     * Move constructor.
     */
    LineDefinition(LineDefinition&& other) noexcept;

    /**
     * Move assignment.
     */
    LineDefinition& operator=(LineDefinition&& other) noexcept;

    /**
     * Builder class to incrementally add components and then build a LineDefinition from them.
     * Usage example:
     *
     * @code
     *   const LineDefinition line =
     *     LineDefinition::Builder().add_tag("a").add_int("i).build();
     * @endcode
     */
    class Builder
    {
     public:
      /**
       * A function type that may be supplied to some of the Add... functions in this class to
       * define how the number of vector entries should be determined from the @p already_read_line.
       */
      using LengthDefinition =
          std::function<std::size_t(const Core::IO::InputParameterContainer& already_read_line)>;

      /**
       * Create a new Builder.
       */
      Builder();

      /**
       * Destructor.
       */
      ~Builder();
      /**
       * Copy constructor.
       */
      Builder(const Builder& other);

      /**
       * Copy assignment.
       */
      Builder& operator=(const Builder& other);

      /**
       * Move constructor.
       */
      Builder(Builder&& other) noexcept;

      /**
       * Move assignment.
       */
      Builder& operator=(Builder&& other) noexcept;

      /**
       * Initialize a Builder with all components from an existing LineDefinition.
       */
      Builder(const LineDefinition& line_definition);

      /**
       * Convert the gathered components into a LineDefinition.
       *
       * @note This overload is chosen for rvalues and can steal the data gathered in the Builder.
       * After calling this function, the builder is in a moved-from state.
       */
      [[nodiscard]] LineDefinition build() &&;

      /**
       * Convert the gathered components into a LineDefinition.
       *
       * @note This overload copies the gathered data to the newly created LineDefinition.
       */
      [[nodiscard]] LineDefinition build() const&;

      /// Add a single string definition without a value.
      Builder& add_optional_tag(const std::string& name);

      /// Add a single string definition without a value.
      Builder& add_tag(std::string name);

      /// Add a name followed by a variable string
      Builder& add_named_string(std::string name);

      /// Add a name followed by an integer variable
      Builder& add_named_int(std::string name);

      /// Add a name followed by a vector of integer variables
      Builder& add_named_int_vector(std::string name, int length);

      /// Add a name followed by a double variable
      Builder& add_named_double(std::string name);

      /// Add a name followed by a vector of double variables
      Builder& add_named_double_vector(std::string name, int length);

      /// Add a name followed by a vector of double variables
      Builder& add_named_string_vector(std::string name, int length);

      /*!
       * Add a name followed by a vector of double variables.
       *
       * The function @p length_definition specifies how to obtain the number of to be read vector
       * entries from the previously added components. See LengthDefinition for details.
       */
      Builder& add_named_double_vector(std::string name, LengthDefinition length_definition);

      /// Add a name followed by a variable string
      Builder& add_optional_named_string(const std::string& name);

      /// Add a name followed by an integer variable
      Builder& add_optional_named_int(const std::string& name);

      /// Add a name followed by a vector of integer variables
      Builder& add_optional_named_int_vector(const std::string& name, int length);

      /// Add a name followed by a double variable
      Builder& add_optional_named_double(const std::string& name);

      /// Add a name followed by a vector of double variables
      Builder& add_optional_named_double_vector(const std::string& name, int length);

      /*!
       * Add a name followed by a vector of double variables.
       *
       * The parameter \p lengthdef specifies the name of an integer component
       * that gives the length of the vector. The integer component has to
       * precede the vector definition on the input line.
       */
      Builder& add_optional_named_double_vector(
          const std::string& name, LengthDefinition lengthdef);

      /// Add a name followed by a vector of string variables.
      Builder& add_optional_named_string_vector(const std::string& name, int length);

      /**
       * Add a name followed by a vector of string variables.
       *
       * The parameter \p lengthdef specifies the name of an integer component
       * that gives the length of the vector. The integer component has to
       * precede the vector definition on the input line.
       * The space defines the separation between a string and the next one.
       */
      Builder& add_optional_named_string_vector(
          const std::string& name, LengthDefinition lengthdef);

      /*!
       * Add a name followed by a vector of double variables.
       *
       * The parameter \p lengthdef specifies the name of an integer component
       * that gives the length of the vector. The integer component has to
       * precede the vector definition on the input line.
       */
      Builder& add_optional_named_pair_of_string_and_double_vector(
          const std::string& name, LengthDefinition lengthdef);

     private:
      /// Implementation details are hidden behind the PIMPL idiom.
      std::unique_ptr<Internal::LineDefinitionImplementation> pimpl_;
    };

    /// print to dat file comment
    void print(std::ostream& stream) const;

    /**
     * If reading succeeds, returns the data. Otherwise, returns an empty std::optional.
     */
    std::optional<Core::IO::InputParameterContainer> read(std::istream& stream);

    [[nodiscard]] const Core::IO::InputParameterContainer& container() const;

    //@}

   private:
    /// Constructor called by the Builder to directly pass on implementation.
    explicit LineDefinition(std::unique_ptr<Internal::LineDefinitionImplementation>&& pimpl);

    /// Implementation details are hidden behind the PIMPL idiom.
    std::unique_ptr<Internal::LineDefinitionImplementation> pimpl_;
  };


  /**
   * Helper functor to parse the length of a vector input from an integer component. This
   * functor is compatible with LengthDefinition.
   * Example:
   *
   * @code
   *    [...].add_double_vector("values", FromIntNamed("NUMVALUES")) [...]
   * @endcode
   */
  struct LengthFromIntNamed
  {
    LengthFromIntNamed(std::string definition_name);

    std::size_t operator()(const Core::IO::InputParameterContainer& already_read_line);

   private:
    std::string definition_name_;
  };

}  // namespace Input

FOUR_C_NAMESPACE_CLOSE

#endif
