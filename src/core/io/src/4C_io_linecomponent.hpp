// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_LINECOMPONENT_HPP
#define FOUR_C_IO_LINECOMPONENT_HPP


#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"

#include <Teuchos_Array.hpp>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Input
{

  /**
   * Interface for components in an input line.
   *
   * TODO: this class unifies duplicated interfaces and is currently a collection of various
   * methods. This is not the final interface.
   *
   */
  class LineComponent
  {
   public:
    /// construct with the name of the corresponding variable in the material
    explicit LineComponent(std::string name, bool optional = false);

    /// virtual destructor is mandatory
    virtual ~LineComponent() = default;

    /// write my part of the default (comment) line of the condition
    virtual void default_line(std::ostream& stream) = 0;

    /// A human-readable description of this component used in help messages.
    virtual void describe(std::ostream& stream) {}

    virtual std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) = 0;

    /* write my part of a default line of the condition
     * as restructuredText for ReadTheDocs
     * For some components it returns the same output as Default line (but as a string).
     * However, for many components the output in ReadTheDocs is more illustrative.
     */
    virtual std::string write_read_the_docs() { return ""; }


    virtual Teuchos::Array<std::string> get_options() { return {}; }

    /// the name of my variable inside a material
    [[nodiscard]] std::string name() const { return name_; }

   protected:
    /// for optional components
    bool optional_;

   private:
    /// my material variable name
    std::string name_;
  };


  /**
   * A function type used to determine the length of other components from the
   * @p already_parsed_container.
   */
  using LengthDefinition =
      std::function<int(const Core::IO::InputParameterContainer& already_parsed_container)>;

  struct LengthFromInt
  {
    /**
     * Determine the length of the vector component at runtime from an Input::IntComponent of
     * given @p name.
     */
    LengthFromInt(std::string name) : name_(std::move(name)) {}
    int operator()(const Core::IO::InputParameterContainer& already_read_line)
    {
      return already_read_line.get<int>(name_);
    }

   private:
    std::string name_;
  };


  /// @brief A fixed string without any effect on Container
  ///
  /// This is really just a separator at the input line.
  ///
  /// The reason we need this is that the we specify the order of the input line
  /// part. It might be reasonable to specify names that have to appear in the
  /// dat file to enhance human readability.
  ///
  class SeparatorComponent : public Input::LineComponent
  {
   public:
    SeparatorComponent(std::string separator, std::string description = {}, bool optional = false);

    void default_line(std::ostream& stream) override;

    void describe(std::ostream& stream) override;

    std::vector<std::string> write_read_the_docs_table_row() const;

    std::string write_read_the_docs() override;

    std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) override;

   private:
    /// separator string, i.e. the NAME of variable in DAT input file
    std::string separator_;

    /// description attached to the field separator
    std::string description_;
  };


  /**
   * Component that parses a single string.
   */
  class StringComponent : public Input::LineComponent
  {
   public:
    StringComponent(std::string name, std::string defaultvalue, bool optional = false);

    void default_line(std::ostream& stream) override;

    void describe(std::ostream& stream) override;

    std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) override;

   private:
    /// default value
    std::string defaultvalue_;
  };



  /**
   * Parse a string from a selection of different strings. The parsed strings are either converted
   * into an integer or a string value.
   */
  class SelectionComponent : public Input::LineComponent
  {
   public:
    SelectionComponent(std::string name, std::string defaultvalue,
        const Teuchos::Array<std::string>& datfilevalues,
        const Teuchos::Array<std::string>& stringcondvalues, bool optional = false);

    SelectionComponent(std::string name, std::string defaultvalue,
        const Teuchos::Array<std::string>& datfilevalues, const Teuchos::Array<int>& intcondvalues,
        bool optional = false);

    void default_line(std::ostream& stream) override;

    std::string write_read_the_docs() override;

    Teuchos::Array<std::string> get_options() override;

    std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) override;

   private:
    std::string defaultvalue_;
    Teuchos::Array<std::string> datfilevalues_;
    Teuchos::Array<std::string> stringcondvalues_;
    Teuchos::Array<int> intcondvalues_;
    const bool stringtostring_;
  };



  /**
   * Additional parameters for IntComponents.
   */
  struct IntComponentData
  {
    int default_value{0};
    bool optional{false};
  };

  /**
   * Parse an integer.
   */
  class IntComponent : public Input::LineComponent
  {
   public:
    explicit IntComponent(std::string name, IntComponentData data = IntComponentData{});

    void default_line(std::ostream& stream) override;

    void describe(std::ostream& stream) override;

    std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) override;

    std::string write_read_the_docs() override;

   private:
    IntComponentData data_;
  };


  /**
   * Parse a vector of integers.
   */
  class IntVectorComponent : public Input::LineComponent
  {
   public:
    IntVectorComponent(std::string name, int length, IntComponentData data = {});

    IntVectorComponent(
        std::string name, LengthDefinition length_from_component, IntComponentData data = {});

    void default_line(std::ostream& stream) override;

    std::string write_read_the_docs() override;

    void describe(std::ostream& stream) override;

    std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) override;

    void set_length(int newlength);

   private:
    std::variant<int, LengthDefinition> length_;

    IntComponentData data_;
  };



  /**
   * Additional data for RealComponents.
   */
  struct RealComponentData
  {
    double default_value{};
    // Legacy: Reals are optional by default
    bool optional{true};
  };

  /**
   * Parse a single double value.
   */
  class RealComponent : public Input::LineComponent
  {
   public:
    RealComponent(std::string name, RealComponentData data = {});

    void default_line(std::ostream& stream) override;

    void describe(std::ostream& stream) override;

    std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) override;

   private:
    RealComponentData data_;
  };


  /**
   * Parse a vector of doubles.
   */
  class RealVectorComponent : public Input::LineComponent
  {
   public:
    RealVectorComponent(std::string name, int length, RealComponentData data = {});

    RealVectorComponent(
        std::string name, LengthDefinition length_from_component, RealComponentData data = {});

    void default_line(std::ostream& stream) override;

    std::string write_read_the_docs() override;

    void set_length(int newlength);

    void describe(std::ostream& stream) override;

    std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) override;

   private:
    std::variant<int, LengthDefinition> length_;
    RealComponentData data_;
  };


  /**
   * Parse a single bool value.
   */
  class BoolComponent : public Input::LineComponent
  {
   public:
    explicit BoolComponent(
        std::string name, const bool defaultvalue = false, bool optional = false);

    void default_line(std::ostream& stream) override;

    void describe(std::ostream& stream) override;

    std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) override;

   private:
    void print_yes_no(std::ostream& stream,  ///< stream to add output
        const bool value                     ///< the Boolean value to transcribe
    ) const;

    /// string constant which is identified with 'true'
    static const std::string lineTrue_;

    /// string constant which is identified with 'false'
    static const std::string lineFalse_;

    /// a default value
    bool defaultvalue_;
  };


  /**
   * This component contains a series of LineComponents that are selected by a key parameter.
   */
  class SwitchComponent : public Input::LineComponent
  {
    //! This component only supports integers for keys. Unscoped enums convert automatically to
    //! int and can be used to increase readability.
    using KeyType = int;

   public:
    /**
     * Define a component that selects one of the @p choices at runtime. This component lets the
     * user create composite structures of nested components. Depending on the integer, the
     * corresponding vector of components from the @p choices map is selected and reading
     * continues with these components. By default, the selection is made based on @p default_key.
     */
    SwitchComponent(std::string name, const KeyType& default_key,
        std::map<KeyType,
            std::pair<std::string, std::vector<std::shared_ptr<Input::LineComponent>>>>
            choices);

    void default_line(std::ostream& stream) override;

    std::string write_read_the_docs() override;

    std::vector<std::string> write_read_the_docs_lines();

    Teuchos::Array<std::string> get_options() override;

    std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) override;

   private:
    KeyType default_key_;
    std::map<KeyType, std::pair<std::string, std::vector<std::shared_ptr<Input::LineComponent>>>>
        choices_;

    //! Helper component to read the selected key from input.
    std::unique_ptr<Input::SelectionComponent> component_for_key_;
  };


  /**
   * A LineComponent where an input string is processed by a user-defined operation.
   */
  class ProcessedComponent : public Input::LineComponent
  {
   public:
    /*!
     * @brief Define a component that reads the component value as string, and conducts a given
     * @p process_operation on this string @p read_string. @p process_operation returns a type
     * @p T object. Add this object to a @p container for the given key @p name .
     *
     * As an example, you can use this class to:
     * - post-process a given string as a LINALG matrix and store this LINALG matrix in the
     *   container. Therefore, define a process_operation parsing the string into the LINALG
     *   matrix.
     * - post-process a file path to read the content of this file and store this content in the
     *   container. Therefore, define a process_operation that reads the file into the desired
     *   object of type T.
     * - post-process a given string into a boolean flag. Therefore, define the logic whether for
     *   the given string, the stored boolean is true or false.
     */
    template <typename T>
    ProcessedComponent(const std::string& name,
        std::function<T(const std::string&)> process_operation, bool optional = false)
        : LineComponent(name, optional),
          insert_operation_([process_operation, name](const std::string& read_string,
                                Core::IO::InputParameterContainer& container)
              { container.add(name, process_operation(read_string)); })
    {
    }

    void default_line(std::ostream& stream) override;

    std::shared_ptr<std::stringstream> read(const std::string& section_name,
        std::shared_ptr<std::stringstream> condline,
        Core::IO::InputParameterContainer& container) override;

   private:
    //! add processed data to the container
    std::function<void(
        const std::string& read_string, Core::IO::InputParameterContainer& container)>
        insert_operation_;
  };


  /// add a separator followed by a single integer value
  ///
  /// The name on the input line becomes the name used to put the int value into
  /// the parsed container.
  template <typename DefinitionType>
  inline void add_named_int(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& description = {}, const int defaultvalue = 0,
      const bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, description, optional));
    IntComponentData data{};
    data.default_value = defaultvalue;
    data.optional = optional;
    definition->add_component(std::make_shared<Input::IntComponent>(name, data));
  }

  /// add a separator followed by a number integer values
  ///
  /// The name on the input line becomes the name used to put the int value into
  /// the parsed Container.
  template <typename DefinitionType>
  inline void add_named_int_vector(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& description, const int size,
      const int defaultvalue = 0, const bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, description, optional));
    IntComponentData data{};
    data.default_value = defaultvalue;
    data.optional = optional;
    definition->add_component(std::make_shared<Input::IntVectorComponent>(name, size, data));
  }

  /// add a separator followed by a number integer values
  ///
  /// The name on the input line becomes the name used to put the int value into
  /// the parsed Container.
  template <typename DefinitionType>
  inline void add_named_int_vector(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& description, const std::string& sizename,
      const int defaultvalue = 0, const bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, description, optional));
    IntComponentData data{};
    data.default_value = defaultvalue;
    data.optional = optional;
    definition->add_component(
        std::make_shared<Input::IntVectorComponent>(name, LengthFromInt(sizename), data));
  }

  /// add a separator followed by a single real value
  ///
  /// The name on the input line becomes the name used to put the value into the parsed Container
  template <typename DefinitionType>
  inline void add_named_real(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& description = {}, const double defaultvalue = 0.0,
      const bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, description, optional));
    definition->add_component(
        std::make_shared<Input::RealComponent>(name, RealComponentData{defaultvalue, optional}));
  }

  /// add a separator followed by a number of real values
  ///
  /// The name on the input line becomes the name used to put the value into the parsed Container.
  template <typename DefinitionType>
  inline void add_named_real_vector(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& description, const int size,
      const double defaultvalue = 0.0, const bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, description, optional));
    definition->add_component(std::make_shared<Input::RealVectorComponent>(
        name, size, RealComponentData{defaultvalue, optional}));
  }

  /// add a separator followed by a number of real values
  ///
  /// The name on the input line becomes the name used to put the value into the parsed Container.
  template <typename DefinitionType>
  inline void add_named_real_vector(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& description, const std::string& sizename,
      const double defaultvalue = 0.0, const bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, description, optional));
    definition->add_component(std::make_shared<Input::RealVectorComponent>(
        name, LengthFromInt(sizename), Input::RealComponentData{defaultvalue, optional}));
  }

  /// add a separator followed by a single string value
  ///
  /// The name on the input line becomes the name used to put the value into the parsed Container
  template <typename DefinitionType>
  inline void add_named_string(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& description, const std::string& defaultvalue,
      const bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, description, optional));
    definition->add_component(
        std::make_shared<Input::StringComponent>(name, defaultvalue, optional));
  }

  /// add a separator followed by a single Boolean value
  ///
  /// The name on the input line becomes the name used to put the bool value into
  /// the parsed Container.
  template <typename DefinitionType>
  inline void add_named_bool(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& description, const bool defaultvalue = false,
      const bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, description, optional));
    definition->add_component(std::make_shared<Input::BoolComponent>(name, defaultvalue, optional));
  }

  /*!
   * @brief Add a separator followed by a post processed component
   *
   * This function adds two components to the @p definition:
   *  1. A SeparatorComponent with a provided @p name, and a @p separator_description .
   *  2. A ProcessedComponent with the same @p name , an @p process_operation function,
   *     and a given @p print_string
   *
   * The @p process_operation function constructs an object of type @p T from the substring that
   * is parsed from the input line definition for the ProcessedComponent. The @p print_string is
   * used to print this ProcessedComponent.
   *
   * The example below serves to clarify the usage of this function. There are several other use
   * cases as well, see e.g. the examples in the documentation of the ProcessedComponent.
   *
   * Assume you specify a file path in your input file and want to store not the actual file path
   * string, but rather the content of the file as an std::vector<int>.
   *
   * You can use this function to add the following two components to the given definition:
   * 1. add a separator "FILE"
   * 2. add a postprocessed component with the name "FILE", a process_operation
   * @code {.cpp}
   * std::function<std::vector<int>(const std::string&)> process_operation =
   *     [](const std::string& file) -> std::vector<int>
   * {
   *    // your logic to read the file and process its content into an integer vector
   * }
   * @endcode
   * and a print_string "integer vector retrieved from the FILE".
   */
  template <typename T, typename DefinitionType>
  inline void add_named_processed_component(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& separator_description,
      const std::function<T(const std::string&)>& process_operation, const bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, separator_description, optional));
    definition->add_component(
        std::make_shared<Input::ProcessedComponent>(name, process_operation, optional));
  }


  /*!
   * @brief Add a separator followed by a selection component
   *
   * This function adds two components to the @p definition:
   *  1. A SeparatorComponent with a provided @p name, and a @p separator_description .
   *  2. A SelectionComponent with the same @p name , @p datfilevalues, @p condvalues and an @p
   * optional tag.
   */
  template <typename DefinitionType>
  inline void add_named_selection_component(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& separator_description,
      const std::string& defaultvalue, const Teuchos::Array<std::string>& datfilevalues,
      const Teuchos::Array<std::string>& condvalues, bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, separator_description, optional));
    definition->add_component(std::make_shared<Input::SelectionComponent>(
        name, defaultvalue, datfilevalues, condvalues, optional));
  }

  /*!
   * @brief Add a separator followed by a selection component
   *
   * This function adds two components to the @p definition:
   *  1. A SeparatorComponent with a provided @p name, and a @p separator_description .
   *  2. A SelectionComponent with the same @p name , @p datfilevalues, @p condvalues and an @p
   * optional tag.
   */
  template <typename DefinitionType>
  inline void add_named_selection_component(const std::shared_ptr<DefinitionType>& definition,
      const std::string& name, const std::string& separator_description,
      const std::string& defaultvalue, const Teuchos::Array<std::string>& datfilevalues,
      const Teuchos::Array<int>& condvalues, bool optional = false)
  {
    definition->add_component(
        std::make_shared<Input::SeparatorComponent>(name, separator_description, optional));
    definition->add_component(std::make_shared<Input::SelectionComponent>(
        name, defaultvalue, datfilevalues, condvalues, optional));
  }

}  // namespace Input

FOUR_C_NAMESPACE_CLOSE

#endif
