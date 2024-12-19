// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_LINE_HPP
#define FOUR_C_IO_INPUT_LINE_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_io_value_parser.hpp"

#include <functional>
#include <optional>
#include <ostream>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  namespace Internal
  {
    /**
     * Helper to print values in the format of a dat file.
     */
    struct DatPrinter
    {
      template <typename T>
      void operator()(std::ostream& out, const T& val) const
      {
        out << " " << val;
      }

      void operator()(std::ostream& out, const std::string& val) const
      {
        if (val.empty())
          out << " ''";
        else
          out << " " << val;
      }

      void operator()(std::ostream& out, bool val) const { out << " " << std::boolalpha << val; }

      template <typename T>
      void operator()(std::ostream& out, const std::vector<T>& val) const
      {
        if (val.empty())
          out << " ...";
        else
        {
          for (const auto& v : val)
          {
            (*this)(out, v);
          }
        }
      }

      template <typename T, typename U>
      void operator()(std::ostream& out, const std::pair<T, U>& val) const
      {
        (*this)(out, val.first);
        (*this)(out, val.second);
      }
    };

    template <typename T, typename AlwaysVoid = void>
    constexpr bool has_print_value = false;

    template <typename T>
    constexpr bool
        has_print_value<T, std::void_t<decltype(std::declval<const std::decay_t<T>>().print_value(
                               std::declval<std::ostream&>(),
                               std::declval<const Core::IO::InputParameterContainer&>()))>> = true;

    template <typename T, typename AlwaysVoid = void>
    constexpr bool has_set_default_value = false;

    template <typename T>
    constexpr bool has_set_default_value<T,
        std::void_t<decltype(std::declval<const std::decay_t<T>>().set_default_value(
            std::declval<Core::IO::InputParameterContainer&>()))>> = true;



  }  // namespace Internal

  /**
   * Type-erased component of an input line. Multiple objects of this class make up an InputLine.
   * Users should create components using the helper functions in the InputLineBuilders namespace.
   * This class can be treated as an implementation detail.
   */
  struct Component
  {
   private:
    struct EraseComponentInterface
    {
      virtual ~EraseComponentInterface() = default;
      virtual void parse_and_store_value(
          ValueParser& parser, InputParameterContainer& container) const = 0;
      virtual void set_default_value(InputParameterContainer& container) const = 0;
      virtual void print_value(
          std::ostream& stream, const InputParameterContainer& container) const = 0;
      [[nodiscard]] virtual std::unique_ptr<EraseComponentInterface> clone() const = 0;
    };

    template <typename T>
    struct EraseComponentImplementation : public EraseComponentInterface
    {
      template <typename T2, std::enable_if_t<!std::is_same_v<std::decay_t<T>, Component>, int> = 0>
      explicit EraseComponentImplementation(T2&& component) : component(std::forward<T2>(component))
      {
      }

      void parse_and_store_value(
          ValueParser& parser, InputParameterContainer& container) const override
      {
        component.parse_and_store_value(parser, container);
      }

      void set_default_value(InputParameterContainer& container) const override
      {
        if constexpr (Internal::has_set_default_value<T>)
        {
          component.set_default_value(container);
        }
        else
        {
          FOUR_C_ASSERT(component.data.default_value.has_value(),
              "Implementation error: this function should only be called if the component has an "
              "optional default value.");

          container.add(component.data.name, *component.data.default_value);
        }
      }

      void print_value(
          std::ostream& stream, const InputParameterContainer& container) const override
      {
        if constexpr (Internal::has_print_value<T>)
        {
          component.print_value(stream, container);
        }
        else
        {
          const auto* value =
              container.get_if<typename T::DataType::StoredType>(component.data.name);
          if (value)
          {
            // Print the value if it is present in the container.
            Internal::DatPrinter{}(stream, *value);
          }
          else
          {
            // Print the type of the value if it is not present in the container.
            // This functionality is only required to print exemplary lines into a default
            // dat file. When no default value, exists there is not much that we can let the
            // user know except the type we expect.
            stream << " <" << Core::Utils::get_type_name<typename T::DataType::StoredType>() << ">";
          }
        }
      }

      [[nodiscard]] std::unique_ptr<EraseComponentInterface> clone() const override
      {
        return std::make_unique<EraseComponentImplementation<T>>(component);
      }

      T component;
    };

   public:
    /**
     * Store a reduced version of the common data entries.
     */
    struct CommonData
    {
      /**
       * The name or key under which the value is found in the input. This name is also used to
       * store the value in the container.
       */
      std::string name;

      /**
       * An optional description of the value.
       */
      std::string description;

      /**
       * Whether the value is required or optional.
       */
      bool required;
    };

    /**
     * A default-constructed component. A value needs to be assigned to this object before it can
     * be used.
     */
    Component() = default;

    /**
     * Construct a Component. The passed object needs to be of a type that:
     *
     * - has a parse_and_store_value() method that takes an ValueParser and an
     *   InputParameterContainer
     * - (optionally) has a set_default_value() method that takes an InputParameterContainer
     * - (optionally) has a print_value() method that takes an std::ostream
     */
    template <typename T, std::enable_if_t<!std::is_same_v<std::decay_t<T>, Component>, int> = 0>
    Component(T&& component, CommonData data);

    Component(const Component& other) : data_(other.data_), component_(other.component_->clone()) {}

    Component& operator=(const Component& other)
    {
      component_ = other.component_->clone();
      data_ = other.data_;
      return *this;
    }

    Component(Component&&) noexcept = default;
    Component& operator=(Component&&) noexcept = default;

    void parse(ValueParser& parser, InputParameterContainer& container) const
    {
      parser.consume(data_.name);
      component_->parse_and_store_value(parser, container);
    }

    void set_default_value(InputParameterContainer& container) const
    {
      component_->set_default_value(container);
    }

    void print(std::ostream& stream, const InputParameterContainer& container) const
    {
      if (!required()) stream << "[";
      stream << data_.name;
      component_->print_value(stream, container);
      if (!required()) stream << "]";
      stream << " ";
    }

    [[nodiscard]] const std::string& name() const { return data_.name; }

    [[nodiscard]] const std::string& description() const { return data_.description; }

    [[nodiscard]] bool required() const { return data_.required; }


   private:
    CommonData data_;

    //! Pointer to type-erased implementation.
    std::unique_ptr<EraseComponentInterface> component_;
  };

  /**
   * Helper functions to create components for an InputLine.
   */
  namespace InputLineBuilders
  {
    template <typename StoredTypeIn>
    struct ScalarComponentData
    {
      using StoredType = StoredTypeIn;
      /**
       * The name or key under which the value is found in the input. This name is also used to
       * store the value in the container.
       */
      std::string name;

      /**
       * An optional description of the value.
       */
      std::string description{};

      /**
       * The default value of the parameter. If this optional fields is set, the parameter is
       * optional. If it is not set, the parameter is required.
       */
      std::optional<StoredType> default_value{};
    };

    template <typename StoredTypeIn, typename ScalarTypeIn>
    struct VectorComponentData
    {
      using StoredType = StoredTypeIn;
      using ScalarType = ScalarTypeIn;

      /**
       * A function that determines the size of the vector. This function is called with the
       * already parsed content of the input line, which may be used to query the size as the
       * value of another parameter.
       */
      using SizeCallback = std::function<int(const InputParameterContainer&)>;

      /**
       * The name or key under which the value is found in the input. This name is also used to
       * store the value in the container.
       */
      std::string name;

      /**
       * An optional description of the value.
       */
      std::string description{};

      /**
       * The default value of the parameter. If this optional fields is set, the parameter is
       * optional. If it is not set, the parameter is required.
       */
      std::optional<StoredType> default_value{};

      /**
       * The size of the data. This can either be a fixed size or a callback that determines the
       * size based on the value of another parameter.
       *
       * @note We use `int` instead of `size_t` since this is what most users write. Template
       *      argument deduction will work a lot better with `int` than with `size_t`.
       */
      std::variant<int, SizeCallback> size;
    };

    template <typename StoredTypeIn>
    struct SelectionComponentData
    {
      using StoredType = StoredTypeIn;

      /**
       * The name or key under which the value is found in the input. This name is also used to
       * store the value in the container.
       */
      std::string name;

      /**
       * An optional description of the value.
       */
      std::string description{};

      /**
       * The default value of the parameter. If this optional fields is set, the parameter is
       * optional. If it is not set, the parameter is required.
       */
      std::optional<StoredType> default_value{};

      /**
       * The choices that are available for the selection.
       */
      std::vector<std::pair<std::string, StoredType>> choices;
    };

    struct GroupComponentData
    {
      /**
       * The name of the Group. This name will be found as a group in the InputParameterContainer.
       */
      std::string name;

      /**
       * An optional description of the Group.
       */
      std::string description{};

      bool required{true};

      std::vector<Component> entries;
    };

    namespace Internal
    {
      template <typename T>
      struct DataForHelper
      {
        using type = ScalarComponentData<T>;
      };

      template <typename T>
      struct DataForHelper<std::vector<T>>
      {
        using type = VectorComponentData<std::vector<T>, T>;
      };

      template <typename T>
      using DataFor = typename DataForHelper<T>::type;


      template <typename DataType>
      static constexpr bool is_sized_data = false;

      template <typename D, typename S>
      static constexpr bool is_sized_data<VectorComponentData<D, S>> = true;

      /**
       * Instead of requiring users to set fields of the structs that parameterize a component in
       * a specific way, we can handles special cases in here. This also allows to validate
       * certain conditions, e.g., that the name of a component is not empty.
       */
      template <typename DataType>
      DataType sanitize_user_input_data(DataType&& data)
      {
        DataType sanitized_data(std::forward<DataType>(data));

        FOUR_C_ASSERT_ALWAYS(!sanitized_data.name.empty(), "The name must not be empty.");

        return sanitized_data;
      }

      template <typename DataTypeIn>
      struct BasicComponent
      {
        using DataType = std::decay_t<DataTypeIn>;
        using StoredType = typename DataType::StoredType;
        DataType data;
        void parse_and_store_value(ValueParser& parser, InputParameterContainer& container) const;
      };


      template <typename DataTypeIn>
      struct UserDefinedComponent
      {
        using DataType = std::decay_t<DataTypeIn>;
        using StoredType = typename DataType::StoredType;
        DataType data;
        std::function<void(ValueParser&, InputParameterContainer&)> parse_and_store_value;
        std::function<void(std::ostream&, const InputParameterContainer&)> print_value;
      };

      struct GroupComponent
      {
        GroupComponentData data;

        void parse_and_store_value(ValueParser& parser, InputParameterContainer& container) const;
        void set_default_value(InputParameterContainer& container) const;
        void print_value(std::ostream& stream, const InputParameterContainer& container) const;
      };
    }  // namespace Internal

    /**
     * Create a callback that returns the value of the parameter with the given @p name. Such a
     * callback can, e.g., be used to determine the size of a vector parameter based on the value
     * of another parameter. Of course, the parameter with the given @p name must be read before
     * the parameter that uses this callback. Example:
     *
     * @code
     *   InputLine line{
     *    entry<int>({.name = "N"}),
     *    entry<std::vector<double>>({.name = "data",
     *    .size = read_from_parameter<int>("N")}),
     *    };
     * @endcode
     *
     * @tparam T The type of the parameter of given @p name.
     */
    template <typename T>
    auto from_parameter(const std::string& name);

    /**
     * Create a normal entry. All entries are parameterized by a struct which contains
     * `name`, `description`,  and `default_value` fields. The following examples demonstrate how
     * entries can be created:
     *
     * @code
     * // An entry with name and description. By default, the entry is required.
     * entry<int>({.name = "my_int", .description = "An integer value."});
     *
     * // An entry with a default value. This entry is implicitly optional because a default value
     * // is given.
     * entry<std::string>({.name = "my_int", .description = "A string value.", .default_value =
     *   "abc"});
     *
     * // A vector entry with a fixed size of 3.
     * entry<std::vector<double>>({.name = "my_vector", .description = "A vector of doubles.",
     *   .size = 3});
     *
     * // A vector entry with a size that is determined by the value of another parameter. The
     * // size is given as a callback.
     * entry<int>({.name = "N"});
     * entry<std::vector<double>>({.name = "my_vector", .description = "A vector of doubles.",
     *   .size = from_parameter<int>("N")});
     * @endcode
     *
     * @tparam T The data type of the entry.
     */
    template <typename T, typename DataType = Internal::DataFor<T>>
    Component entry(DataType&& data);

    /**
     * Create a "tag" for an input line. A tag is essentially a `bool` parameter that is `true` if
     * the tag is present in the input and `false` otherwise.
     *
     * @note Tags are always optional. If not present in the input, the value is `false`.
     *
     * @deprecated Use `entry<bool>` instead, to be more explicit in input files. A "tag" cannot
     * explicitly be set to `false` in the input file, as this requires leaving out the tag.
     */
    Component tag(ScalarComponentData<bool> data);

    /**
     * A user-defined entry. This is a more flexible version of the `entry` function. The user can
     * provide a custom function to parse and store the value. The function must take a
     * `ValueParser` and an `InputParameterContainer` as arguments. The user can also provide a
     * custom function to print the value. If no print function is provided, a default print
     * function fitting the data type is used. The struct that parameterizes the entry follows the
     * same rules as for the entry() function.
     */
    template <typename T, typename DataType = Internal::DataFor<T>>
    Component user_defined(DataType&& data,
        const std::function<void(ValueParser&, InputParameterContainer&)>& parse_and_store_value,
        const std::function<void(std::ostream&, const Core::IO::InputParameterContainer&)>&
            print_value = nullptr);

    /**
     * An entry that is a a selection from a list of choices. For example:
     *
     * @code
     * selection<int>({.name = "my_int", .choices = {{"one", 1}, {"two", 2}}});
     * @endcode
     *
     * The choices are given as a vector of pairs. The first element of the pair is the string
     * that is expected in the input file. The second element is the value that is stored. The
     * remaining parameterization options follow the same rules as for the entry() function.
     */
    template <typename T>
    Component selection(SelectionComponentData<T> data);

    /**
     * A group of entries. This groups one or more entries under a name. A group can be required
     * (default) or optional. If the group is optional, all entries in the group are optional.
     * Examples:
     *
     * @code
     * // A required group with three entries.
     * group({.name = "my_group", .entries = {
     *   entry<int>({.name = "a"}),
     *   entry<double>({.name = "b"}),
     *   entry<std::string>({.name = "c"}),
     *   }});
     *
     * // An optional group. If the group is not present in the input, none of the entries are
     * // required.
     * group({.name = "my_group", .required = false, .entries = {
     *   entry<int>({.name = "a"}),
     *   entry<double>({.name = "b", .required = false}),
     *   }});
     *
     * @endcode
     *
     */
    Component group(GroupComponentData data);
  }  // namespace InputLineBuilders


  /**
   * Additional options for parsing the InputLine.
   */
  struct InputLineOptions
  {
    /**
     * When `true`, failing to parse a line will throw an exception. When `false`, errors
     * are suppressed. Check the return value of the parse function to see if an error occurred.
     */
    bool throw_on_error{true};

    /**
     * When `true`, store default values of optional entries that are not encountered.
     * When `false`, do not store default values.
     */
    bool store_default_values{true};
  };

  /**
   * A line in the input file.
   */
  class InputLine
  {
   public:
    explicit InputLine(std::initializer_list<Component> components);

    explicit InputLine(std::vector<Component> components);

    /**
     * Parse the @p input and store the values in the @p container. Additional options can be
     * specified in @p options. Returns `true` if parsing was successful, `false` otherwise.
     */
    bool parse(std::string_view input, InputParameterContainer& container,
        InputLineOptions options = {}, ValueParserContext context = {}) const;

    /**
     * Print the components of the line to the @p stream. A value must exist for each component in
     * @p container. All components are printed in the order they were given to the constructor.
     */
    void print(std::ostream& stream, const InputParameterContainer& container) const;

    /**
     * Print the components of the line to the @p stream.
     */
    void print_default(std::ostream& stream) const;



   private:
    //! Internal storage of the components that make up the line.
    std::vector<Component> components_;
  };

}  // namespace Core::IO


// --- template and inline functions --- //


template <typename DataType>
void Core::IO::InputLineBuilders::Internal::BasicComponent<DataType>::parse_and_store_value(
    ValueParser& parser, InputParameterContainer& container) const
{
  if constexpr (InputLineBuilders::Internal::is_sized_data<DataType>)
  {
    struct SizeVisitor
    {
      std::size_t operator()(std::size_t size) const { return size; }
      std::size_t operator()(const typename DataType::SizeCallback& callback) const
      {
        return callback(container);
      }
      InputParameterContainer& container;
    };

    std::size_t size = std::visit(SizeVisitor{container}, data.size);
    FOUR_C_ASSERT(size > 0, "Size must be greater than 0.");
    auto parsed = parser.read<typename DataType::StoredType>(size);
    container.add(data.name, parsed);
  }
  else
  {
    container.add(data.name, parser.read<typename DataType::StoredType>());
  }
}

template <typename T>
auto Core::IO::InputLineBuilders::from_parameter(const std::string& name)
{
  return [name](const InputParameterContainer& container) -> T
  {
    const T* val = container.get_if<T>(name);
    FOUR_C_ASSERT_ALWAYS(val, "Parameter '%s' not found in container.", name.c_str());
    return *val;
  };
}

template <typename T, typename DataType>
Core::IO::Component Core::IO::InputLineBuilders::entry(DataType&& data)
{
  auto sanitized_data = Internal::sanitize_user_input_data(std::forward<DataType>(data));
  return Component(Internal::BasicComponent<DataType>{.data = sanitized_data},
      {
          .name = sanitized_data.name,
          .description = sanitized_data.description,
          .required = !sanitized_data.default_value.has_value(),
      });
}

template <typename T, typename DataType>
Core::IO::Component Core::IO::InputLineBuilders::user_defined(DataType&& data,
    const std::function<void(ValueParser&, InputParameterContainer&)>& parse_and_store_value,
    const std::function<void(std::ostream&, const Core::IO::InputParameterContainer&)>& print_value)
{
  auto sanitized_data = Internal::sanitize_user_input_data(std::forward<DataType>(data));

  return Component(
      Internal::UserDefinedComponent<DataType>{
          .data = sanitized_data,
          .parse_and_store_value = parse_and_store_value,
          .print_value =
              print_value ? print_value : [](std::ostream&, const InputParameterContainer&) {},
      },
      {
          .name = sanitized_data.name,
          .description = sanitized_data.description,
          .required = !sanitized_data.default_value.has_value(),
      });
}

template <typename T>
Core::IO::Component Core::IO::InputLineBuilders::selection(
    Core::IO::InputLineBuilders::SelectionComponentData<T> data)
{
  return user_defined<T, SelectionComponentData<T>>(
      std::move(data),
      [data](ValueParser& parser, InputParameterContainer& container)
      {
        auto value = parser.read<std::string>();
        for (const auto& choice : data.choices)
        {
          if (choice.first == value)
          {
            container.add(data.name, choice.second);
            return;
          }
        }
        FOUR_C_THROW("Invalid value '%s'", value.c_str());
      },
      [data](std::ostream& stream, const Core::IO::InputParameterContainer& container)
      {
        stream << " [";
        for (const auto& choice : data.choices)
        {
          stream << choice.first << " ";
        }
        stream << "]";
      });
}

template <typename T, std::enable_if_t<!std::is_same_v<std::decay_t<T>, Core::IO::Component>, int>>
Core::IO::Component::Component(T&& component, CommonData data)
    : data_(std::move(data)),
      component_(std::make_unique<EraseComponentImplementation<std::decay_t<T>>>(
          std::forward<T>(component)))
{
  FOUR_C_ASSERT_ALWAYS(!data_.name.empty(), "The name of a component must not be empty.");
}


FOUR_C_NAMESPACE_CLOSE

#endif
