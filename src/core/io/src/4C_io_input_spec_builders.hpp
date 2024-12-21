// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_SPEC_BUILDERS_HPP
#define FOUR_C_IO_INPUT_SPEC_BUILDERS_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec.hpp"
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

      void operator()(std::ostream& out, const std::string& val) const { out << " " << val; }

      void operator()(std::ostream& out, bool val) const { out << " " << std::boolalpha << val; }

      template <typename T>
      void operator()(std::ostream& out, const std::vector<T>& val) const
      {
        for (const auto& v : val)
        {
          (*this)(out, v);
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
    constexpr bool has_print = false;

    template <typename T>
    constexpr bool has_print<T, std::void_t<decltype(std::declval<const std::decay_t<T>>().print(
                                    std::declval<std::ostream&>(),
                                    std::declval<const Core::IO::InputParameterContainer&>()))>> =
        true;

    template <typename T, typename AlwaysVoid = void>
    constexpr bool has_set_default_value = false;

    template <typename T>
    constexpr bool has_set_default_value<T,
        std::void_t<decltype(std::declval<const std::decay_t<T>>().set_default_value(
            std::declval<Core::IO::InputParameterContainer&>()))>> = true;


    template <typename T, typename AlwaysVoid = void>
    struct PrettyTypeName
    {
      std::string operator()() { return Utils::try_demangle(typeid(T).name()); }
    };

    template <>
    struct PrettyTypeName<std::string>
    {
      std::string operator()() { return "string"; }
    };

    template <>
    struct PrettyTypeName<std::filesystem::path>
    {
      std::string operator()() { return "char"; }
    };

    template <typename T>
    struct PrettyTypeName<std::vector<T>>
    {
      std::string operator()() { return "vector<" + PrettyTypeName<T>{}() + ">"; }
    };

    template <typename T, typename U>
    struct PrettyTypeName<std::pair<T, U>>
    {
      std::string operator()()
      {
        return "pair<" + PrettyTypeName<T>{}() + ", " + PrettyTypeName<U>{}() + ">";
      }
    };

    template <typename T>
    std::string get_pretty_type_name()
    {
      return PrettyTypeName<T>{}();
    }

    class InputSpecTypeErasedInterface
    {
     public:
      virtual ~InputSpecTypeErasedInterface() = default;
      virtual void parse(ValueParser& parser, InputParameterContainer& container) const = 0;
      virtual void set_default_value(InputParameterContainer& container) const = 0;
      virtual void print(std::ostream& stream, const InputParameterContainer& container) const = 0;
      [[nodiscard]] virtual std::unique_ptr<InputSpecTypeErasedInterface> clone() const = 0;
    };

    template <typename T>
    struct InputSpecTypeErasedImplementation : public InputSpecTypeErasedInterface
    {
      template <typename T2, std::enable_if_t<!std::is_same_v<std::decay_t<T>, InputSpec>, int> = 0>
      explicit InputSpecTypeErasedImplementation(T2&& wrapped) : wrapped(std::forward<T2>(wrapped))
      {
      }

      void parse(ValueParser& parser, InputParameterContainer& container) const override
      {
        wrapped.parse(parser, container);
      }

      void set_default_value(InputParameterContainer& container) const override
      {
        if constexpr (Internal::has_set_default_value<T>)
        {
          wrapped.set_default_value(container);
        }
        else
        {
          FOUR_C_ASSERT(wrapped.data.default_value.has_value(),
              "Implementation error: this function should only be called if the wrapped type has "
              "an optional default value.");

          container.add(wrapped.name, *wrapped.data.default_value);
        }
      }

      void print(std::ostream& stream, const InputParameterContainer& container) const override
      {
        if constexpr (Internal::has_print<T>)
        {
          wrapped.print(stream, container);
        }
        else
        {
          stream << wrapped.name;

          const auto* value = container.get_if<typename T::DataType::StoredType>(wrapped.name);
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
            stream << " <" << Internal::get_pretty_type_name<typename T::DataType::StoredType>()
                   << ">";
          }
        }
      }

      [[nodiscard]] std::unique_ptr<InputSpecTypeErasedInterface> clone() const override
      {
        return std::make_unique<InputSpecTypeErasedImplementation<T>>(wrapped);
      }

      T wrapped;
    };
  }  // namespace Internal

  /**
   * Objects of this class encapsulate knowledge about the input. Users can create objects using
   * the helper functions in the InputSpecBuilders namespace. See the function
   * InputSpecBuilders::entry() for an explanation how to create InputSpecs. This class can be
   * treated as an implementation detail from the user's perspective.
   */
  class InputSpec
  {
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

      /**
       * Whether the spec has a default value.
       */
      bool has_default_value;
    };

    /**
     * A default-constructed spec. A value needs to be assigned to this object before it can
     * be used.
     */
    InputSpec() = default;

    /**
     * Construct a InputSpec. The passed object needs to be of a type that:
     *
     * - has a parse() method that takes an ValueParser and an
     *   InputParameterContainer
     * - (optionally) has a set_default_value() method that takes an InputParameterContainer
     * - (optionally) has a print() method that takes an std::ostream
     */
    template <typename T, std::enable_if_t<!std::is_same_v<std::decay_t<T>, InputSpec>, int> = 0>
    InputSpec(T&& spec, CommonData data);

    InputSpec(const InputSpec& other) : data_(other.data_), spec_(other.spec_->clone()) {}

    InputSpec& operator=(const InputSpec& other)
    {
      spec_ = other.spec_->clone();
      data_ = other.data_;
      return *this;
    }

    InputSpec(InputSpec&&) noexcept = default;
    InputSpec& operator=(InputSpec&&) noexcept = default;

    /**
     * Parse this spec from the input. This is often not the correct function to call, if you
     * want to ensure that the input is fully parsed. Use the fully_parse() function instead.
     */
    void parse(ValueParser& parser, InputParameterContainer& container) const
    {
      spec_->parse(parser, container);
    }

    void set_default_value(InputParameterContainer& container) const
    {
      spec_->set_default_value(container);
    }

    /**
     * Legacy dat-style printing. This format is intended to be human-readable and is not used
     * for further processing.
     */
    void print(std::ostream& stream, const InputParameterContainer& container) const
    {
      if (!required()) stream << "[";
      spec_->print(stream, container);
      if (!required()) stream << "]";
    }

    [[nodiscard]] const std::string& name() const { return data_.name; }

    [[nodiscard]] const std::string& description() const { return data_.description; }

    [[nodiscard]] bool required() const { return data_.required; }

    [[nodiscard]] bool has_default_value() const { return data_.has_default_value; }

   private:
    CommonData data_;

    //! Pointer to type-erased implementation.
    std::unique_ptr<Internal::InputSpecTypeErasedInterface> spec_;
  };

  /**
   * Helper functions to create InputSpec objects.
   */
  namespace InputSpecBuilders
  {
    template <typename StoredTypeIn>
    struct ScalarData
    {
      using StoredType = StoredTypeIn;
      /**
       * An optional description of the value.
       */
      std::string description{};

      /**
       * Whether the value is required or optional.
       */
      std::optional<bool> required{};

      /**
       * The default value of the parameter. If this optional fields is set, the parameter is
       * optional. If it is not set, the parameter is required.
       */
      std::optional<StoredType> default_value{};
    };

    template <typename StoredTypeIn, typename ScalarTypeIn>
    struct VectorData
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
       * An optional description of the value.
       */
      std::string description{};

      /**
       * Whether the value is required or optional.
       */
      std::optional<bool> required{};

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
    struct SelectionData
    {
      using StoredType = StoredTypeIn;

      /**
       * An optional description of the value.
       */
      std::string description{};

      /**
       * Whether the value is required or optional.
       */
      std::optional<bool> required{};

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

    struct GroupData
    {
      /**
       * An optional description of the Group.
       */
      std::string description{};

      bool required{true};

      std::vector<InputSpec> entries;
    };

    namespace Internal
    {
      template <typename T>
      struct DataForHelper
      {
        using type = ScalarData<T>;
      };

      template <typename T>
      struct DataForHelper<std::vector<T>>
      {
        using type = VectorData<std::vector<T>, T>;
      };

      template <typename T>
      using DataFor = typename DataForHelper<T>::type;


      template <typename DataType>
      static constexpr bool is_sized_data = false;

      template <typename D, typename S>
      static constexpr bool is_sized_data<VectorData<D, S>> = true;

      //! Make .required field consistent with .default_value.
      template <typename DataType>
      void sanitize_required_default(DataType& data)
      {
        if (data.default_value.has_value())
        {
          if (data.required.has_value())
          {
            FOUR_C_ASSERT_ALWAYS(!data.required.value(),
                "A parameter cannot be both required and have a default value.");
          }
          else
          {
            data.required = false;
          }
        }
        else
        {
          if (!data.required.has_value()) data.required = true;
        }
        FOUR_C_ASSERT(data.required.has_value(), "Required field must now be set.");
      }

      template <typename DataTypeIn>
      struct BasicSpec
      {
        std::string name;
        using DataType = std::decay_t<DataTypeIn>;
        using StoredType = typename DataType::StoredType;
        DataType data;
        void parse(ValueParser& parser, InputParameterContainer& container) const;
      };


      template <typename DataTypeIn>
      struct UserDefinedSpec
      {
        std::string name;
        using DataType = std::decay_t<DataTypeIn>;
        using StoredType = typename DataType::StoredType;
        DataType data;
        std::function<void(ValueParser&, InputParameterContainer&)> parse;
        std::function<void(std::ostream&, const InputParameterContainer&)> print;
      };

      struct GroupSpec
      {
        std::string name;
        GroupData data;

        void parse(ValueParser& parser, InputParameterContainer& container) const;
        void set_default_value(InputParameterContainer& container) const;
        void print(std::ostream& stream, const InputParameterContainer& container) const;
      };

      struct OneOfSpec
      {
        // A one_of spec is a group with special logic.
        GroupData data;

        void parse(ValueParser& parser, InputParameterContainer& container) const;

        void set_default_value(InputParameterContainer& container) const;

        void print(std::ostream& stream, const InputParameterContainer& container) const;
      };
    }  // namespace Internal

    /**
     * Create a normal entry. All entries are parameterized by a struct which contains the optional
     * `description`, `required` and `default_value` fields. The following examples demonstrate how
     * entries can be created:
     *
     * @code
     * // An entry with name and description. By default, the entry is required.
     * entry<int>("my_int", {.description = "An integer value."});
     *
     * // An entry with a default value. This entry is implicitly optional because a default value
     * // is given.
     * entry<double>("my_double", {.default_value = 3.14});
     * // This is equivalent to:
     * entry<double>("my_double", {.required = false, .default_value = 3.14});
     *
     * // An optional entry. This value does not have a default value.
     * entry<std::string>("my_string", {.required = false});
     *
     * // A vector entry with a fixed size of 3.
     * entry<std::vector<double>>("my_vector", {.size = 3});
     *
     * // A vector entry with a size that is determined by the value of another parameter. The
     * // size is given as a callback.
     * entry<int>("N");
     * entry<std::vector<double>>("my_vector", {.size = from_parameter<int>("N")});
     * @endcode
     *
     * After parsing an InputSpec with fully_parse(), the value of the entry can be retrieved from
     * an InputParameterContainer. The details depend on the `required` and `default_value` fields:
     *
     *   - `required` is used to determine whether the parameter is required in the input file. Note
     *     that by default, if `required` is not set, the parameter is implicitly required. Failing
     *     to read a required parameter will result in an exception. On the other hand, if a
     *     parameter is not required, it is optional and can be left out in the input file. The
     *     container that is filled by the fully_parse() function will not contain the parameter.
     *
     *   - `default_value` is used to provide a value that is used if the parameter is not present
     *     in the input file. If `default_value` is set, the parameter is implicitly optional. If
     *     the parameter is not present in the input file, the default value will be stored in the
     *     container that is filled by the fully_parse() function.
     *
     *   - Setting both `required = true` and a `default_value` is a logical error and will result
     *     in an exception.
     *
     * When you decide how to set the `required` and `default_value` fields, consider the following
     * three cases:
     *
     *   - If you always require a parameter and there is no reasonable default value, do not set
     *     `required` or `default_value`. This will make the parameter required by default. Parsing
     *     will fail if the parameter is not present, but after parsing you can be sure that the
     *     parameter can safely be retrieved with InputParameterContainer::get(). A good example is
     *     the time step size in a time integration scheme: this parameter is always required and
     *     taking an arbitrary default value is not a good idea.
     *
     *   - Is there a reasonable default value for the parameter, which works in most situations? If
     *     yes, set `default_value` to this value (`required` implicitly is `false` then). This
     *     guarantees that you can always read the a value from the container with
     *     InputParameterContainer::get(). A good example is a parameter that activates or
     *     deactivates a feature, e.g., defaulting the EAS element technology to off might be
     *     reasonable.
     *
     *   - If the parameter is not required, but there is no reasonable default value, set
     *     `required = false`. This makes the parameter optional and the container might not contain
     *     the parameter if it is not present in the input file. Use
     *     InputParameterContainer::get_if() or InputParameterContainer::get_or() to safely retrieve
     *     the value from the container. Since this complicates retrieval of input data, this case
     *     should be used with caution. Try to formulate your input requirements in a way that one
     *     of the cases above applies. To still give an example, the present case may be useful for
     *     a damping parameter that, when present, activates damping using the provided value. This
     *     example demonstrates that the parameter has a double role: its presence activates
     *     damping and its value determines the damping strength. An often better way to selectively
     *     activate parameters can be achieved with the group() function, especially if a set of
     *     parameters is always required together.
     *
     * @tparam T The data type of the entry.
     */
    template <typename T, typename DataType = Internal::DataFor<T>>
    [[nodiscard]] InputSpec entry(std::string name, DataType&& data = {});

    /**
     * Create a callback that returns the value of the parameter with the given @p name. Such a
     * callback can, e.g., be used to determine the size of a vector parameter based on the value
     * of another parameter. Of course, the parameter with the given @p name must be read before
     * the parameter that uses this callback. Example:
     *
     * @code
     *   auto input_spec = group({
     *    entry<int>("N"),
     *    entry<std::vector<double>>("data", {.size = read_from_parameter<int>("N")}),
     *    });
     * @endcode
     *
     * @tparam T The type of the parameter of given @p name.
     */
    template <typename T>
    [[nodiscard]] auto from_parameter(const std::string& name);

    /**
     * Create a "tag" for an input line. A tag is essentially a `bool` parameter that is `true` if
     * the tag is present in the input and `false` otherwise.
     *
     * @note Tags are always optional. If not present in the input, the value is `false`.
     *
     * @deprecated Use `entry<bool>` instead, to be more explicit in input files. A "tag" cannot
     * explicitly be set to `false` in the input file, as this requires leaving out the tag.
     */
    [[nodiscard]] InputSpec tag(std::string name, ScalarData<bool> data = {});

    /**
     * A user-defined entry. This is a more flexible version of the `entry` function. It takes
     * a custom function to parse and store the value. The function must take a `ValueParser` and an
     * `InputParameterContainer` as arguments. You can also provide a custom function to print the
     * value. If no print function is provided, a default print function fitting the data type is
     * used. The struct that parameterizes the entry follows the same rules as for the entry()
     * function.
     *
     * @note This function is a last resort. If what you are parsing is so special that it
     *       is not covered by the other functions, you can use this function. Please consider,
     *       enhancing the library with a new function if you think what you are doing is a
     *       missing common use case.
     */
    template <typename T, typename DataType = Internal::DataFor<T>>
    [[nodiscard]] InputSpec user_defined(std::string name, DataType&& data = {},
        const std::function<void(ValueParser&, InputParameterContainer&)>& parse = nullptr,
        const std::function<void(std::ostream&, const Core::IO::InputParameterContainer&)>& print =
            nullptr);

    /**
     * An entry whose value is a a selection from a list of choices. For example:
     *
     * @code
     * selection<int>("my_selection", {.choices = {{"a", 1}, {"b", 2}, {"c", 3}}});
     * @endcode
     *
     * The choices are given as a vector of pairs. The first element of the pair is the string
     * that is expected in the input file. The second element is the value that is stored and may be
     * any type. This function is for convenience, as you do not need to convert parsed string
     * values to another type yourself. A frequent use case is to map strings to enum constants.
     *
     * The remaining parameterization options follow the same rules as for the entry() function.
     */
    template <typename T>
    [[nodiscard]] InputSpec selection(std::string name, SelectionData<T> data = {});

    /**
     * A group of InputSpecs. This groups one or more InputSpecs under a name. A group can be
     * required (default) or optional. If the group is optional and not present in the input, all
     * InputSpecs in the group are implicitly optional. Examples:
     *
     * @code
     * // A required group with three entries.
     * group("GenAlpha",
     *    {
     *    .entries = {
     *      entry<double>("alpha_f"),
     *      entry<double>("alpha_m"),
     *      entry<double>("gamma"),
     *    }
     *    });
     *
     * // An optional group. If the group is not present in the input, none of the entries are
     * // required. This is useful to require a group of parameters together.
     * group("GenAlpha",
     *   {
     *   .required = false,
     *   .entries = {
     *     entry<double>("alpha_f"),
     *     entry<double>("alpha_m"),
     *     entry<double>("gamma"),
     *   }
     *   });
     *
     * //Groups may be nested
     * group("outer",
     *  {
     *  .entries = {
     *    entry<int>("a"),
     *    group("inner",
     *    {
     *    .entries = {
     *      entry<double>("b"),
     *      entry<std::string>("c"),
     *    }
     *  }),
     * });
     *
     * @endcode
     *
     * A group introduces a new scope in the input. This group scope is not only useful for
     * structuring the input file, but also to selectively activate a set of parameters, as
     * demonstrated in the second example. If an optional group is not present in the input, none of
     * the entries are required. If the group is present, all entries are required. This is often
     * exactly what you need: a group activates a feature which requires certain parameters to
     * be present.
     *
     * Whether a group has a default value is determined by the default values of its entries. If
     * all entries have default values, the group implicitly has a default value. In this case, if
     * the group is optional and not present in the input, the default values of the entries are
     * stored under the group name in the container.
     *
     * @note If you want to group multiple InputSpecs without creating a new scope, use the other
     * group() function which does not take a name.
     *
     */
    [[nodiscard]] InputSpec group(std::string name, GroupData data);

    /**
     * A group of multiple InputSpecs. In contrast to the other group() function, this function
     * produces a group without a name and, consequently, no scope is created in the input.
     * Example:
     *
     * @code
     * group({
     *   entry<int>("a"),
     *   entry<double>("b"),
     *   entry<std::string>("c"),
     *   });
     * @endcode
     *
     * This functions gathers multiple InputSpecs on the same level and treats them as a single
     * InputSpec.
     */
    [[nodiscard]] InputSpec group(std::vector<InputSpec> specs);

    /**
     * Exactly one of the given InputSpecs is expected. There is no support for default values; one
     * of the InputSpecs must fit the input.
     *
     * @note The specs are tried in the order they are given. The first InputSpec that fits the
     * input is used. An exception is thrown if other parts of the input would also fit.
     */
    [[nodiscard]] InputSpec one_of(std::vector<InputSpec> specs);
  }  // namespace InputSpecBuilders
}  // namespace Core::IO


// --- template definitions --- //

template <typename DataType>
void Core::IO::InputSpecBuilders::Internal::BasicSpec<DataType>::parse(
    ValueParser& parser, InputParameterContainer& container) const
{
  parser.consume(name);

  if constexpr (InputSpecBuilders::Internal::is_sized_data<DataType>)
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
    container.add(name, parsed);
  }
  else
  {
    container.add(name, parser.read<typename DataType::StoredType>());
  }
}


template <typename T, std::enable_if_t<!std::is_same_v<std::decay_t<T>, Core::IO::InputSpec>, int>>
Core::IO::InputSpec::InputSpec(T&& spec, CommonData data)
    : data_(std::move(data)),
      spec_(std::make_unique<Internal::InputSpecTypeErasedImplementation<std::decay_t<T>>>(
          std::forward<T>(spec)))
{
}


template <typename T>
auto Core::IO::InputSpecBuilders::from_parameter(const std::string& name)
{
  return [name](const InputParameterContainer& container) -> T
  {
    const T* val = container.get_if<T>(name);
    FOUR_C_ASSERT_ALWAYS(val, "Parameter '%s' not found in container.", name.c_str());
    return *val;
  };
}


template <typename T, typename DataType>
Core::IO::InputSpec Core::IO::InputSpecBuilders::entry(std::string name, DataType&& data)
{
  Internal::sanitize_required_default(data);
  return InputSpec(Internal::BasicSpec<DataType>{.name = name, .data = data},
      {
          .name = name,
          .description = data.description,
          .required = data.required.value(),
          .has_default_value = data.default_value.has_value(),
      });
}


template <typename T, typename DataType>
Core::IO::InputSpec Core::IO::InputSpecBuilders::user_defined(std::string name, DataType&& data,
    const std::function<void(ValueParser&, InputParameterContainer&)>& parse,
    const std::function<void(std::ostream&, const Core::IO::InputParameterContainer&)>& print)
{
  Internal::sanitize_required_default(data);
  return InputSpec(
      Internal::UserDefinedSpec<DataType>{
          .name = name,
          .data = std::forward<DataType>(data),
          .parse = parse,
          .print = print ? print : [](std::ostream&, const InputParameterContainer&) {},
      },
      {
          .name = name,
          .description = data.description,
          .required = data.required.value(),
          .has_default_value = data.default_value.has_value(),
      });
}


template <typename T>
Core::IO::InputSpec Core::IO::InputSpecBuilders::selection(
    std::string name, Core::IO::InputSpecBuilders::SelectionData<T> data)
{
  FOUR_C_ASSERT_ALWAYS(!data.choices.empty(), "Selection must have at least one choice.");

  if (data.default_value.has_value())
  {
    auto default_value_it = std::find_if(data.choices.begin(), data.choices.end(),
        [&](const auto& choice) { return choice.second == data.default_value.value(); });

    FOUR_C_ASSERT_ALWAYS(
        default_value_it != data.choices.end(), "Default value of selection not found in choices.");
  }

  return user_defined<T, SelectionData<T>>(
      name, std::move(data),
      [name, data](ValueParser& parser, InputParameterContainer& container)
      {
        parser.consume(name);
        auto value = parser.read<std::string>();
        for (const auto& choice : data.choices)
        {
          if (choice.first == value)
          {
            container.add(name, choice.second);
            return;
          }
        }
        FOUR_C_THROW("Invalid value '%s'", value.c_str());
      },
      [name, data](std::ostream& stream, const Core::IO::InputParameterContainer& container)
      {
        stream << name;
        auto* val = container.get_if<T>(name);
        if (val)
        {
          for (const auto& choice : data.choices)
          {
            if (choice.second == *val)
            {
              stream << " " << choice.first;
              return;
            }
          }
          FOUR_C_ASSERT(false, "Invalid value.");
        }
        else
        {
          stream << " (";
          for (const auto& choice : data.choices)
          {
            stream << choice.first << "|";
          }
          stream << ")";
        }
      });
}

FOUR_C_NAMESPACE_CLOSE

#endif