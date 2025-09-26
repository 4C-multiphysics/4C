// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_SPEC_BUILDERS_HPP
#define FOUR_C_IO_INPUT_SPEC_BUILDERS_HPP

#include "4C_config.hpp"

#include "4C_io_input_field.hpp"
#include "4C_io_input_parameter_container.templates.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_io_input_spec_storage.hpp"
#include "4C_io_input_spec_validators.hpp"
#include "4C_io_input_types.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_io_yaml.hpp"
#include "4C_utils_compile_time_string.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_string.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <ostream>
#include <tuple>
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
        out << "<unprintable>";
      }

      template <SupportedType T>
      void operator()(std::ostream& out, const T& val) const
      {
        using EnumTools::operator<<;
        out << val;
      }

      void operator()(std::ostream& out, bool val) const { out << " " << (val ? "true" : "false"); }

      template <typename T>
      void operator()(std::ostream& out, const std::optional<T>& val) const
      {
        if (val.has_value())
          (*this)(out, *val);
        else
          out << "none";
      }

      template <typename T>
      void operator()(std::ostream& out, const std::vector<T>& val) const
      {
        for (const auto& v : val)
        {
          (*this)(out, v);
          out << " ";
        }
      }

      template <typename T, std::size_t n>
      void operator()(std::ostream& out, const std::array<T, n>& val) const
      {
        for (const auto& v : val)
        {
          (*this)(out, v);
          out << " ";
        }
      }

      template <typename T>
      void operator()(std::ostream& out, const std::map<std::string, T>& val) const
      {
        for (const auto& [key, v] : val)
        {
          out << key << " ";
          (*this)(out, v);
          out << " ";
        }
      }

      template <typename... Ts>
      void operator()(std::ostream& out, const std::tuple<Ts...>& vals) const
      {
        std::apply([&](const auto&... vals) { (((*this)(out, vals), out << " "), ...); }, vals);
      }

      template <typename T1, typename T2>
      void operator()(std::ostream& out, const std::pair<T1, T2>& val) const
      {
        (*this)(out, val.first);
        out << " ";
        (*this)(out, val.second);
        out << " ";
      }
    };

    template <typename T>
    concept CustomDatPrintable = requires(const T& t, std::ostream& stream, std::size_t indent) {
      { t.print(stream, indent) } -> std::same_as<void>;
    };

    template <typename T, typename AlwaysVoid = void>
    constexpr bool has_set_default_value = false;

    template <typename T>
    constexpr bool has_set_default_value<T,
        std::void_t<decltype(std::declval<const std::decay_t<T>>().set_default_value(
            std::declval<Core::IO::InputParameterContainer&>()))>> = true;


    template <typename T>
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
      std::string operator()() { return "path"; }
    };

    template <typename Enum>
      requires(std::is_enum_v<Enum>)
    struct PrettyTypeName<Enum>
    {
      std::string operator()() { return std::string{EnumTools::enum_type_name<Enum>()}; }
    };

    template <typename T>
    struct PrettyTypeName<std::vector<T>>
    {
      std::string operator()() { return "vector<" + PrettyTypeName<T>{}() + ">"; }
    };

    template <typename T, typename U>
    struct PrettyTypeName<std::map<T, U>>
    {
      std::string operator()()
      {
        return "map<" + PrettyTypeName<T>{}() + ", " + PrettyTypeName<U>{}() + ">";
      }
    };

    template <typename... Ts>
    struct PrettyTypeName<std::tuple<Ts...>>
    {
      std::string operator()()
      {
        std::string result = "tuple<";

        auto append_types = [&]<std::size_t... index>(std::index_sequence<index...>)
        { ((result += PrettyTypeName<Ts>{}() + (index + 1 < sizeof...(Ts) ? ", " : "")), ...); };

        append_types(std::index_sequence_for<Ts...>{});

        return result += ">";
      }
    };

    template <typename T1, typename T2>
    struct PrettyTypeName<std::pair<T1, T2>>
    {
      std::string operator()()
      {
        return "pair<" + PrettyTypeName<T1>{}() + ", " + PrettyTypeName<T2>{}() + ">";
      }
    };

    template <typename T>
    struct PrettyTypeName<std::optional<T>>
    {
      std::string operator()() { return "std::optional<" + PrettyTypeName<T>{}() + ">"; }
    };

    template <typename T>
    std::string get_pretty_type_name()
    {
      return PrettyTypeName<T>{}();
    }

    template <typename T>
    struct YamlTypeEmitter
    {
      void operator()(ryml::NodeRef node) = delete;
    };

    template <YamlSupportedType T>
      requires(!std::is_enum_v<T>)
    struct YamlTypeEmitter<T>
    {
      void operator()(ryml::NodeRef node, size_t*)
      {
        FOUR_C_ASSERT(node.is_map(), "Expected a map node.");
        node["type"] << get_pretty_type_name<T>();
      }
    };

    template <typename Enum>
      requires(std::is_enum_v<Enum>)
    struct YamlTypeEmitter<Enum>
    {
      void operator()(ryml::NodeRef node, size_t*)
      {
        FOUR_C_ASSERT(node.is_map(), "Expected a map node.");
        node["type"] = "enum";
      }
    };

    template <typename T>
    struct YamlTypeEmitter<std::vector<T>>
    {
      void operator()(ryml::NodeRef node, size_t* size)
      {
        FOUR_C_ASSERT(node.is_map(), "Expected a map node.");
        node["type"] = "vector";
        if (*size > 0)
        {
          node["size"] << *size;
        }
        node["value_type"] |= ryml::MAP;
        YamlTypeEmitter<T>{}(node["value_type"], size + 1);
      }
    };

    template <typename T, std::size_t size_n>
    struct YamlTypeEmitter<std::array<T, size_n>>
    {
      void operator()(ryml::NodeRef node, size_t* size)
      {
        FOUR_C_ASSERT(node.is_map(), "Expected a map node.");
        // In the schema we do not distinguish between arrays and vectors, so we use the same type.
        node["type"] = "vector";
        node["size"] << size_n;
        node["value_type"] |= ryml::MAP;
        YamlTypeEmitter<T>{}(node["value_type"], size + 1);
      }
    };

    template <typename T>
    struct YamlTypeEmitter<std::map<std::string, T>>
    {
      void operator()(ryml::NodeRef node, size_t* size)
      {
        FOUR_C_ASSERT(node.is_map(), "Expected a map node.");
        node["type"] = "map";
        if (*size > 0)
        {
          node["size"] << *size;
        }
        node["value_type"] |= ryml::MAP;
        YamlTypeEmitter<T>{}(node["value_type"], size + 1);
      }
    };

    template <typename... Ts>
    struct YamlTypeEmitter<std::tuple<Ts...>>
    {
      template <std::size_t tuple_index = 0>
      void emit_elements(ryml::NodeRef node, size_t* size)
      {
        if constexpr (tuple_index < sizeof...(Ts))
        {
          using Te = std::tuple_element_t<tuple_index, std::tuple<Ts...>>;
          ryml::NodeRef child = node["value_types"].append_child();
          child |= ryml::MAP;
          YamlTypeEmitter<Te>{}(child, size + 1);

          emit_elements<tuple_index + 1>(node, size);  // recurse to next element
        }
      }

      void operator()(ryml::NodeRef node, size_t* size)
      {
        FOUR_C_ASSERT(node.is_map(), "Expected a map node.");
        node["type"] = "tuple";
        node["size"] << sizeof...(Ts);
        node["value_types"] |= ryml::SEQ;
        emit_elements(node, size);
      }
    };

    template <typename T1, typename T2>
    struct YamlTypeEmitter<std::pair<T1, T2>>
    {
      void operator()(ryml::NodeRef node, size_t* size)
      {
        YamlTypeEmitter<std::tuple<T1, T2>>{}(node, size);
      }
    };

    template <typename T>
    struct YamlTypeEmitter<std::optional<T>>
    {
      void operator()(ryml::NodeRef node, size_t* size)
      {
        FOUR_C_ASSERT(node.is_map(), "Expected a map node.");
        // Pull up the std::optional aspect. The fact that this type wraps another type is specific
        // to C++ and not relevant to other tools. Simply knowing that a type can be empty is
        // enough for them.
        emit_value_as_yaml(YamlNodeRef{node["noneable"], ""}, true);
        YamlTypeEmitter<T>{}(node, size);
      }
    };

    template <typename T>
      requires(dynamic_rank<T>() == 0)
    void emit_type_as_yaml(ryml::NodeRef node)
    {
      YamlTypeEmitter<T>{}(node, nullptr);
    }

    template <typename T>
    void emit_type_as_yaml(ryml::NodeRef node, std::array<std::size_t, dynamic_rank<T>()> size)
    {
      YamlTypeEmitter<T>{}(node, size.data());
    }


    class MatchTree;

    /**
     * Entries in the MatchTree.
     */
    struct MatchEntry
    {
      MatchTree* tree;
      const InputSpec* spec;
      std::vector<MatchEntry*> children;
      std::string additional_info;

      /**
       * A MatchEntry can only match a single node. For parameters and groups, this is the node
       * that contains the key. For logical specs (all_of, one_of), this is the parent node that
       * contains the content of the spec as children. This is always set to a valid node, even
       * if the spec is not matched. In that case, the node ID refers to the node that the match
       * was attempted on.
       */
      ryml::id_type matched_node{ryml::npos};

      enum class State : std::uint8_t
      {
        /**
         * Not a match at all.
         */
        unmatched,
        /**
         * A perfect match.
         */
        matched,
        /**
         * At least something matched. Depending on the type of spec, this can e.g. mean that
         * the key of a group matched.
         */
        partial,
        /**
         * The match was successfully defaulted.
         */
        defaulted,
        /**
         * The spec was not required and was not matched. This is OK.
         */
        not_required,
      };

      State state{State::unmatched};

      /**
       * Append a child for the @p in_spec to the current entry and return a reference to it. This
       * child is passed on to the match function of @p in_spec.
       */
      MatchEntry& append_child(const InputSpec* in_spec);

      /**
       * Reset the state of this entry. This includes dropping all children from the MatchTree.
       * The state of the entry and MatchTree is the same as if append_child was just called.
       */
      void reset();
    };

    /**
     * A tree that tracks how well a spec matches a yaml tree.
     */
    class MatchTree
    {
     public:
      MatchTree(const InputSpec& root, ConstYamlNodeRef node);

      MatchEntry& root() { return entries_.front(); }

      ConstYamlNodeRef node() const { return node_; }

      MatchEntry& append_child(const InputSpec* spec);

      void dump(std::ostream& stream) const;

      /**
       * Throw an exception that contains the input and match tree in a user-friendly format, if
       * the match was not successful.
       */
      void assert_match() const;

      /**
       * A helper function to remove every entry added after @p entry. This function is especially
       * useful to keep the number of stored entries low when matching a list().
       */
      void erase_everything_after(const MatchEntry& entry);

     private:
      std::vector<MatchEntry> entries_;
      ConstYamlNodeRef node_;
    };


    /**
     * Distinguish between different types of specs in the implementation.
     */
    enum class InputSpecType : std::uint8_t
    {
      parameter,
      group,
      list,
      selection,
      all_of,
      one_of,
      deprecated_selection,
    };


    class InputSpecImpl
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

        /**
         * The total number of specs that make up this spec. This includes the spec itself, meaning,
         * that the minimum value is 1. This value can be used to reserve memory ahead of time.
         */
        std::size_t n_specs;

        /**
         * The type of the spec.
         */
        InputSpecType type;

        /**
         * The type_info for the type that this spec will write to. Providing this here allows
         * consistency checks when combining specs.
         */
        const std::type_info* stores_to;
      };


      virtual ~InputSpecImpl() = default;

      /**
       * @param data The common data of the spec.
       */
      InputSpecImpl(CommonData data);

      virtual void parse(ValueParser& parser, InputParameterContainer& container) const = 0;

      /**
       * Returns true if the node matches the spec and stores the value in the container. The passed
       * @p node is the parent node which might contain data matching the spec. Every spec needs
       * to check if it can find required data in this node. If yes, the InputSpec should report
       * itself as matched in the @p match_entry. Note that the @p match_entry already refers to the
       * spec that is being matched. When matching more specs internally, the spec needs to append
       * children to the match_entry.
       */
      virtual bool match(ConstYamlNodeRef node, InputSpecBuilders::Storage& container,
          MatchEntry& match_entry) const = 0;

      virtual void set_default_value(InputSpecBuilders::Storage& container) const = 0;

      //! Emit metadata. This function always emits into a map, i.e., the implementation must
      //! insert keys and values into the yaml emitter.
      virtual void emit_metadata(YamlNodeRef node) const = 0;

      [[nodiscard]] virtual bool emit(YamlNodeRef node, const InputParameterContainer&,
          const InputSpecEmitOptions& options) const = 0;

      [[nodiscard]] virtual std::string pretty_type_name() const = 0;

      [[nodiscard]] virtual std::unique_ptr<InputSpecImpl> clone() const = 0;

      void print(std::ostream& stream, std::size_t indent) const { do_print(stream, indent); }

      [[nodiscard]] const std::string& name() const { return data.name; }

      [[nodiscard]] const std::string& description() const { return data.description; }

      [[nodiscard]] std::string description_one_line() const;

      [[nodiscard]] bool required() const { return data.required; }

      [[nodiscard]] bool has_default_value() const { return data.has_default_value; }


      CommonData data;

     protected:
      InputSpecImpl(const InputSpecImpl&) = default;
      InputSpecImpl& operator=(const InputSpecImpl&) = default;
      InputSpecImpl(InputSpecImpl&&) noexcept = default;
      InputSpecImpl& operator=(InputSpecImpl&&) noexcept = default;

     private:
      virtual void do_print(std::ostream& stream, std::size_t indent) const = 0;
    };

    template <typename T>
    concept StoresType = requires(T t) { typename T::StoredType; };

    template <typename T>
    struct InputSpecTypeErasedImplementation : public InputSpecImpl
    {
      template <typename T2>
      explicit InputSpecTypeErasedImplementation(T2&& wrapped, CommonData data)
          : InputSpecImpl(std::move(data)), wrapped(std::forward<T2>(wrapped))
      {
      }

      void parse(ValueParser& parser, InputParameterContainer& container) const override
      {
        wrapped.parse(parser, container);
      }

      bool match(ConstYamlNodeRef node, InputSpecBuilders::Storage& container,
          MatchEntry& match_entry) const override
      {
        return wrapped.match(node, container, match_entry);
      }

      void set_default_value(InputSpecBuilders::Storage& container) const override
      {
        wrapped.set_default_value(container);
      }

      void emit_metadata(YamlNodeRef node) const override { wrapped.emit_metadata(node); }

      [[nodiscard]] bool emit(YamlNodeRef node, const InputParameterContainer& container,
          const InputSpecEmitOptions& options) const override
      {
        return wrapped.emit(node, container, options);
      }

      void do_print(std::ostream& stream, std::size_t indent) const override
      {
        if constexpr (CustomDatPrintable<T>)
        {
          wrapped.print(stream, indent);
        }
        else
        {
          stream << "// " << std::string(indent, ' ') << name();

          // pretty printed type of the parameter
          stream << " <" << pretty_type_name() << ">";

          if (has_default_value())
          {
            stream << " (default: ";
            Internal::DatPrinter{}(stream, std::get<1>(wrapped.data.default_value));
            stream << ")";
          }

          if (!description().empty())
          {
            stream << " " << std::quoted(description_one_line());
          }
          stream << "\n";
        }
      }

      [[nodiscard]] std::string pretty_type_name() const override
      {
        if constexpr (StoresType<T>)
        {
          return Internal::get_pretty_type_name<typename T::StoredType>();
        }
        else
        {
          FOUR_C_ASSERT(false,
              "Implementation error: this function should only be called for "
              "types that store a type.");
          return "unknown";
        }
      }

      [[nodiscard]] std::unique_ptr<InputSpecImpl> clone() const override
      {
        return std::make_unique<InputSpecTypeErasedImplementation<T>>(wrapped, data);
      }

      T wrapped;
    };

    template <typename T>
    InputSpec make_spec(T&& wrapped, InputSpecImpl::CommonData data)
    {
      return InputSpec(std::make_unique<InputSpecTypeErasedImplementation<std::decay_t<T>>>(
          std::forward<T>(wrapped), std::move(data)));
    }

    //! Validate if possible. Returns false only if validation was attempted and failed.
    template <typename T>
    [[nodiscard]] bool validate_helper(
        const T& val, const std::optional<InputSpecBuilders::Validators::Validator<T>>& validator)
    {
      return !validator.has_value() || (*validator)(val);
    }

    //! Validate if possible. Returns false only if validation was attempted and failed.
    template <typename T>
    [[nodiscard]] bool validate_helper(const std::optional<T>& val,
        const std::optional<InputSpecBuilders::Validators::Validator<T>>& validator)
    {
      return !val.has_value() || validate_helper(*val, validator);
    }

    enum class InputFieldType : std::uint8_t
    {
      constant,
      from_file,
      field_reference,
      from_mesh,
    };

    template <typename T>
    auto make_input_field_data_transform(
        std::string name, InputSpecBuilders::StoreFunction<InputField<T>> store)
    {
      return InputSpecBuilders::StoreFunction<InputSpecBuilders::DefaultStorage>(
          [=](InputSpecBuilders::Storage& out, InputSpecBuilders::DefaultStorage&& in)
          {
            auto selector = in.get<InputFieldType>("_selector");
            switch (selector)
            {
              case InputFieldType::from_file:
              {
                std::unordered_map<int, T> map_data;
                std::filesystem::path file_path = in.get<std::filesystem::path>("from_file");
                IO::read_value_from_yaml(file_path, name, map_data);
                auto field = IO::InputField<T>(std::move(map_data));
                return store(out, std::move(field));
                break;
              }
              case InputFieldType::constant:
              {
                T data = in.get<T>("constant");
                auto field = IO::InputField<T>(data);
                return store(out, std::move(field));
                break;
              }
              case InputFieldType::from_mesh:
              {
                const auto& array_name = in.get<std::string>("from_mesh");
                MeshDataReference ref =
                    global_mesh_data_input_field_registry().register_field_reference(array_name);
                auto field = IO::InputField<T>(ref);
                return store(out, std::move(field));
                break;
              }
              case InputFieldType::field_reference:
              {
                std::string field_name = in.get<std::string>("field_reference");
                InputFieldReference ref =
                    global_input_field_registry().register_field_reference(field_name);
                auto field = IO::InputField<T>(ref);
                return store(out, std::move(field));
                break;
              }
              default:
              {
                FOUR_C_THROW("Unsupported input field type: {}",
                    in.get<Core::IO::Internal::InputFieldType>("type"));
              }
            }
          },
          store.stores_to());
    }
  }  // namespace Internal

  /**
   * Helper functions to create InputSpec objects. When you want to create an InputSpec, you
   * can "import" the functions from this namespace by using the following line:
   *
   * @code
   * using namespace Core::IO::InputSpecBuilders;
   * @endcode
   *
   * This allows you to create InputSpec objects in a concise notation like this:
   *
   * @code
   * auto input = group("params",
   *   {
   *   parameter<int>("a", {.description = "An integer value"}),
   *   parameter<std::vector<double>>("b", {.description = "A vector of doubles", .size = 4}),
   *   }
   * );
   * @endcode
   */
  namespace InputSpecBuilders
  {
    /**
     * This constant signifies that a size is dynamic and will be determined at runtime. Pass this
     * value to the size parameter of a vector-valued parameter() or the list() function.
     *
     * @note Dynamic sizes do not work for the legacy dat file format and will throw a runtime
     * error.
     */
    constexpr int dynamic_size = 0;

    /**
     * Callback to determine the size of a vector or map at runtime from info that has already been
     * parsed.
     */
    using SizeCallback = std::function<int(const InputParameterContainer&)>;

    /**
     * Size of a vector or map. This can be a fixed size, #dynamic_size, or a callback that
     * determines the size at runtime.
     */
    using Size = std::variant<int, SizeCallback>;

    /**
     * Callback function that may be attached to parameter().
     */
    using ParameterCallback = std::function<void(InputParameterContainer&)>;

    /**
     * Callback function to transform data from a selection.
     */
    using SelectionCallback = std::function<void(const DefaultStorage&, Storage&)>;

    /**
     * A tag type to indicate that a field in the parameter data cannot take a value.
     */
    struct RejectField
    {
      RejectField() = default;

      template <typename T>
      RejectField(const T&)
      {
        static_assert(false,
            "You try to assign a value to a field that does not take a value for this parameter "
            "type.");
      }
    };

    /**
     * The type used for the default value of a parameter. If the parameter is optional, the user
     * cannot specify a default value, so this type becomes RejectField. Otherwise, the default
     * value can be either not set, resulting in the std::monostate, or set to a value of the
     * parameter type.
     */
    template <typename T>
    using DefaultType =
        std::conditional_t<OptionalType<T>, RejectField, std::variant<std::monostate, T>>;

    /**
     * Function type for a function that describes enum values.
     */
    template <typename T>
    using EnumValueDescriptionFunction = std::function<std::string(T)>;

    /**
     * The type used for the enum value description. If T is not an enum type, this is
     * equal to RejectField.
     */
    template <typename T>
    using EnumValueDescription =
        std::conditional_t<std::is_enum_v<T>, EnumValueDescriptionFunction<T>, RejectField>;

    //! Additional parameters for a parameter().
    template <typename T>
    struct ParameterDataIn
    {
      using StoredType = T;
      /**
       * An optional description of the value.
       */
      std::string description{};

      /**
       * An optional description of the enum values. This is only possible for enum types.
       */
      EnumValueDescription<T> enum_value_description{};

      /**
       * The default value of the parameter. If this field is set, the parameter does not need to be
       * entered in the input. If the parameter is not entered, this default value is used.
       */
      DefaultType<T> default_value{};

      /**
       * An optional callback that is called after the value has been parsed. This can be used to
       * set additional values in the container.
       */
      ParameterCallback on_parse_callback{nullptr};

      /**
       * An optional validator that is called after the value has been parsed and is guaranteed to
       * be of the correct type. This validator can then check the value for more specific
       * conditions such as assuring that a value is within a certain range of values. See the
       * Validators namespace for some available validators.
       */
      std::optional<Validators::Validator<T>> validator{std::nullopt};

      /**
       * An optional function to store a parsed value. See the in_struct() function for more
       * details.
       */
      StoreFunction<T> store{nullptr};
    };

    template <typename T>
      requires(dynamic_rank<T>() == 1)
    struct ParameterDataIn<T>
    {
      using StoredType = T;

      std::string description{};

      DefaultType<T> default_value{};

      ParameterCallback on_parse_callback{nullptr};

      std::optional<Validators::Validator<T>> validator{std::nullopt};

      StoreFunction<T> store{nullptr};

      /**
       * The size of the vector. This can be a fixed size, #dynamic_size, or a callback that
       * determines the size at runtime.
       */
      Size size{dynamic_size};
    };

    template <typename T>
      requires(dynamic_rank<T>() > 1)
    struct ParameterDataIn<T>
    {
      using StoredType = T;

      std::string description{};

      DefaultType<T> default_value{};

      ParameterCallback on_parse_callback{nullptr};

      std::optional<Validators::Validator<T>> validator{std::nullopt};

      StoreFunction<T> store{nullptr};

      std::array<Size, dynamic_rank<T>()> size;
    };

    template <typename Selector, typename StorageType>
    struct SelectionData
    {
      /**
       * An optional description of the selection.
       */
      std::string description{};

      /**
       * An optional function to store the selection group.
       */
      StoreFunction<StorageType> store{nullptr};

      /**
       * An optional function to store the value of the selector.
       */
      StoreFunction<Selector> store_selector{nullptr};

      /**
       * An optional function to transform the data of the selection and store it differently.
       * This is a powerful expert function to implement advanced specs. For example, it is
       * used to implement input_field().
       */
      StoreFunction<DefaultStorage> transform_data{nullptr};
    };

    template <typename T>
    struct GroupData
    {
      /**
       * An optional description of the Group.
       */
      std::string description{};

      /**
       * Whether the Group is required or optional.
       */
      bool required{true};

      /**
       * An optional function to store a parsed value. See the in_struct() function for more
       * details.
       */
      StoreFunction<T> store{nullptr};
    };

    //! Additional parameters for a list().
    struct ListData
    {
      /**
       * An optional description of the List.
       */
      std::string description{};

      /**
       * Whether the List is required or optional.
       */
      bool required{true};

      /**
       * The size of the List.
       */
      int size{dynamic_size};
    };

    /**
     * Data for an InputField. Note that T is the type of the data stored inside the InputField.
     */
    template <typename T>
    struct InputFieldData
    {
      /**
       * An optional description of the value.
       */
      std::string description{};

      /**
       * An optional function to store the parsed InputField. By default, the result is stored in
       * an InputParameterContainer. See the in_struct() function for more details on how to
       * store the InputField in a struct.
       */
      StoreFunction<InputField<T>> store{nullptr};
    };

    template <typename Number, Utils::CompileTimeString... variables>
    struct SymbolicExpressionData;
  }  // namespace InputSpecBuilders

  namespace Internal
  {
    template <typename T>
    struct ParameterData
    {
      using StoredType = T;

      std::string description{};

      InputSpecBuilders::EnumValueDescriptionFunction<T> enum_value_description{};

      std::variant<std::monostate, StoredType> default_value{};

      InputSpecBuilders::ParameterCallback on_parse_callback{nullptr};

      std::optional<InputSpecBuilders::Validators::Validator<T>> validator{std::nullopt};

      InputSpecBuilders::StoreFunction<T> store;

      std::array<InputSpecBuilders::Size, dynamic_rank<T>()> size{};
    };

    template <typename T>
    struct DeprecatedSelectionData
    {
      using StoredType = T;

      std::string description{};

      std::variant<std::monostate, StoredType> default_value{};

      InputSpecBuilders::ParameterCallback on_parse_callback{nullptr};

      InputSpecBuilders::StoreFunction<T> store;
    };

    template <SupportedType T>
    struct ParameterSpec
    {
      std::string name;
      using StoredType = T;
      ParameterData<T> data;
      void parse(ValueParser& parser, InputParameterContainer& container) const;
      bool match(ConstYamlNodeRef node, InputSpecBuilders::Storage& container,
          IO::Internal::MatchEntry& match_entry) const;
      void emit_metadata(YamlNodeRef node) const;
      bool emit(YamlNodeRef node, const InputParameterContainer& container,
          const InputSpecEmitOptions& options) const;
      void set_default_value(InputSpecBuilders::Storage& container) const;
      [[nodiscard]] bool has_correct_size(
          const T& val, const InputSpecBuilders::Storage& container) const;
    };

    /**
     * Note that a DeprecatedSelectionSpec can store any type since we never need to read or write
     * values of this type.
     */
    template <typename T>
    struct DeprecatedSelectionSpec
    {
      std::string name;
      using StoredType = T;
      //! The type that is used in the input file.
      using InputType =
          std::conditional_t<OptionalType<T>, std::optional<std::string>, std::string>;
      using ChoiceMap = std::map<InputType, StoredType>;
      DeprecatedSelectionData<T> data;
      ChoiceMap choices;
      //! The string representation of the choices.
      std::string choices_string;
      void parse(ValueParser& parser, InputParameterContainer& container) const;
      bool match(ConstYamlNodeRef node, InputSpecBuilders::Storage& container,
          IO::Internal::MatchEntry& match_entry) const;
      void print(std::ostream& stream, std::size_t indent) const;
      void emit_metadata(YamlNodeRef node) const;
      bool emit(YamlNodeRef node, const InputParameterContainer& container,
          const InputSpecEmitOptions& options) const;
      void set_default_value(InputSpecBuilders::Storage& container) const;
    };

    template <typename T>
      requires(std::is_enum_v<T>)
    struct SelectionSpec
    {
      std::string group_name;
      std::map<T, InputSpec> choices;

      struct
      {
        std::string description;
        InputSpecBuilders::StoreFunction<InputSpecBuilders::DefaultStorage> transform_data;
        InputSpecBuilders::StoreFunction<T> store_selector;
      } data;

      std::function<void(InputSpecBuilders::Storage& my_storage)> init_my_storage{};
      InputSpecBuilders::StoreFunction<InputSpecBuilders::Storage> move_my_storage{};

      void parse(ValueParser& parser, InputParameterContainer& container) const;
      bool match(ConstYamlNodeRef node, InputSpecBuilders::Storage& container,
          IO::Internal::MatchEntry& match_entry) const;
      void print(std::ostream& stream, std::size_t indent) const;
      void emit_metadata(YamlNodeRef node) const;
      bool emit(YamlNodeRef node, const InputParameterContainer& container,
          const InputSpecEmitOptions& options) const;
      void set_default_value(InputSpecBuilders::Storage& container) const;
    };

    struct AllOfSpec
    {
      std::vector<InputSpec> specs;

      void parse(ValueParser& parser, InputParameterContainer& container) const;
      bool match(ConstYamlNodeRef node, InputSpecBuilders::Storage& container,
          IO::Internal::MatchEntry& match_entry) const;
      void set_default_value(InputSpecBuilders::Storage& container) const;
      void print(std::ostream& stream, std::size_t indent) const;
      void emit_metadata(YamlNodeRef node) const;
      bool emit(YamlNodeRef node, const InputParameterContainer& container,
          const InputSpecEmitOptions& options) const;
    };

    struct GroupSpec
    {
      std::string name;
      struct
      {
        std::string description;
        bool required;
        bool defaultable;
      } data;
      //! A logical spec with the content of the group.
      InputSpec spec;

      std::function<void(InputSpecBuilders::Storage& my_storage)> init_my_storage{};
      InputSpecBuilders::StoreFunction<InputSpecBuilders::Storage> move_my_storage{};

      void parse(ValueParser& parser, InputParameterContainer& container) const;
      bool match(ConstYamlNodeRef node, InputSpecBuilders::Storage& container,
          IO::Internal::MatchEntry& match_entry) const;
      void set_default_value(InputSpecBuilders::Storage& container) const;
      void print(std::ostream& stream, std::size_t indent) const;
      void emit_metadata(YamlNodeRef node) const;
      bool emit(YamlNodeRef node, const InputParameterContainer& container,
          const InputSpecEmitOptions& options) const;
    };


    struct OneOfSpec
    {
      std::vector<InputSpec> specs;

      //! This callback may be used to perform additional actions after parsing one of the specs.
      //! The index of the parsed spec as given inside #specs is passed as an argument.
      std::function<void(InputParameterContainer& container, std::size_t index)> on_parse_callback;

      void parse(ValueParser& parser, InputParameterContainer& container) const;

      bool match(ConstYamlNodeRef node, InputSpecBuilders::Storage& container,
          IO::Internal::MatchEntry& match_entry) const;

      void set_default_value(InputSpecBuilders::Storage& container) const;

      void print(std::ostream& stream, std::size_t indent) const;

      void emit_metadata(YamlNodeRef node) const;
      bool emit(YamlNodeRef node, const InputParameterContainer& container,
          const InputSpecEmitOptions& options) const;
    };

    struct ListSpec
    {
      //! The name of the list.
      std::string name;
      //! The spec that fits the list elements.
      InputSpec spec;

      InputSpecBuilders::ListData data;

      void parse(ValueParser& parser, InputParameterContainer& container) const;
      bool match(ConstYamlNodeRef node, InputSpecBuilders::Storage& container,
          IO::Internal::MatchEntry& match_entry) const;
      void set_default_value(InputSpecBuilders::Storage& container) const;
      void print(std::ostream& stream, std::size_t indent) const;
      void emit_metadata(YamlNodeRef node) const;
      bool emit(YamlNodeRef node, const InputParameterContainer& container,
          const InputSpecEmitOptions& options) const;
    };


    //! Helper to create selection() specs.
    //! Note that the type can be anything since we never read or write values of this type.
    template <typename T>
    [[nodiscard]] InputSpec selection_internal(std::string name,
        std::map<std::string, RemoveOptional<T>> choices,
        InputSpecBuilders::ParameterDataIn<T> data = {});


    struct SizeChecker
    {
      constexpr bool operator()(const auto& val, std::size_t* size_info) const
      {
        static_assert(dynamic_rank<decltype(val)>() == 0, "Missing overload.");
        return true;
      }

      template <typename U>
      constexpr bool operator()(const std::vector<U>& v, std::size_t* size_info) const
      {
        return ((*size_info == InputSpecBuilders::dynamic_size) || (v.size() == *size_info)) &&
               std::ranges::all_of(
                   v, [&](const auto& val) { return this->operator()(val, size_info + 1); });
      }

      template <typename T, std::size_t n>
      constexpr bool operator()(const std::array<T, n>& arr, std::size_t* size_info) const
      {
        return std::ranges::all_of(
            arr, [&](const auto& val) { return this->operator()(val, size_info); });
      }

      template <typename U>
      constexpr bool operator()(const std::map<std::string, U>& m, std::size_t* size_info) const
      {
        return ((*size_info == InputSpecBuilders::dynamic_size) || (m.size() == *size_info)) &&
               std::ranges::all_of(
                   m, [&](const auto& val) { return this->operator()(val.second, size_info + 1); });
      }

      template <typename... Ts>
      constexpr bool operator()(const std::tuple<Ts...>& t, std::size_t* size_info) const
      {
        return std::invoke(
            [&]<std::size_t... is>(std::index_sequence<is...>)
            {
              bool result = true;
              std::size_t offset = 0;

              ((result = result && this->operator()(std::get<is>(t), size_info + offset),
                   offset += dynamic_rank<std::tuple_element_t<is, std::tuple<Ts...>>>()),
                  ...);

              return result;
            },
            std::index_sequence_for<Ts...>{});
      }

      template <typename T1, typename T2>
      constexpr bool operator()(const std::pair<T1, T2>& p, std::size_t* size_info) const
      {
        return this->operator()(p.first, size_info) &&
               this->operator()(p.second, size_info + dynamic_rank<T1>());
      }

      template <typename T>
      constexpr bool operator()(const std::optional<T>& opt, std::size_t* size_info) const
      {
        if (opt.has_value())
        {
          return this->operator()(opt.value(), size_info);
        }
        else
        {
          return true;
        }
      }
    };

    /**
     * If the given @p spec is not already an all_of spec, this function wraps it into an
     * all_of spec. This is useful for the match function of InputSpec to ensure a consistent
     * starting point.
     */
    Core::IO::InputSpec wrap_with_all_of(Core::IO::InputSpec spec);
  }  // namespace Internal

  namespace InputSpecBuilders
  {
    /**
     * Create a normal parameter with given @p name. All parameters are parameterized by a struct
     * which contains the optional `description` and `default_value` fields. The following examples
     * demonstrate how parameters can be created:
     *
     * @code
     * // A parameter with name and description. By default, the parameter is required in the input.
     * parameter<std::string>("my_string", {.description = "A string value."});
     *
     * // A parameter with a default value. This parameter is implicitly optional because a default
     * // value is given.
     * parameter<double>("my_double", {.default_value = 3.14});
     *
     * // An alternative way to create an optional double parameter is achieved with a
     * // std::optional type. This value is optional and has an empty value in the input file. You
     * // cannot set a default value in this case.
     * parameter<std::optional<my_double>>("my_double");
     *
     * // A vector parameter with a fixed size of 3.
     * parameter<std::vector<double>>("my_vector", {.size = 3});
     *
     * // A vector may also contain std::optional values.
     * parameter<std::vector<std::optional<double>>>("my_vector", {.size = 3});
     *
     * // A vector parameter with a size that is determined by the value of another parameter. The
     * // size is given as a callback.
     * parameter<int>("N");
     * parameter<std::vector<double>>("my_vector", {.size = from_parameter<int>("N")});
     *
     * // A vector parameter which performs an additional action after parsing.
     * parameter<std::filesystem::path>("data_file", {.description = "A path to a file.",
     *   .on_parse_callback = [](InputParameterContainer& container) {
     *     // Perform an action with the parsed path.
     *     std::filesystem::path path = container.get<std::filesystem::path>("my_path");
     *     // e.g. read the file content and also store it in the container.
     *     // auto data_table = ...
     *     container.add("data_table", data_table);
     *   }});
     * @endcode
     *
     * There are two ways how the parameters can be stored after parsing. By default, the parsed
     * value is stored in an InputParameterContainer. Another powerful way is to use the
     * in_struct() function to attach a struct member to the parameter. In this case, the
     * parsed value will be stored in the struct member. Note that you will need to provide this
     * struct by using the group() function. See the documentation there for a usage example.
     *
     * After parsing an InputSpec, the value of the parameter can be retrieved from storage, either
     * an InputParameterContainer or a struct. If `default_value` is set, the parameter is
     * implicitly optional. In this case, if the parameter is not present in the input file, the
     * default value will be stored.
     *
     * When you decide how to set the `default_value` field or whether to make a parameter optional,
     * consider the following cases:
     *
     *   - If you always require a parameter and there is no reasonable default value, do not set
     *     a `default_value`. This will make the parameter required. Parsing will fail if the
     *     parameter is not present, but after parsing you can be sure that the parameter can safely
     *     be retrieved with InputParameterContainer::get() or from the struct. A good example is
     *     the time step size in a time integration scheme: this parameter is always required and
     *     taking an arbitrary default value is not a good idea.
     *
     *   - Is there a reasonable default value for the parameter, which works in most situations? If
     *     yes, set `default_value` to this value. This guarantees that you can always read the
     *     value from the container with InputParameterContainer::get() or from the struct. A good
     *     example is a parameter that activates or deactivates a feature, e.g., defaulting the EAS
     *     element technology to off might be reasonable.
     *
     *   - If the parameter is not required and there is no reasonable default value, wrap the type
     *     in `std::optional`. As an example, this may be useful for a damping parameter that, when
     *     set, activates damping using the provided value. This example demonstrates that the
     *     parameter has a double role: its presence activates damping and its value determines the
     *     damping strength. An often better way to selectively activate parameters can be achieved
     *     with the group() function, especially if a set of parameters is always required together.
     *
     * @tparam T The data type of the parameter. Must be a SupportedType.
     *
     * @relatedalso InputSpec
     */
    template <SupportedType T>
    [[nodiscard]] InputSpec parameter(std::string name, ParameterDataIn<T>&& data = {});

    /**
     * Create a callback that returns the value of the parameter with the given @p name. Such a
     * callback can, e.g., be used to determine the size of a vector parameter based on the value
     * of another parameter. Of course, the parameter with the given @p name must be read before
     * the parameter that uses this callback. Example:
     *
     * @code
     *   auto input_spec = group({
     *    parameter<int>("N"),
     *    parameter<std::vector<double>>("data", {.size = read_from_parameter<int>("N")}),
     *    });
     * @endcode
     *
     * @tparam T The type of the parameter of given @p name.
     *
     * @relatedalso InputSpec
     */
    template <typename T>
    [[nodiscard]] auto from_parameter(const std::string& name);

    /**
     * A parameter whose value is a selection from a list of choices.
     *
     * The choices are given as a map from string to stored type T. This function is for
     * convenience, as you do not need to convert parsed string values to another type yourself. A
     * frequent use case is to map strings to enum constants.
     *
     * The remaining parameterization options follow the same rules as for the parameter() function.
     *
     * @note If you want to store the choices as strings and not map them to another type, use the
     * other deprecated_selection() function.
     *
     * @deprecated If you want to select from a set of enum constants use the parameter() function
     * with the enum type.
     *
     * @relatedalso InputSpec
     */
    template <typename T>
      requires(std::is_enum_v<T>)
    [[nodiscard]] InputSpec deprecated_selection(std::string name,
        std::map<std::string, RemoveOptional<T>> choices, ParameterDataIn<T> data = {});


    /**
     * Like the other deprecated_selection() function, but the choices are stored as strings and not
     * mapped to another type.
     *
     * @note Although this function only works with strings, you still need to provide a type for
     * the first template parameter for consistency with the other functions.
     *
     * @deprecated If you want to select from a set of strings, create an enum with the name
     * of the strings and use the parameter() function with the enum type.
     *
     * @relatedalso InputSpec
     */
    template <std::same_as<std::string> T>
    [[nodiscard]] InputSpec deprecated_selection(
        std::string name, std::vector<std::string> choices, ParameterDataIn<T> data = {});

    /**
     * A group of InputSpecs. This groups one or more InputSpecs under a name. A group can be
     * required (default) or optional. If the group is optional and not present in the input, all
     * InputSpecs in the group are implicitly optional. Examples:
     *
     * @code
     * // A required group.
     * group("group",
     *    {
     *      parameter<double>("a"),
     *      parameter<int>("b"),
     *    });
     *
     * // An optional group. If the group is not present in the input, none of the parameters are
     * // required. This is useful to require a group of parameters together.
     * group("GenAlpha",
     *   {
     *      parameter<double>("alpha_f"),
     *      parameter<double>("alpha_m"),
     *      parameter<double>("gamma"),
     *   },
     *   {.required = false}
     *   );
     *
     * //Groups may be nested
     * group("outer",
     *  {
     *    parameter<int>("a"),
     *    group("inner",
     *    {
     *      parameter<double>("b"),
     *      parameter<std::string>("c"),
     *    }
     *    ),
     *  });
     *
     * @endcode
     *
     * A group introduces a new scope in the input. This group scope is not only useful for
     * structuring the input file, but also to selectively activate a set of parameters, as
     * demonstrated in the second example. If an optional group is not present in the input, none of
     * its children are required. If the group is present, all children are required. This is often
     * exactly what you need: a group activates a feature which requires certain parameters to
     * be present.
     *
     * An alternative way to use groups is to store the parsed values in a struct. In this case,
     * you can use the `store` option in the GroupData struct to specify how the parsed values
     * should be stored.
     *
     * Example:
     *
     * @code
     * struct Data
     * {
     *   double alpha_f;
     *   double alpha_m;
     * };
     *
     * struct TimeIntegration
     * {
     *   double time_step;
     *   Data data;
     * };
     *
     * auto spec = group<TimeIntegration>("time_integration",
     *     {
     *         parameter<double>("time_step", {.store = in_struct(&TimeIntegration::time_step)}),
     *         group<Data>("data",
     *             {
     *                 parameter<double>("alpha_f", {.store = in_struct(&Data::alpha_f)}),
     *                 parameter<double>("alpha_m", {.store = in_struct(&Data::alpha_m)}),
     *             },
     *             {.store = in_struct(&TimeIntegration::data)}),
     *     });
     * @endcode
     *
     * The innermost parameters are stored in a `Data` struct, which is itself stored in the
     * `data` member of the `TimeIntegration` struct. The outer group() does not specify
     * any special store behavior, so it stores the parsed value in an InputParameterContainer.
     *
     * @note Providing inconsistent types for the store functions and group() will cause
     * a runtime error.
     *
     * Whether a group can have a default value is determined by the default values of its children.
     * If all of its children have default values, the group can have a default value. In this case,
     * setting `.required = false` guarantees that the default values of the children are
     * stored under the group name in the container, even if the group is not present in the input.
     * This behavior is analogous to the behavior of the parameter() function. While parameters with
     * default values are often a good idea (if the default value is meaningful), groups with
     * default values are less common. If you follow the advice to use groups to activate features,
     * you will usually have required parameters in the group and, consequently, the group will not
     * be defaultable. Put differently, if you have a group with many default values, you are likely
     * not using the InputSpecs to their full potential. Consider splitting the group into multiple
     * smaller groups or use selection() to select between different groups.
     *
     * @note If you want to group multiple InputSpecs without creating a new named scope, use the
     * all_of() function.
     *
     * @relatedalso InputSpec
     */
    template <typename StorageType = DefaultStorage>
    [[nodiscard]] InputSpec group(
        std::string name, std::vector<InputSpec> specs, GroupData<StorageType> data = {});

    /**
     * This function shares all the properties of group(), but it allows you to create a group that
     * stores the parsed values in a struct rather than in a container. You want to use this
     * function in combination with a parameter() that stores its parsed value into a struct.
     *
     */

    /**
     * This function is used to select one of multiple InputSpecs based on the value of an
     * enum parameter. For every possible value of the selector enum, a different InputSpec
     * is expected. For example:
     *
     * @code
     * enum class TimeIntegration
     * {
     *   OST,
     *   GenAlpha,
     * };
     *
     * auto spec = selection<TimeIntegration>(
     *     "time_integration", {
     *         group("OST",{
     *             parameter<double>("theta"),
     *         }),
     *         group("GenAlpha",{
     *             parameter<double>("alpha_f"),
     *             parameter<double>("alpha_m"),
     *         }),
     *     });
     * @endcode
     *
     * will match the following input:
     *
     * @code
     * time_integration:
     *   OST:
     *     theta: 0.5
     * @endcode
     *
     * or the following input:
     *
     * @code
     * time_integration:
     *   GenAlpha:
     *     alpha_f: 1
     *     alpha_m: 0.5
     * @endcode
     *
     * Every name of a choice must match one of the enum values of the @p Selector type (in the
     * example above: `OST` or `GenAlpha`).
     * During matching, a parameter named "_selector" with the enum constant corresponding to the
     * active choice is automatically added inside the group @p name by this function.
     *
     * Since this function is a clever combination of the functionalities of group() and
     * parameter(), it also provides support for storing the parsed values in a struct. Enhancing
     * the example above, you can use the following code:
     *
     * @code
     * struct Ost
     * {
     *   double theta;
     * };
     *
     * struct GenAlpha
     * {
     *   double alpha_f;
     *   double alpha_m;
     * };
     *
     * struct TimeIntegrationParameters
     * {
     *   TimeIntegration scheme;
     *   std::variant<Ost, GenAlpha> parameters;
     * };
     *
     * auto ost = group<Ost>("ost",
     *     {
     *         parameter<double>("theta", {.store = in_struct(&Ost::theta)}),
     *     },
     *     {.store = as_variant<Ost>(&TimeIntegrationParameters::parameters)});
     *
     * auto gen_alpha = group<GenAlpha>("gen_alpha",
     *     {
     *         parameter<double>("alpha_f", {.store = in_struct(&GenAlpha::alpha_f)}),
     *         parameter<double>("alpha_m", {.store = in_struct(&GenAlpha::alpha_m)}),
     *     },
     *     {.store = as_variant<GenAlpha>(&TimeIntegrationParameters::parameters)});
     *
     * auto spec = selection<TimeIntegration, TimeIntegrationParameters>(
     *     "time_integration", {ost, gen_alpha}
     *     {.store_selector = in_struct(&TimeIntegrationParameters::scheme)});
     * @endcode
     *
     * Note that you need to enhance every choice of the selection to be a group(), so that
     * there is a clear type for the parsed value. Every group() can store to the variant
     * that encodes the selection in a type.
     */
    template <typename Selector, typename StorageType = DefaultStorage>
      requires(std::is_enum_v<Selector>)
    [[nodiscard]] InputSpec selection(std::string name, std::vector<InputSpec> choices,
        SelectionData<Selector, StorageType> data = {});

    /**
     * Defines an input field for spatially dependent parameters.
     *
     * The input field allows a parameter of a certain type to be specified as either a constant
     * or an element-wise value. This is useful for cases where a parameter, such as material
     * stiffness, needs to be defined differently for different parts of a model.
     *
     * Example usage:
     * @code
     * auto spec = input_field<double>("stiffness");
     * @endcode
     *
     * This will match the following input:
     * @code
     *  stiffness:
     *    constant: 100.0
     * @endcode
     *  or
     * @code
     *  stiffness:
     *    from_file: /some/file.json
     * @endcode
     *
     * The json input field json file could look like this:
     * @code
     * {
     *  "stiffness": {
     *    "1": 2.0,
     *    "2": 3.5,
     *    "3": 4.0,
     *    "4": 5.5
     *    }
     * }
     * @endcode
     *
     * After matching, the data can be retrieved as an InputField<T>, either from an
     * InputParameterContainer or from a struct. For the latter, you need use the
     * in_struct() function to store the InputField in an appropriate struct member.
     */
    template <typename T>
    [[nodiscard]] InputSpec input_field(std::string name, InputFieldData<T> data = {});

    /**
     * @brief Define an input parameter that is a SymbolicExpression.
     *
     * The template parameters are the same as for the SymbolicExpression class. Refer to the
     * class documentation for more details.
     *
     * Example:
     *
     * @code
     * symbolic_expression<double, "x", "y">("expr");
     * @endcode
     *
     * will match the following input:
     *
     * @code
     * expr: "x + y"
     * @endcode
     *
     * Note that the SymbolicExpression is parsed right away, and you will need to retrieve the
     * parameter as a SymbolicExpression<double, "x", "y"> from the InputParameterContainer or store
     * it in an appropriate struct member using the in_struct() function.
     */
    template <typename Number, Utils::CompileTimeString... variables>
    [[nodiscard]] InputSpec symbolic_expression(
        std::string name, SymbolicExpressionData<Number, variables...> data = {});

    /**
     * All of the given InputSpecs are expected, e.g.,
     *
     * @code
     * all_of({
     *   parameter<int>("a"),
     *   parameter<double>("b"),
     *   parameter<std::string>("c"),
     *   });
     * @endcode
     *
     * will require all three parameters to be present in the input.
     *
     * The main application of this function is to gather multiple InputSpecs on the same level and
     * treat them as a single InputSpec. Nesting multiple all_of() specs is possible but does not
     * have any effect on the structure of the input. The following two examples are equivalent:
     *
     * @code
     * // version 1
     * group("outer",
     * {
     *   all_of({
     *     parameter<int>("a"),
     *     all_of({
     *       parameter<double>("b"),
     *     }),
     *   }),
     *   parameter<std::string>("c"),
     * });
     *
     * // version 2
     * group("outer",
     * {
     *   parameter<int>("a"),
     *   parameter<double>("b"),
     *   parameter<std::string>("c"),
     * });
     * @endcode
     *
     * In practice, all_of() is essentially a group() without a name and without an associated scope
     * in the input file. An all_of() InputSpec will be required if at least one of its contained
     * InputSpecs is required. If none of the contained InputSpecs are required, the all_of()
     * InputSpec is not required.
     *
     * @relatedalso InputSpec
     */
    [[nodiscard]] InputSpec all_of(std::vector<InputSpec> specs);

    /**
     * Exactly one of the given InputSpecs is expected. For example, to require exactly one of two
     * groups:
     *
     * @code
     * one_of({
     *  group("OneStepTheta",
     *  {
     *    parameter<double>("theta"),
     *  }),
     *  group("GenAlpha",
     *  {
     *    parameter<double>("alpha_f"),
     *    parameter<double>("alpha_m"),
     *    parameter<double>("gamma"),
     *    parameter<bool>("do_logging", {.default_value = false}),
     *  }),
     *  });
     * @endcode
     *
     * Here, one_of() requires either the "OneStepTheta" group or the "GenAlpha" group to be
     * present in the input. If both or none of them are present, an exception is thrown. Note that
     * all InputSpecs handed to one_of() need to be required, i.e., they may not have a default
     * value. While this could silently be changed internally, you will instead encounter an error
     * if any InputSpec is not required to avoid confusion and stop you from constructing difficult
     * to understand InputSpecs. You can use parameters with default values nested inside
     * other InputSpecs, see e.g. the `do_logging` parameter in the example code. The returned
     * one_of() InputSpec is always treated as required.
     *
     * The optional @p on_parse_callback may be used to perform additional actions after parsing one
     * of the specs. The index of the parsed spec inside the @p specs vector is passed as an
     * argument. An exemplary use case is to map the index to an enum value and store it in the
     * container. This lets you perform a switch on the enum value to easily obtain the correct
     * parsed data from the container. The store_index_as() function can be used to create such a
     * callback.
     *
     * @note The one_of() function is not intended to be used for selecting from a fixed set of
     * different values of the same type. Use the selection() function for this purpose.
     *
     * @relatedalso InputSpec
     */
    [[nodiscard]] InputSpec one_of(std::vector<InputSpec> specs,
        std::function<void(InputParameterContainer& container, std::size_t index)>
            on_parse_callback = nullptr);

    /**
     * This function may be used to produce the optional argument of the one_of() function. It
     * returns a callback that stores the index of the parsed spec in the container under @p name.
     * The index is stored on the same level as the parsed spec from the one_of() function.
     * Example:
     *
     * @code
     * enum class TimeIntegrationMethod
     * {
     *   OneStepTheta,
     *   GenAlpha,
     * };
     *
     * one_of({
     *  group("OneStepTheta",
     *  {
     *    parameter<double>("theta"),
     *  }),
     *  group("GenAlpha",
     *  {
     *    parameter<double>("alpha_f"),
     *    parameter<double>("alpha_m"),
     *    parameter<double>("gamma"),
     *  }),
     *  },
     *  store_index_as<TimeIntegrationMethod>("index",
     *    {TimeIntegrationMethod::OneStepTheta,
     *     TimeIntegrationMethod::GenAlpha})
     * );
     * @endcode
     *
     * Additionally, you can provide a reindexing vector to map the indices to other values. This is
     * especially useful if you map the often arbitrarily ordered indices to enum constants that
     * document the meaning of the index. This is demonstrated in the example above. If the
     * @p reindexing vector is not provided, the index is stored as is.
     *
     * @relatedalso InputSpec
     */
    template <typename T>
    auto store_index_as(std::string name, std::vector<T> reindexing = {});

    /**
     * The InputSpec returned by this function represents a list where each element matches the
     * given @p spec. The size of the list can be specified in the @p data, either as a fixed value
     * or dynamic_size (the default). For example,
     *
     * @code
     * list("my_list", parameter<int>("a"), {.size = 3});
     * @endcode
     *
     * will match the following yaml input:
     *
     * @code
     * my_list:
     *   - a: 42
     *   - a: 7
     *   - a: 3
     * @endcode
     *
     * @note The list() function is not intended to specify an array of primitive values of type T.
     * Use the `parameter<std::vector<T>>()` function for this purpose. As a developer of a new
     * InputSpec, you should rarely need the list() function.
     */
    [[nodiscard]] InputSpec list(std::string name, InputSpec spec, ListData data = {});

  }  // namespace InputSpecBuilders
}  // namespace Core::IO


// --- template definitions --- //

template <Core::IO::SupportedType T>
void Core::IO::Internal::ParameterSpec<T>::parse(
    ValueParser& parser, InputParameterContainer& container) const
{
  if (parser.peek() == name)
    parser.consume(name);
  else if (data.default_value.index() == 1)
  {
    container.add(name, std::get<1>(data.default_value));
    return;
  }
  else
  {
    std::string next_token{parser.peek()};
    FOUR_C_THROW("Could not parse '{}'. Next token is '{}'.", name, next_token);
  }

  if constexpr (dynamic_rank<T>() == 0)
  {
    container.add(name, parser.read<T>());
  }
  else
  {
    struct SizeVisitor
    {
      int operator()(int size) const
      {
        if (size > 0) return size;

        FOUR_C_THROW("Reading a vector from a dat-style string requires a known size.");
      }
      int operator()(const InputSpecBuilders::SizeCallback& callback) const
      {
        return callback(container);
      }
      InputParameterContainer& container;
    };

    if constexpr (dynamic_rank<T>() > 0)
    {
      std::array<std::size_t, dynamic_rank<T>()> size_info;
      for (std::size_t i = 0; i < dynamic_rank<T>(); ++i)
      {
        size_info[i] = std::visit(SizeVisitor{container}, data.size[i]);
      }
      auto parsed = parser.read<T>(size_info);
      container.add(name, parsed);
    }
  }

  if (data.on_parse_callback) data.on_parse_callback(container);
}


template <Core::IO::SupportedType T>
bool Core::IO::Internal::ParameterSpec<T>::match(ConstYamlNodeRef node,
    InputSpecBuilders::Storage& container, IO::Internal::MatchEntry& match_entry) const
{
  auto spec_name = ryml::to_csubstr(name);

  if (!node.node.is_map() || !node.node.has_child(spec_name))
  {
    // It is OK to not encounter an optional parameter
    if (data.default_value.index() == 1)
    {
      set_default_value(container);
      match_entry.state = IO::Internal::MatchEntry::State::defaulted;
      return true;
    }
    else
    {
      return false;
    }
  }

  // A child with the name of the spec exists, so this is at least a partial match.
  match_entry.state = IO::Internal::MatchEntry::State::partial;
  auto entry_node = node.wrap(node.node[spec_name]);
  match_entry.matched_node = entry_node.node.id();

  FOUR_C_ASSERT(entry_node.node.key() == name, "Internal error.");

  T value;

  auto status = read_value_from_yaml(entry_node, value);
  if (status != YamlReadStatus::success)
  {
    if (has_flag(status, YamlReadStatus::wrong_size))
    {
      match_entry.additional_info = "has incorrect size";
      return false;
    }
    if (has_flag(status, YamlReadStatus::wrong_type))
    {
      if constexpr (std::is_enum_v<T>)
      {
        std::string choices_string;
        for (const auto& e : EnumTools::enum_names<T>())
        {
          choices_string += std::string(e) + "|";
        }
        choices_string.pop_back();

        match_entry.additional_info = "has wrong value, possible values: " + choices_string;
      }
      else
      {
        match_entry.additional_info = "has wrong type, expected type: " + get_pretty_type_name<T>();
      }
      return false;
    }
  }

  // During reading of YAML values we can only check for static sizes. Check for dynamic sizes here.
  if constexpr (dynamic_rank<T>() > 0)
  {
    if (!has_correct_size(value, container))
    {
      match_entry.additional_info = "has incorrect size";
      return false;
    }
  }
  if (!validate_helper(value, data.validator))
  {
    std::ostringstream ss;
    ss << "does not pass validation: ";
    data.validator->describe(ss);
    match_entry.additional_info = ss.str();
    return false;
  }

  auto [ok, msg] = data.store(container, std::move(value));
  if (!ok)
  {
    match_entry.additional_info = msg;
    return false;
  }

  match_entry.state = IO::Internal::MatchEntry::State::matched;
  if (data.on_parse_callback && Internal::holds<InputParameterContainer>(container))
    data.on_parse_callback(std::any_cast<InputParameterContainer&>(container));
  return true;
}


template <Core::IO::SupportedType T>
void Core::IO::Internal::ParameterSpec<T>::emit_metadata(YamlNodeRef node) const
{
  node.node |= ryml::MAP;
  node.node["name"] << name;

  if constexpr (dynamic_rank<T>() == 0)
    IO::Internal::emit_type_as_yaml<StoredType>(node.node);
  else
  {
    struct DynamicSizeVisitor
    {
      int operator()(int size) const { return size; }
      int operator()(const InputSpecBuilders::SizeCallback& callback) const
      {
        return InputSpecBuilders::dynamic_size;
      }
    };

    std::array<std::size_t, dynamic_rank<T>()> size_info;
    for (std::size_t i = 0; i < dynamic_rank<T>(); ++i)
    {
      size_info[i] = std::visit(DynamicSizeVisitor{}, data.size[i]);
    }
    IO::Internal::emit_type_as_yaml<StoredType>(node.node, size_info);
  }

  if (!data.description.empty())
  {
    emit_value_as_yaml(node.wrap(node.node["description"]), data.description);
  }
  emit_value_as_yaml(node.wrap(node.node["required"]), !(data.default_value.index() == 1));
  if (data.default_value.index() == 1)
  {
    emit_value_as_yaml(node.wrap(node.node["default"]), std::get<1>(data.default_value));
  }

  // Enums are special: their validation is built in
  if constexpr (std::is_enum_v<T>)
  {
    if (data.validator)
    {
      // Only use the valid values.
      data.validator->emit_metadata(node);
    }
    else
    {
      // If no validator is set, we emit all enum values as valid values.
      auto choices_node = node.node["choices"];
      choices_node |= ryml::SEQ;
      for (const auto& choice : EnumTools::enum_values<T>())
      {
        auto choice_node = choices_node.append_child();
        choice_node |= ryml::MAP;
        emit_value_as_yaml(node.wrap(choice_node["name"]), choice);
        if (data.enum_value_description)
        {
          emit_value_as_yaml(
              node.wrap(choice_node["description"]), data.enum_value_description(choice));
        }
      }
    }
  }
  else if (data.validator)
  {
    data.validator->emit_metadata(node.wrap(node.node["validator"]));
  }
}

template <Core::IO::SupportedType T>
bool Core::IO::Internal::ParameterSpec<T>::emit(YamlNodeRef node,
    const InputParameterContainer& container, const InputSpecEmitOptions& options) const
{
  node.node |= ryml::MAP;
  // Value present in container
  if (auto value = container.get_if<StoredType>(name))
  {
    if (options.emit_defaulted_values || !(data.default_value.index() == 1) ||
        std::get<1>(data.default_value) != *value)
    {
      auto value_node = node.node.append_child();
      value_node << ryml::key(name);
      emit_value_as_yaml(node.wrap(value_node), *value);
    }
    return true;
  }
  // Not present but we have a default
  else if (data.default_value.index() == 1)
  {
    if (options.emit_defaulted_values)
    {
      auto value_node = node.node.append_child();
      value_node << ryml::key(name);
      emit_value_as_yaml(node.wrap(value_node), std::get<1>(data.default_value));
    }
    return true;
  }
  else
  {
    return false;
  }
}

template <Core::IO::SupportedType T>
void Core::IO::Internal::ParameterSpec<T>::set_default_value(
    InputSpecBuilders::Storage& container) const
{
  auto [ok, _] = data.store(container, T{std::get<1>(data.default_value)});
  FOUR_C_ASSERT(ok, "Internal error: could not set default value for parameter '{}'.", name);
}


template <Core::IO::SupportedType T>
bool Core::IO::Internal::ParameterSpec<T>::has_correct_size(
    const T& val, const InputSpecBuilders::Storage& container) const
{
  if constexpr (dynamic_rank<T>() == 0)
  {
    return true;
  }
  else
  {
    struct SizeVisitor
    {
      int operator()(int size) const { return size; }
      int operator()(const InputSpecBuilders::SizeCallback& callback) const
      {
        FOUR_C_ASSERT(Internal::holds<InputParameterContainer>(container),
            "Size callback can only be used with InputParameterContainer.");
        return callback(std::any_cast<const InputParameterContainer&>(container));
      }
      const InputSpecBuilders::Storage& container;
    };

    std::array<std::size_t, dynamic_rank<T>()> size_info;
    for (std::size_t i = 0; i < dynamic_rank<T>(); ++i)
    {
      size_info[i] = std::visit(SizeVisitor{container}, data.size[i]);
    }
    SizeChecker size_checker;
    return size_checker(val, size_info.data());
  }
}


template <typename T>
void Core::IO::Internal::DeprecatedSelectionSpec<T>::parse(
    ValueParser& parser, InputParameterContainer& container) const
{
  if (parser.peek() == name)
    parser.consume(name);
  else if (data.default_value.index() == 1)
  {
    container.add(name, std::get<1>(data.default_value));
    return;
  }
  else
  {
    std::string next_token{parser.peek()};
    FOUR_C_THROW("Could not parse '{}'. Next token is '{}'.", name, next_token);
  }

  auto value = parser.read<InputType>();
  for (const auto& choice : choices)
  {
    if (choice.first == value)
    {
      container.add(name, choice.second);
      return;
    }
  }
  std::stringstream parsed_value_str;
  IO::Internal::DatPrinter{}(parsed_value_str, value);

  FOUR_C_THROW("Could not parse parameter '{}': invalid value '{}'. Valid options are: {}",
      name.c_str(), parsed_value_str.str(), choices_string);
}


template <typename T>
bool Core::IO::Internal::DeprecatedSelectionSpec<T>::match(ConstYamlNodeRef node,
    InputSpecBuilders::Storage& container, IO::Internal::MatchEntry& match_entry) const
{
  auto spec_name = ryml::to_csubstr(name);

  if (!node.node.is_map() || !node.node.has_child(spec_name))
  {
    // It is OK to not encounter an optional parameter
    if (data.default_value.index() == 1)
    {
      [[maybe_unused]] auto [ok, _] = data.store(container, T(std::get<1>(data.default_value)));
      FOUR_C_ASSERT(ok, "Internal error: could not set default value for parameter '{}'.", name);
      match_entry.state = IO::Internal::MatchEntry::State::defaulted;
      return true;
    }
    else
    {
      return false;
    }
  }

  // A child with the name of the spec exists, so this is at least a partial match.
  match_entry.state = IO::Internal::MatchEntry::State::partial;
  auto entry_node = node.wrap(node.node[spec_name]);
  match_entry.matched_node = entry_node.node.id();

  FOUR_C_ASSERT(entry_node.node.key() == name, "Internal error.");

  InputType value;
  auto status = read_value_from_yaml(entry_node, value);
  if (status == YamlReadStatus::success)
  {
    for (const auto& choice : choices)
    {
      if (choice.first == value)
      {
        auto [ok, msg] = data.store(container, T(choice.second));
        if (!ok)
        {
          match_entry.additional_info = msg;
          return false;
        }
        match_entry.state = IO::Internal::MatchEntry::State::matched;
        match_entry.matched_node = entry_node.node.id();
        if (data.on_parse_callback && Internal::holds<InputParameterContainer>(container))
          data.on_parse_callback(std::any_cast<InputParameterContainer&>(container));
        return true;
      }
    }
  }

  match_entry.additional_info = "has wrong value, possible values: " + choices_string;
  return false;
}

template <typename T>
void Core::IO::Internal::DeprecatedSelectionSpec<T>::print(
    std::ostream& stream, std::size_t indent) const
{
  stream << "// " << std::string(indent, ' ') << name;

  if (data.default_value.index() == 1)
  {
    // Find the choice that corresponds to the default value.
    auto default_value_it = std::find_if(choices.begin(), choices.end(),
        [&](const auto& choice) { return choice.second == std::get<1>(data.default_value); });
    FOUR_C_ASSERT(
        default_value_it != choices.end(), "Internal error: default value not found in choices.");

    stream << " (default: ";
    IO::Internal::DatPrinter{}(stream, default_value_it->first);
    stream << ")";
  }
  {
    stream << " (choices: ";
    stream << choices_string;
    stream << ")";
  }
  if (!data.description.empty())
  {
    stream << " " << std::quoted(Core::Utils::trim(data.description));
  }
  stream << "\n";
}

template <typename T>
void Core::IO::Internal::DeprecatedSelectionSpec<T>::emit_metadata(YamlNodeRef node) const
{
  node.node |= ryml::MAP;
  node.node["name"] << name;

  if constexpr (OptionalType<T>) emit_value_as_yaml(node.wrap(node.node["noneable"]), true);
  node.node["type"] = "enum";

  if (!data.description.empty())
  {
    emit_value_as_yaml(node.wrap(node.node["description"]), data.description);
  }
  emit_value_as_yaml(node.wrap(node.node["required"]), !(data.default_value.index() == 1));
  if (data.default_value.index() == 1)
  {
    // Find the choice that corresponds to the default value.
    auto default_value_it = std::find_if(choices.begin(), choices.end(),
        [&](const auto& choice) { return choice.second == std::get<1>(data.default_value); });
    FOUR_C_ASSERT(
        default_value_it != choices.end(), "Internal error: default value not found in choices.");
    emit_value_as_yaml(node.wrap(node.node["default"]), default_value_it->first);
  }
  node.node["choices"] |= ryml::SEQ;
  for (const auto& [choice_string, _] : choices)
  {
    auto entry = node.node["choices"].append_child();
    // Write every choice entry as a map to easily extend the information at a later point.
    entry |= ryml::MAP;
    emit_value_as_yaml(node.wrap(entry["name"]), choice_string);
  }
}

template <typename T>
bool Core::IO::Internal::DeprecatedSelectionSpec<T>::emit(YamlNodeRef node,
    const InputParameterContainer& container, const InputSpecEmitOptions& options) const
{
  node.node |= ryml::MAP;

  const auto emit_key_value = [&](const std::string& key, const StoredType& value)
  {
    for (const auto& choice : choices)
    {
      if (choice.second == value)
      {
        auto value_node = node.node.append_child();
        value_node << ryml::key(key);
        emit_value_as_yaml(node.wrap(value_node), choice.first);
        return true;
      }
    }
    return false;
  };

  // Value present in container
  if (auto value = container.get_if<StoredType>(name))
  {
    if (options.emit_defaulted_values || !(data.default_value.index() == 1) ||
        std::get<1>(data.default_value) != *value)
    {
      return emit_key_value(name, *value);
    }
    return true;
  }
  // Not present but we have a default
  else if (data.default_value.index() == 1)
  {
    if (options.emit_defaulted_values)
    {
      return emit_key_value(name, std::get<1>(data.default_value));
    }
    return true;
  }
  else
  {
    return false;
  }
}

template <typename T>
void Core::IO::Internal::DeprecatedSelectionSpec<T>::set_default_value(
    InputSpecBuilders::Storage& container) const
{
  [[maybe_unused]] auto [ok, _] = data.store(container, T{std::get<1>(data.default_value)});
  FOUR_C_ASSERT(ok, "Internal error: could not set default value for parameter '{}'.", name);
}

template <typename T>
  requires(std::is_enum_v<T>)
void Core::IO::Internal::SelectionSpec<T>::parse(
    ValueParser& parser, InputParameterContainer& container) const
{
  FOUR_C_THROW("Not implemented. You cannot use a selection in the legacy dat format.");
}


template <typename T>
  requires(std::is_enum_v<T>)
bool Core::IO::Internal::SelectionSpec<T>::match(ConstYamlNodeRef node,
    InputSpecBuilders::Storage& container, IO::Internal::MatchEntry& match_entry) const
{
  const auto group_name_substr = ryml::to_csubstr(group_name);

  const bool group_node_is_input = node.node.has_key() && (node.node.key() == group_name);
  const bool group_exists_nested = node.node.is_map() && node.node.has_child(group_name_substr) &&
                                   node.node[group_name_substr].is_map();

  if (!group_exists_nested && !group_node_is_input)
  {
    return false;
  }

  auto group_node = group_node_is_input ? node : node.wrap(node.node[group_name_substr]);

  // Matching the key of the group is at least a partial match.
  match_entry.state = IO::Internal::MatchEntry::State::partial;
  match_entry.matched_node = group_node.node.id();

  // The selection group must have exactly one child group with one of the selector values as key.
  if (group_node.node.num_children() != 1)
  {
    match_entry.additional_info = "needs exactly one child with selector value as key";
    return false;
  }

  const auto selected_node = group_node.wrap(group_node.node.first_child());
  const auto selector_str = selected_node.node.key();
  std::optional<T> selector_value =
      EnumTools::enum_cast<T>(std::string_view{selector_str.data(), selector_str.size()});

  if (!selector_value)
  {
    match_entry.additional_info = "has wrong selector value, expected one of: ";
    for (const auto& [choice_value, _] : choices)
    {
      match_entry.additional_info += std::string{EnumTools::enum_name<T>(choice_value)} + "|";
    }
    match_entry.additional_info.pop_back();  // Remove the last '|'
    return false;
  }

  FOUR_C_ASSERT(
      choices.contains(*selector_value), "Internal error: selector not found in choices.");
  const auto& selected_spec = choices.at(*selector_value);

  InputSpecBuilders::Storage subcontainer;
  init_my_storage(subcontainer);
  auto choice_matched = selected_spec.impl().match(
      group_node, subcontainer, match_entry.append_child(&selected_spec));
  if (!choice_matched) return false;

  if (data.store_selector)
  {
    auto [ok, msg] = data.store_selector(subcontainer, std::move(*selector_value));
    if (!ok)
    {
      match_entry.additional_info = msg;
      return false;
    }
  }

  if (data.transform_data)
  {
    auto [ok, msg] = data.transform_data(
        container, std::any_cast<InputParameterContainer&&>(std::move(subcontainer)));
    if (!ok)
    {
      match_entry.additional_info = msg;
      return false;
    }
  }
  else
  {
    auto [ok, msg] = move_my_storage(container, std::move(subcontainer));
    if (!ok)
    {
      match_entry.additional_info = msg;
      return false;
    }
  }

  // Everything was correctly matched, so mark the whole node as matched.
  match_entry.state = IO::Internal::MatchEntry::State::matched;
  return true;
}

template <typename T>
  requires(std::is_enum_v<T>)
void Core::IO::Internal::SelectionSpec<T>::print(std::ostream& stream, std::size_t indent) const
{
  stream << "// " << std::string(indent, ' ') << group_name;
  stream << "// " << std::string(indent + 2, ' ') << "<choices>\n";
  for (const auto& [selector_value, spec] : choices)
  {
    stream << "\n";
    spec.impl().print(stream, indent + 4);
  }
  stream << "\n";
}

template <typename T>
  requires(std::is_enum_v<T>)
void Core::IO::Internal::SelectionSpec<T>::emit_metadata(YamlNodeRef node) const
{
  node.node |= ryml::MAP;
  node.node["name"] << group_name;

  if constexpr (OptionalType<T>) emit_value_as_yaml(node.wrap(node.node["noneable"]), true);
  node.node["type"] = "selection";

  if (!data.description.empty())
  {
    emit_value_as_yaml(node.wrap(node.node["description"]), data.description);
  }
  emit_value_as_yaml(node.wrap(node.node["required"]), true);

  node.node["choices"] |= ryml::SEQ;
  for (const auto& [choice, spec] : choices)
  {
    auto entry = node.node["choices"].append_child();
    entry |= ryml::MAP;
    emit_value_as_yaml(node.wrap(entry["name"]), choice);
    spec.impl().emit_metadata(node.wrap(entry["spec"]));
  }
}

template <typename T>
  requires(std::is_enum_v<T>)
bool Core::IO::Internal::SelectionSpec<T>::emit(YamlNodeRef node,
    const InputParameterContainer& container, const InputSpecEmitOptions& options) const
{
  node.node |= ryml::MAP;
  if (container.has_group(group_name))
  {
    auto group_node = node.node.append_child();
    group_node << ryml::key(group_name);
    group_node |= ryml::MAP;

    // Try to emit each of the choices, only one of them will succeed.
    bool success = false;
    for (const auto& [selector_value, spec] : choices)
    {
      if (spec.impl().emit(node.wrap(group_node), container.group(group_name), options))
      {
        success = true;
        break;
      }
    }

    return success;
  }
  return false;
}

template <typename T>
  requires(std::is_enum_v<T>)
void Core::IO::Internal::SelectionSpec<T>::set_default_value(
    InputSpecBuilders::Storage& container) const
{
  FOUR_C_THROW("Internal error: set_default_value() called on a SelectionSpec.");
}


template <typename T>
auto Core::IO::InputSpecBuilders::from_parameter(const std::string& name)
{
  return [name](const InputParameterContainer& container) -> T
  {
    const T* val = container.get_if<T>(name);
    FOUR_C_ASSERT_ALWAYS(val, "Parameter '{}' not found in container.", name);
    return *val;
  };
}


template <Core::IO::SupportedType T>
Core::IO::InputSpec Core::IO::InputSpecBuilders::parameter(
    std::string name, ParameterDataIn<T>&& data)
{
  Internal::ParameterData<T> internal_data;
  internal_data.description = data.description;
  if constexpr (OptionalType<T>)
  {
    // An optional<T> implies a default_value corresponding to the empty state.
    internal_data.default_value = std::nullopt;
  }
  else
  {
    internal_data.default_value = data.default_value;
  }
  if constexpr (dynamic_rank<T>() == 1)
  {
    internal_data.size[0] = data.size;
  }
  else if constexpr (dynamic_rank<T>() > 1)
  {
    internal_data.size = data.size;
  }
  internal_data.on_parse_callback = data.on_parse_callback;

  internal_data.validator = data.validator;

  internal_data.store = data.store ? data.store : in_container<T>(name);

  if constexpr (std::is_enum_v<T>)
    internal_data.enum_value_description = data.enum_value_description;

  if (internal_data.default_value.index() == 1 &&
      !Internal::validate_helper(std::get<1>(internal_data.default_value), data.validator))
  {
    std::stringstream validation_error_stream;
    validation_error_stream << "Default value '";
    Core::IO::Internal::DatPrinter{}(
        validation_error_stream, std::get<1>(internal_data.default_value));
    validation_error_stream << "' does not pass validation: ";
    data.validator->describe(validation_error_stream);

    FOUR_C_THROW("{}", validation_error_stream.str());
  }


  return IO::Internal::make_spec(Internal::ParameterSpec<T>{.name = name, .data = internal_data},
      {
          .name = name,
          .description = data.description,
          .required = !(internal_data.default_value.index() == 1),
          .has_default_value = internal_data.default_value.index() == 1,
          .n_specs = 1,
          .type = Internal::InputSpecType::parameter,
          .stores_to = &internal_data.store.stores_to(),
      });
}

template <typename Selector, typename StorageType>
  requires(std::is_enum_v<Selector>)
Core::IO::InputSpec Core::IO::InputSpecBuilders::selection(
    std::string name, std::vector<InputSpec> choices, SelectionData<Selector, StorageType> data)
{
  std::map<Selector, InputSpec> choices_map;

  for (auto&& choice : choices)
  {
    const auto& choice_name = choice.impl().name();

    FOUR_C_ASSERT_ALWAYS(!name.empty(),
        "Selection '{}' has at least one choice with an empty name. "
        "Every choice needs to be a named InputSpec.",
        name);

    const auto selector_value = EnumTools::enum_cast<Selector>(choice_name);
    FOUR_C_ASSERT_ALWAYS(selector_value,
        "Selection '{}' has a choice with name '{}' that does not match any enum constant of type "
        "'{}'.",
        name, choice_name, EnumTools::enum_type_name<Selector>());

    FOUR_C_ASSERT_ALWAYS(!choices_map.contains(*selector_value),
        "Selection '{}' has at least two choices with the same name '{}'. "
        "Every choice needs to be a uniquely named InputSpec.",
        name, *selector_value);

    choices_map.emplace(*selector_value, std::move(choice));
  }

  // Ensure that every enum constant is mapped to a spec.
  for (const auto& e : EnumTools::enum_values<Selector>())
  {
    FOUR_C_ASSERT_ALWAYS(choices_map.contains(e),
        "You need to give an InputSpec for every possible value of enum '{}'. Missing "
        "choice for enum constant '{}'.",
        EnumTools::enum_type_name<Selector>(), EnumTools::enum_name(e));
  }

  std::size_t max_specs_for_choices = std::ranges::max_element(
      choices_map, {}, [](const auto& spec) { return spec.second.impl().data.n_specs; })
                                          ->second.impl()
                                          .data.n_specs;

  StoreFunction<Storage> move_my_storage;
  if constexpr (std::is_same_v<StorageType, DefaultStorage>)
  {
    move_my_storage = Internal::store_container_in_container(name);
  }
  else
  {
    move_my_storage = Internal::wrap_group_in_container<StorageType>(
        data.store ? data.store : in_container<StorageType>(name));
  }

  Internal::InputSpecImpl::CommonData common_data{
      .name = name,
      .description = data.description,
      .required = true,
      .has_default_value = false,
      // one for the group plus the number of specs for the choices.
      .n_specs = 1 + max_specs_for_choices,
      .type = Internal::InputSpecType::selection,
      .stores_to =
          data.transform_data ? &data.transform_data.stores_to() : &move_my_storage.stores_to(),
  };

  StoreFunction<Selector> store_selector;
  if constexpr (std::is_same_v<StorageType, DefaultStorage>)
  {
    store_selector =
        data.store_selector ? data.store_selector : in_container<Selector>("_selector");
  }
  else
  {
    // Use whatever was given. This may be null by default, which will be checked during match.
    store_selector = data.store_selector;
  }

  return IO::Internal::make_spec(
      Internal::SelectionSpec<Selector>{
          .group_name = name,
          .choices = std::move(choices_map),
          .data =
              {
                  .description = data.description,
                  .transform_data = data.transform_data,
                  .store_selector = store_selector,
              },
          .init_my_storage = [](Storage& storage) { storage.emplace<StorageType>(); },
          .move_my_storage = std::move(move_my_storage),
      },
      common_data);
}


template <typename T>
Core::IO::InputSpec Core::IO::InputSpecBuilders::input_field(
    const std::string name, InputFieldData<T> data)
{
  auto store = data.store ? data.store : in_container<InputField<T>>(name);
  auto spec = selection<Internal::InputFieldType>(name,
      {
          parameter<T>("constant", {.description = "Constant value for the field."}),
          parameter<std::filesystem::path>(
              "from_file", {.description = "Path to a file containing the input field data."}),
          parameter<std::string>("from_mesh",
              {.description = "Refer to a field defined in the input mesh by a name."}),
          parameter<std::string>(
              "field_reference", {.description = "Refer to a globally defined field by a name."}),
      },
      {
          .description = data.description,
          .transform_data = Internal::make_input_field_data_transform<T>(name, std::move(store)),
      });
  return spec;
};


template <typename T>
Core::IO::InputSpec Core::IO::Internal::selection_internal(std::string name,
    std::map<std::string, RemoveOptional<T>> choices, InputSpecBuilders::ParameterDataIn<T> data)
{
  FOUR_C_ASSERT_ALWAYS(!choices.empty(), "Selection must have at least one choice.");

  // In case all strings match the enum values, we can use the enum type directly.
  if constexpr (std::is_enum_v<RemoveOptional<T>>)
  {
    using Enum = RemoveOptional<T>;
    if (EnumTools::enum_count<Enum>() == choices.size())
    {
      const bool all_enum_values_match = std::ranges::all_of(EnumTools::enum_names<Enum>(),
          [&choices](const auto& enum_name) { return choices.contains(std::string(enum_name)); });
      if (all_enum_values_match)
      {
        FOUR_C_THROW(
            "All choices for selection '{}' match enum values of type '{}'. "
            "Use parameter<{}>() instead.",
            name, EnumTools::enum_type_name<Enum>(), EnumTools::enum_type_name<Enum>());
      }
    }
  }


  // If we have a std::optional type, we need to convert the choices.
  typename DeprecatedSelectionSpec<T>::ChoiceMap modified_choices;
  std::string choices_string;
  for (auto&& [key, value] : choices)
  {
    modified_choices.emplace(std::move(key), std::move(value));
    choices_string += key + "|";
  }
  choices_string.pop_back();

  if constexpr (OptionalType<T>)
  {
    modified_choices[std::nullopt] = T{};
    choices_string += "|none";
  }

  DeprecatedSelectionData<T> internal_data;
  internal_data.description = data.description;
  if constexpr (OptionalType<T>)
  {
    // An optional<T> implies a default_value corresponding to the empty state.
    internal_data.default_value = std::nullopt;
  }
  else
  {
    internal_data.default_value = data.default_value;
  }
  internal_data.on_parse_callback = data.on_parse_callback;
  internal_data.store = data.store ? data.store : InputSpecBuilders::in_container<T>(name);

  const bool has_default_value = internal_data.default_value.index() == 1;

  // Check that we have a default value that is in the choices.
  if (has_default_value)
  {
    const auto& default_value = std::get<1>(internal_data.default_value);
    auto default_value_it = std::find_if(modified_choices.begin(), modified_choices.end(),
        [&](const auto& choice) { return choice.second == default_value; });

    if (default_value_it == modified_choices.end())
    {
      std::stringstream default_value_stream;
      Core::IO::Internal::DatPrinter{}(default_value_stream, default_value);
      FOUR_C_THROW("Default value '{}' of selection not found in choices '{}'.",
          default_value_stream.str(), choices_string);
    }
  }

  return IO::Internal::make_spec(
      Internal::DeprecatedSelectionSpec<T>{
          .name = name,
          .data = internal_data,
          .choices = modified_choices,
          .choices_string = choices_string,
      },
      {
          .name = name,
          .description = data.description,
          .required = !has_default_value,
          .has_default_value = has_default_value,
          .n_specs = 1,
          .type = InputSpecType::deprecated_selection,
          .stores_to = &internal_data.store.stores_to(),
      });
}


template <typename T>
  requires(std::is_enum_v<T>)
Core::IO::InputSpec Core::IO::InputSpecBuilders::deprecated_selection(
    std::string name, std::map<std::string, RemoveOptional<T>> choices, ParameterDataIn<T> data)
{
  return Internal::selection_internal(name, choices, data);
}


template <std::same_as<std::string> T>
Core::IO::InputSpec Core::IO::InputSpecBuilders::deprecated_selection(
    std::string name, std::vector<std::string> choices, ParameterDataIn<T> data)
{
  std::map<std::string, std::string> choices_with_strings;
  for (const auto& choice : choices)
  {
    choices_with_strings.emplace(choice, choice);
  }
  return Internal::selection_internal(name, choices_with_strings, data);
}



template <typename T>
auto Core::IO::InputSpecBuilders::store_index_as(std::string name, std::vector<T> reindexing)
{
  return [name, reindexing](InputParameterContainer& container, std::size_t index)
  {
    if (reindexing.empty())
    {
      container.add(name, static_cast<T>(index));
    }
    else
    {
      FOUR_C_ASSERT(index < reindexing.size(), "Index out of bounds.");
      container.add(name, reindexing[index]);
    }
  };
}

template <typename StorageType>
Core::IO::InputSpec Core::IO::InputSpecBuilders::group(std::string name,
    std::vector<InputSpec> specs, Core::IO::InputSpecBuilders::GroupData<StorageType> data)
{
  auto internal_all_of = all_of(std::move(specs));

  if (auto* stores_to = internal_all_of.impl().data.stores_to;
      stores_to && *stores_to != typeid(StorageType))
  {
    FOUR_C_THROW("Group '{}' is stored as '{}' but contains specs that store to '{}'.", name,
        Core::Utils::try_demangle(typeid(StorageType).name()),
        Core::Utils::try_demangle(stores_to->name()));
  }

  // The group is defaultable if it is not required and all child specs are defaultable.
  const bool defaultable = !data.required && internal_all_of.impl().has_default_value();

  StoreFunction<Storage> move_my_storage;
  if constexpr (std::is_same_v<StorageType, DefaultStorage>)
  {
    move_my_storage = Internal::store_container_in_container(name);
  }
  else
  {
    move_my_storage = Internal::wrap_group_in_container<StorageType>(
        data.store ? data.store : in_container<StorageType>(name));
  }

  Internal::InputSpecImpl::CommonData common_data{
      .name = name,
      .description = data.description,
      .required = data.required,
      .has_default_value = defaultable,
      .n_specs = internal_all_of.impl().data.n_specs + 1,
      .type = Internal::InputSpecType::group,
      .stores_to = &move_my_storage.stores_to(),
  };

  return Internal::make_spec(
      Internal::GroupSpec{
          .name = name,
          .data =
              {
                  .description = data.description,
                  .required = data.required,
                  .defaultable = defaultable,
              },
          .spec = std::move(internal_all_of),
          .init_my_storage = [](Storage& storage) { storage.emplace<StorageType>(); },
          .move_my_storage = move_my_storage,
      },
      common_data);
}

FOUR_C_NAMESPACE_CLOSE

#endif
