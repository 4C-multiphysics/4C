// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_SPEC_BUILDERS_HPP
#define FOUR_C_IO_INPUT_SPEC_BUILDERS_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.templates.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_io_input_types.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_io_yaml.hpp"
#include "4C_utils_string.hpp"

#include <magic_enum/magic_enum_iostream.hpp>

#include <algorithm>
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
        out << "<unprintable>";
      }

      template <SupportedType T>
      void operator()(std::ostream& out, const T& val) const
      {
        using magic_enum::iostream_operators::operator<<;
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
      std::string operator()() { return "enum"; }
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
        node["choices"] |= ryml::SEQ;
        for (const auto& choice_string : magic_enum::enum_values<Enum>())
        {
          auto entry = node["choices"].append_child();
          // Write every choice entry as a map to easily extend the information at a later point.
          entry |= ryml::MAP;
          emit_value_as_yaml(entry["name"], choice_string);
        }
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

    template <typename T>
    struct YamlTypeEmitter<std::optional<T>>
    {
      void operator()(ryml::NodeRef node, size_t* size)
      {
        FOUR_C_ASSERT(node.is_map(), "Expected a map node.");
        // Pull up the std::optional aspect. The fact that this type wraps another type is specific
        // to C++ and not relevant to other tools. Simply knowing that a type can be empty is
        // enough for them.
        emit_value_as_yaml(node["noneable"], true);
        YamlTypeEmitter<T>{}(node, size);
      }
    };

    template <typename T>
      requires(rank<T>() == 0)
    void emit_type_as_yaml(ryml::NodeRef node)
    {
      YamlTypeEmitter<T>{}(node, nullptr);
    }

    template <typename T>
    void emit_type_as_yaml(ryml::NodeRef node, std::array<std::size_t, rank<T>()> size)
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

      /**
       * A MatchEntry can only match a single node. Logical specs like all_of and one_of are not
       * considered to match nodes themselves, as this is done by their children.
       */
      ryml::id_type matched_node{ryml::npos};

      enum class State : std::uint8_t
      {
        unmatched,
        matched,
        partial,
        defaulted,
      };

      enum class Type : std::uint8_t
      {
        unknown,
        parameter,
        group,
        all_of,
        one_of,
        list,
      };

      State state{State::unmatched};
      Type type{Type::unknown};


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

    class InputSpecTypeErasedBase
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
      };

      virtual ~InputSpecTypeErasedBase() = default;

      /**
       * @param data The common data of the spec.
       */
      InputSpecTypeErasedBase(CommonData data);

      virtual void parse(ValueParser& parser, InputParameterContainer& container) const = 0;

      /**
       * Returns true if the node matches the spec and stores the value in the container. The passed
       * @p node is the parent node which might contain data matching the spec. Every spec needs
       * to check if it can find required data in this node. If yes, the InputSpec should report
       * itself as matched in the @p match_entry. Note that the @p match_entry already refers to the
       * spec that is being matched. When matching more specs internally, the spec needs to append
       * children to the match_entry.
       */
      virtual bool match(ConstYamlNodeRef node, InputParameterContainer& container,
          MatchEntry& match_entry) const = 0;

      virtual void set_default_value(InputParameterContainer& container) const = 0;

      //! Emit metadata. This function always emits into a map, i.e., the implementation must
      //! insert keys and values into the yaml emitter.
      virtual void emit_metadata(ryml::NodeRef node) const = 0;

      virtual bool emit(YamlNodeRef node, const InputParameterContainer&,
          const InputSpecEmitOptions& options) const = 0;

      [[nodiscard]] virtual std::string pretty_type_name() const = 0;

      [[nodiscard]] virtual std::unique_ptr<InputSpecTypeErasedBase> clone() const = 0;

      void print(std::ostream& stream, std::size_t indent) const { do_print(stream, indent); }

      [[nodiscard]] const std::string& name() const { return data.name; }

      [[nodiscard]] const std::string& description() const { return data.description; }

      [[nodiscard]] std::string description_one_line() const;

      [[nodiscard]] bool required() const { return data.required; }

      [[nodiscard]] bool has_default_value() const { return data.has_default_value; }


      CommonData data;

     protected:
      InputSpecTypeErasedBase(const InputSpecTypeErasedBase&) = default;
      InputSpecTypeErasedBase& operator=(const InputSpecTypeErasedBase&) = default;
      InputSpecTypeErasedBase(InputSpecTypeErasedBase&&) noexcept = default;
      InputSpecTypeErasedBase& operator=(InputSpecTypeErasedBase&&) noexcept = default;

     private:
      virtual void do_print(std::ostream& stream, std::size_t indent) const = 0;
    };

    template <typename T>
    concept StoresType = requires(T t) { typename T::StoredType; };

    template <typename T>
    struct InputSpecTypeErasedImplementation : public InputSpecTypeErasedBase
    {
      template <typename T2>
      explicit InputSpecTypeErasedImplementation(T2&& wrapped, CommonData data)
          : InputSpecTypeErasedBase(std::move(data)), wrapped(std::forward<T2>(wrapped))
      {
      }

      void parse(ValueParser& parser, InputParameterContainer& container) const override
      {
        wrapped.parse(parser, container);
      }

      bool match(ConstYamlNodeRef node, InputParameterContainer& container,
          MatchEntry& match_entry) const override
      {
        return wrapped.match(node, container, match_entry);
      }

      void set_default_value(InputParameterContainer& container) const override
      {
        if constexpr (Internal::has_set_default_value<T>)
        {
          wrapped.set_default_value(container);
        }
        else
        {
          FOUR_C_ASSERT(has_default_value(),
              "Implementation error: this function should only be called if the wrapped type has "
              "an optional default value.");

          container.add(wrapped.name, std::get<1>(wrapped.data.default_value));
        }
      }

      void emit_metadata(ryml::NodeRef node) const override { wrapped.emit_metadata(node); }

      bool emit(YamlNodeRef node, const InputParameterContainer& container,
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

      std::string pretty_type_name() const override
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

      [[nodiscard]] std::unique_ptr<InputSpecTypeErasedBase> clone() const override
      {
        return std::make_unique<InputSpecTypeErasedImplementation<T>>(wrapped, data);
      }

      T wrapped;
    };

    template <typename T>
    InputSpec make_spec(T&& wrapped, InputSpecTypeErasedBase::CommonData data)
    {
      return InputSpec(std::make_unique<InputSpecTypeErasedImplementation<std::decay_t<T>>>(
          std::forward<T>(wrapped), std::move(data)));
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
     * A tag type to indicate that a parameter cannot take default values.
     */
    struct NoDefault
    {
    };

    /**
     * The type used for the default value of a parameter. If the parameter is optional, the user
     * cannot specify a default value, so this type becomes NoDefault. Otherwise, the default value
     * can be either not set, resulting in the std::monostate, or set to a value of the parameter
     * type.
     */
    template <typename T>
    using DefaultType =
        std::conditional_t<OptionalType<T>, NoDefault, std::variant<std::monostate, T>>;

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
       * The default value of the parameter. If this field is set, the parameter does not need to be
       * entered in the input. If the parameter is not entered, this default value is used.
       */
      DefaultType<T> default_value{};

      /**
       * An optional callback that is called after the value has been parsed. This can be used to
       * set additional values in the container.
       */
      ParameterCallback on_parse_callback{nullptr};
    };

    template <typename T>
      requires(rank<T>() == 1)
    struct ParameterDataIn<T>
    {
      using StoredType = T;

      std::string description{};

      DefaultType<T> default_value{};

      ParameterCallback on_parse_callback{nullptr};

      /**
       * The size of the vector. This can be a fixed size, #dynamic_size, or a callback that
       * determines the size at runtime.
       */
      Size size{dynamic_size};
    };

    template <typename T>
      requires(rank<T>() > 1)
    struct ParameterDataIn<T>
    {
      using StoredType = T;

      std::string description{};

      DefaultType<T> default_value{};

      ParameterCallback on_parse_callback{nullptr};

      std::array<Size, rank<T>()> size;
    };

    //! Additional parameters for a group().
    struct GroupData
    {
      /**
       * An optional description of the Group.
       */
      std::string description{};

      /**
       * Whether the Group is required or optional.
       */
      std::optional<bool> required{};

      /**
       * Whether the Group will store itself and its children with defaulted values, if the Group
       * is not encountered in the input. This only works if all children have default values.
       */
      bool defaultable{};
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

    namespace Internal
    {
      template <typename T>
      struct ParameterData
      {
        using StoredType = T;

        std::string description{};

        std::variant<std::monostate, StoredType> default_value{};

        ParameterCallback on_parse_callback{nullptr};

        std::array<Size, rank<T>()> size{};
      };

      template <typename T>
      struct SelectionData
      {
        using StoredType = T;

        std::string description{};

        std::variant<std::monostate, StoredType> default_value{};

        ParameterCallback on_parse_callback{nullptr};
      };

      template <SupportedType T>
      struct ParameterSpec
      {
        std::string name;
        using StoredType = T;
        ParameterData<T> data;
        void parse(ValueParser& parser, InputParameterContainer& container) const;
        bool match(ConstYamlNodeRef node, InputParameterContainer& container,
            IO::Internal::MatchEntry& match_entry) const;
        void emit_metadata(ryml::NodeRef node) const;
        bool emit(YamlNodeRef node, const InputParameterContainer& container,
            const InputSpecEmitOptions& options) const;
        [[nodiscard]] bool has_correct_size(
            const T& val, const InputParameterContainer& container) const;
      };

      /**
       * Note that a SelectionSpec can store any type since we never need to read or write values of
       * this type.
       */
      template <typename T>
      struct SelectionSpec
      {
        std::string name;
        using StoredType = T;
        //! The type that is used in the input file.
        using InputType =
            std::conditional_t<OptionalType<T>, std::optional<std::string>, std::string>;
        using ChoiceMap = std::map<InputType, StoredType>;
        SelectionData<T> data;
        ChoiceMap choices;
        //! The string representation of the choices.
        std::string choices_string;
        void parse(ValueParser& parser, InputParameterContainer& container) const;
        bool match(ConstYamlNodeRef node, InputParameterContainer& container,
            IO::Internal::MatchEntry& match_entry) const;
        void print(std::ostream& stream, std::size_t indent) const;
        void emit_metadata(ryml::NodeRef node) const;
        bool emit(YamlNodeRef node, const InputParameterContainer& container,
            const InputSpecEmitOptions& options) const;
      };

      struct GroupSpec
      {
        std::string name;
        GroupData data;
        std::vector<InputSpec> specs;

        void parse(ValueParser& parser, InputParameterContainer& container) const;
        bool match(ConstYamlNodeRef node, InputParameterContainer& container,
            IO::Internal::MatchEntry& match_entry) const;
        void set_default_value(InputParameterContainer& container) const;
        void print(std::ostream& stream, std::size_t indent) const;
        void emit_metadata(ryml::NodeRef node) const;
        bool emit(YamlNodeRef node, const InputParameterContainer& container,
            const InputSpecEmitOptions& options) const;
      };

      struct AllOfSpec
      {
        GroupData data;
        std::vector<InputSpec> specs;

        void parse(ValueParser& parser, InputParameterContainer& container) const;
        bool match(ConstYamlNodeRef node, InputParameterContainer& container,
            IO::Internal::MatchEntry& match_entry) const;
        void set_default_value(InputParameterContainer& container) const;
        void print(std::ostream& stream, std::size_t indent) const;
        void emit_metadata(ryml::NodeRef node) const;
        bool emit(YamlNodeRef node, const InputParameterContainer& container,
            const InputSpecEmitOptions& options) const;
      };

      struct OneOfSpec
      {
        // A one_of spec is essentially an unnamed group with additional logic to ensure that
        // exactly one of the contained specs is present.
        GroupData data;
        std::vector<InputSpec> specs;

        //! This callback may be used to perform additional actions after parsing one of the specs.
        //! The index of the parsed spec as given inside #specs is passed as an argument.
        std::function<void(InputParameterContainer& container, std::size_t index)>
            on_parse_callback;

        void parse(ValueParser& parser, InputParameterContainer& container) const;

        bool match(ConstYamlNodeRef node, InputParameterContainer& container,
            IO::Internal::MatchEntry& match_entry) const;

        void set_default_value(InputParameterContainer& container) const;

        void print(std::ostream& stream, std::size_t indent) const;

        void emit_metadata(ryml::NodeRef node) const;
        bool emit(YamlNodeRef node, const InputParameterContainer& container,
            const InputSpecEmitOptions& options) const;
      };

      struct ListSpec
      {
        //! The name of the list.
        std::string name;
        //! The spec that fits the list elements.
        InputSpec spec;

        ListData data;

        void parse(ValueParser& parser, InputParameterContainer& container) const;
        bool match(ConstYamlNodeRef node, InputParameterContainer& container,
            IO::Internal::MatchEntry& match_entry) const;
        void set_default_value(InputParameterContainer& container) const;
        void print(std::ostream& stream, std::size_t indent) const;
        void emit_metadata(ryml::NodeRef node) const;
        bool emit(YamlNodeRef node, const InputParameterContainer& container,
            const InputSpecEmitOptions& options) const;
      };


      //! Helper to create selection() specs.
      //! Note that the type can be anything since we never read or write values of this type.
      template <typename T>
      [[nodiscard]] InputSpec selection_internal(std::string name,
          std::map<std::string, RemoveOptional<T>> choices, ParameterDataIn<T> data = {});


      struct SizeChecker
      {
        constexpr bool operator()(const auto& val, std::size_t* size_info) const { return true; }

        template <typename U>
        constexpr bool operator()(const std::vector<U>& v, std::size_t* size_info) const
        {
          return ((*size_info == dynamic_size) || (v.size() == *size_info)) &&
                 std::ranges::all_of(
                     v, [&](const auto& val) { return this->operator()(val, size_info + 1); });
        }

        template <typename U>
        constexpr bool operator()(const std::map<std::string, U>& m, std::size_t* size_info) const
        {
          return ((*size_info == dynamic_size) || (m.size() == *size_info)) &&
                 std::ranges::all_of(m,
                     [&](const auto& val) { return this->operator()(val.second, size_info + 1); });
        }
      };
    }  // namespace Internal

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
     * After parsing an InputSpec, the value of the parameter can be retrieved from an
     * InputParameterContainer. If `default_value` is set, the parameter is implicitly optional. In
     * this case, if the parameter is not present in the input file, the default value will be
     * stored in the container.
     *
     * When you decide how to set the `default_value` field or whether to make a parameter optional,
     * consider the following cases:
     *
     *   - If you always require a parameter and there is no reasonable default value, do not set
     *     a `default_value`. This will make the parameter required. Parsing will fail if the
     *     parameter is not present, but after parsing you can be sure that the parameter can safely
     *     be retrieved with InputParameterContainer::get(). A good example is the time step size in
     *     a time integration scheme: this parameter is always required and taking an arbitrary
     *     default value is not a good idea.
     *
     *   - Is there a reasonable default value for the parameter, which works in most situations? If
     *     yes, set `default_value` to this value. This guarantees that you can always read the
     *     value from the container with InputParameterContainer::get(). A good example is a
     *     parameter that activates or deactivates a feature, e.g., defaulting the EAS element
     *     technology to off might be reasonable.
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
     * A parameter whose value is a selection from a list of choices. For example:
     *
     * @code
     * selection<int>("my_selection", {{"a", 1}, {"b", 2}, {"c", 3}});
     * @endcode
     *
     * The choices are given as a map from string to stored type T. This function is for
     * convenience, as you do not need to convert parsed string values to another type yourself. A
     * frequent use case is to map strings to enum constants.
     *
     * The remaining parameterization options follow the same rules as for the parameter() function.
     *
     * @note If you want to store the choices as strings and not map them to another type, use the
     * other selection() function.
     *
     * @relatedalso InputSpec
     */
    template <typename T>
      requires(!std::same_as<T, std::string>)
    [[nodiscard]] InputSpec selection(std::string name,
        std::map<std::string, RemoveOptional<T>> choices, ParameterDataIn<T> data = {});


    /**
     * Like the other selection() function, but the choices are stored as strings and not mapped
     * to another type.
     *
     * @note Although this function only works with strings, you still need to provide a type for
     * the first template parameter for consistency with the other functions.
     *
     * @relatedalso InputSpec
     */
    template <std::same_as<std::string> T>
    [[nodiscard]] InputSpec selection(
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
     * Whether a group can have a default value is determined by the default values of its children.
     * If all of its children have default values, the group can have a default value. In this case,
     * you may set the `defaultable` option to guarantee that the default values of the children are
     * stored under the group name in the container, even if the group is not present in the input.
     * Obviously, this only makes sense if the group is not required.
     * This behavior is analogous to the behavior of the parameter() function. While parameters with
     * default values are often a good idea (if the default value is meaningful), groups with
     * default values are less common. If you follow the advice to use groups to activate features,
     * you will usually have required parameters in the group and, consequently, the group cannot be
     * `defaultable`. Put differently, if you have a group with many default values, you are likely
     * not using the InputSpecs to their full potential. Consider splitting the group into multiple
     * smaller groups or use the one_of() function to select between different groups.
     *
     * @note If you want to group multiple InputSpecs without creating a new named scope, use the
     * all_of() function.
     *
     * @relatedalso InputSpec
     */
    [[nodiscard]] InputSpec group(
        std::string name, std::vector<InputSpec> specs, GroupData data = {});

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
void Core::IO::InputSpecBuilders::Internal::ParameterSpec<T>::parse(
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
    FOUR_C_THROW("Could not parse '%s'. Next token is '%s'.", name.c_str(), next_token.c_str());
  }

  if constexpr (rank<T>() == 0)
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
      int operator()(const SizeCallback& callback) const { return callback(container); }
      InputParameterContainer& container;
    };

    if constexpr (rank<T>() > 0)
    {
      std::array<std::size_t, rank<T>()> size_info;
      for (std::size_t i = 0; i < rank<T>(); ++i)
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
bool Core::IO::InputSpecBuilders::Internal::ParameterSpec<T>::match(ConstYamlNodeRef node,
    InputParameterContainer& container, IO::Internal::MatchEntry& match_entry) const
{
  match_entry.type = IO::Internal::MatchEntry::Type::parameter;
  auto spec_name = ryml::to_csubstr(name);

  // If we are not even in a map, we refuse to do anything and let the MatchTree handle this case.
  // Setting a default would confuse the user, since something fundamental must be wrong in the
  // input file.
  if (!node.node.is_map()) return false;

  if (!node.node.has_child(spec_name))
  {
    // It is OK to not encounter an optional parameter
    if (data.default_value.index() == 1)
    {
      container.add(name, std::get<1>(data.default_value));
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

  FOUR_C_ASSERT(entry_node.node.key() == name, "Internal error.");

  try
  {
    T value;
    read_value_from_yaml(entry_node, value);
    // Perform validation of the value here if necessary. Currently, we only validate sizes.
    if constexpr (rank<T>() > 0)
    {
      if (!has_correct_size(value, container))
      {
        return false;
      }
    }
    container.add(name, value);
    match_entry.state = IO::Internal::MatchEntry::State::matched;
    match_entry.matched_node = entry_node.node.id();
  }
  catch (const Core::Exception& e)
  {
    return false;
  }
  return true;
}


template <Core::IO::SupportedType T>
void Core::IO::InputSpecBuilders::Internal::ParameterSpec<T>::emit_metadata(
    ryml::NodeRef node) const
{
  node |= ryml::MAP;
  node["name"] << name;

  if constexpr (rank<T>() == 0)
    IO::Internal::emit_type_as_yaml<StoredType>(node);
  else
  {
    struct DynamicSizeVisitor
    {
      int operator()(int size) const { return size; }
      int operator()(const SizeCallback& callback) const { return dynamic_size; }
    };

    std::array<std::size_t, rank<T>()> size_info;
    for (std::size_t i = 0; i < rank<T>(); ++i)
    {
      size_info[i] = std::visit(DynamicSizeVisitor{}, data.size[i]);
    }
    IO::Internal::emit_type_as_yaml<StoredType>(node, size_info);
  }

  if (!data.description.empty())
  {
    emit_value_as_yaml(node["description"], data.description);
  }
  emit_value_as_yaml(node["required"], !(data.default_value.index() == 1));
  if (data.default_value.index() == 1)
  {
    emit_value_as_yaml(node["default"], std::get<1>(data.default_value));
  }
}

template <Core::IO::SupportedType T>
bool Core::IO::InputSpecBuilders::Internal::ParameterSpec<T>::emit(YamlNodeRef node,
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
      emit_value_as_yaml(value_node, *value);
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
      emit_value_as_yaml(value_node, std::get<1>(data.default_value));
    }
    return true;
  }
  else
  {
    return false;
  }
}


template <Core::IO::SupportedType T>
bool Core::IO::InputSpecBuilders::Internal::ParameterSpec<T>::has_correct_size(
    const T& val, const InputParameterContainer& container) const
{
  if constexpr (rank<T>() == 0)
  {
    return true;
  }
  else
  {
    struct SizeVisitor
    {
      int operator()(int size) const { return size; }
      int operator()(const SizeCallback& callback) const
      {
        // For yaml, we do not validate the size based on the callback. This feature can be removed
        // once we remove dat.
        return dynamic_size;
      }
      const InputParameterContainer& container;
    };

    std::array<std::size_t, rank<T>()> size_info;
    for (std::size_t i = 0; i < rank<T>(); ++i)
    {
      size_info[i] = std::visit(SizeVisitor{container}, data.size[i]);
    }
    SizeChecker size_checker;
    return size_checker(val, size_info.data());
  }
}


template <typename T>
void Core::IO::InputSpecBuilders::Internal::SelectionSpec<T>::parse(
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
    FOUR_C_THROW("Could not parse '%s'. Next token is '%s'.", name.c_str(), next_token.c_str());
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

  FOUR_C_THROW("Could not parse parameter '%s': invalid value '%s'. Valid options are: %s",
      name.c_str(), parsed_value_str.str().c_str(), choices_string.c_str());
}


template <typename T>
bool Core::IO::InputSpecBuilders::Internal::SelectionSpec<T>::match(ConstYamlNodeRef node,
    InputParameterContainer& container, IO::Internal::MatchEntry& match_entry) const
{
  match_entry.type = IO::Internal::MatchEntry::Type::parameter;
  auto spec_name = ryml::to_csubstr(name);

  // If we are not even in a map, we refuse to do anything and let the MatchTree handle this case.
  // Setting a default would confuse the user, since something fundamental must be wrong in the
  // input file.
  if (!node.node.is_map()) return false;

  if (!node.node.has_child(spec_name))
  {
    // It is OK to not encounter an optional parameter
    if (data.default_value.index() == 1)
    {
      container.add(name, std::get<1>(data.default_value));
      match_entry.state = IO::Internal::MatchEntry::State::defaulted;
      return true;
    }
    else
    {
      return false;
    }
  }

  auto entry_node = node.wrap(node.node[spec_name]);

  FOUR_C_ASSERT(entry_node.node.key() == name, "Internal error.");

  try
  {
    InputType value;
    read_value_from_yaml(entry_node, value);

    for (const auto& choice : choices)
    {
      if (choice.first == value)
      {
        container.add(name, choice.second);
        match_entry.state = IO::Internal::MatchEntry::State::matched;
        match_entry.matched_node = entry_node.node.id();
        if (data.on_parse_callback) data.on_parse_callback(container);
        return true;
      }
    }
  }
  catch (const std::exception& e)
  {
    return false;
  }

  return false;
}

template <typename T>
void Core::IO::InputSpecBuilders::Internal::SelectionSpec<T>::print(
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
void Core::IO::InputSpecBuilders::Internal::SelectionSpec<T>::emit_metadata(
    ryml::NodeRef node) const
{
  node |= ryml::MAP;
  node["name"] << name;

  if constexpr (OptionalType<T>) emit_value_as_yaml(node["noneable"], true);
  node["type"] = "enum";

  if (!data.description.empty())
  {
    emit_value_as_yaml(node["description"], data.description);
  }
  emit_value_as_yaml(node["required"], !(data.default_value.index() == 1));
  if (data.default_value.index() == 1)
  {
    // Find the choice that corresponds to the default value.
    auto default_value_it = std::find_if(choices.begin(), choices.end(),
        [&](const auto& choice) { return choice.second == std::get<1>(data.default_value); });
    FOUR_C_ASSERT(
        default_value_it != choices.end(), "Internal error: default value not found in choices.");
    emit_value_as_yaml(node["default"], default_value_it->first);
  }
  node["choices"] |= ryml::SEQ;
  for (const auto& [choice_string, _] : choices)
  {
    auto entry = node["choices"].append_child();
    // Write every choice entry as a map to easily extend the information at a later point.
    entry |= ryml::MAP;
    emit_value_as_yaml(entry["name"], choice_string);
  }
}

template <typename T>
bool Core::IO::InputSpecBuilders::Internal::SelectionSpec<T>::emit(YamlNodeRef node,
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
        emit_value_as_yaml(value_node, choice.first);
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
auto Core::IO::InputSpecBuilders::from_parameter(const std::string& name)
{
  return [name](const InputParameterContainer& container) -> T
  {
    const T* val = container.get_if<T>(name);
    FOUR_C_ASSERT_ALWAYS(val, "Parameter '%s' not found in container.", name.c_str());
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
  if constexpr (rank<T>() == 1)
  {
    internal_data.size[0] = data.size;
  }
  else if constexpr (rank<T>() > 1)
  {
    internal_data.size = data.size;
  }
  internal_data.on_parse_callback = data.on_parse_callback;

  return IO::Internal::make_spec(Internal::ParameterSpec<T>{.name = name, .data = internal_data},
      {
          .name = name,
          .description = data.description,
          .required = !(internal_data.default_value.index() == 1),
          .has_default_value = internal_data.default_value.index() == 1,
          .n_specs = 1,
      });
}


template <typename T>
Core::IO::InputSpec Core::IO::InputSpecBuilders::Internal::selection_internal(
    std::string name, std::map<std::string, RemoveOptional<T>> choices, ParameterDataIn<T> data)
{
  FOUR_C_ASSERT_ALWAYS(!choices.empty(), "Selection must have at least one choice.");

  // If we have a std::optional type, we need to convert the choices.
  typename SelectionSpec<T>::ChoiceMap modified_choices;
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

  SelectionData<T> internal_data;
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
      FOUR_C_THROW("Default value '%s' of selection not found in choices '%s'.",
          default_value_stream.str().c_str(), choices_string.c_str());
    }
  }

  return IO::Internal::make_spec(
      Internal::SelectionSpec<T>{
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
      });
}


template <typename T>
  requires(!std::same_as<T, std::string>)
Core::IO::InputSpec Core::IO::InputSpecBuilders::selection(
    std::string name, std::map<std::string, RemoveOptional<T>> choices, ParameterDataIn<T> data)
{
  return Internal::selection_internal(name, choices, data);
}


template <std::same_as<std::string> T>
Core::IO::InputSpec Core::IO::InputSpecBuilders::selection(
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

FOUR_C_NAMESPACE_CLOSE

#endif
