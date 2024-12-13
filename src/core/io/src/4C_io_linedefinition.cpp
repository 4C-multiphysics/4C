// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_linedefinition.hpp"

#include "4C_io_input_file.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_utils_exceptions.hpp"

#include <functional>
#include <iterator>
#include <memory>
#include <utility>

FOUR_C_NAMESPACE_OPEN

// Note on the namespace: while implementation details can often be put into an anonymous namespace,
// this is trickier when these details are part of a forward declared implementation class (in this
// case LineDefinitionImplementation). Compiling this file multiple times would then lead to
// different implementations of that class because the anonymous namespace is always unique.
// Compilers do not always warn about this and this is not guaranteed to be a problem (after all,
// why would we compile the class twice?). Here we encountered the warning in unity builds, so this
// is not an anonymous namespace but the INTERNAL one.
namespace Input::Internal
{
  template <class T>
  std::string string_from_data_type()
  {
    if constexpr (std::is_same_v<T, int>)
      return "integer";
    else if constexpr (std::is_same_v<T, double>)
      return "double";
    else if constexpr (std::is_same_v<T, std::string>)
      return "string";
    else
    {
      // Trick to use static_assert in constexpr: make the expression dependent on template
      // parameter. In this case, the assert always fails if the type is not included in the cases
      // above.
      static_assert(!sizeof(T), "No match for given type.");
    }
  }


  //! Decide on behavior of the component name when reading or printing.
  enum class Behavior
  {
    // The name is printed to and read from lines.
    read_print_name,
    // The name is only used to name a component internally and not part of the input line.
    ignore_name,
  };


  /**
   * Visitor to read a ComponentValue from an input stream.
   */
  struct ReadValue
  {
    template <typename T>
    bool operator()(T& value)
    {
      in >> value;

      check_stream_for_unparsed_characters(string_from_data_type<T>());
      return !in.fail();
    }

    template <typename T>
    bool operator()(std::vector<T>& values)
    {
      for (auto& v : values)
      {
        in >> v;
      }

      check_stream_for_unparsed_characters(string_from_data_type<T>());
      return !in.fail();
    }

    template <typename T1, typename T2>
    bool operator()(std::vector<std::pair<T1, T2>>& values)
    {
      for (auto& [a, b] : values)
      {
        in >> a;
        check_stream_for_unparsed_characters(string_from_data_type<T1>());

        in >> b;
        check_stream_for_unparsed_characters(string_from_data_type<T2>());
      }

      return !in.fail();
    }

    std::istream& in;

    //! Store the name to throw better errors.
    const std::string& name;

   private:
    void check_stream_for_unparsed_characters(const std::string& correctDataType)
    {
      if (!(in.eof()))  // otherwise peek writes a failbit
      {
        if ((in.peek() != ' ') && (in.peek() != '/'))  // exclude whitespaces and comments
        {
          FOUR_C_THROW(
              "Data type of component %s doesn't match the specified value. Needs to be of type %s",
              name.c_str(), correctDataType.c_str());
        }
      }
    }
  };


  /**
   * Visitor to print a ComponentValue.
   *
   * The call operator is overloaded on the type in ComponentValue since this type determines
   * whether and how printing happens.
   */
  struct PrintComponent
  {
    template <typename T>
    void operator()(const T& value)
    {
      if (behavior == Behavior::read_print_name) out << name << " ";
      out << value;
    }

    template <typename T>
    void operator()(const std::vector<T>& values)
    {
      if (behavior == Behavior::read_print_name) out << name << " ";
      std::copy(values.begin(), values.end(), std::ostream_iterator<T>(out, " "));
    }

    template <typename T1, typename T2>
    void operator()(const std::vector<std::pair<T1, T2>>& values)
    {
      if (behavior == Behavior::read_print_name) out << name << " ";
      for (const auto& [a, b] : values)
      {
        out << a << " " << b << " ";
      }
    }

    std::ostream& out;
    const Behavior& behavior;
    const std::string& name;
  };


  /**
   * A type-erased container for components of a LineDefinition.
   */
  class LineDefinitionComponent
  {
   private:
    //! The type-erased interface.
    class LineDefinitionComponentConcept
    {
     public:
      virtual ~LineDefinitionComponentConcept() = default;

      [[nodiscard]] virtual std::unique_ptr<LineDefinitionComponentConcept> clone() const = 0;

      virtual void print(std::ostream& stream) const = 0;

      virtual bool read_required(
          Core::IO::InputParameterContainer& container, std::istream& stream) = 0;

      virtual bool read_optional(Core::IO::InputParameterContainer& container, std::string& name,
          std::istream& stream) = 0;

      [[nodiscard]] virtual bool is_named(const std::string& name) const = 0;

      virtual void store_default(Core::IO::InputParameterContainer& container) const = 0;
    };

    //! The wrapper for a concrete type T compatible with the interface.
    //! Forward all calls to the concrete type T.
    template <typename T>
    class LineDefinitionComponentModel : public LineDefinitionComponentConcept
    {
     public:
      template <typename T2,
          // prevent matching a copy or move constructor
          typename = std::enable_if_t<
              !std::is_same_v<std::decay_t<T2>, LineDefinitionComponentModel<T>>, void>>
      LineDefinitionComponentModel(T2&& in) : component_(std::forward<T2>(in))
      {
      }

      [[nodiscard]] std::unique_ptr<LineDefinitionComponentConcept> clone() const override
      {
        return std::make_unique<LineDefinitionComponentModel<T>>(*this);
      }

      void print(std::ostream& stream) const override { component_.print(stream); }

      bool read_required(
          Core::IO::InputParameterContainer& container, std::istream& stream) override
      {
        return component_.read_required(container, stream);
      }

      bool read_optional(Core::IO::InputParameterContainer& container, std::string& name,
          std::istream& stream) override
      {
        return component_.read_optional(container, name, stream);
      }

      [[nodiscard]] bool is_named(const std::string& name) const override
      {
        return component_.is_named(name);
      }

      void store_default(Core::IO::InputParameterContainer& container) const override
      {
        component_.store_default(container);
      }

     private:
      T component_;
    };

   public:
    /**
     * Templated constructor creating a bridge from concrete type T to type-erase interface
     * LinearComponentConcept.
     */
    template <typename T,
        // prevent matching a copy or move constructor
        typename =
            std::enable_if_t<!std::is_same_v<std::decay_t<T>, LineDefinitionComponent>, void>>
    LineDefinitionComponent(T&& in)
        : pimpl_(std::make_unique<LineDefinitionComponentModel<T>>(std::forward<T>(in)))
    {
    }

    //! Copy constructor.
    LineDefinitionComponent(const LineDefinitionComponent& other) : pimpl_(other.pimpl_->clone()) {}

    //! Copy assignment.
    LineDefinitionComponent& operator=(const LineDefinitionComponent& other)
    {
      this->pimpl_ = other.pimpl_->clone();
      return *this;
    }

    //! Move constructor.
    LineDefinitionComponent(LineDefinitionComponent&& other) = default;

    //! Move assignment.
    LineDefinitionComponent& operator=(LineDefinitionComponent&& other) = default;

    /// print to a dat file comment
    void print(std::ostream& stream) const { pimpl_->print(stream); }

    /// Try to read component from input line
    /// This function is called for required components.
    bool read_required(Core::IO::InputParameterContainer& container, std::istream& stream)
    {
      return pimpl_->read_required(container, stream);
    }

    /// Try to read component from input line
    /// This function is called for optional components.
    bool read_optional(
        Core::IO::InputParameterContainer& container, std::string& name, std::istream& stream)
    {
      return pimpl_->read_optional(container, name, stream);
    }

    /// tell if the component has the specified name tag
    [[nodiscard]] bool is_named(const std::string& name) const { return pimpl_->is_named(name); }

    /// store the default value in the container
    void store_default(Core::IO::InputParameterContainer& container) const
    {
      return pimpl_->store_default(container);
    }

   private:
    std::unique_ptr<LineDefinitionComponentConcept> pimpl_;
  };


  /**
   * Component that works for various primitive types and STL containers.
   */
  template <typename T>
  class GenericComponent
  {
   public:
    GenericComponent(std::string name, T value, Behavior behavior = Behavior::read_print_name,
        std::function<void(Core::IO::InputParameterContainer&, T& value)> value_prepare = {})
        : name_(std::move(name)),
          behavior_(behavior),
          value_prepare_(std::move(value_prepare)),
          default_value_(std::move(value))
    {
    }

    void print(std::ostream& stream) const
    {
      if (value_prepare_)
      {
        // Assume that the custom parsing logic is not printable. Print a placeholder instead.
        if (behavior_ == Behavior::read_print_name) stream << this->name_ << " ";
        stream << "[...]";
      }
      else
        PrintComponent{stream, behavior_, name_}(default_value_);
    }

    bool read_optional(
        Core::IO::InputParameterContainer& container, std::string& name, std::istream& stream)
    {
      if (name != name_) return false;

      T value = default_value_;

      if (value_prepare_) value_prepare_(container, value);

      bool read_succeeded = ReadValue{stream, name_}(value);

      if (read_succeeded) container.add(name, value);

      return read_succeeded;
    }

    bool read_required(Core::IO::InputParameterContainer& container, std::istream& stream)
    {
      if (behavior_ == Behavior::read_print_name)
      {
        std::string name;
        stream >> name;
        if (name != name_) return false;
      }
      return read_optional(container, name_, stream);
    }

    [[nodiscard]] bool is_named(const std::string& name) const { return name == name_; }

    void store_default(Core::IO::InputParameterContainer& container) const
    {
      // In the current state, GenericComponents do not store a name and the default value for
      // non-existent optional components. For optional components that are non-existent in the
      // input stream, the name key does not exist in the container.
    }

   private:
    //! Name of the component.
    std::string name_;

    //! Store how the component behaves when reading or writing.
    Behavior behavior_{Behavior::read_print_name};

    //! An optional function that determines how to parse the value. This is useful to
    //! do prelimiary work, like querying already read data.
    std::function<void(Core::IO::InputParameterContainer&, T&)> value_prepare_;

    T default_value_;
  };


  /**
   * Tag component that behaves like a boolean: an existing tag is treated as a true value
   */
  class TagComponent
  {
   public:
    TagComponent(std::string name) : name_(std::move(name)) {}

    void print(std::ostream& stream) const { stream << name_; }

    bool read_optional(
        Core::IO::InputParameterContainer& container, std::string& name, std::istream& stream)
    {
      if (name != name_) return false;

      // if we read a tag means we can store true
      container.add(name, true);

      return true;
    }

    bool read_required(Core::IO::InputParameterContainer& container, std::istream& stream)
    {
      std::string name;
      stream >> name;
      if (name != name_) return false;
      return read_optional(container, name_, stream);
    }

    [[nodiscard]] bool is_named(const std::string& name) const { return name == name_; }

    void store_default(Core::IO::InputParameterContainer& container) const
    {
      // if we did not read an optional tag, we store the default false
      container.add(name_, false);
    }

   private:
    //! Name of the component.
    std::string name_;
  };

}  // namespace Input::Internal


namespace Input
{
  namespace Internal
  {
    /**
     * Internal data used in the implementation. This type has value semantics.
     */
    class LineDefinitionImplementation
    {
     public:
      /// Gather all added required components.
      std::vector<LineDefinitionComponent> components_;

      /// Gather all added optional components.
      std::unordered_map<std::string, LineDefinitionComponent> optionaltail_;

      /// Store which optional components have been read.
      std::set<std::string> readtailcomponents_;

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
    pimpl_->components_.emplace_back(Internal::TagComponent{std::move(name)});
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_string(std::string name)
  {
    pimpl_->components_.emplace_back(
        Internal::GenericComponent<std::string>{std::move(name), "''"});
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_int(std::string name)
  {
    pimpl_->components_.emplace_back(Internal::GenericComponent<int>{std::move(name), 0});
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_int_vector(
      std::string name, int length)
  {
    pimpl_->components_.emplace_back(
        Internal::GenericComponent(std::move(name), std::vector<int>(length)));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_double(std::string name)
  {
    pimpl_->components_.emplace_back(Internal::GenericComponent<double>{std::move(name), 0.0});
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_double_vector(
      std::string name, int length)
  {
    pimpl_->components_.emplace_back(
        Internal::GenericComponent(std::move(name), std::vector<double>(length)));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_string_vector(
      std::string name, int length)
  {
    pimpl_->components_.emplace_back(
        Internal::GenericComponent(std::move(name), std::vector<std::string>(length)));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_named_double_vector(
      std::string name, LengthDefinition length_definition)
  {
    pimpl_->components_.emplace_back(Internal::GenericComponent<std::vector<double>>(name,
        std::vector<double>{}, Internal::Behavior::read_print_name,
        [lengthdef = std::move(length_definition)](
            Core::IO::InputParameterContainer& linedef, std::vector<double>& values)
        {
          // Find expected vector on line. It has to be read already!
          const std::size_t length = lengthdef(linedef);
          values.resize(length);
        }));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_tag(const std::string& name)
  {
    if (pimpl_->optionaltail_.find(name) != pimpl_->optionaltail_.end())
      FOUR_C_THROW("optional component '%s' already defined", name.c_str());
    pimpl_->optionaltail_.emplace(name, Internal::TagComponent{name});
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_string(
      const std::string& name)
  {
    if (pimpl_->optionaltail_.find(name) != pimpl_->optionaltail_.end())
      FOUR_C_THROW("optional component '%s' already defined", name.c_str());
    pimpl_->optionaltail_.emplace(name, Internal::GenericComponent<std::string>{name, "''"});
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_int(const std::string& name)
  {
    if (pimpl_->optionaltail_.find(name) != pimpl_->optionaltail_.end())
      FOUR_C_THROW("optional component '%s' already defined", name.c_str());
    pimpl_->optionaltail_.emplace(name, Internal::GenericComponent<int>{name, 0});
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_int_vector(
      const std::string& name, int length)
  {
    if (pimpl_->optionaltail_.find(name) != pimpl_->optionaltail_.end())
      FOUR_C_THROW("optional component '%s' already defined", name.c_str());
    pimpl_->optionaltail_.emplace(name, Internal::GenericComponent(name, std::vector<int>(length)));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_double(
      const std::string& name)
  {
    if (pimpl_->optionaltail_.find(name) != pimpl_->optionaltail_.end())
      FOUR_C_THROW("optional component '%s' already defined", name.c_str());
    pimpl_->optionaltail_.emplace(name, Internal::GenericComponent<double>{name, 0.});
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_double_vector(
      const std::string& name, int length)
  {
    if (pimpl_->optionaltail_.find(name) != pimpl_->optionaltail_.end())
      FOUR_C_THROW("optional component '%s' already defined", name.c_str());
    pimpl_->optionaltail_.emplace(
        name, Internal::GenericComponent(name, std::vector<double>(length)));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_double_vector(
      const std::string& name, LengthDefinition lengthdef)
  {
    if (pimpl_->optionaltail_.find(name) != pimpl_->optionaltail_.end())
      FOUR_C_THROW("optional component '%s' already defined", name.c_str());

    pimpl_->optionaltail_.emplace(
        name, Internal::GenericComponent<std::vector<double>>(name, std::vector<double>{},
                  Internal::Behavior::read_print_name,
                  [lengthdef = std::move(lengthdef)](
                      Core::IO::InputParameterContainer& linedef, std::vector<double>& values)
                  {
                    // Find expected vector on line. It has to be read already!
                    const std::size_t length = lengthdef(linedef);
                    values.resize(length);
                  }));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_string_vector(
      const std::string& name, int length)
  {
    if (pimpl_->optionaltail_.find(name) != pimpl_->optionaltail_.end())
      FOUR_C_THROW("optional component '%s' already defined", name.c_str());
    pimpl_->optionaltail_.emplace(
        name, Internal::GenericComponent(name, std::vector<std::string>(length, "''")));
    return *this;
  }



  LineDefinition::Builder& LineDefinition::Builder::add_optional_named_string_vector(
      const std::string& name, LengthDefinition lengthdef)
  {
    if (pimpl_->optionaltail_.find(name) != pimpl_->optionaltail_.end())
      FOUR_C_THROW("optional component '%s' already defined", name.c_str());
    using T = std::vector<std::string>;

    pimpl_->optionaltail_.emplace(
        name, Internal::GenericComponent<T>(name, T{}, Internal::Behavior::read_print_name,
                  [lengthdef = std::move(lengthdef)](
                      Core::IO::InputParameterContainer& linedef, T& values)
                  {
                    // Find expected vector on line. It has to be read already!
                    const std::size_t length = lengthdef(linedef);
                    values.resize(length);
                  }));
    return *this;
  }



  LineDefinition::Builder&
  LineDefinition::Builder::add_optional_named_pair_of_string_and_double_vector(
      const std::string& name, LengthDefinition lengthdef)
  {
    if (pimpl_->optionaltail_.find(name) != pimpl_->optionaltail_.end())
      FOUR_C_THROW("optional component '%s' already defined", name.c_str());
    using T = std::vector<std::pair<std::string, double>>;

    pimpl_->optionaltail_.emplace(
        name, Internal::GenericComponent<T>(name, T{}, Internal::Behavior::read_print_name,
                  [lengthdef = std::move(lengthdef)](
                      Core::IO::InputParameterContainer& linedef, T& values)
                  {
                    // Find expected vector on line. It has to be read already!
                    const std::size_t length = lengthdef(linedef);
                    values.resize(length);
                  }));
    return *this;
  }



  void LineDefinition::print(std::ostream& stream) const
  {
    for (const auto& component : pimpl_->components_)
    {
      component.print(stream);
      stream << ' ';
    }
    if (!pimpl_->optionaltail_.empty())
    {
      stream << "[ ";
      for (const auto& [_, component] : pimpl_->optionaltail_)
      {
        component.print(stream);
        stream << ' ';
      }
      stream << "] ";
    }
  }



  std::optional<Core::IO::InputParameterContainer> LineDefinition::read(std::istream& stream)
  {
    pimpl_->container_ = Core::IO::InputParameterContainer();

    pimpl_->readtailcomponents_.clear();
    for (auto& component : pimpl_->components_)
    {
      if (not component.read_required(pimpl_->container_, stream))
      {
        return std::nullopt;
      }
    }

    // we expect as many optional components as are defined (or less)
    for (unsigned a = 0; a < pimpl_->optionaltail_.size(); ++a)
    {
      std::string name;
      stream >> name;
      if (not stream) break;
      if (pimpl_->readtailcomponents_.find(name) != pimpl_->readtailcomponents_.end())
      {
        // duplicated optional component
        return std::nullopt;
      }
      auto i = pimpl_->optionaltail_.find(name);
      if (i == pimpl_->optionaltail_.end()) return std::nullopt;
      if (not i->second.read_optional(pimpl_->container_, name, stream)) return std::nullopt;
      pimpl_->readtailcomponents_.insert(name);
    }

    // for any optional added TagComponents that have not been read, store the default value in the
    // container
    for (const auto& [name, component] : pimpl_->optionaltail_)
    {
      if (pimpl_->readtailcomponents_.count(name) == 0)
      {
        component.store_default(pimpl_->container_);
      }
    }

    // check if any other unused strings except from comments and whitespaces are given
    std::string superfluousstring;
    stream >> superfluousstring;  // stream strips whitespaces


    // Check that remaining string is either empty or is a comment, i.e., starts with "//"
    if (!superfluousstring.empty() && (superfluousstring.rfind("//", 0) != 0))
      return std::nullopt;
    else
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
