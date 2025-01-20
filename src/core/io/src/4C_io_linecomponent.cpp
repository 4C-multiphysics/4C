// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_linecomponent.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_utils_exceptions.hpp"

#include <iterator>
#include <optional>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace
{
  /*! @brief Convert the string @p snumber to number of type T and return that value number via
   * stoi or stod depending on whether T is type int or double.
   *
   * @param[in] position of the next character in string after the numerical value
   */
  template <typename T>
  T convert_string_to_number(const std::string& snumber, std::size_t& pos);

  /// specialization for int
  template <>
  inline int convert_string_to_number<int>(const std::string& snumber, std::size_t& pos)
  {
    return stoi(snumber, &pos);
  }

  /// specialization for double
  template <>
  inline double convert_string_to_number<double>(const std::string& snumber, std::size_t& pos)
  {
    return stod(snumber, &pos);
  }

  // Throw an error for the wrong data type in case 'number' is of type int
  void throw_error_wrong_data_type(const std::string& snumbersubstring, int number,
      const std::string& variablename, const std::string& sectionname)
  {
    FOUR_C_THROW(
        "Failed to read value '%s' while reading variable '%s' in '%s'. 4C could only read "
        "'%d', so the specified number format is probably not supported. The variable '%s' "
        "has to be an integer.",
        snumbersubstring.c_str(), variablename.c_str(), sectionname.c_str(), number,
        variablename.c_str());
  }

  // Throw an error for the wrong data type in case 'number' is of type double
  void throw_error_wrong_data_type(const std::string& snumbersubstring, double number,
      const std::string& variablename, const std::string& sectionname)
  {
    FOUR_C_THROW(
        "Failed to read value '%s' while reading variable '%s' in '%s'. 4C could only read "
        "'%f', so the specified number format is probably not supported. The variable '%s' "
        "has to be a floating point.",
        snumbersubstring.c_str(), variablename.c_str(), sectionname.c_str(), number,
        variablename.c_str());
  }

  // Convert a string to a number, i.e. to an int or a double
  // Perform the appropriate error checks
  template <typename T>
  T convert_and_validate_string_to_number(const std::string& snumber,
      const std::string& variablename, const std::string& sectionname, int variablelength,
      bool optional)
  {
    // value is set by the function stoi or stod to position of the next character in str after the
    // numerical value. Needed to check for remaining characters after string to int or double
    // conversion.
    T number;
    std::size_t pos = 0;

    try
    {
      // convert to int or double, depending on type T
      number = convert_string_to_number<T>(snumber, pos);
    }
    catch (std::invalid_argument& e)
    {
      // in case the parameter is mandatory and no value is given
      if (!optional and snumber.empty())
      {
        FOUR_C_THROW(
            "Invalid argument! No value of variable '%s' in '%s' specified. Possibly you "
            "didn't give enough input values. The variable '%s' expects %i input values.",
            variablename.c_str(), sectionname.c_str(), variablename.c_str(), variablelength);
      }
      // any other weird input values
      else
      {
        FOUR_C_THROW("Invalid argument! Failed to read the value '%s' of variable '%s' in '%s'.",
            snumber.c_str(), variablename.c_str(), sectionname.c_str(), variablename.c_str());
      }
    }
    // check if there are any other characters that were not converted
    if (pos != snumber.size())
    {
      throw_error_wrong_data_type(snumber.substr(pos), number, variablename, sectionname);
    }

    return number;
  }
}  // namespace

namespace Input
{

  LineComponent::LineComponent(std::string name, bool optional)
      : optional_(optional), name_(std::move(name))
  {
  }


  SeparatorComponent::SeparatorComponent(
      std::string separator, std::string description, bool optional)
      : Input::LineComponent("*SEPARATOR*", optional),
        separator_(std::move(separator)),
        description_(std::move(description))
  {
  }

  void SeparatorComponent::default_line(std::ostream& stream) { stream << separator_; }

  void SeparatorComponent::describe(std::ostream& stream)
  {
    stream << "    " << std::setw(15) << std::left << separator_ << std::setw(15) << std::left
           << (optional_ ? "(optional)" : "") << description_;
  }

  std::string SeparatorComponent::write_read_the_docs() { return separator_; }

  std::vector<std::string> SeparatorComponent::write_read_the_docs_table_row() const
  {
    std::vector<std::string> tablerow;

    tablerow.push_back(separator_);
    tablerow.emplace_back((optional_ ? "yes" : ""));
    std::string descriptionstring = "";
    tablerow.push_back(description_);
    return tablerow;
  }

  std::shared_ptr<std::stringstream> SeparatorComponent::read(const std::string& section_name,
      std::shared_ptr<std::stringstream> condline, Core::IO::InputParameterContainer& container)
  {
    // try to find line parameter label "separator_" (with leading and trailing white spaces for
    // uniqueness) in stringstream "condline"
    size_t position = condline->str().find(" " + separator_ + " ");

    // case: line parameter label "separator_" not found
    if (position == std::string::npos)
    {
      if (optional_)
        // move stringstream position to end of "condline" in case of optional line parameter
        condline->seekg(0, condline->end);
      else
        // return error in case a required line parameter is not specified
        FOUR_C_THROW("Required parameter '%s' for section '%s' not specified in input file!",
            separator_.c_str(), section_name.c_str());
    }
    // case: found line parameter label "separator_"
    else
    {
      // care for leading white space in search string ("position" should indicate position of first
      // actual character of parameter label)
      position++;

      // remove line parameter label "separator_" from stringstream "condline"
      condline->str(condline->str().erase(position, separator_.size()));

      // set current position in stringstream "condline" in front of value associated with line
      // parameter label "separator_"
      condline->seekg((std::streampos)position);
    }

    return condline;
  }


  StringComponent::StringComponent(std::string name, std::string defaultvalue, bool optional)
      : Input::LineComponent(std::move(name), optional), defaultvalue_(std::move(defaultvalue))
  {
  }

  void StringComponent::default_line(std::ostream& stream) { stream << defaultvalue_; }

  void StringComponent::describe(std::ostream& stream) {}

  std::shared_ptr<std::stringstream> StringComponent::read(const std::string& section_name,
      std::shared_ptr<std::stringstream> condline, Core::IO::InputParameterContainer& container)
  {
    // initialize string parameter value to be read
    std::string str = defaultvalue_;

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read string parameter value in case the associated parameter label appears in
    // line line of input file
    if ((size_t)position != condline->str().size())
    {
      // extract string parameter value from stringstream "condline"
      *condline >> str;

      // return error in case the extraction was not successful
      if (str.empty())
        FOUR_C_THROW(
            "Value of parameter '%s' for section '%s' not properly specified in input file!",
            name().c_str(), section_name.c_str());

      // remove string parameter value from stringstream "condline"
      condline->str(condline->str().erase((size_t)condline->tellg() - str.size(), str.size()));

      // reset current position in stringstream "condline"
      condline->seekg(position);
    }

    // add double parameter value to line parameter list
    container.add(name(), str);

    return condline;
  }


  SelectionComponent::SelectionComponent(std::string name, std::string defaultvalue,
      const Teuchos::Array<std::string>& datfilevalues,
      const Teuchos::Array<std::string>& stringcondvalues, bool optional)
      : Input::LineComponent(std::move(name)),
        defaultvalue_(std::move(defaultvalue)),
        datfilevalues_(datfilevalues),
        stringcondvalues_(stringcondvalues),
        intcondvalues_(Teuchos::tuple<int>(-1)),
        stringtostring_(true)
  {
    if (std::find(datfilevalues_.begin(), datfilevalues_.end(), defaultvalue_) ==
        datfilevalues_.end())
    {
      FOUR_C_THROW("Invalid default value '%s'.", defaultvalue_.c_str());
    }
    if (datfilevalues_.size() != stringcondvalues_.size())
    {
      FOUR_C_THROW("Input file values must match condition values.");
    }
  }

  SelectionComponent::SelectionComponent(std::string name, std::string defaultvalue,
      const Teuchos::Array<std::string>& datfilevalues, const Teuchos::Array<int>& intcondvalues,
      bool optional)
      : Input::LineComponent(std::move(name)),
        defaultvalue_(std::move(defaultvalue)),
        datfilevalues_(datfilevalues),
        stringcondvalues_(Teuchos::tuple<std::string>("notdefined")),
        intcondvalues_(intcondvalues),
        stringtostring_(false)
  {
    if (std::find(datfilevalues_.begin(), datfilevalues_.end(), defaultvalue_) ==
        datfilevalues_.end())
    {
      FOUR_C_THROW("Invalid default value '%s'.", defaultvalue_.c_str());
    }
    if (datfilevalues_.size() != intcondvalues_.size())
    {
      FOUR_C_THROW("Input file values must match condition values.");
    }
  }

  void SelectionComponent::default_line(std::ostream& stream) { stream << defaultvalue_; }

  std::string SelectionComponent::write_read_the_docs() { return "<" + name() + ">"; }

  Teuchos::Array<std::string> SelectionComponent::get_options() { return datfilevalues_; }

  std::shared_ptr<std::stringstream> SelectionComponent::read(const std::string& section_name,
      std::shared_ptr<std::stringstream> condline, Core::IO::InputParameterContainer& container)
  {
    std::size_t position{};
    std::string selected_value{defaultvalue_};
    for (unsigned i = 0; i < datfilevalues_.size(); ++i)
    {
      position = condline->str().find(" " + datfilevalues_[i] + " ");
      if (position != std::string::npos)
      {
        selected_value = datfilevalues_[i];
        break;
      }
    }

    {
      // care for leading white space in search string ("position" should indicate position of first
      // actual character of parameter label)
      position++;

      // Is this necessary?
      condline->str(condline->str().erase(position, selected_value.size()));

      // set current position in stringstream "condline" in front of value associated with line
      // parameter label "separator_"
      condline->seekg((std::streampos)position);
    }

    auto i = std::find(datfilevalues_.begin(), datfilevalues_.end(), selected_value);
    const unsigned pos = std::distance(datfilevalues_.begin(), i);
    // choose, if we have an array based on std::string or int
    if (stringtostring_)
      container.add(name(), stringcondvalues_[pos]);
    else
      container.add(name(), intcondvalues_[pos]);

    return condline;
  }


  IntComponent::IntComponent(std::string name, IntComponentData data)
      : Input::LineComponent(std::move(name), data.optional), data_(data)
  {
  }

  void IntComponent::default_line(std::ostream& stream)
  {
    if (data_.none_allowed)
      stream << "none";
    else
      stream << data_.default_value;
  }

  std::string IntComponent::write_read_the_docs()
  {
    return data_.none_allowed ? "none" : std::to_string(data_.default_value);
  }

  void IntComponent::describe(std::ostream& stream) {}

  std::shared_ptr<std::stringstream> IntComponent::read(const std::string& section_name,
      std::shared_ptr<std::stringstream> condline, Core::IO::InputParameterContainer& container)
  {
    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // consider two cases: either the data is noneable or not
    if (data_.none_allowed)
    {
      // initialize integer parameter value to be read
      Core::IO::Noneable<int> noneable_number = data_.default_value;

      // only try to read integer parameter value in case the associated parameter label appears in
      // line line of input file
      if ((size_t)position != condline->str().size())
      {
        // extract integer component as string
        std::string snumber;
        *condline >> snumber;
        if (!optional_ or !snumber.empty())
        {
          if (snumber == "none")
          {
            noneable_number = Core::IO::none<int>;
          }
          else
          {
            noneable_number = convert_and_validate_string_to_number<int>(
                snumber, name(), section_name, 1, optional_);
          }
        }

        // remove parameter value from stringstream "condline"
        condline->str(
            condline->str().erase((size_t)condline->tellg() - snumber.size(), snumber.size()));

        // reset current position in stringstream "condline"
        condline->seekg(position);
      }

      // add a nonable int parameter value to line parameter list
      container.add(name(), noneable_number);
    }
    else
    {
      // initialize integer parameter value to be read
      int number = data_.default_value;

      // only try to read integer parameter value in case the associated parameter label appears in
      // line line of input file
      if ((size_t)position != condline->str().size())
      {
        // extract integer vector component as string
        std::string snumber;
        *condline >> snumber;

        // if the value is required or the value is not empty
        if (!optional_ or !snumber.empty())
        {
          number = convert_and_validate_string_to_number<int>(
              snumber, name(), section_name, 1, optional_);
        }

        // remove parameter value from stringstream "condline"
        condline->str(
            condline->str().erase((size_t)condline->tellg() - snumber.size(), snumber.size()));

        // reset current position in stringstream "condline"
        condline->seekg(position);
      }

      // add int parameter value to line parameter list
      container.add(name(), number);
    }

    return condline;
  }


  IntVectorComponent::IntVectorComponent(std::string name, int length, IntComponentData data)
      : Input::LineComponent(std::move(name), data.optional), length_(length), data_(data)
  {
  }

  IntVectorComponent::IntVectorComponent(
      std::string name, LengthDefinition length_from_component, IntComponentData data)
      : Input::LineComponent(std::move(name)),
        length_(std::move(length_from_component)),
        data_(data)
  {
  }

  namespace
  {
    struct DefaultLengthVisitor
    {
      int operator()(int length) { return length; }
      int operator()(const LengthDefinition& length) { return 1; }
    };
  }  // namespace

  void IntVectorComponent::default_line(std::ostream& stream)
  {
    using namespace std::string_literals;
    const int default_length = std::visit(DefaultLengthVisitor{}, length_);
    const std::string default_value = std::invoke(
        [&]()
        {
          if (data_.none_allowed)
            return "none "s;
          else
            return std::to_string(data_.default_value) + " ";
        });

    for (int i = 0; i < default_length; ++i) stream << default_value;
  }

  std::string IntVectorComponent::write_read_the_docs()
  {
    std::string parameterstring = "<int vec";
    if (data_.none_allowed) parameterstring += " [incl none]";
    parameterstring += ":" + name() + "> ";
    return parameterstring;
  }

  namespace
  {
    struct LengthVisitor
    {
      int operator()(int length) { return length; }
      int operator()(const LengthDefinition& length) { return length(condition); }

      const Core::IO::InputParameterContainer& condition;
    };
  }  // namespace

  std::shared_ptr<std::stringstream> IntVectorComponent::read(const std::string& section_name,
      std::shared_ptr<std::stringstream> condline, Core::IO::InputParameterContainer& container)
  {
    const int initialize_value = data_.default_value;
    const int dynamic_length = std::visit(LengthVisitor{container}, length_);

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    if (data_.none_allowed)
    {
      // initialize integer parameter vector to be read
      std::vector<Core::IO::Noneable<int>> nnumbers(dynamic_length, initialize_value);

      // only try to read integer parameter vector in case the associated parameter label appears in
      // line line of input file
      if ((size_t)position != condline->str().size())
      {
        // extract integer parameter vector from stringstream "condline"
        for (auto& current_number : nnumbers)
        {
          // extract integer vector component as string
          std::string snumber;
          *condline >> snumber;

          // if the input is none
          if (snumber == "none")
          {
            current_number = Core::IO::none<int>;
          }
          // in case the parameter is optional and no value is given
          else if (optional_ and snumber.empty())
          {
            break;
          }
          // all other cases
          else
          {
            current_number = convert_and_validate_string_to_number<int>(
                snumber, name(), section_name, dynamic_length, optional_);
          }

          // remove parameter value from stringstream "condline"
          condline->str(
              condline->str().erase((size_t)condline->tellg() - snumber.size(), snumber.size()));

          // reset current position in stringstream "condline"
          condline->seekg(position);
        }
      }

      // add Noneable int parameter vector to line parameter list
      container.add(name(), nnumbers);
    }
    else
    {
      // initialize integer parameter vector to be read
      std::vector<int> nnumbers(dynamic_length, initialize_value);

      // only try to read integer parameter vector in case the associated parameter label appears in
      // line line of input file
      if ((size_t)position != condline->str().size())
      {
        // extract integer parameter vector from stringstream "condline"
        for (auto& current_number : nnumbers)
        {
          // extract integer vector component as string
          std::string snumber;
          *condline >> snumber;

          // in case the parameter is optional and no value is given
          if (optional_ and snumber.empty())
          {
            break;
          }
          // all other cases
          else
          {
            current_number = convert_and_validate_string_to_number<int>(
                snumber, name(), section_name, dynamic_length, optional_);
          }

          // remove parameter value from stringstream "condline"
          condline->str(
              condline->str().erase((size_t)condline->tellg() - snumber.size(), snumber.size()));

          // reset current position in stringstream "condline"
          condline->seekg(position);
        }
      }

      // add int parameter vector to line parameter list
      container.add(name(), nnumbers);
    }
    return condline;
  }

  void IntVectorComponent::set_length(int length) { length_ = length; }

  void IntVectorComponent::describe(std::ostream& stream) {}


  RealComponent::RealComponent(std::string name, RealComponentData data)
      : Input::LineComponent(std::move(name), data.optional), data_(data)
  {
  }

  void RealComponent::default_line(std::ostream& stream) { stream << data_.default_value; }

  void RealComponent::describe(std::ostream& stream) {}

  std::shared_ptr<std::stringstream> RealComponent::read(const std::string& section_name,
      std::shared_ptr<std::stringstream> condline, Core::IO::InputParameterContainer& container)
  {
    // initialize double parameter value to be read
    double number = data_.default_value;

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read double parameter value in case the associated parameter label appears in
    // line line of input file
    if ((size_t)position != condline->str().size())
    {
      // read string from stream (need to handle doubles and "none" arguments)
      std::string snumber;
      *condline >> snumber;

      if (!(optional_ && snumber.empty()))
      {
        number = convert_and_validate_string_to_number<double>(
            snumber, name(), section_name, 1, optional_);

        // remove parameter value from stringstream "condline"
        condline->str(
            condline->str().erase((size_t)condline->tellg() - snumber.size(), snumber.size()));

        // reset current position in stringstream "condline"
        condline->seekg(position);
      }
    }

    // add double parameter value to line parameter list
    container.add(name(), number);

    return condline;
  }


  RealVectorComponent::RealVectorComponent(std::string name, int length, RealComponentData data)
      : Input::LineComponent(std::move(name), data.optional), length_(length), data_(data)
  {
  }

  RealVectorComponent::RealVectorComponent(
      std::string name, LengthDefinition length, RealComponentData data)
      : Input::LineComponent(std::move(name), data.optional),
        length_(std::move(length)),
        data_(data)
  {
  }

  void RealVectorComponent::default_line(std::ostream& stream)
  {
    const int default_length = std::visit(DefaultLengthVisitor{}, length_);
    for (int i = 0; i < default_length; ++i) stream << data_.default_value;
  }

  std::string RealVectorComponent::write_read_the_docs() { return "<real vec:" + name() + "> "; }

  void RealVectorComponent::describe(std::ostream& stream) {}

  std::shared_ptr<std::stringstream> RealVectorComponent::read(const std::string& section_name,
      std::shared_ptr<std::stringstream> condline, Core::IO::InputParameterContainer& container)
  {
    const int dynamic_length = std::visit(LengthVisitor{container}, length_);
    std::vector<double> nnumbers(dynamic_length, data_.default_value);

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read double parameter vector in case the associated parameter label appears in
    // line line of input file
    if ((size_t)position != condline->str().size())
    {
      // extract double parameter vector from stringstream "condline"
      for (auto& current_number : nnumbers)
      {
        // extract double vector component as string
        std::string snumber;
        *condline >> snumber;

        // in case the parameter is optional and no value is
        // given
        if (optional_ and snumber.empty())
        {
          break;
        }
        // all other cases
        else
        {
          current_number = convert_and_validate_string_to_number<double>(
              snumber, name(), section_name, dynamic_length, optional_);
        }

        // remove parameter value from stringstream "condline"
        condline->str(
            condline->str().erase((size_t)condline->tellg() - snumber.size(), snumber.size()));

        // reset current position in stringstream "condline"
        condline->seekg(position);
      }
    }

    // add double parameter vector to line parameter list
    container.add(name(), nnumbers);

    return condline;
  }

  void RealVectorComponent::set_length(int length) { length_ = length; }


  const std::string BoolComponent::lineTrue_ = "Yes";
  const std::string BoolComponent::lineFalse_ = "No";
  BoolComponent::BoolComponent(std::string name, const bool defaultvalue, bool optional)
      : Input::LineComponent(std::move(name), optional), defaultvalue_(defaultvalue)
  {
  }

  void BoolComponent::default_line(std::ostream& stream) { print_yes_no(stream, defaultvalue_); }

  void BoolComponent::print_yes_no(std::ostream& stream, const bool value) const
  {
    if (value)
      stream << lineTrue_;
    else
      stream << lineFalse_;
  }

  void BoolComponent::describe(std::ostream& stream) {}

  std::shared_ptr<std::stringstream> BoolComponent::read(const std::string& section_name,
      std::shared_ptr<std::stringstream> condline, Core::IO::InputParameterContainer& container)
  {
    // initialize boolean parameter value to be read
    bool boolean = defaultvalue_;

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read boolean parameter value in case the associated parameter label appears in
    // line line of input file
    if ((size_t)position != condline->str().size())
    {
      // extract boolean parameter value from stringstream "condline" as string
      std::string sboolean;
      *condline >> sboolean;

      // try to convert to bool
      if (sboolean == "Yes" or sboolean == "YES" or sboolean == "yes" or sboolean == "True" or
          sboolean == "TRUE" or sboolean == "true")
        boolean = true;
      else if (sboolean == "No" or sboolean == "NO" or sboolean == "no" or sboolean == "False" or
               sboolean == "FALSE" or sboolean == "false")
        boolean = false;
      else
      {
        // return error in case the conversion was not successful
        FOUR_C_THROW(
            "Value of parameter '%s' for section '%s' not properly specified in input file!",
            name().c_str(), section_name.c_str());
      }

      // remove boolean parameter value from stringstream "condline"
      condline->str(
          condline->str().erase((size_t)condline->tellg() - sboolean.size(), sboolean.size()));

      // reset current position in stringstream "condline"
      condline->seekg(position);
    }

    // add boolean parameter value to line parameter list
    container.add(name(), boolean);

    return condline;
  }


  SwitchComponent::SwitchComponent(std::string name, const KeyType& default_key,
      std::map<KeyType, std::pair<std::string, std::vector<std::shared_ptr<Input::LineComponent>>>>
          choices)
      : Input::LineComponent(std::move(name)),
        default_key_(default_key),
        choices_(std::move(choices))
  {
    Teuchos::Array<int> keys;
    keys.reserve(choices_.size());
    Teuchos::Array<std::string> names_for_keys;
    names_for_keys.reserve(choices_.size());

    for (const auto& [key, choice] : choices_)
    {
      keys.push_back(key);
      names_for_keys.push_back(choice.first);
    }

    component_for_key_ = std::make_unique<Input::SelectionComponent>(
        this->name(), choices_[default_key_].first, names_for_keys, keys);
  }

  void SwitchComponent::default_line(std::ostream& stream)
  {
    component_for_key_->default_line(stream);
    stream << " ";

    for (const auto& component : choices_[default_key_].second)
    {
      component->default_line(stream);
      stream << " ";
    }
  }

  std::vector<std::string> SwitchComponent::write_read_the_docs_lines()
  {
    std::vector<std::string> all_choices_as_rtd;
    std::transform(choices_.begin(), choices_.end(), std::back_inserter(all_choices_as_rtd),
        [this](const auto& key_components)
        {
          const auto& [key, components] = key_components;

          std::stringstream stream;
          stream << choices_[key].first << " ";
          for (const auto& c : components.second) stream << c->write_read_the_docs() << " ";
          return stream.str();
        });

    return all_choices_as_rtd;
  }

  std::string SwitchComponent::write_read_the_docs()
  {
    return component_for_key_->write_read_the_docs() + " [further parameters]";
  }

  Teuchos::Array<std::string> SwitchComponent::get_options()
  {
    return component_for_key_->get_options();
  }

  std::shared_ptr<std::stringstream> SwitchComponent::read(const std::string& section_name,
      std::shared_ptr<std::stringstream> condline, Core::IO::InputParameterContainer& container)
  {
    component_for_key_->read(section_name, condline, container);
    const KeyType key = static_cast<KeyType>(container.get<int>(component_for_key_->name()));

    FOUR_C_ASSERT(choices_.count(key) == 1, "Internal error.");

    for (const auto& component : choices_[key].second)
    {
      component->read(section_name, condline, container);
    }

    return condline;
  }


  void ProcessedComponent::default_line(std::ostream& stream) { stream << "none"; }

  std::shared_ptr<std::stringstream> ProcessedComponent::read(const std::string& section_name,
      std::shared_ptr<std::stringstream> condline, Core::IO::InputParameterContainer& container)
  {
    // initialize string parameter value to be read
    std::string str = "";

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read string parameter value in case the associated parameter label appears in
    // line of input file
    if ((size_t)position != condline->str().size())
    {
      // extract string parameter value from stringstream "condline"
      *condline >> str;

      // return error in case the extraction was not successful
      if (str.empty())
        FOUR_C_THROW(
            "Value of parameter '%s' for section '%s' not properly specified in input file!",
            name().c_str(), section_name.c_str());

      // remove string parameter value from stringstream "condline"
      condline->str(condline->str().erase((size_t)condline->tellg() - str.size(), str.size()));

      // reset current position in stringstream "condline"
      condline->seekg(position);

      // add parameter value to line parameter list
      insert_operation_(str, container);
    }

    return condline;
  }
}  // namespace Input

FOUR_C_NAMESPACE_CLOSE
