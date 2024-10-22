// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_definition.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io_inputreader.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <iterator>
#include <utility>

FOUR_C_NAMESPACE_OPEN



/* -----------------------------------------------------------------------------------------------*
 | Class ConditionDefinition                                                                      |
 * -----------------------------------------------------------------------------------------------*/

Core::Conditions::ConditionDefinition::ConditionDefinition(std::string sectionname,
    std::string conditionname, std::string description, Core::Conditions::ConditionType condtype,
    bool buildgeometry, Core::Conditions::GeometryType gtype)
    : sectionname_(std::move(sectionname)),
      conditionname_(std::move(conditionname)),
      description_(std::move(description)),
      condtype_(condtype),
      buildgeometry_(buildgeometry),
      gtype_(gtype)
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::ConditionDefinition::add_component(
    const Teuchos::RCP<Input::LineComponent>& c)
{
  inputline_.push_back(c);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::ConditionDefinition::read(Core::IO::DatFileReader& reader,
    std::multimap<int, Teuchos::RCP<Core::Conditions::Condition>>& cmap)
{
  // read the range into a vector
  std::vector<std::string> section_vec;
  {
    const auto& section = reader.lines_in_section("--" + sectionname_);
    for (const auto& line : section) section_vec.push_back(std::string{line});
  }

  if (section_vec.empty()) return;

  // First we read a header for the current section: It needs to start with the
  // geometry type followed by the number of lines:
  //
  // ("DPOINT" | "DLINE" | "DSURF" | "DVOL" ) <number>

  std::stringstream line(section_vec[0]);

  Core::IO::ValueParser parser_header(
      line, "While reading header of condition section '" + sectionname_ + "': ");

  const std::string expected_geometry_type = std::invoke(
      [this]()
      {
        switch (gtype_)
        {
          case Core::Conditions::geometry_type_point:
            return "DPOINT";
          case Core::Conditions::geometry_type_line:
            return "DLINE";
          case Core::Conditions::geometry_type_surface:
            return "DSURF";
          case Core::Conditions::geometry_type_volume:
            return "DVOL";
          default:
            FOUR_C_THROW("Geometry type unspecified");
        }
      });

  parser_header.consume(expected_geometry_type);
  const int condition_count = parser_header.read<int>();

  if (condition_count != static_cast<int>(section_vec.size() - 1))
  {
    FOUR_C_THROW("Got %d condition lines but expected %d in section '%s'", section_vec.size() - 1,
        condition_count, sectionname_.c_str());
  }

  for (auto i = section_vec.begin() + 1; i != section_vec.end(); ++i)
  {
    Teuchos::RCP<std::stringstream> condline = Teuchos::make_rcp<std::stringstream>(*i);

    // add trailing white space to stringstream "condline" to avoid deletion of stringstream upon
    // reading the last entry inside This is required since the material parameters can be
    // specified in an arbitrary order in the input file. So it might happen that the last entry
    // is extracted before all of the previous ones are.
    condline->seekp(0, condline->end);
    *condline << " ";

    Core::IO::ValueParser parser_content(
        *condline, "While reading content of condition section '" + sectionname_ + "': ");

    parser_content.consume("E");
    // Read a one-based condition number but convert it to zero-based for internal use.
    const int dobjid = parser_content.read<int>() - 1;
    parser_content.consume("-");

    Teuchos::RCP<Core::Conditions::Condition> condition =
        Teuchos::make_rcp<Core::Conditions::Condition>(dobjid, condtype_, buildgeometry_, gtype_);

    for (auto& j : inputline_)
    {
      condline = j->read(section_name(), condline, condition->parameters());
    }

    //------------------------------- put condition in map of conditions
    cmap.insert(std::pair<int, Teuchos::RCP<Core::Conditions::Condition>>(dobjid, condition));
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::ostream& Core::Conditions::ConditionDefinition::print(
    std::ostream& stream, const Core::FE::Discretization* dis)
{
  unsigned l = sectionname_.length();
  stream << "--";
  for (int i = 0; i < std::max<int>(65 - l, 0); ++i) stream << '-';
  stream << sectionname_ << '\n';

  std::string name;
  switch (gtype_)
  {
    case Core::Conditions::geometry_type_point:
      name = "DPOINT";
      break;
    case Core::Conditions::geometry_type_line:
      name = "DLINE";
      break;
    case Core::Conditions::geometry_type_surface:
      name = "DSURF";
      break;
    case Core::Conditions::geometry_type_volume:
      name = "DVOL";
      break;
    default:
      FOUR_C_THROW("geometry type unspecified");
      break;
  }

  int count = 0;
  if (dis != nullptr)
  {
    std::vector<Core::Conditions::Condition*> conds;
    dis->get_condition(conditionname_, conds);
    for (auto& cond : conds)
    {
      if (cond->g_type() == gtype_)
      {
        count += 1;
      }
    }
  }

  stream << name;
  l = name.length();
  for (int i = 0; i < std::max<int>(31 - l, 0); ++i) stream << ' ';
  stream << ' ' << count << '\n';

  stream << "//"
         << "E num - ";
  for (auto& i : inputline_)
  {
    i->default_line(stream);
    stream << " ";
  }

  stream << "\n";

  if (dis != nullptr)
  {
    std::vector<Core::Conditions::Condition*> conds;
    dis->get_condition(conditionname_, conds);

    for (auto& cond : conds)
    {
      if (cond->g_type() == gtype_)
      {
        stream << "E " << cond->id() << " - ";
        for (auto& i : inputline_)
        {
          i->print(stream, cond->parameters());
          stream << " ";
        }
        stream << "\n";
      }
    }
  }

  return stream;
}

FOUR_C_NAMESPACE_CLOSE
