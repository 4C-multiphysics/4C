// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_string.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Core::Utils
{
  std::string trim(const std::string& line)
  {
#if (BOOST_MAJOR_VERSION == 1) && (BOOST_MINOR_VERSION >= 47)
    return boost::algorithm::trim_all_copy(boost::algorithm::replace_all_copy(line, "\t", " "));
#else
    std::istringstream t;
    std::string s;
    std::string newline;
    t.str(line);
    while (t >> s)
    {
      newline.append(s);
      newline.append(" ");
    }
    if (newline.size() > 0) newline.resize(newline.size() - 1);
    return newline;
#endif
  }

  std::vector<std::string> split(const std::string& input, const std::string& delimiter)
  {
    std::vector<std::string> return_value{};
    return boost::split(return_value, input, boost::is_any_of(delimiter));
  }

  std::string strip_comment(const std::string& line)
  {
    std::string newline = line;

    // remove comments
    std::string::size_type loc = line.find("//");
    if (loc != std::string::npos)
    {
      newline = newline.substr(0, loc);
    }

    // remove trailing and leading whitespaces
    // compact internal whitespaces
    newline = trim(newline);

    return newline;
  }

  std::string to_lower(const std::string& line) { return boost::algorithm::to_lower_copy(line); }

  std::vector<std::string> split_string_list(const std::string& str, const std::string& separator)
  {
    // Keep the currently remaining part of the input string in 'tmp' and
    // keep chopping elements of the list off the front
    std::string tmp = str;

    // Remove whitespace from the end of the string
    while (tmp.size() != 0 && tmp.back() == ' ') tmp.erase(tmp.size() - 1, 1);

    // Split the input list until it is empty. In every iteration, 'tmp' represents the remaining
    // portion of the string after the next separator. Since trailing spaces have already been
    // removed, 'tmp' will eventually be empty if 'str' ended with a separator, even if there was
    // space after the last separator.
    std::vector<std::string> split_list;
    while (tmp.size() != 0)
    {
      std::string name;
      name = tmp;

      if (name.find(separator) != std::string::npos)
      {
        name.erase(name.find(separator), std::string::npos);
        tmp.erase(0, tmp.find(separator) + separator.size());
      }
      else
        tmp = "";

      // Strip spaces from this element's front and end
      while ((name.size() != 0) && (name[0] == ' ')) name.erase(0, 1);
      while (name.size() != 0 && name.back() == ' ') name.erase(name.size() - 1, 1);

      split_list.push_back(name);
    }

    return split_list;
  }

  std::vector<std::string> split_string_list(const std::string& str, const char separator)
  {
    return split_string_list(str, std::string(1, separator));
  }
}  // namespace Core::Utils


FOUR_C_NAMESPACE_CLOSE
