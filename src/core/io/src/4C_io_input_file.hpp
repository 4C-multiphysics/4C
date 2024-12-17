// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_FILE_HPP
#define FOUR_C_IO_INPUT_FILE_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_utils_string.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <ranges>
#include <set>
#include <string>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  namespace Nurbs
  {
    class Knotvector;
  }
}  // namespace Core::FE

namespace Core::IO
{
  namespace Internal
  {
    /**
     * An internal iterator that walks over lines in a stream. This iterator is used as an
     * ingredient for the DatFileLineIterator.
     */
    class StreamLineIterator
    {
     public:
      using iterator_category = std::input_iterator_tag;
      using value_type = std::string_view;
      using pointer = const value_type*;
      using reference = value_type;
      using difference_type = int;

      /**
       * Read lines from the given @p stream.
       */
      explicit StreamLineIterator(std::shared_ptr<std::istream> stream);

      /**
       * Read lines rom the given @p stream but a maximum of @p max_reads times. After reading
       * that often, the iterator will be considered past-the-end.
       *
       * @note @p max_reads may be zero. The iterator will not read any lines in that case.
       */
      StreamLineIterator(std::shared_ptr<std::istream> stream, int max_reads);

      /**
       * Construct a past-the-end iterator.
       */
      StreamLineIterator();

      StreamLineIterator& operator++();

      reference operator*() const;

      bool operator==(const StreamLineIterator& other) const;

      bool operator!=(const StreamLineIterator& other) const;

     private:
      //! Stream to read from.
      std::shared_ptr<std::istream> stream_{};

      //! Current line number.
      int line_number_{};

      //! Maximum number of lines to read.
      int max_reads_{};

      //! Currently read line.
      std::string line_;
    };

    /**
     * An internal iterator that walks over lines in a dat file. Providing
     * a custom iterator allows us to abstract away the differences between
     * reading from a file directly or querying lines from a pre-read vector.
     */
    class DatFileLineIterator
    {
     public:
      using PreReadIterator = std::vector<std::string_view>::const_iterator;
      using iterator_category = StreamLineIterator::iterator_category;
      using value_type = StreamLineIterator::value_type;
      using pointer = StreamLineIterator::pointer;
      using reference = StreamLineIterator::reference;
      using difference_type = StreamLineIterator::difference_type;

      DatFileLineIterator() = default;

      //! Construct an iterator that reads from the given stream.
      explicit DatFileLineIterator(std::variant<StreamLineIterator, PreReadIterator> iterator);

      DatFileLineIterator& operator++();

      DatFileLineIterator operator++(int);

      reference operator*() const;

      bool operator==(const DatFileLineIterator& other) const;

      bool operator!=(const DatFileLineIterator& other) const;

     private:
      //! The two ways to iterate over dat file content are handled by the iterator variants.
      std::variant<StreamLineIterator, PreReadIterator> iterator_;
    };

    static_assert(std::input_iterator<DatFileLineIterator>);
    static_assert(std::sentinel_for<DatFileLineIterator, DatFileLineIterator>);

    //! Helper to store section positions in dat files.
    struct SectionPosition
    {
      std::filesystem::path file;
      std::ifstream::pos_type pos;
      unsigned int length;

      void pack(Core::Communication::PackBuffer& data) const;

      void unpack(Core::Communication::UnpackBuffer& buffer);
    };
  }  // namespace Internal

  /**
   * This class encapsulates input files independent of their format.
   *
   * Objects of this class read the content of a file and grant access to it. The input is not
   * yet interpreted in any way. Input files contain different sections. A section either contains
   * key-value pairs or a list of arbitrary lines of text. Sections may appear in an arbitrary order
   * and each section name must be unique. An exception is the special section named "INCLUDES"
   * which can contain a list of other files that should be read in addition to the current file.
   *
   * Three file formats are supported: the custom .dat file format and the standard .yaml (or .yml)
   * and .json formats. The format of a file is detected based on its ending. If the ending is not
   * one of the above mentioned, the file is assumed to be in the .dat format.
   * Included files do not have to use the same format as the including file.
   *
   * The following example shows the structure of a .dat file:
   *
   * @code
   * // A comment. Sections start with at least two dashes in the .dat file format.
   * --INCLUDES
   * include1.yml
   * include2.json
   * // More dashes are allowed to start a section.
   * -------------SECTION1
   * key1 = value1
   * key2 = value2
   * // The "=" is optional and can be replaced by whitespace. This only works if key and value do
   * // not contain whitespace themselves.
   * key3 value3
   * --SECTION2
   * A line with content that is not a key-value pair. It can contain anything.
   * @endcode
   *
   * A similar example looks like this in .yaml format:
   *
   * @code
   * INCLUDES:
   *   - include1.yml
   *   - include2.json
   * # A comment. Note that sections do NOT start with dashes in the .yaml file format.
   * SECTION1:
   *   key1: value1
   *   key2: value2
   * SECTION2:
   *   - A line with content that is not a key-value pair.
   *   - It can contain anything.
   * @endcode
   *
   *
   * @note The file is only read on rank 0 and the content is distributed to all ranks.
   */
  class InputFile
  {
   public:
    /// Format flags for dumping the content of the input file.
    enum class Format
    {
      /// Print the input file as a dat file.
      dat,
      /// Print the input file as a yaml file.
      yaml,
    };

    /// construct a reader for a given file
    InputFile(std::string filename, MPI_Comm comm);

    /**
     * Get the (absolute) file path of the input file that contained a section. If the section is
     * unknown or was not read from any file, an empty path is returned.
     */
    [[nodiscard]] std::filesystem::path file_for_section(const std::string& section_name) const;

    /**
     * Get a a range of lines inside a section that have actual content, i.e., they contain
     * something other than whitespace or comments. Any line returned will have comment stripped
     * and whitespace trimmed. The usual way to do something with the lines is
     *
     * @code
     *   for (const auto& line : input.lines_in_section("section_name"))
     *   {
     *     // do something with line
     *   }
     * @endcode
     *
     * Depending on the @p section_name, the lines were either pre-read or are read from the file
     * on-the-fly.
     *
     * @return A range of string_views to the lines in this section.
     */
    std::ranges::view auto lines_in_section(const std::string& section_name);

    /**
     * Returns whether a section with the given name exists in the input file and contains any
     * content.
     */
    [[nodiscard]] bool has_section(const std::string& section_name) const;

    /**
     * Access MPI communicator associated with this object.
     */
    [[nodiscard]] MPI_Comm get_comm() const { return comm_; }

    /**
     * Print a list of all sections that are contained in the input file but never
     * accessed through this object.
     *
     * @return True if there were unknown sections, false otherwise.
     */
    bool print_unknown_sections(std::ostream& out) const;

    /**
     * Dump the content of the input file to the given output stream.
     * Various output format settings are controlled by the format flags.
     */
    void dump(std::ostream& output, Format format) const;

   private:
    /**
     * The shared part of reading a file and postprocessing its content.
     */
    void read_generic(const std::filesystem::path& top_level_file);

    //! Remember that a section was used.
    void record_section_used(const std::string& section_name);

    //! Internal helper to get the range of lines in a section.
    //! Does not record the section as used.
    [[nodiscard]] std::ranges::view auto line_range(const std::string& section_name) const;

    /// The communicator associated with this object.
    MPI_Comm comm_;

    /// The whole input file.
    std::vector<char> inputfile_;

    /// The lines of the #input_file_ as std::string_view.
    std::vector<std::string_view> lines_;

    /// file positions of skipped sections
    std::map<std::string, Internal::SectionPosition> excludepositions_;

    /// Section positions of all sections inside the #inputfile_ array.
    std::map<std::string, std::pair<std::size_t, std::size_t>> positions_;

    /// Mapping of section names to the file they were read from.
    std::unordered_map<std::string, std::string> section_to_file_mapping_;

    /// Protocol of known and unknown section names
    std::map<std::string, bool> knownsections_;
  };

  /**
   * Split the given @p line into a key-value pair. Key and value are normally separated by
   * whitespace. In case there are multiple distinct whitespace groups in one line, the first of
   * these is assumed to be the separator and all the other whitespace is assumed to be part of
   * the value. Key and value may also be separated by an equals sign "=" and at least one
   * whitespace character on both sides. In this case, key and value may contain spaces
   * internally. Leading and trailing whitespace is trimmed from both key and value.
   *
   * @throws Core::Exception If the @p line cannot be read.
   *
   * @return A pair of key and value.
   */
  std::pair<std::string, std::string> read_key_value(const std::string& line);

  /**
   * Read a @p section_name from the input file and store the key-value pairs in the given @p list.
   */
  bool read_parameters_in_section(
      InputFile& input, const std::string& section_name, Teuchos::ParameterList& list);

  /**
   * Read a node-design topology section
   *
   * @param input The input file.
   * @param name Name of the topology to read
   * @param dobj_fenode Resulting collection of all nodes that belong to a design.
   * @param get_discretization Callback to return a discretization by name.
   */
  void read_design(InputFile& input, const std::string& name,
      std::vector<std::vector<int>>& dobj_fenode,
      const std::function<const Core::FE::Discretization&(const std::string& name)>&
          get_discretization);

  /**
   * \brief read the knotvector section (for isogeometric analysis)
   *
   * \param  reader         (in ): InputFile object
   * \param  name           (in ): Name/type of discretisation
   * \param  disknots       (out): node vector coordinates
   *
   */
  void read_knots(InputFile& input, const std::string& name,
      std::shared_ptr<Core::FE::Nurbs::Knotvector>& disknots);


  /// -- template and inline functions --- //

  inline std::ranges::view auto InputFile::line_range(const std::string& section_name) const
  {
    auto filter = [](std::string_view line)
    { return !Core::Utils::strip_comment(std::string(line)).empty(); };

    if (excludepositions_.count(section_name) > 0)
    {
      const auto [path, start_pos, length] = excludepositions_.at(section_name);

      auto file = std::make_shared<std::ifstream>(path);
      file->seekg(start_pos);

      return std::views::filter(
          std::ranges::subrange<Internal::DatFileLineIterator>(
              Internal::DatFileLineIterator(Internal::StreamLineIterator(std::move(file), length)),
              Internal::DatFileLineIterator(Internal::StreamLineIterator())),
          filter);
    }

    const auto [start_line, end_line] = std::invoke(
        [&]() -> std::pair<std::size_t, std::size_t>
        {
          auto entry_it = positions_.find(section_name);
          if (entry_it == positions_.end()) return {lines_.size(), lines_.size()};
          return entry_it->second;
        });

    return std::views::filter(
        std::ranges::subrange(Internal::DatFileLineIterator(lines_.begin() + start_line),
            Internal::DatFileLineIterator(lines_.begin() + end_line)),
        filter);
  }

  inline std::ranges::view auto InputFile::lines_in_section(const std::string& section_name)
  {
    record_section_used(section_name);
    return line_range(section_name);
  }


}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
