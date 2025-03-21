// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_FILE_HPP
#define FOUR_C_IO_INPUT_FILE_HPP

#include "4C_config.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_pack_buffer.hpp"
#include "4C_io_input_parameter_container.hpp"

#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputSpec;

  namespace Internal
  {
    class InputFileImpl;
    class InputFileFragmentImpl;
  }  // namespace Internal

  /**
   * This class encapsulates input files independent of their format.
   *
   * Objects of this class read the content of a file and grant access to it. The input is not
   * yet interpreted in any way. Input files contain different sections. A section either contains
   * key-value pairs or a list of arbitrary lines of text. The content is accessed via the
   * in_section() function which return InputFile::Fragment objects. These hide what exactly is
   * contained and in which format. The content can be matched against an expected InputSpec to
   * extract meaningful data.
   *
   * Sections may appear in an arbitrary order and each section name must be unique. An exception is
   * the special section named "INCLUDES" which can contain a list of other files that should be
   * read in addition to the current file.
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
   * @note The file is only read on rank 0 to save memory. Sections that are huge are only
   * distributed to other ranks if they are accessed through line_in_section(). If you only
   * want to read a section on rank 0, use in_section_rank_0_only().
   */
  class InputFile
  {
   public:
    /**
     * A Fragment is a part of the input file. Fragments are used to store the content of a section.
     * and are used to match() against an InputSpec. Instead of using a concrete data type, the
     * Fragment class encapsulates details of the input file format and provides a few methods to
     * interact with the content. A Fragment is a lightweight object that can be copied and passed
     * around by value. The associated InputFile must outlive any of its fragments.
     */
    class Fragment
    {
     public:
      /**
       * Return the content of the fragment as a string formatted in the .dat style. You need to
       * manually parse this string. Depending on the format of the input file, this might be rather
       * inefficient, since you convert a differently structured format into a one-liner string,
       * just to reparse it later.
       *
       * @deprecated Do not use this function in new code. Say what you expect with an InputSpec and
       * use the match() function instead.
       */
      [[nodiscard]] std::string_view get_as_dat_style_string() const;

      /**
       * Match the fragment against a given InputSpec @p spec. If the fragment does not match the
       * spec, an empty optional is returned. If the fragment matches the spec, an
       * InputParameterContainer with the parsed content is returned.
       */
      [[nodiscard]] std::optional<InputParameterContainer> match(const InputSpec& spec) const;

      /**
       * A raw pointer to implementation details. This pointer will be valid as long as the
       * associated InputFile object is alive.
       *
       * @note Can be public since you cannot do anything useful with this opaque pointer.
       */
      Internal::InputFileFragmentImpl* pimpl_;
    };

    /**
     * Iterator over the Fragments making up a section.
     */
    using FragmentIterator = std::vector<Fragment>::const_iterator;

    /**
     * A range of FragmentIterators.
     */
    using FragmentIteratorRange = std::ranges::subrange<FragmentIterator>;

    /**
     * Construct an InputFile that understands the given @p valid_sections. These sections can
     * be fully validated based on their InputSpec. The @p legacy_section_names are also accepted as
     * valid section names but the section content cannot be validated. We implicitly assume a list
     * of strings as valid input for these sections. Additionally, the following special sections
     * are always accepted:
     *
     * - "INCLUDES" which contains a list of files to include.
     * - "TITLE" which may contain anything. This section is not validated in any way. In contrast
     *   to a comment, this section is useful to store arbitrary data, e.g. generated by a script,
     *   which can be emitted again.
     *
     * Since reading and distributing input is a collective operation, an MPI communicator @p comm
     * is required.
     */
    InputFile(std::map<std::string, InputSpec> valid_sections,
        std::vector<std::string> legacy_section_names, MPI_Comm comm);

    /**
     * Like the other constructor, but also accepts partial specs for legacy sections. This is
     * useful to provide at least some partial information about the expected format of a legacy
     * section in the metadata output. 4C will not do anything with the partial specs.
     *
     * @note This constructor is meant to ease the transition from legacy sections to fully
     * parseable sections.
     */
    InputFile(std::map<std::string, InputSpec> valid_sections,
        std::vector<std::string> legacy_section_names,
        std::map<std::string, InputSpec> legacy_partial_specs, MPI_Comm comm);

    /**
     * Destructor.
     */
    ~InputFile();

    /**
     * Copy constructor is deleted. InputFile can contain large amounts of data and copying it
     * is almost certainly not what you want.
     */
    InputFile(const InputFile&) = delete;

    /**
     * Copy assignment is deleted. InputFile can contain large amounts of data and copying it is
     * almost certainly not what you want.
     */
    InputFile& operator=(const InputFile&) = delete;

    /**
     * Move constructor.
     */
    InputFile(InputFile&&) noexcept = default;

    /**
     * Move assignment.
     */
    InputFile& operator=(InputFile&&) noexcept = default;

    /**
     * Read the content of the input file. The file is read on rank 0 and distributed to all
     * other ranks if necessary. The file format is detected based on the file ending. If the
     * file ending is not recognized, the .dat format is assumed.
     */
    void read(const std::filesystem::path& top_level_file);

    /**
     * Get the (absolute) file path of the input file that contained a section. If the section
     * is unknown or was not read from any file, an empty path is returned.
     */
    [[nodiscard]] std::filesystem::path file_for_section(const std::string& section_name) const;

    /**
     * Get a range of Fragments inside a section. A Fragment always contains meaningful content,
     * i.e., something other than whitespace or comments. The usual way to interact with the
     * Fragments is as follows:
     *
     * @code
     *   for (const auto& input_fragment : input.in_section("section_name"))
     *   {
     *      auto parameters = fragment.match(spec);
     *   }
     * @endcode
     *
     * @return A range of Fragments which encapsulate the content of the section. Use the
     * Fragment::match() function to extract the content.
     *
     * @note This is a collective call that needs to be called on all MPI ranks in the
     * communicator associated with this object. Depending on the section size, the content
     * might need to be distributed from rank 0 to all other ranks. This happens automatically.
     */
    FragmentIteratorRange in_section(const std::string& section_name) const;

    /**
     * This function is similar to in_section(), but it only returns the lines on rank 0 and
     * returns an empty range on all other ranks. This is useful for sections that might be huge
     * and are not processed on all ranks.
     */
    FragmentIteratorRange in_section_rank_0_only(const std::string& section_name) const;

    /**
     * Match a whole section named @p section_name against the input file content. The results
     * are stored in the @p container. If the section is not known, an exception is thrown. Note
     * that you do not need to pass an InputSpec to this function, since the InputFile object
     * already knows about the expected format of the section. Nevertheless, this function only
     * makes sense for sections with a known InputSpec. Legacy string sections cannot use this
     * function and must be processed with in_section() or in_section_rank_0_only().
     */
    void match_section(const std::string& section_name, InputParameterContainer& container) const;

    /**
     * Returns true if the input file contains a section with the given name.
     *
     * @note This is a collective call that needs to be called on all MPI ranks in the
     * communicator.
     */
    [[nodiscard]] bool has_section(const std::string& section_name) const;

    /**
     * Access MPI communicator associated with this object.
     */
    [[nodiscard]] MPI_Comm get_comm() const;

    /**
     * Emit metadata about the input file to the given output stream @p out. The metadata
     * contains information about all sections and parameters that are known to this object. The
     * metadata information can be useful for additional tools that generate schema files or
     * documentation. The output is formatted as YAML.
     */
    void emit_metadata(std::ostream& out) const;

    /**
     * Write the content of the input file to the given output stream @p out. The content is
     * written in .yaml format.
     *
     * @note You probably only want to call this function on rank 0.
     */
    void write_as_yaml(std::ostream& out) const;

   private:
    std::unique_ptr<Internal::InputFileImpl> pimpl_;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
