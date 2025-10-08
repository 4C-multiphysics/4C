// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CREATE_RTDFILES_UTILS_HPP
#define FOUR_C_CREATE_RTDFILES_UTILS_HPP
#include "4C_config.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"

#include <map>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace RTD
{
  /*!
   * /brief The Table class is used to create a table in a readthedocs document (written in
   * restructuredText).
   *
   * The table for readthedocs is a so-called list-table.
   * One may enter directives (like "align", "header-rows"), and also enter the relative widths of
   * columns, which is then converted into a directive as well.
   * Then, one may enter table content row by row.
   *
   * The construction parameter defines the number of columns in the table.
   * Of course, if you add a row with a different number of cells than predefined, an error is
   * thrown.

  */
  class Table
  {
   public:
    /*!
     * The table constructor doesn't do much besides defining the number of columns of the table.
     *
     * @param size Number of columns
     */
    Table(const unsigned& size);
    /*!
     * The content of the table is entered row by row.
     *
     * @param row This vector of strings fills the row. Of course, the size of this vector must be
     * equal to the size of the table (construction parameter). Otherwise an error is thrown.
     */
    void add_row(const std::vector<std::string>& row);
    /*!
     * The relative widths of the table columns are set here. The numbers should be given as
     * characters after which a new line is created, that is, each cell gets an enforced line
     * break at a space closest to the i'th width character position. Can be set to 0 for any
     * cell. For this case the cell is not split! If the size of the vector does not correspond to
     * the size of the table, an error is thrown.
     *
     * @param widths vector of unsigned ints, defining the characters that should fit into each
     * cell.
     */
    void set_widths(const std::vector<unsigned>& widths);
    /*!
     * The directives are entered as a key-value pair, which is also the way, how it is written in
     * a list-table. If the type of table is changed some day, the key-value pairs can probably
     * still be used to provide the controls needed in any other table style (csv-table, ...)
     *
     * @param key The control parameter of the table (e.g., width, widths, header-rows, align,...)
     * @param value The value of the control parameter, given as a string no matter what it
     *              actually is.
     */
    void add_directive(const std::string& key, const std::string& value);
    /*!
     * A getter method for the current number of rows in the table
     *
     * @return Number of rows of the table (that is, the size of the private vector tablerows_
     */
    unsigned get_rows() const;
    /*!
     * Here the table is printed as a restructuredText list-table
     *
     * @param stream The output file stream of the restructuredText file.
     */
    void print(std::ostream& stream) const;

   private:
    /*!
     * The construction parameter: Number of columns of the table
     */
    const unsigned tablewidth_;
    std::vector<unsigned> widths_;
    /*!
     * tablerows_ is a vector, that contains the content rows of table as each entry.
     *            Each row is again a vector of strings that contains the cells.
     */
    std::vector<std::vector<std::string>> tablerows_;
    /*!
     * directives_ is a mapping containing all table controls which are given for the table.
     */
    std::map<std::string, std::string> directives_;
  };

  /*!
    \brief Write a link target to the restructuredText stream

    \param[in] stream: stream for the restructuredText file
    \param[in] line: The link target to be printed

  */
  void write_linktarget(std::ostream& stream, const std::string& line);

  /*!
   * \brief Create a yaml file containing the cell type information about
   *
   * - nodes incl. coordinates
   * - lines
   * - surfaces
   *
   * \param[in] stream: filename for the the yaml file, should be elementinformation.yaml
   */
  void write_yaml_cell_type_information(std::ostream& yamlfile);


  /*----------------------------------------------------------------------*/
  /*!
   * \brief Write a paragraph to the restructuredText stream, with an optional additional
   * indentation
   *
   * \param[in] stream: stream for the restructuredText file
   * \param[in] paragraph: The text to be printed
   * \param[in] indent: Optional indentation of the paragraph (default 0)
   *
   */
  void write_paragraph(std::ostream& stream, std::string paragraph, size_t indent = 0);
  /*----------------------------------------------------------------------*/
  /*!
   * \brief Write a code block to the restructuredText stream
   *
   * \param[in] stream: stream for the restructuredText file
   * \param[in] lines: A vector of strings that make up the code
   *
   */
  void write_code(std::ostream& stream, const std::vector<std::string>& lines);
  /*----------------------------------------------------------------------*/
  /*!
   * \brief Write a note block to the restructuredText stream
   *
   * \param[in] stream: stream for the restructuredText file
   * \param[in] line: The link target to be printed
   *
   */
  void write_note(std::ostream& stream, const std::string& paragraph);

  /*!
   *  \brief write the information for all available cell types for readthedocs
   *
   *  \param[in] stream: stream for the restructuredText file
   */
  void write_celltype_reference(std::ostream& stream);

  /*!
   * replace control chars with restructuredText control commands
   * at this time, only equations are considered
   *
   *  @param[inout] documentation_string string to be modified.
   */
  void replace_restructuredtext_keys(std::string& documentation_string);

}  // namespace RTD

std::ostream& operator<<(std::ostream& os, const RTD::Table& table);


FOUR_C_NAMESPACE_CLOSE

#endif
