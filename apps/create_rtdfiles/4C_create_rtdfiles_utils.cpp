// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_create_rtdfiles_utils.hpp"

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_string.hpp"

#include <Teuchos_StrUtils.hpp>

#include <format>

FOUR_C_NAMESPACE_OPEN


namespace RTD
{
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  Table::Table(const unsigned& size) : tablewidth_(size)
  {
    for (unsigned i = 0; i < tablewidth_; ++i)
    {
      widths_.push_back(0);
    }
  }
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void Table::add_row(const std::vector<std::string>& row)
  {
    if (row.size() != tablewidth_)
    {
      FOUR_C_THROW(
          "Trying to add {} row elements into a table with {} rows", row.size(), tablewidth_);
    }
    tablerows_.push_back(row);
  }
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void Table::set_widths(const std::vector<unsigned>& widths)
  {
    if (widths.size() != tablewidth_)
    {
      FOUR_C_THROW(
          "Number of given cell widths ({}) "
          "does not correspond to the number of rows ({})",
          widths.size(), tablewidth_);
    }
    widths_ = widths;
  }
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void Table::add_directive(const std::string& key, const std::string& value)
  {
    directives_[key] = value;
  }
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  unsigned Table::get_rows() const { return tablerows_.size(); }
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void Table::print(std::ostream& stream) const
  {
    // if the widths are not set, they are currently set to 100/tablewidth_
    unsigned defaultcolsize = 100 / tablewidth_;
    bool isWidthDirectiveGiven = false;
    stream << ".. list-table::\n";
    // write directives
    for (const auto& directive : directives_)
    {
      stream << "   :" << directive.first << ": " << directive.second << "\n";
      if (directive.first.substr(0, 5) == "width") isWidthDirectiveGiven = true;
    }
    // add the cell widths to the directives if not already given
    if (!isWidthDirectiveGiven)
    {
      stream << "   :widths: ";
      unsigned wd;
      for (auto itw = widths_.begin(); itw != widths_.end(); ++itw)
      {
        if (itw != widths_.begin()) stream << ",";
        wd = (*itw == 0) ? defaultcolsize : *itw;
        stream << wd;
      }
      stream << "\n";
    }
    stream << "\n";
    //
    // now write table content (split if necessary, i.e., more characters than given in widths_)
    for (const auto& tablerow : tablerows_)
    {
      for (unsigned i = 0; i < tablewidth_; ++i)
      {
        std::string cellstring = (i == 0) ? "   * - " : "     - ";
        if ((widths_[i] != 0) and (tablerow[i].length() > widths_[i]))
        {
          std::string cellstringPart = tablerow[i];
          std::size_t spacepos = cellstringPart.rfind(" ", widths_[i]);
          if (spacepos < cellstringPart.npos)
          {
            cellstring += cellstringPart.substr(0, spacepos) + " |break| \n";
            cellstringPart = cellstringPart.substr(spacepos + 1);
            // print the rest of the description with two empty columns before
            while (cellstringPart.length() > widths_[i])
            {
              spacepos = cellstringPart.rfind(" ", widths_[i]);
              if (spacepos == cellstringPart.npos) break;
              cellstring += "       " + cellstringPart.substr(0, spacepos) + " |break| \n";
              cellstringPart = cellstringPart.substr(spacepos + 1);
            }
            cellstring += "       ";
          }
          cellstring += cellstringPart;
        }
        else
        {
          cellstring += tablerow[i];
        }
        stream << cellstring << "\n";
      }
    }
    stream << "\n";
  }
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_linktarget(std::ostream& stream, const std::string& line)
  {
    stream << ".. _" << line << ":\n\n";
  }
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_header(std::ostream& stream, unsigned level, const std::string& line)
  {
    const std::vector<char> headerchar{'=', '-', '~', '^'};
    unsigned headerlength = line.length();
    stream << line << "\n";
    if (level > headerchar.size())
    {
      FOUR_C_THROW("Header level for ReadTheDocs output must be [0,3], but is {}", level);
    }
    stream << std::string(headerlength, headerchar[level]);
    stream << "\n\n";
  }
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_paragraph(std::ostream& stream, std::string paragraph, size_t indent)
  {
    size_t mathstartpos = paragraph.find("$");
    size_t mathendpos = 0;
    while (mathstartpos != paragraph.npos)
    {
      mathendpos = paragraph.find("$", mathstartpos + 1);
      if (mathendpos == paragraph.npos)
      {
        FOUR_C_THROW(
            "Math tags in a ReadTheDocs paragraph must occur pairwise. "
            "Error found in: {}\n",
            paragraph);
      }
      paragraph.replace(mathendpos, 1, "`");
      paragraph.replace(mathstartpos, 1, ":math:`");
      mathstartpos = paragraph.find("$");
    }
    stream << std::string(" ", indent) << paragraph << "\n\n";
  }
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_code(std::ostream& stream, const std::vector<std::string>& lines)
  {
    stream << "::\n\n";
    for (const auto& line : lines)
    {
      stream << "   " << line << "\n";
    }
    stream << "\n";
  }
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_note(std::ostream& stream, const std::string& paragraph)
  {
    stream << ".. note::\n\n";
    stream << "   " << paragraph << "\n\n";
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  std::ostream& operator<<(std::ostream& stream, const Table& table)
  {
    table.print(stream);
    return stream;
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_celltype_reference(std::ostream& stream)
  {
    write_linktarget(stream, "celltypes");
    write_header(stream, 1, "Cell types");

    // We run the loop over the cell types four times to sort the cell types after their dimension
    for (unsigned outputdim = 0; outputdim < 4; ++outputdim)
    {
      ;
      write_linktarget(stream, std::format("{}D_cell_types", outputdim));
      write_header(stream, 2, std::format("{}D cell types", outputdim));

      for (auto celltype : Core::FE::celltype_array<Core::FE::all_physical_celltypes>)
      {
        std::string celltypename = Core::FE::cell_type_to_string(celltype);
        // Skip the cell type if it has not the desired dimension
        const unsigned celldimension = Core::FE::get_dimension(celltype);
        if (celldimension != outputdim) continue;

        std::string celltypelinkname = Core::Utils::to_lower(celltypename);
        write_linktarget(stream, celltypelinkname);
        write_header(stream, 3, celltypename);

        std::stringstream celltypeinfostream;
        celltypeinfostream << "- Nodes: " << Core::FE::get_number_of_element_nodes(celltype)
                           << std::endl;
        celltypeinfostream << "- Dimension: " << celldimension << std::endl;
        if (Core::FE::get_order(celltype, -1) >= 0)
        {
          celltypeinfostream << "- Shape function order (element): "
                             << Core::FE::get_degree(celltype) << std::endl;
          celltypeinfostream << "- Shape function order (edges): " << Core::FE::get_order(celltype)
                             << std::endl;
        }
        std::string celltypeinformation = celltypeinfostream.str();
        write_paragraph(stream, celltypeinformation);

        if (celldimension >= 2)
        {
          const std::string figurename("reference_images/" + celltypename + ".png");
          std::string captionstring = "**" + celltypename + ":** ";
          if (celldimension == 2)
            captionstring += "Line and node numbering";
          else
            captionstring += "Left: Line and node numbering, right: Face numbering";
          std::string figureincludestring = ".. figure:: " + figurename + "\n";
          figureincludestring += "    :alt: Figure not available for " + celltypename + "\n";
          figureincludestring += "    :width: ";
          figureincludestring += (outputdim == 3) ? "100%" : "50%";
          figureincludestring += "\n\n";
          figureincludestring += "    " + captionstring;
          write_paragraph(stream, figureincludestring);
        }
      }
    }
  }

  void write_yaml_cell_type_information(std::ostream& yamlfile)
  {
    for (auto celltype : Core::FE::celltype_array<Core::FE::all_physical_celltypes>)
    {
      std::string celltypename = Core::FE::cell_type_to_string(celltype);
      std::string yamlcelltypestring = celltypename + ":\n";
      // 0. information: dimension of the element
      yamlcelltypestring +=
          "  dimension: " + std::to_string(Core::FE::get_dimension(celltype)) + "\n";
      // 1. information: nodal coordinates
      Core::LinAlg::SerialDenseMatrix coordmap;
      try
      {
        coordmap = Core::FE::get_ele_node_numbering_nodes_paramspace(celltype);
      }
      catch (...)
      {
        std::cout << "could not read coords\n";
        continue;
      }
      const unsigned num_nodes = coordmap.numCols();
      yamlcelltypestring += "  nodes:\n";
      for (unsigned int node = 0; node < num_nodes; ++node)
      {
        yamlcelltypestring += "    - [";
        for (int indx = 0; indx < coordmap.numRows(); ++indx)
        {
          if (indx > 0) yamlcelltypestring += ",";
          yamlcelltypestring += std::format("{:6.2f}", coordmap(indx, node));
        }
        yamlcelltypestring += "]\n";
      }
      // 2. information: line vectors of internal node numbers
      bool nodes_exist = true;
      std::vector<std::vector<int>> linevector;
      try
      {
        linevector = Core::FE::get_ele_node_numbering_lines(celltype);
      }
      catch (...)
      {
        std::cout << "could not read lines\n";
        continue;
      }
      yamlcelltypestring += "  lines:\n";
      for (auto line : linevector)
      {
        yamlcelltypestring += "    - [";
        for (size_t indx = 0; indx < line.size(); ++indx)
        {
          if (indx > 0) yamlcelltypestring += ",";
          yamlcelltypestring += std::format("{:3d}", line[indx]);
          if ((unsigned int)line[indx] >= num_nodes) nodes_exist = false;
        }
        yamlcelltypestring += "]\n";
      }
      if (not nodes_exist)
      {
        std::cout << "line nodes are not contained\n";
        continue;
      }
      // 3. information: surface vectors of internal node numbers (for 3D elements)
      if (Core::FE::get_dimension(celltype) == 3)
      {
        std::vector<std::vector<int>> surfacevector;
        try
        {
          surfacevector = Core::FE::get_ele_node_numbering_surfaces(celltype);
        }
        catch (...)
        {
          std::cout << "could not read surfaces\n";
          continue;
        }
        yamlcelltypestring += "  surfaces:\n";
        for (auto surface : surfacevector)
        {
          yamlcelltypestring += "    - [";
          for (size_t indx = 0; indx < surface.size(); ++indx)
          {
            if (indx > 0) yamlcelltypestring += ",";
            yamlcelltypestring += std::format("{:3d}", surface[indx]);
            if ((unsigned)surface[indx] >= num_nodes) nodes_exist = false;
          }
          yamlcelltypestring += "]\n";
        }
        if (not nodes_exist)
        {
          std::cout << "surface nodes are not contained\n";
          continue;
        }
        // 4. information: vector of number of nodes for all surfaces
        std::vector<int> surfacecorners;
        try
        {
          surfacecorners = Core::FE::get_number_of_face_element_corner_nodes(celltype);
        }
        catch (...)
        {
          std::cout << "could not read surface corners\n";
          continue;
        }
        yamlcelltypestring += "  surfacecorners: [";
        for (size_t indx = 0; indx < surfacecorners.size(); ++indx)
        {
          if (indx > 0) yamlcelltypestring += ",";
          yamlcelltypestring += std::format("{:3d}", surfacecorners[indx]);
        }
        yamlcelltypestring += "]\n";
      }
      std::cout << "Writing information on cell type " << celltypename << " to yaml file\n";
      yamlfile << yamlcelltypestring;
    }
  }
  void replace_restructuredtext_keys(std::string& documentation_string)
  {
    size_t mathstartpos = documentation_string.find("$");
    size_t mathendpos = 0;
    while (mathstartpos != documentation_string.npos)
    {
      mathendpos = documentation_string.find("$", mathstartpos + 1);
      if (mathendpos == documentation_string.npos)
      {
        FOUR_C_THROW(
            "Math tags in a ReadTheDocs paragraph must occur pairwise. "
            "Error found in: {}\n",
            documentation_string);
      }
      documentation_string.replace(mathendpos, 1, "`");
      documentation_string.replace(mathstartpos, 1, ":math:`");
      mathstartpos = documentation_string.find("$");
    }
  }

}  // namespace RTD
FOUR_C_NAMESPACE_CLOSE
