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

#include <boost/algorithm/string.hpp>
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
          "Trying to add %i row elements into a table with %i rows", row.size(), tablewidth_);
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
          "Number of given cell widths (%i) "
          "does not correspond to the number of rows (%i)",
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
      FOUR_C_THROW("Header level for ReadTheDocs output must be [0,3], but is %i", level);
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
            "Error found in: \n" +
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

        std::string celltypelinkname = boost::algorithm::to_lower_copy(celltypename);
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

  void write_single_material_read_the_docs(
      std::ostream& stream, const Core::IO::InputSpec& material)
  {
    /* Each entry consists of a number of fields:
    - header
    - description
    - code line
    - parameter description */

    // the Material title
    write_linktarget(stream, material.impl().name());
    write_header(stream, 1, material.impl().name());

    // the description of the material
    std::string materialDescription = material.impl().description();
    write_paragraph(stream, materialDescription);

    std::stringstream specs_string;
    material.print_as_dat(specs_string);

    // Split on newline because this is what the write_code function expects
    std::vector<std::string> specs_list = Core::Utils::split(specs_string.str(), "\n");

    write_code(stream, specs_list);
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_material_reference(std::ostream& stream,
      const std::unordered_map<Core::Materials::MaterialType, Core::IO::InputSpec>& materials)
  {
    write_linktarget(stream, "materialsreference");
    write_header(stream, 0, "Material reference");

    std::vector<std::string> materialsectionstring{std::string(58, '-') + "MATERIALS"};
    write_code(stream, materialsectionstring);

    for (auto& material : materials)
    {
      write_single_material_read_the_docs(stream, material.second);
    }
    //
    // adding the section for the CLONING MATERIAL MAP
    write_linktarget(stream, "cloningmaterialsreference");
    write_header(stream, 0, "Cloning material reference");
    write_paragraph(stream,
        "This section is used for multi physics simulations, where one wants to discretize two "
        "different fields on the same mesh. "
        "Instead of creating the same mesh twice, the user only needs to create it once. "
        "The pre-defined mesh is read in and results in a discretization object with material "
        "SRC_MAT. "
        "This discretization is then cloned/duplicated such that the resulting discretization "
        "is assigned the material TAR_MAT.");

    const auto spec = Core::FE::valid_cloning_material_map();
    std::stringstream cloningMatStream;
    Core::IO::print_section(cloningMatStream, "CLONING MATERIAL MAP", spec);
    const std::vector<std::string> cloningMatList =
        Core::Utils::split(cloningMatStream.str(), "\n");

    write_code(stream, cloningMatList);
  }


  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_header_reference(
      std::ostream& stream, const Teuchos::ParameterList& list, std::string parentname)
  {
    // prevent invalid ordering of parameters caused by alphabetical output:
    // in the first run, print out all list elements that are not a sublist
    // in the second run, do the recursive call for all the sublists in the list

    for (int j = 0; j < 2; ++j)
    {
      // bool loop_isEntry = (j == 0);
      bool loop_isList = (j == 1);
      for (Teuchos::ParameterList::ConstIterator it = list.begin(); it != list.end(); ++it)
      {
        const Teuchos::ParameterEntry& entry = list.entry(it);
        if (entry.isList() != loop_isList) continue;
        const std::string& name = list.name(it);
        Teuchos::RCP<const Teuchos::ParameterEntryValidator> validator = entry.validator();

        std::string doc = (entry.docString() == "") ? "no description yet" : entry.docString();

        std::string fullname = parentname;
        bool issubsection = false;
        if (fullname != "")
        {
          fullname += "/";
          issubsection = true;
        }
        fullname += name;
        std::string linktarget = boost::algorithm::replace_all_copy(fullname, "/", "_");
        linktarget = Teuchos::StrUtils::removeAllSpaces(Core::Utils::to_lower(linktarget));

        if (entry.isList())  // it is a section header
        {
          unsigned l = fullname.length();
          // write link:
          write_linktarget(stream, "SEC" + linktarget);
          // write section header
          unsigned level = (issubsection) ? 2 : 1;
          write_header(stream, level, fullname);

          write_paragraph(stream, doc);

          std::vector<std::string> codelines;
          codelines.push_back("--" + std::string(std::max<int>(65 - l, 0), '-') + fullname);
          write_code(stream, codelines);

          if (Core::IO::need_to_print_equal_sign(list.sublist(name)))
          {
            write_note(stream,
                "   The parameters in this section need an equal sign (=) "
                "between the parameter name and its value!");
          }

          write_header_reference(stream, list.sublist(name), fullname);
        }
        else  // it is a parameter entry
        {
          write_linktarget(stream, linktarget);

          const Teuchos::any& v = entry.getAny(false);

          std::string s =
              std::format("**{}** | *default:* {} |break| {}", name, Teuchos::toString(v), doc);
          write_paragraph(stream, s);
          if (validator != Teuchos::null)  // it can only take specific values
          {
            Teuchos::RCP<const Teuchos::Array<std::string>> values = validator->validStringValues();
            if (values != Teuchos::null)
            {
              stream << "   **Possible values:**\n\n";
              for (auto val : *values) stream << "   - " << val << "\n";
              stream << "\n";
            }
          }
        }
      }
    }
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_conditions_reference(
      std::ostream& stream, const std::vector<Core::Conditions::ConditionDefinition>& condlist)
  {
    write_linktarget(stream, "prescribedconditionreference");
    write_header(stream, 0, "Prescribed Condition Reference");

    for (const auto& condition : condlist)
    {
      write_single_condition_read_the_docs(stream, condition);
    }
  }


  void write_single_condition_read_the_docs(
      std::ostream& stream, const Core::Conditions::ConditionDefinition& condition)
  {
    std::string sectionname = condition.section_name();
    const std::string sectionlinktarget =
        Teuchos::StrUtils::removeAllSpaces(Core::Utils::to_lower(sectionname));
    //
    // boundary condition header
    //
    /*------ PART 1 --------------------------
     * Boundary condition header (incl. link target)
     */
    // link target line
    write_linktarget(stream, sectionlinktarget);
    // condition name as section header
    write_header(stream, 1, sectionname);

    /*------ PART 2 -------------------------
     * boundary condition description string
     */
    std::string descriptionline =
        (condition.description() == "") ? "no description yet" : condition.description();
    write_paragraph(stream, descriptionline);

    /*------ PART 3 -------------------------
     * boundary condition input lines
     * In this section, the table for parameter description is filled as well.
     */
    // First line: condition name as a section
    unsigned l = sectionname.length();
    std::vector<std::string> conditioncode{
        "--" + std::string(std::max<int>(65 - l, 0), '-') + sectionname};
    write_code(stream, conditioncode);

    // Avoid writing an empty code block if there are no specs for this condition
    if (!condition.specs().empty())
    {
      auto cond_spec = Core::IO::InputSpecBuilders::all_of(condition.specs());

      std::stringstream specs_string;
      cond_spec.print_as_dat(specs_string);

      // Split on newline because this is what the write_code function expects
      std::vector<std::string> specs_list = Core::Utils::split(specs_string.str(), "\n");

      write_code(stream, specs_list);
    }
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_contact_law_reference(std::ostream& stream, const Core::IO::InputSpec& specs)
  {
    write_linktarget(stream, "contactconstitutivelawreference");
    write_header(stream, 0, "Contact Constitutive Law Reference");


    std::vector<std::string> contactlawsectionstring{
        std::string(43, '-') + "CONTACT CONSTITUTIVE LAW"};
    write_code(stream, contactlawsectionstring);

    std::stringstream specs_string;
    specs.print_as_dat(specs_string);

    // Split on newline because this is what the write_code function expects
    std::vector<std::string> specs_list = Core::Utils::split(specs_string.str(), "\n");

    write_code(stream, specs_list);
  }


  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_various_reference(std::ostream& stream)
  {
    //
    // adding the sections for the RESULT DESCRIPTION
    {
      write_linktarget(stream, "restultdescriptionreference");
      write_header(stream, 0, "Result description reference");

      auto result_spec = global_legacy_module_callbacks().valid_result_description_lines();
      write_paragraph(stream,
          "The result of the simulation with respect to specific quantities at concrete points "
          "can be tested against particular values with a given tolerance.");
      std::stringstream resultDescriptionStream;
      Core::IO::print_section(resultDescriptionStream, "RESULT DESCRIPTION", result_spec);
      const std::vector<std::string> resultDescriptionList =
          Core::Utils::split(resultDescriptionStream.str(), "\n");
      write_code(stream, resultDescriptionList);
    }
    //
    // adding the sections for the FUNCTION
    {
      write_linktarget(stream, "functionreference");
      write_header(stream, 0, "Functions reference");
      Core::Utils::FunctionManager function_manager;
      global_legacy_module_callbacks().AttachFunctionDefinitions(function_manager);

      const auto lines = function_manager.valid_function_lines();

      write_paragraph(
          stream, "Definition of functions for various cases, mainly boundary conditions");
      std::stringstream functionStream;
      Core::IO::print_section(functionStream, "FUNCT", lines);
      const std::vector<std::string> functionList = Core::Utils::split(functionStream.str(), "\n");
      write_code(stream, functionList);
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
            "Error found in: \n" +
            documentation_string);
      }
      documentation_string.replace(mathendpos, 1, "`");
      documentation_string.replace(mathstartpos, 1, ":math:`");
      mathstartpos = documentation_string.find("$");
    }
  }

}  // namespace RTD
FOUR_C_NAMESPACE_CLOSE
