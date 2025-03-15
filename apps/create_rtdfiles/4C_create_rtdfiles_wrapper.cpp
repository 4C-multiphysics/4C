// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config_revision.hpp"

#include "4C_create_rtdfiles_wrapper.hpp"

#include "4C_comm_utils.hpp"
#include "4C_contact_constitutivelaw_valid_laws.hpp"
#include "4C_create_rtdfiles_utils.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_global_legacy_module_validmaterials.hpp"
#include "4C_inpar_validconditions.hpp"
#include "4C_inpar_validparameters.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_string.hpp"

#include <Teuchos_StrUtils.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN

namespace RTD
{
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_cell_type_information(const std::string& elementinformationfilename)
  {
    // open ascii file for writing the cell type information
    std::ofstream elementinformationfile(elementinformationfilename.c_str());
    if (!elementinformationfile)
      FOUR_C_THROW("failed to open file: %s", elementinformationfilename.c_str());
    elementinformationfile << "# yaml file created using 4C version (git SHA1):\n";
    elementinformationfile << "# " << VersionControl::git_hash << "\n#\n";

    write_yaml_cell_type_information(elementinformationfile);
  }


  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_read_the_docs_header(const std::string& headerdocumentationfilename)
  {
    // open ascii file for writing all header parameters
    std::ofstream headerdocumentationfile(headerdocumentationfilename.c_str());
    if (!headerdocumentationfile)
      FOUR_C_THROW("failed to open file: %s", headerdocumentationfilename.c_str());
    headerdocumentationfile << "..\n   Created using 4C version (git SHA1):\n";
    headerdocumentationfile << "   " << VersionControl::git_hash << "\n\n";
    headerdocumentationfile << ".. _headerparameters:\n\n";
    headerdocumentationfile << "Header parameters\n";
    headerdocumentationfile << "=================\n\n";
    auto parameters = Input::valid_parameters();

    for (const auto& [section_name, spec] : parameters)
    {
      std::string linktarget = boost::algorithm::replace_all_copy(section_name, "/", "_");
      linktarget = Teuchos::StrUtils::removeAllSpaces(Core::Utils::to_lower(linktarget));

      // write link:
      write_linktarget(headerdocumentationfile, "SEC" + linktarget);
      // write section header
      write_header(headerdocumentationfile, 1, section_name);

      headerdocumentationfile << section_name << "\n";
      std::stringstream ss;
      spec.print_as_dat(ss);

      // Split on newline because this is what the write_code function expects
      std::vector<std::string> specs_list = Core::Utils::split(ss.str(), "\n");
      write_code(headerdocumentationfile, specs_list);
    }
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_read_the_docs_celltypes(const std::string& celltypedocumentationfilename)
  {
    // open ascii file for writing all header parameters
    std::ofstream celltypeocumentationfile(celltypedocumentationfilename.c_str());
    if (!celltypeocumentationfile)
      FOUR_C_THROW("failed to open file: %s", celltypedocumentationfilename.c_str());
    celltypeocumentationfile << "..\n   Created using 4C version (git SHA1):\n";
    celltypeocumentationfile << "   " << VersionControl::git_hash << "\n\n";

    write_celltype_reference(celltypeocumentationfile);
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_read_the_docs_material(const std::string& materialdocumentationfilename)
  {
    //
    // open ascii file for writing all material parameters
    std::ofstream materialdocumentationfile(materialdocumentationfilename.c_str());
    if (!materialdocumentationfile)
      FOUR_C_THROW("failed to open file: %s", materialdocumentationfilename.c_str());
    materialdocumentationfile << "..\n   Created using 4C version (git SHA1):\n";
    materialdocumentationfile << "   " << VersionControl::git_hash << "\n\n";
    write_material_reference(materialdocumentationfile, Global::valid_materials());
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_read_the_docs_condition(const std::string& conditiondocumentationfilename)
  {
    //
    // open ascii file for writing all constrains / conditions parameters
    std::ofstream conditiondocumentationfile(conditiondocumentationfilename.c_str());
    if (!conditiondocumentationfile)
      FOUR_C_THROW("failed to open file: %s", conditiondocumentationfilename.c_str());
    conditiondocumentationfile << "..\n   Created using 4C version (git SHA1):\n";
    conditiondocumentationfile << "   " << VersionControl::git_hash << "\n\n";
    write_conditions_reference(conditiondocumentationfile, Input::valid_conditions());

    write_contact_law_reference(
        conditiondocumentationfile, CONTACT::CONSTITUTIVELAW::valid_contact_constitutive_laws());
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_read_the_docs_various(const std::string& variousdocumentationfilename)
  {
    //
    // open ascii file for writing other (non header) parameters
    std::ofstream variousdocumentationfile(variousdocumentationfilename.c_str());
    if (!variousdocumentationfile)
      FOUR_C_THROW("failed to open file: %s", variousdocumentationfilename.c_str());
    variousdocumentationfile << "..\n   Created using 4C version (git SHA1):\n";
    variousdocumentationfile << "   " << VersionControl::git_hash << "\n\n";
    write_various_reference(variousdocumentationfile);
  }

  void print_help_message()
  {
    std::cout << "This program writes all necessary reference files for the main documentation.\n";
    std::cout << "Usage:\n    create_rtd [pathanem]\n";
    std::cout << " Parameter:\n   pathname (str) path where the reference files are stored.\n";
    std::cout << "                   Default: reference_docs";
  }

}  // namespace RTD

FOUR_C_NAMESPACE_CLOSE
