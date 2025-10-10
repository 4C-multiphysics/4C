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
#include "4C_global_legacy_module_validconditions.hpp"
#include "4C_global_legacy_module_validmaterials.hpp"
#include "4C_global_legacy_module_validparameters.hpp"
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
      FOUR_C_THROW("failed to open file: {}", elementinformationfilename);
    elementinformationfile << "# yaml file created using 4C version (git SHA1):\n";
    elementinformationfile << "# " << VersionControl::git_hash << "\n#\n";

    write_yaml_cell_type_information(elementinformationfile);
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void write_read_the_docs_celltypes(const std::string& celltypedocumentationfilename)
  {
    // open ascii file for writing all header parameters
    std::ofstream celltypeocumentationfile(celltypedocumentationfilename.c_str());
    if (!celltypeocumentationfile)
      FOUR_C_THROW("failed to open file: {}", celltypedocumentationfilename);
    celltypeocumentationfile << "..\n   Created using 4C version (git SHA1):\n";
    celltypeocumentationfile << "   " << VersionControl::git_hash << "\n\n";

    write_celltype_reference(celltypeocumentationfile);
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
