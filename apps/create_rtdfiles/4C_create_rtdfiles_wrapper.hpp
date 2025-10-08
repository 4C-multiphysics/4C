// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CREATE_RTDFILES_WRAPPER_HPP
#define FOUR_C_CREATE_RTDFILES_WRAPPER_HPP
#include "4C_config.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

namespace RTD
{
  /*!
    \brief Create a yaml file containing the cell type information

    \param[in] string: filename for the the yaml file, should be elementinformation.yaml
  */
  void write_cell_type_information(const std::string& elementinformationfilename);


  /*!
    \brief Create a restructuredText file containing the cell type section
      including node numbers, etc.
      TODO face IDs, etc.

    \param[in] string: filename for the the header parameters, should be celltypereference.rst

  */
  void write_read_the_docs_celltypes(const std::string& celltypedocumentationfilename);


  void print_help_message();

}  // namespace RTD

FOUR_C_NAMESPACE_CLOSE

#endif
