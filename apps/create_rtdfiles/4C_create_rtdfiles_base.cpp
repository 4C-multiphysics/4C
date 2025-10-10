// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_create_rtdfiles_wrapper.hpp"

#include <mpi.h>

#include <cstring>
#include <filesystem>
#include <iostream>


int main(int argc, char* argv[])
{
  using namespace FourC;

  MPI_Init(&argc, &argv);

  printf(
      "\n"
      "**********************************************\n"
      "*                    4C                      *\n"
      "*          Reference files creator           *\n"
      "*              for ReadTheDocs               *\n"
      "**********************************************\n\n");

  if ((argc == 2) && ((strcmp(argv[1], "-h") == 0) || (strcmp(argv[1], "--help") == 0)))
  {
    printf("\n\n");
    RTD::print_help_message();
    printf("\n\n");
  }
  else
  {
    std::string reference_path = (argc == 2) ? argv[1] : "reference_docs";
    if (not std::filesystem::exists(reference_path))
      std::filesystem::create_directory(reference_path);

    RTD::write_cell_type_information(reference_path + "/elementinformation.yaml");
    std::cout << "Writing cell type information to yaml finished\n";
    std::cout << "Writing headerreference.rst finished\n";
    RTD::write_read_the_docs_celltypes(reference_path + "/celltypereference.rst");
    std::cout << "Writing celltypes.rst finished\n";
  }

  MPI_Finalize();

  return (0);
}
