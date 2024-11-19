// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_dofset.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_integrator.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_mortar_node.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Visualize mortar stuff with gmsh                          popp 08/08|
 *----------------------------------------------------------------------*/
void Mortar::Interface::visualize_gmsh(
    const int step, const int iter, const std::string& file_name_only_prefix) const
{
  //**********************************************************************
  // GMSH output of all interface elements
  //**********************************************************************
  // construct unique filename for gmsh output
  // basic information
  std::ostringstream filename;
  filename << "o/gmsh_output/" << file_name_only_prefix << "_mt_id";
  if (id_ < 10)
    filename << 0;
  else if (id_ > 99)
    FOUR_C_THROW("Gmsh output implemented for a maximum of 99 iterations");
  filename << id_;

  // construct unique filename for gmsh output
  // first index = time step index
  filename << "_step";
  if (step < 10)
    filename << 0 << 0 << 0 << 0;
  else if (step < 100)
    filename << 0 << 0 << 0;
  else if (step < 1000)
    filename << 0 << 0;
  else if (step < 10000)
    filename << 0;
  else if (step > 99999)
    FOUR_C_THROW("Gmsh output implemented for a maximum of 99.999 time steps");
  filename << step;

  // construct unique filename for gmsh output
  // second index = Newton iteration index
  filename << "_iter";
  if (iter >= 0)
  {
    if (iter < 10)
      filename << 0;
    else if (iter > 99)
      FOUR_C_THROW("Gmsh output implemented for a maximum of 99 iterations");
    filename << iter;
  }
  else
    filename << "XX";

  // create three files (slave, master and whole interface)
  std::ostringstream filenameslave;
  std::ostringstream filenamemaster;
  filenameslave << filename.str();
  filenamemaster << filename.str();
  filename << "_if.pos";
  filenameslave << "_sl.pos";
  filenamemaster << "_ma.pos";

  // do output to file in c-style
  FILE* fp = nullptr;
  FILE* fps = nullptr;
  FILE* fpm = nullptr;

  //**********************************************************************
  // Start GMSH output
  //**********************************************************************
  for (int proc = 0; proc < get_comm().NumProc(); ++proc)
  {
    if (proc == Core::Communication::my_mpi_rank(get_comm()))
    {
      // open files (overwrite if proc==0, else append)
      if (proc == 0)
      {
        fp = fopen(filename.str().c_str(), "w");
        fps = fopen(filenameslave.str().c_str(), "w");
        fpm = fopen(filenamemaster.str().c_str(), "w");
      }
      else
      {
        fp = fopen(filename.str().c_str(), "a");
        fps = fopen(filenameslave.str().c_str(), "a");
        fpm = fopen(filenamemaster.str().c_str(), "a");
      }

      // write output to temporary std::stringstream
      std::stringstream gmshfilecontent;
      std::stringstream gmshfilecontentslave;
      std::stringstream gmshfilecontentmaster;
      if (proc == 0)
      {
        gmshfilecontent << "View \" Mt-Id " << id_ << " Step " << step << " Iter " << iter
                        << " Iface\" {" << std::endl;
        gmshfilecontentslave << "View \" Mt-Id " << id_ << " Step " << step << " Iter " << iter
                             << " Slave\" {" << std::endl;
        gmshfilecontentmaster << "View \" Mt-Id " << id_ << " Step " << step << " Iter " << iter
                              << " Master\" {" << std::endl;
      }

      //******************************************************************
      // plot elements
      //******************************************************************
      for (int i = 0; i < idiscret_->num_my_row_elements(); ++i)
      {
        Mortar::Element* element = dynamic_cast<Mortar::Element*>(idiscret_->l_row_element(i));
        int nnodes = element->num_node();
        Core::LinAlg::SerialDenseMatrix coord(3, nnodes);
        element->get_nodal_coords(coord);
        double color = (double)element->owner();

        // local center
        double xi[2] = {0.0, 0.0};

        // 2D linear case (2noded line elements)
        if (element->shape() == Core::FE::CellType::line2)
        {
          if (element->is_slave())
          {
            gmshfilecontent << "SL(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "};" << std::endl;
            gmshfilecontentslave << "SL(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                 << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                 << "," << coord(2, 1) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "};"
                                 << std::endl;
          }
          else
          {
            gmshfilecontent << "SL(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "};" << std::endl;
            gmshfilecontentmaster << "SL(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                  << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                  << "," << coord(2, 1) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "};"
                                  << std::endl;
          }
        }

        // 2D quadratic case (3noded line elements)
        if (element->shape() == Core::FE::CellType::line3)
        {
          if (element->is_slave())
          {
            gmshfilecontent << "SL2(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontentslave << "SL2(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                 << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                 << "," << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2)
                                 << "," << coord(2, 2) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "," << color
                                 << "};" << std::endl;
          }
          else
          {
            gmshfilecontent << "SL2(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontentmaster << "SL2(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                  << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                  << "," << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2)
                                  << "," << coord(2, 2) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "," << color
                                  << "};" << std::endl;
          }
        }

        // 3D linear case (3noded triangular elements)
        if (element->shape() == Core::FE::CellType::tri3)
        {
          if (element->is_slave())
          {
            gmshfilecontent << "ST(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontentslave << "ST(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                 << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                 << "," << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2)
                                 << "," << coord(2, 2) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "," << color
                                 << "};" << std::endl;
          }
          else
          {
            gmshfilecontent << "ST(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontentmaster << "ST(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                  << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                  << "," << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2)
                                  << "," << coord(2, 2) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "," << color
                                  << "};" << std::endl;
          }
          xi[0] = 1.0 / 3;
          xi[1] = 1.0 / 3;
        }

        // 3D bilinear case (4noded quadrilateral elements)
        if (element->shape() == Core::FE::CellType::quad4)
        {
          if (element->is_slave())
          {
            gmshfilecontent << "SQ(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3) << ","
                            << coord(2, 3) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "," << color << "};" << std::endl;
            gmshfilecontentslave << "SQ(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                 << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                 << "," << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2)
                                 << "," << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3)
                                 << "," << coord(2, 3) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "," << color
                                 << "," << color << "};" << std::endl;
          }
          else
          {
            gmshfilecontent << "SQ(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3) << ","
                            << coord(2, 3) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "," << color << "};" << std::endl;
            gmshfilecontentmaster << "SQ(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                  << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                  << "," << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2)
                                  << "," << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3)
                                  << "," << coord(2, 3) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "," << color
                                  << "," << color << "};" << std::endl;
          }
        }

        // 3D quadratic case (6noded triangular elements)
        if (element->shape() == Core::FE::CellType::tri6)
        {
          if (element->is_slave())
          {
            gmshfilecontent << "ST2(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3) << ","
                            << coord(2, 3) << "," << coord(0, 4) << "," << coord(1, 4) << ","
                            << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5) << ","
                            << coord(2, 5) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "," << color << "," << color << "," << color << "};" << std::endl;
            gmshfilecontentslave << "ST2(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                 << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                 << "," << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2)
                                 << "," << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3)
                                 << "," << coord(2, 3) << "," << coord(0, 4) << "," << coord(1, 4)
                                 << "," << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5)
                                 << "," << coord(2, 5) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "," << color
                                 << "," << color << "," << color << "," << color << "};"
                                 << std::endl;
          }
          else
          {
            gmshfilecontent << "ST2(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3) << ","
                            << coord(2, 3) << "," << coord(0, 4) << "," << coord(1, 4) << ","
                            << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5) << ","
                            << coord(2, 5) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "," << color << "," << color << "," << color << "};" << std::endl;
            gmshfilecontentmaster << "ST2(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                  << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                  << "," << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2)
                                  << "," << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3)
                                  << "," << coord(2, 3) << "," << coord(0, 4) << "," << coord(1, 4)
                                  << "," << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5)
                                  << "," << coord(2, 5) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "," << color
                                  << "," << color << "," << color << "," << color << "};"
                                  << std::endl;
          }
          xi[0] = 1.0 / 3;
          xi[1] = 1.0 / 3;
        }

        // 3D serendipity case (8noded quadrilateral elements)
        if (element->shape() == Core::FE::CellType::quad8)
        {
          if (element->is_slave())
          {
            gmshfilecontent << "ST(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 4) << "," << coord(1, 4) << ","
                            << coord(2, 4) << "," << coord(0, 7) << "," << coord(1, 7) << ","
                            << coord(2, 7) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontent << "ST(" << std::scientific << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 5) << "," << coord(1, 5) << ","
                            << coord(2, 5) << "," << coord(0, 4) << "," << coord(1, 4) << ","
                            << coord(2, 4) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontent << "ST(" << std::scientific << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << "," << coord(0, 6) << "," << coord(1, 6) << ","
                            << coord(2, 6) << "," << coord(0, 5) << "," << coord(1, 5) << ","
                            << coord(2, 5) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontent << "ST(" << std::scientific << coord(0, 3) << "," << coord(1, 3) << ","
                            << coord(2, 3) << "," << coord(0, 7) << "," << coord(1, 7) << ","
                            << coord(2, 7) << "," << coord(0, 6) << "," << coord(1, 6) << ","
                            << coord(2, 6) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontent << "SQ(" << std::scientific << coord(0, 4) << "," << coord(1, 4) << ","
                            << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5) << ","
                            << coord(2, 5) << "," << coord(0, 6) << "," << coord(1, 6) << ","
                            << coord(2, 6) << "," << coord(0, 7) << "," << coord(1, 7) << ","
                            << coord(2, 7) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "," << color << "};" << std::endl;
            gmshfilecontentslave << "ST(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                 << "," << coord(2, 0) << "," << coord(0, 4) << "," << coord(1, 4)
                                 << "," << coord(2, 4) << "," << coord(0, 7) << "," << coord(1, 7)
                                 << "," << coord(2, 7) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "," << color
                                 << "};" << std::endl;
            gmshfilecontentslave << "ST(" << std::scientific << coord(0, 1) << "," << coord(1, 1)
                                 << "," << coord(2, 1) << "," << coord(0, 5) << "," << coord(1, 5)
                                 << "," << coord(2, 5) << "," << coord(0, 4) << "," << coord(1, 4)
                                 << "," << coord(2, 4) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "," << color
                                 << "};" << std::endl;
            gmshfilecontentslave << "ST(" << std::scientific << coord(0, 2) << "," << coord(1, 2)
                                 << "," << coord(2, 2) << "," << coord(0, 6) << "," << coord(1, 6)
                                 << "," << coord(2, 6) << "," << coord(0, 5) << "," << coord(1, 5)
                                 << "," << coord(2, 5) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "," << color
                                 << "};" << std::endl;
            gmshfilecontentslave << "ST(" << std::scientific << coord(0, 3) << "," << coord(1, 3)
                                 << "," << coord(2, 3) << "," << coord(0, 7) << "," << coord(1, 7)
                                 << "," << coord(2, 7) << "," << coord(0, 6) << "," << coord(1, 6)
                                 << "," << coord(2, 6) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "," << color
                                 << "};" << std::endl;
            gmshfilecontentslave << "SQ(" << std::scientific << coord(0, 4) << "," << coord(1, 4)
                                 << "," << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5)
                                 << "," << coord(2, 5) << "," << coord(0, 6) << "," << coord(1, 6)
                                 << "," << coord(2, 6) << "," << coord(0, 7) << "," << coord(1, 7)
                                 << "," << coord(2, 7) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "," << color
                                 << "," << color << "};" << std::endl;
          }
          else
          {
            gmshfilecontent << "ST(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 4) << "," << coord(1, 4) << ","
                            << coord(2, 4) << "," << coord(0, 7) << "," << coord(1, 7) << ","
                            << coord(2, 7) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontent << "ST(" << std::scientific << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 5) << "," << coord(1, 5) << ","
                            << coord(2, 5) << "," << coord(0, 4) << "," << coord(1, 4) << ","
                            << coord(2, 4) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontent << "ST(" << std::scientific << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << "," << coord(0, 6) << "," << coord(1, 6) << ","
                            << coord(2, 6) << "," << coord(0, 5) << "," << coord(1, 5) << ","
                            << coord(2, 5) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontent << "ST(" << std::scientific << coord(0, 3) << "," << coord(1, 3) << ","
                            << coord(2, 3) << "," << coord(0, 7) << "," << coord(1, 7) << ","
                            << coord(2, 7) << "," << coord(0, 6) << "," << coord(1, 6) << ","
                            << coord(2, 6) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "};" << std::endl;
            gmshfilecontent << "SQ(" << std::scientific << coord(0, 4) << "," << coord(1, 4) << ","
                            << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5) << ","
                            << coord(2, 5) << "," << coord(0, 6) << "," << coord(1, 6) << ","
                            << coord(2, 6) << "," << coord(0, 7) << "," << coord(1, 7) << ","
                            << coord(2, 7) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "," << color << "};" << std::endl;
            gmshfilecontentmaster << "ST(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                  << "," << coord(2, 0) << "," << coord(0, 4) << "," << coord(1, 4)
                                  << "," << coord(2, 4) << "," << coord(0, 7) << "," << coord(1, 7)
                                  << "," << coord(2, 7) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "," << color
                                  << "};" << std::endl;
            gmshfilecontentmaster << "ST(" << std::scientific << coord(0, 1) << "," << coord(1, 1)
                                  << "," << coord(2, 1) << "," << coord(0, 5) << "," << coord(1, 5)
                                  << "," << coord(2, 5) << "," << coord(0, 4) << "," << coord(1, 4)
                                  << "," << coord(2, 4) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "," << color
                                  << "};" << std::endl;
            gmshfilecontentmaster << "ST(" << std::scientific << coord(0, 2) << "," << coord(1, 2)
                                  << "," << coord(2, 2) << "," << coord(0, 6) << "," << coord(1, 6)
                                  << "," << coord(2, 6) << "," << coord(0, 5) << "," << coord(1, 5)
                                  << "," << coord(2, 5) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "," << color
                                  << "};" << std::endl;
            gmshfilecontentmaster << "ST(" << std::scientific << coord(0, 3) << "," << coord(1, 3)
                                  << "," << coord(2, 3) << "," << coord(0, 7) << "," << coord(1, 7)
                                  << "," << coord(2, 7) << "," << coord(0, 6) << "," << coord(1, 6)
                                  << "," << coord(2, 6) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "," << color
                                  << "};" << std::endl;
            gmshfilecontentmaster << "SQ(" << std::scientific << coord(0, 4) << "," << coord(1, 4)
                                  << "," << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5)
                                  << "," << coord(2, 5) << "," << coord(0, 6) << "," << coord(1, 6)
                                  << "," << coord(2, 6) << "," << coord(0, 7) << "," << coord(1, 7)
                                  << "," << coord(2, 7) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "," << color
                                  << "," << color << "};" << std::endl;
          }
        }

        // 3D biquadratic case (9noded quadrilateral elements)
        if (element->shape() == Core::FE::CellType::quad9)
        {
          if (element->is_slave())
          {
            gmshfilecontent << "SQ2(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3) << ","
                            << coord(2, 3) << "," << coord(0, 4) << "," << coord(1, 4) << ","
                            << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5) << ","
                            << coord(2, 5) << "," << coord(0, 6) << "," << coord(1, 6) << ","
                            << coord(2, 6) << "," << coord(0, 7) << "," << coord(1, 7) << ","
                            << coord(2, 7) << "," << coord(0, 8) << "," << coord(1, 8) << ","
                            << coord(2, 8) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "," << color << "," << color << "," << color << "," << color << ","
                            << color << "," << color << "};" << std::endl;
            gmshfilecontentslave << "SQ2(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                 << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                 << "," << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2)
                                 << "," << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3)
                                 << "," << coord(2, 3) << "," << coord(0, 4) << "," << coord(1, 4)
                                 << "," << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5)
                                 << "," << coord(2, 5) << "," << coord(0, 6) << "," << coord(1, 6)
                                 << "," << coord(2, 6) << "," << coord(0, 7) << "," << coord(1, 7)
                                 << "," << coord(2, 7) << "," << coord(0, 8) << "," << coord(1, 8)
                                 << "," << coord(2, 8) << ")";
            gmshfilecontentslave << "{" << std::scientific << color << "," << color << "," << color
                                 << "," << color << "," << color << "," << color << "," << color
                                 << "," << color << "," << color << "};" << std::endl;
          }
          else
          {
            gmshfilecontent << "SQ2(" << std::scientific << coord(0, 0) << "," << coord(1, 0) << ","
                            << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1) << ","
                            << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2) << ","
                            << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3) << ","
                            << coord(2, 3) << "," << coord(0, 4) << "," << coord(1, 4) << ","
                            << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5) << ","
                            << coord(2, 5) << "," << coord(0, 6) << "," << coord(1, 6) << ","
                            << coord(2, 6) << "," << coord(0, 7) << "," << coord(1, 7) << ","
                            << coord(2, 7) << "," << coord(0, 8) << "," << coord(1, 8) << ","
                            << coord(2, 8) << ")";
            gmshfilecontent << "{" << std::scientific << color << "," << color << "," << color
                            << "," << color << "," << color << "," << color << "," << color << ","
                            << color << "," << color << "};" << std::endl;
            gmshfilecontentmaster << "SQ2(" << std::scientific << coord(0, 0) << "," << coord(1, 0)
                                  << "," << coord(2, 0) << "," << coord(0, 1) << "," << coord(1, 1)
                                  << "," << coord(2, 1) << "," << coord(0, 2) << "," << coord(1, 2)
                                  << "," << coord(2, 2) << "," << coord(0, 3) << "," << coord(1, 3)
                                  << "," << coord(2, 3) << "," << coord(0, 4) << "," << coord(1, 4)
                                  << "," << coord(2, 4) << "," << coord(0, 5) << "," << coord(1, 5)
                                  << "," << coord(2, 5) << "," << coord(0, 6) << "," << coord(1, 6)
                                  << "," << coord(2, 6) << "," << coord(0, 7) << "," << coord(1, 7)
                                  << "," << coord(2, 7) << "," << coord(0, 8) << "," << coord(1, 8)
                                  << "," << coord(2, 8) << ")";
            gmshfilecontentmaster << "{" << std::scientific << color << "," << color << "," << color
                                  << "," << color << "," << color << "," << color << "," << color
                                  << "," << color << "," << color << "};" << std::endl;
          }
        }

        // plot element number in element center
        double elec[3];
        element->local_to_global(xi, elec, 0);

        if (element->is_slave())
        {
          gmshfilecontent << "T3(" << std::scientific << elec[0] << "," << elec[1] << "," << elec[2]
                          << "," << 17 << ")";
          gmshfilecontent << "{\""
                          << "S" << element->id() << "\"};" << std::endl;
          gmshfilecontentslave << "T3(" << std::scientific << elec[0] << "," << elec[1] << ","
                               << elec[2] << "," << 17 << ")";
          gmshfilecontentslave << "{\""
                               << "S" << element->id() << "\"};" << std::endl;
        }
        else
        {
          gmshfilecontent << "T3(" << std::scientific << elec[0] << "," << elec[1] << "," << elec[2]
                          << "," << 17 << ")";
          gmshfilecontent << "{\""
                          << "M" << element->id() << "\"};" << std::endl;
          gmshfilecontentmaster << "T3(" << std::scientific << elec[0] << "," << elec[1] << ","
                                << elec[2] << "," << 17 << ")";
          gmshfilecontentmaster << "{\""
                                << "M" << element->id() << "\"};" << std::endl;
        }

        // plot node numbers at the nodes
        for (int j = 0; j < nnodes; ++j)
        {
          if (element->is_slave())
          {
            gmshfilecontent << "T3(" << std::scientific << coord(0, j) << "," << coord(1, j) << ","
                            << coord(2, j) << "," << 17 << ")";
            gmshfilecontent << "{\""
                            << "SN" << element->node_ids()[j] << "\"};" << std::endl;
            gmshfilecontentslave << "T3(" << std::scientific << coord(0, j) << "," << coord(1, j)
                                 << "," << coord(2, j) << "," << 17 << ")";
            gmshfilecontentslave << "{\""
                                 << "SN" << element->node_ids()[j] << "\"};" << std::endl;
          }
          else
          {
            gmshfilecontent << "T3(" << std::scientific << coord(0, j) << "," << coord(1, j) << ","
                            << coord(2, j) << "," << 17 << ")";
            gmshfilecontent << "{\""
                            << "MN" << element->node_ids()[j] << "\"};" << std::endl;
            gmshfilecontentmaster << "T3(" << std::scientific << coord(0, j) << "," << coord(1, j)
                                  << "," << coord(2, j) << "," << 17 << ")";
            gmshfilecontentmaster << "{\""
                                  << "MN" << element->node_ids()[j] << "\"};" << std::endl;
          }
        }
      }

      //******************************************************************
      // plot normal vectors
      //******************************************************************
      for (int i = 0; i < snoderowmap_->NumMyElements(); ++i)
      {
        int gid = snoderowmap_->GID(i);
        Core::Nodes::Node* node = idiscret_->g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        Node* mtrnode = dynamic_cast<Node*>(node);
        if (!mtrnode) FOUR_C_THROW("Static Cast to Node* failed");

        double nc[3];
        double nn[3];

        for (int j = 0; j < 3; ++j)
        {
          nc[j] = mtrnode->xspatial()[j];
          nn[j] = mtrnode->mo_data().n()[j];
        }

        gmshfilecontentslave << "VP(" << std::scientific << nc[0] << "," << nc[1] << "," << nc[2]
                             << ")";
        gmshfilecontentslave << "{" << std::scientific << nn[0] << "," << nn[1] << "," << nn[2]
                             << "};" << std::endl;
      }

      // end GMSH output section in all files
      if (proc == get_comm().NumProc() - 1)
      {
        gmshfilecontent << "};" << std::endl;
        gmshfilecontentslave << "};" << std::endl;
        gmshfilecontentmaster << "};" << std::endl;
      }

      // move everything to gmsh post-processing files and close them
      fprintf(fp, "%s", gmshfilecontent.str().c_str());
      fprintf(fps, "%s", gmshfilecontentslave.str().c_str());
      fprintf(fpm, "%s", gmshfilecontentmaster.str().c_str());
      fclose(fp);
      fclose(fps);
      fclose(fpm);
    }
    get_comm().Barrier();
  }


  //**********************************************************************
  // GMSH output of all treenodes (DOPs) on all layers
  //**********************************************************************
#ifdef MORTARGMSHTN
  // get max. number of layers for every proc.
  // (master elements are equal on each proc)
  int lnslayers = binarytree_->Streenodesmap().size();
  int gnmlayers = binarytree_->Mtreenodesmap().size();
  int gnslayers = 0;
  Comm().MaxAll(&lnslayers, &gnslayers, 1);

  // create files for visualization of slave dops for every layer
  std::ostringstream filenametn;
  filenametn << "o/gmsh_output/" << file_name_only_prefix << "_";

  if (step < 10)
    filenametn << 0 << 0 << 0 << 0;
  else if (step < 100)
    filenametn << 0 << 0 << 0;
  else if (step < 1000)
    filenametn << 0 << 0;
  else if (step < 10000)
    filenametn << 0;
  else if (step > 99999)
    FOUR_C_THROW("Gmsh output implemented for a maximum of 99.999 time steps");
  filenametn << step;

  // construct unique filename for gmsh output
  // second index = Newton iteration index
  if (iter >= 0)
  {
    filenametn << "_";
    if (iter < 10)
      filenametn << 0;
    else if (iter > 99)
      FOUR_C_THROW("Gmsh output implemented for a maximum of 99 iterations");
    filenametn << iter;
  }

  if (Core::Communication::my_mpi_rank(Comm()) == 0)
  {
    for (int i = 0; i < gnslayers; i++)
    {
      std::ostringstream currentfilename;
      currentfilename << filenametn.str().c_str() << "_s_tnlayer_" << i << ".pos";
      // std::cout << std::endl << Core::Communication::my_mpi_rank(Comm())<< "filename: " <<
      // currentfilename.str().c_str();
      fp = fopen(currentfilename.str().c_str(), "w");
      std::stringstream gmshfile;
      gmshfile << "View \" Step " << step << " Iter " << iter << " stl " << i << " \" {"
               << std::endl;
      fprintf(fp, gmshfile.str().c_str());
      fclose(fp);
    }
  }

  Comm().Barrier();

  // for every proc, one after another, put data of slabs into files
  for (int i = 0; i < Comm().NumProc(); i++)
  {
    if ((i == Core::Communication::my_mpi_rank(Comm())) && (binarytree_->Sroot()->Type() != 4))
    {
      // print full tree with treenodesmap
      for (int j = 0; j < (int)binarytree_->Streenodesmap().size(); j++)
      {
        for (int k = 0; k < (int)binarytree_->Streenodesmap()[j].size(); k++)
        {
          // if proc !=0 and first treenode to plot->create new sheet in gmsh
          if (i != 0 && k == 0)
          {
            // create new sheet "Treenode" in gmsh
            std::ostringstream currentfilename;
            currentfilename << filenametn.str().c_str() << "_s_tnlayer_" << j << ".pos";
            fp = fopen(currentfilename.str().c_str(), "a");
            std::stringstream gmshfile;
            gmshfile << "};" << std::endl << "View \" Treenode \" { " << std::endl;
            fprintf(fp, gmshfile.str().c_str());
            fclose(fp);
          }
          // std::cout << std::endl << "plot streenode level: " << j << "treenode: " << k;
          std::ostringstream currentfilename;
          currentfilename << filenametn.str().c_str() << "_s_tnlayer_" << j << ".pos";
          binarytree_->Streenodesmap()[j][k]->PrintDopsForGmsh(currentfilename.str().c_str());

          // if there is another treenode to plot
          if (k < ((int)binarytree_->Streenodesmap()[j].size() - 1))
          {
            // create new sheet "Treenode" in gmsh
            std::ostringstream currentfilename;
            currentfilename << filenametn.str().c_str() << "_s_tnlayer_" << j << ".pos";
            fp = fopen(currentfilename.str().c_str(), "a");
            std::stringstream gmshfile;
            gmshfile << "};" << std::endl << "View \" Treenode \" { " << std::endl;
            fprintf(fp, gmshfile.str().c_str());
            fclose(fp);
          }
        }
      }
    }

    Comm().Barrier();
  }

  Comm().Barrier();
  // close all slave-gmsh files
  if (Core::Communication::my_mpi_rank(Comm()) == 0)
  {
    for (int i = 0; i < gnslayers; i++)
    {
      std::ostringstream currentfilename;
      currentfilename << filenametn.str().c_str() << "_s_tnlayer_" << i << ".pos";
      // std::cout << std::endl << Core::Communication::my_mpi_rank(Comm())<< "current filename: "
      // << currentfilename.str().c_str();
      fp = fopen(currentfilename.str().c_str(), "a");
      std::stringstream gmshfilecontent;
      gmshfilecontent << "};";
      fprintf(fp, gmshfilecontent.str().c_str());
      fclose(fp);
    }
  }
  Comm().Barrier();

  // create master slabs
  if (Core::Communication::my_mpi_rank(Comm()) == 0)
  {
    for (int i = 0; i < gnmlayers; i++)
    {
      std::ostringstream currentfilename;
      currentfilename << filenametn.str().c_str() << "_m_tnlayer_" << i << ".pos";
      // std::cout << std::endl << Core::Communication::my_mpi_rank(Comm())<< "filename: " <<
      // currentfilename.str().c_str();
      fp = fopen(currentfilename.str().c_str(), "w");
      std::stringstream gmshfile;
      gmshfile << "View \" Step " << step << " Iter " << iter << " mtl " << i << " \" {"
               << std::endl;
      fprintf(fp, gmshfile.str().c_str());
      fclose(fp);
    }

    // print full tree with treenodesmap
    for (int j = 0; j < (int)binarytree_->Mtreenodesmap().size(); j++)
    {
      for (int k = 0; k < (int)binarytree_->Mtreenodesmap()[j].size(); k++)
      {
        std::ostringstream currentfilename;
        currentfilename << filenametn.str().c_str() << "_m_tnlayer_" << j << ".pos";
        binarytree_->Mtreenodesmap()[j][k]->PrintDopsForGmsh(currentfilename.str().c_str());

        // if there is another treenode to plot
        if (k < ((int)binarytree_->Mtreenodesmap()[j].size() - 1))
        {
          // create new sheet "Treenode" in gmsh
          std::ostringstream currentfilename;
          currentfilename << filenametn.str().c_str() << "_m_tnlayer_" << j << ".pos";
          fp = fopen(currentfilename.str().c_str(), "a");
          std::stringstream gmshfile;
          gmshfile << "};" << std::endl << "View \" Treenode \" { " << std::endl;
          fprintf(fp, gmshfile.str().c_str());
          fclose(fp);
        }
      }
    }

    // close all master files
    for (int i = 0; i < gnmlayers; i++)
    {
      std::ostringstream currentfilename;
      currentfilename << filenametn.str().c_str() << "_m_tnlayer_" << i << ".pos";
      fp = fopen(currentfilename.str().c_str(), "a");
      std::stringstream gmshfilecontent;
      gmshfilecontent << std::endl << "};";
      fprintf(fp, gmshfilecontent.str().c_str());
      fclose(fp);
    }
  }
#endif


  //**********************************************************************
  // GMSH output of all active treenodes (DOPs) on leaf level
  //**********************************************************************
#ifdef MORTARGMSHCTN
  std::ostringstream filenamectn;
  filenamectn << "o/gmsh_output/" << file_name_only_prefix << "_";
  if (step < 10)
    filenamectn << 0 << 0 << 0 << 0;
  else if (step < 100)
    filenamectn << 0 << 0 << 0;
  else if (step < 1000)
    filenamectn << 0 << 0;
  else if (step < 10000)
    filenamectn << 0;
  else if (step > 99999)
    FOUR_C_THROW("Gmsh output implemented for a maximum of 99.999 time steps");
  filenamectn << step;

  // construct unique filename for gmsh output
  // second index = Newton iteration index
  if (iter >= 0)
  {
    filenamectn << "_";
    if (iter < 10)
      filenamectn << 0;
    else if (iter > 99)
      FOUR_C_THROW("Gmsh output implemented for a maximum of 99 iterations");
    filenamectn << iter;
  }

  int lcontactmapsize = (int)(binarytree_->coupling_map()[0].size());
  int gcontactmapsize;

  Comm().MaxAll(&lcontactmapsize, &gcontactmapsize, 1);

  if (gcontactmapsize > 0)
  {
    // open/create new file
    if (Core::Communication::my_mpi_rank(Comm()) == 0)
    {
      std::ostringstream currentfilename;
      currentfilename << filenamectn.str().c_str() << "_ct.pos";
      // std::cout << std::endl << Core::Communication::my_mpi_rank(Comm())<< "filename: " <<
      // currentfilename.str().c_str();
      fp = fopen(currentfilename.str().c_str(), "w");
      std::stringstream gmshfile;
      gmshfile << "View \" Step " << step << " Iter " << iter << " contacttn  \" {" << std::endl;
      fprintf(fp, gmshfile.str().c_str());
      fclose(fp);
    }

    // every proc should plot its contacting treenodes!
    for (int i = 0; i < Comm().NumProc(); i++)
    {
      if (Core::Communication::my_mpi_rank(Comm()) == i)
      {
        if ((int)(binarytree_->coupling_map()[0]).size() !=
            (int)(binarytree_->coupling_map()[1]).size())
          FOUR_C_THROW("Binarytree coupling_map does not have right size!");

        for (int j = 0; j < (int)((binarytree_->coupling_map()[0]).size()); j++)
        {
          std::ostringstream currentfilename;
          std::stringstream gmshfile;
          std::stringstream newgmshfile;

          // create new sheet for slave
          if (Core::Communication::my_mpi_rank(Comm()) == 0 && j == 0)
          {
            currentfilename << filenamectn.str().c_str() << "_ct.pos";
            fp = fopen(currentfilename.str().c_str(), "w");
            gmshfile << "View \" Step " << step << " Iter " << iter << " CS  \" {" << std::endl;
            fprintf(fp, gmshfile.str().c_str());
            fclose(fp);
            (binarytree_->coupling_map()[0][j])->PrintDopsForGmsh(currentfilename.str().c_str());
          }
          else
          {
            currentfilename << filenamectn.str().c_str() << "_ct.pos";
            fp = fopen(currentfilename.str().c_str(), "a");
            gmshfile << "};" << std::endl
                     << "View \" Step " << step << " Iter " << iter << " CS  \" {" << std::endl;
            fprintf(fp, gmshfile.str().c_str());
            fclose(fp);
            (binarytree_->coupling_map()[0][j])->PrintDopsForGmsh(currentfilename.str().c_str());
          }

          // create new sheet for master
          fp = fopen(currentfilename.str().c_str(), "a");
          newgmshfile << "};" << std::endl
                      << "View \" Step " << step << " Iter " << iter << " CM  \" {" << std::endl;
          fprintf(fp, newgmshfile.str().c_str());
          fclose(fp);
          (binarytree_->coupling_map()[1][j])->PrintDopsForGmsh(currentfilename.str().c_str());
        }
      }
      Comm().Barrier();
    }

    // close file
    if (Core::Communication::my_mpi_rank(Comm()) == 0)
    {
      std::ostringstream currentfilename;
      currentfilename << filenamectn.str().c_str() << "_ct.pos";
      // std::cout << std::endl << Core::Communication::my_mpi_rank(Comm())<< "filename: " <<
      // currentfilename.str().c_str();
      fp = fopen(currentfilename.str().c_str(), "a");
      std::stringstream gmshfile;
      gmshfile << "};";
      fprintf(fp, gmshfile.str().c_str());
      fclose(fp);
    }
  }
#endif  // MORTARGMSHCTN

  return;
}

FOUR_C_NAMESPACE_CLOSE
