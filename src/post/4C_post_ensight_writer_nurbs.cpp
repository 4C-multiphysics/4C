// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_post_common.hpp"
#include "4C_post_ensight_writer.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*
    Write the coordinates for a Nurbs discretization
    The coordinates of the visualisation points (i.e. the corner
    nodes of elements displayed in paraview) are not the control point
    coordinates of the nodes in the discretization but the points the
    knot values are mapped to.
*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_coordinates_for_nurbs_shapefunctions(std::ofstream& geofile,
    Core::FE::Discretization& dis, std::shared_ptr<Core::LinAlg::Map>& proc0map)
{
  using namespace FourC;

  // refcountpointer to vector of all coordinates
  // distributed among all procs
  std::shared_ptr<Core::LinAlg::MultiVector<double>> nodecoords;

  // the ids of the visualisation points on this proc
  std::vector<int> local_vis_point_ids;
  local_vis_point_ids.clear();

  // the coordinates of the visualisation points on this proc
  // used to construct the multivector nodecoords
  std::vector<std::vector<double>> local_vis_point_x;
  local_vis_point_x.clear();

  // cast dis to NurbsDiscretisation
  Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(dis));

  if (nurbsdis == nullptr)
  {
    FOUR_C_THROW("This probably isn't a NurbsDiscretization\n");
  }

  // get dimension
  int dim = (nurbsdis->return_nele_x_mele_x_lele(0)).size();

  // get the knotvector itself
  std::shared_ptr<Core::FE::Nurbs::Knotvector> knotvec = nurbsdis->get_knot_vector();

  // determine number of patches from knotvector
  int npatches = knotvec->return_np();

  // get vispoint offsets among patches
  std::vector<int> vpoff(npatches);

  vpoff[0] = 0;

  // loop all patches
  for (int np = 1; np < npatches; ++np)
  {
    // get nurbs dis' knotvector sizes
    std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(np - 1));

    int numvisp = 1;

    for (unsigned rr = 0; rr < nele_x_mele_x_lele.size(); ++rr)
    {
      numvisp *= 2 * nele_x_mele_x_lele[rr] + 1;
    }

    vpoff[np] = vpoff[np - 1] + numvisp;
  }

  // get element map
  const Core::LinAlg::Map* elementmap = nurbsdis->element_row_map();

  // loop all available elements
  for (int iele = 0; iele < elementmap->num_my_elements(); ++iele)
  {
    Core::Elements::Element* actele = nurbsdis->g_element(elementmap->gid(iele));

    // access elements knot span
    std::vector<Core::LinAlg::SerialDenseVector> knots(dim);
    bool zero_size = (*knotvec).get_ele_knots(knots, actele->id());

    // get gid, location in the patch
    int gid = actele->id();

    std::vector<int> ele_cart_id(dim);
    int np = -1;

    knotvec->convert_ele_gid_to_knot_ids(gid, np, ele_cart_id);

    // zero sized elements in knot span cannot be visualised
    if (zero_size)
    {
      // if we just skip them, we would loose some connectivity
      // in the result;
      // as a work-around, we replace the zero-sized element
      // with the next nonzero element --- this preserves the
      // connectivity, and everything looks nice and smooth again.
      // Note: This work-around will not work in parallel (or a
      //       special ghosting for elements along interpolated
      //       boundaries has to be applied)
      // Note: The following element will be plotted twice

      actele = nurbsdis->g_element(knotvec->return_next_nonzero_ele_gid(actele->id()));
      (*knotvec).get_ele_knots(knots, actele->id());
    }

    Core::Nodes::Node** nodes = actele->nodes();

    // get nurbs dis' element numbers
    std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(np));

    // want to loop all control points of the element,
    // so get the number of points
    const int numnp = actele->num_node();

    // access elements knot span
    zero_size = (*knotvec).get_ele_knots(knots, actele->id());

    // acquire weights from nodes
    Core::LinAlg::SerialDenseVector weights(numnp);

    for (int inode = 0; inode < numnp; ++inode)
    {
      Core::FE::Nurbs::ControlPoint* cp =
          dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes[inode]);

      weights(inode) = cp->w();
    }

    // get shapefunctions, compute all visualisation point positions
    Core::LinAlg::SerialDenseVector nurbs_shape_funct(numnp);

    switch (actele->shape())
    {
      case Core::FE::CellType::nurbs4:
      {
        // element local point position
        Core::LinAlg::SerialDenseVector uv(2);

        // standard

        // 3           4
        //  X---------X
        //  |         |
        //  |         |
        //  |         |
        //  |         |
        //  |         |
        //  X---------X
        // 1           2
        // append 4 points
        local_vis_point_ids.push_back(
            (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1) + 2 * ele_cart_id[0]);
        local_vis_point_ids.push_back(
            (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1) + 2 * ele_cart_id[0] + 1);
        local_vis_point_ids.push_back(
            (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1) + 2 * ele_cart_id[0]);
        local_vis_point_ids.push_back(
            (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1) + 2 * ele_cart_id[0] + 1);

        // temporary x vector
        std::vector<double> x(3);
        x[2] = 0;

        // point 1
        uv(0) = -1.0;
        uv(1) = -1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(nurbs_shape_funct, uv, knots, weights, actele->shape());
        for (int isd = 0; isd < 2; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
          }
          x[isd] = val;
        }
        local_vis_point_x.push_back(x);

        // point 2
        uv(0) = 1.0;
        uv(1) = -1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(nurbs_shape_funct, uv, knots, weights, actele->shape());
        for (int isd = 0; isd < 2; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
          }
          x[isd] = val;
        }
        local_vis_point_x.push_back(x);

        // point 3
        uv(0) = -1.0;
        uv(1) = 1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(nurbs_shape_funct, uv, knots, weights, actele->shape());
        for (int isd = 0; isd < 2; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
          }
          x[isd] = val;
        }
        local_vis_point_x.push_back(x);

        // point 4
        uv(0) = 1.0;
        uv(1) = 1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(nurbs_shape_funct, uv, knots, weights, actele->shape());
        for (int isd = 0; isd < 2; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
          }
          x[isd] = val;
        }
        local_vis_point_x.push_back(x);

        break;
      }
      case Core::FE::CellType::nurbs9:
      {
        // element local point position
        Core::LinAlg::SerialDenseVector uv(2);

        {
          // standard

          //
          //  +---------+
          //  |         |
          //  |         |
          //  X    X    |
          // 3|   4     |
          //  |         |
          //  X----X----+
          // 1    2
          // append 4 points
          local_vis_point_ids.push_back(vpoff[np] +
                                        (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1) +
                                        2 * ele_cart_id[0]);
          local_vis_point_ids.push_back(vpoff[np] +
                                        (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1) +
                                        2 * ele_cart_id[0] + 1);
          local_vis_point_ids.push_back(vpoff[np] +
                                        (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1) +
                                        2 * ele_cart_id[0]);
          local_vis_point_ids.push_back(vpoff[np] +
                                        (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1) +
                                        2 * ele_cart_id[0] + 1);

          // temporary x vector
          std::vector<double> x(3);
          x[2] = 0;

          // point 1
          uv(0) = -1.0;
          uv(1) = -1.0;
          Core::FE::Nurbs::nurbs_get_2d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 2; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 2
          uv(0) = 0.0;
          uv(1) = -1.0;
          Core::FE::Nurbs::nurbs_get_2d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 2; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 3
          uv(0) = -1.0;
          uv(1) = 0.0;
          Core::FE::Nurbs::nurbs_get_2d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 2; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 4
          uv(0) = 0.0;
          uv(1) = 0.0;
          Core::FE::Nurbs::nurbs_get_2d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 2; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }

        if (ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
        {
          // top line

          //
          //  X----X----+
          // 5|   6     |
          //  |         |
          //  X    X    |
          // 3|   4     |
          //  |         |
          //  X----X----+
          // 1    2
          //

          // append points 5 and 6
          local_vis_point_ids.push_back(vpoff[np] +
                                        (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1) +
                                        2 * ele_cart_id[0]);
          local_vis_point_ids.push_back(vpoff[np] +
                                        (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1) +
                                        2 * ele_cart_id[0] + 1);

          // temporary x vector
          std::vector<double> x(3);
          x[2] = 0;

          // point 5
          uv(0) = -1.0;
          uv(1) = 1.0;
          Core::FE::Nurbs::nurbs_get_2d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());

          for (int isd = 0; isd < 2; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 6
          uv(0) = 0.0;
          uv(1) = 1.0;
          Core::FE::Nurbs::nurbs_get_2d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 2; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }

        if (ele_cart_id[0] + 1 == nele_x_mele_x_lele[0])
        {
          // right line

          //
          //  +---------+
          //  |         |
          //  |         |
          //  X    X    X
          // 4|   5    6|
          //  |         |
          //  X----X----X
          // 1    2    3

          // append points 3 and 6
          local_vis_point_ids.push_back(vpoff[np] +
                                        (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1) +
                                        2 * ele_cart_id[0] + 2);
          local_vis_point_ids.push_back(vpoff[np] +
                                        (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1) +
                                        2 * ele_cart_id[0] + 2);

          // temporary x vector
          std::vector<double> x(3);
          x[2] = 0;

          // point 3
          uv(0) = 1.0;
          uv(1) = -1.0;
          Core::FE::Nurbs::nurbs_get_2d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 2; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 6
          uv(0) = 1.0;
          uv(1) = 0.0;
          Core::FE::Nurbs::nurbs_get_2d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 2; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }
        if (ele_cart_id[1] + 1 == nele_x_mele_x_lele[1] &&
            ele_cart_id[0] + 1 == nele_x_mele_x_lele[0])
        {
          // top right corner

          //
          //  X----X----X
          // 7|   8    9|
          //  |         |
          //  X    X    X
          // 4|   5    6|
          //  |         |
          //  X----X----X
          // 1    2    3

          // append point 9
          local_vis_point_ids.push_back(vpoff[np] +
                                        (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1) +
                                        2 * ele_cart_id[0] + 2);

          // temporary x vector
          std::vector<double> x(3);
          x[2] = 0;

          // point 9
          uv(0) = 1.0;
          uv(1) = 1.0;
          Core::FE::Nurbs::nurbs_get_2d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 2; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }
        break;
      }
      case Core::FE::CellType::nurbs27:
      {
        // element local point position
        Core::LinAlg::SerialDenseVector uv(3);

        int idu;
        int idv;
        int idw;

        // number of visualisation points in u direction
        int nvpu = 2 * (nurbsdis->return_nele_x_mele_x_lele(np))[0] + 1;

        // number of visualisation points in v direction
        int nvpv = 2 * (nurbsdis->return_nele_x_mele_x_lele(np))[1] + 1;

        {
          // standard

          //               v
          //              /
          //  w          /
          //  ^   +---------+
          //  |  /         /|
          //  | /         / |
          //   /         /  |
          //  +---------+   |
          //  | A----A  |   |
          //  |/|   /|  |   +
          //  A----A |  |  /
          //  | A--|-A  | /
          //  |/   |/   |/
          //  A----A----+ ----->u
          //
          // append 8 points

          idu = (2 * ele_cart_id[0]);
          idv = (2 * ele_cart_id[1]) * nvpu;
          idw = (2 * ele_cart_id[2]) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 1);
          idv = (2 * ele_cart_id[1]) * nvpu;
          idw = (2 * ele_cart_id[2]) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0]);
          idv = (2 * ele_cart_id[1] + 1) * nvpu;
          idw = (2 * ele_cart_id[2]) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 1);
          idv = (2 * ele_cart_id[1] + 1) * nvpu;
          idw = (2 * ele_cart_id[2]) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0]);
          idv = (2 * ele_cart_id[1]) * nvpu;
          idw = (2 * ele_cart_id[2] + 1) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 1);
          idv = (2 * ele_cart_id[1]) * nvpu;
          idw = (2 * ele_cart_id[2] + 1) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0]);
          idv = (2 * ele_cart_id[1] + 1) * nvpu;
          idw = (2 * ele_cart_id[2] + 1) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 1);
          idv = (2 * ele_cart_id[1] + 1) * nvpu;
          idw = (2 * ele_cart_id[2] + 1) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);


          // temporary x vector
          std::vector<double> x(3);

          // point 1
          uv(0) = -1.0;
          uv(1) = -1.0;
          uv(2) = -1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);


          // point 2
          uv(0) = 0.0;
          uv(1) = -1.0;
          uv(2) = -1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 3
          uv(0) = -1.0;
          uv(1) = 0.0;
          uv(2) = -1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 4
          uv(0) = 0.0;
          uv(1) = 0.0;
          uv(2) = -1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 5
          uv(0) = -1.0;
          uv(1) = -1.0;
          uv(2) = 0.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 6
          uv(0) = 0.0;
          uv(1) = -1.0;
          uv(2) = 0.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 7
          uv(0) = -1.0;
          uv(1) = 0.0;
          uv(2) = 0.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 8
          uv(0) = 0.0;
          uv(1) = 0.0;
          uv(2) = 0.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }

        if (ele_cart_id[0] + 1 == nele_x_mele_x_lele[0])
        {
          //               v
          //              /
          //  w          /
          //  ^   +---------+
          //  |  /         /|
          //  | /         / |
          //   /         /  |
          //  +---------+   |
          //  | X----X--|-A |
          //  |/|   /|  |/| +
          //  X----X----A |/
          //  | X--|-X--|-A
          //  |/   |/   |/
          //  X----X----A ----->u
          //
          // append 4 additional points

          idu = (2 * ele_cart_id[0] + 2);
          idv = (2 * ele_cart_id[1]) * nvpu;
          idw = (2 * ele_cart_id[2]) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 2);
          idv = (2 * ele_cart_id[1] + 1) * nvpu;
          idw = (2 * ele_cart_id[2]) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 2);
          idv = (2 * ele_cart_id[1]) * nvpu;
          idw = (2 * ele_cart_id[2] + 1) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 2);
          idv = (2 * ele_cart_id[1] + 1) * nvpu;
          idw = (2 * ele_cart_id[2] + 1) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);


          // temporary x vector
          std::vector<double> x(3);

          // point 1
          uv(0) = 1.0;
          uv(1) = -1.0;
          uv(2) = -1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);


          // point 2
          uv(0) = 1.0;
          uv(1) = 0.0;
          uv(2) = -1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 3
          uv(0) = 1.0;
          uv(1) = -1.0;
          uv(2) = 0.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 4
          uv(0) = 1.0;
          uv(1) = 0.0;
          uv(2) = 0.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }

        if (ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
        {
          //               v
          //              /
          //  w          /
          //  ^   +---------+
          //  |  /         /|
          //  | /         / |
          //   /  A----A /  |
          //  +---------+   |
          //  | X----X ||   |
          //  |/| A-/|-A|   +
          //  X----X |/ |  /
          //  | X--|-X  | /
          //  |/   |/   |/
          //  X----X----+ ----->u
          //
          // append 4 additional points

          idu = (2 * ele_cart_id[0]);
          idv = (2 * ele_cart_id[1] + 2) * nvpu;
          idw = (2 * ele_cart_id[2]) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 1);
          idv = (2 * ele_cart_id[1] + 2) * nvpu;
          idw = (2 * ele_cart_id[2]) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0]);
          idv = (2 * ele_cart_id[1] + 2) * nvpu;
          idw = (2 * ele_cart_id[2] + 1) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 1);
          idv = (2 * ele_cart_id[1] + 2) * nvpu;
          idw = (2 * ele_cart_id[2] + 1) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);


          // temporary x vector
          std::vector<double> x(3);

          // point 1
          uv(0) = -1.0;
          uv(1) = 1.0;
          uv(2) = -1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);


          // point 2
          uv(0) = 0.0;
          uv(1) = 1.0;
          uv(2) = -1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 3
          uv(0) = -1.0;
          uv(1) = 1.0;
          uv(2) = 0.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 4
          uv(0) = 0.0;
          uv(1) = 1.0;
          uv(2) = 0.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }

        if (ele_cart_id[0] + 1 == nele_x_mele_x_lele[0] &&
            ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
        {
          //               v
          //              /
          //  w          /
          //  ^   +---------+
          //  |  /         /|
          //  | /         / |
          //   /  X----X-/--A
          //  +---------+  /|
          //  | X----X--|-X |
          //  |/| X-/|-X|/|-A
          //  X----X----X |/
          //  | X--|-X--|-X
          //  |/   |/   |/
          //  X----X----X ----->u
          //
          // append 2 additional points

          idu = (2 * ele_cart_id[0] + 2);
          idv = (2 * ele_cart_id[1] + 2) * nvpu;
          idw = (2 * ele_cart_id[2]) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 2);
          idv = (2 * ele_cart_id[1] + 2) * nvpu;
          idw = (2 * ele_cart_id[2] + 1) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          // temporary x vector
          std::vector<double> x(3);

          // point 1
          uv(0) = 1.0;
          uv(1) = 1.0;
          uv(2) = -1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 2
          uv(0) = 1.0;
          uv(1) = 1.0;
          uv(2) = 0.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }


        if (ele_cart_id[2] + 1 == nele_x_mele_x_lele[2])
        {
          //               v
          //              /
          //  w          /
          //  ^   +---------+
          //  |  /         /|
          //  | A----A    / |
          //   /    /|   /  |
          //  A----A----+   |
          //  | X--|-X  |   |
          //  |/|  |/|  |   +
          //  X----X |  |  /
          //  | X--|-X  | /
          //  |/   |/   |/
          //  X----X----+ ----->u
          //
          //
          // append 4 additional points

          idu = (2 * ele_cart_id[0]);
          idv = (2 * ele_cart_id[1]) * nvpu;
          idw = (2 * ele_cart_id[2] + 2) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 1);
          idv = (2 * ele_cart_id[1]) * nvpu;
          idw = (2 * ele_cart_id[2] + 2) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0]);
          idv = (2 * ele_cart_id[1] + 1) * nvpu;
          idw = (2 * ele_cart_id[2] + 2) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 1);
          idv = (2 * ele_cart_id[1] + 1) * nvpu;
          idw = (2 * ele_cart_id[2] + 2) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);


          // temporary x vector
          std::vector<double> x(3);

          // point 1
          uv(0) = -1.0;
          uv(1) = -1.0;
          uv(2) = 1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);


          // point 2
          uv(0) = 0.0;
          uv(1) = -1.0;
          uv(2) = 1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 3
          uv(0) = -1.0;
          uv(1) = 0.0;
          uv(2) = 1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 4
          uv(0) = 0.0;
          uv(1) = 0.0;
          uv(2) = 1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }


        if (ele_cart_id[2] + 1 == nele_x_mele_x_lele[2] &&
            ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
        {
          //               v
          //              /
          //  w          /
          //  ^   A----A----+
          //  |  /|   /|   /|
          //  | X----X |  / |
          //   /| X /| X /  |
          //  X----X----+   |
          //  | X--|-X ||   |
          //  |/|  |/| X|   +
          //  X----X |/ |  /
          //  | X--|-X  | /
          //  |/   |/   |/
          //  X----X----+ ----->u
          //
          //
          // append 2 additional points

          idu = (2 * ele_cart_id[0]);
          idv = (2 * ele_cart_id[1] + 2) * nvpu;
          idw = (2 * ele_cart_id[2] + 2) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 1);
          idv = (2 * ele_cart_id[1] + 2) * nvpu;
          idw = (2 * ele_cart_id[2] + 2) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);
          // temporary x vector
          std::vector<double> x(3);

          // point 1
          uv(0) = -1.0;
          uv(1) = 1.0;
          uv(2) = 1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 2
          uv(0) = 0.0;
          uv(1) = 1.0;
          uv(2) = 1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }

        if (ele_cart_id[2] + 1 == nele_x_mele_x_lele[2] &&
            ele_cart_id[0] + 1 == nele_x_mele_x_lele[0])
        {
          //               v
          //              /
          //  w          /
          //  ^   +---------+
          //  |  /         /|
          //  | X----X----A |
          //   /    /|   /| |
          //  X----X----A | |
          //  | X--|-X--|-X |
          //  |/|  |/|  |/| +
          //  X----X----X |/
          //  | X--|-X--|-X
          //  |/   |/   |/
          //  X----X----X ----->u
          //
          //
          // append 2 additional points

          idu = (2 * ele_cart_id[0] + 2);
          idv = (2 * ele_cart_id[1]) * nvpu;
          idw = (2 * ele_cart_id[2] + 2) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          idu = (2 * ele_cart_id[0] + 2);
          idv = (2 * ele_cart_id[1] + 1) * nvpu;
          idw = (2 * ele_cart_id[2] + 2) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          // temporary x vector
          std::vector<double> x(3);

          // point 1
          uv(0) = 1.0;
          uv(1) = -1.0;
          uv(2) = 1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);

          // point 2
          uv(0) = 1.0;
          uv(1) = 0.0;
          uv(2) = 1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }

        if (ele_cart_id[2] + 1 == nele_x_mele_x_lele[2] &&
            ele_cart_id[1] + 1 == nele_x_mele_x_lele[1] &&
            ele_cart_id[0] + 1 == nele_x_mele_x_lele[0])
        {
          //               v
          //              /
          //  w          /
          //  ^   X----X----A
          //  |  /|   /    /|
          //  | X----X----X |
          //   /| X-/|-X-/|-X
          //  X----X----X |/|
          //  | X--|-X--|-X |
          //  |/| X|/|-X|/|-X
          //  X----X----X |/
          //  | X--|-X--|-X
          //  |/   |/   |/
          //  X----X----X ----->u
          //
          // append 1 additional point


          idu = (2 * ele_cart_id[0] + 2);
          idv = (2 * ele_cart_id[1] + 2) * nvpu;
          idw = (2 * ele_cart_id[2] + 2) * nvpu * nvpv;
          local_vis_point_ids.push_back(vpoff[np] + idu + idv + idw);

          // temporary x vector
          std::vector<double> x(3);

          // point 1
          uv(0) = 1.0;
          uv(1) = 1.0;
          uv(2) = 1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }
          local_vis_point_x.push_back(x);
        }

        break;
      }
      default:
        std::cout << *actele;
        FOUR_C_THROW("Unknown distype for nurbs element output\n");
        break;
    }
  }

  // construct map for visualisation points. Store it in
  // class variable for access in data interpolation
  int numvispoints = 0;

  // loop all patches
  for (int np = 0; np < npatches; ++np)
  {
    // get nurbs dis' knotvector sizes
    std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(np));

    int numvisp = 1;

    for (unsigned rr = 0; rr < nele_x_mele_x_lele.size(); ++rr)
    {
      numvisp *= 2 * (nele_x_mele_x_lele[rr]) + 1;
    }
    numvispoints += numvisp;
  }

  vispointmap_ = std::make_shared<Core::LinAlg::Map>(numvispoints, local_vis_point_ids.size(),
      local_vis_point_ids.data(), 0, nurbsdis->get_comm());

  // allocate the coordinates of the visualisation points
  nodecoords = std::make_shared<Core::LinAlg::MultiVector<double>>(*vispointmap_, 3);

  // loop over the nodes on this proc and store the coordinate information
  for (int inode = 0; inode < (int)local_vis_point_x.size(); inode++)
  {
    for (int isd = 0; isd < 3; ++isd)
    {
      double val = (local_vis_point_x[inode])[isd];
      nodecoords->replace_local_value(inode, isd, val);
    }
  }

  // new procmap
  proc0map = Core::LinAlg::allreduce_e_map(*vispointmap_, 0);

  // import my new values (proc0 gets everything, other procs empty)
  Core::LinAlg::Import proc0importer(*proc0map, *vispointmap_);
  Core::LinAlg::MultiVector<double> allnodecoords(*proc0map, 3);
  allnodecoords.import(*nodecoords, proc0importer, Insert);

  // write the node coordinates (only proc 0)
  // ensight format requires x_1 .. x_n, y_1 .. y_n, z_1 ... z_n
  // this is fulfilled automatically due to Core::LinAlg::MultiVector<double> usage (columnwise
  // writing data)
  if (myrank_ == 0)
  {
    double* coords = allnodecoords.get_values();
    int numentries = (3 * (allnodecoords.global_length()));

    if (nodeidgiven_)
    {
      // first write node global ids (default)
      for (int inode = 0; inode < proc0map->num_global_elements(); ++inode)
      {
        write(geofile, static_cast<float>(proc0map->gid(inode)) + 1);
        // gid+1 delivers the node numbering of the input file starting with 1
      }
    }
    // now write the coordinate information
    for (int i = 0; i < numentries; ++i)
    {
      write(geofile, static_cast<float>(coords[i]));
    }
  }

  return;
}

/*----------------------------------------------------------------------
         Write the cells for a Nurbs discretization
         quadratic nurbs split one element in knot space into
         four(2d)/eight(3d) cells. The global numbering of the
         visualisation points (i.e. the corner points of the
         cells) is computed from the local patch numbering and
         the patch offset.                             (gammi)
----------------------------------------------------------------------*/
void EnsightWriter::write_nurbs_cell(const Core::FE::CellType distype, const int gid,
    std::ofstream& geofile, std::vector<int>& nodevector, Core::FE::Discretization& dis,
    Core::LinAlg::Map& proc0map) const
{
  using namespace FourC;

  // cast dis to NurbsDiscretisation
  Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(dis));

  if (nurbsdis == nullptr)
  {
    FOUR_C_THROW("This probably isn't a NurbsDiscretization\n");
  }

  // get the knotvector itself
  std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = nurbsdis->get_knot_vector();

  // determine number of patches from knotvector
  int npatches = knots->return_np();

  // get vispoint offsets among patches
  std::vector<int> vpoff(npatches);

  vpoff[0] = 0;

  // loop all patches
  for (int np = 1; np < npatches; ++np)
  {
    // get nurbs dis' knotvector sizes
    std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(np - 1));

    int numvisp = 1;

    for (unsigned rr = 0; rr < nele_x_mele_x_lele.size(); ++rr)
    {
      numvisp *= 2 * nele_x_mele_x_lele[rr] + 1;
    }

    vpoff[np] = vpoff[np - 1] + numvisp;
  }

  switch (distype)
  {
    case Core::FE::CellType::nurbs4:
    {
      // get dimension
      const int dim = 2;

      // get the knotvector itself
      std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = nurbsdis->get_knot_vector();

      // get location in the patch and the number of the patch
      int npatch = -1;
      std::vector<int> ele_cart_id(dim);
      knots->convert_ele_gid_to_knot_ids(gid, npatch, ele_cart_id);

      // number of visualisation points in u direction
      int nvpu = (nurbsdis->return_nele_x_mele_x_lele(npatch))[0] + 1;

      // 3           4
      //  X---------X
      //  |         |
      //  |         |
      //  |         |
      //  |         |
      //  |         |
      //  X---------X
      // 1           2

      // append 4 elements
      if (myrank_ == 0)  // proc0 can write its elements immediately
      {
        write(geofile, proc0map.lid(((ele_cart_id[1]) * (nvpu) + ele_cart_id[0])) + 1);
        write(geofile, proc0map.lid(((ele_cart_id[1]) * (nvpu) + ele_cart_id[0] + 1)) + 1);
        write(geofile, proc0map.lid(((ele_cart_id[1] + 1) * (nvpu) + ele_cart_id[0] + 1)) + 1);
        write(geofile, proc0map.lid(((ele_cart_id[1] + 1) * (nvpu) + ele_cart_id[0])) + 1);
      }
      else  // elements on other procs have to store their global node ids
      {
        nodevector.push_back((ele_cart_id[1]) * (nvpu) + ele_cart_id[0]);
        nodevector.push_back((ele_cart_id[1]) * (nvpu) + ele_cart_id[0] + 1);
        nodevector.push_back((ele_cart_id[1] + 1) * (nvpu) + ele_cart_id[0] + 1);
        nodevector.push_back((ele_cart_id[1] + 1) * (nvpu) + ele_cart_id[0]);
      }
    }
    break;
    case Core::FE::CellType::nurbs9:
    {
      // get dimension
      const int dim = 2;

      // get location in the patch from gid
      int npatch = -1;
      std::vector<int> ele_cart_id(dim);
      knots->convert_ele_gid_to_knot_ids(gid, npatch, ele_cart_id);

      // number of visualisation points in u direction
      int nvpu = 2 * (nurbsdis->return_nele_x_mele_x_lele(npatch))[0] + 1;

      //
      //  X----X----X
      // 7|   8    9|
      //  |         |
      //  X    X    X
      // 4|   5    6|
      //  |         |
      //  X----X----X
      // 1    2    3

      // append 4 elements
      if (myrank_ == 0)  // proc0 can write its elements immediately
      {
        write(geofile,
            proc0map.lid(vpoff[npatch] + ((2 * ele_cart_id[1]) * (nvpu) + 2 * ele_cart_id[0])) + 1);
        write(geofile,
            proc0map.lid(vpoff[npatch] + ((2 * ele_cart_id[1]) * (nvpu) + 2 * ele_cart_id[0] + 1)) +
                1);
        write(geofile, proc0map.lid(vpoff[npatch] +
                                    ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 1)) +
                           1);
        write(geofile,
            proc0map.lid(vpoff[npatch] + ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0])) +
                1);

        write(geofile,
            proc0map.lid(vpoff[npatch] + ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0])) +
                1);
        write(geofile, proc0map.lid(vpoff[npatch] +
                                    ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 1)) +
                           1);
        write(geofile, proc0map.lid(vpoff[npatch] +
                                    ((2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0] + 1)) +
                           1);
        write(geofile,
            proc0map.lid(vpoff[npatch] + ((2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0])) +
                1);

        write(geofile,
            proc0map.lid(vpoff[npatch] + ((2 * ele_cart_id[1]) * (nvpu) + 2 * ele_cart_id[0] + 1)) +
                1);
        write(geofile,
            proc0map.lid(vpoff[npatch] + ((2 * ele_cart_id[1]) * (nvpu) + 2 * ele_cart_id[0] + 2)) +
                1);
        write(geofile, proc0map.lid(vpoff[npatch] +
                                    ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 2)) +
                           1);
        write(geofile, proc0map.lid(vpoff[npatch] +
                                    ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 1)) +
                           1);

        write(geofile, proc0map.lid(vpoff[npatch] +
                                    ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 1)) +
                           1);
        write(geofile, proc0map.lid(vpoff[npatch] +
                                    ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 2)) +
                           1);
        write(geofile, proc0map.lid(vpoff[npatch] +
                                    ((2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0] + 2)) +
                           1);
        write(geofile, proc0map.lid(vpoff[npatch] +
                                    ((2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0] + 1)) +
                           1);
      }
      else  // elements on other procs have to store their global node ids
      {
        nodevector.push_back(vpoff[npatch] + ((2 * ele_cart_id[1]) * (nvpu) + 2 * ele_cart_id[0]));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1]) * (nvpu) + 2 * ele_cart_id[0] + 1));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 1));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0]));

        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0]));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 1));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0] + 1));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0]));

        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1]) * (nvpu) + 2 * ele_cart_id[0] + 1));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1]) * (nvpu) + 2 * ele_cart_id[0] + 2));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 2));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 1));

        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 1));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 2));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0] + 2));
        nodevector.push_back(
            vpoff[npatch] + ((2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0] + 1));
      }
    }
    break;
    case Core::FE::CellType::nurbs27:
    {
      //               v
      //              /
      //  w          /
      //  ^   X----X----A
      //  |  /|   /    /|
      //  | X----X----X |
      //   /| X-/|-X-/|-X
      //  X----X----X |/|
      //  | X--|-X--|-X |
      //  |/| X|/|-X|/|-X
      //  X----X----X |/
      //  | X--|-X--|-X
      //  |/   |/   |/
      //  X----X----X ----->u
      //

      // get dimension
      const int dim = 3;

      // get location in the patch
      int npatch = -1;
      std::vector<int> ele_cart_id(dim);
      knots->convert_ele_gid_to_knot_ids(gid, npatch, ele_cart_id);

      // number of visualisation points in u direction
      int nvpu = 2 * (nurbsdis->return_nele_x_mele_x_lele(npatch))[0] + 1;

      // number of visualisation points in v direction
      int nvpv = 2 * (nurbsdis->return_nele_x_mele_x_lele(npatch))[1] + 1;

      // vector containing node connectivity for all sub hexes (in blocks of 8)
      std::vector<int> cellnodes;

      // bottom, left front
      append_nurbs_sub_hex(cellnodes, 0, 0, 0, ele_cart_id, nvpu, nvpv, npatch);
      // bottom, right front
      append_nurbs_sub_hex(cellnodes, 1, 0, 0, ele_cart_id, nvpu, nvpv, npatch);
      // bottom, left rear
      append_nurbs_sub_hex(cellnodes, 0, 1, 0, ele_cart_id, nvpu, nvpv, npatch);
      // bottom, right rear
      append_nurbs_sub_hex(cellnodes, 1, 1, 0, ele_cart_id, nvpu, nvpv, npatch);
      // top, left front
      append_nurbs_sub_hex(cellnodes, 0, 0, 1, ele_cart_id, nvpu, nvpv, npatch);
      // top, right front
      append_nurbs_sub_hex(cellnodes, 1, 0, 1, ele_cart_id, nvpu, nvpv, npatch);
      // top, left rear
      append_nurbs_sub_hex(cellnodes, 0, 1, 1, ele_cart_id, nvpu, nvpv, npatch);
      // top, right rear
      append_nurbs_sub_hex(cellnodes, 1, 1, 1, ele_cart_id, nvpu, nvpv, npatch);

      if (cellnodes.size() != 64)
      {
        FOUR_C_THROW("something went wrong with the construction of cellnode connectivity\n");
      }

      if (myrank_ == 0)  // proc0 can write its elements immediately
      {
        for (unsigned id = 0; id < cellnodes.size(); ++id)
        {
          write(geofile, proc0map.lid(vpoff[npatch] + cellnodes[id]) + 1);
        }
      }
      else  // elements on other procs have to store their global node ids
      {
        for (unsigned id = 0; id < cellnodes.size(); ++id)
        {
          nodevector.push_back(vpoff[npatch] + cellnodes[id]);
        }
      }
    }
    break;
    default:
    {
      FOUR_C_THROW("unknown nurbs discretisation type\n");
      break;
    }
  }  // end switch distype
  return;
}

/*----------------------------------------------------------------------*/
/*
    Write the results for a NURBS discretisation (dof based).

    On input, result data for an ndimensional computation
    is provided (from the result file)

    This element data is communicated in such a way that
    all elements have access to their (dof-accessible) data.
    Here we separate velocity/displacement and pressure
    output, since for velocity/displacement and pressure
    different dofs are required.

    Then, all elements are looped and function values are
    evaluated at visualisation points. This is the place
    where we need the dof data (again, different data for
    velocity/displacement and pressure output)

    The resulting vector is allreduced on proc0 and written.

    gammi
*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_dof_result_step_for_nurbs(std::ofstream& file, const int numdf,
    Core::LinAlg::Vector<double>& data, const std::string name, const int offset) const
{
  using namespace FourC;

  // a multivector for the interpolated data
  std::shared_ptr<Core::LinAlg::MultiVector<double>> idata;
  idata = std::make_shared<Core::LinAlg::MultiVector<double>>(*vispointmap_, numdf);

  Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*field_->discretization()));

  if (nurbsdis == nullptr)
  {
    FOUR_C_THROW("This probably isn't a NurbsDiscretization\n");
  }

  // get number of patches
  int npatches = (nurbsdis->get_knot_vector())->return_np();

  // assuming that dimension of the manifold is
  // equal to spatial dimension
  int dim = (int)(nurbsdis->return_nele_x_mele_x_lele(0)).size();

  // the number of visualisation points
  int numvispoints = 0;

  for (int np = 0; np < npatches; ++np)
  {
    int numvisp = 1;

    // get nurbs dis' knotvector sizes
    std::vector<int> n_x_m_x_l(nurbsdis->return_n_x_m_x_l(np));

    // get nurbs dis' knotvector sizes
    std::vector<int> degree(nurbsdis->return_degree(np));

    for (unsigned rr = 0; rr < n_x_m_x_l.size(); ++rr)
    {
      numvisp *= 2 * (n_x_m_x_l[rr] - 2 * degree[rr]) - 1;
    }
    numvispoints += numvisp;
  }  // end loop over patches

  // get the knotvector itself
  std::shared_ptr<Core::FE::Nurbs::Knotvector> knotvec = nurbsdis->get_knot_vector();

  // get vispoint offsets among patches
  std::vector<int> vpoff(npatches);

  vpoff[0] = 0;

  // loop all patches
  for (int np = 1; np < npatches; ++np)
  {
    // get nurbs dis' knotvector sizes
    std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(np - 1));

    int numvisp = 1;

    for (unsigned rr = 0; rr < nele_x_mele_x_lele.size(); ++rr)
    {
      numvisp *= 2 * nele_x_mele_x_lele[rr] + 1;
    }

    vpoff[np] = vpoff[np - 1] + numvisp;
  }

  // get element map
  const Core::LinAlg::Map* elementmap = nurbsdis->element_row_map();

  // construct a colmap for data to have it available at
  // all elements (the critical ones are the ones at the
  // processor boundary)
  // loop all available elements
  std::set<int> coldofset;
  for (int iele = 0; iele < elementmap->num_my_elements(); ++iele)
  {
    Core::Elements::Element* actele = nurbsdis->g_element(elementmap->gid(iele));

    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;

    // extract local values from the global vectors
    actele->location_vector(*nurbsdis, lm, lmowner, lmstride);

    // do not forget to consider a (potential) offset in dof numbering for all results!
    for (int inode = 0; inode < actele->num_node(); ++inode)
    {
      if (name == "velocity" || name == "averaged_velocity" || name == "ale_displacement" ||
          name == "convective_velocity" || name == "grid_velocity")
      {
        if (dim != numdf)
        {
          FOUR_C_THROW("dim and numdf not matching for field {}", name);
        }

        for (int rr = 0; rr < dim; ++rr)
        {
          coldofset.insert(lm[inode * (dim + 1) + rr] + offset);
        }
      }
      else if (name == "displacement")
      {
        if (dim != numdf)
        {
          FOUR_C_THROW("dim and numdf not matching for field {}", name);
        }
        for (int rr = 0; rr < dim; ++rr)
        {
          coldofset.insert(lm[inode * dim + rr] + offset);
        }
      }
      else if (name == "pressure" || name == "averaged_pressure")
      {
        // offset should be equal to dim for pressure case!
        coldofset.insert(lm[inode * (dim + 1) + dim] + (offset - dim));
      }
      else if (name == "averaged_scalar_field")
      {
        // offset should be equal to dim for pressure case!
        coldofset.insert(lm[inode * (dim + 1) + dim] + offset);
      }
      else if ((name == "phi") or (name == "averaged_phi"))
      {
        Core::Nodes::Node* n = nurbsdis->l_row_node(inode);
        int numdofpernode = actele->num_dof_per_node(*n);
        if (numdofpernode == 1)  // one passive scalar (Scalar_Transport problem)
          coldofset.insert(lm[inode] + offset);
        else  // result for electric potential (ELCH problem)
          coldofset.insert(lm[inode * numdofpernode + (numdofpernode - 1)] + offset);
      }
      else if (name.substr(0, 2) == "c_" or name.substr(0, 11) == "averaged_c_")  // c_1, c_2 ,...
      {
        int k(0);
        if ((name == "c_1") or (name == "averaged_c_1"))
          k = 0;
        else if ((name == "c_2") or (name == "averaged_c_2"))
          k = 1;
        else if ((name == "c_3") or (name == "averaged_c_3"))
          k = 2;
        else if ((name == "c_4") or (name == "averaged_c_4"))
          k = 3;
        else if ((name == "c_5") or (name == "averaged_c_5"))
          k = 4;
        else
          FOUR_C_THROW("Up to now, I'm not able to write a field named {}\n", name);

        Core::Nodes::Node* n = nurbsdis->l_row_node(inode);
        int numdofpernode = actele->num_dof_per_node(*n);
        coldofset.insert(lm[inode * numdofpernode + k] + offset);
      }
      else if (name == "normalflux")
      {
        Core::Nodes::Node* n = nurbsdis->l_row_node(inode);
        int numdofpernode = actele->num_dof_per_node(*n);
        coldofset.insert(lm[inode * numdofpernode] + offset);
      }
      //---------------------------------------------------
      // contact - specific output:
      else if (name == "norcontactstress" || name == "tancontactstress" ||
               name == "interfacetraction" || name == "slaveforces" || name == "masterforces" ||
               name == "norslaveforce" || name == "tanslaveforce" || name == "normasterforce" ||
               name == "tanmasterforce" || name == "wear" || name == "norslaveforcelm" ||
               name == "norslaveforceg" || name == "normasterforcelm" || name == "normasterforceg")
      {
        if (dim != numdf)
        {
          FOUR_C_THROW("dim and numdf not matching for field {}", name);
        }
        for (int rr = 0; rr < dim; ++rr)
        {
          coldofset.insert(lm[inode * dim + rr] + offset);
        }
      }
      else if (name == "temperature")
      {
        coldofset.insert(lm[inode] + offset);
      }
      else
      {
        FOUR_C_THROW("Up to now, I'm not able to write a field named {}\n", name);
      }
    }
  }

  std::vector<int> coldofmapvec;
  coldofmapvec.reserve(coldofset.size());
  coldofmapvec.assign(coldofset.begin(), coldofset.end());
  coldofset.clear();
  Core::LinAlg::Map coldofmap(
      -1, coldofmapvec.size(), coldofmapvec.data(), 0, nurbsdis->get_comm());
  coldofmapvec.clear();

  const Core::LinAlg::Map* fulldofmap = &(coldofmap);
  Core::LinAlg::Vector<double> coldata(*fulldofmap, true);

  // create an importer and import the data
  Core::LinAlg::Import importer((coldata).get_map(), (data).get_map());
  int imerr = (coldata).import((data), importer, Insert);
  if (imerr)
  {
    FOUR_C_THROW("import failed\n");
  }

  // loop all available elements
  for (int iele = 0; iele < elementmap->num_my_elements(); ++iele)
  {
    Core::Elements::Element* actele = nurbsdis->g_element(elementmap->gid(iele));

    // get gid, location in the patch
    std::vector<int> ele_cart_id(dim);
    int gid = actele->id();
    int np = -1;

    // access elements knot span
    std::vector<Core::LinAlg::SerialDenseVector> knots(dim);
    bool zero_size = (*knotvec).get_ele_knots(knots, actele->id());

    knotvec->convert_ele_gid_to_knot_ids(gid, np, ele_cart_id);

    // zero sized elements in knot span cannot be visualised
    if (zero_size)
    {
      // if we just skip them, we would loose some connectivity
      // in the result;
      // as a work-around, we replace the zero-sized element
      // with the next nonzero element --- this preserves the
      // connectivity, and everything looks nice and smooth again.
      // Note: This work-around will not work in parallel (or a
      //       special ghosting for elements along interpolated
      //       boundaries has to be applied)
      // Note: The following element will be plotted twice

      actele = nurbsdis->g_element(knotvec->return_next_nonzero_ele_gid(actele->id()));
      (*knotvec).get_ele_knots(knots, actele->id());
    }

    Core::Nodes::Node** nodes = actele->nodes();

    // number of all control points of the element
    const int numnp = actele->num_node();

    // acquire weights from nodes
    Core::LinAlg::SerialDenseVector weights(numnp);

    for (int inode = 0; inode < numnp; ++inode)
    {
      Core::FE::Nurbs::ControlPoint* cp =
          dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes[inode]);
      weights(inode) = cp->w();
    }

    // extract local values from the global vectors
    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;

    actele->location_vector(*nurbsdis, lm, lmowner,
        lmstride);  // get gid, location in the patch and the number of the patch

    int npatch = np;

    // access elements knot span
    std::vector<Core::LinAlg::SerialDenseVector> eleknots(dim);
    knotvec->get_ele_knots(eleknots, actele->id());

    // access solution data
    std::vector<double> my_data(lm.size());
    if (name == "velocity" || name == "averaged_velocity" || name == "ale_displacement" ||
        name == "convective_velocity" || name == "grid_velocity")
    {
      my_data.resize(dim * numnp);

      for (int inode = 0; inode < numnp; ++inode)
      {
        for (int rr = 0; rr < dim; ++rr)
        {
          my_data[dim * inode + rr] =
              (coldata)[(coldata).get_map().lid(lm[inode * (dim + 1) + rr] + offset)];
        }
      }
    }
    else if (name == "displacement")
    {
      my_data.resize(dim * numnp);

      for (int inode = 0; inode < numnp; ++inode)
      {
        for (int rr = 0; rr < dim; ++rr)
        {
          my_data[dim * inode + rr] =
              (coldata)[(coldata).get_map().lid(lm[inode * dim + rr] + offset)];
        }
      }
    }
    else if (name == "pressure" || name == "averaged_pressure")
    {
      my_data.resize(numnp);

      for (int inode = 0; inode < numnp; ++inode)
      {
        // offset should be equal to dim for pressure case!
        my_data[inode] =
            (coldata)[(coldata).get_map().lid(lm[inode * (dim + 1) + dim] + offset - dim)];
      }
    }
    else if (name == "averaged_scalar_field")
    {
      my_data.resize(numnp);

      for (int inode = 0; inode < numnp; ++inode)
      {
        // offset should be equal to dim for pressure case!
        my_data[inode] = (coldata)[(coldata).get_map().lid(lm[inode * (dim + 1) + dim] + offset)];
      }
    }
    else if ((name == "phi") or (name == "averaged_phi"))
    {
      my_data.resize(numnp);

      for (int inode = 0; inode < numnp; ++inode)
      {
        Core::Nodes::Node* n = nurbsdis->l_row_node(inode);
        int numdofpernode = actele->num_dof_per_node(*n);
        if (numdofpernode == 1)  // one passive scalar (Scalar_Transport problem)
          my_data[inode] = (coldata)[(coldata).get_map().lid(lm[inode * numdofpernode] + offset)];
        else  // result for electric potential (ELCH problem)
          my_data[inode] = (coldata)[(coldata).get_map().lid(
              lm[inode * numdofpernode + (numdofpernode - 1)] + offset)];
      }
    }
    else if (name.substr(0, 2) == "c_" or name.substr(0, 11) == "averaged_c_")  // c_1, c_2 ,...
    {
      my_data.resize(numnp);
      int k(0);
      if ((name == "c_1") or (name == "averaged_c_1"))
        k = 0;
      else if ((name == "c_2") or (name == "averaged_c_2"))
        k = 1;
      else if ((name == "c_3") or (name == "averaged_c_3"))
        k = 2;
      else if ((name == "c_4") or (name == "averaged_c_4"))
        k = 3;
      else if ((name == "c_5") or (name == "averaged_c_5"))
        k = 4;
      else
        FOUR_C_THROW("Up to now, I'm not able to write a field named {}\n", name);

      for (int inode = 0; inode < numnp; ++inode)
      {
        Core::Nodes::Node* n = nurbsdis->l_row_node(inode);
        int numdofpernode = actele->num_dof_per_node(*n);
        my_data[inode] = (coldata)[(coldata).get_map().lid(lm[inode * numdofpernode + k] + offset)];
      }
    }
    else if (name == "normalflux")
    {
      my_data.resize(numnp);

      for (int inode = 0; inode < numnp; ++inode)
      {
        // Core::Nodes::Node* n = nurbsdis->lRowNode(inode);
        int numdofpernode = 1;  // actele->NumDofPerNode(*n);
        my_data[inode] = (coldata)[(coldata).get_map().lid(lm[inode * numdofpernode] + offset)];
      }
    }
    //---------------------------------------------------
    // contact - specific output:
    else if (name == "norcontactstress" || name == "tancontactstress" ||
             name == "interfacetraction" || name == "slaveforces" || name == "masterforces" ||
             name == "norslaveforce" || name == "tanslaveforce" || name == "normasterforce" ||
             name == "tanmasterforce" || name == "wear" || name == "norslaveforcelm" ||
             name == "norslaveforceg" || name == "normasterforcelm" || name == "normasterforceg")
    {
      my_data.resize(dim * numnp);

      for (int inode = 0; inode < numnp; ++inode)
      {
        for (int rr = 0; rr < dim; ++rr)
        {
          my_data[dim * inode + rr] =
              (coldata)[(coldata).get_map().lid(lm[inode * dim + rr] + offset)];
        }
      }
    }
    else if (name == "temperature")
    {
      for (int inode = 0; inode < numnp; ++inode)
        my_data[inode] = (coldata)[(coldata).get_map().lid(lm[inode] + offset)];
    }
    else
    {
      FOUR_C_THROW("Up to now, I'm not able to write a field named {}\n", name);
    }

    interpolate_nurbs_result_to_viz_points(*idata, dim, npatch, vpoff, ele_cart_id, actele,
        nurbsdis, eleknots, weights, numdf, my_data);

  }  // loop over available elements

  // import my new values (proc0 gets everything, other procs empty)
  Core::LinAlg::Import proc0importer(*proc0map_, *vispointmap_);
  Core::LinAlg::MultiVector<double> allsols(*proc0map_, numdf);
  allsols.import(*idata, proc0importer, Insert);

  // write the node results (only proc 0)
  // ensight format requires u_1 .. u_n, v_1 .. v_n, w_1 ... w_n, as for nodes
  // this is fulfilled automatically due to Core::LinAlg::MultiVector<double> usage (columnwise
  // writing data)
  if (myrank_ == 0)
  {
    double* solvals = allsols.get_values();
    int numentries = (numdf * (allsols.global_length()));

    // now write the solution
    for (int i = 0; i < numentries; ++i)
    {
      write(file, static_cast<float>(solvals[i]));
    }

    // 2 component vectors in a 3d problem require a row of zeros.
    // do we really need this?
    if (numdf == 2)
    {
      for (int inode = 0; inode < numvispoints; inode++)
      {
        write<float>(file, 0.);
      }
    }
  }
}  // EnsightWriter::write_dof_result_step_for_nurbs


/*----------------------------------------------------------------------*/
/*
    Perform interpolation of result data to visualization points.
    This routine is used for dofmap-based as well as for nodemap-based
    data. The results for the current element have to be provided in
    my_data accordingly.
*/
/*----------------------------------------------------------------------*/
void EnsightWriter::interpolate_nurbs_result_to_viz_points(Core::LinAlg::MultiVector<double>& idata,
    const int dim, const int npatch, const std::vector<int>& vpoff,
    const std::vector<int>& ele_cart_id, const Core::Elements::Element* actele,
    Core::FE::Nurbs::NurbsDiscretization* nurbsdis,
    const std::vector<Core::LinAlg::SerialDenseVector>& eleknots,
    const Core::LinAlg::SerialDenseVector& weights, const int numdf,
    const std::vector<double>& my_data) const
{
  using namespace FourC;

  // number of all control points of the element
  const int numnp = actele->num_node();

  // get shapefunctions, compute all visualisation point positions
  Core::LinAlg::SerialDenseVector nurbs_shape_funct(numnp);

  // element local visualisation point position
  Core::LinAlg::SerialDenseVector uv(dim);

  // get nele_x_mele_x_lele array
  std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(npatch));

  switch (actele->shape())
  {
    case Core::FE::CellType::nurbs4:
    {
      // number of visualisation points in u direction
      int nvpu = (nurbsdis->return_nele_x_mele_x_lele(npatch))[0] + 1;

      {
        // standard

        // 3           4
        //  X---------X
        //  |         |
        //  |         |
        //  |         |
        //  |         |
        //  |         |
        //  X---------X
        // 1           2

        // point 1
        uv(0) = -1.0;
        uv(1) = -1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid((ele_cart_id[1]) * (nvpu) + ele_cart_id[0]);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 2
        uv(0) = 1.0;
        uv(1) = -1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid((ele_cart_id[1]) * (nvpu) + ele_cart_id[0] + 1);
          (idata).replace_local_value(lid, isd, val);
        }


        // point 3
        uv(0) = -1.0;
        uv(1) = 1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid((ele_cart_id[1] + 1) * (nvpu) + ele_cart_id[0]);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 4
        uv(0) = 1.0;
        uv(1) = 1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid((ele_cart_id[1] + 1) * (nvpu) + ele_cart_id[0] + 1);
          (idata).replace_local_value(lid, isd, val);
        }
      }
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      int idu;
      int idv;

      // number of visualisation points in u direction
      int nvpu = 2 * (nurbsdis->return_nele_x_mele_x_lele(npatch))[0] + 1;

      {
        // standard

        //
        //  +---------+
        //  |         |
        //  |         |
        //  X    X    |
        // 3|   4     |
        //  |         |
        //  X----X----+
        // 1    2

        // point 1
        uv(0) = -1.0;
        uv(1) = -1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        idu = 2 * ele_cart_id[0];
        idv = 2 * ele_cart_id[1] * (nvpu);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 2
        uv(0) = 0.0;
        uv(1) = -1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        idu = 2 * ele_cart_id[0] + 1;
        idv = 2 * ele_cart_id[1] * (nvpu);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idv + idu);
          (idata).replace_local_value(lid, isd, val);
        }


        // point 3
        uv(0) = -1.0;
        uv(1) = 0.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        idu = 2 * ele_cart_id[0];
        idv = (2 * ele_cart_id[1] + 1) * (nvpu);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 4
        uv(0) = 0.0;
        uv(1) = 0.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        idu = 2 * ele_cart_id[0] + 1;
        idv = (2 * ele_cart_id[1] + 1) * (nvpu);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv);
          (idata).replace_local_value(lid, isd, val);
        }
      }
      // top line

      //
      //  X----X----+
      // 5|   6     |
      //  |         |
      //  X    X    |
      // 3|   4     |
      //  |         |
      //  X----X----+
      // 1    2
      //
      // two additional points

      if (ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
      {
        // point 5
        uv(0) = -1.0;
        uv(1) = 1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid =
              (*vispointmap_)
                  .lid(vpoff[npatch] + (2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0]);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 6
        uv(0) = 0.0;
        uv(1) = 1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid =
              (*vispointmap_)
                  .lid(vpoff[npatch] + (2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0] + 1);
          (idata).replace_local_value(lid, isd, val);
        }
      }

      // right line
      if (ele_cart_id[0] + 1 == nele_x_mele_x_lele[0])
      {
        //
        //  +---------+
        //  |         |
        //  |         |
        //  x    x    X
        // 4|   5    6|
        //  |         |
        //  x----x----X
        // 1    2    3

        // two additional points
        // point 5
        uv(0) = 1.0;
        uv(1) = -1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid =
              (*vispointmap_)
                  .lid(vpoff[npatch] + (2 * ele_cart_id[1]) * (nvpu) + 2 * ele_cart_id[0] + 2);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 6
        uv(0) = 1.0;
        uv(1) = 0.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid =
              (*vispointmap_)
                  .lid(vpoff[npatch] + (2 * ele_cart_id[1] + 1) * (nvpu) + 2 * ele_cart_id[0] + 2);
          (idata).replace_local_value(lid, isd, val);
        }
      }

      // top right corner
      if (ele_cart_id[0] + 1 == nele_x_mele_x_lele[0] &&
          ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
      {
        //
        //  x----x----X
        // 7|   8    9|
        //  |         |
        //  x    x    x
        // 4|   5    6|
        //  |         |
        //  x----x----x
        // 1    2    3

        // point 9
        uv(0) = 1.0;
        uv(1) = 1.0;
        Core::FE::Nurbs::nurbs_get_2d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid =
              (*vispointmap_)
                  .lid(vpoff[npatch] + (2 * ele_cart_id[1] + 2) * (nvpu) + 2 * ele_cart_id[0] + 2);
          (idata).replace_local_value(lid, isd, val);
        }
      }
      break;
    }
    case Core::FE::CellType::nurbs27:
    {
      // element local point position
      Core::LinAlg::SerialDenseVector uv(3);

      int idu;
      int idv;
      int idw;

      {
        // standard

        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /         /  |
        //  +---------+   |
        //  | A----A  |   |
        //  |/|   /|  |   +
        //  A----A |  |  /
        //  | A--|-A  | /
        //  |/   |/   |/
        //  A----A----+ ----->u
        //
        // append 8 points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0) = -1.0;
        uv(1) = -1.0;
        uv(2) = -1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0]);
        idv = (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2]) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 2
        uv(0) = 0.0;
        uv(1) = -1.0;
        uv(2) = -1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 1);
        idv = (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2]) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 3
        uv(0) = -1.0;
        uv(1) = 0.0;
        uv(2) = -1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0]);
        idv = (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2]) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 4
        uv(0) = 0.0;
        uv(1) = 0.0;
        uv(2) = -1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 1);
        idv = (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2]) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 5
        uv(0) = -1.0;
        uv(1) = -1.0;
        uv(2) = 0.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0]);
        idv = (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 1) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 6
        uv(0) = 0.0;
        uv(1) = -1.0;
        uv(2) = 0.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 1);
        idv = (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 1) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 7
        uv(0) = -1.0;
        uv(1) = 0.0;
        uv(2) = 0.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0]);
        idv = (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 1) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 8
        uv(0) = 0.0;
        uv(1) = 0.0;
        uv(2) = 0.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 1);
        idv = (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 1) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }
      }

      if (ele_cart_id[0] + 1 == nele_x_mele_x_lele[0])
      {
        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /         /  |
        //  +---------+   |
        //  | X----X--|-A |
        //  |/|   /|  |/| +
        //  X----X----A |/
        //  | X--|-X--|-A
        //  |/   |/   |/
        //  X----X----A ----->u
        //
        // append 4 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0) = 1.0;
        uv(1) = -1.0;
        uv(2) = -1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        idu = (2 * ele_cart_id[0] + 2);
        idv = (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2]) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 2
        uv(0) = 1.0;
        uv(1) = 0.0;
        uv(2) = -1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 2);
        idv = (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2]) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 3
        uv(0) = 1.0;
        uv(1) = -1.0;
        uv(2) = 0.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 2);
        idv = (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 1) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 4
        uv(0) = 1.0;
        uv(1) = 0.0;
        uv(2) = 0.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 2);
        idv = (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 1) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }
      }

      if (ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
      {
        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /  A----A /  |
        //  +---------+   |
        //  | X----X ||   |
        //  |/| A-/|-A|   +
        //  X----X |/ |  /
        //  | X--|-X  | /
        //  |/   |/   |/
        //  X----X----+ ----->u
        //
        // append 4 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0) = -1.0;
        uv(1) = 1.0;
        uv(2) = -1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0]);
        idv = (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2]) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }


        // point 2
        uv(0) = 0.0;
        uv(1) = 1.0;
        uv(2) = -1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 1);
        idv = (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2]) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 3
        uv(0) = -1.0;
        uv(1) = 1.0;
        uv(2) = 0.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0]);
        idv = (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 1) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 4
        uv(0) = 0.0;
        uv(1) = 1.0;
        uv(2) = 0.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 1);
        idv = (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 1) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }
      }

      if (ele_cart_id[0] + 1 == nele_x_mele_x_lele[0] &&
          ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
      {
        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /  X----X-/--A
        //  +---------+  /|
        //  | X----X--|-X |
        //  |/| X-/|-X|/|-A
        //  X----X----X |/
        //  | X--|-X--|-X
        //  |/   |/   |/
        //  X----X----X ----->u
        //
        // append 2 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0) = 1.0;
        uv(1) = 1.0;
        uv(2) = -1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());
        idu = (2 * ele_cart_id[0] + 2);
        idv = (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2]) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 2
        uv(0) = 1.0;
        uv(1) = 1.0;
        uv(2) = 0.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 2);
        idv = (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 1) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }
      }


      if (ele_cart_id[2] + 1 == nele_x_mele_x_lele[2])
      {
        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | A----A    / |
        //   /    /|   /  |
        //  A----A----+   |
        //  | X--|-X  |   |
        //  |/|  |/|  |   +
        //  X----X |  |  /
        //  | X--|-X  | /
        //  |/   |/   |/
        //  X----X----+ ----->u
        //
        //
        // append 4 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0) = -1.0;
        uv(1) = -1.0;
        uv(2) = 1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0]);
        idv = (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 2) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 2
        uv(0) = 0.0;
        uv(1) = -1.0;
        uv(2) = 1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 1);
        idv = (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 2) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 3
        uv(0) = -1.0;
        uv(1) = 0.0;
        uv(2) = 1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0]);
        idv = (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 2) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 4
        uv(0) = 0.0;
        uv(1) = 0.0;
        uv(2) = 1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 1);
        idv = (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 2) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }
      }


      if (ele_cart_id[2] + 1 == nele_x_mele_x_lele[2] &&
          ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
      {
        //               v
        //              /
        //  w          /
        //  ^   A----A----+
        //  |  /|   /|   /|
        //  | X----X |  / |
        //   /| X /| X /  |
        //  X----X----+   |
        //  | X--|-X ||   |
        //  |/|  |/| X|   +
        //  X----X |/ |  /
        //  | X--|-X  | /
        //  |/   |/   |/
        //  X----X----+ ----->u
        //
        //
        // append 2 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0) = -1.0;
        uv(1) = 1.0;
        uv(2) = 1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0]);
        idv = (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 2) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 2
        uv(0) = 0.0;
        uv(1) = 1.0;
        uv(2) = 1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 1);
        idv = (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 2) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }
      }

      if (ele_cart_id[2] + 1 == nele_x_mele_x_lele[2] &&
          ele_cart_id[0] + 1 == nele_x_mele_x_lele[0])
      {
        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | X----X----A |
        //   /    /|   /| |
        //  X----X----A | |
        //  | X--|-X--|-X |
        //  |/|  |/|  |/| +
        //  X----X----X |/
        //  | X--|-X--|-X
        //  |/   |/   |/
        //  X----X----X ----->u
        //
        //
        // append 2 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0) = 1.0;
        uv(1) = -1.0;
        uv(2) = 1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 2);
        idv = (2 * ele_cart_id[1]) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 2) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }

        // point 2
        uv(0) = 1.0;
        uv(1) = 0.0;
        uv(2) = 1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 2);
        idv = (2 * ele_cart_id[1] + 1) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 2) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }
      }

      if (ele_cart_id[2] + 1 == nele_x_mele_x_lele[2] &&
          ele_cart_id[1] + 1 == nele_x_mele_x_lele[1] &&
          ele_cart_id[0] + 1 == nele_x_mele_x_lele[0])
      {
        //               v
        //              /
        //  w          /
        //  ^   X----X----A
        //  |  /|   /    /|
        //  | X----X----X |
        //   /| X-/|-X-/|-X
        //  X----X----X |/|
        //  | X--|-X--|-X |
        //  |/| X|/|-X|/|-X
        //  X----X----X |/
        //  | X--|-X--|-X
        //  |/   |/   |/
        //  X----X----X ----->u
        //
        // append 1 additional point

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0) = 1.0;
        uv(1) = 1.0;
        uv(2) = 1.0;
        Core::FE::Nurbs::nurbs_get_3d_funct(
            nurbs_shape_funct, uv, eleknots, weights, actele->shape());

        idu = (2 * ele_cart_id[0] + 2);
        idv = (2 * ele_cart_id[1] + 2) * (2 * nele_x_mele_x_lele[0] + 1);
        idw = (2 * ele_cart_id[2] + 2) * (2 * nele_x_mele_x_lele[1] + 1) *
              (2 * nele_x_mele_x_lele[0] + 1);

        for (int isd = 0; isd < numdf; ++isd)
        {
          double val = 0;
          for (int inode = 0; inode < numnp; ++inode)
          {
            val += my_data[numdf * inode + isd] * nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).lid(vpoff[npatch] + idu + idv + idw);
          (idata).replace_local_value(lid, isd, val);
        }
      }

      break;
    }
    default:
      FOUR_C_THROW("unable to visualise this as a nurbs discretisation\n");
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_nodal_result_step_for_nurbs(std::ofstream& file, const int numdf,
    Core::LinAlg::MultiVector<double>& data, const std::string name, const int offset) const
{
  using namespace FourC;

  // a multivector for the interpolated data
  std::shared_ptr<Core::LinAlg::MultiVector<double>> idata;
  idata = std::make_shared<Core::LinAlg::MultiVector<double>>(*vispointmap_, numdf);

  Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*field_->discretization()));

  if (nurbsdis == nullptr)
  {
    FOUR_C_THROW("This probably isn't a NurbsDiscretization\n");
  }

  // get number of patches
  int npatches = (nurbsdis->get_knot_vector())->return_np();

  // assuming that dimension of the manifold is
  // equal to spatial dimension
  int dim = (int)(nurbsdis->return_nele_x_mele_x_lele(0)).size();

  for (int np = 0; np < npatches; ++np)
  {
    // get nurbs dis' knotvector sizes
    std::vector<int> n_x_m_x_l(nurbsdis->return_n_x_m_x_l(np));

    // get nurbs dis' knotvector sizes
    std::vector<int> degree(nurbsdis->return_degree(np));

  }  // end loop over patches

  // get the knotvector itself
  std::shared_ptr<Core::FE::Nurbs::Knotvector> knotvec = nurbsdis->get_knot_vector();

  // get vispoint offsets among patches
  std::vector<int> vpoff(npatches);

  vpoff[0] = 0;

  // loop all patches
  for (int np = 1; np < npatches; ++np)
  {
    // get nurbs dis' knotvector sizes
    std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(np - 1));

    int numvisp = 1;

    for (unsigned rr = 0; rr < nele_x_mele_x_lele.size(); ++rr)
    {
      numvisp *= 2 * nele_x_mele_x_lele[rr] + 1;
    }

    vpoff[np] = vpoff[np - 1] + numvisp;
  }

  // get element map
  const Core::LinAlg::Map* elementmap = nurbsdis->element_row_map();

  // construct a colmap for nodal data to have it available at
  // all elements (the critical ones are the ones at the
  // processor boundary)
  // loop all available elements
  std::set<int> colnodeset;
  for (int iele = 0; iele < elementmap->num_my_elements(); ++iele)
  {
    Core::Elements::Element* actele = nurbsdis->g_element(elementmap->gid(iele));
    /*
        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;

        // extract local values from the global vectors
        actele->LocationVector(*nurbsdis,lm,lmowner,lmstride);
    */
    // do not forget to consider a (potential) offset in dof numbering for all results!
    const int* nodeids = actele->node_ids();
    for (int inode = 0; inode < actele->num_node(); ++inode)
    {
      colnodeset.insert(nodeids[inode]);
    }
  }

  std::vector<int> colnodemapvec;
  colnodemapvec.reserve(colnodeset.size());
  colnodemapvec.assign(colnodeset.begin(), colnodeset.end());
  colnodeset.clear();
  Core::LinAlg::Map colnodemap(
      -1, colnodemapvec.size(), colnodemapvec.data(), 0, nurbsdis->get_comm());
  colnodemapvec.clear();

  const Core::LinAlg::Map* fullnodemap = &(colnodemap);
  Core::LinAlg::MultiVector<double> coldata(*fullnodemap, numdf, true);  // numdf important!!!

  // create an importer and import the data
  Core::LinAlg::Import importer((coldata).get_map(), (data).get_map());
  coldata.import((data), importer, Insert);

  // loop all available elements
  for (int iele = 0; iele < elementmap->num_my_elements(); ++iele)
  {
    Core::Elements::Element* actele = nurbsdis->g_element(elementmap->gid(iele));

    // get gid, location in the patch
    std::vector<int> ele_cart_id(dim);
    int gid = actele->id();
    int np = -1;

    // access elements knot span
    std::vector<Core::LinAlg::SerialDenseVector> knots(dim);
    bool zero_size = (*knotvec).get_ele_knots(knots, actele->id());

    knotvec->convert_ele_gid_to_knot_ids(gid, np, ele_cart_id);

    // zero sized elements in knot span cannot be visualised
    if (zero_size)
    {
      // if we just skip them, we would loose some connectivity
      // in the result;
      // as a work-around, we replace the zero-sized element
      // with the next nonzero element --- this preserves the
      // connectivity, and everything looks nice and smooth again.
      // Note: This work-around will not work in parallel (or a
      //       special ghosting for elements along interpolated
      //       boundaries has to be applied)
      // Note: The following element will be plotted twice

      actele = nurbsdis->g_element(knotvec->return_next_nonzero_ele_gid(actele->id()));
      (*knotvec).get_ele_knots(knots, actele->id());
    }

    Core::Nodes::Node** nodes = actele->nodes();

    // number of all control points of the element
    const int numnp = actele->num_node();

    // acquire weights from nodes
    Core::LinAlg::SerialDenseVector weights(numnp);

    for (int inode = 0; inode < numnp; ++inode)
    {
      Core::FE::Nurbs::ControlPoint* cp =
          dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes[inode]);
      weights(inode) = cp->w();
    }

    int npatch = np;

    // access elements knot span
    std::vector<Core::LinAlg::SerialDenseVector> eleknots(dim);
    knotvec->get_ele_knots(eleknots, actele->id());

    // access solution data
    std::vector<double> my_data(numnp * numdf);

    const int* nodeids = actele->node_ids();

    for (int inode = 0; inode < numnp; ++inode)
    {
      for (int rr = 0; rr < numdf; ++rr)
      {
        // value of nodemap-based column rr
        my_data[numdf * inode + rr] = (((coldata)(rr)))[(coldata).get_map().lid(nodeids[inode])];
      }
    }

    // intoplate solution to desired visualization points
    interpolate_nurbs_result_to_viz_points(*idata, dim, npatch, vpoff, ele_cart_id, actele,
        nurbsdis, eleknots, weights, numdf, my_data);

  }  // loop over available elements

  // import my new values (proc0 gets everything, other procs empty)
  Core::LinAlg::Import proc0importer(*proc0map_, *vispointmap_);
  Core::LinAlg::MultiVector<double> allsols(*proc0map_, numdf);
  allsols.import(*idata, proc0importer, Insert);

  //---------------
  // write results
  //---------------

  const int finalnumnode = proc0map_->num_global_elements();

  if (myrank_ == 0)
  {
    for (int idf = 0; idf < numdf; ++idf)
    {
      Core::LinAlg::Vector<double> column((allsols)(idf));
      for (int inode = 0; inode < finalnumnode;
          inode++)  // inode == lid of node because we use proc0map_
      {
        //        Write(file, static_cast<float>(idf));
        write(file, static_cast<float>((column)[inode]));
      }
    }
  }  // if (myrank_==0)

}  // EnsightWriter::write_nodal_result_step_for_nurbs


FOUR_C_NAMESPACE_CLOSE
