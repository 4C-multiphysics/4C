// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_output.hpp"

#include "4C_cut_boundarycell.hpp"
#include "4C_cut_cycle.hpp"
#include "4C_cut_edge.hpp"
#include "4C_cut_element.hpp"
#include "4C_cut_facet.hpp"
#include "4C_cut_kernel.hpp"
#include "4C_cut_line.hpp"
#include "4C_cut_point.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_volumecell.hpp"

#include <iosfwd>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_facets_only(
    const plain_facet_set& facets, Element* ele, const std::string& file_affix)
{
  // write details of volume cells
  std::stringstream str;
  str << ".facets" << (not file_affix.empty() ? "_" + file_affix : "") << "_CUTFAIL.pos";
  std::string filename(Cut::Output::generate_gmsh_output_filename(str.str()));

  std::cout << "\nFacets of element " << ele->id() << " are written to " << filename << "\n";

  std::ofstream file(filename.c_str());
  int count = 0;
  for (plain_facet_set::const_iterator it = facets.begin(); it != facets.end(); ++it)
  {
    std::ostringstream sectionname;
    sectionname << "Facet_" << count++;
    Output::gmsh_new_section(file, sectionname.str(), false);
    Output::gmsh_facet_dump(file, *it, "sides", true, false, ele);
    Output::gmsh_end_section(file);
  }
  file.close();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_edges_only(const plain_edge_set& edges)
{
  std::stringstream str;
  str << ".edges"
      << "_CUTFAIL.pos";
  std::string filename(Cut::Output::generate_gmsh_output_filename(str.str()));
  std::ofstream file(filename.c_str());
  Core::IO::cout << "\nEdges are written to " << filename << "\n";

  int count = 0;
  for (plain_edge_set::const_iterator it = edges.begin(); it != edges.end(); ++it)
  {
    std::ostringstream sectionname;
    sectionname << "Edge_" << count++;
    Output::gmsh_new_section(file, sectionname.str(), false);
    Output::gmsh_edge_dump(file, *it);
    Output::gmsh_end_section(file);
  }
  file.close();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_volume_cells_only(const plain_volumecell_set& vcells)
{
  // write details of volume cells
  Core::IO::cout << "\nVolumeCells are written to [...].volumecells_CUTFAIL.pos\n";

  std::stringstream str;
  str << ".volumecells"
      << "_CUTFAIL.pos";
  std::string filename(Cut::Output::generate_gmsh_output_filename(str.str()));
  std::ofstream file(filename.c_str());
  int count = 0;
  for (plain_volumecell_set::const_iterator its = vcells.begin(); its != vcells.end(); ++its)
  {
    std::ostringstream sectionname;
    sectionname << "VolumeCell_" << count++;
    Output::gmsh_new_section(file, sectionname.str(), false);
    Output::gmsh_volumecell_dump(file, *its, "sides", true, false, (*its)->parent_element());
    Output::gmsh_end_section(file);
  }
  file.close();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
char Cut::Output::gmsh_element_type(Core::FE::CellType shape)
{
  switch (shape)
  {
    case Core::FE::CellType::point1:
    {
      return 'P';
    }
    case Core::FE::CellType::line2:
    {
      return 'L';
    }
    case Core::FE::CellType::tri3:
    {
      return 'T';
    }
    case Core::FE::CellType::quad4:
    {
      return 'Q';
    }
    case Core::FE::CellType::hex8:
    {
      return 'H';
    }
    case Core::FE::CellType::tet4:
    {
      return 'S';
    }
    case Core::FE::CellType::wedge6:
    {
      return 'I';
    }
    case Core::FE::CellType::pyramid5:
    {
      return 'P';
    }
    case Core::FE::CellType::dis_none:
    {
      // arbitrary cells are not yet supported
      return ' ';
    }
    default:
      FOUR_C_THROW(
          "Unsupported cell shape! ( shape = {} )", Core::FE::cell_type_to_string(shape).c_str());
      exit(EXIT_FAILURE);
  }
  // impossible to reach this point
  exit(EXIT_FAILURE);
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given element                                           sudhakar 03/14
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_element_dump(std::ofstream& file, Element* ele, bool to_local)
{
  const std::vector<Node*>& nodes = ele->nodes();
  char elementtype = gmsh_element_type(ele->shape());
  gmsh_element_dump(file, nodes, elementtype, to_local, ele);
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given element                                           sudhakar 03/14
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_element_dump(std::ofstream& file, const std::vector<Cut::Node*>& nodes,
    char elementtype, bool to_local, Element* ele)
{
  file << "S" << elementtype << "(";
  for (std::vector<Cut::Node*>::const_iterator i = nodes.begin(); i != nodes.end(); ++i)
  {
    if (i != nodes.begin()) file << ",";
    gmsh_write_coords(file, *i, to_local, ele);
  }
  file << "){";
  for (std::vector<Cut::Node*>::const_iterator i = nodes.begin(); i != nodes.end(); ++i)
  {
    Cut::Node* n = *i;
    Cut::Point* p = n->point();
    if (i != nodes.begin()) file << ",";
#if CUT_DEVELOP
    file << p->Id();
#else
    file << p->position();
#endif
  }
  file << "};\n";
}



/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_cell_dump(std::ofstream& file, Core::FE::CellType shape,
    const Core::LinAlg::SerialDenseMatrix& xyze, const Point::PointPosition* position,
    const int* value)
{
  char elementtype = gmsh_element_type(shape);

  file.precision(16);
  file << "S" << elementtype << "(";
  for (unsigned i = 0; i < static_cast<unsigned>(xyze.numCols()); ++i)
  {
    if (i > 0) file << ", ";
    for (unsigned j = 0; j < static_cast<unsigned>(xyze.numRows()); ++j)
    {
      if (j > 0) file << ",";
      file << xyze(j, i);
    }
  }
  file << "){";
  for (unsigned i = 0; i < static_cast<unsigned>(xyze.numCols()); ++i)
  {
    if (i > 0) file << ",";
    if (value)
      file << (*value);
    else if (position)
      file << (*position);
    // dummy value
    else
      file << 0.0;
  }
  file << "};\n";
}

void Cut::Output::gmsh_side_dump(
    std::ofstream& file, const Side* s, const std::string& sname, bool to_local, Element* ele)
{
  gmsh_new_section(file, sname, false);
  gmsh_side_dump(file, s, false, ele);
  file << "};\n";
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given side                                       sudhakar 03/14
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_side_dump(std::ofstream& file, const Side* s, bool to_local, Element* ele)
{
  const std::vector<Node*>& nodes = s->nodes();
  char elementtype;
  switch (nodes.size())
  {
    case 0:
      return;  // I'm a Levelset Side - do nothing!
    case 2:
      elementtype = 'L';
      break;
    case 3:
      elementtype = 'T';
      break;
    case 4:
      elementtype = 'Q';
      break;
    default:
      std::stringstream str;
      str << "unknown element type in gmsh_side_dump for " << nodes.size() << " nodes!";
      FOUR_C_THROW("{}", str.str());
      exit(EXIT_FAILURE);
  }
  gmsh_element_dump(file, nodes, elementtype, to_local, ele);
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given side                                                ager 04/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_tri_side_dump(
    std::ofstream& file, std::vector<Point*> points, bool to_local, Element* ele)
{
  char elementtype;
  switch (points.size())
  {
    case 3:
      elementtype = 'T';
      break;
    case 4:
      elementtype = 'Q';
      break;
    default:
    {
      std::stringstream str;
      str << "unknown element type in GmshTriSideDump for " << points.size() << " points!";
      FOUR_C_THROW("{}", str.str());
      exit(EXIT_FAILURE);
    }
  }

  file << "S" << elementtype << "(";
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    if (i != 0) file << ",";
    gmsh_write_coords(file, points[i], to_local, ele);
  }
  file << "){";
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    Cut::Point* p = points[i];
    if (i != 0) file << ",";
    file << p->position();
  }
  file << "};\n";
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given facet                                                ager 08/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_facet_dump(std::ofstream& file, Facet* facet,
    const std::string& visualizationtype, bool print_all, bool to_local, Element* ele)
{
  if (to_local)
  {
    if (ele == nullptr)
      FOUR_C_THROW("GmshWriteCoords: Didn't get a parent element for the Coordinate!");
  }

  if (visualizationtype == "sides")
  {
    if (facet->points().size() == 1)
    {
      gmsh_facet_dump(file, facet, "points", print_all, to_local, ele);
      return;
    }

    if (facet->points().size() == 2)
    {
      gmsh_facet_dump(file, facet, "lines", print_all, to_local, ele);
      return;
    }

    if (facet->is_triangulated())
    {
      for (unsigned j = 0; j < facet->triangulation().size(); ++j)
      {
        Cut::Output::gmsh_tri_side_dump(file, facet->triangulation()[j], to_local, ele);
      }
    }
    else if (facet->is_facet_split())
    {
      for (unsigned j = 0; j < facet->get_split_cells().size(); ++j)
      {
        Cut::Output::gmsh_tri_side_dump(file, facet->get_split_cells()[j], to_local, ele);
      }
    }
    else if (facet->belongs_to_level_set_side() || facet->corner_points().size() == 3 ||
             facet->corner_points().size() == 4)
    {
      Cut::Output::gmsh_tri_side_dump(file, facet->corner_points(), to_local, ele);
    }
    else if (facet->corner_points().size() > 2 &&
             print_all)  // do mitpoint triangulation just for visualization reasons! (not usedfull
                         // if you want to check if a triangulation exists!)
    {
      std::vector<double> xmid(3, 0);
      for (unsigned int i = 0; i < facet->corner_points().size(); ++i)
      {
        for (unsigned int dim = 0; dim < 3; ++dim) xmid[dim] += facet->corner_points()[i]->x()[dim];
      }
      for (unsigned int dim = 0; dim < 3; ++dim)
        xmid[dim] = xmid[dim] / facet->corner_points().size();

      ConcretePoint<3> midpoint = ConcretePoint<3>(-1, xmid.data(), nullptr, nullptr, 0.0);

      std::vector<Point*> tri;
      for (unsigned int i = 0; i < facet->corner_points().size(); ++i)
      {
        tri.clear();
        tri.push_back(facet->corner_points()[i]);
        tri.push_back(facet->corner_points()[(i + 1) % facet->corner_points().size()]);
        tri.push_back(&midpoint);
        Cut::Output::gmsh_tri_side_dump(file, tri, to_local, ele);
      }
    }
  }
  else if (visualizationtype == "lines")
  {
    for (std::size_t pidx = 0; pidx < facet->points().size(); ++pidx)
    {
      Cut::Output::gmsh_line_dump(file, facet->points()[pidx],
          facet->points()[(pidx + 1) % facet->points().size()], to_local, ele);
    }
  }
  else if (visualizationtype == "points")
  {
    for (std::size_t pidx = 0; pidx < facet->points().size(); ++pidx)
      Cut::Output::gmsh_point_dump(file, facet->points()[pidx], facet->side_id(), to_local, ele);
  }
  else
    FOUR_C_THROW("gmsh_facet_dump: unknown visualizationtype!");
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given volumecell                                       ager 08/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_volumecell_dump(std::ofstream& file, VolumeCell* VC,
    const std::string& visualizationtype, bool print_all, bool to_local, Element* ele)
{
  for (plain_facet_set::const_iterator j = VC->facets().begin(); j != VC->facets().end(); j++)
    gmsh_facet_dump(file, *j, visualizationtype, print_all, to_local, ele);
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given cycle                                                ager 08/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_cycle_dump(std::ofstream& file, Cycle* cycle,
    const std::string& visualizationtype, bool to_local, Element* ele)
{
  if (visualizationtype == "points")
  {
    for (unsigned i = 0; i != (*cycle)().size(); ++i)
      gmsh_point_dump(file, (*cycle)()[i], to_local, ele);
  }
  else if (visualizationtype == "lines")
  {
    for (unsigned i = 0; i != (*cycle)().size(); ++i)
      gmsh_line_dump(file, (*cycle)()[i], (*cycle)()[(i + 1) % (*cycle)().size()], to_local, ele);
  }
  else
    FOUR_C_THROW("gmsh_facet_dump: unknown visualizationtype!");
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of element along with all its cut sides                sudhakar 03/14
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_complete_cut_element(std::ofstream& file, Element* ele, bool to_local)
{
  // write details of background element
  gmsh_new_section(file, "Element");
  gmsh_element_dump(file, ele, to_local);

  // write details of points
  gmsh_new_section(file, "Points", true);
  const std::vector<Point*>& ps = ele->points();
  for (std::vector<Point*>::const_iterator its = ps.begin(); its != ps.end(); its++)
    gmsh_point_dump(file, *its, to_local, ele);

  // write details of cut sides
  gmsh_new_section(file, "Cut_Facets", true);
  const plain_facet_set& facets = ele->facets();
  for (plain_facet_set::const_iterator its = facets.begin(); its != facets.end(); its++)
  {
    if ((*its)->parent_side()->is_cut_side())
      gmsh_facet_dump(file, *its, "sides", true, to_local, ele);
  }

  // write details of cut sides
  gmsh_new_section(file, "Ele_Facets", true);
  for (plain_facet_set::const_iterator its = facets.begin(); its != facets.end(); its++)
  {
    if (!(*its)->parent_side()->is_cut_side())
      gmsh_facet_dump(file, *its, "sides", true, to_local, ele);
  }

  // write details of volumecells
  const plain_volumecell_set& vcs = ele->volume_cells();
  for (plain_volumecell_set::const_iterator its = vcs.begin(); its != vcs.end(); its++)
  {
    gmsh_new_section(file, "Volumecells", true);
    gmsh_volumecell_dump(file, *its, "sides", true, to_local, ele);
    gmsh_end_section(file);
  }

  // write details of cut sides
  gmsh_new_section(file, "Cut sides", false);
  const plain_side_set& cutsides = ele->cut_sides();
  for (plain_side_set::const_iterator its = cutsides.begin(); its != cutsides.end(); its++)
    gmsh_side_dump(file, *its, to_local, ele);
  gmsh_end_section(file);

  if (ele->has_level_set_side())
  {
    gmsh_new_section(file, "LevelSetValues");
    gmsh_level_set_value_dump(
        file, ele, true, to_local);  // true -> dumps LS values at nodes as well.

    gmsh_new_section(file, "LevelSetGradient", true);
    gmsh_level_set_gradient_dump(file, ele, to_local);

    gmsh_new_section(file, "LevelSetOrientation", true);
    gmsh_level_set_orientation_dump(file, ele, to_local);

    gmsh_new_section(file, "LevelSetZeroShape", true);
    gmsh_level_set_value_zero_surface_dump(file, ele, to_local);
    gmsh_end_section(file);
  }
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given line                                           ager 04/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_line_dump(std::ofstream& file, Cut::Line* line, bool to_local, Element* ele)
{
  gmsh_line_dump(file, line->begin_point(), line->end_point(), to_local, ele);
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given line                                           ager 08/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_line_dump(std::ofstream& file, Cut::Point* p1, Cut::Point* p2, int idx1,
    int idx2, bool to_local, Element* ele)
{
  file << "SL (";
  gmsh_write_coords(file, p1, to_local, ele);
  file << ",";
  gmsh_write_coords(file, p2, to_local, ele);
  file << "){";
  file << idx1 << ",";
  file << idx2;
  file << "};\n";
}


void Cut::Output::gmsh_cut_pair_dump(
    std::ofstream& file, Side* side, Edge* edge, int id, const std::string& suffix)
{
  std::stringstream side_section_name;
  side_section_name << "Side" << suffix << id;
  Cut::Output::gmsh_new_section(file, side_section_name.str());
  Cut::Output::gmsh_side_dump(file, side, false, nullptr);
  Cut::Output::gmsh_end_section(file, false);

  std::stringstream edge_section_name;
  edge_section_name << "Edge" << suffix << id;
  Cut::Output::gmsh_new_section(file, edge_section_name.str());
  Cut::Output::gmsh_edge_dump(file, edge, false, nullptr);
  Cut::Output::gmsh_end_section(file, false);
}


void Cut::Output::gmsh_cut_pair_dump(
    std::ofstream& file, const std::pair<Side*, Edge*>& pair, int id, const std::string& suffix)
{
  gmsh_cut_pair_dump(file, pair.first, pair.second, id, suffix);
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given edge                                           ager 04/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_edge_dump(std::ofstream& file, Cut::Edge* edge, bool to_local, Element* ele)
{
#if CUT_DEVELOP
  GmshLineDump(file, edge->BeginNode()->point(), edge->EndNode()->point(),
      edge->BeginNode()->point()->Id(), edge->EndNode()->point()->Id(), to_local, ele);
#else
  gmsh_line_dump(file, edge->begin_node()->point(), edge->end_node()->point(),
      edge->begin_node()->id(), edge->end_node()->id(), to_local, ele);
#endif
}

void Cut::Output::gmsh_edge_dump(
    std::ofstream& file, Cut::Edge* edge, const std::string& ename, bool to_local, Element* ele)
{
  gmsh_new_section(file, ename, false);
  gmsh_edge_dump(file, edge, to_local, ele);
  file << "};\n";
}


/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given node                                           ager 04/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_node_dump(std::ofstream& file, Cut::Node* node, bool to_local, Element* ele)
{
  gmsh_point_dump(file, node->point(), node->id(), to_local, ele);
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given point                                           ager 04/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_point_dump(
    std::ofstream& file, Cut::Point* point, int idx, bool to_local, Element* ele)
{
  file << "SP (";
  gmsh_write_coords(file, point, to_local, ele);
  file << "){";
  file << idx;
  file << "};\n";
}



void Cut::Output::gmsh_point_dump(std::ofstream& file, Cut::Point* point, int idx,
    const std::string& pname, bool to_local, Element* ele)
{
  gmsh_new_section(file, pname, false);
  gmsh_point_dump(file, point, idx, to_local, ele);
  file << "};\n";
}


/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given point                                           ager 04/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_point_dump(
    std::ofstream& file, Cut::Point* point, bool to_local, Element* ele)
{
  gmsh_point_dump(file, point, point->position(), to_local, ele);
}

/*--------------------------------------------------------------------------------------*
 * Write Level Set Gradient for given Element
 *
 * The gradients are written at the midpoint of the facets and if the facet is triangulated,
 * also in the midpoint of the triangles.
 *                                                                           winter 07/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_level_set_gradient_dump(std::ofstream& file, Element* ele, bool to_local)
{
  const plain_facet_set facets = ele->facets();
  for (plain_facet_set::const_iterator j = facets.begin(); j != facets.end(); j++)
  {
    Facet* facet = *j;

    std::vector<double> normal_triag_midp;
    if (facet->on_cut_side())
    {
      Core::LinAlg::Matrix<3, 1> facet_triang_midpoint_coord(Core::LinAlg::Initialization::zero);

      if (facet->is_triangulated())
      {
        std::vector<std::vector<Point*>> facet_triang = facet->triangulation();
        Point* facet_triang_midpoint = (facet_triang[0])[0];

        facet_triang_midpoint->coordinates(&facet_triang_midpoint_coord(0, 0));
        normal_triag_midp = ele->get_level_set_gradient(facet_triang_midpoint_coord);

        for (std::vector<std::vector<Point*>>::iterator k = facet_triang.begin();
            k != facet_triang.end(); k++)
        {
          std::vector<Point*> facet_triang_tri = *k;

          Core::LinAlg::Matrix<3, 1> cur;
          Core::LinAlg::Matrix<3, 1> f_triang_tri_midp(Core::LinAlg::Initialization::zero);
          for (std::vector<Point*>::iterator i = facet_triang_tri.begin();
              i != facet_triang_tri.end(); i++)
          {
            Point* p1 = *i;
            p1->coordinates(cur.data());
            f_triang_tri_midp.update(1.0, cur, 1.0);
          }
          f_triang_tri_midp.scale(1.0 / facet_triang_tri.size());

          std::vector<double> normal = ele->get_level_set_gradient(f_triang_tri_midp);

          gmsh_vector(file, f_triang_tri_midp, normal, true, to_local, ele);
        }
      }
      else
      {
        Core::LinAlg::Matrix<3, 1> cur;
        std::vector<Point*> pts = facet->points();
        for (std::vector<Point*>::iterator i = pts.begin(); i != pts.end(); i++)
        {
          Point* p1 = *i;
          p1->coordinates(cur.data());
          facet_triang_midpoint_coord.update(1.0, cur, 1.0);
        }
        facet_triang_midpoint_coord.scale(1.0 / pts.size());
        normal_triag_midp = ele->get_level_set_gradient(facet_triang_midpoint_coord);
      }

      std::vector<double> normal = ele->get_level_set_gradient(facet_triang_midpoint_coord);
      gmsh_vector(file, facet_triang_midpoint_coord, normal, true, to_local, ele);

      // Write Corner-points of LS:
      std::vector<Point*> cornerpts = facet->corner_points();
      for (std::vector<Point*>::iterator i = cornerpts.begin(); i != cornerpts.end(); i++)
      {
        Core::LinAlg::Matrix<3, 1> cornercoord;
        Point* p1 = *i;
        p1->coordinates(cornercoord.data());
        std::vector<double> normal = ele->get_level_set_gradient(cornercoord);

        gmsh_vector(file, cornercoord, normal, true, to_local, ele);
      }
    }
  }
}

/*--------------------------------------------------------------------------------------*
 * Write Level Set Values for given Element
 *
 * The LS-value written at the midpoint of the facets and if the facet is triangulated,
 * also in the midpoint of the triangles.
 * Values at the nodes are also written.
 *                                                                           winter 07/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_level_set_value_dump(
    std::ofstream& file, Element* ele, bool dumpnodevalues, bool to_local)
{
  const plain_facet_set facets = ele->facets();
  for (plain_facet_set::const_iterator j = facets.begin(); j != facets.end(); j++)
  {
    Facet* facet = *j;

    if (facet->on_cut_side())
    {
      Core::LinAlg::Matrix<3, 1> facet_triang_midpoint_coord(Core::LinAlg::Initialization::zero);

      if (facet->is_triangulated())
      {
        std::vector<std::vector<Point*>> facet_triang = facet->triangulation();
        for (std::vector<std::vector<Point*>>::iterator k = facet_triang.begin();
            k != facet_triang.end(); k++)
        {
          std::vector<Point*> facet_triang_tri = *k;

          Core::LinAlg::Matrix<3, 1> cur;
          Core::LinAlg::Matrix<3, 1> f_triang_tri_midp(Core::LinAlg::Initialization::zero);
          for (std::vector<Point*>::iterator i = facet_triang_tri.begin();
              i != facet_triang_tri.end(); i++)
          {
            Point* p1 = *i;
            p1->coordinates(cur.data());
            f_triang_tri_midp.update(1.0, cur, 1.0);
          }
          f_triang_tri_midp.scale(1.0 / facet_triang_tri.size());

          double ls_value = ele->get_level_set_value(f_triang_tri_midp);
          gmsh_scalar(file, f_triang_tri_midp, ls_value, to_local, ele);
        }
        Point* facet_triang_midpoint = (facet_triang[0])[0];
        facet_triang_midpoint->coordinates(&facet_triang_midpoint_coord(0, 0));
      }
      else
      {
        Core::LinAlg::Matrix<3, 1> cur;
        std::vector<Point*> pts = facet->points();
        for (std::vector<Point*>::iterator i = pts.begin(); i != pts.end(); i++)
        {
          Point* p1 = *i;
          p1->coordinates(cur.data());
          facet_triang_midpoint_coord.update(1.0, cur, 1.0);
        }
        facet_triang_midpoint_coord.scale(1.0 / pts.size());
      }

      double ls_value = ele->get_level_set_value(facet_triang_midpoint_coord);
      gmsh_scalar(file, facet_triang_midpoint_coord, ls_value, to_local, ele);
    }
  }

  if (dumpnodevalues)
  {
    std::vector<Node*> nodes = ele->nodes();
    for (std::vector<Node*>::iterator j = nodes.begin(); j != nodes.end(); j++)
    {
      Node* node = *j;
      Core::LinAlg::Matrix<3, 1> node_coord(Core::LinAlg::Initialization::zero);
      node->coordinates(&node_coord(0, 0));

      gmsh_scalar(file, node_coord, node->lsv(), to_local, ele);
    }
  }
}

/*--------------------------------------------------------------------------------------*
 * Write GMSH output of given coord as point                                   ager 02/17
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_coord_dump(
    std::ofstream& file, Core::LinAlg::Matrix<3, 1> coord, double idx, bool to_local, Element* ele)
{
  file << "SP (";
  gmsh_write_coords(file, coord, to_local, ele);
  file << "){";
  file << idx;
  file << "};\n";
}

/*--------------------------------------------------------------------------------------*
 * Write Level Set Values for given Element
 *
 * The LS-value written at the midpoint of the facets and if the facet is triangulated,
 * also in the midpoint of the triangles.
 * Values at the nodes are also written.
 *                                                                           winter 07/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_level_set_value_zero_surface_dump(
    std::ofstream& file, Element* ele, bool to_local)
{
  std::vector<double> lsv_value(8);

  std::vector<Node*> nodes = ele->nodes();
  int mm = 0;
  for (std::vector<Node*>::iterator j = nodes.begin(); j != nodes.end(); j++)
  {
    Node* node = *j;
    lsv_value[mm] = node->lsv();
    mm++;
  }

  double lsv_max = lsv_value[0];
  double lsv_min = lsv_value[0];

  for (unsigned l = 1; l < lsv_value.size(); l++)
  {
    if (lsv_max < lsv_value[l]) lsv_max = lsv_value[l];

    if (lsv_min > lsv_value[l]) lsv_min = lsv_value[l];
  }

  // localcoord [-1,-1,-1] x [1,1,1]
  int z_sp = 150;
  int y_sp = 150;
  int x_sp = 150;


  double fac = 5 * 1e-3;
  double tolerance = (lsv_max - lsv_min) * fac;  //(0.001;

  //  double* x(3);
  Core::LinAlg::Matrix<3, 1> coord;

  for (int i = 0; i < x_sp; i++)
  {
    // std::cout << "i: " << i << std::endl;
    coord(0, 0) = -1.0 + (2.0 / (double(x_sp) - 1)) * double(i);
    for (int j = 0; j < y_sp; j++)
    {
      // std::cout << "j: " << j << std::endl;
      coord(1, 0) = -1.0 + (2.0 / (double(y_sp) - 1)) * double(j);

      for (int k = 0; k < z_sp; k++)
      {
        // std::cout << "k: " << k << std::endl;
        coord(2, 0) = -1.0 + (2.0 / (double(z_sp) - 1)) * double(k);

        double ls_value = ele->get_level_set_value_at_local_coords(coord);
        if (fabs(ls_value) < tolerance)
        {
          Core::LinAlg::Matrix<3, 1> coord_global;
          ele->global_coordinates(coord, coord_global);
          gmsh_scalar(file, coord_global, ls_value, to_local, ele);
        }
      }
    }
  }
}


/*--------------------------------------------------------------------------------------*
 * Write Level Set Gradient Orientation of Boundary-Cell Normal and LevelSet
 *                                                                           winter 07/15
 *--------------------------------------------------------------------------------------*/
void Cut::Output::gmsh_level_set_orientation_dump(std::ofstream& file, Element* ele, bool to_local)
{
  const plain_volumecell_set volcells = ele->volume_cells();
  for (plain_volumecell_set::const_iterator i = volcells.begin(); i != volcells.end(); i++)
  {
    VolumeCell* volcell = *i;

    if (volcell->position() == Cut::Point::inside) continue;

    //    const plain_facet_set facets = volcells->Facets();

    volcell->boundary_cells();
    plain_boundarycell_set bc_cells = volcell->boundary_cells();
    for (plain_boundarycell_set::iterator j = bc_cells.begin(); j != bc_cells.end(); ++j)
    {
      BoundaryCell* bc = *j;

      //      Facet *facet = *bc->GetFacet();
      Core::LinAlg::Matrix<3, 1> midpoint_bc;
      bc->element_center(midpoint_bc);

      Core::LinAlg::Matrix<3, 1> normal_bc;
      Core::LinAlg::Matrix<2, 1> xsi;
      bc->normal(xsi, normal_bc);

      std::vector<std::vector<double>> coords_bc = bc->coordinates_v();
      // const Core::LinAlg::SerialDenseMatrix ls_coordEp = bc->coordinates();
      Core::LinAlg::Matrix<3, 1> ls_coord(Core::LinAlg::Initialization::zero);
      ls_coord(0, 0) = coords_bc[1][0];
      ls_coord(1, 0) = coords_bc[1][1];
      ls_coord(2, 0) = coords_bc[1][2];

      std::vector<double> normal_ls = ele->get_level_set_gradient(ls_coord);

      double dotProduct = 0.0;
      for (unsigned d = 0; d < normal_ls.size(); ++d) dotProduct += normal_ls[d] * normal_bc(d, 0);

      double normalized_dotproduct = 0;
      if (dotProduct < 1e-15)
        normalized_dotproduct = 0;
      else
        normalized_dotproduct = dotProduct / fabs(dotProduct);

      gmsh_scalar(file, midpoint_bc, normalized_dotproduct, to_local, ele);
    }
  }
}


/*!
\brief Write Eqn of plane normal for facet (used for DirectDivergence).
 */
void Cut::Output::gmsh_eqn_plane_normal_dump(
    std::ofstream& file, Element* ele, bool normalize, bool to_local)
{
  const plain_facet_set facets = ele->facets();
  for (plain_facet_set::const_iterator j = facets.begin(); j != facets.end(); j++)
  {
    Facet* facet = *j;
    gmsh_eqn_plane_normal_dump(file, facet, normalize, to_local, ele);
  }
}

/*!
\brief Write Eqn of plane normal for all facets (used for DirectDivergence).
 */
void Cut::Output::gmsh_eqn_plane_normal_dump(
    std::ofstream& file, Facet* facet, bool normalize, bool to_local, Element* ele)
{
  Core::LinAlg::Matrix<3, 1> facet_triang_midpoint_coord(Core::LinAlg::Initialization::zero);
  std::vector<Point*> f_cornpts = facet->corner_points();
  std::vector<double> eqn_plane = get_eq_of_plane(f_cornpts);

  if (facet->is_triangulated())
  {
    std::vector<std::vector<Point*>> facet_triang = facet->triangulation();
    Point* facet_triang_midpoint = (facet_triang[0])[0];
    facet_triang_midpoint->coordinates(&facet_triang_midpoint_coord(0, 0));

    for (std::vector<std::vector<Point*>>::iterator k = facet_triang.begin();
        k != facet_triang.end(); k++)
    {
      std::vector<Point*> facet_triang_tri = *k;

      Core::LinAlg::Matrix<3, 1> cur;
      Core::LinAlg::Matrix<3, 1> f_triang_tri_midp(Core::LinAlg::Initialization::zero);
      for (std::vector<Point*>::iterator i = facet_triang_tri.begin(); i != facet_triang_tri.end();
          i++)
      {
        Point* p1 = *i;
        p1->coordinates(cur.data());
        f_triang_tri_midp.update(1.0, cur, 1.0);
      }
      f_triang_tri_midp.scale(1.0 / facet_triang_tri.size());

      gmsh_vector(
          file, f_triang_tri_midp, get_eq_of_plane(facet_triang_tri), normalize, to_local, ele);
    }
  }
  else
  {
    Core::LinAlg::Matrix<3, 1> cur;
    std::vector<Point*> pts = facet->points();
    for (std::vector<Point*>::iterator i = pts.begin(); i != pts.end(); i++)
    {
      Point* p1 = *i;
      p1->coordinates(cur.data());
      facet_triang_midpoint_coord.update(1.0, cur, 1.0);
    }
    facet_triang_midpoint_coord.scale(1.0 / pts.size());
  }

  gmsh_vector(file, facet_triang_midpoint_coord, eqn_plane, normalize, to_local, ele);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_scalar(std::ofstream& file, Core::LinAlg::Matrix<3, 1> coord, double scalar,
    bool to_local, Element* ele)  // use gmshpoint?
{
  file << "SP(";
  gmsh_write_coords(file, coord, to_local, ele);
  file << "){";
  file << scalar;
  file << "};\n";
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_vector(std::ofstream& file, Core::LinAlg::Matrix<3, 1> coord,
    std::vector<double> vector, bool normalize, bool to_local, Element* ele)
{
  file << "VP(";
  gmsh_write_coords(file, coord, to_local, ele);
  file << "){";

  if (normalize)
  {
    double norm = 0.0;
    for (unsigned i = 0; i < vector.size(); ++i) norm += vector[i] * vector[i];

    norm = sqrt(norm);
    if (norm > 1e-15)
    {
      for (unsigned i = 0; i < vector.size(); ++i) vector[i] = vector[i] / norm;
    }
    else
    {
      std::fill(vector.begin(), vector.end(), 0.0);
    }
  }
  gmsh_write_coords(file, vector, to_local, ele);
  file << "};\n";
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_write_coords(
    std::ofstream& file, std::vector<double> coord, bool to_local, Element* ele)
{
  Core::LinAlg::Matrix<3, 1> xyz(Core::LinAlg::Initialization::zero);

  if (coord.size() <= 3)
    std::copy(coord.begin(), coord.end(), xyz.data());
  else
    FOUR_C_THROW("The coord vector dimension is wrong! (coord.size() = {})", coord.size());

  if (to_local)
  {
    if (ele == nullptr)
      FOUR_C_THROW("GmshWriteCoords: Didn't get a parent element for the Coordinate!");

    Core::LinAlg::Matrix<3, 1> rst(Core::LinAlg::Initialization::zero);

    ele->local_coordinates(xyz, rst);
    gmsh_write_coords(file, rst, false, nullptr);  // rst are already local coords!
    return;
  }
  file << std::setprecision(15) << xyz(0) << "," << xyz(1) << "," << xyz(2);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_write_coords(
    std::ofstream& file, Core::LinAlg::Matrix<3, 1> coord, bool to_local, Element* ele)
{
  if (to_local)
  {
    if (ele == nullptr)
      FOUR_C_THROW("GmshWriteCoords: Didn't get a parent element for the Coordinate!");

    Core::LinAlg::Matrix<3, 1> xyz = coord;
    ele->local_coordinates(xyz, coord);
  }
  file << std::setprecision(15) << coord(0, 0) << "," << coord(1, 0) << "," << coord(2, 0);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_write_coords(std::ofstream& file, Node* node, bool to_local, Element* ele)
{
  gmsh_write_coords(file, node->point(), to_local, ele);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_write_coords(std::ofstream& file, Point* point, bool to_local, Element* ele)
{
  Core::LinAlg::Matrix<3, 1> coord;
  point->coordinates(coord.data());

  gmsh_write_coords(file, coord, to_local, ele);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::string Cut::Output::generate_gmsh_output_filename(const std::string& filename_tail)
{
  //  std::string filename = ::Global::Problem::instance()->output_control_file()->file_name();
  std::string filename("xxx");
  filename.append(filename_tail);
  return filename;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_new_section(
    std::ofstream& file, const std::string& section, bool first_endsection)
{
  if (first_endsection) file << "};\n";
  file << "View \"" << section << "\" {\n";
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Output::gmsh_end_section(std::ofstream& file, bool close_file)
{
  file << "};\n";
  if (close_file) file.close();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::vector<double> Cut::Output::get_eq_of_plane(std::vector<Point*> pts)
{
  int mm = 0;

  std::vector<std::vector<double>> corners(pts.size());

  for (std::vector<Point*>::iterator k = pts.begin(); k != pts.end(); k++)
  {
    Point* p1 = *k;
    Core::LinAlg::Matrix<3, 1> cur;
    p1->coordinates(cur.data());

    std::vector<double> pt(3);

    pt[0] = cur(0, 0);
    pt[1] = cur(1, 0);
    pt[2] = cur(2, 0);

    corners[mm] = pt;
    mm++;
  }
  return Kernel::eqn_plane_of_polygon(corners);
}


/*-------------------------------------------------------------------------------*
 * Write cuttest for this element!                                     ager 04/15
 *-------------------------------------------------------------------------------*/
void Cut::Output::gmsh_element_cut_test(
    std::ofstream& file, Cut::Element* ele, bool haslevelsetside)
{
  std::cout << "Write Cut Test for Element " << ele->id() << " ... " << std::flush;

  // default precision for coordinates
  file << std::setprecision(20);
  // -- 1 -- header of cut_test -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  file << "// This test was automatically generated by Cut::Output::GmshElementCutTest(), "
       << "\n";
  file << "// as the cut crashed for this configuration!"
       << "\n";
  file << ""
       << "\n";
  file << "#include <iostream>"
       << "\n";
  file << "#include <map>"
       << "\n";
  file << "#include <string>"
       << "\n";
  file << "#include <vector>"
       << "\n";
  file << ""
       << "\n";
  file << "#include \"cut_test_utils.H\""
       << "\n";
  file << ""
       << "\n";
  file << "#include \"../../src/cut/cut_side.H\""
       << "\n";
  file << "#include \"../../src/cut/cut_meshintersection.H\""
       << "\n";
  file << "#include \"../../src/cut/cut_levelsetintersection.H\""
       << "\n";
  file << "#include \"../../src/cut/cut_combintersection.H\""
       << "\n";
  file << "#include \"../../src/cut/cut_tetmeshintersection.H\""
       << "\n";
  file << "#include \"../../src/cut/cut_options.H\""
       << "\n";
  file << "#include \"../../src/cut/cut_volumecell.H\""
       << "\n";
  file << ""
       << "\n";
  file << "#include \"../../src/fem_general/utils_local_connectivity_matrices.H\""
       << "\n";
  file << ""
       << "\n";
  file << "void test_generated_" << ele->id() << "()"
       << "\n";
  file << "{"
       << "\n";
  if (not haslevelsetside)
    file << "  Cut::MeshIntersection intersection;"
         << "\n";
  else
    file << "  Cut::CombIntersection intersection(-1);"
         << "\n";
  file << "  intersection.GetOptions().Init_for_Cuttests();  // use full cln\n";
  file << "  std::vector<int> nids;"
       << "\n";
  file << ""
       << "\n";
  file << "  int sidecount = 0;"
       << "\n";
  file << "  std::vector<double> lsvs(" << ele->nodes().size() << ");"
       << "\n";

  // --- get all neighboring elements and cutsides
  plain_element_set eles;
  plain_side_set csides;
  for (std::vector<Side*>::const_iterator sit = ele->sides().begin(); sit != ele->sides().end();
      ++sit)
  {
    for (plain_element_set::const_iterator it = (*sit)->elements().begin();
        it != (*sit)->elements().end(); ++it)
    {
      eles.insert(*it);
      for (plain_side_set::const_iterator csit = (*it)->cut_sides().begin();
          csit != (*it)->cut_sides().end(); ++csit)
        csides.insert(*csit);
    }
  }


  if (not haslevelsetside)
  {
    // -- 2 -- add sides -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    const plain_side_set& cutsides = csides;
    for (plain_side_set::const_iterator i = cutsides.begin(); i != cutsides.end(); ++i)
    {
      file << "  {"
           << "\n";
      file << "    Core::LinAlg::SerialDenseMatrix tri3_xyze( 3, 3 );"
           << "\n";
      file << ""
           << "\n";
      Side* s = *i;
      const std::vector<Node*>& side_nodes = s->nodes();
      int nodelid = -1;
      file << "    nids.clear();"
           << "\n";
      for (std::vector<Node*>::const_iterator j = side_nodes.begin(); j != side_nodes.end(); ++j)
      {
        nodelid++;
        Node* n = *j;
        for (int dim = 0; dim < 3; ++dim)
        {
          file << "    tri3_xyze(" << dim << "," << nodelid << ") = " << n->point()->x()[dim] << ";"
               << "\n";
        }
        file << "    nids.push_back( " << n->id() << " );"
             << "\n";
      }
      file << "    intersection.AddCutSide( ++sidecount, nids, tri3_xyze, "
              "Core::FE::CellType::tri3 );"
           << "\n";
      file << "  }"
           << "\n";
    }
  }
  else
  {
    file << "  ci.AddLevelSetSide(1);"
         << "\n";
    for (std::size_t i = 0; i < ele->nodes().size(); ++i)
    {
      file << "  lsvs[" << i << "] = " << ele->nodes()[i]->lsv() << ";"
           << "\n";
    }
  }

  // -- 3 -- add background element -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  const plain_element_set& elements = eles;
  for (plain_element_set::const_iterator it = elements.begin(); it != elements.end(); ++it)
  {
    Element* aele = *it;
    file << "  {"
         << "\n";
    file << "  LinAlg::SerialDenseMatrix hex" << aele->nodes().size() << "_xyze( 3, "
         << aele->nodes().size() << " );"
         << "\n";
    file << ""
         << "\n";
    file << "    nids.clear();"
         << "\n";
    for (std::size_t i = 0; i < aele->nodes().size(); ++i)
    {
      for (unsigned dim = 0; dim < 3; ++dim)
      {
        file << "  hex8_xyze(" << dim << "," << i << ") = " << aele->nodes()[i]->point()->x()[dim]
             << ";"
             << "\n";
      }
      file << "  nids.push_back( " << aele->nodes()[i]->id() << " );"
           << "\n";
    }
    file << ""
         << "\n";
    if (not haslevelsetside)
      file << "  intersection.add_element( " << aele->id()
           << ", nids, hex8_xyze, Core::FE::CellType::hex8);"
           << "\n";
    else
      file << "  intersection.add_element( " << aele->id()
           << ", nids, hex8_xyze, Core::FE::CellType::hex8, &lsvs[0], false );"
           << "\n";
    file << "  }"
         << "\n";
    file << ""
         << "\n";
  }
  file << "  intersection.CutTest_Cut( true,Cut::VCellGaussPts_DirectDivergence, "
          "Cut::BCellGaussPts_Tessellation );"
       << "\n";
  file << "  intersection.Cut_Finalize( true, Cut::VCellGaussPts_DirectDivergence, "
          "Cut::BCellGaussPts_Tessellation, false, true );"
       << "\n";
  file << ""
       << "\n";

  if (not haslevelsetside && 0)
  {
    // -- 4 -- compare integration methods -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    file << "  std::vector<double> tessVol,momFitVol,dirDivVol;"
         << "\n";
    file << ""
         << "\n";
    file << "  Cut::Mesh mesh = intersection.NormalMesh();"
         << "\n";
    file << "  const std::list<std::shared_ptr<Cut::VolumeCell> > & other_cells = "
            "mesh.VolumeCells();"
         << "\n";
    file << "  for ( std::list<std::shared_ptr<Cut::VolumeCell> >::const_iterator "
            "i=other_cells.begin();"
         << "\n";
    file << "        i!=other_cells.end();"
         << "\n";
    file << "        ++i )"
         << "\n";
    file << "  {"
         << "\n";
    file << "    Cut::VolumeCell * vc = &**i;"
         << "\n";
    file << "    tessVol.push_back(vc->Volume());"
         << "\n";
    file << "  for ( std::list<std::shared_ptr<Cut::VolumeCell> >::const_iterator "
            "i=other_cells.begin();"
         << "\n";
    file << "        i!=other_cells.end();"
         << "\n";
    file << "        ++i )"
         << "\n";
    file << "  {"
         << "\n";
    file << "    Cut::VolumeCell * vc = &**i;"
         << "\n";
    file << "    "
            "vc->moment_fit_gauss_weights(vc->parent_element(),mesh,true,Cut::"
            "BCellGaussPts_"
            "Tessellation);"
         << "\n";
    file << "    momFitVol.push_back(vc->Volume());"
         << "\n";
    file << "  }"
         << "\n";
    file << ""
         << "\n";
    file << "  for ( std::list<std::shared_ptr<Cut::VolumeCell> >::const_iterator "
            "i=other_cells.begin();"
         << "\n";
    file << "           i!=other_cells.end();"
         << "\n";
    file << "           ++i )"
         << "\n";
    file << "   {"
         << "\n";
    file << "     Cut::VolumeCell * vc = &**i;"
         << "\n";
    file << "     "
            "vc->direct_divergence_gauss_rule(vc->parent_element(),mesh,true,Cut::"
            "BCellGaussPts_"
            "Tessellation);"
         << "\n";
    file << "     dirDivVol.push_back(vc->Volume());"
         << "\n";
    file << "   }"
         << "\n";
    file << ""
         << "\n";
    file << "  std::cout<<\"the volumes predicted by\\n tessellation \\t MomentFitting \\t "
            "DirectDivergence\\n\";"
         << "\n";
    file << "  for(unsigned i=0;i<tessVol.size();i++)"
         << "\n";
    file << "  {"
         << "\n";
    file << "    std::cout<<tessVol[i]<<\"\\t\"<<momFitVol[i]<<\"\\t\"<<dirDivVol[i]<<\"\\n\";"
         << "\n";
    file << "    if( fabs(tessVol[i]-momFitVol[i])>1e-9 || fabs(dirDivVol[i]-momFitVol[i])>1e-9 )"
         << "\n";
    file << "    {"
         << "\n";
    file << "      mesh.DumpGmsh(\"Cuttest_Debug_Output.pos\");"
         << "\n";
    file << "      intersection.CutMesh().get_element(1)->DebugDump();"
         << "\n";
    file << "      FOUR_C_THROW(\"volume predicted by either one of the method is wrong\");"
         << "\n";
    file << "      }"
         << "\n";
    file << "    }"
         << "\n";
  }
  file << "}"
       << "\n";
  std::cout << "done " << std::endl;
}

// Debug related functions

void Cut::Output::debug_dump_three_points_on_edge(
    Side* first, Side* second, Edge* e, Point* p, const PointSet& cut)
{
  std::ofstream file("three_points_on_the_edge.pos");
  Cut::Output::gmsh_side_dump(file, first, std::string("FirstSide"));
  Cut::Output::gmsh_side_dump(file, second, std::string("SecondSide"));
  file << "// Edge is " << e->begin_node()->point() << "->" << e->end_node()->point() << std::endl;
  Cut::Output::gmsh_edge_dump(file, e, std::string("NotTouchedEdge"));
  Cut::Output::gmsh_point_dump(file, p, p->id(), std::string("NotTouchedPoint"), false, nullptr);
  for (PointSet::const_iterator it = cut.begin(); it != cut.end(); ++it)
  {
    std::stringstream point_name;
    point_name << "CutPoint" << std::distance(cut.begin(), it);
    Cut::Output::gmsh_point_dump(file, *it, (*it)->id(), point_name.str(), false, nullptr);
    (*it)->dump_connectivity_info();
  }

  const PointPositionSet& edge_cut_points = e->cut_points();
  for (PointPositionSet::const_iterator it = edge_cut_points.begin(); it != edge_cut_points.end();
      ++it)
  {
    std::stringstream point_name;
    point_name << "EdgeCutPoint" << std::distance(edge_cut_points.begin(), it);
    Cut::Output::gmsh_point_dump(file, *it, (*it)->id(), point_name.str(), false, nullptr);
  }
  file.close();
}

void Cut::Output::debug_dump_more_than_two_intersection_points(
    Edge* edge, Side* other, const std::vector<Point*>& point_stack)
{
  std::ofstream file("multiple_intersection_points.pos");
  // dump everything
  for (std::vector<Point*>::const_iterator it = point_stack.begin(); it != point_stack.end(); ++it)
  {
    std::stringstream section_name;
    section_name << "Point" << (*it)->id();
    Cut::Output::gmsh_new_section(file, section_name.str());
    Cut::Output::gmsh_point_dump(file, *it, (*it)->id(), false, nullptr);
    Cut::Output::gmsh_end_section(file, false);
#if CUT_CREATION_INFO
    const std::pair<Side*, Edge*>& cu_pair = std::make_pair(other, edge);
    const std::pair<Side*, Edge*>& or_pair = (*it)->AddedFrom(cu_pair);
    Cut::Output::GmshCutPairDump(file, or_pair, 0, std::string("added_from"));
    if (or_pair != cu_pair)
    {
      file << "// original added because: " << (*it)->GetCreationInfo(or_pair) << "\n";
      file << "// common added because: " << (*it)->GetCreationInfo(cu_pair) << "\n";
    }
    else
      file << "// original added because: " << (*it)->GetCreationInfo(cu_pair) << "\n";
#endif

    (*it)->dump_connectivity_info();
  }
  // Compute all possible differences between points

  for (std::vector<Point*>::const_iterator it = point_stack.begin(); it != point_stack.end(); ++it)
  {
    for (std::vector<Point*>::const_iterator jt = point_stack.begin(); jt != point_stack.end();
        ++jt)
    {
      if (*it != *jt)
      {
        std::cout << "Difference between " << (*it)->id() << " and " << (*jt)->id() << " is "
                  << Cut::distance_between_points(*jt, *it) << std::endl;
      }
    }
  }
  Cut::Output::gmsh_cut_pair_dump(file, other, edge, 0, std::string("common"));
  file.close();
}

void Cut::Output::debug_dump_multiple_cut_points_special(Side* first, Side* second,
    const PointSet& cut, const PointSet& collected_points, const point_line_set& new_lines)
{
  std::ofstream file("special_case_multiple_cut_points.pos");
  Cut::Output::gmsh_side_dump(file, first, std::string("ThisSide"));
  Cut::Output::gmsh_side_dump(file, second, std::string("OtherSide"));
  for (PointSet::const_iterator it = cut.begin(); it != cut.end(); ++it)
  {
    std::stringstream pname;
    pname << "CutPoint" << std::distance(cut.begin(), it);
    Cut::Output::gmsh_point_dump(file, *it, (*it)->id(), pname.str(), false, nullptr);
    (*it)->dump_connectivity_info();
  }
  file.close();

  std::cout << "Collected " << collected_points.size() << " points " << std::endl;
  for (PointSet::const_iterator it = collected_points.begin(); it != collected_points.end(); ++it)
  {
    std::cout << (*it)->id() << "  " << std::endl;
  }
  std::cout << "Totally there are  " << cut.size() << " points " << std::endl;
  for (PointSet::const_iterator it = cut.begin(); it != cut.end(); ++it)
  {
    std::cout << (*it)->id() << "  " << std::endl;
  }
  std::cout << "Got " << new_lines.size() << " cut lines"
            << " for " << cut.size() << " cut points\n";
  std::cout << "Cut lines are: \n";
  for (point_line_set::const_iterator it = new_lines.begin(); it != new_lines.end(); ++it)
  {
    std::cout << it->first << "--" << it->second << std::endl;
  }
}

FOUR_C_NAMESPACE_CLOSE
