// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beamcontact_beam3contact_octtree.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_beamcontact_beam3contact_manager.hpp"
#include "4C_beamcontact_input.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_contact_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_rigidsphere.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Teuchos_Time.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  constructor (public)                                     meier 01/11|
 *----------------------------------------------------------------------*/
Beam3ContactOctTree::Beam3ContactOctTree(Teuchos::ParameterList& params,
    Core::FE::Discretization& discret, Core::FE::Discretization& searchdis)
    : btsol_(false),
      discret_(discret),
      searchdis_(searchdis),
      basisnodes_(discret.num_global_nodes())
{
  extrusionvalue_ = std::make_shared<std::vector<double>>();
  extrusionvalue_->clear();

  BeamContact::OctreeType bboxtype_input = BeamContact::boct_none;

  // This OctTree may be used for contact search
  // it will thus be called with slightly different parameter sets
  // first find out which params we got and extract octtree specifications

  if (params.name() == "ROOT->BEAM CONTACT")
  {
    // octree specs
    bboxtype_input = Teuchos::getIntegralValue<BeamContact::OctreeType>(params, "BEAMS_OCTREE");

    // additive or multiplicative extrusion of bounding boxes
    if (params.get<bool>("BEAMS_ADDITEXT"))
      additiveextrusion_ = true;
    else
      additiveextrusion_ = false;

    // extrusion factor(s)
    // depending on the kind of extrusion, more than one extrusion value will be necessary
    // AABB: one value for all dimensions
    // COBB: 1. value for axial extrusion, 2. value for radial extrusion
    // SPBB: one value for radial extrusion

    std::string extrusion_value_in(Teuchos::getNumericStringParameter(params, "BEAMS_EXTVAL"));

    Core::IO::ValueParser extrusionvalue_parser(
        extrusion_value_in, {.user_scope_message = "While reading extrusion values: "});

    while (!extrusionvalue_parser.at_end())
    {
      extrusionvalue_->push_back(extrusionvalue_parser.read<double>());
    }

    // max tree depth
    maxtreedepth_ = params.get<int>("BEAMS_TREEDEPTH", 6);
    // max number of bounding boxes per leaf octant
    minbboxesinoctant_ = params.get<int>("BEAMS_BOXESINOCT", 8);

    btsol_ = params.get<bool>("BEAMS_BTSOL");
  }
  else
  {
    FOUR_C_THROW("OctTree called with unknown parameter list named '{}'!", params.name());
  }

  // sanity checks for extrusion value(s)
  if ((int)extrusionvalue_->size() > 2)
    FOUR_C_THROW("You gave {} values for BEAMS_EXTVAL! Check your input file.",
        (int)extrusionvalue_->size());
  if ((int)extrusionvalue_->size() == 1) extrusionvalue_->push_back(extrusionvalue_->at(0));
  for (int i = 0; i < (int)extrusionvalue_->size(); i++)
    if (extrusionvalue_->at(i) < 1.0 && !additiveextrusion_)
      FOUR_C_THROW(
          "BEAMS_EXTVAL( {} ) = {:4.2f} < 1.0 does not make any sense! Check your input file.", i,
          extrusionvalue_->at(i));
  if (boundingbox_ == Beam3ContactOctTree::cyloriented && (int)extrusionvalue_->size() != 2)
    FOUR_C_THROW(
        "For cylindrical, oriented bounding boxes, TWO extrusion factors are mandatory! Check "
        "BEAMS_EXTVAL in your input file.");

  // statmech input section no longer existent, get parameters out of new section if needed
  periodlength_ = std::make_shared<std::vector<double>>();
  periodlength_->clear();
  for (int i = 0; i < 2; i++) periodlength_->push_back(0);
  periodic_bc_ = false;

  // determine bounding box type
  switch (bboxtype_input)
  {
    case BeamContact::boct_aabb:
    {
      if (!Core::Communication::my_mpi_rank(discret_.get_comm()))
        std::cout << "Search routine:\nOctree + Axis Aligned BBs" << std::endl;
      boundingbox_ = Beam3ContactOctTree::axisaligned;
    }
    break;
    case BeamContact::boct_cobb:
    {
      if (!Core::Communication::my_mpi_rank(discret_.get_comm()))
        std::cout << "Search routine:\nOctree + Cylindrical Oriented BBs" << std::endl;
      boundingbox_ = Beam3ContactOctTree::cyloriented;

      if (btsol_)
        FOUR_C_THROW(
            "Only axis aligned or spherical bounding boxes possible for beam-to-solid contact!");
    }
    break;
    case BeamContact::boct_spbb:
    {
      if (!Core::Communication::my_mpi_rank(discret_.get_comm()))
        std::cout << "Search routine:\nOctree + Spherical BBs" << std::endl;
      boundingbox_ = Beam3ContactOctTree::spherical;
    }
    break;
    default:
      FOUR_C_THROW("No octree (i.e. none) declared in your input file!");
      break;
  }

  if (!Core::Communication::my_mpi_rank(discret_.get_comm()))
  {
    int numextval = (int)extrusionvalue_->size();
    if (additiveextrusion_)
    {
      numextval = 1;
      std::cout << "additive extrusion     = ";
    }
    else
      std::cout << "multiplicat. extrusion = ";
    for (int i = 0; i < numextval; i++) std::cout << extrusionvalue_->at(i) << " ";
    std::cout << std::endl;
    std::cout << "max. tree depth        = " << maxtreedepth_
              << "\nmax. BB per octant     = " << minbboxesinoctant_ << std::endl;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  calls the almighty Octtree (public)                    mueller 01/11|
 *----------------------------------------------------------------------*/
std::vector<std::vector<Core::Elements::Element*>> Beam3ContactOctTree::oct_tree_search(
    std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions, int step)
{
  // initialize class vectors
  initialize_octree_search();

  // build axis aligned bounding boxes
  create_bounding_boxes(currentpositions);

  // call recursive octtree build
  // clear vector for assigning bounding boxes to octants to be on the safe side before
  // (re)assigning bounding boxes
  bool bboxesfound = locate_all();

  // intersection checks
  std::vector<std::vector<Core::Elements::Element*>> contactpairelements;
  if (bboxesfound)
  {
    bounding_box_intersection(currentpositions, contactpairelements);

    // output
    octree_output(contactpairelements, step);
  }

  // return contactpairs;
  return contactpairelements;
}  // OctTreeSearch()

/*----------------------------------------------------------------------*
 |  Return the octants to which this bounding box belongs               |
 |  (public)                                               mueller 01/11|
 *----------------------------------------------------------------------*/
std::vector<int> Beam3ContactOctTree::in_which_octant_lies(const int& thisBBoxID)
{
  std::vector<int> octants;
  octants.clear();
  int bboxcolid = searchdis_.element_col_map()->lid(thisBBoxID);
  int i = 0;
  while ((*bbox2octant_)(i).local_values_as_span()[bboxcolid] >= 0.0)
  {
    octants.push_back((int)(*bbox2octant_)(i).local_values_as_span()[bboxcolid]);
    i++;
  }
  return octants;
}
/*----------------------------------------------------------------------*
 |  Intersect the bounding boxes of a certain octant with a given       |
 |  bounding box (public)                                  mueller 01/11|
 *----------------------------------------------------------------------*/
bool Beam3ContactOctTree::intersect_b_boxes_with(
    Core::LinAlg::SerialDenseMatrix& nodecoords, Core::LinAlg::SerialDenseMatrix& nodeLID)
{
  FOUR_C_THROW("Not in use!");

  /* notes:
   * 1) do not apply this before having constructed the octree. This is merely a query tool
   * 2)"boxid" does not necessarily coincide with the bounding box we are going to intersect with
   * the other boxes in the octant. The reason: The bounding box may actually not exist. Of course,
   * if it does exist, "boxid" will be the id of the bounding box we actually want to check the
   * other boxes against. However, if the bounding box is merely a hypothetical construct (i.e.
   * there is no actual beam element), then we have to give a box id that does exist in order to
   * find the correct octant. Ideally, that means that "boxid" should be the ID of a bounding box
   * which is a direct neighbor of our (hypothetical) bounding box. 3) nodecoords are the
   * coordinates of the nodes of the (non-)existing element.
   */
  bool intersection = false;

  // determine bounding box limits
  std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits =
      std::make_shared<Core::LinAlg::SerialDenseMatrix>(1, 1);

  // build bounding box according to given type
  switch (boundingbox_)
  {
    case Beam3ContactOctTree::axisaligned:
      create_aabb(nodecoords, 0, bboxlimits);
      break;
    case Beam3ContactOctTree::cyloriented:
      create_cobb(nodecoords, 0, bboxlimits);
      break;
    case Beam3ContactOctTree::spherical:
      create_spbb(nodecoords, 0, bboxlimits);
      break;
    default:
      FOUR_C_THROW("No or an invalid Octree type was chosen. Check your input file!");
      break;
  }

  // retrieve octants in which the bounding box with ID thisBBoxID is located
  std::vector<std::vector<int>> octants;
  octants.clear();
  // get the octants for two bounding boxes (element) GIDs adjacent to each given node LID
  for (int i = 0; i < nodeLID.numRows(); i++)
    octants.push_back(in_which_octant_lies(
        searchdis_.l_col_node((int)nodeLID(i, 0))->adjacent_elements()[0].global_id()));

  // intersection of given bounding box with all other bounding boxes in given octant
  for (int ibox = 0; ibox < (int)octants.size(); ibox++)
  {
    for (int oct = 0; oct < (int)octants[ibox].size(); oct++)
    {
      if (octants[ibox][oct] != -1)
      {
        for (int i = 0; i < bboxesinoctants_->num_vectors(); i++)
        {
          // take only values of existing bounding boxes and not the filler values (-9)
          if ((int)(*bboxesinoctants_)(i).local_values_as_span()[octants[ibox][oct]] > -0.9)
          {
            // get the second bounding box ID
            std::vector<int> bboxinoct(2, -1);
            bboxinoct[0] = (int)(*bboxesinoctants_)(i).local_values_as_span()[octants[ibox][oct]];
            /*check for adjacent nodes: if there are adjacent nodes, then, of course, there
             * the intersection test will turn out positive. We skip those cases.*/
            // note: bounding box IDs are equal to element GIDs
            bool sharednode = false;
            for (int j = 0; j < searchdis_.g_element(bboxinoct[0])->num_node(); j++)
            {
              for (int k = 0; k < (int)nodeLID.numRows(); k++)
              {
                if (searchdis_.node_col_map()->lid(
                        searchdis_.g_element(bboxinoct[0])->node_ids()[j]) == (int)nodeLID(k, 0))
                {
                  sharednode = true;
                  break;
                }
              }
              if (sharednode) break;
            }
            // apply different bounding box intersection schemes
            if (!sharednode)
            {
              switch (boundingbox_)
              {
                case Beam3ContactOctTree::axisaligned:
                  intersection = intersection_aabb(bboxinoct, bboxlimits);
                  break;
                case Beam3ContactOctTree::cyloriented:
                  intersection = intersection_cobb(bboxinoct, bboxlimits);
                  break;
                case Beam3ContactOctTree::spherical:
                  intersection = intersection_spbb(bboxinoct, bboxlimits);
                  break;
                default:
                  FOUR_C_THROW("No or an invalid Octree type was chosen. Check your input file!");
                  break;
              }
            }

            if (intersection) break;
          }
          else  // loop reached the first bogus value (-9)
            break;
        }
      }
      else
        break;
      if (intersection) break;
    }
    if (intersection) break;
  }

  return intersection;
}

/*-----------------------------------------------------------------------------------*
 |  Output of octants, bounding boxes and contact pairs (public)       mueller 01/12 |
 *----------------------------------------------------------------------------------.*/
void Beam3ContactOctTree::octree_output(
    std::vector<std::vector<Core::Elements::Element*>> contactpairelements, int step)
{
  if (!Core::Communication::my_mpi_rank(discret_.get_comm()) && step != -1)
  {
    // active contact pairs
    if ((int)contactpairelements.size() > 0)
    {
      // Print contact_pairs to custom file format and plot with Matlab....................
      std::ostringstream filename;
      if (step != -2)
        filename << "contact_pairs" << std::setw(6) << std::setfill('0') << step << ".dat";
      else
        filename << "ContactPairsInit.dat";
      std::ofstream out(filename.str());
      for (int i = 0; i < (int)contactpairelements.size(); i++)
        out << ((contactpairelements[i])[0])->id() << "  " << ((contactpairelements[i][1]))->id()
            << '\n';
    }
    // octant limits output
    if ((int)octreelimits_.size() > 0)
    {
      std::ostringstream filename;
      if (step != -2)
        filename << "OctreeLimits" << std::setw(6) << std::setfill('0') << step << ".dat";
      else
        filename << "OctreeLimitsInit.dat" << std::endl;
      std::ofstream out(filename.str());
      for (auto& octreelimit : octreelimits_)
      {
        for (int v = 0; v < (int)octreelimit.num_rows(); v++)
          out << std::scientific << octreelimit(v) << " ";
        out << '\n';
      }
      // root box
      for (int u = 0; u < (int)rootbox_.num_rows(); u++)
        out << std::scientific << rootbox_(u) << " ";
      out << '\n';
    }
    // bounding box coords output
    if (allbboxes_ != nullptr)
    {
      std::ostringstream filename;
      if (step != -2)
        filename << "BoundingBoxCoords" << std::setw(6) << std::setfill('0') << step << ".dat";
      else
        filename << "BoundingBoxCoordsInit.dat" << std::endl;
      std::ofstream out(filename.str());
      for (int u = 0; u < allbboxes_->local_length(); u++)
      {
        for (int v = 0; v < allbboxes_->num_vectors(); v++)
          out << std::scientific << std::setprecision(10)
              << (*allbboxes_)(v).local_values_as_span()[u] << " ";
        out << diameter_->local_values_as_span()[u] << '\n';
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Bounding Box creation function (private)               mueller 01/12|
 |  Initialize octree class vectors and specifications                  |
 *----------------------------------------------------------------------*/
void Beam3ContactOctTree::initialize_octree_search()
{
  // number of shifts across volume boundaries in case of periodic boundary conditions (for
  // intersection optimization)
  if (periodic_bc_)
    numshifts_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*(searchdis_.element_col_map()), true);

  // determine radius factor by looking at the absolute mean variance of a bounding box (not quite
  // sure...) beam diameter
  diameter_ = std::make_shared<Core::LinAlg::Vector<double>>(*(searchdis_.element_col_map()));
  switch (boundingbox_)
  {
    // for this case, the diameter is calculated in create_spbb()
    case Beam3ContactOctTree::spherical:
      diameter_->put_scalar(0.0);
      break;
    default:
    {
      for (int i = 0; i < searchdis_.element_col_map()->num_my_elements(); i++)
      {
        Core::Elements::Element* element = searchdis_.l_col_element(i);
        const Core::Elements::ElementType& eot = element->element_type();

        (*diameter_).get_values()[i] =
            -1.0;  // TODO get diameter from call of abstract Beam3Base::Radius();

        if (eot == Discret::Elements::Beam3rType::instance() or
            eot == Discret::Elements::Beam3ebType::instance())
        {
          (*diameter_).get_values()[i] =
              2.0 * (dynamic_cast<Discret::Elements::Beam3Base*>(element))
                        ->get_circular_cross_section_radius_for_interactions();
        }
        else if (eot == Discret::Elements::RigidsphereType::instance())
          (*diameter_).get_values()[i] =
              2.0 * (dynamic_cast<Discret::Elements::Rigidsphere*>(element))->radius();
        // If we have a solid element, we don't need its diameter and can it set to zero:
        else if (!BeamInteraction::Utils::is_beam_element(*element) and
                 !BeamInteraction::Utils::is_rigid_sphere_element(*element))
          (*diameter_).get_values()[i] = 0.0;
        // feasibility check
        if ((*diameter_).get_values()[i] < 0.0)
          FOUR_C_THROW("ERROR: Did not receive feasible element radius.");
      }
    }
  }
  // storage of Bounding Boxes
  // (components 0,...,5 contain bounding box limits)
  // (components 6,...,23 contain bounding box limits in case of periodic boundary conditions):
  // a box may b subject to a boundary shift up to 3 times -> 4 bounding boxes -> up to 24 values +
  // 1 bounding box ID note: for spherical bounding boxes, we only need the center point. Hence
  // numbboxcoords = 3.
  int numbboxcoords = 6;
  if (boundingbox_ == Beam3ContactOctTree::spherical) numbboxcoords = 3;

  // number of shifts across periodic boundaries (std: no periodic boundary conditions ->
  // maxnumshifts = 0)
  int maxnumshifts = 0;
  if (periodic_bc_) maxnumshifts = 3;
  allbboxes_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *(searchdis_.element_col_map()), (maxnumshifts + 1) * numbboxcoords + 1, true);

  return;
}

/*----------------------------------------------------------------------*
 |  Bounding Box creation function (private)               mueller 01/11|
 |  generates bounding boxes extended with factor 1.05                  |
 *----------------------------------------------------------------------*/
void Beam3ContactOctTree::create_bounding_boxes(
    std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions)
{
#ifdef MEASURETIME
  double t_AABB = Teuchos::Time::wallTime();
#endif
  // Get Nodes from discretization....................
  // build bounding boxes according to input parameter
  for (int elecolid = 0; elecolid < searchdis_.element_col_map()->num_my_elements(); elecolid++)
  {
    int elegid = searchdis_.element_col_map()->gid(elecolid);
    // only do stuff for Row Elements
    if (searchdis_.element_row_map()->lid(elegid) > -1)
    {
      // get the element with local ID (LID) elecolid
      Core::Elements::Element* element = searchdis_.l_col_element(elecolid);

      // store nodal positions into matrix coords
      Core::LinAlg::SerialDenseMatrix coord(3, 2, true);

      if (BeamInteraction::Utils::is_beam_element(*element))
      {
        for (int i = 0; i < 2; i++)
        {
          int gid = element->nodes()[i]->id();
          Core::LinAlg::Matrix<3, 1> coord_aux = currentpositions[gid];
          for (int j = 0; j < 3; j++) coord(j, i) = coord_aux(j);
        }
      }
      else if (BeamInteraction::Utils::is_rigid_sphere_element(*element))
      {
        int gid = element->nodes()[0]->id();
        Core::LinAlg::Matrix<3, 1> coord_aux = currentpositions[gid];
        for (int j = 0; j < 3; j++)
        {
          // write sphere center coordinates into both columns, then the creation of bbs will work
          // with the following code for beams
          coord(j, 0) = coord_aux(j);
          coord(j, 1) = coord_aux(j);
        }
        if (boundingbox_ == Beam3ContactOctTree::cyloriented)
          FOUR_C_THROW(
              "Only axis-aligned or spherical bounding boxes possible for beam-to-sphere contact!");
      }
      else
      {
        // for solid elements only axis-aligned bounding boxes are possible
        switch (boundingbox_)
        {
          case Beam3ContactOctTree::axisaligned:
            calc_corner_pos(element, currentpositions, coord);
            break;
          case Beam3ContactOctTree::cyloriented:
            FOUR_C_THROW(
                "Only axis aligned or spherical bounding boxes possible for beam-to-solid "
                "contact!");
            break;
          case Beam3ContactOctTree::spherical:
            calc_corner_pos(element, currentpositions, coord);
            break;
          default:
            FOUR_C_THROW("No or an invalid Octree type was chosen. Check your input file!");
            break;
        }
      }
      // build bounding box according to given type
      switch (boundingbox_)
      {
        case Beam3ContactOctTree::axisaligned:
          create_aabb(coord, elecolid);
          break;
        case Beam3ContactOctTree::cyloriented:
          create_cobb(coord, elecolid);
          break;
        case Beam3ContactOctTree::spherical:
          create_spbb(coord, elecolid);
          break;
        default:
          FOUR_C_THROW("No or an invalid Octree type was chosen. Check your input file!");
          break;
      }
    }
  }  // end for-loop which goes through all elements

  // Communicate diameters to all procs, in case of spherical bounding boxes, this has to be done
  // separately, i.e., here
  if (boundingbox_ == Beam3ContactOctTree::spherical)
  {
    Core::LinAlg::Vector<double> diameterrow(*(searchdis_.element_row_map()), true);
    communicate_vector(diameterrow, *diameter_);
  }
  // communication of findings
  Core::LinAlg::MultiVector<double> allbboxesrow(
      *(searchdis_.element_row_map()), allbboxes_->num_vectors(), true);
  communicate_multi_vector(allbboxesrow, *allbboxes_);

  if (periodic_bc_)
  {
    Core::LinAlg::Vector<double> numshiftsrow(*(searchdis_.element_row_map()), true);
    communicate_vector(numshiftsrow, *numshifts_);
  }

#ifdef MEASURETIME
  double bbgentimelocal = Teuchos::Time::wallTime() - t_AABB;
  double bbgentimeglobal = 0.0;

  bbgentimeglobal = Core::Communication::max_all(bbgentimelocal, searchdis_.Comm());

  if (!Core::Communication::my_mpi_rank(searchdis_.Comm()))
    std::cout << "\n\nBBox creation time:\t\t" << bbgentimeglobal << " seconds" << std::endl;
#endif

  return;
}

/*-----------------------------------------------------------------------------------------*
 |  Create an Axis Aligned Bounding Box   (private)                           mueller 11/11|
 *----------------------------------------------------------------------------------------*/
void Beam3ContactOctTree::create_aabb(Core::LinAlg::SerialDenseMatrix& coord, const int& elecolid,
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits)
{
  // Why bboxlimits separately: The idea is that we can use this method to check whether a
  // hypothetical bounding box (i.e. without an element) can be tested for intersection. Hence, we
  // store the limits of this bounding box into bboxlimits if needed. Since the hypothetical
  // bounding box stands for a crosslinker to be set, we just need the exact dimensions of the
  // element
  if (bboxlimits != nullptr) FOUR_C_THROW("Not in use!");

  // factor by which the box is extruded in each dimension
  double extrusionvalue = get_bounding_box_extrusion_value();

  if (elecolid < 0 || elecolid >= searchdis_.element_col_map()->num_my_elements())
    FOUR_C_THROW("Given Element Column Map ID is {} !", elecolid);

  int elegid = searchdis_.element_col_map()->gid(elecolid);

  // make the bounding box continuous
  std::vector<int> cut(3, 0);
  int numshifts = 0;
  if (periodic_bc_) undo_effect_of_periodic_boundary_condition(coord, cut, numshifts);

  // directional vector (of the non-interrupted bounding box/ beam element
  Core::LinAlg::Matrix<3, 1> dir;
  for (int dof = 0; dof < (int)dir.num_rows(); dof++) dir(dof) = coord(dof, 1) - coord(dof, 0);

  // extrude bounding box in axial direction
  if (additiveextrusion_)
    dir.scale(1.0 + extrusionvalue);
  else  // multiplicative extrusion
    dir.scale(extrusionvalue);

  // extrude coords
  for (int dof = 0; dof < coord.numRows(); dof++)
  {
    double tmpcoord = coord(dof, 1);
    coord(dof, 1) = coord(dof, 0) + dir(dof);
    coord(dof, 0) = tmpcoord - dir(dof);
  }

  // first bounding box (independent of periodicBC_)
  // Calculate Center Point of AABB
  Core::LinAlg::Matrix<3, 1> midpoint;
  for (int i = 0; i < (int)midpoint.num_rows(); i++)
    midpoint(i) = 0.5 * (coord(i, 0) + coord(i, 1));

  // Calculate edgelength of AABB, select max value
  double maxedgelength = -1.0;
  for (int i = 0; i < (int)midpoint.num_rows(); i++)
    if (fabs(coord(i, 1) - coord(i, 0)) > maxedgelength)
      maxedgelength = fabs(coord(i, 1) - coord(i, 0));
  // Check for edgelength of AABB
  if (maxedgelength < diameter_->local_values_as_span()[elecolid])
    maxedgelength = diameter_->local_values_as_span()[elecolid];

  // Calculate limits of AABB
  // first bounding box
  // mins
  (*allbboxes_)(0).get_values()[elecolid] = midpoint(0) - 0.5 * maxedgelength;  // x
  (*allbboxes_)(2).get_values()[elecolid] = midpoint(1) - 0.5 * maxedgelength;  // y
  (*allbboxes_)(4).get_values()[elecolid] = midpoint(2) - 0.5 * maxedgelength;  // z
  // maxes
  (*allbboxes_)(1).get_values()[elecolid] = midpoint(0) + 0.5 * maxedgelength;  // x
  (*allbboxes_)(3).get_values()[elecolid] = midpoint(1) + 0.5 * maxedgelength;  // y
  (*allbboxes_)(5).get_values()[elecolid] = midpoint(2) + 0.5 * maxedgelength;  // z

  // periodic shifts of bounding box
  if (numshifts > 0)
  {
    int shift = 0;

    for (int dof = 0; dof < (int)cut.size(); dof++)
    {
      if (cut[dof] > 0)
      {
        double periodicshift = 0.0;
        switch (cut[dof])
        {
          case 1:
            periodicshift = (*periodlength_)[dof];
            break;
          case 2:
            periodicshift = -(*periodlength_)[dof];
            break;
        }
        for (int i = 0; i < 6; i++)
        {
          (*allbboxes_)(6 * shift + i).get_values()[elecolid] =
              (*allbboxes_)(i).get_values()[elecolid];
          if (i % 3 == dof) (*allbboxes_)(6 * shift + i).get_values()[elecolid] += periodicshift;
        }
      }
    }
    (*numshifts_).get_values()[elecolid] = shift;  // store number of shifts
  }

  if (periodic_bc_ && numshifts < 3)
    for (int i = allbboxes_->num_vectors() - 1; i > (numshifts + 1) * 6 - 1; i--)
      (*allbboxes_)(i).get_values()[elecolid] = -1e9;

  // store GID (=box number)
  (*allbboxes_)(allbboxes_->num_vectors() - 1).get_values()[elecolid] = elegid;

  return;
}

/*-----------------------------------------------------------------------------------------*
 |  Create Cylindrical an Oriented Bounding Box   (private)                   mueller 11/11|
 *----------------------------------------------------------------------------------------*/
void Beam3ContactOctTree::create_cobb(Core::LinAlg::SerialDenseMatrix& coord, const int& elecolid,
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits)
{
  // Why bboxlimits separately: The idea is that we can use this method to check whether a
  // hypothetical bounding box (i.e. without an element) can be tested for intersection. Hence, we
  // store the limits of this bounding box into bboxlimits if needed. Since the hypothetical
  // bounding box stands for a crosslinker to be set, we just need the exact dimensions of the
  // element
  if (bboxlimits != nullptr) FOUR_C_THROW("Not in use!");
  // make the bounding box continuous
  std::vector<int> cut(3, 0);
  int numshifts = 0;
  if (periodic_bc_) undo_effect_of_periodic_boundary_condition(coord, cut, numshifts);

  double extrusionvalue = get_bounding_box_extrusion_value();


  int elegid = searchdis_.element_col_map()->gid(elecolid);

  // directional vector (of the non-interrupted bounding box/ beam element
  Core::LinAlg::Matrix<3, 1> dir;
  for (int dof = 0; dof < (int)dir.num_rows(); dof++) dir(dof) = coord(dof, 1) - coord(dof, 0);

  // extrude bounding box in axial direction
  if (additiveextrusion_)
  {
    Core::LinAlg::Matrix<3, 1> additivedir(dir);
    additivedir.scale(extrusionvalue / dir.norm2());
    dir += additivedir;
  }
  else  // multiplicative extrusion
    dir.scale(extrusionvalue);

  Core::LinAlg::Matrix<3, 1> midpoint;
  for (int i = 0; i < (int)midpoint.num_rows(); i++)
    midpoint(i) = 0.5 * (coord(i, 0) + coord(i, 1));

  // First bounding box is the untreated set of coordinates!
  for (int dof = 0; dof < coord.numRows(); dof++)
  {
    (*allbboxes_)(dof).get_values()[elecolid] = midpoint(dof) - 0.5 * dir(dof);
    (*allbboxes_)(dof + 3).get_values()[elecolid] = midpoint(dof) + 0.5 * dir(dof);
  }

  int shift = 0;
  if (numshifts > 0)
  {
    // create shifted copies of the bounding boxes if there is an interruption by periodic BCs
    for (int dof = 0; dof < (int)cut.size(); dof++)
    {
      if (cut[dof] > 0)
      {
        shift++;
        double periodicshift = 0.0;
        switch (cut[dof])
        {
          case 1:
            periodicshift = (*periodlength_)[dof];
            break;
          case 2:
            periodicshift = -(*periodlength_)[dof];
            break;
        }
        // (node) position 1
        (*allbboxes_)(6 * shift + dof).get_values()[elecolid] =
            (*allbboxes_)(dof).local_values_as_span()[elecolid] + periodicshift;
        // (node) position 2
        (*allbboxes_)(6 * shift + dof + 3).get_values()[elecolid] =
            (*allbboxes_)(dof + 3).local_values_as_span()[elecolid] + periodicshift;

        // unshifted DOFs
        // (node) position 1
        (*allbboxes_)(6 * shift + ((dof + 1) % 3)).get_values()[elecolid] =
            (*allbboxes_)((dof + 1) % 3).local_values_as_span()[elecolid];
        (*allbboxes_)(6 * shift + ((dof + 2) % 3)).get_values()[elecolid] =
            (*allbboxes_)((dof + 2) % 3).local_values_as_span()[elecolid];
        // (node) position 2
        (*allbboxes_)(6 * shift + ((dof + 1) % 3) + 3).get_values()[elecolid] =
            (*allbboxes_)((dof + 1) % 3 + 3).local_values_as_span()[elecolid];
        (*allbboxes_)(6 * shift + ((dof + 2) % 3) + 3).get_values()[elecolid] =
            (*allbboxes_)((dof + 2) % 3 + 3).local_values_as_span()[elecolid];
      }
    }
    // fill all latter entries  except for the last one (->ID) with bogus values (in case of
    // periodic BCs)
    (*numshifts_).get_values()[elecolid] = numshifts;

    //    std::cout<<"Shifted BBOX "<<elecolid<<": ";
    //    for(int i=0; i<allbboxes_->NumVectors(); i++)
    //      std::cout<<(*allbboxes_)(i)[elecolid]<<" ";
    //    std::cout<<std::endl;
  }
  if (periodic_bc_ && numshifts < 3)
    for (int i = allbboxes_->num_vectors() - 1; i > (numshifts + 1) * 6 - 1; i--)
      (*allbboxes_)(i).get_values()[elecolid] = -1e9;

  // last entry: element GID
  (*allbboxes_)(allbboxes_->num_vectors() - 1).get_values()[elecolid] = elegid;

  return;
}

/*-----------------------------------------------------------------------------------------*
 |  Create Spherical Bounding Box   (private)                                  mueller 1/12|
 *-----------------------------------------------------------------------------------------*/
void Beam3ContactOctTree::create_spbb(Core::LinAlg::SerialDenseMatrix& coord, const int& elecolid,
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits)
{
  // Why bboxlimits separately: The idea is that we can use this method to check whether a
  // hypothetical bounding box (i.e. without an element) can be tested for intersection. Hence, we
  // store the limits of this bounding box into bboxlimits if needed. Since the hypothetical
  // bounding box stands for a crosslinker to be set, we just need the exact dimensions of the
  // element
  if (bboxlimits != nullptr) FOUR_C_THROW("Not in use!");
  // make the bounding box continuous
  std::vector<int> cut(3, 0);
  int numshifts = 0;
  if (periodic_bc_) undo_effect_of_periodic_boundary_condition(coord, cut, numshifts);

  // Note: diameter_ is used as the sphere diameter in the context of SPBBs
  double extrusionvalue = get_bounding_box_extrusion_value();

  int elegid = searchdis_.element_col_map()->gid(elecolid);

  // get the element with local ID (LID) elecolid
  Core::Elements::Element* element = searchdis_.l_col_element(elecolid);
  double diameter = 0.0;

  if (BeamInteraction::Utils::is_beam_element(*element))
  {
    if (coord.numRows() == 3 and coord.numCols() == 2)
    {
      diameter = sqrt((coord(0, 0) - coord(0, 1)) * (coord(0, 0) - coord(0, 1)) +
                      (coord(1, 0) - coord(1, 1)) * (coord(1, 0) - coord(1, 1)) +
                      (coord(2, 0) - coord(2, 1)) * (coord(2, 0) - coord(2, 1)));
    }
    else
      FOUR_C_THROW("coord matrix of nodal positions has wrong dimensions here!");
  }
  else if (BeamInteraction::Utils::is_rigid_sphere_element(*element))
  {
    diameter = diameter_->local_values_as_span()[elecolid];
  }
  else
  {
    if (coord.numRows() == 3 and coord.numCols() == 2)
    {
      diameter = sqrt((coord(0, 0) - coord(0, 1)) * (coord(0, 0) - coord(0, 1)) +
                      (coord(1, 0) - coord(1, 1)) * (coord(1, 0) - coord(1, 1)) +
                      (coord(2, 0) - coord(2, 1)) * (coord(2, 0) - coord(2, 1)));
    }
    else
      FOUR_C_THROW("coord matrix of nodal positions has wrong dimensions here!");
  }

  // we want an additive extrusion of the radius, hence "2.0"
  if (additiveextrusion_)
    (*diameter_).get_values()[elecolid] = diameter + 2.0 * extrusionvalue;
  else
    (*diameter_).get_values()[elecolid] = diameter * extrusionvalue;

  // first bounding box
  for (int dof = 0; dof < coord.numRows(); dof++)
    (*allbboxes_)(dof).get_values()[elecolid] = 0.5 * (coord(dof, 0) + coord(dof, 1));

  bool periodicbbox = false;
  int shift = 0;
  if (numshifts > 0)
  {
    for (int dof = 0; dof < coord.numRows(); dof++)
    {
      for (int pos = 0; pos < coord.numCols(); pos++)
      {
        if (coord(dof, pos) >=
                (*periodlength_)[dof] - 0.5 * diameter_->local_values_as_span()[elecolid] ||
            coord(dof, pos) <= 0.5 * diameter_->local_values_as_span()[elecolid])
        {
          shift++;
          // shift downwards
          if (coord(dof, pos) >=
              (*periodlength_)[dof] - 0.5 * diameter_->local_values_as_span()[elecolid])
          {
            (*allbboxes_)(3 * shift + dof).get_values()[elecolid] =
                (*allbboxes_)(dof).local_values_as_span()[elecolid] - (*periodlength_)[dof];
            (*allbboxes_)(3 * shift + (dof + 1) % 3).get_values()[elecolid] =
                (*allbboxes_)((dof + 1) % 3).local_values_as_span()[elecolid];
            (*allbboxes_)(3 * shift + (dof + 2) % 3).get_values()[elecolid] =
                (*allbboxes_)((dof + 2) % 3).local_values_as_span()[elecolid];
            // no need to check the second position if pos==0
            periodicbbox = true;
            break;
          }
          // shift upwards
          else if (coord(dof, pos) <= 0.5 * diameter_->local_values_as_span()[elecolid])
          {
            (*allbboxes_)(3 * shift + dof).get_values()[elecolid] =
                (*allbboxes_)(dof).local_values_as_span()[elecolid] + (*periodlength_)[dof];
            (*allbboxes_)(3 * shift + (dof + 1) % 3).get_values()[elecolid] =
                (*allbboxes_)((dof + 1) % 3).local_values_as_span()[elecolid];
            (*allbboxes_)(3 * shift + (dof + 2) % 3).get_values()[elecolid] =
                (*allbboxes_)((dof + 2) % 3).local_values_as_span()[elecolid];
            // no need to check the second position if pos==0
            periodicbbox = true;
            break;
          }
        }
      }
      if (periodicbbox) break;
    }
    (*numshifts_).get_values()[elecolid] = shift;  // store number of shifts
  }

  if (periodic_bc_ && shift < 3)
    for (int i = allbboxes_->num_vectors() - 1; i > (shift + 1) * 3 - 1; i--)
      (*allbboxes_)(i).get_values()[elecolid] = -1e9;

  // last entry: element GID
  (*allbboxes_)(allbboxes_->num_vectors() - 1).get_values()[elecolid] = elegid;
  return;
}
/*-----------------------------------------------------------------------------------------*
 |  locate_all function (private); Recursive division of a 3-dimensional set.    meier 02/11|
 |  [Oct_ID, element-ID] = locate_all( &allbboxes_)                                          |
 |  Performs recursive tree-like division of a set of Bounding Boxes.                      |
 |  N0 is maximum permissible number of "counted" boxes in the leaf octant.                |
 |  Returns vector IND of the same size as rows of allbboxes_ showing which region each     |
 |  box of a set belongs to; binary matrices BX, BY, BZ where each row shows               |
 |  "binary address" of each region are written to files each.                       |
 *----------------------------------------------------------------------------------------*/
bool Beam3ContactOctTree::locate_all()
{
#ifdef MEASURETIME
  double t_octree = Teuchos::Time::wallTime();
#endif

  numcrit1_ = 0;
  numcrit2_ = 0;

  // Convert Core::LinAlg::MultiVector<double> allbboxes_ to vector(std::vector<double>)
  //  std::cout<<allbboxes_->MyLength()<<", "<<allbboxes_->NumVectors()<<std::endl;
  std::vector<std::vector<double>> allbboxesstdvec(
      allbboxes_->local_length(), std::vector<double>(allbboxes_->num_vectors(), 0.0));
  epetra_multi_vec_to_std_vec(*allbboxes_, allbboxesstdvec);

  // initial tree depth value (will be incremented with each recursive call of locate_box()
  int treedepth = 0;

  // Parameters and Initialization
  std::vector<std::vector<int>> bboxesinoctants;
  bboxesinoctants.clear();
  octreelimits_.clear();

  // explanation: bbox2octant[i][j]: bounding box i has a vector of octants j, to which the bounding
  // box belongs
  std::vector<std::vector<int>> bbox2octant(allbboxes_->local_length(), std::vector<int>());

  // Recursively construct octree; Proc 0 only (parallel computing impossible)
  // max depth of OctreeMap
  int maxdepthlocal = 0;
  int bboxlengthlocal = 0;
  // max depth of octree
  int maxnumoctlocal = 0;
  int maxnumoctglobal = 0;
  int maxdepthglobal = 0;
  int bboxlengthglobal = 0;


  // calculate root box limits
  rootbox_ = get_root_box();
  if (Core::Communication::my_mpi_rank(searchdis_.get_comm()) == 0)
  {
    locate_box(allbboxesstdvec, rootbox_, octreelimits_, bboxesinoctants, bbox2octant, treedepth);

    for (int i = 0; i < (int)bbox2octant.size(); i++)
      if ((int)bbox2octant[i].size() > maxnumoctlocal) maxnumoctlocal = (int)bbox2octant[i].size();

    bboxlengthlocal = (int)bboxesinoctants.size();
    for (int i = 0; i < (int)bboxesinoctants.size(); i++)
      if ((int)bboxesinoctants[i].size() > maxdepthlocal)
        maxdepthlocal = (int)bboxesinoctants[i].size();
  }

  // communicate bbox2octant_ to all Procs
  maxnumoctglobal = Core::Communication::max_all(maxnumoctlocal, searchdis_.get_comm());
  bbox2octant_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *(searchdis_.element_col_map()), maxnumoctglobal, true);

  // fill vector
  if (!Core::Communication::my_mpi_rank(searchdis_.get_comm()))
  {
    bbox2octant_->put_scalar(-9.0);
    std_vec_to_epetra_multi_vec(bbox2octant, *bbox2octant_);
  }

  Core::LinAlg::MultiVector<double> bbox2octantrow(
      *(searchdis_.element_row_map()), maxnumoctglobal, true);
  communicate_multi_vector(bbox2octantrow, *bbox2octant_);

  maxdepthglobal = Core::Communication::max_all(maxdepthlocal, searchdis_.get_comm());
  bboxlengthglobal = Core::Communication::max_all(bboxlengthlocal, searchdis_.get_comm());

  /* build temporary, fully overlapping map and row map for octree
   * Note: maxdepthglobal does not occur for a converging Newton iteration. Yet, in some cases, when
   * encountering divergence for the Newton scheme, this might happen.
   * In biopolymer network simulations, this setting is not unlikely and unavoidable. a maximum
   * depth of 0 means, there are no bounding boxes/elements in any octants. Hence, we will not
   * detect any contact and therefore skip the rest of the octree algorithm.*/
  if (maxdepthglobal > 0)
  {
    // create octree maps
    std::vector<int> gids;
    for (int i = 0; i < bboxlengthglobal; i++) gids.push_back(i);
    // crosslinker column and row map
    Core::LinAlg::Map octtreerowmap((int)gids.size(), 0, discret_.get_comm());
    Core::LinAlg::Map octtreemap(-1, (int)gids.size(), gids.data(), 0, discret_.get_comm());

    // build Core::LinAlg::MultiVector<double>s which hold the BBs of the OctreeMap; for
    // communication
    bboxesinoctants_ =
        std::make_shared<Core::LinAlg::MultiVector<double>>(octtreemap, maxdepthglobal);
    Core::LinAlg::MultiVector<double> bboxinoctrow(octtreerowmap, maxdepthglobal, true);

    // fill bboxinoct for Proc 0
    if (Core::Communication::my_mpi_rank(searchdis_.get_comm()) == 0)
    {
      bboxesinoctants_->put_scalar(-9.0);
      //      for (int i=0 ; i<(int)bboxesinoctants.size(); i++ )
      //        for(int j=0; j<(int)bboxesinoctants[i].size(); j++)
      //          (*bboxesinoctants_)[j][i] = bboxesinoctants[i][j];
      std_vec_to_epetra_multi_vec(bboxesinoctants, *bboxesinoctants_);
    }

    // Communication
    communicate_multi_vector(bboxinoctrow, *bboxesinoctants_);

#ifdef MEASURETIME
    if (!Core::Communication::my_mpi_rank(searchdis_.Comm()))
      std::cout << "\nOctree building time:\t\t" << Teuchos::Time::wallTime() - t_octree
                << " seconds" << std::endl;
#endif
    return true;
  }
  else
  {
    return false;
  }
}  // end of method locate_all


/*-----------------------------------------------------------------------------------------*
 |  locate_box function (private);                                               meier 02/11|
 *----------------------------------------------------------------------------------------*/
void Beam3ContactOctTree::locate_box(std::vector<std::vector<double>>& allbboxesstdvec,
    Core::LinAlg::Matrix<6, 1>& lim, std::vector<Core::LinAlg::Matrix<6, 1>>& octreelimits,
    std::vector<std::vector<int>>& bboxesinoctants, std::vector<std::vector<int>>& bbox2octant,
    int& treedepth)
{
  double extrusionvalue = get_bounding_box_extrusion_value();

  // edge length vector of the suboctants
  Core::LinAlg::Matrix<3, 1> newedgelength;
  // vector with limits of the 8 sub octants
  std::vector<Core::LinAlg::Matrix<6, 1>> suboctlimits;
  create_sub_octants(lim, newedgelength, suboctlimits);

  /* Decision to which child box belongs....................
   *
   *           5 ======================== 7
   *           //|                       /||
   *          // |                      //||
   *         //  |                     // ||
   *        //   |                    //  ||
   *       //    |                   //   ||
   *      //     |                  //    ||
   *     //      |                 //     ||
   *    1 ========================= 3     ||
   *    ||       |                ||      ||
   *    ||       |                ||      ||
   *    ||       |      o (center)||      ||
   *    ||      4 ----------------||------ 6
   *    ||      /                 ||     //
   *    ||     /                  ||    //
   *    ||    /                   ||   //
   *    ||   /                    ||  //
   *    ||  /                     || //      y  z
   *    || /                      ||//       | /
   *    ||/                       ||/        |/
   *    0 ========================= 2        ---> x
   *
   */
  std::shared_ptr<Core::LinAlg::Matrix<3, 1>> octcenter = nullptr;

  // Goes through all suboctants
  for (int oct = 0; oct < 8; oct++)
  {
    // Define temporary vector of same size as current allbboxesstdvec
    std::vector<std::vector<double>> bboxsubset;
    bboxsubset.clear();

    // we need the octant centers when applying cylindrical or spherical bounding boxes.
    if (boundingbox_ == Beam3ContactOctTree::cyloriented ||
        boundingbox_ == Beam3ContactOctTree::spherical)
    {
      octcenter = std::make_shared<Core::LinAlg::Matrix<3, 1>>();
      for (int i = 0; i < (int)octcenter->m(); i++)
        (*octcenter)(i) = 0.5 * (suboctlimits[oct](2 * i) + suboctlimits[oct](2 * i + 1));
    }

    int numshifts = 0;

    for (int i = 0; i < (int)allbboxesstdvec.size(); i++)
    {
      int lid =
          searchdis_.element_col_map()->lid((int)allbboxesstdvec[i][allbboxesstdvec[i].size() - 1]);
      if (periodic_bc_) numshifts = numshifts_->local_values_as_span()[lid];

      for (int shift = 0; shift < numshifts + 1; shift++)
      {
        // Processes columns indices 1 to 6
        // 2)loop over the limits of the current octant and check if the current bounding box lies
        // within this octant. 3)Then, check component-wise and leave after first "hit"
        bool assignbboxtooctant = false;
        switch (boundingbox_)
        {
          case Beam3ContactOctTree::axisaligned:
          {
            if (aabb_is_in_this_octant(suboctlimits[oct], allbboxesstdvec[i], shift))
              assignbboxtooctant = true;
          }
          break;
          case Beam3ContactOctTree::cyloriented:
          {
            //            std::cout<<"Octant-Lim:"<<suboctlimits[oct]<<std::endl;
            if (cobb_is_in_this_octant(
                    *octcenter, newedgelength, allbboxesstdvec[i], extrusionvalue, lid, shift))
              assignbboxtooctant = true;
          }
          break;
          case Beam3ContactOctTree::spherical:
          {
            if (spbb_is_in_this_octant(*octcenter, newedgelength, allbboxesstdvec[i], lid, shift))
              assignbboxtooctant = true;
          }
          break;
          default:
            FOUR_C_THROW("No or an invalid Octree type was chosen. Check your input file!");
            break;
        }
        if (assignbboxtooctant)
        {
          bboxsubset.push_back(allbboxesstdvec[i]);
          break;
        }
      }
    }  // end of for-loop which goes through all elements of input

    // current tree depth
    int currtreedepth = treedepth + 1;
    // Check for further recursion by checking number of boxes in octant (first
    // criterion)....................
    int N = (int)bboxsubset.size();

    // If to divide further, let LocateBox call itself with updated inputs
    if (N > minbboxesinoctant_ && currtreedepth < maxtreedepth_ - 1)
      locate_box(
          bboxsubset, suboctlimits[oct], octreelimits, bboxesinoctants, bbox2octant, currtreedepth);
    else
    {
      // no further discretization of the volume because either the maximal tree depth or the
      // minimal number of bounding boxes per octant has been reached this vector holds the IDs of
      // the bounding boxes in this octant
      if (N > 0)
      {
        std::vector<int> boxids;
        boxids.clear();
        // Push back Limits of suboctants to octreelimits
        octreelimits.push_back(suboctlimits[oct]);

        for (int m = 0; m < (int)bboxsubset.size(); m++)
        {
          // note: the Bounding Box ID is the last column entry of the m-th entry vector bboxsubset
          int bboxgid = (int)bboxsubset[m][(int)bboxsubset[m].size() - 1];
          boxids.push_back(bboxgid);
          // assign current octant number to the bounding box
          int collid = searchdis_.element_col_map()->lid(bboxgid);
          // std::cout<<"Collid = "<<collid<<std::endl;
          bbox2octant[collid].push_back((int)bboxesinoctants.size());
        }
        // add bounding box IDs of this octant to the global vector
        bboxesinoctants.push_back(boxids);

        //        std::cout<<"Octant "<<bboxesinoctants.size()<<", treedepth = "<<currtreedepth<<":
        //        "; for(int i=0; i<(int)bboxesinoctants[bboxesinoctants.size()-1].size(); i++)
        //          std::cout<<bboxesinoctants[bboxesinoctants.size()-1][i]<<" ";
        //        std::cout<<", num of bboxes: "<<bboxsubset.size()<<std::endl;
      }
    }
  }  // end of loop which goes through all suboctants
  return;
}  // end of method locate_box

/*----------------------------------------------------------------------------------*
 |  Calculate limits of the root box   (public)                        mueller 11/11|
 *----------------------------------------------------------------------------------*/
Core::LinAlg::Matrix<6, 1> Beam3ContactOctTree::get_root_box()
{
  int entriesperbbox = 0;
  if (boundingbox_ == Beam3ContactOctTree::spherical)
    entriesperbbox = 3;
  else
    entriesperbbox = 6;

  Core::LinAlg::Matrix<6, 1> lim(Core::LinAlg::Initialization::zero);
  // determine globally extremal coordinates and use them as root box.
  // initialize
  lim(0) = 1.0e9;
  lim(1) = -1.0e9;
  // loop over allbboxes_ and determine the extremes

  // number of shifts due to periodic BCs (if present, else 0)
  int numshifts = 0;
  for (int i = 0; i < allbboxes_->local_length(); i++)
  {
    if (periodic_bc_) numshifts = (int)numshifts_->local_values_as_span()[i];
    for (int j = 0; j < entriesperbbox * numshifts + entriesperbbox; j++)
    {
      if ((*allbboxes_)(j).local_values_as_span()[i] < lim(0))  // mins
        lim(0) = (*allbboxes_)(j).local_values_as_span()[i];
      if ((*allbboxes_)(j).local_values_as_span()[i] > lim(1))  // maxes
        lim(1) = (*allbboxes_)(j).local_values_as_span()[i];
    }
  }
  lim(2) = lim(0);
  lim(4) = lim(0);
  lim(3) = lim(1);
  lim(5) = lim(1);
  // safety factor
  lim.scale(1.5);

  return lim;
}
/*-----------------------------------------------------------------------------------*
 |  Create limits of sub octants      (private)                        mueller 02/15|
 |----------------------------------------------------------------------------------*/
void Beam3ContactOctTree::create_sub_octants(Core::LinAlg::Matrix<6, 1>& parentoctlimits,
    Core::LinAlg::Matrix<3, 1>& suboctedgelength,
    std::vector<Core::LinAlg::Matrix<6, 1>>& suboctlimits)
{
  // Center of parent octant
  Core::LinAlg::Matrix<3, 1> parentcenter;

  for (int i = 0; i < (int)parentcenter.num_rows(); i++)
  {
    parentcenter(i) = 0.5 * (parentoctlimits(2 * i) + parentoctlimits(2 * i + 1));
    suboctedgelength(i) = 0.5 * fabs(parentoctlimits(2 * i + 1) - (parentoctlimits)(2 * i));
  }

  // safety: only cubic octants for now.
  if (fabs(suboctedgelength(0) - suboctedgelength(1)) > 1e-8 ||
      fabs(suboctedgelength(0) - suboctedgelength(2)) > 1e-8)
    FOUR_C_THROW("Check get_root_box()! Octants must be cubic!");

  suboctlimits.clear();
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      for (int k = 0; k < 2; k++)
      {
        Core::LinAlg::Matrix<6, 1> sublim;
        sublim(0) = parentcenter(0) + (i - 1) * suboctedgelength(0);
        sublim(1) = parentcenter(0) + i * suboctedgelength(0);
        sublim(2) = parentcenter(1) + (j - 1) * suboctedgelength(1);
        sublim(3) = parentcenter(1) + j * suboctedgelength(1);
        sublim(4) = parentcenter(2) + (k - 1) * suboctedgelength(2);
        sublim(5) = parentcenter(2) + k * suboctedgelength(2);

        suboctlimits.push_back(sublim);
      }
    }
  }
  return;
}

/*-----------------------------------------------------------------------------------*
 |  Check if AABB in octant           (private)                        mueller 02/15|
 |----------------------------------------------------------------------------------*/
bool Beam3ContactOctTree::aabb_is_in_this_octant(
    Core::LinAlg::Matrix<6, 1>& suboctlimits, std::vector<double>& bboxcoords, int& shift)
{
  if (!((suboctlimits(0) >= bboxcoords[6 * shift + 1]) ||
          (bboxcoords[6 * shift] >= suboctlimits(1)) ||
          (suboctlimits(2) >= bboxcoords[6 * shift + 3]) ||
          (bboxcoords[6 * shift + 2] >= suboctlimits(3)) ||
          (suboctlimits(4) >= bboxcoords[6 * shift + 5]) ||
          (bboxcoords[6 * shift + 4] >= suboctlimits(5))))
    return true;
  else
    return false;
}

/*-----------------------------------------------------------------------------------*
 |  Check if COBB in octant           (private)                        mueller 02/15|
 |----------------------------------------------------------------------------------*/
bool Beam3ContactOctTree::cobb_is_in_this_octant(Core::LinAlg::Matrix<3, 1>& octcenter,
    Core::LinAlg::Matrix<3, 1>& newedgelength, std::vector<double>& bboxcoords,
    double& extrusionvalue, int& lid, int& shift)
{
  /* General idea:
   * Two checks ensure whether a bounding box lies within the given octant
   * 1) Check the two end points of the bounding box
   * 2) Check the linear connection between the end points for an intersection with the octant
   * faces.
   * @1) Evaluate the closest intersection point of a line going through the octant center and end
   * points. If the distance to the end point is smaller than the distance between octant center and
   * the intersection with the nearest octant face plane-> intersection
   * @2) Calculate the intersection points of the line connecting the two end point of the bounding
   * box with the octant face planes. If the signed distance parameter lambda_k lies within the
   * boundaries of the bounding box -> intersection.
   */
  // increase the radius of the box in order to have some additional buffer volume.
  double boxradius = 0.0;
  if (additiveextrusion_)
    boxradius = extrusionvalue + 0.5 * diameter_->local_values_as_span()[lid];
  else
    boxradius = 0.5 * extrusionvalue * diameter_->local_values_as_span()[lid];

  // criterion I: distance
  // check for both bounding box end points
  Core::LinAlg::Matrix<3, 1> dir;
  for (int j = 0; j < 2; j++)
  {
    // std::vector to Core::LinAlg::Matrix (slower but nicer to read)
    Core::LinAlg::Matrix<3, 1> bboxendpoint;
    bboxendpoint(0) = bboxcoords[6 * shift + 3 * j];
    bboxendpoint(1) = bboxcoords[6 * shift + 3 * j + 1];
    bboxendpoint(2) = bboxcoords[6 * shift + 3 * j + 2];

    dir = bboxendpoint;
    dir -= octcenter;
    double dist = dir.norm2();
    dir.scale(1.0 / dist);

    // find shortest distance to face planes
    Core::LinAlg::Matrix<3, 1> lambda;
    for (int k = 0; k < (int)lambda.num_rows(); k++)
    {
      double normal = 0.0;
      if (bboxendpoint(k) > octcenter(k))
        normal = 1.0;
      else if (bboxendpoint(k) < octcenter(k))
        normal = -1.0;
      else  // no component in direction k -> no intersection -> sinularity -> lambda ->infty ->
            // insert high enough dummy value
      {
        lambda(k) = 1.0e9;
        break;
      }
      lambda(k) = 0.5 * (newedgelength(k) * normal) / (dir(k));
    }
    // closest intersection point with face plane
    double lambda_min = lambda.min_value();

    // only one of the endpoint must lie in the octant
    if (lambda_min + boxradius >= dist) return true;
  }

  // Criterion II: some part between the two given end points of the bbox may lie within the octant
  // determine unit ofrientation for line equation
  Core::LinAlg::Matrix<3, 1> bboxorient;
  for (int k = 0; k < (int)bboxorient.num_rows(); k++)
    bboxorient(k) = bboxcoords[6 * shift + k + 3] - bboxcoords[6 * shift + k];
  double bboxlength = bboxorient.norm2();
  //    std::cout<<"bboxlentgth = "<<bboxlength<<std::endl;
  bboxorient.scale(1.0 / bboxorient.norm2());
  // component-wise check if the intersection point of the line lies outside or inside the octant
  // (using an outward-facing normal)
  for (int k = 0; k < (int)bboxorient.num_rows(); k++)
  {
    // no intersection will occur since bbox is not slanted towards the face
    if (bboxorient(k) == 0.0) continue;

    // retrieve the correct value for the outward-facing normal
    //      std::cout<<"bboxcoords[6*shift+k,"<<6*shift+k<<"] = "<<bboxcoords[6*shift+k]<<std::endl;
    double normal = 0.0;
    if (bboxcoords[6 * shift + k] >= octcenter(k))
      normal = 1.0;
    else if (bboxcoords[6 * shift + k] < octcenter(k))
      normal = -1.0;

    // signed distance from the first endpoint to the intersection point
    double lambda_k = (octcenter(k) + 0.5 * newedgelength(k) * normal - bboxcoords[6 * shift + k]) /
                      bboxorient(k);
    // calculate intersection point coords and subsequently the distance from octcenter to the
    // intersecion
    //(only for promising candidates)
    if (lambda_k > -boxradius && lambda_k < bboxlength + boxradius)
    {
      Core::LinAlg::Matrix<3, 1> dist;
      for (int l = 0; l < (int)dist.num_rows(); l++)
        dist(l) = bboxcoords[6 * shift + l] + lambda_k * bboxorient(l) - octcenter(l);
      // bbox is in octant if one intersection point lies on the octant surface.
      if (dist.norm2() <= 0.5 * newedgelength(k) * sqrt(3.0) + boxradius)
      {
        //        if((int)(*numshifts_)[lid]>0)
        //        {
        //          Core::LinAlg::Matrix<3,1> aux(dist);
        //          aux += octcenter;
        //          std::cout<<"BBOX "<<lid<<": ";
        //          for(int k=0; k<(int)bboxcoords.size(); k++)
        //            if(bboxcoords[k]>-1e9)
        //              std::cout<<bboxcoords[k]<<" ";
        //          std::cout<<std::endl;
        //          std::cout<<"; \nISEC : "<<aux(0)<<" "<<aux(1)<<" "<<aux(2)<<"\nLambda =
        //          "<<lambda_k<<", normal = "<<normal<<std::endl; FOUR_C_THROW("CRIT II");
        //        }
        return true;
      }
    }
  }
  return false;
}

/*-----------------------------------------------------------------------------------*
 |  Check if SPBB in octant           (private)                        mueller 02/15|
 |----------------------------------------------------------------------------------*/
bool Beam3ContactOctTree::spbb_is_in_this_octant(Core::LinAlg::Matrix<3, 1>& octcenter,
    Core::LinAlg::Matrix<3, 1>& newedgelength, std::vector<double>& bboxcoords, int& lid,
    int& shift)
{
  double maxdistx = 0.5 * (newedgelength(0) + diameter_->local_values_as_span()[lid]);
  double maxdisty = 0.5 * (newedgelength(1) + diameter_->local_values_as_span()[lid]);
  double maxdistz = 0.5 * (newedgelength(2) + diameter_->local_values_as_span()[lid]);

  double distx = fabs(octcenter(0) - bboxcoords[3 * shift]);
  double disty = fabs(octcenter(1) - bboxcoords[3 * shift + 1]);
  double distz = fabs(octcenter(2) - bboxcoords[3 * shift + 2]);
  // check: component-wise distance
  if (distx <= maxdistx && disty <= maxdisty && distz <= maxdistz)
    return true;
  else
    return false;
}

/*-----------------------------------------------------------------------------------*
 |  Bounding Box intersection function (private)                          meier 01/11|
 |  Intersects Bounding Boxes in same line of map OctreeMap                          |
 |  Gives back vector of intersection pairs                                          |
 *----------------------------------------------------------------------------------*/
void Beam3ContactOctTree::bounding_box_intersection(
    std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    std::vector<std::vector<Core::Elements::Element*>>& contactpairelements)
{
#ifdef MEASURETIME
  double t_search = Teuchos::Time::wallTime();
#endif
  // Build contact pair Map
  std::map<int, std::vector<int>> contactpairmap;
  // create contact pair vector, redundant on all Procs; including redundant pairs
  // for-loop lines of map
  for (int i = 0; i < bboxesinoctants_->local_length(); i++)
  {
    // for-loop index first box
    for (int j = 0; j < bboxesinoctants_->num_vectors(); j++)
    {
      // first box ID
      std::vector<int> bboxIDs(2, 0);
      bboxIDs[0] = (int)(*bboxesinoctants_)(j).local_values_as_span()[i];
      // for-loop second box
      for (int k = j + 1; k < bboxesinoctants_->num_vectors(); k++)
      {
        bboxIDs[1] = (int)(*bboxesinoctants_)(k).local_values_as_span()[i];

        // exclude element pairs sharing one node
        // contact flag
        bool considerpair = false;
        // only consider existing bounding boxes, i.e. no dummy entries "-9.0"
        if (bboxIDs[0] > -1 && bboxIDs[1] > -1)
        {
          considerpair = true;
          Core::Elements::Element* element1 = searchdis_.g_element(bboxIDs[0]);
          Core::Elements::Element* element2 = searchdis_.g_element(bboxIDs[1]);

          // Here we have introduced an criteria in order to sort neighboring contact element pairs
          // out. This method is based on the assumption, that two elements sharing the same node
          // should not get into contact. In contrary to a former criterion based on filament
          // numbers (which forbids self contact), this method works for arbitrary element types and
          // still allows for self contact!!!
          if (BeamInteraction::elements_share_node(*element1, *element2)) considerpair = false;
        }
        if (considerpair)
        {
          // apply different bounding box intersection schemes
          bool intersection = false;
          switch (boundingbox_)
          {
            case Beam3ContactOctTree::axisaligned:
              intersection = intersection_aabb(bboxIDs);
              break;
            case Beam3ContactOctTree::cyloriented:
              intersection = intersection_cobb(bboxIDs);
              break;
            case Beam3ContactOctTree::spherical:
              intersection = intersection_spbb(bboxIDs);
              break;
            default:
              FOUR_C_THROW("No or an invalid Octree type was chosen. Check your input file!");
              break;
          }

          if (intersection)
          {
            // note: creation of unique "first" entries in map, attention: IDs identical to
            // crosslinker GIDs!!
            int mapfirst = (bboxIDs[0] + 1) * basisnodes_ + bboxIDs[1];
            contactpairmap.insert(std::pair<int, std::vector<int>>(mapfirst, bboxIDs));
          }
        }
      }
    }
  }
  // build Pair Vector from contactpairmap
  std::map<int, std::vector<int>>::iterator it;
  for (it = contactpairmap.begin(); it != contactpairmap.end(); it++)
  {
    // if(!Core::Communication::my_mpi_rank(discret_.Comm()))
    // std::cout << std::scientific << (*it).first <<"  "<< ((*it).second)[0]<<" "<<
    // ((*it).second)[1]<<std::endl;
    int collid1 = searchdis_.element_col_map()->lid(((*it).second)[0]);
    int collid2 = searchdis_.element_col_map()->lid(((*it).second)[1]);
    Core::Elements::Element* tempele1 = searchdis_.l_col_element(collid1);
    Core::Elements::Element* tempele2 = searchdis_.l_col_element(collid2);
    // matrices to store nodal coordinates
    Core::LinAlg::SerialDenseMatrix ele1pos(3, tempele1->num_node());
    Core::LinAlg::SerialDenseMatrix ele2pos(3, tempele2->num_node());
    // store nodal coordinates of element 1
    for (int m = 0; m < tempele1->num_node(); ++m)
    {
      int tempGID = (tempele1->node_ids())[m];
      Core::LinAlg::Matrix<3, 1> temppos = currentpositions[tempGID];
      for (int n = 0; n < 3; n++) ele1pos(n, m) = temppos(n);
    }
    // store nodal coordinates of element 2
    for (int m = 0; m < tempele2->num_node(); ++m)
    {
      int tempGID = (tempele2->node_ids())[m];
      Core::LinAlg::Matrix<3, 1> temppos = currentpositions[tempGID];
      for (int n = 0; n < 3; n++) ele2pos(n, m) = temppos(n);
    }
    // add to pair vector
    std::vector<Core::Elements::Element*> temp_vec(2);
    temp_vec[0] = tempele1;
    temp_vec[1] = tempele2;
    contactpairelements.push_back(temp_vec);
  }
  // if(!Core::Communication::my_mpi_rank(discret_.Comm()))
  // std::cout<<"number of boxes: "<<counter<<std::endl;

#ifdef MEASURETIME
  double isectimelocal = Teuchos::Time::wallTime() - t_search;
  double isectimeglobal = 0.0;

  isectimeglobal = Core::Communication::max_all(isectimelocal, searchdis_.Comm());
  Core::Communication::barrier(discret_.Comm());
  if (!Core::Communication::my_mpi_rank(searchdis_.Comm()))
    std::cout << "intersection time:\t\t" << isectimeglobal << " seconds" << std::endl;
#endif

  return;
}  // end of method bounding_box_intersection()

/*-----------------------------------------------------------------------------------*
 |  Axis Aligned Bounding Box intersection function when both bounding boxes         |
 |  represent actual finite elements  (private)                         mueller 11/11|
 *----------------------------------------------------------------------------------*/
bool Beam3ContactOctTree::intersection_aabb(
    const std::vector<int>& bboxIDs, std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits)
{
  // Why bboxlimits separately: The idea is that we can use this method to check whether a
  // hypothetical bounding box (i.e. without an element) can be tested for intersection. Hence, we
  // store the limits of this bounding box into bboxlimits if needed. Since the hypothetical
  // bounding box stands for a crosslinker to be set, we just need the exact dimensions of the
  // element
  if (bboxlimits != nullptr) FOUR_C_THROW("Not in use!");

  // translate box / element GIDs to ElementColMap()-LIDs
  // note: GID and ColumnMap LID are usually the same except for crosslinker elements from
  // statistical mechanics
  int entry1 = searchdis_.element_col_map()->lid(bboxIDs[0]);
  int entry2 = searchdis_.element_col_map()->lid(bboxIDs[1]);

  // Initialization....................
  double a_xmin, a_xmax, a_ymin, a_ymax, a_zmin, a_zmax;
  double b_xmin, b_xmax, b_ymin, b_ymax, b_zmin, b_zmax;

  int I = 0;
  int J = 0;

  if (periodic_bc_)
  {
    I = (int)numshifts_->local_values_as_span()[entry1];
    J = (int)numshifts_->local_values_as_span()[entry2];
  }
  // note: n shifts means n+1 segments
  for (int i = 0; i < I + 1; i++)
  {
    for (int j = 0; j < J + 1; j++)
    {
      // intersection Test
      a_xmin = (*allbboxes_)(i * 6).local_values_as_span()[entry1];
      a_xmax = (*allbboxes_)(i * 6 + 1).local_values_as_span()[entry1];
      a_ymin = (*allbboxes_)(i * 6 + 2).local_values_as_span()[entry1];
      a_ymax = (*allbboxes_)(i * 6 + 3).local_values_as_span()[entry1];
      a_zmin = (*allbboxes_)(i * 6 + 4).local_values_as_span()[entry1];
      a_zmax = (*allbboxes_)(i * 6 + 5).local_values_as_span()[entry1];

      b_xmin = (*allbboxes_)(j * 6).local_values_as_span()[entry2];
      b_xmax = (*allbboxes_)(j * 6 + 1).local_values_as_span()[entry2];
      b_ymin = (*allbboxes_)(j * 6 + 2).local_values_as_span()[entry2];
      b_ymax = (*allbboxes_)(j * 6 + 3).local_values_as_span()[entry2];
      b_zmin = (*allbboxes_)(j * 6 + 4).local_values_as_span()[entry2];
      b_zmax = (*allbboxes_)(j * 6 + 5).local_values_as_span()[entry2];

      // if intersection exists, return true
      if (!((a_xmin >= b_xmax || b_xmin >= a_xmax) || (a_ymin >= b_ymax || b_ymin >= a_ymax) ||
              (a_zmin >= b_zmax || b_zmin >= a_zmax)))
        return true;
    }
  }
  return false;
}

/*-----------------------------------------------------------------------------------*
 |  Cylindrical Oriented Bounding Box intersection function when both bounding boxes |
 |  represent actual finite elements  (private)                         mueller 11/11|
 *----------------------------------------------------------------------------------*/
bool Beam3ContactOctTree::intersection_cobb(
    const std::vector<int>& bboxIDs, std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits)
{
  // Why bboxlimits separately: The idea is that we can use this method to check whether a
  // hypothetical bounding box (i.e. without an element) can be tested for intersection. Hence, we
  // store the limits of this bounding box into bboxlimits if needed. Since the hypothetical
  // bounding box stands for a crosslinker to be set, we just need the exact dimensions of the
  // element
  if (bboxlimits != nullptr) FOUR_C_THROW("Not in use!");

  /* intersection test by calculating the distance between the two bounding box center lines
   * and comparing it to the respective diameters of the beams*/
  bool intersection = false;


  int bboxid0 = searchdis_.element_col_map()->lid(bboxIDs[0]);
  int bboxid1 = searchdis_.element_col_map()->lid(bboxIDs[1]);

  // In case of a hypothetical BB, simply take the last beam element's diameter (does the job for
  // now)
  double bbox0diameter = 0.0;
  double bbox1diameter = 0.0;
  bbox0diameter = diameter_->local_values_as_span()[bboxid0];
  bbox1diameter = diameter_->local_values_as_span()[bboxid1];

  double radiusextrusion = extrusionvalue_->at(1);
  if (additiveextrusion_)
  {
    bbox0diameter += 2.0 * radiusextrusion;
    bbox1diameter += 2.0 * radiusextrusion;
  }
  else
  {
    bbox1diameter *= radiusextrusion;
    bbox0diameter *= radiusextrusion;
  }

  // Distance at which intersection happens
  double distancelimit = 0.5 * (bbox0diameter + bbox1diameter);

  Core::LinAlg::Matrix<3, 1, double> t1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> t2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> r1_a;
  Core::LinAlg::Matrix<3, 1, double> r1_b;
  Core::LinAlg::Matrix<3, 1, double> r2_a;
  Core::LinAlg::Matrix<3, 1, double> r2_b;

  int numshifts_bbox0 = 0;
  int numshifts_bbox1 = 0;

  if (periodic_bc_)
  {
    numshifts_bbox0 = (int)numshifts_->local_values_as_span()[bboxid0];
    numshifts_bbox1 = (int)numshifts_->local_values_as_span()[bboxid1];
  }

  //  if(numshifts_bbox0>0 || numshifts_bbox1>0)
  //    std::cout<<"PAIR "<<bboxid0<<", "<<bboxid1<<"; numshifts: "<<numshifts_bbox0<<",
  //    "<<numshifts_bbox1<<std::endl;

  //  double t_start = Teuchos::Time::wallTime();
  // intersect
  for (int i = 0; i < numshifts_bbox0 + 1; i++)
  {
    for (int j = 0; j < numshifts_bbox1 + 1; j++)
    {
      for (int k = 0; k < (int)r1_a.num_rows(); k++)
      {
        // first bbox
        r1_a(k) = (*allbboxes_)(6 * i + k).local_values_as_span()[bboxid0];
        r1_b(k) = (*allbboxes_)(6 * i + k + 3).local_values_as_span()[bboxid0];
        // second bbox
        r2_a(k) = (*allbboxes_)(6 * j + k).local_values_as_span()[bboxid1];
        r2_b(k) = (*allbboxes_)(6 * j + k + 3).local_values_as_span()[bboxid1];
      }

      //      double t_02 = Teuchos::Time::wallTime();
      t1 = Core::FADUtils::diff_vector(r1_b, r1_a);
      t2 = Core::FADUtils::diff_vector(r2_b, r2_a);
      double angle = BeamInteraction::calc_angle(t1, t2);
      //      std::cout<<"(i,j) = "<<i<<", "<<j<<",  Angle Calc : "<<
      //      Teuchos::Time::wallTime()-t_02<<std::endl;

      //      double t_03 = Teuchos::Time::wallTime();
      // non-parallel case
      if (angle > ANGLETOL)
      {
        std::pair<double, double> closestpoints(std::make_pair(0.0, 0.0));
        bool etaset = false;
        intersection = BeamInteraction::intersect_arbitrary_cylinders(
            r1_a, r1_b, r2_a, r2_b, distancelimit, closestpoints, etaset);
        //        std::cout<<"(i,j) = "<<i<<", "<<j<<",  Isec nonpar: "<<
        //        Teuchos::Time::wallTime()-t_03<<std::endl;
      }
      else  // parallel case
      {
        intersection =
            BeamInteraction::intersect_parallel_cylinders(r1_a, r1_b, r2_a, r2_b, distancelimit);
        //        std::cout<<"(i,j) = "<<i<<", "<<j<<",  Isec par   : "<<
        //        Teuchos::Time::wallTime()-t_03<<std::endl;
      }
      //      if(numshifts_bbox0>0 || numshifts_bbox1>0)
      //        std::cout<<"  (i,j) = "<<i<<", "<<j<<": intersection = "<<intersection<<std::endl;
      if (intersection) break;
    }
    // return upon first found intersection, the remainder does not matter anyway+
    if (intersection) break;
  }

  //  std::cout<<"intersection total: "<< Teuchos::Time::wallTime()-t_start<<std::endl;
  return intersection;
}

/*-----------------------------------------------------------------------------------*
 |  Spherical Bounding Box intersection function when both bounding boxes           |
 |  for linkers                                       (private)        mueller 01/12|
 *----------------------------------------------------------------------------------*/
bool Beam3ContactOctTree::intersection_spbb(
    const std::vector<int>& bboxIDs, std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits)
{
  // Why bboxlimits separately: The idea is that we can use this method to check whether a
  // hypothetical bounding box (i.e. without an element) can be tested for intersection. Hence, we
  // store the limits of this bounding box into bboxlimits if needed. Since the hypothetical
  // bounding box stands for a crosslinker to be set, we just need the exact dimensions of the
  // element
  if (bboxlimits != nullptr) FOUR_C_THROW("Not in use!");

  int bboxid0 = searchdis_.element_col_map()->lid(bboxIDs[0]);
  int bboxid1 = searchdis_.element_col_map()->lid(bboxIDs[1]);

  int numshifts_bbox0 = 0;
  int numshifts_bbox1 = 0;

  if (periodic_bc_)
  {
    numshifts_bbox0 = numshifts_->local_values_as_span()[bboxid0];
    numshifts_bbox1 = numshifts_->local_values_as_span()[bboxid1];
  }

  for (int i = 0; i < numshifts_bbox0 + 1; i++)
  {
    for (int j = 0; j < numshifts_bbox1 + 1; j++)
    {
      // calculate distance between sphere centers
      double centerdist = sqrt(((*allbboxes_)(0).local_values_as_span()[bboxid0] -
                                   (*allbboxes_)(0).local_values_as_span()[bboxid1]) *
                                   ((*allbboxes_)(0).local_values_as_span()[bboxid0] -
                                       (*allbboxes_)(0).local_values_as_span()[bboxid1]) +
                               ((*allbboxes_)(1).local_values_as_span()[bboxid0] -
                                   (*allbboxes_)(1).local_values_as_span()[bboxid1]) *
                                   ((*allbboxes_)(1).local_values_as_span()[bboxid0] -
                                       (*allbboxes_)(1).local_values_as_span()[bboxid1]) +
                               ((*allbboxes_)(2).local_values_as_span()[bboxid0] -
                                   (*allbboxes_)(2).local_values_as_span()[bboxid1]) *
                                   ((*allbboxes_)(2).local_values_as_span()[bboxid0] -
                                       (*allbboxes_)(2).local_values_as_span()[bboxid1]));

      double bboxradius0 = 0.5 * diameter_->local_values_as_span()[bboxid0];
      double bboxradius1 = 0.5 * diameter_->local_values_as_span()[bboxid1];

      // intersect spheres
      if (centerdist <= bboxradius0 + bboxradius1) return true;  // spheres have overlapping volumes
    }
  }
  return false;  // no overlap
}

/*-----------------------------------------------------------------------*
 | communicate Vector to all Processors                    mueller 11/11 |
 *-----------------------------------------------------------------------*/
void Beam3ContactOctTree::communicate_vector(Core::LinAlg::Vector<double>& InVec,
    Core::LinAlg::Vector<double>& OutVec, bool zerofy, bool doexport, bool doimport)
{
  /* zerofy InVec at the beginning of each search except for Proc 0
   * for subsequent export and reimport. This way, we guarantee redundant information
   * on all processors. */

  // first, export the values of OutVec on Proc 0 to InVecs of all participating processors
  Core::LinAlg::Export exporter(OutVec.get_map(), InVec.get_map());
  Core::LinAlg::Import importer(OutVec.get_map(), InVec.get_map());
  if (doexport)
  {
    // zero out all vectors which are not Proc 0. Then, export Proc 0 data to InVec map.
    if (Core::Communication::my_mpi_rank(discret_.get_comm()) != 0 && zerofy)
      OutVec.put_scalar(0.0);
    InVec.export_to(OutVec, exporter, Add);
  }
  if (doimport) OutVec.import(InVec, importer, Insert);
  return;
}

/*-----------------------------------------------------------------------*
 | communicate MultiVector to all Processors               mueller 11/11 |
 *-----------------------------------------------------------------------*/
void Beam3ContactOctTree::communicate_multi_vector(Core::LinAlg::MultiVector<double>& InVec,
    Core::LinAlg::MultiVector<double>& OutVec, bool zerofy, bool doexport, bool doimport)
{
  // first, export the values of OutVec on Proc 0 to InVecs of all participating processors
  Core::LinAlg::Export exporter(OutVec.get_map(), InVec.get_map());
  Core::LinAlg::Import importer(OutVec.get_map(), InVec.get_map());
  if (doexport)
  {
    // zero out all vectors which are not Proc 0. Then, export Proc 0 data to InVec map.
    if (Core::Communication::my_mpi_rank(discret_.get_comm()) != 0 && zerofy)
      OutVec.put_scalar(0.0);
    InVec.export_to(OutVec, exporter, Add);
  }
  if (doimport) OutVec.import(InVec, importer, Insert);
  return;
}

/*-----------------------------------------------------------------------*
 | Calc. max and min values of node coordinates              meier 05/14 |
 *-----------------------------------------------------------------------*/
void Beam3ContactOctTree::calc_corner_pos(Core::Elements::Element* element,
    std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    Core::LinAlg::SerialDenseMatrix& coord)
{
  Core::LinAlg::Matrix<3, 1> coord_max(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> coord_min(Core::LinAlg::Initialization::zero);
  for (int k = 0; k < 3; ++k)
  {
    coord_max(k) = -1.0e12;
    coord_min(k) = 1.0e12;
  }

  for (int i = 0; i < element->num_node(); i++)
  {
    int gid = element->nodes()[i]->id();
    Core::LinAlg::Matrix<3, 1> coord_aux = currentpositions[gid];
    for (int j = 0; j < 3; j++)
    {
      if (coord_aux(j) < coord_min(j)) coord_min(j) = coord_aux(j);
      if (coord_aux(j) > coord_max(j)) coord_max(j) = coord_aux(j);
    }
  }

  for (int k = 0; k < 3; ++k)
  {
    if (coord_max(k) == -1.0e12 || coord_min(k) == 1.0e12)
      FOUR_C_THROW("ERROR: calc_corner_pos was not successful");
  }
  /*  We don't fill real node coordinates in the matrix coord, but only the min/max x-,y-,z- values
     over all nodes of the considered elements. These coordinates coord_max and coord_min can thus
     be interpreted as fictitious nodes which are sufficient in order to create a correct bounding
     box. */

  for (int j = 0; j < 3; j++)
  {
    coord(j, 0) = coord_min(j);
    coord(j, 1) = coord_max(j);
  }

  return;
}

/*-----------------------------------------------------------------------*
 | unshift coordinates               mueller 02/15 |
 *-----------------------------------------------------------------------*/
void Beam3ContactOctTree::undo_effect_of_periodic_boundary_condition(
    Core::LinAlg::SerialDenseMatrix& coord, std::vector<int>& cut, int& numshifts)
{
  if (coord.numRows() != 3 || coord.numCols() != 2)
    FOUR_C_THROW("coord must have the dimension m()==3, n()==2!");
  if ((int)cut.size() != 3) FOUR_C_THROW("cut is of wrong size {}!", (int)cut.size());
  // By definition, we shift the second position!
  for (int dof = 0; dof < coord.numRows(); dof++)
  {
    if (fabs(coord(dof, 1) - periodlength_->at(dof) - coord(dof, 0)) <
        fabs(coord(dof, 1) - coord(dof, 0)))
    {
      numshifts++;
      cut[dof] = 1.0;
      coord(dof, 1) -= periodlength_->at(dof);
    }
    if (fabs(coord(dof, 1) + periodlength_->at(dof) - coord(dof, 0)) <
        fabs(coord(dof, 1) - coord(dof, 0)))
    {
      numshifts++;
      cut[dof] = 2.0;
      coord(dof, 1) += periodlength_->at(dof);
    }
  }
  return;
}
/*-----------------------------------------------------------------------*
 | Get bounding-box-specific extrusion value               mueller 02/15 |
 *-----------------------------------------------------------------------*/
double Beam3ContactOctTree::get_bounding_box_extrusion_value()
{
  double extrusionvalue = -1.0;
  switch (boundingbox_)
  {
    case Beam3ContactOctTree::axisaligned:
      extrusionvalue = extrusionvalue_->at(0);
      break;
    case Beam3ContactOctTree::cyloriented:
      extrusionvalue = std::max(extrusionvalue_->at(0), extrusionvalue_->at(1));
      break;
    case Beam3ContactOctTree::spherical:
      extrusionvalue = extrusionvalue_->at(0);
      break;
    default:
      FOUR_C_THROW("No bounding box type chosen!");
  }
  if (extrusionvalue < 0.0)
    FOUR_C_THROW("Check bounding box extrusion value {} < 0.0!", extrusionvalue);
  return extrusionvalue;
}

FOUR_C_NAMESPACE_CLOSE
