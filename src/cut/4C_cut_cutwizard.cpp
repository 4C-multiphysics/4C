// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_cutwizard.hpp"

#include "4C_cut_combintersection.hpp"
#include "4C_cut_elementhandle.hpp"
#include "4C_cut_node.hpp"
#include "4C_cut_parallel.hpp"
#include "4C_cut_sidehandle.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::CutWizard::BackMesh::init(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& back_disp_col,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& back_levelset_col)
{
  back_disp_col_ = back_disp_col;
  back_levelset_col_ = back_levelset_col;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Cut::CutWizard::BackMesh::num_my_col_elements() const
{
  return back_discret_->num_my_col_elements();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::Elements::Element* Cut::CutWizard::BackMesh::l_col_element(int lid) const
{
  return back_discret_->l_col_element(lid);
}

/*-------------------------------------------------------------*
 * constructor
 *-------------------------------------------------------------*/
Cut::CutWizard::CutWizard(const std::shared_ptr<Core::FE::Discretization>& backdis,
    std::function<void(const Core::Nodes::Node& node, std::vector<int>& lm)> global_dof_indices)
    : back_mesh_(std::make_shared<CutWizard::BackMesh>(backdis, this)),
      global_dof_indices_(std::move(global_dof_indices)),
      comm_(backdis->get_comm()),
      myrank_(Core::Communication::my_mpi_rank(backdis->get_comm())),
      intersection_(std::make_shared<Cut::CombIntersection>(myrank_)),
      do_mesh_intersection_(false),
      do_levelset_intersection_(false),
      level_set_sid_(-1),
      v_cellgausstype_(Cut::VCellGaussPts_Tessellation),
      b_cellgausstype_(Cut::BCellGaussPts_Tessellation),
      gmsh_output_(false),
      tetcellsonly_(false),
      screenoutput_(false),
      lsv_only_plus_domain_(true),
      is_set_options_(false),
      is_cut_prepare_performed_(false)
{
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::CutWizard::CutWizard(MPI_Comm comm)
    : back_mesh_(nullptr),
      comm_(comm),
      myrank_(Core::Communication::my_mpi_rank(comm)),
      intersection_(std::make_shared<Cut::CombIntersection>(myrank_)),
      do_mesh_intersection_(false),
      do_levelset_intersection_(false),
      level_set_sid_(-1),
      v_cellgausstype_(Cut::VCellGaussPts_Tessellation),
      b_cellgausstype_(Cut::BCellGaussPts_Tessellation),
      gmsh_output_(false),
      tetcellsonly_(false),
      screenoutput_(false),
      lsv_only_plus_domain_(false),
      is_set_options_(false),
      is_cut_prepare_performed_(false)
{
}

/*========================================================================*/
//! @name Setters
/*========================================================================*/

/*-------------------------------------------------------------*
 * set options and flags used during the cut
 *--------------------------------------------------------------*/
void Cut::CutWizard::set_options(
    const Teuchos::ParameterList& cutparams,         //!< parameter list for cut options
    Cut::NodalDofSetStrategy nodal_dofset_strategy,  //!< strategy for nodal dofset management
    Cut::VCellGaussPts VCellgausstype,  //!< Gauss point generation method for Volumecell
    Cut::BCellGaussPts BCellgausstype,  //!< Gauss point generation method for Boundarycell
    std::string output_prefix,          //!< prefix for output files
    bool gmsh_output,                   //!< print write gmsh output for cut
    bool positions,     //!< set inside and outside point, facet and volumecell positions
    bool tetcellsonly,  //!< generate only tet cells
    bool screenoutput   //!< print screen output
)
{
  v_cellgausstype_ = VCellgausstype;
  b_cellgausstype_ = BCellgausstype;
  output_prefix_ = output_prefix;
  gmsh_output_ = gmsh_output;
  tetcellsonly_ = tetcellsonly;
  screenoutput_ = screenoutput;

  // set position option to the intersection class
  intersection_->set_find_positions(positions);
  intersection_->set_nodal_dof_set_strategy(nodal_dofset_strategy);

  // Initialize Cut Parameters based on input file section CUT GENERAL
  intersection_->get_options().init_by_paramlist(cutparams);

  is_set_options_ = true;
}


/*-------------------------------------------------------------*
 * set displacement and level-set vectors used during the cut
 *--------------------------------------------------------------*/
void Cut::CutWizard::set_background_state(
    std::shared_ptr<const Core::LinAlg::Vector<double>>
        back_disp_col,  //!< col vector holding background ALE displacements for backdis
    std::shared_ptr<const Core::LinAlg::Vector<double>>
        back_levelset_col,  //!< col vector holding nodal level-set values based on backdis
    int level_set_sid       //!< global id for level-set side
)
{
  // set state vectors used in cut
  back_mesh_->init(back_disp_col, back_levelset_col);
  level_set_sid_ = level_set_sid;

  do_levelset_intersection_ = back_mesh_->is_level_set();
}


/*-------------------------------------------------------------*
 * set displacement and level-set vectors used during the cut
 *--------------------------------------------------------------*/
void Cut::CutWizard::add_cutter_state(const int mc_idx,
    std::shared_ptr<Core::FE::Discretization> cutter_dis,
    std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col)
{
  add_cutter_state(0, cutter_dis, cutter_disp_col, 0);
}

/*-------------------------------------------------------------*
 * set displacement and level-set vectors used during the cut
 *--------------------------------------------------------------*/
void Cut::CutWizard::add_cutter_state(const int mc_idx,
    std::shared_ptr<Core::FE::Discretization> cutter_dis,
    std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col, const int start_ele_gid)
{
  std::map<int, std::shared_ptr<CutterMesh>>::iterator cm = cutter_meshes_.find(mc_idx);

  if (cm != cutter_meshes_.end())
    FOUR_C_THROW("cutter mesh with mesh coupling index {} already set", mc_idx);

  cutter_meshes_[mc_idx] = std::make_shared<CutterMesh>(cutter_dis, cutter_disp_col, start_ele_gid);

  do_mesh_intersection_ = true;
}

/*-------------------------------------------------------------*
 * Mark surfaces loaded into cut with background surfaces
 *--------------------------------------------------------------*/
void Cut::CutWizard::set_marked_condition_sides(
    // const int mc_idx,                                       //Not needed (for now?)
    Core::FE::Discretization& cutter_dis,
    // std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col,      //Not needed (for
    // now?)
    const int start_ele_gid)
{
  // Set the counter to the gid.
  //  -- Set ids in correspondence to this ID.
  //  -- Loop over the surface elements and find (if it exists) a corresponding side loaded into the
  //  cut
  //  ## WARNING: Not sure what happens if it doesn't find a surface?
  for (int lid = 0; lid < cutter_dis.num_my_row_elements(); ++lid)
  {
    Core::Elements::Element* cutter_dis_ele = cutter_dis.l_row_element(lid);

    const int numnode = cutter_dis_ele->num_node();
    const int* nodeids = cutter_dis_ele->node_ids();
    std::vector<int> node_ids_of_cutterele(nodeids, nodeids + numnode);

    const int eid = cutter_dis_ele->id();  // id of marked side based on the cutter discretization
    const int marked_sid = eid + start_ele_gid;  // id of marked side within the cut library

    // Get sidehandle to corresponding background surface discretization
    // -- if it exists!!!
    Cut::SideHandle* cut_sidehandle =
        intersection_->get_mesh_handle().get_side(node_ids_of_cutterele);

    if (cut_sidehandle != nullptr)
    {
      Cut::plain_side_set cut_sides;
      cut_sidehandle->collect_sides(cut_sides);

      // Set Id's and mark the sides in correspondence with the coupling manager object.
      for (Cut::plain_side_set::iterator it = cut_sides.begin(); it != cut_sides.end(); ++it)
      {
        (*it)->set_marked_side_properties(marked_sid, Cut::mark_and_create_boundarycells);
      }
    }
    else
      FOUR_C_THROW(
          "If we don't find a marked side it's not sure what happens... You are on your own!");
  }
}

/*========================================================================*/
//! @name main Cut call
/*========================================================================*/
/*-------------------------------------------------------------*
 * main Cut call
 *--------------------------------------------------------------*/
void Cut::CutWizard::cut(bool include_inner  //!< perform cut in the interior of the cutting mesh
)
{
  // safety checks if the cut is initialized correctly
  if (!safety_checks(false)) return;

  TEUCHOS_FUNC_TIME_MONITOR("Cut::CutWizard::Cut");

  if (myrank_ == 0 and screenoutput_) Core::IO::cout << "\nCut::CutWizard::Cut:" << Core::IO::endl;

  const double t_start = Teuchos::Time::wallTime();

  /* wirtz 08/14:
   * preprocessing: everything above should only be done once in a simulation;
   *                so it should be moved before the time loop into a preprocessing
   *                step
   * runtime:       everything below should be done in every Newton increment */

  //--------------------------------------
  // perform the actual cut, the intersection
  //--------------------------------------
  run_cut(include_inner);


  const double t_end = Teuchos::Time::wallTime() - t_start;
  if (myrank_ == 0 and screenoutput_)
  {
    Core::IO::cout << "\n\t\t\t\t\t\t\t... Success (" << t_end << " secs)\n" << Core::IO::endl;
  }

  //--------------------------------------
  // write statistics and output to screen and files
  //--------------------------------------
  output(include_inner);
}

/*-------------------------------------------------------------*
 * prepare the cut, add background elements and cutting sides
 *--------------------------------------------------------------*/
void Cut::CutWizard::prepare()
{
  // safety checks if the cut is initialized correctly
  if (!safety_checks(true)) return;

  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 1/6 --- Cut_Initialize");

  const double t_start = Teuchos::Time::wallTime();

  if (myrank_ == 0 and screenoutput_)
    Core::IO::cout << "\nCut::CutWizard::Prepare:" << Core::IO::endl;

  if (myrank_ == 0 and screenoutput_) Core::IO::cout << "\n\t * 1/6 Cut_Initialize ...";

  // fill the cutwizard cw with information:
  // build up the mesh_ (normal background mesh) and the cut_mesh_ (cutter mesh) created by the
  // meshhandle: REMARK: DO NOT CHANGE THE ORDER of 1. and 2.
  // 1. Add CutSides (sides of the cutterdiscretization)
  //      -> Update the current position of all cutter-nodes dependent on displacement idispcol
  // 2. Add Elements (elements of the background discretization)

  // Ordering is very important because first we add all cut sides, and create a bounding box which
  // contains all the cut sides Then, when adding elements from background discret, only the
  // elements that intersect this bounding box are added Changing the order would render in problems
  // when all bg-elements on one proc are within the structure, then the bb around the bg-mesh on
  // this proc has no intersection with an an bb around an side element


  // 1. Add CutSides (possible sides of the cutter-discretization and a possible level-set side)
  add_cutting_sides();

  // 2. Add background elements dependent on bounding box created by the CutSides in 1.
  add_background_elements();


  // build the static search tree for the collision detection in the self cut
  intersection_->build_self_cut_tree();

  // build the static search tree for the collision detection
  intersection_->build_static_search_tree();

  const double t_mid = Teuchos::Time::wallTime() - t_start;
  if (myrank_ == 0 and screenoutput_)
  {
    Core::IO::cout << "\t\t\t... Success (" << t_mid << " secs)" << Core::IO::endl;
  }

  is_cut_prepare_performed_ = true;
}

/*-------------------------------------------------------------*
 * add all cutting sides (mesh and level-set sides)
 *--------------------------------------------------------------*/
void Cut::CutWizard::add_cutting_sides()
{
  // add all mesh cutting sides
  if (do_mesh_intersection_) add_mesh_cutting_side();

  // add a new level-set side
  if (do_levelset_intersection_) add_ls_cutting_side();
}

/*-------------------------------------------------------------*
 * add level-set cutting side
 *--------------------------------------------------------------*/
void Cut::CutWizard::add_ls_cutting_side()
{
  // add a new level-set side
  intersection_->add_level_set_side(level_set_sid_);
}


/*-------------------------------------------------------------*
 * add all mesh-cutting sides of all cutting discretizations
 *--------------------------------------------------------------*/
void Cut::CutWizard::add_mesh_cutting_side()
{
  // loop all mesh coupling objects
  for (std::map<int, std::shared_ptr<CutterMesh>>::iterator it = cutter_meshes_.begin();
      it != cutter_meshes_.end(); it++)
  {
    std::shared_ptr<CutterMesh> cutter_mesh = it->second;

    add_mesh_cutting_side(
        cutter_mesh->cutterdis_, cutter_mesh->cutter_disp_col_, cutter_mesh->start_ele_gid_);
  }
}

/*
 * Helper function to evaluate the position on a nurbs9 face, so it can be represented as a quad9
 * cell
 */
void evaluate_position_on_nurbs9(Core::Elements::Element* element,
    Core::LinAlg::SerialDenseMatrix& element_current_position,
    std::shared_ptr<Core::FE::Discretization>& cutterdis,
    const Core::LinAlg::Vector<double>& cutter_disp_col)
{
  // Initialize the information needed for NURBS elements
  Core::LinAlg::Matrix<9, 1, double> weights(Core::LinAlg::Initialization::zero);
  std::vector<Core::LinAlg::SerialDenseVector> myknots(2);
  std::vector<Core::LinAlg::SerialDenseVector> mypknots(3);

  const int num_nodes = Core::FE::num_nodes<Core::FE::CellType::nurbs9>;
  const int nurbs_dim = Core::FE::dim<Core::FE::CellType::nurbs9>;
  const int prob_dim = 3;

  // The element pointer has to be a face element.
  auto face_element = dynamic_cast<const Core::Elements::FaceElement*>(element);
  FOUR_C_ASSERT_ALWAYS(
      face_element, "Element given in evaluate_position_on_nurbs9 has to be a face element.");

  // Factor for surface orientation.
  double normalfac = 1.0;

  // Get the knots and weights for this element.
  const bool zero_size = Core::FE::Nurbs::get_knot_vector_and_weights_for_nurbs_boundary(element,
      face_element->face_master_number(), face_element->parent_element_id(), *cutterdis.get(),
      mypknots, myknots, weights, normalfac);
  if (zero_size)
    FOUR_C_THROW("get_knot_vector_and_weights_for_nurbs_boundary has to return a non zero size.");

  // Get the position of the control points in the reference configuration and their
  // displacements.
  std::vector<int> lm;
  std::vector<double> mydisp;
  Core::LinAlg::Matrix<num_nodes * prob_dim, 1, double> ref_pos_controlpoints;
  Core::LinAlg::Matrix<num_nodes * prob_dim, 1, double> displacement_controlpoints;

  for (unsigned int i_controlpoint = 0; i_controlpoint < (unsigned int)element->num_node();
      ++i_controlpoint)
  {
    const Core::Nodes::Node* controlpoint = element->nodes()[i_controlpoint];

    // Obtain the reference position of the control point
    for (int i_dim = 0; i_dim < prob_dim; ++i_dim)
      ref_pos_controlpoints(prob_dim * i_controlpoint + i_dim) = controlpoint->x()[i_dim];

    lm.clear();
    mydisp.clear();
    cutterdis->dof(controlpoint, lm);
    mydisp = Core::FE::extract_values(cutter_disp_col, lm);

    // Obtain the displacements on control points
    for (int i_dim = 0; i_dim < prob_dim; ++i_dim)
      displacement_controlpoints(prob_dim * i_controlpoint + i_dim) = mydisp[i_dim];
  }

  // Evaluate the NURBS shape functions to obtain the reference nodal positions and displacement
  Core::LinAlg::Matrix<prob_dim, 1, double> nodal_position;
  Core::LinAlg::Matrix<prob_dim, 1, double> nodal_displacement;
  Core::LinAlg::Matrix<prob_dim, 1, double> current_nodal_position;
  for (unsigned int i_node = 0; i_node < num_nodes; i_node++)
  {
    Core::LinAlg::Matrix<nurbs_dim, 1, double> xi;
    for (unsigned int i = 0; i < nurbs_dim; i++)
      xi(i) = Core::FE::eleNodeNumbering_quad9_nodes_reference[i_node][i];

    nodal_position = Core::FE::Nurbs::eval_nurbs_interpolation<num_nodes, nurbs_dim, prob_dim>(
        ref_pos_controlpoints, xi, weights, myknots, element->shape());

    nodal_displacement = Core::FE::Nurbs::eval_nurbs_interpolation<num_nodes, nurbs_dim, prob_dim>(
        displacement_controlpoints, xi, weights, myknots, element->shape());

    for (unsigned int i_dim = 0; i_dim < prob_dim; i_dim++)
      current_nodal_position(i_dim) = nodal_position(i_dim) + nodal_displacement(i_dim);

    // Save current nodal position in element_current_position
    std::copy(current_nodal_position.data(), current_nodal_position.data() + 3,
        &element_current_position(0, i_node));
  }
}

/*
 * Helper function to evaluate the position of a classic Lagrange element
 */
void evaluate_position_on_lagrange_element(Core::Elements::Element* element,
    Core::LinAlg::SerialDenseMatrix& element_current_position, Core::FE::Discretization& cutterdis,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& cutter_disp_col)
{
  std::vector<int> lm;
  std::vector<double> mydisp;
  const int numnode = element->num_node();
  Core::Nodes::Node** nodes = element->nodes();

  for (int i_node = 0; i_node < numnode; ++i_node)
  {
    Core::Nodes::Node& node = *nodes[i_node];

    lm.clear();
    mydisp.clear();
    cutterdis.dof(&node, lm);

    // Initialize the current nodal position with its reference position
    Core::LinAlg::Matrix<3, 1> current_nodal_position(node.x().data());

    if (cutter_disp_col != nullptr)
    {
      if (lm.size() == 3)  // case for BELE3 boundary elements
      {
        mydisp = Core::FE::extract_values(*cutter_disp_col, lm);
      }
      else if (lm.size() == 4)  // case for BELE3_4 boundary elements
      {
        // copy the first three entries for the displacement, the fourth entry should be zero if
        // BELE3_4 is used for cutdis instead of BELE3
        std::vector<int> lm_red;  // reduced local map
        lm_red.clear();
        for (int k = 0; k < 3; k++) lm_red.push_back(lm[k]);

        mydisp = Core::FE::extract_values(*cutter_disp_col, lm_red);
      }
      else
        FOUR_C_THROW("wrong number of dofs for node {}", lm.size());

      Core::LinAlg::Matrix<3, 1> disp(mydisp.data(), true);

      // update the reference nodal position of cutter node for current time step (update with
      // displacement)
      current_nodal_position.update(1, disp, 1);
    }

    // Save the current nodal position of i_node in element_current_position
    std::copy(current_nodal_position.data(), current_nodal_position.data() + 3,
        &element_current_position(0, i_node));
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::CutWizard::add_mesh_cutting_side(std::shared_ptr<Core::FE::Discretization> cutterdis,
    std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col,
    const int start_ele_gid  ///< mesh coupling index
)
{
  if (cutterdis == nullptr)
    FOUR_C_THROW("cannot add mesh cutting sides for invalid cutter discretization!");

  std::vector<int> lm;
  std::vector<double> mydisp;
  int numcutelements = cutterdis->num_my_col_elements();


  for (int lid = 0; lid < numcutelements; ++lid)
  {
    Core::Elements::Element* element = cutterdis->l_col_element(lid);

    const int numnode = element->num_node();
    Core::LinAlg::SerialDenseMatrix element_current_position(3, numnode);

    // obtain the current position of the cutting element depending on its cell type
    if (element->shape() == Core::FE::CellType::nurbs9)
      evaluate_position_on_nurbs9(element, element_current_position, cutterdis, *cutter_disp_col);
    else if (element->shape() == Core::FE::CellType::nurbs4)
      FOUR_C_THROW(
          "This type of NURBS element is not implemented to be used as a cutter discretization.");
    else
      evaluate_position_on_lagrange_element(
          element, element_current_position, *cutterdis, cutter_disp_col);

    // Determine the cutting side id. For embedded mesh cases, we use the id of the parent element
    // of the cutter discretization. Otherwise, we use directly the id of the cutter discretization
    int sid;
    std::vector<Core::Conditions::Condition*> embeddedmesh_cond;
    cutterdis->get_condition("EmbeddedMeshSolidSurfCoupling", embeddedmesh_cond);

    if (embeddedmesh_cond.size() == 0)
      sid = element->id() + start_ele_gid;
    else
    {
      Core::Elements::FaceElement* face_ele = dynamic_cast<Core::Elements::FaceElement*>(element);
      if (!face_ele) FOUR_C_THROW("Cast to FaceElement failed!");

      sid = face_ele->parent_element_id();
    }

    // add the side of the cutter-discretization
    add_mesh_cutting_side(0, element, element_current_position, sid);
  }
}

/*-------------------------------------------------------------*
 * prepare the cut, add background elements and cutting sides
 *--------------------------------------------------------------*/
void Cut::CutWizard::add_mesh_cutting_side(int mi, Core::Elements::Element* ele,
    const Core::LinAlg::SerialDenseMatrix& element_current_position, int sid)
{
  const int numnode = ele->num_node();
  const int* nodeids = ele->node_ids();

  std::vector<int> nids(numnode);
  if (ele->shape() != Core::FE::CellType::nurbs9)
  {
    for (int iter = 0; iter < numnode; ++iter) nids[iter] = nodeids[iter];

    intersection_->add_mesh_cutting_side(sid, nids, element_current_position, ele->shape(), mi);
  }
  else
  {
    // For element_current_position, new nodal ids are assigned to them.
    // These nodal ids are generated by looking to the last id number
    // in the mesh and start counting from there
    auto cutmesh_nodes = intersection_->MeshIntersection::cut_mesh(0).nodes();

    // Find the highest value of the ids of the nodes in the cutmesh
    auto max = std::max_element(std::begin(cutmesh_nodes), std::end(cutmesh_nodes),
        [](const auto& p1, const auto& p2) { return p1.second->id() < p2.second->id(); });

    // Fill out the vector of nodes ids by counting from the maximum id value found
    // in the cut mesh
    int start_id = max->first + 1;
    std::iota(std::begin(nids), std::end(nids), start_id);

    intersection_->add_mesh_cutting_side(sid, nids, element_current_position, ele->shape(), mi);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::CutWizard::add_background_elements()
{
  // Check if the background mesh has an embedded mesh coupling condition
  std::vector<Core::Conditions::Condition*> embeddedmesh_cond;
  back_mesh_->get().get_condition("EmbeddedMeshSolidVolBackground", embeddedmesh_cond);

  if (!embeddedmesh_cond.size())
    add_background_elements_general();
  else
    add_background_elements_embeddedmesh(embeddedmesh_cond);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::CutWizard::add_background_elements_general()
{
  // vector with nodal level-set values
  std::vector<double> myphinp;

  // Loop over all Elements to find cut elements and add them to the LevelsetIntersection class
  // Brute force method.
  int numelements = back_mesh_->num_my_col_elements();

  for (int lid = 0; lid < numelements; ++lid)
  {
    const Core::Elements::Element* element = back_mesh_->l_col_element(lid);

    Core::LinAlg::SerialDenseMatrix current_element_position =
        get_current_element_position(element);

    if (back_mesh_->is_level_set())
    {
      myphinp.clear();

      Core::FE::extract_my_node_based_values(element, myphinp, back_mesh_->back_level_set_col());
      add_element(element, current_element_position, myphinp.data(), lsv_only_plus_domain_);
    }
    else
    {
      add_element(element, current_element_position, nullptr);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::CutWizard::add_background_elements_embeddedmesh(
    std::vector<Core::Conditions::Condition*>& embeddedmesh_cond)
{
  // Loop over all Elements to find background elements for embedded mesh condition
  int numelements = back_mesh_->num_my_col_elements();

  for (int lid = 0; lid < numelements; ++lid)
  {
    const Core::Elements::Element* element = back_mesh_->l_col_element(lid);
    Core::LinAlg::SerialDenseMatrix element_position = get_current_element_position(element);

    // Checking if an element is part of the background mesh
    if (embeddedmesh_cond.size())
    {
      for (unsigned int i = 0; i < embeddedmesh_cond.size(); ++i)
      {
        for (int num_node = 0; num_node < element->num_node(); ++num_node)
        {
          if (embeddedmesh_cond[i]->contains_node(element->nodes()[num_node]->id()))
            add_element(element, element_position, nullptr);
        }
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Cut::CutWizard::get_current_element_position(
    const Core::Elements::Element* element)
{
  std::vector<int> lm;
  std::vector<double> mydisp;

  const int numnode = element->num_node();
  const Core::Nodes::Node* const* nodes = element->nodes();

  Core::LinAlg::SerialDenseMatrix current_element_position;
  current_element_position.shape(3, numnode);

  for (int i = 0; i < numnode; ++i)
  {
    const Core::Nodes::Node& node = *nodes[i];

    // Initialize node_current_position with its reference position
    Core::LinAlg::Matrix<3, 1> node_current_position(node.x().data());

    if (back_mesh_->is_back_disp())
    {
      lm.clear();
      mydisp.clear();

      FOUR_C_ASSERT(global_dof_indices_, "global_dof_indices callback not set.");
      global_dof_indices_(node, lm);

      FOUR_C_ASSERT(lm.size() == 4, "Wrong number of dofs for node {}", lm.size());

      // copy the first three entries for the displacement, the fourth entry an all others
      std::vector<int> lm_red;  // reduced local map
      lm_red.clear();
      for (int k = 0; k < 3; k++) lm_red.push_back(lm[k]);

      mydisp = Core::FE::extract_values(back_mesh_->back_disp_col(), lm_red);

      if (mydisp.size() != 3) FOUR_C_THROW("we need 3 displacements here");

      Core::LinAlg::Matrix<3, 1> node_displacement(mydisp.data(), true);

      // update position of cutter node for current time step (update with displacement)
      node_current_position.update(1, node_displacement, 1);
    }
    std::copy(node_current_position.data(), node_current_position.data() + 3,
        &current_element_position(0, i));
  }

  return current_element_position;
}


/*-------------------------------------------------------------*
 * Add this background mesh element to the intersection class
 *--------------------------------------------------------------*/
void Cut::CutWizard::add_element(const Core::Elements::Element* ele,
    const Core::LinAlg::SerialDenseMatrix& xyze, double* myphinp, bool lsv_only_plus_domain)
{
  const int numnode = ele->num_node();
  const int* nodeids = ele->node_ids();

  std::vector<int> nids(nodeids, nodeids + numnode);

  // If include_inner == false then add elements with negative level set values to discretization.
  intersection_->add_element(ele->id(), nids, xyze, ele->shape(), myphinp, lsv_only_plus_domain);
}

/*------------------------------------------------------------------------------------------------*
 * perform the actual cut, the intersection
 *------------------------------------------------------------------------------------------------*/
void Cut::CutWizard::run_cut(
    bool include_inner  //!< perform cut in the interior of the cutting mesh
)
{
  // just for time measurement
  Core::Communication::barrier(comm_);

  if (do_mesh_intersection_)
  {
    //----------------------------------------------------------
    // Selfcut (2/6 Cut_SelfCut)
    {
      const double t_start = Teuchos::Time::wallTime();

      // cut the mesh
      intersection_->cut_self_cut(include_inner, screenoutput_);

      // just for time measurement
      Core::Communication::barrier(comm_);

      const double t_diff = Teuchos::Time::wallTime() - t_start;
      if (myrank_ == 0 and screenoutput_)
        Core::IO::cout << "\t\t\t... Success (" << t_diff << " secs)" << Core::IO::endl;
    }
    //----------------------------------------------------------
    // Cut Part I: Collision Detection (3/6 cut_collision_detection)
    {
      const double t_start = Teuchos::Time::wallTime();

      // cut the mesh
      intersection_->cut_collision_detection(include_inner, screenoutput_);

      // just for time measurement
      Core::Communication::barrier(comm_);

      const double t_diff = Teuchos::Time::wallTime() - t_start;
      if (myrank_ == 0 and screenoutput_)
        Core::IO::cout << "\t\t... Success (" << t_diff << " secs)" << Core::IO::endl;
    }
  }

  //----------------------------------------------------------
  // Cut Part II: intersection (4/6 Cut_Intersection)
  {
    const double t_start = Teuchos::Time::wallTime();

    intersection_->cut(screenoutput_);

    // just for time measurement
    Core::Communication::barrier(comm_);

    const double t_diff = Teuchos::Time::wallTime() - t_start;
    if (myrank_ == 0 and screenoutput_)
      Core::IO::cout << "\t\t\t... Success (" << t_diff << " secs)" << Core::IO::endl;
  }

  //----------------------------------------------------------
  // Cut Part III & IV: Element Selection and DOF-Set Management (5/6 cut_positions_dofsets)
  {
    const double t_start = Teuchos::Time::wallTime();

    find_position_dof_sets(include_inner);

    // just for time measurement
    Core::Communication::barrier(comm_);

    const double t_diff = Teuchos::Time::wallTime() - t_start;
    if (myrank_ == 0 and screenoutput_)
      Core::IO::cout << "\t... Success (" << t_diff << " secs)" << Core::IO::endl;
  }

  //----------------------------------------------------------
  // Cut Part V & VI: Polyhedra Integration and Boundary Tessellation (6/6 Cut_Finalize)
  {
    const double t_start = Teuchos::Time::wallTime();

    // perform tessellation or moment fitting on the mesh
    intersection_->cut_finalize(
        include_inner, v_cellgausstype_, b_cellgausstype_, tetcellsonly_, screenoutput_);

    // just for time measurement
    Core::Communication::barrier(comm_);

    const double t_diff = Teuchos::Time::wallTime() - t_start;
    if (myrank_ == 0 and screenoutput_)
      Core::IO::cout << "\t\t\t... Success (" << t_diff << " secs)" << Core::IO::endl;
  }

  is_cut_performed_ = true;
}


/*------------------------------------------------------------------------------------------------*
 * routine for finding node positions and computing vc dofsets in a parallel way
 *------------------------------------------------------------------------------------------------*/
void Cut::CutWizard::find_position_dof_sets(bool include_inner)
{
  Core::Communication::barrier(comm_);

  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 5/6 --- cut_positions_dofsets (parallel)");

  if (myrank_ == 0 and screenoutput_)
    Core::IO::cout << "\t * 5/6 cut_positions_dofsets (parallel) ...";

  //  const double t_start = Teuchos::Time::wallTime();

  //----------------------------------------------------------

  if (intersection_->get_options().find_positions())
  {
    Cut::Mesh& m = intersection_->normal_mesh();

    bool communicate = (Core::Communication::num_mpi_ranks(comm_) > 1);

    // create a parallel Cut object for the current background mesh to communicate missing data
    std::shared_ptr<Cut::Parallel> cut_parallel = nullptr;

    if (communicate)
    {
      cut_parallel = std::make_shared<Cut::Parallel>(back_mesh_->get_ptr(), m, *intersection_);
    }

    // find inside and outside positions of nodes
    // first for mesh cut and distribute data in parallel, after that do the same for the level-set
    // cut

    //--------------------------------------------
    // first, set the position for the mesh cut
    if (do_mesh_intersection_)
    {
      m.find_node_positions();

      if (communicate) cut_parallel->communicate_node_positions();
    }

    //--------------------------------------------
    // second, set the position for the level-set cut (no parallel communication necessary)
    if (do_levelset_intersection_)
    {
      m.find_ls_node_positions();
    }

    if (do_mesh_intersection_)
    {
      m.find_facet_positions();
    }

    //--------------------------------------------
    Core::Communication::barrier(comm_);

    // find number and connection of dofsets at nodes from cut volumes
    intersection_->create_nodal_dof_set(include_inner, back_mesh_->get());

    if (communicate) cut_parallel->communicate_node_dof_set_numbers(include_inner);
  }
}


bool Cut::CutWizard::safety_checks(bool is_prepare_cut_call)
{
  if (!is_set_options_)
    FOUR_C_THROW("You have to call SetOptions() before you can use the CutWizard");

  if (!is_prepare_cut_call and !is_cut_prepare_performed_)
    FOUR_C_THROW("You have to call PrepareCut() before you can call the Cut-routine");

  if (!do_mesh_intersection_ and !do_levelset_intersection_)
  {
    if (myrank_ == 0 and is_prepare_cut_call)
    {
      std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n";
      std::cout << "WARNING: No mesh intersection and no level-set intersection! \n"
                << "         Why do you call the CUT-library?\n";
      std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n";
    }
    return false;
  }

  return true;
}

/*------------------------------------------------------------------------------------------------*
 * write statistics and output to screen and files
 *------------------------------------------------------------------------------------------------*/
void Cut::CutWizard::output(bool include_inner)
{
  if (gmsh_output_) dump_gmsh_num_dof_sets(include_inner);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  print_cell_stats();
#endif

  if (gmsh_output_)
  {
    dump_gmsh_integration_cells();
    dump_gmsh_volume_cells(include_inner);
  }
}


/*------------------------------------------------------------------------------------------------*
 * Print the number of volumecells and boundarycells generated over the whole mesh during the cut *
 *------------------------------------------------------------------------------------------------*/
void Cut::CutWizard::print_cell_stats() { intersection_->print_cell_stats(); }


/*------------------------------------------------------------------------------------------------*
 * Write the DOF details of the nodes                                                             *
 *------------------------------------------------------------------------------------------------*/
void Cut::CutWizard::dump_gmsh_num_dof_sets(bool include_inner)
{
  intersection_->dump_gmsh_num_dof_sets(output_prefix_, include_inner, back_mesh_->get());
}


/*------------------------------------------------------------------------------------------------*
 * Write volumecell output in GMSH format throughout the domain                                   *
 *------------------------------------------------------------------------------------------------*/
void Cut::CutWizard::dump_gmsh_volume_cells(bool include_inner)
{
  std::stringstream str;
  str << output_prefix_ << ".CUT_volumecells." << myrank_ << ".pos";
  intersection_->dump_gmsh_volume_cells(str.str(), include_inner);
}

/*------------------------------------------------------------------------------------------------*
 * Write the integrationcells and boundarycells in GMSH format throughout the domain              *
 *------------------------------------------------------------------------------------------------*/
void Cut::CutWizard::dump_gmsh_integration_cells()
{
  std::stringstream str;
  str << output_prefix_ << ".CUT_integrationcells." << myrank_ << ".pos";
  intersection_->dump_gmsh_integration_cells(str.str());
}


/*========================================================================*/
//! @name Getters
/*========================================================================*/

Cut::SideHandle* Cut::CutWizard::get_side(std::vector<int>& nodeids)
{
  return intersection_->get_side(nodeids);
}

Cut::SideHandle* Cut::CutWizard::get_side(int sid) { return intersection_->get_side(sid); }

Cut::SideHandle* Cut::CutWizard::get_cut_side(int sid)
{
  if (intersection_ == nullptr) FOUR_C_THROW("No intersection object available!");
  std::shared_ptr<Cut::MeshIntersection> meshintersection =
      std::dynamic_pointer_cast<Cut::MeshIntersection>(intersection_);
  if (meshintersection == nullptr) FOUR_C_THROW("Cast to MeshIntersection failed!");
  return meshintersection->get_cut_side(sid);
}

Cut::ElementHandle* Cut::CutWizard::get_element(const int eleid) const
{
  return intersection_->get_element(eleid);
}

Cut::ElementHandle* Cut::CutWizard::get_element(const Core::Elements::Element* ele) const
{
  return get_element(ele->id());
}

Cut::Node* Cut::CutWizard::get_node(int nid) { return intersection_->get_node(nid); }

Cut::SideHandle* Cut::CutWizard::get_mesh_cutting_side(int sid, int mi)
{
  return intersection_->get_cut_side(sid, mi);
}

bool Cut::CutWizard::has_ls_cutting_side(int sid)
{
  return intersection_->has_ls_cutting_side(sid);
}

void Cut::CutWizard::update_boundary_cell_coords(
    std::shared_ptr<Core::FE::Discretization> cutterdis,
    std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col, const int start_ele_gid)
{
  if (cutterdis == nullptr)
    FOUR_C_THROW("cannot add mesh cutting sides for invalid cutter discretization!");

  std::vector<int> lm;
  std::vector<double> mydisp;

  int numcutelements = cutterdis->num_my_col_elements();


  for (int lid = 0; lid < numcutelements; ++lid)
  {
    Core::Elements::Element* element = cutterdis->l_col_element(lid);

    const int numnode = element->num_node();
    Core::Nodes::Node** nodes = element->nodes();

    Core::LinAlg::SerialDenseMatrix xyze(3, numnode);
    std::vector<int> dofs;

    for (int i = 0; i < numnode; ++i)
    {
      Core::Nodes::Node& node = *nodes[i];

      lm.clear();
      mydisp.clear();

      Core::LinAlg::Matrix<3, 1> x(node.x().data());

      cutterdis->dof(&node, lm);

      dofs.push_back(lm[0]);
      dofs.push_back(lm[1]);
      dofs.push_back(lm[2]);

      if (cutter_disp_col != nullptr)
      {
        if (lm.size() == 3)  // case for BELE3 boundary elements
        {
          mydisp = Core::FE::extract_values(*cutter_disp_col, lm);
        }
        else if (lm.size() == 4)  // case for BELE3_4 boundary elements
        {
          // copy the first three entries for the displacement, the fourth entry should be zero if
          // BELE3_4 is used for cutdis instead of BELE3
          std::vector<int> lm_red;  // reduced local map
          lm_red.clear();
          for (int k = 0; k < 3; k++) lm_red.push_back(lm[k]);

          mydisp = Core::FE::extract_values(*cutter_disp_col, lm_red);
        }
        else
          FOUR_C_THROW("wrong number of dofs for node {}", lm.size());

        if (mydisp.size() != 3) FOUR_C_THROW("we need 3 displacements here");

        Core::LinAlg::Matrix<3, 1> disp(mydisp.data(), true);

        // update x-position of cutter node for current time step (update with displacement)
        x.update(1, disp, 1);
      }
      std::copy(x.data(), x.data() + 3, &xyze(0, i));
    }

    Cut::SideHandle* sh = get_cut_side(element->id() + start_ele_gid);
    if (!sh) FOUR_C_THROW("couldn't get sidehandle!");

    if (xyze.numCols() == 4 && sh->shape() == Core::FE::CellType::quad4)
    {
      Core::LinAlg::Matrix<3, 4> XYZE(xyze.values(), true);

      Cut::plain_side_set sides;
      sh->collect_sides(sides);

      for (Cut::plain_side_set::iterator sit = sides.begin(); sit != sides.end(); ++sit)
      {
        Cut::Side* side = *sit;

        Cut::plain_boundarycell_set bcs;
        side->get_boundary_cells(bcs);

        for (Cut::plain_boundarycell_set::iterator bit = bcs.begin(); bit != bcs.end(); ++bit)
        {
          Cut::BoundaryCell* bc = *bit;

          for (std::size_t bcpoint = 0; bcpoint < bc->points().size(); ++bcpoint)
          {
            // get local coord on sidehandle
            Core::LinAlg::Matrix<2, 1> xsi = sh->local_coordinates(bc->points()[bcpoint]);

            // eval shape function
            Core::LinAlg::Matrix<4, 1> funct;
            Core::FE::shape_function_2d(funct, xsi(0, 0), xsi(1, 0), sh->shape());

            Core::LinAlg::Matrix<3, 1> newpos(Core::LinAlg::Initialization::zero);
            newpos.multiply(XYZE, funct);
            bc->reset_pos(bcpoint, newpos);
          }
        }
      }
    }
    else
      FOUR_C_THROW("Shape not implemented!");
  }
}

int Cut::CutWizard::get_bc_cubaturedegree() const
{
  if (is_set_options_)
    return intersection_->get_options().bc_cubaturedegree();
  else
    FOUR_C_THROW("get_bc_cubaturedegree: Options are not set!");
  return -1;  // dummy to make compiler happy :)
}

bool Cut::CutWizard::do_inside_cells_have_physical_meaning()
{
  return intersection_->get_options().do_inside_cells_have_physical_meaning();
}

std::shared_ptr<Cut::CombIntersection> Cut::CutWizard::get_intersection()
{
  if (!intersection_) FOUR_C_THROW("nullptr pointer!");

  return intersection_;
}

void Cut::CutWizard::check_if_mesh_intersection_and_cut()
{
  if (!do_mesh_intersection_ or !is_cut_performed_)
  {
    FOUR_C_THROW(
        "Not possible to create coupling pairs, first perform cut using a mesh intersection.");
  }
}

FOUR_C_NAMESPACE_CLOSE
