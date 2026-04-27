// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_edgestab.hpp"

#include "4C_cut_cutwizard.hpp"
#include "4C_cut_elementhandle.hpp"
#include "4C_cut_sidehandle.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_intfaces_calc.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_material_base.hpp"
#include "4C_xfem_utils.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  prepares edge based stabilization and ghost penalty in case of XFEM |
 |  and calls evaluate routine                             schott 03/12 |
 *----------------------------------------------------------------------*/
void XFEM::XfemEdgeStab::evaluate_edge_stab_ghost_penalty(
    Teuchos::ParameterList& eleparams,                           ///< element parameter list
    std::shared_ptr<Core::FE::Discretization> discret,           ///< discretization
    Discret::Elements::FluidIntFace* faceele,                    ///< face element
    std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix,    ///< systemmatrix
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector,  ///< systemvector
    Cut::CutWizard& wizard,                                      ///< cut wizard
    bool include_inner,        ///< stabilize also facets with inside position
    bool include_inner_faces,  ///< stabilize also faces with inside position if possible
    bool gmsh_eos_out          ///< stabilization gmsh output
)
{
  //====================================================================================================
  // implementation of edge-based fluid stabilization and ghost penalty
  //====================================================================================================

  // EDGE-based fluid stabilization and EDGE-based Ghost-penalty stabilization
  // REMARK: the current implementation of edge-based stabilization is based on the
  // DiscretizationXFEM extension
  //         using additional information about faces between two volume elements
  // * fluid stabilization has to be integrated for all internal faces
  // * ghost penalty has to be integrated if there is at least one cut element
  //   (because all faces between two elements for that at least one element is cut by the interface
  //   has to be stabilized) NOTE: the limit case that a cut side just touches a node or if the cut
  //   side touches an element side completely, the check
  //         e->IsIntersected() returns false and we do not stabilize the face.
  //         This avoids over-stabilization as it e.g. does not switch on the ghost-penalties for
  //         standard FEM situations. by a more appropriate check which tells you if the neighboring
  //         volumecell is equal to the element itself.
  //   NOTE: it might be helpful and might lead to better results when weighting ghost-penalties by
  //   e.g. volume-fractions.
  //         In that case it still has to be guaranteed not to loose coercivity. To guarantee weak
  //         consistency the scalings have to be bounded by h. Such scalings are not available yet.
  //
  // we distinguish different stabilization cases
  //  1. the master element and slave element (connected via current side)
  //     do not have an elementhandle (standard fluid case)
  //     -> standard fluid stabilization
  //                               => EOS(fluid): YES         GHOST-PENALTY: NO
  //  2. element handles for both parent elements
  //     -> stabilization for each facet and corresponding volumecells of parent elements
  //                               => EOS(fluid): YES         GHOST-PENALTY: Yes (if at least one
  //                               parent element is cut)
  //                                                                         NO  (if both parent
  //                                                                         elements are uncut)
  //  3. just one elementhandle available (at limit of bounding box)
  //     -> stabilization for each facet and corresponding volumecells of parent elements
  //                               => EOS(fluid): YES         GHOST-PENALTY: Yes (if at least one
  //                               parent element is cut)
  //                                                                         NO  (if both parent
  //                                                                         elements are uncut)


  std::shared_ptr<Core::FE::DiscretizationFaces> xdiscret =
      std::dynamic_pointer_cast<Core::FE::DiscretizationFaces>(discret);
  if (xdiscret == nullptr)
    FOUR_C_THROW(
        "Failed to cast Core::FE::Discretization to "
        "Core::FE::DiscretizationFaces.");


  // get the parent fluid elements
  Discret::Elements::Fluid* p_master = faceele->parent_target_element();
  Discret::Elements::Fluid* p_slave = faceele->parent_source_element();

  // get corresponding element handles if available
  Cut::ElementHandle* p_master_handle = wizard.get_element(p_master);
  Cut::ElementHandle* p_slave_handle = wizard.get_element(p_slave);

  size_t p_master_numnode = p_master->num_node();
  size_t p_slave_numnode = p_slave->num_node();

  // get the parent element
  int p_master_id = p_master->id();

  std::vector<int> nds_master;
  nds_master.reserve(p_master_numnode);

  std::vector<int> nds_slave;
  nds_slave.reserve(p_slave_numnode);

  XFEM::FaceType face_type;

  int num_edgestab = 0;      // how often to stabilize this face for edgebased stabilizations
  int num_ghostpenalty = 0;  // how often to stabilize this face for ghost penalty stabilizations


  // Provide material at both sides:
  std::shared_ptr<Core::Mat::Material> matptr_m;
  std::shared_ptr<Core::Mat::Material> matptr_s;
  matptr_m = p_master->material();
  matptr_s = p_slave->material();

  //------------------------------------------------------------------------------
  // simplest case: no element handles for both parent elements
  // two uncut elements / standard fluid case
  // problems cut with levelset will not enter here!
  //------------------------------------------------------------------------------
  if (p_master_handle == nullptr and p_slave_handle == nullptr)
  {
    num_edgestab++;

    if (matptr_m->material_type() == Core::Materials::m_matlist)
      FOUR_C_THROW("The edgebased algo can not handle matlist at the moment, for this entry!");

    face_type = XFEM::face_type_std;

    {
      TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: create nds");

      for (size_t i = 0; i < p_master_numnode; i++) nds_master.push_back(0);

      for (size_t i = 0; i < p_slave_numnode; i++) nds_slave.push_back(0);
    }

    //--------------------------------------------------------------------------------------------

    // call evaluate and assemble routine
    assemble_edge_stab_ghost_penalty(eleparams, face_type, faceele, matptr_m, matptr_s, nds_master,
        nds_slave, *xdiscret, systemmatrix, systemvector);

    //--------------------------------------------------------------------------------------------
  }
  //------------------------------------------------------------------------------
  // second case: element handles for both parent elements
  // two elements that are maybe cut
  //------------------------------------------------------------------------------
  else if (p_master_handle != nullptr and p_slave_handle != nullptr)
  {
    // linear elements
    if (p_master->shape() == Core::FE::CellType::hex8 or
        p_master->shape() == Core::FE::CellType::tet4 or
        p_master->shape() == Core::FE::CellType::wedge6 or
        p_master->shape() == Core::FE::CellType::pyramid5)
    {
      Cut::SideHandle* side = get_face(faceele, wizard);

      //-------------------------------- loop facets of this side -----------------------------
      // facet of current side
      std::vector<Cut::Facet*> facets;
      side->facets(facets);

      if (facets.size() == 0)
        FOUR_C_THROW("there is no facet between two elements with elementhandle!");

      // each facet should have 2 volumecells
      for (std::vector<Cut::Facet*>::const_iterator f = facets.begin(); f != facets.end(); f++)
      {
        if ((*f)->position() == Cut::Point::outside or
            ((*f)->position() == Cut::Point::inside and (include_inner || include_inner_faces)))
        {
          Cut::plain_volumecell_set vcs = (*f)->cells();

          // how many volumecells found?
          if (vcs.size() ==
              2)  // standard XFEM case (facet between two vcs of two neighbouring cut elements
          {
            Cut::plain_volumecell_set::iterator vc_it = vcs.begin();

            Cut::VolumeCell* vc1 = *(vc_it);
            vc_it++;
            Cut::VolumeCell* vc2 = *(vc_it);


            // get the parent element
            int vc_ele1_id = vc1->parent_element()->id();
            int vc_ele2_id = vc2->parent_element()->id();

            bool all_dofs = (facets.size() == 1 && include_inner_faces);
            if ((*f)->position() == Cut::Point::outside || include_inner)
            {
              //------------------------ create nodal dof sets
              TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: create nds");
              // which element is the parent element
              if (vc_ele1_id == p_master_id)
              {
                nds_master = vc1->nodal_dof_set();
                nds_slave = vc2->nodal_dof_set();
              }
              else if (vc_ele2_id == p_master_id)
              {  // switch ele 1 <-> ele 2
                nds_master = vc2->nodal_dof_set();
                nds_slave = vc1->nodal_dof_set();
              }
              else
                FOUR_C_THROW("no element (ele1 and ele2) is the parent element!!! WHY?");
            }
            else if ((*f)->position() == Cut::Point::inside && all_dofs)
            {
              for (std::size_t n = 0; n < vc1->parent_element()->nodes().size(); ++n)
              {
                if (!vc1->parent_element()->nodes()[n]->nodal_dof_sets().size())
                {
                  all_dofs = false;
                  break;
                }
              }
              if (all_dofs)
                for (std::size_t n = 0; n < vc2->parent_element()->nodes().size(); ++n)
                {
                  if (!vc2->parent_element()->nodes()[n]->nodal_dof_sets().size())
                  {
                    all_dofs = false;
                    break;
                  }
                }
              if (all_dofs)
              {
                nds_master.clear();
                nds_slave.clear();
                if (vc1->parent_element()->num_nodes() == vc2->parent_element()->num_nodes())
                {
                  //------------------------ create nodal dof sets
                  TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: create nds");
                  for (std::size_t n = 0; n < vc2->parent_element()->nodes().size(); ++n)
                  {
                    nds_master.push_back(0);
                    nds_slave.push_back(0);
                  }
                }
                else
                  FOUR_C_THROW("Number of Nodes different between Master and Slave Element!");
              }
            }

            if ((*f)->position() == Cut::Point::inside && !include_inner && !all_dofs) continue;
            //------------------------

            num_edgestab++;

            // at least one element has to be cut
            if (p_master_handle->is_intersected() or p_slave_handle->is_intersected())
            {
              num_ghostpenalty++;

              face_type = XFEM::face_type_ghost_penalty;
            }
            else
              face_type = XFEM::face_type_std;

            XFEM::Utils::get_volume_cell_material(p_master, matptr_m, (*f)->position());
            XFEM::Utils::get_volume_cell_material(p_slave, matptr_s, (*f)->position());

            //--------------------------------------------------------------------------------------------

            // call evaluate and assemble routine
            assemble_edge_stab_ghost_penalty(eleparams, face_type, faceele, matptr_m, matptr_s,
                nds_master, nds_slave, *xdiscret, systemmatrix, systemvector);

            //--------------------------------------------------------------------------------------------
          }
          else if (vcs.size() == 1)
          {
            FOUR_C_THROW("just one vcs reasonable?! face {}", faceele->id());
          }
        }  // facet outside or (inside and include_inner)
        else if ((*f)->position() == Cut::Point::undecided)
        {
          FOUR_C_THROW("the position of this facet is undecided, how to stabilize???");
        }
        else if ((*f)->position() == Cut::Point::oncutsurface)
        {
#ifdef FOUR_C_ENABLE_ASSERTIONS
          std::cout << "the position of this facet of face " << faceele->id()
                    << " is oncutsurface, we do not stabilize it!!! " << std::endl;
#endif
          // if a facet lies oncutsurface, then there is only one neighbor, we do not stabilize this
          // facet REMARK: in case of one part of the facet is physical and the other part lies on
          // cutsurface,
          //         then the physical part is stabilized via another facet lying on the same fluid
          //         element's side
        }
        else
        {
          // facet is inside!
          face_type = XFEM::face_type_ghost;
        }

      }  // loop facets
    }  // if linear elements
    else if (p_master->shape() == Core::FE::CellType::hex20 or
             p_master->shape() == Core::FE::CellType::hex27 or
             p_master->shape() == Core::FE::CellType::tet10 or
             p_master->shape() == Core::FE::CellType::wedge15)
    {
      Cut::SideHandle* side = get_face(faceele, wizard);  // the side of the quadratic element
      //-------------------------------- loop facets of this side -----------------------------
      // facet of current side
      std::vector<Cut::Facet*> facets;

      side->facets(facets);  // all facets of this quadratic element side
      if (facets.size() == 0)
        FOUR_C_THROW("there is no facet between two elements with elementhandle!");
      // each facet should have 2 volumecells
      std::vector<std::vector<int>> all_used_nds_master;
      std::vector<std::vector<int>> all_used_nds_slave;
      for (std::vector<Cut::Facet*>::const_iterator f = facets.begin(); f != facets.end(); f++)
      {
        if ((*f)->position() == Cut::Point::outside or
            ((*f)->position() == Cut::Point::inside and include_inner))
        {
          Cut::plain_volumecell_set vcs = (*f)->cells();
          // how many volumecells found?
          if (vcs.size() ==
              2)  // standard XFEM case (facet between two vcs of two neighbouring cut elements
          {
            //------------------------ create nodal dof sets
            {
              TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: create nds");

              Cut::plain_volumecell_set::iterator vc_it = vcs.begin();

              Cut::VolumeCell* vc1 = *(vc_it);
              vc_it++;
              Cut::VolumeCell* vc2 = *(vc_it);

              // get the parent element
              int vc_ele1_id = vc1->parent_element()->get_parent_id();
              int vc_ele2_id = vc2->parent_element()->get_parent_id();

              // which element is the parent element
              if (vc_ele1_id == p_master_id)
              {
                nds_master = vc1->nodal_dof_set();
                nds_slave = vc2->nodal_dof_set();
              }
              else if (vc_ele2_id == p_master_id)
              {  // switch ele 1 <-> ele 2
                nds_master = vc2->nodal_dof_set();
                nds_slave = vc1->nodal_dof_set();
              }
              else
              {
                FOUR_C_THROW("no element (ele1 and ele2) is the parent element!!! WHY?");
              }
            }
            bool new_nds_master = true;
            bool new_nds_slave = true;
            for (std::vector<std::vector<int>>::iterator i = all_used_nds_master.begin();
                i != all_used_nds_master.end(); ++i)
            {
              std::vector<int> used_nds_master = *i;
              if (used_nds_master == nds_master)
              {
                new_nds_master = false;
              }
            }
            for (std::vector<std::vector<int>>::iterator i = all_used_nds_slave.begin();
                i != all_used_nds_slave.end(); ++i)
            {
              std::vector<int> used_nds_slave = *i;
              if (used_nds_slave == nds_slave)
              {
                new_nds_slave = false;
              }
            }
            if (new_nds_master == false and new_nds_slave == false)
            {
              continue;
            }
            if (new_nds_master == true)
            {
              all_used_nds_master.push_back(nds_master);
            }
            if (new_nds_slave == true)
            {
              all_used_nds_slave.push_back(nds_slave);
            }
            //------------------------
            num_edgestab++;
            // at least one element has to be cut
            if (p_master_handle->is_intersected() or p_slave_handle->is_intersected())
            {
              num_ghostpenalty++;

              face_type = XFEM::face_type_ghost_penalty;
            }
            else
              face_type = XFEM::face_type_std;

            XFEM::Utils::get_volume_cell_material(p_master, matptr_m, (*f)->position());
            XFEM::Utils::get_volume_cell_material(p_slave, matptr_s, (*f)->position());

            //--------------------------------------------------------------------------------------------
            // call evaluate and assemble routine
            assemble_edge_stab_ghost_penalty(eleparams, face_type, faceele, matptr_m, matptr_s,
                nds_master, nds_slave, *xdiscret, systemmatrix, systemvector);
            //--------------------------------------------------------------------------------------------
          }
          else if (vcs.size() == 1)
          {
            FOUR_C_THROW("just one vcs reasonable?! face {}", faceele->id());
          }
        }  // facet outside or (inside and include_inner)
        else if ((*f)->position() == Cut::Point::undecided)
        {
          FOUR_C_THROW("the position of this facet is undecided, how to stabilize???");
        }
        else if ((*f)->position() == Cut::Point::oncutsurface)
        {
#ifdef FOUR_C_ENABLE_ASSERTIONS
          std::cout << "the position of this facet of face " << faceele->id()
                    << " is oncutsurface, we do not stabilize it!!! " << std::endl;
#endif
          // if a facet lies oncutsurface, then there is only one neighbor, we do not stabilize this
          // facet REMARK: in case of one part of the facet is physical and the other part lies on
          // cutsurface,
          //         then the physical part is stabilized via another facet lying on the same fluid
          //         element's side
        }
        else
        {
          // facet is inside!
          face_type = XFEM::face_type_ghost;
        }
      }  // loop facets
    }
    else
      FOUR_C_THROW("not supported for this elements");
  }  // end second case: element handles for both parent elements
  //------------------------------------------------------------------------------
  // third case: element handle only for master element or for slave element available
  // at most one element cut
  //------------------------------------------------------------------------------
  else if ((p_master_handle != nullptr and p_slave_handle == nullptr) or
           (p_master_handle == nullptr and p_slave_handle != nullptr))
  {
    // linear elements
    if (p_master->shape() == Core::FE::CellType::hex8 or
        p_master->shape() == Core::FE::CellType::tet4 or
        p_master->shape() == Core::FE::CellType::wedge6 or
        p_master->shape() == Core::FE::CellType::pyramid5 or
        p_master->shape() == Core::FE::CellType::hex20 or
        p_master->shape() == Core::FE::CellType::hex27 or
        p_master->shape() == Core::FE::CellType::tet10 or
        p_master->shape() == Core::FE::CellType::wedge15)
    {
      Cut::SideHandle* side = get_face(faceele, wizard);

      // facet of current side
      std::vector<Cut::Facet*> facets;
      side->facets(facets);

      if (p_master->shape() == Core::FE::CellType::hex8 or
          p_master->shape() == Core::FE::CellType::tet4 or
          p_master->shape() == Core::FE::CellType::wedge6 or
          p_master->shape() == Core::FE::CellType::pyramid5)
      {
        if (facets.size() != 1) FOUR_C_THROW("there has to be 1 facet equal to the side");
      }

      // get the unique single facet
      Cut::Facet* f = facets[0];
      if (f->position() == Cut::Point::outside or
          (f->position() == Cut::Point::inside and include_inner))
      {
        Cut::plain_volumecell_set vcs = f->cells();

        if (vcs.size() != 1)
          FOUR_C_THROW("there has to be 1 volumecell equal to the side");
        else
        {
          //------------------------ create nodal dof sets
          {
            TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: create nds");

            Cut::VolumeCell* vc = *(vcs.begin());

            // get the parent element
            int vc_ele_id = vc->parent_element()->id();
            if (vc_ele_id == -1)
            {
              vc_ele_id = vc->parent_element()->get_parent_id();
            }


            // which element is the parent element
            if (p_master_handle != nullptr)
            {
              nds_master = vc->nodal_dof_set();

              for (size_t i = 0; i < p_slave_numnode; i++) nds_slave.push_back(0);
            }
            else if (p_slave_handle != nullptr)
            {
              for (size_t i = 0; i < p_master_numnode; i++) nds_master.push_back(0);

              nds_slave = vc->nodal_dof_set();
            }
            else
              FOUR_C_THROW("no element (ele1 and ele2) is the parent element!!! WHY?");
          }
          //------------------------

          num_edgestab++;

          // at most one element can be a cut one
          if (p_master_handle != nullptr)
          {
            if (p_master_handle->is_intersected())
            {
              num_ghostpenalty++;
              face_type = XFEM::face_type_ghost_penalty;
            }
            else
              face_type = XFEM::face_type_std;
          }
          else if (p_slave_handle != nullptr)
          {
            if (p_slave_handle->is_intersected())
            {
              num_ghostpenalty++;
              face_type = XFEM::face_type_ghost_penalty;
            }
            else
              face_type = XFEM::face_type_std;
          }
          else
            face_type = XFEM::face_type_std;


          // Get materials:
          XFEM::Utils::get_volume_cell_material(p_master, matptr_m, f->position());
          XFEM::Utils::get_volume_cell_material(p_slave, matptr_s, f->position());

          //--------------------------------------------------------------------------------------------

          // call evaluate and assemble routine
          assemble_edge_stab_ghost_penalty(eleparams, face_type, faceele, matptr_m, matptr_s,
              nds_master, nds_slave, *xdiscret, systemmatrix, systemvector);

          //--------------------------------------------------------------------------------------------
        }

      }  // if outside or (inside and include_inner)
    }
  }  // end last case



  if (gmsh_eos_out)
  {
    ghost_penalty_stab_.insert(std::pair<int, int>(faceele->id(), num_ghostpenalty));
    edge_based_stab_.insert(std::pair<int, int>(faceele->id(), num_edgestab));
  }

  //--------------------------------------------------------------------------------------------

  return;
}


/*----------------------------------------------------------------------*
 | calls the evaluate and assemble routine for edge based stabilization |
 | and ghost penalty in the XFEM                            schott 03/12 |
 *----------------------------------------------------------------------*/
void XFEM::XfemEdgeStab::assemble_edge_stab_ghost_penalty(
    Teuchos::ParameterList& eleparams,         ///< element parameter list
    const XFEM::FaceType& face_type,           ///< which type of face std, ghost, ghost-penalty
    Discret::Elements::FluidIntFace* intface,  ///< internal face element
    std::shared_ptr<Core::Mat::Material>& material_m,  ///< material of the master side
    std::shared_ptr<Core::Mat::Material>& material_s,  ///< material of the slave side
    std::vector<int>& nds_master,             ///< nodal dofset vector w.r.t. master element
    std::vector<int>& nds_slave,              ///< nodal dofset vector w.r.t. slave element
    Core::FE::DiscretizationFaces& xdiscret,  ///< XFEM discretization
    std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix,   ///< systemmatrix
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector  ///< systemvector
)
{
  // If Safety check is passed, both elements contain the same material and with the same settings
  XFEM::Utils::safety_check_materials(material_m, material_s);

  //======================================================================================
  // call the internal faces stabilization routine for the current side/surface
  TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: assemble_edge_stab_ghost_penalty");

  // set action and facetype for elements

  // TODO: set here the right stab-type LPS or EOS
  eleparams.set<FLD::IntFaceAction>("action", FLD::EOS_and_GhostPenalty_stabilization);

  // call the edge-based assemble and evaluate routine
  Discret::Elements::FluidIntFaceImplInterface::impl(intface)
      ->assemble_internal_faces_using_neighbor_data(intface, material_m, nds_master, nds_slave,
          face_type, eleparams, xdiscret, systemmatrix, systemvector);


  return;
}


/*----------------------------------------------------------------------*
 | get the cut side for face's element identified using the sorted      |
 | node ids                                                schott 04/12 |
 *----------------------------------------------------------------------*/
Cut::SideHandle* XFEM::XfemEdgeStab::get_face(
    Core::Elements::Element* faceele, Cut::CutWizard& wizard)
{
  TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: get_face");

  // get faceele's nodes
  const int numnode = faceele->num_node();
  std::vector<int> nodeids(numnode);

  for (int inode = 0; inode < numnode; inode++)
  {
    nodeids[inode] = faceele->node_ids()[inode];
  }

  std::sort(nodeids.begin(), nodeids.end());

  return wizard.get_side(nodeids);
}

/*----------------------------------------------------------------------*
 | reset maps for output                                    kruse 04/15 |
 *----------------------------------------------------------------------*/
void XFEM::XfemEdgeStab::reset()
{
  ghost_penalty_stab_.clear();
  edge_based_stab_.clear();
}



/*----------------------------------------------------------------------*
 |  prepares edge based stabilization for standard fluid   schott 05/12 |
 *----------------------------------------------------------------------*/
void XFEM::XfemEdgeStab::evaluate_edge_stab_std(
    Teuchos::ParameterList& eleparams,                          ///< element parameter list
    std::shared_ptr<Core::FE::Discretization> discret,          ///< discretization
    Discret::Elements::FluidIntFace* faceele,                   ///< face element
    std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix,   ///< systemmatrix
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector  ///< systemvector
)
{
  std::shared_ptr<Core::FE::DiscretizationFaces> xdiscret =
      std::dynamic_pointer_cast<Core::FE::DiscretizationFaces>(discret);
  if (xdiscret == nullptr)
    FOUR_C_THROW(
        "Failed to cast Core::FE::Discretization to "
        "Core::FE::DiscretizationFaces.");


  // get the parent fluid elements
  Discret::Elements::Fluid* p_master = faceele->parent_target_element();
  Discret::Elements::Fluid* p_slave = faceele->parent_source_element();

  size_t p_master_numnode = p_master->num_node();
  size_t p_slave_numnode = p_slave->num_node();

  //------------------------------------------------------------------------------
  // simplest case: no element handles for both parent elements
  // two uncut elements / standard fluid case
  //------------------------------------------------------------------------------

  std::vector<int> nds_master(p_master_numnode, 0);
  std::vector<int> nds_slave(p_slave_numnode, 0);

  // Provide material at both sides:
  std::shared_ptr<Core::Mat::Material> matptr_m;
  std::shared_ptr<Core::Mat::Material> matptr_s;
  matptr_m = p_master->material();
  matptr_s = p_slave->material();

  //--------------------------------------------------------------------------------------------

  // call evaluate and assemble routine
  assemble_edge_stab_ghost_penalty(eleparams, XFEM::face_type_std, faceele, matptr_m, matptr_s,
      nds_master, nds_slave, *xdiscret, systemmatrix, systemvector);

  //--------------------------------------------------------------------------------------------

  return;
}

/*----------------------------------------------------------------------*
 |  prepares edge based stabilization for fluid-fluid applications      |
 |  where EOS pressure stab. shall be applied to interface-contributing |
 |  embedded fluid elements                               (kruse 10/14) |
 *----------------------------------------------------------------------*/
void XFEM::XfemEdgeStab::evaluate_edge_stab_boundary_gp(
    Teuchos::ParameterList& eleparams,                  ///< element parameter list
    std::shared_ptr<Core::FE::Discretization> discret,  ///< discretization
    Core::FE::Discretization&
        boundarydiscret,  ///< auxiliary discretization of interface-contributing elements
    Discret::Elements::FluidIntFace* faceele,                   ///< face element
    std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix,   ///< systemmatrix
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector  ///< systemvector
)
{
  std::shared_ptr<Core::FE::DiscretizationFaces> xdiscret =
      std::dynamic_pointer_cast<Core::FE::DiscretizationFaces>(discret);
  if (xdiscret == nullptr)
    FOUR_C_THROW(
        "Failed to cast Core::FE::Discretization to "
        "Core::FE::DiscretizationFaces.");


  // get the parent fluid elements
  Discret::Elements::Fluid* p_master = faceele->parent_target_element();
  Discret::Elements::Fluid* p_slave = faceele->parent_source_element();

  size_t p_master_numnode = p_master->num_node();
  size_t p_slave_numnode = p_slave->num_node();


  std::vector<int> nds_master;
  nds_master.reserve(p_master_numnode);

  std::vector<int> nds_slave;
  nds_slave.reserve(p_slave_numnode);

  //------------------------------------------------------------------------------
  // simplest case: no element handles for both parent elements
  // two uncut elements / standard fluid case
  //------------------------------------------------------------------------------

  {
    TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: create nds");

    for (size_t i = 0; i < p_master_numnode; i++) nds_master.push_back(0);

    for (size_t i = 0; i < p_slave_numnode; i++) nds_slave.push_back(0);
  }

  //--------------------------------------------------------------------------------------------
  // leave, if neither slave nor master element of this face contributes to the fluid-fluid
  // interface
  if (!(boundarydiscret.have_global_element(p_master->id()) ||
          boundarydiscret.have_global_element(p_slave->id())))
    return;

  // Provide material at both sides:
  std::shared_ptr<Core::Mat::Material> matptr_m;
  std::shared_ptr<Core::Mat::Material> matptr_s;
  matptr_m = p_master->material();
  matptr_s = p_slave->material();


  // call evaluate and assemble routine
  assemble_edge_stab_ghost_penalty(eleparams, XFEM::face_type_boundary_ghost_penalty, faceele,
      matptr_m, matptr_s, nds_master, nds_slave, *xdiscret, systemmatrix, systemvector);

  //--------------------------------------------------------------------------------------------

  return;
}

FOUR_C_NAMESPACE_CLOSE
