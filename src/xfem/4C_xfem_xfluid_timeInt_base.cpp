// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_xfluid_timeInt_base.hpp"

#include "4C_bele_bele3.hpp"
#include "4C_comm_exporter.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_cut_cutwizard.hpp"
#include "4C_cut_elementhandle.hpp"
#include "4C_cut_intersection.hpp"
#include "4C_cut_position.hpp"
#include "4C_cut_sidehandle.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_gmsh.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_enum.hpp"
#include "4C_xfem_dofset.hpp"

FOUR_C_NAMESPACE_OPEN

// #define DEBUG_TIMINT_STD

/*------------------------------------------------------------------------------------------------*
 * basic XFEM time-integration constructor                                           schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
XFEM::XfluidTimeintBase::XfluidTimeintBase(
    const std::shared_ptr<Core::FE::Discretization> discret,      /// background discretization
    const std::shared_ptr<Core::FE::Discretization> boundarydis,  /// cut discretization
    std::shared_ptr<Cut::CutWizard> wizard_old,    /// cut wizard w.r.t. old interface position
    std::shared_ptr<Cut::CutWizard> wizard_new,    /// cut wizard w.r.t. new interface position
    std::shared_ptr<XFEM::XFEMDofSet> dofset_old,  /// XFEM dofset w.r.t. old interface position
    std::shared_ptr<XFEM::XFEMDofSet> dofset_new,  /// XFEM dofset w.r.t. new interface position
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>
        oldVectors,  /// vector of col-vectors w.r.t. old interface position
    std::shared_ptr<Core::LinAlg::Vector<double>> dispn,   /// old col displacement vector
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnp,  /// col displacement n +1
    const Core::LinAlg::Map& olddofcolmap,  /// dofcolmap w.r.t. old interface position
    const Core::LinAlg::Map& newdofrowmap,  /// dofcolmap w.r.t. new interface position
    const std::shared_ptr<std::map<int, std::vector<int>>>
        pbcmap  /// map of periodic boundary conditions
    )
    : discret_(discret),
      boundarydis_(boundarydis),
      wizard_old_(wizard_old),
      wizard_new_(wizard_new),
      dofset_old_(dofset_old),
      dofset_new_(dofset_new),
      olddofcolmap_(olddofcolmap),
      newdofrowmap_(newdofrowmap),
      oldVectors_(oldVectors),
      dispn_(dispn),
      dispnp_(dispnp),
      timeIntData_(nullptr),
      pbcmap_(pbcmap),
      myrank_(Core::Communication::my_mpi_rank(discret_->get_comm())),
      numproc_(Core::Communication::num_mpi_ranks(discret_->get_comm())),
      newton_max_iter_(10),  /// maximal number of newton iterations for Semi-Lagrangean algorithm
      limits_tol_(1.0e-10),  /// newton tolerance for Semi-Lagrangean algorithm
      TOL_dist_(1.0e-12)  /// tolerance to find the shortest distance of point to its projection on
                          /// the surface dis
{
  return;
}  // end constructor


/*------------------------------------------------------------------------------------------------*
 * set the computation type with help of the iteration counter                       schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::type(int iter, int iterMax)
{
  // TODO: update this for FSI
  if (iter == 1)
    FGIType_ = FRS1FGI1_;
  else if (iterMax == 1 or iter % iterMax == 1)
    FGIType_ = FRS1FGINot1_;
  else
    FGIType_ = FRSNot1_;

  return;
}  // end function type



/*------------------------------------------------------------------------------------------------*
 * assign the Epetra vectors which shall be computed to the                                       *
 * algorithms data structure                                                         schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::handle_vectors(
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& newRowVectorsn)
{
  newVectors_ = newRowVectorsn;

  if (oldVectors_.size() != newVectors_.size())
    FOUR_C_THROW(
        "Number of state-vectors at new and old discretization are different!");  // but they have
                                                                                  // different maps
                                                                                  // (row vs col)

}  // end function handle_vectors



/*------------------------------------------------------------------------------------------------*
 * check if the current point x2 changed the side compared to x1                     schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
bool XFEM::XfluidTimeintBase::changed_side_same_time(
    const bool newTimeStep,          /// new/old timestep for both points x1 and x2
    Core::Elements::Element* ele1,   /// first element where x1 lies in
    Core::LinAlg::Matrix<3, 1>& x1,  /// global coordinates of point x1
    Core::Elements::Element* ele2,   /// second element where x2 lies in
    Core::LinAlg::Matrix<3, 1>& x2   /// global coordinates of point x2
) const
{
#ifdef DEBUG_TIMINT_STD
  Core::IO::cout << "\n\t\t\t check if point changed the side ";
#endif

  //-----------------------------------------------------------------------
  // special case of equal coordinates x1 and x2 -> no line
  Core::LinAlg::Matrix<3, 1> diff(Core::LinAlg::Initialization::set_zero);
  diff.update(1.0, x1, -1.0, x2);

  if (diff.norm2() < 1.0e-13) return false;

  //-----------------------------------------------------------------------
  // standard case of a real line between x1 and x2

  std::shared_ptr<Cut::CutWizard> wizard = newTimeStep ? wizard_new_ : wizard_old_;

  // REMARK:
  // changing the side of a point at two times (newton steps) with coordinates x1 and x2 is done
  // via changing the cut of the linear trace between x1 and x2 with surrounding cutting sides
  // if there is a cut between the ray tracer and a cutting side, then the point changed the side
  // while moving from Position x1 to Position x2

  //-----------------------------------------------------------------------
  //-----------------------------------------------------------------------
  // first: find involved elements which are possibly passed through by the ray between x1 and x2

#ifdef DEBUG_TIMINT_STD
  Core::IO::cout << "\n\t\t\t\t\t --------------------------------- ";
  Core::IO::cout << "\n\t\t\t\t\t Find involved background elements ";
  Core::IO::cout << "\n\t\t\t\t\t --------------------------------- ";
#endif

  int mi = 0;  // we assume only cutting sides with mesh 0 (meshintersection)

  bool check_allsides =
      false;  // flag to check all sides in case of one common node is a non-row node

  std::set<int> cut_sides;

  std::set<int> eids;
  std::set<int> common_nodes;

  if (ele1 == nullptr or ele2 == nullptr)
  {
    check_allsides = true;  // no efficient search possible
  }
  else
  {
    if (ele1->id() == ele2->id())  // line within one element
    {
      eids.insert(ele1->id());  // just one element to check

#ifdef DEBUG_TIMINT_STD
      Core::IO::cout << "\n\t\t\t\t\t check changing side just for the unique element" << ele1->Id()
                     << Core::IO::endl;
#endif
    }
    else if (neighbors(ele1, ele2, common_nodes))
    {
#ifdef DEBUG_TIMINT_STD
      Core::IO::cout << "\n\t\t\t\t\t check changing side for neighboring elements: "
                     << Core::IO::endl;
#endif

      // get all elements adjacent to common nodes in ele1 and ele2

      // REMARK: if at least one common node is not a row node, then there is possibly one adjacent
      // element missing, but then there are also involved sides missing -> check all sides in
      // cutdiscret, also if it is very slow!

      for (std::set<int>::iterator it = common_nodes.begin(); it != common_nodes.end(); it++)
      {
        Core::Nodes::Node* n = discret_->g_node(*it);

        if (n->owner() != myrank_)
        {
          check_allsides =
              true;  // flag to check all sides in case of one common node is a non-row node
          break;
        }

        const int numele = n->num_element();

        Core::Elements::Element** elements = n->elements();

        for (int e_it = 0; e_it < numele; e_it++)
        {
          Core::Elements::Element* ele = elements[e_it];

          eids.insert(ele->id());

#ifdef DEBUG_TIMINT_STD
          Core::IO::cout << "\n\t\t\t\t\t add element " << ele->Id() << Core::IO::endl;
#endif
        }
      }
    }
    else
    {
      // check all sides ghosted in boundarydis
      check_allsides = true;

#ifdef DEBUG_TIMINT_STD
      Core::IO::cout
          << "\n\t\t\t\t\t all sides on boundary discretization have to be check, ele1 = "
          << ele1->Id() << " and ele2 = " << ele2->Id() << " are not Neighbors" << Core::IO::endl;
#endif
    }
  }
  //-----------------------------------------------------------------------
  //-----------------------------------------------------------------------
  // second: find all boundary sides involved in cutting the determined background elements


#ifdef DEBUG_TIMINT_STD
  Core::IO::cout << "\n\t\t\t\t\t --------------------------------- ";
  Core::IO::cout << "\n\t\t\t\t\t Find involved sides with possible cuts between trace and side ";
  Core::IO::cout << "\n\t\t\t\t\t --------------------------------- " << Core::IO::endl;
#endif


  // collect all cutting sides
  if (check_allsides)
  {
    // add all sides of cut_discret to check
    for (int i = 0; i < boundarydis_->num_my_col_elements(); i++)
    {
      cut_sides.insert(boundarydis_->element_col_map()->GID(i));
    }
  }
  else
  {
    // loop all found element
    for (std::set<int>::iterator e_it = eids.begin(); e_it != eids.end(); e_it++)
    {
      Core::Elements::Element* ele = discret_->g_element(*e_it);

      Cut::ElementHandle* eh = wizard->get_element(ele);

      // no cutsides within this element, then no side changing possible
      if (eh == nullptr)
      {
        continue;  // next element
      }

      //--------------------------------------------------------
      // get involved side ids
      Cut::plain_element_set elements;
      eh->collect_elements(elements);

      // get all side-ids
      for (Cut::plain_element_set::iterator eles = elements.begin(); eles != elements.end(); eles++)
      {
        Cut::Element* sub_ele = *eles;

        Cut::plain_facet_set facets = sub_ele->facets();

        for (Cut::plain_facet_set::const_iterator facet_it = facets.begin();
            facet_it != facets.end(); facet_it++)
        {
          Cut::Facet* facet = *facet_it;

          Cut::Side* parent_side = facet->parent_side();

          bool is_elements_side = sub_ele->owned_side(parent_side);

          if (!is_elements_side)  // is a cutting side
          {
            cut_sides.insert(parent_side->id());

#ifdef DEBUG_TIMINT_STD
            Core::IO::cout << "\n\t\t\t\t\t add side with Id=" << parent_side->Id()
                           << Core::IO::endl;
#endif
          }  // !element's side
        }  // facets
      }  // sub elements
    }
  }


  //-----------------------------------------------------------------------
  //-----------------------------------------------------------------------
  // third: check cut between the line and all cutting sides
  for (std::set<int>::iterator side_it = cut_sides.begin(); side_it != cut_sides.end(); side_it++)
  {
    // get the side via sidehandle
    Cut::SideHandle* sh = wizard->get_mesh_cutting_side(*side_it, mi);

#ifdef DEBUG_TIMINT_STD
    Core::IO::cout << "\n\t\t\t\t\t SIDE-CHECK with side=" << *side_it << Core::IO::endl;
#endif

    if (call_side_edge_intersection(sh, *side_it, x1, x2))
    {
#ifdef DEBUG_TIMINT_STD
      Core::IO::cout << "\n\t\t\t\t\t <<< POINT CHANGED THE SIDE >>>" << Core::IO::endl;
#endif

      return true;
    }
  }

  return false;
}


/*------------------------------------------------------------------------------------------------*
 * check if both element are neighbors sharing at least one common node              schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
bool XFEM::XfluidTimeintBase::neighbors(
    Core::Elements::Element* ele1, Core::Elements::Element* ele2, std::set<int>& common_nodes) const
{
  bool is_neighbor = false;

  const int numnode1 = ele1->num_node();
  const int numnode2 = ele2->num_node();

  const int* nodeids1 = ele1->node_ids();
  const int* nodeids2 = ele2->node_ids();

  std::vector<int> nodeids1_vec;
  std::vector<int> nodeids2_vec;

  for (int i = 0; i < numnode1; i++) nodeids1_vec.push_back(nodeids1[i]);

  for (int i = 0; i < numnode2; i++) nodeids2_vec.push_back(nodeids2[i]);

  sort(nodeids1_vec.begin(), nodeids1_vec.end());
  sort(nodeids2_vec.begin(), nodeids2_vec.end());

  //------------------------------------------------------
  // find common nodes
  int index1 = 0;
  int index2 = 0;
  while (true)
  {
    // if at least one vector of nodes is checked
    if (index1 >= numnode1 or index2 >= numnode2) break;

    if (nodeids1_vec[index1] < nodeids2_vec[index2])
      index1++;
    else if (nodeids1_vec[index1] > nodeids2_vec[index2])
      index2++;
    else  // nodeids1_vec[index1] == nodeids2_vec[index2]
    {
      common_nodes.insert(nodeids1_vec[index1]);

      is_neighbor = true;  // true if at least one node is common, find also further common nodes

      index1++;
      index2++;
    }
  }

  return is_neighbor;
}


/*------------------------------------------------------------------------------------------------*
 * check if edge between x1 and x2 cuts the side                                     schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
bool XFEM::XfluidTimeintBase::call_side_edge_intersection(Cut::SideHandle* sh,  /// side handle
    int sid,                                                                    /// side id
    Core::LinAlg::Matrix<3, 1>& x1,  /// coordinates of edge's start point
    Core::LinAlg::Matrix<3, 1>& x2   /// coordinates of edge's end point
) const
{
  switch (sh->shape())
  {
    case Core::FE::CellType::tri3:
    {
      return call_side_edge_intersection_t<Core::FE::CellType::tri3>(sh, sid, x1, x2);
    }
    case Core::FE::CellType::quad4:
    {
      return call_side_edge_intersection_t<Core::FE::CellType::quad4>(sh, sid, x1, x2);
    }
    case Core::FE::CellType::quad8:
    {
      return call_side_edge_intersection_t<Core::FE::CellType::quad8>(sh, sid, x1, x2);
    }
    case Core::FE::CellType::quad9:
    {
      return call_side_edge_intersection_t<Core::FE::CellType::quad9>(sh, sid, x1, x2);
    }
    default:
    {
      FOUR_C_THROW("unknown side shape");
      break;
    }
  }

  return false;
}

/*------------------------------------------------------------------------------------------------*
 * check if edge between x1 and x2 cuts the side (templated)                         schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType sidetype>
bool XFEM::XfluidTimeintBase::call_side_edge_intersection_t(Cut::SideHandle* sh,  /// side handle
    int sid,                                                                      /// side id
    Core::LinAlg::Matrix<3, 1>& x1,  /// coordinates of edge's start point
    Core::LinAlg::Matrix<3, 1>& x2   /// coordinates of edge's end point
) const
{
  const int nsd = 3;
  const int numNodesSurface = Core::FE::num_nodes<sidetype>;

  Core::LinAlg::Matrix<nsd, 2> xyze_lineElement(Core::LinAlg::Initialization::set_zero);

  for (int i = 0; i < nsd; i++)
  {
    xyze_lineElement(i, 0) = x1(i);
    xyze_lineElement(i, 1) = x2(i);
  }

  Core::LinAlg::SerialDenseMatrix xyze_side;
  sh->coordinates(xyze_side);

  Core::LinAlg::Matrix<nsd, numNodesSurface> xyze_surfaceElement(xyze_side);

  Core::LinAlg::Matrix<3, 1> xsi(Core::LinAlg::Initialization::set_zero);


  std::shared_ptr<Cut::IntersectionBase> intersect =
      Cut::IntersectionBase::create(Core::FE::CellType::line2, sidetype);
  std::shared_ptr<Cut::Options> options =
      std::make_shared<Cut::Options>();  // Create cut options for intersection
                                         // (specify to use double prec.)
  intersect->init(xyze_lineElement, xyze_surfaceElement, false, false, false, options.get());

  // check also limits during the newton scheme and when converged
  double itol;
  return (static_cast<bool>(intersect->compute_edge_side_intersection(itol)));
}


/*------------------------------------------------------------------------------------------------*
 * call the computation of local coordinates for an integration cell                 schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::call_x_to_xi_coords(
    const Core::Elements::Element* ele,  /// pointer to element
    Core::LinAlg::Matrix<3, 1>& x,       /// global coordinates of point
    Core::LinAlg::Matrix<3, 1>& xi,      /// determined local coordinates w.r.t ele
    const std::string state,             ///< state dispn or dispnp?
    bool& pointInDomain                  /// lies point in element ?
) const
{
  Core::LinAlg::SerialDenseMatrix xyz(3, ele->num_node(), true);
  Core::Geo::fill_initial_position_array(ele, xyz);

  // add ale displacements to initial position
  if (state != "reference" && dispnp_ != nullptr &&
      dispn_ != nullptr)  // add no displacements in case of state == "reference" or //is ale?
  {
    int nen = ele->num_node();
    int numdof = ele->num_dof_per_node(*(ele->nodes()[0]));

    std::vector<int> nds(nen, 0);

    Core::Elements::LocationArray la(1);
    ele->location_vector(*discret_, nds, la, false);

    // extract local values of the global vectors
    std::vector<double> mydispnp(la[0].lm_.size());

    if (state == "dispnp")
      mydispnp = Core::FE::extract_values(*dispnp_, la[0].lm_);
    else if (state == "dispn")
      mydispnp = Core::FE::extract_values(*dispn_, la[0].lm_);
    else
      FOUR_C_THROW("XFEM::XfluidTimeintBase::call_x_to_xi_coords: Undefined state!");

    for (int inode = 0; inode < nen; ++inode)  // number of nodes
    {
      for (int idim = 0; idim < 3; ++idim)  // number of dimensions
      {
        (xyz)(idim, inode) +=
            mydispnp[idim + (inode * numdof)];  // attention! dispnp vector has 3+1 dofs for
                                                // displacement (the same as for (u,p))
      }
    }
  }

  call_x_to_xi_coords(xyz, ele->shape(), x, xi, pointInDomain);
}  // end function call_x_to_xi_coords



/*------------------------------------------------------------------------------------------------*
 * call the computation of local coordinates for a polytop                                        *
 * with corners given by the coordinates                                             schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::call_x_to_xi_coords(
    Core::LinAlg::SerialDenseMatrix& nodecoords,  /// node coordinates of element
    Core::FE::CellType DISTYPE,                   /// discretization type
    Core::LinAlg::Matrix<3, 1>& x,                /// global coordinates of point
    Core::LinAlg::Matrix<3, 1>& xi,               /// determined local coordinates w.r.t ele
    bool& pointInDomain                           /// lies point in element ?
) const
{
  switch (DISTYPE)
  {
    case Core::FE::CellType::hex8:
      x_to_xi_coords<Core::FE::CellType::hex8>(nodecoords, x, xi, pointInDomain);
      break;
    case Core::FE::CellType::hex20:
      x_to_xi_coords<Core::FE::CellType::hex20>(nodecoords, x, xi, pointInDomain);
      break;
    case Core::FE::CellType::tet4:
      x_to_xi_coords<Core::FE::CellType::tet4>(nodecoords, x, xi, pointInDomain);
      break;
    default:
      FOUR_C_THROW("add your 3D distype and the according transformation!");
      break;
  }  // end switch

  return;
}  // end function call_x_to_xi_coords


//! compute local element coordinates and check whether the according point is inside the element
template <Core::FE::CellType distype>
void XFEM::XfluidTimeintBase::x_to_xi_coords(
    Core::LinAlg::SerialDenseMatrix& xyz,  /// node coordinates of element
    Core::LinAlg::Matrix<3, 1>& x,         /// global coordinates of point
    Core::LinAlg::Matrix<3, 1>& xi,        /// determined local coordinates w.r.t ele
    bool& pointInCell                      /// lies point in element?
) const
{
  const int nsd = 3;                                 // dimension
  const int numnode = Core::FE::num_nodes<distype>;  // number of nodes of
                                                     // element

  Core::LinAlg::Matrix<nsd, numnode> xyze(xyz);

  std::shared_ptr<Cut::Position> pos = Cut::PositionFactory::build_position<3, distype>(xyze, x);
  pos->compute();
  pos->local_coordinates(xi);  // local coordinates

  pointInCell = pos->within_limits_tol(limits_tol_);  // check if point is in element
};


//! data at an arbitrary point lying in an element
template <const int numnode, Core::FE::CellType distype>
void XFEM::XfluidTimeintBase::eval_shape_and_deriv(
    Core::Elements::Element* element,            /// pointer to element
    Core::LinAlg::Matrix<3, 1>& xi,              /// local coordinates of point w.r.t element
    Core::LinAlg::Matrix<3, 3>& xji,             /// inverse of jacobian
    Core::LinAlg::Matrix<numnode, 1>& shapeFcn,  /// shape functions at point
    Core::LinAlg::Matrix<3, numnode>&
        shapeFcnDerivXY,  /// derivatives of shape function w.r.t global coordinates xyz
    bool compute_deriv    /// shall derivatives and jacobian be computed
) const
{
  const int* elenodeids = element->node_ids();  // nodeids of element
  const int nsd = 3;                            // dimension

  // clear data that shall be filled
  shapeFcn.clear();
  shapeFcnDerivXY.clear();

  //-------------------------------------------------------
  Core::FE::shape_function_3d(
      shapeFcn, xi(0), xi(1), xi(2), distype);  // evaluate shape functions at xi

  if (compute_deriv)
  {
    Core::LinAlg::Matrix<nsd, numnode> nodecoords(
        Core::LinAlg::Initialization::set_zero);         // node coordinates of the element
    for (size_t nodeid = 0; nodeid < numnode; nodeid++)  // fill node coordinates
    {
      Core::Nodes::Node* currnode = discret_->g_node(elenodeids[nodeid]);
      for (int i = 0; i < nsd; i++) nodecoords(i, nodeid) = currnode->x()[i];
    }

    // add ale displacements to initial position for state np
    if (dispnp_ != nullptr)  // is ale?
    {
      int nen = element->num_node();
      int numdof = element->num_dof_per_node(*(element->nodes()[0]));

      std::vector<int> nds(nen, 0);
      Core::Elements::LocationArray la(1);
      element->location_vector(*discret_, nds, la, false);

      // extract local values of the global vectors
      std::vector<double> mydispnp = Core::FE::extract_values(*dispnp_, la[0].lm_);

      for (int inode = 0; inode < nen; ++inode)  // number of nodes
      {
        for (int idim = 0; idim < nsd; ++idim)  // number of dimensions
        {
          (nodecoords)(idim, inode) +=
              mydispnp[idim + (inode * numdof)];  // attention! dispnp vector has 3+1 dofs for
                                                  // displacement (the same as for (u,p))
        }
      }
    }

    // shape function derivatives w.r.t local coordinates
    Core::LinAlg::Matrix<3, numnode> shapeFcnDeriv;
    Core::FE::shape_function_3d_deriv1(shapeFcnDeriv, xi(0), xi(1), xi(2), distype);

    Core::LinAlg::Matrix<nsd, nsd> xjm(Core::LinAlg::Initialization::set_zero);  // jacobi matrix
    xjm.multiply_nt(shapeFcnDeriv, nodecoords);  // jacobian J = (dx/dxi)^T
    xji.clear();
    xji.invert(xjm);  // jacobian inverted J^(-1) = dxi/dx

    shapeFcnDerivXY.multiply(xji, shapeFcnDeriv);  // (dN/dx)^T = (dN/dxi)^T * J^(-T)
  }

  return;
};  // end template eval_shape_and_deriv



// explicit instantiations of this function...
template void XFEM::XfluidTimeintBase::eval_shape_and_deriv<8, Core::FE::CellType::hex8>(
    Core::Elements::Element*, Core::LinAlg::Matrix<3, 1>&, Core::LinAlg::Matrix<3, 3>&,
    Core::LinAlg::Matrix<8, 1>&, Core::LinAlg::Matrix<3, 8>&, bool) const;
template void XFEM::XfluidTimeintBase::eval_shape_and_deriv<20, Core::FE::CellType::hex20>(
    Core::Elements::Element*, Core::LinAlg::Matrix<3, 1>&, Core::LinAlg::Matrix<3, 3>&,
    Core::LinAlg::Matrix<20, 1>&, Core::LinAlg::Matrix<3, 20>&, bool) const;
template void XFEM::XfluidTimeintBase::eval_shape_and_deriv<4, Core::FE::CellType::tet4>(
    Core::Elements::Element*, Core::LinAlg::Matrix<3, 1>&, Core::LinAlg::Matrix<3, 3>&,
    Core::LinAlg::Matrix<4, 1>&, Core::LinAlg::Matrix<3, 4>&, bool) const;



/*------------------------------------------------------------------------------------------------*
 * add adjacent elements for a periodic boundary node                            winklmaier 05/11 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::add_pb_celements(
    const Core::Nodes::Node* node, std::vector<const Core::Elements::Element*>& eles) const
{
  FOUR_C_THROW("what to do in add_pb_celements?");

  const Core::Elements::Element* const* elements = node->elements();  // element around current node

  for (int iele = 0; iele < node->num_element(); iele++) eles.push_back(elements[iele]);

  // get pbcnode
  bool pbcnodefound = false;  // boolean indicating whether this node is a pbc node
  Core::Nodes::Node* pbcnode = nullptr;
  find_pbc_node(node, pbcnode, pbcnodefound);

  // add elements located around the coupled pbc node
  if (pbcnodefound)
  {
    // get adjacent elements of this node
    const Core::Elements::Element* const* pbcelements = pbcnode->elements();
    // add elements to list
    for (int iele = 0; iele < pbcnode->num_element(); iele++)  // = ptToNode->Elements();
    {
      eles.push_back(pbcelements[iele]);
    }
  }  // end if pbcnode true
}


/*------------------------------------------------------------------------------------------------*
 * find the PBC node                                                             winklmaier 05/11 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::find_pbc_node(
    const Core::Nodes::Node* node, Core::Nodes::Node*& pbcnode, bool& pbcnodefound) const
{
  const int nodegid = node->id();  // global node id
  pbcnodefound = false;            // boolean indicating whether this node is a pbc node
  int coupnodegid = -1;
  // loop all nodes with periodic boundary conditions (master nodes)
  for (std::map<int, std::vector<int>>::const_iterator pbciter = (*pbcmap_).begin();
      pbciter != (*pbcmap_).end(); ++pbciter)
  {
    if (pbciter->first == nodegid)  // node is a pbc master node
    {
      pbcnodefound = true;
      // coupled node is the (first) slave node
      coupnodegid = pbciter->second[0];
      if (pbciter->second.size() != 1)
        FOUR_C_THROW("this might need to be modified for more than 1 slave per master");
    }
    else
    {
      // loop all slave nodes
      for (size_t islave = 0; islave < pbciter->second.size(); islave++)
      {
        if (pbciter->second[islave] == nodegid)  // node is a pbc slave node
        {
          pbcnodefound = true;
          coupnodegid = pbciter->first;  // coupled node is the master node
        }
      }  // end loop over slave nodes
    }  // end if
  }  // end loop over pbc map

  // get pbcnode
  if (pbcnodefound) pbcnode = discret_->g_node(coupnodegid);
}  // end if pbcnode true}



/*------------------------------------------------------------------------------------------------*
 * reset a special state with another state in the data class                        schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::reset_state(
    TimeIntData::State oldState, TimeIntData::State newState) const
{
  for (std::vector<TimeIntData>::iterator data = timeIntData_->begin(); data != timeIntData_->end();
      data++)
  {
    if (data->state_ == oldState) data->state_ = newState;
  }

  return;
}  // end function resetState



/*------------------------------------------------------------------------------------------------*
 * clear the data of all nodes having a special state                                schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::clear_state(TimeIntData::State state  /// state of time int to clear
) const
{
  std::vector<TimeIntData>::iterator data;
  while (true)  // while loop over data to be cleared
  {
    for (data = timeIntData_->begin(); data != timeIntData_->end();
        data++)  // for loop over all data
    {
      if (data->state_ == state)
      {
        timeIntData_->erase(data);
        break;
      }
    }
    if (data == timeIntData_->end()) break;
  }  // end while loop over data to be cleared

  return;
}  // end function clear state



/*------------------------------------------------------------------------------------------------*
 * basic function sending data to dest and receiving data from source                schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::send_data(Core::Communication::PackBuffer& dataSend, int& dest,
    int& source, std::vector<char>& dataRecv) const
{
  std::vector<int> lengthSend(1, 0);
  lengthSend[0] = dataSend().size();
  int size_one = 1;

#ifdef FOUR_C_ENABLE_ASSERTIONS
  std::cout << "--- sending " << lengthSend[0] << " bytes: from proc " << myrank_ << " to proc "
            << dest << std::endl;
#endif

  // exporter for sending
  Core::Communication::Exporter exporter(discret_->get_comm());

  // send length of the data to be received ...
  MPI_Request req_length_data;
  int length_tag = 0;
  exporter.i_send(myrank_, dest, lengthSend.data(), size_one, length_tag, req_length_data);
  // ... and receive length
  std::vector<int> lengthRecv(1, 0);
  exporter.receive(source, length_tag, lengthRecv, size_one);
  exporter.wait(req_length_data);

  // send actual data ...
  int data_tag = 4;
  MPI_Request req_data;
  exporter.i_send(myrank_, dest, dataSend().data(), lengthSend[0], data_tag, req_data);

  // ... and receive data
  dataRecv.clear();
  dataRecv.resize(lengthRecv[0]);
  exporter.receive_any(source, data_tag, dataRecv, lengthRecv[0]);
  exporter.wait(req_data);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  std::cout << "--- receiving " << lengthRecv[0] << " bytes: to proc " << myrank_ << " from proc "
            << source << std::endl;
#endif
}  // end send_data



/*------------------------------------------------------------------------------------------------*
 * packing a node for parallel communication only with the basic nodal data                       *
 * without an underlying discretization fitting to the node's new prozessor          schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::pack_node(
    Core::Communication::PackBuffer& dataSend, Core::Nodes::Node& node) const
{
  const int nsd = 3;
  add_to_pack(dataSend, node.id());
  add_to_pack(dataSend, Core::LinAlg::Matrix<nsd, 1>(node.x().data()));
  add_to_pack(dataSend, node.owner());
}  // end packNode



/*------------------------------------------------------------------------------------------------*
 * unpacking a node after parallel communication only with the basic nodal data                   *
 * without an underlying discretization fitting to the node's new prozessor          schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidTimeintBase::unpack_node(
    Core::Communication::UnpackBuffer& buffer, Core::Nodes::Node& node) const
{
  const int nsd = 3;                    // dimension
  int id;                               // global id
  Core::LinAlg::Matrix<nsd, 1> coords;  // coordinates
  int owner;                            // processor

  extract_from_pack(buffer, id);
  extract_from_pack(buffer, coords);
  extract_from_pack(buffer, owner);

  if (owner == myrank_)  // real node with all data
  {
    node = *discret_->g_node(id);
  }
  else  // just id, coords and owner
  {
    std::vector<double> coordinates(nsd, 0.0);
    for (int dim = 0; dim < nsd; dim++) coordinates[dim] = coords(dim);

    Core::Nodes::Node newNode(id, coordinates, owner);
    node = newNode;
  }  // end if correct processor
}  // end function unpackNode



/*------------------------------------------------------------------------------------------------*
 * basic XFEM time-integration constructor for standard degrees of freedom           schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
XFEM::XfluidStd::XfluidStd(
    XFEM::XfluidTimeintBase& timeInt,  ///< time integration base class object
    const std::map<int, std::vector<Inpar::XFEM::XFluidTimeInt>>&
        reconstr_method,                      ///< reconstruction map for nodes and its dofsets
    Inpar::XFEM::XFluidTimeInt& timeIntType,  ///< type of time integration
    const std::shared_ptr<Core::LinAlg::Vector<double>> veln,  ///< velocity at time t^n in col map
    const double& dt,                                          ///< time step size
    const bool initialize                                      ///< is initialization?
    )
    : XFEM::XfluidTimeintBase::XfluidTimeintBase(timeInt),
      timeIntType_(timeIntType),
      veln_(veln),
      dt_(dt)
{
  if (initialize)
  {
    const int nsd = 3;                                            // dimension
    timeIntData_ = std::make_shared<std::vector<TimeIntData>>();  // vector containing all data
                                                                  // used for computation

    Core::LinAlg::Matrix<nsd, 1> dummyStartpoint;  // dummy startpoint for comparison
    for (int i = 0; i < nsd; i++) dummyStartpoint(i) = 777.777;

    //--------------------------------------------------------------------------------------
    // fill timeIntData_ structure with the data for the nodes which are marked for SEMILAGRANGEAN
    // reconstruction

#ifdef DEBUG_TIMINT_STD
    Core::IO::cout
        << "\n\t ---------------------------------------------------------------------- ";
    Core::IO::cout << "\n\t PREPARE SEMI-LAGRANGEAN time integration: fill timeIntData for nodes";
    Core::IO::cout << "\n\t ---------------------------------------------------------------------- "
                   << Core::IO::endl;
#endif

    // loop over processor nodes
    for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
    {
      Core::Nodes::Node* node = discret_->l_row_node(lnodeid);

      std::map<int, std::vector<Inpar::XFEM::XFluidTimeInt>>::const_iterator it =
          reconstr_method.find(node->id());
      if (it != reconstr_method.end())
      {
        // reconstruction methods just for marked dofsets
        for (size_t i = 0; i < (it->second).size(); i++)
        {
          if ((it->second)[i] == Inpar::XFEM::Xf_TimeInt_STD_by_SL)
          {
#ifdef DEBUG_TIMINT_STD
            Core::IO::cout << "\t * fill timeIntData for node " << node->Id() << " and dofset " << i
                           << Core::IO::endl;
#endif

            Core::LinAlg::Matrix<3, 1> nodedispnp(Core::LinAlg::Initialization::set_zero);
            if (dispnp_ != nullptr)  // is alefluid
            {
              //------------------------------------------------------- add ale disp
              // get node location vector, dirichlet flags and ownerships (discret, nds, la,
              // doDirichlet)
              std::vector<int> lm;
              std::vector<int> dofs;
              dofset_new_->dof(dofs, node, 0);  // dofs for standard dofset
              for (int j = 0; j < 4; ++j) lm.push_back(dofs[j]);

              Core::LinAlg::Matrix<1, 1> nodepredummy(Core::LinAlg::Initialization::set_zero);
              extract_nodal_values_from_vector<1>(nodedispnp, nodepredummy, dispnp_, lm);
            }

            // constructor for standard computation
            timeIntData_->push_back(TimeIntData(*node,  //! node for which SL-algorithm is called
                i,  //! nds (nodal dofset) number w.r.t new interface position, for which SL-algo is
                    //! called
                Core::LinAlg::Matrix<nsd, 1>(true),  //!  velocity at point x (=x_Lagr(t^n+1))
                std::vector<Core::LinAlg::Matrix<nsd, nsd>>(oldVectors_.size(),
                    Core::LinAlg::Matrix<nsd, nsd>(
                        true)),  //! velocity gradient at point x (=x_Lagr(t^n+1))
                std::vector<Core::LinAlg::Matrix<1, nsd>>(oldVectors_.size(),
                    Core::LinAlg::Matrix<1, nsd>(
                        true)),             //! pressure gradient at point x (=x_Lagr(t^n+1))
                nodedispnp,                 //!  displacement at point x (=x_Lagr(t^n+1))
                dummyStartpoint,            // dummy-startpoint
                1,                          // searchedProcs
                0,                          // counter
                INFINITY,                   // minimal distance
                TimeIntData::predictor_));  // data created for the node
          }
        }  // nodaldofsets
      }  // some dofsets have to be reconstructed
    }  // end loop over processor nodes

    //--------------------------------------------------------------------------------------
    // compute initial points on the right side of the interface to start finding the Lagrangean
    // origin
    startpoints();

    //--------------------------------------------------------------------------------------

    // test loop if all initial startpoints have been computed
    for (std::vector<TimeIntData>::iterator data = timeIntData_->begin();
        data != timeIntData_->end(); data++)
    {
      if (data->startpoint_ == dummyStartpoint)  // startpoint unchanged
        FOUR_C_THROW(
            "WARNING! No enriched node on one interface side found!\nThis "
            "indicates that the whole area is at one side of the interface!");
    }  // end loop over nodes
  }

  return;
}  // end constructor



/*------------------------------------------------------------------------------------------------*
 * out of order!                                                                     schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::compute(
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& newRowVectorsn)
{
  FOUR_C_THROW("Unused function! Use a function of the derived classes");
}



///*------------------------------------------------------------------------------------------------*
// * initialize data when called in a new FGI                                      winklmaier 10/11
// *
// *------------------------------------------------------------------------------------------------*/
// void XFEM::XFLUID_STD::importNewFGIData(
//    const std::shared_ptr<Core::FE::Discretization> discret,
//    const std::shared_ptr<XFEM::DofManager> newdofman,
//    const std::shared_ptr<COMBUST::FlameFront> flamefront,
//    const Core::LinAlg::Map& newdofrowmap,
//    const std::map<DofKey, DofGID>& newNodalDofRowDistrib)
//{
//  discret_ = discret;
//  newdofman_ = newdofman;
//  phinpi_ = phinp_;
//  phinp_ = flamefront->Phinp();
//  flamefront_ = flamefront;
//  newinterfacehandle_ = flamefront->InterfaceHandle();
//  newdofrowmap_ = newdofrowmap;
//  newNodalDofRowDistrib_ = newNodalDofRowDistrib;
//  return;
//}



/*------------------------------------------------------------------------------------------------*
 * identify an element containing a point and additional data                        schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::element_search(
    Core::Elements::Element*& ele,   /// pointer to element if point lies in a found element
    Core::LinAlg::Matrix<3, 1>& x,   /// global coordinates of point
    Core::LinAlg::Matrix<3, 1>& xi,  /// determined local coordinates w.r.t ele
    bool& found                      /// is element found?
) const
{
  // REMARK: if ele!= nullptr, then check that element first, before loop all row elements

  int startid;  // local row element id
  if (ele == nullptr)
    startid = 0;  // start with first local row element
  else
    startid =
        -1;  // pseudo-id so that id+1 will be 0 (for additional element check, if ele!=nullptr)

  Core::Elements::Element* currele = nullptr;  // current element

  // loop over elements
  for (int ieleid = startid; ieleid < discret_->num_my_row_elements(); ieleid++)
  {
    // if ele != nullptr additional check
    // first it should be checked if it is fitting
    if (ieleid == -1)
    {
      currele = ele;
      ele = nullptr;  // reset ele, ele will be set if an element is found finally
    }
    else
      currele = discret_->l_row_element(ieleid);

    call_x_to_xi_coords(currele, x, xi, "dispn", found);

    if (found)
    {
      ele = currele;
      break;
    }
  }  // end loop over processor elements

  return;
}  // end elementSearch



/*------------------------------------------------------------------------------------------------*
 * interpolate velocity and derivatives for a point in an element                    schott 06/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::get_gp_values(Core::Elements::Element* ele,  ///< pointer to element
    Core::LinAlg::Matrix<3, 1>& xi,          ///< local coordinates of point w.r.t element
    std::vector<int>& nds,                   ///< nodal dofset of point for elemental nodes
    XFEM::XFEMDofSet& dofset,                ///< XFEM dofset
    Core::LinAlg::Matrix<3, 1>& vel,         ///< determine velocity at point
    Core::LinAlg::Matrix<3, 3>& vel_deriv,   ///< determine velocity derivatives at point
    double& pres,                            ///< pressure
    Core::LinAlg::Matrix<1, 3>& pres_deriv,  ///< pressure gradient
    std::shared_ptr<const Core::LinAlg::Vector<double>>
        vel_vec,        ///< vector used for interpolating at gp
    bool compute_deriv  ///< shall derivatives be computed?
) const
{
  switch (ele->shape())
  {
    case Core::FE::CellType::hex8:
      get_gp_values_t<Core::FE::CellType::hex8>(
          ele, xi, nds, dofset, vel, vel_deriv, pres, pres_deriv, vel_vec, compute_deriv);
      break;
    case Core::FE::CellType::hex20:
      get_gp_values_t<Core::FE::CellType::hex20>(
          ele, xi, nds, dofset, vel, vel_deriv, pres, pres_deriv, vel_vec, compute_deriv);
      break;
    case Core::FE::CellType::tet4:
      get_gp_values_t<Core::FE::CellType::tet4>(
          ele, xi, nds, dofset, vel, vel_deriv, pres, pres_deriv, vel_vec, compute_deriv);
      break;
    default:
      FOUR_C_THROW("add your 3D distype here!");
      break;
  }  // end switch

  return;
}  // end function getGPValues


//! interpolate velocity and derivatives for a point in an element
template <Core::FE::CellType distype>
void XFEM::XfluidStd::get_gp_values_t(Core::Elements::Element* ele,  ///< pointer to element
    Core::LinAlg::Matrix<3, 1>& xi,          ///< local coordinates of point w.r.t element
    std::vector<int>& nds,                   ///< nodal dofset of point for elemental nodes
    XFEM::XFEMDofSet& dofset,                ///< XFEM dofset
    Core::LinAlg::Matrix<3, 1>& vel,         ///< determine velocity at point
    Core::LinAlg::Matrix<3, 3>& vel_deriv,   ///< determine velocity derivatives at point
    double& pres,                            ///< pressure
    Core::LinAlg::Matrix<1, 3>& pres_deriv,  ///< pressure gradient
    std::shared_ptr<const Core::LinAlg::Vector<double>>
        vel_vec,        ///< vector used for interpolating at gp
    bool compute_deriv  ///< shall derivatives be computed?
) const
{
  if (vel_vec == nullptr) FOUR_C_THROW("vector is not filled");

  const int nsd = 3;  // dimension

  const int numdofpernode = nsd + 1;
  const int numnode = Core::FE::num_nodes<distype>;  // number of element
                                                     // nodes

  //-------------------------------------------------------
  // initialization
  Core::LinAlg::Matrix<numnode, 1> shapeFcn(
      Core::LinAlg::Initialization::set_zero);  /// shape function at point
  Core::LinAlg::Matrix<nsd, numnode> shapeFcnDeriv(
      Core::LinAlg::Initialization::set_zero);  /// xyz shape derivatives at point
  Core::LinAlg::Matrix<nsd, nsd> xji(
      Core::LinAlg::Initialization::set_zero);  /// inverse of jacobian

  //-------------------------------------------------------
  // get element location vector, dirichlet flags and ownerships (discret, nds, la, doDirichlet)
  if ((int)(nds.size()) != numnode)
    FOUR_C_THROW("size of nds-vector ({}) != numnode({})", nds.size(), numnode);

  std::vector<int> lm;

  for (int inode = 0; inode < numnode; inode++)
  {
    Core::Nodes::Node* node = ele->nodes()[inode];
    std::vector<int> dofs;
    dofset.dof(dofs, node, nds[inode]);

    for (int j = 0; j < 4; ++j)
    {
      lm.push_back(dofs[j]);
    }
  }

  //-------------------------------------------------------
  // get element-wise velocity/pressure field
  Core::LinAlg::Matrix<nsd, numnode> evel(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<numnode, 1> epre(Core::LinAlg::Initialization::set_zero);


  // extract local values of the global vectors
  std::vector<double> mymatrix = Core::FE::extract_values(*vel_vec, lm);

  for (int inode = 0; inode < numnode; ++inode)  // number of nodes
  {
    for (int idim = 0; idim < nsd; ++idim)  // number of dimensions
    {
      (evel)(idim, inode) = mymatrix[idim + (inode * numdofpernode)];
    }
    (epre)(inode, 0) = mymatrix[nsd + (inode * numdofpernode)];
  }


  //-------------------------------------------------------
  eval_shape_and_deriv<numnode, distype>(ele, xi, xji, shapeFcn, shapeFcnDeriv, compute_deriv);

  //-------------------------------------------------------
  // interpolate velocity and pressure values at starting point
  vel.clear();  // set to zero
  vel.multiply(evel, shapeFcn);

  pres = epre.dot(shapeFcn);

  if (compute_deriv)
  {
    vel_deriv.multiply_nt(evel, shapeFcnDeriv);
    pres_deriv.multiply_tt(epre, shapeFcnDeriv);
  }

  return;
};  // end template getGPValuesT



/*------------------------------------------------------------------------------------------------*
 * Compute starting values for nodes marked for semi-lagrangian algo                 schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::startpoints()
{
#ifdef DEBUG_TIMINT_STD
  Core::IO::cout << "\n\t ---------------------------------------------------------------------- ";
  Core::IO::cout << "\n\t compute initial start points for finding the Lagrangean origin";
  Core::IO::cout << "\n\t ---------------------------------------------------------------------- "
                 << Core::IO::endl;
#endif

  // REMARK: we do not need parallel communication for start point values because
  // structural surface is ghosted on all procs

  // loop over nodes which changed interface side
  for (std::vector<TimeIntData>::iterator data = timeIntData_->begin(); data != timeIntData_->end();
      data++)
  {
    if (data->state_ == TimeIntData::basicStd_)  // correct state
    {
      // find a start approximation for Semi-Lagrange backtracking
      // this start approximation can be found by tracking back the projection of current node
      // along structural movement
      project_and_trackback(*data);

    }  // end if correct state
  }  // end loop over nodes which changed interface side


  return;
}  // end startpoints()



/*------------------------------------------------------------------------------------------------*
 * Project the current point onto the structural surface, track it back along structural         *
 * movement and project it back into the fluid domain                                schott 06/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::project_and_trackback(TimeIntData& data)
{
#ifdef DEBUG_TIMINT_STD
  Core::IO::cout << "\n\t * project_and_trackback for node " << data.node_.Id() << Core::IO::endl;
#endif

  const std::string state = "idispnp";

  const int nsd = 3;
  Core::LinAlg::Matrix<nsd, 1> newNodeCoords(
      data.node_.x().data());  // coords of endpoint of Lagrangian characteristics
  for (int i = 0; i < nsd; ++i) newNodeCoords(i) += data.dispnp_(i);

  // determine the smallest distance of node at t^(n+1) to the current side elements

  Cut::Node* n_new = wizard_new_->get_node(data.node_.id());

  //------------------------------------
  // find all involved nodes and sides (edges) for computing distance to node

  std::set<int> points;
  std::set<int> sides;

  // is there a nodehandle to get information about cutting sides near this node?
  if (n_new != nullptr)
  {
    //--------------------------------------------------------
    // get involved side ids for projection and distance computation
    //--------------------------------------------------------

    // set of side-ids involved in cutting the current connection of volumecells at t^(n+1)
    std::map<int, std::vector<Cut::BoundaryCell*>> bcells_new;

    // the std-set
    const std::set<Cut::plain_volumecell_set,
        Cut::Cmp>& cell_set_new = n_new->get_nodal_dof_set(data.nds_np_)
                                      ->volume_cell_composite();  // always the standard dofset

    // get all side-ids w.r.t to all volumecells contained in current new set around the current
    // node
    for (std::set<Cut::plain_volumecell_set>::const_iterator adj_eles = cell_set_new.begin();
        adj_eles != cell_set_new.end(); adj_eles++)
    {
      const Cut::plain_volumecell_set ele_vc = *adj_eles;

      for (Cut::plain_volumecell_set::const_iterator vcs = ele_vc.begin(); vcs != ele_vc.end();
          vcs++)
      {
        Cut::VolumeCell* vc = *vcs;

        // get sides involved in creation boundary cells (std::map<sideId,bcells>)
        vc->get_boundary_cells(bcells_new);
      }
    }

    //--------------------------------------------------------
    // distance computation w.r.t surface involved in cutting the adjacent elements(n_new to
    // sides/edges/nodes)
    //--------------------------------------------------------

    // loop bcs and extract sides and nodes
    for (std::map<int, std::vector<Cut::BoundaryCell*>>::iterator bcells_it = bcells_new.begin();
        bcells_it != bcells_new.end(); bcells_it++)
    {
      int sid = bcells_it->first;
      sides.insert(sid);

      // get the side
      Core::Elements::Element* side = boundarydis_->g_element(sid);
      int numnode = side->num_node();
      const int* side_nodes = side->node_ids();

      // get the nodes
      for (int i = 0; i < numnode; i++)
      {
        points.insert(side_nodes[i]);
      }
    }
  }

  if ((points.size() == 0 and sides.size() == 0) or n_new == nullptr)
  {
    // node does not carry a node handle -> Semilagrangean algorithm for nodes with too much
    // structural movement
    // TODO: can this improved?!
    // -> compute projection for all sides and nodes, that's really expensive!

    // loop all column sides and extract sides and nodes
    for (int i = 0; i < boundarydis_->num_my_col_elements(); i++)
    {
      // get the side
      Core::Elements::Element* side = boundarydis_->l_col_element(i);
      int numnode = side->num_node();
      const int* side_nodes = side->node_ids();

      sides.insert(side->id());

      // get the nodes
      for (int i = 0; i < numnode; i++)
      {
        points.insert(side_nodes[i]);
      }
    }
  }

  if (points.size() == 0 and sides.size() == 0)
    FOUR_C_THROW("there are no cutting sides for node {}, that cannot be anymore", data.node_.id());

  //-------------------------------------
  // initialize data holding information about minimal distance

  // smallest distance
  Core::LinAlg::Matrix<3, 1> proj_x_np(
      Core::LinAlg::Initialization::set_zero);  ///< projected point at t^(n+1)
  Core::LinAlg::Matrix<3, 1> proj_x_n(
      true);  ///< projected point at t^n (tracked back along structural movement)
  Core::LinAlg::Matrix<3, 1> start_point(
      Core::LinAlg::Initialization::set_zero);  ///< final start point for SemiLagrange algo
  int proj_sid = -1;                            ///< id of side that contains the projected point
  std::map<std::vector<int>, std::vector<int>>
      proj_lineid;  ///< std::map< sorted nids, global side IDs >
  // smallest distance w.r.t side
  Core::LinAlg::Matrix<2, 1> proj_xi_side(
      true);  ///< local coordinates of projected point if projection w.r.t side
  // smallest distance w.r.t line
  std::map<std::vector<int>, std::vector<double>>
      proj_xi_line;  ///< std::map<sorted nids,local line coordinates w.r.t lines of different
                     ///< sides>
  std::map<std::vector<int>, std::vector<int>> proj_nid_line;  ///< std::map<sorted nids, sideids>
  // smallest distance w.r.t point
  int proj_nid_np = -1;  ///< nid of projected point if projection w.r.t point (structural node)

  double min_dist = INFINITY;  ///< minimal distance

  data.proj_ = TimeIntData::failed_;  ///< projection method yielding minimal distance

  //----------------------------------------------------------------------
  // check distance to sides and its lines
  //----------------------------------------------------------------------
  for (std::set<int>::iterator it = sides.begin(); it != sides.end(); it++)
  {
    //-------------------------------------------------

    Core::Elements::Element* side = boundarydis_->g_element(*it);

    call_project_on_side(
        side, state, newNodeCoords, proj_sid, min_dist, proj_x_np, proj_xi_side, data.proj_);

    //------------------------------------
    // edges are checked twice (w.r.t both sides)
    //------------------------------------

    // loop all edges of current side
    std::vector<std::shared_ptr<Core::Elements::Element>> lines = side->lines();

    int line_count = 0;

    for (std::vector<std::shared_ptr<Core::Elements::Element>>::iterator line_it = lines.begin();
        line_it != lines.end(); line_it++)
    {
      call_project_on_line(side, &(**line_it),
          line_count,  ///< local line id w.r.t side element
          state,
          newNodeCoords,  ///< node coordinates of point that has to be projected
          min_dist,       ///< minimal distance, potentially updated
          proj_x_np,      ///< projection of point on this side
          proj_xi_line,   ///< std::map<side ID, local coordinates of projection of point w.r.t to
                          ///< this line>
          proj_lineid,    ///< std::map<side ID, local line id>
          proj_nid_line,  ///< std::map<side ID, vec<line Ids>>
          proj_sid,       ///< id of side that contains the projected point
          data.proj_      ///< reference to data
      );

      line_count++;
    }  // loop lines of side
  }  // loop sides


  //------------------------------------
  // check minimal distance w.r.t points (structural nodes)
  //------------------------------------
  // loop all points
  for (std::set<int>::iterator it = points.begin(); it != points.end(); it++)
  {
    // compute distance to following point
    Core::Nodes::Node* node = boundarydis_->g_node(*it);

    call_project_on_point(node,  ///< pointer to node
        state,
        newNodeCoords,  ///< node coordinates of point that has to be projected
        min_dist,       ///< minimal distance, potentially updated
        proj_nid_np,    ///< nid id of projected point on surface
        proj_x_np,      ///< projection of point on this side
        proj_sid,       ///< id of side that contains the projected point
        proj_xi_line,  ///< std::map<side ID, local coordinates of projection of point w.r.t to this
                       ///< line>
        proj_lineid,   ///< std::map<side ID, local line id>
        proj_nid_line,  ///< std::map<side ID, vec<line Ids>>
        data.proj_      ///< reference to data
    );


  }  // loop points


  if (data.proj_ == TimeIntData::failed_)
    FOUR_C_THROW("projection of node {} not successful!", n_new->id());

#ifdef DEBUG_TIMINT_STD
  Core::IO::cout << "\n\t => Projection of node lies on (side=0, line=1, point=2, failed=3): "
                 << data.proj_ << " with distance " << min_dist << Core::IO::endl;
#endif


  // track back the projected points on structural surface from t^(n+1)->t^n
  if (data.proj_ == TimeIntData::onSide_)
  {
    Core::Elements::Element* side = boundarydis_->g_element(proj_sid);

    if (side == nullptr) FOUR_C_THROW("side with id {} not found ", proj_sid);

    // side geometry at initial state t^0
    const int numnodes = side->num_node();
    Core::Nodes::Node** nodes = side->nodes();
    Core::LinAlg::SerialDenseMatrix side_xyze(3, numnodes);
    for (int i = 0; i < numnodes; ++i)
    {
      const double* x = nodes[i]->x().data();
      std::copy(x, x + 3, &side_xyze(0, i));
    }

    Core::Elements::LocationArray cutla(1);
    side->location_vector(*boundarydis_, cutla, false);

    compute_start_point_side(
        side, side_xyze, cutla[0].lm_, proj_xi_side, min_dist, proj_x_n, start_point);
  }
  else if (data.proj_ == TimeIntData::onLine_)
  {
    // check if both lines are identical
    for (std::map<std::vector<int>, std::vector<int>>::iterator line = proj_nid_line.begin();
        line != proj_nid_line.end(); line++)
    {
      std::vector<int>& sides = line->second;

      std::vector<int>& local_lineIds = proj_lineid.find(line->first)->second;
      std::vector<double>& local_xi_line = proj_xi_line.find(line->first)->second;


      int sid_1 = -1;
      int sid_2 = -1;

      if (sides.size() <= 0)
      {
        FOUR_C_THROW(
            "there must be at least one found side adjacent to this line, but there are {} sides",
            sides.size());
      }
      else if (sides.size() == 1)
      {
        sid_1 = sides[0];

        // find the second neighboring side
        Core::Elements::Element* side_1 = boundarydis_->g_element(sid_1);

        Core::Nodes::Node** nodes = side_1->nodes();

        std::set<int> neighbor_sides;

        for (int i = 0; i < side_1->num_node(); i++)
        {
          Core::Elements::Element** surr_sides = nodes[i]->elements();
          for (int s = 0; s < nodes[i]->num_element(); s++)
          {
            neighbor_sides.insert(surr_sides[s]->id());
          }
        }

        // check which neighbor side is the right one
        for (std::set<int>::iterator s = neighbor_sides.begin(); s != neighbor_sides.end(); s++)
        {
          Core::Elements::Element* side_tmp = boundarydis_->g_element(*s);

          std::vector<std::shared_ptr<Core::Elements::Element>> lines_tmp = side_tmp->lines();

          for (std::vector<std::shared_ptr<Core::Elements::Element>>::iterator line_it =
                   lines_tmp.begin();
              line_it != lines_tmp.end(); line_it++)
          {
            Core::Nodes::Node** line_nodes = (*line_it)->nodes();
            std::vector<int> line_nids;
            for (int i = 0; i < (*line_it)->num_node(); ++i)
            {
              line_nids.push_back(line_nodes[i]->id());
            }

            sort(line_nids.begin(), line_nids.end());

            if (line_nids == line->first)
            {
              // side found
              if (sid_2 == -1 and side_tmp->id() != sid_1) sid_2 = side_tmp->id();
              break;
            }
          }
        }
        if (sid_2 == -1)
        {
#ifdef DEBUG_TIMINT_STD
          std::cout
              << "\t\t\t\tneighbor side not found, current line seems to be a line at the boundary"
              << std::endl;
          std::cout << "\t\t\t\t -> the same distance should have been found for a side ? "
                    << std::endl;
          std::cout << "line nodeids: " << std::endl;
          for (int i = 0; i < (int)(line->first).size(); i++)
            std::cout << "\t nid: " << (line->first)[i] << std::endl;
          std::cout << "\t\t\t\In case your simulation is a pseudo 2d case, every thing is fine "
                       "and we will just use"
                    << std::endl;
          std::cout << "\t\t\t\tthe normal on the first side!" << std::endl;
#endif
        }
      }
      else if (sides.size() == 2)
      {
        sid_1 = sides[0];
        sid_2 = sides[1];
      }
      else
        FOUR_C_THROW("non valid number of sides adjacent to line");

      //---------------------------------------------------------
      // side 1
      Core::Elements::Element* side_1 = boundarydis_->g_element(sid_1);

      // side geometry at initial state t^0
      const int numnodes_1 = side_1->num_node();
      Core::Nodes::Node** nodes_1 = side_1->nodes();
      Core::LinAlg::SerialDenseMatrix side_xyze_1(3, numnodes_1);
      for (int i = 0; i < numnodes_1; ++i)
      {
        const double* x = nodes_1[i]->x().data();
        std::copy(x, x + 3, &side_xyze_1(0, i));
      }

      Core::Elements::LocationArray cutla_1(1);
      side_1->location_vector(*boundarydis_, cutla_1, false);

      //---------------------------------------------------------
      // side 2
      Core::Elements::Element* side_2;
      Core::LinAlg::SerialDenseMatrix side_xyze_2;
      Core::Elements::LocationArray cutla_2(1);
      if (sid_2 != -1)
      {
        side_2 = boundarydis_->g_element(sid_2);
        // side geometry at initial state t^0
        const int numnodes_2 = side_2->num_node();
        Core::Nodes::Node** nodes_2 = side_2->nodes();
        side_xyze_2.shape(3, numnodes_2);
        for (int i = 0; i < numnodes_2; ++i)
        {
          const double* x = nodes_2[i]->x().data();
          std::copy(x, x + 3, &side_xyze_2(0, i));
        }
        side_2->location_vector(*boundarydis_, cutla_2, false);
      }
      else
      {
        side_2 = nullptr;
      }
      //---------------------------------------------------------
      // line geometry at initial state t^0
      std::vector<std::shared_ptr<Core::Elements::Element>> lines = side_1->lines();

      std::shared_ptr<Core::Elements::Element> line_ele = lines[local_lineIds[0]];

      call_get_projxn_line(side_1, &*line_ele, proj_x_n, local_xi_line[0]);


      compute_start_point_line(side_1,  ///< pointer to side element
          side_xyze_1,                  ///< side's node coordinates
          side_2,                       ///< pointer to side element
          side_xyze_2,                  ///< side's node coordinates
          cutla_1[0].lm_,               ///< local map
          cutla_2[0].lm_,               ///< local map
          min_dist,                     ///< distance from point to its projection
          proj_x_n,                     ///< projected point at t^n
          start_point                   ///< final start point
      );
    }
  }
  else if (data.proj_ == TimeIntData::onPoint_)
  {
    std::vector<Core::Elements::Element*> surr_sides;              ///< pointer to side element
    std::vector<Core::LinAlg::SerialDenseMatrix> surr_sides_xyze;  ///< side's node coordinates
    std::vector<std::vector<int>> surr_sides_lm;                   ///< local map


    // get the node
    Core::Nodes::Node* node = boundarydis_->g_node(proj_nid_np);

    // get all surrounding sides
    Core::Elements::Element** node_sides = node->elements();

    for (int s = 0; s < node->num_element(); s++)
    {
      // side geometry at initial state t^0
      Core::Elements::Element* side = node_sides[s];
      const int numnodes = side->num_node();
      Core::Nodes::Node** nodes = side->nodes();
      Core::LinAlg::SerialDenseMatrix side_xyze(3, numnodes);
      for (int i = 0; i < numnodes; ++i)
      {
        const double* x = nodes[i]->x().data();
        std::copy(x, x + 3, &side_xyze(0, i));
      }

      Core::Elements::LocationArray cutla(1);
      side->location_vector(*boundarydis_, cutla, false);

      surr_sides.push_back(side);
      surr_sides_xyze.push_back(side_xyze);
      surr_sides_lm.push_back(cutla[0].lm_);
    }

    //----------------------------------------------------

    // its point geometry
    Core::LinAlg::SerialDenseMatrix point_xyze(3, 1);

    const double* x = node->x().data();
    std::copy(x, x + 3, &point_xyze(0, 0));

    std::vector<int> lm = boundarydis_->dof(0, node);

    // get the coordinates of the projection point at time tn
    const std::string state = "idispn";

    std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
        boundarydis_->get_state(state);
    if (matrix_state == nullptr) FOUR_C_THROW("Cannot get state vector {}", state.c_str());

    // extract local values of the global vectors
    std::vector<double> mymatrix = Core::FE::extract_values(*matrix_state, lm);

    // add the displacement of the interface
    for (int idim = 0; idim < 3; ++idim)  // number of dimensions
    {
      point_xyze(idim, 0) += mymatrix[idim];  // attention! disp state vector has 3+1 dofs for
                                              // displacement (the same as for (u,p))
    }

    // fill Core::LINALG matrix
    for (int idim = 0; idim < 3; ++idim)  // number of dimensions
    {
      proj_x_n(idim) = point_xyze(idim,
          0);  // attention! disp state vector has 3+1 dofs for displacement (the same as for (u,p))
    }

    //----------------------------------------------------


    compute_start_point_avg(surr_sides,  ///< pointer to side element
        surr_sides_xyze,                 ///< side's node coordinates
        surr_sides_lm,                   ///< local map
        min_dist,                        ///< distance from point to its projection
        proj_x_n,                        ///< projected point at t^n
        start_point                      ///< final start point
    );
  }
  else
    FOUR_C_THROW("unknown projection method for tracking back the structural surface point");


  /// set the data
  if (data.proj_ != TimeIntData::failed_)
  {
    // fill data
    //  data.startpoint_.update(1.0,newNodeCoords,-dt_,nodevel);
    data.startpoint_.update(1.0, start_point, 0.0);
    data.initialpoint_.update(1.0, start_point, 0.0);
    data.initial_eid_ = -1;
    data.initial_ele_owner_ = -1;
    data.dMin_ = min_dist;
  }
}

bool XFEM::XfluidStd::find_nearest_surf_point(
    Core::LinAlg::Matrix<3, 1>& x,       ///< coords of point to be projected
    Core::LinAlg::Matrix<3, 1>& proj_x,  ///< coords of projected point
    Cut::VolumeCell* vc,                 ///< volumcell on that's cut-surface we want to project
    const std::string state              ///< state n or np?
)
{
  if (vc == nullptr)
    FOUR_C_THROW("do not call find_nearest_surf_point with Null-Pointer for Volumecell");

  //--------------------------------------------------------
  // get involved side ids for projection and distance computation
  //--------------------------------------------------------

  // set of side-ids involved in cutting the current connection of volumecells at t^(n+1)
  std::map<int, std::vector<Cut::BoundaryCell*>> bcells_new;

  // get sides involved in creation boundary cells (std::map<sideId,bcells>)
  vc->get_boundary_cells(bcells_new);



  std::set<int> points;
  std::set<int> sides;

  // loop bcs and extract sides and nodes
  for (std::map<int, std::vector<Cut::BoundaryCell*>>::iterator bcells_it = bcells_new.begin();
      bcells_it != bcells_new.end(); bcells_it++)
  {
    int sid = bcells_it->first;
    sides.insert(sid);

    // get the side
    Core::Elements::Element* side = boundarydis_->g_element(sid);
    int numnode = side->num_node();
    const int* side_nodes = side->node_ids();

    // get the nodes
    for (int i = 0; i < numnode; i++)
    {
      points.insert(side_nodes[i]);
    }
  }


  return project_to_surface(x, proj_x, state, points, sides);
}


/*------------------------------------------------------------------------------------------------*
 * Project the current point onto the structural surface given by point and side Ids schott 06/12
 **
 *------------------------------------------------------------------------------------------------*/
bool XFEM::XfluidStd::project_to_surface(
    Core::LinAlg::Matrix<3, 1>& x,       ///< coords of point to be projected
    Core::LinAlg::Matrix<3, 1>& proj_x,  ///< coords of projected point
    const std::string state,             ///< state n or np?
    std::set<int>& points,  ///< node Ids of surface for that distance has to be computed
    std::set<int>&
        sides  ///< side Ids of surface for that and it's lines the distances have to be computed
)
{
#ifdef DEBUG_TIMINT_STD
  Core::IO::cout << "\n\t * find_nearest_surf_point for point " << x << Core::IO::endl;
#endif

  if (points.size() == 0 and sides.size() == 0)
    FOUR_C_THROW(
        "there are no cutting sides around current point ({},{},{}), Projection on surface not "
        "possible here",
        x(0), x(1), x(2));

  //-------------------------------------
  // initialize data holding information about minimal distance

  // smallest distance
  int proj_sid = -1;  ///< id of side that contains the projected point
  std::map<std::vector<int>, std::vector<int>>
      proj_lineid;  ///< std::map< sorted nids, global side IDs >
  // smallest distance w.r.t side
  Core::LinAlg::Matrix<2, 1> proj_xi_side(
      true);  ///< local coordinates of projected point if projection w.r.t side
  // smallest distance w.r.t line
  std::map<std::vector<int>, std::vector<double>>
      proj_xi_line;  ///< std::map<sorted nids,local line coordinates w.r.t lines of different
                     ///< sides>
  std::map<std::vector<int>, std::vector<int>> proj_nid_line;  ///< std::map<sorted nids, sideids>
  // smallest distance w.r.t point
  int proj_nid_np = -1;  ///< nid of projected point if projection w.r.t point (structural node)

  double min_dist = INFINITY;  ///< minimal distance

  TimeIntData::Projection proj =
      TimeIntData::failed_;  ///< projection method yielding minimal distance

  //----------------------------------------------------------------------
  // check distance to sides and its lines
  //----------------------------------------------------------------------
  for (std::set<int>::iterator it = sides.begin(); it != sides.end(); it++)
  {
    //-------------------------------------------------

    Core::Elements::Element* side = boundarydis_->g_element(*it);

    call_project_on_side(side, state, x, proj_sid, min_dist, proj_x, proj_xi_side, proj);

    //------------------------------------
    // edges are checked twice (w.r.t both sides)
    //------------------------------------

    // loop all edges of current side
    std::vector<std::shared_ptr<Core::Elements::Element>> lines = side->lines();

    int line_count = 0;

    for (std::vector<std::shared_ptr<Core::Elements::Element>>::iterator line_it = lines.begin();
        line_it != lines.end(); line_it++)
    {
      call_project_on_line(side, &(**line_it),
          line_count,  ///< local line id w.r.t side element
          state,
          x,              ///< node coordinates of point that has to be projected
          min_dist,       ///< minimal distance, potentially updated
          proj_x,         ///< projection of point on this side
          proj_xi_line,   ///< std::map<side ID, local coordinates of projection of point w.r.t to
                          ///< this line>
          proj_lineid,    ///< std::map<side ID, local line id>
          proj_nid_line,  ///< std::map<side ID, vec<line Ids>>
          proj_sid,       ///< id of side that contains the projected point
          proj            ///< reference to data
      );

      line_count++;
    }  // loop lines of side
  }  // loop sides


  //------------------------------------
  // check minimal distance w.r.t points (structural nodes)
  //------------------------------------
  // loop all points
  for (std::set<int>::iterator it = points.begin(); it != points.end(); it++)
  {
    // compute distance to following point
    Core::Nodes::Node* node = boundarydis_->g_node(*it);

    call_project_on_point(node,  ///< pointer to node on which we want to project
        state,
        x,              ///< node coordinates of point that has to be projected
        min_dist,       ///< minimal distance, potentially updated
        proj_nid_np,    ///< nid id of projected point on surface
        proj_x,         ///< projection of point on this side
        proj_sid,       ///< id of side that contains the projected point
        proj_xi_line,   ///< std::map<side ID, local coordinates of projection of point w.r.t to
                        ///< this line>
        proj_lineid,    ///< std::map<side ID, local line id>
        proj_nid_line,  ///< std::map<side ID, vec<line Ids>>
        proj            ///< reference to data
    );


  }  // loop points


  if (proj == TimeIntData::failed_)
    FOUR_C_THROW("projection of point ({},{},{}) not successful!", x(0), x(1), x(2));

#ifdef DEBUG_TIMINT_STD
  Core::IO::cout << "\n\t => Projection of node lies on (side=0, line=1, point=2, failed=3): "
                 << proj << " with distance " << min_dist << " coords: " << proj_x
                 << Core::IO::endl;
#endif

  return (proj != TimeIntData::failed_);
}



void XFEM::XfluidStd::compute_start_point_side(
    Core::Elements::Element* side,               ///< pointer to side element
    Core::LinAlg::SerialDenseMatrix& side_xyze,  ///< side's node coordinates
    const std::vector<int>& lm,                  ///< local map
    Core::LinAlg::Matrix<2, 1>& xi_side,     ///< local coordinates of projected point w.r.t side
    double& dist,                            ///< distance from point to its projection
    Core::LinAlg::Matrix<3, 1>& proj_x_n,    ///< projected point at t^n
    Core::LinAlg::Matrix<3, 1>& start_point  ///< final start point
)
{
  Core::LinAlg::Matrix<3, 1> normal(Core::LinAlg::Initialization::set_zero);

  callget_normal_side_tn(side, normal, side_xyze, lm, proj_x_n, xi_side);

  // map point into fluid domain along normal vector
  start_point.update(1.0, proj_x_n, dist, normal);

  return;
}

void XFEM::XfluidStd::compute_start_point_line(
    Core::Elements::Element* side1,               ///< pointer to side element
    Core::LinAlg::SerialDenseMatrix& side1_xyze,  ///< side's node coordinates
    Core::Elements::Element* side2,               ///< pointer to side element
    Core::LinAlg::SerialDenseMatrix& side2_xyze,  ///< side's node coordinates
    const std::vector<int>& lm1,                  ///< local map
    const std::vector<int>& lm2,                  ///< local map
    double& dist,                                 ///< distance from point to its projection
    Core::LinAlg::Matrix<3, 1>& proj_x_n,         ///< projected point at t^n
    Core::LinAlg::Matrix<3, 1>& start_point       ///< final start point
)
{
  Core::LinAlg::Matrix<3, 1> normal_avg(
      Core::LinAlg::Initialization::set_zero);  // averaged normal vector
  Core::LinAlg::Matrix<3, 1> normal1(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> normal2(Core::LinAlg::Initialization::set_zero);

  Core::LinAlg::Matrix<2, 1> xi_side1(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<2, 1> xi_side2(Core::LinAlg::Initialization::set_zero);


  Core::LinAlg::Matrix<3, 1> xi_1_avg(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> xi_2_avg(Core::LinAlg::Initialization::set_zero);

  Core::LinAlg::Matrix<3, 1> proj_x_n_dummy1(Core::LinAlg::Initialization::set_zero);


  for (int i = 0; i < side1->num_node(); i++)
    xi_1_avg.update(1.0, Core::FE::get_node_coordinates(i, side1->shape()), 1.0);

  xi_1_avg.scale(1.0 / side1->num_node());

  xi_side1(0, 0) = xi_1_avg(0, 0);
  xi_side1(1, 0) = xi_1_avg(1, 0);

  callget_normal_side_tn(side1, normal1, side1_xyze, lm1, proj_x_n_dummy1, xi_side1);

  normal_avg.update(1.0, normal1, 1.0);

  if (side2 != nullptr)  // in case we have side2, use averaged normal
  {
    for (int i = 0; i < side2->num_node(); i++)
      xi_2_avg.update(1.0, Core::FE::get_node_coordinates(i, side2->shape()), 1.0);

    xi_2_avg.scale(1.0 / side2->num_node());

    xi_side2(0, 0) = xi_2_avg(0, 0);
    xi_side2(1, 0) = xi_2_avg(1, 0);

    callget_normal_side_tn(side2, normal2, side2_xyze, lm2, proj_x_n_dummy1, xi_side2);

    normal_avg.update(1.0, normal2, 1.0);
  }

  normal_avg.scale(1.0 / normal_avg.norm2());

  // map point into fluid domain along normal vector
  start_point.update(1.0, proj_x_n, dist, normal_avg);

  return;
}


void XFEM::XfluidStd::compute_start_point_avg(
    const std::vector<Core::Elements::Element*>& sides,        ///< pointer to side element
    std::vector<Core::LinAlg::SerialDenseMatrix>& sides_xyze,  ///< side's node coordinates
    const std::vector<std::vector<int>>& sides_lm,             ///< local map
    double& dist,                            ///< distance from point to its projection
    Core::LinAlg::Matrix<3, 1>& proj_x_n,    ///< projected point at t^n
    Core::LinAlg::Matrix<3, 1>& start_point  ///< final start point
)
{
  if (sides.size() != sides_xyze.size() or sides.size() != sides_lm.size())
    FOUR_C_THROW("not equal number of sides, xyze-coordinates or lm-vectors ");

  Core::LinAlg::Matrix<3, 1> normal_avg(
      Core::LinAlg::Initialization::set_zero);  // averaged normal vector

  ///
  for (std::vector<Core::Elements::Element*>::const_iterator it = sides.begin(); it != sides.end();
      it++)
  {
    Core::Elements::Element* side = *it;

    Core::LinAlg::Matrix<2, 1> side_center(Core::LinAlg::Initialization::set_zero);

    Core::LinAlg::Matrix<3, 1> local_node_coord(Core::LinAlg::Initialization::set_zero);

    // get the side-center
    for (int i = 0; i < side->num_node(); i++)
    {
      local_node_coord = Core::FE::get_node_coordinates(i, side->shape());
      side_center(0) += local_node_coord(0);
      side_center(1) += local_node_coord(1);
    }

    side_center.scale(1.0 / side->num_node());

    // get the normal at the side center
    Core::LinAlg::Matrix<3, 1> proj_x_n_dummy1(Core::LinAlg::Initialization::set_zero);
    Core::LinAlg::Matrix<3, 1> side_normal(Core::LinAlg::Initialization::set_zero);

    callget_normal_side_tn(side, side_normal, sides_xyze[it - sides.begin()],
        sides_lm[it - sides.begin()], proj_x_n_dummy1, side_center);

    normal_avg += side_normal;
  }

  normal_avg.scale(1.0 / normal_avg.norm2());

  // map point into fluid domain along normal vector
  start_point.update(1.0, proj_x_n, dist, normal_avg);

  return;
}



void XFEM::XfluidStd::callget_normal_side_tn(
    Core::Elements::Element* side,               ///< pointer to side element
    Core::LinAlg::Matrix<3, 1>& normal,          ///< normal vector w.r.t side
    Core::LinAlg::SerialDenseMatrix& side_xyze,  ///< side's node coordinates
    const std::vector<int>& lm,                  ///< local map
    Core::LinAlg::Matrix<3, 1>& proj_x_n,        ///< projected point on side
    Core::LinAlg::Matrix<2, 1>& xi_side  ///< local coordinates of projected point w.r.t side
)
{
  // get number of dofs for this side element
  const int numdofpernode = side->num_dof_per_node(*side->nodes()[0]);

  if (numdofpernode == 3)  // three dofs per node, for standard Dirichlet coupling
  {
    switch (side->shape())
    {
        //      case Core::FE::CellType::tri3:
        //      {
        //        get_normal_side_tn<Core::FE::CellType::tri3, 3>(normal, side_xyze, lm,
        //        proj_x_n, xi_side); break;
        //      }
        //      case Core::FE::CellType::tri6:
        //      {
        //        get_normal_side_tn<Core::FE::CellType::tri6, 3>(normal, side_xyze, lm,
        //        proj_x_n, xi_side); break;
        //      }
      case Core::FE::CellType::quad4:
      {
        get_normal_side_tn<Core::FE::CellType::quad4, 3>(normal, side_xyze, lm, proj_x_n, xi_side);
        break;
      }
      case Core::FE::CellType::quad8:
      {
        get_normal_side_tn<Core::FE::CellType::quad8, 3>(normal, side_xyze, lm, proj_x_n, xi_side);
        break;
      }
        //      case Core::FE::CellType::quad9:
        //      {
        //        get_normal_side_tn<Core::FE::CellType::quad9, 3>(normal, side_xyze,
        //        lm, proj_x_n, xi_side); break;
        //      }
      default:
        FOUR_C_THROW("unsupported side shape {}", side->shape());
        break;
    }
  }
  else if (numdofpernode == 4)  // four dofs per node, for standard Dirichlet coupling
  {
    switch (side->shape())
    {
        //      case Core::FE::CellType::tri3:
        //      {
        //        get_normal_side_tn<Core::FE::CellType::tri3, 4>(normal, side_xyze, lm,
        //        proj_x_n, xi_side); break;
        //      }
        //      case Core::FE::CellType::tri6:
        //      {
        //        get_normal_side_tn<Core::FE::CellType::tri6, 4>(normal, side_xyze, lm,
        //        proj_x_n, xi_side); break;
        //      }
      case Core::FE::CellType::quad4:
      {
        get_normal_side_tn<Core::FE::CellType::quad4, 4>(normal, side_xyze, lm, proj_x_n, xi_side);
        break;
      }
      case Core::FE::CellType::quad8:
      {
        get_normal_side_tn<Core::FE::CellType::quad8, 4>(normal, side_xyze, lm, proj_x_n, xi_side);
        break;
      }
        //      case Core::FE::CellType::quad9:
        //      {
        //        get_normal_side_tn<Core::FE::CellType::quad9, 4>(normal, side_xyze,
        //        lm, proj_x_n, xi_side); break;
        //      }
      default:
        FOUR_C_THROW("unsupported side shape {}", side->shape());
        break;
    }
  }
}

template <Core::FE::CellType side_distype, const int numdof>
void XFEM::XfluidStd::get_normal_side_tn(
    Core::LinAlg::Matrix<3, 1>& normal,          ///< normal vector w.r.t side
    Core::LinAlg::SerialDenseMatrix& side_xyze,  ///< side's node coordinates
    const std::vector<int>& lm,                  ///< local map
    Core::LinAlg::Matrix<3, 1>& proj_x_n,        ///< projected point on side
    Core::LinAlg::Matrix<2, 1>& xi_side  ///< local coordinates of projected point w.r.t side
)
{
  const int side_nen_ = Core::FE::num_nodes<side_distype>;

  // add displacements
  addeidisp<side_distype, numdof>(side_xyze, *boundarydis_, "idispn", lm);

  // fill Linalg matrix with current node coordinates including displacements
  Core::LinAlg::Matrix<3, side_nen_> xyze_(side_xyze);


  // Initialization
  Core::LinAlg::Matrix<side_nen_, 1> funct(
      Core::LinAlg::Initialization::set_zero);  // shape functions
  Core::LinAlg::Matrix<2, side_nen_> deriv(
      Core::LinAlg::Initialization::set_zero);  // derivatives dr, ds


  Core::LinAlg::Matrix<3, 1> x(Core::LinAlg::Initialization::set_zero);

  Core::LinAlg::Matrix<3, 2> derxy(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_dr(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_ds(Core::LinAlg::Initialization::set_zero);

  // get current values
  Core::FE::shape_function_2d(funct, xi_side(0), xi_side(1), side_distype);
  Core::FE::shape_function_2d_deriv1(deriv, xi_side(0), xi_side(1), side_distype);

  proj_x_n.multiply(xyze_, funct);

  derxy.multiply_nt(xyze_, deriv);


  // set dx_dr and dx_ds
  for (int i = 0; i < 3; i++)
  {
    dx_dr(i) = derxy(i, 0);
    dx_ds(i) = derxy(i, 1);
  }


  normal(0) = dx_dr(1) * dx_ds(2) - dx_ds(1) * dx_dr(2);
  normal(1) = dx_dr(2) * dx_ds(0) - dx_ds(2) * dx_dr(0);
  normal(2) = dx_dr(0) * dx_ds(1) - dx_ds(0) * dx_dr(1);

  normal.scale(1.0 / normal.norm2());

  return;
}



/*------------------------------------------------------------------------------------------------*
 * call and prepare the projection of point to side                                  schott 07/12
 **
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::call_get_projxn_line(
    Core::Elements::Element* side,  ///< pointer to structural side element
    Core::Elements::Element* line,  ///< pointer to structural line of side element
    Core::LinAlg::Matrix<3, 1>& proj_x_n, double& xi_line)
{
  //-------------------------------------------------

  // line geometry at initial state t^0

  // line geometry
  const int numnodes = line->num_node();
  Core::Nodes::Node** nodes = line->nodes();
  Core::LinAlg::SerialDenseMatrix line_xyze(3, numnodes);
  for (int i = 0; i < numnodes; ++i)
  {
    const double* x = nodes[i]->x().data();
    std::copy(x, x + 3, &line_xyze(0, i));
  }


  Core::Elements::LocationArray cutla(1);
  line->location_vector(*boundarydis_, cutla, false);

  // get number of dofs for this side element
  const int numdofpernode = side->num_dof_per_node(*side->nodes()[0]);

  if (numdofpernode == 3)
  {
    switch (line->shape())
    {
      case Core::FE::CellType::line2:
      {
        get_projxn_line<Core::FE::CellType::line2, 3>(line_xyze, cutla[0].lm_, proj_x_n, xi_line);
        break;
      }
      default:
        FOUR_C_THROW("unsupported line shape {}", line->shape());
        break;
    }
  }
  else if (numdofpernode == 4)  // four dofs per node, for standard Dirichlet coupling
  {
    switch (line->shape())
    {
      case Core::FE::CellType::line2:
      {
        get_projxn_line<Core::FE::CellType::line2, 4>(line_xyze, cutla[0].lm_, proj_x_n, xi_line);
        break;
      }
      default:
        FOUR_C_THROW("unsupported line shape {}", line->shape());
        break;
    }
  }
}


template <Core::FE::CellType line_distype, const int numdof>
void XFEM::XfluidStd::get_projxn_line(
    Core::LinAlg::SerialDenseMatrix& line_xyze,  ///< line's node coordinates
    const std::vector<int>& lm,                  ///< local map
    Core::LinAlg::Matrix<3, 1>& proj_x_n,        ///< projected point on side
    double& xi_line  ///< local coordinates of projected point w.r.t line
)
{
  const int line_nen_ = Core::FE::num_nodes<line_distype>;

  // add displacements
  addeidisp<line_distype, numdof>(line_xyze, *boundarydis_, "idispn", lm);

  // fill Core::LINALG matrix
  Core::LinAlg::Matrix<3, line_nen_> xyze_(line_xyze);


  // Initialization
  Core::LinAlg::Matrix<line_nen_, 1> funct(
      Core::LinAlg::Initialization::set_zero);  // shape functions
  Core::LinAlg::Matrix<1, line_nen_> deriv(
      Core::LinAlg::Initialization::set_zero);  // derivatives dr

  Core::LinAlg::Matrix<3, 1> x(Core::LinAlg::Initialization::set_zero);


  // get current values
  Core::FE::shape_function_1d(funct, xi_line, line_distype);

  // projected point tracked back at t^n
  proj_x_n.multiply(xyze_, funct);


  return;
}


/*--------------------------------------------------------------------------------
 * add side's or line's interface displacements and set current node coordinates
 *--------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, const int numdof>
void XFEM::XfluidStd::addeidisp(
    Core::LinAlg::SerialDenseMatrix& xyze,   ///< node coordinates of side or line
    const Core::FE::Discretization& cutdis,  ///< cut discretization
    const std::string state,                 ///< state
    const std::vector<int>& lm               ///< local map
)
{
  const int nen = Core::FE::num_nodes<distype>;

  Core::LinAlg::Matrix<3, nen> eidisp(Core::LinAlg::Initialization::set_zero);


  // get state of the global vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state = cutdis.get_state(state);
  if (matrix_state == nullptr) FOUR_C_THROW("Cannot get state vector {}", state.c_str());

  // extract local values of the global vectors
  std::vector<double> mymatrix = Core::FE::extract_values(*matrix_state, lm);

  for (int inode = 0; inode < nen; ++inode)  // number of nodes
  {
    for (int idim = 0; idim < 3; ++idim)  // number of dimensions
    {
      (eidisp)(idim, inode) =
          mymatrix[idim + (inode * numdof)];  // attention! disp state vector has 3+1 dofs for
                                              // displacement (the same as for (u,p))
    }
  }

  // add the displacement of the interface
  for (int inode = 0; inode < nen; ++inode)
  {
    xyze(0, inode) += eidisp(0, inode);
    xyze(1, inode) += eidisp(1, inode);
    xyze(2, inode) += eidisp(2, inode);
  }

  return;
}  // addeidisp


/*------------------------------------------------------------------------------------------------*
 * call and prepare the projection of point to side                                  schott 07/12
 **
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::call_project_on_side(
    Core::Elements::Element* side,  ///< pointer to structural side element
    const std::string state,        ///< state n or np?
    Core::LinAlg::Matrix<3, 1>&
        newNodeCoords,                      ///< node coordinates of point that has to be projected
    int& proj_sid,                          ///< id of side when projection lies on side
    double& min_dist,                       ///< minimal distance, potentially updated
    Core::LinAlg::Matrix<3, 1>& proj_x_np,  ///< projection of point on this side
    Core::LinAlg::Matrix<2, 1>&
        proj_xi_side,              ///< local coordinates of projection of point w.r.t to this side
    TimeIntData::Projection& proj  ///< reference to data
)
{
  bool on_side = false;         ///< lies projection on side?
  double curr_dist = INFINITY;  ///< resulting distance
  Core::LinAlg::Matrix<3, 1> x_side(
      Core::LinAlg::Initialization::set_zero);  ///< resulting projected point
  Core::LinAlg::Matrix<2, 1> xi_side(
      Core::LinAlg::Initialization::set_zero);  ///< local coordinates resulting projected point


  // side geometry at initial state t^0
  const int numnodes = side->num_node();
  Core::Nodes::Node** nodes = side->nodes();
  Core::LinAlg::SerialDenseMatrix side_xyze(3, numnodes);
  for (int i = 0; i < numnodes; ++i)
  {
    const double* x = nodes[i]->x().data();
    std::copy(x, x + 3, &side_xyze(0, i));
  }


  Core::Elements::LocationArray cutla(1);
  side->location_vector(*boundarydis_, cutla, false);

  // get number of dofs for this side element
  const int numdofpernode = side->num_dof_per_node(*side->nodes()[0]);

  if (numdofpernode == 3)  // three dofs per node, for standard Dirichlet coupling
  {
    switch (side->shape())
    {
        //      case Core::FE::CellType::tri3:
        //      {
        //        on_side = project_on_side<Core::FE::CellType::tri3,3>(side_xyze,
        //        cutla[0].lm_,state,newNodeCoords,x_side,xi_side); break;
        //      }
        //      case Core::FE::CellType::tri6:
        //      {
        //        on_side = project_on_side<Core::FE::CellType::tri6,3>(side_xyze,
        //        cutla[0].lm_,state,newNodeCoords,x_side,xi_side); break;
        //      }
      case Core::FE::CellType::quad4:
      {
        on_side = project_on_side<Core::FE::CellType::quad4, 3>(
            side_xyze, cutla[0].lm_, state, newNodeCoords, x_side, xi_side, curr_dist);
        break;
      }
      case Core::FE::CellType::quad8:
      {
        on_side = project_on_side<Core::FE::CellType::quad8, 3>(
            side_xyze, cutla[0].lm_, state, newNodeCoords, x_side, xi_side, curr_dist);
        break;
      }
      case Core::FE::CellType::quad9:
      {
        on_side = project_on_side<Core::FE::CellType::quad9, 3>(
            side_xyze, cutla[0].lm_, state, newNodeCoords, x_side, xi_side, curr_dist);
        break;
      }
      default:
        FOUR_C_THROW("unsupported side shape {}", side->shape());
        break;
    }
  }
  else if (numdofpernode == 4)  // four dofs per node, for standard Dirichlet coupling
  {
    switch (side->shape())
    {
        //      case Core::FE::CellType::tri3:
        //      {
        //        on_side = project_on_side<Core::FE::CellType::tri3,4>(side_xyze,
        //        cutla[0].lm_,state,newNodeCoords,x_side,xi_side, curr_dist); break;
        //      }
        //      case Core::FE::CellType::tri6:
        //      {
        //        on_side = project_on_side<Core::FE::CellType::tri6,4>(side_xyze,
        //        cutla[0].lm_,state,newNodeCoords,x_side,xi_side, curr_dist); break;
        //      }
      case Core::FE::CellType::quad4:
      {
        on_side = project_on_side<Core::FE::CellType::quad4, 4>(
            side_xyze, cutla[0].lm_, state, newNodeCoords, x_side, xi_side, curr_dist);
        break;
      }
      case Core::FE::CellType::quad8:
      {
        on_side = project_on_side<Core::FE::CellType::quad8, 4>(
            side_xyze, cutla[0].lm_, state, newNodeCoords, x_side, xi_side, curr_dist);
        break;
      }
        //      case Core::FE::CellType::quad9:
        //      {
        //        //      on_side =
        //        project_on_side<Core::FE::CellType::quad9,4>(side_xyze,
        //        cutla[0].lm_,state,newNodeCoords,x_side,xi_side, curr_dist); break;
        //      }
      default:
        FOUR_C_THROW("unsupported side shape {}", side->shape());
        break;
    }
  }

  // update minimal distance if possible
  if (on_side)
  {
#ifdef DEBUG_TIMINT_STD
    std::cout << "\t\tprojection of current node lies on side " << side->Id() << " with distance "
              << curr_dist << std::endl;
#endif

    if (curr_dist < min_dist - TOL_dist_ && curr_dist >= 0)
    {
#ifdef DEBUG_TIMINT_STD
      std::cout << "\t\t\t>>> updated smallest distance! " << curr_dist << std::endl;
#endif


      //--------------------
      // set current minimal distance w.r.t side
      proj = TimeIntData::onSide_;
      // set side that contains the projected point
      proj_sid = side->id();
      // update minimal distance
      min_dist = curr_dist;
      // update projection point
      proj_x_np.update(1.0, x_side, 0.0);
      // update local coordinates w.r.t side
      proj_xi_side.update(1.0, xi_side, 0.0);
      //--------------------
    }
    else if (curr_dist < 0)
    {
#ifdef DEBUG_TIMINT_STD
      std::cout << "\t\t\tnegative distance -> do not update" << std::endl;
#endif
    }
  }

  return;
}



/*------------------------------------------------------------------------------------------------*
 * call and prepare the projection of point to side                                  schott 07/12
 **
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::call_project_on_line(
    Core::Elements::Element* side,  ///< pointer to structural side element
    Core::Elements::Element* line,  ///< pointer to structural line of side element
    int line_count,                 ///< local line id w.r.t side element
    const std::string state,        ///< state n or np?
    Core::LinAlg::Matrix<3, 1>&
        newNodeCoords,                      ///< node coordinates of point that has to be projected
    double& min_dist,                       ///< minimal distance, potentially updated
    Core::LinAlg::Matrix<3, 1>& proj_x_np,  ///< projection of point on this side
    std::map<std::vector<int>, std::vector<double>>&
        proj_xi_line,  ///< std::map<sorted nids, local line coordinates of projection of point
                       ///< w.r.t sides >
    std::map<std::vector<int>, std::vector<int>>&
        proj_lineid,  ///< std::map<sorted nids, local line id w.r.t sides>
    std::map<std::vector<int>, std::vector<int>>&
        proj_nid_line,             ///< std::map<sorted nids, side Ids>
    int& proj_sid,                 ///< id of side that contains the projected point
    TimeIntData::Projection& proj  ///< reference to data
)
{
  bool on_line = false;         ///< is projection on line
  double curr_dist = INFINITY;  ///< resulting distance
  Core::LinAlg::Matrix<3, 1> x_line(
      Core::LinAlg::Initialization::set_zero);  ///< resulting projected point
  double xi_line = INFINITY;                    ///< local coordinates resulting projected point

  //-------------------------------------------------

  // line geometry at initial state t^0

  // line geometry
  const int numnodes = line->num_node();
  Core::Nodes::Node** nodes = line->nodes();
  Core::LinAlg::SerialDenseMatrix line_xyze(3, numnodes);
  for (int i = 0; i < numnodes; ++i)
  {
    const double* x = nodes[i]->x().data();
    std::copy(x, x + 3, &line_xyze(0, i));
  }


  Core::Elements::LocationArray cutla(1);
  line->location_vector(*boundarydis_, cutla, false);

  // get number of dofs for this side element
  const int numdofpernode = side->num_dof_per_node(*side->nodes()[0]);

  if (numdofpernode == 3)
  {
    switch (line->shape())
    {
      case Core::FE::CellType::line2:
      {
        on_line = project_on_line<Core::FE::CellType::line2, 3>(
            line_xyze, cutla[0].lm_, state, newNodeCoords, x_line, xi_line, curr_dist);
        break;
      }
      default:
        FOUR_C_THROW("unsupported line shape {}", line->shape());
        break;
    }
  }
  else if (numdofpernode == 4)  // four dofs per node, for standard Dirichlet coupling
  {
    switch (line->shape())
    {
      case Core::FE::CellType::line2:
      {
        on_line = project_on_line<Core::FE::CellType::line2, 4>(
            line_xyze, cutla[0].lm_, state, newNodeCoords, x_line, xi_line, curr_dist);
        break;
      }
      case Core::FE::CellType::line3:
      {
        on_line = project_on_line<Core::FE::CellType::line3, 4>(
            line_xyze, cutla[0].lm_, state, newNodeCoords, x_line, xi_line, curr_dist);
        break;
      }
      default:
        FOUR_C_THROW("unsupported line shape {}", line->shape());
        break;
    }
  }

  // update minimal distance if possible
  if (on_line)
  {
#ifdef DEBUG_TIMINT_STD
    std::cout << "\t\tline-proj: projection lies on line " << line_count << " of side "
              << side->Id() << std::endl;
#endif

    // distance has to be smaller than the last one
    if (curr_dist < (min_dist - TOL_dist_) && curr_dist >= 0)
    {
#ifdef DEBUG_TIMINT_STD
      std::cout.precision(15);
      std::cout << "\t\t\t>>> updated smallest distance!" << curr_dist << std::endl;
#endif

      if (curr_dist < (min_dist - TOL_dist_))  // smaller distance found
      {
        // another line found, reset already found lines
        proj_lineid.clear();
        proj_xi_line.clear();
        proj_nid_line.clear();
      }
      else
      {
        // add lines, that have the same distance up to TOL_dist
      }

      std::vector<int> line_nids;
      for (int i = 0; i < numnodes; ++i)
      {
        line_nids.push_back(nodes[i]->id());
      }

      sort(line_nids.begin(), line_nids.end());

      // reset side id for projection on side
      proj_sid = -1;

      //--------------------
      // set current minimal distance w.r.t line
      proj = TimeIntData::onLine_;
      // update minimal distance
      min_dist = curr_dist;
      // update projection point
      proj_x_np.update(1.0, x_line, 0.0);



      std::map<std::vector<int>, std::vector<int>>::iterator lines = proj_nid_line.find(line_nids);

      // line already inserted via other side
      if (lines != proj_nid_line.end())
      {
        // set line id w.r.t side->Id() that contains the projected point
        (lines->second).push_back(side->id());
        // update local line id w.r.t side
        (proj_lineid.find(line_nids))->second.push_back(line_count);
        // update local coordinates w.r.t line
        (proj_xi_line.find(line_nids))->second.push_back(xi_line);
      }
      else
      {
        std::vector<int> sideids;
        sideids.push_back(side->id());
        std::vector<int> locallineids;
        locallineids.push_back(line_count);
        std::vector<double> locallineXiCoords;
        locallineXiCoords.push_back(xi_line);

        // set line id w.r.t side->Id() that contains the projected point
        proj_nid_line.insert(std::pair<std::vector<int>, std::vector<int>>(line_nids, sideids));
        // update local line id w.r.t side
        proj_lineid.insert(std::pair<std::vector<int>, std::vector<int>>(line_nids, locallineids));
        // update local coordinates w.r.t line
        proj_xi_line.insert(
            std::pair<std::vector<int>, std::vector<double>>(line_nids, locallineXiCoords));
      }

      //--------------------
    }
    else if (curr_dist < 0)
    {
      FOUR_C_THROW(" no negative distance to line possible");
    }
  }  // on_line

  return;
}

/*------------------------------------------------------------------------------------------------*
 * call and prepare the projection of point to point (distance computation)          schott 07/12
 **
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::call_project_on_point(Core::Nodes::Node* node,  ///< pointer to node
    const std::string state,                                          ///< state n or np?
    Core::LinAlg::Matrix<3, 1>&
        newNodeCoords,                      ///< node coordinates of point that has to be projected
    double& min_dist,                       ///< minimal distance, potentially updated
    int& proj_nid_np,                       ///< nid id of projected point on surface
    Core::LinAlg::Matrix<3, 1>& proj_x_np,  ///< projection of point on this side
    int& proj_sid,                          ///< id of side that contains the projected point
    std::map<std::vector<int>, std::vector<double>>
        proj_xi_line,  ///< std::map<side ID, local coordinates of projection of point w.r.t to
                       ///< this line>
    std::map<std::vector<int>, std::vector<int>> proj_lineid,  ///< std::map<side ID, local line id>
    std::map<std::vector<int>, std::vector<int>>
        proj_nid_line,             ///< std::map<side ID, vec<line Ids>>
    TimeIntData::Projection& proj  ///< projection type
)
{
  double curr_dist = INFINITY;  ///< resulting distance
  Core::LinAlg::Matrix<3, 1> x_point(
      Core::LinAlg::Initialization::set_zero);  ///< resulting projected point


  // its point geometry
  Core::LinAlg::SerialDenseMatrix point_xyze(3, 1);

  const double* x = node->x().data();
  std::copy(x, x + 3, &point_xyze(0, 0));

  std::vector<int> lm = boundarydis_->dof(0, node);


  // compute distance between two points
  project_on_point(point_xyze, lm, state, newNodeCoords, x_point, curr_dist);


  // update minimal distance if possible
  if (curr_dist < min_dist - TOL_dist_ and curr_dist >= 0)
  {
#ifdef DEBUG_TIMINT_STD
    std::cout << "\t\t\t>>> updated smallest distance w.r.t point-proj!" << curr_dist << std::endl;
#endif
    //--------------------
    // set current minimal distance w.r.t point
    proj = TimeIntData::onPoint_;
    // reset side id
    proj_sid = -1;
    // reset line id w.r.t side->Id() that contains the projected point
    proj_lineid.clear();
    proj_xi_line.clear();
    proj_nid_line.clear();
    // update minimal distance
    min_dist = curr_dist;
    // update point with minimal distance
    proj_x_np.update(1.0, x_point, 0.0);
    // update node id of point with minimal distance
    proj_nid_np = node->id();
    //--------------------
  }

  return;
}


/*--------------------------------------------------------------------------------*
 * project point from in normal direction onto corresponding side    schott 07/12 *
 *--------------------------------------------------------------------------------*/
template <Core::FE::CellType side_distype, const int numdof>
bool XFEM::XfluidStd::project_on_side(
    Core::LinAlg::SerialDenseMatrix& side_xyze,  ///< side's node coordinates
    const std::vector<int>& lm,                  ///< local map
    const std::string state,                     ///< state n or np
    Core::LinAlg::Matrix<3, 1>& x_gp_lin,  ///< global coordinates of point that has to be projected
    Core::LinAlg::Matrix<3, 1>& x_side,    ///< projected point on side
    Core::LinAlg::Matrix<2, 1>& xi_side,   ///< local coordinates of projected point w.r.t side
    double& dist                           ///< distance from point to its projection
)
{
  const int side_nen_ = Core::FE::num_nodes<side_distype>;

  // add displacements
  addeidisp<side_distype, numdof>(side_xyze, *boundarydis_, state, lm);

  // fill Linalg matrix with current node coordinates including displacements
  Core::LinAlg::Matrix<3, side_nen_> xyze_(side_xyze);


  // Initialization
  Core::LinAlg::Matrix<side_nen_, 1> funct(
      Core::LinAlg::Initialization::set_zero);  // shape functions
  Core::LinAlg::Matrix<2, side_nen_> deriv(
      Core::LinAlg::Initialization::set_zero);  // derivatives dr, ds
  Core::LinAlg::Matrix<3, side_nen_> deriv2(
      Core::LinAlg::Initialization::set_zero);  // 2nd derivatives drdr, dsds, drds


  Core::LinAlg::Matrix<3, 1> x(Core::LinAlg::Initialization::set_zero);

  Core::LinAlg::Matrix<3, 2> derxy(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_dr(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_ds(Core::LinAlg::Initialization::set_zero);

  Core::LinAlg::Matrix<3, 3> derxy2(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_drdr(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_dsds(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_drds(Core::LinAlg::Initialization::set_zero);

  Core::LinAlg::Matrix<3, 1> dx_drdr_times_dx_ds(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_dr_times_dx_drds(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_drds_times_dx_ds(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_dr_times_dx_dsds(Core::LinAlg::Initialization::set_zero);
  Core::LinAlg::Matrix<3, 1> dx_dr_times_dx_ds(Core::LinAlg::Initialization::set_zero);

  Core::LinAlg::Matrix<3, 1> residuum(
      Core::LinAlg::Initialization::set_zero);  // residuum of the newton iteration
  Core::LinAlg::Matrix<3, 3> sysmat(
      Core::LinAlg::Initialization::set_zero);  // matrix for the newton system
  Core::LinAlg::Matrix<3, 1> incr(
      Core::LinAlg::Initialization::set_zero);  // increment of the newton system

  Core::LinAlg::Matrix<3, 1> sol(
      Core::LinAlg::Initialization::set_zero);  // sol carries xi_1, xi_2, d (distance)

  if (side_distype == Core::FE::CellType::tri3 or side_distype == Core::FE::CellType::tri6)
  {
    sol(0) = 0.333333333333333;
    sol(1) = 0.333333333333333;
  }
  else if (side_distype == Core::FE::CellType::quad4 or side_distype == Core::FE::CellType::quad8 or
           side_distype == Core::FE::CellType::quad9)
  {
    sol(0) = 0.0;
    sol(1) = 0.0;
  }
  else
  {
    FOUR_C_THROW("define start side xi-coordinates for unsupported cell type");
  }

  const double absTolIncr = 1.0e-9;  // absolute tolerance for the local coordinates increment
  const double absTolRes = 1.0e-9;   // absolute tolerance for the whole residual
  const double absTOLdist = 1.0e-9;  // absolute tolerance for distance

  int iter = 0;
  const int maxiter = 7;

  bool converged = false;

  while (iter < maxiter && !converged)
  {
    iter++;


    // get current values
    Core::FE::shape_function_2d(funct, sol(0), sol(1), side_distype);
    Core::FE::shape_function_2d_deriv1(deriv, sol(0), sol(1), side_distype);
    Core::FE::shape_function_2d_deriv2(deriv2, sol(0), sol(1), side_distype);

    x.multiply(xyze_, funct);

    derxy.multiply_nt(xyze_, deriv);

    derxy2.multiply_nt(xyze_, deriv2);

    // set dx_dr and dx_ds
    for (int i = 0; i < 3; i++)
    {
      dx_dr(i) = derxy(i, 0);
      dx_ds(i) = derxy(i, 1);

      dx_drdr(i) = derxy2(i, 0);
      dx_dsds(i) = derxy2(i, 1);
      dx_drds(i) = derxy2(i, 2);
    }

    // get vector products

    dx_drdr_times_dx_ds(0) = dx_drdr(1) * dx_ds(2) - dx_ds(1) * dx_drdr(2);
    dx_drdr_times_dx_ds(1) = dx_drdr(2) * dx_ds(0) - dx_ds(2) * dx_drdr(0);
    dx_drdr_times_dx_ds(2) = dx_drdr(0) * dx_ds(1) - dx_ds(0) * dx_drdr(1);

    dx_dr_times_dx_drds(0) = dx_dr(1) * dx_drds(2) - dx_drds(1) * dx_dr(2);
    dx_dr_times_dx_drds(1) = dx_dr(2) * dx_drds(0) - dx_drds(2) * dx_dr(0);
    dx_dr_times_dx_drds(2) = dx_dr(0) * dx_drds(1) - dx_drds(0) * dx_dr(1);

    dx_drds_times_dx_ds(0) = dx_drds(1) * dx_ds(2) - dx_ds(1) * dx_drds(2);
    dx_drds_times_dx_ds(1) = dx_drds(2) * dx_ds(0) - dx_ds(2) * dx_drds(0);
    dx_drds_times_dx_ds(2) = dx_drds(0) * dx_ds(1) - dx_ds(0) * dx_drds(1);

    dx_dr_times_dx_dsds(0) = dx_dr(1) * dx_dsds(2) - dx_dsds(1) * dx_dr(2);
    dx_dr_times_dx_dsds(1) = dx_dr(2) * dx_dsds(0) - dx_dsds(2) * dx_dr(0);
    dx_dr_times_dx_dsds(2) = dx_dr(0) * dx_dsds(1) - dx_dsds(0) * dx_dr(1);

    dx_dr_times_dx_ds(0) = dx_dr(1) * dx_ds(2) - dx_ds(1) * dx_dr(2);
    dx_dr_times_dx_ds(1) = dx_dr(2) * dx_ds(0) - dx_ds(2) * dx_dr(0);
    dx_dr_times_dx_ds(2) = dx_dr(0) * dx_ds(1) - dx_ds(0) * dx_dr(1);

    // define sysmat
    for (int i = 0; i < 3; i++)
    {
      // d/dr
      sysmat(i, 0) = dx_dr(i) - sol(2) * (dx_drdr_times_dx_ds(i) + dx_dr_times_dx_drds(i));

      // d/ds
      sysmat(i, 1) = dx_ds(i) - sol(2) * (dx_drds_times_dx_ds(i) + dx_dr_times_dx_dsds(i));

      // d/d(dist)
      sysmat(i, 2) = -dx_dr_times_dx_ds(i);


      // residual
      residuum(i) = x(i) - sol(2) * dx_dr_times_dx_ds(i) - x_gp_lin(i);
    }



    sysmat.invert();

    // solve Newton iteration
    incr.clear();
    incr.multiply(-1.0, sysmat, residuum);  // incr = -Systemmatrix^-1 * residuum

    // update solution
    sol.update(1.0, incr, 1.0);

    if ((incr.norm2() < absTolIncr) && (residuum.norm2() < absTolRes))
    {
      converged = true;
    }

    // check relative criterion for local coordinates (between [-1,1]^2)
    //       absolute criterion for distance (-> 0)
    //       relative criterion for whole residuum
    if (  // sqrt(incr(0)*incr(0)+incr(1)*incr(1))/sqrt(sol(0)*sol(0)+sol(1)*sol(1)) <  relTolIncr
        sqrt(incr(0) * incr(0) + incr(1) * incr(1)) < absTolIncr && incr(2) < absTOLdist &&
        residuum.norm2() < absTolRes)
    {
      converged = true;
    }
  }

  bool on_side = false;

  if (!converged)
  {
#ifdef DEBUG_TIMINT_STD
    std::cout.precision(15);

    std::cout << "increment criterion loc coord "
              //<< sqrt(incr(0)*incr(0)+incr(1)*incr(1))/sqrt(sol(0)*sol(0)+sol(1)*sol(1))
              << sqrt(incr(0) * incr(0) + incr(1) * incr(1)) << " \tabsTOL: " << absTolIncr
              << std::endl;
    std::cout << "absolute criterion for distance " << incr(2) << " \tabsTOL: " << absTOLdist
              << std::endl;
    std::cout << "absolute criterion whole residuum " << residuum.norm2()
              << " \tabsTOL: " << absTolRes << std::endl;


    std::cout << "sysmat.invert" << sysmat << std::endl;
    std::cout << "sol-norm " << sol.norm2() << std::endl;
    std::cout << "sol " << sol << std::endl;
    std::cout << "x_gp_lin" << x_gp_lin << std::endl;
    std::cout << "side " << xyze_ << std::endl;
#endif

    // FOUR_C_THROW( "newton scheme in project_on_side not converged! " );

    xi_side(0) = INFINITY;
    xi_side(1) = INFINITY;

    dist = INFINITY;

    on_side = false;
  }
  else
  {
    Core::LinAlg::Matrix<3, 1> xsi(Core::LinAlg::Initialization::set_zero);
    xsi(0) = sol(0);
    xsi(1) = sol(1);
    xsi(2) = 0;

    if (within_limits<side_distype>(xsi, 1e-10))
    {
      on_side = true;


      // get length of current normal vector
      double normal_length = dx_dr_times_dx_ds.norm2();


      dist = -sol(2) * normal_length;  // negative sol(2)!!! and scaling with normal length

      // evaluate shape function at solution
      Core::FE::shape_function_2d(funct, sol(0), sol(1), side_distype);

      // get projected gauss point
      x_side.multiply(xyze_, funct);

      // set local coordinates w.r.t side
      xi_side(0) = sol(0);
      xi_side(1) = sol(1);

#ifdef DEBUG_TIMINT_STD
      std::cout << "\t\tside-proj: converged with local coordinates " << sol(0) << " " << sol(1)
                << " "
                << " dist: " << sol(2) * normal_length << std::endl;
#endif
    }
    else
    {
      on_side = false;

      xi_side(0) = INFINITY;
      xi_side(1) = INFINITY;

      dist = INFINITY;

      x_side(0) = INFINITY;
      x_side(1) = INFINITY;
      x_side(2) = INFINITY;

#ifdef DEBUG_TIMINT_STD
      std::cout << "\t\tside-proj: converged with local coordinates " << sol(0) << " " << sol(1)
                << ". Check if is on side: false" << std::endl;
#endif
    }
  }

  return on_side;
}  // project_on_side

/*--------------------------------------------------------------------------------*
 * project point in normal direction onto corresponding line         schott 07/12 *
 *--------------------------------------------------------------------------------*/
template <Core::FE::CellType line_distype, const int numdof>
bool XFEM::XfluidStd::project_on_line(
    Core::LinAlg::SerialDenseMatrix& line_xyze,  ///< line's node coordinates
    const std::vector<int>& lm,                  ///< local map
    const std::string state,                     ///< state n or np?
    Core::LinAlg::Matrix<3, 1>&
        x_point_np,                      ///< global coordinates of point that has to be projected
    Core::LinAlg::Matrix<3, 1>& x_line,  ///< projected point on line
    double& xi_line,                     ///< local coordinates of projected point w.r.t line
    double& dist                         ///< distance from point to its projection
)
{
  bool on_line = false;

  const int line_nen_ = Core::FE::num_nodes<line_distype>;

  // add displacements
  addeidisp<line_distype, numdof>(line_xyze, *boundarydis_, state, lm);

  // fill Core::LINALG matrix
  Core::LinAlg::Matrix<3, line_nen_> xyze_(line_xyze);

  // linear line
  if (line_nen_ == 2)
  {
    // get line direction
    Core::LinAlg::Matrix<3, 1> line_dir(Core::LinAlg::Initialization::set_zero);  // x2-x1
    Core::LinAlg::Matrix<3, 1> dir_tmp(Core::LinAlg::Initialization::set_zero);  // x - 0.5(x1 + x2)
    for (int isd = 0; isd < 3; isd++)
    {
      line_dir(isd) = xyze_(isd, 1) - xyze_(isd, 0);
      dir_tmp(isd) = x_point_np(isd, 0) - 0.5 * (xyze_(isd, 1) + xyze_(isd, 0));
    }

    double line_length = line_dir.norm2();  // || (x2-x1) ||

    if (line_length < 1e-12) FOUR_C_THROW("line has length smaller than 1e-12");

    // 2.0/|| (x2-x1) ||^2 * < x-0.5(x1+x2),x2-x1 >
    xi_line = 2.0 / (line_length * line_length) * line_dir.dot(dir_tmp);

    // check if projection within line segment between x1 and x2
    Core::LinAlg::Matrix<3, 1> xsi(Core::LinAlg::Initialization::set_zero);
    xsi(0) = xi_line;
    xsi(1) = 0;
    xsi(2) = 0;

    if (within_limits<line_distype>(xsi, 1e-10))
    {
      on_line = true;

      // set projection
      for (int i = 0; i < 3; i++)
      {
        x_line(i) = 0.5 * (1 + xi_line) * xyze_(i, 1) + 0.5 * (1 - xi_line) * xyze_(i, 0);
      }

      // set distance vector between x and projection
      Core::LinAlg::Matrix<3, 1> dist_vec(Core::LinAlg::Initialization::set_zero);
      for (int i = 0; i < 3; i++)
      {
        dist_vec(i) = x_point_np(i) - x_line(i);
      }

      // compute distance
      dist = dist_vec.norm2();
    }
    else
    {
      on_line = false;
      xi_line = INFINITY;
      dist = INFINITY;

      x_line(0) = INFINITY;
      x_line(1) = INFINITY;
      x_line(2) = INFINITY;
    }
  }
  else
  {
    FOUR_C_THROW("projection on line just for linear lines implemented!");
  }

  return on_line;
}  // project_on_line

/*--------------------------------------------------------------------------------*
 * compute distance (project) between two points                     schott 07/12 *
 *--------------------------------------------------------------------------------*/
void XFEM::XfluidStd::project_on_point(
    Core::LinAlg::SerialDenseMatrix& point_xyze,  ///< point's node coordinates
    const std::vector<int>& lm,                   ///< local map
    const std::string state,                      ///< state n or np?
    Core::LinAlg::Matrix<3, 1>&
        x_point_np,                       ///< global coordinates of point that has to be projected
    Core::LinAlg::Matrix<3, 1>& x_point,  ///< global coordinates of point on surface
    double& dist                          ///< distance from point to its projection
)
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state = boundarydis_->get_state(state);
  if (matrix_state == nullptr) FOUR_C_THROW("Cannot get state vector {}", state.c_str());

  // extract local values of the global vectors
  std::vector<double> mymatrix = Core::FE::extract_values(*matrix_state, lm);

  // add the displacement of the interface
  for (int idim = 0; idim < 3; ++idim)  // number of dimensions
  {
    (point_xyze)(idim, 0) += mymatrix[idim];  // attention! disp state vector has 3+1 dofs for
                                              // displacement (the same as for (u,p))
  }

  // fill Core::LINALG matrix
  Core::LinAlg::Matrix<3, 1> p(point_xyze);

  // compute direction vector between two points
  Core::LinAlg::Matrix<3, 1> direction(Core::LinAlg::Initialization::set_zero);
  direction.update(1.0, x_point_np, -1.0, p);

  // compute distance
  dist = direction.norm2();

  x_point.update(1.0, p, 0.0);

  return;
}  // project_on_point


/*--------------------------------------------------------------------------------*
 * check if local coordinates are within limits                      schott 07/12 *
 *--------------------------------------------------------------------------------*/
template <Core::FE::CellType elementtype>
bool XFEM::XfluidStd::within_limits(Core::LinAlg::Matrix<3, 1>& xsi_, const double TOL)
{
  switch (elementtype)
  {
    case Core::FE::CellType::line2:
    case Core::FE::CellType::line3:
      return (xsi_(0) >= -1 - TOL and xsi_(0) <= 1 + TOL);
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
      return (xsi_(0) >= -TOL and xsi_(1) >= -TOL and xsi_(0) <= 1 + TOL and xsi_(1) <= 1 + TOL and
              (xsi_(0) + xsi_(1)) <= 1 + TOL);
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
      return (xsi_(0) >= -1 - TOL and xsi_(1) >= -1 - TOL and xsi_(0) <= 1 + TOL and
              xsi_(1) <= 1 + TOL);
    case Core::FE::CellType::hex8:
    case Core::FE::CellType::hex20:
    case Core::FE::CellType::hex27:
      return (xsi_(0) >= -1 - TOL and xsi_(1) >= -1 - TOL and xsi_(2) >= -1 - TOL and
              xsi_(0) <= 1 + TOL and xsi_(1) <= 1 + TOL and xsi_(2) <= 1 + TOL);
    case Core::FE::CellType::tet4:
    case Core::FE::CellType::tet10:
      return (xsi_(0) >= -TOL and xsi_(1) >= -TOL and xsi_(2) >= -TOL and
              xsi_(0) + xsi_(1) + xsi_(2) <= 1 + TOL);
    case Core::FE::CellType::pyramid5:
      return (xsi_(0) >= -1 - TOL and xsi_(1) >= -1 - TOL and xsi_(2) >= -TOL and
              xsi_(0) <= 1 + TOL and xsi_(1) <= 1 + TOL and
              (fabs(xsi_(0)) > fabs(xsi_(1)) ? (xsi_(2) + fabs(xsi_(0)) <= 1 + TOL)
                                             : (xsi_(2) + fabs(xsi_(1)) <= 1 + TOL)));
    case Core::FE::CellType::wedge6:
    case Core::FE::CellType::wedge15:
      return (xsi_(0) >= -TOL and xsi_(1) >= -TOL and xsi_(2) >= -1 - TOL and xsi_(2) <= 1 + TOL and
              xsi_(0) + xsi_(1) <= 1 + TOL);
    default:
      FOUR_C_THROW("unsupported element type in XFEM::XFLUID_STD::within_limits");
  }
  return false;
}

/*------------------------------------------------------------------------------------------------*
 * setting the computed data for the standard degrees of freedom into the according * Epetra
 *Vectors for all handled nodes                                              schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::set_final_data()
{
  const int nsd = 3;  // 3 dimensions for a 3d fluid element

  // loop over data
  for (std::vector<TimeIntData>::iterator data = timeIntData_->begin(); data != timeIntData_->end();
      data++)
  {
    if (data->state_ != TimeIntData::doneStd_)
      FOUR_C_THROW("when data is set, all computation has to be done");

    std::vector<Core::LinAlg::Matrix<nsd, 1>>& velValues(
        data->velValues_);                               // velocities of the node
    std::vector<double>& presValues(data->presValues_);  // pressures of the node

    const int gnodeid = data->node_.id();  // global node id

    //-------------------------------------------------------
    Core::Nodes::Node* node = discret_->g_node(gnodeid);

#ifdef DEBUG_TIMINT_STD
    Core::IO::cout << "dofset at new timestep " << data->nds_np_ << Core::IO::endl;
#endif

    if (data->nds_np_ == -1)
      FOUR_C_THROW("cannot get dofs for dofset with number {}", data->nds_np_);

    std::vector<int> dofs;
    dofset_new_->dof(dofs, node, data->nds_np_);

    if (dofs.size() != (nsd + 1))
      FOUR_C_THROW("not the right number of dofs {} for this node ", dofs.size());


    // set velocity dofs
    for (int i = 0; i < nsd; i++)
    {
      for (size_t index = 0; index < vector_size(data->type_); index++)
        (*newVectors_[index])[newdofrowmap_.LID(dofs[i])] =
            velValues[index](i, 0);  // set the value
    }
    for (size_t index = 0; index < vector_size(data->type_); index++)
      (*newVectors_[index])[newdofrowmap_.LID(dofs[nsd])] = presValues[index];  // set the value

    data->type_ = TimeIntData::standard_;  // predictor is done, so next time standard
  }  // end loop over nodes
}  // end set_final_data



/*------------------------------------------------------------------------------------------------*
 * export start data to neighbour proc                                           winklmaier 06/10
 **
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::export_start_data()
{
  FOUR_C_THROW("this function has still to be adapted for XFSI");

  const int nsd = 3;  // 3 dimensions for a 3d fluid element

  // destination proc (the "next" one)
  int dest = myrank_ + 1;
  if (myrank_ == (numproc_ - 1)) dest = 0;

  // source proc (the "last" one)
  int source = myrank_ - 1;
  if (myrank_ == 0) source = numproc_ - 1;

  Core::Communication::PackBuffer dataSend;  // data to be sent
  for (std::vector<TimeIntData>::iterator data = timeIntData_->begin(); data != timeIntData_->end();
      data++)
  {
    pack_node(dataSend, data->node_);
    add_to_pack(dataSend, data->nds_np_);
    add_to_pack(dataSend, data->vel_);
    add_to_pack(dataSend, data->velDeriv_);
    add_to_pack(dataSend, data->presDeriv_);
    add_to_pack(dataSend, data->dispnp_);
    add_to_pack(dataSend, data->startpoint_);
    add_to_pack(dataSend, data->searchedProcs_);
    add_to_pack(dataSend, data->counter_);
    add_to_pack(dataSend, data->dMin_);
    add_to_pack(dataSend, data->type_);
  }

  std::vector<char> dataRecv;
  send_data(dataSend, dest, source, dataRecv);

  // clear vector that should be filled
  timeIntData_->clear();

  // unpack received data
  Core::Communication::UnpackBuffer buffer(dataRecv);
  while (!buffer.at_end())
  {
    std::vector<double> coords(nsd, 0.0);
    Core::Nodes::Node node(0, coords, 0);  // initialize node
    int nds_np = -1;
    Core::LinAlg::Matrix<nsd, 1> vel;                      // velocity at point x
    std::vector<Core::LinAlg::Matrix<nsd, nsd>> velDeriv;  // derivation of velocity at point x
    std::vector<Core::LinAlg::Matrix<1, nsd>> presDeriv;   // derivation of pressure at point x
    Core::LinAlg::Matrix<nsd, 1> dispnp;                   // displacement at point x
    Core::LinAlg::Matrix<nsd, 1> startpoint;               // startpoint
    int searchedProcs;                                     // number of searched processors
    int counter;                                           // iteration counter
    double dMin;                                           // minimal distance
    int newtype;                                           // type of the data

    unpack_node(buffer, node);
    extract_from_pack(buffer, nds_np);
    extract_from_pack(buffer, vel);
    extract_from_pack(buffer, velDeriv);
    extract_from_pack(buffer, presDeriv);
    extract_from_pack(buffer, dispnp);
    extract_from_pack(buffer, startpoint);
    extract_from_pack(buffer, searchedProcs);
    extract_from_pack(buffer, counter);
    extract_from_pack(buffer, dMin);
    extract_from_pack(buffer, newtype);

    timeIntData_->push_back(TimeIntData(node, nds_np, vel, velDeriv, presDeriv, dispnp, startpoint,
        searchedProcs, counter, dMin, (TimeIntData::Type)newtype));
  }

  Core::Communication::barrier(discret_->get_comm());  // processors wait for each other
}  // end exportStartData



/*------------------------------------------------------------------------------------------------*
 * export final data to node's proc                                                  schott 07/12
 **
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidStd::export_final_data()
{
#ifdef DEBUG_TIMINT_STD
  Core::IO::cout << "\n\t=============================";
  Core::IO::cout << "\n\t  export Final Data  ";
  Core::IO::cout << "\n\t=============================" << Core::IO::endl;
#endif


  const int nsd = 3;  // 3 dimensions for a 3d fluid element

  // array of vectors which stores data for
  // every processor in one vector
  std::vector<std::vector<TimeIntData>> dataVec(numproc_);

  // fill vectors with the data
  // fill data into a vector where the index corresponds to the destination which is equal to the
  // node's owner
  for (std::vector<TimeIntData>::iterator data = timeIntData_->begin(); data != timeIntData_->end();
      data++)
  {
    if (data->state_ != TimeIntData::doneStd_)
      FOUR_C_THROW("All data should be set here, having status 'done'. Thus something is wrong!");
    dataVec[data->node_.owner()].push_back(*data);
  }

  timeIntData_->clear();
  *timeIntData_ = dataVec[myrank_];  // set final data of own processor
  dataVec[myrank_].clear();          // clear data about current proc

  // send data to the processor where the point lies (1. nearest higher neighbour 2. 2nd nearest
  // higher neighbour...)
  for (int dest = (myrank_ + 1) % numproc_; dest != myrank_;
      dest = (dest + 1) % numproc_)  // dest is the target processor
  {
    // Initialization
    int source = myrank_ - (dest - myrank_);  // source proc (sends (dest-myrank_) far and gets
                                              // from (dest-myrank_) earlier)
    if (source < 0)
      source += numproc_;
    else if (source >= numproc_)
      source -= numproc_;

    Core::Communication::PackBuffer dataSend;

    for (std::vector<TimeIntData>::iterator data = dataVec[dest].begin();
        data != dataVec[dest].end(); data++)
    {
      add_to_pack(dataSend, data->node_.id());
      add_to_pack(dataSend, data->nds_np_);
      add_to_pack(dataSend, data->startpoint_);
      add_to_pack(dataSend, data->velValues_);
      add_to_pack(dataSend, data->presValues_);
      add_to_pack(dataSend, data->type_);
    }

    // clear the no more needed data
    dataVec[dest].clear();

    std::vector<char> dataRecv;
    send_data(dataSend, dest, source, dataRecv);

    Core::Communication::UnpackBuffer buffer(dataRecv);
    // unpack received data
    while (!buffer.at_end())
    {
      int gid;                                              // global id of node
      int nds_np;                                           // dofset number of node at new timestep
      Core::LinAlg::Matrix<nsd, 1> startpoint;              // startpoint
      std::vector<Core::LinAlg::Matrix<nsd, 1>> velValues;  // velocity values
      std::vector<double> presValues;                       // pressure values
      TimeIntData::Type newtype;                            // type of the data

      extract_from_pack(buffer, gid);
      extract_from_pack(buffer, nds_np);
      extract_from_pack(buffer, startpoint);
      extract_from_pack(buffer, velValues);
      extract_from_pack(buffer, presValues);
      extract_from_pack(buffer, newtype);

      Core::LinAlg::Matrix<3, 1> nodedispnp(Core::LinAlg::Initialization::set_zero);
      if (dispnp_ != nullptr)  // is alefluid
      {
        //------------------------------------------------------- add ale disp
        // get node location vector, dirichlet flags and ownerships (discret, nds, la,
        // doDirichlet)
        std::vector<int> lm;
        std::vector<int> dofs;
        dofset_new_->dof(dofs, discret_->g_node(gid), 0);  // dofs for standard dofset
        for (int j = 0; j < 4; ++j) lm.push_back(dofs[j]);

        Core::LinAlg::Matrix<1, 1> nodepredummy(Core::LinAlg::Initialization::set_zero);
        extract_nodal_values_from_vector<1>(nodedispnp, nodepredummy, dispnp_, lm);
      }

      timeIntData_->push_back(TimeIntData(
          *discret_->g_node(gid), nds_np, nodedispnp, startpoint, velValues, presValues, newtype));
    }  // end loop over number of nodes to get

    // processors wait for each other
    Core::Communication::barrier(discret_->get_comm());
  }  // end loop over processors
}  // end exportfinalData

FOUR_C_NAMESPACE_CLOSE
