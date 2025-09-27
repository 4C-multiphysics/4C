// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_xfluid_timeInt.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_cut_boundingbox.hpp"
#include "4C_cut_cutwizard.hpp"
#include "4C_cut_elementhandle.hpp"
#include "4C_cut_point.hpp"
#include "4C_cut_position.hpp"
#include "4C_cut_sidehandle.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_factory.hpp"
#include "4C_fluid_ele_interface.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_xfem_condition_manager.hpp"
#include "4C_xfem_dofset.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

// #define DEBUG_TIMINT


// -------------------------------------------------------------------
// constructor
// -------------------------------------------------------------------
XFEM::XFluidTimeInt::XFluidTimeInt(
    const bool is_newton_increment_transfer,  /// monolithic newton increment transfer or time step
                                              /// transfer?
    const std::shared_ptr<Core::FE::Discretization>& dis,              /// discretization
    const std::shared_ptr<XFEM::ConditionManager>& condition_manager,  /// condition manager
    const std::shared_ptr<Cut::CutWizard>& wizard_old,                 /// cut wizard at t^n
    const std::shared_ptr<Cut::CutWizard>& wizard_new,                 /// cut wizard at t^(n+1)
    const std::shared_ptr<XFEM::XFEMDofSet>& dofset_old,               /// XFEM dofset at t^n
    const std::shared_ptr<XFEM::XFEMDofSet>& dofset_new,               /// XFEM dofset at t^(n+1)
    const Inpar::XFEM::XFluidTimeIntScheme xfluid_timintapproach,      /// xfluid_timintapproch
    std::map<int, std::vector<Inpar::XFEM::XFluidTimeInt>>&
        node_to_reconstr_method,  /// reconstruction map for nodes and its dofsets
    std::map<Inpar::XFEM::XFluidTimeInt, std::map<int, std::set<int>>>&
        reconstr_method_to_node,  /// inverse reconstruction map for nodes and its dofsets
    const int step,               /// global time step
    const bool xfluid_timint_check_interfacetips,      /// check interfacetips?
    const bool xfluid_timint_check_sliding_on_surface  /// check sliding on surface?
    )
    : is_newton_increment_transfer_(is_newton_increment_transfer),
      dis_(dis),
      condition_manager_(condition_manager),
      wizard_old_(wizard_old),
      wizard_new_(wizard_new),
      dofset_old_(dofset_old),
      dofset_new_(dofset_new),
      timeint_scheme_(xfluid_timintapproach),
      node_to_reconstr_method_(node_to_reconstr_method),
      reconstr_method_to_node_(reconstr_method_to_node),
      xfluid_timint_check_interfacetips_(xfluid_timint_check_interfacetips),
      xfluid_timint_check_sliding_on_surface_(xfluid_timint_check_sliding_on_surface)
{
  myrank_ = Core::Communication::my_mpi_rank(dis->get_comm());
  numproc_ = Core::Communication::num_mpi_ranks(dis->get_comm());

  permutation_map_ = std::make_shared<std::map<int, int>>();

  return;
}  // end constructor


// -------------------------------------------------------------------
// set and print reconstruction status for nodes
// -------------------------------------------------------------------
void XFEM::XFluidTimeInt::set_and_print_status(const bool screenout)
{
  reconstr_counts_.clear();
  for (std::map<int, std::vector<Inpar::XFEM::XFluidTimeInt>>::const_iterator node_it =
           node_to_reconstr_method_.begin();
      node_it != node_to_reconstr_method_.end(); node_it++)
  {
    const std::vector<Inpar::XFEM::XFluidTimeInt>& nodesets = node_it->second;

    for (std::vector<Inpar::XFEM::XFluidTimeInt>::const_iterator sets = nodesets.begin();
        sets != nodesets.end(); sets++)
    {
      std::map<Inpar::XFEM::XFluidTimeInt, int>::iterator it = reconstr_counts_.find(*sets);

      if (it != reconstr_counts_.end())
        (it->second)++;  // increase counter
      else
        reconstr_counts_.insert(
            std::pair<Inpar::XFEM::XFluidTimeInt, int>(*sets, 1));  // initialize counter with 1
    }
  }

  int nummethods = Inpar::XFEM::Xf_TimeInt_undefined +
                   1;  // has to be larger than the maximum of the enum Inpar::XFEM::XFluidTimeInt

  std::vector<int> cpu_methods(nummethods);
  std::vector<int> glob_methods(nummethods);

  for (int i = 0; i < nummethods; ++i)
  {
    cpu_methods[i] = 0;
    glob_methods[i] = 0;
  }

  for (std::map<Inpar::XFEM::XFluidTimeInt, int>::iterator reconstrMethod =
           reconstr_counts_.begin();
      reconstrMethod != reconstr_counts_.end(); reconstrMethod++)
  {
    int index = (int)(reconstrMethod->first);
    cpu_methods[index] = reconstrMethod->second;
  }

  // reduce and sum over all procs
  glob_methods = Core::Communication::sum_all(cpu_methods, dis_->get_comm());

  if (screenout)
  {
    Core::IO::cout << "\n+-------------------------------------------------------+"
                   << "\nProc " << myrank_ << ": XFEM::XFluidTimeInt::Status:" << Core::IO::endl;

    for (int method_idx = 0; method_idx < nummethods; method_idx++)
    {
      printf("\n%s:\t #dofsets(P%i/allprocs):\t(%i/%i)",
          map_method_enum_to_string(Inpar::XFEM::XFluidTimeInt(method_idx)).c_str(), myrank_,
          cpu_methods[method_idx], glob_methods[method_idx]);
    }
    Core::IO::cout << "\n+-------------------------------------------------------+\n"
                   << Core::IO::endl;
  }

  return;
}


/*----------------------------------------------------------------------*
| returns matching std::string for each reconstruction method   schott 07/12 |
*----------------------------------------------------------------------*/
std::string XFEM::XFluidTimeInt::map_method_enum_to_string(
    const enum Inpar::XFEM::XFluidTimeInt term)
{
  // length of return std::string is 14 due to usage in formatted screen output
  switch (term)
  {
    case Inpar::XFEM::Xf_TimeInt_STD_by_SL:
      return "Semi-Lagrange: STD(n+1)                    ";
      break;
    case Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_GHOST:
      return "Copy Dofset:   GHOST(n)   -> STD(n+1)      ";
      break;
    case Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_STD:
      return "Copy Dofset:   STD(n)     -> STD(n+1)      ";
      break;
    case Inpar::XFEM::Xf_TimeInt_GHOST_by_GP:
      return "Ghost-Penalty: GHOST(n+1)                  ";
      break;
    case Inpar::XFEM::Xf_TimeInt_GHOST_by_COPY_from_GHOST:
      return "Copy Dofset:   GHOST(n)   -> GHOST(n+1)    ";
      break;
    case Inpar::XFEM::Xf_TimeInt_GHOST_by_COPY_from_STD:
      return "Copy Dofset:   STD(n)     -> GHOST(n+1)    ";
      break;
    case Inpar::XFEM::Xf_TimeInt_by_PROJ_from_DIS:
      return "Copy Dofset:   STD_EMB(n) -> STD/GHOST(n+1)";
      break;
    case Inpar::XFEM::Xf_TimeInt_undefined:
      return "undefined:                                 ";
    default:
      FOUR_C_THROW("Cannot cope with name enum {}", term);
      return "";
      break;
  }

  return "";
}  // ScaTraTimIntImpl::map_tim_int_enum_to_string


// -------------------------------------------------------------------
// transfer standard and ghost dofs to new map as far as possible and
// mark all dofs for reconstruction
// -------------------------------------------------------------------
void XFEM::XFluidTimeInt::transfer_dofs_to_new_map(
    const std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
        oldRowStateVectors,  /// row map based vectors w.r.t old interface position
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
        newRowStateVectors,  /// row map based vectors w.r.t new interface position
    const std::shared_ptr<std::set<int>>
        dbcgids  /// set of dof gids that must not be changed by ghost penalty reconstruction
)
{
  //------------------------------------------------------------
  // loop background fluid row nodes
  const Core::LinAlg::Map* noderowmap = dis_->node_row_map();
  const int numrownode = noderowmap->num_my_points();

  for (int lid = 0; lid < numrownode; ++lid)
  {
    // get global id of a node
    int gid = noderowmap->gid(lid);
    transfer_nodal_dofs_to_new_map(oldRowStateVectors, newRowStateVectors, dbcgids, gid);
  }

  // export the reconstruction method information to other procs
  // this has to be done, when SL-nodes mark non-row surrounding nodes (and its some of its dofsets)
  // as Ghost-penalty dofsets
  export_methods(newRowStateVectors, dbcgids);
}

// -------------------------------------------------------------------
// transfer standard and ghost dofs to new map as far as possible and
// mark all dofs for reconstruction
// -------------------------------------------------------------------
void XFEM::XFluidTimeInt::transfer_dofs_to_new_map(
    const std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
        oldRowStateVectors,  /// row map based vectors w.r.t old interface position
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
        newRowStateVectors,  /// row map based vectors w.r.t new interface position
    const std::shared_ptr<std::set<int>>
        dbcgids,  /// set of dof gids that must not be changed by ghost penalty reconstruction
    const std::vector<int>& node_gids  /// vector of node gids
)
{
  for (size_t i = 0; i < node_gids.size(); ++i)
  {
    // get global id of a node
    transfer_nodal_dofs_to_new_map(oldRowStateVectors, newRowStateVectors, dbcgids, node_gids[i]);
  }
  // export the reconstruction method information to other procs
  // this has to be done, when SL-nodes mark non-row surrounding nodes (and its some of its dofsets)
  // as Ghost-penalty dofsets
  export_methods(newRowStateVectors, dbcgids);
}

// -------------------------------------------------------------------
// transfer standard and ghost dofs to new map as far as possible
// and mark nodal dofs for reconstruction
// -------------------------------------------------------------------
void XFEM::XFluidTimeInt::transfer_nodal_dofs_to_new_map(
    const std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
        oldRowStateVectors,  /// row map based vectors w.r.t old interface position
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
        newRowStateVectors,  /// row map based vectors w.r.t new interface position
    const std::shared_ptr<std::set<int>>
        dbcgids,  /// set of dof gids that must not be changed by ghost penalty reconstruction
    int gid       /// nodal gid
)
{
#ifdef DEBUG_TIMINT
  const int numdofpernode = 4;
#endif

  if (oldRowStateVectors.size() != newRowStateVectors.size())
    FOUR_C_THROW("transfer_dofs_to_new_map: not equal number of old and new vectors");

  // get the node
  Core::Nodes::Node* node = dis_->g_node(gid);

  // get cut nodes with respect to cut wizards at t^n and t^(n+1)
  Cut::Node* n_new = wizard_new_->get_node(gid);
  Cut::Node* n_old = wizard_old_->get_node(gid);


  //---------------------------------------------------------------------------------
  // switch over different cases dependent of surrounding elements are cut or not at t^n and t^(n+1)
  //     case A: surrounding elements not cut at t^n AND t^(n+1) => copy dofs
  //     case B: at least one surrounding element cut at t^n, uncut elements at t^(n+1)
  //     case C: uncut elements at t^n, but at least one surrounding element cut at t^(n+1)
  //     case D: surrounding elements cut at old and new timestep t^n and t^(n+1)
  //---------------------------------------------------------------------------------

  // initialize to true, as in case that no cut information we have standard FE and the dofsets are
  // unique standard dofsets
  bool unique_std_uncut_np = true;
  bool unique_std_uncut_n = true;

  // check for unique std dofset and surrounding uncut elements at t^(n+1)
  if (n_new != nullptr)
  {
    const int numDofSets_new = n_new->num_dof_sets();  //= dof_cellsets_new.size()

    // just one dofset at new timestep t^(n+1)
    if (numDofSets_new == 1)
      unique_std_uncut_np = non_intersected_elements(node, *wizard_new_);
    else
      unique_std_uncut_np = false;
  }

  // check for unique std dofset and surrounding uncut elements at t^n
  if (n_old != nullptr)
  {
    const int numDofSets_old = n_old->num_dof_sets();  //= dof_cellsets_old.size()

    // just one dofset at new timestep t^n
    if (numDofSets_old == 1)
      unique_std_uncut_n = non_intersected_elements(node, *wizard_old_);
    else
      unique_std_uncut_n = false;
  }

  //===========================================================
  // switch cases A-D
  //===========================================================
  if ((n_new == nullptr and n_old == nullptr) or (n_new == nullptr and unique_std_uncut_n) or
      (unique_std_uncut_np and n_old == nullptr) or (unique_std_uncut_np and unique_std_uncut_n))
  {
    //---------------------------------------------------------------------------------
    // case A: surrounding elements not cut at t^n AND t^(n+1) => copy dofs
    //---------------------------------------------------------------------------------

    // holes are included within the bounding box
    // REMARK: we assume that the structure does not move more than one element and
    //         nodes do not change the side w.r.t structure (not to large movement!!!)

    // just one dofset available at t^n and t^(n+1)
    const int nds_new = 0;
    const int nds_old = 0;

    // copy dofs from this node from t^n -> t^(n+1)
    copy_dofs(node, nds_new, nds_old, Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_STD,
        newRowStateVectors, oldRowStateVectors, dbcgids);

  }  // end case A
  else if (n_new == nullptr or unique_std_uncut_np)
  {
    if (n_old == nullptr) FOUR_C_THROW("you should call case A here");

    //---------------------------------------------------------------------------------
    // case B: at least one surrounding element cut at t^(n), uncut elements at t^(n+1)
    //---------------------------------------------------------------------------------

    // case a): a similar uncut position at old and new timestep (but node in bounding box at old
    // timestep) case b): at least one std-dof at t^n (  -> std-dof can be copied, std-dof should
    // not have changed the side within one timestep!)
    //                                         -> all ghost-dofs have to be computed newly)
    // case c): only ghost dofs at t^n (-> to much structural movement within one timestep)

    //-------------------------------
    // t^(n+1)
    // assume one unique dofset
    const int nds_new = 0;
    Cut::Point::PointPosition pos_new = Cut::Point::undecided;

    if (n_new == nullptr)
      pos_new = Cut::Point::outside;  // by default for nodes outside the cut-boundingbox
    else
    {
      if (n_new->nodal_dof_sets().size() == 1)
        pos_new = n_new->nodal_dof_sets()[0]->position();
      else
        FOUR_C_THROW(
            "XFEM::XFluidTimeInt::transfer_dofs_to_new_map() - Case B: "
            "n_new->NodalDofSets().size() != "
            "1");
    }

    if (pos_new != Cut::Point::outside and pos_new != Cut::Point::inside)
      FOUR_C_THROW("position of unique std-dofset is not inside and not outside, can this happen?");

    std::vector<int> dofs_new;
    dofset_new_->dof(dofs_new, node, nds_new);

#ifdef DEBUG_TIMINT
    const int numdofs_new = (int)(dofs_new.size());
    if (numdofs_new != numdofpernode)
    {
      FOUR_C_THROW(
          "XFLUID-TIMINIT CASE B: node {},\t {} dofs for node without nodehandle at timestep "
          "t^(n+1) ?!",
          gid, numdofs_new);
    }
#endif

    //-------------------------------
    // t^(n)

    // how many sets of dofs?
    const int numDofSets_old = n_old->num_dof_sets();  //= dof_cellsets_old.size()

    //-------------------------------------
    // just one dofset at old timestep t^n
    //-------------------------------------
    if (numDofSets_old == 1 and unique_std_uncut_n)
    {
      FOUR_C_THROW("here you should call case A!");
    }
    else if (numDofSets_old == 1 and !unique_std_uncut_n)
    {
      // check if there is a standard dofset
      int nds_old = n_old->get_standard_nodal_dof_set(pos_new);

      if (nds_old != -1)  // case b)
      {
        // copy values
        copy_dofs(node, nds_new, nds_old, Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_STD,
            newRowStateVectors, oldRowStateVectors, dbcgids);
      }
      else  // case c)
      {
        // case: NEWLY CREATED NODEHANDLE: just one ghost dofset at old time!
        //      structural movement more than one element near node, (here SEMILAGRANGE possible!)

        if (timeint_scheme_ == Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP)
          FOUR_C_THROW("structural movement more than one element near node {}", gid);
        else if (timeint_scheme_ ==
                     Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP or
                 timeint_scheme_ ==
                     Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP)
        {
          mark_dofs(node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_STD_by_SL, dbcgids);
        }
        else if (timeint_scheme_ ==
                 Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP)
        {
          mark_dofs(
              node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_by_PROJ_from_DIS, dbcgids);
        }
        else
          FOUR_C_THROW("unknown Inpar::XFEM::Xf_TimIntScheme");
      }
    }  // some elements cut
    else if (numDofSets_old == 0)
    {
      // case: "XFLUID-TIMINIT CASE B: node %d,\t no dofset at t^n available,
      //        structural movement more than one element for node ? (here SEMILAGRANGE possible!",
      //        gid);

      if (timeint_scheme_ == Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP)
        FOUR_C_THROW(
            "no dofset at t^n available, structural movement more than one element for node {}",
            gid);
      else if (timeint_scheme_ ==
                   Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP or
               timeint_scheme_ == Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP)
      {
        mark_dofs(node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_STD_by_SL, dbcgids);
      }
      else if (timeint_scheme_ ==
               Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP)
      {
        mark_dofs(
            node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_by_PROJ_from_DIS, dbcgids);
      }
      else
        FOUR_C_THROW("unknown Inpar::XFEM::Xf_TimIntScheme");
    }
    else
    {
      //-------------------------------------
      // more than one dofset at old timestep
      //-------------------------------------

      // check if there is a standard dofset
      int nds_old = n_old->get_standard_nodal_dof_set(pos_new);

      if (nds_old == -1)
      {
        // all dofsets at t^n are not std-dofsets, structural movement more than one element

        if (timeint_scheme_ == Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP)
          FOUR_C_THROW(
              "all dofset at t^n are ghost-dofsets, structural movement more than one element for "
              "node {}",
              gid);
        else if (timeint_scheme_ ==
                     Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP or
                 timeint_scheme_ ==
                     Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP)
        {
          mark_dofs(node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_STD_by_SL, dbcgids);
        }
        else if (timeint_scheme_ ==
                 Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP)
        {
          mark_dofs(
              node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_by_PROJ_from_DIS, dbcgids);
        }
        else
          FOUR_C_THROW("unknown Inpar::XFEM::Xf_TimIntScheme");
      }
      else
      {
        // copy values
        copy_dofs(node, nds_new, nds_old, Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_STD,
            newRowStateVectors, oldRowStateVectors, dbcgids);
      }
    }  // end more than one dofset
  }  // end case B
  else if (n_old == nullptr or unique_std_uncut_n)
  {
    if (n_new == nullptr) FOUR_C_THROW("you should call case A here");

    // how many sets of dofs?
    const int numDofSets_new = n_new->num_dof_sets();  //= dof_cellsets_new.size()

    if (numDofSets_new == 0)  // no values to set
    {
      return;  // do nothing for this node
    }

    //---------------------------------------------------------------------------------
    // case C: C: uncut elements at t^n, but at least one surrounding element cut at t^(n+1)
    //---------------------------------------------------------------------------------

    // case a): a similar uncut position at old and new timestep (but node now in bounding box)
    // case b): at least one std-dof at t^(n+1)(  -> std-dof can be copied, std-dof should not have
    // changed the side within one timestep!)
    //                                            -> all ghost-dofs have to be computed newly)
    // case c): only ghost dofs at t^(n+1) (-> to much structural movement within one timestep)

    //-------------------------------
    // t^n
    // assume one unique dofset
    const int nds_old = 0;

    std::vector<int> dofs_old;
    dofset_old_->dof(dofs_old, node, nds_old);

#ifdef DEBUG_TIMINT
    const int numdofs_old = (int)(dofs_old.size());
    if (numdofs_old != numdofpernode)
    {
      FOUR_C_THROW(
          "XFLUID-TIMINIT CASE C: node {},\t {} dofs for node without nodehandle at timestep t^n "
          "?!",
          gid, numdofs_old);
    }
#endif

    //-------------------------------
    // t^(n+1)
    const std::vector<std::shared_ptr<Cut::NodalDofSet>>& dof_cellsets_new =
        n_new->nodal_dof_sets();



    //------------------------------------------
    // just one dofset at new timestep t^(n+1)
    //------------------------------------------
    if (numDofSets_new == 1 and unique_std_uncut_np)
    {
      FOUR_C_THROW("here you should call case A!");
    }
    else if (numDofSets_new == 1 and !unique_std_uncut_np)
    {
      const int nds_new = 0;

      // get the unique cellset
      const Cut::NodalDofSet* nodaldofset = &*(dof_cellsets_new[nds_new]);

      if (nodaldofset->is_standard_dof_set())  // case b)
      {
        // copy values or SL
        if (timeint_scheme_ == Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP or
            timeint_scheme_ ==
                Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP or
            timeint_scheme_ ==
                Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP)
        {
          copy_dofs(node, nds_new, nds_old, Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_STD,
              newRowStateVectors, oldRowStateVectors, dbcgids);
        }
        else if (timeint_scheme_ ==
                 Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP)
        {
          mark_dofs(node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_STD_by_SL, dbcgids);
        }
      }
      else  // case c)
      {
        // REMARK: Ghost penalty reconstruction is more safe than the copy method
        // GHOST_by_COPY_from_STD the ghost dof can be also newly created w.r.t a second other
        // interface, e.g. at t^n a fluid node between two contacting thin structures can become a
        // ghost node w.r.t to a neighbored independent fluid domain (F|S|F -node- F|S|F ->
        // F|S-node-S|F -- F|S|F)
        mark_dofs(node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_GHOST_by_GP, dbcgids);
      }
    }  // just one dofset at new timestep
    else
    {
      //------------------------------------------
      // more than one dofset at new timestep t^(n+1)
      //------------------------------------------

      // loop new dofsets
      for (std::vector<std::shared_ptr<Cut::NodalDofSet>>::const_iterator sets =
               dof_cellsets_new.begin();
          sets != dof_cellsets_new.end(); sets++)
      {
        std::shared_ptr<Cut::NodalDofSet> nodaldofset_new = *sets;
        int nds_new = sets - dof_cellsets_new.begin();

        if (nodaldofset_new
                ->is_standard_dof_set())  // first dofset (has been checked to be a std-dofset)
        {
          // copy values or SL
          if (timeint_scheme_ ==
                  Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP or
              timeint_scheme_ ==
                  Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP or
              timeint_scheme_ ==
                  Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP)
          {
            copy_dofs(node, nds_new, nds_old, Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_STD,
                newRowStateVectors, oldRowStateVectors, dbcgids);
          }
          else if (timeint_scheme_ ==
                   Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP)
          {
            mark_dofs(
                node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_STD_by_SL, dbcgids);
          }
        }
        else
        {
          // for newly created ghost dofsets we have to reconstruct ghost dofs
          mark_dofs(
              node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_GHOST_by_GP, dbcgids);
        }
      }
    }  // more than one dofset
  }  // end case C
  else
  {
    if (n_new == nullptr or n_old == nullptr) FOUR_C_THROW("this case should be done before");

    // how many sets of dofs?
    const int numDofSets_new = n_new->num_dof_sets();  //= dof_cellsets_new.size()

    if (numDofSets_new == 0)  // no values to set
    {
      return;  // do nothing for this node
    }

    //---------------------------------------------------------------------------------
    // D: surrounding elements cut at old and new timestep t^n and t^(n+1)
    //---------------------------------------------------------------------------------

    // case a): std-set at t^(n+1) -> try to identify vc-connection
    //                                    std found: - Check special case of tips with changed side!
    //                                                  -> Yes: Copy value
    //                                                  -> No:  Semilagrange
    //                                    ghost found and only one ghost dof:
    //                                                   -> copy (or Semilagrange)
    //                                    ghost found but not unique
    //                                                   -> Semilagrange
    // case b): ghost dofset at t^(n+1) -> try to identify vc-connection
    //                                    std or ghost found -> Copy
    //                                    not found -> mark for Ghost Penalty!

    //-------------------------------
    // t^n
    const std::vector<std::shared_ptr<Cut::NodalDofSet>>& dof_cellsets_old =
        n_old->nodal_dof_sets();


    //-------------------------------
    // t^(n+1)
    const std::vector<std::shared_ptr<Cut::NodalDofSet>>& dof_cellsets_new =
        n_new->nodal_dof_sets();

    // loop new dofsets
    for (std::vector<std::shared_ptr<Cut::NodalDofSet>>::const_iterator sets =
             dof_cellsets_new.begin();
        sets != dof_cellsets_new.end(); sets++)
    {
      const Cut::NodalDofSet* cell_set = &**sets;

      int nds_new = sets - dof_cellsets_new.begin();  // nodal dofset counter

      // is current set at t^(n+1) std or ghost or dofset
      bool is_std_set_np = cell_set->is_standard_dof_set();

      //-----------------------------------------------------------------
      // identify cellsets at t^n with current dofset at t^(n+1)
      int nds_old = identify_old_sets(
          n_old, n_new, dof_cellsets_old, cell_set);  // get nds number of corresponding old dofset
      //-----------------------------------------------------------------

      if (timeint_scheme_ ==
          Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP)
      {
        // check, if we already have an entry for that node!
        if (node_to_reconstr_method_.find(gid) != node_to_reconstr_method_.end())
        {
          // disable safety check in case of double cut if we project from other discretization
          nds_old = 0;

          if (nds_old < 0)
            FOUR_C_THROW(
                "Projection failed and there is no old dofset to be copied for node {}", gid);
        }
      }

      if (nds_old < 0)  // no set or not a unique set found
      {
#ifdef DEBUG_TIMINT
        Core::IO::cout << "XFLUID-TIMINIT CASE D: node " << gid
                       << ",\t no corresponding dofset found at time t^n for dofset " << nds_new
                       << " at time t^(n+1)" << Core::IO::endl;
#endif
        if (is_std_set_np)  // std at t^(n+1)
        {
          if (timeint_scheme_ == Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP)
            FOUR_C_THROW(
                "no corresponding dofset at t^n, choose a SL-based-approach for node {}", gid);
          else if (timeint_scheme_ ==
                       Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP or
                   timeint_scheme_ ==
                       Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP)
          {
            mark_dofs(
                node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_STD_by_SL, dbcgids);
          }
          else if (timeint_scheme_ ==
                   Inpar::XFEM::
                       Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP)
          {
            mark_dofs(node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_by_PROJ_from_DIS,
                dbcgids);
          }
          else
            FOUR_C_THROW("unknown Inpar::XFEM::Xf_TimIntScheme");
        }
        else  // ghost at t^(n+1)
        {
          mark_dofs(
              node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_GHOST_by_GP, dbcgids);
        }
      }
      else  // unique set found
      {
        bool is_std_set_n = dof_cellsets_old[nds_old]->is_standard_dof_set();

#ifdef DEBUG_TIMINT
        Core::IO::cout << "XFLUID-TIMINIT CASE D: node " << gid
                       << ",\t corresponding std(yes/no: " << is_std_set_n << ") dofset " << nds_old
                       << " found at time t^n for dofset " << nds_new << " at time t^(n+1)"
                       << Core::IO::endl;
#endif

        if (is_std_set_np and is_std_set_n)  // std at t^n and t^(n+1)
        {
          // copy values or use SL
          if (timeint_scheme_ ==
                  Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP or
              timeint_scheme_ ==
                  Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP or
              timeint_scheme_ ==
                  Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP)
          {
            copy_dofs(node, nds_new, nds_old, Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_STD,
                newRowStateVectors, oldRowStateVectors, dbcgids);
          }
          else if (timeint_scheme_ ==
                   Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP)
          {
            mark_dofs(
                node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_STD_by_SL, dbcgids);
          }
          else
            FOUR_C_THROW("unknown Inpar::XFEM::Xf_TimIntScheme");
        }
        else if (is_std_set_np and !is_std_set_n)  // ghost at t^n and std at t^(n+1)
        {
          // check number of ghost-dofsets
          // get the number of ghost dofsets -> this check is done in identify_old_sets
          bool ghost_set_unique = true;

          if (ghost_set_unique)  // ghost dofset is unique
          {
            // copy values

            if (timeint_scheme_ ==
                Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP)
            {
              copy_dofs(node, nds_new, nds_old, Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_GHOST,
                  newRowStateVectors, oldRowStateVectors, dbcgids);
            }
            else if (timeint_scheme_ ==
                     Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP)
            {
              // is a copy from ghost->std reasonable or not?
              copy_dofs(node, nds_new, nds_old, Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_GHOST,
                  newRowStateVectors, oldRowStateVectors, dbcgids);
            }
            else if (timeint_scheme_ ==
                     Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP)
            {
              mark_dofs(
                  node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_STD_by_SL, dbcgids);
            }
            else if (timeint_scheme_ ==
                     Inpar::XFEM::
                         Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP)
            {
              mark_dofs(node, nds_new, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_by_PROJ_from_DIS,
                  dbcgids);
            }
            else
              FOUR_C_THROW("unknown Inpar::XFEM::Xf_TimIntScheme");
          }
          else
          {
            FOUR_C_THROW("case should not be called");
            // not unique ghost dofset
            // mark_dofs(node, nds_new, newRowStateVectors,
            // Inpar::XFEM::Xf_TimeInt_SemiLagrange,dbcgids);
          }
        }
        else if (!is_std_set_np and is_std_set_n)  // ghost at t^(n+1)
        {
          // copy values, should be reasonable since dofsets identified
          copy_dofs(node, nds_new, nds_old, Inpar::XFEM::Xf_TimeInt_GHOST_by_COPY_from_STD,
              newRowStateVectors, oldRowStateVectors, dbcgids);
          //            mark_dofs(node, nds_new, newRowStateVectors,
          //            Inpar::XFEM::Xf_TimeInt_GHOST_by_GP,dbcgids);
        }
        else if (!is_std_set_np and !is_std_set_n)
        {
          // copy values, should be reasonable since dofsets identified
          //  FOUR_C_THROW("Copy for node {}", gid);
          copy_dofs(node, nds_new, nds_old, Inpar::XFEM::Xf_TimeInt_GHOST_by_COPY_from_GHOST,
              newRowStateVectors, oldRowStateVectors, dbcgids);
          //            mark_dofs(node, nds_new, newRowStateVectors,
          //            Inpar::XFEM::Xf_TimeInt_GHOST_by_GP,dbcgids);
        }
      }  // end found set at t^n

    }  // loop new dofsets
  }  // end case D (node handles at t^n and t^(n+1))

  return;
}

// -------------------------------------------------------------------
// get nodes and dofsets for given reconstruction method
// -------------------------------------------------------------------
std::map<int, std::set<int>>& XFEM::XFluidTimeInt::get_node_to_dof_map_for_reconstr(
    Inpar::XFEM::XFluidTimeInt reconstr)
{
  // returns empty map if reconstruction method is not present
  return reconstr_method_to_node_[reconstr];
}

// -------------------------------------------------------------------
// all surrounding elements non-intersected ?
// -------------------------------------------------------------------
bool XFEM::XFluidTimeInt::non_intersected_elements(Core::Nodes::Node* n, Cut::CutWizard& wizard)
{
  // loop surrounding elements
  for (auto ele : n->adjacent_elements())
  {
    // we have to check elements and its sub-elements in case of quadratic elements
    Cut::ElementHandle* ehandle = wizard.get_element(ele.user_element());

    // elements which do not have an element-handle are non-intersected anyway
    if (ehandle == nullptr) continue;

    // check if the element is intersected or not.
    // If the element is not intersected or is just fully or partially touched at a facet
    // a unique volume-cell for the element (each sub-element for quadratic elementhandles holding
    // the same position info) has been produced
    if (ehandle->is_intersected())
    {
      return false;  // at least one element is intersected by a cut-side
    }
  }

  return true;  // all surrounding elements uncut
}


// -------------------------------------------------------------------
// find all ghost dofsets around this node and its std-dofset
// -------------------------------------------------------------------
void XFEM::XFluidTimeInt::find_surrounding_ghost_dofsets(
    std::map<int, std::set<int>>&
        ghostDofsets,  /// surrounding ghost dofsets to be filled, map of ghost nodes and
                       /// corresponding ghost nds index w.r.t given std nodal dofset
    const Core::Nodes::Node* node,  /// node
    const int nds_new               /// dofset of node used for finding the surrounding ghost dofs
)
{
  if (nds_new != 0)
    FOUR_C_THROW("do you really want to find ghost dofsets surrounding a non-std node with id {}?",
        node->id());

  Cut::Node* n = wizard_new_->get_node(node->id());

  if (n == nullptr)
  {
    // it can happen that the node was std at t^n, however cut elements around it (nodehandle
    // available at t^n and SL called) and at new time t^(n+1) it is a std-node without a
    // nodehandle, then there are no surrounding ghost dofs at t^(n+1)
    return;
  }

  const std::vector<std::shared_ptr<Cut::NodalDofSet>>& dof_cellsets = n->nodal_dof_sets();


  // get the corresponding cellset
  const std::set<Cut::plain_volumecell_set, Cut::Cmp>& cellset =
      dof_cellsets[nds_new]->volume_cell_composite();

  // get for each plain_volumecell_set of the surrounding elements the ghost dofs

  // loop the surrounding (sub-)elements
  for (std::set<Cut::plain_volumecell_set, Cut::Cmp>::const_iterator e_vcset = cellset.begin();
      e_vcset != cellset.end(); e_vcset++)
  {
    const Cut::plain_volumecell_set& vcs = *e_vcset;

    // get the element, ask the first vc
    int peid = (*vcs.begin())->parent_element()->get_parent_id();
    Core::Elements::Element* ele = dis_->g_element(peid);
    Core::Nodes::Node** nodes = ele->nodes();

    const std::vector<int>& nds = (*vcs.begin())->nodal_dof_set();

    // which dofset is the corresponding to the current vc-connection and is a ghost dofset?
    for (int n_it = 0; n_it < ele->num_node(); n_it++)
    {
      const Core::Nodes::Node* ghost_node = nodes[n_it];
      const int ghost_nid = ghost_node->id();

      Cut::Node* ghost_node_cut = wizard_new_->get_node(ghost_nid);

      if (ghost_node_cut == nullptr)
        continue;  // this node is then a standard node or not on this proc

      // check if the neighbored node is a ghost node w.r.t to the cellset of the SL-node
      // if ghost node w.r.t. the cellset then add it to ghostDofsets with the corresponding
      // nds-number
      if (!(dof_cellsets[nds_new]->contains(ghost_node_cut->point())))
      {
        std::map<int, std::set<int>>::iterator map_it = ghostDofsets.find(ghost_nid);
        if (map_it == ghostDofsets.end())
        {
          std::set<int> tmp_map;

          tmp_map.insert(
              nds[n_it]);  // insert in map, use map as for hex20 elements, the subelements
          ghostDofsets.insert(std::pair<int, std::set<int>>(ghost_nid, tmp_map));
        }
        else
        {
          map_it->second.insert(nds[n_it]);
        }
      }
    }
  }
}



// -------------------------------------------------------------------
// copy dofs from old vectors to new vector for all row vectors
// -------------------------------------------------------------------
void XFEM::XFluidTimeInt::copy_dofs(const Core::Nodes::Node* node,  /// drt node
    const int nds_new,                                              /// nodal dofset at t^(n+1)
    const int nds_old,                                              /// nodal dofset at t^n
    const Inpar::XFEM::XFluidTimeInt method,                        /// reconstruction method
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
        newRowStateVectors,  /// row map based state vectors at t^(n+1)
    const std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
        oldRowStateVectors,                       /// row map based state vectors at t^n
    const std::shared_ptr<std::set<int>> dbcgids  /// set of DBC global ids)
)
{
  if (method != Inpar::XFEM::Xf_TimeInt_GHOST_by_COPY_from_GHOST and
      method != Inpar::XFEM::Xf_TimeInt_GHOST_by_COPY_from_STD and
      method != Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_GHOST and
      method != Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_STD)
    FOUR_C_THROW("don't call copy_dofs for non-copy reconstruction type");


  std::vector<int> dofs_old;
  std::vector<int> dofs_new;

  dofset_new_->dof(dofs_new, node, nds_new);
  dofset_old_->dof(dofs_old, node, nds_old);

#ifdef DEBUG_TIMINT
  if (dofs_new.size() != dofs_old.size())
  {
    FOUR_C_THROW(
        "XFLUID-TIMINIT: unequal number of dofs for node {} at old timestep (set {}, {} dofs) and "
        "new timestep (set {}, {} dofs)",
        node->Id(), nds_old, dofs_old.size(), nds_new, dofs_new.size());
  }
#endif

  // try to set the reconstruction method, in case that it has been already set to Ghost-penalty, do
  // not overwrite it
  if (!set_reconstr_method(node, nds_new, method)) return;

  int vec_count = 0;

  // copy values for all vectors
  for (std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>::const_iterator it =
           oldRowStateVectors.begin();
      it != oldRowStateVectors.end(); it++)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> vec_new = newRowStateVectors[vec_count];
    std::shared_ptr<const Core::LinAlg::Vector<double>> vec_old = oldRowStateVectors[vec_count];

    // copy values from old vector to new vector
    for (size_t i = 0; i < dofs_new.size(); i++)
    {
      int dof_lid_new = vec_new->get_map().lid(dofs_new[i]);
      if (dof_lid_new == -1) FOUR_C_THROW("new dof {} not local on this proc!", dofs_new[i]);

      int dof_lid_old = vec_old->get_map().lid(dofs_old[i]);
      if (dof_lid_old == -1) FOUR_C_THROW("old dof {} not local on this proc!", dofs_old[i]);

      (*vec_new).get_values()[dof_lid_new] = (*vec_old).get_values()[dof_lid_old];

      // set Dirichlet BC for ghost penalty reconstruction
      int gid = dofs_new[i];

      if (dbcgids != nullptr) (*dbcgids).insert(gid);
    }

    vec_count++;
  }

  // create permutation cycles
  if ((method == Inpar::XFEM::Xf_TimeInt_GHOST_by_COPY_from_GHOST or
          method == Inpar::XFEM::Xf_TimeInt_GHOST_by_COPY_from_STD or
          method == Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_GHOST or
          method ==
              Inpar::XFEM::Xf_TimeInt_STD_by_COPY_from_STD  // no permutation for std-to-std as std
                                                            // is always the first, nevertheless
                                                            // std-to-std reasonable for some cases?
          ) and
      nds_new != nds_old)
  {
    // std::cout << "copying from ghost to ghost for node " << node->Id() << " set old: " << nds_old
    // << " set new: " << nds_new << std::endl;

    // copy values from old vector to new vector
    for (size_t i = 0; i < dofs_new.size(); i++)
    {
      int dof_gid_new = dofs_new[i];
      int dof_gid_old = dofs_old[i];

      permutation_map_->insert(std::pair<int, int>(dof_gid_old, dof_gid_new));
    }
  }
  return;
}


// -------------------------------------------------------------------
// mark nodal dofs of vector w.r.t new interface position for reconstruction
// -------------------------------------------------------------------
void XFEM::XFluidTimeInt::mark_dofs(const Core::Nodes::Node* node,  /// drt node
    const int nds_new,                                              /// nodal dofset at t^(n+1)
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
        newRowStateVectors,                   /// row map based state vectors at t^(n+1)
    const Inpar::XFEM::XFluidTimeInt method,  /// reconstruction method
    const std::shared_ptr<std::set<int>>
        dbcgids  /// set of dof gids that must not be changed by ghost penalty reconstruction
)
{
  // try to set the reconstruction method, in case that it has been already set to Ghost-penalty, do
  // not overwrite it
  if (!set_reconstr_method(node, nds_new, method)) return;


  //-------------------------------------
  // for the Semi-Lagrangean algorithm applied to single nodes we reconstruct the surrounding ghost
  // nodes with the Ghost-penalty approach this is done to avoid kinks in the reconstructed field
  // when only std-values are computed using SL but ghost-values would be copied in case of the full
  // SL-algorithm in the boundary zone, all ghost-values will be reconstructed using the GP-approach
  // (this is done in the transfer_dofs_to_new_map directly)
  if ((timeint_scheme_ == Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP or
          timeint_scheme_ == Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP) and
      method == Inpar::XFEM::Xf_TimeInt_STD_by_SL)
  {
    // map of global nid of a neighbored ghost node and the corresponding ghost-dofset-number which
    // corresponds to the given std dofset at the node
    std::map<int, std::set<int>> ghostDofsets;
    find_surrounding_ghost_dofsets(ghostDofsets, node, nds_new);

    for (std::map<int, std::set<int>>::iterator it = ghostDofsets.begin(); it != ghostDofsets.end();
        it++)
    {
      // mark the ghost dofset in case that the node is a row node on this proc
      // otherwise mark it for export at the end of transfer_dofs_to_new_map
      int nid = it->first;
      std::set<int>& nds = it->second;
      for (std::set<int>::iterator nds_it = nds.begin(); nds_it != nds.end(); nds_it++)
      {
        const int dofset = *nds_it;  // dofset number for ghost node

        // comment this block if we do not want to reconstruct ghost values of ghost nodes around
        // SL-standard nodes
        if (dis_->node_row_map()->lid(nid) != -1)  // is row node on this proc
        {
          Core::Nodes::Node* n = dis_->g_node(nid);
          mark_dofs(n, dofset, newRowStateVectors, Inpar::XFEM::Xf_TimeInt_GHOST_by_GP, dbcgids);
        }
        else
        {
          // in parallel we have to export the info to other procs
          mark_dofs_for_export(nid, dofset, Inpar::XFEM::Xf_TimeInt_GHOST_by_GP);
        }
      }
    }
  }
  //-------------------------------------

  // get nodal dofs for current dofset w.r.t new interface position
  std::vector<int> dofs_new;
  dofset_new_->dof(dofs_new, node, nds_new);

  // loop vectors
  for (std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>::const_iterator it =
           newRowStateVectors.begin();
      it != newRowStateVectors.end(); it++)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> vec_new = *it;

    // set a dummy value for dofs in the vector
    for (size_t i = 0; i < dofs_new.size(); i++)
    {
      int dof_lid_new = vec_new->get_map().lid(dofs_new[i]);
      if (dof_lid_new == -1) FOUR_C_THROW("new dof {} not local on this proc!", dofs_new[i]);

      (*vec_new).get_values()[dof_lid_new] =
          0.0;  // zero value as start value for newton for gradient reconstruction

      int gid = dofs_new[i];

      // set Dirichlet BC for ghost penalty reconstruction
      if (method != Inpar::XFEM::Xf_TimeInt_GHOST_by_GP)
      {
        if (dbcgids != nullptr) (*dbcgids).insert(gid);
      }
      else
      {
        // find if the dbc has been already set by a previous set non-ghost-penalty reconstruction
        // and ensure that no dbc is set
        std::set<int>::iterator it = (*dbcgids).find(gid);
        if (it != (*dbcgids).end()) (*dbcgids).erase(gid);  // remove the already set dbc
      }
    }  // dofs
  }  // state vectors

  return;
}

// -------------------------------------------------------------------
// mark one specific nodal dofset with used for export
// -------------------------------------------------------------------
void XFEM::XFluidTimeInt::mark_dofs_for_export(const int nid,  /// node id
    const int dofset,                                          /// ghost dofset number
    const Inpar::XFEM::XFluidTimeInt
        method  /// reconstruction method used for marking the nodal dofset
)
{
  std::map<int, std::map<int, int>>::iterator it = dofset_marker_export_.find(nid);

  if (it != dofset_marker_export_.end())
  {
    std::map<int, int>& dofsets_map = it->second;
    std::map<int, int>::iterator nds_it = dofsets_map.find(dofset);
    if (nds_it != dofsets_map.end())
    {
      if (nds_it->second != (int)method)
        FOUR_C_THROW(
            "you want to mark one dofset twice with different methods! nid: {}, dofset: {}, set "
            "method: {}, new method: {}",
            nid, dofset, (int)nds_it->second, (int)method);
      else
        return;  // already set
    }
    else
    {
      dofsets_map.insert(std::pair<int, int>(dofset, (int)method));
    }
  }
  else
  {
    std::map<int, int> dofsets_map;
    dofsets_map.insert(std::pair<int, int>(dofset, (int)method));

    dofset_marker_export_.insert(std::pair<int, std::map<int, int>>(nid, dofsets_map));
  }
}


// -------------------------------------------------------------------
// set the reconstruction method for current nodal dofset, return if set
// -------------------------------------------------------------------
bool XFEM::XFluidTimeInt::set_reconstr_method(const Core::Nodes::Node* node,  /// drt node
    const int nds_new,                       /// nodal dofset w.r.t new interface position
    const Inpar::XFEM::XFluidTimeInt method  /// which type of reconstruction method
)
{
  Cut::Node* n = wizard_new_->get_node(node->id());

  int numdofsets = -1;
  if (n != nullptr)
  {
    numdofsets = n->num_dof_sets();
  }
  else
    numdofsets = 1;  // one std dofset

  // find the node in map
  std::map<int, std::vector<Inpar::XFEM::XFluidTimeInt>>::iterator it;
  it = node_to_reconstr_method_.find(node->id());

  if (it != node_to_reconstr_method_.end())
  {
    // assume that we call this function for all nodal dofsets in the right order
    if (nds_new >= numdofsets) FOUR_C_THROW("no valid nds {} for node {}", nds_new, n->id());

    //---------------------------------------------------------
    // check if the current set method can be overwritten

    // do not overwrite Ghost-penalty reconstruction method which have been already marked by
    // surrounding SL-dofsets
    if (it->second[nds_new] == Inpar::XFEM::Xf_TimeInt_GHOST_by_GP) return false;

    // do not overwrite already values, expect that method is Xf_TimeInt_GHOST_by_GP
    if (it->second[nds_new] != Inpar::XFEM::Xf_TimeInt_undefined and  // overwriting
        it->second[nds_new] != method and                             // methods not equal
        method != Inpar::XFEM::Xf_TimeInt_GHOST_by_GP and  // overwriting with non-GP approach
        it->second[nds_new] !=
            Inpar::XFEM::Xf_TimeInt_by_PROJ_from_DIS  // overwriting projection after previous
                                                      // failure (label correction)
    )
    {
      FOUR_C_THROW(
          "inconsistency in reconstruction method, why do want to replace reconstruction method {} "
          "with {} for node={} and nds={}",
          it->second[nds_new], method, node->id(), nds_new);
    }
    //---------------------------------------------------------

    // remove the node from the old method in the reconstr. method-to-node map
    std::map<Inpar::XFEM::XFluidTimeInt, std::map<int, std::set<int>>>::iterator itm;
    itm = reconstr_method_to_node_.find(it->second[nds_new]);

    if (itm != reconstr_method_to_node_.end())
    {
      std::map<int, std::set<int>>::iterator itn = itm->second.find(node->id());
      if (itn != itm->second.end()) itn->second.erase(nds_new);

      if (itn->second.empty()) itm->second.erase(node->id());
    }

    (it->second)[nds_new] = method;
    reconstr_method_to_node_[method][node->id()].insert(nds_new);

    return true;
  }
  else
  {
    std::vector<Inpar::XFEM::XFluidTimeInt> vec(
        numdofsets, Inpar::XFEM::Xf_TimeInt_undefined);  // initialize with undefined status
    vec[nds_new] = method;
    node_to_reconstr_method_.insert(
        std::pair<int, std::vector<Inpar::XFEM::XFluidTimeInt>>(node->id(), vec));
    reconstr_method_to_node_[method][node->id()].insert(nds_new);
    return true;
  }

  return false;
}


// -------------------------------------------------------------------
// identify cellsets at time t^n with cellsets at time t^(n+1)
// -------------------------------------------------------------------
int XFEM::XFluidTimeInt::identify_old_sets(const Cut::Node* n_old,  /// node w.r.t to old wizard
    const Cut::Node* n_new,                                         /// node w.r.t to new wizard
    const std::vector<std::shared_ptr<Cut::NodalDofSet>>&
        dof_cellsets_old,                 /// all dofcellsets at t^n
    const Cut::NodalDofSet* cell_set_new  /// dofcellset at t^(n+1) which has to be identified
)
{
  // is current set at t^(n+1) std or ghost or dofset
  bool is_std_set_np = cell_set_new->is_standard_dof_set();

  const Cut::Point::PointPosition pos_new = cell_set_new->position();

  //--------------------------------------------------------
  // t^(n+1)
  // set of side-ids involved in cutting the current connection of volumecells at t^(n+1)
  // get all side-ids w.r.t to all volumecells contained in current new set around the current node

  Cut::plain_int_set cutsides_new;
  cell_set_new->collect_cut_sides(cutsides_new);

  //--------------------------------------------------------
  // t^n

  std::map<int, std::vector<int>>
      identified_old_sets;  // map of possible nds_sets and related identified sides

  //--------------------------------------------------------
  // PRESELECTION via common cutting sides (for level-sets all sets have the same level-set side and
  // are possible sets)
  //--------------------------------------------------------
  // check each old dofset for identification with new dofset
  for (std::vector<std::shared_ptr<Cut::NodalDofSet>>::const_iterator old_sets =
           dof_cellsets_old.begin();
      old_sets != dof_cellsets_old.end(); old_sets++)
  {
    const int setnumber = old_sets - dof_cellsets_old.begin();

    Cut::plain_int_set cutsides_old;
    (*old_sets)->collect_cut_sides(cutsides_old);

    //--------------------------------------------------------------
    // check if identification of sets is possible
    // -> check if any side is involved in both cuts (find side-Id involved at time t^(n+1) in set
    // of involved side-Ids at t^n)
    //--------------------------------------------------------------

    // get the common sides of both sets!
    std::vector<int> common_sides;
    common_sides.reserve(std::max(cutsides_new.size(), cutsides_old.size()));

    // sorted vectors are already sorted
    if (cutsides_old.size() > 0 and cutsides_new.size() > 0)
      std::set_intersection(cutsides_new.begin(), cutsides_new.end(), cutsides_old.begin(),
          cutsides_old.end(), std::back_inserter(common_sides));  // back_inserter uses pushback


    if (common_sides.size() > 0)
      identified_old_sets[setnumber] =
          common_sides;  // [] creates a new vector of common side-ids if not created yet
  }

  // "rotation" of interface around a point outside the volumecell composite, such that no common
  // cut side is available for the two sets of cut-sides Nevertheless the standard dofset (if there
  // exists one) could be still a good choice check if there is a standard dofset which is
  // reasonable nevertheless
  if (identified_old_sets.size() == 0  // no identification via common cutsides possible
      and is_std_set_np  // it is a standard dofset at the new time step, for which the alternative
                         // would be only to use SemiLagrangean
      and
      timeint_scheme_ ==
          Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP  // just when we need to
                                                                             // copy data, otherwise
                                                                             // simulation will stop
  )
  {
    // look again in the old sets if there was a standard dofset
    for (std::vector<std::shared_ptr<Cut::NodalDofSet>>::const_iterator old_sets =
             dof_cellsets_old.begin();
        old_sets != dof_cellsets_old.end(); old_sets++)
    {
      if ((*old_sets)->is_standard_dof_set())  // this set is a unique standard set at tn, might it
                                               // be a good choice nevertheless?
      {
        const int setnumber = old_sets - dof_cellsets_old.begin();

        //  the following variable is used to define if the unique standard dofset should be used.
        //  However, this might be unsafe. Here, the standard set is used since it is assumed that
        //  there is at least a common node which is shared by the two sets of cutsides
        //  TODO: an alternative would be to check if there is at least one common node or a common
        //  edge between the two sets of cutsides, this might be more safe!
        bool use_old_std_set = true;

        if (use_old_std_set)
        {
          Core::IO::cout
              << "WARNING (xfluid_timeInt.cpp): for std-dofset (t^(n+1)) at node " << n_new->id()
              << " no std dofset at t^n could be identified via common sides."
              << " However, there is a std-dofset (t^n) which could be used nevertheless! "
                 "This might be unsafe! Be aware of that!"
              << Core::IO::endl;

          std::vector<int> dummy;  // dummy for common sides ( actually no common sides available! )
          identified_old_sets[setnumber] =
              dummy;  // [] creates a new vector of common side-ids if not created yet
        }
      }
    }
  }

  //---------------------------------------
  // try to find a unique candidate and do safety check if there is only one possible set
  //---------------------------------------

  std::set<int> invalid_old_sets;  // find invalid cell-sets and erase them from identified_old_sets
                                   // afterwards

  if (identified_old_sets.size() > 1)
  {
    bool unique_set_found = false;

    for (std::map<int, std::vector<int>>::iterator it = identified_old_sets.begin();
        it != identified_old_sets.end(); it++)
    {
      const int nds_old = it->first;

      if (unique_set_found)
      {
        // make the other sets invalid
        invalid_old_sets.insert(nds_old);
        continue;
      }

      const Cut::Point::PointPosition pos_old = dof_cellsets_old[nds_old]->position();

      if (pos_old == Cut::Point::undecided or pos_old == Cut::Point::oncutsurface)
        FOUR_C_THROW(
            "why is the cellcet position undecided or oncutsurface {}, something wrong", pos_old);

      // dof-cellsets at new and old time have to correspond to each other w.r.t position of their
      // cellsets (same fluid phase)
      if (pos_old == pos_new)
      {
        bool is_std_set_n = dof_cellsets_old[nds_old]->is_standard_dof_set();

        if (is_std_set_n)  // standard sets are unique, therefore accept the unique set and neglect
                           // the others
          unique_set_found = true;
        else
          unique_set_found = false;
      }
      else  // changing dofset positions, not allowed here
      {
        // make the other sets invalid
        invalid_old_sets.insert(nds_old);
        continue;
      }
    }  // loop old sets
  }  // identified_dofsets > 1

  //---------------------------------------
  // remove invalid sets
  //---------------------------------------
  for (std::set<int>::iterator set_it = invalid_old_sets.begin(); set_it != invalid_old_sets.end();
      set_it++)
  {
    identified_old_sets.erase(*set_it);
  }

  //---------------------------------------
  // no unique reasonable candidate found
  //---------------------------------------
  if (identified_old_sets.size() > 1)
  {
#ifdef DEBUG_TIMINT
    Core::IO::cout << "Warning: found dofset at t^n not unique, found "
                   << identified_old_sets.size() << " dofsets, set status to not-found!"
                   << Core::IO::endl;
#endif
    return -1;
  }

  //---------------------------------------
  // no reasonable candidate available
  //---------------------------------------
  if (identified_old_sets.size() <= 0)
  {
#ifdef DEBUG_TIMINT
    Core::IO::cout << "Warning: no found dofset at t^n, set status to not-found!" << Core::IO::endl;
#endif
    return -1;
  }

  //---------------------------------------
  // special check for the unique candidate / can we really accept the value?
  //---------------------------------------
  if (identified_old_sets.size() == 1)
  {
    const int nds_old = identified_old_sets.begin()->first;

#ifdef DEBUG_TIMINT
    Core::IO::cout << "Exactly one dofset found: nds= " << nds_old
                   << " perform further special checks:" << Core::IO::endl;
#endif

    bool is_std_set_n = dof_cellsets_old[nds_old]->is_standard_dof_set();

    //---------------------------------------
    // do special checks
    //---------------------------------------


    bool did_node_change_side = false;
    bool successful_check = true;

    if (xfluid_timint_check_sliding_on_surface_)
    {
      //--------------------------------------
      // special check whether node slides on cut surface
      successful_check = special_check_sliding_on_surface(did_node_change_side, n_old, n_new);

      if (!successful_check or (successful_check and did_node_change_side))
        return -1;  // do not accept the old value
    }

    if (xfluid_timint_check_interfacetips_)
    {
      //--------------------------------------
      // special check for interface tips if the node has changed the side w.r.t identified sides at
      // t^n and t^(n+1)
      if ((is_std_set_np and is_std_set_n) or (!is_std_set_np and !is_std_set_n))
      {
        std::vector<int>& identified_sides = identified_old_sets[nds_old];
        successful_check =
            special_check_interface_tips(did_node_change_side, identified_sides, n_old, n_new);

        if (!successful_check or (successful_check and did_node_change_side))
          return -1;  // do not accept the old value
      }
    }

    //---------------------------------------
    // if the unique candidate passed all checks we accept the value
    //---------------------------------------
#ifdef DEBUG_TIMINT
    Core::IO::cout << "\t ACCEPT the unique dofset!" << Core::IO::endl;
#endif

    return nds_old;  // all special tests passed -> accept the old value
  }

  return -1;
}


bool XFEM::XFluidTimeInt::special_check_sliding_on_surface(bool& changed_side,
    const Cut::Node* n_old,  /// node w.r.t to old wizard
    const Cut::Node* n_new   /// node w.r.t to new wizard
)
{
  if (n_old->id() != n_new->id())
    FOUR_C_THROW("XFLUID-TIMINT: not the same node for CheckChangingSide");

  //--------------------------------------------------
  // do some simple checks for changing side based on point positions to reduce checks based on
  // space-time sides

  // check when point moves on the surface of the same side
  Cut::Point* p_old = n_old->point();
  Cut::Point* p_new = n_new->point();

  Cut::Point::PointPosition pos_old = p_old->position();
  Cut::Point::PointPosition pos_new = p_new->position();

  if ((pos_old == Cut::Point::oncutsurface and pos_new == Cut::Point::oncutsurface))
  {
    // first case: point slides on the cut surface (within one side or from 'within the side' to
    // point or edge and vice versa) second case: point moves from one side to another side (->
    // decide if the sides are neighbors and so on)

    const Cut::plain_side_set& cut_sides_old = p_old->cut_sides();
    const Cut::plain_side_set& cut_sides_new = p_new->cut_sides();


#ifdef DEBUG_TIMINT
    Core::IO::cout << "!!WARNING point is on cut surface at time t^n and t^(n+1)" << Core::IO::endl;
#endif


    // check if there is at least one common cutside at both times
    if (cut_sides_old.size() == 0 or cut_sides_new.size() == 0)
      FOUR_C_THROW("there are no cutsides but point is 'oncutsurface' at both times");

    // REMARK: we have to check this based on the real sides not based on the triangulated cutsides
    // get the side-Ids

    std::set<int> on_cut_sides_old;
    std::set<int> on_cut_sides_new;

    // loop cutsides and extract side ids
    for (Cut::plain_side_set::const_iterator side_it = cut_sides_old.begin();
        side_it != cut_sides_old.end(); side_it++)
    {
      int sid = (*side_it)->id();
      if (sid != -1) on_cut_sides_old.insert(sid);  // do not insert sides of the background element
    }

    // loop cutsides and extract side ids
    for (Cut::plain_side_set::const_iterator side_it = cut_sides_new.begin();
        side_it != cut_sides_new.end(); side_it++)
    {
      int sid = (*side_it)->id();
      if (sid != -1) on_cut_sides_new.insert(sid);  // do not insert sides of the background element
    }


    // check if there is at least one common cut side the point lies on
    // this checks the case when point moves from inside the side to an edge or cornerpoint of the
    // same side
    //------------------------------------------------------
    // find common side

    while (true)
    {
      // if at least one vector of nodes is checked
      if ((int)on_cut_sides_new.size() == 0 or (int) on_cut_sides_old.size() == 0) break;

      if (*(on_cut_sides_new.begin()) < *(on_cut_sides_old.begin()))
        on_cut_sides_new.erase(on_cut_sides_new.begin());
      else if (*(on_cut_sides_new.begin()) > *(on_cut_sides_old.begin()))
        on_cut_sides_old.erase(on_cut_sides_old.begin());
      else  // on_cut_sides_new.begin() == on_cut_sides_old.begin()
      {
#ifdef DEBUG_TIMINT
        Core::IO::cout << "point moves within side " << *(on_cut_sides_new.begin())
                       << " oncutsurface!" << Core::IO::endl;
#endif

        // no changing side
        changed_side = false;
        return true;
      }
    }

    //------------------------------------------------------
    // if this check was not successful, check if two sides are at least neighbors
    FOUR_C_THROW(
        "point moves oncutsurface from t^n to t^(n+1), but remains not within the same side, check "
        "if a 'changed side' maybe based on side neighbors would be possible");
    //------------------------------------------------------

    changed_side = false;
    return true;
  }

  changed_side = false;
  return true;
}


// -------------------------------------------------------------------
// check if the node has changed the side w.r.t identified sides at t^n and t^(n+1)
// return if check was successful
// -------------------------------------------------------------------
bool XFEM::XFluidTimeInt::special_check_interface_tips(
    bool& changed_side,                  /// did the node change the side ?
    std::vector<int>& identified_sides,  /// side Id of identified side
    const Cut::Node* n_old,              /// node w.r.t to old wizard
    const Cut::Node* n_new               /// node w.r.t to new wizard
)
{
  bool successful_check = true;

  changed_side = false;

  if (n_old->id() != n_new->id())
    FOUR_C_THROW("XFLUID-TIMINT: not the same node for CheckChangingSide");

  //--------------------------------------------------
  // do some simple checks for changing side based on point positions to reduce checks based on
  // space-time sides

  // check when point moves on the surface of the same side
  Cut::Point* p_old = n_old->point();
  Cut::Point* p_new = n_new->point();


  Cut::Point::PointPosition pos_old = p_old->position();
  Cut::Point::PointPosition pos_new = p_new->position();

  if (pos_old == Cut::Point::undecided or pos_new == Cut::Point::undecided)
    FOUR_C_THROW("at least one node position undecided!");

  if ((pos_old == Cut::Point::outside and pos_new == Cut::Point::outside) or
      (pos_old == Cut::Point::inside and pos_new == Cut::Point::inside))
  {
    // continue with check based on space time sides
    // REMARK: the point could have moved through an inside/outside volumecell

    // do nothing and continue with space-time side check
  }
  else  // otherwise check not required, continue with other special checks
  {
    changed_side = false;
    return true;
  }


  //------------------------------------

  Core::LinAlg::Matrix<3, 1> n_coord_old(Core::LinAlg::Initialization::zero);
  n_old->coordinates(&n_coord_old(0, 0));

  Core::LinAlg::Matrix<3, 1> n_coord_new(Core::LinAlg::Initialization::zero);
  n_new->coordinates(&n_coord_new(0, 0));

  // check if moving node (ALE case)
  Core::LinAlg::Matrix<3, 1> n_diff(Core::LinAlg::Initialization::zero);
  n_diff.update(1.0, n_coord_new, -1.0, n_coord_old);

  // TODO: for ALE we have to check whether the path of the point crosses at least one space-time
  // side element
  if (n_diff.norm2() > 1e-14)
  {
    // TODO: currently we expect that node did not change the side around a tip
    // TODO: USE the FOUR_C_THROW, at the moment we just throw a warning
    Core::IO::cout
        << "WARNING: node " << n_old->id()
        << " seems to move (background ALE?). Check the special_check_interface_tips-Check for ALE!"
        << Core::IO::endl;

    changed_side = false;
    return true;

    //  FOUR_C_THROW("node {} seems to move. Check the SpaceTimeChangingSide-Check for ALE!",
    // n_old->Id());
  }



  // check the change of the node w.r.t to sides using space-time side elements
  // create space-time side elements using the same side at t^n and t^(n+1)
  // check if the node is contained in such space-time surface elements at any time between t^n and
  // t^(n+1)

  // loop sides
  for (std::vector<int>::iterator sides = identified_sides.begin(); sides != identified_sides.end();
      sides++)
  {
    const int coup_sid = *sides;  // side id used within the cut

    if (condition_manager_->is_mesh_coupling(coup_sid))
    {
      Core::Elements::Element* side = condition_manager_->get_side(coup_sid);

      successful_check =
          special_check_interface_tips_space_time(changed_side, side, coup_sid, n_coord_old);
    }
    else
    {
      if (n_diff.norm2() > 1e-14)
        FOUR_C_THROW(
            "background fluid ALE with level-sets interface??? Think about that, does Scatra "
            "support this?");

      successful_check = special_check_interface_tips_levelset(changed_side);
    }

    if (!successful_check) return false;  // return non-successful check

    if (changed_side)
    {
#ifdef DEBUG_TIMINT
      Core::IO::cout
          << "\t\t node " << n_new->Id()
          << " changed interface side, node within space-time side element for at least one side"
          << Core::IO::endl;
#endif
      break;  // stop loop over sides
    }
  }

  return successful_check;
}

bool XFEM::XFluidTimeInt::special_check_interface_tips_levelset(
    bool& changed_side  /// did the node change the side ?
)
{
  // interface tips moving/rotating around a node not supported yet
  // TODO: if necessary, implement a check based on the gradients of the level-set field or the
  // normal vectors of the boundary cells possibility: check the different directions of the normal
  // vectors up to a certain tolerance

  // assume that the interface did not change the side and accept the value
  changed_side = false;
  return true;
}

bool XFEM::XFluidTimeInt::special_check_interface_tips_space_time(
    bool& changed_side,  /// did the node change the side ?
    Core::Elements::Element* side, const int coup_sid,
    const Core::LinAlg::Matrix<3, 1>& n_coord  /// node coordinates
)
{
  bool node_within_Space_Time_Side = false;

  Core::FE::CellType side_distype = side->shape();

  bool successful_check = false;

  switch (side_distype)
  {
    case Core::FE::CellType::tri3:
    {
      successful_check =
          within_space_time_side<Core::FE::CellType::tri3, Core::FE::CellType::wedge6>(
              node_within_Space_Time_Side, side, coup_sid, n_coord);
      break;
    }
    case Core::FE::CellType::quad4:
    {
      successful_check =
          within_space_time_side<Core::FE::CellType::quad4, Core::FE::CellType::hex8>(
              node_within_Space_Time_Side, side, coup_sid, n_coord);
      break;
    }
    case Core::FE::CellType::quad8:
    {
      successful_check =
          within_space_time_side<Core::FE::CellType::quad8, Core::FE::CellType::hex16>(
              node_within_Space_Time_Side, side, coup_sid, n_coord);
      break;
    }
    case Core::FE::CellType::quad9:
    {
      successful_check =
          within_space_time_side<Core::FE::CellType::quad9, Core::FE::CellType::hex18>(
              node_within_Space_Time_Side, side, coup_sid, n_coord);
      break;
    }
    default:
      FOUR_C_THROW("side-distype {} not handled", Core::FE::cell_type_to_string(side_distype));
      break;
  }

  changed_side = node_within_Space_Time_Side;

  return successful_check;
}

// -------------------------------------------------------------------
// check if the node is within the space time side
// -------------------------------------------------------------------
template <Core::FE::CellType side_distype,
    Core::FE::CellType space_time_distype>
bool XFEM::XFluidTimeInt::within_space_time_side(
    bool& within_space_time_side,  /// within the space time side
    Core::Elements::Element* side, const int coup_sid,
    const Core::LinAlg::Matrix<3, 1>& n_coord  /// node coordinates
)
{
  // get the right cutter discretization for the given side
  std::shared_ptr<Core::FE::Discretization> cutter_dis =
      condition_manager_->get_cutter_dis(coup_sid);

  std::string state_new = "idispnp";
  std::string state_old = "";

  if (is_newton_increment_transfer_)
    state_old = "idispnpi";  // get displacements from last newton increment
  else
    state_old = "idispn";  // get displacements from last time step

  // get state of the global vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> idisp_new = cutter_dis->get_state(state_new);
  std::shared_ptr<const Core::LinAlg::Vector<double>> idisp_old = cutter_dis->get_state(state_old);


  const int numnode_space_time = Core::FE::num_nodes(space_time_distype);

  const int numnode_side = Core::FE::num_nodes(space_time_distype) / 2;

  // space time side coordinates
  Core::LinAlg::Matrix<3, numnode_space_time> xyze_st;

  const int numnode = side->num_node();
  Core::Nodes::Node** nodes = side->nodes();


  Core::LinAlg::SerialDenseMatrix xyze_old(3, numnode);
  Core::LinAlg::SerialDenseMatrix xyze_new(3, numnode);

  for (int i = 0; i < numnode; ++i)
  {
    Core::Nodes::Node& node = *nodes[i];

    Core::LinAlg::Matrix<3, 1> x_old(node.x().data());
    Core::LinAlg::Matrix<3, 1> x_new(node.x().data());

    std::vector<int> lm;
    std::vector<double> mydisp_new;

    cutter_dis->dof(&node, lm);

    std::vector<double> mydisp_old = Core::FE::extract_values(*idisp_old, lm);
    mydisp_new = Core::FE::extract_values(*idisp_new, lm);

    // add displacements
    x_old(0) += mydisp_old.at(0);
    x_old(1) += mydisp_old.at(1);
    x_old(2) += mydisp_old.at(2);

    x_new(0) += mydisp_new.at(0);
    x_new(1) += mydisp_new.at(1);
    x_new(2) += mydisp_new.at(2);

    std::copy(x_old.data(), x_old.data() + 3, &xyze_old(0, i));
    std::copy(x_new.data(), x_new.data() + 3, &xyze_new(0, i));
  }

  //------------------------------------------------------------------

  for (int i = 0; i < numnode_side; ++i)
  {
    for (int j = 0; j < 3; j++)
    {
      xyze_st(j, i) = xyze_old(j, i);
      xyze_st(j, i + numnode_side) = xyze_new(j, i);
    }
  }

  // check the space time side volume
  bool successful_check = check_st_side_volume<space_time_distype, numnode_space_time>(xyze_st);

  if (!successful_check)  // apply small artificial movement in normal direction(translation)
  {
    double perturbation = 1e-008;

#ifdef DEBUG_TIMINT
    Core::IO::cout
        << "\t\t\t Not Successful! -> Apply small artificial translation in normal direction "
           "at t^(n+1) of "
        << perturbation << Core::IO::endl;
#endif

    //------------------------------------------------------------------
    // get normal vector on side at new interface position at center of side

    Core::LinAlg::Matrix<3, 1> normal(Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<2, 1> xi_side(Core::LinAlg::Initialization::zero);

    if (numnode_side == 3)
    {
      xi_side(0) = 0.3333333333333333;
      xi_side(1) = 0.3333333333333333;
    }
    else if (numnode_side == 4 or numnode_side == 8 or numnode_side == 9)
    {
      xi_side(0) = 0.0;
      xi_side(1) = 0.0;
    }
    else
      FOUR_C_THROW("unknown side type with {} nodes", numnode_side);

    // Initialization
    Core::LinAlg::Matrix<2, numnode_side> deriv(
        Core::LinAlg::Initialization::zero);  // derivatives dr, ds

    Core::LinAlg::Matrix<3, 2> derxy(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> dx_dr(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> dx_ds(Core::LinAlg::Initialization::zero);

    // get current values
    Core::FE::shape_function_2d_deriv1(deriv, xi_side(0), xi_side(1), side_distype);

    Core::LinAlg::Matrix<3, numnode_side> xyz_side_new(xyze_new);

    derxy.multiply_nt(xyz_side_new, deriv);


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

    //------------------------------------------------------------------

    for (int i = 0; i < numnode_side; ++i)
    {
      for (int j = 0; j < 3; j++)
      {
        xyze_st(j, i) = xyze_old(j, i);
        xyze_st(j, i + numnode_side) = xyze_new(j, i) + perturbation * normal(j, 0);
      }
    }

    successful_check = check_st_side_volume<space_time_distype, numnode_space_time>(xyze_st);

#ifdef DEBUG_TIMINT
    if (successful_check)
      Core::IO::cout << "\t\t\t Successful check!" << Core::IO::endl;
    else
      Core::IO::cout << "\t\t\t Again not successful check!" << Core::IO::endl;
#endif
  }

  if (!successful_check) return successful_check;

  std::shared_ptr<Cut::Position> pos =
      Cut::PositionFactory::build_position<3, space_time_distype>(xyze_st, n_coord);
  within_space_time_side = pos->compute();

#ifdef DEBUG_TIMINT
  Core::LinAlg::Matrix<3, 1> rst(
      Core::LinAlg::Initialization::zero);  // local coordinates w.r.t space time element (r,s,t
                                            // !!!)
  pos->local_coordinates(rst);

  if (within_space_time_side)
  {
    Core::IO::cout << "rst " << rst << Core::IO::endl;
    Core::IO::cout << "\t\t\t changing side found!"
                   << "new: " << xyze_new << " old: " << xyze_old << " \n global coords " << n_coord
                   << " \n local coords " << rst << Core::IO::endl;
  }
#endif

  return successful_check;
}


// -------------------------------------------------------------------
// check the volume of the space time side, distorted space-time side ?
// -------------------------------------------------------------------
template <Core::FE::CellType space_time_distype, const int numnode_space_time>
bool XFEM::XFluidTimeInt::check_st_side_volume(
    const Core::LinAlg::Matrix<3, numnode_space_time>& xyze_st)
{
  bool successful = true;

  const int nsd = Core::FE::dim<space_time_distype>;

  // use one-point Gauss rule
  Core::FE::IntPointsAndWeights<nsd> intpoints_stab(
      Discret::Elements::DisTypeToStabGaussRule<space_time_distype>::rule);

  Core::LinAlg::Matrix<nsd, 1> xsi(Core::LinAlg::Initialization::zero);

  // coordinates of the current integration point
  const double* gpcoord = (intpoints_stab.ip().qxg)[0];
  for (int idim = 0; idim < nsd; idim++)
  {
    xsi(idim) = gpcoord[idim];
  }


  Core::LinAlg::Matrix<nsd, numnode_space_time> deriv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> xjm(Core::LinAlg::Initialization::zero);

  Core::FE::shape_function_deriv1<space_time_distype>(xsi, deriv);

  xjm.multiply_nt(deriv, xyze_st);

  double det = 0.0;
  det = xjm.determinant();
  if (abs(det) < 1E-14)
  {
    successful = false;
#ifdef DEBUG_TIMINT
    Core::IO::cout
        << "\t\t\t det for space-time side element (det == 0.0) => distorted flat st-element"
        << Core::IO::endl;
#endif
    return successful;
  }

#ifdef DEBUG_TIMINT
  const double wquad = intpoints_stab.IP().qwgt[0];
  std::cout << "\t\t\t space-time side element volume " << wquad * det << std::endl;
#endif

  if (det < 1E-14 and det > -1E-14)
  {
#ifdef DEBUG_TIMINT
    Core::IO::cout << "\t\t\t det for space-time side element (det == " << det
                   << ") => distorted flat st-element" << Core::IO::endl;
#endif
    successful = false;
    return successful;
  }


  return successful;
}


/*------------------------------------------------------------------------------------------------*
 * export data about reconstruction method to neighbor proc and receive data from previous proc
 *schott 04/13 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XFluidTimeInt::export_methods(
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
        newRowStateVectors,  /// row map based vectors w.r.t new interface position
    const std::shared_ptr<std::set<int>>
        dbcgids  /// set of dof gids that must not be changed by ghost penalty reconstruction
)
{
  // send data to the processor where the point lies (1. nearest higher neighbour 2. 2nd nearest
  // higher neighbour...)
  for (int dest = (myrank_ + 1) % numproc_; dest != myrank_;
      dest = (dest + 1) % numproc_)  // dest is the target processor
  {
    // Initialization of sending
    Core::Communication::PackBuffer
        dataSend;  // vector including all data that has to be send to dest proc

    // Initialization
    int source = myrank_ - (dest - myrank_);  // source proc (sends (dest-myrank_) far and gets from
                                              // (dest-myrank_) earlier)
    if (source < 0)
      source += numproc_;
    else if (source >= numproc_)
      source -= numproc_;

    //---------------------------------------------------------------------------------------------------------------
    //--------------------- send data to next proc and receive from previous proc
    //-----------------------------------
    //---------------------------------------------------------------------------------------------------------------
    // send current DofSetData to next proc and receive a new map from previous proc
    {
      Core::Communication::PackBuffer dataSend;  // data to be sent
      for (std::map<int, std::map<int, int>>::iterator node_it = dofset_marker_export_.begin();
          node_it != dofset_marker_export_.end(); node_it++)
      {
        add_to_pack(dataSend, node_it->first);
        add_to_pack(dataSend, node_it->second);
      }

      std::vector<char> dataRecv;
      send_data(dataSend, dest, source, dataRecv);


      // unpack received data
      Core::Communication::UnpackBuffer buffer(dataRecv);
      while (!buffer.at_end())
      {
        // unpack volumecell
        int nid = -1;                   // node id
        std::map<int, int> dofset_map;  // dofset map <nds, Method>

        // unpack reconstruction method data
        extract_from_pack(buffer, nid);
        extract_from_pack(buffer, dofset_map);

        // distribute the received information on this proc if the info is required on this node

        // set reconstruction method on this proc only for received row node information
        int lid = dis_->node_row_map()->lid(nid);


        if (lid == -1) continue;  // this is not a row node on this proc

        // get the node
        Core::Nodes::Node* node = dis_->g_node(nid);

        for (std::map<int, int>::iterator it = dofset_map.begin(); it != dofset_map.end(); it++)
        {
          mark_dofs(
              node, it->first, newRowStateVectors, (Inpar::XFEM::XFluidTimeInt)it->second, dbcgids);
        }
      }

      Core::Communication::barrier(dis_->get_comm());  // processors wait for each other
    }

    //---------------------------------------------------------------------------------------------------------------

  }  // end loop over procs

  return;
}  // end export_methods


/*------------------------------------------------------------------------------------------------*
 * basic function sending data to dest and receiving data from source                schott 03/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XFluidTimeInt::send_data(Core::Communication::PackBuffer& dataSend, int& dest,
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
  Core::Communication::Exporter exporter(dis_->get_comm());

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

FOUR_C_NAMESPACE_CLOSE
