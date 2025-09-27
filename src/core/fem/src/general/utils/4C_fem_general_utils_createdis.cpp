// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_createdis.hpp"

#include "4C_fem_dofset_transparent_independent.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_rebalance_binning_based.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::FE::DiscretizationCreatorBase::initial_checks(
    const Core::FE::Discretization& sourcedis, const Core::FE::Discretization& targetdis) const
{
  // are the source and target discretizations ready?
  if (!sourcedis.filled())
    FOUR_C_THROW("The source discretization '{}' is not filled!", sourcedis.name());
  if (!targetdis.filled())
    FOUR_C_THROW("The target discretization '{}' is not filled!", targetdis.name());

  // is the target discretization really empty?
  if (targetdis.num_global_elements() or targetdis.num_global_nodes())
  {
    FOUR_C_THROW("There are {} elements and {} nodes in target discretization. Panic.",
        targetdis.num_global_elements(), targetdis.num_global_nodes());
  }
  // Ok. Let's go on
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::FE::DiscretizationCreatorBase::create_nodes(const Core::FE::Discretization& sourcedis,
    Core::FE::Discretization& targetdis, const std::set<int>& rownodeset,
    const std::set<int>& colnodeset, const bool isnurbsdis) const
{
  // prepare some variables we need
  int myrank = Core::Communication::my_mpi_rank(targetdis.get_comm());
  const Core::LinAlg::Map* sourcenoderowmap = sourcedis.node_row_map();

  // construct nodes / control points in the new discretization
  if (isnurbsdis == false)
  {
    for (int i = 0; i < sourcenoderowmap->num_my_elements(); ++i)
    {
      int gid = sourcenoderowmap->gid(i);
      if (rownodeset.find(gid) != rownodeset.end())
      {
        Core::Nodes::Node* node_to_create = sourcedis.l_row_node(i);
        targetdis.add_node(node_to_create->x(), gid, nullptr);
      }
    }
  }
  else
  {
    for (int i = 0; i < sourcenoderowmap->num_my_elements(); ++i)
    {
      const int gid = sourcenoderowmap->gid(i);
      if (rownodeset.find(gid) != rownodeset.end())
      {
        Core::FE::Nurbs::ControlPoint* node_to_create =
            dynamic_cast<Core::FE::Nurbs::ControlPoint*>(sourcedis.l_row_node(i));
        targetdis.add_node(node_to_create->x(), gid,
            std::make_shared<Core::FE::Nurbs::ControlPoint>(
                gid, node_to_create->x(), node_to_create->w(), myrank));
      }
    }
  }

  // ensure reset() is called on targetdis on all procs (including procs without rownodes)
  targetdis.check_filled_globally();

  return;
}  // Core::FE::DiscretizationCreatorBase::CreateNodes

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::FE::DiscretizationCreatorBase::create_map(
    std::set<int>& gidset, const Core::FE::Discretization& targetdis) const
{
  // we get the node maps almost for free
  std::vector<int> targetgidvec(gidset.begin(), gidset.end());
  gidset.clear();

  std::shared_ptr<Core::LinAlg::Map> map = std::make_shared<Core::LinAlg::Map>(
      -1, targetgidvec.size(), targetgidvec.data(), 0, targetdis.get_comm());
  targetgidvec.clear();

  return map;
}  // Core::FE::DiscretizationCreatorBase::CreateMap

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::FE::DiscretizationCreatorBase::copy_conditions(const Core::FE::Discretization& sourcedis,
    Core::FE::Discretization& targetdis,
    const std::map<std::string, std::string>& conditions_to_copy) const
{
  // copy selected conditions to the new discretization (and rename them if desired)
  for (const auto& condition_pair : conditions_to_copy)
  {
    std::vector<const Core::Conditions::Condition*> conds;
    sourcedis.get_condition(condition_pair.first, conds);
    for (const auto& cond : conds)
    {
      // We use the same nodal ids and therefore we can just copy the conditions.
      // The string-map gives the new condition names
      // (e.g. renaming from TransportDirichlet to Dirichlet)
      targetdis.set_condition(condition_pair.second, cond->copy_without_geometry());
    }
    conds.clear();
  }
}  // Core::FE::DiscretizationCreatorBase::CopyConditions

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Discretization>
Core::FE::DiscretizationCreatorBase::create_matching_discretization(
    Core::FE::Discretization& sourcedis, const std::string& targetdisname, bool clonedofs,
    bool assigndegreesoffreedom, bool initelements, bool doboundaryconditions) const
{
  // initialize identical clone discretization
  std::shared_ptr<Core::FE::Discretization> targetdis = std::make_shared<Core::FE::Discretization>(
      targetdisname, sourcedis.get_comm(), sourcedis.n_dim());

  // clone nodes
  for (int i = 0; i < sourcedis.node_col_map()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = sourcedis.l_col_node(i);
    if (!node) FOUR_C_THROW("Cannot find node with local id {}", i);
    std::shared_ptr<Core::Nodes::Node> newnode(node->clone());
    targetdis->add_node(newnode->x(), newnode->id(), newnode);
  }

  // clone elements
  for (int i = 0; i < sourcedis.element_col_map()->num_my_elements(); ++i)
  {
    Core::Elements::Element* ele = sourcedis.l_col_element(i);
    if (!ele) FOUR_C_THROW("Cannot find element with local id {}", i);
    std::shared_ptr<Core::Elements::Element> newele(ele->clone());
    targetdis->add_element(newele);
  }

  // clone conditions (prescribed in input file)
  std::vector<std::string> allcond;
  sourcedis.get_condition_names(allcond);
  // loop all conditions types
  for (unsigned numcond = 0; numcond < allcond.size(); ++numcond)
  {
    // get condition
    std::vector<const Core::Conditions::Condition*> actcond;
    sourcedis.get_condition(allcond[numcond], actcond);

    // loop all condition of the current type
    for (unsigned numactcond = 0; numactcond < actcond.size(); ++numactcond)
      targetdis->set_condition(allcond[numcond], actcond[numactcond]->copy_without_geometry());
  }

  // make auxiliary discretization have the same dofs as the coupling discretization
  if (clonedofs)
    targetdis->replace_dof_set(std::make_shared<Core::DOFSets::IndependentDofSet>(), false);
  targetdis->fill_complete({
      .assign_degrees_of_freedom = assigndegreesoffreedom,
      .init_elements = initelements,
      .do_boundary_conditions = doboundaryconditions,
  });

  // at the end, we do several checks to ensure that we really have generated
  // an identical discretization
  if (not sourcedis.node_row_map()->same_as(*(targetdis->node_row_map())))
    FOUR_C_THROW("NodeRowMaps of source and target discretization are different!");
  if (not sourcedis.node_col_map()->same_as(*(targetdis->node_col_map())))
    FOUR_C_THROW("NodeColMaps of source and target discretization are different!");
  if (not sourcedis.element_row_map()->same_as(*(targetdis->element_row_map())))
    FOUR_C_THROW("ElementRowMaps of source and target discretization are different!");
  if (not sourcedis.element_col_map()->same_as(*(targetdis->element_col_map())))
    FOUR_C_THROW("ElementColMaps of source and target discretization are different!");
  if (clonedofs)
  {
    if (not sourcedis.dof_row_map()->same_as(*(targetdis->dof_row_map())))
      FOUR_C_THROW("DofRowMaps of source and target discretization are different!");
    if (not sourcedis.dof_col_map()->same_as(*(targetdis->dof_col_map())))
      FOUR_C_THROW("DofColMaps of source and target discretization are different!");
  }

  // return identical dis
  return targetdis;

}  // Core::FE::DiscretizationCreatorBase::create_matching_discretization

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::FE::DiscretizationCreatorBase::finalize(
    const Core::FE::Discretization& sourcedis, Core::FE::Discretization& targetdis) const
{
  // export according to previously filled maps
  targetdis.redistribute({*targetnoderowmap_, *targetnodecolmap_},
      {*targetelerowmap_, *targetelecolmap_}, {.fill_complete = OptionsFillComplete::none()});

  // extra work for NURBS discretizations

  // try to cast sourcedis to NurbsDiscretisation
  const Core::FE::Nurbs::NurbsDiscretization* nurbsdis_ptr =
      dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&sourcedis);

  if (nurbsdis_ptr != nullptr)
  {
    Core::FE::Nurbs::NurbsDiscretization* targetnurbsdis_ptr =
        dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&targetdis);

    if (targetnurbsdis_ptr == nullptr)
    {
      FOUR_C_THROW("Nurbs source discretization but no nurbs target discretization\n");
    }

    std::shared_ptr<Core::FE::Nurbs::Knotvector> knots =
        std::make_shared<Core::FE::Nurbs::Knotvector>(*(nurbsdis_ptr->get_knot_vector()));

    // reset offsets
    int smallest_gid_in_dis = targetnurbsdis_ptr->element_row_map()->min_all_gid();
    knots->finish_knots(smallest_gid_in_dis);

    targetnurbsdis_ptr->set_knot_vector(knots);
    targetnurbsdis_ptr->fill_complete();
  }

  // at the end, we do several checks to ensure that we really have identical
  // distributions of elements and nodes over processors (as expected!)
  // We do not perform this check if the new discretization is only a subset of the
  // source discretization.
  int sumeleskips = 0;
  int lnumeleskips = numeleskips_;
  sumeleskips = Core::Communication::sum_all(lnumeleskips, sourcedis.get_comm());

  if (sumeleskips == 0)
  {
    if (not sourcedis.node_row_map()->same_as(*(targetdis.node_row_map())))
      FOUR_C_THROW("NodeRowMaps of source and target discretization are different!");
    if (not sourcedis.node_col_map()->same_as(*(targetdis.node_col_map())))
      FOUR_C_THROW("NodeColMaps of source and target discretization are different!");
    if (not sourcedis.element_row_map()->same_as(*(targetdis.element_row_map())))
      FOUR_C_THROW("ElementRowMaps of source and target discretization are different!");
    if (not sourcedis.element_col_map()->same_as(*(targetdis.element_col_map())))
      FOUR_C_THROW("ElementColMaps of source and target discretization are different!");
  }
}  // Core::FE::DiscretizationCreatorBase::Finalize

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::InputSpec Core::FE::valid_cloning_material_map()
{
  using namespace Core::IO::InputSpecBuilders;
  return all_of({
      parameter<std::string>("SRC_FIELD"),
      parameter<int>("SRC_MAT"),
      parameter<std::string>("TAR_FIELD"),
      parameter<int>("TAR_MAT"),

  });
}

FOUR_C_NAMESPACE_CLOSE
