// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_manager.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::evaluate_dirichlet(const Core::FE::Discretization& discret,
    const Teuchos::ParameterList& params,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvector,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvectord,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvectordd,
    const std::shared_ptr<Core::LinAlg::Vector<int>>& toggle,
    const std::shared_ptr<Core::LinAlg::MapExtractor>& dbcmapextractor)
{
  // create const version
  const std::shared_ptr<const Core::FE::Dbc> dbc = build_dbc(&discret);
  (*dbc)(discret, params, systemvector, systemvectord, systemvectordd, toggle, dbcmapextractor);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::FE::Dbc> Core::FE::build_dbc(
    const Core::FE::Discretization* discret_ptr)
{
  // HDG discretization
  if (dynamic_cast<const Core::FE::DiscretizationHDG*>(discret_ptr) != nullptr)
    return std::make_shared<const Core::FE::DbcHDG>();

  // Nurbs discretization
  if (dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(discret_ptr) != nullptr)
    return std::make_shared<const Core::FE::DbcNurbs>();

  // default case
  return std::make_shared<const Core::FE::Dbc>();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Dbc::operator()(const Core::FE::Discretization& discret,
    const Teuchos::ParameterList& params,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvector,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvectord,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvectordd,
    const std::shared_ptr<Core::LinAlg::Vector<int>>& toggle,
    const std::shared_ptr<Core::LinAlg::MapExtractor>& dbcmapextractor) const
{
  if (!discret.filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!discret.have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  // get the current time
  double time = -1.0;
  if (params.isParameter("total time"))
  {
    time = params.get<double>("total time");
  }
  else
    FOUR_C_THROW("The 'total time' needs to be specified in your parameter list!");

  // vector of DOF-IDs which are Dirichlet BCs
  std::array<std::shared_ptr<std::set<int>>, 2> dbcgids = {nullptr, nullptr};
  if (dbcmapextractor != nullptr) dbcgids[set_row] = std::make_shared<std::set<int>>();

  const std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> systemvectors = {
      systemvector, systemvectord, systemvectordd};

  /* If no toggle vector is provided we have to create a temporary one,
   * i.e. we create a temporary toggle if nullptr.
   * We need this to assess the entity hierarchy and to determine which
   * dof has a Dirichlet BC in the end. The highest entity defined for
   * a certain dof in the input file overwrites the corresponding entry
   * in the toggle vector. The entity hierarchy is:
   * point>line>surface>volume */
  std::shared_ptr<Core::LinAlg::Vector<int>> toggleaux =
      create_toggle_vector(toggle, systemvectors.data());

  // --------------------------------------------------------------------------
  // start to evaluate the dirichlet boundary conditions...
  // --------------------------------------------------------------------------
  DbcInfo info(*toggleaux);
  evaluate(params, discret, time, systemvectors.data(), info, dbcgids.data());

  // --------------------------------------------------------------------------
  // create DBC and free map and build their common extractor
  // --------------------------------------------------------------------------
  build_dbc_map_extractor(discret, dbcgids[set_row], dbcmapextractor);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<int>> Core::FE::Dbc::create_toggle_vector(
    const std::shared_ptr<Core::LinAlg::Vector<int>> toggle_input,
    const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors) const
{
  std::shared_ptr<Core::LinAlg::Vector<int>> toggleaux = nullptr;

  if (toggle_input)
    toggleaux = toggle_input;
  else
  {
    if (systemvectors[0])
    {
      toggleaux = std::make_shared<Core::LinAlg::Vector<int>>(systemvectors[0]->get_map());
    }
    else if (systemvectors[1])
    {
      toggleaux = std::make_shared<Core::LinAlg::Vector<int>>(systemvectors[1]->get_map());
    }
    else if (systemvectors[2])
    {
      toggleaux = std::make_shared<Core::LinAlg::Vector<int>>(systemvectors[2]->get_map());
    }
    else if (systemvectors[0] == nullptr and systemvectors[1] == nullptr and
             systemvectors[2] == nullptr)
    {
      FOUR_C_THROW(
          "At least one systemvector must be provided. Otherwise, calling "
          "this method makes no sense.");
    }
  }

  return toggleaux;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Dbc::evaluate(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, double time,
    const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors, DbcInfo& info,
    std::shared_ptr<std::set<int>>* dbcgids) const
{
  // --------------------------------------------------------------------------
  // loop through Dirichlet conditions and evaluate them
  // --------------------------------------------------------------------------
  std::vector<const Conditions::Condition*> conds;
  discret.get_condition("Dirichlet", conds);
  read_dirichlet_condition(params, discret, conds, time, info, dbcgids);
  // --------------------------------------------------------------------------
  // Now, as we know from the toggle vector which dofs actually have
  // Dirichlet BCs, we can assign the values to the system vectors.
  // --------------------------------------------------------------------------
  do_dirichlet_condition(params, discret, conds, time, systemvectors, info.toggle, dbcgids);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Dbc::read_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, std::span<const Conditions::Condition*> conds,
    double time, DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids) const
{
  // read the DBC in descending order of the geometrical hierarchy.
  // Since lower geometry DBC can override the higher one, the order is important for inconsistency
  // check. This logic can be understood by contraposition: if, for example, Point DBC is read
  // first, the dof values will not be altered by Line/Surface/Volume DBC. Hence inconsistency in
  // Line/Surface/Volume DBC cannot be detected.
  read_dirichlet_condition(
      params, discret, conds, time, info, dbcgids, Core::Conditions::VolumeDirichlet);
  read_dirichlet_condition(
      params, discret, conds, time, info, dbcgids, Core::Conditions::SurfaceDirichlet);
  read_dirichlet_condition(
      params, discret, conds, time, info, dbcgids, Core::Conditions::LineDirichlet);
  read_dirichlet_condition(
      params, discret, conds, time, info, dbcgids, Core::Conditions::PointDirichlet);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Dbc::read_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, std::span<const Conditions::Condition*> conds,
    double time, DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids,
    const enum Core::Conditions::ConditionType& type) const
{
  int hierarchical_order;
  switch (type)
  {
    case Core::Conditions::PointDirichlet:
      hierarchical_order = 0;
      break;
    case Core::Conditions::LineDirichlet:
      hierarchical_order = 1;
      break;
    case Core::Conditions::SurfaceDirichlet:
      hierarchical_order = 2;
      break;
    case Core::Conditions::VolumeDirichlet:
      hierarchical_order = 3;
      break;
    default:
      FOUR_C_THROW("Unknown condition type");
      break;
  }

  // Gather dbcgids of given type
  for (const auto& cond : conds)
  {
    // skip conditions of different type
    if (cond->type() != type) continue;

    read_dirichlet_condition(params, discret, *cond, time, info, dbcgids, hierarchical_order);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Dbc::read_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond, double time,
    DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids, int hierarchical_order) const
{
  // get ids of conditioned nodes
  const std::vector<int>* nodeids = cond.get_nodes();
  if (!nodeids) FOUR_C_THROW("Dirichlet condition does not have nodal cloud");
  // determine number of conditioned nodes
  const unsigned nnode = (*nodeids).size();
  // get onoff toggles from condition
  const auto onoff = cond.parameters().get<std::vector<int>>("ONOFF");
  // get val from condition
  const auto val = cond.parameters().get<std::vector<double>>("VAL");
  // get funct from condition
  const auto funct = cond.parameters().get<std::vector<std::optional<int>>>("FUNCT");

  // loop nodes to identify spatial distributions of Dirichlet boundary conditions
  for (unsigned i = 0; i < nnode; ++i)
  {
    // do only nodes in my row map
    Core::Nodes::Node* actnode = nullptr;
    bool isrow = true;
    int nlid = discret.node_row_map()->lid((*nodeids)[i]);
    if (nlid < 0)
    {
      // just skip this node, if we are not interested in column information
      if (dbcgids[set_col] == nullptr) continue;
      // ----------------------------------------------------------------------
      // get a column node, if desired
      // NOTE: the following is not supported for discretization wrappers
      // ----------------------------------------------------------------------
      const auto* dis_ptr = dynamic_cast<const Core::FE::Discretization*>(&discret);
      if (not dis_ptr)
        FOUR_C_THROW(
            "Sorry! The given discretization is of wrong type. There is "
            "probably no column information available!");

      nlid = dis_ptr->node_col_map()->lid((*nodeids)[i]);

      // node not on this processor -> next node
      if (nlid < 0) continue;

      actnode = dis_ptr->l_col_node(nlid);
      isrow = false;
    }
    else
      actnode = discret.l_row_node(nlid);

    // call explicitly the main dofset, i.e. the first column
    std::vector<int> dofs = discret.dof(0, actnode);
    const unsigned total_numdf = dofs.size();

    // only continue if there are node dofs
    if (total_numdf == 0) continue;

    // Get number of non-enriched dofs at this node. There might be several
    // nodal dof-sets (in xfem cases), thus the size of the dofs vector might
    // be a multiple of this value. Otherwise you get the same number of dofs
    // as total_numdf
    int numdf = discret.num_standard_dof(0, actnode);

    if ((total_numdf % numdf) != 0)
      FOUR_C_THROW(
          "Illegal number of DoF's at this node! (nGID={})\n"
          "{} is not a multiple of {}",
          actnode->id(), total_numdf, numdf);

    // is the number of degrees of freedom given in the constraint definition sufficient?
    const int num_dbc_dofs = static_cast<int>(onoff.size());
    if (num_dbc_dofs < numdf)
      FOUR_C_THROW("{} DOFs given but {} expected in {}", num_dbc_dofs, numdf,
          Core::Conditions::to_string(cond.type()).data());

    // loop over dofs of current nnode
    for (unsigned j = 0; j < total_numdf; ++j)
    {
      // get dof gid
      const int gid = dofs[j];

      // get corresponding lid
      const int lid = info.toggle.get_map().lid(gid);
      if (lid < 0)
        FOUR_C_THROW("Global id {} not on this proc {} in system vector", dofs[j],
            Core::Communication::my_mpi_rank(discret.get_comm()));

      // get position of label for this dof in condition line ( e.g. for XFEM )
      int onesetj = j % numdf;

      // get the current hierarchical order this dof is currently applying to
      const int current_order = info.hierarchy.get_local_values()[lid];

      if (onoff[onesetj] == 0)
      {
        // the dof at geometry of lower hierarchical order can reset the toggle value
        // Note: this check is crucial to avoid DBC at the same geometrical level to not override
        // each other, so the 3D patch test can pass without additional DBC on line
        if (hierarchical_order < current_order)
        {
          // no DBC on this dof, set toggle zero
          info.toggle.get_local_values()[lid] = 0;

          // get rid of entry in row DBC map - if it exists
          if (isrow and (dbcgids[set_row])) (*dbcgids[set_row]).erase(gid);

          // get rid of entry in column DBC map - if it exists
          if (dbcgids[set_col]) (*dbcgids[set_col]).erase(gid);

          // record the current hierarchical order of the DBC dof
          info.hierarchy.get_local_values()[lid] = hierarchical_order;
        }
      }
      else  // if (onoff[onesetj]==1)
      {
        // evaluate the DBC prescribed value based on time curve
        // here we only compute based on time curve and not the derivative, hence degree = 0
        double functfac = 1.0;

        if (funct[onesetj].has_value() && funct[onesetj].value() > 0)
        {
          functfac = params.get<const Core::Utils::FunctionManager*>("function_manager")
                         ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[onesetj].value())
                         .evaluate(actnode->x(), time, onesetj);
        }

        const double value = val[onesetj] * functfac;

        // check: if the dof has been fixed before and the DBC set it to a different value, then an
        // inconsistency is detected.
        if ((hierarchical_order == current_order) && (info.toggle.get_local_values()[lid] == 1))
        {
          // get the current prescribed value of dof
          const double current_val = info.values.local_values_as_span()[lid];

          // get the current condition that prescribed value of dof
          const int current_cond = info.condition.get_local_values()[lid];

          // if the current condition set the dof value to other value, then we found an
          // inconsistency. The basis for this is: Overwriting should be allowed over hierarchies
          // (line overwrites surface, and so on) which is one of the main features of our DBC
          // application work flow. And in such a case it is fully okay if the line prescribes also
          // an inconsistent value (regarding the surface value). Of course, an error/warning is
          // given if different values are prescribed on the same hierarchy level. Here,
          // inconsistency matters.
          const double dbc_tol = 1.0e-13;
          if (std::abs(current_val - value) > dbc_tol)
          {
            std::string geom_name;
            if (hierarchical_order == 0)
              geom_name = "POINT";
            else if (hierarchical_order == 1)
              geom_name = "LINE";
            else if (hierarchical_order == 2)
              geom_name = "SURF";
            else if (hierarchical_order == 3)
              geom_name = "VOL";
            std::stringstream ss;
            ss << "Error!!! Inconsistency is detected at " << geom_name << " DBC " << cond.id() + 1
               << " (node " << actnode->id() + 1 << ", dof " << j
               << ").\nIt tried to override the previous fixed value of " << current_val
               << " prescribed by " << geom_name << " DBC " << current_cond + 1
               << " with new value of " << value << " at time " << time << ".\nThe difference is "
               << std::setprecision(13) << std::abs(current_val - value) << " > " << dbc_tol
               << ".\nPlease try to adjust the input.";
            FOUR_C_THROW("{}", ss.str());
          }
        }

        if (hierarchical_order > current_order)
          FOUR_C_THROW(
              "This couldn't happen, except if you try to read DBC not in descending order.");

        // dof has DBC, set toggle vector one
        info.toggle.get_local_values()[lid] = 1;

        // amend set of row DOF-IDs which are dirichlet BCs
        if (isrow and (dbcgids[set_row])) (*dbcgids[set_row]).insert(gid);

        // amend set of column DOF-IDs which are dirichlet BCs
        if (dbcgids[set_col])
        {
          (*dbcgids[set_col]).insert(gid);
        }

        // record the lowest hierarchical order of the DBC dof
        if (hierarchical_order < current_order)
          info.hierarchy.get_local_values()[lid] = hierarchical_order;

        // record the prescribed value of dof if it is fixed
        info.values.get_values()[lid] = value;

        // record the condition that assign the value
        info.condition.get_local_values()[lid] = cond.id();
      }
    }  // loop over nodal DOFs
  }  // loop over nodes
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Dbc::do_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, std::span<const Conditions::Condition*> conds,
    double time, const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
    const Core::LinAlg::Vector<int>& toggle, const std::shared_ptr<std::set<int>>* dbcgids) const
{
  do_dirichlet_condition(params, discret, conds, time, systemvectors, toggle, dbcgids,
      Core::Conditions::VolumeDirichlet);
  do_dirichlet_condition(params, discret, conds, time, systemvectors, toggle, dbcgids,
      Core::Conditions::SurfaceDirichlet);
  do_dirichlet_condition(params, discret, conds, time, systemvectors, toggle, dbcgids,
      Core::Conditions::LineDirichlet);
  do_dirichlet_condition(params, discret, conds, time, systemvectors, toggle, dbcgids,
      Core::Conditions::PointDirichlet);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Dbc::do_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, std::span<const Core::Conditions::Condition*> conds,
    double time, const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
    const Core::LinAlg::Vector<int>& toggle, const std::shared_ptr<std::set<int>>* dbcgids,
    const enum Core::Conditions::ConditionType& type) const
{
  for (const auto& cond : conds)
  {
    // skip conditions of different type
    if (cond->type() != type) continue;

    do_dirichlet_condition(params, discret, *cond, time, systemvectors, toggle, dbcgids);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Dbc::do_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond, double time,
    const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
    const Core::LinAlg::Vector<int>& toggle, const std::shared_ptr<std::set<int>>* dbcgids) const
{
  if (systemvectors[0] == nullptr and systemvectors[1] == nullptr and systemvectors[2] == nullptr)
    FOUR_C_THROW(
        "At least one systemvector must be provided. Otherwise, "
        "calling this method makes no sense.");

  // get ids of conditioned nodes
  const std::vector<int>* nodeids = cond.get_nodes();
  if (!nodeids) FOUR_C_THROW("Dirichlet condition does not have nodal cloud");
  // determine number of conditioned nodes
  const unsigned nnode = (*nodeids).size();
  // get onoff, funct, and val from condition
  const auto onoff = cond.parameters().get<std::vector<int>>("ONOFF");
  const auto funct = cond.parameters().get<std::vector<std::optional<int>>>("FUNCT");
  const auto val = cond.parameters().get<std::vector<double>>("VAL");

  // determine highest degree of time derivative
  // and first existent system vector to apply DBC to
  unsigned deg = 0;  // highest degree of requested time derivative
  if (systemvectors[0] != nullptr) deg = 0;

  if (systemvectors[1] != nullptr) deg = 1;

  if (systemvectors[2] != nullptr) deg = 2;

  // loop nodes to identify and evaluate load curves and spatial distributions
  // of Dirichlet boundary conditions
  for (unsigned i = 0; i < nnode; ++i)
  {
    // do only nodes in my row map
    const int nlid = discret.node_row_map()->lid((*nodeids)[i]);
    if (nlid < 0) continue;
    Core::Nodes::Node* actnode = discret.l_row_node(nlid);

    // call explicitly the main dofset, i.e. the first column
    std::vector<int> dofs = discret.dof(0, actnode);
    const unsigned total_numdf = dofs.size();

    // only continue if there are node dofs
    if (total_numdf == 0) continue;

    // Get number of non-enriched dofs at this node. There might be several
    // nodal dof-sets (in xfem cases), thus the size of the dofs vector might
    // be a multiple of this value. Otherwise you get the same number of dofs
    // as total_numdf
    const int numdf = discret.num_standard_dof(0, actnode);

    if ((total_numdf % numdf) != 0)
      FOUR_C_THROW(
          "Illegal number of DoF's at this node! (nGID={})\n"
          "{} is not a multiple of {}",
          actnode->id(), total_numdf, numdf);

    // loop over dofs of current nnode
    for (unsigned j = 0; j < total_numdf; ++j)
    {
      // get dof gid
      const int gid = dofs[j];
      // get corresponding lid
      const int lid = toggle.get_map().lid(gid);
      if (lid < 0)
        FOUR_C_THROW("Global id {} not on this proc {} in system vector", dofs[j],
            Core::Communication::my_mpi_rank(discret.get_comm()));
      // get position of label for this dof in condition line
      const int onesetj = j % numdf;

      // check whether dof gid is a dbc gid and is prescribed only by the current condition
      const bool dbc_on_dof_is_off = (onoff[onesetj] == 0);  // dof is not DBC by current condition
      const bool dbc_toggle_is_off =
          (toggle.get_local_values()[lid] == 0);  // dof is not prescribed by current condition or
                                                  // is unprescribed by lower hierarchy condition
      if (dbc_on_dof_is_off || dbc_toggle_is_off) continue;

      std::vector<double> value(deg + 1, val[onesetj]);

      // factor given by temporal and spatial function
      std::vector<double> functimederivfac(deg + 1, 0.0);
      functimederivfac[0] = 1.0;

      if (funct[onesetj].has_value() && funct[onesetj].value() > 0)
      {
        functimederivfac =
            params.get<const Core::Utils::FunctionManager*>("function_manager")
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[onesetj].value())
                .evaluate_time_derivative(actnode->x(), time, deg, onesetj);
      }

      // apply factors to Dirichlet value
      for (unsigned i = 0; i < deg + 1; ++i)
      {
        value[i] *= functimederivfac[i];
      }

      // assign value
      if (systemvectors[0] != nullptr) (*systemvectors[0]).get_values()[lid] = value[0];
      if (systemvectors[1] != nullptr) (*systemvectors[1]).get_values()[lid] = value[1];
      if (systemvectors[2] != nullptr) (*systemvectors[2]).get_values()[lid] = value[2];

    }  // loop over nodal DOFs
  }  // loop over nodes
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Dbc::build_dbc_map_extractor(const Core::FE::Discretization& discret,
    const std::shared_ptr<const std::set<int>>& dbcrowgids,
    const std::shared_ptr<Core::LinAlg::MapExtractor>& dbcmapextractor) const
{
  if (!dbcmapextractor) return;

  FOUR_C_ASSERT(dbcrowgids,
      "The variable `dbcrowgids` in `Core::FE::Dbc::build_dbc_map_extractor` is a null "
      "pointer. This violates the implicit assumption that it must be non-null when "
      "`dbcmapextractor` is non-null.");

  // build map of Dirichlet DOFs
  int nummyelements = 0;
  int* myglobalelements = nullptr;
  std::vector<int> dbcgidsv;
  if (dbcrowgids->size() > 0)
  {
    dbcgidsv.reserve(dbcrowgids->size());
    dbcgidsv.assign(dbcrowgids->begin(), dbcrowgids->end());
    nummyelements = dbcgidsv.size();
    myglobalelements = dbcgidsv.data();
  }
  std::shared_ptr<Core::LinAlg::Map> dbcmap = std::make_shared<Core::LinAlg::Map>(-1, nummyelements,
      myglobalelements, discret.dof_row_map()->index_base(), discret.dof_row_map()->get_comm());
  // build the map extractor of Dirichlet-conditioned and free DOFs
  dbcmapextractor->setup(*(discret.dof_row_map()), dbcmap);
}

FOUR_C_NAMESPACE_CLOSE
