// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization_utils.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_map.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_io_input_field.hpp"

FOUR_C_NAMESPACE_OPEN

std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::FE::extract_node_coordinates(
    const Discretization& discretization)
{
  return extract_node_coordinates(discretization, *discretization.node_row_map());
}



std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::FE::extract_node_coordinates(
    const Discretization& discretization, const Core::LinAlg::Map& node_row_map)
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> coordinates =
      std::make_shared<Core::LinAlg::MultiVector<double>>(node_row_map, 3, true);

  for (int lid = 0; lid < node_row_map.num_my_elements(); ++lid)
  {
    const int gid = node_row_map.gid(lid);
    if (!discretization.have_global_node(gid)) continue;

    auto x = discretization.g_node(gid)->x();
    for (size_t dim = 0; dim < 3; ++dim)
    {
      if (dim >= discretization.n_dim())
        coordinates->replace_local_value(lid, dim, 0.0);
      else
        coordinates->replace_local_value(lid, dim, x[dim]);
    }
  }

  return coordinates;
}

std::unique_ptr<Core::LinAlg::MultiVector<double>> Core::FE::extract_retained_node_coordinates(
    const Discretization& discretization, const Core::LinAlg::Map& node_row_map)
{
  std::set<int> skipped_nodes = {};
  if (discretization.get_pbc_source_to_target_node_connectivity())
  {
    const auto& pbcconnectivity = *(discretization.get_pbc_source_to_target_node_connectivity());
    for (const auto& [source, _] : pbcconnectivity)
    {
      skipped_nodes.insert(source);
    }
  }
  std::set<int> fully_redundant_skipped_nodes =
      Core::Communication::all_reduce(skipped_nodes, node_row_map.get_comm());

  std::vector<int> my_filtered_global_node_ids{};
  my_filtered_global_node_ids.reserve(node_row_map.num_my_elements());
  for (int lid = 0; lid < node_row_map.num_my_elements(); ++lid)
  {
    const int gid = node_row_map.gid(lid);

    // filter pbc source nodes
    if (fully_redundant_skipped_nodes.contains(gid)) continue;
    my_filtered_global_node_ids.emplace_back(gid);
  }
  Core::LinAlg::Map filtered_row_map(
      node_row_map.num_global_elements() - static_cast<int>(fully_redundant_skipped_nodes.size()),
      static_cast<int>(my_filtered_global_node_ids.size()), my_filtered_global_node_ids.data(), 0,
      node_row_map.get_comm());

  std::unique_ptr<Core::LinAlg::MultiVector<double>> coordinates =
      std::make_unique<Core::LinAlg::MultiVector<double>>(filtered_row_map, 3, true);

  for (int lid = 0; lid < filtered_row_map.num_my_elements(); ++lid)
  {
    const int gid = filtered_row_map.gid(lid);
    if (!discretization.have_global_node(gid)) continue;

    auto x = discretization.g_node(gid)->x();
    for (size_t dim = 0; dim < 3; ++dim)
    {
      if (dim >= discretization.n_dim())
        coordinates->replace_local_value(lid, dim, 0.0);
      else
        coordinates->replace_local_value(lid, dim, x[dim]);
    }
  }

  return coordinates;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::evaluate_initial_field(const Core::Utils::FunctionManager& function_manager,
    const Core::FE::Discretization& discret, const std::string& fieldstring,
    Core::LinAlg::Vector<double>& fieldvector, const std::vector<int>& locids)
{
  // get initial field conditions
  std::vector<const Core::Conditions::Condition*> initfieldconditions;
  discret.get_condition("Initfield", initfieldconditions);

  //--------------------------------------------------------
  // loop through Initfield conditions and evaluate them
  //--------------------------------------------------------
  // Note that this method does not sum up but 'sets' values in fieldvector.
  // For this reason, Initfield BCs are evaluated hierarchical meaning
  // in this order (just like Dirichlet BCs):
  //                VolumeInitfield
  //                SurfaceInitfield
  //                LineInitfield
  //                PointInitfield
  // This way, lower entities override higher ones.
  const std::vector<Core::Conditions::ConditionType> evaluation_type_order = {
      Core::Conditions::VolumeInitfield, Core::Conditions::SurfaceInitfield,
      Core::Conditions::LineInitfield, Core::Conditions::PointInitfield};

  for (const auto& type : evaluation_type_order)
  {
    for (const auto& initfieldcondition : initfieldconditions)
    {
      if (initfieldcondition->type() != type) continue;
      const std::string condstring = initfieldcondition->parameters().get<std::string>("FIELD");
      if (condstring != fieldstring) continue;
      do_initial_field(function_manager, discret, *initfieldcondition, fieldvector, locids);
    }
  }
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::do_initial_field(const Core::Utils::FunctionManager& function_manager,
    const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond,
    Core::LinAlg::Vector<double>& fieldvector, const std::vector<int>& locids)
{
  const std::vector<int> cond_nodeids = *cond.get_nodes();
  if (cond_nodeids.empty()) FOUR_C_THROW("Initfield condition does not have nodal cloud.");

  // loop nodes to identify and evaluate spatial distributions
  // of Initfield boundary conditions
  const auto& params = cond.parameters();

  for (const int cond_nodeid : cond_nodeids)
  {
    // do only nodes in my row map
    int cond_node_lid = discret.node_row_map()->lid(cond_nodeid);
    if (cond_node_lid < 0) continue;
    Core::Nodes::Node* node = discret.l_row_node(cond_node_lid);

    // call explicitly the main dofset, i.e. the first column
    std::vector<int> node_dofs = discret.dof(0, node);
    const int total_numdof = static_cast<int>(node_dofs.size());

    // Get native number of dofs at this node. There might be multiple dofsets
    // (in xfem cases), thus the size of the dofs vector might be a multiple
    // of this.
    auto elements = node->adjacent_elements();
    auto ele_with_max_dof = std::ranges::max_element(elements,
        [&](ElementRef a, ElementRef b)
        {
          return a.user_element()->num_dof_per_node(*node) <
                 b.user_element()->num_dof_per_node(*node);
        });
    const int numdof = ele_with_max_dof->user_element()->num_dof_per_node(*node);

    if ((total_numdof % numdof) != 0) FOUR_C_THROW("illegal dof set number");

    // now loop over all relevant DOFs
    for (int j = 0; j < total_numdof; ++j)
    {
      int localdof = j % numdof;

      // evaluate function if local DOF id exists
      // in the given locids vector
      for (const int locid : locids)
      {
        if (localdof == locid)
        {
          const double time = 0.0;  // dummy time here
          double val = 0.0; // default value
          bool val_assigned = false;

          // Case 1: We defined a function's ID
          if (auto funct_num = params.get_if<int>("FUNCT"); funct_num && *funct_num > 0)
          {
            val = function_manager.function_by_id<Core::Utils::FunctionOfSpaceTime>(*funct_num)
                      .evaluate(node->x(), time, localdof);
            val_assigned = true;
          }
          // Case 2a: Point-based SCALAR input field (basis: points) - lookup by node ID
          else if (auto point_scalar_field = params.get_if<Core::IO::InputField<double>>("POINT_SCALAR_INPUT_FIELD"))
          {
            try {
              val = point_scalar_field->at(node->id());
              val_assigned = true;
            }
            catch (...) {
              // node not in field (outside defined zone)
            }
          }

          // Case 2b: Cell-based SCALAR input field (basis: cells) - lookup by adjacent element IDs
          else if (auto cell_scalar_field = params.get_if<Core::IO::InputField<double>>("CELL_SCALAR_INPUT_FIELD"))
          {
            double sum = 0.0;
            int count = 0;
            for (auto elem : elements)
            {
              try
              {
                sum += cell_scalar_field->at(elem.user_element()->id());
                count++;
              }
              catch (...)
              {
                // element not in field (outside defined zone)
              }
            }

            if (count > 0)
            {
              val = sum / count; // We take the mean value of adjacent elements
              val_assigned = true;
            }
          }

          // Case 3a. Point-based VECTOR input field (basis: points) - lookup by node ID
          else if (auto point_vector_field = params.get_if<Core::IO::InputField<std::vector<double>>>("POINT_VECTOR_INPUT_FIELD"))
          {
            std::vector<double> vec_vals;
            try {
              vec_vals = point_vector_field->at(node->id());
            }
            catch (...) {
              // node not in field (outside defined zone)
            }

            if (localdof < static_cast<int>(vec_vals.size()))
            {
              val = vec_vals[localdof];
              val_assigned = true;
            }
          }

          // Case 3b. Cell-based VECTOR input field (basis: cells) - lookup by adjacent element IDs
          else if (auto cell_vector_field = params.get_if<Core::IO::InputField<std::vector<double>>>("CELL_VECTOR_INPUT_FIELD"))
          {
            std::vector<double> sum_vec;
            int count = 0;
            for (auto elem : elements)
            {
              try
              {
                std::vector<double> current_vec = cell_vector_field->at(elem.user_element()->id());
                if (sum_vec.empty()) sum_vec.resize(current_vec.size(), 0.0);
                for (size_t k = 0; k < current_vec.size(); ++k)
                  sum_vec[k] += current_vec[k];
                count++;
              }
              catch (...)
              {
                // element not in field (outside defined zone)
              }
            }

            if (count > 0)
            {
              for (double& v : sum_vec) v /= count;
              if (localdof < static_cast<int>(sum_vec.size()))
              {
                val = sum_vec[localdof];
                val_assigned = true;
              }
            }
          }

          // assign value
          if(val_assigned)
          {
          const int gid = node_dofs[j];
          const int lid = fieldvector.get_map().lid(gid);
          if (lid < 0) FOUR_C_THROW("Global id {} not on this proc in system vector", gid);
          fieldvector.get_values()[lid] = val;
          }
        }
      }
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
