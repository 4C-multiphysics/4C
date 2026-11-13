// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_restart.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::Restart
{
  namespace
  {
    //! Name of the per-element dof values in the restart file.
    const std::string element_dofs_name = "element_dofs";

    /**
     * Call @p visit(name, values) for every history vector of the airway block @p model. Writing
     * and reading share this list, so they cannot get out of sync.
     */
    template <typename Visitor>
    void visit_history(Airways::AirwayModel& model, Visitor&& visit)
    {
      visit("airway_p1_n", model.data.p1_n);
      visit("airway_p2_n", model.data.p2_n);
      visit("airway_q1_n", model.data.q1_n);
      visit("airway_q2_n", model.data.q2_n);
      std::visit(
          [&](auto& wall_model)
          {
            using WallModelType = std::decay_t<decltype(wall_model)>;
            if constexpr (std::is_same_v<WallModelType, Airways::KelvinVoigtWall>)
            {
              visit("airway_area_n", wall_model.area_n);
            }
          },
          model.wall_model);
    }

    /**
     * Call @p visit(name, values) for every history vector of the terminal unit block @p model.
     */
    template <typename Visitor>
    void visit_history(TerminalUnits::TerminalUnitModel& model, Visitor&& visit)
    {
      visit("terminal_unit_volume", model.data.volume_v);
      std::visit(
          [&](auto& rheological_model)
          {
            using ModelType = std::decay_t<decltype(rheological_model)>;
            if constexpr (std::is_same_v<ModelType, TerminalUnits::FourElementMaxwell>)
            {
              visit("terminal_unit_maxwell_pressure", rheological_model.maxwell_pressure_p_m);
            }
          },
          model.rheological_model);
      std::visit(
          [&](auto& recruitment_model)
          {
            using ModelType = std::decay_t<decltype(recruitment_model)>;
            if constexpr (std::is_same_v<ModelType, TerminalUnits::LinearPressureRecruitment>)
            {
              visit("terminal_unit_v0_n", recruitment_model.v0_n);
              visit("terminal_unit_v0_target", recruitment_model.v0_target);
              visit("terminal_unit_active_path_n", recruitment_model.pressure_law.active_path_n);
            }
          },
          model.recruitment_model);
    }

    /**
     * Call @p visit(name, values, local_element_ids) for every history vector of all blocks.
     */
    template <typename Visitor>
    void visit_all_history(const State& state, Visitor&& visit)
    {
      for (auto& model : state.airways.models)
      {
        visit_history(model, [&](const std::string& name, auto& values)
            { visit(name, values, model.data.local_element_id); });
      }
      for (auto& model : state.terminal_units.models)
      {
        visit_history(model, [&](const std::string& name, auto& values)
            { visit(name, values, model.data.local_element_id); });
      }
    }

    //! The history is written as doubles; enums such as the hysteresis path are converted.
    template <typename T>
    double to_double(T value)
    {
      if constexpr (std::is_enum_v<T>)
        return static_cast<double>(static_cast<std::underlying_type_t<T>>(value));
      else
        return static_cast<double>(value);
    }

    template <typename T>
    T from_double(double value)
    {
      if constexpr (std::is_enum_v<T>)
        return static_cast<T>(static_cast<std::underlying_type_t<T>>(std::lround(value)));
      else
        return static_cast<T>(value);
    }

    /**
     * Names of the history vectors of all blocks on all ranks. Writing and reading a vector is
     * collective, so every rank has to handle every name, even if it owns no block of that kind.
     */
    std::set<std::string> all_history_names(const State& state)
    {
      std::set<std::string> names;
      visit_all_history(state, [&](const std::string& name, const auto&, const std::vector<int>&)
          { names.insert(name); });
      return Core::Communication::all_reduce(names, state.discretization.get_comm());
    }

    /**
     * Largest number of dofs of any element, the number of columns of the element dof vector.
     * The map is the same on all ranks.
     */
    int max_dofs_per_element(const State& state)
    {
      int max_dofs = 0;
      for (const auto& [element_id, n_dofs] : state.global_dof_per_ele)
      {
        max_dofs = std::max(max_dofs, n_dofs);
      }
      return max_dofs;
    }
  }  // namespace

  void write(Core::IO::DiscretizationWriter& writer, const State& state)
  {
    const Core::LinAlg::Map& element_row_map = *state.discretization.element_row_map();

    // The dofs of every element, in the element's local dof order.
    Core::LinAlg::MultiVector<double> element_dofs(
        element_row_map, max_dofs_per_element(state), true);
    for (int lid = 0; lid < element_row_map.num_my_elements(); ++lid)
    {
      const int element_id = element_row_map.gid(lid);
      const int first_dof = state.first_global_dof_of_ele.at(element_id);
      for (int k = 0; k < state.global_dof_per_ele.at(element_id); ++k)
      {
        const int dof_lid = state.locally_relevant_dofs.get_map().lid(first_dof + k);
        FOUR_C_ASSERT(dof_lid >= 0, "Dof {} of element {} is not locally relevant.", first_dof + k,
            element_id);
        element_dofs.replace_local_value(
            lid, k, state.locally_relevant_dofs.local_values_as_span()[dof_lid]);
      }
    }
    writer.write_multi_vector(element_dofs_name, element_dofs, Core::IO::elementvector);

    // Every block of the same kind contributes to the same vector, at its own elements.
    std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>> history;
    for (const auto& name : all_history_names(state))
    {
      history[name] = std::make_shared<Core::LinAlg::Vector<double>>(element_row_map, true);
    }
    visit_all_history(state,
        [&](const std::string& name, const auto& values, const std::vector<int>& local_element_ids)
        {
          auto& vector = *history.at(name);
          for (std::size_t i = 0; i < values.size(); ++i)
            vector.replace_local_value(local_element_ids[i], to_double(values[i]));
        });

    for (const auto& [name, vector] : history)
    {
      writer.write_vector(name, vector, Core::IO::elementvector);
    }
  }

  void read(Core::IO::DiscretizationReader& reader, State& state)
  {
    const Core::LinAlg::Map& element_row_map = *state.discretization.element_row_map();

    auto element_dofs = std::make_shared<Core::LinAlg::MultiVector<double>>(
        element_row_map, max_dofs_per_element(state), true);
    reader.read_multi_vector(element_dofs, element_dofs_name);

    // Each rank only reads the dofs of its own elements, but the solution vector distributes its
    // dofs independently of the elements, and the locally relevant dofs include those of ghosted
    // elements. Collect the dofs of the own elements and export them to both.
    std::vector<int> dof_gids;
    std::vector<double> dof_values;
    for (int lid = 0; lid < element_row_map.num_my_elements(); ++lid)
    {
      const int element_id = element_row_map.gid(lid);
      const int first_dof = state.first_global_dof_of_ele.at(element_id);
      for (int k = 0; k < state.global_dof_per_ele.at(element_id); ++k)
      {
        dof_gids.push_back(first_dof + k);
        dof_values.push_back(element_dofs->get_vector(k).local_values_as_span()[lid]);
      }
    }
    const Core::LinAlg::Map element_dof_map(
        -1, static_cast<int>(dof_gids.size()), dof_gids.data(), 0, state.discretization.get_comm());
    Core::LinAlg::Vector<double> element_dof_values(element_dof_map, false);
    for (std::size_t i = 0; i < dof_values.size(); ++i)
      element_dof_values.replace_local_value(static_cast<int>(i), dof_values[i]);
    Core::LinAlg::export_to(element_dof_values, state.x);
    Core::LinAlg::export_to(element_dof_values, state.locally_relevant_dofs);

    std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>> history;
    for (const auto& name : all_history_names(state))
    {
      FOUR_C_ASSERT_ALWAYS(reader.has_vector(name),
          "The reduced lung restart data has no history field '{}'. Was it written with a "
          "different model setup?",
          name);
      auto vector = std::make_shared<Core::LinAlg::Vector<double>>(element_row_map, true);
      reader.read_vector(vector, name);
      history[name] = vector;
    }

    visit_all_history(state,
        [&](const std::string& name, auto& values, const std::vector<int>& local_element_ids)
        {
          using ValueType = typename std::decay_t<decltype(values)>::value_type;
          const auto stored = history.at(name)->local_values_as_span();
          for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = from_double<ValueType>(stored[local_element_ids[i]]);
        });
  }
}  // namespace ReducedLung::Restart

FOUR_C_NAMESPACE_CLOSE
