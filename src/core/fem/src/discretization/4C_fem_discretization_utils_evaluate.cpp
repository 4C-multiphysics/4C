// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_general_element.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Utils::evaluate(Core::FE::Discretization& discret, Teuchos::ParameterList& eparams,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvector,
    const Core::LinAlg::Map* col_ele_map)
{
  std::vector<std::shared_ptr<Core::LinAlg::SparseOperator>> systemmatrices(2, nullptr);
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> systemvectors(3, nullptr);

  systemmatrices[0] = systemmatrix;
  systemvectors[0] = systemvector;

  evaluate(discret, eparams, systemmatrices, systemvectors, col_ele_map);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Utils::evaluate(Core::FE::Discretization& discret, Teuchos::ParameterList& eparams,
    std::vector<std::shared_ptr<Core::LinAlg::SparseOperator>>& systemmatrices,
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& systemvectors,
    const Core::LinAlg::Map* col_ele_map)
{
  FOUR_C_ASSERT(systemmatrices.size() <= 2,
      "Currently a maximum number of two "
      "system-matrices is supported!");
  FOUR_C_ASSERT(systemvectors.size() <= 3,
      "Currently a maximum number of three "
      "system-vectors is supported!");

  if (systemmatrices.size() < 2) systemmatrices.resize(2, nullptr);
  if (systemvectors.size() < 3) systemvectors.resize(3, nullptr);

  Core::FE::AssembleStrategy strategy(0, 0, systemmatrices[0], systemmatrices[1], systemvectors[0],
      systemvectors[1], systemvectors[2]);
  evaluate(discret, eparams, strategy, col_ele_map);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::Utils::evaluate(Core::FE::Discretization& discret, Teuchos::ParameterList& eparams,
    Core::FE::AssembleStrategy& strategy, const Core::LinAlg::Map* col_ele_map)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::FE::Utils::Evaluate");

  if (!discret.filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!discret.have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  int row = strategy.first_dof_set();
  int col = strategy.second_dof_set();

  Core::Elements::LocationArray la(discret.num_dof_sets());

  bool is_subset = false;
  if (not col_ele_map)
    col_ele_map = discret.element_col_map();
  else
    is_subset = true;

  // loop over column elements
  const int numcolele = col_ele_map->NumMyElements();
  const int* ele_gids = col_ele_map->MyGlobalElements();

  for (int i = 0; i < numcolele; ++i)
  {
    Core::Elements::Element* actele = nullptr;
    if (is_subset)
    {
      const int egid = ele_gids[i];
      actele = discret.g_element(egid);
    }
    else
      actele = discret.l_col_element(i);

    {
      TEUCHOS_FUNC_TIME_MONITOR("Core::FE::Utils::Evaluate LocationVector");
      // get element location vector, dirichlet flags and ownerships
      actele->location_vector(discret, la, false);
    }

    {
      TEUCHOS_FUNC_TIME_MONITOR("Core::FE::Utils::Evaluate Resize");

      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      strategy.clear_element_storage(la[row].size(), la[col].size());
    }

    {
      TEUCHOS_FUNC_TIME_MONITOR("Core::FE::Utils::Evaluate elements");
      // call the element evaluate method
      int err = actele->evaluate(eparams, discret, la, strategy.elematrix1(), strategy.elematrix2(),
          strategy.elevector1(), strategy.elevector2(), strategy.elevector3());
      if (err)
        FOUR_C_THROW("Proc {}: Element {} returned err={}",
            Core::Communication::my_mpi_rank(discret.get_comm()), actele->id(), err);
    }

    {
      TEUCHOS_FUNC_TIME_MONITOR("Core::FE::Utils::Evaluate assemble");
      int eid = actele->id();
      strategy.assemble_matrix1(eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
      strategy.assemble_matrix2(eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
      strategy.assemble_vector1(la[row].lm_, la[row].lmowner_);
      strategy.assemble_vector2(la[row].lm_, la[row].lmowner_);
      strategy.assemble_vector3(la[row].lm_, la[row].lmowner_);
    }

  }  // loop over all considered elements

  return;
}

FOUR_C_NAMESPACE_CLOSE
