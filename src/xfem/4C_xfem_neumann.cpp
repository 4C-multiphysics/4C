// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_neumann.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  evaluate Neumann conditions (public)                    schott 08/11|
 *----------------------------------------------------------------------*/
void XFEM::evaluate_neumann(Teuchos::ParameterList& params,
    std::shared_ptr<Core::FE::Discretization> discret, Core::LinAlg::Vector<double>& systemvector,
    Core::LinAlg::SparseOperator* systemmatrix)
{
  TEUCHOS_FUNC_TIME_MONITOR("FLD::XFluid::XFluidState::Evaluate 5) evaluate_neumann");


  if (!discret->filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!discret->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  bool assemblemat = (systemmatrix != nullptr);

  // get the current time
  const double time = params.get("total time", -1.0);

  std::multimap<std::string, Core::Conditions::Condition*>::iterator fool;
  std::multimap<std::string, Core::Conditions::Condition*> condition;

  // vector for conditions of one special type
  std::vector<Core::Conditions::Condition*> condition_vec;

  //================================================
  // Load Neumann conditions from discretization
  // REMARK: standard volume Neumann conditions are not loaded -> evaluated in Evaluate
  //         for XFEM Neumann boundaries: we assumme only XFEM Surface(!) Neumann conditions
  //================================================

  // get standard Point Neumann conditions
  condition_vec.clear();
  discret->get_condition("PointNeumann", condition_vec);
  // copy conditions to a condition multimap
  for (size_t i = 0; i < condition_vec.size(); i++)
  {
    condition.insert(std::pair<std::string, Core::Conditions::Condition*>(
        std::string("PointNeumann"), condition_vec[i]));
  }

  // get standard Surface Neumann conditions
  condition_vec.clear();
  discret->get_condition("LineNeumann", condition_vec);
  for (size_t i = 0; i < condition_vec.size(); i++)
  {
    condition.insert(std::pair<std::string, Core::Conditions::Condition*>(
        std::string("LineNeumann"), condition_vec[i]));
  }

  // get standard Surface Neumann conditions
  condition_vec.clear();
  discret->get_condition("SurfaceNeumann", condition_vec);
  for (size_t i = 0; i < condition_vec.size(); i++)
  {
    condition.insert(std::pair<std::string, Core::Conditions::Condition*>(
        std::string("SurfaceNeumann"), condition_vec[i]));
  }

  // NOTE: WE SKIP VolumeNeumann conditions -> there are evaluated at the fluid element level
  // (Bodyforce!)

  // TODO: we should introduce safety checks, when the Neumann boundary is cut!
  // TODO: shift the functionality to XFEM Neumann conditions and automatically detect, whether the
  // element is cut by an interface
  // similar to Weak Dirichlet conditions!

  // evaluate standard Neumann conditions
  evaluate_neumann_standard(
      condition, time, assemblemat, params, *discret, systemvector, systemmatrix);
}



/*----------------------------------------------------------------------*
 |  evaluate Neumann for standard conditions (public)       schott 08/11|
 *----------------------------------------------------------------------*/
void XFEM::evaluate_neumann_standard(
    std::multimap<std::string, Core::Conditions::Condition*>& condition, const double time,
    bool assemblemat, Teuchos::ParameterList& params, Core::FE::Discretization& discret,
    Core::LinAlg::Vector<double>& systemvector, Core::LinAlg::SparseOperator* systemmatrix)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "FLD::XFluid::XFluidState::EvaluateNeumannStandard" );

  std::multimap<std::string, Core::Conditions::Condition*>::iterator fool;

  //--------------------------------------------------------
  // loop through Point Neumann conditions and evaluate them
  //--------------------------------------------------------
  for (fool = condition.begin(); fool != condition.end(); ++fool)
  {
    if (fool->first != (std::string) "PointNeumann") continue;
    if (assemblemat && !Core::Communication::my_mpi_rank(systemvector.Comm()))
      std::cout << "WARNING: No linearization of PointNeumann conditions" << std::endl;
    Core::Conditions::Condition& cond = *(fool->second);
    const std::vector<int>* nodeids = cond.get_nodes();
    if (!nodeids) FOUR_C_THROW("PointNeumann condition does not have nodal cloud");
    const int nnode = (*nodeids).size();
    const auto* funct = cond.parameters().get_if<std::vector<int>>("FUNCT");
    const auto* onoff = cond.parameters().get_if<std::vector<int>>("ONOFF");
    const auto* val = cond.parameters().get_if<std::vector<double>>("VAL");
    // Neumann BCs for some historic reason only have one curve
    int functnum = -1;
    if (funct) functnum = (*funct)[0];
    double functfac = 1.0;
    if (functnum >= 0)
    {
      functfac = Global::Problem::instance()
                     ->function_by_id<Core::Utils::FunctionOfTime>(functnum)
                     .evaluate(time);
    }
    for (int i = 0; i < nnode; ++i)
    {
      // do only nodes in my row map
      if (!discret.node_row_map()->MyGID((*nodeids)[i])) continue;
      Core::Nodes::Node* actnode = discret.g_node((*nodeids)[i]);
      if (!actnode) FOUR_C_THROW("Cannot find global node %d", (*nodeids)[i]);
      // call explicitly the main dofset, i.e. the first column
      std::vector<int> dofs = discret.dof(0, actnode);
      const unsigned numdf = dofs.size();
      for (unsigned j = 0; j < numdf; ++j)
      {
        if ((*onoff)[j] == 0) continue;
        const int gid = dofs[j];
        double value = (*val)[j];
        value *= functfac;
        const int lid = systemvector.Map().LID(gid);
        if (lid < 0) FOUR_C_THROW("Global id %d not on this proc in system vector", gid);
        systemvector[lid] += value;
      }
    }
  }

  //--------------------------------------------------------
  // loop through line/surface Neumann BCs and evaluate them
  // ATTENTION: VolumeNeumann conditions (bodyforces) are evaluated in Evaluate
  //--------------------------------------------------------
  for (fool = condition.begin(); fool != condition.end(); ++fool)
    if (fool->first == (std::string) "LineNeumann" || fool->first == (std::string) "SurfaceNeumann")
    {
      Core::Conditions::Condition& cond = *(fool->second);
      std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = cond.geometry();
      std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator curr;
      Core::LinAlg::SerialDenseVector elevector;
      Core::LinAlg::SerialDenseMatrix elematrix;
      for (curr = geom.begin(); curr != geom.end(); ++curr)
      {
        // get element location vector, dirichlet flags and ownerships
        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;
        curr->second->location_vector(discret, lm, lmowner, lmstride);
        elevector.size((int)lm.size());
        if (!assemblemat)
        {
          curr->second->evaluate_neumann(params, discret, cond, lm, elevector);
          Core::LinAlg::assemble(systemvector, elevector, lm, lmowner);
        }
        else
        {
          const int size = (int)lm.size();
          if (elematrix.numRows() != size)
            elematrix.shape(size, size);
          else
            elematrix.putScalar(0.0);
          curr->second->evaluate_neumann(params, discret, cond, lm, elevector, &elematrix);
          Core::LinAlg::assemble(systemvector, elevector, lm, lmowner);
          systemmatrix->assemble(lmstride, elematrix, lm, lmowner);
        }
      }
    }
}

FOUR_C_NAMESPACE_CLOSE
