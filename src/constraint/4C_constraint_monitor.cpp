// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_monitor.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"

#include <Teuchos_ParameterList.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
Constraints::Monitor::Monitor(std::shared_ptr<Core::FE::Discretization> discr,
    const std::string& conditionname, int& minID, int& maxID)
    : actdisc_(discr)
{
  actdisc_->get_condition(conditionname, moncond_);
  if (moncond_.size())
  {
    montype_ = get_moni_type(conditionname);
    for (auto& i : moncond_)
    {
      int condID = i->parameters().get<int>("ConditionID");

      if (condID > maxID)
      {
        maxID = condID;
      }
      if (condID < minID)
      {
        minID = condID;
      }
    }
  }
  else
  {
    montype_ = none;
  }
}


/*-----------------------------------------------------------------------*
|(private)                                                       tk 07/08|
*-----------------------------------------------------------------------*/
Constraints::Monitor::MoniType Constraints::Monitor::get_moni_type(const std::string& name)
{
  if (name == "VolumeMonitor_3D")
    return volmonitor3d;
  else if (name == "AreaMonitor_3D")
    return areamonitor3d;
  else if (name == "AreaMonitor_2D")
    return areamonitor2d;
  return none;
}


/*-----------------------------------------------------------------------*
|(public)                                                        tk 07/08|
|Evaluate Monitors, choose the right action based on type             |
*-----------------------------------------------------------------------*/
void Constraints::Monitor::evaluate(
    Teuchos::ParameterList& params, Core::LinAlg::Vector<double>& systemvector)
{
  switch (montype_)
  {
    case volmonitor3d:
      params.set("action", "calc_struct_constrvol");
      break;
    case areamonitor3d:
      params.set("action", "calc_struct_monitarea");
      break;
    case areamonitor2d:
      params.set("action", "calc_struct_constrarea");
      break;
    case none:
      return;
    default:
      FOUR_C_THROW("Unknown monitor type to be evaluated in Monitor class!");
  }
  evaluate_monitor(params, systemvector);
}


/*-----------------------------------------------------------------------*
 |(private)                                                     tk 08/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void Constraints::Monitor::evaluate_monitor(
    Teuchos::ParameterList& params, Core::LinAlg::Vector<double>& systemvector)
{
  if (!(actdisc_->filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!actdisc_->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (auto* cond : moncond_)
  {
    // Get ConditionID of current condition if defined and write value in parameterlist
    const int condID = cond->parameters().get<int>("ConditionID");
    const int offsetID = params.get("OffsetID", 0);
    params.set<const Core::Conditions::Condition*>("condition", cond);

    // define element matrices and vectors
    Core::LinAlg::SerialDenseMatrix elematrix1;
    Core::LinAlg::SerialDenseMatrix elematrix2;
    Core::LinAlg::SerialDenseVector elevector1;
    Core::LinAlg::SerialDenseVector elevector2;
    Core::LinAlg::SerialDenseVector elevector3;

    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = cond->geometry();
    // no check for empty geometry here since in parallel computations
    // can exist processors which do not own a portion of the elements belonging
    // to the condition geometry
    for (const auto& ele : geom | std::views::values)
    {
      // get element location vector and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->location_vector(*actdisc_, lm, lmowner, lmstride);

      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      elevector3.size(1);

      // call the element specific evaluate method
      int err = ele->evaluate(
          params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("error while evaluating elements");

      // assembly
      std::vector<int> constrlm;
      std::vector<int> constrowner;
      constrlm.push_back(condID - offsetID);
      constrowner.push_back(ele->owner());
      Core::LinAlg::assemble(systemvector, elevector3, constrlm, constrowner);
    }
  }
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Constraints::Monitor::set_state(const std::string& state, Core::LinAlg::Vector<double>& V)
{
  actdisc_->set_state(state, V);
}

FOUR_C_NAMESPACE_CLOSE
