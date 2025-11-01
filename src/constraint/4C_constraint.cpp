// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_utils_function_of_time.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
Constraints::Constraint::Constraint(std::shared_ptr<Core::FE::Discretization> discr,
    const std::string& conditionname, int& minID, int& maxID)
    : actdisc_(discr)
{
  actdisc_->get_condition(conditionname, constrcond_);
  if (constrcond_.size())
  {
    constrtype_ = get_constr_type(conditionname);
    for (auto& i : constrcond_)
    {
      int condID = (i->parameters().get<int>("ConditionID"));
      if (condID > maxID)
      {
        maxID = condID;
      }
      if (condID < minID)
      {
        minID = condID;
      }

      auto const myinittime = i->parameters().get<double>("activeTime");

      inittimes_.insert(std::pair<int, double>(condID, myinittime));
      activecons_.insert(std::pair<int, bool>(condID, false));
    }
  }
  else
  {
    constrtype_ = none;
  }
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
Constraints::Constraint::Constraint(
    std::shared_ptr<Core::FE::Discretization> discr, const std::string& conditionname)
    : actdisc_(discr)
{
  actdisc_->get_condition(conditionname, constrcond_);

  if (constrcond_.size())
  {
    constrtype_ = get_constr_type(conditionname);

    for (auto& i : constrcond_)
    {
      int condID = i->parameters().get<int>("ConditionID");
      auto const myinittime = i->parameters().get<double>("activeTime");

      inittimes_.insert(std::pair<int, double>(condID, myinittime));
      activecons_.insert(std::pair<int, bool>(condID, false));
    }
  }
  else
  {
    constrtype_ = none;
  }
}

/*-----------------------------------------------------------------------*
|(private)                                                       tk 07/08|
*-----------------------------------------------------------------------*/
Constraints::Constraint::ConstrType Constraints::Constraint::get_constr_type(
    const std::string& name)
{
  if (name == "VolumeConstraint_3D" or name == "VolumeConstraint_3D_Pen")
    return volconstr3d;
  else if (name == "AreaConstraint_3D" or name == "AreaConstraint_3D_Pen")
    return areaconstr3d;
  else if (name == "AreaConstraint_2D")
    return areaconstr2d;
  else if (name == "MPC_NodeOnPlane_3D")
    return mpcnodeonplane3d;
  else if (name == "MPC_NodeOnLine_2D")
    return mpcnodeonline2d;
  else if (name == "MPC_NormalComponent_3D" or name == "MPC_NormalComponent_3D_Pen")
    return mpcnormalcomp3d;
  return none;
}

/*------------------------------------------------------------------------*
|(public)                                                       tk 08/08  |
|Initialization routine computes ref base values and activates conditions |
*------------------------------------------------------------------------*/
void Constraints::Constraint::initialize(
    Teuchos::ParameterList& params, Core::LinAlg::Vector<double>& systemvector3)
{
  // choose action
  switch (constrtype_)
  {
    case volconstr3d:
      params.set("action", "calc_struct_constrvol");
      break;
    case areaconstr3d:
      params.set("action", "calc_struct_constrarea");
      break;
    case areaconstr2d:
      params.set("action", "calc_struct_constrarea");
      break;
    case none:
      return;
    default:
      FOUR_C_THROW("Unknown constraint/monitor type to be evaluated in Constraint class!");
  }
  // start computing
  initialize_constraint(params, systemvector3);
  return;
}

/*------------------------------------------------------------------------*
|(public)                                                       tk 08/08  |
|Initialization routine activates conditions (restart)                    |
*------------------------------------------------------------------------*/
void Constraints::Constraint::initialize(const double& time)
{
  for (auto* cond : constrcond_)
  {
    // Get ConditionID of current condition if defined and write value in parameterlist
    int condID = cond->parameters().get<int>("ConditionID");

    // if current time (at) is larger than activation time of the condition, activate it
    if ((inittimes_.find(condID)->second <= time) && (activecons_.find(condID)->second == false))
    {
      activecons_.find(condID)->second = true;
      if (Core::Communication::my_mpi_rank(actdisc_->get_comm()) == 0)
      {
        std::cout << "Encountered another active condition (Id = " << condID
                  << ")  for restart time t = " << time << std::endl;
      }
    }
  }
}

/*-----------------------------------------------------------------------*
|(public)                                                        tk 07/08|
|Evaluate Constraints, choose the right action based on type             |
*-----------------------------------------------------------------------*/
void Constraints::Constraint::evaluate(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3)
{
  switch (constrtype_)
  {
    case volconstr3d:
      params.set("action", "calc_struct_volconstrstiff");
      break;
    case areaconstr3d:
      params.set("action", "calc_struct_areaconstrstiff");
      break;
    case areaconstr2d:
      params.set("action", "calc_struct_areaconstrstiff");
      break;
    case none:
      return;
    default:
      FOUR_C_THROW("Wrong constraint type to evaluate systemvector!");
  }
  evaluate_constraint(
      params, systemmatrix1, systemmatrix2, systemvector1, *systemvector2, systemvector3);
  return;
}

/*-----------------------------------------------------------------------*
 |(private)                                                     tk 07/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void Constraints::Constraint::evaluate_constraint(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
    Core::LinAlg::Vector<double>& systemvector2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3)
{
  if (!(actdisc_->filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!actdisc_->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");
  // get the current time
  const double time = params.get("total time", -1.0);

  const bool assemblemat1 = systemmatrix1 != nullptr;
  const bool assemblemat2 = systemmatrix2 != nullptr;
  const bool assemblevec1 = systemvector1 != nullptr;
  const bool assemblevec3 = systemvector3 != nullptr;

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (auto* cond : constrcond_)
  {
    // get values from time integrator to scale matrices with
    double scStiff = params.get("scaleStiffEntries", 1.0);
    double scConMat = params.get("scaleConstrMat", 1.0);

    // Get ConditionID of current condition if defined and write value in parameterlist
    int condID = cond->parameters().get<int>("ConditionID");
    params.set("ConditionID", condID);

    // is conditions supposed to be active?
    if (inittimes_.find(condID)->second <= time)
    {
      // is conditions already labeled as active?
      if (activecons_.find(condID)->second == false)
      {
        const std::string action = params.get<std::string>("action");
        std::shared_ptr<Core::LinAlg::Vector<double>> displast =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("old disp");
        actdisc_->set_state("displacement", *displast);
        initialize(params, systemvector2);
        std::shared_ptr<Core::LinAlg::Vector<double>> disp =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("new disp");
        actdisc_->set_state("displacement", *disp);
        params.set("action", action);
      }

      // Evaluate loadcurve if defined. Put current load factor in parameterlist
      const auto curvenum = cond->parameters().get<std::optional<int>>("curve");
      double curvefac = 1.0;
      if (curvenum.has_value() && curvenum.value() > 0)
      {
        // function_by_id takes a zero-based index
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curvenum.value())
                       .evaluate(time);
      }

      // global and local ID of this bc in the redundant vectors
      const int offsetID = params.get<int>("OffsetID");
      int gindex = condID - offsetID;
      const int lindex = (systemvector3->get_map()).lid(gindex);

      // store loadcurve values
      std::shared_ptr<Core::LinAlg::Vector<double>> timefact =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("vector curve factors");
      timefact->replace_global_values(1, &curvefac, &gindex);

      // Get the current lagrange multiplier value for this condition
      const std::shared_ptr<Core::LinAlg::Vector<double>> lagramul =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("LagrMultVector");
      const double lagraval = lagramul->local_values_as_span()[lindex];

      // elements might need condition
      params.set<const Core::Conditions::Condition*>("condition", cond);

      // define element matrices and vectors
      Core::LinAlg::SerialDenseMatrix elematrix1;
      Core::LinAlg::SerialDenseMatrix elematrix2;
      Core::LinAlg::SerialDenseVector elevector1;
      Core::LinAlg::SerialDenseVector elevector2;
      Core::LinAlg::SerialDenseVector elevector3;

      const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = cond->geometry();
      // if (geom.empty()) FOUR_C_THROW("evaluation of condition with empty geometry");
      // no check for empty geometry here since in parallel computations
      // can exist processors which do not own a portion of the elements belonging
      // to the condition geometry
      for (const auto& [id, ele] : geom)
      {
        // get element location vector and ownerships
        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;
        ele->location_vector(*actdisc_, lm, lmowner, lmstride);

        // get dimension of element matrices and vectors
        // Reshape element matrices and vectors and init to zero
        const int eledim = (int)lm.size();
        elematrix1.shape(eledim, eledim);
        elematrix2.shape(eledim, eledim);
        elevector1.size(eledim);
        elevector2.size(eledim);
        elevector3.size(1);

        // call the element specific evaluate method
        int err = ele->evaluate(
            params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
        if (err) FOUR_C_THROW("error while evaluating elements");

        // assembly
        int eid = ele->id();
        if (assemblemat1)
        {
          // scale with time integrator dependent value
          elematrix1.scale(scStiff * lagraval);
          systemmatrix1->assemble(eid, lmstride, elematrix1, lm, lmowner);
        }
        if (assemblemat2)
        {
          // assemble to rectangular matrix. The column corresponds to the constraint ID.
          // scale with time integrator dependent value
          std::vector<int> colvec(1);
          colvec[0] = gindex;
          elevector2.scale(scConMat);
          systemmatrix2->assemble(eid, lmstride, elevector2, lm, lmowner, colvec);
        }
        if (assemblevec1)
        {
          elevector1.scale(lagraval);
          Core::LinAlg::assemble(*systemvector1, elevector1, lm, lmowner);
        }
        if (assemblevec3)
        {
          std::vector<int> constrlm;
          std::vector<int> constrowner;
          constrlm.push_back(gindex);
          constrowner.push_back(ele->owner());
          Core::LinAlg::assemble(*systemvector3, elevector3, constrlm, constrowner);
        }
      }
    }
  }
  return;
}  // end of evaluate_condition

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Constraints::Constraint::initialize_constraint(
    Teuchos::ParameterList& params, Core::LinAlg::Vector<double>& systemvector)
{
  if (!(actdisc_->filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!actdisc_->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");
  // get the current time
  const double time = params.get("total time", -1.0);

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (auto* cond : constrcond_)
  {
    // Get ConditionID of current condition if defined and write value in parameterlist

    int condID = cond->parameters().get<int>("ConditionID");
    params.set("ConditionID", condID);

    // if current time is larger than initialization time of the condition, start computing
    if ((inittimes_.find(condID)->second <= time) && (!(activecons_.find(condID)->second)))
    {
      params.set<const Core::Conditions::Condition*>("condition", cond);

      // define element matrices and vectors
      Core::LinAlg::SerialDenseMatrix elematrix1;
      Core::LinAlg::SerialDenseMatrix elematrix2;
      Core::LinAlg::SerialDenseVector elevector1;
      Core::LinAlg::SerialDenseVector elevector2;
      Core::LinAlg::SerialDenseVector elevector3;

      const auto& geom = cond->geometry();
      // no check for empty geometry here since in parallel computations
      // can exist processors which do not own a portion of the elements belonging
      // to the condition geometry
      for (const auto& [id, ele] : geom)
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
        int offsetID = params.get<int>("OffsetID");
        constrlm.push_back(condID - offsetID);
        constrowner.push_back(ele->owner());
        Core::LinAlg::assemble(systemvector, elevector3, constrlm, constrowner);
      }
      // remember next time, that this condition is already initialized, i.e. active
      activecons_.find(condID)->second = true;

      if (Core::Communication::my_mpi_rank(actdisc_->get_comm()) == 0)
      {
        std::cout << "Encountered a new active Lagrange condition (Id = " << condID
                  << ")  at time t = " << time << std::endl;
      }
    }
  }
}  // end of Initialize Constraint


/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
std::vector<int> Constraints::Constraint::get_active_cond_id()
{
  std::vector<int> condID;
  std::map<int, bool>::const_iterator mapit;
  for (mapit = activecons_.begin(); mapit != activecons_.end(); mapit++)
  {
    if (mapit->second) condID.push_back(mapit->first);
  }
  return condID;
}

FOUR_C_NAMESPACE_CLOSE
