// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_penalty.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_utils_function_of_time.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Constraints::ConstraintPenalty::ConstraintPenalty(
    std::shared_ptr<Core::FE::Discretization> discr, const std::string& conditionname)
    : Constraint(discr, conditionname)
{
  if (constrcond_.size())
  {
    for (auto* i : constrcond_)
    {
      const double mypenalty = i->parameters().get<double>("penalty");
      const double myrho = i->parameters().get<double>("rho");
      int condID = i->parameters().get<int>("ConditionID");

      penalties_.insert(std::pair<int, double>(condID, mypenalty));
      rho_.insert(std::pair<int, double>(condID, myrho));
    }
    int nummyele = 0;
    int numele = penalties_.size();
    if (!Core::Communication::my_mpi_rank(actdisc_->get_comm()))
    {
      nummyele = numele;
    }
    // initialize maps and importer
    errormap_ = std::make_shared<Core::LinAlg::Map>(numele, nummyele, 0, actdisc_->get_comm());
    rederrormap_ = Core::LinAlg::allreduce_e_map(*errormap_);
    errorexport_ = std::make_shared<Core::LinAlg::Export>(*rederrormap_, *errormap_);
    errorimport_ = std::make_shared<Core::LinAlg::Import>(*rederrormap_, *errormap_);
    acterror_ = std::make_shared<Core::LinAlg::Vector<double>>(*rederrormap_);
    initerror_ = std::make_shared<Core::LinAlg::Vector<double>>(*rederrormap_);
    lagrvalues_ = std::make_shared<Core::LinAlg::Vector<double>>(*rederrormap_);
    lagrvalues_force_ = std::make_shared<Core::LinAlg::Vector<double>>(*rederrormap_);
  }
  else
  {
    constrtype_ = none;
  }
}

void Constraints::ConstraintPenalty::initialize(
    Teuchos::ParameterList& params, Core::LinAlg::Vector<double>& systemvector3)
{
  FOUR_C_THROW("method not used for penalty formulation!");
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void Constraints::ConstraintPenalty::initialize(Teuchos::ParameterList& params)
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
  evaluate_error(params, *initerror_);
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void Constraints::ConstraintPenalty::initialize(const double& time)
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
 *-----------------------------------------------------------------------*/
void Constraints::ConstraintPenalty::evaluate(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1)
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
  acterror_->put_scalar(0.0);
  evaluate_error(params, *acterror_);

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
  evaluate_constraint(params, systemmatrix1, systemvector1);
}

/*-----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Constraints::ConstraintPenalty::evaluate_constraint(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1)
{
  if (!(actdisc_->filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!actdisc_->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");
  // get the current time
  const double time = params.get("total time", -1.0);

  const bool assemblemat1 = systemmatrix1 != nullptr;
  const bool assemblevec1 = systemvector1 != nullptr;

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (auto* cond : constrcond_)
  {
    // get values from time integrator to scale matrices with
    double scStiff = params.get("scaleStiffEntries", 1.0);

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
        // last converged step is used reference
        initialize(params);
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

      double diff = (curvefac * (*initerror_).local_values_as_span()[condID - 1] -
                     (*acterror_).local_values_as_span()[condID - 1]);

      // take care when calling this evaluate function separately (evaluate force / evaluate
      // force+stiff)
      if (assemblemat1) (*lagrvalues_).get_values()[condID - 1] += rho_[condID] * diff;
      if (assemblevec1 and !(assemblemat1))
        (*lagrvalues_force_).get_values()[condID - 1] =
            (*lagrvalues_).local_values_as_span()[condID - 1] + rho_[condID] * diff;

      // elements might need condition
      params.set<const Core::Conditions::Condition*>("condition", cond);

      // define element matrices and vectors
      Core::LinAlg::SerialDenseMatrix elematrix1;
      Core::LinAlg::SerialDenseMatrix elematrix2;
      Core::LinAlg::SerialDenseVector elevector1;
      Core::LinAlg::SerialDenseVector elevector2;
      Core::LinAlg::SerialDenseVector elevector3;

      const auto& geom = cond->geometry();
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

        elevector1.size(eledim);
        elevector3.size(1);

        // call the element specific evaluate method
        int err = ele->evaluate(
            params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
        if (err) FOUR_C_THROW("error while evaluating elements");

        elematrix2 = elematrix1;
        elevector2 = elevector1;

        // assembly
        int eid = ele->id();

        // scale with time integrator dependent value
        elematrix1.scale(diff);
        for (int i = 0; i < eledim; i++)
          for (int j = 0; j < eledim; j++) elematrix1(i, j) += elevector1(i) * elevector1(j);

        if (assemblemat1)
        {
          elematrix1.scale(scStiff * penalties_[condID]);
          elematrix2.scale((*lagrvalues_).local_values_as_span()[condID - 1] * scStiff);
          systemmatrix1->assemble(eid, lmstride, elematrix1, lm, lmowner);
          systemmatrix1->assemble(eid, lmstride, elematrix2, lm, lmowner);
        }

        if (assemblevec1)
        {
          elevector1.scale(penalties_[condID] * diff);
          //          elevector2.Scale((*lagrvalues_).local_values_as_span()[condID-1]);
          // take care when calling this evaluate function separately (evaluate force / evaluate
          // force+stiff)
          if (!assemblemat1)
            elevector2.scale((*lagrvalues_force_).local_values_as_span()[condID - 1]);
          if (assemblemat1) elevector2.scale((*lagrvalues_).local_values_as_span()[condID - 1]);
          Core::LinAlg::assemble(*systemvector1, elevector1, lm, lmowner);
          Core::LinAlg::assemble(*systemvector1, elevector2, lm, lmowner);
        }
      }
    }
  }
}  // end of evaluate_condition

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Constraints::ConstraintPenalty::evaluate_error(
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
    if (inittimes_.find(condID)->second <= time)
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
        constrlm.push_back(condID - 1);
        constrowner.push_back(ele->owner());
        Core::LinAlg::assemble(systemvector, elevector3, constrlm, constrowner);
      }

      if (Core::Communication::my_mpi_rank(actdisc_->get_comm()) == 0 &&
          (!(activecons_.find(condID)->second)))
      {
        std::cout << "Encountered a new active penalty condition (Id = " << condID
                  << ")  at time t = " << time << std::endl;
      }

      // remember next time, that this condition is already initialized, i.e. active
      activecons_.find(condID)->second = true;
    }
  }
  Core::LinAlg::Vector<double> acterrdist(*errormap_);
  acterrdist.export_to(systemvector, *errorexport_, Add);
  systemvector.import(acterrdist, *errorimport_, Insert);
}  // end of evaluate_error

FOUR_C_NAMESPACE_CLOSE
