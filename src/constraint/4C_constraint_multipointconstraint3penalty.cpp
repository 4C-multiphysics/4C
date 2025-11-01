// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_multipointconstraint3penalty.hpp"

#include "4C_constraint_element3.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_transparent.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_utils_function_of_time.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Constraints::MPConstraint3Penalty::MPConstraint3Penalty(
    std::shared_ptr<Core::FE::Discretization> discr,  ///< discretization constraint lives on
    const std::string& CondName  ///< Name of condition to create constraint from
    )
    : MPConstraint(discr, CondName)
{
  if (constrcond_.size())
  {
    // control the constraint by absolute or relative values
    for (auto* conditer : constrcond_)
    {
      const int condID = conditer->parameters().get<int>("ConditionID");
      penalties_[condID] = conditer->parameters().get<double>("penalty");
      const std::string type = conditer->parameters().get<std::string>("control");
      if (type == "abs")
        absconstraint_[condID] = true;
      else
      {
        absconstraint_[condID] = false;
      }
    }

    int startID = 0;
    constraintdis_ = create_discretization_from_condition(
        actdisc_, constrcond_, "ConstrDisc", "CONSTRELE3", startID);

    std::map<int, std::shared_ptr<Core::FE::Discretization>>::iterator discriter;
    for (discriter = constraintdis_.begin(); discriter != constraintdis_.end(); discriter++)
    {
      std::shared_ptr<Core::LinAlg::Map> newcolnodemap =
          Core::Rebalance::compute_node_col_map(*actdisc_, *discriter->second);
      actdisc_->redistribute({*actdisc_->node_row_map(), *newcolnodemap});
      std::shared_ptr<Core::DOFSets::DofSet> newdofset =
          std::make_shared<Core::DOFSets::TransparentDofSet>(actdisc_);
      (discriter->second)->replace_dof_set(newdofset);
      newdofset = nullptr;
      (discriter->second)->fill_complete();
    }

    int nummyele = 0;
    int numele = eletocond_id_.size();
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
  }
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void Constraints::MPConstraint3Penalty::initialize(const double& time)
{
  for (auto* cond : constrcond_)
  {
    // Get ConditionID of current condition if defined and write value in parameterlist
    int condID = cond->parameters().get<int>("ConditionID");

    // if current time (at) is larger than activation time of the condition, activate it
    if ((inittimes_.find(condID)->second < time) && (!activecons_.find(condID)->second))
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
void Constraints::MPConstraint3Penalty::initialize(
    Teuchos::ParameterList& params, std::shared_ptr<Core::LinAlg::Vector<double>> systemvector)
{
  FOUR_C_THROW("method not used for penalty formulation!");
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Constraints::MPConstraint3Penalty::initialize(Teuchos::ParameterList& params)
{
  const double time = params.get("total time", -1.0);

  for (auto* cond : constrcond_)
  {
    int condID = cond->parameters().get<int>("ConditionID");
    // control absolute values
    switch (type())
    {
      case mpcnodeonplane3d:
      case mpcnormalcomp3d:
        params.set("action", "calc_MPC_state");
        break;
      case none:
        return;
      default:
        FOUR_C_THROW("Constraint/monitor is not an multi point constraint!");
    }

    evaluate_error(*constraintdis_.find(condID)->second, params, *initerror_, true);

    activecons_.find(condID)->second = true;
    if (Core::Communication::my_mpi_rank(actdisc_->get_comm()) == 0)
    {
      std::cout << "Encountered a new active condition (Id = " << condID
                << ")  at time t = " << time << std::endl;
    }
    //    std::cout << "initial error "<< *initerror_<<std::endl;
  }
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Constraints::MPConstraint3Penalty::evaluate(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3)
{
  switch (type())
  {
    case mpcnodeonplane3d:
    case mpcnormalcomp3d:
      params.set("action", "calc_MPC_state");
      break;
    case none:
      return;
    default:
      FOUR_C_THROW("Constraint/monitor is not an multi point constraint!");
  }

  acterror_->put_scalar(0.0);
  std::map<int, std::shared_ptr<Core::FE::Discretization>>::iterator discriter;
  for (discriter = constraintdis_.begin(); discriter != constraintdis_.end(); discriter++)
    evaluate_error(*discriter->second, params, *acterror_);

  //    std::cout << "current error "<< *acterror_<<std::endl;

  switch (type())
  {
    case mpcnodeonplane3d:
    case mpcnormalcomp3d:
      params.set("action", "calc_MPC_stiff");
      break;
    case none:
      return;
    default:
      FOUR_C_THROW("Constraint/monitor is not an multi point constraint!");
  }
  for (discriter = constraintdis_.begin(); discriter != constraintdis_.end(); discriter++)
    evaluate_constraint(discriter->second, params, systemmatrix1, systemmatrix2, systemvector1,
        systemvector2, systemvector3);

  return;
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
std::map<int, std::shared_ptr<Core::FE::Discretization>>
Constraints::MPConstraint3Penalty::create_discretization_from_condition(
    std::shared_ptr<Core::FE::Discretization> actdisc,
    std::span<const Core::Conditions::Condition*> constrcondvec, const std::string& discret_name,
    const std::string& element_name, int& startID)
{
  // start with empty map
  std::map<int, std::shared_ptr<Core::FE::Discretization>> newdiscmap;

  if (!actdisc->filled())
  {
    actdisc->fill_complete();
  }

  if (constrcondvec.size() == 0)
    FOUR_C_THROW(
        "number of multi point constraint conditions = 0 --> cannot create constraint "
        "discretization");

  // Loop all conditions in constrcondvec and build discretization for any condition ID

  int index = 0;  // counter for the index of condition in vector
  std::span<const Core::Conditions::Condition*>::iterator conditer;
  for (conditer = constrcondvec.begin(); conditer != constrcondvec.end(); conditer++)
  {
    // initialize a new discretization
    MPI_Comm com(actdisc->get_comm());
    std::shared_ptr<Core::FE::Discretization> newdis =
        std::make_shared<Core::FE::Discretization>(discret_name, com, actdisc->n_dim());
    const int myrank = Core::Communication::my_mpi_rank(newdis->get_comm());
    std::set<int> rownodeset;
    std::set<int> colnodeset;
    const Core::LinAlg::Map* actnoderowmap = actdisc->node_row_map();
    // get node IDs, this vector will only contain FREE nodes in the end
    std::vector<int> ngid = *((*conditer)->get_nodes());
    std::vector<int> defnv;
    switch (type())
    {
      case mpcnodeonplane3d:
      {
        // take three nodes defining plane as specified by user and put them into a set
        const auto& defnvp = (*conditer)->parameters().get<std::vector<int>>("planeNodes");
        defnv = defnvp;
      }
      break;
      case mpcnormalcomp3d:
      {
        // take master node
        const int defn = (*conditer)->parameters().get<int>("masterNode");
        defnv.push_back(defn);
      }
      break;
      default:
        FOUR_C_THROW("not good!");
    }
    std::set<int> defns(defnv.begin(), defnv.end());
    std::set<int>::iterator nsit;
    // safe gids of definition nodes in a vector
    std::vector<int> defnodeIDs;

    int counter = 1;  // counter is used to keep track of deleted node ids from the vector, input
                      // starts with 1

    for (nsit = defns.begin(); nsit != defns.end(); ++nsit)
    {
      defnodeIDs.push_back(ngid.at((*nsit) - counter));
      ngid.erase(ngid.begin() + (*nsit) - counter);
      counter++;
    }

    unsigned int nodeiter;
    // loop over all free nodes of condition
    for (nodeiter = 0; nodeiter < ngid.size(); nodeiter++)
    {
      std::vector<int> ngid_ele = defnodeIDs;
      ngid_ele.push_back(ngid[nodeiter]);
      const int numnodes = ngid_ele.size();
      remove_copy_if(ngid_ele.data(), ngid_ele.data() + numnodes,
          inserter(rownodeset, rownodeset.begin()),
          std::not_fn(Core::Conditions::MyGID(actnoderowmap)));
      // copy node ids specified in condition to colnodeset
      copy(ngid_ele.data(), ngid_ele.data() + numnodes, inserter(colnodeset, colnodeset.begin()));

      // construct constraint nodes, which use the same global id as the standard nodes
      for (int i = 0; i < actnoderowmap->num_my_elements(); ++i)
      {
        const int gid = actnoderowmap->gid(i);
        if (rownodeset.find(gid) != rownodeset.end())
        {
          const Core::Nodes::Node* standardnode = actdisc->l_row_node(i);
          newdis->add_node(standardnode->x(), gid,
              std::make_shared<Core::Nodes::Node>(gid, standardnode->x(), myrank));
        }
      }

      if (myrank == 0)
      {
        std::shared_ptr<Core::Elements::Element> constraintele =
            Core::Communication::factory(element_name, "Polynomial", nodeiter + startID, myrank);
        // set the same global node ids to the ale element
        constraintele->set_node_ids(ngid_ele.size(), ngid_ele.data());
        // add constraint element
        newdis->add_element(constraintele);
      }
      // save the connection between element and condition
      eletocond_id_[nodeiter + startID] = (*conditer)->parameters().get<int>("ConditionID");
      eletocondvecindex_[nodeiter + startID] = index;
    }
    // adjust starting ID for next condition, in this case nodeiter=ngid.size(), hence the counter
    // is larger than the ID
    // of the last element
    startID += nodeiter;

    // now care about the parallel distribution and ghosting.
    // So far every processor only knows about its nodes

    // build unique node row map
    std::vector<int> boundarynoderowvec(rownodeset.begin(), rownodeset.end());
    rownodeset.clear();
    Core::LinAlg::Map constraintnoderowmap(
        -1, boundarynoderowvec.size(), boundarynoderowvec.data(), 0, newdis->get_comm());
    boundarynoderowvec.clear();

    // build overlapping node column map
    std::vector<int> constraintnodecolvec(colnodeset.begin(), colnodeset.end());
    colnodeset.clear();
    Core::LinAlg::Map constraintnodecolmap(
        -1, constraintnodecolvec.size(), constraintnodecolvec.data(), 0, newdis->get_comm());

    constraintnodecolvec.clear();
    newdis->redistribute({constraintnoderowmap, constraintnodecolmap});
    // put new discretization into the map
    newdiscmap[(*conditer)->parameters().get<int>("ConditionID")] = newdis;
    // increase counter
    index++;
  }

  startID--;  // set counter back to ID of the last element
  return newdiscmap;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Constraints::MPConstraint3Penalty::evaluate_constraint(
    std::shared_ptr<Core::FE::Discretization> disc, Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3)
{
  if (!(disc->filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!(disc->have_dofs())) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  // define element matrices and vectors
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector1;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  const double time = params.get("total time", -1.0);
  const int numcolele = disc->num_my_col_elements();

  // get values from time integrator to scale matrices with
  double scStiff = params.get("scaleStiffEntries", 1.0);

  // loop over column elements
  for (int i = 0; i < numcolele; ++i)
  {
    // some useful data for computation
    Core::Elements::Element* actele = disc->l_col_element(i);
    int eid = actele->id();
    int condID = eletocond_id_.find(eid)->second;
    const Core::Conditions::Condition* cond = constrcond_[eletocondvecindex_.find(eid)->second];
    params.set<const Core::Conditions::Condition*>("condition", cond);

    // computation only if time is larger or equal than initialization time for constraint
    if (inittimes_.find(condID)->second <= time)
    {
      // initialize if it is the first time condition is evaluated
      if (activecons_.find(condID)->second == false)
      {
        const std::string action = params.get<std::string>("action");
        std::shared_ptr<Core::LinAlg::Vector<double>> displast =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("old disp");
        set_constr_state("displacement", *displast);
        // last converged step is used reference
        initialize(params);
        std::shared_ptr<Core::LinAlg::Vector<double>> disp =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("new disp");
        set_constr_state("displacement", *disp);
        params.set("action", action);
      }

      // get element location vector, dirichlet flags and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      actele->location_vector(*disc, lm, lmowner, lmstride);
      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      const int eledim = (int)lm.size();
      elematrix1.shape(eledim, eledim);
      elevector1.size(eledim);
      elevector3.size(1);
      params.set("ConditionID", eid);

      // call the element evaluate method
      int err = actele->evaluate(
          params, *disc, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err)
        FOUR_C_THROW("Proc {}: Element {} returned err={}",
            Core::Communication::my_mpi_rank(disc->get_comm()), eid, err);

      // loadcurve business
      const auto curvenum = cond->parameters().get<std::optional<int>>("curve");
      double curvefac = 1.0;
      if (curvenum.has_value() && curvenum.value() > 0 && time >= 0.0)
      {
        // function_by_id takes a zero-based index
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curvenum.value())
                       .evaluate(time);
      }

      double diff = (curvefac * (*initerror_).local_values_as_span()[eid] -
                     (*acterror_).local_values_as_span()[eid]);
      elematrix1.scale(diff);
      for (int i = 0; i < eledim; i++)
        for (int j = 0; j < eledim; j++) elematrix1(i, j) += elevector1(i) * elevector1(j);
      elematrix1.scale(2 * scStiff * penalties_[condID]);

      systemmatrix1->assemble(eid, lmstride, elematrix1, lm, lmowner);
      elevector1.scale(2. * penalties_[condID] * diff);
      Core::LinAlg::assemble(*systemvector1, elevector1, lm, lmowner);
    }
  }
}  // end of evaluate_condition

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Constraints::MPConstraint3Penalty::evaluate_error(Core::FE::Discretization& disc,
    Teuchos::ParameterList& params, Core::LinAlg::Vector<double>& systemvector, bool init)
{
  if (!(disc.filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!(disc.have_dofs())) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  // define element matrices and vectors
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector1;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  // loop over column elements
  const double time = params.get("total time", -1.0);
  const int numcolele = disc.num_my_col_elements();
  for (int i = 0; i < numcolele; ++i)
  {
    // some useful data for computation
    Core::Elements::Element* actele = disc.l_col_element(i);
    int eid = actele->id();
    int condID = eletocond_id_.find(eid)->second;
    const Core::Conditions::Condition* cond = constrcond_[eletocondvecindex_.find(eid)->second];
    params.set<const Core::Conditions::Condition*>("condition", cond);

    // get element location vector, dirichlet flags and ownerships
    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    actele->location_vector(disc, lm, lmowner, lmstride);
    elevector3.size(1);
    params.set("ConditionID", eid);

    if (absconstraint_.find(condID)->second && init)
    {
      elevector3[0] = cond->parameters().get<double>("amplitude");
    }
    else
    {
      // call the element evaluate method
      int err = actele->evaluate(
          params, disc, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err)
        FOUR_C_THROW("Proc {}: Element {} returned err={}",
            Core::Communication::my_mpi_rank(disc.get_comm()), eid, err);
    }

    // assembly
    std::vector<int> constrlm;
    std::vector<int> constrowner;
    constrlm.push_back(eid);
    constrowner.push_back(actele->owner());
    Core::LinAlg::assemble(systemvector, elevector3, constrlm, constrowner);

    activecons_.find(condID)->second = true;

    if (Core::Communication::my_mpi_rank(actdisc_->get_comm()) == 0 &&
        (!(activecons_.find(condID)->second)))
    {
      std::cout << "Encountered a new active penalty mp condition (Id = " << condID
                << ")  at time t = " << time << std::endl;
    }
  }

  Core::LinAlg::Vector<double> acterrdist(*errormap_);
  acterrdist.export_to(systemvector, *errorexport_, Add);
  systemvector.import(acterrdist, *errorimport_, Insert);
  return;
}  // end of evaluate_error

FOUR_C_NAMESPACE_CLOSE
