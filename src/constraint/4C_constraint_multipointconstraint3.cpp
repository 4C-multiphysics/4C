// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_multipointconstraint3.hpp"

#include "4C_constraint_element3.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_transparent.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_utils_function_of_time.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
Constraints::MPConstraint3::MPConstraint3(std::shared_ptr<Core::FE::Discretization> discr,
    const std::string& conditionname, int& offsetID, int& maxID)
    : MPConstraint(discr, conditionname)
{
  if (constrcond_.size())
  {
    maxID++;
    // control the constraint by absolute or relative values
    for (auto* cond : constrcond_)
    {
      const int condID = cond->parameters().get<int>("ConditionID");
      if (offsetID > maxID) offsetID = maxID;
      const std::string type = cond->parameters().get<std::string>("control");
      if (type == "abs")
        absconstraint_[condID] = true;
      else
        absconstraint_[condID] = false;
    }

    constraintdis_ = create_discretization_from_condition(
        actdisc_, constrcond_, "ConstrDisc", "CONSTRELE3", maxID);

    std::map<int, std::shared_ptr<Core::FE::Discretization>>::iterator discriter;
    for (discriter = constraintdis_.begin(); discriter != constraintdis_.end(); discriter++)
    {
      // ReplaceNumDof(actdisc_,discriter->second);
      std::shared_ptr<Core::LinAlg::Map> newcolnodemap =
          Core::Rebalance::compute_node_col_map(*actdisc_, *discriter->second);
      actdisc_->redistribute({*actdisc_->node_row_map(), *newcolnodemap});
      std::shared_ptr<Core::DOFSets::DofSet> newdofset =
          std::make_shared<Core::DOFSets::TransparentDofSet>(actdisc_);
      (discriter->second)->replace_dof_set(newdofset);
      newdofset = nullptr;
      (discriter->second)->fill_complete();
    }
  }
}

/*------------------------------------------------------------------------*
|(public)                                                       tk 08/08  |
|Initialization routine activates conditions (restart)                    |
*------------------------------------------------------------------------*/
void Constraints::MPConstraint3::initialize(const double& time)
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
|(public)                                                        tk 07/08|
|Evaluate Constraints, choose the right action based on type             |
*-----------------------------------------------------------------------*/
void Constraints::MPConstraint3::initialize(
    Teuchos::ParameterList& params, std::shared_ptr<Core::LinAlg::Vector<double>> systemvector)
{
  const double time = params.get("total time", -1.0);
  const int mid = params.get("OffsetID", 0);

  // Collect only valid abs entries
  std::vector<double> vals;
  std::vector<int> gids;
  vals.reserve(constrcond_.size());
  gids.reserve(constrcond_.size());

  const int myrank = Core::Communication::my_mpi_rank(actdisc_->get_comm());

  for (std::size_t i = 0; i < constrcond_.size(); ++i)
  {
    const auto& cond = *(constrcond_[i]);
    const int condID = cond.parameters().get<int>("ConditionID");

    // activate at/after init time and only once
    if (inittimes_.find(condID)->second <= time && !activecons_.find(condID)->second)
    {
      if (absconstraint_.find(condID)->second)
      {
        // absolute value - set a single entry (owner only)
        const int gid = condID - mid;
        const Core::LinAlg::Map& ownedMap = systemvector->get_map();
        if (ownedMap.my_gid(gid))
        {
          const double MPCampl = constrcond_[i]->parameters().get<double>("amplitude");
          vals.push_back(MPCampl);
          gids.push_back(gid);
        }
      }
      else
      {
        switch (type())
        {
          case mpcnormalcomp3d:
          case mpcnodeonplane3d:
            params.set("action", "calc_MPC_state");
            break;
          case none:
            return;
          default:
            FOUR_C_THROW("Constraint is not an multi point constraint!");
        }

        // Let the owner(s) assemble inside initialize_constraint (they should gate by ownership)
        initialize_constraint(*constraintdis_.find(condID)->second, params, *systemvector);
      }

      // mark active locally
      activecons_.find(condID)->second = true;

      if (myrank == 0)
      {
        std::cout << "Encountered a new active condition (Id = " << condID
                  << ")  at time t = " << time << std::endl;
      }
    }
  }

  // Apply absolute amplitudes only if we actually collected some
  // and distribute only from proc 0
  if (myrank == 0 and !vals.empty())
  {
    systemvector->sum_into_global_values(static_cast<int>(gids.size()), vals.data(), gids.data());
  }
}

/*-----------------------------------------------------------------------*
|(public)                                                        tk 07/08|
|Evaluate Constraints, choose the right action based on type             |
*-----------------------------------------------------------------------*/
void Constraints::MPConstraint3::evaluate(Teuchos::ParameterList& params,
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
      params.set("action", "calc_MPC_stiff");
      break;
    case none:
      return;
    default:
      FOUR_C_THROW("Constraint/monitor is not an multi point constraint!");
  }
  std::map<int, std::shared_ptr<Core::FE::Discretization>>::iterator discriter;
  for (discriter = constraintdis_.begin(); discriter != constraintdis_.end(); discriter++)
    evaluate_constraint(discriter->second, params, systemmatrix1, systemmatrix2, systemvector1,
        systemvector2, systemvector3);

  return;
}

/*------------------------------------------------------------------------*
 |(private)                                                   tk 04/08    |
 |subroutine creating a new discretization containing constraint elements |
 *------------------------------------------------------------------------*/
std::map<int, std::shared_ptr<Core::FE::Discretization>>
Constraints::MPConstraint3::create_discretization_from_condition(
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
      ngid.erase(ngid.begin() + ((*nsit) - counter));
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



/*-----------------------------------------------------------------------*
 |(private)                                                     tk 07/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void Constraints::MPConstraint3::evaluate_constraint(std::shared_ptr<Core::FE::Discretization> disc,
    Teuchos::ParameterList& params, std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3)
{
  if (!(disc->filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!(disc->have_dofs())) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  // see what we have for input
  bool assemblemat1 = systemmatrix1 != nullptr;
  bool assemblemat2 = systemmatrix2 != nullptr;
  bool assemblevec1 = systemvector1 != nullptr;
  bool assemblevec3 = systemvector3 != nullptr;

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
  double scConMat = params.get("scaleConstrMat", 1.0);

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
        initialize(params, systemvector2);
        std::shared_ptr<Core::LinAlg::Vector<double>> disp =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("new disp");
        set_constr_state("displacement", *disp);
        params.set("action", action);
      }

      // define global and local index of this bc in redundant vectors
      const int offsetID = params.get<int>("OffsetID");
      int gindex = eid - offsetID;
      const int lindex = (systemvector3->get_map()).lid(gindex);

      // Get the current lagrange multiplier value for this condition
      const std::shared_ptr<Core::LinAlg::Vector<double>> lagramul =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("LagrMultVector");
      const double lagraval = (*lagramul).local_values_as_span()[lindex];

      // get element location vector, dirichlet flags and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      actele->location_vector(*disc, lm, lmowner, lmstride);
      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      const int eledim = (int)lm.size();
      elematrix1.shape(eledim, eledim);
      elematrix2.shape(eledim, eledim);
      elevector1.size(eledim);
      elevector2.size(eledim);
      elevector3.size(1);
      params.set("ConditionID", eid);

      // call the element evaluate method
      int err = actele->evaluate(
          params, *disc, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err)
        FOUR_C_THROW("Proc {}: Element {} returned err={}",
            Core::Communication::my_mpi_rank(disc->get_comm()), eid, err);

      if (assemblemat1)
      {
        // scale with time integrator dependent value
        elematrix1.scale(scStiff * lagraval);
        systemmatrix1->assemble(eid, lmstride, elematrix1, lm, lmowner);
      }
      if (assemblemat2)
      {
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
        constrowner.push_back(actele->owner());
        Core::LinAlg::assemble(*systemvector3, elevector3, constrlm, constrowner);
      }

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

      std::shared_ptr<Core::LinAlg::Vector<double>> timefact =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("vector curve factors");
      timefact->replace_global_values(1, &curvefac, &gindex);
    }
  }
}  // end of evaluate_condition

/*-----------------------------------------------------------------------*
 |(private)                                                     tk 07/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void Constraints::MPConstraint3::initialize_constraint(Core::FE::Discretization& disc,
    Teuchos::ParameterList& params, Core::LinAlg::Vector<double>& systemvector)
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
    // get dimension of element matrices and vectors
    // Reshape element matrices and vectors and init to zero
    const int eledim = (int)lm.size();
    elematrix1.shape(eledim, eledim);
    elematrix2.shape(eledim, eledim);
    elevector1.size(eledim);
    elevector2.size(eledim);
    elevector3.size(1);
    // call the element evaluate method
    int err = actele->evaluate(
        params, disc, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
    if (err)
      FOUR_C_THROW("Proc {}: Element {} returned err={}",
          Core::Communication::my_mpi_rank(disc.get_comm()), actele->id(), err);

    // assembly
    std::vector<int> constrlm;
    std::vector<int> constrowner;
    int offsetID = params.get<int>("OffsetID");
    constrlm.push_back(eid - offsetID);
    constrowner.push_back(actele->owner());
    Core::LinAlg::assemble(systemvector, elevector3, constrlm, constrowner);

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

    // Get ConditionID of current condition if defined and write value in parameterlist
    constexpr unsigned character_length = 30;
    char factorname[character_length];
    snprintf(factorname, character_length, "LoadCurveFactor %d", condID);
    params.set(factorname, curvefac);
  }
}  // end of initialize_constraint

FOUR_C_NAMESPACE_CLOSE
