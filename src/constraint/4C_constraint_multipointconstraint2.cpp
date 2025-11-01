// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_multipointconstraint2.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_transparent.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_utils_function_of_time.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
Constraints::MPConstraint2::MPConstraint2(std::shared_ptr<Core::FE::Discretization> discr,
    const std::string& conditionname, int& minID, int& maxID)
    : MPConstraint(discr, conditionname, minID, maxID)
{
  if (constrcond_.size())
  {
    int dummy = 0;
    // create constraint discretization and store it with label 0, within the map
    constraintdis_ = create_discretization_from_condition(
        actdisc_, constrcond_, "ConstrDisc", "CONSTRELE2", dummy);
    std::shared_ptr<Core::LinAlg::Map> newcolnodemap =
        Core::Rebalance::compute_node_col_map(*actdisc_, *constraintdis_.find(0)->second);
    actdisc_->redistribute({*actdisc_->node_row_map(), *newcolnodemap});
    std::shared_ptr<Core::DOFSets::DofSet> newdofset =
        std::make_shared<Core::DOFSets::TransparentDofSet>(actdisc_);
    (constraintdis_.find(0)->second)->replace_dof_set(newdofset);
    newdofset = nullptr;
    (constraintdis_.find(0)->second)->fill_complete();
  }
}

/*------------------------------------------------------------------------*
|(public)                                                       tk 08/08  |
|Initialization routine activates conditions (restart)                    |
*------------------------------------------------------------------------*/
void Constraints::MPConstraint2::initialize(const double& time)
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
void Constraints::MPConstraint2::initialize(
    Teuchos::ParameterList& params, std::shared_ptr<Core::LinAlg::Vector<double>> systemvector)
{
  const double time = params.get("total time", -1.0);
  // in case init is set to true we want to set systemvector1 to the amplitudes defined
  // in the input file
  // allocate vectors for amplitudes and IDs


  std::vector<double> amplit(constrcond_.size());
  std::vector<int> IDs(constrcond_.size());
  // read data of the input files
  for (unsigned int i = 0; i < constrcond_.size(); i++)
  {
    const Core::Conditions::Condition& cond = *(constrcond_[i]);
    int condID = cond.parameters().get<int>("ConditionID");
    if (inittimes_.find(condID)->second <= time)
    {
      const int MPCcondID = constrcond_[i]->parameters().get<int>("ConditionID");
      amplit[i] = constrcond_[i]->parameters().get<double>("amplitude");
      const int mid = params.get("OffsetID", 0);
      IDs[i] = MPCcondID - mid;
      // remember next time, that this condition is already initialized, i.e. active
      activecons_.find(condID)->second = true;
      if (Core::Communication::my_mpi_rank(actdisc_->get_comm()) == 0)
      {
        std::cout << "Encountered a new active condition (Id = " << condID
                  << ")  at time t = " << time << std::endl;
      }
    }
  }
  // replace systemvector by the given amplitude values
  // systemvector is supposed to be the vector with initial values of the constraints
  if (Core::Communication::my_mpi_rank(actdisc_->get_comm()) == 0)
  {
    systemvector->replace_global_values(amplit.size(), amplit.data(), IDs.data());
  }
}

/*-----------------------------------------------------------------------*
|(public)                                                        tk 07/08|
|Evaluate Constraints, choose the right action based on type             |
*-----------------------------------------------------------------------*/
void Constraints::MPConstraint2::evaluate(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3)
{
  switch (type())
  {
    case mpcnodeonline2d:
      params.set("action", "calc_MPC_stiff");
      break;
    case none:
      return;
    default:
      FOUR_C_THROW("Constraint/monitor is not an multi point constraint!");
  }
  evaluate_constraint(constraintdis_.find(0)->second, params, systemmatrix1, systemmatrix2,
      systemvector1, systemvector2, systemvector3);
}

/*------------------------------------------------------------------------*
 |(private)                                                   tk 04/08    |
 |subroutine creating a new discretization containing constraint elements |
 *------------------------------------------------------------------------*/
std::map<int, std::shared_ptr<Core::FE::Discretization>>
Constraints::MPConstraint2::create_discretization_from_condition(
    std::shared_ptr<Core::FE::Discretization> actdisc,
    std::span<const Core::Conditions::Condition*> constrcondvec, const std::string& discret_name,
    const std::string& element_name, int& startID)
{
  MPI_Comm com(actdisc->get_comm());

  std::shared_ptr<Core::FE::Discretization> newdis =
      std::make_shared<Core::FE::Discretization>(discret_name, com, actdisc->n_dim());

  if (!actdisc->filled())
  {
    actdisc->fill_complete();
  }

  const int myrank = Core::Communication::my_mpi_rank(newdis->get_comm());

  if (constrcondvec.size() == 0)
    FOUR_C_THROW(
        "number of multi point constraint conditions = 0 --> cannot create constraint "
        "discretization");

  std::set<int> rownodeset;
  std::set<int> colnodeset;
  const Core::LinAlg::Map* actnoderowmap = actdisc->node_row_map();

  // Loop all conditions in constrcondvec
  for (unsigned int j = 0; j < constrcondvec.size(); j++)
  {
    std::vector<int> ngid = *(constrcondvec[j]->get_nodes());
    const int numnodes = ngid.size();
    // We sort the global node ids according to the definition of the boundary condition
    reorder_constraint_nodes(ngid, constrcondvec[j]);

    remove_copy_if(ngid.data(), ngid.data() + numnodes, inserter(rownodeset, rownodeset.begin()),
        std::not_fn(Core::Conditions::MyGID(actnoderowmap)));
    // copy node ids specified in condition to colnodeset
    copy(ngid.data(), ngid.data() + numnodes, inserter(colnodeset, colnodeset.begin()));

    // construct boundary nodes, which use the same global id as the cutter nodes
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
          Core::Communication::factory(element_name, "Polynomial", j, myrank);
      // set the same global node ids to the ale element
      constraintele->set_node_ids(ngid.size(), ngid.data());

      // add constraint element
      newdis->add_element(constraintele);
    }
    // now care about the parallel distribution and ghosting.
    // So far every processor only knows about its nodes
  }

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

  std::map<int, std::shared_ptr<Core::FE::Discretization>> newdismap;
  newdismap[startID] = newdis;
  return newdismap;
}

/*----------------------------------------------------------------------*
 |(private)                                                 tk 04/08    |
 |reorder MPC nodes based on condition input                            |
 *----------------------------------------------------------------------*/
void Constraints::MPConstraint2::reorder_constraint_nodes(
    std::vector<int>& nodeids, const Core::Conditions::Condition* cond)
{
  // get this condition's nodes
  std::vector<int> temp = nodeids;
  if (nodeids.size() == 3)
  {
    nodeids[0] = temp[cond->parameters().get<int>("constrNode1") - 1];
    nodeids[1] = temp[cond->parameters().get<int>("constrNode2") - 1];
    nodeids[2] = temp[cond->parameters().get<int>("constrNode3") - 1];
  }
  else
  {
    FOUR_C_THROW("strange number of nodes for an MPC! Should be 3 in 2D.");
  }
}

/*-----------------------------------------------------------------------*
 |(private)                                                     tk 07/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void Constraints::MPConstraint2::evaluate_constraint(std::shared_ptr<Core::FE::Discretization> disc,
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
  bool assemblevec2 = systemvector2 != nullptr;
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
    Core::Elements::Element* actele = disc->l_col_element(i);
    const Core::Conditions::Condition& cond = *(constrcond_[actele->id()]);
    int condID = cond.parameters().get<int>("ConditionID");

    // computation only if time is larger or equal than initialization time for constraint
    if (inittimes_.find(condID)->second <= time)
    {
      // initialize if it is the first time condition is evaluated
      if (activecons_.find(condID)->second == false)
      {
        const std::string action = params.get<std::string>("action");
        initialize(params, systemvector2);
        params.set("action", action);
      }

      // define global and local index of this bc in redundant vectors
      const int offsetID = params.get<int>("OffsetID");
      int gindex = condID - offsetID;
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
      if (assemblemat1) elematrix1.shape(eledim, eledim);
      if (assemblemat2) elematrix2.shape(eledim, eledim);
      if (assemblevec1) elevector1.size(eledim);
      if (assemblevec2) elevector2.size(eledim);
      if (assemblevec3) elevector3.size(1);  // elevector3 always contains a scalar

      params.set("ConditionID", condID);
      params.set<const Core::Conditions::Condition*>("condition", &cond);
      // call the element evaluate method
      int err = actele->evaluate(
          params, *disc, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err)
        FOUR_C_THROW("Proc {}: Element {} returned err={}",
            Core::Communication::my_mpi_rank(disc->get_comm()), actele->id(), err);

      int eid = actele->id();

      // Assembly
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

      // Load curve business
      const auto curvenum = cond.parameters().get<std::optional<int>>("curve");
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

FOUR_C_NAMESPACE_CLOSE
