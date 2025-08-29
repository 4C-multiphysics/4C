// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elements_paramsinterface.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  evaluate (public)                                        mwgee 12/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3)
{
  Core::FE::AssembleStrategy strategy(
      0, 0, systemmatrix1, systemmatrix2, systemvector1, systemvector2, systemvector3);
  evaluate(params, strategy);
}


void Core::FE::Discretization::evaluate(
    Teuchos::ParameterList& params, Core::FE::AssembleStrategy& strategy)
{
  // Call the Evaluate method for the specific element
  evaluate(params, strategy,
      [&](Core::Elements::Element& ele, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
          Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3)
      {
        const int err =
            ele.evaluate(params, *this, la, strategy.elematrix1(), strategy.elematrix2(),
                strategy.elevector1(), strategy.elevector2(), strategy.elevector3());
        if (err)
          FOUR_C_THROW("Proc {}: Element {} returned err={}",
              Core::Communication::my_mpi_rank(get_comm()), ele.id(), err);
      });
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate(Teuchos::ParameterList& params,
    Core::FE::AssembleStrategy& strategy,
    const std::function<void(Core::Elements::Element&, Core::Elements::LocationArray&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseVector&, Core::LinAlg::SerialDenseVector&,
        Core::LinAlg::SerialDenseVector&)>& element_action)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::FE::Discretization::Evaluate");

  if (!filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  int row = strategy.first_dof_set();
  int col = strategy.second_dof_set();

  Core::Elements::LocationArray la(dofsets_.size());

  // loop over column elements
  for (auto* actele : my_col_element_range())
  {
    // get element location vector, dirichlet flags and ownerships
    actele->location_vector(*this, la);

    // get dimension of element matrices and vectors
    // Reshape element matrices and vectors and init to zero
    strategy.clear_element_storage(la[row].size(), la[col].size());

    // call the element evaluate method
    element_action(*actele, la, strategy.elematrix1(), strategy.elematrix2(), strategy.elevector1(),
        strategy.elevector2(), strategy.elevector3());

    int eid = actele->id();
    strategy.assemble_matrix1(eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
    strategy.assemble_matrix2(eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
    strategy.assemble_vector1(la[row].lm_, la[row].lmowner_);
    strategy.assemble_vector2(la[row].lm_, la[row].lmowner_);
    strategy.assemble_vector3(la[row].lm_, la[row].lmowner_);
  }
}


/*----------------------------------------------------------------------*
 |  evaluate (public)                                        u.kue 01/08|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector)
{
  evaluate(params, systemmatrix, nullptr, systemvector, nullptr, nullptr);
}

void Core::FE::Discretization::evaluate(
    const std::function<void(Core::Elements::Element&)>& element_action)
{
  // test only for Filled()!Dof information is not required
  if (!filled()) FOUR_C_THROW("fill_complete() was not called");

  for (auto* actele : my_col_element_range())
  {
    // call the element evaluate method
    element_action(*actele);
  }
}


/*----------------------------------------------------------------------*
 |  evaluate (public)                                        a.ger 03/09|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate(Teuchos::ParameterList& params)
{
  // define empty element matrices and vectors
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector1;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  Core::Elements::LocationArray la(dofsets_.size());

  evaluate(
      [&](Core::Elements::Element& ele)
      {
        const int err = ele.evaluate(
            params, *this, la, elematrix1, elematrix2, elevector1, elevector2, elevector3);
        if (err)
          FOUR_C_THROW("Proc {}: Element {} returned err={}",
              Core::Communication::my_mpi_rank(get_comm()), ele.id(), err);
      });
}


/*----------------------------------------------------------------------*
 |  evaluate Neumann conditions (public)                     mwgee 12/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate_neumann(Teuchos::ParameterList& params,
    Core::LinAlg::Vector<double>& systemvector, Core::LinAlg::SparseOperator* systemmatrix)
{
  if (!filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  bool assemblemat = (systemmatrix != nullptr);

  // get the current time
  double time = params.get("total time", -1.0);

  if (params.isParameter("interface"))
  {
    time =
        params.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface")->get_total_time();
  }

  //--------------------------------------------------------
  // loop through Point Neumann conditions and evaluate them
  //--------------------------------------------------------
  for (const auto& [name, cond] : condition_)
  {
    if (name != (std::string) "PointNeumann") continue;
    if (assemblemat && !Core::Communication::my_mpi_rank(systemvector.get_comm()))
    {
      std::cout << "WARNING: System matrix handed in but no linearization of "
                   "PointNeumann conditions implemented. Did you set the LOADLIN-flag "
                   "accidentally?\n";
    }
    const std::vector<int>* nodeids = cond->get_nodes();
    if (!nodeids) FOUR_C_THROW("PointNeumann condition does not have nodal cloud");
    const auto& tmp_funct = cond->parameters().get<std::vector<std::optional<int>>>("FUNCT");
    const auto& onoff = cond->parameters().get<std::vector<int>>("ONOFF");
    const auto& val = cond->parameters().get<std::vector<double>>("VAL");

    for (const int nodeid : *nodeids)
    {
      // do only nodes in my row map
      if (!node_row_map()->my_gid(nodeid)) continue;
      Core::Nodes::Node* actnode = g_node(nodeid);
      if (!actnode) FOUR_C_THROW("Cannot find global node {}", nodeid);
      // call explicitly the main dofset, nodeid.e. the first column
      std::vector<int> dofs = dof(0, actnode);
      const unsigned numdf = dofs.size();
      for (unsigned j = 0; j < numdf; ++j)
      {
        if (onoff[j] == 0) continue;
        const int gid = dofs[j];
        double value = val[j];

        const double functfac = std::invoke(
            [&]()
            {
              if (tmp_funct[j].has_value() && tmp_funct[j].value() > 0)
              {
                const auto* function_manager =
                    params.isParameter("interface")
                        ? params.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface")
                              ->get_function_manager()
                        : params.get<const Core::Utils::FunctionManager*>("function_manager");

                return function_manager
                    ->function_by_id<Core::Utils::FunctionOfTime>((tmp_funct[j]).value())
                    .evaluate(time);
              }
              else
                return 1.0;
            });

        value *= functfac;
        const int lid = systemvector.get_map().lid(gid);
        if (lid < 0) FOUR_C_THROW("Global id {} not on this proc in system vector", gid);
        systemvector.get_values()[lid] += value;
      }
    }
  }

  //--------------------------------------------------------
  // loop through line/surface/volume Neumann BCs and evaluate them
  //--------------------------------------------------------
  for (const auto& [name, cond] : condition_)
  {
    if (name == (std::string) "LineNeumann" || name == (std::string) "SurfaceNeumann" ||
        name == (std::string) "VolumeNeumann")
    {
      std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = cond->geometry();
      Core::LinAlg::SerialDenseVector elevector;
      Core::LinAlg::SerialDenseMatrix elematrix;
      for (const auto& [_, ele] : geom)
      {
        // get element location vector, dirichlet flags and ownerships
        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;
        ele->location_vector(*this, lm, lmowner, lmstride);
        elevector.size((int)lm.size());
        if (!assemblemat)
        {
          ele->evaluate_neumann(params, *this, *cond, lm, elevector);
          Core::LinAlg::assemble(systemvector, elevector, lm, lmowner);
        }
        else
        {
          const int size = lm.size();
          if (elematrix.numRows() != size)
            elematrix.shape(size, size);
          else
            elematrix.putScalar(0.0);
          ele->evaluate_neumann(params, *this, *cond, lm, elevector, &elematrix);
          Core::LinAlg::assemble(systemvector, elevector, lm, lmowner);
          systemmatrix->assemble(ele->id(), lmstride, elematrix, lm, lmowner);
        }
      }
    }
  }

  //--------------------------------------------------------
  // loop through Point Moment EB conditions and evaluate them
  //--------------------------------------------------------
  for (const auto& [name, cond] : condition_)
  {
    if (name != (std::string) "PointNeumannEB") continue;
    const std::vector<int>* nodeids = cond->get_nodes();
    if (!nodeids) FOUR_C_THROW("Point Moment condition does not have nodal cloud");

    for (const int nodeid : *nodeids)
    {
      // create matrices for fext and fextlin
      Core::LinAlg::SerialDenseVector elevector;
      Core::LinAlg::SerialDenseMatrix elematrix;

      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;

      // do only nodes in my row map
      if (!node_row_map()->my_gid(nodeid)) continue;

      // get global node
      Core::Nodes::Node* actnode = g_node(nodeid);
      if (!actnode) FOUR_C_THROW("Cannot find global node {}", nodeid);

      // get elements attached to global node
      Core::Elements::Element** curreleptr = actnode->elements();

      // find element from pointer
      // please note, that external force will be applied to the first element [0] attached to a
      // node this needs to be done, otherwise it will be applied several times on several elements.
      Core::Elements::Element* currele = curreleptr[0];

      // get information from location
      currele->location_vector(*this, lm, lmowner, lmstride);
      const int size = (int)lm.size();
      elevector.size(size);

      // evaluate linearized point moment conditions and assemble f_ext and f_ext_lin into global
      // matrix
      //-----if the stiffness matrix was given in-------
      if (assemblemat)
      {
        // resize f_ext_lin matrix
        if (elematrix.numRows() != size)
          elematrix.shape(size, size);
        else
          elematrix.putScalar(0.0);
        // evaluate linearized point moment conditions and assemble matrices
        currele->evaluate_neumann(params, *this, *cond, lm, elevector, &elematrix);
        systemmatrix->assemble(currele->id(), lmstride, elematrix, lm, lmowner);
      }
      //-----if no stiffness matrix was given in-------
      else
        currele->evaluate_neumann(params, *this, *cond, lm, elevector);
      Core::LinAlg::assemble(systemvector, elevector, lm, lmowner);
    }
  }
}


/*----------------------------------------------------------------------*
 |  evaluate Dirichlet conditions (public)                  rauch 06/16 |
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate_dirichlet(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvectord,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvectordd,
    std::shared_ptr<Core::LinAlg::Vector<int>> toggle,
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmapextractor) const
{
  Core::FE::evaluate_dirichlet(
      *this, params, systemvector, systemvectord, systemvectordd, toggle, dbcmapextractor);
}


/*----------------------------------------------------------------------*
 |  evaluate a condition (public)                               tk 02/08|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate_condition(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3, const std::string& condstring,
    const int condid)
{
  Core::FE::AssembleStrategy strategy(
      0, 0, systemmatrix1, systemmatrix2, systemvector1, systemvector2, systemvector3);
  evaluate_condition(params, strategy, condstring, condid);
}  // end of Core::FE::Discretization::evaluate_condition


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate_condition(Teuchos::ParameterList& params,
    Core::FE::AssembleStrategy& strategy, const std::string& condstring, const int condid)
{
  if (!filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  int row = strategy.first_dof_set();
  int col = strategy.second_dof_set();

  Core::Elements::LocationArray la(dofsets_.size());

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (const auto& [name, cond] : condition_)
  {
    if (name == condstring)
    {
      if (condid == -1 || condid == cond->parameters().get<int>("ConditionID"))
      {
        std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = cond->geometry();
        // if (geom.empty()) FOUR_C_THROW("evaluation of condition with empty geometry");
        // no check for empty geometry here since in parallel computations
        // can exist processors which do not own a portion of the elements belonging
        // to the condition geometry

        // Get ConditionID of current condition if defined and write value in parameter list
        const auto* condID = cond->parameters().get_if<int>("ConditionID");
        if (condID)
        {
          params.set("ConditionID", *condID);
        }
        params.set<const Core::Conditions::Condition*>("condition", cond.get());

        for (const auto& [_, ele] : geom)
        {
          // get element location vector and ownerships
          // the LocationVector method will return the location vector
          // of the dofs this condition is meant to assemble into.
          // These dofs do not need to be the same as the dofs of the element
          // (this is the standard case, though). Special boundary conditions,
          // like weak Dirichlet conditions, assemble into the dofs of the parent element.
          ele->location_vector(*this, la, condstring, params);

          // get dimension of element matrices and vectors
          // Reshape element matrices and vectors and initialize to zero
          strategy.clear_element_storage(la[row].size(), la[col].size());

          // call the element specific evaluate method
          int err = ele->evaluate(params, *this, la, strategy.elematrix1(), strategy.elematrix2(),
              strategy.elevector1(), strategy.elevector2(), strategy.elevector3());
          if (err) FOUR_C_THROW("error while evaluating elements");

          // assembly
          /* If BlockMatrixes are used, the decision which assemble strategy is
           * used, is based on the element id. As this id is compared to a list
           * of conditioned volume elements, always the volume element id should
           * be given to the Assembling! (comment: eid is not used by
           * sysmat.assemble(...,eid,...))*/
          int eid;
          if (auto* faceele = dynamic_cast<Core::Elements::FaceElement*>(ele.get()))
            eid = faceele->parent_element()->id();
          else
            eid = ele->id();

          strategy.assemble_matrix1(
              eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
          strategy.assemble_matrix2(
              eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
          strategy.assemble_vector1(la[row].lm_, la[row].lmowner_);
          strategy.assemble_vector2(la[row].lm_, la[row].lmowner_);
          strategy.assemble_vector3(la[row].lm_, la[row].lmowner_);
        }
      }
    }
  }
}

void Core::FE::Discretization::evaluate_condition(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector, const std::string& condstring,
    const int condid)
{
  evaluate_condition(params, nullptr, nullptr, systemvector, nullptr, nullptr, condstring, condid);
}


void Core::FE::Discretization::evaluate_condition(
    Teuchos::ParameterList& params, const std::string& condstring, const int condid)
{
  evaluate_condition(params, nullptr, nullptr, nullptr, nullptr, nullptr, condstring, condid);
}


/*----------------------------------------------------------------------*
 |  evaluate/assemble scalars across elements (public)       bborn 08/08|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate_scalars(
    Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector& scalars)
{
  if (!filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  // number of scalars
  const int numscalars = scalars.length();
  if (numscalars <= 0) FOUR_C_THROW("scalars vector of interest has size <=0");
  // intermediate sum of each scalar on each processor
  Core::LinAlg::SerialDenseVector cpuscalars(numscalars);

  // define element matrices and vectors
  // -- which are empty and unused, just to satisfy element evaluate()
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  // loop over _row_ elements
  for (const auto& actele : my_row_element_range())
  {
    // get element location vector
    Core::Elements::LocationArray la(dofsets_.size());
    actele->location_vector(*this, la);

    // define element vector
    Core::LinAlg::SerialDenseVector elescalars(numscalars);

    // call the element evaluate method
    {
      int err = actele->evaluate(
          params, *this, la, elematrix1, elematrix2, elescalars, elevector2, elevector3);
      if (err)
        FOUR_C_THROW("Proc {}: Element {} returned err={}",
            Core::Communication::my_mpi_rank(get_comm()), actele->id(), err);
    }

    // sum up (on each processor)
    cpuscalars += elescalars;
  }

  // reduce
  for (int i = 0; i < numscalars; ++i) (scalars)(i) = 0.0;
  Core::Communication::sum_all(cpuscalars.values(), scalars.values(), numscalars, get_comm());
}  // Core::FE::Discretization::EvaluateScalars


/*-----------------------------------------------------------------------------*
 | evaluate/assemble scalars across conditioned elements (public)   fang 02/15 |
 *-----------------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate_scalars(
    Teuchos::ParameterList& params,  //! (in) parameter list
    Core::LinAlg::SerialDenseVector&
        scalars,                    //! (out) result vector for scalar quantities to be computed
    const std::string& condstring,  //! (in) name of condition to be evaluated
    const int condid                //! (in) condition ID (optional)
)
{
  // safety checks
  if (!filled()) FOUR_C_THROW("fill_complete() has not been called on discretization!");
  if (!have_dofs())
    FOUR_C_THROW("assign_degrees_of_freedom() has not been called on discretization!");

  // determine number of scalar quantities to be computed
  const int numscalars = scalars.length();

  // safety check
  if (numscalars <= 0)
    FOUR_C_THROW("Result vector for EvaluateScalars routine must have positive length!");

  // initialize vector for intermediate results of scalar quantities on single processor
  Core::LinAlg::SerialDenseVector cpuscalars(numscalars);

  // define empty dummy element matrices and residuals
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  // loop over all conditions on discretization
  for (const auto& [name, condition] : condition_)
  {
    // consider only conditions with specified label
    if (name == condstring)
    {
      // additional filtering by condition ID if explicitly provided
      if (condid == -1 or condid == condition->parameters().get<int>("ConditionID"))
      {
        // extract geometry map of current condition
        std::map<int, std::shared_ptr<Core::Elements::Element>>& geometry = condition->geometry();

        // add condition to parameter list for elements
        params.set<const Core::Conditions::Condition*>("condition", condition.get());

        // loop over all elements associated with current condition
        for (auto& [_, element] : geometry)
        {
          // consider only unghosted elements for evaluation
          if (element->owner() == Core::Communication::my_mpi_rank(get_comm()))
          {
            // construct location vector for current element
            Core::Elements::LocationArray la(dofsets_.size());
            element->location_vector(*this, la);

            // initialize result vector for current element
            Core::LinAlg::SerialDenseVector elescalars(numscalars);

            // call element evaluation routine
            int error = element->evaluate(
                params, *this, la, elematrix1, elematrix2, elescalars, elevector2, elevector3);

            // safety check
            if (error)
            {
              FOUR_C_THROW(
                  "Element evaluation failed for element {} on processor {} with error code {}!",
                  element->id(), Core::Communication::my_mpi_rank(get_comm()), error);
            }

            // update result vector on single processor
            cpuscalars += elescalars;
          }  // if(element.Owner() == Core::Communication::my_mpi_rank(Comm()))
        }  // loop over elements
      }  // if(condid == -1 or condid == condition.get<int>("ConditionID"))
    }  // if(conditionpair->first == condstring)
  }  // loop over conditions

  // communicate results across all processors
  Core::Communication::sum_all(cpuscalars.values(), scalars.values(), numscalars, get_comm());
}  // Core::FE::Discretization::EvaluateScalars


/*----------------------------------------------------------------------*
 |  evaluate/assemble scalars across elements (public)         gee 05/11|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate_scalars(
    Teuchos::ParameterList& params, Core::LinAlg::MultiVector<double>& scalars)
{
  if (!filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  Core::LinAlg::MultiVector<double>& sca = scalars;

  // number of scalars
  const int numscalars = scalars.NumVectors();
  if (numscalars <= 0) FOUR_C_THROW("scalars vector of interest has size <=0");

  // define element matrices and vectors
  // -- which are empty and unused, just to satisfy element evaluate()
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  // loop over _row_ elements
  const int numrowele = num_my_row_elements();
  for (int i = 0; i < numrowele; ++i)
  {
    // pointer to current element
    Core::Elements::Element* actele = l_row_element(i);

    if (!scalars.get_map().my_gid(actele->id()))
      FOUR_C_THROW("Proc does not have global element {}", actele->id());

    // get element location vector
    Core::Elements::LocationArray la(dofsets_.size());
    actele->location_vector(*this, la);

    // define element vector
    Core::LinAlg::SerialDenseVector elescalars(numscalars);

    // call the element evaluate method
    {
      int err = actele->evaluate(
          params, *this, la, elematrix1, elematrix2, elescalars, elevector2, elevector3);
      if (err)
        FOUR_C_THROW("Proc {}: Element {} returned err={}",
            Core::Communication::my_mpi_rank(get_comm()), actele->id(), err);
    }

    for (int j = 0; j < numscalars; ++j)
    {
      sca(j).get_values()[i] = elescalars(j);
    }

  }  // for (int i=0; i<numrowele; ++i)
}  // Core::FE::Discretization::EvaluateScalars


/*----------------------------------------------------------------------*
 |  evaluate an initial scalar or vector field (public)       popp 06/11|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::evaluate_initial_field(
    const Core::Utils::FunctionManager& function_manager, const std::string& fieldstring,
    Core::LinAlg::Vector<double>& fieldvector, const std::vector<int>& locids) const
{
  Core::FE::evaluate_initial_field(function_manager, *this, fieldstring, fieldvector, locids);
}  // Core::FE::Discretization::EvaluateInitialField

FOUR_C_NAMESPACE_CLOSE
