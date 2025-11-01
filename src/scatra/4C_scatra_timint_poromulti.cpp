// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_poromulti.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_utils.hpp"
#include "4C_scatra_ele_action.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                             vuong  08/16 |
 *----------------------------------------------------------------------*/
ScaTra::ScaTraTimIntPoroMulti::ScaTraTimIntPoroMulti(std::shared_ptr<Core::FE::Discretization> dis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(dis, solver, sctratimintparams, extraparams, output), L2_projection_(false)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
  return;
}


/*----------------------------------------------------------------------*
 | initialize algorithm                                    vuong  08/16 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMulti::init() { return; }

/*----------------------------------------------------------------------*
 | set solution fields on given dof sets                    vuong  08/16 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMulti::set_l2_flux_of_multi_fluid(
    std::shared_ptr<const Core::LinAlg::MultiVector<double>> multiflux)
{
  // set L2-projection to true
  L2_projection_ = true;

  // safety check
  if (nds_vel() >= discret_->num_dof_sets())
    FOUR_C_THROW("Too few dofsets on scatra discretization!");

  if (multiflux->num_vectors() % nsd_ != 0)
    FOUR_C_THROW("Unexpected length of flux vector: {}", multiflux->num_vectors());

  const int totalnumdof = multiflux->num_vectors() / nsd_;

  std::string stateprefix = "flux";

  for (int curphase = 0; curphase < totalnumdof; ++curphase)
  {
    // initialize velocity vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> phaseflux =
        Core::LinAlg::create_vector(*discret_->dof_row_map(nds_vel()), true);

    std::stringstream statename;
    statename << stateprefix << curphase;

    int count = 0;
    for (auto lnode : discret_->my_row_node_range())
    {
      // get dofs associated with current node
      std::vector<int> nodedofs = discret_->dof(nds_vel(), lnode);

      if ((int)nodedofs.size() != nsd_)
        FOUR_C_THROW(
            "Expected number of DOFs to be equal to the number of space dimensions for flux "
            "state!");

      for (int index = 0; index < nsd_; ++index)
      {
        // get global and local dof IDs
        const int gid = nodedofs[index];
        const int lid = phaseflux->get_map().lid(gid);
        if (lid < 0) FOUR_C_THROW("Local ID not found in map for given global ID!");

        const double value = ((*multiflux)(curphase * nsd_ + index)).local_values_as_span()[count];

        phaseflux->replace_local_value(lid, value);
      }
      ++count;
    }

    // provide scatra discretization with convective velocity
    discret_->set_state(nds_vel(), statename.str(), *phaseflux);
  }
}  // ScaTraTimIntImpl::SetSolutionFields

/*----------------------------------------------------------------------*
 | set solution fields on given dof sets              kremheller  07/17 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMulti::set_solution_field_of_multi_fluid(
    std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_fluid,
    std::shared_ptr<const Core::LinAlg::Vector<double>> phin_fluid)
{
  if (nds_pressure() >= discret_->num_dof_sets())
    FOUR_C_THROW("Too few dofsets on scatra discretization!");

  // provide scatra discretization with fluid primary variable field
  discret_->set_state(nds_pressure(), "phinp_fluid", *phinp_fluid);
  discret_->set_state(nds_pressure(), "phin_fluid", *phin_fluid);
}

/*----------------------------------------------------------------------*
 | add parameters depending on the problem                  vuong  08/16 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMulti::add_problem_specific_parameters_and_vectors(
    Teuchos::ParameterList& params  //!< parameter list
)
{
  // provide pressure field
  params.set<bool>("L2-projection", L2_projection_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMulti::collect_runtime_output_data()
{
  visualization_writer().append_element_owner("Owner");

  visualization_writer().append_result_data_vector_with_context(
      *phinp_, Core::IO::OutputEntity::dof, phi_components_);

  // displacement field
  if (isale_)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
        discret_->get_state(nds_disp(), "dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot extract displacement field from discretization");

    // convert dof-based vector into node-based multi-vector for postprocessing
    auto dispnp_multi = Core::IO::convert_dof_vector_to_node_based_multi_vector(
        *discret_, *dispnp, nds_disp(), nsd_);

    std::vector<std::optional<std::string>> context(nsd_, "ale-displacement");
    visualization_writer().append_result_data_vector_with_context(
        *dispnp_multi, Core::IO::OutputEntity::node, context);
  }

  // extract conditions for oxygen partial pressure
  std::vector<const Core::Conditions::Condition*> conditions;
  discret_->get_condition("PoroMultiphaseScatraOxyPartPressCalcCond", conditions);

  // perform all following operations only if there is at least one condition for oxygen partial
  // pressure
  if (conditions.size() > 0)
  {
    auto oxypartpress = Core::LinAlg::Vector<double>(*discret_->node_row_map(), true);

    // this condition is supposed to be for output of oxygen partial pressure over whole domain
    // it does not make sense to have more than one condition
    if (conditions.size() != 1)
      FOUR_C_THROW(
          "Should have only one PoroMultiphaseScatraOxyPartPressCalcCond per discretization");

    // extract nodal cloud from condition
    const std::vector<int>* nodegids = conditions[0]->get_nodes();

    // output
    double Pb = 0.0;

    // read input from condition
    const int oxyscalar = conditions[0]->parameters().get<int>("SCALARID") - 1;
    const double CaO2_max = conditions[0]->parameters().get<double>("CaO2_max");
    const double Pb50 = conditions[0]->parameters().get<double>("Pb50");
    const double n = conditions[0]->parameters().get<double>("n");
    const double alpha_eff = conditions[0]->parameters().get<double>("alpha_bl_eff");
    const double rho_oxy = conditions[0]->parameters().get<double>("rho_oxy");
    const double rho_bl = conditions[0]->parameters().get<double>("rho_bl");

    // loop over all nodes
    for (int nodegid : *nodegids)
    {
      // process only nodes stored by current processor
      if (discret_->have_global_node(nodegid))
      {
        // extract current node
        const Core::Nodes::Node* const node = discret_->g_node(nodegid);

        // process only nodes owned by current processor
        if (node->owner() == Core::Communication::my_mpi_rank(discret_->get_comm()))
        {
          // get dof
          int myoxydof = discret_->dof(0, node, oxyscalar);
          const int lidoxydof = discret_->dof_row_map()->lid(myoxydof);
          if (lidoxydof < 0) FOUR_C_THROW("Couldn't extract local ID of oxygen dof!");
          // compute CaO2
          const double CaO2 = phinp_->local_values_as_span()[lidoxydof] * rho_bl / rho_oxy;
          // compute Pb
          PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
              Pb, CaO2, CaO2_max, Pb50, n, alpha_eff);
          // replace value
          oxypartpress.replace_global_value(node->id(), Pb);
        }
      }
    }

    std::vector<std::optional<std::string>> context(oxypartpress.num_vectors(), "oxypartpress");
    visualization_writer().append_result_data_vector_with_context(
        oxypartpress, Core::IO::OutputEntity::node, context);
  }

  strategy_->collect_output_data();
}

/*----------------------------------------------------------------------*
 |  Constructor (public)                                    vuong  08/16 |
 *----------------------------------------------------------------------*/
ScaTra::ScaTraTimIntPoroMultiOST::ScaTraTimIntPoroMultiOST(
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, sctratimintparams, extraparams, output),
      ScaTraTimIntPoroMulti(actdis, solver, params, sctratimintparams, extraparams, output),
      TimIntOneStepTheta(actdis, solver, sctratimintparams, extraparams, output)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                             vuong  08/16 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMultiOST::init()
{
  // call init()-functions of base classes
  // note: this order is important
  TimIntOneStepTheta::init();
  ScaTraTimIntPoroMulti::init();

  return;
}



/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                            gjb 08/08 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMultiOST::update()
{
  TimIntOneStepTheta::update();
  ScaTraTimIntPoroMulti::update();

  return;
}

/*----------------------------------------------------------------------*
 |  Constructor (public)                                    vuong  08/16 |
 *----------------------------------------------------------------------*/
ScaTra::ScaTraTimIntPoroMultiBDF2::ScaTraTimIntPoroMultiBDF2(
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, sctratimintparams, extraparams, output),
      ScaTraTimIntPoroMulti(actdis, solver, params, sctratimintparams, extraparams, output),
      TimIntBDF2(actdis, solver, sctratimintparams, extraparams, output)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                             vuong  08/16 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMultiBDF2::init()
{
  // call init()-functions of base classes
  // note: this order is important
  TimIntBDF2::init();
  ScaTraTimIntPoroMulti::init();

  return;
}



/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                            gjb 08/08 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMultiBDF2::update()
{
  TimIntBDF2::update();
  ScaTraTimIntPoroMulti::update();

  return;
}


/*----------------------------------------------------------------------*
 |  Constructor (public)                                    vuong  08/16 |
 *----------------------------------------------------------------------*/
ScaTra::ScaTraTimIntPoroMultiGenAlpha::ScaTraTimIntPoroMultiGenAlpha(
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, sctratimintparams, extraparams, output),
      ScaTraTimIntPoroMulti(actdis, solver, params, sctratimintparams, extraparams, output),
      TimIntGenAlpha(actdis, solver, sctratimintparams, extraparams, output)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                             vuong  08/16 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMultiGenAlpha::init()
{
  // call init()-functions of base classes
  // note: this order is important
  TimIntGenAlpha::init();
  ScaTraTimIntPoroMulti::init();

  return;
}



/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                            gjb 08/08 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMultiGenAlpha::update()
{
  TimIntGenAlpha::update();
  ScaTraTimIntPoroMulti::update();

  return;
}

/*----------------------------------------------------------------------*
 |  Constructor (public)                                    vuong  08/16 |
 *----------------------------------------------------------------------*/
ScaTra::ScaTraTimIntPoroMultiStationary::ScaTraTimIntPoroMultiStationary(
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, sctratimintparams, extraparams, output),
      ScaTraTimIntPoroMulti(actdis, solver, params, sctratimintparams, extraparams, output),
      TimIntStationary(actdis, solver, sctratimintparams, extraparams, output)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                             vuong  08/16 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMultiStationary::init()
{
  // call init()-functions of base classes
  // note: this order is important
  TimIntStationary::init();
  ScaTraTimIntPoroMulti::init();

  return;
}



/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                         vuong  08/16 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntPoroMultiStationary::update()
{
  TimIntStationary::update();
  ScaTraTimIntPoroMulti::update();

  return;
}

FOUR_C_NAMESPACE_CLOSE
