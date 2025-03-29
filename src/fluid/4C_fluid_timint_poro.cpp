// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_timint_poro.hpp"

#include "4C_fem_general_node.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_parameter_poro.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

FLD::TimIntPoro::TimIntPoro(const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis, solver, params, output, alefluid)
{
}

void FLD::TimIntPoro::init()
{
  Teuchos::ParameterList* stabparams;
  stabparams = &(params_->sublist("RESIDUAL-BASED STABILIZATION"));

  if (stabparams->get<Inpar::FLUID::StabType>("STABTYPE") ==
      Inpar::FLUID::StabType::stabtype_residualbased)
  {
    if (stabparams->get<Inpar::FLUID::SubscalesTD>("TDS") ==
        Inpar::FLUID::SubscalesTD::subscales_time_dependent)
    {
      FOUR_C_THROW(
          "TDS is not implemented for Poro yet. An error will occur in "
          "FluidImplicitTimeInt::TimeUpdate().");
    }
  }

  if (not alefluid_) FOUR_C_THROW("poro fluid has to be an ale fluid!");

  // set some poro-specific parameters
  set_element_custom_parameter();
}

void FLD::TimIntPoro::set_element_general_fluid_parameter()
{
  // set some poro-specific parameters only in specific poro cases
  FluidImplicitTimeInt::set_element_general_fluid_parameter();
}

void FLD::TimIntPoro::set_element_turbulence_parameters()
{
  // set some poro-specific parameters only in specific poro cases
  FluidImplicitTimeInt::set_element_turbulence_parameters();
}

void FLD::TimIntPoro::assemble_mat_and_rhs()
{
  FluidImplicitTimeInt::assemble_mat_and_rhs();
  poro_int_update();
}

void FLD::TimIntPoro::read_restart(int step)
{
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step);
  reader.read_vector(gridv_, "gridv");
}

void FLD::TimIntPoro::set_element_custom_parameter()
{
  Teuchos::ParameterList eleparams;

  // set general element parameters
  eleparams.set("form of convective term", convform_);
  eleparams.set<Inpar::FLUID::LinearisationAction>("Linearisation", newton_);
  eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

  // set poro specific element parameters
  eleparams.set<bool>("conti partial integration", params_->get<bool>("conti partial integration"));
  eleparams.set<Inpar::PoroElast::TransientEquationsOfPoroFluid>("Transient Terms Poro Fluid",
      params_->get<Inpar::PoroElast::TransientEquationsOfPoroFluid>("Transient Terms Poro Fluid"));
  eleparams.set<bool>("convective term", params_->get<bool>("convective term"));

  // parameter for stabilization
  eleparams.sublist("RESIDUAL-BASED STABILIZATION") =
      params_->sublist("RESIDUAL-BASED STABILIZATION");
  eleparams.sublist("EDGE-BASED STABILIZATION") = params_->sublist("EDGE-BASED STABILIZATION");

  Discret::Elements::FluidEleParameterPoro* fldpara =
      Discret::Elements::FluidEleParameterPoro::instance();
  fldpara->set_element_poro_parameter(
      eleparams, Core::Communication::my_mpi_rank(discret_->get_comm()));
}

void FLD::TimIntPoro::set_initial_porosity_field(
    const Inpar::PoroElast::InitialField init, const int startfuncno)
{
  std::cout << "FLD::TimIntPoro::set_initial_porosity_field()" << std::endl;

  switch (init)
  {
    case Inpar::PoroElast::initfield_field_by_function:
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      // loop all nodes on the processor
      for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
      {
        // get the processor local node
        Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(lnode);

        int numdofs = nodedofset.size();
        double initialval = Global::Problem::instance()
                                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(startfuncno)
                                .evaluate(lnode->x().data(), time_, 0);

        // check whether there are invalid values of porosity
        if (initialval < 1e-15) FOUR_C_THROW("zero or negative initial porosity");
        if (initialval > 1.0) FOUR_C_THROW("initial porosity greater than 1");
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->LID(dofgid);
          // evaluate component k of spatial function
          int err = init_porosity_field_->replace_local_values(1, &initialval, &doflid);
          if (err != 0) FOUR_C_THROW("dof not on proc");
        }
      }

      break;
    }
    default:
      FOUR_C_THROW("Unknown option for initial field: {}", init);
      break;
  }
}

void FLD::TimIntPoro::update_iter_incrementally(
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel)  //!< input residual velocities

{
  FluidImplicitTimeInt::update_iter_incrementally(vel);
  // set the new solution we just got
  if (vel != nullptr)
  {
    // Take Dirichlet values from velnp and add vel to veln for non-Dirichlet
    // values.
    std::shared_ptr<Core::LinAlg::Vector<double>> aux =
        Core::LinAlg::create_vector(*(discret_->dof_row_map(0)), true);

    // only one step theta
    // new end-point accelerations
    aux->update(1.0 / (theta_ * dta_), *velnp_, -1.0 / (theta_ * dta_), *veln_, 0.0);
    aux->update(-(1.0 - theta_) / theta_, *accn_, 1.0);
    // put only to free/non-DBC DOFs
    dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*accnp_), *aux);
    *accnp_ = *aux;
  }
}

void FLD::TimIntPoro::output()
{
  FluidImplicitTimeInt::output();
  // output of solution
  if (step_ % upres_ == 0)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> convel =
        std::make_shared<Core::LinAlg::Vector<double>>(*velnp_);
    convel->update(-1.0, *gridv_, 1.0);
    output_->write_vector("convel", convel);
    output_->write_vector("gridv", gridv_);
  }
  // write restart also when uprestart_ is not a integer multiple of upres_
  else if (uprestart_ > 0 && step_ % uprestart_ == 0)
  {
    output_->write_vector("gridv", gridv_);
  }
}

void FLD::TimIntPoro::set_custom_ele_params_assemble_mat_and_rhs(Teuchos::ParameterList& eleparams)
{
  eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

  // just for poroelasticity
  discret_->set_state("dispn", *dispn_);
  discret_->set_state("accnp", *accnp_);
  discret_->set_state("accn", *accn_);
  discret_->set_state("gridvn", *gridvn_);

  eleparams.set("total time", time_);
  eleparams.set("delta time", dta_);
}

void FLD::TimIntPoro::poro_int_update()
{
  sysmat_->un_complete();

  std::string condname = "PoroPartInt";
  std::vector<Core::Conditions::Condition*> poroPartInt;
  discret_->get_condition(condname, poroPartInt);
  if (poroPartInt.size())
  {
    Teuchos::ParameterList eleparams;

    // set action for elements
    eleparams.set<FLD::BoundaryAction>("action", FLD::poro_boundary);
    eleparams.set("total time", time_);
    eleparams.set("delta time", dta_);
    eleparams.set<PoroElast::Coupltype>("coupling", PoroElast::fluidfluid);
    eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

    discret_->clear_state();
    discret_->set_state("dispnp", *dispnp_);
    discret_->set_state("gridv", *gridv_);
    discret_->set_state("velnp", *velnp_);
    discret_->set_state("scaaf", *scaaf_);
    discret_->evaluate_condition(
        eleparams, sysmat_, nullptr, residual_, nullptr, nullptr, condname);
    discret_->clear_state();
  }

  condname = "PoroPresInt";
  std::vector<Core::Conditions::Condition*> poroPresInt;
  discret_->get_condition(condname, poroPresInt);
  if (poroPresInt.size())
  {
    Teuchos::ParameterList eleparams;

    // set action for elements
    eleparams.set<FLD::BoundaryAction>("action", FLD::poro_prescoupl);
    eleparams.set<PoroElast::Coupltype>("coupling", PoroElast::fluidfluid);
    eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

    discret_->clear_state();
    discret_->set_state("dispnp", *dispnp_);
    discret_->set_state("gridv", *gridv_);
    discret_->set_state("velnp", *velnp_);
    discret_->evaluate_condition(
        eleparams, sysmat_, nullptr, residual_, nullptr, nullptr, condname);
    discret_->clear_state();
  }
  sysmat_->complete();
}

void FLD::TimIntPoro::tim_int_calculate_acceleration()
{
  // for poro problems, there is a time derivative of the porosity/pressure
  // in the continuity equation. Therefore, we potentially need time
  // derivatives of the pressure and thus do not split the state vectors
  calculate_acceleration(velnp_, veln_, velnm_, accn_, accnp_);
}

FOUR_C_NAMESPACE_CLOSE
