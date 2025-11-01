// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_meshtying_strategy_s2i_elch.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils_gid_vector.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_soret.hpp"
#include "4C_mortar_element.hpp"
#include "4C_scatra_ele_boundary_calc_elch_electrode.hpp"
#include "4C_scatra_ele_boundary_calc_elch_electrode_sti_thermo.hpp"
#include "4C_scatra_ele_boundary_calc_elch_electrode_utils.hpp"
#include "4C_scatra_ele_boundary_calc_sti_electrode.hpp"
#include "4C_scatra_ele_parameter_boundary.hpp"
#include "4C_scatra_ele_parameter_elch.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_parameter_list.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                               fang 12/14 |
 *----------------------------------------------------------------------*/
ScaTra::MeshtyingStrategyS2IElch::MeshtyingStrategyS2IElch(
    ScaTra::ScaTraTimIntElch* elchtimint, const Teuchos::ParameterList& parameters)
    : MeshtyingStrategyS2I(elchtimint, parameters),
      etagrowthmin_(0.),
      intlayergrowth_startstep_(-1),
      intlayergrowth_timestep_active_(false)
{
}  // ScaTra::MeshtyingStrategyS2IElch::MeshtyingStrategyS2IElch


/*---------------------------------------------------------------------------*
 | compute time step size                                         fang 02/18 |
 *---------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2IElch::compute_time_step_size(double& dt)
{
  // consider adaptive time stepping for scatra-scatra interface layer growth if necessary
  if (intlayergrowth_timestep_ > 0.0)
  {
    // add state vectors to discretization
    scatratimint_->discretization()->clear_state();
    scatratimint_->add_time_integration_specific_vectors();

    // create parameter list
    Teuchos::ParameterList condparams;

    // action for elements
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
        "action", ScaTra::BoundaryAction::calc_elch_minmax_overpotential, condparams);

    // initialize results
    condparams.set<double>("etagrowthmin", std::numeric_limits<double>::infinity());
    condparams.set<double>("etagrowthmax", -std::numeric_limits<double>::infinity());

    // extract boundary conditions for scatra-scatra interface layer growth
    std::vector<const Core::Conditions::Condition*> conditions;
    scatratimint_->discretization()->get_condition("S2IKineticsGrowth", conditions);

    // collect condition specific data and store to scatra boundary parameter class
    set_condition_specific_scatra_parameters(*conditions[0]);
    // evaluate minimum and maximum interfacial overpotential associated with scatra-scatra
    // interface layer growth
    scatratimint_->discretization()->evaluate_condition(
        condparams, nullptr, nullptr, nullptr, nullptr, nullptr, "S2IKineticsGrowth");
    scatratimint_->discretization()->clear_state();

    // communicate minimum interfacial overpotential associated with scatra-scatra interface layer
    // growth
    double etagrowthmin(0.0);
    etagrowthmin = Core::Communication::min_all(
        condparams.get<double>("etagrowthmin"), scatratimint_->discretization()->get_comm());

    // adaptive time stepping for scatra-scatra interface layer growth is currently inactive
    if (not intlayergrowth_timestep_active_)
    {
      // check whether adaptive time stepping for scatra-scatra interface layer growth needs to be
      // activated this is the case if the minimum interfacial overpotential is currently positive,
      // but would turn negative after adding twice the change in the minimum interfacial
      // overpotential during the previous time step, i.e., eta - 2*(eta_old - eta) < 0, so that
      // lithium plating could take place after the current time step
      if (etagrowthmin > 0.0 and etagrowthmin - 2.0 * (etagrowthmin_ - etagrowthmin) < 0.0)
        // activate adaptive time stepping for scatra-scatra interface layer growth
        intlayergrowth_timestep_active_ = true;
    }

    // adaptive time stepping for scatra-scatra interface layer growth is currently active
    else
    {
      // communicate maximum interfacial overpotential associated with scatra-scatra interface layer
      // growth
      double etagrowthmax(0.0);
      etagrowthmax = Core::Communication::max_all(
          condparams.get<double>("etagrowthmax"), scatratimint_->discretization()->get_comm());

      // check whether maximum interfacial overpotential has become negative
      if (etagrowthmax < 0.0 and intlayergrowth_startstep_ < 0)
      {
        // store current time step as indicator for completed onset of scatra-scatra interface layer
        // growth
        intlayergrowth_startstep_ = scatratimint_->step();
      }

      // check whether adaptive time stepping for scatra-scatra interface layer growth needs to be
      // deactivated this is the case if ten time steps have passed since the completed onset of
      // scatra-scatra interface layer growth or if the minimum interfacial overpotential is
      // positive and increasing
      if (scatratimint_->step() == intlayergrowth_startstep_ + 10 or
          (etagrowthmin > 0.0 and etagrowthmin > etagrowthmin_))
      {
        // deactivate adaptive time stepping for scatra-scatra interface layer growth
        intlayergrowth_timestep_active_ = false;

        // reset time step tracker
        intlayergrowth_startstep_ = -1;
      }
    }

    // update minimum interfacial overpotential associated with scatra-scatra interface layer growth
    etagrowthmin_ = etagrowthmin;

    // reduce time step size if necessary
    if (dt > intlayergrowth_timestep_ and intlayergrowth_timestep_active_)
      dt = intlayergrowth_timestep_;
  }
}  // ScaTra::MeshtyingStrategyS2IElch::compute_time_step_size


/*--------------------------------------------------------------------------------------*
 | evaluate scatra-scatra interface coupling conditions (electrochemistry)   fang 10/14 |
 *--------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2IElch::evaluate_meshtying()
{
  // call base class routine
  ScaTra::MeshtyingStrategyS2I::evaluate_meshtying();
}  // ScaTra::MeshtyingStrategyS2IElch::evaluate_meshtying

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2IElch::evaluate_point_coupling()
{
  // extract multi-scale coupling conditions
  // loop over conditions
  for (const auto& slave_condition : kinetics_conditions_meshtying_slave_side())
  {
    auto* cond_slave = slave_condition.second;

    // only evaluate point coupling conditions
    if (cond_slave->g_type() != Core::Conditions::geometry_type_point) continue;

    auto* cond_master = master_conditions()[slave_condition.first];

    // extract nodal cloud
    const std::vector<int>* const nodeids_slave = cond_slave->get_nodes();
    const std::vector<int>* const nodeids_master = cond_master->get_nodes();

    if (nodeids_slave->size() != 1 or nodeids_master->size() != 1)
      FOUR_C_THROW("only one node per condition allowed");

    const int nodeid_slave = (*nodeids_slave)[0];
    const int nodeid_master = (*nodeids_master)[0];

    auto dis = scatratimint_->discretization();

    auto* slave_node = dis->g_node(nodeid_slave);
    auto* master_node = dis->g_node(nodeid_master);

    // extract degrees of freedom from node
    const std::vector<int> slave_dofs = dis->dof(0, slave_node);
    const std::vector<int> master_dofs = dis->dof(0, master_node);

    const int ed_conc_gid = slave_dofs[0];
    const int ed_pot_gid = slave_dofs[1];
    const int el_conc_gid = master_dofs[0];
    const int el_pot_gid = master_dofs[1];

    auto dof_row_map = scatratimint_->dof_row_map();
    const int ed_conc_lid = dof_row_map->lid(ed_conc_gid);
    const int ed_pot_lid = dof_row_map->lid(ed_pot_gid);
    const int el_conc_lid = dof_row_map->lid(el_conc_gid);
    const int el_pot_lid = dof_row_map->lid(el_pot_gid);

    // extract electrode-side and electrolyte-side values at coupling point
    auto phinp = scatratimint_->phinp();
    const double ed_conc = phinp->local_values_as_span()[ed_conc_lid];
    const double ed_pot = phinp->local_values_as_span()[ed_pot_lid];
    const double el_conc = phinp->local_values_as_span()[el_conc_lid];
    const double el_pot = phinp->local_values_as_span()[el_pot_lid];

    // compute matrix and vector contributions according to kinetic model for current point coupling
    // condition
    const int kinetic_model =
        cond_slave->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL");
    switch (kinetic_model)
    {
      case Inpar::S2I::kinetics_butlervolmer:
      case Inpar::S2I::kinetics_butlervolmerreduced:
      {
        // access material of electrode
        auto matelectrode = std::dynamic_pointer_cast<const Mat::Electrode>(
            slave_node->adjacent_elements()[0].user_element()->material());
        if (matelectrode == nullptr)
          FOUR_C_THROW("Invalid electrode material for multi-scale coupling!");

        // access input parameters associated with current condition
        const int nume = cond_slave->parameters().get<int>("E-");
        if (nume != 1)
        {
          FOUR_C_THROW(
              "Invalid number of electrons involved in charge transfer at "
              "electrode-electrolyte interface!");
        }
        const std::vector<int>* stoichiometries =
            cond_slave->parameters().get_if<std::vector<int>>("STOICHIOMETRIES");
        if (stoichiometries == nullptr)
        {
          FOUR_C_THROW(
              "Cannot access vector of stoichiometric coefficients for multi-scale "
              "coupling!");
        }
        if (stoichiometries->size() != 1)
          FOUR_C_THROW("Number of stoichiometric coefficients does not match number of scalars!");
        if ((*stoichiometries)[0] != -1) FOUR_C_THROW("Invalid stoichiometric coefficient!");
        const double faraday =
            Global::Problem::instance(0)->elch_control_params().get<double>("FARADAY_CONSTANT");
        const double gasconstant =
            Global::Problem::instance(0)->elch_control_params().get<double>("GAS_CONSTANT");
        const double frt =
            faraday /
            (gasconstant *
                (Global::Problem::instance(0)->elch_control_params().get<double>("TEMPERATURE")));
        const double alphaa = cond_slave->parameters().get<double>("ALPHA_A");
        const double alphac = cond_slave->parameters().get<double>("ALPHA_C");
        const double kr = cond_slave->parameters().get<double>("K_R");
        if (kr < 0.0) FOUR_C_THROW("Charge transfer constant k_r is negative!");

        // extract saturation value of intercalated lithium concentration from electrode material
        const double cmax = matelectrode->c_max();
        if (cmax < 1.0e-12)
          FOUR_C_THROW(
              "Saturation value c_max of intercalated lithium concentration is too small!");

        // compute domain integration factor
        constexpr double four_pi = 4.0 * std::numbers::pi;
        const double fac = scatratimint_->scatra_parameter_list()->get<bool>("SPHERICALCOORDS")
                               ? *slave_node->x().data() * *slave_node->x().data() * four_pi
                               : 1.0;
        const double timefacfac =
            Discret::Elements::ScaTraEleParameterTimInt::instance(dis->name())->time_fac() * fac;
        const double timefacrhsfac =
            Discret::Elements::ScaTraEleParameterTimInt::instance(dis->name())->time_fac_rhs() *
            fac;
        if (timefacfac < 0.0 or timefacrhsfac < 0.0)
          FOUR_C_THROW("Integration factor is negative!");
        // no deformation available
        const double dummy_detF(1.0);

        // equilibrium electric potential difference and its derivative w.r.t. concentration
        // at electrode surface
        const double epd =
            matelectrode->compute_open_circuit_potential(ed_conc, faraday, frt, dummy_detF);
        const double epdderiv = matelectrode->compute_d_open_circuit_potential_d_concentration(
            ed_conc, faraday, frt, dummy_detF);

        // overpotential
        const double eta = ed_pot - el_pot - epd;

        // Butler-Volmer exchange mass flux density
        const double j0 = cond_slave->parameters().get<Inpar::S2I::KineticModels>(
                              "KINETIC_MODEL") == Inpar::S2I::kinetics_butlervolmerreduced
                              ? kr
                              : kr * std::pow(el_conc, alphaa) * std::pow(cmax - ed_conc, alphaa) *
                                    std::pow(ed_conc, alphac);

        // exponential Butler-Volmer terms
        const double expterm1 = std::exp(alphaa * frt * eta);
        const double expterm2 = std::exp(-alphac * frt * eta);
        const double expterm = expterm1 - expterm2;

        // core residual term associated with Butler-Volmer mass flux density
        const double j = j0 * expterm;

        // initialize a dummy resistance as the method below requires a resistance which is not
        // relevant in this case
        const double dummyresistance(0.0);
        // define flux linearization terms
        double dj_ded_conc(0.0), dj_del_conc(0.0), dj_ded_pot(0.0), dj_del_pot(0.0);
        // calculate flux linearizations
        Discret::Elements::calculate_butler_volmer_elch_linearizations(kinetic_model, j0, frt,
            epdderiv, alphaa, alphac, dummyresistance, expterm1, expterm2, kr, faraday, el_conc,
            ed_conc, cmax, eta, dj_ded_conc, dj_del_conc, dj_ded_pot, dj_del_pot);

        // assemble concentration residuals
        auto residual = scatratimint_->residual();
        (*residual).get_values()[ed_conc_lid] -= timefacrhsfac * j;
        (*residual).get_values()[el_conc_lid] -= timefacrhsfac * j * -1.0;

        // assemble potential residuals
        (*residual).get_values()[ed_pot_lid] -= timefacrhsfac * nume * j;
        (*residual).get_values()[el_pot_lid] -= timefacrhsfac * nume * j * -1.0;

        // assemble concentration linearizations
        auto sys_mat = scatratimint_->system_matrix_operator();
        sys_mat->assemble(timefacfac * dj_ded_conc, ed_conc_gid, ed_conc_gid);
        sys_mat->assemble(timefacfac * dj_del_conc, ed_conc_gid, el_conc_gid);
        sys_mat->assemble(timefacfac * dj_ded_pot, ed_conc_gid, ed_pot_gid);
        sys_mat->assemble(timefacfac * dj_del_pot, ed_conc_gid, el_pot_gid);

        sys_mat->assemble(timefacfac * dj_ded_conc * -1.0, el_conc_gid, ed_conc_gid);
        sys_mat->assemble(timefacfac * dj_del_conc * -1.0, el_conc_gid, el_conc_gid);
        sys_mat->assemble(timefacfac * dj_ded_pot * -1.0, el_conc_gid, ed_pot_gid);
        sys_mat->assemble(timefacfac * dj_del_pot * -1.0, el_conc_gid, el_pot_gid);

        // assemble potential linearizations
        sys_mat->assemble(timefacfac * nume * dj_ded_conc, ed_pot_gid, ed_conc_gid);
        sys_mat->assemble(timefacfac * nume * dj_del_conc, ed_pot_gid, el_conc_gid);
        sys_mat->assemble(timefacfac * nume * dj_ded_pot, ed_pot_gid, ed_pot_gid);
        sys_mat->assemble(timefacfac * nume * dj_del_pot, ed_pot_gid, el_pot_gid);

        sys_mat->assemble(timefacfac * nume * dj_ded_conc * -1.0, el_pot_gid, ed_conc_gid);
        sys_mat->assemble(timefacfac * nume * dj_del_conc * -1.0, el_pot_gid, el_conc_gid);
        sys_mat->assemble(timefacfac * nume * dj_ded_pot * -1.0, el_pot_gid, ed_pot_gid);
        sys_mat->assemble(timefacfac * nume * dj_del_pot * -1.0, el_pot_gid, el_pot_gid);

        break;
      }
      case Inpar::S2I::kinetics_nointerfaceflux:
        break;

      default:
      {
        FOUR_C_THROW("Kinetic model for s2i coupling not yet implemented!");
      }
    }
  }
}

/*------------------------------------------------------------------------*
 | instantiate strategy for Newton-Raphson convergence check   fang 02/16 |
 *------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2IElch::init_conv_check_strategy()
{
  if (couplingtype_ == Inpar::S2I::coupling_mortar_saddlepoint_petrov or
      couplingtype_ == Inpar::S2I::coupling_mortar_saddlepoint_bubnov)
  {
    convcheckstrategy_ = std::make_shared<ScaTra::ConvCheckStrategyS2ILMElch>(
        scatratimint_->scatra_parameter_list()->sublist("NONLINEAR"));
  }
  else if (elch_tim_int()->macro_scale())
  {
    convcheckstrategy_ = std::make_shared<ScaTra::ConvCheckStrategyStdMacroScaleElch>(
        scatratimint_->scatra_parameter_list()->sublist("NONLINEAR"));
  }
  else
    convcheckstrategy_ = std::make_shared<ScaTra::ConvCheckStrategyStdElch>(
        scatratimint_->scatra_parameter_list()->sublist("NONLINEAR"));
}  // ScaTra::MeshtyingStrategyS2IElch::init_conv_check_strategy


/*------------------------------------------------------------------------------------------*
 | update solution after convergence of the nonlinear Newton-Raphson iteration   fang 01/17 |
 *------------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2IElch::update() const
{
  // update scatra-scatra interface layer thicknesses in case of semi-implicit solution approach
  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_semi_implicit)
  {
    // extract boundary conditions for scatra-scatra interface layer growth
    std::vector<const Core::Conditions::Condition*> conditions;
    scatratimint_->discretization()->get_condition("S2IKineticsGrowth", conditions);

    // loop over all conditions
    for (const auto& condition : conditions)
    {
      // extract current condition
      // extract kinetic model from current condition
      switch (condition->parameters().get<Inpar::S2I::GrowthKineticModels>("KINETIC_MODEL"))
      {
        case Inpar::S2I::growth_kinetics_butlervolmer:
        {
          // extract parameters from current condition
          const auto kr = condition->parameters().get<double>("K_R");
          const auto alphaa = condition->parameters().get<double>("ALPHA_A");
          const auto alphac = condition->parameters().get<double>("ALPHA_C");
          const double frt = elch_tim_int()->frt();
          const double conductivity_inverse =
              1. / condition->parameters().get<double>("CONDUCTIVITY");
          const double faraday =
              Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();

          // pre-compute integration factor
          const double integrationfac(condition->parameters().get<double>("MOLMASS") *
                                      scatratimint_->dt() /
                                      (condition->parameters().get<double>("DENSITY") * faraday));

          // extract nodal cloud from current condition
          const std::vector<int>* nodegids = condition->get_nodes();

          // loop over all nodes
          for (int nodegid : *nodegids)
          {
            // extract global ID of current node
            // process only nodes stored by current processor
            if (scatratimint_->discretization()->have_global_node(nodegid))
            {
              // extract current node
              const Core::Nodes::Node* const node =
                  scatratimint_->discretization()->g_node(nodegid);

              // process only nodes owned by current processor
              if (node->owner() ==
                  Core::Communication::my_mpi_rank(scatratimint_->discretization()->get_comm()))
              {
                // extract local ID of first scalar transport degree of freedom associated with
                // current node
                const int doflid_scatra = scatratimint_->discretization()->dof_row_map()->lid(
                    scatratimint_->discretization()->dof(
                        0, node, 0));  // Do not remove the first zero, i.e., the first function
                                       // argument, otherwise an error is thrown in debug mode!
                if (doflid_scatra < 0)
                  FOUR_C_THROW("Couldn't extract local ID of scalar transport degree of freedom!");

                // extract local ID of scatra-scatra interface layer thickness variable associated
                // with current node
                const int doflid_growth = scatratimint_->discretization()->dof_row_map(2)->lid(
                    scatratimint_->discretization()->dof(2, node, 0));
                if (doflid_growth < 0)
                  FOUR_C_THROW(
                      "Couldn't extract local ID of scatra-scatra interface layer thickness!");

                // extract slave-side electric potential associated with current node
                const double slavepot =
                    scatratimint_->phiafnp()->local_values_as_span()[doflid_scatra + 1];

                // extract master-side lithium concentration associated with current node
                const double masterphi =
                    imasterphi_on_slave_side_np_->local_values_as_span()[doflid_scatra];

                // extract master-side electric potential associated with current node
                const double masterpot =
                    imasterphi_on_slave_side_np_->local_values_as_span()[doflid_scatra + 1];

                // compute interface layer resistance associated with current node
                const double resistance =
                    growthn_->local_values_as_span()[doflid_growth] * conductivity_inverse;

                // check existence of interface layer and set Heaviside value accordingly
                const unsigned heaviside(resistance > 0. ? 1 : 0);

                // compute exchange current density
                const double i0 = kr * faraday * pow(masterphi, alphaa);

                // compute initial guess of Butler-Volmer current density associated with lithium
                // plating, neglecting overpotential due to resistance of plated lithium
                double eta = slavepot - masterpot;
                double i = i0 * (heaviside * exp(alphaa * frt * eta) - exp(-alphac * frt * eta));

                // initialize Newton-Raphson iteration counter
                unsigned iternum(0);

                // apply Newton-Raphson method to compute Butler-Volmer current density associated
                // with lithium plating, involving overpotential due to resistance of plated lithium
                while (true)
                {
                  // increment counter
                  ++iternum;

                  // compute current Newton-Raphson residual
                  eta = slavepot - masterpot -
                        resistance *
                            i;  // open-circuit potential is zero for lithium plating reaction
                  const double expterm1 = heaviside * exp(alphaa * frt * eta);
                  const double expterm2 = exp(-alphac * frt * eta);
                  const double residual = i0 * (expterm1 - expterm2) - i;

                  // convergence check
                  if (std::abs(residual) < intlayergrowth_convtol_)
                    break;
                  else if (iternum == intlayergrowth_itemax_)
                  {
                    FOUR_C_THROW(
                        "Local Newton-Raphson iteration for scatra-scatra interface layer growth "
                        "did not converge!");
                  }

                  // compute linearization of current Newton-Raphson residual w.r.t. Butler-Volmer
                  // current density associated with lithium plating
                  const double linearization =
                      -i0 * resistance * frt * (alphaa * expterm1 + alphac * expterm2) - 1.;

                  // update Butler-Volmer current density
                  i -= residual / linearization;
                }

                // enforce plating condition, i.e., consider initial lithium plating only in case of
                // negative overpotential
                if (!heaviside and eta >= 0.) i = 0.;

                // update lithium plating variable
                (*growthn_).get_values()[doflid_growth] -= i * integrationfac;
              }  // nodes owned by current processor
            }  // nodes stored by current processor
          }  // loop over all nodes

          break;
        }

        default:
        {
          FOUR_C_THROW(
              "Kinetic model for scatra-scatra interface layer growth is not yet implemented!");
          break;
        }
      }  // kinetic models
    }  // loop over all conditions
  }  // semi-implicit evaluation of scatra-scatra interface layer growth

  else
    // call base class routine
    MeshtyingStrategyS2I::update();
}


/*----------------------------------------------------------------------*
 | singleton access method                                   fang 01/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
ScaTra::MortarCellCalcElch<distype_s, distype_m>*
ScaTra::MortarCellCalcElch<distype_s, distype_m>::instance(
    const Inpar::S2I::CouplingType& couplingtype,  //!< flag for meshtying method
    const Inpar::S2I::InterfaceSides&
        lmside,  //!< flag for interface side underlying Lagrange multiplier definition
    const int& numdofpernode_slave,   //!< number of slave-side degrees of freedom per node
    const int& numdofpernode_master,  //!< number of master-side degrees of freedom per node
    const std::string& disname        //!< name of mortar discretization
)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const Inpar::S2I::CouplingType& couplingtype, const Inpar::S2I::InterfaceSides& lmside,
          const int& numdofpernode_slave, const int& numdofpernode_master)
      {
        return std::unique_ptr<MortarCellCalcElch<distype_s, distype_m>>(
            new MortarCellCalcElch<distype_s, distype_m>(
                couplingtype, lmside, numdofpernode_slave, numdofpernode_master));
      });

  return singleton_map[disname].instance(Core::Utils::SingletonAction::create, couplingtype, lmside,
      numdofpernode_slave, numdofpernode_master);
}


/*----------------------------------------------------------------------*
 | protected constructor for singletons                      fang 01/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
ScaTra::MortarCellCalcElch<distype_s, distype_m>::MortarCellCalcElch(
    const Inpar::S2I::CouplingType& couplingtype,  //!< flag for meshtying method
    const Inpar::S2I::InterfaceSides&
        lmside,  //!< flag for interface side underlying Lagrange multiplier definition
    const int& numdofpernode_slave,  //!< number of slave-side degrees of freedom per node
    const int& numdofpernode_master  //!< number of master-side degrees of freedom per node
    )
    : my::MortarCellCalc(couplingtype, lmside, numdofpernode_slave, numdofpernode_master)
{
}

/*---------------------------------------------------------------------------*
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalcElch<distype_s, distype_m>::evaluate_condition(
    const Core::FE::Discretization& idiscret, Mortar::IntCell& cell, Mortar::Element& slaveelement,
    Mortar::Element& masterelement, Core::Elements::LocationArray& la_slave,
    Core::Elements::LocationArray& la_master, const Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_sm,
    Core::LinAlg::SerialDenseMatrix& k_ms, Core::LinAlg::SerialDenseMatrix& k_mm,
    Core::LinAlg::SerialDenseVector& r_s, Core::LinAlg::SerialDenseVector& r_m)
{
  // safety checks
  if (my::numdofpernode_slave_ != 2 or my::numdofpernode_master_ != 2)
    FOUR_C_THROW("Invalid number of degrees of freedom per node!");
  if (Discret::Elements::ScaTraEleParameterElch::instance("scatra")->equ_pot() != ElCh::equpot_divi)
    FOUR_C_THROW("Invalid closing equation for electric potential!");

  // extract condition from parameter list
  const Core::Conditions::Condition* condition =
      params.get<const Core::Conditions::Condition*>("condition");
  if (condition == nullptr)
    FOUR_C_THROW("Cannot access scatra-scatra interface coupling condition!");

  // access material of slave element
  std::shared_ptr<const Mat::Electrode> matelectrode =
      std::dynamic_pointer_cast<const Mat::Electrode>(slaveelement.material());
  if (matelectrode == nullptr)
    FOUR_C_THROW("Invalid electrode material for scatra-scatra interface coupling!");

  // extract nodal state variables associated with slave and master elements
  this->extract_node_values(idiscret, la_slave, la_master);

  // determine quadrature rule
  const Core::FE::IntPointsAndWeights<2> intpoints(Core::FE::GaussRule2D::tri_7point);

  // dummy matrix of nodal temperature values
  Core::LinAlg::Matrix<nen_slave_, 1> dummy_slave_temp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_master_, 1> dummy_master_temp(Core::LinAlg::Initialization::zero);
  // always in contact
  const double pseudo_contact_fac = 1.0;

  // loop over all integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_dom_int_fac_at_int_point(
        slaveelement, masterelement, cell, intpoints, iquad);
    // no deformation available
    const double dummy_detF(1.0);

    // overall integration factors
    const double timefacfac =
        Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time_fac() * fac;
    const double timefacrhsfac =
        Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time_fac_rhs() * fac;
    if (timefacfac < 0.0 or timefacrhsfac < 0.0) FOUR_C_THROW("Integration factor is negative!");

    Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<
        distype_s>::template evaluate_s2_i_coupling_at_integration_point<distype_m>(matelectrode,
        my::ephinp_slave_, my::ephinp_master_, dummy_slave_temp, dummy_master_temp,
        pseudo_contact_fac, my::funct_slave_, my::funct_master_, my::test_lm_slave_,
        my::test_lm_master_, my::scatraparamsboundary_, timefacfac, timefacrhsfac, dummy_detF,
        get_frt(), my::numdofpernode_slave_, k_ss, k_sm, k_ms, k_mm, r_s, r_m);
  }
}

/*---------------------------------------------------------------------------*
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalcElch<distype_s, distype_m>::evaluate_condition_nts(
    const Core::Conditions::Condition& condition, const Mortar::Node& slavenode,
    const double& lumpedarea, Mortar::Element& slaveelement, Mortar::Element& masterelement,
    const std::vector<Core::LinAlg::Matrix<nen_slave_, 1>>& ephinp_slave,
    const std::vector<Core::LinAlg::Matrix<nen_master_, 1>>& ephinp_master,
    Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_sm,
    Core::LinAlg::SerialDenseMatrix& k_ms, Core::LinAlg::SerialDenseMatrix& k_mm,
    Core::LinAlg::SerialDenseVector& r_s, Core::LinAlg::SerialDenseVector& r_m)
{
  // safety checks
  if (my::numdofpernode_slave_ != 2 or my::numdofpernode_master_ != 2)
    FOUR_C_THROW("Invalid number of degrees of freedom per node!");
  if (Discret::Elements::ScaTraEleParameterElch::instance("scatra")->equ_pot() != ElCh::equpot_divi)
    FOUR_C_THROW("Invalid closing equation for electric potential!");

  // access material of slave element
  std::shared_ptr<const Mat::Electrode> matelectrode =
      std::dynamic_pointer_cast<const Mat::Electrode>(
          std::dynamic_pointer_cast<Core::Elements::FaceElement>(
              condition.geometry().at(slaveelement.id()))
              ->parent_element()
              ->material());
  if (matelectrode == nullptr)
    FOUR_C_THROW("Invalid electrode material for scatra-scatra interface coupling!");

  // evaluate shape functions at position of slave-side node
  my::eval_shape_func_at_slave_node(slavenode, slaveelement, masterelement);

  // dummy matrix of nodal temperature values
  Core::LinAlg::Matrix<nen_slave_, 1> dummy_slave_temp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_master_, 1> dummy_master_temp(Core::LinAlg::Initialization::zero);
  // always in contact
  const double pseudo_contact_fac = 1.0;

  // overall integration factors
  const double timefacfac =
      Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time_fac() * lumpedarea;
  const double timefacrhsfac =
      Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time_fac_rhs() * lumpedarea;
  if (timefacfac < 0. or timefacrhsfac < 0.) FOUR_C_THROW("Integration factor is negative!");

  // no deformation available
  const double dummy_detF(1.0);

  Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<
      distype_s>::template evaluate_s2_i_coupling_at_integration_point<distype_m>(matelectrode,
      ephinp_slave, ephinp_master, dummy_slave_temp, dummy_master_temp, pseudo_contact_fac,
      my::funct_slave_, my::funct_master_, my::funct_slave_, my::funct_master_,
      my::scatraparamsboundary_, timefacfac, timefacrhsfac, dummy_detF,
      Discret::Elements::ScaTraEleParameterElch::instance("scatra")->frt(),
      my::numdofpernode_slave_, k_ss, k_sm, k_ms, k_mm, r_s, r_m);
}


/*----------------------------------------------------------------------*
 | evaluate factor F/RT                                      fang 01/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
double ScaTra::MortarCellCalcElch<distype_s, distype_m>::get_frt() const
{
  // fetch factor F/RT from electrochemistry parameter list
  return Discret::Elements::ScaTraEleParameterElch::instance("scatra")->frt();
}


/*----------------------------------------------------------------------*
 | singleton access method                                   fang 01/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
ScaTra::MortarCellCalcElchSTIThermo<distype_s, distype_m>*
ScaTra::MortarCellCalcElchSTIThermo<distype_s, distype_m>::instance(
    const Inpar::S2I::CouplingType& couplingtype,  //!< flag for meshtying method
    const Inpar::S2I::InterfaceSides&
        lmside,  //!< flag for interface side underlying Lagrange multiplier definition
    const int& numdofpernode_slave,   //!< number of slave-side degrees of freedom per node
    const int& numdofpernode_master,  //!< number of master-side degrees of freedom per node
    const std::string& disname        //!< name of mortar discretization
)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const Inpar::S2I::CouplingType& couplingtype, const Inpar::S2I::InterfaceSides& lmside,
          const int& numdofpernode_slave, const int& numdofpernode_master)
      {
        return std::unique_ptr<MortarCellCalcElchSTIThermo<distype_s, distype_m>>(
            new MortarCellCalcElchSTIThermo<distype_s, distype_m>(
                couplingtype, lmside, numdofpernode_slave, numdofpernode_master));
      });

  return singleton_map[disname].instance(Core::Utils::SingletonAction::create, couplingtype, lmside,
      numdofpernode_slave, numdofpernode_master);
}


/*----------------------------------------------------------------------*
 | private constructor for singletons                        fang 01/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
ScaTra::MortarCellCalcElchSTIThermo<distype_s, distype_m>::MortarCellCalcElchSTIThermo(
    const Inpar::S2I::CouplingType& couplingtype,  //!< flag for meshtying method
    const Inpar::S2I::InterfaceSides&
        lmside,  //!< flag for interface side underlying Lagrange multiplier definition
    const int& numdofpernode_slave,  //!< number of slave-side degrees of freedom per node
    const int& numdofpernode_master  //!< number of master-side degrees of freedom per node
    )
    :  // call base class constructor
      myelch::MortarCellCalcElch(couplingtype, lmside, numdofpernode_slave, numdofpernode_master),

      // initialize member variable
      etempnp_slave_(Core::LinAlg::Initialization::zero)
{
}


/*--------------------------------------------------------------------------------------------------------------------*
 | evaluate single mortar integration cell of particular slave-side and master-side discretization
 types   fang 01/17 |
 *--------------------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalcElchSTIThermo<distype_s, distype_m>::evaluate(
    const Core::FE::Discretization& idiscret,      //!< interface discretization
    Mortar::IntCell& cell,                         //!< mortar integration cell
    Mortar::Element& slaveelement,                 //!< slave-side mortar element
    Mortar::Element& masterelement,                //!< master-side mortar element
    Core::Elements::LocationArray& la_slave,       //!< slave-side location array
    Core::Elements::LocationArray& la_master,      //!< master-side location array
    const Teuchos::ParameterList& params,          //!< parameter list
    Core::LinAlg::SerialDenseMatrix& cellmatrix1,  //!< cell matrix 1
    Core::LinAlg::SerialDenseMatrix& cellmatrix2,  //!< cell matrix 2
    Core::LinAlg::SerialDenseMatrix& cellmatrix3,  //!< cell matrix 3
    Core::LinAlg::SerialDenseMatrix& cellmatrix4,  //!< cell matrix 4
    Core::LinAlg::SerialDenseVector& cellvector1,  //!< cell vector 1
    Core::LinAlg::SerialDenseVector& cellvector2   //!< cell vector 2
)
{
  // extract and evaluate action
  switch (Teuchos::getIntegralValue<Inpar::S2I::EvaluationActions>(params, "action"))
  {
    // evaluate and assemble off-diagonal interface linearizations
    case Inpar::S2I::evaluate_condition_od:
    {
      evaluate_condition_od(idiscret, cell, slaveelement, masterelement, la_slave, la_master,
          params, cellmatrix1, cellmatrix3);

      break;
    }

    // call base class routine
    default:
    {
      my::evaluate(idiscret, cell, slaveelement, masterelement, la_slave, la_master, params,
          cellmatrix1, cellmatrix2, cellmatrix3, cellmatrix4, cellvector1, cellvector2);

      break;
    }
  }
}


/*---------------------------------------------------------------------------*
 | evaluate and assemble off-diagonal interface linearizations    fang 01/17 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalcElchSTIThermo<distype_s, distype_m>::evaluate_condition_od(
    const Core::FE::Discretization& idiscret,  //!< interface discretization
    Mortar::IntCell& cell,                     //!< mortar integration cell
    Mortar::Element& slaveelement,             //!< slave-side mortar element
    Mortar::Element& masterelement,            //!< master-side mortar element
    Core::Elements::LocationArray& la_slave,   //!< slave-side location array
    Core::Elements::LocationArray& la_master,  //!< master-side location array
    const Teuchos::ParameterList& params,      //!< parameter list
    Core::LinAlg::SerialDenseMatrix&
        k_ss,  //!< linearizations of slave-side residuals w.r.t. slave-side dofs
    Core::LinAlg::SerialDenseMatrix&
        k_ms  //!< linearizations of master-side residuals w.r.t. slave-side dofs
)
{
  // safety checks
  if (my::numdofpernode_slave_ != 2 or my::numdofpernode_master_ != 2)
    FOUR_C_THROW("Invalid number of degrees of freedom per node!");
  if (Discret::Elements::ScaTraEleParameterElch::instance("scatra")->equ_pot() != ElCh::equpot_divi)
    FOUR_C_THROW("Invalid closing equation for electric potential!");

  // extract condition from parameter list
  const Core::Conditions::Condition* s2icondition =
      params.get<const Core::Conditions::Condition*>("condition");
  if (s2icondition == nullptr)
    FOUR_C_THROW("Cannot access scatra-scatra interface coupling condition!");

  // access material of slave element
  std::shared_ptr<const Mat::Electrode> matelectrode =
      std::dynamic_pointer_cast<const Mat::Electrode>(
          std::dynamic_pointer_cast<Core::Elements::FaceElement>(
              s2icondition->geometry().at(slaveelement.id()))
              ->parent_element()
              ->material());
  if (matelectrode == nullptr)
    FOUR_C_THROW("Invalid electrode material for scatra-scatra interface coupling!");

  // extract nodal state variables associated with slave and master elements
  extract_node_values(idiscret, la_slave, la_master);

  // determine quadrature rule
  const Core::FE::IntPointsAndWeights<2> intpoints(Core::FE::GaussRule2D::tri_7point);

  // dummy matrix of nodal master temperature values and shape derivatives
  Core::LinAlg::Matrix<nen_master_, 1> dummy_master_temp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_slave_ + 1, nen_slave_> dummy_shapederivatives(
      Core::LinAlg::Initialization::zero);
  // always in contact
  const double pseudo_contact_fac = 1.0;

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_dom_int_fac_at_int_point(
        slaveelement, masterelement, cell, intpoints, gpid);

    // evaluate overall integration factor
    const double timefac =
        Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time_fac();
    const double timefacfac = timefac * fac;
    if (timefacfac < 0.) FOUR_C_THROW("Integration factor is negative!");

    const double timefacwgt = timefac * intpoints.ip().qwgt[gpid];

    // no deformation available
    const double dummy_detF(1.0);

    Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype_s>::
        template evaluate_s2_i_coupling_od_at_integration_point<distype_m>(*matelectrode,
            my::ephinp_slave_, etempnp_slave_, dummy_master_temp, my::ephinp_master_,
            pseudo_contact_fac, my::funct_slave_, my::funct_master_, my::test_lm_slave_,
            my::test_lm_master_, dummy_shapederivatives, dummy_shapederivatives,
            my::scatraparamsboundary_, ScaTra::DifferentiationType::temp, timefacfac, timefacwgt,
            dummy_detF, my::numdofpernode_slave_, k_ss, k_ms);
  }  // loop over integration points
}


/*------------------------------------------------------------------------------------*
 | extract nodal state variables associated with mortar integration cell   fang 01/17 |
 *------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalcElchSTIThermo<distype_s, distype_m>::extract_node_values(
    const Core::FE::Discretization& idiscret,  //!< interface discretization
    Core::Elements::LocationArray& la_slave,   //!< slave-side location array
    Core::Elements::LocationArray& la_master   //!< master-side location array
)
{
  // call base class routine
  my::extract_node_values(idiscret, la_slave, la_master);

  // extract nodal temperature variables associated with mortar integration cell
  my::extract_node_values(etempnp_slave_, idiscret, la_slave, "thermo", 1);
}


/*----------------------------------------------------------------------*
 | evaluate factor F/RT                                      fang 01/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
double ScaTra::MortarCellCalcElchSTIThermo<distype_s, distype_m>::get_frt() const
{
  // evaluate local temperature value
  const double temperature = my::funct_slave_.dot(etempnp_slave_);

  // safety check
  if (temperature <= 0.) FOUR_C_THROW("Temperature is non-positive!");

  const double faraday = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
  const double gasconstant =
      Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();

  // evaluate factor F/RT
  return faraday / (gasconstant * temperature);
}


/*----------------------------------------------------------------------*
 | singleton access method                                   fang 01/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
ScaTra::MortarCellCalcSTIElch<distype_s, distype_m>*
ScaTra::MortarCellCalcSTIElch<distype_s, distype_m>::instance(
    const Inpar::S2I::CouplingType& couplingtype,  //!< flag for meshtying method
    const Inpar::S2I::InterfaceSides&
        lmside,  //!< flag for interface side underlying Lagrange multiplier definition
    const int& numdofpernode_slave,   //!< number of slave-side degrees of freedom per node
    const int& numdofpernode_master,  //!< number of master-side degrees of freedom per node
    const std::string& disname        //!< name of mortar discretization
)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const Inpar::S2I::CouplingType& couplingtype, const Inpar::S2I::InterfaceSides& lmside,
          const int& numdofpernode_slave, const int& numdofpernode_master)
      {
        return std::unique_ptr<MortarCellCalcSTIElch<distype_s, distype_m>>(
            new MortarCellCalcSTIElch<distype_s, distype_m>(
                couplingtype, lmside, numdofpernode_slave, numdofpernode_master));
      });

  return singleton_map[disname].instance(Core::Utils::SingletonAction::create, couplingtype, lmside,
      numdofpernode_slave, numdofpernode_master);
}


/*----------------------------------------------------------------------*
 | private constructor for singletons                        fang 01/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
ScaTra::MortarCellCalcSTIElch<distype_s, distype_m>::MortarCellCalcSTIElch(
    const Inpar::S2I::CouplingType& couplingtype,  //!< flag for meshtying method
    const Inpar::S2I::InterfaceSides&
        lmside,  //!< flag for interface side underlying Lagrange multiplier definition
    const int& numdofpernode_slave,  //!< number of slave-side degrees of freedom per node
    const int& numdofpernode_master  //!< number of master-side degrees of freedom per node
    )
    :  // call base class constructor
      my::MortarCellCalc(couplingtype, lmside, numdofpernode_slave, numdofpernode_master),

      // initialize member variables
      eelchnp_slave_(2, Core::LinAlg::Matrix<nen_slave_, 1>(Core::LinAlg::Initialization::zero)),
      eelchnp_master_(2, Core::LinAlg::Matrix<nen_master_, 1>(Core::LinAlg::Initialization::zero))
{
}


/*--------------------------------------------------------------------------------------------------------------------*
 | evaluate single mortar integration cell of particular slave-side and master-side discretization
 types   fang 01/17 |
 *--------------------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalcSTIElch<distype_s, distype_m>::evaluate(
    const Core::FE::Discretization& idiscret,      //!< interface discretization
    Mortar::IntCell& cell,                         //!< mortar integration cell
    Mortar::Element& slaveelement,                 //!< slave-side mortar element
    Mortar::Element& masterelement,                //!< master-side mortar element
    Core::Elements::LocationArray& la_slave,       //!< slave-side location array
    Core::Elements::LocationArray& la_master,      //!< master-side location array
    const Teuchos::ParameterList& params,          //!< parameter list
    Core::LinAlg::SerialDenseMatrix& cellmatrix1,  //!< cell matrix 1
    Core::LinAlg::SerialDenseMatrix& cellmatrix2,  //!< cell matrix 2
    Core::LinAlg::SerialDenseMatrix& cellmatrix3,  //!< cell matrix 3
    Core::LinAlg::SerialDenseMatrix& cellmatrix4,  //!< cell matrix 4
    Core::LinAlg::SerialDenseVector& cellvector1,  //!< cell vector 1
    Core::LinAlg::SerialDenseVector& cellvector2   //!< cell vector 2
)
{
  // extract and evaluate action
  switch (Teuchos::getIntegralValue<Inpar::S2I::EvaluationActions>(params, "action"))
  {
    // evaluate and assemble interface linearizations and residuals
    case Inpar::S2I::evaluate_condition:
    {
      evaluate_condition(idiscret, cell, slaveelement, masterelement, la_slave, la_master, params,
          cellmatrix1, cellvector1);

      break;
    }

    // evaluate and assemble off-diagonal interface linearizations
    case Inpar::S2I::evaluate_condition_od:
    {
      evaluate_condition_od(idiscret, cell, slaveelement, masterelement, la_slave, la_master,
          params, cellmatrix1, cellmatrix2);

      break;
    }

    // call base class routine
    default:
    {
      my::evaluate(idiscret, cell, slaveelement, masterelement, la_slave, la_master, params,
          cellmatrix1, cellmatrix2, cellmatrix3, cellmatrix4, cellvector1, cellvector2);

      break;
    }
  }
}


/*---------------------------------------------------------------------------*
 | evaluate and assemble interface linearizations and residuals   fang 01/17 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalcSTIElch<distype_s, distype_m>::evaluate_condition(
    const Core::FE::Discretization& idiscret,  //!< interface discretization
    Mortar::IntCell& cell,                     //!< mortar integration cell
    Mortar::Element& slaveelement,             //!< slave-side mortar element
    Mortar::Element& masterelement,            //!< master-side mortar element
    Core::Elements::LocationArray& la_slave,   //!< slave-side location array
    Core::Elements::LocationArray& la_master,  //!< master-side location array
    const Teuchos::ParameterList& params,      //!< parameter list
    Core::LinAlg::SerialDenseMatrix&
        k_ss,  //!< linearizations of slave-side residuals w.r.t. slave-side dofs
    Core::LinAlg::SerialDenseVector& r_s  //!< slave-side residual vector
)
{
  // safety check
  if (my::numdofpernode_slave_ != 1 or my::numdofpernode_master_ != 1)
    FOUR_C_THROW("Invalid number of degrees of freedom per node!");

  // extract condition from parameter list
  const Core::Conditions::Condition* s2icondition =
      params.get<const Core::Conditions::Condition*>("condition");
  if (s2icondition == nullptr)
    FOUR_C_THROW("Cannot access scatra-scatra interface coupling condition!");

  // access primary and secondary materials of slave element
  const std::shared_ptr<const Mat::Soret> matsoret = std::dynamic_pointer_cast<const Mat::Soret>(
      std::dynamic_pointer_cast<Core::Elements::FaceElement>(
          s2icondition->geometry().at(slaveelement.id()))
          ->parent_element()
          ->material());
  const std::shared_ptr<const Mat::Electrode> matelectrode =
      std::dynamic_pointer_cast<const Mat::Electrode>(
          std::dynamic_pointer_cast<Core::Elements::FaceElement>(
              s2icondition->geometry().at(slaveelement.id()))
              ->parent_element()
              ->material(1));
  if (matsoret == nullptr or matelectrode == nullptr)
    FOUR_C_THROW("Invalid electrode material for scatra-scatra interface coupling!");

  // extract nodal state variables associated with slave and master elements
  extract_node_values(idiscret, la_slave, la_master);

  // determine quadrature rule
  const Core::FE::IntPointsAndWeights<2> intpoints(Core::FE::GaussRule2D::tri_7point);

  // dummy matrix for derivative of slave fluxes w.r.t. master side temperatures
  Core::LinAlg::SerialDenseMatrix dummy_ksm;
  // always in contact
  const double pseudo_contact_fac = 1.0;

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_dom_int_fac_at_int_point(
        slaveelement, masterelement, cell, intpoints, gpid);

    // evaluate overall integration factors
    const double timefacfac =
        Discret::Elements::ScaTraEleParameterTimInt::instance("thermo")->time_fac() * fac;
    const double timefacrhsfac =
        Discret::Elements::ScaTraEleParameterTimInt::instance("thermo")->time_fac_rhs() * fac;
    if (timefacfac < 0. or timefacrhsfac < 0.) FOUR_C_THROW("Integration factor is negative!");

    // no deformation available
    const double dummy_detF(1.0);

    Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<
        distype_s>::template evaluate_s2_i_coupling_at_integration_point<distype_m>(*matelectrode,
        my::ephinp_slave_[0], my::ephinp_master_[0], eelchnp_slave_, eelchnp_master_,
        pseudo_contact_fac, my::funct_slave_, my::funct_master_, my::scatraparamsboundary_,
        timefacfac, timefacrhsfac, dummy_detF, k_ss, dummy_ksm, r_s);
  }  // loop over integration points
}


/*---------------------------------------------------------------------------*
 | evaluate and assemble off-diagonal interface linearizations    fang 01/17 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalcSTIElch<distype_s, distype_m>::evaluate_condition_od(
    const Core::FE::Discretization& idiscret,  //!< interface discretization
    Mortar::IntCell& cell,                     //!< mortar integration cell
    Mortar::Element& slaveelement,             //!< slave-side mortar element
    Mortar::Element& masterelement,            //!< master-side mortar element
    Core::Elements::LocationArray& la_slave,   //!< slave-side location array
    Core::Elements::LocationArray& la_master,  //!< master-side location array
    const Teuchos::ParameterList& params,      //!< parameter list
    Core::LinAlg::SerialDenseMatrix&
        k_ss,  //!< linearizations of slave-side residuals w.r.t. slave-side dofs
    Core::LinAlg::SerialDenseMatrix&
        k_sm  //!< linearizations of slave-side residuals w.r.t. master-side dofs
)
{
  // safety check
  if (my::numdofpernode_slave_ != 1 or my::numdofpernode_master_ != 1)
    FOUR_C_THROW("Invalid number of degrees of freedom per node!");

  // extract condition from parameter list
  const Core::Conditions::Condition* s2icondition =
      params.get<const Core::Conditions::Condition*>("condition");
  if (s2icondition == nullptr)
    FOUR_C_THROW("Cannot access scatra-scatra interface coupling condition!");

  // access primary and secondary materials of parent element
  std::shared_ptr<const Mat::Soret> matsoret = std::dynamic_pointer_cast<const Mat::Soret>(
      std::dynamic_pointer_cast<Core::Elements::FaceElement>(
          s2icondition->geometry().at(slaveelement.id()))
          ->parent_element()
          ->material());
  std::shared_ptr<const Mat::Electrode> matelectrode =
      std::dynamic_pointer_cast<const Mat::Electrode>(
          std::dynamic_pointer_cast<Core::Elements::FaceElement>(
              s2icondition->geometry().at(slaveelement.id()))
              ->parent_element()
              ->material(1));
  if (matsoret == nullptr or matelectrode == nullptr)
    FOUR_C_THROW("Invalid electrode or soret material for scatra-scatra interface coupling!");

  // extract nodal state variables associated with slave and master elements
  extract_node_values(idiscret, la_slave, la_master);

  // determine quadrature rule
  const Core::FE::IntPointsAndWeights<2> intpoints(Core::FE::GaussRule2D::tri_7point);

  // dummy matrix for shape derivatives
  Core::LinAlg::Matrix<3, nen_slave_> dummy_shape_deriv;
  // always in contact
  const double pseudo_contact_fac = 1.0;

  // loop over all integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_dom_int_fac_at_int_point(
        slaveelement, masterelement, cell, intpoints, iquad);

    // overall integration factors
    const double timefacfac =
        Discret::Elements::ScaTraEleParameterTimInt::instance("thermo")->time_fac() * fac;
    if (timefacfac < 0.) FOUR_C_THROW("Integration factor is negative!");

    // no deformation available
    const double dummy_detF(1.0);

    Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype_s>::
        template evaluate_s2_i_coupling_od_at_integration_point<distype_m>(*matelectrode,
            my::ephinp_slave_[0], my::ephinp_master_[0], eelchnp_slave_, eelchnp_master_,
            pseudo_contact_fac, my::funct_slave_, my::funct_master_, my::scatraparamsboundary_,
            timefacfac, fac, dummy_detF, ScaTra::DifferentiationType::elch, dummy_shape_deriv,
            dummy_shape_deriv, k_ss, k_sm);
  }  // loop over integration points
}


/*------------------------------------------------------------------------------------*
 | extract nodal state variables associated with mortar integration cell   fang 01/17 |
 *------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalcSTIElch<distype_s, distype_m>::extract_node_values(
    const Core::FE::Discretization& idiscret,  //!< interface discretization
    Core::Elements::LocationArray& la_slave,   //!< slave-side location array
    Core::Elements::LocationArray& la_master   //!< master-side location array
)
{
  // extract nodal temperature variables associated with slave element
  my::extract_node_values(my::ephinp_slave_[0], idiscret, la_slave);

  // extract nodal electrochemistry variables associated with mortar integration cell
  my::extract_node_values(
      eelchnp_slave_, eelchnp_master_, idiscret, la_slave, la_master, "scatra", 1);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
ScaTra::MeshtyingStrategyS2IElchSCL::MeshtyingStrategyS2IElchSCL(
    ScaTra::ScaTraTimIntElch* elchtimint, const Teuchos::ParameterList& parameters)
    : MeshtyingStrategyS2IElch(elchtimint, parameters)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2IElchSCL::setup_meshtying()
{
  // extract scatra-scatra coupling conditions from discretization
  std::vector<const Core::Conditions::Condition*> s2imeshtying_conditions;
  scatratimint_->discretization()->get_condition("S2IMeshtying", s2imeshtying_conditions);

  std::set<int> islavenodegidset;
  std::set<int> imasternodegidset;

  for (const auto& s2imeshtying_condition : s2imeshtying_conditions)
  {
    if (s2imeshtying_condition->parameters().get<int>("S2I_KINETICS_ID") != -1)
      FOUR_C_THROW("No kinetics condition is allowed for the coupled space-charge layer problem.");

    switch (s2imeshtying_condition->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE"))
    {
      case Inpar::S2I::side_slave:
      {
        Core::Communication::add_owned_node_gid_from_list(*scatratimint_->discretization(),
            *s2imeshtying_condition->get_nodes(), islavenodegidset);
        break;
      }
      case Inpar::S2I::side_master:
      {
        Core::Communication::add_owned_node_gid_from_list(*scatratimint_->discretization(),
            *s2imeshtying_condition->get_nodes(), imasternodegidset);
        break;
      }
      default:
      {
        FOUR_C_THROW("interface side must bee slave or master");
        break;
      }
    }
  }

  std::vector<int> islavenodegidvec(islavenodegidset.begin(), islavenodegidset.end());
  std::vector<int> imasternodegidvec(imasternodegidset.begin(), imasternodegidset.end());

  icoup_ = std::make_shared<Coupling::Adapter::Coupling>();
  icoup_->setup_coupling(*(scatratimint_->discretization()), *(scatratimint_->discretization()),
      imasternodegidvec, islavenodegidvec, 2, true, 1.0e-8);
}

/*------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2IElchSCL::solve(const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& increment,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& residual,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& phinp, const int iteration,
    Core::LinAlg::SolverParams& solver_params) const
{
  solver_params.refactor = true;
  solver_params.reset = iteration == 1;
  solver->solve(systemmatrix, increment, residual, solver_params);
}



// forward declarations
template class ScaTra::MortarCellCalcElch<Core::FE::CellType::tri3, Core::FE::CellType::tri3>;
template class ScaTra::MortarCellCalcElch<Core::FE::CellType::tri3, Core::FE::CellType::quad4>;
template class ScaTra::MortarCellCalcElch<Core::FE::CellType::quad4, Core::FE::CellType::tri3>;
template class ScaTra::MortarCellCalcElch<Core::FE::CellType::quad4, Core::FE::CellType::quad4>;
template class ScaTra::MortarCellCalcElchSTIThermo<Core::FE::CellType::tri3,
    Core::FE::CellType::tri3>;
template class ScaTra::MortarCellCalcElchSTIThermo<Core::FE::CellType::tri3,
    Core::FE::CellType::quad4>;
template class ScaTra::MortarCellCalcElchSTIThermo<Core::FE::CellType::quad4,
    Core::FE::CellType::tri3>;
template class ScaTra::MortarCellCalcElchSTIThermo<Core::FE::CellType::quad4,
    Core::FE::CellType::quad4>;
template class ScaTra::MortarCellCalcSTIElch<Core::FE::CellType::tri3, Core::FE::CellType::tri3>;
template class ScaTra::MortarCellCalcSTIElch<Core::FE::CellType::tri3, Core::FE::CellType::quad4>;
template class ScaTra::MortarCellCalcSTIElch<Core::FE::CellType::quad4, Core::FE::CellType::tri3>;
template class ScaTra::MortarCellCalcSTIElch<Core::FE::CellType::quad4, Core::FE::CellType::quad4>;

FOUR_C_NAMESPACE_CLOSE
