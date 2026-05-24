// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_basedatasdyn.hpp"

#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_global_data.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_new_utils.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

Solid::TimeInt::DynamicRebalanceConfig Solid::TimeInt::parse_dynamic_rebalance_config(
    const Teuchos::ParameterList& rebalance_params)
{
  DynamicRebalanceConfig config;
  config.enabled = rebalance_params.get<bool>("ENABLED");
  config.imbalance_threshold = rebalance_params.get<double>("IMBALANCE_THRESHOLD");
  config.window_steps = rebalance_params.get<int>("WINDOW_STEPS");
  config.cooldown_steps = rebalance_params.get<int>("COOLDOWN_STEPS");
  config.mesh_partitioning_parameters.rebalance_type =
      Teuchos::getIntegralValue<Core::Rebalance::RebalanceType>(rebalance_params, "REBALANCE_TYPE");
  config.mesh_partitioning_parameters.imbalance_tol = rebalance_params.get<double>("IMBALANCE_TOL");
  config.mesh_partitioning_parameters.min_ele_per_proc =
      rebalance_params.get<int>("MIN_ELE_PER_PROC");
  config.edge_weight_multiplier = rebalance_params.get<double>("EDGE_WEIGHT_MULTIPLIER");
  return config;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataSDyn::BaseDataSDyn()
    : isinit_(false),
      issetup_(false),
      timemax_(-1.0),
      stepmax_(-1),
      timer_(nullptr),
      damptype_(Inpar::Solid::damp_none),
      dampk_(-1.0),
      dampm_(-1.0),
      masslintype_(Inpar::Solid::MassLin::ml_none),
      lumpmass_(false),
      neglectinertia_(false),
      modeltypes_(nullptr),
      eletechs_(nullptr),
      coupling_model_ptr_(nullptr),
      dyntype_(Inpar::Solid::DynamicType::Statics),
      stcscale_(Inpar::Solid::stc_inactive),
      stclayer_(-1),
      itermin_(-1),
      itermax_(-1),
      loadlin_(false),
      prestresstype_(Inpar::Solid::PreStress::none),
      predtype_(Inpar::Solid::pred_vague),
      nlnsolvertype_(Inpar::Solid::soltech_vague),
      divergenceaction_(Inpar::Solid::divcont_stop),
      mid_time_energy_type_(Inpar::Solid::midavg_vague),
      maxdivconrefinementlevel_(-1),
      noxparams_(nullptr),
      ptc_delta_init_(0.0),
      linsolvers_(nullptr),
      normtype_(Inpar::Solid::norm_vague),
      nox_normtype_(::NOX::Abstract::Vector::TwoNorm),
      tol_disp_incr_(-1.0),
      toltype_disp_incr_(Inpar::Solid::convnorm_abs),
      tol_fres_(-1.0),
      toltype_fres_(Inpar::Solid::convnorm_abs),
      tol_pres_(-1.0),
      toltype_pres_(Inpar::Solid::convnorm_abs),
      tol_inco_(-1.0),
      toltype_inco_(Inpar::Solid::convnorm_abs),
      normcombo_disp_pres_(Inpar::Solid::bop_and),
      normcombo_fres_inco_(Inpar::Solid::bop_and),
      normcombo_fres_eas_res_(Inpar::Solid::bop_and),
      normcombo_disp_eas_incr_(Inpar::Solid::bop_and),
      normcombo_fres_disp_(Inpar::Solid::bop_and),
      toltype_cardvasc0d_res_(Inpar::Solid::convnorm_abs),
      tol_cardvasc0d_res_(-1.0),
      toltype_cardvasc0d_incr_(Inpar::Solid::convnorm_abs),
      tol_cardvasc0d_incr_(-1.0),
      toltype_constr_res_(Inpar::Solid::convnorm_abs),
      tol_constr_res_(-1.0),
      toltype_constr_incr_(Inpar::Solid::convnorm_abs),
      tol_constr_incr_(-1.0),
      toltype_contact_res_(Inpar::Solid::convnorm_abs),
      tol_contact_res_(-1.0),
      toltype_contact_lm_incr_(Inpar::Solid::convnorm_abs),
      tol_contact_lm_incr_(-1.0),
      normcombo_fres_contact_res_(Inpar::Solid::bop_and),
      normcombo_disp_contact_lm_incr_(Inpar::Solid::bop_and),
      normcombo_fres_cardvasc0d_res_(Inpar::Solid::bop_and),
      normcombo_disp_cardvasc0d_incr_(Inpar::Solid::bop_and),
      normcombo_fres_constr_res_(Inpar::Solid::bop_and),
      normcombo_disp_constr_incr_(Inpar::Solid::bop_and),
      rand_tsfac_(1.0),
      divconrefinementlevel_(0),
      divconnumfinestep_(0),
      sdynparams_ptr_(nullptr)
{
  // empty constructor
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataSDyn::init(const std::shared_ptr<Core::FE::Discretization> discret,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    const std::shared_ptr<std::set<Inpar::Solid::ModelType>> modeltypes,
    const std::shared_ptr<std::set<Inpar::Solid::EleTech>> eletechs,
    const std::shared_ptr<std::map<Inpar::Solid::ModelType, std::shared_ptr<Core::LinAlg::Solver>>>
        linsolvers)
{
  // We have to call setup() after init()
  issetup_ = false;

  // ---------------------------------------------------------------------------
  // initialize general variables
  // ---------------------------------------------------------------------------
  {
    timemax_ = sdynparams.get<double>("MAXTIME");
    stepmax_ = sdynparams.get<int>("NUMSTEP");

    timer_ = std::make_shared<Teuchos::Time>("", true);

    dyntype_ = Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdynparams, "DYNAMICTYPE");

    stcscale_ = Teuchos::getIntegralValue<Inpar::Solid::StcScale>(sdynparams, "STC_SCALING");

    stclayer_ = sdynparams.get<int>("STC_LAYER");

    isrestarting_initial_state_ =
        Teuchos::getIntegralValue<bool>(sdynparams, "CALC_ACC_ON_RESTART");
  }
  // ---------------------------------------------------------------------------
  // initialize the damping control parameters
  // ---------------------------------------------------------------------------
  {
    damptype_ = Teuchos::getIntegralValue<Inpar::Solid::DampKind>(sdynparams, "DAMPING");
    dampk_ = sdynparams.get<double>("K_DAMP");
    dampm_ = sdynparams.get<double>("M_DAMP");
  }
  // ---------------------------------------------------------------------------
  // initialize the mass and inertia control parameters
  // ---------------------------------------------------------------------------
  {
    masslintype_ = Teuchos::getIntegralValue<Inpar::Solid::MassLin>(sdynparams, "MASSLIN");
    lumpmass_ = sdynparams.get<bool>("LUMPMASS");
    neglectinertia_ = sdynparams.get<bool>("NEGLECTINERTIA");
  }
  // ---------------------------------------------------------------------------
  // initialize model evaluator control parameters
  // ---------------------------------------------------------------------------
  {
    modeltypes_ = modeltypes;
    eletechs_ = eletechs;
    if (modeltypes_->find(Inpar::Solid::model_partitioned_coupling) != modeltypes->end())
    {
      if (modeltypes_->find(Inpar::Solid::model_monolithic_coupling) != modeltypes->end())
        FOUR_C_THROW("Cannot have both monolithic and partitioned coupling at the same time!");
      coupling_model_ptr_ = sdynparams.get<std::shared_ptr<Solid::ModelEvaluator::Generic>>(
          "Partitioned Coupling Model");
    }
    else if (modeltypes_->find(Inpar::Solid::model_monolithic_coupling) != modeltypes->end())
    {
      coupling_model_ptr_ = sdynparams.get<std::shared_ptr<Solid::ModelEvaluator::Generic>>(
          "Monolithic Coupling Model");
    }
    else if (modeltypes_->find(Inpar::Solid::model_basic_coupling) != modeltypes->end())
    {
      coupling_model_ptr_ =
          sdynparams.get<std::shared_ptr<Solid::ModelEvaluator::Generic>>("Basic Coupling Model");
    }
  }
  // ---------------------------------------------------------------------------
  // initialize implicit variables
  // ---------------------------------------------------------------------------
  {
    itermin_ = sdynparams.get<int>("MINITER");
    itermax_ = sdynparams.get<int>("MAXITER");
    loadlin_ = (sdynparams.get<bool>("LOADLIN"));
    prestresstime_ =
        Global::Problem::instance()->structural_dynamic_params().get<double>("PRESTRESSTIME");
    prestresstype_ = Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
        Global::Problem::instance()->structural_dynamic_params(), "PRESTRESS");
    prestress_displacement_tolerance_ = sdynparams.get<double>("PRESTRESSTOLDISP");
    prestress_min_number_of_load_steps_ = sdynparams.get<int>("PRESTRESSMINLOADSTEPS");
    predtype_ = Teuchos::getIntegralValue<Inpar::Solid::PredEnum>(sdynparams, "PREDICT");
    nlnsolvertype_ = Teuchos::getIntegralValue<Inpar::Solid::NonlinSolTech>(sdynparams, "NLNSOL");
    divergenceaction_ =
        Teuchos::getIntegralValue<Inpar::Solid::DivContAct>(sdynparams, "DIVERCONT");
    mid_time_energy_type_ =
        Teuchos::getIntegralValue<Inpar::Solid::MidAverageEnum>(sdynparams, "MIDTIME_ENERGY_TYPE");
    maxdivconrefinementlevel_ = sdynparams.get<int>("MAXDIVCONREFINEMENTLEVEL");
    noxparams_ = std::make_shared<Teuchos::ParameterList>(xparams.sublist("NOX"));
    ptc_delta_init_ = sdynparams.get<double>("PTCDT");
  }
  // ---------------------------------------------------------------------------
  // initialize linear solver variables
  // ---------------------------------------------------------------------------
  {
    linsolvers_ = linsolvers;
  }
  // ---------------------------------------------------------------------------
  // initialize the status test control parameters
  // ---------------------------------------------------------------------------
  {
    normtype_ = Teuchos::getIntegralValue<Inpar::Solid::VectorNorm>(sdynparams, "ITERNORM");
    nox_normtype_ = Solid::Nln::convert2_nox_norm_type(normtype_);

    // -------------------------------------------------------------------------
    // primary variables
    // -------------------------------------------------------------------------
    tol_disp_incr_ = sdynparams.get<double>("TOLDISP");
    toltype_disp_incr_ = Teuchos::getIntegralValue<Inpar::Solid::ConvNorm>(sdynparams, "NORM_DISP");

    tol_fres_ = sdynparams.get<double>("TOLRES");
    toltype_fres_ = Teuchos::getIntegralValue<Inpar::Solid::ConvNorm>(sdynparams, "NORM_RESF");

    tol_pres_ = sdynparams.get<double>("TOLPRE");
    toltype_pres_ = Inpar::Solid::convnorm_abs;

    tol_inco_ = sdynparams.get<double>("TOLINCO");
    toltype_inco_ = Inpar::Solid::convnorm_abs;

    toltype_eas_res_ = Inpar::Solid::convnorm_abs;

    toltype_eas_incr_ = Inpar::Solid::convnorm_abs;

    normcombo_disp_pres_ =
        Teuchos::getIntegralValue<Inpar::Solid::BinaryOp>(sdynparams, "NORMCOMBI_DISPPRES");
    normcombo_fres_inco_ =
        Teuchos::getIntegralValue<Inpar::Solid::BinaryOp>(sdynparams, "NORMCOMBI_RESFINCO");
    normcombo_fres_disp_ =
        Teuchos::getIntegralValue<Inpar::Solid::BinaryOp>(sdynparams, "NORMCOMBI_RESFDISP");

    // -------------------------------------------------------------------------
    // constraint variables
    // -------------------------------------------------------------------------
    tol_constr_res_ = sdynparams.get<double>("TOLCONSTR");
    toltype_constr_res_ = Inpar::Solid::convnorm_abs;

    tol_constr_incr_ = sdynparams.get<double>("TOLCONSTRINCR");
    toltype_constr_incr_ = Inpar::Solid::convnorm_abs;

    tol_cardvasc0d_res_ =
        Global::Problem::instance()->cardiovascular0_d_structural_params().get<double>(
            "TOL_CARDVASC0D_RES");
    toltype_cardvasc0d_res_ = Inpar::Solid::convnorm_abs;

    tol_cardvasc0d_incr_ =
        Global::Problem::instance()->cardiovascular0_d_structural_params().get<double>(
            "TOL_CARDVASC0D_DOFINCR");
    toltype_cardvasc0d_incr_ = Inpar::Solid::convnorm_abs;

    tol_contact_res_ =
        Global::Problem::instance()->contact_dynamic_params().get<double>("TOLCONTCONSTR");
    toltype_contact_res_ = Inpar::Solid::convnorm_abs;

    tol_contact_lm_incr_ =
        Global::Problem::instance()->contact_dynamic_params().get<double>("TOLLAGR");
    toltype_contact_lm_incr_ = Inpar::Solid::convnorm_abs;

    normcombo_fres_contact_res_ = Teuchos::getIntegralValue<Inpar::Solid::BinaryOp>(
        Global::Problem::instance()->contact_dynamic_params(), "NORMCOMBI_RESFCONTCONSTR");
    normcombo_disp_contact_lm_incr_ = Teuchos::getIntegralValue<Inpar::Solid::BinaryOp>(
        Global::Problem::instance()->contact_dynamic_params(), "NORMCOMBI_DISPLAGR");
  }

  {
    // store the structural dynamics parameter list for derived Setup routines
    sdynparams_ptr_ = Core::Utils::shared_ptr_from_ref(sdynparams);
  }

  // -------------------------------------------------------------------------
  // initial displacement variables
  // -------------------------------------------------------------------------
  {
    initial_disp_ = Teuchos::getIntegralValue<Inpar::Solid::InitialDisp>(sdynparams, "INITIALDISP");
    start_func_no_ = sdynparams.get<int>("STARTFUNCNO");
  }

  isinit_ = true;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataSDyn::setup()
{
  check_init();

  dynamic_rebalance_config_ =
      parse_dynamic_rebalance_config(get_sdyn_params().sublist("DYNAMIC REBALANCE"));

  if (dynamic_rebalance_config_.enabled &&
      (modeltypes_->size() != 1 || !modeltypes_->contains(Inpar::Solid::model_structure)))
  {
    FOUR_C_THROW("Dynamic rebalance is currently only supported for pure structure problems.");
  }

  std::set<Inpar::Solid::ModelType>::const_iterator it;
  // setup model type specific data containers
  for (it = (*modeltypes_).begin(); it != (*modeltypes_).end(); ++it)
  {
    switch (*it)
    {
      case Inpar::Solid::model_beaminteraction:
      case Inpar::Solid::model_beam_interaction_old:
      case Inpar::Solid::model_browniandyn:
      {
        periodic_boundingbox_ = std::make_shared<Core::Geo::MeshFree::BoundingBox>();
        periodic_boundingbox_->init(Global::Problem::instance()->binning_strategy_params());
        std::shared_ptr<Core::FE::Discretization> boundingbox_dis =
            Global::Problem::instance()->does_exist_dis("boundingbox")
                ? Global::Problem::instance()->get_dis("boundingbox")
                : nullptr;
        periodic_boundingbox_->setup(Global::Problem::instance()->io_params(), boundingbox_dis,
            Global::Problem::instance()->get_communicators().global_comm(),
            Global::Problem::instance()->n_dim(),
            *Global::Problem::instance()->output_control_file());
        break;
      }
      default:
      {
        // nothing to do
        break;
      }
    }
  }

  issetup_ = true;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::BaseDataSDyn::get_res_tolerance(
    const NOX::Nln::StatusTest::QuantityType& qtype) const
{
  check_init_setup();
  switch (qtype)
  {
    case NOX::Nln::StatusTest::quantity_structure:
      return tol_fres_;
      break;
    case NOX::Nln::StatusTest::quantity_contact_normal:
    case NOX::Nln::StatusTest::quantity_contact_friction:
    case NOX::Nln::StatusTest::quantity_meshtying:
      return tol_contact_res_;
      break;
    case NOX::Nln::StatusTest::quantity_cardiovascular0d:
      return tol_cardvasc0d_res_;
      break;
    case NOX::Nln::StatusTest::quantity_lag_pen_constraint:
      return tol_constr_res_;
      break;
    case NOX::Nln::StatusTest::quantity_pressure:
      return tol_inco_;
      break;
    case NOX::Nln::StatusTest::quantity_eas:
      return tol_eas_res_;
      break;
    default:
      FOUR_C_THROW(
          "There is no residual tolerance for the given quantity type! "
          "(quantity: {})",
          NOX::Nln::StatusTest::quantity_type_to_string(qtype).c_str());
      break;
  }

  return -1.0;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::BaseDataSDyn::get_incr_tolerance(
    const NOX::Nln::StatusTest::QuantityType& qtype) const
{
  check_init_setup();
  switch (qtype)
  {
    case NOX::Nln::StatusTest::quantity_structure:
      return tol_disp_incr_;
      break;
    case NOX::Nln::StatusTest::quantity_contact_normal:
    case NOX::Nln::StatusTest::quantity_contact_friction:
    case NOX::Nln::StatusTest::quantity_meshtying:
      return tol_contact_lm_incr_;
      break;
    case NOX::Nln::StatusTest::quantity_cardiovascular0d:
      return tol_cardvasc0d_incr_;
      break;
    case NOX::Nln::StatusTest::quantity_lag_pen_constraint:
      return tol_constr_incr_;
      break;
    case NOX::Nln::StatusTest::quantity_pressure:
      return tol_pres_;
      break;
    default:
      FOUR_C_THROW(
          "There is no increment tolerance for the given quantity type! "
          "(quantity: {})",
          NOX::Nln::StatusTest::quantity_type_to_string(qtype).c_str());
      break;
  }

  return -1.0;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::ConvNorm Solid::TimeInt::BaseDataSDyn::get_res_tolerance_type(
    const NOX::Nln::StatusTest::QuantityType& qtype) const
{
  check_init_setup();
  switch (qtype)
  {
    case NOX::Nln::StatusTest::quantity_structure:
      return toltype_fres_;
      break;
    case NOX::Nln::StatusTest::quantity_contact_normal:
    case NOX::Nln::StatusTest::quantity_contact_friction:
    case NOX::Nln::StatusTest::quantity_meshtying:
      return toltype_contact_res_;
      break;
    case NOX::Nln::StatusTest::quantity_cardiovascular0d:
      return toltype_cardvasc0d_res_;
      break;
    case NOX::Nln::StatusTest::quantity_lag_pen_constraint:
      return toltype_constr_res_;
      break;
    case NOX::Nln::StatusTest::quantity_pressure:
      return toltype_inco_;
      break;
    case NOX::Nln::StatusTest::quantity_eas:
      return toltype_eas_res_;
      break;
    default:
      FOUR_C_THROW(
          "There is no residual tolerance type for the given quantity type! "
          "(quantity: {})",
          NOX::Nln::StatusTest::quantity_type_to_string(qtype).c_str());
      break;
  }

  return Inpar::Solid::convnorm_abs;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::ConvNorm Solid::TimeInt::BaseDataSDyn::get_incr_tolerance_type(
    const NOX::Nln::StatusTest::QuantityType& qtype) const
{
  check_init_setup();
  switch (qtype)
  {
    case NOX::Nln::StatusTest::quantity_structure:
      return toltype_disp_incr_;
      break;
    case NOX::Nln::StatusTest::quantity_contact_normal:
    case NOX::Nln::StatusTest::quantity_contact_friction:
    case NOX::Nln::StatusTest::quantity_meshtying:
      return toltype_contact_lm_incr_;
      break;
    case NOX::Nln::StatusTest::quantity_cardiovascular0d:
      return toltype_cardvasc0d_incr_;
      break;
    case NOX::Nln::StatusTest::quantity_lag_pen_constraint:
      return toltype_constr_incr_;
      break;
    case NOX::Nln::StatusTest::quantity_pressure:
      return toltype_pres_;
      break;
    case NOX::Nln::StatusTest::quantity_eas:
      return toltype_eas_incr_;
      break;
    default:
      FOUR_C_THROW(
          "There is no increment tolerance type for the given quantity type! "
          "(quantity: {})",
          NOX::Nln::StatusTest::quantity_type_to_string(qtype).c_str());
      break;
  }

  return Inpar::Solid::convnorm_abs;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::BinaryOp Solid::TimeInt::BaseDataSDyn::get_res_combo_type(
    const NOX::Nln::StatusTest::QuantityType& qtype) const
{
  return get_res_combo_type(NOX::Nln::StatusTest::quantity_structure, qtype);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::BinaryOp Solid::TimeInt::BaseDataSDyn::get_res_combo_type(
    const NOX::Nln::StatusTest::QuantityType& qtype_1,
    const NOX::Nln::StatusTest::QuantityType& qtype_2) const
{
  check_init_setup();
  // combination: STRUCTURE <--> PRESSURE
  if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
          qtype_2 == NOX::Nln::StatusTest::quantity_pressure) or
      (qtype_1 == NOX::Nln::StatusTest::quantity_pressure and
          qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_fres_inco_;
  // combination: STRUCTURE <--> EAS
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_eas) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_eas and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_fres_eas_res_;
  // combination: STRUCTURE <--> CONTACT
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_contact_normal) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_contact_normal and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_fres_contact_res_;
  // combination: STRUCTURE <--> frictional CONTACT
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_contact_friction) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_contact_friction and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_fres_contact_res_;
  // combination: STRUCTURE <--> mesh tying
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_meshtying) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_meshtying and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_fres_contact_res_;
  // combination: STRUCTURE <--> CARDIOVASCULAR0D
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_cardiovascular0d) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_cardiovascular0d and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_fres_cardvasc0d_res_;
  // combination: STRUCTURE <--> LAG-PEN-CONSTRAINT
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_lag_pen_constraint) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_lag_pen_constraint and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_fres_constr_res_;
  // no combination was found
  else
    FOUR_C_THROW(
        "There is no combination type for the given quantity types! "
        "(quantity_1: {}, quantity_2: {})",
        NOX::Nln::StatusTest::quantity_type_to_string(qtype_1).c_str(),
        NOX::Nln::StatusTest::quantity_type_to_string(qtype_2).c_str());

  return Inpar::Solid::bop_and;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::BinaryOp Solid::TimeInt::BaseDataSDyn::get_incr_combo_type(
    const NOX::Nln::StatusTest::QuantityType& qtype) const
{
  return get_incr_combo_type(NOX::Nln::StatusTest::quantity_structure, qtype);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::BinaryOp Solid::TimeInt::BaseDataSDyn::get_incr_combo_type(
    const NOX::Nln::StatusTest::QuantityType& qtype_1,
    const NOX::Nln::StatusTest::QuantityType& qtype_2) const
{
  check_init_setup();
  // combination: STRUCTURE <--> PRESSURE
  if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
          qtype_2 == NOX::Nln::StatusTest::quantity_pressure) or
      (qtype_1 == NOX::Nln::StatusTest::quantity_pressure and
          qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_disp_pres_;
  // combination: STRUCTURE <--> EAS
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_eas) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_eas and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_disp_eas_incr_;
  // combination: STRUCTURE <--> CONTACT
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_contact_normal) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_contact_normal and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_disp_contact_lm_incr_;
  // combination: STRUCTURE <--> frictional CONTACT
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_contact_friction) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_contact_friction and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_disp_contact_lm_incr_;
  // combination: STRUCTURE <--> mesh tying
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_meshtying) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_meshtying and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_disp_contact_lm_incr_;
  // combination: STRUCTURE <--> CARDIOVASCULAR0D
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_cardiovascular0d) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_cardiovascular0d and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_disp_cardvasc0d_incr_;
  // combination: STRUCTURE <--> LAG-PEN-CONSTRAINT
  else if ((qtype_1 == NOX::Nln::StatusTest::quantity_structure and
               qtype_2 == NOX::Nln::StatusTest::quantity_lag_pen_constraint) or
           (qtype_1 == NOX::Nln::StatusTest::quantity_lag_pen_constraint and
               qtype_2 == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_disp_constr_incr_;
  // no combination was found
  else
    FOUR_C_THROW(
        "There is no combination type for the given quantity types! "
        "(quantity_1: {}, quantity_2: {})",
        NOX::Nln::StatusTest::quantity_type_to_string(qtype_1).c_str(),
        NOX::Nln::StatusTest::quantity_type_to_string(qtype_2).c_str());

  return Inpar::Solid::bop_and;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::BinaryOp Solid::TimeInt::BaseDataSDyn::get_res_incr_combo_type(
    const NOX::Nln::StatusTest::QuantityType& qtype_res,
    const NOX::Nln::StatusTest::QuantityType& qtype_incr) const
{
  check_init_setup();
  // combination: STRUCTURE (force/res) <--> STRUCTURE (displ/incr)
  if ((qtype_res == NOX::Nln::StatusTest::quantity_structure and
          qtype_incr == NOX::Nln::StatusTest::quantity_structure))
    return normcombo_fres_disp_;
  // no combination was found
  else
    FOUR_C_THROW(
        "There is no res-incr-combination type for the given quantity types! "
        "(quantity_res: {}, quantity_incr: {})",
        NOX::Nln::StatusTest::quantity_type_to_string(qtype_res).c_str(),
        NOX::Nln::StatusTest::quantity_type_to_string(qtype_incr).c_str());

  return Inpar::Solid::bop_and;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::TimeInt::BaseDataSDyn::have_model_type(const Inpar::Solid::ModelType& modeltype) const
{
  check_init_setup();
  return (get_model_types().find(modeltype) != get_model_types().end());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::TimeInt::BaseDataSDyn::have_ele_tech(const Inpar::Solid::EleTech& eletech) const
{
  check_init_setup();
  return (get_element_technologies().find(eletech) != get_element_technologies().end());
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::GenAlphaDataSDyn::GenAlphaDataSDyn()
    : midavg_(Inpar::Solid::midavg_vague),
      beta_(-1.0),
      gamma_(-1.0),
      alphaf_(-1.0),
      alpham_(-1.0),
      rhoinf_(-1.0)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::GenAlphaDataSDyn::setup()
{
  check_init();

  // call base class setup
  Solid::TimeInt::BaseDataSDyn::setup();

  midavg_ = Teuchos::getIntegralValue<Inpar::Solid::MidAverageEnum>(
      get_sdyn_params().sublist("GENALPHA"), "GENAVG");
  beta_ = get_sdyn_params().sublist("GENALPHA").get<double>("BETA");
  gamma_ = get_sdyn_params().sublist("GENALPHA").get<double>("GAMMA");
  alphaf_ = get_sdyn_params().sublist("GENALPHA").get<double>("ALPHA_F");
  alpham_ = get_sdyn_params().sublist("GENALPHA").get<double>("ALPHA_M");
  rhoinf_ = get_sdyn_params().sublist("GENALPHA").get<double>("RHO_INF");

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::OneStepThetaDataSDyn::OneStepThetaDataSDyn() : theta_(-1.0)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::OneStepThetaDataSDyn::setup()
{
  check_init();

  // call base class setup
  Solid::TimeInt::BaseDataSDyn::setup();

  theta_ = get_sdyn_params().sublist("ONESTEPTHETA").get<double>("THETA");

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::ExplEulerDataSDyn::ExplEulerDataSDyn() : modexpleuler_(true)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::ExplEulerDataSDyn::setup()
{
  check_init();

  // call base class setup
  Solid::TimeInt::BaseDataSDyn::setup();

  modexpleuler_ =
      Global::Problem::instance()->structural_dynamic_params().get<bool>("MODIFIEDEXPLEULER");

  issetup_ = true;
}

FOUR_C_NAMESPACE_CLOSE
