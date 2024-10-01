/*----------------------------------------------------------------------*/
/*! \file

\brief Monolithic coupling of 3D structure and 0D cardiovascular flow models

\level 2


*----------------------------------------------------------------------*/

#include "4C_cardiovascular0d_manager.hpp"

#include "4C_adapter_str_structure.hpp"
#include "4C_cardiovascular0d.hpp"
#include "4C_cardiovascular0d_4elementwindkessel.hpp"
#include "4C_cardiovascular0d_arterialproxdist.hpp"
#include "4C_cardiovascular0d_dofset.hpp"
#include "4C_cardiovascular0d_mor_pod.hpp"
#include "4C_cardiovascular0d_respiratory_syspulperiphcirculation.hpp"
#include "4C_cardiovascular0d_resulttest.hpp"
#include "4C_cardiovascular0d_syspulcirculation.hpp"
#include "4C_fem_condition.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_utils_parameter_list.hpp"

#include <stdio.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                              mhv 11/13|
 *----------------------------------------------------------------------*/
UTILS::Cardiovascular0DManager::Cardiovascular0DManager(
    Teuchos::RCP<Core::FE::Discretization> discr,
    Teuchos::RCP<const Core::LinAlg::Vector<double>> disp, Teuchos::ParameterList strparams,
    Teuchos::ParameterList cv0dparams, Core::LinAlg::Solver& solver,
    Teuchos::RCP<FourC::Cardiovascular0D::ProperOrthogonalDecomposition> mor)
    : actdisc_(discr),
      myrank_(actdisc_->get_comm().MyPID()),
      dbcmaps_(Teuchos::rcp(new Core::LinAlg::MapExtractor())),
      cardiovascular0ddofset_full_(Teuchos::null),
      cardiovascular0dmap_full_(Teuchos::null),
      redcardiovascular0dmap_(Teuchos::null),
      cardvasc0dimpo_(Teuchos::null),
      cv0ddofincrement_(Teuchos::null),
      cv0ddof_n_(Teuchos::null),
      cv0ddof_np_(Teuchos::null),
      cv0ddof_m_(Teuchos::null),
      dcv0ddof_m_(Teuchos::null),
      v_n_(Teuchos::null),
      v_np_(Teuchos::null),
      v_m_(Teuchos::null),
      cv0ddof_t_n_(Teuchos::null),
      cv0ddof_t_np_(Teuchos::null),
      cardvasc0d_res_m_(Teuchos::null),
      cardvasc0d_df_n_(Teuchos::null),
      cardvasc0d_df_np_(Teuchos::null),
      cardvasc0d_df_m_(Teuchos::null),
      cardvasc0d_f_n_(Teuchos::null),
      cardvasc0d_f_np_(Teuchos::null),
      cardvasc0d_f_m_(Teuchos::null),
      t_period_(cv0dparams.get("T_PERIOD", -1.0)),
      eps_periodic_(cv0dparams.get("EPS_PERIODIC", 1.0e-16)),
      is_periodic_(false),
      cycle_error_(1.0),
      num_cardiovascular0_did_(0),
      cardiovascular0_did_(0),
      offset_id_(10000),
      current_id_(false),
      havecardiovascular0d_(false),
      cardvasc0d_model_(
          Teuchos::rcp(new Cardiovascular0D4ElementWindkessel(actdisc_, "", current_id_))),
      cardvasc0d_4elementwindkessel_(Teuchos::rcp(new Cardiovascular0D4ElementWindkessel(
          actdisc_, "Cardiovascular0D4ElementWindkesselStructureCond", current_id_))),
      cardvasc0d_arterialproxdist_(Teuchos::rcp(new Cardiovascular0DArterialProxDist(
          actdisc_, "Cardiovascular0DArterialProxDistStructureCond", current_id_))),
      cardvasc0d_syspulcirculation_(Teuchos::rcp(new Cardiovascular0DSysPulCirculation(
          actdisc_, "Cardiovascular0DSysPulCirculationStructureCond", current_id_))),
      cardvascrespir0d_syspulperiphcirculation_(
          Teuchos::rcp(new CardiovascularRespiratory0DSysPulPeriphCirculation(actdisc_,
              "CardiovascularRespiratory0DSysPulPeriphCirculationStructureCond", current_id_))),
      solver_(Teuchos::null),
      cardiovascular0dstiffness_(Teuchos::null),
      mat_dcardvasc0d_dd_(Teuchos::null),
      mat_dstruct_dcv0ddof_(Teuchos::null),
      counter_(0),
      isadapttol_(false),
      adaptolbetter_(0.01),
      tolres_struct_(strparams.get("TOLRES", 1.0e-8)),
      tolres_cardvasc0d_(cv0dparams.get("TOL_CARDVASC0D_RES", 1.0e-8)),
      algochoice_(Teuchos::getIntegralValue<Inpar::Cardiovascular0D::Cardvasc0DSolveAlgo>(
          cv0dparams, "SOLALGORITHM")),
      dirichtoggle_(Teuchos::null),
      zeros_(Core::LinAlg::create_vector(*(actdisc_->dof_row_map()), true)),
      theta_(cv0dparams.get("TIMINT_THETA", 0.5)),
      enhanced_output_(cv0dparams.get<bool>("ENHANCED_OUTPUT")),
      ptc_3d0d_(cv0dparams.get<bool>("PTC_3D0D")),
      k_ptc_(cv0dparams.get("K_PTC", 0.0)),
      totaltime_(0.0),
      linsolveerror_(0),
      strparams_(strparams),
      cv0dparams_(cv0dparams),
      intstrat_(
          Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(strparams, "INT_STRATEGY")),
      mor_(mor),
      have_mor_(false)
{
  // Check what kind of Cardiovascular0D boundary conditions there are
  havecardiovascular0d_ = (cardvasc0d_4elementwindkessel_->have_cardiovascular0_d() or
                           cardvasc0d_arterialproxdist_->have_cardiovascular0_d() or
                           cardvasc0d_syspulcirculation_->have_cardiovascular0_d() or
                           cardvascrespir0d_syspulperiphcirculation_->have_cardiovascular0_d());

  if (!havecardiovascular0d_) return;


  switch (intstrat_)
  {
    case Inpar::Solid::int_standard:
      break;
    case Inpar::Solid::int_old:
      // setup solver
      solver_setup(solver, strparams);
      break;
    default:
      FOUR_C_THROW("Unknown integration strategy!");
      break;
  }

  // Map containing Dirichlet DOFs
  {
    Teuchos::ParameterList p;
    const double time = 0.0;
    p.set("total time", time);
    p.set<const Core::UTILS::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());
    actdisc_->evaluate_dirichlet(p, zeros_, Teuchos::null, Teuchos::null, Teuchos::null, dbcmaps_);
    zeros_->PutScalar(0.0);  // just in case of change
  }

  if (cardvasc0d_4elementwindkessel_->have_cardiovascular0_d())
  {
    cardvasc0d_model_ = cardvasc0d_4elementwindkessel_;
    // dof vector for ONE 0D cardiovascular condition of this type: [p  q  s]^T
    num_cardiovascular0_did_ =
        3 * cardvasc0d_4elementwindkessel_->get_cardiovascular0_d_condition().size();
  }
  if (cardvasc0d_arterialproxdist_->have_cardiovascular0_d())
  {
    cardvasc0d_model_ = cardvasc0d_arterialproxdist_;
    // dof vector for ONE 0D cardiovascular condition of this type: [p_v  p_arp  q_arp  p_ard]^T
    num_cardiovascular0_did_ =
        4 * cardvasc0d_arterialproxdist_->get_cardiovascular0_d_condition().size();
  }
  if (cardvasc0d_syspulcirculation_->have_cardiovascular0_d())
  {
    cardvasc0d_model_ = cardvasc0d_syspulcirculation_;
    // dof vector for 0D cardiovascular condition of this type:
    // [p_at_l  q_vin_l  q_vout_l  p_v_l  p_ar_sys  q_ar_sys  p_ven_sys  q_ven_sys  p_at_r  q_vin_r
    // q_vout_r  p_v_r  p_ar_pul  q_ar_pul  p_ven_pul  q_ven_pul]^T
    num_cardiovascular0_did_ = 16;
  }

  if (cardvascrespir0d_syspulperiphcirculation_->have_cardiovascular0_d())
  {
    cardvasc0d_model_ = cardvascrespir0d_syspulperiphcirculation_;
    // set number of degrees of freedom
    switch (cardvasc0d_model_->get_respiratory_model())
    {
      case Inpar::Cardiovascular0D::resp_none:
        num_cardiovascular0_did_ = 34;
        break;
      case Inpar::Cardiovascular0D::resp_standard:
        num_cardiovascular0_did_ = 82;
        break;
      default:
        FOUR_C_THROW("Undefined respiratory_model!");
        break;
    }
  }

  // are we using model order reduction?
  if (mor_ != Teuchos::null)
    if (mor_->have_mor()) have_mor_ = true;

  if (cardvasc0d_4elementwindkessel_->have_cardiovascular0_d() or
      cardvasc0d_arterialproxdist_->have_cardiovascular0_d() or
      cardvasc0d_syspulcirculation_->have_cardiovascular0_d() or
      cardvascrespir0d_syspulperiphcirculation_->have_cardiovascular0_d())
  {
    cardiovascular0ddofset_ = Teuchos::rcp(new Cardiovascular0DDofSet());
    cardiovascular0ddofset_->assign_degrees_of_freedom(actdisc_, num_cardiovascular0_did_, 0, mor_);
    cardiovascular0ddofset_full_ = Teuchos::rcp(new Cardiovascular0DDofSet());
    cardiovascular0ddofset_full_->assign_degrees_of_freedom(
        actdisc_, num_cardiovascular0_did_, 0, Teuchos::null);
    offset_id_ = cardiovascular0ddofset_->first_gid();

    cardiovascular0dmap_full_ =
        Teuchos::rcp(new Epetra_Map(*(cardiovascular0ddofset_full_->dof_row_map())));
    cardiovascular0dmap_ = Teuchos::rcp(new Epetra_Map(*(cardiovascular0ddofset_->dof_row_map())));
    redcardiovascular0dmap_ = Core::LinAlg::allreduce_e_map(*cardiovascular0dmap_);
    cardvasc0dimpo_ =
        Teuchos::rcp(new Epetra_Export(*redcardiovascular0dmap_, *cardiovascular0dmap_));
    cv0ddofincrement_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cv0ddof_n_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cv0ddof_np_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cv0ddof_m_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    dcv0ddof_m_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    v_n_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    v_np_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    v_m_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cv0ddof_t_n_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cv0ddof_t_np_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cardvasc0d_res_m_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cardvasc0d_df_n_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cardvasc0d_df_np_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cardvasc0d_df_m_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cardvasc0d_f_n_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cardvasc0d_f_np_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));
    cardvasc0d_f_m_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*cardiovascular0dmap_));

    cardiovascular0dstiffness_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
        *cardiovascular0dmap_, num_cardiovascular0_did_, false, true));
    mat_dcardvasc0d_dd_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
        *(actdisc_->dof_row_map()), num_cardiovascular0_did_, false, true));
    mat_dstruct_dcv0ddof_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
        *(actdisc_->dof_row_map()), num_cardiovascular0_did_, false, true));

    Teuchos::ParameterList p;
    const double time = strparams.get<double>("total time", 0.0);
    const double sc_timint = strparams.get("scale_timint", 1.0);
    const double ts_size = strparams.get("time_step_size", 1.0);

    if ((theta_ <= 0.0) or (theta_ > 1.0))
      FOUR_C_THROW("theta for 0D cardiovascular model time integration out of range (0.0,1.0] !");

    // Initialize vectors
    actdisc_->clear_state();

    cv0ddofincrement_->PutScalar(0.0);

    cv0ddof_n_->PutScalar(0.0);
    cv0ddof_np_->PutScalar(0.0);
    cv0ddof_m_->PutScalar(0.0);
    dcv0ddof_m_->PutScalar(0.0);
    v_n_->PutScalar(0.0);
    v_np_->PutScalar(0.0);
    v_m_->PutScalar(0.0);
    cardvasc0d_res_m_->PutScalar(0.0);

    cardvasc0d_df_n_->PutScalar(0.0);
    cardvasc0d_df_np_->PutScalar(0.0);
    cardvasc0d_df_m_->PutScalar(0.0);
    cardvasc0d_f_n_->PutScalar(0.0);
    cardvasc0d_f_np_->PutScalar(0.0);
    cardvasc0d_f_m_->PutScalar(0.0);

    cv0ddof_t_n_->PutScalar(0.0);
    cv0ddof_t_np_->PutScalar(0.0);

    cardiovascular0dstiffness_->zero();

    p.set("total time", time);
    p.set("OffsetID", offset_id_);
    p.set("NumberofID", num_cardiovascular0_did_);
    p.set("scale_timint", sc_timint);
    p.set("time_step_size", ts_size);
    actdisc_->set_state("displacement", disp);

    Teuchos::RCP<Core::LinAlg::Vector<double>> v_n_red =
        Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
    Teuchos::RCP<Core::LinAlg::Vector<double>> v_n_red2 =
        Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
    Teuchos::RCP<Core::LinAlg::Vector<double>> cv0ddof_n_red =
        Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));

    // initialize everything
    cardvasc0d_model_->initialize(p, v_n_red, cv0ddof_n_red);

    v_n_->PutScalar(0.0);
    v_n_->Export(*v_n_red, *cardvasc0dimpo_, Add);

    cv0ddof_n_->Export(*cv0ddof_n_red, *cardvasc0dimpo_, Insert);


    Core::LinAlg::export_to(*v_n_, *v_n_red2);

    // evaluate initial 0D right-hand side at t_{n}
    Teuchos::RCP<Core::LinAlg::Vector<double>> cardvasc0d_df_n_red =
        Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
    Teuchos::RCP<Core::LinAlg::Vector<double>> cardvasc0d_f_n_red =
        Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
    cardvasc0d_model_->evaluate(p, Teuchos::null, Teuchos::null, Teuchos::null, cardvasc0d_df_n_red,
        cardvasc0d_f_n_red, Teuchos::null, cv0ddof_n_red, v_n_red2);

    // insert compartment volumes into vol vector
    v_n_->Export(*v_n_red2, *cardvasc0dimpo_, Insert);

    cardvasc0d_df_n_->PutScalar(0.0);
    cardvasc0d_df_n_->Export(*cardvasc0d_df_n_red, *cardvasc0dimpo_, Insert);
    cardvasc0d_f_n_->PutScalar(0.0);
    cardvasc0d_f_n_->Export(*cardvasc0d_f_n_red, *cardvasc0dimpo_, Insert);

    // predict with initial values
    cv0ddof_np_->Update(1.0, *cv0ddof_n_, 0.0);

    cardvasc0d_df_np_->Update(1.0, *cardvasc0d_df_n_, 0.0);
    cardvasc0d_f_np_->Update(1.0, *cardvasc0d_f_n_, 0.0);

    v_np_->Update(1.0, *v_n_, 0.0);

    cv0ddof_t_n_->Update(1.0, *cv0ddof_np_, 0.0);
    cv0ddof_t_np_->Update(1.0, *cv0ddof_np_, 0.0);


    // Create resulttest
    Teuchos::RCP<Core::UTILS::ResultTest> resulttest =
        Teuchos::rcp(new Cardiovascular0DResultTest(*this, actdisc_));

    // Resulttest for 0D problem
    Global::Problem::instance()->add_field_test(resulttest);
  }

  return;
}

/*-----------------------------------------------------------------------*
|(public)                                                       mhv 11/13|
|do all the time integration, evaluation and assembling of stiffnesses   |
|and right-hand sides                                                    |
 *-----------------------------------------------------------------------*/
void UTILS::Cardiovascular0DManager::evaluate_force_stiff(const double time,
    Teuchos::RCP<const Core::LinAlg::Vector<double>> disp,
    Teuchos::RCP<Core::LinAlg::Vector<double>> fint,
    Teuchos::RCP<Core::LinAlg::SparseOperator> stiff, Teuchos::ParameterList scalelist)
{
  const double sc_strtimint = scalelist.get("scale_timint", 1.0);
  const double ts_size = scalelist.get("time_step_size", 1.0);

  // create the parameters for the discretization
  Teuchos::ParameterList p;
  const Epetra_Map* dofrowmap = actdisc_->dof_row_map();

  cardiovascular0dstiffness_->zero();
  mat_dcardvasc0d_dd_->zero();
  mat_dstruct_dcv0ddof_->zero();

  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("OffsetID", offset_id_);
  p.set("NumberofID", num_cardiovascular0_did_);
  p.set("new disp", disp);
  p.set("scale_timint", sc_strtimint);
  p.set("scale_theta", theta_);
  p.set("time_step_size", ts_size);

  totaltime_ = time;
  Teuchos::RCP<Core::LinAlg::Vector<double>> v_np_red =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
  Teuchos::RCP<Core::LinAlg::Vector<double>> v_np_red2 =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
  Teuchos::RCP<Core::LinAlg::Vector<double>> cv0ddof_np_red =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
  Teuchos::RCP<Core::LinAlg::Vector<double>> cardvasc0d_df_np_red =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
  Teuchos::RCP<Core::LinAlg::Vector<double>> cardvasc0d_f_np_red =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));

  actdisc_->clear_state();
  actdisc_->set_state("displacement", disp);

  // evaluate current 3D volume only
  cardvasc0d_model_->evaluate(p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null,
      Teuchos::null, v_np_red, Teuchos::null, Teuchos::null);

  // import into vol vector at end-point
  v_np_->PutScalar(0.0);
  v_np_->Export(*v_np_red, *cardvasc0dimpo_, Add);

  // solution and rate of solution at generalized mid-point t_{n+theta}
  // for post-processing only - residual midpoint evaluation done separately!
  cv0ddof_m_->Update(theta_, *cv0ddof_np_, 1. - theta_, *cv0ddof_n_, 0.0);
  dcv0ddof_m_->Update(1. / ts_size, *cv0ddof_np_, -1. / ts_size, *cv0ddof_n_, 0.0);

  // export end-point values
  Core::LinAlg::export_to(*cv0ddof_np_, *cv0ddof_np_red);
  Core::LinAlg::export_to(*v_np_, *v_np_red2);

  // assemble Cardiovascular0D stiffness and offdiagonal coupling matrices as well as rhs
  // contributions
  cardvasc0d_model_->evaluate(p, cardiovascular0dstiffness_, mat_dcardvasc0d_dd_,
      mat_dstruct_dcv0ddof_, cardvasc0d_df_np_red, cardvasc0d_f_np_red, Teuchos::null,
      cv0ddof_np_red, v_np_red2);

  // insert compartment volumes into vol vector
  v_np_->Export(*v_np_red2, *cardvasc0dimpo_, Insert);

  // volume at generalized mid-point t_{n+theta} - for post-processing only
  v_m_->Update(theta_, *v_np_, 1. - theta_, *v_n_, 0.0);

  cardvasc0d_df_np_->PutScalar(0.0);
  cardvasc0d_df_np_->Export(*cardvasc0d_df_np_red, *cardvasc0dimpo_, Insert);
  cardvasc0d_f_np_->PutScalar(0.0);
  cardvasc0d_f_np_->Export(*cardvasc0d_f_np_red, *cardvasc0dimpo_, Insert);
  // df_m = (df_np - df_n) / dt
  cardvasc0d_df_m_->Update(1. / ts_size, *cardvasc0d_df_np_, -1. / ts_size, *cardvasc0d_df_n_, 0.0);
  // f_m = theta * f_np + (1-theta) * f_n
  cardvasc0d_f_m_->Update(theta_, *cardvasc0d_f_np_, 1. - theta_, *cardvasc0d_f_n_, 0.0);
  // total 0D residual r_m = df_m + f_m
  cardvasc0d_res_m_->Update(1., *cardvasc0d_df_m_, 1., *cardvasc0d_f_m_, 0.0);

  // Complete matrices
  cardiovascular0dstiffness_->complete(*cardiovascular0dmap_, *cardiovascular0dmap_);
  mat_dcardvasc0d_dd_->complete(*cardiovascular0dmap_, *dofrowmap);
  mat_dstruct_dcv0ddof_->complete(*cardiovascular0dmap_, *dofrowmap);

  // ATTENTION: We necessarily need the end-point and NOT the generalized mid-point pressure here
  // since fint will be set to the generalized mid-point by the respective structural
  // time-integrator!
  // Core::LinAlg::export_to(*cv0ddof_np_,*cv0ddof_np_red);
  evaluate_neumann_cardiovascular0_d_coupling(p, cv0ddof_np_red, fint, stiff);

  return;
}

void UTILS::Cardiovascular0DManager::update_time_step()
{
  if (t_period_ > 0.0 and modulo_is_realtive_zero(totaltime_, t_period_, totaltime_))
  {
    cv0ddof_t_np_->Update(1.0, *cv0ddof_np_, 0.0);
    check_periodic();
    cv0ddof_t_n_->Update(1.0, *cv0ddof_t_np_, 0.0);
  }

  cv0ddof_n_->Update(1.0, *cv0ddof_np_, 0.0);
  v_n_->Update(1.0, *v_np_, 0.0);

  cardvasc0d_df_n_->Update(1.0, *cardvasc0d_df_np_, 0.0);
  cardvasc0d_f_n_->Update(1.0, *cardvasc0d_f_np_, 0.0);

  if (t_period_ > 0.0) printf("Cycle error (error in periodicity): %10.6e \n", cycle_error_);

  if (is_periodic_)
  {
    if (actdisc_->get_comm().MyPID() == 0)
      std::cout << "============ PERIODIC STATE REACHED ! ============" << std::endl;
  }

  return;
}

void UTILS::Cardiovascular0DManager::check_periodic()  // not yet thoroughly tested!
{
  Teuchos::RCP<Core::LinAlg::Vector<double>> cv0ddof_T_N_red =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
  Teuchos::RCP<Core::LinAlg::Vector<double>> cv0ddof_T_NP_red =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
  Core::LinAlg::export_to(*cv0ddof_t_n_, *cv0ddof_T_N_red);
  Core::LinAlg::export_to(*cv0ddof_t_np_, *cv0ddof_T_NP_red);

  std::vector<double> vals;
  for (int j = 0; j < num_cardiovascular0_did_; j++)
  {
    //    if(j<34 or j>53) // exclude oscillatory lung dofs
    vals.push_back(fabs(
        ((*cv0ddof_T_NP_red)[j] - (*cv0ddof_T_N_red)[j]) / fmax(1.0, fabs((*cv0ddof_T_N_red)[j]))));
    //      vals.push_back( fabs(
    //      ((*cv0ddof_T_NP_red)[j]-(*cv0ddof_T_N_red)[j])/fabs((*cv0ddof_T_N_red)[j]) ) );
  }

  cycle_error_ = *std::max_element(vals.begin(), vals.end());


  if (cycle_error_ <= eps_periodic_)
    is_periodic_ = true;
  else
    is_periodic_ = false;

  return;
}


/*----------------------------------------------------------------------*
 | Compare if two doubles are relatively equal               Thon 08/15 |
 *----------------------------------------------------------------------*/
bool UTILS::Cardiovascular0DManager::is_realtive_equal_to(
    const double A, const double B, const double Ref)
{
  return ((fabs(A - B) / Ref) < 1e-12);
}

/*----------------------------------------------------------------------*
 | Compare if A mod B is relatively equal to zero            Thon 08/15 |
 *----------------------------------------------------------------------*/
bool UTILS::Cardiovascular0DManager::modulo_is_realtive_zero(
    const double value, const double modulo, const double Ref)
{
  return is_realtive_equal_to(fmod(value + modulo / 2, modulo) - modulo / 2, 0.0, Ref);
}



void UTILS::Cardiovascular0DManager::reset_step()
{
  cv0ddof_np_->Update(1.0, *cv0ddof_n_, 0.0);
  v_np_->Update(1.0, *v_n_, 0.0);

  cardvasc0d_df_np_->Update(1.0, *cardvasc0d_df_n_, 0.0);
  cardvasc0d_f_np_->Update(1.0, *cardvasc0d_f_n_, 0.0);

  return;
}

/*----------------------------------------------------------------------*/
/* iterative iteration update of state */
void UTILS::Cardiovascular0DManager::update_cv0_d_dof(
    Teuchos::RCP<Core::LinAlg::Vector<double>> cv0ddofincrement)
{
  // new end-point solution
  // cv0ddof_{n+1}^{i+1} := cv0ddof_{n+1}^{i} + Inccv0ddof_{n+1}^{i}
  cv0ddof_np_->Update(1.0, *cv0ddofincrement, 1.0);

  return;
}

/*----------------------------------------------------------------------*
|(public)                                                      mhv 03/15|
|Read restart information                                               |
 *-----------------------------------------------------------------------*/
void UTILS::Cardiovascular0DManager::read_restart(
    Core::IO::DiscretizationReader& reader, const double& time)
{
  // check if restart from non-Cardiovascular0D simulation is desired
  const bool restartwithcardiovascular0d =
      cardvasc0_d_params().get<bool>("RESTART_WITH_CARDVASC0D");

  if (!restartwithcardiovascular0d)
  {
    Teuchos::RCP<Epetra_Map> cardvasc0d = get_cardiovascular0_d_map();
    Teuchos::RCP<Core::LinAlg::Vector<double>> tempvec =
        Core::LinAlg::create_vector(*cardvasc0d, true);
    // old rhs contributions
    reader.read_vector(tempvec, "cv0d_df_np");
    set0_d_df_n(tempvec);
    reader.read_vector(tempvec, "cv0d_f_np");
    set0_d_f_n(tempvec);
    // old dof and vol vector
    reader.read_vector(tempvec, "cv0d_dof_np");
    set0_d_dof_n(tempvec);
    reader.read_vector(tempvec, "vol_np");
    set0_d_v_n(tempvec);
  }

  totaltime_ = time;

  if (restartwithcardiovascular0d) print_pres_flux(true);

  return;
}

/*----------------------------------------------------------------------*/
void UTILS::Cardiovascular0DManager::evaluate_neumann_cardiovascular0_d_coupling(
    Teuchos::ParameterList params, const Teuchos::RCP<Core::LinAlg::Vector<double>> actpres,
    Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector,
    Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix)
{
  const bool assvec = systemvector != Teuchos::null;
  const bool assmat = systemmatrix != Teuchos::null;

  std::vector<Core::Conditions::Condition*> surfneumcond;
  std::vector<Core::Conditions::Condition*> cardvasc0dstructcoupcond;
  std::vector<int> tmp;
  Teuchos::RCP<Core::FE::Discretization> structdis =
      Global::Problem::instance()->get_dis("structure");
  if (structdis == Teuchos::null) FOUR_C_THROW("No structure discretization available!");

  // get all coupling conditions on structure
  structdis->get_condition("SurfaceNeumannCardiovascular0D", cardvasc0dstructcoupcond);
  unsigned int numcoupcond = cardvasc0dstructcoupcond.size();
  if (numcoupcond == 0) FOUR_C_THROW("No coupling conditions found!");

  // fill the i-sorted wk coupling conditions vector with the id-sorted values of the wk pressure
  // vector, at the respective coupling_id
  for (unsigned int i = 0; i < numcoupcond; ++i)
  {
    int id_strcoupcond = cardvasc0dstructcoupcond[i]->parameters().get<int>("coupling_id");

    Core::Conditions::Condition* coupcond = cardvasc0dstructcoupcond[i];
    std::vector<double> newval(6, 0.0);
    if (cardvasc0d_4elementwindkessel_->have_cardiovascular0_d())
      newval[0] = -(*actpres)[3 * id_strcoupcond];
    if (cardvasc0d_arterialproxdist_->have_cardiovascular0_d())
      newval[0] = -(*actpres)[4 * id_strcoupcond];

    if (cardvasc0d_syspulcirculation_->have_cardiovascular0_d())
    {
      for (unsigned int j = 0;
           j < cardvasc0d_syspulcirculation_->get_cardiovascular0_d_condition().size(); ++j)
      {
        Core::Conditions::Condition& cond =
            *(cardvasc0d_syspulcirculation_->get_cardiovascular0_d_condition()[j]);
        int id_cardvasc0d = cond.parameters().get<int>("id");

        if (id_strcoupcond == id_cardvasc0d)
        {
          const std::string& conditiontype =
              cardvasc0d_syspulcirculation_->get_cardiovascular0_d_condition()[j]
                  ->parameters()
                  .get<std::string>("TYPE");
          if (conditiontype == "ventricle_left") newval[0] = -(*actpres)[3];
          if (conditiontype == "ventricle_right") newval[0] = -(*actpres)[11];
          if (conditiontype == "atrium_left") newval[0] = -(*actpres)[0];
          if (conditiontype == "atrium_right") newval[0] = -(*actpres)[8];
          if (conditiontype == "dummy") newval[0] = 0.;
        }
      }
    }

    if (cardvascrespir0d_syspulperiphcirculation_->have_cardiovascular0_d())
    {
      for (unsigned int j = 0;
           j < cardvascrespir0d_syspulperiphcirculation_->get_cardiovascular0_d_condition().size();
           ++j)
      {
        Core::Conditions::Condition& cond =
            *(cardvascrespir0d_syspulperiphcirculation_->get_cardiovascular0_d_condition()[j]);
        int id_cardvasc0d = cond.parameters().get<int>("id");

        if (id_strcoupcond == id_cardvasc0d)
        {
          const std::string conditiontype =
              cardvascrespir0d_syspulperiphcirculation_->get_cardiovascular0_d_condition()[j]
                  ->parameters()
                  .get<std::string>("TYPE");
          if (conditiontype == "ventricle_left") newval[0] = -(*actpres)[3];
          if (conditiontype == "ventricle_right") newval[0] = -(*actpres)[27];
          if (conditiontype == "atrium_left") newval[0] = -(*actpres)[0];
          if (conditiontype == "atrium_right") newval[0] = -(*actpres)[24];
          if (conditiontype == "dummy") newval[0] = 0.;
        }
      }
    }
    if (assvec) coupcond->parameters().add("VAL", newval);


    Teuchos::RCP<const Core::LinAlg::Vector<double>> disp =
        params.get<Teuchos::RCP<const Core::LinAlg::Vector<double>>>("new disp");
    actdisc_->set_state("displacement new", disp);

    Core::LinAlg::SerialDenseVector elevector;
    Core::LinAlg::SerialDenseMatrix elematrix;
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& geom = coupcond->geometry();

    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator curr;
    for (curr = geom.begin(); curr != geom.end(); ++curr)
    {
      // get element location vector, dirichlet flags and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      curr->second->location_vector(*actdisc_, lm, lmowner, lmstride);
      elevector.size((int)lm.size());

      const int size = (int)lm.size();
      if (elematrix.numRows() != size)
        elematrix.shape(size, size);
      else
        elematrix.putScalar(0.0);
      curr->second->evaluate_neumann(params, *actdisc_, *coupcond, lm, elevector, &elematrix);
      // minus sign here since we sum into fint_ !!
      elevector.scale(-1.0);
      if (assvec) Core::LinAlg::assemble(*systemvector, elevector, lm, lmowner);
      // plus sign here since evaluate_neumann already assumes that an fext vector enters, and thus
      // puts a minus infront of the load linearization matrix !!
      // elematrix.Scale(1.0);
      if (assmat) systemmatrix->assemble(curr->second->id(), lmstride, elematrix, lm, lmowner);
    }
  }

  return;
}


void UTILS::Cardiovascular0DManager::print_pres_flux(bool init) const
{
  // prepare stuff for printing to screen
  // ATTENTION: we print the mid-point pressure (NOT the end-point pressure at t_{n+1}),
  // since this is the one where mechanical equilibrium is guaranteed
  Teuchos::RCP<Core::LinAlg::Vector<double>> cv0ddof_m_red =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
  Teuchos::RCP<Core::LinAlg::Vector<double>> dcv0ddof_m_red =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
  Teuchos::RCP<Core::LinAlg::Vector<double>> v_m_red =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
  Teuchos::RCP<Core::LinAlg::Vector<double>> cv0ddof_np_red =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*redcardiovascular0dmap_));
  if (init)
  {
    Core::LinAlg::export_to(*cv0ddof_n_, *cv0ddof_m_red);
    Core::LinAlg::export_to(*v_n_, *v_m_red);
  }
  else
  {
    Core::LinAlg::export_to(*cv0ddof_m_, *cv0ddof_m_red);
    Core::LinAlg::export_to(*v_m_, *v_m_red);
  }

  Core::LinAlg::export_to(*dcv0ddof_m_, *dcv0ddof_m_red);

  Core::LinAlg::export_to(*cv0ddof_n_, *cv0ddof_np_red);

  if (myrank_ == 0)
  {
    for (unsigned int i = 0; i < current_id_.size(); ++i)
    {
      if (cardvasc0d_4elementwindkessel_->have_cardiovascular0_d())
      {
        printf("Cardiovascular0D output id%2d:\n", current_id_[i]);
        printf("%2d p: %10.16e \n", current_id_[i], (*cv0ddof_m_red)[3 * i]);
        printf("%2d V: %10.16e \n", current_id_[i], (*v_m_red)[3 * i]);
      }
      if (cardvasc0d_arterialproxdist_->have_cardiovascular0_d())
      {
        printf("Cardiovascular0D output id%2d:\n", current_id_[i]);
        printf("%2d p_v: %10.16e \n", current_id_[i], (*cv0ddof_m_red)[4 * i]);
        printf("%2d p_ar_prox: %10.16e \n", current_id_[i], (*cv0ddof_m_red)[4 * i + 1]);
        printf("%2d q_ar_prox: %10.16e \n", current_id_[i], (*cv0ddof_m_red)[4 * i + 2]);
        printf("%2d p_ar_dist: %10.16e \n", current_id_[i], (*cv0ddof_m_red)[4 * i + 3]);
        printf("%2d V_v: %10.16e \n", current_id_[i], (*v_m_red)[4 * i]);
        if (enhanced_output_ and !(init))
        {
          printf("%2d dp_v/dt: %10.16e \n", current_id_[i], (*dcv0ddof_m_red)[4 * i]);
          printf("%2d dp_ar_prox/dt: %10.16e \n", current_id_[i], (*dcv0ddof_m_red)[4 * i + 1]);
          printf("%2d dq_ar_prox/dt: %10.16e \n", current_id_[i], (*dcv0ddof_m_red)[4 * i + 2]);
          printf("%2d dp_ar_dist/dt: %10.16e \n", current_id_[i], (*dcv0ddof_m_red)[4 * i + 3]);
        }
      }
    }

    if (cardvasc0d_syspulcirculation_->have_cardiovascular0_d())
    {
      printf("p_at_l: %10.16e \n", (*cv0ddof_m_red)[0]);
      printf("q_vin_l: %10.16e \n", (*cv0ddof_m_red)[1]);
      printf("q_vout_l: %10.16e \n", (*cv0ddof_m_red)[2]);
      printf("p_v_l: %10.16e \n", (*cv0ddof_m_red)[3]);
      printf("p_ar_sys: %10.16e \n", (*cv0ddof_m_red)[4]);
      printf("q_ar_sys: %10.16e \n", (*cv0ddof_m_red)[5]);
      printf("p_ven_sys: %10.16e \n", (*cv0ddof_m_red)[6]);
      printf("q_ven_sys: %10.16e \n", (*cv0ddof_m_red)[7]);
      printf("p_at_r: %10.16e \n", (*cv0ddof_m_red)[8]);
      printf("q_vin_r: %10.16e \n", (*cv0ddof_m_red)[9]);
      printf("q_vout_r: %10.16e \n", (*cv0ddof_m_red)[10]);
      printf("p_v_r: %10.16e \n", (*cv0ddof_m_red)[11]);
      printf("p_ar_pul: %10.16e \n", (*cv0ddof_m_red)[12]);
      printf("q_ar_pul: %10.16e \n", (*cv0ddof_m_red)[13]);
      printf("p_ven_pul: %10.16e \n", (*cv0ddof_m_red)[14]);
      printf("q_ven_pul: %10.16e \n", (*cv0ddof_m_red)[15]);
      // print volumes (no state variables)
      printf("V_at_l: %10.16e \n", (*v_m_red)[0]);
      printf("V_v_l: %10.16e \n", (*v_m_red)[2]);
      printf("V_ar_sys: %10.16e \n", (*v_m_red)[4]);
      printf("V_ven_sys: %10.16e \n", (*v_m_red)[6]);
      printf("V_at_r: %10.16e \n", (*v_m_red)[8]);
      printf("V_v_r: %10.16e \n", (*v_m_red)[10]);
      printf("V_ar_pul: %10.16e \n", (*v_m_red)[12]);
      printf("V_ven_pul: %10.16e \n", (*v_m_red)[14]);
    }

    if (cardvascrespir0d_syspulperiphcirculation_->have_cardiovascular0_d())
    {
      printf("p_at_l: %10.16e \n", (*cv0ddof_m_red)[0]);
      printf("q_vin_l: %10.16e \n", (*cv0ddof_m_red)[1]);
      printf("q_vout_l: %10.16e \n", (*cv0ddof_m_red)[2]);
      printf("p_v_l: %10.16e \n", (*cv0ddof_m_red)[3]);
      printf("p_ar_sys: %10.16e \n", (*cv0ddof_m_red)[4]);
      printf("q_ar_sys: %10.16e \n", (*cv0ddof_m_red)[5]);
      printf("p_arperi_sys: %10.16e \n", (*cv0ddof_m_red)[6]);
      printf("q_arspl_sys: %10.16e \n", (*cv0ddof_m_red)[7]);
      printf("q_arespl_sys: %10.16e \n", (*cv0ddof_m_red)[8]);
      printf("q_armsc_sys: %10.16e \n", (*cv0ddof_m_red)[9]);
      printf("q_arcer_sys: %10.16e \n", (*cv0ddof_m_red)[10]);
      printf("q_arcor_sys: %10.16e \n", (*cv0ddof_m_red)[11]);
      printf("p_venspl_sys: %10.16e \n", (*cv0ddof_m_red)[12]);
      printf("q_venspl_sys: %10.16e \n", (*cv0ddof_m_red)[13]);
      printf("p_venespl_sys: %10.16e \n", (*cv0ddof_m_red)[14]);
      printf("q_venespl_sys: %10.16e \n", (*cv0ddof_m_red)[15]);
      printf("p_venmsc_sys: %10.16e \n", (*cv0ddof_m_red)[16]);
      printf("q_venmsc_sys: %10.16e \n", (*cv0ddof_m_red)[17]);
      printf("p_vencer_sys: %10.16e \n", (*cv0ddof_m_red)[18]);
      printf("q_vencer_sys: %10.16e \n", (*cv0ddof_m_red)[19]);
      printf("p_vencor_sys: %10.16e \n", (*cv0ddof_m_red)[20]);
      printf("q_vencor_sys: %10.16e \n", (*cv0ddof_m_red)[21]);
      printf("p_ven_sys: %10.16e \n", (*cv0ddof_m_red)[22]);
      printf("q_ven_sys: %10.16e \n", (*cv0ddof_m_red)[23]);
      printf("p_at_r: %10.16e \n", (*cv0ddof_m_red)[24]);
      printf("q_vin_r: %10.16e \n", (*cv0ddof_m_red)[25]);
      printf("q_vout_r: %10.16e \n", (*cv0ddof_m_red)[26]);
      printf("p_v_r: %10.16e \n", (*cv0ddof_m_red)[27]);
      printf("p_ar_pul: %10.16e \n", (*cv0ddof_m_red)[28]);
      printf("q_ar_pul: %10.16e \n", (*cv0ddof_m_red)[29]);
      printf("p_cap_pul: %10.16e \n", (*cv0ddof_m_red)[30]);
      printf("q_cap_pul: %10.16e \n", (*cv0ddof_m_red)[31]);
      printf("p_ven_pul: %10.16e \n", (*cv0ddof_m_red)[32]);
      printf("q_ven_pul: %10.16e \n", (*cv0ddof_m_red)[33]);
      // print volumes (no state variables)
      printf("V_at_l: %10.16e \n", (*v_m_red)[0]);
      printf("V_v_l: %10.16e \n", (*v_m_red)[2]);
      printf("V_ar_sys: %10.16e \n", (*v_m_red)[4]);
      printf("V_arperi_sys: %10.16e \n", (*v_m_red)[6]);
      printf("V_venspl_sys: %10.16e \n", (*v_m_red)[12]);
      printf("V_venespl_sys: %10.16e \n", (*v_m_red)[14]);
      printf("V_venmsc_sys: %10.16e \n", (*v_m_red)[16]);
      printf("V_vencer_sys: %10.16e \n", (*v_m_red)[18]);
      printf("V_vencor_sys: %10.16e \n", (*v_m_red)[20]);
      printf("V_ven_sys: %10.16e \n", (*v_m_red)[22]);
      printf("V_at_r: %10.16e \n", (*v_m_red)[24]);
      printf("V_v_r: %10.16e \n", (*v_m_red)[26]);
      printf("V_ar_pul: %10.16e \n", (*v_m_red)[28]);
      printf("V_cap_pul: %10.16e \n", (*v_m_red)[30]);
      printf("V_ven_pul: %10.16e \n", (*v_m_red)[32]);

      if (cardvasc0d_model_->get_respiratory_model())
      {
        // 0D lung
        printf("V_alv: %10.16e \n", (*cv0ddof_m_red)[34]);
        printf("q_alv: %10.16e \n", (*cv0ddof_m_red)[35]);
        printf("p_alv: %10.16e \n", (*cv0ddof_m_red)[36]);
        printf("fCO2_alv: %10.16e \n", (*cv0ddof_m_red)[37]);
        printf("fO2_alv: %10.16e \n", (*cv0ddof_m_red)[38]);
        // (auxiliary) incoming systemic capillary fluxes
        printf("q_arspl_sys_in: %10.16e \n", (*cv0ddof_m_red)[39]);
        printf("q_arespl_sys_in: %10.16e \n", (*cv0ddof_m_red)[40]);
        printf("q_armsc_sys_in: %10.16e \n", (*cv0ddof_m_red)[41]);
        printf("q_arcer_sys_in: %10.16e \n", (*cv0ddof_m_red)[42]);
        printf("q_arcor_sys_in: %10.16e \n", (*cv0ddof_m_red)[43]);
        // the partial pressures
        printf("ppCO2_at_r: %10.16e \n", (*cv0ddof_m_red)[44]);
        printf("ppO2_at_r: %10.16e \n", (*cv0ddof_m_red)[45]);
        printf("ppCO2_v_r: %10.16e \n", (*cv0ddof_m_red)[46]);
        printf("ppO2_v_r: %10.16e \n", (*cv0ddof_m_red)[47]);
        printf("ppCO2_ar_pul: %10.16e \n", (*cv0ddof_m_red)[48]);
        printf("ppO2_ar_pul: %10.16e \n", (*cv0ddof_m_red)[49]);
        printf("ppCO2_cap_pul: %10.16e \n", (*cv0ddof_m_red)[50]);
        printf("ppO2_cap_pul: %10.16e \n", (*cv0ddof_m_red)[51]);
        printf("ppCO2_ven_pul: %10.16e \n", (*cv0ddof_m_red)[52]);
        printf("ppO2_ven_pul: %10.16e \n", (*cv0ddof_m_red)[53]);
        printf("ppCO2_at_l: %10.16e \n", (*cv0ddof_m_red)[54]);
        printf("ppO2_at_l: %10.16e \n", (*cv0ddof_m_red)[55]);
        printf("ppCO2_v_l: %10.16e \n", (*cv0ddof_m_red)[56]);
        printf("ppO2_v_l: %10.16e \n", (*cv0ddof_m_red)[57]);
        printf("ppCO2_ar_sys: %10.16e \n", (*cv0ddof_m_red)[58]);
        printf("ppO2_ar_sys: %10.16e \n", (*cv0ddof_m_red)[59]);
        printf("ppCO2_arspl_sys: %10.16e \n", (*cv0ddof_m_red)[60]);
        printf("ppO2_arspl_sys: %10.16e \n", (*cv0ddof_m_red)[61]);
        printf("ppCO2_arespl_sys: %10.16e \n", (*cv0ddof_m_red)[62]);
        printf("ppO2_arespl_sys: %10.16e \n", (*cv0ddof_m_red)[63]);
        printf("ppCO2_armsc_sys: %10.16e \n", (*cv0ddof_m_red)[64]);
        printf("ppO2_armsc_sys: %10.16e \n", (*cv0ddof_m_red)[65]);
        printf("ppCO2_arcer_sys: %10.16e \n", (*cv0ddof_m_red)[66]);
        printf("ppO2_arcer_sys: %10.16e \n", (*cv0ddof_m_red)[67]);
        printf("ppCO2_arcor_sys: %10.16e \n", (*cv0ddof_m_red)[68]);
        printf("ppO2_arcor_sys: %10.16e \n", (*cv0ddof_m_red)[69]);
        printf("ppCO2_venspl_sys: %10.16e \n", (*cv0ddof_m_red)[70]);
        printf("ppO2_venspl_sys: %10.16e \n", (*cv0ddof_m_red)[71]);
        printf("ppCO2_venespl_sys: %10.16e \n", (*cv0ddof_m_red)[72]);
        printf("ppO2_venespl_sys: %10.16e \n", (*cv0ddof_m_red)[73]);
        printf("ppCO2_venmsc_sys: %10.16e \n", (*cv0ddof_m_red)[74]);
        printf("ppO2_venmsc_sys: %10.16e \n", (*cv0ddof_m_red)[75]);
        printf("ppCO2_vencer_sys: %10.16e \n", (*cv0ddof_m_red)[76]);
        printf("ppO2_vencer_sys: %10.16e \n", (*cv0ddof_m_red)[77]);
        printf("ppCO2_vencor_sys: %10.16e \n", (*cv0ddof_m_red)[78]);
        printf("ppO2_vencor_sys: %10.16e \n", (*cv0ddof_m_red)[79]);
        printf("ppCO2_ven_sys: %10.16e \n", (*cv0ddof_m_red)[80]);
        printf("ppO2_ven_sys: %10.16e \n", (*cv0ddof_m_red)[81]);

        if (enhanced_output_)
        {
          // oxygen saturations (no state variables - stored in volume vector for convenience!)
          printf("SO2_ar_pul: %10.16e \n", (*v_m_red)[49]);
          printf("SO2_ar_sys: %10.16e \n", (*v_m_red)[59]);
        }
      }
    }
    printf("total time: %10.16e \n", totaltime_);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  set-up (public)                                            mhv 11/13|
 *----------------------------------------------------------------------*/
void UTILS::Cardiovascular0DManager::solver_setup(
    Core::LinAlg::Solver& solver, Teuchos::ParameterList params)
{
  solver_ = Teuchos::rcp(&solver, false);

  // different setup for #adapttol_
  isadapttol_ = true;
  isadapttol_ = (params.get<bool>("ADAPTCONV"));

  // simple parameters
  adaptolbetter_ = params.get<double>("ADAPTCONV_BETTER", 0.01);

  counter_ = 0;

  return;
}



int UTILS::Cardiovascular0DManager::solve(Teuchos::RCP<Core::LinAlg::SparseMatrix> mat_structstiff,
    Teuchos::RCP<Core::LinAlg::Vector<double>> dispinc,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> rhsstruct, const double k_ptc)
{
  // create old style dirichtoggle vector (supposed to go away)
  dirichtoggle_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*(dbcmaps_->full_map())));
  Teuchos::RCP<Core::LinAlg::Vector<double>> temp =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*(dbcmaps_->cond_map())));
  temp->PutScalar(1.0);
  Core::LinAlg::export_to(*temp, *dirichtoggle_);

  // allocate additional vectors and matrices
  Teuchos::RCP<Core::LinAlg::Vector<double>> rhscardvasc0d =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*(get_cardiovascular0_drhs())));
  Teuchos::RCP<Core::LinAlg::Vector<double>> cv0ddofincr =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*(get_cardiovascular0_d_map())));
  Teuchos::RCP<Core::LinAlg::SparseMatrix> mat_cardvasc0dstiff =
      (Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(get_cardiovascular0_d_stiffness()));
  Teuchos::RCP<Core::LinAlg::SparseMatrix> mat_dcardvasc0d_dd =
      (Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(get_mat_dcardvasc0d_dd()));
  Teuchos::RCP<Core::LinAlg::SparseMatrix> mat_dstruct_dcv0ddof =
      (Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(get_mat_dstruct_dcv0ddof()));

  // prepare residual cv0ddof
  cv0ddofincr->PutScalar(0.0);


  // apply DBC to additional offdiagonal coupling matrices
  mat_dcardvasc0d_dd->apply_dirichlet(*(dbcmaps_->cond_map()), false);
  mat_dstruct_dcv0ddof->apply_dirichlet(*(dbcmaps_->cond_map()), false);

  // define maps of standard dofs and additional pressures
  Teuchos::RCP<Epetra_Map> standrowmap = Teuchos::rcp(new Epetra_Map(mat_structstiff->row_map()));
  Teuchos::RCP<Epetra_Map> cardvasc0drowmap =
      Teuchos::rcp(new Epetra_Map(*cardiovascular0dmap_full_));


  if (ptc_3d0d_)
  {
    // PTC on structural matrix
    Teuchos::RCP<Core::LinAlg::Vector<double>> tmp3D =
        Core::LinAlg::create_vector(mat_structstiff->row_map(), false);
    tmp3D->PutScalar(k_ptc);
    Teuchos::RCP<Core::LinAlg::Vector<double>> diag3D =
        Core::LinAlg::create_vector(mat_structstiff->row_map(), false);
    mat_structstiff->extract_diagonal_copy(*diag3D);
    diag3D->Update(1.0, *tmp3D, 1.0);
    mat_structstiff->replace_diagonal_values(*diag3D);
  }

  // merge maps to one large map
  Teuchos::RCP<Epetra_Map> mergedmap =
      Core::LinAlg::merge_map(standrowmap, cardvasc0drowmap, false);
  // define MapExtractor
  // Core::LinAlg::MapExtractor mapext(*mergedmap,standrowmap,cardvasc0drowmap);

  std::vector<Teuchos::RCP<const Epetra_Map>> myMaps;
  myMaps.push_back(standrowmap);
  myMaps.push_back(cardvasc0drowmap);
  Core::LinAlg::MultiMapExtractor mapext(*mergedmap, myMaps);

  // initialize blockmat, mergedrhs, mergedsol and mapext to keep them in scope after the following
  // if-condition
  Teuchos::RCP<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>> blockmat;
  Teuchos::RCP<Core::LinAlg::Vector<double>> mergedrhs;
  Teuchos::RCP<Core::LinAlg::Vector<double>> mergedsol;
  Core::LinAlg::MultiMapExtractor mapext_R;

  if (have_mor_)
  {
    // reduce linear system
    Teuchos::RCP<Core::LinAlg::SparseMatrix> mat_structstiff_R =
        mor_->reduce_diagnoal(mat_structstiff);
    Teuchos::RCP<Core::LinAlg::SparseMatrix> mat_dcardvasc0d_dd_R =
        mor_->reduce_off_diagonal(mat_dcardvasc0d_dd);
    Teuchos::RCP<Core::LinAlg::SparseMatrix> mat_dstruct_dcv0ddof_R =
        mor_->reduce_off_diagonal(mat_dstruct_dcv0ddof);
    Teuchos::RCP<Epetra_MultiVector> rhsstruct_R =
        mor_->reduce_rhs(rhsstruct->get_ptr_of_Epetra_MultiVector());

    // define maps of reduced standard dofs and additional pressures
    Teuchos::RCP<Epetra_Map> structmap_R =
        Teuchos::rcp(new Epetra_Map(mor_->get_red_dim(), 0, actdisc_->get_comm()));
    Teuchos::RCP<Epetra_Map> standrowmap_R = Teuchos::rcp(new Epetra_Map(*structmap_R));
    Teuchos::RCP<Epetra_Map> cardvasc0drowmap_R =
        Teuchos::rcp(new Epetra_Map(mat_cardvasc0dstiff->row_map()));

    // merge maps of reduced standard dofs and additional pressures to one large map
    Teuchos::RCP<Epetra_Map> mergedmap_R =
        Core::LinAlg::merge_map(standrowmap_R, cardvasc0drowmap_R, false);

    std::vector<Teuchos::RCP<const Epetra_Map>> myMaps_R;
    myMaps_R.push_back(standrowmap_R);
    myMaps_R.push_back(cardvasc0drowmap_R);
    mapext_R.setup(*mergedmap_R, myMaps_R);

    // initialize BlockMatrix and Core::LinAlg::Vectors
    blockmat =
        Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
            mapext_R, mapext_R, 81, false, false));
    mergedrhs = Teuchos::rcp(new Core::LinAlg::Vector<double>(*mergedmap_R));
    mergedsol = Teuchos::rcp(new Core::LinAlg::Vector<double>(*mergedmap_R));

    // use BlockMatrix
    blockmat->assign(0, 0, Core::LinAlg::View, *mat_structstiff_R);
    blockmat->assign(1, 0, Core::LinAlg::View, *mat_dcardvasc0d_dd_R);
    blockmat->assign(
        0, 1, Core::LinAlg::View, *Core::LinAlg::matrix_transpose(*mat_dstruct_dcv0ddof_R));
    blockmat->assign(1, 1, Core::LinAlg::View, *mat_cardvasc0dstiff);
    blockmat->complete();

    // export 0D part of rhs
    Core::LinAlg::export_to(*rhscardvasc0d, *mergedrhs);
    // make the 0D part of the rhs negative
    mergedrhs->Scale(-1.0);
    // export reduced structure part of rhs -> no need to make it negative since this has been done
    // by the structural time integrator already!
    Core::LinAlg::export_to(*rhsstruct_R, *mergedrhs);
  }
  else
  {
    // initialize BlockMatrix and Core::LinAlg::Vectors
    blockmat =
        Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
            mapext, mapext, 81, false, false));
    mergedrhs = Teuchos::rcp(new Core::LinAlg::Vector<double>(*mergedmap));
    mergedsol = Teuchos::rcp(new Core::LinAlg::Vector<double>(*mergedmap));

    // use BlockMatrix
    blockmat->assign(0, 0, Core::LinAlg::View, *mat_structstiff);
    blockmat->assign(
        1, 0, Core::LinAlg::View, *Core::LinAlg::matrix_transpose(*mat_dcardvasc0d_dd));
    blockmat->assign(0, 1, Core::LinAlg::View, *mat_dstruct_dcv0ddof);
    blockmat->assign(1, 1, Core::LinAlg::View, *mat_cardvasc0dstiff);
    blockmat->complete();

    // export 0D part of rhs
    Core::LinAlg::export_to(*rhscardvasc0d, *mergedrhs);
    // make the 0D part of the rhs negative
    mergedrhs->Scale(-1.0);
    // export structure part of rhs -> no need to make it negative since this has been done by the
    // structural time integrator already!
    Core::LinAlg::export_to(*rhsstruct, *mergedrhs);
  }

  // ONLY compatability
  // dirichtoggle_ changed and we need to rebuild associated DBC maps
  if (dirichtoggle_ != Teuchos::null)
    dbcmaps_ = Core::LinAlg::convert_dirichlet_toggle_vector_to_maps(dirichtoggle_);


  Teuchos::ParameterList sfparams =
      solver_->params();  // save copy of original solver parameter list
  const Teuchos::ParameterList& cardvasc0dstructparams =
      Global::Problem::instance()->cardiovascular0_d_structural_params();
  const int linsolvernumber = cardvasc0dstructparams.get<int>("LINEAR_COUPLED_SOLVER");
  solver_->params() = Core::LinAlg::Solver::translate_solver_parameters(
      Global::Problem::instance()->solver_params(linsolvernumber),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));
  switch (algochoice_)
  {
    case Inpar::Cardiovascular0D::cardvasc0dsolve_direct:
      break;
    case Inpar::Cardiovascular0D::cardvasc0dsolve_block:
    {
      solver_->put_solver_params_to_sub_params("Inverse1",
          Global::Problem::instance()->solver_params(linsolvernumber),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));
      actdisc_->compute_null_space_if_necessary(solver_->params().sublist("Inverse1"), true);

      solver_->put_solver_params_to_sub_params("Inverse2",
          Global::Problem::instance()->solver_params(linsolvernumber),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));
      actdisc_->compute_null_space_if_necessary(solver_->params().sublist("Inverse2"), true);
      break;
    }
    default:
      FOUR_C_THROW("Unknown 0D cardiovascular-structural solution technique!");
  }

  linsolveerror_ = 0;

  double norm_res_full;
  mergedrhs->Norm2(&norm_res_full);

  // solve for disi
  // Solve K . IncD = -R  ===>  IncD_{n+1}
  Core::LinAlg::SolverParams solver_params;
  if (isadapttol_ && counter_)
  {
    solver_params.nonlin_tolerance = tolres_struct_;
    solver_params.nonlin_residual = norm_res_full;
    solver_params.lin_tol_better = adaptolbetter_;
  }

  // solve with merged matrix
  // solver_->Solve(mergedmatrix->EpetraMatrix(),mergedsol,mergedrhs,true,counter_==0);
  // solve with BlockMatrix
  solver_params.refactor = true;
  solver_params.reset = counter_ == 0;
  linsolveerror_ = solver_->solve(blockmat, mergedsol, mergedrhs, solver_params);
  solver_->reset_tolerance();

  // initialize mergedsol_full to keep it in scope after the following if-condition
  Teuchos::RCP<Core::LinAlg::Vector<double>> mergedsol_full =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*mergedmap));

  if (have_mor_)
  {
    // initialize and write vector with reduced displacement dofs
    Teuchos::RCP<Core::LinAlg::Vector<double>> disp_R =
        Teuchos::rcp(new Core::LinAlg::Vector<double>(*mapext_R.Map(0)));
    mapext_R.extract_vector(mergedsol, 0, disp_R);

    // initialize and write vector with pressure dofs, replace row map
    Teuchos::RCP<Core::LinAlg::Vector<double>> cv0ddof =
        Teuchos::rcp(new Core::LinAlg::Vector<double>(*mapext_R.Map(1)));
    mapext_R.extract_vector(mergedsol, 1, cv0ddof);
    cv0ddof->ReplaceMap(*cardvasc0drowmap);

    // extend reduced displacement dofs to high dimension
    Teuchos::RCP<Core::LinAlg::Vector<double>> disp_full = mor_->extend_solution(disp_R);

    // assemble displacement and pressure dofs
    mergedsol_full = mapext.insert_vector(disp_full, 0);
    mapext.add_vector(cv0ddof, 1, mergedsol_full, 1);
  }
  else
    mergedsol_full = mergedsol;

  // store results in smaller vectors
  mapext.extract_vector(mergedsol_full, 0, dispinc);
  mapext.extract_vector(mergedsol_full, 1, cv0ddofincr);

  cv0ddofincrement_->Update(1., *cv0ddofincr, 0.);

  counter_++;

  // update 0D cardiovascular dofs
  update_cv0_d_dof(cv0ddofincr);

  return linsolveerror_;
}

FOUR_C_NAMESPACE_CLOSE
