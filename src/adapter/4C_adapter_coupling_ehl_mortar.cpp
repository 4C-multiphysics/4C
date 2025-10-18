// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_coupling_ehl_mortar.hpp"

#include "4C_contact_friction_node.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_lagrange_strategy_tsi.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN

Adapter::CouplingEhlMortar::CouplingEhlMortar(int spatial_dimension,
    Teuchos::ParameterList mortar_coupling_params, Teuchos::ParameterList contact_dynamic_params,
    Core::FE::ShapeFunctionType shape_function_type)
    : CouplingNonLinMortar(
          spatial_dimension, mortar_coupling_params, contact_dynamic_params, shape_function_type),
      contact_regularization_(Global::Problem::instance()->contact_dynamic_params().get<bool>(
          "REGULARIZED_NORMAL_CONTACT")),
      regularization_thickness_(Global::Problem::instance()->contact_dynamic_params().get<double>(
          "REGULARIZATION_THICKNESS")),
      regularization_compliance_(Global::Problem::instance()->contact_dynamic_params().get<double>(
          "REGULARIZATION_STIFFNESS"))
{
  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(
          Global::Problem::instance()->mortar_coupling_params().sublist("PARALLEL REDISTRIBUTION"),
          "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none)
    FOUR_C_THROW(
        "EHL does not support parallel redistribution. Set \"PARALLEL_REDIST none\" in section "
        "\"MORTAR COUPLING\"");

  if (contact_regularization_)
    if (regularization_compliance_ <= 0. || regularization_thickness_ <= 0.)
      FOUR_C_THROW("need positive REGULARIZATION_THICKNESS and REGULARIZATION_STIFFNESS");
  if (contact_regularization_) regularization_compliance_ = 1. / regularization_compliance_;
  if (Global::Problem::instance()->contact_dynamic_params().get<bool>(
          "REGULARIZED_NORMAL_CONTACT") &&
      not Global::Problem::instance()->elasto_hydro_dynamic_params().get<bool>("DRY_CONTACT_MODEL"))
    FOUR_C_THROW("for dry contact model you need REGULARIZED_NORMAL_CONTACT and DRY_CONTACT_MODEL");
}

/*----------------------------------------------------------------------*
 |  read mortar condition                                               |
 *----------------------------------------------------------------------*/
void Adapter::CouplingEhlMortar::read_mortar_condition(
    std::shared_ptr<Core::FE::Discretization> masterdis,
    std::shared_ptr<Core::FE::Discretization> slavedis, std::vector<int> coupleddof,
    const std::string& couplingcond, Teuchos::ParameterList& input,
    std::map<int, Core::Nodes::Node*>& mastergnodes, std::map<int, Core::Nodes::Node*>& slavegnodes,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& masterelements,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& slaveelements)
{
  Adapter::CouplingNonLinMortar::read_mortar_condition(masterdis, slavedis, coupleddof,
      couplingcond, input, mastergnodes, slavegnodes, masterelements, slaveelements);

  input.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::ehl);
}

void Adapter::CouplingEhlMortar::setup(std::shared_ptr<Core::FE::Discretization> masterdis,
    std::shared_ptr<Core::FE::Discretization> slavedis, std::vector<int> coupleddof,
    const std::string& couplingcond)
{
  Adapter::CouplingNonLinMortar::setup(masterdis, slavedis, coupleddof, couplingcond);
  z_ = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_dofs(), true);
  fscn_ = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_dofs(), true);

  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(
      Global::Problem::instance()->contact_dynamic_params(), "FRICTION");

  std::vector<const Core::Conditions::Condition*> ehl_conditions;
  masterdis->get_condition(couplingcond, ehl_conditions);
  std::vector<const std::string*> sides((int)ehl_conditions.size());
  double fr_coeff = -1.;
  for (int i = 0; i < (int)ehl_conditions.size(); ++i)
  {
    [[maybe_unused]] const int group1id = ehl_conditions[i]->parameters().get<int>("InterfaceID");
    const auto fr = ehl_conditions[i]->parameters().get<double>("FrCoeffOrBound");
    if (fr != ehl_conditions[0]->parameters().get<double>("FrCoeffOrBound"))
      FOUR_C_THROW("inconsistency in friction coefficients");
    fr_coeff = fr;
  }

  switch (ftype)
  {
    case CONTACT::FrictionType::tresca:
      FOUR_C_THROW("no tresca friction supported");
      break;
    case CONTACT::FrictionType::none:
      break;
    case CONTACT::FrictionType::coulomb:
      interface_->interface_params().set<double>("FRCOEFF", fr_coeff);
      interface_->interface_params().set<double>("FRBOUND", -1.);
      break;
    default:
      FOUR_C_THROW("don't know what to do with this friction type");
      break;
  }
}


/*----------------------------------------------------------------------*
 |  perform interface integration and assembly                          |
 *----------------------------------------------------------------------*/
void Adapter::CouplingEhlMortar::integrate(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp, const double dt)
{
  // safety check
  check_setup();

  // return if this state has already been evaluated
  if (already_evaluated(disp)) return;

  // set current displ state
  interface_->set_state(Mortar::state_new_displacement, *disp);

  // init internal data
  interface_->initialize();
  interface_->set_element_areas();
  // call interface evaluate (d,m,gap...)
  interface_->evaluate();

  // some first assemblies, that don't require any additional states
  D_ = std::make_shared<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);
  M_ = std::make_shared<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);
  interface_->assemble_dm(*D_, *M_);
  D_->complete();
  M_->complete(*masterdofrowmap_, *slavedofrowmap_);
  N_->complete(*smdofrowmap_, *slavedofrowmap_);
  assemble_real_gap();
  assemble_real_gap_deriv();
  assemble_normals();
  assemble_normals_deriv();
  assemble_surf_grad();
  assemble_interface_velocities(dt);

  // save that state as the last evaluated one
  evaluated_state_ = std::make_shared<Core::LinAlg::Vector<double>>(*disp);

  // all done
  return;
}

/*----------------------------------------------------------------------*
 |  perform interface integration and assembly                          |
 *----------------------------------------------------------------------*/
void Adapter::CouplingEhlMortar::condense_contact(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>>& combined_RHS,
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp, const double dt)
{
  const double alphaf_ = 0.;  // statics!
  const CONTACT::ConstraintDirection& constr_direction_ =
      Teuchos::getIntegralValue<CONTACT::ConstraintDirection>(
          interface()->interface_params(), "CONSTRAINT_DIRECTIONS");

  // return if this state has already been evaluated
  if (not already_evaluated(disp)) integrate(disp, dt);

  // get the relative movement for frictional contact
  evaluate_rel_mov();

  // update active set
  as_converged_ = interface_->update_active_set_semi_smooth();
  interface_->build_active_set();

  // assemble the constraint lines for the active contact nodes
  std::shared_ptr<Core::LinAlg::SparseMatrix> dcsdd = std::make_shared<Core::LinAlg::SparseMatrix>(
      *interface_->active_dofs(), 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  std::shared_ptr<Core::LinAlg::SparseMatrix> dcsdLMc =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *interface_->active_dofs(), 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  std::shared_ptr<Core::LinAlg::Vector<double>> fcsa =
      Core::LinAlg::create_vector(*interface_->active_dofs(), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> g_all;
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
    g_all = Core::LinAlg::create_vector(*interface_->slave_row_dofs(), true);
  else
    g_all = Core::LinAlg::create_vector(*interface_->slave_row_nodes(), true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> dmatrix =
      std::make_shared<Core::LinAlg::SparseMatrix>(*interface_->slave_row_dofs(), 10);
  std::shared_ptr<Core::LinAlg::SparseMatrix> mmatrix =
      std::make_shared<Core::LinAlg::SparseMatrix>(*interface_->slave_row_dofs(), 100);
  interface_->assemble_dm(*dmatrix, *mmatrix);
  dmatrix->complete();
  mmatrix->complete(*masterdofrowmap_, *slavedofrowmap_);

  // setup some linearizations
  Core::LinAlg::SparseMatrix linDcontactLM(
      *interface_->slave_row_dofs(), 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix linMcontactLM(
      *interface_->master_row_dofs(), 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  interface_->assemble_lin_dm(linDcontactLM, linMcontactLM);

  // D and M matrix for the active nodes
  Core::LinAlg::SparseMatrix dInv(*interface_->slave_row_dofs(), 100, true, false);

  // linearized normal contact
  interface_->assemble_s(*dcsdd);
  interface_->assemble_g(*g_all);

  if (contact_regularization_)
  {
    interface_->assemble_normal_contact_regularization(*dcsdd, *dcsdLMc, *fcsa);

    // linearized tangential contact (friction)
    if (interface_->is_friction())
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> rcsa_fr =
          Core::LinAlg::create_vector(*interface_->active_dofs(), true);
      interface_->assemble_lin_slip_normal_regularization(*dcsdLMc, *dcsdd, *rcsa_fr);
      interface_->assemble_lin_stick(*dcsdLMc, *dcsdd, *rcsa_fr);
      rcsa_fr->scale(-1.);
      CONTACT::Utils::add_vector(*rcsa_fr, *fcsa);
    }
    else
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> rcsa_fr =
          Core::LinAlg::create_vector(*interface_->active_dofs(), true);
      interface_->assemble_tn(dcsdLMc, nullptr);
      interface_->assemble_t_nderiv(dcsdd, nullptr);
      interface_->assemble_tangrhs(*rcsa_fr);
      rcsa_fr->scale(-1.);
      CONTACT::Utils::add_vector(*rcsa_fr, *fcsa);
    }
  }
  else
    FOUR_C_THROW("stop");

  // complete all those linearizations
  //                             colmap        rowmap
  linDcontactLM.complete(*s_mdof_map(), *interface_->slave_row_dofs());
  linMcontactLM.complete(*s_mdof_map(), *interface_->master_row_dofs());

  // normal contact
  std::shared_ptr<Core::LinAlg::Vector<double>> gact;
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    gact = Core::LinAlg::create_vector(*interface_->active_dofs(), true);
    if (gact->global_length()) Core::LinAlg::export_to(*g_all, *gact);
  }
  else
  {
    gact = Core::LinAlg::create_vector(*interface_->active_nodes(), true);
    if (gact->global_length())
    {
      Core::LinAlg::export_to(*g_all, *gact);
      if (gact->replace_map(*interface_->active_n_dofs())) FOUR_C_THROW("replaceMap went wrong");
    }
  }
  CONTACT::Utils::add_vector(*gact, *fcsa);
  fcsa->norm_2(&contact_rhs_norm_);

  // complete all the new matrix blocks
  // Note: since the contact interface assemled them, they are all based
  //       on displacement row and col maps. Hence, some still need to be transformed
  dcsdd->complete(*s_mdof_map(), *interface_->active_dofs());
  dcsdLMc->complete(*interface_->active_dofs(), *interface_->active_dofs());

  // get the separate blocks of the 2x2 TSI block system
  // View mode!!! Since we actually want to add things there
  std::shared_ptr<Core::LinAlg::SparseMatrix> kss = std::make_shared<Core::LinAlg::SparseMatrix>(
      sysmat->matrix(0, 0), Core::LinAlg::DataAccess::Copy);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kst = std::make_shared<Core::LinAlg::SparseMatrix>(
      sysmat->matrix(0, 1), Core::LinAlg::DataAccess::Copy);
  Core::LinAlg::SparseMatrix kts(sysmat->matrix(1, 0), Core::LinAlg::DataAccess::Copy);
  Core::LinAlg::SparseMatrix ktt(sysmat->matrix(1, 1), Core::LinAlg::DataAccess::Copy);

  // get some maps
  std::shared_ptr<Core::LinAlg::Map> gdisp_DofRowMap =
      std::make_shared<Core::LinAlg::Map>(kss->row_map());
  std::shared_ptr<Core::LinAlg::Map> gpres_DofRowMap =
      std::make_shared<Core::LinAlg::Map>(ktt.row_map());
  std::shared_ptr<Core::LinAlg::Map> gmdof =
      std::make_shared<Core::LinAlg::Map>(*interface_->master_row_dofs());
  std::shared_ptr<Core::LinAlg::Map> active_dofs =
      std::make_shared<Core::LinAlg::Map>(*interface_->active_dofs());

  // split rhs
  Core::LinAlg::Vector<double> rs(kss->row_map(), true);
  Core::LinAlg::Vector<double> rt(ktt.row_map(), true);
  Core::LinAlg::export_to(*combined_RHS, rs);
  Core::LinAlg::export_to(*combined_RHS, rt);

  // we don't want the rhs but the residual
  rs.scale(-1.);
  rt.scale(-1.);

  // add last time step contact forces to rhs
  if (fscn_ != nullptr)  // in the first time step, we don't have any history of the
                         // contact force, after that, fscn_ should be initialized properly
  {
    Core::LinAlg::Vector<double> tmp(kss->row_map());
    Core::LinAlg::export_to(*fscn_, tmp);
    rs.update(alphaf_, tmp, 1.);  // fscn already scaled with alphaf_ in update
  }


  // map containing the inactive and non-contact structural dofs
  std::shared_ptr<Core::LinAlg::Map> str_gni_dofs = Core::LinAlg::split_map(
      *Core::LinAlg::split_map(Core::LinAlg::Map(kss->row_map()), *interface_->master_row_dofs()),
      *interface_->active_dofs());

  // add to kss
  kss->un_complete();
  kss->add(linDcontactLM, false, 1. - alphaf_, 1.);
  kss->add(linMcontactLM, false, 1. - alphaf_, 1.);

  // complete the matrix blocks again, now that we have added
  // the additional displacement linearizations
  kss->complete();

  // now we have added the additional linearizations.
  // if there are no active nodes, we can leave now
  if (interface_->active_nodes()->num_global_elements() == 0)
  {
    sysmat->reset();
    sysmat->assign(0, 0, Core::LinAlg::DataAccess::Copy, *kss);
    sysmat->assign(0, 1, Core::LinAlg::DataAccess::Copy, *kst);
    sysmat->assign(1, 0, Core::LinAlg::DataAccess::Copy, kts);
    sysmat->assign(1, 1, Core::LinAlg::DataAccess::Copy, ktt);
    return;
  }

  // split matrix blocks in 3 rows: Active, Master and (Inactive+others)
  std::shared_ptr<Core::LinAlg::SparseMatrix> kss_ni, kss_m, kss_a, kst_ni, kst_m, kst_a, kts_ni,
      kts_m, kts_a, ktt_ni, ktt_m, ktt_a, dummy1, dummy2, dummy3;

  // temporary matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> tmp;
  std::shared_ptr<Core::LinAlg::Vector<double>> tmpv;

  // an empty dummy map
  std::shared_ptr<Core::LinAlg::Map> dummy_map1, dummy_map2;

  // ****************************************************
  // split kss block*************************************
  // ****************************************************
  // split first row
  Core::LinAlg::split_matrix2x2(
      kss, str_gni_dofs, dummy_map1, gdisp_DofRowMap, dummy_map2, kss_ni, dummy1, tmp, dummy2);

  // this should be a split in rows, so that two blocks should have zero columns
  if (dummy1->domain_map().num_global_elements() != 0 ||
      dummy2->domain_map().num_global_elements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = nullptr;
  dummy2 = nullptr;
  dummy_map1 = nullptr;
  dummy_map2 = nullptr;

  // split the remaining two rows
  Core::LinAlg::split_matrix2x2(
      tmp, gmdof, dummy_map1, gdisp_DofRowMap, dummy_map2, kss_m, dummy1, kss_a, dummy2);

  // this should be a split in rows, so that two blocks should have zero columns
  if (dummy1->domain_map().num_global_elements() != 0 ||
      dummy2->domain_map().num_global_elements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = nullptr;
  dummy2 = nullptr;
  dummy_map1 = nullptr;
  dummy_map2 = nullptr;
  tmp = nullptr;
  // ****************************************************
  // split kss block*************************************
  // ****************************************************

  // ****************************************************
  // split kst block*************************************
  // ****************************************************
  // split first row
  Core::LinAlg::split_matrix2x2(
      kst, str_gni_dofs, dummy_map1, gpres_DofRowMap, dummy_map2, kst_ni, dummy1, tmp, dummy2);

  // this should be a split in rows, so that two blocks should have zero columns
  if (dummy1->domain_map().num_global_elements() != 0 ||
      dummy2->domain_map().num_global_elements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = nullptr;
  dummy2 = nullptr;
  dummy_map1 = nullptr;
  dummy_map2 = nullptr;

  // split the remaining two rows
  Core::LinAlg::split_matrix2x2(
      tmp, gmdof, dummy_map1, gpres_DofRowMap, dummy_map2, kst_m, dummy1, kst_a, dummy2);

  // this should be a split in rows, so that two blocks should have zero columns
  if (dummy1->domain_map().num_global_elements() != 0 ||
      dummy2->domain_map().num_global_elements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = nullptr;
  dummy2 = nullptr;
  dummy_map1 = nullptr;
  dummy_map2 = nullptr;
  tmp = nullptr;
  // ****************************************************
  // split kst block*************************************
  // ****************************************************

  // ****************************************************
  // split rhs vectors***********************************
  // ****************************************************
  // split structural rhs
  Core::LinAlg::Vector<double> rsni(*str_gni_dofs);
  Core::LinAlg::export_to(rs, rsni);
  Core::LinAlg::Vector<double> rsm(*interface_->master_row_dofs());
  Core::LinAlg::export_to(rs, rsm);
  std::shared_ptr<Core::LinAlg::Vector<double>> rsa =
      std::make_shared<Core::LinAlg::Vector<double>>(*interface_->active_dofs());
  Core::LinAlg::export_to(rs, *rsa);
  // ****************************************************
  // split rhs vectors***********************************
  // ****************************************************

  // D and M matrix for the active nodes
  std::shared_ptr<Core::LinAlg::SparseMatrix> dInvA =
      std::make_shared<Core::LinAlg::SparseMatrix>(*interface_->active_dofs(), 100, true, false);
  std::shared_ptr<Core::LinAlg::SparseMatrix> mA =
      std::make_shared<Core::LinAlg::SparseMatrix>(*interface_->active_dofs(), 100, true, false);

  dummy_map1 = dummy_map2 = nullptr;
  dummy1 = dummy2 = dummy3 = nullptr;
  Core::LinAlg::split_matrix2x2(
      dmatrix, active_dofs, dummy_map1, active_dofs, dummy_map2, dInvA, dummy1, dummy2, dummy3);
  dInvA->complete(*interface_->active_dofs(), *interface_->active_dofs());
  // invert D-matrix
  Core::LinAlg::Vector<double> dDiag(*interface_->active_dofs());
  dInvA->extract_diagonal_copy(dDiag);
  dDiag.reciprocal(dDiag);
  dInvA->replace_diagonal_values(dDiag);

  dummy_map1 = dummy_map2 = nullptr;
  dummy1 = dummy2 = dummy3 = nullptr;
  Core::LinAlg::split_matrix2x2(
      mmatrix, active_dofs, dummy_map1, gmdof, dummy_map2, mA, dummy1, dummy2, dummy3);
  mA->complete(*interface_->master_row_dofs(), *interface_->active_dofs());

  // get dinv * M
  std::shared_ptr<Core::LinAlg::SparseMatrix> dInvMa =
      Core::LinAlg::matrix_multiply(*dInvA, false, *mA, false, false, false, true);

  // we need to add another term, since AssembleLinStick/Slip assumes that we solve
  // for the Lagrange multiplier increments. However, we solve for the LM directly.
  // We can do that, since the system is linear in the LMs.
  tmpv = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->active_dofs());
  std::shared_ptr<Core::LinAlg::Vector<double>> tmpv2 =
      std::make_shared<Core::LinAlg::Vector<double>>(*interface_->active_dofs());
  Core::LinAlg::export_to(*z_, *tmpv2);
  dcsdLMc->multiply(false, *tmpv2, *tmpv);
  tmpv->scale(-1.);
  CONTACT::Utils::add_vector(*tmpv, *fcsa);
  tmpv = nullptr;
  tmpv2 = nullptr;


  // save some matrix blocks for recovery
  dinvA_ = dInvA;
  kss_a_ = kss_a;
  kst_a_ = kst_a;
  rs_a_ = rsa;
  // apply contact symmetry conditions
  if (!sdirichtoggle_) FOUR_C_THROW("you didn't call store_dirichlet_status");
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    double haveDBC = 0;
    sdirichtoggle_->norm_1(&haveDBC);
    if (haveDBC > 0.)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> diag =
          Core::LinAlg::create_vector(*interface_->active_dofs(), true);
      dInvA->extract_diagonal_copy(*diag);
      std::shared_ptr<Core::LinAlg::Vector<double>> lmDBC =
          Core::LinAlg::create_vector(*interface_->active_dofs(), true);
      Core::LinAlg::export_to(*sdirichtoggle_, *lmDBC);
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          Core::LinAlg::create_vector(*interface_->active_dofs(), true);
      tmp->multiply(1., *diag, *lmDBC, 0.);
      diag->update(-1., *tmp, 1.);
      dInvA->replace_diagonal_values(*diag);
      dInvMa = Core::LinAlg::matrix_multiply(*dInvA, false, *mA, false, false, false, true);
    }
  }

  // reset the tangent stiffness
  // (for the condensation we have constructed copies above)
  sysmat->un_complete();

  // need diagonal block kss with explicitdirichtlet_=true
  // to be able to apply dirichlet values for contact symmetry condition
  Core::LinAlg::SparseMatrix tmpkss(
      *gdisp_DofRowMap, 100, false, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  sysmat->assign(0, 0, Core::LinAlg::DataAccess::Copy, tmpkss);

  // get references to the blocks (just for convenience)
  Core::LinAlg::SparseMatrix& kss_new = sysmat->matrix(0, 0);
  Core::LinAlg::SparseMatrix& kst_new = sysmat->matrix(0, 1);
  kss_new.reset();
  kst_new.reset();
  // reynolds equation blocks remain untouched

  // reset rhs
  combined_RHS->put_scalar(0.);
  CONTACT::Utils::add_vector(rt, *combined_RHS);

  // **********************************************************************
  // **********************************************************************
  // BUILD CONDENSED SYSTEM
  // **********************************************************************
  // **********************************************************************

  // (1) add the blocks, we do nothing with (i.e. (Inactive+others))
  kss_new.add(*kss_ni, false, 1., 1.);
  kst_new.add(*kst_ni, false, 1., 1.);
  CONTACT::Utils::add_vector(rsni, *combined_RHS);

  // (2) add the 'uncondensed' blocks (i.e. everything w/o a D^-1
  // (2)a actual stiffness blocks of the master-rows
  kss_new.add(*kss_m, false, 1., 1.);
  kst_new.add(*kst_m, false, 1., 1.);
  CONTACT::Utils::add_vector(rsm, *combined_RHS);

  // (2)b active constraints in the active slave rows
  kss_new.add(*dcsdd, false, 1., 1.);
  CONTACT::Utils::add_vector(*fcsa, *combined_RHS);

  // (3) condensed parts
  // second row
  kss_new.add(*Core::LinAlg::matrix_multiply(*dInvMa, true, *kss_a, false, false, false, true),
      false, 1., 1.);
  kst_new.add(*Core::LinAlg::matrix_multiply(*dInvMa, true, *kst_a, false, false, false, true),
      false, 1., 1.);
  tmpv = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->master_row_dofs());
  if (dInvMa->multiply(true, *rsa, *tmpv)) FOUR_C_THROW("multiply failed");
  CONTACT::Utils::add_vector(*tmpv, *combined_RHS);
  tmpv = nullptr;

  // third row
  std::shared_ptr<Core::LinAlg::SparseMatrix> wDinv =
      Core::LinAlg::matrix_multiply(*dcsdLMc, false, *dInvA, true, false, false, true);
  kss_new.add(*Core::LinAlg::matrix_multiply(*wDinv, false, *kss_a, false, false, false, true),
      false, -1. / (1. - alphaf_), 1.);
  kst_new.add(*Core::LinAlg::matrix_multiply(*wDinv, false, *kst_a, false, false, false, true),
      false, -1. / (1. - alphaf_), 1.);
  tmpv = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->active_dofs());
  wDinv->multiply(false, *rsa, *tmpv);
  tmpv->scale(-1. / (1. - alphaf_));
  CONTACT::Utils::add_vector(*tmpv, *combined_RHS);
  tmpv = nullptr;
  wDinv = nullptr;

  // and were done with the system matrix
  sysmat->complete();

  // we need to return the rhs, not the residual
  combined_RHS->scale(-1.);

  return;
}


void Adapter::CouplingEhlMortar::evaluate_rel_mov()
{
  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = interface()->discret().l_row_node(i);
    if (!node) FOUR_C_THROW("node not found");
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);
    if (!cnode) FOUR_C_THROW("not a contact node");

    cnode->fri_data().get_deriv_jump().resize(3);
    // write it to nodes
    for (int dim = 0; dim < interface_->n_dim(); dim++)
    {
      cnode->fri_data().jump()[dim] = cnode->ehl_data().get_weighted_rel_tang_vel()(dim);
      for (auto p = cnode->ehl_data().get_weighted_rel_tang_vel_deriv().begin();
          p != cnode->ehl_data().get_weighted_rel_tang_vel_deriv().end(); ++p)
        cnode->fri_data().get_deriv_jump()[dim][p->first] = p->second(dim);
    }
  }
}

void Adapter::CouplingEhlMortar::recover_coupled(std::shared_ptr<Core::LinAlg::Vector<double>> sinc,
    std::shared_ptr<Core::LinAlg::Vector<double>> tinc)
{
  const double alphaf_ = 0.;  // statics!

  std::shared_ptr<Core::LinAlg::Vector<double>> z_old = nullptr;
  if (z_ != nullptr) z_old = std::make_shared<Core::LinAlg::Vector<double>>(*z_);

  // recover contact LM
  if (interface_->active_nodes()->num_global_elements() > 0)
  {
    // do we have everything we need?
    if (rs_a_ == nullptr || kss_a_ == nullptr || kst_a_ == nullptr || dinvA_ == nullptr)
      FOUR_C_THROW("some data for LM recovery is missing");

    Core::LinAlg::Vector<double> lmc_a_new(*interface_->active_dofs(), false);
    Core::LinAlg::Vector<double> tmp(*interface_->active_dofs(), false);
    lmc_a_new.update(1., *rs_a_, 0.);
    kss_a_->multiply(false, *sinc, tmp);
    lmc_a_new.update(1., tmp, 1.);
    kst_a_->multiply(false, *tinc, tmp);
    lmc_a_new.update(1., tmp, 1.);
    dinvA_->multiply(false, lmc_a_new, tmp);
    tmp.scale(-1. / (1. - alphaf_));
    z_ = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_dofs());

    Core::LinAlg::export_to(tmp, *z_);
  }

  else
    z_ = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_dofs());

  if (z_old != nullptr)
  {
    z_old->update(-1., *z_, 1.);
    z_old->norm_2(&contact_LM_incr_norm_);
  }

  // store updated LM into nodes
  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(interface_->discret().l_row_node(i));
    for (int dof = 0; dof < interface_->n_dim(); ++dof)
      cnode->mo_data().lm()[dof] = z_->operator[](z_->get_map().lid(cnode->dofs()[dof]));
  }

  return;
}



/*----------------------------------------------------------------------*
 |  Store dirichlet B.C. status into CNode                    popp 06/09|
 *----------------------------------------------------------------------*/
void Adapter::CouplingEhlMortar::store_dirichlet_status(const Core::LinAlg::MapExtractor& dbcmaps)
{
  // loop over all slave row nodes on the current interface
  for (int j = 0; j < interface_->slave_row_nodes()->num_my_elements(); ++j)
  {
    int gid = interface_->slave_row_nodes()->gid(j);
    Core::Nodes::Node* node = interface_->discret().g_node(gid);
    if (!node) FOUR_C_THROW("ERROR: Cannot find node with gid %", gid);
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

    // check if this node's dofs are in dbcmap
    for (int k = 0; k < cnode->num_dof(); ++k)
    {
      int currdof = cnode->dofs()[k];
      int lid = (dbcmaps.cond_map())->lid(currdof);

      // store dbc status if found
      if (lid >= 0 && cnode->dbc_dofs()[k] == false) cnode->set_dbc() = true;

      // check compatibility of contact symmetry condition and displacement dirichlet conditions
      if (lid < 0 && cnode->dbc_dofs()[k] == true)
      {
        std::cout << "node " << cnode->id() << " at: " << cnode->x()[0] << " " << cnode->x()[1]
                  << " " << cnode->x()[2] << std::endl;
        std::cout << "dbcdofs: " << cnode->dbc_dofs()[0] << cnode->dbc_dofs()[1]
                  << cnode->dbc_dofs()[2] << std::endl;
        FOUR_C_THROW(
            "Inconsistency in structure Dirichlet conditions and Mortar symmetry conditions");
      }
    }
  }
  // create old style dirichtoggle vector (supposed to go away)
  sdirichtoggle_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_dofs(), true);
  Core::LinAlg::Vector<double> temp(*(dbcmaps.cond_map()));
  temp.put_scalar(1.0);
  Core::LinAlg::export_to(temp, *sdirichtoggle_);

  return;
}



bool Adapter::CouplingEhlMortar::already_evaluated(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp)
{
  if (!evaluated_state_) return false;
  Core::LinAlg::Vector<double> diff(*disp);
  diff.update(-1., *evaluated_state_, 1.);
  double inf_diff = -1.;
  diff.norm_inf(&inf_diff);
  if (inf_diff < 1.e-13) return true;

  return false;
}

std::shared_ptr<Core::LinAlg::SparseMatrix> Adapter::CouplingEhlMortar::assemble_ehl_lin_d(
    const std::shared_ptr<Core::LinAlg::Vector<double>> x  // slave dof vector
)
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> DLinEHL =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *slavedofrowmap_, 81, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  DLinEHL->zero();
  DLinEHL->un_complete();

  interface_->assemble_coup_lin_d(*DLinEHL, x);

  DLinEHL->complete(*smdofrowmap_, *slavedofrowmap_);

  return DLinEHL;
}

std::shared_ptr<Core::LinAlg::SparseMatrix> Adapter::CouplingEhlMortar::assemble_ehl_lin_m(
    const std::shared_ptr<Core::LinAlg::Vector<double>> x  // slave dof vector
)
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> MLinEHL =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *masterdofrowmap_, 81, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  MLinEHL->zero();
  MLinEHL->un_complete();

  interface_->assemble_coup_lin_m(*MLinEHL, x);

  MLinEHL->complete(*smdofrowmap_, *masterdofrowmap_);

  return MLinEHL;
}

void Adapter::CouplingEhlMortar::assemble_normals()
{
  normals_ = std::make_shared<Core::LinAlg::Vector<double>>(*slave_dof_map(), true);

  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = interface()->discret().g_node(interface_->slave_row_nodes()->gid(i));
    if (!node) FOUR_C_THROW("node not found");
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
    if (!cnode) FOUR_C_THROW("not a contact node");

    for (int d = 0; d < interface_->n_dim(); ++d)
      normals_->replace_global_value(cnode->dofs()[d], cnode->mo_data().n()[d]);
  }
}


void Adapter::CouplingEhlMortar::assemble_normals_deriv()
{
  Nderiv_ = std::make_shared<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);
  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = interface()->discret().g_node(interface_->slave_row_nodes()->gid(i));
    if (!node) FOUR_C_THROW("node not found");
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
    if (!cnode) FOUR_C_THROW("not a contact node");

    for (int d = 0; d < interface()->n_dim(); ++d)
      for (auto p = cnode->data().get_deriv_n()[d].begin();
          p != cnode->data().get_deriv_n()[d].end(); ++p)
        Nderiv_->assemble(p->second, cnode->dofs()[d], p->first);
  }
  Nderiv_->complete();
}

void Adapter::CouplingEhlMortar::assemble_real_gap()
{
  nodal_gap_ = std::make_shared<Core::LinAlg::Vector<double>>(*slavenoderowmap_, true);

  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = interface()->discret().g_node(interface_->slave_row_nodes()->gid(i));
    if (!node) FOUR_C_THROW("node not found");
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
    if (!cnode) FOUR_C_THROW("not a contact node");
    double real_gap = cnode->data().getg();
    switch (cnode->mo_data().get_d().size())
    {
      case 0:
        break;
      case 1:
        if (cnode->mo_data().get_d().begin()->first != cnode->id())
          FOUR_C_THROW("something is wrong. Here should by my own Id");
        real_gap /= cnode->mo_data().get_d().at(cnode->id());
        break;
      default:
        FOUR_C_THROW(
            "GetD should be of size 0 (unprojectable) or 1 (projectable). Are you not using "
            "duals?");
    }
    nodal_gap_->replace_global_value(cnode->id(), real_gap);
  }

  static const double offset =
      Global::Problem::instance()->lubrication_dynamic_params().get<double>("GAP_OFFSET");
  for (int i = 0; i < nodal_gap_->get_map().num_my_elements(); ++i)
    nodal_gap_->get_values()[i] += offset;
}

void Adapter::CouplingEhlMortar::assemble_real_gap_deriv()
{
  deriv_nodal_gap_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);

  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = interface()->discret().g_node(interface_->slave_row_nodes()->gid(i));
    if (!node) FOUR_C_THROW("node not found");
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
    if (!cnode) FOUR_C_THROW("not a contact node");

    if (cnode->data().get_deriv_d().size() != cnode->mo_data().get_d().size())
      FOUR_C_THROW("size inconsistency");

    const double w_gap = cnode->data().getg();
    double d = -1.;
    switch (cnode->data().get_deriv_d().size())
    {
      case 0:
        break;
      case 1:
        if (cnode->data().get_deriv_d().begin()->first != cnode->id())
          FOUR_C_THROW("something is wrong. Here should by my own Id");
        d = cnode->mo_data().get_d().at(cnode->id());
        break;
      default:
        FOUR_C_THROW(
            "GetDerivD should be of size 0 (unprojectable) or 1 (projectable). Are you not using "
            "duals?");
    }

    if (cnode->data().get_deriv_d().size())
      for (auto p = cnode->data().get_deriv_d().at(cnode->id()).begin();
          p != cnode->data().get_deriv_d().at(cnode->id()).end(); ++p)
      {
        const double val = -w_gap / (d * d) * p->second;
        for (int d = 0; d < interface_->n_dim(); ++d)
          deriv_nodal_gap_->assemble(val, cnode->dofs()[d], p->first);
      }

    if (d == -1 && cnode->data().get_deriv_g().size() != 0) FOUR_C_THROW("inconsistency");

    if (cnode->data().get_deriv_g().size())
      for (auto p = cnode->data().get_deriv_g().begin(); p != cnode->data().get_deriv_g().end();
          ++p)
      {
        const double val = p->second / d;
        for (int d = 0; d < interface_->n_dim(); ++d)
          deriv_nodal_gap_->assemble(val, cnode->dofs()[d], p->first);
      }
  }
  deriv_nodal_gap_->complete(*smdofrowmap_, *slavedofrowmap_);
}

void Adapter::CouplingEhlMortar::assemble_interface_velocities(const double dt)
{
  relTangVel_ = std::make_shared<Core::LinAlg::Vector<double>>(*slavedofrowmap_);
  avTangVel_ = std::make_shared<Core::LinAlg::Vector<double>>(*slavedofrowmap_);
  relTangVel_deriv_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);
  avTangVel_deriv_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);

  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = interface()->discret().g_node(interface_->slave_row_nodes()->gid(i));
    if (!node) FOUR_C_THROW("node not found");
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
    if (!cnode) FOUR_C_THROW("not a contact node");


    double d_val = 0.;
    switch (cnode->mo_data().get_d().size())
    {
      case 0:
        break;
      case 1:
        if (cnode->mo_data().get_d().begin()->first != cnode->id())
          FOUR_C_THROW("something is wrong. Here should by my own Id");
        d_val = cnode->mo_data().get_d().at(cnode->id());
        break;
      default:
        FOUR_C_THROW(
            "GetD should be of size 0 (unprojectable) or 1 (projectable). Are you not using "
            "duals?");
    }

    if (d_val == 0.) continue;

    for (int d = 0; d < interface()->n_dim(); ++d)
    {
      relTangVel_->replace_global_value(
          cnode->dofs()[d], cnode->ehl_data().get_weighted_rel_tang_vel()(d) / d_val);
      avTangVel_->replace_global_value(
          cnode->dofs()[d], cnode->ehl_data().get_weighted_av_tang_vel()(d) / d_val);
    }

    for (auto p = cnode->data().get_deriv_d().at(cnode->id()).begin();
        p != cnode->data().get_deriv_d().at(cnode->id()).end(); ++p)
    {
      const int col = p->first;
      for (int d = 0; d < interface()->n_dim(); ++d)
      {
        const int row = cnode->dofs()[d];
        const double rel_val =
            -cnode->ehl_data().get_weighted_rel_tang_vel()(d) / (d_val * d_val) * p->second;
        const double av_val =
            -cnode->ehl_data().get_weighted_av_tang_vel()(d) / (d_val * d_val) * p->second;
        relTangVel_deriv_->assemble(rel_val, row, col);
        avTangVel_deriv_->assemble(av_val, row, col);
      }
    }
    for (auto p = cnode->ehl_data().get_weighted_av_tang_vel_deriv().begin();
        p != cnode->ehl_data().get_weighted_av_tang_vel_deriv().end(); ++p)
    {
      const int col = p->first;
      for (int d = 0; d < interface()->n_dim(); ++d)
      {
        const int row = cnode->dofs()[d];
        const double val = p->second(d) / d_val;
        avTangVel_deriv_->assemble(val, row, col);
      }
    }
    for (auto p = cnode->ehl_data().get_weighted_rel_tang_vel_deriv().begin();
        p != cnode->ehl_data().get_weighted_rel_tang_vel_deriv().end(); ++p)
    {
      const int col = p->first;
      for (int d = 0; d < interface()->n_dim(); ++d)
      {
        const int row = cnode->dofs()[d];
        const double val = p->second(d) / d_val;
        relTangVel_deriv_->assemble(val, row, col);
      }
    }
  }

  relTangVel_->scale(1. / dt);
  avTangVel_->scale(1. / dt);
  relTangVel_deriv_->complete(*smdofrowmap_, *slavedofrowmap_);
  avTangVel_deriv_->complete(*smdofrowmap_, *slavedofrowmap_);
  relTangVel_deriv_->scale(1. / dt);
  avTangVel_deriv_->scale(1. / dt);
}

void Adapter::CouplingEhlMortar::assemble_surf_grad()
{
  SurfGrad_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *slavedofrowmap_, 81, false, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = interface()->discret().g_node(interface_->slave_row_nodes()->gid(i));
    if (!node) FOUR_C_THROW("ERROR: Cannot find node");
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
    if (!cnode) FOUR_C_THROW("this is not a contact node");

    double dval = 1.;
    switch (cnode->mo_data().get_d().size())
    {
      case 0:
        dval = 1.e32;
        break;  // large number so no tangential gradient
      case 1:
        if (cnode->mo_data().get_d().begin()->first != cnode->id())
          FOUR_C_THROW("something is wrong. Here should by my own Id");
        dval = cnode->mo_data().get_d().at(cnode->id());
        break;
      default:
        FOUR_C_THROW(
            "GetD should be of size 0 (unprojectable) or 1 (projectable). Are you not using "
            "duals?");
    }

    for (auto p = cnode->ehl_data().get_surf_grad().begin();
        p != cnode->ehl_data().get_surf_grad().end(); ++p)
      for (int d = 0; d < interface()->n_dim(); ++d)
        SurfGrad_->assemble(p->second(d) / dval, cnode->dofs()[d], p->first);
  }

  SurfGrad_->complete();
}

std::shared_ptr<Core::LinAlg::SparseMatrix> Adapter::CouplingEhlMortar::assemble_surf_grad_deriv(
    const Core::LinAlg::Vector<double>& x)
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> SurfGradDeriv =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *slavedofrowmap_, 81, false, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = interface()->discret().g_node(interface_->slave_row_nodes()->gid(i));
    if (!node) FOUR_C_THROW("ERROR: Cannot find node");
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
    if (!cnode) FOUR_C_THROW("this is not a contact node");

    double dval = 1.;
    switch (cnode->mo_data().get_d().size())
    {
      case 0:
        dval = 1.e32;
        break;  // large number so no tangential gradient
      case 1:
        if (cnode->mo_data().get_d().begin()->first != cnode->id())
          FOUR_C_THROW("something is wrong. Here should by my own Id");
        dval = cnode->mo_data().get_d().at(cnode->id());
        break;
      default:
        FOUR_C_THROW(
            "GetD should be of size 0 (unprojectable) or 1 (projectable). Are you not using "
            "duals?");
    }

    for (auto p = cnode->ehl_data().get_surf_grad_deriv().begin();
        p != cnode->ehl_data().get_surf_grad_deriv().end(); ++p)
    {
      const int col = p->first;
      for (auto q = p->second.begin(); q != p->second.end(); ++q)
      {
        const int lid = x.get_map().lid(q->first);
        if (lid < 0) FOUR_C_THROW("not my gid");
        const double x_val = x.operator[](lid);
        for (int d = 0; d < interface()->n_dim(); ++d)
        {
          const double val = x_val * q->second(d) / dval;
          SurfGradDeriv->assemble(val, cnode->dofs()[d], col);
        }
      }
    }

    if (cnode->data().get_deriv_d().size())
      for (auto p = cnode->data().get_deriv_d().at(cnode->id()).begin();
          p != cnode->data().get_deriv_d().at(cnode->id()).end(); ++p)
      {
        const int col = p->first;

        for (auto q = cnode->ehl_data().get_surf_grad().begin();
            q != cnode->ehl_data().get_surf_grad().end(); ++q)
          for (int d = 0; d < interface()->n_dim(); ++d)
          {
            const int row = cnode->dofs()[d];
            const int x_gid = q->first;
            const int x_lid = x.get_map().lid(x_gid);
            if (x_lid < 0) FOUR_C_THROW("not my gid");
            double x_val = x.operator[](x_lid);
            const double val = -x_val * q->second(d) / (dval * dval) * p->second;
            SurfGradDeriv->assemble(val, row, col);
          }
      }
  }
  SurfGradDeriv->complete(*smdofrowmap_, *slavedofrowmap_);
  return SurfGradDeriv;
}


void Adapter::CouplingEhlMortar::create_force_vec(std::shared_ptr<Core::LinAlg::Vector<double>>& n,
    std::shared_ptr<Core::LinAlg::Vector<double>>& t)
{
  n = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_dofs());
  t = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_dofs());
  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(interface_->discret().l_row_node(i));
    if (!cnode) FOUR_C_THROW("cast failed");
    const Core::LinAlg::Matrix<3, 1> lm(cnode->mo_data().lm(), true);
    const Core::LinAlg::Matrix<3, 1> nor(cnode->mo_data().n(), true);
    Core::LinAlg::Matrix<3, 3> nn;
    nn.multiply_nt(nor, nor);
    Core::LinAlg::Matrix<3, 1> lmn;
    lmn.multiply(nn, lm);
    Core::LinAlg::Matrix<3, 1> lmt(lm);
    lmt.update(-1., lmn, 1.);
    for (int d = 0; d < 3; ++d)
    {
      n->get_values()[(n->get_map().lid(cnode->dofs()[d]))] = lmn(d);
      t->get_values()[(t->get_map().lid(cnode->dofs()[d]))] = lmt(d);
    }
  }
}

void Adapter::CouplingEhlMortar::create_active_slip_toggle(
    std::shared_ptr<Core::LinAlg::Vector<double>>* active,
    std::shared_ptr<Core::LinAlg::Vector<double>>* slip,
    std::shared_ptr<Core::LinAlg::Vector<double>>* active_old)
{
  *active = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_nodes());
  *slip = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_nodes());
  if (active_old != nullptr)
    *active_old = std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_nodes());
  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(interface_->discret().l_row_node(i));
    if (!cnode) FOUR_C_THROW("cast failed");
    if (cnode->active())
      (*active)->get_values()[i] = 1.;
    else
      (*active)->get_values()[i] = 0.;
    if (cnode->fri_data().slip())
      (*slip)->get_values()[i] = 1.;
    else
      (*slip)->get_values()[i] = 0.;

    if (active_old != nullptr)
    {
      if (cnode->data().active_old())
        (*active_old)->get_values()[i] = 1.;
      else
        (*active_old)->get_values()[i] = 0.;
    }
  }
}

void Adapter::CouplingEhlMortar::write_restart(Core::IO::DiscretizationWriter& output)
{
  if (!contact_regularization_) return;

  output.write_vector("last_contact_force", fscn_);
  output.write_vector("contact_lm", z_);

  std::shared_ptr<Core::LinAlg::Vector<double>> active_toggle, active_old_toggle, slip_toggle;
  create_active_slip_toggle(&active_toggle, &slip_toggle, &active_old_toggle);

  output.write_vector("active_toggle", active_toggle);
  output.write_vector("active_old_toggle", active_old_toggle);
  output.write_vector("slip_toggle", slip_toggle);
}

void Adapter::CouplingEhlMortar::read_restart(Core::IO::DiscretizationReader& reader)
{
  if (!contact_regularization_) return;

  reader.read_vector(fscn_, "last_contact_force");
  reader.read_vector(z_, "contact_lm");

  std::shared_ptr<Core::LinAlg::Vector<double>> active_toggle =
      std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_nodes());
  std::shared_ptr<Core::LinAlg::Vector<double>> active_old_toggle =
      std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_nodes());
  std::shared_ptr<Core::LinAlg::Vector<double>> slip_toggle =
      std::make_shared<Core::LinAlg::Vector<double>>(*interface_->slave_row_nodes());
  reader.read_vector(active_toggle, "active_toggle");
  reader.read_vector(active_old_toggle, "active_old_toggle");
  reader.read_vector(slip_toggle, "slip_toggle");

  for (int i = 0; i < interface_->slave_row_nodes()->num_my_elements(); ++i)
  {
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(interface_->discret().l_row_node(i));
    if (!cnode) FOUR_C_THROW("cast failed");
    cnode->active() = active_toggle->operator[](i);
    cnode->fri_data().slip() = slip_toggle->operator[](i);
    cnode->data().active_old() = active_old_toggle->operator[](i);
    for (int d = 0; d < interface_->n_dim(); ++d)
      cnode->mo_data().lm()[d] = z_->operator[](z_->get_map().lid(cnode->dofs()[d]));
  }
}

int Adapter::CouplingEhlMortar::active_contact()
{
  return interface_->active_nodes()->num_global_elements();
}

int Adapter::CouplingEhlMortar::slip_contact()
{
  return interface_->slip_nodes()->num_global_elements();
}

FOUR_C_NAMESPACE_CLOSE
