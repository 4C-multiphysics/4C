// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_lagrange_strategy_tsi.hpp"

#include "4C_contact_defines.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_tsi_interface.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_io.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_thermo_input.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                             seitz 08/15|
 *----------------------------------------------------------------------*/
CONTACT::LagrangeStrategyTsi::LagrangeStrategyTsi(
    const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
    const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
    Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interface,
    int dim, MPI_Comm comm, double alphaf, int maxdof)
    : LagrangeStrategy(
          data_ptr, dof_row_map, NodeRowMap, params, interface, dim, comm, alphaf, maxdof),
      tsi_alpha_(1.)
{
}

/*------------------------------------------------------------------------*
 | Assign general thermo contact state                         seitz 08/15|
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyTsi::set_state(
    const Mortar::StateType& statetype, const Core::LinAlg::Vector<double>& vec)
{
  switch (statetype)
  {
    case Mortar::state_temperature:
    {
      for (int j = 0; j < (int)interface_.size(); ++j)
      {
        Core::FE::Discretization& idiscr = interface_[j]->discret();
        Core::LinAlg::Vector<double> global(*idiscr.dof_col_map(), false);
        Core::LinAlg::export_to(vec, global);

        for (int i = 0; i < idiscr.num_my_col_nodes(); ++i)
        {
          CONTACT::Node* node = dynamic_cast<CONTACT::Node*>(idiscr.l_col_node(i));
          if (node == nullptr) FOUR_C_THROW("cast failed");
          std::vector<int> lm(1, node->dofs()[0]);

          std::vector<double> mytemp = Core::FE::extract_values(global, lm);
          if (node->has_tsi_data())  // in case the interface has not been initialized yet
            node->tsi_data().temp() = mytemp[0];
        }
      }
      break;
    }
    case Mortar::state_thermo_lagrange_multiplier:
    {
      for (int j = 0; j < (int)interface_.size(); ++j)
      {
        Core::FE::Discretization& idiscr = interface_[j]->discret();

        Core::LinAlg::Vector<double> global(*idiscr.dof_col_map(), false);
        Core::LinAlg::export_to(vec, global);

        for (int i = 0; i < idiscr.num_my_col_nodes(); ++i)
        {
          CONTACT::Node* node = dynamic_cast<CONTACT::Node*>(idiscr.l_col_node(i));
          std::vector<int> lm(1, node->dofs()[0]);
          std::vector<double> myThermoLM = Core::FE::extract_values(global, lm);
          node->tsi_data().thermo_lm() = myThermoLM[0];
        }
      }
      break;
    }
    default:
    {
      CONTACT::AbstractStrategy::set_state(statetype, vec);
      break;
    }
  }

  return;
}


void CONTACT::LagrangeStrategyTsi::evaluate(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>>& combined_RHS,
    std::shared_ptr<Coupling::Adapter::Coupling> coupST,
    std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<const Core::LinAlg::Vector<double>> temp)
{
  if (thermo_s_dofs_ == nullptr) thermo_s_dofs_ = coupST->master_to_slave_map(*gsdofrowmap_);

  // set the new displacements
  set_state(Mortar::state_new_displacement, *dis);

  for (unsigned i = 0; i < interface_.size(); ++i) interface_[i]->initialize();

  // set new temperatures
  std::shared_ptr<Core::LinAlg::Vector<double>> temp2 = coupST->slave_to_master(*temp);
  set_state(Mortar::state_temperature, *temp2);

  // error checks
  if (Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM") !=
      CONTACT::SystemType::condensed)
    FOUR_C_THROW("only condensed system implemented");

  // First, we need to evaluate all the interfaces
  initialize_mortar();                  // initialize mortar matrices and vectors
  initialize_and_evaluate_interface();  // evaluate mortar terms (integrate...)
  assemble_mortar();

  // get the relative movement for frictional contact
  evaluate_relative_movement();

  // update active set
  update_active_set_semi_smooth();

  // init lin-matrices
  initialize();

  // get the necessary maps on the thermo dofs
  std::shared_ptr<Core::LinAlg::Map> gactive_themo_dofs =
      coupST->master_to_slave_map(*gactivedofs_);
  std::shared_ptr<Core::LinAlg::Map> master_thermo_dofs =
      coupST->master_to_slave_map(*gmdofrowmap_);
  std::shared_ptr<Core::LinAlg::Map> thermo_act_dofs = coupST->master_to_slave_map(*gactivedofs_);
  std::shared_ptr<Core::LinAlg::Map> thermo_m_dofs = coupST->master_to_slave_map(*gmdofrowmap_);
  std::shared_ptr<Core::LinAlg::Map> thermo_sm_dofs = coupST->master_to_slave_map(*gsmdofrowmap_);
  std::shared_ptr<Core::LinAlg::Map> thermo_all_dofs =
      std::make_shared<Core::LinAlg::Map>(*coupST->slave_dof_map());

  // assemble the constraint lines for the active contact nodes
  std::shared_ptr<Core::LinAlg::SparseMatrix> dcsdd = std::make_shared<Core::LinAlg::SparseMatrix>(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix dcsdT(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  std::shared_ptr<Core::LinAlg::SparseMatrix> dcsdLMc =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  std::shared_ptr<Core::LinAlg::Vector<double>> rcsa =
      std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_, true);
  std::shared_ptr<Core::LinAlg::Vector<double>> g_all;
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
    g_all = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_, true);
  else
    g_all = std::make_shared<Core::LinAlg::Vector<double>>(*gsnoderowmap_, true);

  // assemble linearization of heat conduction (thermal contact)
  Core::LinAlg::SparseMatrix dcTdd(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix dcTdT(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix dcTdLMc(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix dcTdLMt(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  std::shared_ptr<Core::LinAlg::Vector<double>> rcTa =
      std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_, true);

  // D and M matrix for the active nodes
  Core::LinAlg::SparseMatrix dInv(*gsdofrowmap_, 100, true, false);
  Core::LinAlg::SparseMatrix s(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  Core::LinAlg::SparseMatrix m_LinDissDISP(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix m_LinDissContactLM(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // setup some linearizations
  Core::LinAlg::SparseMatrix linDcontactLM(
      *gsdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix linMcontactLM(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix linMdiss(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix linMThermoLM(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix linDThermoLM(
      *gsdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix m_LinDissContactLM_thermoRow(
      *thermo_m_dofs, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // stick / slip linearization
  for (unsigned i = 0; i < interface_.size(); ++i)
  {
    CONTACT::TSIInterface* tsi_interface = dynamic_cast<CONTACT::TSIInterface*>(&(*interface_[i]));
    if (!tsi_interface) FOUR_C_THROW("in TSI contact, this should be a TSIInterface!");

    // linearized normal contact
    interface_[i]->assemble_s(s);
    interface_[i]->assemble_g(*g_all);

    // linearized tangential contact (friction)
    if (friction_)
    {
      tsi_interface->assemble_lin_slip(*dcsdLMc, *dcsdd, dcsdT, *rcsa);
      tsi_interface->assemble_lin_stick(*dcsdLMc, *dcsdd, dcsdT, *rcsa);
    }
    else
    {
      tsi_interface->assemble_tn(dcsdLMc, nullptr);
      tsi_interface->assemble_t_nderiv(dcsdd, nullptr);
      tsi_interface->assemble_tangrhs(*rcsa);
    }

    // linearized thermal contact (heat conduction)
    tsi_interface->assemble_lin_conduct(dcTdd, dcTdT, dcTdLMt, dcTdLMc);

    tsi_interface->assemble_dm_lin_diss(nullptr, &m_LinDissDISP, nullptr, &m_LinDissContactLM, 1.);

    tsi_interface->assemble_lin_dm(linDcontactLM, linMcontactLM);
    tsi_interface->assemble_lin_dm_x(
        nullptr, &linMdiss, 1., CONTACT::TSIInterface::LinDM_Diss, gsnoderowmap_);
    tsi_interface->assemble_lin_dm_x(
        &linDThermoLM, &linMThermoLM, 1., CONTACT::TSIInterface::LinDM_ThermoLM, gsnoderowmap_);
  }

  // complete all those linearizations
  //                             colmap        rowmap
  linDcontactLM.complete(*gsmdofrowmap_, *gsdofrowmap_);
  linMcontactLM.complete(*gsmdofrowmap_, *gmdofrowmap_);
  m_LinDissDISP.complete(*gsmdofrowmap_, *gmdofrowmap_);
  linMdiss.complete(*gsmdofrowmap_, *gmdofrowmap_);
  linMThermoLM.complete(*gsmdofrowmap_, *gmdofrowmap_);
  linDThermoLM.complete(*gsmdofrowmap_, *gsdofrowmap_);
  m_LinDissContactLM.complete(*gactivedofs_, *gmdofrowmap_);
  s.complete(*gsmdofrowmap_, *gactivedofs_);

  Core::LinAlg::matrix_add(s, false, 1., *dcsdd, -1.);
  dcsdLMc->scale(-1.);
  dcsdT.scale(-1.);
  rcsa->scale(1.);

  // normal contact
  std::shared_ptr<Core::LinAlg::Vector<double>> gact;
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    gact = std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_, true);
    if (gact->global_length()) Core::LinAlg::export_to(*g_all, *gact);
  }
  else
  {
    gact = std::make_shared<Core::LinAlg::Vector<double>>(*gactivenodes_, true);
    if (gact->global_length())
    {
      Core::LinAlg::export_to(*g_all, *gact);
      gact->replace_map(*gactiven_);
    }
  }
  CONTACT::Utils::add_vector(*gact, *rcsa);
  rcsa->norm_2(&mech_contact_res_);

  // complete all the new matrix blocks
  // Note: since the contact interface assemled them, they are all based
  //       on displacement row and col maps. Hence, some still need to be transformed
  dcsdd->complete(*gsmdofrowmap_, *gactivedofs_);
  dcsdT.complete(*gsmdofrowmap_, *gactivedofs_);
  dcsdLMc->complete(*gactivedofs_, *gactivedofs_);
  dcTdd.complete(*gsmdofrowmap_, *gactivedofs_);
  dcTdT.complete(*gsmdofrowmap_, *gactivedofs_);
  dcTdLMc.complete(*gactivedofs_, *gactivedofs_);

  // get the separate blocks of the 2x2 TSI block system
  // View mode!!! Since we actually want to add things there
  std::shared_ptr<Core::LinAlg::SparseMatrix> kss = std::make_shared<Core::LinAlg::SparseMatrix>(
      sysmat->matrix(0, 0), Core::LinAlg::DataAccess::Copy);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kst = std::make_shared<Core::LinAlg::SparseMatrix>(
      sysmat->matrix(0, 1), Core::LinAlg::DataAccess::Copy);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kts = std::make_shared<Core::LinAlg::SparseMatrix>(
      sysmat->matrix(1, 0), Core::LinAlg::DataAccess::Copy);
  std::shared_ptr<Core::LinAlg::SparseMatrix> ktt = std::make_shared<Core::LinAlg::SparseMatrix>(
      sysmat->matrix(1, 1), Core::LinAlg::DataAccess::Copy);
  kss->un_complete();
  kts->un_complete();

  // split rhs
  Core::LinAlg::Vector<double> rs(*gdisprowmap_, true);
  Core::LinAlg::Vector<double> rt(*coupST->slave_dof_map(), true);
  Core::LinAlg::export_to(*combined_RHS, rs);
  Core::LinAlg::export_to(*combined_RHS, rt);

  // we don't want the rhs but the residual
  rs.scale(-1.);
  rt.scale(-1.);

  // add last time step contact forces to rhs
  if (fscn_ != nullptr)  // in the first time step, we don't have any history of the
                         // contact force, after that, fscn_ should be initialized properly
  {
    Core::LinAlg::Vector<double> tmp(*gdisprowmap_);
    Core::LinAlg::export_to(*fscn_, tmp);
    rs.update(alphaf_, tmp, 1.);  // fscn already scaled with alphaf_ in update
  }

  if (ftcn_ != nullptr)
  {
    Core::LinAlg::Vector<double> tmp(*coupST->slave_dof_map());
    Core::LinAlg::export_to(*ftcn_, tmp);
    rt.update((1. - tsi_alpha_), tmp, 1.);
  }

  // map containing the inactive and non-contact structural dofs
  std::shared_ptr<Core::LinAlg::Map> str_gni_dofs = Core::LinAlg::split_map(
      *Core::LinAlg::split_map(*gdisprowmap_, *gmdofrowmap_), *gactivedofs_);
  // map containing the inactive and non-contact thermal dofs
  std::shared_ptr<Core::LinAlg::Map> thermo_gni_dofs = coupST->master_to_slave_map(*str_gni_dofs);

  // add to kss
  Core::LinAlg::matrix_add(linDcontactLM, false, 1. - alphaf_, *kss, 1.);
  Core::LinAlg::matrix_add(linMcontactLM, false, 1. - alphaf_, *kss, 1.);

  // transform and add to kts
  Coupling::Adapter::MatrixRowTransform()(
      m_LinDissDISP, +tsi_alpha_, Coupling::Adapter::CouplingMasterConverter(*coupST), *kts, true);
  Coupling::Adapter::MatrixRowTransform()(linMdiss,
      -tsi_alpha_,  // this minus sign is there, since assemble linM does not actually
      Coupling::Adapter::CouplingMasterConverter(*coupST), *kts,
      true);  // assemble the linearization of M but the negative linearization of M
  Coupling::Adapter::MatrixRowTransform()(
      linMThermoLM, tsi_alpha_, Coupling::Adapter::CouplingMasterConverter(*coupST), *kts, true);
  Coupling::Adapter::MatrixRowTransform()(
      linDThermoLM, tsi_alpha_, Coupling::Adapter::CouplingMasterConverter(*coupST), *kts, true);

  Coupling::Adapter::MatrixRowTransform().operator()(m_LinDissContactLM, 1.,
      Coupling::Adapter::CouplingMasterConverter(*coupST), m_LinDissContactLM_thermoRow, false);
  m_LinDissContactLM_thermoRow.complete(*gactivedofs_, *thermo_m_dofs);

  // complete the matrix blocks again, now that we have added
  // the additional displacement linearizations
  kss->complete();
  kts->complete(*gdisprowmap_, *coupST->slave_dof_map());

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
      kss, str_gni_dofs, dummy_map1, gdisprowmap_, dummy_map2, kss_ni, dummy1, tmp, dummy2);

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
      tmp, gmdofrowmap_, dummy_map1, gdisprowmap_, dummy_map2, kss_m, dummy1, kss_a, dummy2);

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
      kst, str_gni_dofs, dummy_map1, thermo_all_dofs, dummy_map2, kst_ni, dummy1, tmp, dummy2);

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
      tmp, gmdofrowmap_, dummy_map1, thermo_all_dofs, dummy_map2, kst_m, dummy1, kst_a, dummy2);

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
  // split kts block*************************************
  // ****************************************************
  // split first row
  Core::LinAlg::split_matrix2x2(
      kts, thermo_gni_dofs, dummy_map1, gdisprowmap_, dummy_map2, kts_ni, dummy1, tmp, dummy2);

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
      tmp, thermo_m_dofs, dummy_map1, gdisprowmap_, dummy_map2, kts_m, dummy1, kts_a, dummy2);

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
  // split kts block*************************************
  // ****************************************************

  // ****************************************************
  // split ktt block*************************************
  // ****************************************************
  // split first row
  Core::LinAlg::split_matrix2x2(
      ktt, thermo_gni_dofs, dummy_map1, thermo_all_dofs, dummy_map2, ktt_ni, dummy1, tmp, dummy2);

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
      tmp, thermo_m_dofs, dummy_map1, thermo_all_dofs, dummy_map2, ktt_m, dummy1, ktt_a, dummy2);

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
  // split ktt block*************************************
  // ****************************************************

  // ****************************************************
  // split rhs vectors***********************************
  // ****************************************************
  // split structural rhs
  Core::LinAlg::Vector<double> rsni(*str_gni_dofs);
  Core::LinAlg::export_to(rs, rsni);
  Core::LinAlg::Vector<double> rsm(*gmdofrowmap_);
  Core::LinAlg::export_to(rs, rsm);
  std::shared_ptr<Core::LinAlg::Vector<double>> rsa =
      std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
  Core::LinAlg::export_to(rs, *rsa);

  // split thermal rhs
  Core::LinAlg::Vector<double> rtni(*thermo_gni_dofs);
  Core::LinAlg::export_to(rt, rtni);
  Core::LinAlg::Vector<double> rtm(*thermo_m_dofs);
  Core::LinAlg::export_to(rt, rtm);
  std::shared_ptr<Core::LinAlg::Vector<double>> rta =
      std::make_shared<Core::LinAlg::Vector<double>>(*thermo_act_dofs);
  Core::LinAlg::export_to(rt, *rta);
  // ****************************************************
  // split rhs vectors***********************************
  // ****************************************************

  // D and M matrix for the active nodes
  std::shared_ptr<Core::LinAlg::SparseMatrix> dInvA =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gactivedofs_, 100, true, false);
  std::shared_ptr<Core::LinAlg::SparseMatrix> mA =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gactivedofs_, 100, true, false);

  dummy_map1 = dummy_map2 = nullptr;
  dummy1 = dummy2 = dummy3 = nullptr;
  Core::LinAlg::split_matrix2x2(
      dmatrix_, gactivedofs_, dummy_map1, gactivedofs_, dummy_map2, dInvA, dummy1, dummy2, dummy3);
  dummy_map1 = dummy_map2 = nullptr;
  dummy1 = dummy2 = dummy3 = nullptr;
  Core::LinAlg::split_matrix2x2(
      mmatrix_, gactivedofs_, dummy_map1, gmdofrowmap_, dummy_map2, mA, dummy1, dummy2, dummy3);

  // now we have added the additional linearizations.
  // if there are no active nodes, we can leave now

  if (gactivenodes_->num_global_elements() == 0)
  {
    sysmat->reset();
    sysmat->assign(0, 0, Core::LinAlg::DataAccess::Copy, *kss);
    sysmat->assign(0, 1, Core::LinAlg::DataAccess::Copy, *kst);
    sysmat->assign(1, 0, Core::LinAlg::DataAccess::Copy, *kts);
    sysmat->assign(1, 1, Core::LinAlg::DataAccess::Copy, *ktt);
    return;
  }


  // we need to add another term, since AssembleLinStick/Slip assumes that we solve
  // for the Lagrange multiplier increments. However, we solve for the LM directly.
  // We can do that, since the system is linear in the LMs.
  tmpv = std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
  std::shared_ptr<Core::LinAlg::Vector<double>> tmpv2 =
      std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
  Core::LinAlg::export_to(*z_, *tmpv2);
  dcsdLMc->multiply(false, *tmpv2, *tmpv);
  tmpv->scale(-1.);
  CONTACT::Utils::add_vector(*tmpv, *rcsa);
  tmpv = nullptr;
  tmpv2 = nullptr;

  dcTdLMt.complete(*gsdofrowmap_, *gactivedofs_);
  Core::LinAlg::SparseMatrix test(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  std::shared_ptr<Core::LinAlg::SparseMatrix> a1 = Core::Utils::shared_ptr_from_ref(dcTdLMt);
  std::shared_ptr<Core::LinAlg::SparseMatrix> a2;
  dummy_map1 = dummy_map2 = nullptr;
  dummy1 = dummy2 = dummy3 = nullptr;
  Core::LinAlg::split_matrix2x2(
      a1, gactivedofs_, dummy_map1, gactivedofs_, dummy_map2, a2, dummy1, dummy2, dummy3);
  dcTdLMt = *a2;

  dcTdLMt.complete(*gactivedofs_, *gactivedofs_);
  dInvA->complete(*gactivedofs_, *gactivedofs_);
  mA->complete(*gmdofrowmap_, *gactivedofs_);

  Core::LinAlg::SparseMatrix dcTdLMc_thermo(
      *thermo_act_dofs, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix dcTdLMt_thermo(
      *thermo_act_dofs, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Coupling::Adapter::MatrixRowTransform()(
      dcTdLMc, 1., Coupling::Adapter::CouplingMasterConverter(*coupST), dcTdLMc_thermo, true);
  Coupling::Adapter::MatrixRowColTransform()(dcTdLMt, 1.,
      Coupling::Adapter::CouplingMasterConverter(*coupST),
      Coupling::Adapter::CouplingMasterConverter(*coupST), dcTdLMt_thermo, true, false);
  dcTdLMc_thermo.complete(*gactivedofs_, *thermo_act_dofs);
  dcTdLMt_thermo.complete(*thermo_act_dofs, *thermo_act_dofs);

  // invert D-matrix
  Core::LinAlg::Vector<double> dDiag(*gactivedofs_);
  dInvA->extract_diagonal_copy(dDiag);
  dDiag.reciprocal(dDiag);
  dInvA->replace_diagonal_values(dDiag);

  // get dinv on thermal dofs
  std::shared_ptr<Core::LinAlg::SparseMatrix> dInvaThermo =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *thermo_act_dofs, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Coupling::Adapter::MatrixRowColTransform()(*dInvA, 1.,
      Coupling::Adapter::CouplingMasterConverter(*coupST),
      Coupling::Adapter::CouplingMasterConverter(*coupST), *dInvaThermo, false, false);
  dInvaThermo->complete(*thermo_act_dofs, *thermo_act_dofs);

  // save some matrix blocks for recovery
  dinvA_ = dInvA;
  dinvAthr_ = dInvaThermo;
  kss_a_ = kss_a;
  kst_a_ = kst_a;
  kts_a_ = kts_a;
  ktt_a_ = ktt_a;
  rs_a_ = rsa;
  rt_a_ = rta;
  thermo_act_dofs_ = thermo_act_dofs;

  // get dinv * M
  std::shared_ptr<Core::LinAlg::SparseMatrix> dInvMa =
      Core::LinAlg::matrix_multiply(*dInvA, false, *mA, false, false, false, true);

  // get dinv * M on the thermal dofs
  Core::LinAlg::SparseMatrix dInvMaThermo(
      *thermo_act_dofs, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Coupling::Adapter::MatrixRowColTransform()(*dInvMa, 1.,
      Coupling::Adapter::CouplingMasterConverter(*coupST),
      Coupling::Adapter::CouplingMasterConverter(*coupST), dInvMaThermo, false, false);
  dInvMaThermo.complete(*thermo_m_dofs, *thermo_act_dofs);

  // apply contact symmetry conditions
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    double haveDBC = 0;
    non_redist_gsdirichtoggle_->norm_1(&haveDBC);
    if (haveDBC > 0.)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> diag =
          std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_, true);
      dInvA->extract_diagonal_copy(*diag);
      std::shared_ptr<Core::LinAlg::Vector<double>> lmDBC =
          std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_, true);
      Core::LinAlg::export_to(*non_redist_gsdirichtoggle_, *lmDBC);
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_, true);
      tmp->multiply(1., *diag, *lmDBC, 0.);
      diag->update(-1., *tmp, 1.);
      dInvA->replace_diagonal_values(*diag);
      dInvMa = Core::LinAlg::matrix_multiply(*dInvA, false, *mA, false, false, false, true);
    }
  }

  // reset the tangent stiffness
  // (for the condensation we have constructed copies above)
  sysmat->reset();
  sysmat->un_complete();

  // need diagonal block kss with explicitdirichtlet_=true
  // to be able to apply dirichlet values for contact symmetry condition
  Core::LinAlg::SparseMatrix tmpkss(
      *gdisprowmap_, 100, false, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  sysmat->assign(0, 0, Core::LinAlg::DataAccess::Copy, tmpkss);

  // get references to the blocks (just for convenience)
  Core::LinAlg::SparseMatrix& kss_new = sysmat->matrix(0, 0);
  Core::LinAlg::SparseMatrix& kst_new = sysmat->matrix(0, 1);
  Core::LinAlg::SparseMatrix& kts_new = sysmat->matrix(1, 0);
  Core::LinAlg::SparseMatrix& ktt_new = sysmat->matrix(1, 1);

  // reset rhs
  combined_RHS->put_scalar(0.0);

  // **********************************************************************
  // **********************************************************************
  // BUILD CONDENSED SYSTEM
  // **********************************************************************
  // **********************************************************************

  // (1) add the blocks, we do nothing with (i.e. (Inactive+others))
  Core::LinAlg::matrix_add(*kss_ni, false, 1., kss_new, 1.);
  Core::LinAlg::matrix_add(*kst_ni, false, 1., kst_new, 1.);
  Core::LinAlg::matrix_add(*kts_ni, false, 1., kts_new, 1.);
  Core::LinAlg::matrix_add(*ktt_ni, false, 1., ktt_new, 1.);
  CONTACT::Utils::add_vector(rsni, *combined_RHS);
  CONTACT::Utils::add_vector(rtni, *combined_RHS);

  // (2) add the 'uncondensed' blocks (i.e. everything w/o a D^-1
  // (2)a actual stiffness blocks of the master-rows
  Core::LinAlg::matrix_add(*kss_m, false, 1., kss_new, 1.);
  Core::LinAlg::matrix_add(*kst_m, false, 1., kst_new, 1.);
  Core::LinAlg::matrix_add(*kts_m, false, 1., kts_new, 1.);
  Core::LinAlg::matrix_add(*ktt_m, false, 1., ktt_new, 1.);
  CONTACT::Utils::add_vector(rsm, *combined_RHS);
  CONTACT::Utils::add_vector(rtm, *combined_RHS);

  // (2)b active constraints in the active slave rows
  Core::LinAlg::matrix_add(*dcsdd, false, 1., kss_new, 1.);

  Coupling::Adapter::MatrixColTransform()(*gactivedofs_, *gsmdofrowmap_, dcsdT, 1.,
      Coupling::Adapter::CouplingMasterConverter(*coupST), kst_new, false, true);
  Coupling::Adapter::MatrixRowTransform()(
      dcTdd, 1., Coupling::Adapter::CouplingMasterConverter(*coupST), kts_new, true);
  Coupling::Adapter::MatrixRowColTransform()(dcTdT, 1.,
      Coupling::Adapter::CouplingMasterConverter(*coupST),
      Coupling::Adapter::CouplingMasterConverter(*coupST), ktt_new, true, true);
  CONTACT::Utils::add_vector(*rcsa, *combined_RHS);

  // (3) condensed parts
  // second row
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*dInvMa, true, *kss_a, false, false, false, true), false, 1.,
      kss_new, 1.);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*dInvMa, true, *kst_a, false, false, false, true), false, 1.,
      kst_new, 1.);
  tmpv = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_);
  dInvMa->multiply(true, *rsa, *tmpv);
  CONTACT::Utils::add_vector(*tmpv, *combined_RHS);
  tmpv = nullptr;

  // third row
  std::shared_ptr<Core::LinAlg::SparseMatrix> wDinv =
      Core::LinAlg::matrix_multiply(*dcsdLMc, false, *dInvA, true, false, false, true);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*wDinv, false, *kss_a, false, false, false, true), false,
      -1. / (1. - alphaf_), kss_new, 1.);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*wDinv, false, *kst_a, false, false, false, true), false,
      -1. / (1. - alphaf_), kst_new, 1.);
  tmpv = std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
  wDinv->multiply(false, *rsa, *tmpv);
  tmpv->scale(-1. / (1. - alphaf_));
  CONTACT::Utils::add_vector(*tmpv, *combined_RHS);
  tmpv = nullptr;
  wDinv = nullptr;

  // fourth row: no condensation. Terms already added in (1)

  // fifth row
  tmp = nullptr;
  tmp = Core::LinAlg::matrix_multiply(
      m_LinDissContactLM_thermoRow, false, *dInvA, false, false, false, true);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*tmp, false, *kss_a, false, false, false, true), false,
      -tsi_alpha_ / (1. - alphaf_), kts_new, 1.);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*tmp, false, *kst_a, false, false, false, true), false,
      -tsi_alpha_ / (1. - alphaf_), ktt_new, 1.);
  tmpv = std::make_shared<Core::LinAlg::Vector<double>>(*thermo_m_dofs);
  tmp->multiply(false, *rsa, *tmpv);
  tmpv->scale(-tsi_alpha_ / (1. - alphaf_));
  CONTACT::Utils::add_vector(*tmpv, *combined_RHS);
  tmpv = nullptr;

  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(dInvMaThermo, true, *kts_a, false, false, false, true), false,
      1., kts_new, 1.);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(dInvMaThermo, true, *ktt_a, false, false, false, true), false,
      1., ktt_new, 1.);
  tmpv = std::make_shared<Core::LinAlg::Vector<double>>(*thermo_m_dofs);
  dInvMaThermo.multiply(true, *rta, *tmpv);
  CONTACT::Utils::add_vector(*tmpv, *combined_RHS);
  tmp = nullptr;

  // sixth row
  std::shared_ptr<Core::LinAlg::SparseMatrix> yDinv =
      Core::LinAlg::matrix_multiply(dcTdLMc_thermo, false, *dInvA, false, false, false, true);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*yDinv, false, *kss_a, false, false, false, true), false,
      -1. / (1. - alphaf_), kts_new, 1.);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*yDinv, false, *kst_a, false, false, false, true), false,
      -1. / (1. - alphaf_), ktt_new, 1.);
  tmpv = std::make_shared<Core::LinAlg::Vector<double>>(*thermo_act_dofs);
  yDinv->multiply(false, *rsa, *tmpv);
  tmpv->scale(-1. / (1. - alphaf_));
  CONTACT::Utils::add_vector(*tmpv, *combined_RHS);
  tmpv = nullptr;

  std::shared_ptr<Core::LinAlg::SparseMatrix> gDinv =
      Core::LinAlg::matrix_multiply(dcTdLMt_thermo, false, *dInvaThermo, false, false, false, true);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*gDinv, false, *kts_a, false, false, false, true), false,
      -1. / (tsi_alpha_), kts_new, 1.);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*gDinv, false, *ktt_a, false, false, false, true), false,
      -1. / (tsi_alpha_), ktt_new, 1.);
  tmpv = std::make_shared<Core::LinAlg::Vector<double>>(*thermo_act_dofs);
  gDinv->multiply(false, *rta, *tmpv);
  tmpv->scale(-1. / tsi_alpha_);
  CONTACT::Utils::add_vector(*tmpv, *combined_RHS);

  // and were done with the system matrix
  sysmat->complete();

  // we need to return the rhs, not the residual
  combined_RHS->scale(-1.);

  return;
}


void CONTACT::Utils::add_vector(
    Core::LinAlg::Vector<double>& src, Core::LinAlg::Vector<double>& dst)
{
  // return if src has no elements
  if (src.global_length() == 0) return;

#ifdef FOUR_C_ENABLE_ASSERTIONS
  for (int i = 0; i < src.get_map().num_my_elements(); ++i)
    if ((dst.get_map().lid(src.get_map().gid(i))) < 0)
      FOUR_C_THROW("src is not a vector on a sub-map of dst");
#endif

  Core::LinAlg::Vector<double> tmp = Core::LinAlg::Vector<double>(dst.get_map(), true);
  Core::LinAlg::export_to(src, tmp);
  dst.update(1., tmp, 1.);
}

void CONTACT::LagrangeStrategyTsi::recover_coupled(
    std::shared_ptr<Core::LinAlg::Vector<double>> sinc,
    std::shared_ptr<Core::LinAlg::Vector<double>> tinc,
    std::shared_ptr<Coupling::Adapter::Coupling> coupST)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> z_old = nullptr;
  if (z_ != nullptr) z_old = std::make_shared<Core::LinAlg::Vector<double>>(*z_);
  std::shared_ptr<Core::LinAlg::Vector<double>> z_thermo_old = nullptr;
  if (z_thermo_ != nullptr)
    z_thermo_old = std::make_shared<Core::LinAlg::Vector<double>>(*z_thermo_);

  // recover contact LM
  if (gactivedofs_->num_global_elements() > 0)
  {
    // do we have everything we need?
    if (rs_a_ == nullptr || kss_a_ == nullptr || kst_a_ == nullptr || dinvA_ == nullptr)
      FOUR_C_THROW("some data for LM recovery is missing");

    Core::LinAlg::Vector<double> lmc_a_new(*gactivedofs_, false);
    Core::LinAlg::Vector<double> tmp(*gactivedofs_, false);
    lmc_a_new.update(1., *rs_a_, 0.);
    kss_a_->multiply(false, *sinc, tmp);
    lmc_a_new.update(1., tmp, 1.);
    kst_a_->multiply(false, *tinc, tmp);
    lmc_a_new.update(1., tmp, 1.);
    dinvA_->multiply(false, lmc_a_new, tmp);
    tmp.scale(-1. / (1. - alphaf_));
    z_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
    Core::LinAlg::export_to(tmp, *z_);

    // recover thermo LM
    // do we have everything we need?
    if (rt_a_ == nullptr || kts_a_ == nullptr || ktt_a_ == nullptr || dinvAthr_ == nullptr)
      FOUR_C_THROW("some data for LM recovery is missing");

    Core::LinAlg::Vector<double> lmt_a_new(*thermo_act_dofs_, false);
    Core::LinAlg::Vector<double> tmp2(*thermo_act_dofs_, false);
    lmt_a_new.update(1., *rt_a_, 0.);
    kts_a_->multiply(false, *sinc, tmp2);
    lmt_a_new.update(1., tmp2, 1.);
    ktt_a_->multiply(false, *tinc, tmp2);
    lmt_a_new.update(1., tmp2, 1.);
    dinvAthr_->multiply(false, lmt_a_new, tmp2);
    tmp2.scale(-1. / (tsi_alpha_));
    z_thermo_ = std::make_shared<Core::LinAlg::Vector<double>>(*thermo_s_dofs_);
    Core::LinAlg::export_to(tmp2, *z_thermo_);
  }

  else
  {
    z_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
    z_thermo_ = std::make_shared<Core::LinAlg::Vector<double>>(*thermo_s_dofs_);
  }

  if (z_old != nullptr)
  {
    z_old->update(-1., *z_, 1.);
    z_old->norm_2(&mech_contact_incr_);
  }
  if (z_thermo_old != nullptr)
  {
    z_thermo_old->update(-1., *z_thermo_, 1.);
    z_thermo_old->norm_2(&thermo_contact_incr_);
  }

  // store updated LM into nodes
  // Note: this does not store coupST
  store_nodal_quantities(Mortar::StrategyBase::lmupdate, *coupST);
  store_nodal_quantities(Mortar::StrategyBase::lmThermo, *coupST);

  return;
};

void CONTACT::LagrangeStrategyTsi::store_nodal_quantities(
    Mortar::StrategyBase::QuantityType type, Coupling::Adapter::Coupling& coupST)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> vectorglobal = nullptr;
  // start type switch
  switch (type)
  {
    case Mortar::StrategyBase::lmThermo:
    {
      Core::LinAlg::Vector<double> tmp(*coupST.slave_dof_map());

      Core::LinAlg::export_to(*z_thermo_, tmp);
      vectorglobal = z_thermo_;
      vectorglobal = coupST.slave_to_master(tmp);
      std::shared_ptr<const Core::LinAlg::Map> sdofmap, snodemap;
      // loop over all interfaces
      for (int i = 0; i < (int)interface_.size(); ++i)
      {
        sdofmap = interface_[i]->slave_col_dofs();
        snodemap = interface_[i]->slave_col_nodes();
        std::shared_ptr<Core::LinAlg::Vector<double>> vectorinterface = nullptr;
        vectorinterface = std::make_shared<Core::LinAlg::Vector<double>>(*sdofmap);
        if (vectorglobal != nullptr) Core::LinAlg::export_to(*vectorglobal, *vectorinterface);

        // loop over all slave nodes (column or row) on the current interface
        for (int j = 0; j < snodemap->num_my_elements(); ++j)
        {
          int gid = snodemap->gid(j);
          Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
          if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
          Node* cnode = dynamic_cast<Node*>(node);

          cnode->tsi_data().thermo_lm() =
              (*vectorinterface)
                  .local_values_as_span()[(vectorinterface->get_map()).lid(cnode->dofs()[0])];
        }
      }
      break;
    }
    default:
      CONTACT::AbstractStrategy::store_nodal_quantities(type);
      break;
  }
}

void CONTACT::LagrangeStrategyTsi::update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  if (fscn_ == nullptr) fscn_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsmdofrowmap_);
  fscn_->put_scalar(0.0);

  if (ftcnp_ == nullptr)
    ftcnp_ = std::make_shared<Core::LinAlg::Vector<double>>(
        *coupST_->master_to_slave_map(*gsmdofrowmap_));
  ftcnp_->put_scalar(0.0);

  std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
      std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  dmatrix_->multiply(false, *z_, *tmp);
  CONTACT::Utils::add_vector(*tmp, *fscn_);

  tmp = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_);
  mmatrix_->multiply(true, *z_, *tmp);
  tmp->scale(-1.);
  CONTACT::Utils::add_vector(*tmp, *fscn_);

  CONTACT::AbstractStrategy::update(dis);

  Core::LinAlg::SparseMatrix dThermo(*coupST_->master_to_slave_map(*gsdofrowmap_), 100, true, false,
      Core::LinAlg::SparseMatrix::FE_MATRIX);
  Coupling::Adapter::MatrixRowColTransform()(*dmatrix_, 1.,
      Coupling::Adapter::CouplingMasterConverter(*coupST_),
      Coupling::Adapter::CouplingMasterConverter(*coupST_), dThermo, false, false);
  dThermo.complete();
  tmp =
      std::make_shared<Core::LinAlg::Vector<double>>(*coupST_->master_to_slave_map(*gsdofrowmap_));
  if (dThermo.Apply(*z_thermo_, *tmp) != 0) FOUR_C_THROW("apply went wrong");
  CONTACT::Utils::add_vector(*tmp, *ftcnp_);

  Core::LinAlg::SparseMatrix mThermo(*coupST_->master_to_slave_map(*gsdofrowmap_), 100, true, false,
      Core::LinAlg::SparseMatrix::FE_MATRIX);
  Coupling::Adapter::MatrixRowColTransform()(*mmatrix_, 1.,
      Coupling::Adapter::CouplingMasterConverter(*coupST_),
      Coupling::Adapter::CouplingMasterConverter(*coupST_), mThermo, false, false);
  mThermo.complete(
      *coupST_->master_to_slave_map(*gmdofrowmap_), *coupST_->master_to_slave_map(*gsdofrowmap_));
  mThermo.UseTranspose();
  tmp =
      std::make_shared<Core::LinAlg::Vector<double>>(*coupST_->master_to_slave_map(*gmdofrowmap_));
  mThermo.multiply(true, *z_thermo_, *tmp);
  tmp->scale(-1.);
  CONTACT::Utils::add_vector(*tmp, *ftcnp_);

  Core::LinAlg::SparseMatrix m_LinDissContactLM(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  for (unsigned i = 0; i < interface_.size(); ++i)
    dynamic_cast<CONTACT::TSIInterface*>(&(*interface_[i]))
        ->assemble_dm_lin_diss(nullptr, nullptr, nullptr, &m_LinDissContactLM, 1.);
  m_LinDissContactLM.complete(*gactivedofs_, *gmdofrowmap_);
  Core::LinAlg::Vector<double> z_act(*gactivedofs_);
  Core::LinAlg::export_to(*z_, z_act);
  tmp = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_);
  m_LinDissContactLM.multiply(false, z_act, *tmp);
  Core::LinAlg::Vector<double> tmp2(*coupST_->master_dof_map());
  Core::LinAlg::export_to(*tmp, tmp2);
  std::shared_ptr<Core::LinAlg::Vector<double>> tmp3 = coupST_->master_to_slave(tmp2);
  Core::LinAlg::Vector<double> tmp4(*coupST_->master_to_slave_map(*gmdofrowmap_));
  Core::LinAlg::export_to(*tmp3, tmp4);
  CONTACT::Utils::add_vector(tmp4, *ftcnp_);

  ftcn_ = ftcnp_;
}

void CONTACT::LagrangeStrategyTsi::set_alphaf_thermo(const Teuchos::ParameterList& tdyn)
{
  auto dyn_type = Teuchos::getIntegralValue<Thermo::DynamicType>(tdyn, "DYNAMICTYPE");
  switch (dyn_type)
  {
    case Thermo::DynamicType::GenAlpha:
      tsi_alpha_ = tdyn.sublist("GENALPHA").get<double>("ALPHA_F");
      break;
    case Thermo::DynamicType::OneStepTheta:
      tsi_alpha_ = tdyn.sublist("ONESTEPTHETA").get<double>("THETA");
      break;
    case Thermo::DynamicType::Statics:
      tsi_alpha_ = 1.;
      break;
    default:
      FOUR_C_THROW("unknown thermal time integration type");
  }
  return;
}


/*----------------------------------------------------------------------*
 |  write restart information for contact                     popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyTsi::do_write_restart(
    std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>>& restart_vectors,
    bool forcedrestart) const
{
  CONTACT::AbstractStrategy::do_write_restart(restart_vectors, forcedrestart);

  if (fscn_ != nullptr)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
        std::make_shared<Core::LinAlg::Vector<double>>(*gsmdofrowmap_);
    Core::LinAlg::export_to(*fscn_, *tmp);
    restart_vectors["last_contact_force"] = tmp;
  }
  if (ftcn_ != nullptr)
  {
    Core::LinAlg::Vector<double> tmp(*coupST_->slave_dof_map());
    Core::LinAlg::export_to(*ftcn_, tmp);
    restart_vectors["last_thermo_force"] = coupST_->slave_to_master(tmp);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyTsi::do_read_restart(Core::IO::DiscretizationReader& reader,
    std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<CONTACT::ParamsInterface> cparams_ptr)
{
  const bool restartwithcontact = params().get<bool>("RESTART_WITH_CONTACT");

  CONTACT::AbstractStrategy::do_read_restart(reader, dis);
  fscn_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsmdofrowmap_);
  if (!restartwithcontact) reader.read_vector(fscn_, "last_contact_force");

  std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
      std::make_shared<Core::LinAlg::Vector<double>>(*coupST_->master_dof_map());
  if (!restartwithcontact) reader.read_vector(tmp, "last_thermo_force");
  ftcn_ = coupST_->master_to_slave(*tmp);
  tmp =
      std::make_shared<Core::LinAlg::Vector<double>>(*coupST_->master_to_slave_map(*gsmdofrowmap_));
  Core::LinAlg::export_to(*ftcn_, *tmp);
  ftcn_ = tmp;
}

FOUR_C_NAMESPACE_CLOSE
