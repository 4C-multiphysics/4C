// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nitsche_strategy_poro_scatra.hpp"

#include "4C_contact_interface.hpp"
#include "4C_contact_nitsche_utils.hpp"
#include "4C_contact_paramsinterface.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_so3_plast_ssn.hpp"

#include <Epetra_FEVector.h>
#include <Epetra_Operator.h>

FOUR_C_NAMESPACE_OPEN

CONTACT::NitscheStrategyPoroScatra::NitscheStrategyPoroScatra(const Epetra_Map* dof_row_map,
    const Epetra_Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<Teuchos::RCP<CONTACT::Interface>> interface, int dim,
    Teuchos::RCP<Epetra_Comm> comm, double alphaf, int maxdof)
    : CONTACT::NitscheStrategySsi(
          dof_row_map, NodeRowMap, params, std::move(interface), dim, comm, alphaf, maxdof),
      no_penetration_(
          Global::Problem::instance()->poroelast_dynamic_params().get<bool>("CONTACTNOPEN"))
{
}

CONTACT::NitscheStrategyPoroScatra::NitscheStrategyPoroScatra(
    const Teuchos::RCP<CONTACT::AbstractStratDataContainer>& data_ptr,
    const Epetra_Map* dof_row_map, const Epetra_Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<Teuchos::RCP<CONTACT::Interface>> interface, int dim,
    Teuchos::RCP<const Epetra_Comm> comm, double alphaf, int maxdof)
    : CONTACT::NitscheStrategySsi(data_ptr, dof_row_map, NodeRowMap, params, std::move(interface),
          dim, comm, alphaf, maxdof),
      no_penetration_(
          Global::Problem::instance()->poroelast_dynamic_params().get<bool>("CONTACTNOPEN"))
{
}

void CONTACT::NitscheStrategyPoroScatra::apply_force_stiff_cmt(
    Teuchos::RCP<Core::LinAlg::Vector<double>> dis, Teuchos::RCP<Core::LinAlg::SparseOperator>& kt,
    Teuchos::RCP<Core::LinAlg::Vector<double>>& f, const int step, const int iter, bool predictor)
{
  if (predictor) return;
  if (kt != Teuchos::null && f != Teuchos::null)
  {
    CONTACT::NitscheStrategy::apply_force_stiff_cmt(dis, kt, f, step, iter, predictor);
  }
  curr_state_eval_ = true;
}

void CONTACT::NitscheStrategyPoroScatra::evaluate_reference_state()
{
  CONTACT::NitscheStrategy::evaluate_reference_state();
}


void CONTACT::NitscheStrategyPoroScatra::integrate(const CONTACT::ParamsInterface& cparams)
{
  CONTACT::NitscheStrategy::integrate(cparams);

  fs_ = create_rhs_block_ptr(CONTACT::VecBlockType::scatra);
  kss_ = create_matrix_block_ptr(CONTACT::MatBlockType::scatra_scatra);
  ksd_ = create_matrix_block_ptr(CONTACT::MatBlockType::scatra_displ);
  kds_ = create_matrix_block_ptr(CONTACT::MatBlockType::displ_scatra);

  fp_ = create_rhs_block_ptr(CONTACT::VecBlockType::porofluid);
  kpp_ = create_matrix_block_ptr(CONTACT::MatBlockType::porofluid_porofluid);
  kpd_ = create_matrix_block_ptr(CONTACT::MatBlockType::porofluid_displ);
  kdp_ = create_matrix_block_ptr(CONTACT::MatBlockType::displ_porofluid);
}

void CONTACT::NitscheStrategyPoroScatra::set_state(
    const enum Mortar::StateType& statename, const Core::LinAlg::Vector<double>& vec)
{
  switch (statename)
  {
    case Mortar::state_elch:
    case Mortar::state_scalar:
    {
      auto scatra_dis = Global::Problem::instance()->get_dis("scatra");
      if (scatra_dis == Teuchos::null) FOUR_C_THROW("didn't get scatra discretization");
      set_parent_state(statename, vec, *scatra_dis);
      break;
    }
    case Mortar::state_svelocity:
    {
      Teuchos::RCP<Core::FE::Discretization> dis =
          Global::Problem::instance()->get_dis("structure");
      if (dis == Teuchos::null) FOUR_C_THROW("didn't get structure discretization");
      set_parent_state(statename, vec, *dis);
      break;
    }
    case Mortar::state_fvelocity:
    case Mortar::state_fpressure:
    {
      Teuchos::RCP<Core::FE::Discretization> dis =
          Global::Problem::instance()->get_dis("porofluid");
      if (dis == Teuchos::null) FOUR_C_THROW("didn't get my discretization");
      set_parent_state(statename, vec, *dis);
      break;
    }
    default:
    {
      CONTACT::NitscheStrategy::set_state(statename, vec);
      break;
    }
  }
}

void CONTACT::NitscheStrategyPoroScatra::set_parent_state(const enum Mortar::StateType& statename,
    const Core::LinAlg::Vector<double>& vec, const Core::FE::Discretization& dis)
{
  //
  if (statename == Mortar::state_fvelocity || statename == Mortar::state_fpressure)
  {
    Teuchos::RCP<Core::LinAlg::Vector<double>> global =
        Teuchos::rcp(new Core::LinAlg::Vector<double>(*dis.dof_col_map(), true));
    Core::LinAlg::export_to(vec, *global);

    // set state on interfaces
    for (const auto& interface : interface_)
    {
      Core::FE::Discretization& idiscret = interface->discret();

      for (int j = 0; j < interface->discret().element_col_map()->NumMyElements(); ++j)
      {
        const int gid = interface->discret().element_col_map()->GID(j);

        auto* ele = dynamic_cast<Mortar::Element*>(idiscret.g_element(gid));

        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;

        if (ele->parent_slave_element())  // if this pointer is nullptr, this parent is impermeable
        {
          // this gets values in local order
          ele->parent_slave_element()->location_vector(dis, lm, lmowner, lmstride);

          std::vector<double> myval;
          Core::FE::extract_my_values(*global, myval, lm);

          std::vector<double> vel;
          std::vector<double> pres;

          for (int n = 0; n < ele->parent_slave_element()->num_node(); ++n)
          {
            for (unsigned dim = 0; dim < 3; ++dim)
            {
              vel.push_back(myval[n * 4 + dim]);
            }
            pres.push_back(myval[n * 4 + 3]);
          }

          ele->mo_data().parent_pf_pres() = pres;
          ele->mo_data().parent_pf_vel() = vel;
          ele->mo_data().parent_pf_dof() = lm;
        }
      }
    }
  }
  else if (statename == Mortar::state_elch || statename == Mortar::state_scalar)
  {
    auto scatra_dofcolmap =
        Teuchos::rcp(new Core::LinAlg::Vector<double>(*dis.dof_col_map(), true));
    Core::LinAlg::export_to(vec, *scatra_dofcolmap);

    // set state on interfaces
    for (const auto& interface : interface_)
    {
      // get the interface discretization
      const auto& interface_dis = interface->discret();

      // loop over all interface column elements owned by current processor
      for (int j = 0; j < interface_dis.element_col_map()->NumMyElements(); ++j)
      {
        const int gid = interface_dis.element_col_map()->GID(j);

        auto* interface_ele = interface_dis.g_element(gid);
        if (interface_ele == nullptr) FOUR_C_THROW("Did not find element.");

        auto* mortar_ele = dynamic_cast<Mortar::Element*>(interface_ele);
        auto* mortar_parent_ele = dis.g_element(mortar_ele->parent_element_id());

        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;

        if (mortar_parent_ele == nullptr)
          FOUR_C_THROW("Did not get parent element to extract scalar values");
        else
          mortar_parent_ele->location_vector(dis, lm, lmowner, lmstride);

        std::vector<double> myval;
        Core::FE::extract_my_values(*scatra_dofcolmap, myval, lm);

        mortar_ele->mo_data().parent_scalar() = myval;
        mortar_ele->mo_data().parent_scalar_dof() = lm;
      }
    }
  }
  else
    CONTACT::NitscheStrategy::set_parent_state(statename, vec, dis);
}

Teuchos::RCP<Epetra_FEVector> CONTACT::NitscheStrategyPoroScatra::setup_rhs_block_vec(
    const enum CONTACT::VecBlockType& bt) const
{
  switch (bt)
  {
    case CONTACT::VecBlockType::porofluid:
      return Teuchos::rcp(
          new Epetra_FEVector(*Global::Problem::instance()->get_dis("porofluid")->dof_row_map()));
    case CONTACT::VecBlockType::elch:
    case CONTACT::VecBlockType::scatra:
      return Teuchos::rcp(
          new Epetra_FEVector(*Global::Problem::instance()->get_dis("scatra")->dof_row_map()));
    default:
      return CONTACT::NitscheStrategy::setup_rhs_block_vec(bt);
  }
}

Teuchos::RCP<const Core::LinAlg::Vector<double>>
CONTACT::NitscheStrategyPoroScatra::get_rhs_block_ptr(const enum CONTACT::VecBlockType& bp) const
{
  if (bp == CONTACT::VecBlockType::constraint) return Teuchos::null;

  if (!curr_state_eval_)
    FOUR_C_THROW("you didn't evaluate this contact state for %s first",
        CONTACT::vec_block_type_to_str(bp).c_str());

  switch (bp)
  {
    case CONTACT::VecBlockType::porofluid:
      return Teuchos::make_rcp<Core::LinAlg::Vector<double>>(*(*fp_)(0));
    case CONTACT::VecBlockType::elch:
    case CONTACT::VecBlockType::scatra:
      return Teuchos::make_rcp<Core::LinAlg::Vector<double>>(*(*fs_)(0));
    default:
      return CONTACT::NitscheStrategy::get_rhs_block_ptr(bp);
  }
}

Teuchos::RCP<Core::LinAlg::SparseMatrix> CONTACT::NitscheStrategyPoroScatra::setup_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt)
{
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_porofluid:
      return Teuchos::rcp(new Core::LinAlg::SparseMatrix(
          *Teuchos::rcpFromRef<const Epetra_Map>(
              *Global::Problem::instance()->get_dis("structure")->dof_row_map()),
          100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));
    case CONTACT::MatBlockType::porofluid_displ:
    case CONTACT::MatBlockType::porofluid_porofluid:
      return Teuchos::rcp(new Core::LinAlg::SparseMatrix(
          *Teuchos::rcpFromRef<const Epetra_Map>(
              *Global::Problem::instance()->get_dis("porofluid")->dof_row_map()),
          100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));
    case CONTACT::MatBlockType::displ_elch:
    case CONTACT::MatBlockType::displ_scatra:
      return Teuchos::rcp(new Core::LinAlg::SparseMatrix(
          *Teuchos::rcpFromRef<const Epetra_Map>(
              *Global::Problem::instance()->get_dis("structure")->dof_row_map()),
          100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));
    case CONTACT::MatBlockType::elch_displ:
    case CONTACT::MatBlockType::elch_elch:
    case CONTACT::MatBlockType::scatra_displ:
    case CONTACT::MatBlockType::scatra_scatra:
      return Teuchos::rcp(new Core::LinAlg::SparseMatrix(
          *Teuchos::rcpFromRef<const Epetra_Map>(
              *Global::Problem::instance()->get_dis("scatra")->dof_row_map()),
          100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));
    default:
      return CONTACT::NitscheStrategy::setup_matrix_block_ptr(bt);
  }
}

void CONTACT::NitscheStrategyPoroScatra::complete_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt, Teuchos::RCP<Core::LinAlg::SparseMatrix> kc)
{
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_porofluid:
      if (dynamic_cast<Epetra_FECrsMatrix&>(*kc->epetra_matrix())
              .GlobalAssemble(
                  *Global::Problem::instance()->get_dis("porofluid")->dof_row_map(),  // col map
                  *Global::Problem::instance()->get_dis("structure")->dof_row_map(),  // row map
                  true, Add))
        FOUR_C_THROW("GlobalAssemble(...) failed");
      break;
    case CONTACT::MatBlockType::porofluid_displ:
      if (dynamic_cast<Epetra_FECrsMatrix&>(*kc->epetra_matrix())
              .GlobalAssemble(
                  *Global::Problem::instance()->get_dis("structure")->dof_row_map(),  // col map
                  *Global::Problem::instance()->get_dis("porofluid")->dof_row_map(),  // row map
                  true, Add))
        FOUR_C_THROW("GlobalAssemble(...) failed");
      break;
    case CONTACT::MatBlockType::porofluid_porofluid:
      if (dynamic_cast<Epetra_FECrsMatrix&>(*kc->epetra_matrix()).GlobalAssemble(true, Add))
        FOUR_C_THROW("GlobalAssemble(...) failed");
      break;
    case CONTACT::MatBlockType::displ_elch:
    case CONTACT::MatBlockType::displ_scatra:
      if (dynamic_cast<Epetra_FECrsMatrix&>(*kc->epetra_matrix())
              .GlobalAssemble(
                  *Global::Problem::instance()->get_dis("scatra")->dof_row_map(),     // col map
                  *Global::Problem::instance()->get_dis("structure")->dof_row_map(),  // row map
                  true, Add))
        FOUR_C_THROW("GlobalAssemble(...) failed");
      break;
    case CONTACT::MatBlockType::elch_displ:
    case CONTACT::MatBlockType::scatra_displ:
      if (dynamic_cast<Epetra_FECrsMatrix&>(*kc->epetra_matrix())
              .GlobalAssemble(
                  *Global::Problem::instance()->get_dis("structure")->dof_row_map(),  // col map
                  *Global::Problem::instance()->get_dis("scatra")->dof_row_map(),     // row map
                  true, Add))
        FOUR_C_THROW("GlobalAssemble(...) failed");
      break;
    case CONTACT::MatBlockType::elch_elch:
    case CONTACT::MatBlockType::scatra_scatra:
      if (dynamic_cast<Epetra_FECrsMatrix&>(*kc->epetra_matrix()).GlobalAssemble(true, Add))
        FOUR_C_THROW("GlobalAssemble(...) failed");
      break;
    default:
      CONTACT::NitscheStrategy::complete_matrix_block_ptr(bt, kc);
      break;
  }
}

Teuchos::RCP<Core::LinAlg::SparseMatrix> CONTACT::NitscheStrategyPoroScatra::get_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bp) const
{
  if (!curr_state_eval_) FOUR_C_THROW("you didn't evaluate this contact state first");

  switch (bp)
  {
    case CONTACT::MatBlockType::porofluid_porofluid:
      return kpp_;
    case CONTACT::MatBlockType::porofluid_displ:
      return kpd_;
    case CONTACT::MatBlockType::displ_porofluid:
      return kdp_;
    case CONTACT::MatBlockType::elch_elch:
    case CONTACT::MatBlockType::scatra_scatra:
      return kss_;
    case CONTACT::MatBlockType::elch_displ:
    case CONTACT::MatBlockType::scatra_displ:
      return ksd_;
    case CONTACT::MatBlockType::displ_elch:
    case CONTACT::MatBlockType::displ_scatra:
      return kds_;
    default:
      return CONTACT::NitscheStrategy::get_matrix_block_ptr(bp, nullptr);
  }
}

FOUR_C_NAMESPACE_CLOSE
