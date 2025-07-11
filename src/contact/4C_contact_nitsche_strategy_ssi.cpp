// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nitsche_strategy_ssi.hpp"

#include "4C_contact_interface.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_element.hpp"

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------*
/-------------------------------------------------------------------------*/
void CONTACT::NitscheStrategySsi::integrate(const CONTACT::ParamsInterface& cparams)
{
  CONTACT::NitscheStrategy::integrate(cparams);

  fs_ = create_rhs_block_ptr(CONTACT::VecBlockType::scatra);
  kss_ = create_matrix_block_ptr(CONTACT::MatBlockType::scatra_scatra);
  ksd_ = create_matrix_block_ptr(CONTACT::MatBlockType::scatra_displ);
  kds_ = create_matrix_block_ptr(CONTACT::MatBlockType::displ_scatra);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::NitscheStrategySsi::evaluate_reference_state()
{
  auto phinp = Global::Problem::instance()->get_dis("structure")->get_state(1, "scalarfield");
  const auto* scatra_dofrowmap = Global::Problem::instance()->get_dis("scatra")->dof_row_map();
  Core::LinAlg::Vector<double> phinp_dofrowmap(*scatra_dofrowmap, true);
  Core::LinAlg::export_to(*phinp, phinp_dofrowmap);

  set_state(Mortar::state_scalar, phinp_dofrowmap);

  // call base class
  CONTACT::NitscheStrategy::evaluate_reference_state();
}

/*------------------------------------------------------------------------*
/-------------------------------------------------------------------------*/
void CONTACT::NitscheStrategySsi::set_state(
    const enum Mortar::StateType& statename, const Core::LinAlg::Vector<double>& vec)
{
  switch (statename)
  {
    case Mortar::state_elch:
    case Mortar::state_scalar:
    {
      double inf_delta = 0.0;
      if (curr_state_scalar_ == nullptr)
      {
        curr_state_scalar_ = std::make_shared<Core::LinAlg::Vector<double>>(vec);
        inf_delta = 1.0e12;
      }
      else
      {
        auto delta(vec);
        delta.update(-1.0, *curr_state_scalar_, 1.0);
        delta.norm_inf(&inf_delta);
      }
      if (inf_delta < 1.0e-16)
        return;
      else
      {
        curr_state_eval_ = false;
        *curr_state_scalar_ = vec;
        auto scatra_dis = Global::Problem::instance()->get_dis("scatra");
        if (scatra_dis == nullptr) FOUR_C_THROW("didn't get scatra discretization");
        set_parent_state(statename, vec, *scatra_dis);
      }
      break;
    }
    default:
    {
      CONTACT::NitscheStrategy::set_state(statename, vec);
      break;
    }
  }
}

/*------------------------------------------------------------------------*
/-------------------------------------------------------------------------*/
void CONTACT::NitscheStrategySsi::set_parent_state(const enum Mortar::StateType& statename,
    const Core::LinAlg::Vector<double>& vec, const Core::FE::Discretization& dis)
{
  switch (statename)
  {
    case Mortar::state_elch:
    case Mortar::state_scalar:
    {
      Core::LinAlg::Vector<double> scatra_dofcolmap(*dis.dof_col_map(), true);
      Core::LinAlg::export_to(vec, scatra_dofcolmap);

      // set state on interfaces
      for (const auto& interface : interface_)
      {
        // get the interface discretization
        const auto& interface_dis = interface->discret();

        // loop over all interface column elements owned by current processor
        for (int j = 0; j < interface_dis.element_col_map()->num_my_elements(); ++j)
        {
          const int gid = interface_dis.element_col_map()->gid(j);

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

          std::vector<double> myval = Core::FE::extract_values(scatra_dofcolmap, lm);

          mortar_ele->mo_data().parent_scalar() = myval;
          mortar_ele->mo_data().parent_scalar_dof() = lm;
        }
      }
      break;
    }
    default:
    {
      CONTACT::NitscheStrategy::set_parent_state(statename, vec, dis);
      break;
    }
  }
}

/*------------------------------------------------------------------------*
/-------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::FEVector<double>> CONTACT::NitscheStrategySsi::setup_rhs_block_vec(
    const enum CONTACT::VecBlockType& bt) const
{
  switch (bt)
  {
    case CONTACT::VecBlockType::elch:
    case CONTACT::VecBlockType::scatra:
      return std::make_shared<Core::LinAlg::FEVector<double>>(
          Global::Problem::instance()->get_dis("scatra")->dof_row_map()->get_epetra_block_map());
    default:
      return CONTACT::NitscheStrategy::setup_rhs_block_vec(bt);
  }
}

/*------------------------------------------------------------------------*
/-------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> CONTACT::NitscheStrategySsi::get_rhs_block_ptr(
    const enum CONTACT::VecBlockType& bp) const
{
  if (bp == CONTACT::VecBlockType::constraint) return nullptr;

  if (!curr_state_eval_)
    FOUR_C_THROW(
        "you didn't evaluate this contact state for {} first", CONTACT::vec_block_type_to_str(bp));

  switch (bp)
  {
    case CONTACT::VecBlockType::elch:
    case CONTACT::VecBlockType::scatra:
      return std::make_shared<Core::LinAlg::Vector<double>>(
          *(fs_->get_ref_of_epetra_fevector())(0));
    default:
      return CONTACT::NitscheStrategy::get_rhs_block_ptr(bp);
  }
}

/*------------------------------------------------------------------------*
/-------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> CONTACT::NitscheStrategySsi::setup_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt)
{
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_elch:
    case CONTACT::MatBlockType::displ_scatra:
      return std::make_shared<Core::LinAlg::SparseMatrix>(
          *Global::Problem::instance()->get_dis("structure")->dof_row_map(), 100, true, false,
          Core::LinAlg::SparseMatrix::FE_MATRIX);
    case CONTACT::MatBlockType::elch_displ:
    case CONTACT::MatBlockType::elch_elch:
    case CONTACT::MatBlockType::scatra_displ:
    case CONTACT::MatBlockType::scatra_scatra:
      return std::make_shared<Core::LinAlg::SparseMatrix>(
          *Global::Problem::instance()->get_dis("scatra")->dof_row_map(), 100, true, false,
          Core::LinAlg::SparseMatrix::FE_MATRIX);
    default:
      return CONTACT::NitscheStrategy::setup_matrix_block_ptr(bt);
  }
}

/*------------------------------------------------------------------------*
/-------------------------------------------------------------------------*/
void CONTACT::NitscheStrategySsi::complete_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt, std::shared_ptr<Core::LinAlg::SparseMatrix> kc)
{
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_elch:
    case CONTACT::MatBlockType::displ_scatra:
      if (dynamic_cast<Epetra_FECrsMatrix&>(*kc->epetra_matrix())
              .GlobalAssemble(Global::Problem::instance()
                                  ->get_dis("scatra")
                                  ->dof_row_map()
                                  ->get_epetra_map(),  // col map
                  Global::Problem::instance()
                      ->get_dis("structure")
                      ->dof_row_map()
                      ->get_epetra_map(),  // row map
                  true, Add))
        FOUR_C_THROW("GlobalAssemble(...) failed");
      break;
    case CONTACT::MatBlockType::elch_displ:
    case CONTACT::MatBlockType::scatra_displ:
      if (dynamic_cast<Epetra_FECrsMatrix&>(*kc->epetra_matrix())
              .GlobalAssemble(Global::Problem::instance()
                                  ->get_dis("structure")
                                  ->dof_row_map()
                                  ->get_epetra_map(),  // col map
                  Global::Problem::instance()
                      ->get_dis("scatra")
                      ->dof_row_map()
                      ->get_epetra_map(),  // row map
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

/*------------------------------------------------------------------------*
/-------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> CONTACT::NitscheStrategySsi::get_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bp) const
{
  if (!curr_state_eval_) FOUR_C_THROW("you didn't evaluate this contact state first");

  switch (bp)
  {
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
