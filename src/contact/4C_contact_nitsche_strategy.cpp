// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nitsche_strategy.hpp"

#include "4C_contact_interface.hpp"
#include "4C_contact_nitsche_utils.hpp"
#include "4C_contact_paramsinterface.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | global evaluation method called from time integrator     seitz 10/16 |
 *----------------------------------------------------------------------*/
void CONTACT::NitscheStrategy::apply_force_stiff_cmt(
    std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
    std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
    bool predictor)
{
  // mortar initialization and evaluation
  set_state(Mortar::state_new_displacement, *dis);

  // just a Nitsche-version
  std::shared_ptr<Core::LinAlg::FEVector<double>> fc =
      std::make_shared<Core::LinAlg::FEVector<double>>(f->get_map().get_epetra_block_map());
  std::shared_ptr<Core::LinAlg::SparseMatrix> kc = std::make_shared<Core::LinAlg::SparseMatrix>(
      Core::LinAlg::Map(dynamic_cast<Core::LinAlg::SparseMatrix*>(&(*kt))->row_map()), 100, true,
      false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // Evaluation for all interfaces
  for (const auto& interface : interface_)
  {
    interface->initialize();
    interface->evaluate(0, step_, iter_);
    for (int e = 0; e < interface->discret().element_col_map()->num_my_elements(); ++e)
    {
      auto* mele = dynamic_cast<Mortar::Element*>(
          interface->discret().g_element(interface->discret().element_col_map()->gid(e)));
      mele->get_nitsche_container().assemble_rhs(mele, CONTACT::VecBlockType::displ, fc);
      mele->get_nitsche_container().assemble_matrix(mele, CONTACT::MatBlockType::displ_displ, kc);
    }
  }

  // now we also did this state
  curr_state_eval_ = true;

  if (fc->global_assemble() != 0) FOUR_C_THROW("GlobalAssemble failed");
  // add negative contact force here since the time integrator handed me a rhs!
  if (f->update(-1., *fc, 1.)) FOUR_C_THROW("update went wrong");
  kc->complete();
  kt->un_complete();
  kt->add(*kc, false, 1., 1.);
  kt->complete();
}


/*----------------------------------------------------------------------*
 |  read restart information for contact                     seitz 10/16|
 *----------------------------------------------------------------------*/
void CONTACT::NitscheStrategy::do_read_restart(Core::IO::DiscretizationReader& reader,
    std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<CONTACT::ParamsInterface> cparams_ptr)
{
  // check whether this is a restart with contact of a previously
  // non-contact simulation run (if yes, we have to be careful not
  // to try to read certain, in this case non-existing, vectors
  // such as the activetoggle or sliptoggle vectors, but rather
  // initialize the restart active and slip sets as being empty)
  const bool restartwithcontact = params().get<bool>("RESTART_WITH_CONTACT");
  if (restartwithcontact) FOUR_C_THROW("not supported for nitsche contact");

  // set restart displacement state
  set_state(Mortar::state_new_displacement, *dis);
  set_state(Mortar::state_old_displacement, *dis);

  // Evaluation for all interfaces
  for (const auto& interface : interface_) interface->initialize();

  if (friction_)
  {
    for (const auto& interface : interface_)
    {
      interface->evaluate_nodal_normals();
      interface->export_nodal_normals();
    }
    store_to_old(Mortar::StrategyBase::n_old);
  }

  if (params().get<bool>("NITSCHE_PENALTY_ADAPTIVE")) update_trace_ineq_etimates();
}

void CONTACT::NitscheStrategy::set_state(
    const enum Mortar::StateType& statename, const Core::LinAlg::Vector<double>& vec)
{
  if (statename == Mortar::state_new_displacement)
  {
    double inf_delta = 0.;
    if (curr_state_ == nullptr)
    {
      curr_state_ = std::make_shared<Core::LinAlg::Vector<double>>(vec);
      inf_delta = 1.e12;
    }
    else
    {
      Core::LinAlg::Vector<double> delta(vec);
      delta.update(-1., *curr_state_, 1.);
      delta.norm_inf(&inf_delta);
    }
    if (inf_delta < 1.e-12)
      return;
    else
    {
      curr_state_eval_ = false;
      (*curr_state_) = vec;
      AbstractStrategy::set_state(statename, vec);
      std::shared_ptr<Core::FE::Discretization> dis =
          Global::Problem::instance()->get_dis("structure");
      if (dis == nullptr) FOUR_C_THROW("didn't get my discretization");
      set_parent_state(statename, vec, *dis);
    }
  }
  else
  {
    curr_state_eval_ = false;
    AbstractStrategy::set_state(statename, vec);
  }
}

/*------------------------------------------------------------------------*
 |                                                             seitz 10/16|
 *------------------------------------------------------------------------*/
void CONTACT::NitscheStrategy::set_parent_state(const enum Mortar::StateType& statename,
    const Core::LinAlg::Vector<double>& vec, const Core::FE::Discretization& dis)
{
  if (statename == Mortar::state_new_displacement || statename == Mortar::state_svelocity)
  {
    Core::LinAlg::Vector<double> global(*dis.dof_col_map(), true);
    Core::LinAlg::export_to(vec, global);

    // set state on interfaces
    for (const auto& interface : interface_)
    {
      Core::FE::Discretization& idiscret = interface->discret();

      for (int j = 0; j < interface->discret().element_col_map()->num_my_elements(); ++j)
      {
        const int gid = interface->discret().element_col_map()->gid(j);

        auto* ele = dynamic_cast<Mortar::Element*>(idiscret.g_element(gid));

        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;

        // this gets values in local order
        ele->parent_element()->location_vector(dis, lm, lmowner, lmstride);

        std::vector<double> myval = Core::FE::extract_values(global, lm);

        switch (statename)
        {
          case Mortar::state_new_displacement:
          {
            ele->mo_data().parent_disp() = myval;
            ele->mo_data().parent_dof() = lm;
            break;
          }
          case Mortar::state_svelocity:
          {
            ele->mo_data().parent_vel() = myval;
            break;
          }
          default:
            FOUR_C_THROW("unknown statename");
        }
      }
    }
  }
}

void CONTACT::NitscheStrategy::evaluate_force(CONTACT::ParamsInterface& cparams)
{
  integrate(cparams);
}

void CONTACT::NitscheStrategy::evaluate_force_stiff(CONTACT::ParamsInterface& cparams)
{
  integrate(cparams);
}

void CONTACT::NitscheStrategy::reset(const CONTACT::ParamsInterface& cparams,
    const Core::LinAlg::Vector<double>& dispnp, const Core::LinAlg::Vector<double>& xnew)
{
  set_state(Mortar::state_new_displacement, dispnp);
}

void CONTACT::NitscheStrategy::run_post_compute_x(const CONTACT::ParamsInterface& cparams,
    const Core::LinAlg::Vector<double>& xold, const Core::LinAlg::Vector<double>& dir,
    const Core::LinAlg::Vector<double>& xnew)
{
  // do nothing
}

void CONTACT::NitscheStrategy::integrate(const CONTACT::ParamsInterface& cparams)
{
  // we already did this displacement state
  if (curr_state_eval_) return;

  // time measurement (on each processor)
  const double t_start = Teuchos::Time::wallTime();

  // Evaluation for all interfaces
  for (const auto& interface : interface_)
  {
    interface->interface_params().set<double>("TIMESTEP", cparams.get_delta_time());
    interface->initialize();
    interface->evaluate(0, step_, iter_);

    // store required integration time
    inttime_ += interface->inttime();
  }

  // check the parallel distribution
  check_parallel_distribution(t_start);

  // now we also did this state
  curr_state_eval_ = true;

  // ... and we can assemble the matrix and rhs
  fc_ = create_rhs_block_ptr(CONTACT::VecBlockType::displ);
  kc_ = create_matrix_block_ptr(CONTACT::MatBlockType::displ_displ);
}

std::shared_ptr<Core::LinAlg::FEVector<double>> CONTACT::NitscheStrategy::setup_rhs_block_vec(
    const enum CONTACT::VecBlockType& bt) const
{
  switch (bt)
  {
    case CONTACT::VecBlockType::displ:
      return std::make_shared<Core::LinAlg::FEVector<double>>(
          Global::Problem::instance()->get_dis("structure")->dof_row_map()->get_epetra_block_map());
    default:
      FOUR_C_THROW("you should not be here");
      break;
  }
  return nullptr;
}

std::shared_ptr<Core::LinAlg::FEVector<double>> CONTACT::NitscheStrategy::create_rhs_block_ptr(
    const enum CONTACT::VecBlockType& bt) const
{
  if (!curr_state_eval_) FOUR_C_THROW("you didn't evaluate this contact state first");

  std::shared_ptr<Core::LinAlg::FEVector<double>> fc = setup_rhs_block_vec(bt);

  for (const auto& interface : interface_)
  {
    for (int e = 0; e < interface->discret().element_col_map()->num_my_elements(); ++e)
    {
      auto* mele = dynamic_cast<Mortar::Element*>(
          interface->discret().g_element(interface->discret().element_col_map()->gid(e)));
      auto& nitsche_container = mele->get_nitsche_container();
      nitsche_container.assemble_rhs(mele, bt, fc);
    }
  }
  if (fc->global_assemble() != 0) FOUR_C_THROW("GlobalAssemble failed");

  return fc;
}

std::shared_ptr<const Core::LinAlg::Vector<double>> CONTACT::NitscheStrategy::get_rhs_block_ptr(
    const enum CONTACT::VecBlockType& bt) const
{
  if (!curr_state_eval_)
    FOUR_C_THROW(
        "you didn't evaluate this contact state for {} first", CONTACT::vec_block_type_to_str(bt));

  switch (bt)
  {
    case CONTACT::VecBlockType::displ:
      return std::make_shared<Core::LinAlg::Vector<double>>(*fc_);
    case CONTACT::VecBlockType::constraint:
      return nullptr;
    default:
      FOUR_C_THROW("get_rhs_block_ptr: your type is no treated properly!");
      break;
  }

  return nullptr;
}

std::shared_ptr<Core::LinAlg::SparseMatrix> CONTACT::NitscheStrategy::setup_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt)
{
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_displ:
      return std::make_shared<Core::LinAlg::SparseMatrix>(
          *Global::Problem::instance()->get_dis("structure")->dof_row_map(), 100, true, false,
          Core::LinAlg::SparseMatrix::FE_MATRIX);
    default:
      FOUR_C_THROW("you should not be here");
      break;
  }
  return nullptr;
}

void CONTACT::NitscheStrategy::complete_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt, std::shared_ptr<Core::LinAlg::SparseMatrix> kc)
{
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_displ:
      kc->complete();
      break;
    default:
      FOUR_C_THROW("you should not be here");
      break;
  }
}

std::shared_ptr<Core::LinAlg::SparseMatrix> CONTACT::NitscheStrategy::create_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt)
{
  if (!curr_state_eval_) FOUR_C_THROW("you didn't evaluate this contact state first");

  std::shared_ptr<Core::LinAlg::SparseMatrix> kc = setup_matrix_block_ptr(bt);

  for (const auto& interface : interface_)
  {
    for (int e = 0; e < interface->discret().element_col_map()->num_my_elements(); ++e)
    {
      auto* mele = dynamic_cast<Mortar::Element*>(
          interface->discret().g_element(interface->discret().element_col_map()->gid(e)));
      mele->get_nitsche_container().assemble_matrix(mele, bt, kc);
    }
  }

  complete_matrix_block_ptr(bt, kc);

  return kc;
}

std::shared_ptr<Core::LinAlg::SparseMatrix> CONTACT::NitscheStrategy::get_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt, const CONTACT::ParamsInterface* cparams) const
{
  if (!curr_state_eval_) FOUR_C_THROW("you didn't evaluate this contact state first");

  if (bt == CONTACT::MatBlockType::displ_displ)
    return kc_;
  else
    FOUR_C_THROW("get_matrix_block_ptr: your type is no treated properly!");

  return nullptr;
}

void CONTACT::NitscheStrategy::setup(bool redistributed, bool init)
{
  // we need to init the isselfcontact_ flag here, as we do not want to call the AbstractStrategy
  if (init)
  {
    // set potential global self contact status
    // (this is TRUE if at least one contact interface is a self contact interface)
    bool selfcontact = false;
    for (const auto& interface : interfaces())
      if (interface->self_contact()) selfcontact = true;

    if (selfcontact) isselfcontact_ = true;
  }
  reconnect_parent_elements();
  curr_state_ = nullptr;
  curr_state_eval_ = false;
}

void CONTACT::NitscheStrategy::update_trace_ineq_etimates()
{
  auto NitWgt = Teuchos::getIntegralValue<CONTACT::NitscheWeighting>(params(), "NITSCHE_WEIGHTING");
  for (const auto& interface : interface_)
  {
    for (int e = 0; e < interface->discret().element_col_map()->num_my_elements(); ++e)
    {
      auto* mele = dynamic_cast<Mortar::Element*>(
          interface->discret().g_element(interface->discret().element_col_map()->gid(e)));
      if (NitWgt == CONTACT::NitscheWeighting::slave && !mele->is_slave()) continue;
      if (NitWgt == CONTACT::NitscheWeighting::master && mele->is_slave()) continue;
      mele->estimate_nitsche_trace_max_eigenvalue();
    }
  }
}

void CONTACT::NitscheStrategy::update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  if (params().get<bool>("NITSCHE_PENALTY_ADAPTIVE")) update_trace_ineq_etimates();
  if (friction_)
  {
    store_to_old(Mortar::StrategyBase::n_old);
    set_state(Mortar::state_old_displacement, *dis);
  }
}

void CONTACT::NitscheStrategy::evaluate_reference_state()
{
  if (friction_)
  {
    for (const auto& interface : interface_)
    {
      interface->evaluate_nodal_normals();
      interface->export_nodal_normals();
    }
    store_to_old(Mortar::StrategyBase::n_old);
  }

  update_trace_ineq_etimates();
}


/*----------------------------------------------------------------------------------------------*
 |  Reconnect Contact Element -- Parent Element Pointers (required for restart)       ager 04/16|
 *---------------------------------------------------------------------------------------------*/
void CONTACT::NitscheStrategy::reconnect_parent_elements()
{
  std::shared_ptr<Core::FE::Discretization> voldis =
      Global::Problem::instance()->get_dis("structure");

  for (const auto& contact_interface : contact_interfaces())
  {
    const Core::LinAlg::Map* elecolmap = voldis->element_col_map();

    const Core::LinAlg::Map* ielecolmap = contact_interface->discret().element_col_map();

    for (int i = 0; i < ielecolmap->num_my_elements(); ++i)
    {
      const int gid = ielecolmap->gid(i);

      Core::Elements::Element* ele = contact_interface->discret().g_element(gid);
      if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
      auto* faceele = dynamic_cast<Core::Elements::FaceElement*>(ele);

      const int volgid = faceele->parent_element_id();
      if (elecolmap->lid(volgid) == -1)  // Volume discretization has not Element
        FOUR_C_THROW(
            "Manager::reconnect_parent_elements: Element {} does not exist on this Proc!", volgid);

      Core::Elements::Element* vele = voldis->g_element(volgid);
      if (!vele) FOUR_C_THROW("Cannot find element with gid %", volgid);

      faceele->set_parent_master_element(vele, faceele->face_parent_number());
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
