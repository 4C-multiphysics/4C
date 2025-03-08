// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_coupling_base.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fluid_ele_parameter_xfem.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_utils_function.hpp"
#include "4C_xfem_interface_utils.hpp"
#include "4C_xfem_utils.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

Inpar::XFEM::EleCouplingCondType XFEM::cond_type_string_to_enum(const std::string& condname)
{
  if (condname == "XFEMSurfFSIPart")
    return Inpar::XFEM::CouplingCond_SURF_FSI_PART;
  else if (condname == "XFEMSurfFSIMono")
    return Inpar::XFEM::CouplingCond_SURF_FSI_MONO;
  else if (condname == "XFEMSurfFPIMono" || condname == "XFEMSurfFPIMono_ps_ps" ||
           condname == "XFEMSurfFPIMono_ps_pf" || condname == "XFEMSurfFPIMono_pf_ps" ||
           condname == "XFEMSurfFPIMono_pf_pf")
    return Inpar::XFEM::CouplingCond_SURF_FPI_MONO;
  else if (condname == "XFEMSurfFluidFluid")
    return Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID;
  else if (condname == "XFEMLevelsetWeakDirichlet")
    return Inpar::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET;
  else if (condname == "XFEMLevelsetNeumann")
    return Inpar::XFEM::CouplingCond_LEVELSET_NEUMANN;
  else if (condname == "XFEMLevelsetNavierSlip")
    return Inpar::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP;
  else if (condname == "XFEMLevelsetTwophase")
    return Inpar::XFEM::CouplingCond_LEVELSET_TWOPHASE;
  else if (condname == "XFEMLevelsetCombustion")
    return Inpar::XFEM::CouplingCond_LEVELSET_COMBUSTION;
  else if (condname == "XFEMSurfWeakDirichlet")
    return Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET;
  else if (condname == "XFEMSurfNeumann")
    return Inpar::XFEM::CouplingCond_SURF_NEUMANN;
  else if (condname == "XFEMSurfNavierSlip")
    return Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP;
  else if (condname == "XFEMSurfNavierSlipTwoPhase")
    return Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE;
  else if (condname == "EmbeddedMeshSolidSurfCoupling")
    return Inpar::XFEM::CouplingCond_EMBEDDEDMESH_SOLID_SURF;
  else if (condname == "EmbeddedMeshSolidVolBackground")
    return Inpar::XFEM::CouplingCond_EMBEDDEDMESH_BACKGROUND_SOLID_VOL;
  // else FOUR_C_THROW("condition type not supported: %s", condname.c_str());

  return Inpar::XFEM::CouplingCond_NONE;
}

/*--------------------------------------------------------------------------*
 * constructor
 *--------------------------------------------------------------------------*/
XFEM::CouplingBase::CouplingBase(
    std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
    const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                   ///< discretization is identified
    std::shared_ptr<Core::FE::Discretization>&
        cond_dis,           ///< full discretization from which the cutter discretization is derived
    const int coupling_id,  ///< id of composite of coupling conditions
    const double time,      ///< time
    const int step          ///< time step
    )
    : nsd_(Global::Problem::instance()->n_dim()),
      bg_dis_(bg_dis),
      cond_name_(cond_name),
      cond_dis_(cond_dis),
      coupling_id_(coupling_id),
      cutter_dis_(nullptr),
      coupl_dis_(nullptr),
      coupl_name_(""),
      averaging_strategy_(Inpar::XFEM::invalid),
      myrank_(Core::Communication::my_mpi_rank(bg_dis_->get_comm())),
      dt_(-1.0),
      time_(time),
      step_(step),
      issetup_(false),
      isinit_(false)
{
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::CouplingBase::init()
{
  // TODO: correct handling of init and setup flags for derived classes

  // ---------------------------------------------------------------------------
  // We need to call setup() after init()
  // ---------------------------------------------------------------------------
  issetup_ = false;

  // ---------------------------------------------------------------------------
  // do Init
  // ---------------------------------------------------------------------------
  set_coupling_dofsets();

  // set the name of the coupling object to allow access from outside via the name
  set_coupling_name();

  // set list of conditions that will be copied to the new cutter discretization
  set_conditions_to_copy();

  // create a cutter discretization from conditioned nodes of the given coupling discretization or
  // simply clone the discretization
  set_cutter_discretization();

  // set unique element conditions
  set_element_conditions();

  // set condition specific parameters
  set_condition_specific_parameters();

  // set the averaging strategy
  set_averaging_strategy();

  // set coupling discretization
  set_coupling_discretization();

  // initialize element level configuration map (no evaluation)
  init_configuration_map();

  // ---------------------------------------------------------------------------
  // set isInit flag
  // ---------------------------------------------------------------------------
  isinit_ = true;

  // good bye
  return;
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::CouplingBase::setup()
{
  check_init();

  // ---------------------------------------------------------------------------
  // do setup
  // ---------------------------------------------------------------------------

  // initialize state vectors according to cutter discretization
  init_state_vectors();

  // prepare the output writer for the cutter discretization
  prepare_cutter_output();

  // do condition specific setup
  do_condition_specific_setup();

  // initialize the configuration map
  setup_configuration_map();

  // ---------------------------------------------------------------------------
  // set isSetup flag
  // ---------------------------------------------------------------------------

  issetup_ = true;
}


/*--------------------------------------------------------------------------*
 * Initialize Configuration Map --> No Terms are evaluated at the interface
 *--------------------------------------------------------------------------*/
void XFEM::CouplingBase::init_configuration_map()
{
  // Configuration of Consistency Terms
  // all components:
  configuration_map_[Inpar::XFEM::F_Con_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Con_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Con_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XF_Con_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XS_Con_Col] = std::pair<bool, double>(false, 0.0);
  // normal terms:
  configuration_map_[Inpar::XFEM::F_Con_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Con_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Con_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XF_Con_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XS_Con_n_Col] = std::pair<bool, double>(false, 0.0);
  // tangential terms:
  configuration_map_[Inpar::XFEM::F_Con_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Con_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Con_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XF_Con_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XS_Con_t_Col] = std::pair<bool, double>(false, 0.0);

  // Configuration of Adjoint Consistency Terms
  // all components:
  configuration_map_[Inpar::XFEM::F_Adj_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XF_Adj_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XS_Adj_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Adj_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Adj_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::FStr_Adj_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XStr_Adj_Col] = std::pair<bool, double>(false, 0.0);
  // normal terms:
  configuration_map_[Inpar::XFEM::F_Adj_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XF_Adj_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XS_Adj_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Adj_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Adj_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::FStr_Adj_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XStr_Adj_n_Col] = std::pair<bool, double>(false, 0.0);
  // tangential terms:
  configuration_map_[Inpar::XFEM::F_Adj_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XF_Adj_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XS_Adj_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Adj_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Adj_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::FStr_Adj_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::XStr_Adj_t_Col] = std::pair<bool, double>(false, 0.0);

  // Configuration of Penalty Terms
  // all components:
  configuration_map_[Inpar::XFEM::F_Pen_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Pen_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Pen_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Pen_Col] = std::pair<bool, double>(false, 0.0);
  // linearization of penalty terms: at the moment exclusively used for inflow stab
  configuration_map_[Inpar::XFEM::F_Pen_Row_linF1] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Pen_Row_linF2] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Pen_Row_linF3] = std::pair<bool, double>(false, 0.0);
  // normal terms:
  configuration_map_[Inpar::XFEM::F_Pen_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Pen_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Pen_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Pen_n_Col] = std::pair<bool, double>(false, 0.0);
  // tangential terms:
  configuration_map_[Inpar::XFEM::F_Pen_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Pen_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_Pen_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_Pen_t_Col] = std::pair<bool, double>(false, 0.0);

  // Starting from here are some special Terms
  configuration_map_[Inpar::XFEM::F_LB_Rhs] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_LB_Rhs] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::F_TJ_Rhs] = std::pair<bool, double>(false, 0.0);
  configuration_map_[Inpar::XFEM::X_TJ_Rhs] = std::pair<bool, double>(false, 0.0);
  return;
}


void XFEM::CouplingBase::set_element_conditions()
{
  // number of column cutter boundary elements
  int nummycolele = cutter_dis_->num_my_col_elements();

  cutterele_conds_.clear();
  cutterele_conds_.reserve(nummycolele);

  // initialize the vector invalid coupling-condition type "NONE"
  EleCoupCond init_pair = EleCoupCond(Inpar::XFEM::CouplingCond_NONE, nullptr);
  for (int lid = 0; lid < nummycolele; lid++) cutterele_conds_.push_back(init_pair);

  //-----------------------------------------------------------------------------------
  // loop all column cutting elements on this processor
  for (int lid = 0; lid < nummycolele; lid++)
  {
    Core::Elements::Element* cutele = cutter_dis_->l_col_element(lid);

    // loop all possible XFEM-coupling conditions
    for (size_t cond = 0; cond < conditions_to_copy_.size(); cond++)
    {
      Inpar::XFEM::EleCouplingCondType cond_type =
          cond_type_string_to_enum(conditions_to_copy_[cond]);

      // non-coupling condition found (e.g. FSI coupling)
      if (cond_type == Inpar::XFEM::CouplingCond_NONE) continue;

      // get all conditions with given condition name
      std::vector<Core::Conditions::Condition*> mycond;
      Core::Conditions::find_element_conditions(cutele, conditions_to_copy_[cond], mycond);

      std::vector<Core::Conditions::Condition*> mynewcond;
      get_condition_by_coupling_id(mycond, coupling_id_, mynewcond);

      Core::Conditions::Condition* cond_unique = nullptr;

      // safety checks
      if (mynewcond.size() == 0)
      {
        continue;  // try the next condition type
      }
      else if (mynewcond.size() == 1)  // unique condition found
      {
        cond_unique = mynewcond[0];
      }
      else if (mynewcond.size() > 1)
      {
        // get the right condition
        FOUR_C_THROW(
            "%i conditions of the same name with coupling id %i, for element %i! %s "
            "coupling-condition not unique!",
            mynewcond.size(), coupling_id_, cutele->id(), conditions_to_copy_[cond].c_str());
      }

      // non-unique conditions for one cutter element
      if (cutterele_conds_[lid].first != Inpar::XFEM::CouplingCond_NONE)
      {
        FOUR_C_THROW(
            "There are two different condition types for the same cutter dis element with id %i: "
            "1st %i, 2nd %i. Make the XFEM coupling conditions unique!",
            cutele->id(), cutterele_conds_[lid].first, cond_type);
      }

      // store the unique condition pointer to the cutting element
      cutterele_conds_[lid] = EleCoupCond(cond_type, cond_unique);
    }
  }

  //-----------------------------------------------------------------------------------
  // check if all column cutter elements have a valid condition type
  // loop all column cutting elements on this processor
  for (int lid = 0; lid < nummycolele; lid++)
  {
    if (cutterele_conds_[lid].first == Inpar::XFEM::CouplingCond_NONE)
      FOUR_C_THROW("cutter element with local id %i has no valid coupling-condition", lid);
  }
}

void XFEM::CouplingBase::get_condition_by_coupling_id(
    const std::vector<Core::Conditions::Condition*>& mycond, const int coupling_id,
    std::vector<Core::Conditions::Condition*>& mynewcond)
{
  mynewcond.clear();

  // select the conditions with specified "couplingID"
  for (auto* cond : mycond)
  {
    const int id = cond->parameters().get<int>("COUPLINGID");

    if (id == coupling_id) mynewcond.push_back(cond);
  }
}

void XFEM::CouplingBase::status(const int coupling_idx, const int side_start_gid)
{
  // -------------------------------------------------------------------
  //                       output to screen
  // -------------------------------------------------------------------
  if (myrank_ == 0)
  {
    printf(
        "   "
        "+----------+-----------+-----------------------------+---------+--------------------------"
        "---+-----------------------------+-----------------------------+--------------------------"
        "---+\n");
    printf("   | %8i | %9i | %27s | %7i | %27s | %27s | %27s | %27s |\n", coupling_idx,
        side_start_gid, type_to_string_for_print(cond_type_string_to_enum(cond_name_)).c_str(),
        coupling_id_, dis_name_to_string(cutter_dis_).c_str(),
        dis_name_to_string(cond_dis_).c_str(), dis_name_to_string(coupl_dis_).c_str(),
        averaging_to_string_for_print(averaging_strategy_).c_str());
  }
}



void XFEM::CouplingBase::set_averaging_strategy()
{
  const Inpar::XFEM::EleCouplingCondType cond_type = cond_type_string_to_enum(cond_name_);

  switch (cond_type)
  {
    case Inpar::XFEM::CouplingCond_SURF_FSI_MONO:
    {
      // ask the first cutter element
      const int lid = 0;
      averaging_strategy_ =
          cutterele_conds_[lid].second->parameters().get<Inpar::XFEM::AveragingStrategy>(
              "COUPSTRATEGY");
      // check unhandled cased
      if (averaging_strategy_ == Inpar::XFEM::Mean || averaging_strategy_ == Inpar::XFEM::Harmonic)
        FOUR_C_THROW(
            "XFEM::CouplingBase::set_averaging_strategy(): Strategy Mean/Harmoninc not available "
            "for "
            "FSI monolithic, ... coming soon!");
      break;
    }
    case Inpar::XFEM::CouplingCond_SURF_FPI_MONO:
    {
      averaging_strategy_ = Inpar::XFEM::Xfluid_Sided;
      break;
    }
    case Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID:
    {
      // ask the first cutter element
      const int lid = 0;
      averaging_strategy_ =
          cutterele_conds_[lid].second->parameters().get<Inpar::XFEM::AveragingStrategy>(
              "COUPSTRATEGY");
      break;
    }
    case Inpar::XFEM::CouplingCond_LEVELSET_TWOPHASE:
    case Inpar::XFEM::CouplingCond_LEVELSET_COMBUSTION:
    {
      averaging_strategy_ = Inpar::XFEM::Harmonic;
      break;
    }
    case Inpar::XFEM::CouplingCond_SURF_FSI_PART:
    case Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET:
    case Inpar::XFEM::CouplingCond_SURF_NEUMANN:
    case Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP:
    case Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE:
    case Inpar::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET:
    case Inpar::XFEM::CouplingCond_LEVELSET_NEUMANN:
    case Inpar::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP:
    {
      averaging_strategy_ = Inpar::XFEM::Xfluid_Sided;
      break;
    }
    case Inpar::XFEM::CouplingCond_EMBEDDEDMESH_BACKGROUND_SOLID_VOL:
    case Inpar::XFEM::CouplingCond_EMBEDDEDMESH_SOLID_SURF:
    {
      averaging_strategy_ = Inpar::XFEM::Mean;
      break;
    }
    default:
      FOUR_C_THROW("which is the averaging strategy for this type of coupling %i?", cond_type);
      break;
  }
}


void XFEM::CouplingBase::set_coupling_discretization()
{
  const Inpar::XFEM::EleCouplingCondType cond_type = cond_type_string_to_enum(cond_name_);

  switch (cond_type)
  {
    case Inpar::XFEM::CouplingCond_SURF_FPI_MONO:
    {
      coupl_dis_ = cutter_dis_;
      break;
    }
    case Inpar::XFEM::CouplingCond_SURF_FSI_MONO:
    case Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID:
    {
      // depending on the weighting strategy
      if (averaging_strategy_ == Inpar::XFEM::Xfluid_Sided)
      {
        coupl_dis_ = cutter_dis_;
      }
      else if (averaging_strategy_ == Inpar::XFEM::Embedded_Sided or
               averaging_strategy_ == Inpar::XFEM::Mean)
      {
        coupl_dis_ = cond_dis_;
      }
      else
        FOUR_C_THROW("Invalid coupling strategy for XFF or XFSI application");
      break;
    }
    case Inpar::XFEM::CouplingCond_LEVELSET_TWOPHASE:
    case Inpar::XFEM::CouplingCond_LEVELSET_COMBUSTION:
    {
      coupl_dis_ = bg_dis_;
      break;
    }
    case Inpar::XFEM::CouplingCond_SURF_FSI_PART:
    case Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET:  // set this to nullptr when the
                                                         // values are read from the function
                                                         // instead of the ivelnp vector
    case Inpar::XFEM::CouplingCond_SURF_NEUMANN:
    case Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP:
    case Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE:
    {
      coupl_dis_ = cutter_dis_;
      break;
    }
    case Inpar::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET:
    case Inpar::XFEM::CouplingCond_LEVELSET_NEUMANN:
    case Inpar::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP:
    {
      coupl_dis_ = nullptr;
      break;
    }
    case Inpar::XFEM::CouplingCond_EMBEDDEDMESH_BACKGROUND_SOLID_VOL:
    case Inpar::XFEM::CouplingCond_EMBEDDEDMESH_SOLID_SURF:
    {
      coupl_dis_ = cond_dis_;
      break;
    }

    default:
      FOUR_C_THROW("which is the coupling discretization for this type of coupling %i?", cond_type);
      break;
  }
}

void XFEM::CouplingBase::evaluate_dirichlet_function(Core::LinAlg::Matrix<3, 1>& ivel,
    const Core::LinAlg::Matrix<3, 1>& x, const Core::Conditions::Condition* cond, double time)
{
  std::vector<double> final_values(3, 0.0);

  evaluate_function(final_values, x.data(), cond, time);

  ivel(0, 0) = final_values[0];
  ivel(1, 0) = final_values[1];
  ivel(2, 0) = final_values[2];
}

void XFEM::CouplingBase::evaluate_neumann_function(Core::LinAlg::Matrix<3, 1>& itraction,
    const Core::LinAlg::Matrix<3, 1>& x, const Core::Conditions::Condition* cond, double time)
{
  std::vector<double> final_values(3, 0.0);

  //---------------------------------------
  const auto condtype = cond->parameters().get<std::string>("TYPE");

  // get usual body force
  if (!(condtype == "Dead" or condtype == "Live")) FOUR_C_THROW("Unknown Neumann condition");
  //---------------------------------------

  evaluate_function(final_values, x.data(), cond, time);

  itraction(0, 0) = final_values[0];
  itraction(1, 0) = final_values[1];
  itraction(2, 0) = final_values[2];
}

void XFEM::CouplingBase::evaluate_neumann_function(Core::LinAlg::Matrix<6, 1>& itraction,
    const Core::LinAlg::Matrix<3, 1>& x, const Core::Conditions::Condition* cond, double time)
{
  std::vector<double> final_values(6, 0.0);

  //---------------------------------------
  const auto condtype = cond->parameters().get<std::string>("TYPE");

  // get usual body force
  if (!(condtype == "Dead" or condtype == "Live")) FOUR_C_THROW("Unknown Neumann condition");
  //---------------------------------------

  evaluate_function(final_values, x.data(), cond, time);

  for (unsigned i = 0; i < 6; ++i) itraction(i, 0) = final_values[i];
}

void XFEM::CouplingBase::evaluate_function(std::vector<double>& final_values, const double* x,
    const Core::Conditions::Condition* cond, const double time)
{
  if (cond == nullptr) FOUR_C_THROW("invalid condition");

  const int numdof = cond->parameters().get<int>("NUMDOF");

  if (numdof != (int)final_values.size())
    FOUR_C_THROW("you specified NUMDOF %i in the input file, however, only %i dofs allowed!",
        numdof, (int)final_values.size());

  //---------------------------------------
  // get values and switches from the condition
  const auto onoff = cond->parameters().get<std::vector<int>>("ONOFF");
  const auto val = cond->parameters().get<std::vector<double>>("VAL");
  const auto functions = cond->parameters().get<std::vector<std::optional<int>>>("FUNCT");

  // uniformly distributed random noise
  auto& secondary = const_cast<Core::Conditions::Condition&>(*cond);
  const auto percentage = secondary.parameters().get_or<double>("RANDNOISE", 0.0);

  if (time < -1e-14) FOUR_C_THROW("Negative time in curve/function evaluation: time = %f", time);

  //---------------------------------------
  // set this condition
  //---------------------------------------
  for (int dof = 0; dof < numdof; ++dof)
  {
    // initialization of time-curve factor and function factor
    double functionfac = 1.0;

    double num = onoff[dof] * val[dof];

    // get factor given by spatial function
    if (functions[dof].has_value() && functions[dof].value() > 0)
    {
      functionfac = Global::Problem::instance()
                        ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functions[dof].value())
                        .evaluate(x, time, dof % numdof);
    }

    // uniformly distributed noise
    double noise = 0.0;
    if (fabs(percentage) > 1e-14)
    {
      const double randomnumber =
          Global::Problem::instance()->random()->uni();  // uniformly distributed between -1.0, 1.0
      noise = percentage * randomnumber;
    }

    final_values[dof] = num * (functionfac + noise);
  }  // loop dofs
}

void XFEM::CouplingBase::evaluate_scalar_function(double& final_values, const double* x,
    const double& val, const Core::Conditions::Condition* cond, const double time)
{
  if (cond == nullptr) FOUR_C_THROW("invalid condition");

  const int numdof = 1;

  //---------------------------------------
  // get values and switches from the condition
  const auto functnum = cond->parameters().get_or<int>("FUNCT", -1);

  // uniformly distributed random noise
  auto& secondary = const_cast<Core::Conditions::Condition&>(*cond);
  const auto percentage = secondary.parameters().get_or<double>("RANDNOISE", 0.0);

  if (time < -1e-14) FOUR_C_THROW("Negative time in curve/function evaluation: time = %f", time);

  //---------------------------------------
  // set this condition
  //---------------------------------------
  for (int dof = 0; dof < numdof; ++dof)
  {
    // initialization of time-curve factor and function factor
    double functionfac = 1.0;

    double num = val;

    if (functnum > 0)
    {
      functionfac = Global::Problem::instance()
                        ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functnum)
                        .evaluate(x, time, dof % numdof);
    }

    // uniformly distributed noise
    double noise = 0.0;
    if (fabs(percentage) > 1e-14)
    {
      const double randomnumber =
          Global::Problem::instance()->random()->uni();  // uniformly distributed between -1.0, 1.0
      noise = percentage * randomnumber;
    }

    final_values = num * (functionfac + noise);
  }  // loop dofs
}

/*--------------------------------------------------------------------------*
 * get viscosity of the master fluid
 *--------------------------------------------------------------------------*/
void XFEM::CouplingBase::get_viscosity_master(Core::Elements::Element* xfele,  ///< xfluid ele
    double& visc_m)  ///< viscosity mastersided
{
  // Get Materials of master
  std::shared_ptr<Core::Mat::Material> mat_m;

  // Todo: As soon as the master side may not be position = outside anymore we need to take that
  // into account
  // by an additional input parameter here (e.g. XFSI with TwoPhase)
  XFEM::Utils::get_volume_cell_material(xfele, mat_m, Cut::Point::outside);
  if (mat_m->material_type() == Core::Materials::m_fluid)
    visc_m = std::dynamic_pointer_cast<Mat::NewtonianFluid>(mat_m)->viscosity();
  else
    FOUR_C_THROW("get_coupling_specific_average_weights: Master Material not a fluid material?");
}

/*--------------------------------------------------------------------------*
 * get weighting parameters
 *--------------------------------------------------------------------------*/
void XFEM::CouplingBase::get_average_weights(Core::Elements::Element* xfele,  ///< xfluid ele
    Core::Elements::Element* coup_ele,                                        ///< coup_ele ele
    double& kappa_m,  ///< Weight parameter (parameter +/master side)
    double& kappa_s,  ///< Weight parameter (parameter -/slave  side)
    bool& non_xfluid_coupling)
{
  non_xfluid_coupling = (get_averaging_strategy() != Inpar::XFEM::Xfluid_Sided);

  if (get_averaging_strategy() != Inpar::XFEM::Harmonic)
    XFEM::Utils::get_std_average_weights(get_averaging_strategy(), kappa_m);
  else
    get_coupling_specific_average_weights(xfele, coup_ele, kappa_m);

  kappa_s = 1.0 - kappa_m;
}

/*--------------------------------------------------------------------------------
 * compute viscous part of Nitsche's penalty term scaling for Nitsche's method
 *--------------------------------------------------------------------------------*/
void XFEM::CouplingBase::get_visc_penalty_stabfac(Core::Elements::Element* xfele,  ///< xfluid ele
    Core::Elements::Element* coup_ele,                                             ///< coup_ele ele
    const double& kappa_m,  ///< Weight parameter (parameter +/master side)
    const double& kappa_s,  ///< Weight parameter (parameter -/slave  side)
    const double& inv_h_k,  ///< the inverse characteristic element length h_k
    const Discret::Elements::FluidEleParameterXFEM*
        params,                     ///< parameterlist which specifies interface configuration
    double& NIT_visc_stab_fac,      ///< viscous part of Nitsche's penalty term
    double& NIT_visc_stab_fac_tang  ///< viscous part of Nitsche's penalty term in tang direction
)
{
  get_visc_penalty_stabfac(xfele, coup_ele, kappa_m, kappa_s, inv_h_k, NIT_visc_stab_fac,
      NIT_visc_stab_fac_tang, params->nit_stab_scaling(), params->nit_stab_scaling_tang(),
      params->is_pseudo_2d(), params->visc_stab_trac_estimate());
}

/*--------------------------------------------------------------------------------
 * compute viscous part of Nitsche's penalty term scaling for Nitsche's method
 *--------------------------------------------------------------------------------*/
void XFEM::CouplingBase::get_visc_penalty_stabfac(Core::Elements::Element* xfele,  ///< xfluid ele
    Core::Elements::Element* coup_ele,                                             ///< coup_ele ele
    const double& kappa_m,           ///< Weight parameter (parameter +/master side)
    const double& kappa_s,           ///< Weight parameter (parameter -/slave  side)
    const double& inv_h_k,           ///< the inverse characteristic element length h_k
    double& NIT_visc_stab_fac,       ///< viscous part of Nitsche's penalty term
    double& NIT_visc_stab_fac_tang,  ///< viscous part of Nitsche's penalty term in tang direction
    const double& NITStabScaling, const double& NITStabScalingTang, const bool& IsPseudo2D,
    const Inpar::XFEM::ViscStabTraceEstimate ViscStab_TraceEstimate)
{
  double penscaling = 0.0;
  if (get_averaging_strategy() != Inpar::XFEM::Embedded_Sided)
  {
    double visc_m = 0.0;
    get_viscosity_master(
        xfele, visc_m);  // As long as mastersided we just have a fluid, directly use this ...
    penscaling = visc_m * kappa_m * inv_h_k;
  }

  if (get_averaging_strategy() != Inpar::XFEM::Xfluid_Sided)
  {
    double penscaling_s = 0.0;
    get_penalty_scaling_slave(coup_ele, penscaling_s);
    penscaling += penscaling_s * kappa_s * inv_h_k;
  }

  XFEM::Utils::nit_compute_visc_penalty_stabfac(xfele->shape(), penscaling, NITStabScaling,
      IsPseudo2D, ViscStab_TraceEstimate, NIT_visc_stab_fac);

  XFEM::Utils::nit_compute_visc_penalty_stabfac(xfele->shape(), penscaling, NITStabScalingTang,
      IsPseudo2D, ViscStab_TraceEstimate, NIT_visc_stab_fac_tang);
  return;
}

FOUR_C_NAMESPACE_CLOSE
