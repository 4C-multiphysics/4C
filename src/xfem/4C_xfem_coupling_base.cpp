/*----------------------------------------------------------------------*/
/*! \file

\brief is the base for the different types of mesh and level-set based coupling conditions and
thereby builds the bridge between the xfluid class and the cut-library

\level 2

*/
/*----------------------------------------------------------------------*/


#include "4C_xfem_coupling_base.hpp"

#include "4C_fluid_ele_parameter_xfem.hpp"
#include "4C_lib_condition_utils.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_utils_function.hpp"
#include "4C_xfem_interface_utils.hpp"
#include "4C_xfem_utils.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

INPAR::XFEM::EleCouplingCondType XFEM::CondType_stringToEnum(const std::string& condname)
{
  if (condname == "XFEMSurfFSIPart")
    return INPAR::XFEM::CouplingCond_SURF_FSI_PART;
  else if (condname == "XFEMSurfFSIMono")
    return INPAR::XFEM::CouplingCond_SURF_FSI_MONO;
  else if (condname == "XFEMSurfFPIMono" || condname == "XFEMSurfFPIMono_ps_ps" ||
           condname == "XFEMSurfFPIMono_ps_pf" || condname == "XFEMSurfFPIMono_pf_ps" ||
           condname == "XFEMSurfFPIMono_pf_pf")
    return INPAR::XFEM::CouplingCond_SURF_FPI_MONO;
  else if (condname == "XFEMSurfFluidFluid")
    return INPAR::XFEM::CouplingCond_SURF_FLUIDFLUID;
  else if (condname == "XFEMLevelsetWeakDirichlet")
    return INPAR::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET;
  else if (condname == "XFEMLevelsetNeumann")
    return INPAR::XFEM::CouplingCond_LEVELSET_NEUMANN;
  else if (condname == "XFEMLevelsetNavierSlip")
    return INPAR::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP;
  else if (condname == "XFEMLevelsetTwophase")
    return INPAR::XFEM::CouplingCond_LEVELSET_TWOPHASE;
  else if (condname == "XFEMLevelsetCombustion")
    return INPAR::XFEM::CouplingCond_LEVELSET_COMBUSTION;
  else if (condname == "XFEMSurfWeakDirichlet")
    return INPAR::XFEM::CouplingCond_SURF_WEAK_DIRICHLET;
  else if (condname == "XFEMSurfNeumann")
    return INPAR::XFEM::CouplingCond_SURF_NEUMANN;
  else if (condname == "XFEMSurfNavierSlip")
    return INPAR::XFEM::CouplingCond_SURF_NAVIER_SLIP;
  else if (condname == "XFEMSurfNavierSlipTwoPhase")
    return INPAR::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE;
  // else FOUR_C_THROW("condition type not supported: %s", condname.c_str());

  return INPAR::XFEM::CouplingCond_NONE;
}

/*--------------------------------------------------------------------------*
 * constructor
 *--------------------------------------------------------------------------*/
XFEM::CouplingBase::CouplingBase(
    Teuchos::RCP<DRT::Discretization>& bg_dis,  ///< background discretization
    const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                   ///< discretization is identified
    Teuchos::RCP<DRT::Discretization>&
        cond_dis,           ///< full discretization from which the cutter discretization is derived
    const int coupling_id,  ///< id of composite of coupling conditions
    const double time,      ///< time
    const int step          ///< time step
    )
    : nsd_(GLOBAL::Problem::Instance()->NDim()),
      bg_dis_(bg_dis),
      cond_name_(cond_name),
      cond_dis_(cond_dis),
      coupling_id_(coupling_id),
      cutter_dis_(Teuchos::null),
      coupl_dis_(Teuchos::null),
      coupl_name_(""),
      averaging_strategy_(INPAR::XFEM::invalid),
      myrank_(bg_dis_->Comm().MyPID()),
      dt_(-1.0),
      time_(time),
      step_(step),
      issetup_(false),
      isinit_(false)
{
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::CouplingBase::Init()
{
  // TODO: correct handling of init and setup flags for derived classes

  // ---------------------------------------------------------------------------
  // We need to call Setup() after Init()
  // ---------------------------------------------------------------------------
  issetup_ = false;

  // ---------------------------------------------------------------------------
  // do Init
  // ---------------------------------------------------------------------------

  if (dofset_coupling_map_.empty()) FOUR_C_THROW("Call SetDofSetCouplingMap() first!");

  SetCouplingDofsets();

  // set the name of the coupling object to allow access from outside via the name
  SetCouplingName();

  // set list of conditions that will be copied to the new cutter discretization
  SetConditionsToCopy();

  // create a cutter discretization from conditioned nodes of the given coupling discretization or
  // simply clone the discretization
  SetCutterDiscretization();

  // set unique element conditions
  SetElementConditions();

  // set condition specific parameters
  SetConditionSpecificParameters();

  // set the averaging strategy
  SetAveragingStrategy();

  // set coupling discretization
  SetCouplingDiscretization();

  // initialize element level configuration map (no evaluation)
  InitConfigurationMap();

  // ---------------------------------------------------------------------------
  // set isInit flag
  // ---------------------------------------------------------------------------
  isinit_ = true;

  // good bye
  return;
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::CouplingBase::Setup()
{
  CheckInit();

  // ---------------------------------------------------------------------------
  // do setup
  // ---------------------------------------------------------------------------

  // initialize state vectors according to cutter discretization
  InitStateVectors();

  // prepare the output writer for the cutter discretization
  PrepareCutterOutput();

  // do condition specific setup
  DoConditionSpecificSetup();

  // initialize the configuration map
  SetupConfigurationMap();

  // ---------------------------------------------------------------------------
  // set isSetup flag
  // ---------------------------------------------------------------------------

  issetup_ = true;
}


/*--------------------------------------------------------------------------*
 * Initialize Configuration Map --> No Terms are evaluated at the interface
 *--------------------------------------------------------------------------*/
void XFEM::CouplingBase::InitConfigurationMap()
{
  // Configuration of Consistency Terms
  // all components:
  configuration_map_[INPAR::XFEM::F_Con_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Con_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Con_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XF_Con_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XS_Con_Col] = std::pair<bool, double>(false, 0.0);
  // normal terms:
  configuration_map_[INPAR::XFEM::F_Con_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Con_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Con_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XF_Con_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XS_Con_n_Col] = std::pair<bool, double>(false, 0.0);
  // tangential terms:
  configuration_map_[INPAR::XFEM::F_Con_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Con_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Con_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XF_Con_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XS_Con_t_Col] = std::pair<bool, double>(false, 0.0);

  // Configuration of Adjoint Consistency Terms
  // all components:
  configuration_map_[INPAR::XFEM::F_Adj_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XF_Adj_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XS_Adj_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Adj_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Adj_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::FStr_Adj_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XStr_Adj_Col] = std::pair<bool, double>(false, 0.0);
  // normal terms:
  configuration_map_[INPAR::XFEM::F_Adj_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XF_Adj_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XS_Adj_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Adj_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Adj_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::FStr_Adj_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XStr_Adj_n_Col] = std::pair<bool, double>(false, 0.0);
  // tangential terms:
  configuration_map_[INPAR::XFEM::F_Adj_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XF_Adj_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XS_Adj_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Adj_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Adj_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::FStr_Adj_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::XStr_Adj_t_Col] = std::pair<bool, double>(false, 0.0);

  // Configuration of Penalty Terms
  // all components:
  configuration_map_[INPAR::XFEM::F_Pen_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Pen_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Pen_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Pen_Col] = std::pair<bool, double>(false, 0.0);
  // linearization of penalty terms: at the moment exclusively used for inflow stab
  configuration_map_[INPAR::XFEM::F_Pen_Row_linF1] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Pen_Row_linF2] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Pen_Row_linF3] = std::pair<bool, double>(false, 0.0);
  // normal terms:
  configuration_map_[INPAR::XFEM::F_Pen_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Pen_n_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Pen_n_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Pen_n_Col] = std::pair<bool, double>(false, 0.0);
  // tangential terms:
  configuration_map_[INPAR::XFEM::F_Pen_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Pen_t_Row] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_Pen_t_Col] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_Pen_t_Col] = std::pair<bool, double>(false, 0.0);

  // Starting from here are some special Terms
  configuration_map_[INPAR::XFEM::F_LB_Rhs] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_LB_Rhs] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::F_TJ_Rhs] = std::pair<bool, double>(false, 0.0);
  configuration_map_[INPAR::XFEM::X_TJ_Rhs] = std::pair<bool, double>(false, 0.0);
  return;
}


void XFEM::CouplingBase::SetElementConditions()
{
  // number of column cutter boundary elements
  int nummycolele = cutter_dis_->NumMyColElements();

  cutterele_conds_.clear();
  cutterele_conds_.reserve(nummycolele);

  // initialize the vector invalid coupling-condition type "NONE"
  EleCoupCond init_pair = EleCoupCond(INPAR::XFEM::CouplingCond_NONE, nullptr);
  for (int lid = 0; lid < nummycolele; lid++) cutterele_conds_.push_back(init_pair);

  //-----------------------------------------------------------------------------------
  // loop all column cutting elements on this processor
  for (int lid = 0; lid < nummycolele; lid++)
  {
    DRT::Element* cutele = cutter_dis_->lColElement(lid);

    // loop all possible XFEM-coupling conditions
    for (size_t cond = 0; cond < conditions_to_copy_.size(); cond++)
    {
      INPAR::XFEM::EleCouplingCondType cond_type = CondType_stringToEnum(conditions_to_copy_[cond]);

      // non-coupling condition found (e.g. FSI coupling)
      if (cond_type == INPAR::XFEM::CouplingCond_NONE) continue;

      // get all conditions with given condition name
      std::vector<DRT::Condition*> mycond;
      DRT::UTILS::FindElementConditions(cutele, conditions_to_copy_[cond], mycond);

      std::vector<DRT::Condition*> mynewcond;
      GetConditionByCouplingId(mycond, coupling_id_, mynewcond);

      DRT::Condition* cond_unique = nullptr;

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
            mynewcond.size(), coupling_id_, cutele->Id(), conditions_to_copy_[cond].c_str());
      }

      // non-unique conditions for one cutter element
      if (cutterele_conds_[lid].first != INPAR::XFEM::CouplingCond_NONE)
      {
        FOUR_C_THROW(
            "There are two different condition types for the same cutter dis element with id %i: "
            "1st %i, 2nd %i. Make the XFEM coupling conditions unique!",
            cutele->Id(), cutterele_conds_[lid].first, cond_type);
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
    if (cutterele_conds_[lid].first == INPAR::XFEM::CouplingCond_NONE)
      FOUR_C_THROW("cutter element with local id %i has no valid coupling-condition", lid);
  }
}

void XFEM::CouplingBase::GetConditionByCouplingId(const std::vector<DRT::Condition*>& mycond,
    const int coupling_id, std::vector<DRT::Condition*>& mynewcond)
{
  mynewcond.clear();

  // select the conditions with specified "couplingID"
  for (auto* cond : mycond)
  {
    const int id = cond->parameters().Get<int>("label");

    if (id == coupling_id) mynewcond.push_back(cond);
  }
}

void XFEM::CouplingBase::Status(const int coupling_idx, const int side_start_gid)
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
        side_start_gid, TypeToStringForPrint(CondType_stringToEnum(cond_name_)).c_str(),
        coupling_id_, DisNameToString(cutter_dis_).c_str(), DisNameToString(cond_dis_).c_str(),
        DisNameToString(coupl_dis_).c_str(),
        AveragingToStringForPrint(averaging_strategy_).c_str());
  }
}



void XFEM::CouplingBase::SetAveragingStrategy()
{
  const INPAR::XFEM::EleCouplingCondType cond_type = CondType_stringToEnum(cond_name_);

  switch (cond_type)
  {
    case INPAR::XFEM::CouplingCond_SURF_FSI_MONO:
    {
      // ask the first cutter element
      const int lid = 0;
      const int val = cutterele_conds_[lid].second->parameters().Get<int>("COUPSTRATEGY");
      averaging_strategy_ = static_cast<INPAR::XFEM::AveragingStrategy>(val);
      // check unhandled cased
      if (averaging_strategy_ == INPAR::XFEM::Mean || averaging_strategy_ == INPAR::XFEM::Harmonic)
        FOUR_C_THROW(
            "XFEM::CouplingBase::SetAveragingStrategy(): Strategy Mean/Harmoninc not available for "
            "FSI monolithic, ... coming soon!");
      break;
    }
    case INPAR::XFEM::CouplingCond_SURF_FPI_MONO:
    {
      averaging_strategy_ = INPAR::XFEM::Xfluid_Sided;
      break;
    }
    case INPAR::XFEM::CouplingCond_SURF_FLUIDFLUID:
    {
      // ask the first cutter element
      const int lid = 0;
      const int val = cutterele_conds_[lid].second->parameters().Get<int>("COUPSTRATEGY");
      averaging_strategy_ = static_cast<INPAR::XFEM::AveragingStrategy>(val);
      break;
    }
    case INPAR::XFEM::CouplingCond_LEVELSET_TWOPHASE:
    case INPAR::XFEM::CouplingCond_LEVELSET_COMBUSTION:
    {
      averaging_strategy_ = INPAR::XFEM::Harmonic;
      break;
    }
    case INPAR::XFEM::CouplingCond_SURF_FSI_PART:
    case INPAR::XFEM::CouplingCond_SURF_WEAK_DIRICHLET:
    case INPAR::XFEM::CouplingCond_SURF_NEUMANN:
    case INPAR::XFEM::CouplingCond_SURF_NAVIER_SLIP:
    case INPAR::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE:
    case INPAR::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET:
    case INPAR::XFEM::CouplingCond_LEVELSET_NEUMANN:
    case INPAR::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP:
    {
      averaging_strategy_ = INPAR::XFEM::Xfluid_Sided;
      break;
    }
    default:
      FOUR_C_THROW("which is the averaging strategy for this type of coupling %i?", cond_type);
      break;
  }
}


void XFEM::CouplingBase::SetCouplingDiscretization()
{
  const INPAR::XFEM::EleCouplingCondType cond_type = CondType_stringToEnum(cond_name_);

  switch (cond_type)
  {
    case INPAR::XFEM::CouplingCond_SURF_FPI_MONO:
    {
      coupl_dis_ = cutter_dis_;
      break;
    }
    case INPAR::XFEM::CouplingCond_SURF_FSI_MONO:
    case INPAR::XFEM::CouplingCond_SURF_FLUIDFLUID:
    {
      // depending on the weighting strategy
      if (averaging_strategy_ == INPAR::XFEM::Xfluid_Sided)
      {
        coupl_dis_ = cutter_dis_;
      }
      else if (averaging_strategy_ == INPAR::XFEM::Embedded_Sided or
               averaging_strategy_ == INPAR::XFEM::Mean)
      {
        coupl_dis_ = cond_dis_;
      }
      else
        FOUR_C_THROW("Invalid coupling strategy for XFF or XFSI application");
      break;
    }
    case INPAR::XFEM::CouplingCond_LEVELSET_TWOPHASE:
    case INPAR::XFEM::CouplingCond_LEVELSET_COMBUSTION:
    {
      coupl_dis_ = bg_dis_;
      break;
    }
    case INPAR::XFEM::CouplingCond_SURF_FSI_PART:
    case INPAR::XFEM::CouplingCond_SURF_WEAK_DIRICHLET:  // set this to Teuchos::null when the
                                                         // values are read from the function
                                                         // instead of the ivelnp vector
    case INPAR::XFEM::CouplingCond_SURF_NEUMANN:
    case INPAR::XFEM::CouplingCond_SURF_NAVIER_SLIP:
    case INPAR::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE:
    {
      coupl_dis_ = cutter_dis_;
      break;
    }
    case INPAR::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET:
    case INPAR::XFEM::CouplingCond_LEVELSET_NEUMANN:
    case INPAR::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP:
    {
      coupl_dis_ = Teuchos::null;
      break;
    }
    default:
      FOUR_C_THROW("which is the coupling discretization for this type of coupling %i?", cond_type);
      break;
  }
}

void XFEM::CouplingBase::EvaluateDirichletFunction(CORE::LINALG::Matrix<3, 1>& ivel,
    const CORE::LINALG::Matrix<3, 1>& x, const DRT::Condition* cond, double time)
{
  std::vector<double> final_values(3, 0.0);

  EvaluateFunction(final_values, x.A(), cond, time);

  ivel(0, 0) = final_values[0];
  ivel(1, 0) = final_values[1];
  ivel(2, 0) = final_values[2];
}

void XFEM::CouplingBase::EvaluateNeumannFunction(CORE::LINALG::Matrix<3, 1>& itraction,
    const CORE::LINALG::Matrix<3, 1>& x, const DRT::Condition* cond, double time)
{
  std::vector<double> final_values(3, 0.0);

  //---------------------------------------
  const auto condtype = cond->parameters().Get<std::string>("type");

  // get usual body force
  if (!(condtype == "neum_dead" or condtype == "neum_live"))
    FOUR_C_THROW("Unknown Neumann condition");
  //---------------------------------------

  EvaluateFunction(final_values, x.A(), cond, time);

  itraction(0, 0) = final_values[0];
  itraction(1, 0) = final_values[1];
  itraction(2, 0) = final_values[2];
}

void XFEM::CouplingBase::EvaluateNeumannFunction(CORE::LINALG::Matrix<6, 1>& itraction,
    const CORE::LINALG::Matrix<3, 1>& x, const DRT::Condition* cond, double time)
{
  std::vector<double> final_values(6, 0.0);

  //---------------------------------------
  const auto condtype = cond->parameters().Get<std::string>("type");

  // get usual body force
  if (!(condtype == "neum_dead" or condtype == "neum_live"))
    FOUR_C_THROW("Unknown Neumann condition");
  //---------------------------------------

  EvaluateFunction(final_values, x.A(), cond, time);

  for (unsigned i = 0; i < 6; ++i) itraction(i, 0) = final_values[i];
}

void XFEM::CouplingBase::EvaluateFunction(std::vector<double>& final_values, const double* x,
    const DRT::Condition* cond, const double time)
{
  if (cond == nullptr) FOUR_C_THROW("invalid condition");

  const int numdof = cond->parameters().Get<int>("numdof");

  if (numdof != (int)final_values.size())
    FOUR_C_THROW("you specified NUMDOF %i in the input file, however, only %i dofs allowed!",
        numdof, (int)final_values.size());

  //---------------------------------------
  // get values and switches from the condition
  const auto* onoff = &cond->parameters().Get<std::vector<int>>("onoff");
  const auto* val = &cond->parameters().Get<std::vector<double>>("val");
  const auto* functions = cond->parameters().GetIf<std::vector<int>>("funct");

  // uniformly distributed random noise

  auto& secondary = const_cast<DRT::Condition&>(*cond);
  const auto* percentage = secondary.parameters().GetIf<double>("randnoise");

  if (time < -1e-14) FOUR_C_THROW("Negative time in curve/function evaluation: time = %f", time);

  //---------------------------------------
  // set this condition
  //---------------------------------------
  for (int dof = 0; dof < numdof; ++dof)
  {
    // get factor given by spatial function
    int functnum = -1;
    if (functions) functnum = (*functions)[dof];

    // initialization of time-curve factor and function factor
    double functionfac = 1.0;

    double num = (*onoff)[dof] * (*val)[dof];

    if (functnum > 0)
    {
      functionfac = GLOBAL::Problem::Instance()
                        ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(functnum - 1)
                        .Evaluate(x, time, dof % numdof);
    }

    // uniformly distributed noise
    double noise = 0.0;
    if (percentage != nullptr)
    {
      const double perc = *percentage;

      if (fabs(perc) > 1e-14)
      {
        const double randomnumber = GLOBAL::Problem::Instance()
                                        ->Random()
                                        ->Uni();  // uniformly distributed between -1.0, 1.0
        noise = perc * randomnumber;
      }
    }

    final_values[dof] = num * (functionfac + noise);
  }  // loop dofs
}

void XFEM::CouplingBase::EvaluateScalarFunction(double& final_values, const double* x,
    const double& val, const DRT::Condition* cond, const double time)
{
  if (cond == nullptr) FOUR_C_THROW("invalid condition");

  const int numdof = 1;

  //---------------------------------------
  // get values and switches from the condition
  const auto* function = cond->parameters().GetIf<int>("funct");

  // uniformly distributed random noise
  auto& secondary = const_cast<DRT::Condition&>(*cond);
  const auto* percentage = secondary.parameters().GetIf<double>("randnoise");

  if (time < -1e-14) FOUR_C_THROW("Negative time in curve/function evaluation: time = %f", time);

  //---------------------------------------
  // set this condition
  //---------------------------------------
  for (int dof = 0; dof < numdof; ++dof)
  {
    // get factor given by spatial function
    int functnum = -1;
    if (function) functnum = *function;

    // initialization of time-curve factor and function factor
    double functionfac = 1.0;

    double num = val;

    if (functnum > 0)
    {
      functionfac = GLOBAL::Problem::Instance()
                        ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(functnum - 1)
                        .Evaluate(x, time, dof % numdof);
    }

    // uniformly distributed noise
    double noise = 0.0;
    if (percentage != nullptr)
    {
      const double perc = *percentage;

      if (fabs(perc) > 1e-14)
      {
        const double randomnumber = GLOBAL::Problem::Instance()
                                        ->Random()
                                        ->Uni();  // uniformly distributed between -1.0, 1.0
        noise = perc * randomnumber;
      }
    }

    final_values = num * (functionfac + noise);
  }  // loop dofs
}

/*--------------------------------------------------------------------------*
 * get viscosity of the master fluid
 *--------------------------------------------------------------------------*/
void XFEM::CouplingBase::GetViscosityMaster(DRT::Element* xfele,  ///< xfluid ele
    double& visc_m)                                               ///< viscosity mastersided
{
  // Get Materials of master
  Teuchos::RCP<CORE::MAT::Material> mat_m;

  // Todo: As soon as the master side may not be position = outside anymore we need to take that
  // into account
  // by an additional input parameter here (e.g. XFSI with TwoPhase)
  XFEM::UTILS::GetVolumeCellMaterial(xfele, mat_m, CORE::GEO::CUT::Point::outside);
  if (mat_m->MaterialType() == CORE::Materials::m_fluid)
    visc_m = Teuchos::rcp_dynamic_cast<MAT::NewtonianFluid>(mat_m)->Viscosity();
  else
    FOUR_C_THROW("GetCouplingSpecificAverageWeights: Master Material not a fluid material?");
  return;
}

/*--------------------------------------------------------------------------*
 * get weighting paramters
 *--------------------------------------------------------------------------*/
void XFEM::CouplingBase::GetAverageWeights(DRT::Element* xfele,  ///< xfluid ele
    DRT::Element* coup_ele,                                      ///< coup_ele ele
    double& kappa_m,  ///< Weight parameter (parameter +/master side)
    double& kappa_s,  ///< Weight parameter (parameter -/slave  side)
    bool& non_xfluid_coupling)
{
  non_xfluid_coupling = (GetAveragingStrategy() != INPAR::XFEM::Xfluid_Sided);

  if (GetAveragingStrategy() != INPAR::XFEM::Harmonic)
    XFEM::UTILS::GetStdAverageWeights(GetAveragingStrategy(), kappa_m);
  else
    GetCouplingSpecificAverageWeights(xfele, coup_ele, kappa_m);

  kappa_s = 1.0 - kappa_m;
  return;
}

/*--------------------------------------------------------------------------------
 * compute viscous part of Nitsche's penalty term scaling for Nitsche's method
 *--------------------------------------------------------------------------------*/
void XFEM::CouplingBase::Get_ViscPenalty_Stabfac(DRT::Element* xfele,  ///< xfluid ele
    DRT::Element* coup_ele,                                            ///< coup_ele ele
    const double& kappa_m,  ///< Weight parameter (parameter +/master side)
    const double& kappa_s,  ///< Weight parameter (parameter -/slave  side)
    const double& inv_h_k,  ///< the inverse characteristic element length h_k
    const DRT::ELEMENTS::FluidEleParameterXFEM*
        params,                     ///< parameterlist which specifies interface configuration
    double& NIT_visc_stab_fac,      ///< viscous part of Nitsche's penalty term
    double& NIT_visc_stab_fac_tang  ///< viscous part of Nitsche's penalty term in tang direction
)
{
  Get_ViscPenalty_Stabfac(xfele, coup_ele, kappa_m, kappa_s, inv_h_k, NIT_visc_stab_fac,
      NIT_visc_stab_fac_tang, params->NITStabScaling(), params->NITStabScalingTang(),
      params->IsPseudo2D(), params->ViscStabTracEstimate());
}

/*--------------------------------------------------------------------------------
 * compute viscous part of Nitsche's penalty term scaling for Nitsche's method
 *--------------------------------------------------------------------------------*/
void XFEM::CouplingBase::Get_ViscPenalty_Stabfac(DRT::Element* xfele,  ///< xfluid ele
    DRT::Element* coup_ele,                                            ///< coup_ele ele
    const double& kappa_m,           ///< Weight parameter (parameter +/master side)
    const double& kappa_s,           ///< Weight parameter (parameter -/slave  side)
    const double& inv_h_k,           ///< the inverse characteristic element length h_k
    double& NIT_visc_stab_fac,       ///< viscous part of Nitsche's penalty term
    double& NIT_visc_stab_fac_tang,  ///< viscous part of Nitsche's penalty term in tang direction
    const double& NITStabScaling, const double& NITStabScalingTang, const bool& IsPseudo2D,
    const INPAR::XFEM::ViscStabTraceEstimate ViscStab_TraceEstimate)
{
  double penscaling = 0.0;
  if (GetAveragingStrategy() != INPAR::XFEM::Embedded_Sided)
  {
    double visc_m = 0.0;
    GetViscosityMaster(
        xfele, visc_m);  // As long as mastersided we just have a fluid, directly use this ...
    penscaling = visc_m * kappa_m * inv_h_k;
  }

  if (GetAveragingStrategy() != INPAR::XFEM::Xfluid_Sided)
  {
    double penscaling_s = 0.0;
    GetPenaltyScalingSlave(coup_ele, penscaling_s);
    penscaling += penscaling_s * kappa_s * inv_h_k;
  }

  XFEM::UTILS::NIT_Compute_ViscPenalty_Stabfac(xfele->Shape(), penscaling, NITStabScaling,
      IsPseudo2D, ViscStab_TraceEstimate, NIT_visc_stab_fac);

  XFEM::UTILS::NIT_Compute_ViscPenalty_Stabfac(xfele->Shape(), penscaling, NITStabScalingTang,
      IsPseudo2D, ViscStab_TraceEstimate, NIT_visc_stab_fac_tang);
  return;
}

FOUR_C_NAMESPACE_CLOSE
