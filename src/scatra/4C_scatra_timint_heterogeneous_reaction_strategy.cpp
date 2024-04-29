/*----------------------------------------------------------------------*/
/*! \file

 \brief Solution strategy for heterogeneous reactions. This is not meshtying!!!

  \level 3

*/
/*----------------------------------------------------------------------*/
#include "4C_scatra_timint_heterogeneous_reaction_strategy.hpp"

#include "4C_lib_discret.hpp"
#include "4C_lib_dofset_gidbased_wrapper.hpp"
#include "4C_lib_dofset_merged_wrapper.hpp"
#include "4C_lib_utils_createdis.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_rebalance_print.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_utils_clonestrategy.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                               vuong 06/16 |
 *----------------------------------------------------------------------*/
SCATRA::HeterogeneousReactionStrategy::HeterogeneousReactionStrategy(
    SCATRA::ScaTraTimIntImpl* scatratimint)
    : MeshtyingStrategyStd(scatratimint), issetup_(false), isinit_(false)
{
  return;
}  // SCATRA::HeterogeneousReactionStrategy::HeterogeneousReactionStrategy


/*------------------------------------------------------------------------*
 | evaluate heterogeneous reactions (actually no mesh tying    vuong 06/16 |
 *------------------------------------------------------------------------*/
void SCATRA::HeterogeneousReactionStrategy::EvaluateMeshtying()
{
  CheckIsInit();
  CheckIsSetup();

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  CORE::UTILS::AddEnumClassToParameterList<SCATRA::Action>(
      "action", SCATRA::Action::calc_heteroreac_mat_and_rhs, condparams);

  // set global state vectors according to time-integration scheme
  discret_->SetState("phinp", scatratimint_->Phiafnp());
  discret_->SetState("hist", scatratimint_->Hist());

  // provide scatra discretization with convective velocity
  discret_->SetState(scatratimint_->NdsVel(), "convective velocity field",
      scatratimint_->Discretization()->GetState(
          scatratimint_->NdsVel(), "convective velocity field"));

  // provide scatra discretization with velocity
  discret_->SetState(scatratimint_->NdsVel(), "velocity field",
      scatratimint_->Discretization()->GetState(scatratimint_->NdsVel(), "velocity field"));

  if (scatratimint_->IsALE())
  {
    discret_->SetState(scatratimint_->NdsDisp(), "dispnp",
        scatratimint_->Discretization()->GetState(scatratimint_->NdsDisp(), "dispnp"));
  }

  discret_->Evaluate(condparams, scatratimint_->SystemMatrix(), scatratimint_->Residual());

  // now we clear all states.
  // it would be nicer to do this directly before all
  // states are set at the beginning of this method.
  // However, in this case we are not able to set states externally
  // before this method is called.
  // See the call hierarchy of HeterogeneousReactionStrategy::SetState()
  // to check in which algorithms states are set on discret_ .
  discret_->ClearState();
  return;
}  // SCATRA::HeterogeneousReactionStrategy::EvaluateMeshtying


/*----------------------------------------------------------------------*
 | initialize meshtying objects                              rauch 09/16 |
 *----------------------------------------------------------------------*/
void SCATRA::HeterogeneousReactionStrategy::SetupMeshtying()
{
  // call Init() of base class
  SCATRA::MeshtyingStrategyStd::SetupMeshtying();

  // make sure we set up everything properly
  HeterogeneousReactionSanityCheck();

  Teuchos::RCP<Epetra_Comm> com = Teuchos::rcp(scatratimint_->Discretization()->Comm().Clone());

  // standard case
  discret_ = Teuchos::rcp(new DRT::Discretization(scatratimint_->Discretization()->Name(), com));

  // call complete without assigning degrees of freedom
  discret_->FillComplete(false, true, false);

  Teuchos::RCP<DRT::Discretization> scatradis = scatratimint_->Discretization();

  // create scatra elements if the scatra discretization is empty
  {
    // fill scatra discretization by cloning fluid discretization
    DRT::UTILS::CloneDiscretizationFromCondition<SCATRA::ScatraReactionCloneStrategy>(
        *scatradis, *discret_, "ScatraHeteroReactionSlave");

    // set implementation type of cloned scatra elements
    for (int i = 0; i < discret_->NumMyColElements(); ++i)
    {
      DRT::ELEMENTS::Transport* element =
          dynamic_cast<DRT::ELEMENTS::Transport*>(discret_->lColElement(i));
      if (element == nullptr) FOUR_C_THROW("Invalid element type!");

      if (element->Material()->MaterialType() == CORE::Materials::m_matlist_reactions)
        element->SetImplType(INPAR::SCATRA::impltype_advreac);
      else
        FOUR_C_THROW("Invalid material type for HeterogeneousReactionStrategy!");
    }  // loop over all column elements
  }

  {
    // build a dofset that merges the DOFs from both sides
    // convention: the order of the merged dofset will be
    //  _            _
    // | slave dofs   |
    // |              |
    // |_master dofs _|
    //
    // slave side is supposed to be the surface discretization
    //
    Teuchos::RCP<DRT::DofSetMergedWrapper> newdofset =
        Teuchos::rcp(new DRT::DofSetMergedWrapper(scatradis->GetDofSetProxy(), scatradis,
            "ScatraHeteroReactionMaster", "ScatraHeteroReactionSlave"));

    // assign the dofset to the reaction discretization
    discret_->ReplaceDofSet(newdofset, false);

    // add all secondary dofsets as proxies
    for (int ndofset = 1; ndofset < scatratimint_->Discretization()->NumDofSets(); ++ndofset)
    {
      Teuchos::RCP<DRT::DofSetGIDBasedWrapper> gidmatchingdofset =
          Teuchos::rcp(new DRT::DofSetGIDBasedWrapper(scatratimint_->Discretization(),
              scatratimint_->Discretization()->GetDofSetProxy(ndofset)));
      discret_->AddDofSet(gidmatchingdofset);
    }

    // done. Rebuild all maps and boundary condition geometries
    discret_->FillComplete(true, true, true);

    if (com->MyPID() == 0 and com->NumProc() > 1)
      std::cout << "parallel distribution of auxiliary discr. with standard ghosting" << std::endl;
    CORE::REBALANCE::UTILS::PrintParallelDistribution(*discret_);
  }

  SetIsSetup(true);
  return;
}


/*----------------------------------------------------------------------*
 | setup meshtying objects                                  vuong 06/16 |
 *----------------------------------------------------------------------*/
void SCATRA::HeterogeneousReactionStrategy::InitMeshtying()
{
  SetIsSetup(false);

  // call Init() of base class
  SCATRA::MeshtyingStrategyStd::InitMeshtying();

  SetIsInit(true);
  return;
}


/*----------------------------------------------------------------------*
 | Evaluate conditioned elements                            rauch 08/16 |
 *----------------------------------------------------------------------*/
void SCATRA::HeterogeneousReactionStrategy::EvaluateCondition(Teuchos::ParameterList& params,
    Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix1,
    Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix2,
    Teuchos::RCP<Epetra_Vector> systemvector1, Teuchos::RCP<Epetra_Vector> systemvector2,
    Teuchos::RCP<Epetra_Vector> systemvector3, const std::string& condstring, const int condid)
{
  CheckIsInit();
  CheckIsSetup();

  // Call EvaluateCondition on auxiliary discretization.
  // This condition has all dofs, both from the volume-
  // bound scalars and from the surface-bound scalars.
  discret_->EvaluateCondition(params, systemmatrix1, systemmatrix2, systemvector1, systemvector2,
      systemvector3, condstring, condid);

  return;
}


/*----------------------------------------------------------------------*
 | Set state on auxiliary discretization                    rauch 12/16 |
 *----------------------------------------------------------------------*/
void SCATRA::HeterogeneousReactionStrategy::SetState(
    unsigned nds, const std::string& name, Teuchos::RCP<const Epetra_Vector> state)
{
  discret_->SetState(nds, name, state);
  return;
}


/*----------------------------------------------------------------------*
 | sanity check for some assumptions and conventions        rauch 06/17 |
 *----------------------------------------------------------------------*/
void SCATRA::HeterogeneousReactionStrategy::HeterogeneousReactionSanityCheck()
{
  bool valid_slave = false;

  const Epetra_Comm& com = scatratimint_->Discretization()->Comm();

  if (com.MyPID() == 0) std::cout << " Sanity check for HeterogeneousReactionStrategy ...";

  DRT::Condition* slave_cond =
      scatratimint_->Discretization()->GetCondition("ScatraHeteroReactionSlave");

  const Epetra_Map* element_row_map = scatratimint_->Discretization()->ElementRowMap();

  // loop over row elements
  for (int lid = 0; lid < element_row_map->NumMyElements(); lid++)
  {
    const int gid = element_row_map->GID(lid);

    DRT::Element* ele = scatratimint_->Discretization()->gElement(gid);
    DRT::Node** nodes = ele->Nodes();
    if (ele->Shape() == CORE::FE::CellType::quad4 or ele->Shape() == CORE::FE::CellType::tri3)
    {
      for (int node = 0; node < ele->NumNode(); node++)
      {
        const int node_gid = nodes[node]->Id();

        if (not slave_cond->ContainsNode(node_gid))
        {
          FOUR_C_THROW(
              "Surface discretization for membrane transport is "
              "supposed to wear ScatraHeteroReactionSlave condition!");
        }
        else
        {
          valid_slave = true;
          break;
        }
      }  // loop over nodes of row ele

      if (valid_slave) break;
    }  // if surface transport element

    else if (ele->Shape() == CORE::FE::CellType::hex8 or ele->Shape() == CORE::FE::CellType::tet4)
    {
      // no check so far
    }  // if volume transport element

    else
    {
      FOUR_C_THROW(
          "please implement check for new combination of volume transport "
          "- surface transport elements.");
    }

  }  // loop over row elements


  com.Barrier();
  if (com.MyPID() == 0) std::cout << " Passed." << std::endl;

  return;
}

FOUR_C_NAMESPACE_CLOSE
