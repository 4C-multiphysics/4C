/*---------------------------------------------------------------------*/
/*! \file
\brief Deprecated wrapper for the mesh-tying and contact managers.

\level 2


*/
/*---------------------------------------------------------------------*/

#include "4C_contact_meshtying_contact_bridge.hpp"

#include "4C_contact_manager.hpp"
#include "4C_contact_meshtying_manager.hpp"
#include "4C_discretization_condition.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_mortar_manager_base.hpp"
#include "4C_mortar_strategy_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 06/14|
 *----------------------------------------------------------------------*/
CONTACT::MeshtyingContactBridge::MeshtyingContactBridge(DRT::Discretization& dis,
    std::vector<CORE::Conditions::Condition*>& meshtyingConditions,
    std::vector<CORE::Conditions::Condition*>& contactConditions, double timeIntegrationMidPoint)
    : cman_(Teuchos::null), mtman_(Teuchos::null)
{
  bool onlymeshtying = false;
  bool onlycontact = false;
  bool meshtyingandcontact = false;

  // check for case
  if (meshtyingConditions.size() != 0 and contactConditions.size() != 0) meshtyingandcontact = true;

  if (meshtyingConditions.size() != 0 and contactConditions.size() == 0) onlymeshtying = true;

  if (meshtyingConditions.size() == 0 and contactConditions.size() != 0) onlycontact = true;

  // create meshtying and contact manager
  if (onlymeshtying)
  {
    mtman_ = Teuchos::rcp(new CONTACT::MtManager(dis, timeIntegrationMidPoint));
  }
  else if (onlycontact)
  {
    cman_ = Teuchos::rcp(new CONTACT::Manager(dis, timeIntegrationMidPoint));
  }
  else if (meshtyingandcontact)
  {
    mtman_ = Teuchos::rcp(new CONTACT::MtManager(dis, timeIntegrationMidPoint));
    cman_ = Teuchos::rcp(new CONTACT::Manager(dis, timeIntegrationMidPoint));
  }

  // Sanity check for writing output for each interface
  {
    const bool writeInterfaceOutput =
        CORE::UTILS::IntegralValue<bool>(GetStrategy().Params(), "OUTPUT_INTERFACES");

    if (writeInterfaceOutput && HaveContact() && ContactManager()->GetStrategy().Friction())
      FOUR_C_THROW(
          "Output for each interface does not work yet, if friction is enabled. Switch off the "
          "interface-based output in the input file (or implement/fix it for frictional contact "
          "problems.");
  }

  return;
}

/*----------------------------------------------------------------------*
 |  StoreDirichletStatus                                     farah 06/14|
 *----------------------------------------------------------------------*/
void CONTACT::MeshtyingContactBridge::StoreDirichletStatus(
    Teuchos::RCP<CORE::LINALG::MapExtractor> dbcmaps)
{
  if (HaveMeshtying()) MtManager()->GetStrategy().StoreDirichletStatus(dbcmaps);
  if (HaveContact()) ContactManager()->GetStrategy().StoreDirichletStatus(dbcmaps);

  return;
}

/*----------------------------------------------------------------------*
 |  Set displacement state                                   farah 06/14|
 *----------------------------------------------------------------------*/
void CONTACT::MeshtyingContactBridge::SetState(Teuchos::RCP<Epetra_Vector> zeros)
{
  if (HaveMeshtying()) MtManager()->GetStrategy().SetState(MORTAR::state_new_displacement, *zeros);
  if (HaveContact())
    ContactManager()->GetStrategy().SetState(MORTAR::state_new_displacement, *zeros);

  return;
}

/*----------------------------------------------------------------------*
 |  Get Strategy                                             farah 06/14|
 *----------------------------------------------------------------------*/
MORTAR::StrategyBase& CONTACT::MeshtyingContactBridge::GetStrategy() const
{
  // if contact is involved use contact strategy!
  // contact conditions/strategies are dominating the algorithm!
  if (HaveMeshtying() and !HaveContact())
    return MtManager()->GetStrategy();
  else
    return ContactManager()->GetStrategy();
}

/*----------------------------------------------------------------------*
 |  PostprocessTractions                                     farah 06/14|
 *----------------------------------------------------------------------*/
void CONTACT::MeshtyingContactBridge::PostprocessQuantities(
    Teuchos::RCP<IO::DiscretizationWriter>& output)
{
  // contact
  if (HaveContact()) ContactManager()->PostprocessQuantities(*output);

  // meshtying
  if (HaveMeshtying()) MtManager()->PostprocessQuantities(*output);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MeshtyingContactBridge::PostprocessQuantitiesPerInterface(
    Teuchos::RCP<Teuchos::ParameterList> outputParams)
{
  // This is an optional feature, so we check if it has been enabled in the input file
  const bool writeInterfaceOutput =
      CORE::UTILS::IntegralValue<bool>(GetStrategy().Params(), "OUTPUT_INTERFACES");
  if (writeInterfaceOutput)
  {
    // contact
    if (HaveContact()) ContactManager()->PostprocessQuantitiesPerInterface(outputParams);

    // meshtying
    if (HaveMeshtying()) MtManager()->PostprocessQuantitiesPerInterface(outputParams);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Recover lagr. mult and slave displ                       farah 06/14|
 *----------------------------------------------------------------------*/
void CONTACT::MeshtyingContactBridge::Recover(Teuchos::RCP<Epetra_Vector> disi)
{
  // meshtying
  if (HaveMeshtying()) MtManager()->GetStrategy().Recover(disi);

  // contact
  if (HaveContact()) ContactManager()->GetStrategy().Recover(disi);

  return;
}

/*----------------------------------------------------------------------*
 |  Recover lagr. mult and slave displ                       farah 06/14|
 *----------------------------------------------------------------------*/
void CONTACT::MeshtyingContactBridge::ReadRestart(IO::DiscretizationReader& reader,
    Teuchos::RCP<Epetra_Vector> dis, Teuchos::RCP<Epetra_Vector> zero)
{
  // contact
  if (HaveContact()) ContactManager()->ReadRestart(reader, dis, zero);

  // meshtying
  if (HaveMeshtying()) MtManager()->ReadRestart(reader, dis, zero);

  return;
}

/*----------------------------------------------------------------------*
 |  Write restart                                            farah 06/14|
 *----------------------------------------------------------------------*/
void CONTACT::MeshtyingContactBridge::WriteRestart(
    Teuchos::RCP<IO::DiscretizationWriter>& output, bool forcedrestart)
{
  // contact
  if (HaveContact()) ContactManager()->WriteRestart(*output, forcedrestart);

  // meshtying
  if (HaveMeshtying()) MtManager()->WriteRestart(*output, forcedrestart);

  return;
}

/*----------------------------------------------------------------------*
 |  Write restart                                            farah 06/14|
 *----------------------------------------------------------------------*/
void CONTACT::MeshtyingContactBridge::Update(Teuchos::RCP<Epetra_Vector> dis)
{
  // contact
  if (HaveContact()) ContactManager()->GetStrategy().Update(dis);

  // meshtying
  if (HaveMeshtying()) MtManager()->GetStrategy().Update(dis);

  return;
}

/*----------------------------------------------------------------------*
 |  Write restart                                             popp 11/14|
 *----------------------------------------------------------------------*/
void CONTACT::MeshtyingContactBridge::VisualizeGmsh(const int istep, const int iter)
{
  // contact
  if (HaveContact()) ContactManager()->GetStrategy().VisualizeGmsh(istep, iter);

  // meshtying
  if (HaveMeshtying()) MtManager()->GetStrategy().VisualizeGmsh(istep, iter);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Comm& CONTACT::MeshtyingContactBridge::Comm() const
{
  if (cman_ != Teuchos::null) return cman_->Comm();

  if (mtman_ != Teuchos::null) return mtman_->Comm();

  FOUR_C_THROW("can't get comm()");
  return cman_->Comm();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<MORTAR::ManagerBase> CONTACT::MeshtyingContactBridge::ContactManager() const
{
  return cman_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<MORTAR::ManagerBase> CONTACT::MeshtyingContactBridge::MtManager() const
{
  return mtman_;
}

FOUR_C_NAMESPACE_CLOSE
