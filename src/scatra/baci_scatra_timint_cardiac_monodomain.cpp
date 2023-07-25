/*----------------------------------------------------------------------*/
/*! \file

\brief scatra time integration for cardiac monodomain

\level 2


 *------------------------------------------------------------------------------------------------*/

#include "baci_scatra_timint_cardiac_monodomain.H"

#include "baci_mat_material.H"
#include "baci_mat_list.H"

#include "baci_lib_globalproblem.H"
#include "baci_lib_utils_parameter_list.H"

#include "baci_scatra_ele_action.H"
#include "baci_nurbs_discret.H"

#include "baci_io.H"
#include "baci_io_control.H"
#include "baci_linalg_utils_sparse_algebra_create.H"
#include "baci_linear_solver_method_linalg.H"
#include "baci_linalg_krylov_projector.H"

/*----------------------------------------------------------------------*
 | constructor                                              ehrl  01/14 |
 *----------------------------------------------------------------------*/
SCATRA::TimIntCardiacMonodomain::TimIntCardiacMonodomain(Teuchos::RCP<DRT::Discretization> dis,
    Teuchos::RCP<CORE::LINALG::Solver> solver, Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<Teuchos::ParameterList> sctratimintparams,
    Teuchos::RCP<Teuchos::ParameterList> extraparams, Teuchos::RCP<IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(dis, solver, sctratimintparams, extraparams, output),
      // Initialization of electrophysiology variables
      activation_time_np_(Teuchos::null),
      activation_threshold_(0.0),
      nb_max_mat_int_state_vars_(0),
      material_internal_state_np_(Teuchos::null),
      material_internal_state_np_component_(Teuchos::null),
      nb_max_mat_ionic_currents_(0),
      material_ionic_currents_np_(Teuchos::null),
      material_ionic_currents_np_component_(Teuchos::null),
      ep_params_(params)
{
  return;
}


void SCATRA::TimIntCardiacMonodomain::Setup()
{
  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices: local <-> global dof numbering
  // -------------------------------------------------------------------
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // Activation time at time n+1
  activation_time_np_ = CORE::LINALG::CreateVector(*dofrowmap, true);
  activation_threshold_ = ep_params_->get<double>("ACTTHRES");
  // Assumes that maximum nb_max_mat_int_state_vars_ internal state variables will be written
  nb_max_mat_int_state_vars_ = ep_params_->get<int>(
      "WRITEMAXINTSTATE");  // number of maximal internal state variables to be postprocessed
  if (nb_max_mat_int_state_vars_)
  {
    material_internal_state_np_ = Teuchos::rcp(
        new Epetra_MultiVector(*(discret_->ElementRowMap()), nb_max_mat_int_state_vars_, true));
    material_internal_state_np_component_ =
        CORE::LINALG::CreateVector(*(discret_->ElementRowMap()), true);
  }
  // Assumes that maximum nb_max_mat_ionic_currents_ ionic_currents variables will be written
  nb_max_mat_ionic_currents_ = ep_params_->get<int>(
      "WRITEMAXIONICCURRENTS");  // number of maximal internal state variables to be postprocessed
  if (nb_max_mat_ionic_currents_)
  {
    material_ionic_currents_np_ = Teuchos::rcp(
        new Epetra_MultiVector(*(discret_->ElementRowMap()), nb_max_mat_ionic_currents_, true));
    material_ionic_currents_np_component_ =
        CORE::LINALG::CreateVector(*(discret_->ElementRowMap()), true);
  }

  // create dofmap for output writing
  std::vector<int> globaldof;
  for (int i = 0; i < discret_->NodeRowMap()->NumMyElements(); ++i)
    globaldof.push_back(discret_->NodeRowMap()->GID(i));
  // create dof map (one dof for each node)
  dofmap_ = Teuchos::rcp(
      new Epetra_Map(-1, (int)globaldof.size(), globaldof.data(), 0, discret_->Comm()));


  return;
}


/*----------------------------------------------------------------------*
 |  write current state to BINIO                             gjb   08/08|
 *----------------------------------------------------------------------*/
void SCATRA::TimIntCardiacMonodomain::OutputState()
{
  // Call function from base class
  SCATRA::ScaTraTimIntImpl::OutputState();

  // electrophysiology

  // Compute and write activation time
  if (activation_time_np_ != Teuchos::null)
  {
    for (int k = 0; k < phinp_->MyLength(); k++)
    {
      if ((*phinp_)[k] >= activation_threshold_ && (*activation_time_np_)[k] <= dta_ * 0.9)
        (*activation_time_np_)[k] = time_;
    }
    output_->WriteVector("activation_time_np", activation_time_np_);
  }

  // Recover internal state of the material (for electrophysiology)
  if (material_internal_state_np_ != Teuchos::null and nb_max_mat_int_state_vars_)
  {
    Teuchos::ParameterList params;
    DRT::UTILS::AddEnumClassToParameterList<SCATRA::Action>(
        "action", SCATRA::Action::get_material_internal_state, params);
    params.set<Teuchos::RCP<Epetra_MultiVector>>("material_internal_state",
        material_internal_state_np_);  // Probably do it once at the beginning
    discret_->Evaluate(params);
    material_internal_state_np_ =
        params.get<Teuchos::RCP<Epetra_MultiVector>>("material_internal_state");

    for (int k = 0; k < material_internal_state_np_->NumVectors(); ++k)
    {
      std::ostringstream temp;
      temp << k + 1;
      material_internal_state_np_component_ =
          Teuchos::rcp((*material_internal_state_np_)(k), false);
      output_->WriteVector(
          "mat_int_state_" + temp.str(), material_internal_state_np_component_, IO::elementvector);
    }
  }

  // Recover internal ionic currents of the material (for electrophysiology)
  if (material_ionic_currents_np_ != Teuchos::null and nb_max_mat_ionic_currents_)
  {
    Teuchos::ParameterList params;
    DRT::UTILS::AddEnumClassToParameterList<SCATRA::Action>(
        "action", SCATRA::Action::get_material_ionic_currents, params);
    params.set<Teuchos::RCP<Epetra_MultiVector>>("material_ionic_currents",
        material_ionic_currents_np_);  // Probably do it once at the beginning
    discret_->Evaluate(params);
    material_internal_state_np_ =
        params.get<Teuchos::RCP<Epetra_MultiVector>>("material_ionic_currents");

    for (int k = 0; k < material_ionic_currents_np_->NumVectors(); ++k)
    {
      std::ostringstream temp;
      temp << k + 1;
      material_ionic_currents_np_component_ =
          Teuchos::rcp((*material_ionic_currents_np_)(k), false);
      output_->WriteVector("mat_ionic_currents_" + temp.str(),
          material_ionic_currents_np_component_, IO::elementvector);
    }
  }
  return;
}  // TimIntCardiacMonodomain::OutputState


/*----------------------------------------------------------------------*
 | time update of time-dependent materials                    gjb 07/12 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntCardiacMonodomain::ElementMaterialTimeUpdate()
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // action for elements
  DRT::UTILS::AddEnumClassToParameterList<SCATRA::Action>(
      "action", SCATRA::Action::time_update_material, p);
  // further required parameter
  p.set<int>("time-step length", dta_);

  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("phinp", phinp_);

  // go to elements
  discret_->Evaluate(p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
  discret_->ClearState();
  return;
}


/*----------------------------------------------------------------------*
 | set ep-specific element parameters                    heormann 06/16 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntCardiacMonodomain::SetElementSpecificScaTraParameters(
    Teuchos::ParameterList& eleparams) const
{
  // safety check
  if (DRT::INPUT::IntegralValue<int>(*params_, "SEMIIMPLICIT"))
    if (INPAR::SCATRA::timeint_gen_alpha ==
        DRT::INPUT::IntegralValue<INPAR::SCATRA::TimeIntegrationScheme>(*params_, "TIMEINTEGR"))
      if (params_->get<double>("ALPHA_M") < 1.0 or params_->get<double>("ALPHA_F") < 1.0)
        dserror(
            "EP calculation with semiimplicit timestepping scheme only tested for gen-alpha with "
            "alpha_f = alpha_m = 1!");

  eleparams.set<bool>("semiimplicit", DRT::INPUT::IntegralValue<int>(*params_, "SEMIIMPLICIT"));

  return;
}
