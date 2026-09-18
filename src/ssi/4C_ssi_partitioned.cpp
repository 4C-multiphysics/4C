// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_partitioned.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_contact_nitsche_strategy_ssi.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_ssi_str_model_evaluator_partitioned.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
SSI::SSIPart::SSIPart(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIBase(comm, globaltimeparams)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIPart::init(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
    const std::string& struct_disname, const std::string& scatra_disname, bool isAle)
{
  // call setup of base class
  SSI::SSIBase::init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);

  // safety check
  if (ssi_interface_meshtying() and Teuchos::getIntegralValue<Solid::PredictorType>(
                                        structparams, "PREDICT") != Solid::PredictorType::tangdis)
  {
    FOUR_C_THROW(
        "Must have TangDis predictor for structural field in partitioned scalar-structure "
        "interaction simulations involving scatra-scatra interface coupling! Otherwise, Dirichlet "
        "boundary conditions on master-side degrees of freedom are not transferred to slave-side "
        "degrees of freedom!");
  }

  if (is_scatra_manifold()) FOUR_C_THROW("Manifold not implemented for partitioned SSI");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIPart::setup()
{
  // call setup of base class
  SSI::SSIBase::setup();

  if (ssi_interface_contact() and !is_restart())
  {
    setup_contact_strategy();
    scatra_field()->set_nitsche_contact_strategy(nitsche_strategy_ssi());
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::SSIPart::setup_model_evaluator()
{
  // build and register ssi model evaluator
  std::shared_ptr<Solid::ModelEvaluator::Generic> ssi_model_ptr =
      std::make_shared<Solid::ModelEvaluator::PartitionedSSI>(
          Core::Utils::shared_ptr_from_ref(*this));
  structure_base_algorithm()->register_model_evaluator("Partitioned Coupling Model", ssi_model_ptr);

  if (is_s2i_kinetics_with_pseudo_contact()) set_modelevaluator_base_ssi(ssi_model_ptr);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SSIPart::read_restart(int restart)
{
  // call base class
  SSIBase::read_restart(restart);

  // do ssi contact specific tasks
  if (ssi_interface_contact())
  {
    setup_contact_strategy();
    scatra_field()->set_nitsche_contact_strategy(nitsche_strategy_ssi());
  }
}

FOUR_C_NAMESPACE_CLOSE
