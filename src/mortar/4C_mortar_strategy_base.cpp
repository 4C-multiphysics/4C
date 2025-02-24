// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_strategy_base.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mortar_defines.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mortar::StrategyDataContainer::StrategyDataContainer()
    : probdofs_(nullptr),
      probnodes_(nullptr),
      comm_(MPI_COMM_NULL),
      scontact_(),
      dim_(0),
      alphaf_(0.0),
      parredist_(false),
      maxdof_(0),
      systype_(Inpar::CONTACT::system_none),
      dyntype_(Inpar::Solid::dyna_statics),
      dynparam_n_(0.0)
{
}

/*----------------------------------------------------------------------*
 | ctor (public)                                             popp 01/10 |
 *----------------------------------------------------------------------*/
Mortar::StrategyBase::StrategyBase(const std::shared_ptr<Mortar::StrategyDataContainer>& data_ptr,
    const Epetra_Map* dof_row_map, const Epetra_Map* NodeRowMap,
    const Teuchos::ParameterList& params, const int spatialDim, const MPI_Comm& comm,
    const double alphaf, const int maxdof)
    : probdofs_(data_ptr->prob_dofs_ptr()),
      probnodes_(data_ptr->prob_nodes_ptr()),
      comm_(data_ptr->comm_ptr()),
      scontact_(data_ptr->s_contact()),
      dim_(data_ptr->n_dim()),
      alphaf_(data_ptr->alpha_f()),
      parredist_(data_ptr->is_par_redist()),
      maxdof_(data_ptr->max_dof()),
      systype_(data_ptr->sys_type()),
      data_ptr_(data_ptr)
{
  // *** set data container variables
  data().prob_dofs_ptr() = std::make_shared<Epetra_Map>(*(dof_row_map));
  data().prob_nodes_ptr() = std::make_shared<Epetra_Map>(*(NodeRowMap));
  data().comm_ptr() = comm;
  data().s_contact() = params;
  data().n_dim() = spatialDim;
  data().alpha_f() = alphaf;
  data().max_dof() = maxdof;
  data().sys_type() = Teuchos::getIntegralValue<Inpar::CONTACT::SystemType>(scontact_, "SYSTEM");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::StrategyBase::set_time_integration_info(
    const double time_fac, const Inpar::Solid::DynamicType dyntype)
{
  // Get weight for contribution from last time step

  data().set_dyn_type(dyntype);
  switch (dyntype)
  {
    case Inpar::Solid::dyna_statics:
      data().set_dyn_parameter_n(0.0);
      break;
    case Inpar::Solid::dyna_genalpha:
    case Inpar::Solid::dyna_genalpha_liegroup:
    case Inpar::Solid::dyna_onesteptheta:
      data().set_dyn_parameter_n(time_fac);
      break;
    default:
      FOUR_C_THROW(
          "Unsupported time integration detected! [\"%s\"]", dynamic_type_string(dyntype).c_str());
      exit(EXIT_FAILURE);
  }

  // Check if we only want to compute the contact force at the time endpoint
  if (data().s_contact().get<bool>("CONTACTFORCE_ENDTIME"))
    alphaf_ = 0.0;
  else
  {
    alphaf_ = data().get_dyn_parameter_n();
  }
}

FOUR_C_NAMESPACE_CLOSE
