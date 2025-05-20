// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_parameter_elch.hpp"

#include "4C_utils_exceptions.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | singleton access method                                   fang 02/15 |
 *----------------------------------------------------------------------*/
Discret::Elements::ScaTraEleParameterElch* Discret::Elements::ScaTraEleParameterElch::instance(
    const std::string& disname  //!< name of discretization
)
{
  static auto singleton_map =
      Core::Utils::make_singleton_map<std::string>([](const std::string& disname)
          { return std::unique_ptr<ScaTraEleParameterElch>(new ScaTraEleParameterElch(disname)); });

  return singleton_map[disname].instance(Core::Utils::SingletonAction::create, disname);
}


/*----------------------------------------------------------------------*
 | protected constructor for singletons                      fang 02/15 |
 *----------------------------------------------------------------------*/
Discret::Elements::ScaTraEleParameterElch::ScaTraEleParameterElch(
    const std::string& disname  //!< name of discretization
    )
    : boundaryfluxcoupling_(true),
      equpot_(ElCh::equpot_undefined),
      faraday_(0.0),
      gas_constant_(0.0),
      epsilon_(1e-4),
      frt_(0.0),
      temperature_(0.0)
{
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraEleParameterElch::set_parameters(Teuchos::ParameterList& parameters)
{
  // coupling of lithium-ion flux density and electric current density at Dirichlet and Neumann
  // boundaries
  boundaryfluxcoupling_ = parameters.get<bool>("boundaryfluxcoupling");

  // type of closing equation for electric potential
  equpot_ = Teuchos::getIntegralValue<ElCh::EquPot>(parameters, "equpot");
  if (equpot_ == ElCh::equpot_undefined)
    FOUR_C_THROW("Invalid type of closing equation for electric potential!");

  // get parameters
  faraday_ = parameters.get<double>("faraday", -1.0);
  gas_constant_ = parameters.get<double>("gas_constant", -1.0);
  frt_ = parameters.get<double>("frt", -1.0);
  temperature_ = parameters.get<double>("temperature", -1.0);

  // safety checks
  if (frt_ <= 0.0) FOUR_C_THROW("Factor F/RT is non-positive!");
  if (faraday_ <= 0.0) FOUR_C_THROW("Faraday constant is non-positive!");
  if (gas_constant_ <= 0.0) FOUR_C_THROW("(universal) gas constant is non-positive!");
  if (temperature_ < 0.0) FOUR_C_THROW("temperature is non-positive!");
}

FOUR_C_NAMESPACE_CLOSE
