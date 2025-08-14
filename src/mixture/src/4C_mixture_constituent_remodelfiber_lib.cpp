// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_remodelfiber_lib.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mixture_constituent_remodelfiber_material.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential_active.hpp"

FOUR_C_NAMESPACE_OPEN

[[nodiscard]] const Mixture::PAR::RemodelFiberMaterial<double>*
Mixture::PAR::fiber_material_factory(int matid)
{
  // for the sake of safety
  if (Global::Problem::instance()->materials() == nullptr)
  {
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");
  }

  // yet another safety check
  if (Global::Problem::instance()->materials()->num() == 0)
  {
    FOUR_C_THROW("List of materials in the global problem instance is empty.");
  }

  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

  // retrieve validated input line of material ID in question
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);

  switch (curmat->type())
  {
    case Core::Materials::mix_remodelfiber_material_exponential:
      return Mat::create_material_parameter_instance<
          Mixture::PAR::RemodelFiberMaterialExponential<double>>(curmat);
    case Core::Materials::mix_remodelfiber_material_exponential_active:
      return Mat::create_material_parameter_instance<
          Mixture::PAR::RemodelFiberMaterialExponentialActive<double>>(curmat);
    default:
      FOUR_C_THROW(
          "The referenced material with id {} is not registered as a remodel fiber material!",
          matid);
  }
}
FOUR_C_NAMESPACE_CLOSE
