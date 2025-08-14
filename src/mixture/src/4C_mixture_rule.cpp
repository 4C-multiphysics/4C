// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_rule.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_rule_function.hpp"
#include "4C_mixture_rule_growthremodel.hpp"
#include "4C_mixture_rule_simple.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::Communication
{
  class PackBuffer;
}
// Constructor of the material parameters
Mixture::PAR::MixtureRule::MixtureRule(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata)
{
}

// Mixture rule factory generates the mixturerule parameters for a specific material id
Mixture::PAR::MixtureRule* Mixture::PAR::MixtureRule::factory(int matid)
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
    case Core::Materials::mix_rule_function:
    {
      return Mat::create_material_parameter_instance<Mixture::PAR::FunctionMixtureRule>(curmat);
    }
    case Core::Materials::mix_rule_simple:
    {
      return Mat::create_material_parameter_instance<Mixture::PAR::SimpleMixtureRule>(curmat);
    }
    case Core::Materials::mix_rule_growthremodel:
    {
      return Mat::create_material_parameter_instance<Mixture::PAR::GrowthRemodelMixtureRule>(
          curmat);
    }
    default:
      FOUR_C_THROW("The referenced material with id {} is not registered as a mixturerule!", matid);
  }
  return nullptr;
}

// Constructor with parameters
Mixture::MixtureRule::MixtureRule(Mixture::PAR::MixtureRule* params)
    : constituents_(nullptr), numgp_(0), has_read_element_(false), is_setup_(false)
{
}

// Pack the mixture rule
void Mixture::MixtureRule::pack_mixture_rule(Core::Communication::PackBuffer& data) const
{
  // Add number of Gauss points
  add_to_pack(data, numgp_);

  // Add flag whether it has already read the element
  add_to_pack(data, has_read_element_);

  // Add flags whether it is setup
  add_to_pack(data, is_setup_);
}

// Unpack the mixture rule
void Mixture::MixtureRule::unpack_mixture_rule(Core::Communication::UnpackBuffer& buffer)
{
  // Read initialized flag
  extract_from_pack(buffer, numgp_);

  // Read element read flag
  extract_from_pack(buffer, has_read_element_);

  // Read is setup flag
  extract_from_pack(buffer, is_setup_);
}

// reads the element definition and set up all quantities
void Mixture::MixtureRule::read_element(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  // Init must only be called once
  if (has_read_element_)
    FOUR_C_THROW("read_element() is called multiple times. Just once allowed.");
  numgp_ = numgp;

  has_read_element_ = true;
}

// Setup the mixture rule
void Mixture::MixtureRule::setup(const Teuchos::ParameterList& params, const int eleGID)
{
  // Setup must be called after read_element()
  if (!has_read_element_) FOUR_C_THROW("read_element() must be called before setup()!");

  // Setup must only be called once
  if (is_setup_) FOUR_C_THROW("setup() is called multiple times. Just once allowed.");
  is_setup_ = true;
}
FOUR_C_NAMESPACE_CLOSE
