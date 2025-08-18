// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent_elasthyper.hpp"
#include "4C_mixture_constituent_elasthyper_damage.hpp"
#include "4C_mixture_constituent_elasthyper_elastin_membrane.hpp"
#include "4C_mixture_constituent_full_constrained_mixture_fiber.hpp"
#include "4C_mixture_constituent_remodelfiber_expl.hpp"
#include "4C_mixture_constituent_remodelfiber_impl.hpp"
#include "4C_mixture_constituent_solidmaterial.hpp"

FOUR_C_NAMESPACE_OPEN

// Constructor of the mixture constituent parameters
Mixture::PAR::MixtureConstituent::MixtureConstituent(const Core::Mat::PAR::Parameter::Data& matdata)
    : Core::Mat::PAR::Parameter(matdata)
{
}

// Create an instance of the constituent from the parameters
std::shared_ptr<Core::Mat::Material> Mixture::PAR::MixtureConstituent::create_material()
{
  FOUR_C_THROW(
      "Cannot create mixture constituent from this method. Use CreateConstituent() instead.");
}

// Create the parameters of the constituents from the material number and the reference mass
// fraction
Mixture::PAR::MixtureConstituent* Mixture::PAR::MixtureConstituent::factory(int matnum)
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
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matnum);

  switch (curmat->type())
  {
    case Core::Materials::mix_elasthyper:
    {
      return dynamic_cast<Mixture::PAR::MixtureConstituent*>(curmat);
    }
    case Core::Materials::mix_elasthyper_damage:
    {
      return dynamic_cast<Mixture::PAR::MixtureConstituent*>(curmat);
    }
    case Core::Materials::mix_elasthyper_elastin_membrane:
    {
      return dynamic_cast<Mixture::PAR::MixtureConstituent*>(curmat);
    }
    case Core::Materials::mix_remodelfiber_expl:
    {
      return dynamic_cast<Mixture::PAR::MixtureConstituent*>(curmat);
    }
    case Core::Materials::mix_full_constrained_mixture_fiber:
    {
      return dynamic_cast<Mixture::PAR::MixtureConstituent*>(curmat);
    }
    case Core::Materials::mix_remodelfiber_impl:
    {
      return dynamic_cast<Mixture::PAR::MixtureConstituent*>(curmat);
    }
    case Core::Materials::mix_solid_material:
    {
      return dynamic_cast<Mixture::PAR::MixtureConstituent*>(curmat);
    }
    default:
      break;
  }
  FOUR_C_THROW(
      "The referenced material with id {} is not registered as a Mixture Constituent!", matnum);
}

Mixture::MixtureConstituent::MixtureConstituent(Mixture::PAR::MixtureConstituent* params, int id)
    : numgp_(0), has_read_element_(false), is_setup_(false), id_(id)
{
}

//! Init is called once at the beginning to setup the number of GPs and the Parameter List
void Mixture::MixtureConstituent::read_element(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  // Init must only be called once
  if (has_read_element_)
    FOUR_C_THROW("read_element() is called multiple times. Just once allowed.");
  has_read_element_ = true;
  numgp_ = numgp;
}

// Setup of the mixture constituents and all its subparts
void Mixture::MixtureConstituent::setup(const Teuchos::ParameterList& params, const int eleGID)
{
  // Setup must be called after init()
  if (!has_read_element_) FOUR_C_THROW("read_element() must be called before setup()");

  // Setup must only be called once
  if (is_setup_) FOUR_C_THROW("setup() is called multiple times. Just once allowed.");
  is_setup_ = true;
}

// Pack everything for distribution to other processors
void Mixture::MixtureConstituent::pack_constituent(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, numgp_);
  add_to_pack(data, has_read_element_);
  add_to_pack(data, is_setup_);
}

// Unpack base constituent data, need to be called by every derived class
void Mixture::MixtureConstituent::unpack_constituent(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  has_read_element_ = false;
  numgp_ = 0;
  is_setup_ = false;

  extract_from_pack(buffer, numgp_);

  extract_from_pack(buffer, has_read_element_);
  extract_from_pack(buffer, is_setup_);
}

void Mixture::MixtureConstituent::evaluate_elastic_part(const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::Tensor<double, 3, 3>& F_in, const Teuchos::ParameterList& params,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  FOUR_C_THROW("This constituent cannot handle an additional inelastic part.");
}

FOUR_C_NAMESPACE_CLOSE
