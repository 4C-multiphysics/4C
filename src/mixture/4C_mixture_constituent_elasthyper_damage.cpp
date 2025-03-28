// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_elasthyper_damage.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mixture_elastin_membrane_prestress_strategy.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

// Constructor for the parameter class
Mixture::PAR::MixtureConstituentElastHyperDamage::MixtureConstituentElastHyperDamage(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureConstituentElastHyperBase(matdata),
      damage_function_id_(matdata.parameters.get<int>("DAMAGE_FUNCT"))
{
  // nothing to do here
}

// Create an instance of Mixture::MixtureConstituentElastHyper from the parameters
std::unique_ptr<Mixture::MixtureConstituent>
Mixture::PAR::MixtureConstituentElastHyperDamage::create_constituent(int id)
{
  return std::unique_ptr<Mixture::MixtureConstituentElastHyperDamage>(
      new Mixture::MixtureConstituentElastHyperDamage(this, id));
}

// Constructor of the constituent holding the material parameters
Mixture::MixtureConstituentElastHyperDamage::MixtureConstituentElastHyperDamage(
    Mixture::PAR::MixtureConstituentElastHyperDamage* params, int id)
    : MixtureConstituentElastHyperBase(params, id), params_(params)
{
  // nothing to do here
}

// Returns the material type
Core::Materials::MaterialType Mixture::MixtureConstituentElastHyperDamage::material_type() const
{
  return Core::Materials::mix_elasthyper_damage;
}

// Pack the constituent
void Mixture::MixtureConstituentElastHyperDamage::pack_constituent(
    Core::Communication::PackBuffer& data) const
{
  MixtureConstituentElastHyperBase::pack_constituent(data);

  add_to_pack(data, current_reference_growth_);
}

// Unpack the constituent
void Mixture::MixtureConstituentElastHyperDamage::unpack_constituent(
    Core::Communication::UnpackBuffer& buffer)
{
  MixtureConstituentElastHyperBase::unpack_constituent(buffer);

  extract_from_pack(buffer, current_reference_growth_);
}

// Reads the element from the input file
void Mixture::MixtureConstituentElastHyperDamage::read_element(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  MixtureConstituentElastHyperBase::read_element(numgp, container);

  current_reference_growth_.resize(numgp, 1.0);
}

// Updates all summands
void Mixture::MixtureConstituentElastHyperDamage::update(Core::LinAlg::Matrix<3, 3> const& defgrd,
    Teuchos::ParameterList& params, const int gp, const int eleGID)
{
  const auto& reference_coordinates = params.get<Core::LinAlg::Matrix<3, 1>>("gp_coords_ref");

  double totaltime = params.get<double>("total time", -1);
  if (totaltime < 0.0)
  {
    FOUR_C_THROW("Parameter 'total time' could not be read!");
  }

  current_reference_growth_[gp] =
      Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(params_->damage_function_id_)
          .evaluate(reference_coordinates.data(), totaltime, 0);

  MixtureConstituentElastHyperBase::update(defgrd, params, gp, eleGID);
}

double Mixture::MixtureConstituentElastHyperDamage::get_growth_scalar(int gp) const
{
  return current_reference_growth_[gp];
}

void Mixture::MixtureConstituentElastHyperDamage::evaluate(const Core::LinAlg::Matrix<3, 3>& F,
    const Core::LinAlg::Matrix<6, 1>& E_strain, Teuchos::ParameterList& params,
    Core::LinAlg::Matrix<6, 1>& S_stress, Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID)
{
  FOUR_C_THROW("This constituent does not support Evaluation without an elastic part.");
}

void Mixture::MixtureConstituentElastHyperDamage::evaluate_elastic_part(
    const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<3, 3>& iFextin,
    Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID)
{
  // Compute total inelastic deformation gradient
  static Core::LinAlg::Matrix<3, 3> iFin(Core::LinAlg::Initialization::uninitialized);
  iFin.multiply_nn(iFextin, prestretch_tensor(gp));

  // Evaluate 3D elastic part
  Mat::elast_hyper_evaluate_elastic_part(
      F, iFin, S_stress, cmat, summands(), summand_properties(), gp, eleGID);
}
FOUR_C_NAMESPACE_CLOSE
