// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_solidmaterial.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_mixture.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"



FOUR_C_NAMESPACE_OPEN

// Constructor for the parameter class
Mixture::PAR::MixtureConstituentSolidMaterial::MixtureConstituentSolidMaterial(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureConstituent(matdata), matid_(matdata.parameters.get<int>("MATID"))
{
}

// Create an instance of Mixture::MixtureConstituentSolidMaterial from the parameters
std::unique_ptr<Mixture::MixtureConstituent>
Mixture::PAR::MixtureConstituentSolidMaterial::create_constituent(int id)
{
  return std::unique_ptr<Mixture::MixtureConstituentSolidMaterial>(
      new Mixture::MixtureConstituentSolidMaterial(this, id));
}

// Constructor of the constituent holding the material parameters
Mixture::MixtureConstituentSolidMaterial::MixtureConstituentSolidMaterial(
    Mixture::PAR::MixtureConstituentSolidMaterial* params, int id)
    : MixtureConstituent(params, id), params_(params), material_()
{
  // take the matid (i.e. here the id of the solid material), read the type and
  // create the corresponding material
  auto mat = Mat::factory(params_->matid_);

  // cast to an So3Material
  material_ = std::dynamic_pointer_cast<Mat::So3Material>(mat);

  // ensure cast was successful
  if (!(mat))
    FOUR_C_THROW(
        "The solid material constituent with ID {} needs to be an So3Material.", params_->matid_);

  if (material_->density() - 1.0 > 1e-16)
    FOUR_C_THROW(
        "Please set the density of the solid material constituent with ID {} to 1.0 and prescribe "
        "a combined density for the entire mixture material.",
        material_->parameter()->id());
}

Core::Materials::MaterialType Mixture::MixtureConstituentSolidMaterial::material_type() const
{
  return Core::Materials::mix_solid_material;
}

void Mixture::MixtureConstituentSolidMaterial::pack_constituent(
    Core::Communication::PackBuffer& data) const
{
  // pack constituent data
  MixtureConstituent::pack_constituent(data);

  // add the matid of the Mixture_SolidMaterial
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // pack data of the solid material

  Core::Communication::PotentiallyUnusedBufferScope solid_scope{data};
  if (params_ != nullptr)
  {
    material_->pack(data);
  }
}

void Mixture::MixtureConstituentSolidMaterial::unpack_constituent(
    Core::Communication::UnpackBuffer& buffer)
{
  // unpack constituent data
  MixtureConstituent::unpack_constituent(buffer);

  // make sure we have a pristine material
  params_ = nullptr;
  material_ = nullptr;

  // extract the matid of the Mixture_SolidMaterial
  int matid;
  extract_from_pack(buffer, matid);

  // recover the params_ of the Mixture_SolidMaterial
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const unsigned int probinst =
          Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
      {
        params_ = dynamic_cast<Mixture::PAR::MixtureConstituentSolidMaterial*>(mat);
      }
      else
      {
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
      }
    }
  }

  // unpack the data of the solid material
  Core::Communication::PotentiallyUnusedBufferScope solid_scope{buffer};
  if (params_ != nullptr)
  {
    auto so3mat = Mat::factory(params_->matid_);
    material_ = std::dynamic_pointer_cast<Mat::So3Material>(so3mat);
    if (!(so3mat)) FOUR_C_THROW("Failed to allocate");

    material_->unpack(buffer);
  }
}

Core::Materials::MaterialType material_type() { return Core::Materials::mix_solid_material; }

void Mixture::MixtureConstituentSolidMaterial::read_element(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  MixtureConstituent::read_element(numgp, container);
  material_->setup(numgp, container);
}

void Mixture::MixtureConstituentSolidMaterial::update() { material_->update(); }

void Mixture::MixtureConstituentSolidMaterial::update(
    Core::LinAlg::Tensor<double, 3, 3> const& defgrd, const Teuchos::ParameterList& params,
    const int gp, const int eleGID)
{
  material_->update(defgrd, gp, params, eleGID);
}

void Mixture::MixtureConstituentSolidMaterial::evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  material_->evaluate(&F, E_strain, params, S_stress, cmat, gp, eleGID);
}

void Mixture::MixtureConstituentSolidMaterial::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  material_->register_output_data_names(names_and_size);
}

bool Mixture::MixtureConstituentSolidMaterial::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  return material_->evaluate_output_data(name, data);
}
FOUR_C_NAMESPACE_CLOSE
