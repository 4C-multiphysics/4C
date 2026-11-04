// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_solid_superposition.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::PAR::SolidSuperposition::SolidSuperposition(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      density_(matdata.parameters.get<double>("DENS")),
      matids_(matdata.parameters.get<std::vector<int>>("MATIDS"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::SolidSuperposition::create_material()
{
  return std::make_shared<Mat::SolidSuperposition>(this);
}

Mat::SolidSuperpositionType Mat::SolidSuperpositionType::instance_;

Core::Communication::ParObject* Mat::SolidSuperpositionType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* mat_solid_superposition = new Mat::SolidSuperposition();
  mat_solid_superposition->unpack(buffer);

  return mat_solid_superposition;
}

Mat::SolidSuperposition::SolidSuperposition() : params_(nullptr), materials_(0) {}

Mat::SolidSuperposition::SolidSuperposition(Mat::PAR::SolidSuperposition* params)
    : params_(params), materials_(0)
{
  // create the materials from the given material ids
  for (const auto& matid : params_->matids_)
  {
    auto mat = std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(matid));
    if (!mat) FOUR_C_THROW("Failed to allocate material for matid {}", matid);

    materials_.push_back(std::move(mat));
  }
}

void Mat::SolidSuperposition::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // pack material id
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // pack all materials
  Core::Communication::PotentiallyUnusedBufferScope materials_scope{data};

  if (params_ != nullptr)  // materials are not accessible during post processing
  {
    for (const auto& m : materials_) m->pack(data);
  }
}

void Mat::SolidSuperposition::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  params_ = nullptr;
  materials_.clear();

  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = dynamic_cast<Mat::PAR::SolidSuperposition*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

    // extract materials
    Core::Communication::PotentiallyUnusedBufferScope materials_scope{buffer};

    if (params_ != nullptr)  // materials are not accessible during post processing
    {
      // recreate the materials from the given material ids
      for (const auto& matid : params_->matids_)
      {
        auto mat = std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(matid));
        if (!mat) FOUR_C_THROW("Failed to allocate material for matid {}", matid);

        materials_.push_back(std::move(mat));
      }

      // unpack the materials
      for (const auto& m : materials_) m->unpack(buffer);
    }
  }
}

void Mat::SolidSuperposition::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  for (const auto& m : materials_) m->setup(numgp, fibers, coord_system);
}

void Mat::SolidSuperposition::post_setup(const Teuchos::ParameterList& params, const int eleGID)
{
  for (const auto& m : materials_) m->post_setup(params, eleGID);
}

void Mat::SolidSuperposition::update()
{
  for (const auto& m : materials_) m->update();
}

void Mat::SolidSuperposition::update(const Core::LinAlg::Tensor<double, 3, 3>& defgrd, int gp,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context, int eleGID)
{
  for (const auto& m : materials_) m->update(defgrd, gp, params, context, eleGID);
}

void Mat::SolidSuperposition::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  stress = {};
  cmat = {};

  Core::LinAlg::SymmetricTensor<double, 3, 3> s_tmp{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> c_tmp{};

  // evaluate stress and material tangent for all materials
  for (const auto& m : materials_)
  {
    s_tmp = {};
    c_tmp = {};

    m->evaluate(defgrad, glstrain, params, context, s_tmp, c_tmp, gp, eleGID);

    // sum up the material contributions
    stress += s_tmp;
    cmat += c_tmp;
  }
}

void Mat::SolidSuperposition::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  for (const auto& m : materials_) m->register_output_data_names(names_and_size);
}

bool Mat::SolidSuperposition::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  bool data_was_set = false;
  for (const auto& m : materials_)
  {
    if (m->evaluate_output_data(name, data))
    {
      data_was_set = true;
    }
  }
  return data_was_set;
}

FOUR_C_NAMESPACE_CLOSE