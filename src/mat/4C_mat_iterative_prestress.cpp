// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_mat_iterative_prestress.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_tensor_svd.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mixture_rule.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ENull.hpp>
#include <Teuchos_ParameterList.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::shared_ptr<Mat::So3Material> create_child_material(const int mat_id)
  {
    auto material = Mat::factory(mat_id);

    auto child_material3d = std::dynamic_pointer_cast<Mat::So3Material>(material);


    FOUR_C_ASSERT_ALWAYS(child_material3d, "The child material is not a 3D solid material.");

    return child_material3d;
  }

  Core::LinAlg::Tensor<double, 3, 3> get_elastic_deformation_gradient(
      const Core::LinAlg::Tensor<double, 3, 3>& deformation_gradient,
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& prestretch_tensor)
  {
    return deformation_gradient * prestretch_tensor;
  }

  Core::LinAlg::SymmetricTensor<double, 3, 3> get_green_lagrange_strain(
      const Core::LinAlg::Tensor<double, 3, 3>& deformation_gradient)
  {
    return 0.5 * (Core::LinAlg::assume_symmetry(
                      Core::LinAlg::transpose(deformation_gradient) * deformation_gradient) -
                     Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
  }

  Core::LinAlg::SymmetricTensor<double, 3, 3> compute_updated_prestretch_tensor(
      const Core::LinAlg::Tensor<double, 3, 3>& deformation_gradient,
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& prestretch_tensor)
  {
    Core::LinAlg::Tensor<double, 3, 3> pre_deformation_gradient =
        deformation_gradient * prestretch_tensor;

    // Singular value decomposition of F = RU
    const auto [Q, S, VT] = Core::LinAlg::svd(pre_deformation_gradient);

    // Compute stretch tensor G = U = V * S * VT
    return Core::LinAlg::assume_symmetry(
        Core::LinAlg::transpose(VT) * Core::LinAlg::TensorGenerators::diagonal(S) * VT);
  }
}  // namespace

Mat::PAR::IterativePrestressMaterial::IterativePrestressMaterial(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Core::Mat::PAR::Parameter(matdata),
      mat_id_(matdata.parameters.get<int>("MATID")),
      is_prestress_active_(matdata.parameters.get<bool>("ACTIVE"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::IterativePrestressMaterial::create_material()
{
  return std::make_shared<Mat::IterativePrestressMaterial>(this);
}

Mat::IterativePrestressMaterialType Mat::IterativePrestressMaterialType::instance_;

Core::Communication::ParObject* Mat::IterativePrestressMaterialType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* mix_elhy = new Mat::IterativePrestressMaterial();
  mix_elhy->unpack(buffer);

  return mix_elhy;
}

Mat::IterativePrestressMaterial::IterativePrestressMaterial(
    Mat::PAR::IterativePrestressMaterial* params)
    : params_(params)
{
  // Create child material
  child_material_ = create_child_material(params_->mat_id_);
}

void Mat::IterativePrestressMaterial::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);


  add_to_pack(data, prestretch_tensor_);

  int matid = -1;
  if (params_ != nullptr) matid = params_->id();
  add_to_pack(data, matid);

  // pack data of the solid material
  child_material_->pack(data);
}


void Mat::IterativePrestressMaterial::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  extract_from_pack(buffer, prestretch_tensor_);

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

      FOUR_C_ASSERT_ALWAYS(mat->type() == material_type(),
          "Type of parameter material {} does not fit to calling type {}", mat->type(),
          material_type());


      params_ = dynamic_cast<Mat::PAR::IterativePrestressMaterial*>(mat);
    }
  }

  if (params_ != nullptr)
  {
    child_material_ = create_child_material(params_->mat_id_);
    child_material_->unpack(buffer);
  }
}

void Mat::IterativePrestressMaterial::post_setup(const Teuchos::ParameterList& params, int eleGID)
{
  child_material_->post_setup(params, eleGID);
}

void Mat::IterativePrestressMaterial::update() { child_material_->update(); }

void Mat::IterativePrestressMaterial::update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd,
    const int gp, const Teuchos::ParameterList& params, const EvaluationContext& context,
    const int eleGID)
{
  if (params_->is_prestress_active_)
  {
    prestretch_tensor_[gp] = compute_updated_prestretch_tensor(defgrd, prestretch_tensor_[gp]);
  }

  Core::LinAlg::Tensor<double, 3, 3> elastic_deformation_gradient =
      get_elastic_deformation_gradient(defgrd, prestretch_tensor_[gp]);

  child_material_->update(elastic_deformation_gradient, gp, params, context, eleGID);
}

void Mat::IterativePrestressMaterial::setup(const int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  prestretch_tensor_.resize(numgp, Core::LinAlg::TensorGenerators::identity<double, 3, 3>);

  child_material_->setup(numgp, fibers, coord_system);
}

void Mat::IterativePrestressMaterial::vis_names(std::map<std::string, int>& names) const
{
  return child_material_->vis_names(names);
}

bool Mat::IterativePrestressMaterial::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  return child_material_->vis_data(name, data, numgp);
}

void Mat::IterativePrestressMaterial::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  child_material_->register_output_data_names(names_and_size);
}

bool Mat::IterativePrestressMaterial::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  return child_material_->evaluate_output_data(name, data);
}

double Mat::IterativePrestressMaterial::strain_energy(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const EvaluationContext& context,
    int gp, int eleGID) const
{
  FOUR_C_THROW("Strain energy computation is currently not implemented for prestressing materials");
}

void Mat::IterativePrestressMaterial::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  Core::LinAlg::Tensor<double, 3, 3> elastic_deformation_gradient =
      get_elastic_deformation_gradient(*defgrad, prestretch_tensor_[gp]);

  // Evaluate green Lagrange strain
  const Core::LinAlg::SymmetricTensor<double, 3, 3> elastic_gl_strain =
      get_green_lagrange_strain(elastic_deformation_gradient);

  Core::LinAlg::SymmetricTensor<double, 3, 3> elastic_stress{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> elastic_cmat{};


  // evaluate child material
  child_material_->evaluate(&elastic_deformation_gradient, elastic_gl_strain, params, context,
      elastic_stress, elastic_cmat, gp, eleGID);

  // push-forward operation
  const double det_pre = Core::LinAlg::det(prestretch_tensor_[gp]);
  stress = 1.0 / det_pre *
           Core::LinAlg::assume_symmetry(prestretch_tensor_[gp] * elastic_stress *
                                         Core::LinAlg::transpose(prestretch_tensor_[gp]));
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);
  cmat_view += Mat::push_forward_four_tensor(det_pre,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(prestretch_tensor_[gp])),
      Core::LinAlg::make_stress_like_voigt_view(elastic_cmat));
}


FOUR_C_NAMESPACE_CLOSE