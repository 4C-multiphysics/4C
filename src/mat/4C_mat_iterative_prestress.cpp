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
#include "4C_linalg_utils_densematrix_svd.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mixture_rule.hpp"
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

  Core::LinAlg::Matrix<3, 3> get_elastic_deformation_gradient(
      const Core::LinAlg::Matrix<3, 3>& deformation_gradient,
      const Core::LinAlg::Matrix<3, 3>& prestretch_tensor)
  {
    Core::LinAlg::Matrix<3, 3> elastic_deformation_gradient(false);
    elastic_deformation_gradient.multiply_nn(deformation_gradient, prestretch_tensor);

    return elastic_deformation_gradient;
  }

  Core::LinAlg::Matrix<6, 1> get_green_lagrange_strain(
      const Core::LinAlg::Matrix<3, 3>& deformation_gradient)
  {
    Core::LinAlg::Matrix<3, 3> green_lagrange;
    Core::LinAlg::Matrix<6, 1> green_lagrange_strain;

    green_lagrange.multiply_tn(deformation_gradient, deformation_gradient);
    green_lagrange.update(-0.5, Core::LinAlg::identity_matrix<3>(), 0.5);

    Core::LinAlg::Voigt::Strains::matrix_to_vector(green_lagrange, green_lagrange_strain);

    return green_lagrange_strain;
  }

  Core::LinAlg::Matrix<3, 3> compute_updated_prestretch_tensor(
      const Core::LinAlg::Matrix<3, 3>& deformation_gradient,
      const Core::LinAlg::Matrix<3, 3>& prestretch_tensor)
  {
    Core::LinAlg::Matrix<3, 3> pre_deformation_gradinet;

    pre_deformation_gradinet.multiply_nn(deformation_gradient, prestretch_tensor);

    // Singular value decomposition of F = RU
    Core::LinAlg::Matrix<3, 3> Q(true);
    Core::LinAlg::Matrix<3, 3> S(true);
    Core::LinAlg::Matrix<3, 3> VT(true);

    Core::LinAlg::svd<3, 3>(pre_deformation_gradinet, Q, S, VT);

    // Compute stretch tensor G = U = V * S * VT
    Core::LinAlg::Matrix<3, 3> VS;
    Core::LinAlg::Matrix<3, 3> updated_prestretch_tensor;
    VS.multiply_tn(VT, S);
    updated_prestretch_tensor.multiply_nn(VS, VT);

    return updated_prestretch_tensor;
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
          "Type of parameter material %d does not fit to calling type %d", mat->type(),
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

void Mat::IterativePrestressMaterial::post_setup(Teuchos::ParameterList& params, int eleGID)
{
  child_material_->post_setup(params, eleGID);
}

void Mat::IterativePrestressMaterial::update() { child_material_->update(); }

void Mat::IterativePrestressMaterial::update(Core::LinAlg::Matrix<3, 3> const& defgrd, const int gp,
    Teuchos::ParameterList& params, const int eleGID)
{
  if (params_->is_prestress_active_)
  {
    prestretch_tensor_[gp] = compute_updated_prestretch_tensor(defgrd, prestretch_tensor_[gp]);
  }

  Core::LinAlg::Matrix<3, 3> elastic_deformation_gradient =
      get_elastic_deformation_gradient(defgrd, prestretch_tensor_[gp]);

  child_material_->update(elastic_deformation_gradient, gp, params, eleGID);
}

void Mat::IterativePrestressMaterial::setup(
    const int numgp, const Core::IO::InputParameterContainer& container)
{
  prestretch_tensor_.resize(numgp, Core::LinAlg::identity_matrix<3>());

  child_material_->setup(numgp, container);
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

void Mat::IterativePrestressMaterial::strain_energy(
    const Core::LinAlg::Matrix<6, 1>& glstrain, double& psi, int gp, int eleGID) const
{
  FOUR_C_THROW("Strain energy computation is currently not implemented for prestressing materials");
}

void Mat::IterativePrestressMaterial::evaluate(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
    Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, int gp, int eleGID)
{
  Core::LinAlg::Matrix<3, 3> elastic_deformation_gradient =
      get_elastic_deformation_gradient(*defgrd, prestretch_tensor_[gp]);

  // Evaluate green Lagrange strain
  const Core::LinAlg::Matrix<6, 1> elastic_gl_strain =
      get_green_lagrange_strain(elastic_deformation_gradient);

  Core::LinAlg::Matrix<6, 1> elastic_stress(true);
  Core::LinAlg::Matrix<6, 6> elastic_cmat(true);


  // evaluate child material
  child_material_->evaluate(&elastic_deformation_gradient, &elastic_gl_strain, params,
      &elastic_stress, &elastic_cmat, gp, eleGID);

  // push-forward operation
  *stress = Mat::push_forward_stress_tensor_voigt(elastic_stress, prestretch_tensor_[gp]);

  *cmat = Mat::push_forward_four_tensor(
      prestretch_tensor_[gp].determinant(), prestretch_tensor_[gp], elastic_cmat);
}


FOUR_C_NAMESPACE_CLOSE