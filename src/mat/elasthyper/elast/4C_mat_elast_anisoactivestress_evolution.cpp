// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_anisoactivestress_evolution.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_mat_anisotropy_extension.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::AnisoActiveStressEvolution::AnisoActiveStressEvolution(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : ParameterAniso(matdata),
      sigma_(matdata.parameters.get<double>("SIGMA")),
      tauc0_(matdata.parameters.get<double>("TAUC0")),
      maxactiv_(matdata.parameters.get<double>("MAX_ACTIVATION")),
      minactiv_(matdata.parameters.get<double>("MIN_ACTIVATION")),
      activationthreshold_(matdata.parameters.get<double>("ACTIVATION_THRES")),
      sourceactiv_(matdata.parameters.get<int>("SOURCE_ACTIVATION")),
      strain_dep_(matdata.parameters.get<bool>("STRAIN_DEPENDENCY")),
      lambda_lower_(matdata.parameters.get<double>("LAMBDA_LOWER")),
      lambda_upper_(matdata.parameters.get<double>("LAMBDA_UPPER")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      init_(matdata.parameters.get<int>("INIT")),
      adapt_angle_(matdata.parameters.get<bool>("ADAPT_ANGLE"))
{
}

Mat::Elastic::AnisoActiveStressEvolution::AnisoActiveStressEvolution(
    Mat::Elastic::PAR::AnisoActiveStressEvolution* params)
    : params_(params),
      tauc_np_(0.0),
      tauc_n_(0.0),
      anisotropy_extension_(params->init_, params_->gamma_, params_->adapt_angle_ != 0,
          params_->structural_tensor_strategy(), {0})
{
  anisotropy_extension_.register_needed_tensors(
      FiberAnisotropyExtension<1>::FIBER_VECTORS | FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
}

void Mat::Elastic::AnisoActiveStressEvolution::pack_summand(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, tauc_n_);
  anisotropy_extension_.pack_anisotropy(data);
}

void Mat::Elastic::AnisoActiveStressEvolution::unpack_summand(
    Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, tauc_n_);
  anisotropy_extension_.unpack_anisotropy(buffer);
}

void Mat::Elastic::AnisoActiveStressEvolution::register_anisotropy_extensions(
    Mat::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void Mat::Elastic::AnisoActiveStressEvolution::setup(int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // Setup of active stress model
  tauc_n_ = params_->tauc0_;
  tauc_np_ = params_->tauc0_;

  // reasonability check...
  if (params_->strain_dep_)
  {
    if (params_->lambda_lower_ >= params_->lambda_upper_)
    {
      FOUR_C_THROW(
          "LAMBDA_LOWER should be lower than LAMBDA_UPPER. Seems reasonable, doesn't it? Dude...");
    }
  }
}

void Mat::Elastic::AnisoActiveStressEvolution::post_setup(const Teuchos::ParameterList& params)
{
  Summand::post_setup(params);
}

void Mat::Elastic::AnisoActiveStressEvolution::add_stress_aniso_principal(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, const Teuchos::ParameterList& params,
    const int gp, const int eleGID)
{
  // Virtual GP (is zero for element fibers, otherwise it is the current GP)
  const Core::LinAlg::SymmetricTensor<double, 3, 3>& A =
      anisotropy_extension_.get_structural_tensor(gp, 0);

  double dt = params.get<double>("delta time");

  double activationFunction = 0.0;
  std::shared_ptr<Core::Mat::Material> scatramat;
  if (params_->sourceactiv_ == 0)
  {
    activationFunction = params.get<double>("scalar");
  }
  else if (params_->sourceactiv_ == -1)
  {
    if (params.isParameter("scalars"))
    {
      // get pointer to vector containing the scalar values at the Gauss points
      std::shared_ptr<std::vector<double>> scalars =
          params.get<std::shared_ptr<std::vector<double>>>("scalars");

      // safety check
      FOUR_C_ASSERT_ALWAYS(scalars != nullptr, "Activation is not provided via scalars parameter");

      // check if activation is above the given threshold
      if (scalars->at(0) >= params_->activationthreshold_)
      {
        activationFunction = 1.0;
      }
    }
  }
  else
  {
    double totaltime = params.get<double>("total time");
    const auto& element_center_coordinates_ref =
        params.get<Core::LinAlg::Tensor<double, 3>>("elecenter_coords_ref");
    activationFunction =
        Global::Problem::instance()
            ->function_by_id<Core::Utils::FunctionOfSpaceTime>(params_->sourceactiv_)
            .evaluate(element_center_coordinates_ref.as_span(), totaltime, 0);
  }

  double lambda = 0.0;
  double scale = 0.0;
  double n0 = 1.0;
  if (params_->strain_dep_)
  {
    // squared stretch along fiber direction
    double I4;
    I4 = Core::LinAlg::ddot(A, rcg);
    // stretch along fiber direction
    lambda = sqrt(I4);
    scale = -4. / (pow((params_->lambda_lower_ - params_->lambda_upper_), 2.0));
    // Frank-Starling factor n0 for strain-dependent contractility
    if (lambda >= params_->lambda_lower_ and lambda <= params_->lambda_upper_)
    {
      n0 = scale * (lambda - params_->lambda_lower_) * (lambda - params_->lambda_upper_);
    }
    else
    {
      n0 = 0.0;
    }
  }

  activationFunction =
      activationFunction * (params_->maxactiv_ - params_->minactiv_) + params_->minactiv_;
  double abs_u_ = abs(activationFunction);
  double absplus_u_ = abs_u_ * static_cast<double>(activationFunction > 0.0);
  tauc_np_ = (tauc_n_ / dt + n0 * params_->sigma_ * absplus_u_) / (1 / dt + abs_u_);
  stress += tauc_np_ * A;

  // only contribution to cmat if we have strain dependency!
  if (params_->strain_dep_)
  {
    double dtauc_np_dC;
    // Cmat_active = 2 * dS_active / dC = 2 * dtau(t) / dC \otimes (a_0 \otimes a_0)
    if (lambda >= params_->lambda_lower_ and lambda <= params_->lambda_upper_)
    {
      dtauc_np_dC = 2. * scale *
                    (1. - (params_->lambda_upper_ + params_->lambda_lower_) / (2. * lambda)) *
                    params_->sigma_ * absplus_u_ / (1 / dt + abs_u_);
    }
    else
    {
      dtauc_np_dC = 0.0;
    }
    cmat += dtauc_np_dC * Core::LinAlg::dyadic(A, A);
  }
}

void Mat::Elastic::AnisoActiveStressEvolution::get_fiber_vecs(
    std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs  ///< vector of all fiber vectors
) const
{
  if (params_->init_ == DefaultAnisotropyExtension<1>::INIT_MODE_NODAL_FIBERS)
  {
    // This method expects constant fibers within this element but the init mode is such that
    // fibers are defined on the Gauss points
    // We therefore cannot return sth here.

    // ToDo: This may needs improvements later on if needed!
    return;
  }

  fibervecs.push_back(anisotropy_extension_.get_fiber(BaseAnisotropyExtension::GPDEFAULT, 0));
}

// Update internal stress variables
void Mat::Elastic::AnisoActiveStressEvolution::update() { tauc_n_ = tauc_np_; }

void Mat::Elastic::AnisoActiveStressEvolution::set_fiber_vecs(const double newgamma,
    const Core::LinAlg::Tensor<double, 3, 3>& locsys,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
{
  anisotropy_extension_.set_fiber_vecs(newgamma, locsys, defgrd);
}

FOUR_C_NAMESPACE_CLOSE
