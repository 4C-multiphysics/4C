// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_stvenantkirchhoff.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
Mat::PAR::StVenantKirchhoff::StVenantKirchhoff(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_(matdata.parameters.get<double>("YOUNG")),
      poissonratio_(matdata.parameters.get<double>("NUE")),
      density_(matdata.parameters.get<double>("DENS"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::StVenantKirchhoff::create_material()
{
  return std::make_shared<Mat::StVenantKirchhoff>(this);
}

Mat::StVenantKirchhoffType Mat::StVenantKirchhoffType::instance_;


Core::Communication::ParObject* Mat::StVenantKirchhoffType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* stvenantk = new Mat::StVenantKirchhoff();
  stvenantk->unpack(buffer);
  return stvenantk;
}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
Mat::StVenantKirchhoff::StVenantKirchhoff() : params_(nullptr) {}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
Mat::StVenantKirchhoff::StVenantKirchhoff(Mat::PAR::StVenantKirchhoff* params) : params_(params) {}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void Mat::StVenantKirchhoff::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void Mat::StVenantKirchhoff::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::StVenantKirchhoff*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }
}

/*----------------------------------------------------------------------*
//calculates stresses using one of the above method to evaluate the elasticity tensor
 *----------------------------------------------------------------------*/
void Mat::StVenantKirchhoff::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  cmat = StVenantKirchhoff::evaluate_stress_linearization(params_->youngs_, params_->poissonratio_);

  stress = StVenantKirchhoff::evaluate_stress(glstrain, params_->youngs_, params_->poissonratio_);
}


/*----------------------------------------------------------------------*
 |  Calculate strain energy                                    gee 10/09|
 *----------------------------------------------------------------------*/
double Mat::StVenantKirchhoff::strain_energy(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const EvaluationContext& context,
    const int gp, const int eleGID) const
{
  auto stress =
      StVenantKirchhoff::evaluate_stress(glstrain, params_->youngs_, params_->poissonratio_);

  return 0.5 * Core::LinAlg::ddot(stress, glstrain);
}

FOUR_C_NAMESPACE_CLOSE
