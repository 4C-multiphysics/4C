// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_fourier.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

using ActivationFieldType = Core::IO::InputField<std::vector<double>>;

Mat::PAR::Fourier::ActivationParameterVariant get_activation_params(
    const Core::Mat::PAR::Parameter::Data& matdata,
    const Inpar::Mat::ActivationType& activation_type)
{
  if (activation_type == Inpar::Mat::ActivationType::tensor)
  {
    return matdata.parameters.get<std::vector<double>>("TENSOR");
  }
  else if (activation_type == Inpar::Mat::ActivationType::map)
  {
    return matdata.parameters.get<ActivationFieldType>("MAPFILE_CONTENT");
  }
  else
    return std::monostate{};
}

struct ConductivityVisitor
{
  std::vector<double> operator()(const std::vector<double>& vector) const { return vector; }

  std::vector<double> operator()(const ActivationFieldType& map) const
  {
    return Mat::Fourier::get_validate_conductivity_tensor(map, eleGID_ + 1, conduct_para_num_);
  }

  std::vector<double> operator()(const std::monostate& /*unused*/) const
  {
    FOUR_C_THROW(
        "Error in ActivationParamsVisitor. You're calling it with the default-constructed "
        "state.");
  }

  const int& eleGID_;
  const std::size_t& conduct_para_num_;
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::Fourier::Fourier(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      capa_(matdata.parameters.get<double>("CAPA")),
      conduct_para_num_(matdata.parameters.get<int>("CONDUCT_PARA_NUM")),
      conductivity_type_(matdata.parameters.get<Inpar::Mat::ActivationType>("CONDUCT")),
      conductivityParams_(get_activation_params(matdata, conductivity_type_))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::Fourier::create_material()
{
  return std::make_shared<Mat::Fourier>(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::FourierType Mat::FourierType::instance_;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::FourierType::create(Core::Communication::UnpackBuffer& buffer)
{
  auto* fourier = new Mat::Fourier();
  fourier->unpack(buffer);
  return fourier;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::Fourier::Fourier() : params_(nullptr) {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::Fourier::Fourier(Mat::PAR::Fourier* params) : params_(params) {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::Fourier::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  int matid = params_->id();
  add_to_pack(data, matid);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::Fourier::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);

      FOUR_C_ASSERT_ALWAYS(mat->type() == material_type(),
          "Type of parameter material {} does not fit to calling type {}", mat->type(),
          material_type());

      params_ = static_cast<Mat::PAR::Fourier*>(mat);
    }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::vector<double> Mat::Fourier::conductivity(int eleGID) const
{
  const std::size_t num_conduct = params_->conduct_para_num_;
  return std::visit(ConductivityVisitor{eleGID, num_conduct}, params_->conductivityParams_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::Fourier::evaluate(const Core::LinAlg::Matrix<1, 1>& gradtemp,
    Core::LinAlg::Matrix<1, 1>& cmat, Core::LinAlg::Matrix<1, 1>& heatflux, const int eleGID) const
{
  const std::size_t num_conduct = params_->conduct_para_num_;
  const std::vector<double> conduct =
      std::visit(ConductivityVisitor{eleGID, num_conduct}, params_->conductivityParams_);

  // conductivity tensor in 1d is always a scalar quantity
  cmat(0, 0) = conduct[0];

  // heatflux
  heatflux.multiply_nn(cmat, gradtemp);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::Fourier::evaluate(const Core::LinAlg::Matrix<2, 1>& gradtemp,
    Core::LinAlg::Matrix<2, 2>& cmat, Core::LinAlg::Matrix<2, 1>& heatflux, const int eleGID) const
{
  const std::size_t num_conduct = params_->conduct_para_num_;
  const std::vector<double> conduct =
      std::visit(ConductivityVisitor{eleGID, num_conduct}, params_->conductivityParams_);

  // conductivity tensor in 2d is a 2x2 matrix with either constant or variable values on the
  // diagonal or a full matrix.
  cmat.clear();
  switch (params_->conduct_para_num_)
  {
    case 1:  // scalar value is given
    {
      cmat(0, 0) = conduct[0];
      cmat(1, 1) = conduct[0];
      break;
    }
    case 2:  // diagonal values are given
    {
      cmat(0, 0) = conduct[0];
      cmat(1, 1) = conduct[1];
      break;
    }
    case 4:  // full tensor is given
    {
      cmat(0, 0) = conduct[0];
      cmat(0, 1) = conduct[1];
      cmat(1, 0) = conduct[2];
      cmat(1, 1) = conduct[3];
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "Given conductivity vector doesn't have a valid size. Either give a scalar "
          "value (size=1), the diagonal values (size=2) or the full tensor (size=4).");
    }
  }

  // heatflux
  heatflux.multiply_nn(cmat, gradtemp);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::Fourier::evaluate(const Core::LinAlg::Matrix<3, 1>& gradtemp,
    Core::LinAlg::Matrix<3, 3>& cmat, Core::LinAlg::Matrix<3, 1>& heatflux, const int eleGID) const
{
  const std::size_t num_conduct = params_->conduct_para_num_;
  const std::vector<double> conduct =
      std::visit(ConductivityVisitor{eleGID, num_conduct}, params_->conductivityParams_);

  // conductivity tensor in 3d is a 3x3 matrix with either constant or variable values on the
  // diagonal or a full matrix.
  cmat.clear();
  switch (params_->conduct_para_num_)
  {
    case 1:  // scalar value is given
    {
      cmat(0, 0) = conduct[0];
      cmat(1, 1) = conduct[0];
      cmat(2, 2) = conduct[0];
      break;
    }
    case 3:  // diagonal values are given
    {
      cmat(0, 0) = conduct[0];
      cmat(1, 1) = conduct[1];
      cmat(2, 2) = conduct[2];
      break;
    }
    case 9:  // full tensor is given
    {
      cmat(0, 0) = conduct[0];
      cmat(0, 1) = conduct[1];
      cmat(0, 2) = conduct[2];
      cmat(1, 0) = conduct[3];
      cmat(1, 1) = conduct[4];
      cmat(1, 2) = conduct[5];
      cmat(2, 0) = conduct[6];
      cmat(2, 1) = conduct[7];
      cmat(2, 2) = conduct[8];
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "Given conductivity vector doesn't have a valid size. Either give a scalar "
          "value (size=1), the diagonal values (size=3) or the full tensor (size=9).");
    }
  }

  // heatflux
  heatflux.multiply_nn(cmat, gradtemp);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::vector<double> Mat::Fourier::get_validate_conductivity_tensor(
    const Core::IO::InputField<std::vector<double>>& conductivity_field, const int ele_id_key,
    const std::size_t conduct_para_num)
{
  std::vector<double> conductivity = conductivity_field.at(ele_id_key);

  if (conductivity.size() != conduct_para_num)
  {
    FOUR_C_THROW(
        "Number of conductivity tensor entries for element id %d does not match the number of "
        "conductivity parameters given %d.",
        ele_id_key, conduct_para_num);
  }

  return conductivity;
}

FOUR_C_NAMESPACE_CLOSE
