// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_fourieraniso.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FourierAniso::FourierAniso(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      // be careful: capa_ := rho * C_V, e.g contains the density
      capa_(matdata.parameters.get<double>("CAPA")),
      conduct_(matdata.parameters.get<std::vector<double>>("CONDUCT"))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::FourierAniso::create_material()
{
  return std::make_shared<Mat::FourierAniso>(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::FourierAnisoType Mat::FourierAnisoType::instance_;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::FourierAnisoType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* fourieraniso = new Mat::FourierAniso();
  fourieraniso->unpack(buffer);
  return fourieraniso;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::FourierAniso::FourierAniso() : params_(nullptr) {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::FourierAniso::FourierAniso(Mat::PAR::FourierAniso* params) : params_(params) {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::FourierAniso::pack(Core::Communication::PackBuffer& data) const
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
 *----------------------------------------------------------------------*/
void Mat::FourierAniso::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::FourierAniso*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
            material_type());
    }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::FourierAniso::evaluate(const Core::LinAlg::Matrix<1, 1>& gradtemp,
    Core::LinAlg::Matrix<1, 1>& cmat, Core::LinAlg::Matrix<1, 1>& heatflux) const
{
  // conductivity tensor
  cmat(0, 0) = params_->conduct_[0];

  // heatflux
  heatflux.multiply_nn(cmat, gradtemp);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::FourierAniso::evaluate(const Core::LinAlg::Matrix<2, 1>& gradtemp,
    Core::LinAlg::Matrix<2, 2>& cmat, Core::LinAlg::Matrix<2, 1>& heatflux) const
{
  // conductivity tensor
  cmat.clear();
  for (int i = 0; i < 2; ++i) cmat(i, i) = params_->conduct_[i];

  // heatflux
  heatflux.multiply_nn(cmat, gradtemp);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::FourierAniso::evaluate(const Core::LinAlg::Matrix<3, 1>& gradtemp,
    Core::LinAlg::Matrix<3, 3>& cmat, Core::LinAlg::Matrix<3, 1>& heatflux) const
{
  // conductivity tensor
  cmat.clear();
  for (int i = 0; i < 3; ++i) cmat(i, i) = params_->conduct_[i];

  // heatflux
  heatflux.multiply_nn(cmat, gradtemp);
}

FOUR_C_NAMESPACE_CLOSE
