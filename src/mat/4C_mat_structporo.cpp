// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_structporo.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"  // for function Factory in Unpack
#include "4C_global_data.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_poro_law.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

Mat::PAR::StructPoro::StructPoro(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      matid_(matdata.parameters.get<int>("MATID")),
      poro_law_ID_(matdata.parameters.get<int>("POROLAWID")),
      init_porosity_(matdata.parameters.get<double>("INITPOROSITY"))
{
  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

  // for the sake of safety
  if (Global::Problem::instance(probinst)->materials() == nullptr)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");
  // yet another safety check
  if (Global::Problem::instance(probinst)->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // retrieve validated input line of material ID in question
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(poro_law_ID_);

  switch (curmat->type())
  {
    case Core::Materials::m_poro_law_linear:
    {
      poro_law_ = static_cast<Mat::PAR::PoroLaw*>(curmat);
      break;
    }
    case Core::Materials::m_poro_law_constant:
    {
      poro_law_ = static_cast<Mat::PAR::PoroLaw*>(curmat);
      break;
    }
    case Core::Materials::m_poro_law_logNeoHooke_Penalty:
    {
      poro_law_ = static_cast<Mat::PAR::PoroLaw*>(curmat);
      break;
    }
    case Core::Materials::m_poro_law_incompr_skeleton:
    {
      poro_law_ = static_cast<Mat::PAR::PoroLawIncompSkeleton*>(curmat);
      break;
    }
    case Core::Materials::m_poro_law_linear_biot:
    {
      poro_law_ = static_cast<Mat::PAR::PoroLawLinBiot*>(curmat);
      break;
    }
    case Core::Materials::m_poro_law_density_dependent:
    {
      poro_law_ = static_cast<Mat::PAR::PoroLawDensityDependent*>(curmat);
      break;
    }
    default:
      FOUR_C_THROW("invalid material for porosity law {}", curmat->type());
      break;
  }
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::StructPoro::create_material()
{
  return std::make_shared<Mat::StructPoro>(this);
}

Mat::StructPoroType Mat::StructPoroType::instance_;

Core::Communication::ParObject* Mat::StructPoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* struct_poro = new Mat::StructPoro();
  struct_poro->unpack(buffer);
  return struct_poro;
}

Mat::StructPoro::StructPoro()
    : params_(nullptr),
      mat_(nullptr),
      porosity_(nullptr),
      surf_porosity_(nullptr),
      is_initialized_(false)
{
}

Mat::StructPoro::StructPoro(Mat::PAR::StructPoro* params)
    : params_(params), porosity_(nullptr), surf_porosity_(nullptr), is_initialized_(false)
{
  mat_ = std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(params_->matid_));
  if (mat_ == nullptr)
    FOUR_C_THROW("Mat::StructPoro: underlying material should be of type Mat::So3Material");
}

void Mat::StructPoro::poro_setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  porosity_ = std::make_shared<std::vector<double>>(numgp, params_->init_porosity_);
  surf_porosity_ = std::make_shared<std::map<int, std::vector<double>>>();

  is_initialized_ = true;
}

inline Core::Materials::MaterialType Mat::StructPoro::poro_law_type() const
{
  return params_->poro_law_->type();
}

double Mat::StructPoro::inv_bulk_modulus() const { return params_->poro_law_->inv_bulk_modulus(); }

double Mat::StructPoro::density() const
{
  if (params_->init_porosity_ == 1.0)
    return mat_->density();
  else
    return ((1.0 - params_->init_porosity_) * mat_->density());
}

double Mat::StructPoro::density_solid_phase() const { return mat_->density(); }

void Mat::StructPoro::pack(Core::Communication::PackBuffer& data) const
{
  if (not is_initialized_) FOUR_C_THROW("poro material not initialized. Not a poro element?");

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // porosity_
  add_to_pack(data, *porosity_);
  add_to_pack(data, *surf_porosity_);

  // Pack data of underlying material
  if (mat_ != nullptr)
  {
    add_to_pack(data, true);
    mat_->pack(data);
  }
  else
  {
    add_to_pack(data, false);
  }
}

void Mat::StructPoro::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
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
        params_ = static_cast<Mat::PAR::StructPoro*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  // porosity_
  porosity_ = std::make_shared<std::vector<double>>();
  extract_from_pack(buffer, *porosity_);

  // surface porosity (i think it is not necessary to pack/unpack this...)
  surf_porosity_ = std::make_shared<std::map<int, std::vector<double>>>();
  extract_from_pack(buffer, *surf_porosity_);

  // Unpack data of sub material (these lines are copied from element.cpp)
  bool mat_exists;
  extract_from_pack(buffer, mat_exists);
  if (mat_exists)
  {
    Core::Communication::ParObject* o =
        Core::Communication::factory(buffer);  // Unpack is done here
    auto* mat = dynamic_cast<Mat::So3Material*>(o);
    if (mat == nullptr) FOUR_C_THROW("failed to unpack elastic material");
    mat_ = std::shared_ptr<Mat::So3Material>(mat);
  }
  else
    mat_ = nullptr;

  is_initialized_ = true;
}

void Mat::StructPoro::post_setup(const Teuchos::ParameterList& params, const int eleGID)
{
  // Forward post_setup call to actual solid material
  mat_->post_setup(params, eleGID);
}

void Mat::StructPoro::compute_porosity(const double& refporosity, const double& press,
    const double& J, const int& gp, double& porosity, double* dphi_dp, double* dphi_dJ,
    double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, double* dphi_dphiref, bool save)
{
  if (refporosity == 1.0)
  {
    // this is pure fluid. The porosity does not change

    porosity = refporosity;
    if (dphi_dp) *dphi_dp = 0.0;
    if (dphi_dJ) *dphi_dJ = 0.0;
    if (dphi_dJdp) *dphi_dJdp = 0.0;
    if (dphi_dJJ) *dphi_dJJ = 0.0;
    if (dphi_dpp) *dphi_dpp = 0.0;
    if (dphi_dphiref) *dphi_dphiref = 0.0;
    return;
  }

  params_->poro_law_->compute_porosity(refporosity, press, J, gp, porosity, dphi_dp, dphi_dJ,
      dphi_dJdp, dphi_dJJ, dphi_dpp, dphi_dphiref);

  // save porosity
  if (save) porosity_->at(gp) = porosity;
}

void Mat::StructPoro::compute_porosity(const Teuchos::ParameterList& params, double press, double J,
    int gp, double& porosity, double* dphi_dp, double* dphi_dJ, double* dphi_dJdp, double* dphi_dJJ,
    double* dphi_dpp, bool save)
{
  compute_porosity(
      params_
          ->init_porosity_,  // reference porosity equals initial porosity for non reactive material
      press, J, gp, porosity, dphi_dp, dphi_dJ, dphi_dJdp, dphi_dJJ, dphi_dpp,
      nullptr,  // reference porosity is constant (non reactive) -> derivative not needed
      save);
}

void Mat::StructPoro::compute_porosity(const Teuchos::ParameterList& params, double press, double J,
    int gp, double& porosity, bool save)
{
  compute_porosity(
      params, press, J, gp, porosity, nullptr, nullptr, nullptr, nullptr, nullptr, save);
}

void Mat::StructPoro::compute_surf_porosity(const Teuchos::ParameterList& params, double press,
    double J, const int surfnum, int gp, double& porosity, double* dphi_dp, double* dphi_dJ,
    double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, bool save)
{
  compute_porosity(
      params, press, J, gp, porosity, dphi_dp, dphi_dJ, dphi_dJdp, dphi_dJJ, dphi_dpp, save);

  if (save)
  {
    if (gp == 0)  // it's a new iteration, so old values are not needed any more
      ((*surf_porosity_)[surfnum]).clear();

    ((*surf_porosity_)[surfnum]).push_back(porosity);
  }
}

void Mat::StructPoro::compute_surf_porosity(const Teuchos::ParameterList& params, double press,
    double J, const int surfnum, int gp, double& porosity, bool save)
{
  compute_surf_porosity(
      params, press, J, surfnum, gp, porosity, nullptr, nullptr, nullptr, nullptr, nullptr, save);
}


double Mat::StructPoro::porosity_av() const
{
  double porosityav = 0.0;

  std::vector<double>::const_iterator m;
  for (m = porosity_->begin(); m != porosity_->end(); ++m)
  {
    porosityav += *m;
  }
  porosityav = porosityav / (porosity_->size());

  return porosityav;
}

void Mat::StructPoro::constitutive_derivatives(const Teuchos::ParameterList& params, double press,
    double J, double porosity, double* dW_dp, double* dW_dphi, double* dW_dJ, double* dW_dphiref,
    double* W)
{
  if (porosity == 0.0) FOUR_C_THROW("porosity equals zero!! Wrong initial porosity?");

  constitutive_derivatives(
      params, press, J, porosity, params_->init_porosity_, dW_dp, dW_dphi, dW_dJ, dW_dphiref, W);
}

void Mat::StructPoro::constitutive_derivatives(const Teuchos::ParameterList& params, double press,
    double J, double porosity, double refporosity, double* dW_dp, double* dW_dphi, double* dW_dJ,
    double* dW_dphiref, double* W)
{
  params_->poro_law_->constitutive_derivatives(
      params, press, J, porosity, refporosity, dW_dp, dW_dphi, dW_dJ, dW_dphiref, W);
}

void Mat::StructPoro::vis_names(std::map<std::string, int>& names) const
{
  mat_->vis_names(names);
  std::string porosity = "porosity";
  names[porosity] = 1;  // scalar
}

bool Mat::StructPoro::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  if (mat_->vis_data(name, data, numgp)) return true;
  if (name == "porosity")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    data[0] = porosity_av();
    return true;
  }
  return false;
}

FOUR_C_NAMESPACE_CLOSE
