// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_structporo_reaction.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::StructPoroReaction::StructPoroReaction(const Core::Mat::PAR::Parameter::Data& matdata)
    : StructPoro(matdata), dofIDReacScalar_(matdata.parameters.get<int>("DOFIDREACSCALAR"))
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::StructPoroReaction::create_material()
{
  return std::make_shared<Mat::StructPoroReaction>(this);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::StructPoroReactionType Mat::StructPoroReactionType::instance_;

Core::Communication::ParObject* Mat::StructPoroReactionType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::StructPoroReaction* struct_poro = new Mat::StructPoroReaction();
  struct_poro->unpack(buffer);
  return struct_poro;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::StructPoroReaction::StructPoroReaction()
    : params_(nullptr), refporosity_(-1.0), dphiDphiref_(0.0), refporositydot_(0.0)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::StructPoroReaction::StructPoroReaction(Mat::PAR::StructPoroReaction* params)
    : StructPoro(params),
      params_(params),
      refporosity_(-1.0),
      dphiDphiref_(0.0),
      refporositydot_(0.0)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::StructPoroReaction::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  StructPoro::setup(numgp, fibers, coord_system);
  refporosity_ = params_->init_porosity_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::StructPoroReaction::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // refporosity_
  add_to_pack(data, refporosity_);

  // add base class material
  StructPoro::pack(data);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::StructPoroReaction::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::StructPoroReaction*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // refporosity_
  extract_from_pack(buffer, refporosity_);

  // extract base class material
  StructPoro::unpack(buffer);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::StructPoroReaction::compute_porosity(const Teuchos::ParameterList& params, double press,
    double J, int gp, double& porosity, double* dphi_dp, double* dphi_dJ, double* dphi_dJdp,
    double* dphi_dJJ, double* dphi_dpp, bool save)
{
  // evaluate change of reference porosity due to reaction

  // TODO: do not read from parameter list!
  if (params.isParameter("scalars"))
  {
    std::shared_ptr<std::vector<double>> scalars =
        params.get<std::shared_ptr<std::vector<double>>>("scalars");
    reaction(porosity, J, scalars, params);
  }

  // call base class to compute porosity
  StructPoro::compute_porosity(refporosity_, press, J, gp, porosity, dphi_dp, dphi_dJ, dphi_dJdp,
      dphi_dJJ, dphi_dpp, &dphiDphiref_, save);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::StructPoroReaction::constitutive_derivatives(const Teuchos::ParameterList& params,
    double press, double J, double porosity, double* dW_dp, double* dW_dphi, double* dW_dJ,
    double* dW_dphiref, double* W)
{
  if (porosity == 0.0)
    FOUR_C_THROW(
        "porosity equals zero!! Wrong initial porosity? (or wrong collagen density for ecm "
        "material)");

  // evaluate change of reference porosity due to reaction

  // TODO: do not read from parameter list!
  if (params.isParameter("scalars"))
  {
    std::shared_ptr<std::vector<double>> scalars =
        params.get<std::shared_ptr<std::vector<double>>>("scalars");
    reaction(porosity, J, scalars, params);
  }

  // call base class
  StructPoro::constitutive_derivatives(
      params, press, J, porosity, refporosity_, dW_dp, dW_dphi, dW_dJ, dW_dphiref, W);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::StructPoroReaction::reaction(const double porosity, const double J,
    std::shared_ptr<std::vector<double>> scalars, const Teuchos::ParameterList& params)
{
  if (params.isParameter("total time"))  // if time is set
  {
    // double dt = params.get<double>("delta time",-1.0);
    double time = params.get<double>("total time");

    // use scalar for this type of reaction
    double cnp = scalars->at(params_->dofIDReacScalar_);

    // double k = 1.0;
    double tau = 200.0 * cnp;     ///(cnp+k); 20.0/(20*cnp+1.0)
    double limitporosity = 0.45;  // 0.8;

    refporosity_ =
        limitporosity - (limitporosity - params_->init_porosity_) * exp(-1.0 * time / tau);
    refporositydot_ = (limitporosity - params_->init_porosity_) / tau * exp(-1.0 * time / tau);
  }
  else  //(time==-1.0) -> time not set (this happens during setup-> no reaction)
  {
    // do nothing
  }
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::StructPoroReaction::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  // call base class
  StructPoro::evaluate(defgrad, glstrain, params, context, stress, cmat, gp, eleGID);

  // scale stresses and cmat
  stress *= (1.0 - refporosity_) / (1.0 - params_->init_porosity_);
  cmat *= (1.0 - refporosity_) / (1.0 - params_->init_porosity_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::StructPoroReaction::ref_porosity_av() const { return refporosity_; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::StructPoroReaction::vis_names(std::map<std::string, int>& names) const
{
  // call base class
  StructPoro::vis_names(names);
  std::string name = "reference_porosity";
  names[name] = 1;  // scalar
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Mat::StructPoroReaction::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  // call base class
  if (StructPoro::vis_data(name, data, numgp, eleID)) return true;
  if (name == "reference_porosity")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    data[0] = ref_porosity_av();
    return true;
  }
  return false;
}

FOUR_C_NAMESPACE_CLOSE
