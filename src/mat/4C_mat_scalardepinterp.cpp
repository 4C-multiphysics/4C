// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scalardepinterp.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ScalarDepInterp::ScalarDepInterp(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      id_lambda_zero_(matdata.parameters.get<int>("IDMATZEROSC")),
      id_lambda_unit_(matdata.parameters.get<int>("IDMATUNITSC")) {

      };

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::ScalarDepInterp::create_material()
{
  return std::make_shared<Mat::ScalarDepInterp>(this);
}


Mat::ScalarDepInterpType Mat::ScalarDepInterpType::instance_;


Core::Communication::ParObject* Mat::ScalarDepInterpType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ScalarDepInterp* ScalarDepInterp = new Mat::ScalarDepInterp();
  ScalarDepInterp->unpack(buffer);
  return ScalarDepInterp;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScalarDepInterp::ScalarDepInterp()
    : params_(nullptr), isinit_(false), lambda_zero_mat_(nullptr), lambda_unit_mat_(nullptr)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScalarDepInterp::ScalarDepInterp(Mat::PAR::ScalarDepInterp* params)
    : params_(params), isinit_(false), lambda_zero_mat_(nullptr), lambda_unit_mat_(nullptr)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  if (isinit_)
    FOUR_C_THROW("This function should just be called, if the material is jet not initialized.");

  // Setup of elastic material for zero concentration
  lambda_zero_mat_ =
      std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(params_->id_lambda_zero_));
  lambda_zero_mat_->setup(numgp, fibers, coord_system);

  // Setup of elastic material for zero concentration
  lambda_unit_mat_ =
      std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(params_->id_lambda_unit_));
  lambda_unit_mat_->setup(numgp, fibers, coord_system);

  // Some safety check
  const double density1 = lambda_zero_mat_->density();
  const double density2 = lambda_unit_mat_->density();
  if (abs(density1 - density2) > 1e-14)
    FOUR_C_THROW(
        "The densities of the materials specified in IDMATZEROSC and IDMATUNITSC must be equal!");

  lambda_ = std::vector<double>(numgp, 1.0);

  // initialization done
  isinit_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  // evaluate elastic material corresponding to zero concentration
  Core::LinAlg::SymmetricTensor<double, 3, 3> stress_lambda_zero{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat_zero_conc{};

  lambda_zero_mat_->evaluate(
      defgrad, glstrain, params, context, stress_lambda_zero, cmat_zero_conc, gp, eleGID);

  // evaluate elastic material corresponding to infinite concentration
  Core::LinAlg::SymmetricTensor<double, 3, 3> stress_lambda_unit{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat_infty_conc{};

  lambda_unit_mat_->evaluate(
      defgrad, glstrain, params, context, stress_lambda_unit, cmat_infty_conc, gp, eleGID);

  double lambda;
  // get the ratio of interpolation
  // NOTE: if no ratio is available, we use the IDMATZEROSC material!
  if (params.isParameter("lambda"))
  {
    lambda = params.get<double>("lambda");

    // NOTE: this would be nice, but since negative concentrations can occur,
    // we have to catch 'unnatural' cases different...
    //  if ( lambda < -1.0e-14 or lambda > (1.0+1.0e-14) )
    //      FOUR_C_THROW("The lambda must be in [0,1]!");

    // e.g. like that:
    if (lambda < -1.0e-14) lambda = 0.0;
    if (lambda > (1.0 + 1.0e-14)) lambda = 1.0;

    lambda_.at(gp) = lambda;
  }
  else
  {
    lambda = lambda_.at(gp);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // \mym Psi = (1-\lambda(C)) * \Psi_0(\mym C) + \lambda(C) * \Psi_1(\mym C) +  = ...
  // \mym S = 2 \frac{\partial}{\partial \mym C} \left( \Psi(\mym C) \right) = ...
  // \mym S = 2 \frac{\partial}{\partial \mym C} \left( (1-\lambda(C)) * \Psi_0(\mym C) + \lambda(C)
  // * \Psi_1(\mym C) \right) = ...
  ///////////////////////////////////////////////////////////////////////////////////////////////

  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  // do the linear interpolation between stresses:
  // ... = (1-\lambda(C)) * 2* \frac{\partial}{\partial \mym C} \Psi_0) + \lambda(C) * 2 *
  // \frac{\partial}{\partial \mym C} \Psi_1
  stress = (1 - lambda) * stress_lambda_zero + lambda * stress_lambda_unit;
  cmat = (1 - lambda) * cmat_zero_conc + lambda * cmat_infty_conc;

  if (params.isParameter("dlambda_dC"))
  {
    // get derivative of interpolation ratio w.r.t. glstrain
    std::shared_ptr<Core::LinAlg::Matrix<6, 1>> dlambda_dC =
        params.get<std::shared_ptr<Core::LinAlg::Matrix<6, 1>>>("dlambda_dC");

    // evaluate strain energy functions
    double psi_lambda_zero = lambda_zero_mat_->strain_energy(glstrain, context, gp, eleGID);

    double psi_lambda_unit = lambda_unit_mat_->strain_energy(glstrain, context, gp, eleGID);

    // and add the stresses due to possible dependency of the ratio w.r.t. to C
    // ... - 2 * \Psi_0 * \frac{\partial}{\partial \mym C} \lambda(C) ) + * 2 * \Psi_1 *
    // \frac{\partial}{\partial \mym C} \lambda(C)
    stress_view.update(
        -2.0 * psi_lambda_zero, *dlambda_dC, +2.0 * psi_lambda_unit, *dlambda_dC, 1.0);

    // Note: for the linearization we do neglect the derivatives of the ratio w.r.t. glstrain
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  int numgp = 0;
  if (isinit_)
  {
    numgp = lambda_.size();
  }
  add_to_pack(data, numgp);

  for (int gp = 0; gp < numgp; gp++)
  {
    add_to_pack(data, lambda_.at(gp));
  }

  // Pack data of both elastic materials
  if (lambda_zero_mat_ != nullptr and lambda_unit_mat_ != nullptr)
  {
    add_to_pack(data, true);
    lambda_zero_mat_->pack(data);
    lambda_unit_mat_->pack(data);
  }
  else
  {
    add_to_pack(data, false);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::unpack(Core::Communication::UnpackBuffer& buffer)
{
  isinit_ = true;


  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
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
        params_ = dynamic_cast<Mat::PAR::ScalarDepInterp*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  int numgp;
  extract_from_pack(buffer, numgp);
  if (not(numgp == 0))
  {
    lambda_ = std::vector<double>(numgp, 1.0);

    for (int gp = 0; gp < numgp; gp++)
    {
      double lambda = 1.0;
      extract_from_pack(buffer, lambda);
      lambda_.at(gp) = lambda;
    }
  }

  // Unpack data of elastic material (these lines are copied from element.cpp)
  bool have_extra_materials;
  extract_from_pack(buffer, have_extra_materials);
  if (have_extra_materials)
  {
    {
      Core::Communication::ParObject* o = Core::Communication::factory(buffer);
      Mat::So3Material* matel = dynamic_cast<Mat::So3Material*>(o);
      if (matel == nullptr) FOUR_C_THROW("failed to unpack elastic material");
      lambda_zero_mat_ = std::shared_ptr<So3Material>(matel);
    }

    {
      Core::Communication::ParObject* o = Core::Communication::factory(buffer);
      Mat::So3Material* matel = dynamic_cast<Mat::So3Material*>(o);
      if (matel == nullptr) FOUR_C_THROW("failed to unpack elastic material");
      lambda_unit_mat_ = std::shared_ptr<So3Material>(matel);
    }
  }
  else
  {
    lambda_zero_mat_ = nullptr;
    lambda_unit_mat_ = nullptr;
  }
}

/*----------------------------------------------------------------------------*/
void Mat::ScalarDepInterp::update()
{
  lambda_zero_mat_->update();
  lambda_unit_mat_->update();
}

/*----------------------------------------------------------------------------*/
void Mat::ScalarDepInterp::reset_step()
{
  lambda_zero_mat_->reset_step();
  lambda_unit_mat_->reset_step();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::vis_names(std::map<std::string, int>& names) const
{
  std::string fiber = "lambda";
  names[fiber] = 1;  // 1-dim vector

  lambda_zero_mat_->vis_names(names);
  lambda_unit_mat_->vis_names(names);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Mat::ScalarDepInterp::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  if (name == "lambda")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");

    double temp = 0.0;
    for (int gp = 0; gp < numgp; gp++)
    {
      temp += lambda_.at(gp);
    }

    data[0] = temp / ((double)numgp);
  }

  bool tmp1 = lambda_zero_mat_->vis_data(name, data, numgp, eleID);
  bool tmp2 = lambda_unit_mat_->vis_data(name, data, numgp, eleID);
  return (tmp1 and tmp2);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::ScalarDepInterp::strain_energy(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const EvaluationContext& context,
    const int gp, const int eleGID) const
{
  // evaluate strain energy functions
  double psi_lambda_zero = lambda_zero_mat_->strain_energy(glstrain, context, gp, eleGID);

  double psi_lambda_unit = lambda_unit_mat_->strain_energy(glstrain, context, gp, eleGID);

  double lambda = 0.0;
  for (double gp : lambda_)
  {
    lambda += gp;
  }
  lambda = lambda / (lambda_.size());

  return (1 - lambda) * psi_lambda_zero + lambda * psi_lambda_unit;
}

FOUR_C_NAMESPACE_CLOSE
