// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_electrode.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function_of_scalar.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::Electrode::Electrode(const Core::Mat::PAR::Parameter::Data& matdata)
    : ElchSingleMat(matdata),
      cmax_(matdata.parameters.get<double>("C_MAX")),
      chimax_(matdata.parameters.get<double>("CHI_MAX")),
      ocpmodel_(matdata.parameters.group("OCP_MODEL").get<OCPModels>("OCP_MODEL"))
{
  const auto& ocpmodel = matdata.parameters.group("OCP_MODEL");

  switch (ocpmodel_)
  {
    case OCPModels::function:
    {
      ocp_function_num_ = ocpmodel.group("Function").get<int>("OCP_FUNCT_NUM");
      break;
    }
    case OCPModels::redlichkister:
    {
      ocppara_ = ocpmodel.group("Redlich-Kister").get<std::vector<double>>("OCP_PARA");
      FOUR_C_ASSERT_ALWAYS(ocppara_.size() == 16 or ocppara_.size() == 21,
          "Currently used parameterization of the Redlich-Kister model can only have 16 or 21 "
          "parameters! You set {} in the input file. Check your input file or adapt the "
          "implementation.",
          ocppara_.size());
      break;
    }
    case OCPModels::taralov:
    {
      ocppara_ = ocpmodel.group("Taralov").get<std::vector<double>>("OCP_PARA");
      break;
    }
    default:
      FOUR_C_THROW("Unknown OCPModel");
  }

  if (ocpmodel.has_group("LITHIATION_BOUNDS"))
  {
    const auto& lithiation_bounds = ocpmodel.group("LITHIATION_BOUNDS");
    lithiation_bounds_.emplace(LithiationBounds{.x_min = lithiation_bounds.get<double>("X_MIN"),
        .x_max = lithiation_bounds.get<double>("X_MAX")});

    const auto x_min = lithiation_bounds_->x_min;
    const auto x_max = lithiation_bounds_->x_max;
    FOUR_C_ASSERT_ALWAYS(x_min < x_max,
        "Lower bound (X_MIN: {}) of range of validity for ocp calculation model needs to be "
        "smaller than upper bound (X_MAX: {})!",
        x_min, x_max);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::Electrode::create_material()
{
  return std::make_shared<Mat::Electrode>(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::Utils::FunctionOfScalar& Mat::PAR::Electrode::ocp_function()
{
  if (!ocp_function_.has_value())
  {
    ocp_function_.emplace(
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfScalar>(
            ocp_function_num_));
  }
  return ocp_function_->get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::Electrode::x_min() const
{
  FOUR_C_ASSERT_ALWAYS(lithiation_bounds_.has_value(),
      "You try to access the lower bound (X_MIN) of the range of validity for the ocp "
      "calculation model, but no lithiation bounds have been set in the input!");

  return lithiation_bounds_->x_min;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::Electrode::x_max() const
{
  FOUR_C_ASSERT_ALWAYS(lithiation_bounds_.has_value(),
      "You try to access the upper bound (X_MAX) of the range of validity for the ocp calculation "
      "model, but no lithiation bounds have been set in the input!");

  return lithiation_bounds_->x_max;
}


Mat::ElectrodeType Mat::ElectrodeType::instance_;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::ElectrodeType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* electrode = new Mat::Electrode();
  electrode->unpack(buffer);
  return electrode;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::Electrode::Electrode(Mat::PAR::Electrode* params) : ElchSingleMat(params), params_(params) {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::Electrode::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  const int type = unique_par_object_id();
  add_to_pack(data, type);

  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::Electrode::unpack(Core::Communication::UnpackBuffer& buffer)
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
      {
        params_ = static_cast<Mat::PAR::Electrode*>(mat);
        set_elch_single_mat_params(params_);
      }
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::Electrode::compute_open_circuit_potential(
    const double concentration, const double faraday, const double frt, const double detF) const
{
  double ocp(0.0);

  // intercalation fraction
  const double X = compute_intercalation_fraction(concentration, chi_max(), c_max(), detF);

  // print warning to screen if prescribed interval of validity for ocp calculation model is given
  // but not satisfied
  if (params_->lithiation_bounds_.has_value() and (X < params_->x_min() or X > params_->x_max()))
  {
    std::cout << "WARNING: intercalation fraction X = c/c_max is violating prescribed bounds of "
                 "ocp calculation model. Calculated values might therefore not be reasonable!"
              << '\n';
    std::cout << "X: " << X << " lower bound is: " << params_->x_min()
              << " upper bound is: " << params_->x_max() << "\n\n";
  }

  // physically reasonable intercalation fraction
  if (X > 0.0 and X < 1.0)
  {
    switch (params_->ocpmodel_)
    {
      // open circuit potential obtained from function
      case Mat::PAR::OCPModels::function:
      {
        ocp = params_->ocp_function().evaluate(X);

        break;
      }

      // open circuit potential according to Redlich-Kister expansion
      case Mat::PAR::OCPModels::redlichkister:
      {
        // cf. Colclasure and Kee, Electrochimica Acta 55 (2010) 8960:
        // ocppara_[0]         = DeltaG
        // ocppara_[1,2,3,...] = Redlich-Kister coefficients

        // terms not associated with any Redlich-Kister coefficient
        ocp = params_->ocppara_[0] + faraday / frt * std::log((1.0 - X) / X);

        // terms associated with first and second Redlich-Kister coefficients
        // these two terms are separated from the remaining sum and simplified thereafter to remove
        // singularities in the expansion in case X == 0.5
        ocp += params_->ocppara_[1] * (2.0 * X - 1.0) +
               params_->ocppara_[2] * (6.0 * X * X - 6.0 * X + 1.0);

        // terms associated with remaining Redlich-Kister coefficients
        for (unsigned int i = 2; i < params_->ocppara_.size() - 1; ++i)
        {
          ocp += params_->ocppara_[i + 1] *
                 (std::pow(2.0 * X - 1.0, i + 1) -
                     2.0 * i * X * (1.0 - X) * std::pow(2.0 * X - 1.0, i - 1));
        }

        // final scaling
        ocp /= faraday;

        break;
      }

      case Mat::PAR::OCPModels::taralov:
      {
        // cf. Taralov, Taralova, Popov, Iliev, Latz, and Zausch (2012)
        ocp = params_->ocppara_[0] +
              params_->ocppara_[1] * std::tanh(params_->ocppara_[2] * X + params_->ocppara_[3]) +
              params_->ocppara_[4] * std::exp(params_->ocppara_[5] * std::pow(X, 8.0)) +
              params_->ocppara_[6] *
                  (1 / (std::pow((params_->ocppara_[7] - X), params_->ocppara_[8])) +
                      params_->ocppara_[9]) +
              params_->ocppara_[10] * std::exp(params_->ocppara_[11] * (X + params_->ocppara_[12]));

        break;
      }

      default:
      {
        FOUR_C_THROW("Model for half cell open circuit potential not recognized!");
      }
    }
  }

  // non-physical intercalation fraction
  else
  {
    ocp = std::numeric_limits<double>::infinity();
  }

  return ocp;
}  // Mat::Electrode::compute_open_circuit_potential

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::Electrode::compute_d_open_circuit_potential_d_concentration(
    const double concentration, const double faraday, const double frt, const double detF) const
{
  const double X = compute_intercalation_fraction(concentration, chi_max(), c_max(), detF);
  const double d_ocp_dX =
      compute_d_open_circuit_potential_d_intercalation_fraction(X, faraday, frt);
  const double d_X_dc = compute_d_intercalation_fraction_d_concentration(chi_max(), c_max(), detF);

  return d_ocp_dX * d_X_dc;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::Electrode::compute_d_open_circuit_potential_d_intercalation_fraction(
    const double X, const double faraday, const double frt) const

{
  double d_ocp_dX(0.0);

  // physically reasonable intercalation fraction
  if (X > 0.0 and X < 1.0)
  {
    switch (params_->ocpmodel_)
    {
      // derivative of the open circuit potential w.r.t. concentration, obtained from function
      case Mat::PAR::OCPModels::function:
      {
        d_ocp_dX = params_->ocp_function().evaluate_derivative(X, 1);

        break;
      }

      // derivative of the open circuit potential w.r.t. concentration according to Redlich-Kister
      // expansion
      case Mat::PAR::OCPModels::redlichkister:
      {
        // cf. Colclasure and Kee, Electrochimica Acta 55 (2010) 8960:
        // ocppara_[0]         = DeltaG
        // ocppara_[1,2,3,...] = Redlich-Kister coefficients

        // term not associated with any Redlich-Kister coefficient
        d_ocp_dX = faraday / (2.0 * frt * X * (X - 1.0));

        // terms associated with first, second, and third Redlich-Kister coefficients
        // these three terms are separated from the remaining sum and simplified thereafter to
        // remove singularities in the derivative of the expansion in case X == 0.5
        d_ocp_dX += params_->ocppara_[1] + params_->ocppara_[2] * (6.0 * X - 3.0) +
                    params_->ocppara_[3] * (24.0 * X * X - 24.0 * X + 5.0);

        // terms associated with remaining Redlich-Kister coefficients
        for (unsigned int i = 3; i < params_->ocppara_.size() - 1; ++i)
        {
          d_ocp_dX += params_->ocppara_[i + 1] *
                      ((2.0 * i + 1.0) * std::pow(2.0 * X - 1.0, i) +
                          2.0 * X * i * (X - 1.0) * (i - 1.0) * std::pow(2.0 * X - 1.0, i - 2));
        }

        // intermediate scaling
        d_ocp_dX *= 2.0 / faraday;

        break;
      }

      // derivative of half cell open circuit potential w.r.t. concentration according to Taralov,
      // Taralova, Popov, Iliev, Latz, and Zausch (2012)
      case Mat::PAR::OCPModels::taralov:
      {
        d_ocp_dX = params_->ocppara_[1] * params_->ocppara_[2] /
                       std::pow(std::cosh(params_->ocppara_[2] * X + params_->ocppara_[3]), 2) +
                   8.0 * params_->ocppara_[4] * params_->ocppara_[5] *
                       std::exp(params_->ocppara_[5] * std::pow(X, 8)) * std::pow(X, 7) +
                   params_->ocppara_[6] * params_->ocppara_[8] /
                       std::pow(params_->ocppara_[7] - X, params_->ocppara_[8] + 1.0) +
                   params_->ocppara_[10] * params_->ocppara_[11] *
                       std::exp(params_->ocppara_[11] * (X + params_->ocppara_[12]));

        break;
      }

      default:
      {
        FOUR_C_THROW("Model for half cell open circuit potential not recognized!");
      }
    }
  }

  // non-physical intercalation fraction
  else
  {
    d_ocp_dX = std::numeric_limits<double>::infinity();
  }

  return d_ocp_dX;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::Electrode::compute_d_open_circuit_potential_d_det_f(
    const double concentration, const double faraday, const double frt, const double detF) const
{
  const double X = compute_intercalation_fraction(concentration, chi_max(), c_max(), detF);
  const double d_OCP_dX =
      compute_d_open_circuit_potential_d_intercalation_fraction(X, faraday, frt);
  const double d_X_ddetF =
      compute_d_intercalation_fraction_d_det_f(concentration, chi_max(), c_max());

  return d_OCP_dX * d_X_ddetF;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::Electrode::compute_d2_open_circuit_potential_d_concentration_d_concentration(
    const double concentration, const double faraday, const double frt, const double detF) const
{
  double d2_ocp_dX2(0.0), d2_ocp_dc2(0.0);

  // intercalation fraction
  const double X = compute_intercalation_fraction(concentration, chi_max(), c_max(), detF);
  const double d_X_dc = compute_d_intercalation_fraction_d_concentration(chi_max(), c_max(), detF);

  // physically reasonable intercalation fraction
  if (X > 0.0 and X < 1.0)
  {
    switch (params_->ocpmodel_)
    {
      // second derivative of the open circuit potential w.r.t. concentration, obtained from
      // function
      case Mat::PAR::OCPModels::function:
      {
        d2_ocp_dX2 = params_->ocp_function().evaluate_derivative(X, 2);

        break;
      }

      // second derivative of the open circuit potential w.r.t. concentration according to
      // Redlich-Kister expansion
      case Mat::PAR::OCPModels::redlichkister:
      {
        // cf. Colclasure and Kee, Electrochimica Acta 55 (2010) 8960:
        // ocppara_[0]         = DeltaG
        // ocppara_[1,2,3,...] = Redlich-Kister coefficients

        // term not associated with any Redlich-Kister coefficient
        d2_ocp_dX2 = -faraday * (2.0 * X - 1.0) / (4.0 * frt * X * X * (X - 1.0) * (X - 1.0));

        // term associated with first Redlich-Kister coefficient vanishes

        // terms associated with second, third, and fourth Redlich-Kister coefficients
        // these three terms are separated from the remaining sum and simplified thereafter to
        // remove singularities in the second derivative of the expansion in case X == 0.5
        d2_ocp_dX2 += 3.0 * params_->ocppara_[2] + params_->ocppara_[3] * (24.0 * X - 12.0) +
                      params_->ocppara_[4] * (120.0 * X * X - 120.0 * X + 27.0);

        // terms associated with remaining Redlich-Kister coefficients
        for (unsigned int i = 4; i < params_->ocppara_.size() - 1; ++i)
        {
          d2_ocp_dX2 +=
              params_->ocppara_[i + 1] *
              (3.0 * i * i * std::pow(2.0 * X - 1.0, i - 1) +
                  2.0 * i * (i - 1.0) * (i - 2.0) * X * (X - 1.0) * std::pow(2.0 * X - 1.0, i - 3));
        }

        // intermediate scaling
        d2_ocp_dX2 *= 4.0 / faraday;

        break;
      }

      // second derivative of half cell open circuit potential w.r.t. concentration according to
      // Taralov, Taralova, Popov, Iliev, Latz, and Zausch (2012)
      case Mat::PAR::OCPModels::taralov:
      {
        d2_ocp_dX2 = -2.0 * params_->ocppara_[1] * std::pow(params_->ocppara_[2], 2) /
                         std::pow(std::cosh(params_->ocppara_[2] * X + params_->ocppara_[3]), 2) *
                         std::tanh(params_->ocppara_[2] * X + params_->ocppara_[3]) +
                     8.0 * params_->ocppara_[4] * params_->ocppara_[5] * std::pow(X, 6) *
                         std::exp(params_->ocppara_[5] * std::pow(X, 8)) *
                         (7.0 + 8.0 * params_->ocppara_[5] * std::pow(X, 8)) +
                     params_->ocppara_[6] * params_->ocppara_[8] * (params_->ocppara_[8] + 1.0) /
                         std::pow(params_->ocppara_[7] - X, params_->ocppara_[8] + 2.0) +
                     params_->ocppara_[10] * std::pow(params_->ocppara_[11], 2) *
                         std::exp(params_->ocppara_[11] * (X + params_->ocppara_[12]));

        break;
      }

      default:
      {
        FOUR_C_THROW("Model for half cell open circuit potential not recognized!");
      }
    }

    d2_ocp_dc2 = d2_ocp_dX2 * d_X_dc * d_X_dc;
  }

  // non-physical intercalation fraction
  else
  {
    d2_ocp_dc2 = std::numeric_limits<double>::infinity();
  }

  return d2_ocp_dc2;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::Electrode::compute_d_open_circuit_potential_d_temperature(
    const double concentration, const double faraday, const double gasconstant) const
{
  double ocpderiv = 0.0;
  switch (params_->ocpmodel_)
  {
    case Mat::PAR::OCPModels::function:
    case Mat::PAR::OCPModels::taralov:
    {
      ocpderiv = 0.0;
      break;
    }
    case Mat::PAR::OCPModels::redlichkister:
    {
      const double X = compute_intercalation_fraction(concentration, chi_max(), c_max(), 1.0);

      ocpderiv = std::log((1.0 - X) / X) * gasconstant / faraday;
      break;
    }
    default:
    {
      FOUR_C_THROW("Model for half cell open circuit potential not recognized!");
    }
  }
  return ocpderiv;
}

FOUR_C_NAMESPACE_CLOSE
