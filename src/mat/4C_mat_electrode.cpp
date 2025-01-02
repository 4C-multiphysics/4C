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

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::Electrode::Electrode(const Core::Mat::PAR::Parameter::Data& matdata)
    : ElchSingleMat(matdata),
      cmax_(matdata.parameters.get<double>("C_MAX")),
      chimax_(matdata.parameters.get<double>("CHI_MAX")),
      ocpmodel_(string_to_ocp_model(matdata.parameters.get<std::string>("OCP_MODEL"))),
      ocpparanum_(matdata.parameters.get<int>("OCP_PARA_NUM")),
      ocppara_(matdata.parameters.get_or<std::vector<double>>(
          "OCP_PARA", std::vector<double>(ocpparanum_, 0.0))),
      X_(0, 0.0),
      b_(0, 0.0),
      a_(0, 0.0),
      m_(0, 0.0),
      xmin_(matdata.parameters.get<double>("X_MIN")),
      xmax_(matdata.parameters.get<double>("X_MAX"))
{
  // safety checks
  if (cmax_ < 1.0e-12)
    FOUR_C_THROW("Saturation value c_max of intercalated Lithium concentration is too small!");
  if (static_cast<int>(ocppara_.size()) != ocpparanum_)
  {
    FOUR_C_THROW(
        "Length of coefficient vector for electrode half cell open circuit potential doesn't match "
        "prescribed number of coefficients!");
  }
  if ((xmin_ > 1.0) or (xmax_ > 1.0))
  {
    FOUR_C_THROW(
        "Lower bound (X_MIN) and upper bound (X_MAX) of range of validity for ocp calculation "
        "model cannot be larger than one since X is calculated as c/c_max! If you do not want to "
        "prescribe bounds, you have to set the two variables to negative values. "
        "If you set the bounds to realistic values (i.e. [0,1]) you will get a warning printed to "
        "the screen if bounds are violated throughout the simulation time!");
  }
  if (xmin_ > xmax_) FOUR_C_THROW("X_MIN cannot be larger than X_MAX!");

  // additional preparations
  std::string ocpcsv(matdata.parameters.get<std::string>("OCP_CSV"));
  switch (ocpmodel_)
  {
    case ocp_csv:
    {
      // safety checks
      if (ocpcsv.length() == 0)
        FOUR_C_THROW(
            "You forgot to specify the *.csv file for the half cell open circuit potential!");
      if (ocpparanum_)
      {
        FOUR_C_THROW(
            "Must not specify any parameters in case half-cell open-circuit potential is to be "
            "determined via a *.csv file!");
      }

      // parse *.csv file
      if (ocpcsv[0] != '/')
      {
        if (Global::Problem::instance()->output_control_file() == nullptr)
        {
          std::cout << "WARNING: could not check, if OCP .csv file in MAT_electrode is correct."
                    << std::endl;

          break;
        }
        std::string ocpcsvpath =
            Global::Problem::instance()->output_control_file()->input_file_name();
        ocpcsvpath = ocpcsvpath.substr(0, ocpcsvpath.rfind('/') + 1);
        ocpcsv.insert(ocpcsv.begin(), ocpcsvpath.begin(), ocpcsvpath.end());
      }
      std::ifstream file(ocpcsv);
      if (!file.good()) FOUR_C_THROW("Invalid file!");
      std::string line, value;
      std::vector<double> ocp(0, 0.0);
      while (getline(file, line))
      {
        std::istringstream linestream(line);
        std::getline(linestream, value, ',');
        try
        {
          X_.push_back(std::stod(value));
        }
        catch (...)
        {
          continue;
        }

        std::getline(linestream, value);

        try
        {
          ocp.push_back(std::stod(value));
        }
        catch (...)
        {
          FOUR_C_THROW("Invalid *.csv file!");
        }
      }
      file.close();

      // safety checks
      if (X_.size() != ocp.size()) FOUR_C_THROW("Internal error! Vector lengths have to match!");
      if (X_.size() < 2)
        FOUR_C_THROW("Need at least two data points for cubic spline interpolation!");
      for (unsigned i = 0; i < X_.size() - 1; ++i)
        if (X_[i + 1] <= X_[i]) FOUR_C_THROW("Data points must be sorted in ascending order!");

      // build coefficient matrix and right-hand side
      const unsigned N = X_.size() - 2;
      Core::LinAlg::SerialDenseMatrix A(N, N);
      Core::LinAlg::SerialDenseVector M(N), B(N);
      for (unsigned i = 0; i < N; ++i)
      {
        const double Xm = X_[i + 1] - X_[i], Xp = X_[i + 2] - X_[i + 1], ocpm = ocp[i + 1] - ocp[i],
                     ocpp = ocp[i + 2] - ocp[i + 1];
        if (i > 0) A(i, i - 1) = Xm;
        A(i, i) = 2.0 * (Xm + Xp);
        if (i < N - 1) A(i, i + 1) = Xp;
        B(i) = -ocpm / Xm + ocpp / Xp;
      }

      // solve for third-order coefficients for cubic spline interpolation
      using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
      using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
      Teuchos::SerialDenseSolver<ordinalType, scalarType> solver;
      solver.setMatrix(Teuchos::rcpFromRef(A));
      solver.setVectors(Teuchos::rcpFromRef(M), Teuchos::rcpFromRef(B));
      solver.factorWithEquilibration(true);
      solver.solveToRefinedSolution(true);
      if (solver.factor() or solver.solve())
        FOUR_C_THROW("Solution of linear system of equations failed!");

      // fill coefficient vectors
      m_.resize(X_.size(), 0.0);
      for (unsigned i = 1; i < m_.size() - 1; ++i) m_[i] = M(i - 1);
      b_.resize(X_.size() - 1, 0.0);
      a_.resize(X_.size() - 1, 0.0);
      for (unsigned i = 0; i < b_.size(); ++i)
      {
        const double Xm = X_[i + 1] - X_[i];
        b_[i] = ocp[i] - Xm * Xm * m_[i];
        a_[i] = (ocp[i + 1] - ocp[i]) / Xm - Xm * (m_[i + 1] - m_[i]);
      }

      break;
    }

    case ocp_polynomial:
    case ocp_redlichkister:
    case ocp_taralov:
    {
      // safety checks
      if (ocpparanum_ < 1)
        FOUR_C_THROW("No parameters found for electrode half cell open circuit potential!");
      if (ocpcsv.length())
      {
        FOUR_C_THROW(
            "Must not specify *.csv file with data points for chosen half cell open circuit "
            "potential model!");
      }
      if (ocpmodel_ == ocp_taralov and ocpparanum_ != 13)
      {
        FOUR_C_THROW(
            "Electrode half cell open circuit potential according to Taralov, Taralova, Popov, "
            "Iliev, Latz, and Zausch (2012) needs to be specified by exactly 13 coefficients!");
      }

      break;
    }

    default:
    {
      // safety check
      FOUR_C_THROW("Invalid model for half-cell open-circuit potential!");

      break;
    }
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
Mat::PAR::OCPModels Mat::PAR::Electrode::string_to_ocp_model(
    const std::string& ocpmodelstring) const
{
  OCPModels ocpmodelenum(ocp_undefined);

  // Redlich-Kister expansion
  if (ocpmodelstring == "Redlich-Kister")
  {
    ocpmodelenum = ocp_redlichkister;
  }
  // empirical correlation given in Taralov, Taralova, Popov, Iliev, Latz, and Zausch (2012)
  else if (ocpmodelstring == "Taralov")
  {
    ocpmodelenum = ocp_taralov;
  }
  // polynomial
  else if (ocpmodelstring == "Polynomial")
  {
    ocpmodelenum = ocp_polynomial;
  }
  // *.csv file
  else if (ocpmodelstring == "csv")
  {
    ocpmodelenum = ocp_csv;
  }
  // unknown model
  else
  {
    ocpmodelenum = ocp_undefined;
  }

  return ocpmodelenum;
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
Mat::Electrode::Electrode(Mat::PAR::Electrode* params) : params_(params) {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::Electrode::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
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
        params_ = static_cast<Mat::PAR::Electrode*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
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
  if (((X < params_->xmin_) or (X > params_->xmax_)) and params_->xmax_ >= 0.0)
  {
    std::cout << "WARNING: intercalation fraction X = c/c_max is violating prescribed bounds of "
                 "ocp calculation model. Calculated values might therefore not be reasonable!"
              << std::endl;
    std::cout << "X: " << X << " lower bound is: " << params_->xmin_
              << " upper bound is: " << params_->xmax_ << std::endl
              << std::endl;
  }

  // physically reasonable intercalation fraction
  if (X > 0.0 and X < 1.0)
  {
    switch (params_->ocpmodel_)
    {
      // half cell open circuit potential obtained from cubic spline interpolation of *.csv data
      // points
      case Mat::PAR::ocp_csv:
      {
        // safety check
        if (X < params_->X_.front() or X > params_->X_.back())
          FOUR_C_THROW("Intercalation fraction X = %lf lies outside sampling point range!", X);

        // evaluate cubic spline interpolation
        for (unsigned i = 0; i < params_->m_.size() - 1; ++i)
        {
          if (X <= params_->X_[i + 1])
          {
            const double invdX = 1.0 / (params_->X_[i + 1] - params_->X_[i]);
            ocp = params_->m_[i] * invdX * std::pow(params_->X_[i + 1] - X, 3) +
                  params_->m_[i + 1] * invdX * std::pow(X - params_->X_[i], 3) +
                  params_->a_[i] * (X - params_->X_[i]) + params_->b_[i];
            break;
          }
        }

        break;
      }

      // half cell open circuit potential according to Redlich-Kister expansion
      case Mat::PAR::ocp_redlichkister:
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
        for (int i = 2; i < params_->ocpparanum_ - 1; ++i)
        {
          ocp += params_->ocppara_[i + 1] *
                 (std::pow(2.0 * X - 1.0, i + 1) -
                     2.0 * i * X * (1.0 - X) * std::pow(2.0 * X - 1.0, i - 1));
        }

        // final scaling
        ocp /= faraday;

        break;
      }

      case Mat::PAR::ocp_taralov:
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

      // polynomial ocp
      case Mat::PAR::ocp_polynomial:
      {
        // add constant
        ocp = params_->ocppara_[0];

        // add higher polynomial order terms
        for (int i = 1; i < params_->ocpparanum_; ++i) ocp += params_->ocppara_[i] * std::pow(X, i);

        break;
      }

      default:
      {
        FOUR_C_THROW("Model for half cell open circuit potential not recognized!");
        break;
      }
    }
  }

  // non-physical intercalation fraction
  else
    ocp = std::numeric_limits<double>::infinity();

  return ocp;
}  // Mat::Electrode::compute_open_circuit_potential

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::Electrode::compute_d_open_circuit_potential_d_concentration(
    double concentration, double faraday, double frt, double detF) const
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
    double X, double faraday, double frt) const

{
  double d_ocp_dX(0.0);

  // physically reasonable intercalation fraction
  if (X > 0.0 and X < 1.0)
  {
    switch (params_->ocpmodel_)
    {
      // derivative of half cell open circuit potential w.r.t. concentration, obtained from cubic
      // spline interpolation of *.csv data points
      case Mat::PAR::ocp_csv:
      {
        // safety check
        if (X < params_->X_.front() or X > params_->X_.back())
          FOUR_C_THROW("Intercalation fraction X = %lf lies outside sampling point range!", X);

        // evaluate derivative of cubic spline interpolation w.r.t. concentration
        for (unsigned i = 0; i < params_->m_.size() - 1; ++i)
        {
          if (X <= params_->X_[i + 1])
          {
            const double invdX = 1.0 / (params_->X_[i + 1] - params_->X_[i]);
            d_ocp_dX = -3.0 * params_->m_[i] * invdX * std::pow(params_->X_[i + 1] - X, 2) +
                       3.0 * params_->m_[i + 1] * invdX * std::pow(X - params_->X_[i], 2) +
                       params_->a_[i];
            break;
          }
        }

        break;
      }

      // derivative of half cell open circuit potential w.r.t. concentration according to
      // Redlich-Kister expansion
      case Mat::PAR::ocp_redlichkister:
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
        for (int i = 3; i < params_->ocpparanum_ - 1; ++i)
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
      case Mat::PAR::ocp_taralov:
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

      // derivative of polynomial half cell open circuit potential w.r.t. concentration
      case Mat::PAR::ocp_polynomial:
      {
        for (int i = 1; i < params_->ocpparanum_; ++i)
          d_ocp_dX += i * params_->ocppara_[i] * std::pow(X, i - 1);

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
    d_ocp_dX = std::numeric_limits<double>::infinity();

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
    double concentration, double faraday, double frt, double detF) const
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
      // second derivative of half cell open circuit potential w.r.t. concentration, obtained from
      // cubic spline interpolation of *.csv data points
      case Mat::PAR::ocp_csv:
      {
        // safety check
        if (X < params_->X_.front() or X > params_->X_.back())
          FOUR_C_THROW("Intercalation fraction X = %lf lies outside sampling point range!", X);

        // evaluate second derivative of cubic spline interpolation w.r.t. concentration
        for (unsigned i = 0; i < params_->m_.size() - 1; ++i)
        {
          if (X <= params_->X_[i + 1])
          {
            const double invdX = 1.0 / (params_->X_[i + 1] - params_->X_[i]);
            d2_ocp_dX2 = 6.0 * params_->m_[i] * invdX * (params_->X_[i + 1] - X) +
                         6.0 * params_->m_[i + 1] * invdX * (X - params_->X_[i]);
            break;
          }
        }

        break;
      }

      // second derivative of half cell open circuit potential w.r.t. concentration according to
      // Redlich-Kister expansion
      case Mat::PAR::ocp_redlichkister:
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
        for (int i = 4; i < params_->ocpparanum_ - 1; ++i)
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
      case Mat::PAR::ocp_taralov:
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

      // second derivative of polynomial half cell open circuit potential w.r.t. concentration
      case Mat::PAR::ocp_polynomial:
      {
        for (int i = 2; i < params_->ocpparanum_; ++i)
          d2_ocp_dX2 += i * (i - 1) * params_->ocppara_[i] * std::pow(X, i - 2);

        break;
      }

      default:
      {
        FOUR_C_THROW("Model for half cell open circuit potential not recognized!");
        break;
      }
    }

    d2_ocp_dc2 = d2_ocp_dX2 * d_X_dc * d_X_dc;
  }

  // non-physical intercalation fraction
  else
    d2_ocp_dc2 = std::numeric_limits<double>::infinity();

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
    case Mat::PAR::ocp_csv:
    case Mat::PAR::ocp_taralov:
    case Mat::PAR::ocp_polynomial:
    {
      ocpderiv = 0.0;
      break;
    }
    case Mat::PAR::ocp_redlichkister:
    {
      const double X = compute_intercalation_fraction(concentration, chi_max(), c_max(), 1.0);

      ocpderiv = std::log((1.0 - X) / X) * gasconstant / faraday;
      break;
    }
    default:
    {
      FOUR_C_THROW("Model for half cell open circuit potential not recognized!");
      break;
    }
  }
  return ocpderiv;
}

FOUR_C_NAMESPACE_CLOSE
