// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_functions.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_utils_function_manager.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  /// returns St. Venant Kirchhof quick access parameters from given material id
  const Mat::PAR::StVenantKirchhoff& get_svk_mat_pars(int mat_id)
  {
    auto* params = Global::Problem::instance()->materials()->parameter_by_id(mat_id);
    if (params->type() != Core::Materials::m_stvenant)
      FOUR_C_THROW("Material %d is not a St.Venant-Kirchhoff structure material", mat_id);
    auto* fparams = dynamic_cast<Mat::PAR::StVenantKirchhoff*>(params);
    if (!fparams) FOUR_C_THROW("Material does not cast to St.Venant-Kirchhoff structure material");
    return *fparams;
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  std::shared_ptr<Core::Utils::FunctionOfSpaceTime> create_structure_function(
      const std::vector<Input::LineDefinition>& function_line_defs)
  {
    if (function_line_defs.size() != 1) return nullptr;

    const auto& function_lin_def = function_line_defs.front();

    if (function_lin_def.container().get_or("WEAKLYCOMPRESSIBLE_ETIENNE_FSI_STRUCTURE", false))
    {
      // read data
      int mat_id_struc = function_lin_def.container().get_or<int>("MAT_STRUC", -1);

      if (mat_id_struc <= 0)
        FOUR_C_THROW(
            "Please give a (reasonable) 'MAT_STRUC' in WEAKLYCOMPRESSIBLE_ETIENNE_FSI_STRUCTURE");

      // get materials
      auto fparams = get_svk_mat_pars(mat_id_struc);

      return std::make_shared<Solid::WeaklyCompressibleEtienneFSIStructureFunction>(fparams);
    }
    else if (function_lin_def.container().get_or(
                 "WEAKLYCOMPRESSIBLE_ETIENNE_FSI_STRUCTURE_FORCE", false))
    {
      // read data
      int mat_id_struc = function_lin_def.container().get_or<int>("MAT_STRUC", -1);

      if (mat_id_struc <= 0)
      {
        FOUR_C_THROW(
            "Please give a (reasonable) 'MAT_STRUC' in "
            "WEAKLYCOMPRESSIBLE_ETIENNE_FSI_STRUCTURE_FORCE");
      }

      // get materials
      auto fparams = get_svk_mat_pars(mat_id_struc);

      return std::make_shared<Solid::WeaklyCompressibleEtienneFSIStructureForceFunction>(fparams);
    }
    else
    {
      return std::shared_ptr<Core::Utils::FunctionOfSpaceTime>(nullptr);
    }
  }
}  // namespace


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Solid::add_valid_structure_functions(Core::Utils::FunctionManager& function_manager)
{
  std::vector<Input::LineDefinition> lines;
  lines.emplace_back(Input::LineDefinition::Builder()
                         .add_tag("WEAKLYCOMPRESSIBLE_ETIENNE_FSI_STRUCTURE")
                         .add_named_int("MAT_STRUC")
                         .build());
  lines.emplace_back(Input::LineDefinition::Builder()
                         .add_tag("WEAKLYCOMPRESSIBLE_ETIENNE_FSI_STRUCTURE_FORCE")
                         .add_named_int("MAT_STRUC")
                         .build());

  function_manager.add_function_definition(std::move(lines), create_structure_function);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Solid::WeaklyCompressibleEtienneFSIStructureFunction::WeaklyCompressibleEtienneFSIStructureFunction(
    const Mat::PAR::StVenantKirchhoff& fparams)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Solid::WeaklyCompressibleEtienneFSIStructureFunction::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  // ease notation
  double x = xp[0];
  double y = xp[1];

  // initialize variables
  Core::LinAlg::Matrix<2, 1> u_ex;

  // evaluate variables
  u_ex(0) = -((cos(2. * M_PI * t) * cos(2. * M_PI * x)) / 6. + 1.) * (y - 1.);
  u_ex(1) =
      -(sin(2. * M_PI * x) * sin(2. * M_PI * (t + 1. / 4.)) * (cos(2. * M_PI * x) - 1.)) / 20.;

  switch (component)
  {
    case 0:
      return u_ex(0);
    case 1:
      return u_ex(1);

    default:
      return 1.0;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<double> Solid::WeaklyCompressibleEtienneFSIStructureFunction::evaluate_time_derivative(
    const double* xp, const double t, const unsigned deg, const std::size_t component) const
{
  // resulting vector holding
  std::vector<double> res(deg + 1);

  // add the value at time t
  res[0] = evaluate(xp, t, component);

  // add the 1st time derivative at time t
  if (deg >= 1)
  {
    // ease notation
    double x = xp[0];
    double y = xp[1];

    // initialize variables
    Core::LinAlg::Matrix<2, 1> dudt_ex;

    // evaluate variables
    dudt_ex(0) = (M_PI * cos(2. * M_PI * x) * sin(2. * M_PI * t) * (y - 1.)) / 3.;
    dudt_ex(1) =
        -(M_PI * sin(2. * M_PI * x) * cos(2. * M_PI * (t + 1. / 4.)) * (cos(2. * M_PI * x) - 1.)) /
        10.;

    switch (component)
    {
      case 0:
        res[1] = dudt_ex(0);
        break;
      case 1:
        res[1] = dudt_ex(1);
        break;

      default:
        res[1] = 0.0;
    }
  }

  // add the 2nd time derivative at time t
  if (deg >= 2)
  {
    res[2] = 0.0;
  }

  // error for higher derivatives
  if (deg >= 3)
  {
    FOUR_C_THROW("Higher time derivatives than second not supported!");
  }

  return res;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Solid::WeaklyCompressibleEtienneFSIStructureForceFunction::
    WeaklyCompressibleEtienneFSIStructureForceFunction(const Mat::PAR::StVenantKirchhoff& fparams)
    : youngmodulus_(fparams.youngs_),
      poissonratio_(fparams.poissonratio_),
      strucdensity_(fparams.density_)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Solid::WeaklyCompressibleEtienneFSIStructureForceFunction::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  // ease notation
  double x = xp[0];
  double y = xp[1];
  double E = youngmodulus_;
  double v = poissonratio_;
  double r = strucdensity_;

  // initialize variables
  Core::LinAlg::Matrix<2, 1> f_u_ex;

  // evaluate variables
  f_u_ex(0) = (2. * (std::pow(M_PI, 2.)) * cos(2. * M_PI * t) * cos(2. * M_PI * x) * (y - 1.) *
                  (E - r - E * v + r * v + 2. * r * (std::pow(v, 2.)))) /
              (3. * (2. * (std::pow(v, 2.)) + v - 1.));
  f_u_ex(1) = ((std::pow(M_PI, 2.)) * r * sin(2. * M_PI * x) * sin(2. * M_PI * (t + 1. / 4.)) *
                  (cos(2. * M_PI * x) - 1.)) /
                  5. -
              (E * ((M_PI * sin(2. * M_PI * x) * sin(2. * M_PI * (t + 1. / 4.))) / 3. +
                       (3. * (std::pow(M_PI, 2.)) * cos(2. * M_PI * x) * sin(2. * M_PI * x) *
                           sin(2. * M_PI * (t + 1. / 4.))) /
                           5. +
                       ((std::pow(M_PI, 2.)) * sin(2. * M_PI * x) * sin(2. * M_PI * (t + 1. / 4.)) *
                           (cos(2. * M_PI * x) - 1.)) /
                           5.)) /
                  (2. * v + 2.) +
              (E * M_PI * v * sin(2. * M_PI * x) * sin(2. * M_PI * (t + 1. / 4.))) /
                  (3. * (2. * v - 1.) * (v + 1.));

  switch (component)
  {
    case 0:
      return f_u_ex(0);
    case 1:
      return f_u_ex(1);

    default:
      return 1.0;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<double>
Solid::WeaklyCompressibleEtienneFSIStructureForceFunction::evaluate_time_derivative(
    const double* xp, const double t, const unsigned deg, const std::size_t component) const
{
  // resulting vector holding
  std::vector<double> res(deg + 1);

  // add the value at time t
  res[0] = evaluate(xp, t, component);

  // add the 1st time derivative at time t
  if (deg >= 1)
  {
    res[1] = 0.0;
  }

  // add the 2nd time derivative at time t
  if (deg >= 2)
  {
    res[2] = 0.0;
  }

  // error for higher derivatives
  if (deg >= 3)
  {
    FOUR_C_THROW("Higher time derivatives than second not supported!");
  }

  return res;
}

FOUR_C_NAMESPACE_CLOSE
