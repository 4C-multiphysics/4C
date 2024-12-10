// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP


#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <map>
#include <string>


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /// enum class for error types in InelasticDefgradTransvIsotropElastViscoplast, used for
  /// triggering the substepping procedures
  enum class ViscoplastErrorType
  {
    NoErrors,
    NegativePlasticStrain,  // negative plastic strain which does not allow for evaluations
                            // inside the viscoplasticity laws
    OverflowError,  // overflow error of the term \f$ \Delta t \dot{\varepsilon}^{\text{p}} \f$ (and
                    // \f$ \mathsymbol{E}^{\text{p}}  = \exp(- \Delta t \dot{\varepsilon}^{\text{p}}
                    // \mathsymbol{N}^{\text{p}}) \f$) checked in the standard substepping procedure
    NoPlasticIncompressibility,  // no plastic incompressibility, meaning that our determinant
                                 // of the inelastic defgrad is far from 1
    FailedSolLinSystLNL,  // solution of the linear system in the Local Newton-Raphson Loop failed
    NoConvergenceLNL,     // the Local Newton Loop did not converge for the given loop settings
    SingularJacobian,     // singular Jacobian after converged LNL, which does not enable our
                          // analytical evaluation of the linearization
    FailedSolAnalytLinearization,  // solution of the linear system in the analytical linearization
                                   // failed
  };

  /// to_string: error types to error messages in InelasticDefgradTransvIsotropElastViscoplast
  inline std::string to_string(Mat::ViscoplastErrorType err_type)
  {
    switch (err_type)
    {
      case Mat::ViscoplastErrorType::NegativePlasticStrain:
        return "Error in InelasticDefgradTransvIsotropElastViscoplast: negative plastic strain!";
      case Mat::ViscoplastErrorType::OverflowError:
        return "Error in InelasticDefgradTransvIsotropElastViscoplast: overflow error related to "
               "the evaluation of the plastic strain increment!";
      case Mat::ViscoplastErrorType::NoPlasticIncompressibility:
        return "Error in InelasticDefgradTransvIsotropElastViscoplast: plastic incompressibility "
               "not satisfied!";
      case Mat::ViscoplastErrorType::FailedSolLinSystLNL:
        return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
               "system in the Local Newton Loop failed!";
      case Mat::ViscoplastErrorType::NoConvergenceLNL:
        return "Error in InelasticDefgradTransvIsotropElastViscoplast: Local Newton Loop did not "
               "converge for the given loop settings!";
      case Mat::ViscoplastErrorType::SingularJacobian:
        return "Error in InelasticDefgradTransvIsotropElastViscoplast: singular Jacobian after  "
               "converged Local Newton Loop, which does not allow for the analytical evaluation of "
               "the linearization!";
      case Mat::ViscoplastErrorType::FailedSolAnalytLinearization:
        return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
               "system  in the analytical linearization failed";
      default:
        FOUR_C_THROW("to_string(Mat::ViscoplastErrorType): You should not be here!");
    }
  }

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif