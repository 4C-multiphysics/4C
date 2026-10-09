// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_str_wrapper.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Solid::StepStatus Adapter::StructureNOXCorrectionWrapper::prepare_time_step_with_status()
{
  const Solid::StepStatus prepare_status = StructureWrapper::prepare_time_step_with_status();
  if (disstepinc_ != nullptr) disstepinc_->put_scalar(0.);
  return prepare_status;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureNOXCorrectionWrapper::evaluate(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disstepinc)
{
  // The field solver always expects an iteration increment only. And
  // there are Dirichlet conditions that need to be preserved. So take
  // the sum of increments we get from NOX and apply the latest iteration
  // increment only.
  // Naming:
  //
  // x^n+1_i+1 = x^n+1_i + disiterinc  (sometimes referred to as residual increment), and
  //
  // x^n+1_i+1 = x^n     + disstepinc

  if (disstepinc != nullptr)
  {
    // iteration increments
    std::shared_ptr<Core::LinAlg::Vector<double>> disiterinc =
        std::make_shared<Core::LinAlg::Vector<double>>(*disstepinc);
    if (disstepinc_ != nullptr)
    {
      disiterinc->update(-1.0, *disstepinc_, 1.0);

      // update incremental displacement member to provided step increments
      // shortly: disinc_^<i> := disp^<i+1>
      disstepinc_->update(1.0, *disstepinc, 0.0);
    }
    else
    {
      disstepinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*disstepinc);
    }

    // do structural update with provided residual displacements - iteration increment
    StructureWrapper::evaluate(disiterinc);
  }
  else
  {
    StructureWrapper::evaluate(nullptr);
  }
}

FOUR_C_NAMESPACE_CLOSE
