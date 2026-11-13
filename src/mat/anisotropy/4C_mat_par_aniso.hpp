// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PAR_ANISO_HPP
#define FOUR_C_MAT_PAR_ANISO_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace Elastic
  {
    class StructuralTensorStrategyBase;
  }
}  // namespace Mat

/*----------------------------------------------------------------------*/
/* declarations */
namespace Mat
{
  namespace PAR
  {
    /// Extension to hold 'quick' access material parameters for anisotropy
    class ParameterAniso : public Core::Mat::PAR::Parameter
    {
     public:
      /// construct the material object given material parameters
      ParameterAniso(const Core::Mat::PAR::Parameter::Data& matdata);

      /// return pointer to strategy
      const std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>&
      structural_tensor_strategy()
      {
        return structural_tensor_strategy_;
      };

     private:
      /// structural tensor strategy
      std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase> structural_tensor_strategy_;

    };  // class ParameterAniso

  }  // namespace PAR

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
