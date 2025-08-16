// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_MATERIAL_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_MATERIAL_HPP


#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  // forward declaration
  template <typename T>
  class RemodelFiberMaterial;

  namespace PAR
  {
    template <typename T>
    class RemodelFiberMaterial : public Core::Mat::PAR::Parameter
    {
     public:
      RemodelFiberMaterial(const Core::Mat::PAR::Parameter::Data& matdata);
      std::shared_ptr<Core::Mat::Material> create_material() override
      {
        FOUR_C_THROW("This type of material is not created with create_material()");
      }

      [[nodiscard]] virtual std::unique_ptr<Mixture::RemodelFiberMaterial<T>>
      create_remodel_fiber_material() const = 0;
    };
  }  // namespace PAR

  template <typename T>
  class RemodelFiberMaterial
  {
   public:
    virtual ~RemodelFiberMaterial() = default;

    /*!
     * @brief Evaluates the Cauchy stress as a function of I4
     *
     * @param I4 Fourth invariant of the Cauchy-Green tensor
     * @return T
     */
    [[nodiscard]] virtual T get_cauchy_stress(T I4) const = 0;

    /*!
     * @brief Evaluates the first derivative of the Cauchy stress w.r.t. I4
     *
     * @param I4 Fourth invariant of the Cauchy-Green tensor
     * @return T
     */
    [[nodiscard]] virtual T get_d_cauchy_stress_d_i4(T I4) const = 0;

    /*!
     * @brief Evaluates the second derivative of the Cauchy stress w.r.t. I4
     *
     * @param I4 Fourth invariant of the Cauchy-Green tensor
     * @return T
     */
    [[nodiscard]] virtual T get_d_cauchy_stress_d_i4_d_i4(T I4) const = 0;
  };

}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
