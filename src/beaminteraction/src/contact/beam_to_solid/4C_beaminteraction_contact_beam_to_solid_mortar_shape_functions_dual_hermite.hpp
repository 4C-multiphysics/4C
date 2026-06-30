// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_MORTAR_SHAPE_FUNCTIONS_DUAL_HERMITE_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_MORTAR_SHAPE_FUNCTIONS_DUAL_HERMITE_HPP

#include "4C_config.hpp"

#include "4C_beam3_base.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_shape_functions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{
  struct HermiteDual : public GeometryPair::ElementDiscretizationBase<Core::FE::CellType::line2, 2>
  {
  };
}  // namespace BeamInteraction

namespace GeometryPair
{
  template <>
  struct ShapeFunctionData<BeamInteraction::HermiteDual>
  {
    double ref_length_;
  };

  template <>
  struct SetShapeFunctionData<BeamInteraction::HermiteDual>
  {
    static void set(ShapeFunctionData<BeamInteraction::HermiteDual>& shape_function_data,
        const Core::Elements::Element* element)
    {
      const auto* beam_element = dynamic_cast<const Discret::Elements::Beam3Base*>(element);
      if (beam_element == nullptr)
        FOUR_C_THROW(
            "The element pointer has to point to a valid beam element when evaluating the shape "
            "function data of a dual Hermite mortar shape function, as we need to get "
            "RefLength()!");

      shape_function_data.ref_length_ = beam_element->ref_length();
    }
  };


  template <>
  struct EvaluateShapeFunction<BeamInteraction::HermiteDual>
  {
    template <typename V, typename T>
    static void evaluate(V& N, const T& xi,
        const ShapeFunctionData<BeamInteraction::HermiteDual>& shape_function_data)
    {
      const T xi2 = xi * xi;
      const T xi3 = xi2 * xi;

      N(0) = -35.0 / 4.0 * xi3 + 15.0 / 4.0 * xi2 + 15.0 / 4.0 * xi - 3.0 / 4.0;

      N(1) = (105.0 * xi3 - 45.0 / 2.0 * xi2 - 60.0 * xi + 15.0 / 2.0) /
             shape_function_data.ref_length_;

      N(2) = 35.0 / 4.0 * xi3 + 15.0 / 4.0 * xi2 - 15.0 / 4.0 * xi - 3.0 / 4.0;

      N(3) = (105.0 * xi3 + 45.0 / 2.0 * xi2 - 60.0 * xi - 15.0 / 2.0) /
             shape_function_data.ref_length_;
    }
  };

  template <>
  struct PrintElementData<BeamInteraction::HermiteDual>
  {
    template <typename ScalarType>
    static void print(const ElementData<BeamInteraction::HermiteDual, ScalarType>& element_data,
        std::ostream& out)
    {
      constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
      out << std::setprecision(max_precision);
      out << "\nElement reference length: " << element_data.shape_function_data_.ref_length_;
      out << "\nElement state vector: ";
      element_data.element_position_.print(out);
    }
  };
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif