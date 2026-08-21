// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_VOLUME_MORTAR_SHAPE_FUNCTIONS_DUAL_HERMITE_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_VOLUME_MORTAR_SHAPE_FUNCTIONS_DUAL_HERMITE_HPP

#include "4C_config.hpp"

#include "4C_beam3_base.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_element_shape_functions.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{
  struct HermiteDual : public GeometryPair::ElementDiscretizationBase<Core::FE::CellType::line2, 2>
  {
  };
  using t_hermite_dual = HermiteDual;
}  // namespace BeamInteraction

namespace GeometryPair
{
  template <>
  struct ShapeFunctionData<BeamInteraction::t_hermite_dual>
  {
    double ref_length_;
    Core::LinAlg::Matrix<4, 4, double> dual_transformation_;
  };

  template <>
  struct SetShapeFunctionData<BeamInteraction::t_hermite_dual>
  {
    static void set(ShapeFunctionData<BeamInteraction::t_hermite_dual>& shape_function_data,
        const Core::Elements::Element* element)
    {
      const auto* beam_element = dynamic_cast<const Discret::Elements::Beam3Base*>(element);
      if (beam_element == nullptr)
        FOUR_C_THROW(
            "The element pointer has to point to a valid beam element when evaluating the shape "
            "function data of a dual Hermite mortar shape function, as we need to get "
            "RefLength()!");

      shape_function_data.ref_length_ = beam_element->ref_length();
      shape_function_data.dual_transformation_.clear();
    }
  };

  template <typename Mortar>
  struct SetBeamDependentShapeFunctionData
  {
    template <typename Beam, typename BeamReferencePosition, typename BeamShapeFunctionData>
    static void set(
        ShapeFunctionData<Mortar>&, const BeamReferencePosition&, const BeamShapeFunctionData&)
    {
    }
  };


  template <>
  struct SetBeamDependentShapeFunctionData<BeamInteraction::t_hermite_dual>
  {
    template <typename Beam, typename BeamReferencePosition, typename BeamShapeFunctionData>
    static void set(ShapeFunctionData<BeamInteraction::t_hermite_dual>& mortar_data,
        const BeamReferencePosition& beam_reference_position,
        const BeamShapeFunctionData& beam_shape_function_data)
    {
      constexpr unsigned int n_shape = 4;

      Core::LinAlg::Matrix<n_shape, n_shape, double> D(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<n_shape, n_shape, double> M(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 1, double> dr_beam_ref(Core::LinAlg::Initialization::zero);

      Core::LinAlg::Matrix<1, n_shape, double> N_primal(Core::LinAlg::Initialization::zero);

      constexpr std::array<double, 6> gauss_points{-0.932469514203152, -0.661209386466264,
          -0.238619186083197, 0.238619186083197, 0.661209386466264, 0.932469514203152};

      constexpr std::array<double, 6> gauss_weights{0.171324492379170, 0.360761573048139,
          0.467913934572691, 0.467913934572691, 0.360761573048139, 0.171324492379170};

      for (unsigned int i_gp = 0; i_gp < gauss_points.size(); ++i_gp)
      {
        const double eta = gauss_points[i_gp];

        GeometryPair::evaluate_position_derivative1<Beam>(
            eta, beam_reference_position, dr_beam_ref);

        const double integration_factor = gauss_weights[i_gp] * dr_beam_ref.norm2();

        N_primal.clear();
        GeometryPair::EvaluateShapeFunction<Beam>::evaluate(
            N_primal, eta, beam_shape_function_data);

        for (unsigned int j = 0; j < n_shape; ++j)
        {
          // D_jj = integral N_j J d eta
          D(j, j) += N_primal(j) * integration_factor;

          // M_jk = integral N_j N_k J d eta
          for (unsigned int k = 0; k < n_shape; ++k)
          {
            M(j, k) += N_primal(j) * N_primal(k) * integration_factor;
          }
        }
      }


      Core::LinAlg::Matrix<n_shape, n_shape, double> M_transpose(
          Core::LinAlg::Initialization::zero);

      for (unsigned int i = 0; i < n_shape; ++i)
        for (unsigned int j = 0; j < n_shape; ++j) M_transpose(i, j) = M(j, i);

      for (unsigned int row = 0; row < n_shape; ++row)
      {
        auto M_work = M_transpose;

        Core::LinAlg::Matrix<n_shape, 1, double> rhs(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<n_shape, 1, double> solution(Core::LinAlg::Initialization::zero);

        for (unsigned int j = 0; j < n_shape; ++j) rhs(j) = D(row, j);

        Core::LinAlg::FixedSizeSerialDenseSolver<n_shape, n_shape, 1> solver;
        solver.set_matrix(M_work);
        solver.set_vectors(solution, rhs);
        solver.solve();

        for (unsigned int j = 0; j < n_shape; ++j)
          mortar_data.dual_transformation_(row, j) = solution(j);
      }
    }
  };


  template <>
  struct EvaluateShapeFunction<BeamInteraction::t_hermite_dual>
  {
    template <typename V, typename T>
    static void evaluate(V& N, const T& xi,
        const ShapeFunctionData<BeamInteraction::t_hermite_dual>& shape_function_data)
    {
      // Primal cubic Hermite basis on [-1, 1].
      const T xi2 = xi * xi;
      const T xi3 = xi2 * xi;

      Core::LinAlg::Matrix<4, 1, T> N_primal(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<4, 1, T> N_dual(Core::LinAlg::Initialization::zero);

      N_primal(0) = 0.25 * (2.0 - 3.0 * xi + xi3);
      N_primal(1) = shape_function_data.ref_length_ / 8.0 * (1.0 - xi - xi2 + xi3);
      N_primal(2) = 0.25 * (2.0 + 3.0 * xi - xi3);
      N_primal(3) = shape_function_data.ref_length_ / 8.0 * (-1.0 - xi + xi2 + xi3);

      N_dual.multiply(shape_function_data.dual_transformation_, N_primal);

      for (unsigned int i = 0; i < 4; ++i)
      {
        N(i) = N_dual(i);
      }
    }
  };

  template <>
  struct PrintElementData<BeamInteraction::t_hermite_dual>
  {
    template <typename ScalarType>
    static void print(const ElementData<BeamInteraction::t_hermite_dual, ScalarType>& element_data,
        std::ostream& out)
    {
      constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
      out << std::setprecision(max_precision);
      out << "\nElement reference length: " << element_data.shape_function_data_.ref_length_;
      out << "\nElement state vector: ";
      element_data.element_position_.print(out);

      out << "\nElement reference length: " << element_data.shape_function_data_.ref_length_;

      out << "\nDual transformation matrix:\n";
      element_data.shape_function_data_.dual_transformation_.print(out);

      out << "\nElement state vector: ";
      element_data.element_position_.print(out);
    }
  };
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif