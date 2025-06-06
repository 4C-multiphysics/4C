// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_solid_3D_ele_calc_interface.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_lib_integration.hpp"
#include "4C_solid_3D_ele_calc_lib_io.hpp"
#include "4C_solid_3D_ele_calc_lib_nitsche.hpp"
#include "4C_solid_3D_ele_formulation.hpp"

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class So3Material;
}
namespace Solid::ModelEvaluator
{
  class GaussPointDataOutputManager;
}
namespace Discret::Elements
{

  /*!
   * @brief A solid evaluator for different element formulations
   *
   * This is a class that takes the element formulation @p ElementFormulation as a template
   * parameter and generates all necessary code that is needed such that a solid element can be
   * evaluated. The two additional template parameter @p PreparationData and @p HistoryData store
   * preparation data and history data.
   *
   * The type @p ElementFormulation defines static member function @p evaluate(...) for the
   * evaluation of the deformation gradient at a specific position @p xi. Additionally, it has to
   * provide @p Prepare(...) for eventual preparation of data that is constant. It should return the
   * type @p PreparationData that is passed to every subsequent call. The static functions
   * @p add_internal_force_vector(...) and @p add_stiffness_matrix(...) are adding the gauss point
   * contribution to the force vector or stiffness matrix, respectively. Some element formulations
   * require to store data @p HistoryData. This type needs to be default constructable. The static
   * member functions @p pack(...) and @p unpack(...) need to be defined to allow parallel
   * distribution and restarts.
   *
   * @tparam celltype : celltype of the evaluator
   * @tparam ElementFormulation : A type that defines the element formulation in static member
   * functions
   */
  template <Core::FE::CellType celltype, typename ElementFormulation>
  class SolidEleCalc
  {
   public:
    SolidEleCalc();

    void pack(Core::Communication::PackBuffer& data) const;

    void unpack(Core::Communication::UnpackBuffer& buffer);

    void setup(
        Mat::So3Material& solid_material, const Core::IO::InputParameterContainer& container);

    void material_post_setup(const Core::Elements::Element& ele, Mat::So3Material& solid_material);

    void evaluate_nonlinear_force_stiffness_mass(const Core::Elements::Element& ele,
        Mat::So3Material& solid_material, const Core::FE::Discretization& discretization,
        const std::vector<int>& lm, Teuchos::ParameterList& params,
        Core::LinAlg::SerialDenseVector* force_vector,
        Core::LinAlg::SerialDenseMatrix* stiffness_matrix,
        Core::LinAlg::SerialDenseMatrix* mass_matrix);

    void recover(Core::Elements::Element& ele, const Core::FE::Discretization& discretization,
        const std::vector<int>& lm, Teuchos::ParameterList& params);

    void update(const Core::Elements::Element& ele, Mat::So3Material& solid_material,
        const Core::FE::Discretization& discretization, const std::vector<int>& lm,
        Teuchos::ParameterList& params);

    void calculate_stress(const Core::Elements::Element& ele, Mat::So3Material& solid_material,
        const StressIO& stressIO, const StrainIO& strainIO,
        const Core::FE::Discretization& discretization, const std::vector<int>& lm,
        Teuchos::ParameterList& params);

    double calculate_internal_energy(const Core::Elements::Element& ele,
        Mat::So3Material& solid_material, const Core::FE::Discretization& discretization,
        const std::vector<int>& lm, Teuchos::ParameterList& params);

    void initialize_gauss_point_data_output(const Core::Elements::Element& ele,
        const Mat::So3Material& solid_material,
        Solid::ModelEvaluator::GaussPointDataOutputManager& gp_data_output_manager) const;

    void evaluate_gauss_point_data_output(const Core::Elements::Element& ele,
        const Mat::So3Material& solid_material,
        Solid::ModelEvaluator::GaussPointDataOutputManager& gp_data_output_manager) const;

    void update_prestress(const Core::Elements::Element& ele, Mat::So3Material& solid_material,
        const Core::FE::Discretization& discretization, const std::vector<int>& lm,
        Teuchos::ParameterList& params);

    void reset_to_last_converged(
        const Core::Elements::Element& ele, Mat::So3Material& solid_material);

    double get_normal_cauchy_stress_at_xi(const Core::Elements::Element& ele,
        Mat::So3Material& solid_material, const std::vector<double>& disp,
        const Core::LinAlg::Tensor<double, 3>& xi, const Core::LinAlg::Tensor<double, 3>& n,
        const Core::LinAlg::Tensor<double, 3>& dir, CauchyNDirLinearizations<3>& linearizations);

    void for_each_gauss_point(const Core::Elements::Element& ele, Mat::So3Material& solid_material,
        const Core::FE::Discretization& discretization, const std::vector<int>& lm,
        const std::function<void(Mat::So3Material&, double integration_factor, int gp)>& integrator)
        const;

    void set_integration_rule(const Core::FE::GaussIntegration& integration_rule)
    {
      stiffness_matrix_integration_ = integration_rule;
      mass_matrix_integration_ = integration_rule;
    }

    const Core::FE::GaussIntegration& get_gauss_rule_stiffness_integration() const
    {
      return stiffness_matrix_integration_;
    };

   private:
    /// static values for matrix sizes
    static constexpr int num_nodes_ = Core::FE::num_nodes(celltype);
    static constexpr int num_dim_ = Core::FE::dim<celltype>;
    static constexpr int num_dof_per_ele_ = num_nodes_ * num_dim_;
    static constexpr int num_str_ = num_dim_ * (num_dim_ + 1) / 2;

    Core::FE::GaussIntegration stiffness_matrix_integration_;
    Core::FE::GaussIntegration mass_matrix_integration_;

    SolidFormulationHistory<ElementFormulation> history_data_{};
    SolidFormulationHistory<ElementFormulation> old_history_data_{};

  };  // class SolidEleCalc
}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE
#endif