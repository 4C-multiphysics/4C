// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_LIB_IO_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_LIB_IO_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_utils_gauss_point_extrapolation.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_utils.hpp"
#include "4C_structure_new_gauss_point_data_output_manager.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{
  namespace Internal
  {
    /*!
     * @brief Assemble a vector into a matrix row
     *
     * @tparam num_str
     * @param vector (in) : Vector to be assembled into matrix
     * @param data (in/out) : Matrix the vector is assembled into
     * @param row (in) : Matrix row
     */
    template <std::size_t size>
    void assemble_symmetric_tensor_to_matrix_row(
        Core::LinAlg::SymmetricTensor<double, size, size> tensor,
        Core::LinAlg::SerialDenseMatrix& data, const int row)
    {
      for (unsigned i = 0; i < tensor.container().size(); ++i)
        data(row, static_cast<int>(i)) = tensor.data()[i];
    }
  }  // namespace Internal

  template <typename T>
  inline std::vector<char>& get_stress_data(const T& ele, const Teuchos::ParameterList& params)
  {
    if (ele.is_solid_params_interface())
    {
      return *ele.get_solid_params_interface().stress_data_ptr();
    }
    else
    {
      return *params.get<std::shared_ptr<std::vector<char>>>("stress");
    }
  }

  template <typename T>
  inline std::vector<char>& get_strain_data(const T& ele, const Teuchos::ParameterList& params)
  {
    if (ele.is_solid_params_interface())
    {
      return *ele.get_solid_params_interface().strain_data_ptr();
    }
    else
    {
      return *params.get<std::shared_ptr<std::vector<char>>>("strain");
    }
  }

  template <typename T>
  inline Inpar::Solid::StressType get_io_stress_type(
      const T& ele, const Teuchos::ParameterList& params)
  {
    if (ele.is_solid_params_interface())
    {
      return ele.get_solid_params_interface().get_stress_output_type();
    }
    else
    {
      return Teuchos::getIntegralValue<Inpar::Solid::StressType>(params, "iostress");
    }
  }

  template <typename T>
  inline Inpar::Solid::StrainType get_io_strain_type(
      const T& ele, const Teuchos::ParameterList& params)
  {
    if (ele.is_solid_params_interface())
    {
      return ele.get_solid_params_interface().get_strain_output_type();
    }
    else
    {
      return Teuchos::getIntegralValue<Inpar::Solid::StrainType>(params, "iostrain");
    }
  }

  /*!
   * @brief Convert Green-Lagrange strains to the desired strain type and assemble to a given matrix
   * row in stress-like Voigt notation
   *
   * @tparam celltype : Cell type
   * @param gl_strain (in) : Green-Lagrange strain
   * @param defgrd (in) : Deformation gradient
   * @param strain_type (in) : Strain type, i.e., Green-Lagrange or Euler-Almansi
   * @param data (in/out) : Matrix the strains are assembled into
   * @param row (in) : Matrix row
   */
  template <Core::FE::CellType celltype>
  void assemble_strain_type_to_matrix_row(
      const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>&
          gl_strain,
      const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& defgrd,
      const Inpar::Solid::StrainType strain_type, Core::LinAlg::SerialDenseMatrix& data,
      const int row)
  {
    switch (strain_type)
    {
      case Inpar::Solid::strain_gl:
      {
        Internal::assemble_symmetric_tensor_to_matrix_row(gl_strain, data, row);
        return;
      }
      case Inpar::Solid::strain_ea:
      {
        const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>,
            Core::FE::dim<celltype>>
            ea = Solid::Utils::green_lagrange_to_euler_almansi(gl_strain, defgrd);
        Internal::assemble_symmetric_tensor_to_matrix_row(ea, data, row);
        return;
      }
      case Inpar::Solid::strain_log:
      {
        const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>,
            Core::FE::dim<celltype>>
            log_strain = Solid::Utils::green_lagrange_to_log_strain(gl_strain);
        Internal::assemble_symmetric_tensor_to_matrix_row(log_strain, data, row);
        return;
      }
      case Inpar::Solid::strain_none:
        return;
      default:
        FOUR_C_THROW("strain type not supported");
        break;
    }
  }

  /*!
   * @brief Convert 2nd Piola-Kirchhoff stresses to the desired stress type and assemble to a given
   * matrix row in stress-like Voigt notation
   *
   * @tparam celltype : Cell type
   * @param defgrd (in) : Deformation gradient
   * @param stress (in) : 2nd Piola-Kirchhoff stress
   * @param stress_type (in) : Stress type, i.e., 2nd Piola-Kirchhoff or Cauchy
   * @param data (in/out) : Matrix the stresses are assembled into
   * @param row (in) : Matrix row
   */
  template <Core::FE::CellType celltype>
  void assemble_stress_type_to_matrix_row(
      const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& defgrd,
      const Stress<celltype>& stress, const Inpar::Solid::StressType stress_type,
      Core::LinAlg::SerialDenseMatrix& data, const int row)
  {
    switch (stress_type)
    {
      case Inpar::Solid::stress_2pk:
      {
        Internal::assemble_symmetric_tensor_to_matrix_row(stress.pk2_, data, row);
        return;
      }
      case Inpar::Solid::stress_cauchy:
      {
        Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>
            cauchy = Solid::Utils::pk2_to_cauchy(stress.pk2_, defgrd);
        Internal::assemble_symmetric_tensor_to_matrix_row(cauchy, data, row);
        return;
      }
      case Inpar::Solid::stress_none:

        return;
      default:
        FOUR_C_THROW("stress type not supported");
        break;
    }
  }

  /*!
   * @brief Serialize a matrix by conversion to a vector representation

   * @param matrix (in) : Matrix
   * @param serialized_matrix (in/out) : Serialized matrix
   */
  inline void serialize(
      const Core::LinAlg::SerialDenseMatrix& matrix, std::vector<char>& serialized_matrix)
  {
    Core::Communication::PackBuffer packBuffer;
    add_to_pack(packBuffer, matrix);
    std::copy(packBuffer().begin(), packBuffer().end(), std::back_inserter(serialized_matrix));
  }

  /*!
   * @brief Asks the material for the Gauss Point output quantities and adds the information to
   * the Gauss point output data manager
   *
   * @param num_gp (in) : Number of Gauss Points of the element
   * @param solid_material (in) : Solid material of the element
   * @param gp_data_output_manager (in/out) : Gauss point data output manager
   *                                          (only for new structure time integration)
   */
  inline void ask_and_add_quantities_to_gauss_point_data_output(const int num_gp,
      const Mat::So3Material& solid_material,
      Solid::ModelEvaluator::GaussPointDataOutputManager& gp_data_output_manager)
  {
    // Save number of Gauss Points of the element for gauss point data output
    gp_data_output_manager.add_element_number_of_gauss_points(num_gp);

    // holder for output quantity names and their size
    std::unordered_map<std::string, int> quantities_map{};

    // Ask material for the output quantity names and sizes
    solid_material.register_output_data_names(quantities_map);

    // Add quantities to the Gauss point output data manager (if they do not already exist)
    gp_data_output_manager.merge_quantities(quantities_map);
  }

  /*!
   * @brief Collect Gauss Point output data from material and assemble/interpolate depending on
   * output type to element center, Gauss Points, or nodes
   *
   * @tparam celltype : Cell type
   * @param stiffness_matrix_integration (in) : Container holding the integration points
   * @param solid_material (in) : Solid material of the element
   * @param ele (in) : Reference to the element
   * @param gp_data_output_manager (in/out) : Gauss point data output manager
   *                                          (only for new structure time integration)
   */
  template <Core::FE::CellType celltype>
  inline void collect_and_assemble_gauss_point_data_output(
      const Core::FE::GaussIntegration& stiffness_matrix_integration,
      const Mat::So3Material& solid_material, const Core::Elements::Element& ele,
      Solid::ModelEvaluator::GaussPointDataOutputManager& gp_data_output_manager)
  {
    // Collection and assembly of gauss point data
    for (const auto& quantity : gp_data_output_manager.get_quantities())
    {
      const std::string& quantity_name = quantity.first;
      const int quantity_size = quantity.second;

      // Step 1: Collect the data for each Gauss point for the material
      Core::LinAlg::SerialDenseMatrix gp_data(
          stiffness_matrix_integration.num_points(), quantity_size, true);
      bool data_available = solid_material.evaluate_output_data(quantity_name, gp_data);

      // Step 2: Assemble data based on output type (elecenter, postprocessed to nodes, Gauss
      // point)
      if (data_available)
      {
        switch (gp_data_output_manager.get_output_type())
        {
          case Inpar::Solid::GaussPointDataOutputType::element_center:
          {
            // compute average of the quantities
            std::shared_ptr<Core::LinAlg::MultiVector<double>> global_data =
                gp_data_output_manager.get_element_center_data().at(quantity_name);
            Core::FE::assemble_averaged_element_values(*global_data, gp_data, ele);
            break;
          }
          case Inpar::Solid::GaussPointDataOutputType::nodes:
          {
            std::shared_ptr<Core::LinAlg::MultiVector<double>> global_data =
                gp_data_output_manager.get_nodal_data().at(quantity_name);

            Core::LinAlg::Vector<int>& global_nodal_element_count =
                *gp_data_output_manager.get_nodal_data_count().at(quantity_name);

            Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<celltype>(
                ele, gp_data, *global_data, false, stiffness_matrix_integration);
            Core::FE::assemble_nodal_element_count(global_nodal_element_count, ele);
            break;
          }
          case Inpar::Solid::GaussPointDataOutputType::gauss_points:
          {
            std::vector<std::shared_ptr<Core::LinAlg::MultiVector<double>>>& global_data =
                gp_data_output_manager.get_gauss_point_data().at(quantity_name);
            Core::FE::assemble_gauss_point_values(global_data, gp_data, ele);
            break;
          }
          case Inpar::Solid::GaussPointDataOutputType::none:
            FOUR_C_THROW(
                "You specified a Gauss point data output type of none, so you should not end up "
                "here.");
          default:
            FOUR_C_THROW("Unknown Gauss point data output type.");
        }
      }
    }
  }
}  // namespace Discret::Elements
FOUR_C_NAMESPACE_CLOSE

#endif